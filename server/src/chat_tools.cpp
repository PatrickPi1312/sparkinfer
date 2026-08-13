#include "chat_tools.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <utility>

namespace sparkinfer_server {
namespace {

using json = nlohmann::json;

constexpr const char* kImStart = "<|im_start|>";
constexpr const char* kImEnd = "<|im_end|>";
constexpr const char* kThinkOpen = "<think>";
constexpr const char* kThinkClose = "</think>";
constexpr const char* kToolCallOpen = "<tool_call>";
constexpr const char* kToolCallClose = "</tool_call>";
constexpr const char* kFunctionOpen = "<function=";
constexpr const char* kFunctionClose = "</function>";
constexpr const char* kParameterOpen = "<parameter=";
constexpr const char* kParameterClose = "</parameter>";
constexpr const char* kToolResponseOpen = "<tool_response>";
constexpr const char* kToolResponseClose = "</tool_response>";

const char* kToolInstructions = R"(# Tools

You have access to the following functions:

<tools>)";

const char* kToolInstructionsTail = R"(
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>)";

bool set_error(std::string& err, const std::string& message) {
    err = message;
    return false;
}

bool parse_strict_json(const std::string& text, json& value, std::string& err,
                       const std::string& where) {
    // nlohmann's ordinary DOM parser accepts duplicate object keys (last value wins). That is
    // dangerous for tool requests: an auditor, prompt renderer, and executor could otherwise
    // see different meanings. Its public parser callback reports every object/key boundary,
    // which lets us reject duplicates without depending on version-specific detail classes.
    std::vector<std::set<std::string>> object_keys;
    std::string duplicate_key;
    const json::parser_callback_t callback =
        [&](int, json::parse_event_t event, json& parsed) -> bool {
            if (event == json::parse_event_t::object_start) {
                object_keys.emplace_back();
            } else if (event == json::parse_event_t::key) {
                const std::string key = parsed.get<std::string>();
                if (object_keys.empty() || !object_keys.back().insert(key).second) {
                    if (duplicate_key.empty()) duplicate_key = key;
                }
            } else if (event == json::parse_event_t::object_end && !object_keys.empty()) {
                object_keys.pop_back();
            }
            return true;
        };
    try {
        value = json::parse(text, callback, true, false);
    } catch (const json::exception& ex) {
        return set_error(err, where + ": " + ex.what());
    }
    if (!duplicate_key.empty())
        return set_error(err, where + ": duplicate JSON object key " + duplicate_key);
    return true;
}

bool safe_protocol_name(const std::string& value) {
    if (value.empty()) return false;
    // Function and top-level argument names are injected into unquoted Qwen protocol tags.
    // Keep the accepted alphabet deliberately narrower than JSON object keys so no control,
    // whitespace, quoting, or tag-delimiter byte can change the rendered structure.
    for (const unsigned char c : value) {
        if (!std::isalnum(c) && c != '_' && c != '-' && c != '.' && c != ':') return false;
    }
    return true;
}

std::string htmlsafe_json_string(const std::string& value) {
    std::string encoded = json(value).dump(-1, ' ', true);
    std::string safe;
    safe.reserve(encoded.size());
    for (const char c : encoded) {
        switch (c) {
            case '<': safe += "\\u003c"; break;
            case '>': safe += "\\u003e"; break;
            case '&': safe += "\\u0026"; break;
            case '\'': safe += "\\u0027"; break;
            default: safe.push_back(c); break;
        }
    }
    return safe;
}

// Jinja's `tojson` filter uses htmlsafe Python json.dumps: sorted object keys, ASCII escapes,
// and a space after commas/colons. Match that byte-for-byte for the JSON shapes used by the
// pinned Qwen3.6 template so tools cannot inject template tags through descriptions/schemas.
std::string qwen_template_json(const json& value) {
    if (value.is_string()) return htmlsafe_json_string(value.get_ref<const std::string&>());
    if (value.is_array()) {
        std::ostringstream out;
        out << '[';
        for (size_t i = 0; i < value.size(); ++i) {
            if (i) out << ", ";
            out << qwen_template_json(value[i]);
        }
        out << ']';
        return out.str();
    }
    if (value.is_object()) {
        std::ostringstream out;
        out << '{';
        bool first = true;
        for (const auto& item : value.items()) {
            if (!first) out << ", ";
            first = false;
            out << htmlsafe_json_string(item.key()) << ": "
                << qwen_template_json(item.value());
        }
        out << '}';
        return out.str();
    }
    return value.dump(-1, ' ', true);
}

bool is_allowed_key(const json& object, const std::set<std::string>& allowed,
                    const std::string& where, std::string& err) {
    for (const auto& item : object.items()) {
        if (!allowed.count(item.key()))
            return set_error(err, where + " contains unsupported field " + item.key());
    }
    return true;
}

bool parse_content(const json& value, std::string& content, bool& is_null,
                   const std::string& where, std::string& err) {
    content.clear();
    is_null = value.is_null();
    if (is_null) return true;
    if (value.is_string()) {
        content = value.get<std::string>();
        return true;
    }
    if (!value.is_array())
        return set_error(err, where + ".content must be a string, null, or array of text parts");
    for (size_t i = 0; i < value.size(); ++i) {
        const json& part = value[i];
        if (!part.is_object() || !part.contains("type") || !part["type"].is_string())
            return set_error(err, where + ".content[" + std::to_string(i) + "] must have a string type");
        if (part["type"] != "text")
            return set_error(err, where + ".content[" + std::to_string(i) + "] has unsupported non-text type");
        if (!part.contains("text") || !part["text"].is_string())
            return set_error(err, where + ".content[" + std::to_string(i) + "].text must be a string");
        content += part["text"].get<std::string>();
    }
    return true;
}

bool parse_arguments(const json& value, std::string& compact, const std::string& where,
                     std::string& err) {
    json arguments;
    if (value.is_object()) {
        arguments = value;
    } else if (value.is_string()) {
        if (!parse_strict_json(value.get_ref<const std::string&>(), arguments, err,
                               where + " is not valid JSON")) return false;
    } else {
        return set_error(err, where + " must be a JSON object or an object encoded as a string");
    }
    if (!arguments.is_object()) return set_error(err, where + " must encode a JSON object");
    compact = arguments.dump();
    return true;
}

bool parse_tool_call(const json& value, ToolCall& call, const std::string& where,
                     std::string& err) {
    if (!value.is_object()) return set_error(err, where + " must be an object");
    if (!is_allowed_key(value, {"id", "type", "function"}, where, err)) return false;
    if (!value.contains("id") || !value["id"].is_string() || value["id"].get_ref<const std::string&>().empty())
        return set_error(err, where + ".id must be a non-empty string");
    if (value.contains("type") && (!value["type"].is_string() || value["type"] != "function"))
        return set_error(err, where + ".type must be function");
    if (!value.contains("function") || !value["function"].is_object())
        return set_error(err, where + ".function must be an object");
    const json& function = value["function"];
    if (!is_allowed_key(function, {"name", "arguments"}, where + ".function", err)) return false;
    if (!function.contains("name") || !function["name"].is_string() ||
        !safe_protocol_name(function["name"].get_ref<const std::string&>()))
        return set_error(err, where + ".function.name must be a non-empty string");
    if (!function.contains("arguments"))
        return set_error(err, where + ".function.arguments is required");
    call.id = value["id"].get<std::string>();
    call.name = function["name"].get<std::string>();
    return parse_arguments(function["arguments"], call.arguments, where + ".function.arguments", err);
}

bool is_nonnegative_integer(const json& value) {
    return value.is_number_unsigned() ||
           (value.is_number_integer() && value.get<json::number_integer_t>() >= 0);
}

bool valid_schema_node(const json& schema, const std::string& where, bool top_level,
                       std::string& err) {
    if (!schema.is_object()) return set_error(err, where + " must be an object");
    const std::set<std::string> allowed_types = {
        "array", "boolean", "integer", "null", "number", "object", "string"};
    if (schema.contains("type")) {
        const json& type = schema["type"];
        if (top_level) {
            if (!type.is_string() || type != "object")
                return set_error(err, where + ".type must be object");
        } else if (type.is_string()) {
            if (!allowed_types.count(type.get<std::string>()))
                return set_error(err, where + ".type is unsupported");
        } else if (type.is_array() && !type.empty()) {
            std::set<std::string> seen;
            for (const auto& item : type) {
                if (!item.is_string() || !allowed_types.count(item.get<std::string>()))
                    return set_error(err, where + ".type contains an unsupported type");
                if (!seen.insert(item.get<std::string>()).second)
                    return set_error(err, where + ".type contains a duplicate type");
            }
        } else {
            return set_error(err, where + ".type must be a string or non-empty array of strings");
        }
    }
    if (schema.contains("properties")) {
        if (!schema["properties"].is_object())
            return set_error(err, where + ".properties must be an object");
        for (const auto& property : schema["properties"].items()) {
            if (!valid_schema_node(property.value(), where + ".properties." + property.key(),
                                   false, err)) return false;
        }
    }
    if (schema.contains("required")) {
        if (!schema["required"].is_array()) return set_error(err, where + ".required must be an array");
        std::set<std::string> seen;
        for (const auto& item : schema["required"]) {
            if (!item.is_string() || item.get_ref<const std::string&>().empty())
                return set_error(err, where + ".required entries must be non-empty strings");
            const std::string name = item.get<std::string>();
            if (!seen.insert(name).second) return set_error(err, where + ".required contains duplicate " + name);
            if (schema.contains("properties") && !schema["properties"].contains(name))
                return set_error(err, where + ".required names unknown property " + name);
        }
    }
    if (schema.contains("additionalProperties") && !schema["additionalProperties"].is_boolean() &&
        !schema["additionalProperties"].is_object())
        return set_error(err, where + ".additionalProperties must be boolean or an object schema");
    if (schema.contains("additionalProperties") && schema["additionalProperties"].is_object() &&
        !valid_schema_node(schema["additionalProperties"], where + ".additionalProperties",
                           false, err)) return false;
    if (schema.contains("items")) {
        if (!schema["items"].is_object())
            return set_error(err, where + ".items must be an object schema");
        if (!valid_schema_node(schema["items"], where + ".items", false, err)) return false;
    }
    if (schema.contains("enum") && !schema["enum"].is_array())
        return set_error(err, where + ".enum must be an array");
    for (const char* keyword : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
        if (schema.contains(keyword) && !schema[keyword].is_number())
            return set_error(err, where + "." + keyword + " must be a number");
    }
    if (schema.contains("minimum") && schema.contains("maximum") &&
        schema["minimum"].get<double>() > schema["maximum"].get<double>())
        return set_error(err, where + ".minimum must not exceed maximum");
    for (const char* keyword : {"minItems", "maxItems", "minLength", "maxLength"}) {
        if (schema.contains(keyword) && !is_nonnegative_integer(schema[keyword]))
            return set_error(err, where + "." + keyword + " must be a non-negative integer");
    }
    if (schema.contains("minItems") && schema.contains("maxItems") &&
        schema["minItems"].get<std::size_t>() > schema["maxItems"].get<std::size_t>())
        return set_error(err, where + ".minItems must not exceed maxItems");
    if (schema.contains("minLength") && schema.contains("maxLength") &&
        schema["minLength"].get<std::size_t>() > schema["maxLength"].get<std::size_t>())
        return set_error(err, where + ".minLength must not exceed maxLength");
    if (schema.contains("pattern")) {
        if (!schema["pattern"].is_string())
            return set_error(err, where + ".pattern must be a string");
        try {
            (void)std::regex(schema["pattern"].get<std::string>(), std::regex::ECMAScript);
        } catch (const std::regex_error&) {
            return set_error(err, where + ".pattern is not a valid ECMAScript regular expression");
        }
    }
    return true;
}

bool valid_schema(const json& schema, const std::string& where, std::string& err) {
    return valid_schema_node(schema, where, true, err);
}

bool parse_tool_definition(const json& value, ToolDefinition& tool, const std::string& where,
                           std::string& err) {
    if (!value.is_object()) return set_error(err, where + " must be an object");
    if (!is_allowed_key(value, {"type", "function"}, where, err)) return false;
    if (!value.contains("type") || !value["type"].is_string() || value["type"] != "function")
        return set_error(err, where + ".type must be function");
    if (!value.contains("function") || !value["function"].is_object())
        return set_error(err, where + ".function must be an object");
    const json& function = value["function"];
    if (!is_allowed_key(function, {"name", "description", "parameters", "strict"},
                        where + ".function", err)) return false;
    if (!function.contains("name") || !function["name"].is_string() ||
        !safe_protocol_name(function["name"].get_ref<const std::string&>()))
        return set_error(err, where + ".function.name is not safe for the Qwen tool protocol");
    if (function.contains("description") && !function["description"].is_string())
        return set_error(err, where + ".function.description must be a string");
    if (!function.contains("parameters"))
        return set_error(err, where + ".function.parameters is required");
    if (!valid_schema(function["parameters"], where + ".function.parameters", err)) return false;
    if (function["parameters"].contains("properties")) {
        for (const auto& property : function["parameters"]["properties"].items()) {
            if (!safe_protocol_name(property.key()))
                return set_error(err, where + ".function.parameters property " + property.key() +
                                      " is not safe for the Qwen tool protocol");
        }
    }
    if (function.contains("strict") && !function["strict"].is_boolean())
        return set_error(err, where + ".function.strict must be boolean");
    tool.name = function["name"].get<std::string>();
    tool.spec = value;
    return true;
}

std::string trim_copy(std::string value) {
    auto ws = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && ws(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && ws(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

void trim_leading_ws(std::string& value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
}

void trim_trailing_ws(std::string& value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
}

void strip_trailing_im_end(std::string& value) {
    const std::string marker = kImEnd;
    const size_t pos = value.rfind(marker);
    if (pos != std::string::npos && value.substr(pos + marker.size()).find_first_not_of(" \t\r\n") == std::string::npos)
        value.resize(pos);
    trim_trailing_ws(value);
}

bool parse_message(const json& value, ChatMessage& message, size_t index, std::string& err) {
    const std::string where = "messages[" + std::to_string(index) + "]";
    if (!value.is_object()) return set_error(err, where + " must be an object");
    if (!is_allowed_key(value,
                        {"role", "content", "reasoning_content", "name", "tool_call_id", "tool_calls"},
                        where, err)) return false;
    if (!value.contains("role") || !value["role"].is_string())
        return set_error(err, where + ".role must be a string");
    message.role = value["role"].get<std::string>();
    if (message.role != "system" && message.role != "user" && message.role != "assistant" &&
        message.role != "tool")
        return set_error(err, where + ".role is unsupported");
    if (value.contains("content")) {
        if (!parse_content(value["content"], message.content, message.content_is_null, where, err)) return false;
    } else {
        message.content_is_null = true;
    }
    if (value.contains("reasoning_content")) {
        if (!value["reasoning_content"].is_string())
            return set_error(err, where + ".reasoning_content must be a string");
        message.reasoning_content = value["reasoning_content"].get<std::string>();
    }
    if (value.contains("name")) {
        if (!value["name"].is_string()) return set_error(err, where + ".name must be a string");
        message.name = value["name"].get<std::string>();
    }
    if (value.contains("tool_call_id")) {
        if (!value["tool_call_id"].is_string() || value["tool_call_id"].get_ref<const std::string&>().empty())
            return set_error(err, where + ".tool_call_id must be a non-empty string");
        message.tool_call_id = value["tool_call_id"].get<std::string>();
    }
    if (value.contains("tool_calls")) {
        if (message.role != "assistant") return set_error(err, where + ".tool_calls is only valid for assistant");
        if (!value["tool_calls"].is_array() || value["tool_calls"].empty())
            return set_error(err, where + ".tool_calls must be a non-empty array");
        std::set<std::string> ids;
        for (size_t i = 0; i < value["tool_calls"].size(); ++i) {
            ToolCall call;
            if (!parse_tool_call(value["tool_calls"][i], call,
                                 where + ".tool_calls[" + std::to_string(i) + "]", err)) return false;
            if (!ids.insert(call.id).second) return set_error(err, where + ".tool_calls contains duplicate id " + call.id);
            message.tool_calls.push_back(std::move(call));
        }
    }
    if (message.role == "system" && index != 0)
        return set_error(err, "system message must be first");
    if (message.role == "tool") {
        if (message.content_is_null) return set_error(err, where + ".content is required for tool messages");
        if (message.tool_call_id.empty()) return set_error(err, where + ".tool_call_id is required for tool messages");
    } else if (!message.tool_call_id.empty()) {
        return set_error(err, where + ".tool_call_id is only valid for tool messages");
    }
    if (message.role != "assistant" && message.content_is_null)
        return set_error(err, where + ".content must not be null for role " + message.role);
    if (message.role == "assistant" && message.content_is_null && message.tool_calls.empty())
        return set_error(err, where + " must contain content or tool_calls");
    return true;
}

bool resolve_history(const ChatRequest& request, std::string& err) {
    std::map<std::string, std::string> pending;
    for (size_t i = 0; i < request.messages.size(); ++i) {
        const ChatMessage& message = request.messages[i];
        if (message.role == "assistant" && !message.tool_calls.empty()) {
            if (!pending.empty())
                return set_error(err, "messages[" + std::to_string(i) + "] starts new tool calls before all prior tool results");
            for (const ToolCall& call : message.tool_calls) pending.emplace(call.id, call.name);
        } else if (message.role == "tool") {
            auto it = pending.find(message.tool_call_id);
            if (it == pending.end())
                return set_error(err, "messages[" + std::to_string(i) + "].tool_call_id does not reference a pending call");
            if (!message.name.empty() && message.name != it->second)
                return set_error(err, "messages[" + std::to_string(i) + "].name disagrees with its tool call");
            pending.erase(it);
        } else if (!pending.empty()) {
            return set_error(err, "tool results must immediately follow their assistant tool calls");
        }
    }
    if (!pending.empty()) return set_error(err, "assistant tool calls are missing tool result messages");
    return true;
}

std::string json_value_for_parameter(const json& value) {
    return value.is_string() ? value.get<std::string>() : qwen_template_json(value);
}

std::string render_tool_call(const ToolCall& call) {
    json arguments = json::parse(call.arguments);
    std::ostringstream out;
    out << kToolCallOpen << '\n' << kFunctionOpen << call.name << ">\n";
    for (const auto& item : arguments.items()) {
        out << kParameterOpen << item.key() << ">\n" << json_value_for_parameter(item.value())
            << '\n' << kParameterClose << '\n';
    }
    out << kFunctionClose << '\n' << kToolCallClose;
    return out.str();
}

bool has_protocol_markup(const std::string& text) {
    // Match incomplete/misspelled openings too. A partial protocol tag must never be returned as
    // assistant content merely because it failed to reach the exact full-marker search below.
    return text.find("<tool") != std::string::npos || text.find("</tool") != std::string::npos ||
           text.find("<function") != std::string::npos || text.find("</function") != std::string::npos ||
           text.find("<parameter") != std::string::npos || text.find("</parameter") != std::string::npos ||
           text.find("<think") != std::string::npos || text.find("</think") != std::string::npos ||
           text.find("<tool_response") != std::string::npos ||
           text.find("</tool_response") != std::string::npos ||
           text.find("<|im_") != std::string::npos;
}

ParsedToolOutput fail_tool_output(ParsedToolOutput out, const std::string& error) {
    out.reasoning_content.clear();
    out.content.clear();
    out.tool_calls.clear();
    out.error = error;
    return out;
}

bool parse_scalar_from_text(const std::string& value, json& parsed) {
    std::string ignored;
    return parse_strict_json(value, parsed, ignored, "parameter value");
}

bool validate_value(const json& value, const json& schema, const std::string& path, std::string& err) {
    if (!schema.is_object()) return true;
    if (schema.contains("type")) {
        auto matches = [&](const std::string& type) {
            if (type == "object") return value.is_object();
            if (type == "array") return value.is_array();
            if (type == "string") return value.is_string();
            if (type == "integer") return value.is_number_integer() || value.is_number_unsigned();
            if (type == "number") return value.is_number();
            if (type == "boolean") return value.is_boolean();
            if (type == "null") return value.is_null();
            return false;
        };
        bool ok = false;
        if (schema["type"].is_string()) ok = matches(schema["type"].get<std::string>());
        else if (schema["type"].is_array())
            for (const auto& type : schema["type"])
                if (type.is_string() && matches(type.get<std::string>())) ok = true;
        if (!ok) return set_error(err, path + " has the wrong JSON type");
    }
    if (schema.contains("enum")) {
        if (!schema["enum"].is_array()) return set_error(err, path + " has invalid enum schema");
        if (std::find(schema["enum"].begin(), schema["enum"].end(), value) == schema["enum"].end())
            return set_error(err, path + " is not one of the allowed enum values");
    }
    if (value.is_number()) {
        const double number = value.get<double>();
        if (schema.contains("minimum") && schema["minimum"].is_number() &&
            number < schema["minimum"].get<double>())
            return set_error(err, path + " is below minimum");
        if (schema.contains("maximum") && schema["maximum"].is_number() &&
            number > schema["maximum"].get<double>())
            return set_error(err, path + " is above maximum");
        if (schema.contains("exclusiveMinimum") && schema["exclusiveMinimum"].is_number() &&
            number <= schema["exclusiveMinimum"].get<double>())
            return set_error(err, path + " is not above exclusiveMinimum");
        if (schema.contains("exclusiveMaximum") && schema["exclusiveMaximum"].is_number() &&
            number >= schema["exclusiveMaximum"].get<double>())
            return set_error(err, path + " is not below exclusiveMaximum");
    }
    if (value.is_string()) {
        const std::string& string_value = value.get_ref<const std::string&>();
        if (schema.contains("minLength") &&
            string_value.size() < schema["minLength"].get<std::size_t>())
            return set_error(err, path + " is shorter than minLength");
        if (schema.contains("maxLength") &&
            string_value.size() > schema["maxLength"].get<std::size_t>())
            return set_error(err, path + " is longer than maxLength");
        if (schema.contains("pattern")) {
            try {
                const std::regex pattern(schema["pattern"].get<std::string>(),
                                         std::regex::ECMAScript);
                if (!std::regex_search(string_value, pattern))
                    return set_error(err, path + " does not match pattern");
            } catch (const std::regex_error&) {
                return set_error(err, path + " has an invalid pattern schema");
            }
        }
    }
    if (value.is_object()) {
        const json properties = schema.value("properties", json::object());
        if (schema.contains("required")) {
            for (const auto& name : schema["required"])
                if (!value.contains(name.get<std::string>()))
                    return set_error(err, path + " is missing required parameter " + name.get<std::string>());
        }
        // JSON Schema defaults additionalProperties to true. Only an explicit false closes
        // the object; this matters for permissive nested schemas such as {"type":"object"}.
        const bool allow_unknown = !schema.contains("additionalProperties") ||
                                   schema["additionalProperties"] != json(false);
        for (const auto& item : value.items()) {
            if (properties.contains(item.key())) {
                if (!validate_value(item.value(), properties[item.key()], path + "." + item.key(), err)) return false;
            } else if (!allow_unknown) {
                return set_error(err, path + " contains unknown parameter " + item.key());
            } else if (schema.contains("additionalProperties") &&
                       schema["additionalProperties"].is_object() &&
                       !validate_value(item.value(), schema["additionalProperties"],
                                       path + "." + item.key(), err)) {
                return false;
            }
        }
    } else if (value.is_array()) {
        if (schema.contains("minItems") && value.size() < schema["minItems"].get<std::size_t>())
            return set_error(err, path + " has fewer than minItems elements");
        if (schema.contains("maxItems") && value.size() > schema["maxItems"].get<std::size_t>())
            return set_error(err, path + " has more than maxItems elements");
        if (schema.contains("items")) {
            for (size_t i = 0; i < value.size(); ++i)
                if (!validate_value(value[i], schema["items"],
                                    path + "[" + std::to_string(i) + "]", err)) return false;
        }
    }
    return true;
}

const ToolDefinition* offered_tool(const ChatRequest& request, const std::string& name) {
    for (const ToolDefinition& tool : request.tools)
        if (tool.name == name) return &tool;
    return nullptr;
}

bool parse_one_xml_call(const std::string& block, const ChatRequest& request, ToolCall& call,
                        std::string& err) {
    size_t pos = 0;
    while (pos < block.size() && std::isspace(static_cast<unsigned char>(block[pos]))) ++pos;
    if (block.compare(pos, std::char_traits<char>::length(kFunctionOpen), kFunctionOpen) != 0)
        return set_error(err, "tool call is missing <function=...>");
    const size_t name_start = pos + std::char_traits<char>::length(kFunctionOpen);
    const size_t name_end = block.find('>', name_start);
    if (name_end == std::string::npos) return set_error(err, "tool call has an unterminated function name");
    call.name = block.substr(name_start, name_end - name_start);
    if (!safe_protocol_name(call.name))
        return set_error(err, "tool call has an invalid function name");
    const ToolDefinition* tool = offered_tool(request, call.name);
    if (!tool) return set_error(err, "model called unoffered function " + call.name);

    const json& schema = tool->spec["function"]["parameters"];
    const json properties = schema.value("properties", json::object());
    json arguments = json::object();
    pos = name_end + 1;
    while (true) {
        while (pos < block.size() && std::isspace(static_cast<unsigned char>(block[pos]))) ++pos;
        if (block.compare(pos, std::char_traits<char>::length(kFunctionClose), kFunctionClose) == 0) {
            pos += std::char_traits<char>::length(kFunctionClose);
            break;
        }
        if (block.compare(pos, std::char_traits<char>::length(kParameterOpen), kParameterOpen) != 0)
            return set_error(err, "tool call contains malformed text outside parameter tags");
        const size_t key_start = pos + std::char_traits<char>::length(kParameterOpen);
        const size_t key_end = block.find('>', key_start);
        if (key_end == std::string::npos) return set_error(err, "tool call has an unterminated parameter name");
        const std::string key = block.substr(key_start, key_end - key_start);
        if (!safe_protocol_name(key))
            return set_error(err, "tool call has an invalid parameter name");
        if (arguments.contains(key)) return set_error(err, "tool call contains duplicate parameter " + key);
        if (!properties.contains(key)) return set_error(err, "tool call contains unknown parameter " + key);
        const size_t value_start = key_end + 1;
        const size_t value_end = block.find(kParameterClose, value_start);
        if (value_end == std::string::npos) return set_error(err, "tool call has an unterminated parameter " + key);
        std::string value = block.substr(value_start, value_end - value_start);
        // The official template puts one newline on each side of the value. They are protocol
        // delimiters, not part of a string argument; preserve every other byte exactly.
        if (!value.empty() && value.front() == '\n') value.erase(value.begin());
        if (!value.empty() && value.back() == '\n') value.pop_back();
        if (has_protocol_markup(value))
            return set_error(err, "parameter " + key + " contains reserved protocol markup");
        const json& property_schema = properties[key];
        std::string type;
        if (property_schema.contains("type") && property_schema["type"].is_string())
            type = property_schema["type"].get<std::string>();
        json parsed;
        if (type == "string") {
            parsed = value;
        } else {
            if (!parse_scalar_from_text(value, parsed))
                return set_error(err, "parameter " + key + " is not valid JSON for its schema type");
        }
        if (!validate_value(parsed, property_schema, "parameter " + key, err)) return false;
        arguments[key] = std::move(parsed);
        pos = value_end + std::char_traits<char>::length(kParameterClose);
    }
    if (block.substr(pos).find_first_not_of(" \t\r\n") != std::string::npos)
        return set_error(err, "tool call contains text after </function>");
    if (!validate_value(arguments, schema, "arguments for " + call.name, err)) return false;
    call.arguments = arguments.dump();
    call.id.clear();
    return true;
}

}  // namespace

bool parse_chat_request_json(const std::string& body, ChatRequest& request, std::string& err) {
    ChatRequest parsed;
    err.clear();
    json root;
    if (!parse_strict_json(body, root, err, "invalid JSON")) return false;
    if (!root.is_object()) return set_error(err, "request body must be a JSON object");
    if (!root.contains("messages") || !root["messages"].is_array() || root["messages"].empty())
        return set_error(err, "messages must be a non-empty array");
    for (size_t i = 0; i < root["messages"].size(); ++i) {
        ChatMessage message;
        if (!parse_message(root["messages"][i], message, i, err)) return false;
        parsed.messages.push_back(std::move(message));
    }
    if (root.contains("tools")) {
        if (!root["tools"].is_array()) return set_error(err, "tools must be an array");
        std::set<std::string> names;
        for (size_t i = 0; i < root["tools"].size(); ++i) {
            ToolDefinition tool;
            if (!parse_tool_definition(root["tools"][i], tool,
                                       "tools[" + std::to_string(i) + "]", err)) return false;
            if (!names.insert(tool.name).second) return set_error(err, "tools contains duplicate function " + tool.name);
            parsed.tools.push_back(std::move(tool));
        }
    }
    if (root.contains("tool_choice")) {
        const json& choice = root["tool_choice"];
        if (!choice.is_string())
            return set_error(err, "named/object tool_choice is not supported; use auto or none");
        const std::string value = choice.get<std::string>();
        if (value == "auto") parsed.tool_choice = ToolChoiceMode::kAuto;
        else if (value == "none") parsed.tool_choice = ToolChoiceMode::kNone;
        else if (value == "required") return set_error(err, "tool_choice=required is not supported");
        else return set_error(err, "unsupported tool_choice " + value + "; use auto or none");
    }
    if (root.contains("parallel_tool_calls")) {
        if (!root["parallel_tool_calls"].is_boolean())
            return set_error(err, "parallel_tool_calls must be a boolean");
        parsed.parallel_tool_calls = root["parallel_tool_calls"].get<bool>();
        if (!parsed.parallel_tool_calls)
            return set_error(err, "parallel_tool_calls=false is not supported");
    }
    if (parsed.tools.empty() && parsed.tool_choice != ToolChoiceMode::kNone)
        parsed.tool_choice = ToolChoiceMode::kNone;
    std::set<std::string> offered;
    for (const ToolDefinition& tool : parsed.tools) offered.insert(tool.name);
    for (size_t i = 0; i < parsed.messages.size(); ++i) {
        for (const ToolCall& call : parsed.messages[i].tool_calls) {
            if (!offered.count(call.name))
                return set_error(err, "messages[" + std::to_string(i) + "] references unoffered function " + call.name);
            const ToolDefinition* tool = offered_tool(parsed, call.name);
            json arguments;
            if (!parse_strict_json(call.arguments, arguments, err,
                                   "messages[" + std::to_string(i) +
                                       "].tool_calls arguments")) return false;
            const json properties =
                tool->spec["function"]["parameters"].value("properties", json::object());
            for (const auto& argument : arguments.items()) {
                if (!safe_protocol_name(argument.key()))
                    return set_error(err, "messages[" + std::to_string(i) +
                                              "].tool_calls contains an unsafe parameter name");
                if (!properties.contains(argument.key()))
                    return set_error(err, "messages[" + std::to_string(i) +
                                              "].tool_calls contains unknown parameter " +
                                              argument.key());
            }
            if (!validate_value(arguments, tool->spec["function"]["parameters"],
                                "messages[" + std::to_string(i) + "].tool_calls arguments", err)) return false;
        }
    }
    if (!resolve_history(parsed, err)) return false;
    request = std::move(parsed);
    return true;
}

std::string apply_qwen36_tools_template(const ChatRequest& request, bool enable_thinking) {
    std::ostringstream out;
    size_t first_message = 0;
    if (!request.tools.empty() && request.tool_choice == ToolChoiceMode::kAuto) {
        out << kImStart << "system\n" << kToolInstructions;
        for (const ToolDefinition& tool : request.tools)
            out << '\n' << qwen_template_json(tool.spec);
        out << kToolInstructionsTail;
        if (!request.messages.empty() && request.messages[0].role == "system") {
            const std::string system_content = trim_copy(request.messages[0].content);
            if (!system_content.empty()) out << "\n\n" << system_content;
        }
        out << kImEnd << '\n';
        if (!request.messages.empty() && request.messages[0].role == "system") first_message = 1;
    }
    size_t last_user = request.messages.size();
    for (size_t i = request.messages.size(); i > 0; --i) {
        if (request.messages[i - 1].role == "user") {
            last_user = i - 1;
            break;
        }
    }
    for (size_t i = first_message; i < request.messages.size(); ++i) {
        const ChatMessage& message = request.messages[i];
        if (message.role == "tool") {
            out << kImStart << "user";
            do {
                out << '\n' << kToolResponseOpen << '\n' << trim_copy(request.messages[i].content)
                    << '\n' << kToolResponseClose;
                ++i;
            } while (i < request.messages.size() && request.messages[i].role == "tool");
            out << kImEnd << '\n';
            --i;
            continue;
        }
        out << kImStart << message.role << '\n';
        if (message.role == "assistant") {
            std::string content = trim_copy(message.content);
            std::string reasoning = trim_copy(message.reasoning_content);
            // Match the pinned tokenizer template's compatibility path for clients which put
            // a previous turn's reasoning and answer together in content instead of sending
            // reasoning_content separately.
            if (reasoning.empty()) {
                const size_t first_close = content.find(kThinkClose);
                if (first_close != std::string::npos) {
                    std::string embedded_reasoning = content.substr(0, first_close);
                    const size_t last_open = embedded_reasoning.rfind(kThinkOpen);
                    if (last_open != std::string::npos)
                        embedded_reasoning.erase(0, last_open +
                                                        std::char_traits<char>::length(kThinkOpen));
                    reasoning = trim_copy(std::move(embedded_reasoning));
                    const size_t last_close = content.rfind(kThinkClose);
                    content = trim_copy(content.substr(last_close +
                                                       std::char_traits<char>::length(kThinkClose)));
                }
            }
            // The pinned template preserves reasoning only for assistant/tool steps after the
            // latest real user query. Older chain-of-thought is intentionally not replayed.
            if (i > last_user)
                out << kThinkOpen << '\n' << reasoning << '\n' << kThinkClose << "\n\n";
            if (!message.content_is_null) out << content;
            for (size_t j = 0; j < message.tool_calls.size(); ++j) {
                if (j == 0 && !content.empty()) out << "\n\n";
                else if (j > 0) out << '\n';
                out << render_tool_call(message.tool_calls[j]);
            }
        } else {
            out << trim_copy(message.content);
        }
        out << kImEnd << '\n';
    }
    out << kImStart << "assistant\n";
    if (enable_thinking) out << kThinkOpen << '\n';
    else out << kThinkOpen << "\n\n" << kThinkClose << "\n\n";
    return out.str();
}

ParsedToolOutput parse_qwen36_tool_output(const std::string& raw, bool enable_thinking,
                                          const ChatRequest& request) {
    ParsedToolOutput out;
    std::string remaining = raw;
    if (enable_thinking) {
        const size_t open = remaining.find(kThinkOpen);
        if (open != std::string::npos) {
            if (remaining.substr(0, open).find_first_not_of(" \t\r\n") != std::string::npos) {
                return fail_tool_output(std::move(out), "unexpected text before <think>");
            }
            const size_t close = remaining.find(kThinkClose, open + std::char_traits<char>::length(kThinkOpen));
            if (close == std::string::npos) {
                return fail_tool_output(std::move(out), "unterminated <think> block");
            }
            out.reasoning_content = remaining.substr(open + std::char_traits<char>::length(kThinkOpen),
                                                     close - open - std::char_traits<char>::length(kThinkOpen));
            trim_leading_ws(out.reasoning_content);
            trim_trailing_ws(out.reasoning_content);
            remaining.erase(0, close + std::char_traits<char>::length(kThinkClose));
            trim_leading_ws(remaining);
        } else {
            // With enable_thinking=true the generation prompt itself ends in "<think>\n", so the
            // generated suffix normally begins with reasoning text and only emits </think>.
            const size_t close = remaining.find(kThinkClose);
            if (close != std::string::npos) {
                out.reasoning_content = remaining.substr(0, close);
                trim_leading_ws(out.reasoning_content);
                trim_trailing_ws(out.reasoning_content);
                remaining.erase(0, close + std::char_traits<char>::length(kThinkClose));
                trim_leading_ws(remaining);
            } else {
                return fail_tool_output(std::move(out), "unterminated implicit <think> block");
            }
        }
    } else {
        const size_t close = remaining.find(kThinkClose);
        if (close != std::string::npos) {
            const size_t open = remaining.rfind(kThinkOpen, close);
            if (open == std::string::npos) {
                return fail_tool_output(std::move(out), "unmatched </think> marker");
            }
            remaining.erase(open, close + std::char_traits<char>::length(kThinkClose) - open);
            trim_leading_ws(remaining);
        }
    }

    if (has_protocol_markup(out.reasoning_content)) {
        return fail_tool_output(std::move(out), "reasoning contains reserved protocol markup");
    }

    const size_t first_call = remaining.find(kToolCallOpen);
    if (first_call == std::string::npos) {
        out.content = remaining;
        strip_trailing_im_end(out.content);
        if (has_protocol_markup(out.content)) {
            return fail_tool_output(std::move(out), "malformed tool-call markup");
        }
        return out;
    }
    if (request.tools.empty() || request.tool_choice == ToolChoiceMode::kNone) {
        return fail_tool_output(std::move(out), "model emitted a tool call when tools were unavailable");
    }
    out.content = remaining.substr(0, first_call);
    trim_trailing_ws(out.content);
    if (has_protocol_markup(out.content)) {
        return fail_tool_output(std::move(out),
                                "assistant content contains reserved protocol markup");
    }
    size_t pos = first_call;
    while (pos < remaining.size()) {
        while (pos < remaining.size() && std::isspace(static_cast<unsigned char>(remaining[pos]))) ++pos;
        if (remaining.compare(pos, std::char_traits<char>::length(kImEnd), kImEnd) == 0) {
            pos += std::char_traits<char>::length(kImEnd);
            while (pos < remaining.size() && std::isspace(static_cast<unsigned char>(remaining[pos]))) ++pos;
            if (pos != remaining.size()) {
                return fail_tool_output(std::move(out), "text follows terminal <|im_end|>");
            }
            break;
        }
        if (remaining.compare(pos, std::char_traits<char>::length(kToolCallOpen), kToolCallOpen) != 0) {
            return fail_tool_output(std::move(out), "text or malformed markup appears after a tool call");
        }
        const size_t body_start = pos + std::char_traits<char>::length(kToolCallOpen);
        const size_t end = remaining.find(kToolCallClose, body_start);
        if (end == std::string::npos) {
            return fail_tool_output(std::move(out), "unterminated <tool_call> block");
        }
        ToolCall call;
        if (!parse_one_xml_call(remaining.substr(body_start, end - body_start), request, call, out.error)) {
            const std::string error = out.error;
            return fail_tool_output(std::move(out), error);
        }
        out.tool_calls.push_back(std::move(call));
        pos = end + std::char_traits<char>::length(kToolCallClose);
    }
    if (!request.parallel_tool_calls && out.tool_calls.size() > 1) {
        return fail_tool_output(std::move(out),
                                "model emitted parallel tool calls when they were disabled");
    }
    return out;
}

}  // namespace sparkinfer_server
