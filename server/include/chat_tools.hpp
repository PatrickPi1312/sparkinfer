#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace sparkinfer_server {

struct ToolCall {
    std::string id;
    std::string name;
    // OpenAI wire representation: a compact JSON object encoded as a string.
    std::string arguments;
};

struct ToolDefinition {
    std::string name;
    // The complete top-level OpenAI tool object ({"type":"function","function":{...}}).
    nlohmann::json spec;
};

enum class ToolChoiceMode {
    kAuto,
    kNone,
};

struct ChatMessage {
    std::string role;
    std::string content;
    bool content_is_null = false;
    std::string reasoning_content;
    std::string name;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
};

enum class ResponseFormatType {
    kText,
    kJsonObject,
    kJsonSchema,
};

struct ResponseFormat {
    ResponseFormatType type = ResponseFormatType::kText;
    std::string schema_name;   // json_schema.name -- steering text / logging only
    nlohmann::json schema;     // json_schema.schema -- empty/ignored unless type == kJsonSchema
    // json_schema.strict -- parsed and stored, but a documented v1 no-op: this backend has no
    // constrained-decoding mechanism, so it cannot honor a real conformance guarantee either way.
    bool strict = false;
};

struct ChatRequest {
    std::vector<ChatMessage> messages;
    std::vector<ToolDefinition> tools;
    ToolChoiceMode tool_choice = ToolChoiceMode::kAuto;
    bool parallel_tool_calls = true;
    ResponseFormat response_format;
};

struct ParsedToolOutput {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string error;
};

bool parse_chat_request_json(const std::string& body, ChatRequest& request, std::string& err);

// Best-effort validation only -- there is no constrained decoding in this backend, so this
// cannot guarantee the model's output actually conforms; it just checks after the fact.
// format.type == kText always returns true (no-op).
bool validate_response_format(const std::string& content, const ResponseFormat& format,
                              std::string& err);

std::string apply_qwen36_tools_template(const ChatRequest& request,
                                        bool enable_thinking = false);

ParsedToolOutput parse_qwen36_tool_output(const std::string& raw,
                                          bool enable_thinking,
                                          const ChatRequest& request);

}  // namespace sparkinfer_server
