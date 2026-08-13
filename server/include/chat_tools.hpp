#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
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

// Decode-control fields that don't influence prompt construction (unlike ChatRequest's
// tools/response_format, which do) -- extracted here rather than kept private to
// sparkinfer_server.cpp so parsing/validation has a unit-test seam without a running server/GPU.
struct RequestControls {
    bool stream = false;
    bool include_usage = false;
    int max_tokens = 256;
    std::vector<std::string> stop;
    // <= 0 (default) is plain greedy argmax, byte-identical to pre-sampling behavior.
    float temperature = 0.f;
    uint64_t seed = 0;       // only meaningful when seed_set
    bool seed_set = false;   // client explicitly supplied `seed`; false => caller should generate one
};

bool parse_request_controls(const std::string& body, RequestControls& out, std::string& err);

// Pure decision function for the DFlash+temperature-sampling incompatibility (kept separate from
// getenv() so it's unit-testable without a process-wide env var): true => the request should be
// rejected with 400. temperature<=0 is always accepted regardless of dflash_env_on.
bool should_reject_dflash_temperature(bool dflash_env_on, float temperature);

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
