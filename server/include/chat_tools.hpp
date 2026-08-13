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

struct ChatRequest {
    std::vector<ChatMessage> messages;
    std::vector<ToolDefinition> tools;
    ToolChoiceMode tool_choice = ToolChoiceMode::kAuto;
    bool parallel_tool_calls = true;
};

struct ParsedToolOutput {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string error;
};

bool parse_chat_request_json(const std::string& body, ChatRequest& request, std::string& err);

std::string apply_qwen36_tools_template(const ChatRequest& request,
                                        bool enable_thinking = false);

ParsedToolOutput parse_qwen36_tool_output(const std::string& raw,
                                          bool enable_thinking,
                                          const ChatRequest& request);

}  // namespace sparkinfer_server
