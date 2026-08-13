#pragma once

#include "chat_tools.hpp"

#include <memory>
#include <string>
#include <vector>

namespace sparkinfer_server {

// HuggingFace tokenizer.json + Qwen3.6 chat template.
class ChatTokenizer {
public:
    ChatTokenizer();
    ~ChatTokenizer();

    bool load(const std::string& tokenizer_json_path, std::string& err);

    // Selects the chat-template/parsing dialect: Qwen3.6's <think> tags (default) vs Muse
    // Glimmer's harmony-style <|start|>/<|message|>/<|eot|>/<|eom|> segments. Call once after
    // the model's architecture is known (ModelEngine::is_museglimmer()), before the first request.
    void set_museglimmer(bool on);

    bool encode_chat_request(const std::string& request_json, std::vector<int>& ids, bool enable_thinking,
                             std::string& err, ChatRequest* parsed_request = nullptr) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_delta(std::vector<int>& acc, int new_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ParsedAssistantOutput {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string error;
};

// Incrementally routes decoded text into reasoning vs answer for SSE streaming.
class ThinkingStreamSplitter {
public:
    explicit ThinkingStreamSplitter(bool enable_thinking, bool museglimmer = false);

    struct Delta {
        std::string reasoning_content;
        std::string content;
    };
    Delta feed(const std::string& piece);
    void finish(Delta& tail);

private:
    bool enable_thinking_ = false;
    enum class Phase { kBeforeThink, kInThink, kInAnswer } phase_ = Phase::kBeforeThink;
    std::string carry_;
    std::string prefix_buffer_;

    // Muse Glimmer (harmony-style) state: segments are <|start|>{header}<|message|>{body}
    // then <|eom|> (more segments follow, same assistant turn) or <|eot|> (turn done). The
    // header names the recipient -- "assistant to=self" routes body to reasoning_content,
    // anything else (typically "assistant to=user") routes it to content.
    bool museglimmer_ = false;
    enum class MgPhase { kHeader, kAwaitStart, kBody } mg_phase_ = MgPhase::kHeader;
    // Primed with "assistant" -- the prompt itself already emits "<|start|>assistant" before
    // handing off to the model, so the first segment's generated header text picks up right
    // after that word (e.g. " to=self"), unlike later segments which regenerate it in full.
    std::string mg_header_ = "assistant";
    bool mg_is_self_ = false;
    bool mg_done_ = false;
};

bool parse_chat_messages(const std::string& request_json, std::vector<ChatMessage>& messages, std::string& err);
bool parse_enable_thinking(const std::string& request_json, bool default_value = false);
bool validate_chat_request_model_support(const ChatRequest& request, bool museglimmer,
                                         std::string& err);
std::string apply_qwen36_chat_template(const std::vector<ChatMessage>& messages, bool enable_thinking = false);
std::string apply_museglimmer_chat_template(const std::vector<ChatMessage>& messages,
                                            const std::string& reasoning_strength = "high");
ParsedAssistantOutput parse_assistant_output(const std::string& raw, bool enable_thinking,
                                             bool museglimmer = false,
                                             const ChatRequest* request = nullptr);

}  // namespace sparkinfer_server
