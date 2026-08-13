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
    // Directly renders + tokenizes an already-parsed/validated ChatRequest, skipping the JSON
    // body parse step encode_chat_request does. Used to build a retry prompt (e.g. a
    // response_format correction turn appended to the original request) without round-tripping
    // through a serialized JSON body. Returns an empty vector if the tokenizer isn't loaded.
    std::vector<int> encode_augmented(const ChatRequest& request, bool enable_thinking) const;
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
    // Qwen (non-museglimmer) starts in Phase::kInThink directly (the prompt already primes
    // "<think>\n"), but a generation can still defensively repeat the opening marker -- see
    // parse_assistant_output's identical handling for the non-streaming path. Checked once,
    // on the first non-empty data seen in kInThink, so a literal "<think>" appearing later in
    // genuine reasoning text is never mistaken for the marker.
    bool think_open_checked_ = false;

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

// Detects OpenAI-style client-supplied `stop` sequences in a live, incrementally-decoded token
// stream. Generalizes the single-compile-time-marker holdback pattern ThinkingStreamSplitter
// uses for <|im_end|>/<think> to an arbitrary list of up to 4 dynamic strings: text that could
// still be an in-progress prefix of any stop string is buffered (never returned) until a later
// feed() either completes it into a full match (matched() becomes true; the held-back bytes and
// the matched text itself are permanently dropped, never returned) or disproves it (the
// held-back bytes are released as ordinary text). An empty stop list makes feed() a cheap
// passthrough, so it's safe to always construct one per request.
class StopSequenceFilter {
public:
    explicit StopSequenceFilter(std::vector<std::string> stops);

    // Feeds the next raw decoded delta. Returns the subset of text now known not to be part of
    // a stop match, safe to forward downstream immediately. Once matched() is true, feed() must
    // not be called again.
    std::string feed(const std::string& piece);

    bool matched() const { return matched_; }

private:
    std::vector<std::string> stops_;
    std::string carry_;
    bool matched_ = false;
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
