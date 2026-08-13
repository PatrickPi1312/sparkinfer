#include "chat_tokenizer.hpp"

#include <cstdio>
#include <string>
#include <utility>

namespace {

#define CHECK(expr)                                                                            \
    do {                                                                                       \
        if (!(expr)) {                                                                         \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr);             \
            return false;                                                                      \
        }                                                                                      \
    } while (0)

bool test_thinking_prompt_and_nonstream_parser() {
    sparkinfer_server::ChatRequest request;
    sparkinfer_server::ChatMessage message;
    message.role = "user";
    message.content = "Explain briefly.";
    request.messages.push_back(std::move(message));
    const std::string prompt = sparkinfer_server::apply_qwen36_tools_template(request, true);
    const std::string suffix = "<|im_start|>assistant\n<think>\n";
    CHECK(prompt.size() >= suffix.size());
    CHECK(prompt.compare(prompt.size() - suffix.size(), suffix.size(), suffix) == 0);

    const auto parsed = sparkinfer_server::parse_assistant_output(
        "private reasoning\n</think>\n\nPublic answer.<|im_end|>", true, false);
    CHECK(parsed.error.empty());
    CHECK(parsed.reasoning_content == "private reasoning");
    CHECK(parsed.content == "Public answer.");
    CHECK(parsed.tool_calls.empty());

    // Defensively tolerate a model that repeats the already-prefilled opening marker.
    const auto repeated = sparkinfer_server::parse_assistant_output(
        "<think>\nreason\n</think>\nanswer", true, false);
    CHECK(repeated.reasoning_content == "reason");
    CHECK(repeated.content == "answer");

    // Same tolerance when the repeated marker isn't the very first byte (e.g. a stray leading
    // token before it) -- the marker itself must never leak into reasoning_content.
    const auto repeated_not_at_start = sparkinfer_server::parse_assistant_output(
        " <think>\nreason\n</think>\nanswer", true, false);
    CHECK(repeated_not_at_start.reasoning_content == "reason");
    CHECK(repeated_not_at_start.content == "answer");
    CHECK(repeated_not_at_start.reasoning_content.find("<think>") == std::string::npos);
    return true;
}

bool test_thinking_stream_boundaries() {
    sparkinfer_server::ThinkingStreamSplitter splitter(true, false);
    std::string reasoning;
    std::string content;
    for (const char* piece : {"private ", "reasoning\n</thi", "nk>\n\nPublic ",
                              "answer.<|im_", "end|>"}) {
        const auto delta = splitter.feed(piece);
        reasoning += delta.reasoning_content;
        content += delta.content;
    }
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    reasoning += tail.reasoning_content;
    content += tail.content;
    CHECK(reasoning == "private reasoning\n");
    CHECK(content == "Public answer.");
    CHECK(content.find("<think>") == std::string::npos);
    CHECK(content.find("</think>") == std::string::npos);
    CHECK(content.find("<|im_end|>") == std::string::npos);
    return true;
}

bool test_thinking_stream_repeated_opening_marker() {
    // Streaming starts directly in the "inside <think>" phase (the prompt already primed
    // "<think>\n"), but must still tolerate -- and strip -- a defensively repeated opening
    // marker the same way the non-streaming parser does, rather than leaking it into
    // reasoning_content.
    sparkinfer_server::ThinkingStreamSplitter splitter(true, false);
    std::string reasoning;
    std::string content;
    for (const char* piece : {"<thi", "nk>\nreason\n</thi", "nk>\nanswer"}) {
        const auto delta = splitter.feed(piece);
        reasoning += delta.reasoning_content;
        content += delta.content;
    }
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    reasoning += tail.reasoning_content;
    content += tail.content;
    // Streaming reasoning_content chunks aren't leading-whitespace-trimmed the way the
    // non-streaming parser's final string is (pre-existing, orthogonal to this fix) -- the
    // marker itself must simply never appear in the output.
    CHECK(reasoning == "\nreason\n");
    CHECK(content == "answer");
    CHECK(reasoning.find("<think>") == std::string::npos);
    return true;
}

bool test_nonthinking_stream_hides_terminal_marker() {
    sparkinfer_server::ThinkingStreamSplitter splitter(false, false);
    std::string content;
    for (const char* piece : {"hello", "<|im_", "end|>"})
        content += splitter.feed(piece).content;
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    content += tail.content;
    CHECK(content == "hello");
    return true;
}

bool test_muse_rejects_all_tool_protocol_history() {
    sparkinfer_server::ChatRequest parsed;
    std::string error;
    const std::string request = R"JSON({
      "messages":[
        {"role":"assistant","content":null,"tool_calls":[{
          "id":"call_1","type":"function",
          "function":{"name":"lookup","arguments":"{\"query\":\"x\"}"}
        }]},
        {"role":"tool","tool_call_id":"call_1","content":"ok"},
        {"role":"user","content":"continue"}
      ],
      "tools":[{"type":"function","function":{
        "name":"lookup",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}}}
      }}],
      "tool_choice":"none"
    })JSON";
    CHECK(sparkinfer_server::parse_chat_request_json(request, parsed, error));
    CHECK(!sparkinfer_server::validate_chat_request_model_support(parsed, true, error));
    CHECK(error.find("supported only for Qwen3.6") != std::string::npos);
    error.clear();
    CHECK(sparkinfer_server::validate_chat_request_model_support(parsed, false, error));
    return true;
}

bool test_tool_choice_none_still_uses_strict_output_parser() {
    const std::string request_json = R"JSON({
      "messages":[{"role":"user","content":"Do not call tools."}],
      "tools":[{"type":"function","function":{
        "name":"terminal",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}}}
      }}],
      "tool_choice":"none"
    })JSON";
    sparkinfer_server::ChatRequest request;
    std::string error;
    CHECK(sparkinfer_server::parse_chat_request_json(request_json, request, error));
    const auto forbidden = sparkinfer_server::parse_assistant_output(
        "<tool_call>\n<function=terminal>\n<parameter=command>\nid\n</parameter>\n"
        "</function>\n</tool_call>",
        false, false, &request);
    CHECK(!forbidden.error.empty());
    CHECK(forbidden.content.empty());
    CHECK(forbidden.tool_calls.empty());

    const auto answer = sparkinfer_server::parse_assistant_output(
        "No tool used.<|im_end|>", false, false, &request);
    CHECK(answer.error.empty());
    CHECK(answer.content == "No tool used.");
    return true;
}

}  // namespace

int main() {
    if (!test_thinking_prompt_and_nonstream_parser()) return 1;
    if (!test_thinking_stream_boundaries()) return 1;
    if (!test_thinking_stream_repeated_opening_marker()) return 1;
    if (!test_nonthinking_stream_hides_terminal_marker()) return 1;
    if (!test_muse_rejects_all_tool_protocol_history()) return 1;
    if (!test_tool_choice_none_still_uses_strict_output_parser()) return 1;
    std::printf("chat_tokenizer_test: OK\n");
    return 0;
}
