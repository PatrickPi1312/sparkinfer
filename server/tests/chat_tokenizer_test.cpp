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

}  // namespace

int main() {
    if (!test_thinking_prompt_and_nonstream_parser()) return 1;
    if (!test_thinking_stream_boundaries()) return 1;
    if (!test_nonthinking_stream_hides_terminal_marker()) return 1;
    std::printf("chat_tokenizer_test: OK\n");
    return 0;
}
