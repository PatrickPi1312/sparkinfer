#include "chat_tokenizer.hpp"
#include "model_engine.hpp"

#include <nlohmann/json.hpp>

// Do not define CPPHTTPLIB_OPENSSL_SUPPORT — even `= 0` enables OpenSSL in httplib.
#include "../third_party/httplib.h"

#include <atomic>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// Per-request output cap. Independent of context length (checked separately against the
// live engine max_seq below) -- this just bounds how long a single request may run.
int max_output_tokens() {
    static int v = []{
        const char* e = getenv("SPARKINFER_MAX_OUTPUT_TOKENS");
        int n = e ? atoi(e) : 4096;
        return n > 0 ? n : 4096;
    }();
    return v;
}

std::string g_api_key;
std::string g_model_name = "qwen3.6-35b-a3b";
sparkinfer_server::ChatTokenizer g_tokenizer;
const auto g_start_time = std::chrono::steady_clock::now();

// Request/error metrics (GET /metrics). Counters only -- no per-request content is retained.
std::atomic<uint64_t> g_requests_total{0};
std::atomic<uint64_t> g_requests_streaming{0};
std::atomic<uint64_t> g_requests_ok{0};
std::atomic<uint64_t> g_requests_client_error{0};   // 4xx
std::atomic<uint64_t> g_requests_overloaded{0};     // 429
std::atomic<uint64_t> g_requests_alloc_failed{0};   // 503 -- real device OOM, not capacity (#779)
std::atomic<uint64_t> g_requests_timeout{0};
std::atomic<uint64_t> g_requests_cancelled{0};
std::atomic<uint64_t> g_requests_server_error{0};   // 5xx
// 502 -- the model produced tool-call output that failed schema/markup validation for a reason
// other than truncation. Distinct from server_error so monitoring doesn't conflate model-output
// quality variance with a real infrastructure fault -- the exact conflation #779 already
// documented once for overloaded (429) vs alloc_failed (503).
std::atomic<uint64_t> g_requests_invalid_tool_output{0};
std::atomic<uint64_t> g_prompt_tokens_total{0};
std::atomic<uint64_t> g_completion_tokens_total{0};

std::atomic<bool> g_shutdown_requested{false};
void on_shutdown_signal(int) { g_shutdown_requested = true; }

std::string repo_root() {
    const char* env = getenv("SPARKINFER_ROOT");
    if (env && *env) return env;
    return ".";
}

std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                else o << c;
        }
    }
    return o.str();
}

std::string random_id(const char* prefix = "chatcmpl-") {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream ss;
    ss << prefix << std::hex << dist(rng);
    return ss.str();
}

bool auth_ok(const httplib::Request& req) {
    if (g_api_key.empty()) return true;
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) return false;
    const std::string prefix = "Bearer ";
    return it->second.size() > prefix.size() &&
           it->second.compare(0, prefix.size(), prefix) == 0 &&
           it->second.substr(prefix.size()) == g_api_key;
}

bool encode_messages(const std::string& body, std::vector<int>& ids, bool enable_thinking,
                     std::string& err, sparkinfer_server::ChatRequest* request = nullptr) {
    return g_tokenizer.encode_chat_request(body, ids, enable_thinking, err, request);
}

struct RequestControls {
    bool stream = false;
    bool include_usage = false;
    int max_tokens = 256;
    std::vector<std::string> stop;
};

bool parse_request_controls(const std::string& body, RequestControls& out, std::string& err) {
    const auto root = nlohmann::json::parse(body, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        err = "request body must be a JSON object";
        return false;
    }
    if (root.contains("stream")) {
        if (!root["stream"].is_boolean()) {
            err = "stream must be a boolean";
            return false;
        }
        out.stream = root["stream"].get<bool>();
    }
    const char* max_key = root.contains("max_completion_tokens") ? "max_completion_tokens" : "max_tokens";
    if (root.contains(max_key)) {
        const auto& value = root[max_key];
        if (!value.is_number_integer() && !value.is_number_unsigned()) {
            err = std::string(max_key) + " must be a positive integer";
            return false;
        }
        try {
            const auto requested = value.get<long long>();
            if (requested <= 0 || requested > std::numeric_limits<int>::max()) {
                err = std::string(max_key) + " is outside the supported range";
                return false;
            }
            out.max_tokens = static_cast<int>(requested);
        } catch (const nlohmann::json::exception&) {
            err = std::string(max_key) + " is outside the supported range";
            return false;
        }
    }
    if (root.contains("stop") && !root["stop"].is_null()) {
        const auto& value = root["stop"];
        std::vector<std::string> stops;
        if (value.is_string()) {
            stops.push_back(value.get<std::string>());
        } else if (value.is_array()) {
            for (const auto& item : value) {
                if (!item.is_string()) {
                    err = "stop entries must be strings";
                    return false;
                }
                stops.push_back(item.get<std::string>());
            }
        } else {
            err = "stop must be a string or an array of strings";
            return false;
        }
        if (stops.size() > 4) {
            err = "stop supports at most 4 strings";
            return false;
        }
        for (const auto& s : stops) {
            if (s.empty()) {
                err = "stop entries must be non-empty strings";
                return false;
            }
        }
        out.stop = std::move(stops);
    }
    if (root.contains("stream_options")) {
        if (!root["stream_options"].is_object()) {
            err = "stream_options must be an object";
            return false;
        }
        const auto& opts = root["stream_options"];
        if (opts.contains("include_usage")) {
            if (!opts["include_usage"].is_boolean()) {
                err = "stream_options.include_usage must be a boolean";
                return false;
            }
            out.include_usage = opts["include_usage"].get<bool>();
        }
    }
    return true;
}

// One-shot scan over a complete decoded string: is there a stop match anywhere, and where.
// Unlike StopSequenceFilter, this needs no holdback machinery -- it's only ever called once
// generation has already stopped and the full text is available in hand.
bool find_stop_match(const std::string& text, const std::vector<std::string>& stops, size_t& pos) {
    bool found = false;
    for (const auto& s : stops) {
        const size_t p = text.find(s);
        if (p != std::string::npos && (!found || p < pos)) {
            pos = p;
            found = true;
        }
    }
    return found;
}

nlohmann::json stream_chunk_base(const std::string& cid, long long created) {
    return {{"id", cid}, {"object", "chat.completion.chunk"}, {"created", created},
            {"model", g_model_name}};
}

bool write_sse_json(httplib::DataSink& sink, const nlohmann::json& value) {
    const std::string event = "data: " + value.dump() + "\n\n";
    return sink.write(event.c_str(), event.size());
}

bool write_stream_role(httplib::DataSink& sink, const std::string& cid, long long created) {
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array({{{"index", 0},
                                                {"delta", {{"role", "assistant"},
                                                           {"content", nullptr}}},
                                                {"finish_reason", nullptr}}});
    return write_sse_json(sink, chunk);
}

bool write_stream_delta(httplib::DataSink& sink, const std::string& cid, long long created, const std::string& field,
                        const std::string& piece) {
    if (piece.empty()) return true;
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array({{{"index", 0},
                                                {"delta", {{field, piece}}},
                                                {"finish_reason", nullptr}}});
    return write_sse_json(sink, chunk);
}

bool write_stream_tool_call(httplib::DataSink& sink, const std::string& cid, long long created,
                            size_t index, const sparkinfer_server::ToolCall& call) {
    auto chunk = stream_chunk_base(cid, created);
    nlohmann::json delta_call = {{"index", index}, {"id", call.id}, {"type", "function"},
                                {"function", {{"name", call.name},
                                              {"arguments", call.arguments}}}};
    chunk["choices"] = nlohmann::json::array({{{"index", 0},
                                                {"delta", {{"tool_calls",
                                                            nlohmann::json::array({delta_call})}}},
                                                {"finish_reason", nullptr}}});
    return write_sse_json(sink, chunk);
}

bool write_stream_finish(httplib::DataSink& sink, const std::string& cid, long long created,
                         const std::string& reason) {
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array({{{"index", 0},
                                                {"delta", nlohmann::json::object()},
                                                {"finish_reason", reason}}});
    return write_sse_json(sink, chunk);
}

bool write_stream_usage(httplib::DataSink& sink, const std::string& cid, long long created,
                        int prompt_tokens, int completion_tokens, double ttft_ms,
                        double generation_ms, double decode_tps) {
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array();
    nlohmann::json usage = {{"prompt_tokens", prompt_tokens},
                            {"completion_tokens", completion_tokens},
                            {"total_tokens", prompt_tokens + completion_tokens}};
    if (ttft_ms >= 0.0) usage["ttft_ms"] = ttft_ms;
    if (generation_ms >= 0.0) usage["generation_ms"] = generation_ms;
    if (decode_tps >= 0.0) usage["decode_tps"] = decode_tps;
    chunk["usage"] = std::move(usage);
    return write_sse_json(sink, chunk);
}

bool write_stream_done(httplib::DataSink& sink) {
    static const std::string done = "data: [DONE]\n\n";
    return sink.write(done.c_str(), done.size());
}

bool decode_ids(const std::vector<int>& ids, std::string& text, std::string& err) {
    text = g_tokenizer.decode(ids);
    if (text.empty() && !ids.empty()) {
        err = "detokenize returned empty text";
        return false;
    }
    return true;
}

std::vector<int> load_prefix_token_ids() {
    std::vector<int> out;
    if (const char* csv = getenv("SPARKINFER_SERVER_PREFIX_TOKEN_IDS")) {
        const char* p = csv;
        while (*p) {
            char* end = nullptr;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            out.push_back((int)v);
            p = end;
            while (*p == ',' || *p == ' ') p++;
        }
        return out;
    }
    const char* path = getenv("SPARKINFER_SERVER_PREFIX_TOKEN_FILE");
    if (!path || !*path) return out;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[sparkinfer-server] WARN: cannot open prefix token file %s\n", path);
        return out;
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (size_t i = 0; i < s.size();) {
        i = s.find_first_of("0123456789", i);
        if (i == std::string::npos) break;
        out.push_back(atoi(s.c_str() + i));
        i = s.find_first_not_of("0123456789", i);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_path;
    std::string tokenizer_json;
    int ctx = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* flag) { return a == flag && i + 1 < argc; };
        if (need("-m") || need("--model")) model_path = argv[++i];
        else if (need("--host")) host = argv[++i];
        else if (need("--port")) port = atoi(argv[++i]);
        else if (need("--ctx")) ctx = atoi(argv[++i]);
        else if (need("--api-key")) g_api_key = argv[++i];
        else if (need("--tokenizer")) tokenizer_json = argv[++i];
        else if (need("--model-name")) g_model_name = argv[++i];
        else if (a == "-h" || a == "--help") {
            fprintf(stderr,
                    "usage: %s -m model.gguf [--host 127.0.0.1] [--port 8080] [--ctx N] "
                    "[--tokenizer path/to/tokenizer.json] [--model-name ID] [--api-key KEY]\n",
                    argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "error: -m model.gguf is required\n");
        return 2;
    }

    const std::string root = repo_root();
    std::string tok_path = tokenizer_json.empty() ? root + "/models/tokenizer.json" : tokenizer_json;
    std::string tok_err;
    if (!g_tokenizer.load(tok_path, tok_err)) {
        fprintf(stderr, "[sparkinfer-server] %s\n", tok_err.c_str());
        return 1;
    }

    sparkinfer_server::ModelEngine engine;
    if (!engine.load(model_path, ctx > 0 ? ctx : 0)) return 1;
    g_tokenizer.set_museglimmer(engine.is_museglimmer());

    const std::vector<int> prefix_ids = load_prefix_token_ids();
    if (!prefix_ids.empty()) {
        engine.set_prefix_tokens(prefix_ids);
        fprintf(stderr, "[sparkinfer-server] prefix cache: %zu tokens (batched prefill per request)\n",
                prefix_ids.size());
    }

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [&engine](const httplib::Request&, httplib::Response& res) {
        std::ostringstream body;
        body << "{\"object\":\"list\",\"data\":[{\"id\":\"" << g_model_name
             << "\",\"object\":\"model\",\"owned_by\":\"sparkinfer\",\"context_length\":"
             << engine.max_seq() << "}]}";
        res.set_content(body.str(), "application/json");
    });

    svr.Get("/v1/info", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        std::ostringstream body;
        body << "{\"model\":\"" << g_model_name << "\",\"max_context\":" << engine.max_seq()
             << ",\"max_output_tokens\":" << max_output_tokens() << "}";
        res.set_content(body.str(), "application/json");
    });

    // Live occupancy -- lets an orchestrator (or a human) see whether this worker has room
    // before routing a request to it, and is what a fleet-level capacity/load-balancing layer
    // would poll. Single-process only: this reports this server's own queue, not fleet-wide
    // capacity across other nodes.
    svr.Get("/v1/capacity", [&engine](const httplib::Request&, httplib::Response& res) {
        std::ostringstream body;
        const int cap = engine.max_queue_depth();
        body << "{\"active_requests\":" << engine.active_requests()
             << ",\"free_kv_blocks\":" << engine.free_kv_blocks()
             << ",\"max_queue_depth\":" << cap
             << ",\"accepting_requests\":" << (g_shutdown_requested.load() ? "false" : "true") << "}";
        res.set_content(body.str(), "application/json");
    });

    svr.Get("/metrics", [&engine](const httplib::Request&, httplib::Response& res) {
        const double uptime_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - g_start_time).count();
        std::ostringstream body;
        body << "# HELP sparkinfer_uptime_seconds Process uptime\n"
                "# TYPE sparkinfer_uptime_seconds gauge\n"
             << "sparkinfer_uptime_seconds " << uptime_s << "\n"
                "# HELP sparkinfer_requests_total Chat completion requests received\n"
                "# TYPE sparkinfer_requests_total counter\n"
             << "sparkinfer_requests_total " << g_requests_total.load() << "\n"
             << "sparkinfer_requests_streaming_total " << g_requests_streaming.load() << "\n"
                "# HELP sparkinfer_requests_by_outcome_total Requests by terminal outcome\n"
                "# TYPE sparkinfer_requests_by_outcome_total counter\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"ok\"} " << g_requests_ok.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"client_error\"} "
             << g_requests_client_error.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"overloaded\"} "
             << g_requests_overloaded.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"alloc_failed\"} "
             << g_requests_alloc_failed.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"timeout\"} "
             << g_requests_timeout.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"cancelled\"} "
             << g_requests_cancelled.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"server_error\"} "
             << g_requests_server_error.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"invalid_tool_output\"} "
             << g_requests_invalid_tool_output.load() << "\n"
                "# HELP sparkinfer_tokens_total Tokens processed\n"
                "# TYPE sparkinfer_tokens_total counter\n"
             << "sparkinfer_tokens_total{kind=\"prompt\"} " << g_prompt_tokens_total.load() << "\n"
             << "sparkinfer_tokens_total{kind=\"completion\"} " << g_completion_tokens_total.load() << "\n"
                "# HELP sparkinfer_active_requests In-flight requests\n"
                "# TYPE sparkinfer_active_requests gauge\n"
             << "sparkinfer_active_requests " << engine.active_requests() << "\n"
                "# HELP sparkinfer_free_kv_blocks Free KV cache blocks\n"
                "# TYPE sparkinfer_free_kv_blocks gauge\n"
             << "sparkinfer_free_kv_blocks " << engine.free_kv_blocks() << "\n";
        // Only emitted when the LMCache bridge is actually enabled (docs/lmcache_bridge_protocol.md)
        // -- omitting the metric entirely when disabled, rather than always emitting zeros, makes
        // "is this feature even on" visible from /metrics itself, not just server startup logs.
        const auto lmc = engine.lmcache_stats();
        if (lmc.enabled) {
            body << "# HELP sparkinfer_lmcache_lookup_hits_total External KV cache tier lookups "
                    "that restored a matched prefix\n"
                    "# TYPE sparkinfer_lmcache_lookup_hits_total counter\n"
                 << "sparkinfer_lmcache_lookup_hits_total " << lmc.lookup_hits << "\n"
                    "# HELP sparkinfer_lmcache_lookup_misses_total External KV cache tier lookups "
                    "that found nothing usable (includes a genuine miss, a timed-out sidecar, and "
                    "the sidecar being unreachable -- see docs/lmcache_bridge_protocol.md's "
                    "degradation invariant, all three fall back to the same recompute path)\n"
                    "# TYPE sparkinfer_lmcache_lookup_misses_total counter\n"
                 << "sparkinfer_lmcache_lookup_misses_total " << lmc.lookup_misses << "\n";
        }
        res.set_content(body.str(), "text/plain; version=0.0.4");
    });

    svr.Post("/v1/tokenize", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        const bool enable_thinking = sparkinfer_server::parse_enable_thinking(req.body, false);
        std::vector<int> ids;
        std::string err;
        if (!encode_messages(req.body, ids, enable_thinking, err)) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}", "application/json");
            return;
        }
        std::ostringstream body;
        body << "{\"tokens\":" << ids.size() << ",\"max_context\":" << engine.max_seq()
             << ",\"max_output_tokens\":" << max_output_tokens() << ",\"model\":\"" << g_model_name << "\"}";
        res.set_content(body.str(), "application/json");
    });

    svr.Post("/v1/chat/completions",
             [&engine](const httplib::Request& req, httplib::Response& res) {
                 if (!auth_ok(req)) {
                     res.status = 401;
                     res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
                     return;
                 }
                 if (g_shutdown_requested.load()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"server is shutting down\"}}",
                                     "application/json");
                     return;
                 }
                 if (!engine.loaded()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"model not loaded\"}}", "application/json");
                     return;
                 }

                 g_requests_total++;
                 RequestControls controls;
                 std::string err;
                 if (!parse_request_controls(req.body, controls, err)) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 const bool stream = controls.stream;
                 if (stream) g_requests_streaming++;
                 const bool enable_thinking = sparkinfer_server::parse_enable_thinking(req.body, false);
                 int max_tokens = controls.max_tokens;
                 if (max_tokens <= 0) max_tokens = 256;
                 if (max_tokens > max_output_tokens()) max_tokens = max_output_tokens();

                 std::vector<int> prompt_ids;
                 sparkinfer_server::ChatRequest chat_request;
                 if (!encode_messages(req.body, prompt_ids, enable_thinking, err, &chat_request)) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 // Presence of a tool protocol requires strict, buffered parsing even when
                 // tool_choice=none. The latter disables execution, not output validation:
                 // native tool markup must never leak through as ordinary assistant content.
                 const bool tool_protocol = !chat_request.tools.empty();
                 if ((int)prompt_ids.size() + max_tokens > engine.max_seq()) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content(
                         "{\"error\":{\"message\":\"context overflow: prompt=" +
                         std::to_string(prompt_ids.size()) + " max_tokens=" + std::to_string(max_tokens) +
                         " exceeds server ctx=" + std::to_string(engine.max_seq()) + "\"}}",
                         "application/json");
                     return;
                 }

                 const std::string cid = random_id();
                 const auto created = (long long)std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();

                 // Maps an engine outcome to the metrics bucket + HTTP status a non-2xx response
                 // should use. Overloaded -> 429 (retry elsewhere / later, not a bad request).
                 // Alloc failed -> 503 (real device OOM -- permanent until restart, never imply
                 // "retry shortly" like 429 does; #779, where this used to fall through to the
                 // same 429 as a full queue and sent operators chasing SPARKINFER_MAX_QUEUE_DEPTH
                 // for nothing while the actual queue sat empty).
                 // Timed out -> 504 (the request was valid, the server just didn't finish in time).
                 auto record_and_status = [](const sparkinfer_server::CompletionResult& o) -> int {
                     if (o.overloaded)    { g_requests_overloaded++;   return 429; }
                     if (o.alloc_failed)  { g_requests_alloc_failed++; return 503; }
                     if (o.timed_out)     { g_requests_timeout++;      return 504; }
                     g_requests_server_error++;
                     return 400;
                 };

                 if (stream) {
                     res.set_chunked_content_provider(
                         "text/event-stream",
                         [&engine, prompt_ids, max_tokens, cid, created, enable_thinking,
                          chat_request, tool_protocol, include_usage = controls.include_usage,
                          stop = controls.stop]
                         (size_t offset, httplib::DataSink& sink) {
                             if (offset > 0) {
                                 sink.done();
                                 return true;
                             }
                             if (!write_stream_role(sink, cid, created)) {
                                 g_requests_cancelled++;
                                 sink.done();
                                 return true;
                             }
                             std::vector<int> stream_ids;
                             stream_ids.reserve((size_t)max_tokens);
                             sparkinfer_server::ThinkingStreamSplitter splitter(enable_thinking, engine.is_museglimmer());
                             sparkinfer_server::StopSequenceFilter stop_filter(stop);
                             std::string tool_stop_text;  // raw accumulator, tool_protocol branch only
                             // Set by either on_tok branch right before it returns false to signal a
                             // matched `stop` sequence -- checked ahead of outcome.cancelled below,
                             // since both now collapse to the same engine-level cancelled flag, but
                             // only a real disconnect means "the client is gone, send nothing".
                             bool stopped_by_sequence = false;
                             // Returning false cancels generation -- the client is gone, so there
                             // is no point spending GPU time finishing the response.
                             auto on_tok = [&](int tid) -> bool {
                                 // Tool-capable responses are intentionally buffered until the
                                 // native Qwen XML is complete and schema-valid. This still emits
                                 // valid atomic OpenAI SSE deltas, and guarantees malformed or
                                 // partial markup can never leak as executable client content.
                                 if (tool_protocol) {
                                     if (stop.empty()) {
                                         stream_ids.push_back(tid);
                                         return sink.is_writable();
                                     }
                                     // Only pay for incremental decode here when the client
                                     // actually set `stop` -- a scoped opt-in, not a blanket cost
                                     // increase for every tool-calling request.
                                     tool_stop_text += g_tokenizer.decode_delta(stream_ids, tid);
                                     size_t pos;
                                     if (find_stop_match(tool_stop_text, stop, pos)) {
                                         stopped_by_sequence = true;
                                         return false;
                                     }
                                     return sink.is_writable();
                                 }
                                 std::string piece = g_tokenizer.decode_delta(stream_ids, tid);
                                 std::string safe = stop_filter.feed(piece);
                                 if (stop_filter.matched()) {
                                     if (!safe.empty()) {
                                         const auto delta = splitter.feed(safe);
                                         if (!delta.reasoning_content.empty())
                                             write_stream_delta(sink, cid, created, "reasoning_content",
                                                                delta.reasoning_content);
                                         if (!delta.content.empty())
                                             write_stream_delta(sink, cid, created, "content", delta.content);
                                     }
                                     stopped_by_sequence = true;
                                     return false;
                                 }
                                 const auto delta = splitter.feed(safe);
                                 bool ok = true;
                                 if (!delta.reasoning_content.empty())
                                     ok = write_stream_delta(sink, cid, created, "reasoning_content",
                                                              delta.reasoning_content) && ok;
                                 if (!delta.content.empty())
                                     ok = write_stream_delta(sink, cid, created, "content", delta.content) && ok;
                                 return ok && sink.is_writable();
                             };
                             const auto outcome = engine.complete_streaming(prompt_ids, max_tokens, on_tok);
                             const int prompt_tokens = (int)prompt_ids.size();
                             const int completion_tokens = (int)stream_ids.size();
                             g_prompt_tokens_total += (uint64_t)prompt_tokens;
                             g_completion_tokens_total += (uint64_t)completion_tokens;
                             if (outcome.cancelled && !stopped_by_sequence) {
                                 // Client is already gone -- nothing left to write to, just record it.
                                 g_requests_cancelled++;
                                 sink.done();
                                 return true;
                             }
                             if (!outcome.error.empty()) {
                                 if (outcome.overloaded) g_requests_overloaded++;
                                 else if (outcome.alloc_failed) g_requests_alloc_failed++;
                                 else if (outcome.timed_out) g_requests_timeout++;
                                 else g_requests_server_error++;
                                 std::ostringstream err_chunk;
                                 err_chunk << "data: {\"error\":{\"message\":\"" << json_escape(outcome.error)
                                           << "\"}}\n\n";
                                 sink.write(err_chunk.str().c_str(), (size_t)err_chunk.str().size());
                                 // A client that explicitly asked for usage still deserves the
                                 // token counts even though generation faulted -- ttft/generation
                                 // timing isn't meaningful for a hard engine error, so omit those
                                 // (write_stream_usage only includes a field when its arg is >= 0).
                                 if (include_usage)
                                     write_stream_usage(sink, cid, created, prompt_tokens,
                                                        completion_tokens, -1.0, -1.0, -1.0);
                                 write_stream_done(sink);
                                 sink.done();
                                 return true;
                             }

                             std::string finish_reason = outcome.reached_token_limit ? "length" : "stop";
                             if (tool_protocol) {
                                 std::string text;
                                 std::string decode_err;
                                 if (!decode_ids(stream_ids, text, decode_err)) {
                                     g_requests_server_error++;
                                     const nlohmann::json error = {{"error", {{"message", decode_err}}}};
                                     write_sse_json(sink, error);
                                     write_stream_done(sink);
                                     sink.done();
                                     return true;
                                 }
                                 if (stopped_by_sequence) {
                                     size_t pos;
                                     if (find_stop_match(text, stop, pos)) text.resize(pos);
                                 }
                                 auto parsed = sparkinfer_server::parse_assistant_output(
                                     text, enable_thinking, engine.is_museglimmer(), &chat_request);
                                 if (!parsed.error.empty()) {
                                     if (outcome.reached_token_limit || stopped_by_sequence) {
                                         // A truncated native call is not an executable result,
                                         // but token exhaustion (or a stop sequence landing mid
                                         // tool-call XML) is still a normal completion.
                                         // Emit no buffered markup and finish with "length"/"stop".
                                         parsed = {};
                                     } else {
                                         g_requests_invalid_tool_output++;
                                         const nlohmann::json error = {
                                             {"error", {{"message", "invalid model tool call: " + parsed.error}}}};
                                         write_sse_json(sink, error);
                                         write_stream_done(sink);
                                         sink.done();
                                         return true;
                                     }
                                 }
                                 write_stream_delta(sink, cid, created, "reasoning_content",
                                                    parsed.reasoning_content);
                                 write_stream_delta(sink, cid, created, "content", parsed.content);
                                 // parsed.tool_calls is only non-empty here when parsed.error was
                                 // empty (the reached_token_limit-and-error case already reset
                                 // parsed to {} above) -- i.e. a complete, valid call, even if the
                                 // model also happened to exhaust its token budget emitting it.
                                 // Gating this on !reached_token_limit dropped that call entirely.
                                 for (size_t i = 0; i < parsed.tool_calls.size(); ++i) {
                                     parsed.tool_calls[i].id = random_id("call_");
                                     write_stream_tool_call(sink, cid, created, i, parsed.tool_calls[i]);
                                 }
                                 if (!parsed.tool_calls.empty()) finish_reason = "tool_calls";
                             } else if (!stopped_by_sequence) {
                                 // On a stop-match exit, on_tok already wrote whatever safe text
                                 // stop_filter cleared before returning false. Deliberately skip
                                 // finish() here: it exists to flush the splitter's OWN unrelated
                                 // holdback on a legitimate end of generation, not bytes stop_filter
                                 // never vetted -- generation ends at the match point regardless.
                                 sparkinfer_server::ThinkingStreamSplitter::Delta flush;
                                 splitter.finish(flush);
                                 write_stream_delta(sink, cid, created, "reasoning_content",
                                                    flush.reasoning_content);
                                 write_stream_delta(sink, cid, created, "content", flush.content);
                             }

                             g_requests_ok++;
                             write_stream_finish(sink, cid, created, finish_reason);
                             if (include_usage)
                                 write_stream_usage(sink, cid, created, prompt_tokens,
                                                    completion_tokens, outcome.ttft_ms,
                                                    outcome.generation_ms, outcome.decode_tps);
                             write_stream_done(sink);
                             sink.done();
                             return true;
                         });
                     return;
                 }

                 // engine.complete() is complete_streaming(..., nullptr); pass a real callback
                 // whenever stop was requested so non-streaming requests can halt early too --
                 // previously stop-checking (and the compute savings of stopping early) were
                 // unavailable here entirely.
                 std::vector<int> nonstream_ids;
                 std::string nonstream_stop_text;
                 bool stopped_by_sequence = false;
                 auto nonstream_on_tok = [&](int tid) -> bool {
                     if (controls.stop.empty()) {
                         nonstream_ids.push_back(tid);
                         return true;
                     }
                     nonstream_stop_text += g_tokenizer.decode_delta(nonstream_ids, tid);
                     size_t pos;
                     if (find_stop_match(nonstream_stop_text, controls.stop, pos)) {
                         stopped_by_sequence = true;
                         return false;
                     }
                     return true;
                 };
                 const auto outcome = engine.complete_streaming(prompt_ids, max_tokens, nonstream_on_tok);
                 std::string text;
                 if (!outcome.error.empty()) {
                     res.status = record_and_status(outcome);
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(outcome.error) + "\"}}",
                                     "application/json");
                     return;
                 }
                 if (!decode_ids(outcome.tokens, text, err)) {
                     g_requests_server_error++;
                     res.status = 500;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 g_prompt_tokens_total += (uint64_t)prompt_ids.size();
                 g_completion_tokens_total += (uint64_t)outcome.tokens.size();
                 if (stopped_by_sequence) {
                     size_t pos;
                     if (find_stop_match(text, controls.stop, pos)) text.resize(pos);
                 }

                 auto parsed = sparkinfer_server::parse_assistant_output(
                     text, enable_thinking, engine.is_museglimmer(),
                     tool_protocol ? &chat_request : nullptr);
                 if (!parsed.error.empty()) {
                     if (outcome.reached_token_limit || stopped_by_sequence) {
                         // Never expose a truncated native tag sequence. A length/stop end is a
                         // valid completion, so return an empty assistant result instead of 5xx.
                         parsed = {};
                     } else {
                         g_requests_invalid_tool_output++;
                         res.status = 502;
                         const nlohmann::json error = {
                             {"error", {{"message", "invalid model tool call: " + parsed.error}}}};
                         res.set_content(error.dump(), "application/json");
                         return;
                     }
                 }

                 nlohmann::json message = {{"role", "assistant"}};
                 if (!parsed.reasoning_content.empty())
                     message["reasoning_content"] = parsed.reasoning_content;
                 std::string finish_reason = outcome.reached_token_limit ? "length" : "stop";
                 // See the streaming path's identical comment: parsed.tool_calls is only
                 // non-empty here for a complete, successfully-parsed call.
                 if (!parsed.tool_calls.empty()) {
                     message["content"] = parsed.content.empty()
                         ? nlohmann::json(nullptr) : nlohmann::json(parsed.content);
                     message["tool_calls"] = nlohmann::json::array();
                     for (auto& call : parsed.tool_calls) {
                         call.id = random_id("call_");
                         message["tool_calls"].push_back({
                             {"id", call.id}, {"type", "function"},
                             {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
                     }
                     finish_reason = "tool_calls";
                 } else {
                     message["content"] = parsed.content;
                 }

                 nlohmann::json usage = {
                     {"prompt_tokens", (int)prompt_ids.size()},
                     {"completion_tokens", (int)outcome.tokens.size()},
                     {"total_tokens", (int)(prompt_ids.size() + outcome.tokens.size())}};
                 if (outcome.ttft_ms >= 0.0) usage["ttft_ms"] = outcome.ttft_ms;
                 if (outcome.generation_ms >= 0.0) usage["generation_ms"] = outcome.generation_ms;
                 if (outcome.decode_tps >= 0.0) usage["decode_tps"] = outcome.decode_tps;
                 const nlohmann::json body = {
                     {"id", cid}, {"object", "chat.completion"}, {"created", created},
                     {"model", g_model_name},
                     {"choices", nlohmann::json::array({{{"index", 0},
                                                          {"message", message},
                                                          {"finish_reason", finish_reason}}})},
                     {"usage", usage}};
                 g_requests_ok++;
                 res.set_content(body.dump(), "application/json");
             });

    // Transport-level deadlines. Defaults are generous, not aggressive: a cold 32k-context
    // prefill has been measured taking ~90s of TTFT alone (see eval/pr_dflash_bot.py's 32k
    // sweep), so a short default here would misfire on legitimate long-context requests.
    // The read timeout resets on each byte received, so a slow streaming response keeps
    // extending it as it goes -- this only fires on a genuinely stalled connection.
    const long read_timeout_s = getenv("SPARKINFER_READ_TIMEOUT_S") ? atol(getenv("SPARKINFER_READ_TIMEOUT_S")) : 300;
    const long write_timeout_s = getenv("SPARKINFER_WRITE_TIMEOUT_S") ? atol(getenv("SPARKINFER_WRITE_TIMEOUT_S")) : 300;
    svr.set_read_timeout(read_timeout_s, 0);
    svr.set_write_timeout(write_timeout_s, 0);

    std::signal(SIGTERM, on_shutdown_signal);
    std::signal(SIGINT, on_shutdown_signal);
    // Signal handlers must stay async-signal-safe (just set the atomic flag above); the actual
    // drain-and-stop happens here, on an ordinary thread. /v1/chat/completions and /v1/capacity
    // already check g_shutdown_requested to refuse new work as soon as the flag flips, so the
    // gap between "signal received" and "svr.stop() called" (at most one poll interval) only
    // means a few new requests might land right before shutdown starts, not that the drain window
    // is unbounded.
    std::thread shutdown_watcher([&svr] {
        while (!g_shutdown_requested.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        fprintf(stderr, "[sparkinfer-server] shutdown signal received, draining in-flight requests...\n");
        svr.stop();
        // svr.stop() only closes the LISTENING socket -- it does not force-close connections
        // already accepted. A client that vanishes without a clean TCP close (RST, dead network,
        // a hard `kill` on a curl process) can leave an httplib worker thread blocked in a read()
        // for up to the read timeout (SPARKINFER_READ_TIMEOUT_S, default 300s) -- measured: this
        // alone can make listen() take minutes to return, defeating the point of a "graceful"
        // shutdown. Bound the total drain time instead: same SIGTERM -> grace period -> force-kill
        // shape as Kubernetes' terminationGracePeriodSeconds.
        const long grace_s = getenv("SPARKINFER_SHUTDOWN_GRACE_S")
                                 ? atol(getenv("SPARKINFER_SHUTDOWN_GRACE_S")) : 30;
        std::this_thread::sleep_for(std::chrono::seconds(grace_s));
        fprintf(stderr, "[sparkinfer-server] shutdown grace period (%lds) elapsed with requests "
                        "still draining -- forcing exit\n", grace_s);
        _exit(0);  // not exit(): other threads may still be mid-flight; skip atexit/static dtors
    });

    const std::string queue_depth_label =
        engine.max_queue_depth() > 0 ? std::to_string(engine.max_queue_depth()) : std::string("unlimited");
    fprintf(stderr,
            "[sparkinfer-server] OpenAI-compatible API on http://%s:%d\n"
            "  GET  /health\n"
            "  GET  /v1/models\n"
            "  GET  /v1/info\n"
            "  GET  /v1/capacity\n"
            "  GET  /metrics\n"
            "  POST /v1/tokenize\n"
            "  POST /v1/chat/completions\n"
            "  read_timeout=%lds write_timeout=%lds max_output_tokens=%d max_queue_depth=%s\n",
            host.c_str(), port, read_timeout_s, write_timeout_s, max_output_tokens(),
            queue_depth_label.c_str());

    const bool bound = svr.listen(host.c_str(), port);
    g_shutdown_requested = true;  // unblock the watcher thread if listen() returned on its own
    shutdown_watcher.join();
    if (!bound) {
        fprintf(stderr, "[sparkinfer-server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
