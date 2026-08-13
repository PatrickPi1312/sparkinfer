#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/scheduler.h"

namespace sparkinfer {

// Continuous-batch serving engine: queues requests, assigns per-request seq_ids,
// right-sizes KV allocation, and interleaves decode steps via the Scheduler.
//
// DFlash multi-token accept in step_job() is intentionally deferred until the
// single-stream dflash_generate path proves SPEC_AGREE=100% and a tok/s win
// (see bench/scripts/dflash_accuracy.sh). Do not wire speculative multi-accept here yet.
class ContinuousBatchEngine {
public:
    struct Request {
        std::vector<int> prompt;
        int max_new_tokens = 0;
        int priority = 0;
        int prefill_start = 0;          // skip tokens already in a shared prefix cache
        bool use_prefix_session = false; // bind to session 0 (cache_prefix KV)
    };

    struct Result {
        std::vector<int> tokens;
        std::string error;
        // true => caller should surface 429 (no capacity right now), not a generic 4xx —
        // the request itself was fine, there was just nowhere to run it.
        bool overloaded = false;
        // true => a real device allocation failed (distinct from the KV pool being full, which
        // sets `overloaded` instead) -- caller should surface 503, not 429: this condition is
        // permanent until the process restarts, not a transient "try again shortly" (#779, where
        // a leak eventually exhausted VRAM and every subsequent request got a misleading 429
        // "overloaded" even though the queue was empty and the KV pool had free blocks).
        bool alloc_failed = false;
        // true => a per-request deadline (SPARKINFER_REQUEST_TIMEOUT_S) was exceeded.
        bool timed_out = false;
        // true => on_token returned false (client went away mid-stream); not an error.
        bool cancelled = false;
        // true => generation exhausted max_new_tokens instead of reaching an EOS token.
        // HTTP callers surface this as finish_reason="length"; in particular, a truncated
        // tool-call payload must never be reported as a successful "stop".
        bool reached_token_limit = false;
        // GPU-side timings (exclude SSE/on_token backpressure).
        double ttft_ms = -1.0;
        double generation_ms = -1.0;
        double decode_tps = -1.0;
    };

    ContinuousBatchEngine(Qwen35Model* model, KVCacheManager* kv,
                          int max_tokens_per_batch = 64,
                          SchedulePolicy policy = SchedulePolicy::CONTINUOUS_BATCHING);
    ~ContinuousBatchEngine();

    ContinuousBatchEngine(const ContinuousBatchEngine&) = delete;
    ContinuousBatchEngine& operator=(const ContinuousBatchEngine&) = delete;

    // Blocking completion (used by the HTTP server).
    Result complete(const Request& req);

    // Streaming completion: on_token is invoked on the worker thread as tokens are produced.
    // on_token returns false to cancel generation early (e.g. the client disconnected) --
    // the request then finishes with Result::cancelled = true, not an error.
    Result complete_streaming(const Request& req, const std::function<bool(int)>& on_token);

    int num_active() const;
    int num_free_kv_blocks() const;
    // Admission-time queue depth cap (SPARKINFER_MAX_QUEUE_DEPTH, 0 = unlimited). Requests
    // beyond this are rejected as overloaded before any KV allocation is attempted.
    int max_queue_depth() const;

private:
    struct Job;
    enum class EnqueueError { NONE, BAD_REQUEST, OVERLOADED, ALLOC_FAILED };
    uint64_t submit_locked(Job job, const std::function<bool(int)>& on_token, EnqueueError* err_out);
    Result wait_locked(uint64_t request_id);
    void worker_loop();
    bool step_job(Job& job, bool chunked = false);

    Qwen35Model* model_;
    KVCacheManager* kv_;
    Scheduler scheduler_;
    SchedulePolicy policy_ = SchedulePolicy::CONTINUOUS_BATCHING;
    std::thread worker_;
    std::atomic<bool> running_{false};
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<uint64_t, std::unique_ptr<Job>> jobs_;
    std::atomic<uint64_t> next_req_id_{1};
};

}  // namespace sparkinfer
