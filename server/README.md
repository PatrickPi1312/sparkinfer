# sparkinfer-server

OpenAI-compatible HTTP API for local GGUF inference — backend for sparkinfer.com.

Enable with `-DBUILD_SERVER=ON` when building this repo (`main`).

## Build

Requires **Rust/cargo** (build-time only) for HuggingFace `tokenizer.json` via [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SERVER=ON
cmake --build build -j$(nproc) --target sparkinfer_server
```

Or reuse the bench harness build root:

```bash
bench/scripts/_common.sh  # optional: sets ARCH
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SERVER=ON
cmake --build build --target sparkinfer_server
```

## Run

```bash
export SPARKINFER_ROOT="$(pwd)"
# Native C++ tokenizer (tokenizers-cpp). Requires rustc/cargo at build time only.

# download model on first bench run, or:
# bench/scripts/bench.sh --download

# Default: unsloth/Qwen3.6-35B-A3B-GGUF UD-Q4_K_M (~22 GB)
# https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
./server/run.sh --download
# or:
./build/server/sparkinfer_server -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --port 8080
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | `{"status":"ok"}` |
| `GET /v1/models` | OpenAI model list (includes live `context_length`) |
| `GET /v1/info` | Model limits (`max_context`, `max_output_tokens`) — live values, not build-time constants |
| `GET /v1/capacity` | This worker's live occupancy: `active_requests`, `free_kv_blocks`, `max_queue_depth`, `accepting_requests`. Single-process only — not fleet-wide. |
| `GET /metrics` | Prometheus text-exposition counters/gauges: request totals by outcome (`ok`/`client_error`/`overloaded`/`timeout`/`cancelled`/`server_error`), prompt/completion token totals, active requests, free KV blocks, uptime. |
| `POST /v1/tokenize` | Token count for a chat request body |
| `POST /v1/chat/completions` | Chat (JSON `messages`, optional `stream`, `enable_thinking`). Responses include OpenAI `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`) plus additive GPU timing fields (`ttft_ms`, `generation_ms`, `decode_tps`) that standard OpenAI SDKs ignore. Streaming sends a final chunk with `choices:[]` + `usage` before `[DONE]`. A streaming client that disconnects mid-response cancels generation (checked via `DataSink::is_writable()`) instead of running to completion for nobody. Overload (no queue capacity) returns `429`; a request that exceeds `SPARKINFER_REQUEST_TIMEOUT_S` returns `504`. |

### Graceful shutdown

`SIGTERM`/`SIGINT` stop accepting new connections and new `/v1/chat/completions` requests
(`503`) immediately, then let in-flight requests finish naturally before the process exits —
no hard-killed streams. Bounded by `SPARKINFER_SHUTDOWN_GRACE_S` (default `30`): a client that
vanishes without a clean TCP close can otherwise leave the process waiting up to the read
timeout, so after the grace period the process force-exits regardless.

### RTX PRO 6000 deploy (32k / 4k)

See [`bench/results/qwen3-30b-a3b_q4km_pro6000.md`](../bench/results/qwen3-30b-a3b_q4km_pro6000.md) for the full 5090→PRO 6000 migration
notes and benchmark table.

```bash
export CTX=36864          # 32k prompt + 4k completion KV pool
export HOST=0.0.0.0
./server/run.sh --download
curl -s http://127.0.0.1:8080/v1/info
# {"model":"qwen3.6-35b-a3b","max_context":32768,"max_output_tokens":4096}
```

On RTX 5090 (32 GB) use a smaller `--ctx` (8k–16k) or `CTX=0` for GGUF defaults — the
same binary, different memory budget.

### Example

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sparkinfer","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

With API key (optional):

```bash
./build/server/sparkinfer_server -m model.gguf --api-key secret
curl ... -H 'Authorization: Bearer secret'
```

## Request isolation & continuous batching

Each `/v1/chat/completions` call is submitted to `ContinuousBatchEngine`, which:

- Allocates a **per-request `seq_id`** with **right-sized KV** (`prompt + max_tokens + headroom`, not `max_seq`)
- Runs **vLLM V1-style iteration-level scheduling**: each step packs pending decode
  requests first (up to `SPARKINFER_BATCH_TOKENS`), then admits at most one prefill into
  the remaining budget. Prefills larger than `SPARKINFER_PREFILL_MIX_MAX` wait until
  decode drains (hybrid batched prefill is atomic — mixing an 8k pass mid-decode would
  spike ITL by hundreds of ms)
- Under `chunked` (or when decode is waiting under `continuous`), non-batched models
  advance prefills in chunks of `SPARKINFER_PREFILL_CHUNK_TOKENS` before yielding
- Frees KV blocks when the request finishes (no cross-request KV leakage)
- Uses per-request hybrid Gated-DeltaNet recurrent buffers when the model is hybrid

Shared prefix cache still works: when the chat prompt starts with configured prefix tokens,
`cache_prefix()` warms session 0 and only the suffix is prefilled per request.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARKINFER_BATCH_TOKENS` | `64` | Scheduler token budget per step (decode packing) |
| `SPARKINFER_SCHED_POLICY` | `continuous` | `continuous` (pack+mix), `chunked` (CHUNKED_PREFILL), or `priority` (exclusive prefill) |
| `SPARKINFER_PREFILL_CHUNK_TOKENS` | `512` | Token-loop prefill yield size when batched prefill is unavailable (`0` = unlimited). Hybrid models always use full batched GEMM prefill. |
| `SPARKINFER_PREFILL_MIX_MAX` | `2048` | Max prompt tokens allowed to mix with decode in one step (`0` = always mix). Larger atomic prefills wait until decode drains to avoid ITL spikes. |

Prior requests cannot leak decode context into later ones (KV is freed after each completion).

## Env

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARKINFER_ROOT` | `.` | Repo root (tokenizer script path) |
| `CTX` | `36864` (PRO 6000) / `0` (5090) | KV pool size passed as `--ctx` |
| `SPARKINFER_KV_INT8` | model-dependent | Same as `qwen3_gguf_generate` |
| `SPARKINFER_TOKENIZER_URL` | Qwen3.6-35B-A3B tokenizer | Override tokenizer download |
| `SPARKINFER_SERVER_PREFIX_TOKEN_FILE` | — | JSON `[id,...]` warmed via `cache_prefix` each request |
| `SPARKINFER_SERVER_PREFIX_TOKEN_IDS` | — | Comma-separated token ids (same as above) |
| `SPARKINFER_PREFILL_BATCHED` | `1` | Batched prefill in `cache_prefix` / cold prompts |
| `SPARKINFER_MAX_OUTPUT_TOKENS` | `4096` | Per-request generation cap (independent of context length, which is checked separately against the live `--ctx`) |
| `SPARKINFER_MAX_QUEUE_DEPTH` | `0` (unlimited) | Admission-time cap on in-flight + queued requests. Beyond it, new requests are rejected as `429` before any KV allocation is attempted, instead of failing later on KV exhaustion. |
| `SPARKINFER_REQUEST_TIMEOUT_S` | `0` (disabled) | Per-request wall-clock deadline from submission to finish; exceeding it returns `504`. Left disabled by default — a cold 32k-context prefill alone has been measured taking ~90s of TTFT, so an aggressive default would misfire on legitimate long-context requests. |
| `SPARKINFER_READ_TIMEOUT_S` / `SPARKINFER_WRITE_TIMEOUT_S` | `300` | Transport-level socket timeouts (httplib). Reset on each byte transferred, so a slow-but-progressing stream doesn't trip them. |
