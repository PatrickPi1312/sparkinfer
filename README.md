![sparkinfer banner](docs/sparkinfer.png)

# SP⚡RKINFER · Powered by SN74

**Agentic AI inference. Optimized for every Blackwell GPU.**

**+86% faster** than llama.cpp with Blackwell-native **Custom CUDA kernels** (Qwen3.6-35B-A3B SOTA, RTX 5090, v0.4.3). SparkInfer is the runtime layer for **Private AI** agents — optimized MoE/LLM decoding from desk-side RTX to workstation PRO 6000. Continuously optimized by competition at **[SN74 on Gittensor](https://gittensor.io/miners/repository?name=gittensor-ai-lab%2Fsparkinfer)** and **Kernel Design Agents**.

**Why fastest?** Faster inference means more intelligence, more responsive agents, and more efficient compute.

> **Fewer models. Deeper optimization. Faster evolution.**

## Frontier models

SparkInfer focuses on the models driving the future of AI — not thousands of legacy architectures.

| Model | Role |
|---|---|
| [**Qwen3.6-35B-A3B**](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | Primary SOTA — hybrid Gated-DeltaNet + full-attention MoE |
| [**Qwen3.8-27B**](https://huggingface.co/Qwen/Qwen3.8-27B) | Dense hybrid Gated-DeltaNet · current eval scope — native NVFP4 from **[our RTX 5090 build](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)** or [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4), both supported |
| [**SparkDistill**](https://github.com/gittensor-model-hub/SparkDistill/) | Fable 5 / OpenAI 5.6-level CoT *(coming soon)* |
| [**MiniMax M3**](https://huggingface.co/MiniMaxAI/MiniMax-M3) | Open MoE frontier *(next)* |

## Blackwell native

**Consumer → Workstation → Datacenter** — built for NVIDIA Blackwell from the beginning (`sm_120` + `sm_121`, not datacenter `sm_100`).

| GPU | Arch | Target |
|---|---|---|
| [RTX Spark GB10](https://nvidianews.nvidia.com/news/nvidia-microsoft-windows-pcs-agents-rtx-spark) | `sm_121` | Personal AI PC · desk-side agents |
| [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) | `sm_121` | AI workstation |
| [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) | `sm_120` | Consumer Blackwell · current dev platform |
| [RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/) | `sm_120` | 96 GB workstation · 32k/4k API profile |

## Benchmark · Qwen3.6-35B-A3B SOTA

RTX 5090 · same `UD-Q4_K_M` GGUF · greedy bs=1 · warm interleaved · **v0.4.4** frontier (same-box main guards).

### Decode

| context | SparkInfer | llama.cpp | Δ |
|---:|---:|---:|---:|
| 128 | **512** tok/s | 276 tok/s | **+86%** |
| 512 | **506** tok/s | 276 tok/s | +83% |
| 4k | **486** tok/s | 276 tok/s | +76% |
| 16k | **467** tok/s | 281 tok/s | +66% |
| 32k | **437** tok/s | 280 tok/s | +56% |

### Prefill

| context | SparkInfer | llama.cpp | Δ |
|---:|---:|---:|---:|
| 4k | **13,800** tok/s | 8,726 tok/s | **+58%** |
| 16k | **17,700** tok/s | 8,390 tok/s | **+111%** |
| 32k | **18,150** tok/s | 7,984 tok/s | **+127%** |

Quality parity vs llama.cpp: top-1 **0.953** · KL **0.031** · IFEval **83%** · BFCL **75%**.

Full competitor matrix (vLLM, SGLang, TensorRT-LLM) and quality tables:
[`bench/competitors/latest-results.md`](bench/competitors/latest-results.md) ·
[`bench/quality/README.md`](bench/quality/README.md).

## Benchmark · Qwen3.8-27B

RTX 5090 · **same `Q4_K_M` GGUF** ([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)) · greedy bs=1 · 5 reps ·
sparkinfer `d8e1c74` vs `llama.cpp d8df12e` (`-ngl 99`; `-fa 1` was measured too and came out
marginally slower on this model, so llama.cpp's better configuration is the one reported).

### Decode

| context | SparkInfer | llama.cpp | Δ |
|---:|---:|---:|---:|
| 128 | **86.9** tok/s | 80.2 tok/s | **+8.4%** |
| 4k | **85.2** tok/s | 77.0 tok/s | **+10.6%** |
| 16k | **82.3** tok/s | 73.9 tok/s | **+11.5%** |

### Prefill

| context | SparkInfer | llama.cpp | Δ |
|---:|---:|---:|---:|
| 128 | 2,033 tok/s | **2,782** tok/s | **−26.9%** |
| 4k | **7,548** tok/s | 3,670 tok/s | **+105.7%** |
| 16k | **7,596** tok/s | 3,496 tok/s | **+117.2%** |

Short-context prefill is the one dimension llama.cpp still wins, and it is listed here rather than
omitted: at ctx=128 there is not enough work to fill the GPU with one batched GEMM pass, so the
per-pass overhead sparkinfer pays to be fast at 4k/16k is not yet amortised. It is a live
optimisation target, tracked by the same automated eval that gates every PR.

### Native NVFP4 — two supported checkpoints

The GGUF numbers above exist to make the engine comparison fair. They are **not** how sparkinfer
runs this model: it reads Qwen3.8-27B's NVFP4 weights directly, from **either** of two supported
checkpoints.

**[gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)**
— ours: uniform NVFP4 on every `Linear`, quantized in house with NVIDIA ModelOpt for the RTX 5090's
FP4 tensor cores, and the checkpoint the automated eval scores every PR against:

| context | decode | prefill | vs same-GGUF sparkinfer | vs llama.cpp |
|---:|---:|---:|---:|---:|
| 128 | **95.8** tok/s | **6,546** tok/s | +10.2% dec · +222% pp | +19.4% dec · **+135%** pp |
| 4k | **93.7** tok/s | **10,021** tok/s | +10.0% dec · +32.8% pp | +21.7% dec · **+173%** pp |
| 16k | **90.1** tok/s | **9,749** tok/s | +9.4% dec · +28.4% pp | +22.0% dec · **+179%** pp |

Same weights the model was released with, re-quantized for the hardware it runs on: **+9–10% decode
and +28–222% prefill over the same engine reading a Q4_K_M GGUF**, and it erases the one dimension
llama.cpp wins above — 6,546 pp at ctx=128 against 2,782.

**[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)** — upstream: NVFP4
FFN + FP8 attention/GDN projections, equally supported, and measured on the same box at the same
commit:

| context | decode | prefill |
|---:|---:|---:|
| 128 | 84.9 tok/s | 5,031 tok/s |
| 4k | 83.2 tok/s | 9,548 tok/s |
| 16k | 80.6 tok/s | 8,727 tok/s |

Both load through the same `load_compressed_tensors` path and share the batched prefill and every
decode kernel — the mixed-precision build is simply a different quantization of the same weights,
so it lands a little slower on this hardware. The eval *scores* PRs on our build and runs a
separate *no-regression guard* on the upstream one, which is what stops an optimisation from
winning on the scored checkpoint by pessimising the other.

Read that last column for what it is. llama.cpp cannot load NVFP4 compressed-tensors at all, so
`vs llama.cpp` there is **each engine on the format it actually runs** — our NVFP4 against
llama.cpp's best GGUF result — not a same-weights benchmark. The same-weights comparison is the
one at the top of this section, GGUF on both sides, prefill@128 loss included.

Runtime footprint (excluding model weights):

| runtime | size | vs sparkinfer |
|---|---:|---:|
| sparkinfer native binary | **2.5 MB** | 1× |
| llama.cpp CUDA | 80 MB | 33× larger |
| vLLM | 605 MB | 243× larger |

## Powered by SN74 — moving at the speed of ⚡

Contributors submit PRs; the bot verifies correctness and speed on real RTX 5090 hardware; SN74 rewards verified marginal speedups. **15 releases in 3 weeks** — from first llama.cpp beat to **+86% decode / +127% prefill @ 32k on Qwen3.6 SOTA**.

1. Pick a narrow bottleneck in the Blackwell decode path.
2. Submit a PR with source changes and benchmark evidence.
3. The bot builds `main` and the PR on the same RTX 5090.
4. Correctness vs llama.cpp; guards at 128 / 512 / 4k / 16k / 32k decode.
5. Strongest context improvement scores; regressions get `regression-*` labels.
6. Frontier merges; the [dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/) updates.

Miner workflow: [`docs/miner-guide.md`](docs/miner-guide.md).

## Roadmap

### Milestone 1 · Now — Fast on every Blackwell edge GPU

*Fastest = cost-effective inference* — more tokens per dollar on Blackwell edge first.

- Qwen3.6 SOTA: **+86%** decode / **+127%** prefill @ 32k vs llama.cpp on RTX 5090 (512 tok/s decode @ 128 ctx)
- RTX PRO 6000 — **32k input + 4k output**, full MoE resident
- RTX Spark + DGX Spark `sm_121` bring-up for desk-side agents
- Fastest AI runtime at the edge · desktop app, RAG, memory

### Milestone 2 · Next — Trustable AI on confidential compute

Attested builds and sealed execution on PRO 6000 server and B200.

- TDX + NVIDIA CC attestation for `sparkinfer-server` workloads
- Source-verified binaries — same eval loop, inside the enclave
- Privacy guardrails and end-to-end encryption
- Domain-specific models via [SparkDistill](https://github.com/gittensor-model-hub/SparkDistill/)
- Licensed on-prem runtime for regulated enterprise

## Quickstart

On NVIDIA Blackwell (CUDA 12.8+) — scripts auto-detect GPU arch, fetch prebuilt binaries (or build from source), and download the model:

```bash
# decode throughput (fetches Qwen3-30B-A3B Q4_K_M on first run)
bench/scripts/bench.sh --download

# head-to-head vs llama.cpp on the same GGUF + GPU
bench/scripts/bench.sh --download --compare

# accuracy gate — token-match / KL vs llama.cpp
bench/scripts/accuracy.sh --download
```

Your own model: `bench/scripts/bench.sh /path/to/model.gguf --tokens 256`. Options: [`bench/scripts/README.md`](bench/scripts/README.md).

## Layout & scoring

| Path | What |
|---|---|
| [`kernels/`](kernels) | CUDA kernels — flash-decode, decode GEMV, fused MoE FFN, GEMM, RMSNorm, RoPE, GGUF dequant |
| [`runtime/`](runtime) | scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward |
| [`moe/`](moe) | sync-free MoE router + expert dispatch |
| [`bench/`](bench) | reproducible benchmarks + eval harness |
| [`dashboard/`](dashboard) | static frontier dashboard (GitHub Pages) |
| [`server/`](server) | OpenAI-compatible HTTP API (`BUILD_SERVER=ON`) |

**Scoring is speedup-only.** SN74 pays verified marginal speedups labeled **XL / L / M / S / XS**. Sub-2% gains are never aggregated across contexts. See [`.gittensor/weights.json`](.gittensor/weights.json).

## Build

Requires **CUDA Toolkit 12.8+** (`sm_120` / `sm_121` codegen).

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120   # or 121 for RTX Spark / Jetson Thor
cmake --build build -j
ctest --test-dir build
```

## Automated evaluation

Open a PR — a bot evaluates every ~30 min: source build on RTX 5090, correctness gate vs llama.cpp, no-regression guards, **`eval:<label>`** verdict. The bot **never auto-merges**. Details: [`eval/`](eval) · **[EVAL-TRUST.md](EVAL-TRUST.md)** (Polaris TDX receipts, reproducible from source today).

| label | meaning |
|---|---|
| `XL · L · M · S · XS` | verified speedup over frontier, by % gain |
| `none` | correct, no verified improvement |
| `REJECT` | failed correctness or regression |
| `BASELINE` | first verified frontier entry |

## Contributing

Source-required and reproducible. Before a PR: `bench/scripts/bench.sh` + `bench/scripts/accuracy.sh`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) · [Changelog](CHANGELOG.md)
