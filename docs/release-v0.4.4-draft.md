# GitHub release notes draft — v0.4.4

Title: `v0.4.4 — DFlash speculative decode for Qwen3.6 + prefill +127% @32k`

Copy into the GitHub release body (same structure as v0.4.3). Attested assets are
produced by tagging `v0.4.4` (workflow `build-attested-binaries`).

---

SparkInfer lands **DFlash block-diffusion speculative decode** for Qwen3.6-35B-A3B — opt-in multi-token draft on the native GGUF path with greedy **SPEC_AGREE 100%** vs AR. Default generate stays autoregressive; set `SPARKINFER_DFLASH=1` with the z-lab draft weights to enable it.

Qwen3.6 **prefill also climbs again**: **+127% vs llama.cpp at 32k** (was +82.7% in v0.4.3). Continuous-batching **mixed-load TTFT drops ~97%** on the decode-first CB path.

## DFlash — the main story

| | |
|---|---|
| **What** | Block-diffusion draft (`z-lab/Qwen3.6-35B-A3B-DFlash`) + target UD-Q4_K_M GGUF |
| **Correctness** | Greedy **SPEC_AGREE 100%** vs AR |
| **Opt-in** | `SPARKINFER_DFLASH=1` when a draft is attached |
| **Tools** | `qwen3_gguf_dflash_check`, `qwen3_gguf_dflash_bench`, `bench/scripts/dflash_accuracy.sh` |

CB/server multi-accept is deferred until single-stream DFlash proves a tok/s win on top of SPEC_AGREE.

```bash
bench/scripts/dflash_accuracy.sh /path/to/Qwen3.6-35B-A3B.gguf /path/to/Qwen3.6-35B-A3B-DFlash
build/runtime/qwen3_gguf_dflash_bench target.gguf draft_dir 64 <token-ids...>
```

## Qwen3.6-35B-A3B · UD-Q4_K_M · RTX 5090

### Decode

| context | SparkInfer | llama.cpp | Δ |
|---:|---:|---:|---:|
| 128 | **512** tok/s | 276 tok/s | **+86%** |
| 512 | **506** tok/s | 276 tok/s | +83% |
| 4k | **486** tok/s | 276 tok/s | +76% |
| 16k | **467** tok/s | 281 tok/s | +66% |
| 32k | **437** tok/s | 280 tok/s | +56% |

### Prefill

| context | sparkinfer (pp tok/s) | llama.cpp (pp tok/s) | vs llama |
|---|---:|---:|---:|
| **4k prefill** | **~13,800** | 8,726 | **+58%** |
| **16k prefill** | **~17,700** | 8,390 | **+111%** |
| **32k prefill** | **~18,150** | 7,984 | **+127%** |

Headline: **32k prefill +127% vs llama.cpp** (was +82.7% in v0.4.3).

## Attested binaries

| Platform | Asset |
|---|---|
| Linux (sm_120) | `sparkinfer-v0.4.4-linux-x86_64-cuda13-sm120.tar.gz` |
| Windows (sm_120) | `sparkinfer-v0.4.4-windows-amd64-cuda13-sm120.zip` |

Each bundle: `sparkinfer-bin/{bin,lib}`, `BUILD_MANIFEST.json`, `SHA256SUMS`. Verified with GitHub Artifact Attestations.

```bash
gh attestation verify sparkinfer-bin/bin/qwen3_gguf_bench -R gittensor-ai-lab/sparkinfer
```

Bench scripts auto-fetch prebuilt tarballs: `bench/scripts/bench.sh --download`.

## Optimizations landed since v0.4.3

- **#633** — DFlash block-diffusion speculative decode for Qwen3.6 (opt-in) + draft KV fix
- **#621** — routed MoE prefill GEMM reads native quantized experts
- **#614** — tensor-core router / warp top-k / pipelined GDN / fused SwiGLU-quant (~+24% pp @32k)
- **#597** — decode-first CB + MoE batched prefill (~−96.6% CB mixed-load TTFT)
- **#595 / #582 / #583 / #577 / #609 / #598 / #549** — MoE FP8, tilemap, dequant, live-expert gather
- **#579 / #573** — Qwen3.5 long-ctx prefill (ldmatrix / fp8 GDN)
- **#608 / #604** — GDN chunk correctness fixes
- **#588 / #589 / #594 / #592 / #600 / #626** — eval TTFT tiers, CB gates, tokenizer guard

**Verified:** RTX 5090 · DFlash **SPEC_AGREE 100%** · Qwen3.6 decode **~512 tok/s @128 (+86%)** · prefill **~18,150 pp/s @32k (+127%)** · llama.cpp `6f4f53f`.

## Contributors

- **@skyrocket2026** — #633 (DFlash), eval CB/TTFT gates, tokenizer guard, bot cron/automerge
- **@widecloud** — #621, #614 (Qwen3.6 MoE / GDN prefill)
- **@Paral1995** — #597 (decode-first CB TTFT)
- **@fansilas** — #595, #582, #579, #573
- **@James-CUDA** — #583, #577
- **@inference2026** — #609, #598, #549
- **@RealDiligent** — #604

Full notes: [CHANGELOG.md](https://github.com/gittensor-ai-lab/sparkinfer/blob/v0.4.4/CHANGELOG.md)
