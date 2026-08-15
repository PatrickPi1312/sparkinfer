# Qwen3.8-27B on SparkInfer

Branch: `qwen-3-8-27b`, cut from `qwen-3-6-27b`.

Qwen3.8-27B is the same dense hybrid as Qwen3.6-27B: 64 layers, hidden 5120,
FFN 17408, vocab 248320, 16×(3 Gated DeltaNet + 1 gated attention), linear
attention 16 Q/K × 48 V @ 128, full attention 24 Q / 4 KV @ 256 with RoPE 64.
The GGUF parser already derives the 16-vs-24 head split from `attn_qkv` /
`attn_gate` shapes, so a Qwen3.8-27B GGUF loads on the existing 27B path.

## What runs today

Native SparkInfer decode is **GGUF**:

```bash
# after cmake --build
build/qwen3_gguf_bench /path/to/Qwen3.8-27B-Q4_K_M.gguf 128 512
```

Use [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
or an equivalent `qwen35` dense-hybrid GGUF.

## What does not load yet: our NVFP4

[gittensor-model-hub/Qwen3.8-27B-NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4)
is NVIDIA ModelOpt W4A4 (~20.6 GB safetensors + FP8 KV). That is the vLLM /
SGLang path on Blackwell, not GGUF.

SparkInfer's 27B kernels dequant GGUF blocks into BF16 and GEMM/GEMV in BF16.
There is no ModelOpt loader and no NVFP4 tensor-core GEMM in this runtime.
Dropping the HF repo into `qwen3_gguf_bench` will fail at open.

Serve NVFP4 today with vLLM:

```bash
vllm serve gittensor-model-hub/Qwen3.8-27B-NVFP4 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --reasoning-parser qwen3
```

Reference decode on 1× RTX PRO 6000 Blackwell (vLLM 0.27.1, 2026-08-14):
67.2 / 476.6 / 1448.6 / 2152.4 total tok/s at batch 1 / 8 / 32 / 64.

## Work to run NVFP4 inside SparkInfer

Tracked in `bench/configs/targets/qwen38_27b_nvfp4_rtx_pro_6000.yaml`:

1. Read ModelOpt / `compressed-tensors` NVFP4 shards (group size 16, FP8 scales).
2. Blackwell W4A4 GEMM (CUTLASS / FlashInfer-class), not Marlin W4A16.
3. FP8 KV in the hybrid GDN + full-attention cache.
4. Prompt-level parity vs the vLLM serve of the same checkpoint.
