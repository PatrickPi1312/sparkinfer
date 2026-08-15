# Qwen3.8-27B on SparkInfer

Branch: `qwen-3-8-27b`, cut from `qwen-3-6-27b`.

Qwen3.8-27B is the same dense hybrid as Qwen3.6-27B: 64 layers, hidden 5120,
FFN 17408, vocab 248320, 16×(3 Gated DeltaNet + 1 gated attention), linear
attention 16 Q/K × 48 V @ 128, full attention 24 Q / 4 KV @ 256 with RoPE 64.

## Native NVFP4 (this branch)

SparkInfer loads [gittensor-model-hub/Qwen3.8-27B-NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4)
directly: ModelOpt packed e2m1 weights + FP8 E4M3 group-16 scales + FP32
`weight_scale_2`. Decode keeps those weights packed and runs an on-read NVFP4
GEMV (W4A16). Prefill dequants a matrix to BF16 scratch and uses the existing
hybrid GEMM / GDN scan. Embeddings, norms, GDN `A_log`/`dt`/`in_proj_a,b`, and
the LM head stay BF16 as in the checkpoint (`lm_head` is in the exclude list).

```bash
# after cmake --build
build/qwen3_gguf_bench /path/to/Qwen3.8-27B-NVFP4 128 512
build/qwen3_gguf_generate /path/to/Qwen3.8-27B-NVFP4 32 248044
./server/run.sh   # MODEL= path to the HF directory
```

This is **not** vLLM's Blackwell W4A4 tensor-core GEMM. Activations stay BF16
at decode; CUTLASS/cuBLASLt NVFP4 MMA is the next kernel job if we need the
vLLM-class batch-32/64 numbers inside SparkInfer.

## GGUF still works

The 3.6-27B parser also loads a Qwen3.8 GGUF on the same kernels:

```bash
build/qwen3_gguf_bench /path/to/Qwen3.8-27B-Q4_K_M.gguf 128 512
```

## vLLM reference (same checkpoint, W4A4)

```bash
vllm serve gittensor-model-hub/Qwen3.8-27B-NVFP4 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --reasoning-parser qwen3
```

Reference decode on 1× RTX PRO 6000 Blackwell (vLLM 0.27.1, 2026-08-14):
67.2 / 476.6 / 1448.6 / 2152.4 total tok/s at batch 1 / 8 / 32 / 64.
