#pragma once
#include <cuda_runtime.h>

// Dequantizes tensors from HuggingFace "compressed-tensors" format checkpoints (NVIDIA ModelOpt /
// llm-compressor convention -- the exact format used by e.g. unsloth/Qwen3.8-27B-NVFP4) straight
// to bf16, so the result can feed the SAME requantize-to-internal-format pipeline load_gguf()
// already uses (bf16 -> Q4_K for decode via launch_proj_requant_q4k_lloyd, bf16 -> this runtime's
// own fresh NVFP4 for prefill via the existing Muse Glimmer conversion path). One-directional,
// one-time, load-only conversion -- no new GEMM/GEMV kernels, no global-scale bookkeeping needed
// downstream, since the re-quantize step produces its own fresh scales the same way it already
// does for every other model.

namespace sparkinfer { namespace kernels {

// FP8 (E4M3) weight, per-output-channel (row) scale: out[r,c] = float(e4m3(w[r,c])) * float(scale[r])
// w: [rows,cols] raw e4m3 bytes, scale: [rows] bf16, out: [rows,cols] bf16.
void launch_ct_dequant_fp8(const void* w_e4m3, const void* scale_bf16, void* out_bf16,
                           int rows, int cols, cudaStream_t stream = nullptr);

// NVFP4 (E2M1), block_size=16 group scale (UE4M3) + a single tensor-wide F32 global scale:
//   out[r,c] = float(e2m1(packed nibble)) * float(ue4m3(group_scale[r, c/16])) * global_scale
// packed: [rows, cols/2] U8 (2 nibbles/byte, low nibble = even column), group_scale: [rows,
// cols/16] raw ue4m3 bytes (note: tagged F8_E4M3 in the safetensors header, but interpreted as
// CUTLASS's unsigned e4m3 -- the standard NVIDIA NVFP4 export convention; see the loader's own
// comment for why), out: [rows,cols] bf16. cols must be a multiple of 16.
void launch_ct_dequant_nvfp4(const void* packed_u8, const void* group_scale_ue4m3,
                             float global_scale, void* out_bf16, int rows, int cols,
                             cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
