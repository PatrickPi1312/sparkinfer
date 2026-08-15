#pragma once
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// Dequant a SparkInfer NVFP4 blob (see nvfp4_format.h) to bf16 [n_out, n_in].
void launch_nvfp4_dequant(const void* blob, void* out_bf16, int n_out, int n_in,
                          cudaStream_t stream = nullptr);

// Decode GEMV: y[N] = x[K] @ W^T with W an NVFP4 blob, x/y bf16 (or y fp32).
void launch_gemv_nvfp4(const void* x, const void* blob, void* y, int N, int K,
                       cudaStream_t stream = nullptr);
void launch_gemv_nvfp4_f32(const void* x, const void* blob, float* y, int N, int K,
                           cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
