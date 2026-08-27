#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/kernels/compressed_tensors.h"

#include <cuda_bf16.h>

// Linkable fallbacks keep the runtime build independent of CUTLASS. The SM120 implementation
// replaces these symbols when BUILD_NVFP4_KERNELS is enabled for si_fused.
#ifndef SPARKINFER_BUILD_NVFP4
namespace sparkinfer::kernels {
namespace {

// CUTLASS-free decoders used when the native SM120 block-scaled kernels are disabled.  The
// encodings are the same ones used by the native implementation: signed E2M1 nibbles and
// unsigned E4M3 group scales.
__device__ __forceinline__ float fallback_e2m1(unsigned n) {
    const unsigned mag_x2 = (0xC8643210u >> ((n & 7u) << 2)) & 15u;
    const float value = 0.5f * __uint2float_rn(mag_x2);
    return __int_as_float(__float_as_int(value) | ((n & 8u) << 28));
}

__device__ __forceinline__ float fallback_ue4m3(unsigned b) {
    const unsigned e = (b >> 3) & 15u;
    const unsigned m = b & 7u;
    if (e == 0) return static_cast<float>(m) * (1.f / 512.f);
    return __int_as_float(static_cast<int>(((e + 120u) << 23) | (m << 20)));
}

__global__ void fallback_ct_dequant_nvfp4(
        const unsigned char* __restrict__ packed,
        const unsigned char* __restrict__ group_scale,
        float global_scale,
        const float* __restrict__ global_scale_dev,
        __nv_bfloat16* __restrict__ out,
        int rows, int cols) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * cols;
    if (i >= count) return;
    if (global_scale_dev) global_scale = *global_scale_dev;
    const int r = static_cast<int>(i / cols);
    const int c = static_cast<int>(i - static_cast<size_t>(r) * cols);
    const unsigned char byte = packed[static_cast<size_t>(r) * (cols / 2) + (c >> 1)];
    const unsigned nibble = (c & 1) ? (byte >> 4) : (byte & 0xFu);
    const unsigned scale = group_scale[static_cast<size_t>(r) * (cols / 16) + (c >> 4)];
    out[i] = __float2bfloat16(fallback_e2m1(nibble) * fallback_ue4m3(scale) / global_scale);
}

void launch_fallback_ct_dequant_nvfp4(
        const void* packed_u8, const void* group_scale_ue4m3,
        float global_scale, const float* global_scale_dev,
        void* out_bf16, int rows, int cols, cudaStream_t stream) {
    if (!packed_u8 || !group_scale_ue4m3 || !out_bf16 || rows <= 0 || cols <= 0) return;
    const size_t count = static_cast<size_t>(rows) * cols;
    const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
    fallback_ct_dequant_nvfp4<<<blocks, 256, 0, stream>>>(
        static_cast<const unsigned char*>(packed_u8),
        static_cast<const unsigned char*>(group_scale_ue4m3),
        global_scale, global_scale_dev, static_cast<__nv_bfloat16*>(out_bf16), rows, cols);
}

} // namespace

bool prefill_nvfp4_supported(int, int, int) { return false; }
size_t prefill_nvfp4_data_bytes(int r, int c) { return ((size_t)r * c + 1) / 2; }
size_t prefill_nvfp4_scale_bytes_a(int, int) { return 0; }
size_t prefill_nvfp4_scale_bytes_b(int, int) { return 0; }
size_t prefill_nvfp4_workspace_bytes(int, int, int) { return 0; }
bool launch_prefill_nvfp4_quant_a(const void*, void*, void*, int, int, cudaStream_t) { return false; }
bool launch_prefill_nvfp4_gate_quant_a(const void*, const void*, void*, void*, int, int,
                                       cudaStream_t) { return false; }
bool launch_prefill_nvfp4_swiglu_quant_a(const void*, const void*, void*, void*, int, int,
                                         cudaStream_t) { return false; }
bool launch_prefill_nvfp4_quant_b(const void*, void*, void*, int, int, cudaStream_t) { return false; }
bool launch_prefill_nvfp4_gemm(const void*, const void*, const void*, const void*, void*, int, int,
                               int, void*, cudaStream_t, float, const void*) { return false; }
bool launch_ct_nvfp4_pack_sfb(const void*, void*, int, int, cudaStream_t) { return false; }

void launch_ct_dequant_nvfp4(const void* packed_u8, const void* group_scale_ue4m3,
                             float global_scale, void* out_bf16, int rows, int cols,
                             cudaStream_t stream) {
    launch_fallback_ct_dequant_nvfp4(packed_u8, group_scale_ue4m3, global_scale, nullptr,
                                     out_bf16, rows, cols, stream);
}

void launch_ct_dequant_nvfp4_dev(const void* packed_u8, const void* group_scale_ue4m3,
                                 const float* global_scale_dev, void* out_bf16,
                                 int rows, int cols, cudaStream_t stream) {
    launch_fallback_ct_dequant_nvfp4(packed_u8, group_scale_ue4m3, 1.f, global_scale_dev,
                                     out_bf16, rows, cols, stream);
}

bool launch_ct_dequant_nvfp4_rows_i8(const void*, const void*, const float*, signed char*, float*,
                                     int, int, cudaStream_t) {
    return false;
}
} // namespace sparkinfer::kernels
#endif
