// ModelOpt NVFP4 (e2m1 + fp8-e4m3 group scales) dequant and decode GEMV.
// Packed layout matches NVIDIA ModelOpt: uint8 [n_out, n_in/2], low nibble = even K.

#include "sparkinfer/kernels/nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace sparkinfer {
namespace kernels {

namespace {

struct Hdr {
    int32_t n_out;
    int32_t n_in;
    float   scale2;
    int32_t _pad;
};

__device__ __forceinline__ float e2m1_f(int nib) {
    // 0,0.5,1,1.5,2,3,4,6 and negatives — float4_e2m1fn
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = mag[nib & 7];
    return (nib & 8) ? -v : v;
}

__device__ __forceinline__ float fp8e4m3_f(unsigned char u) {
    const int s = u >> 7;
    const int e = (u >> 3) & 0xF;
    const int m = u & 7;
    float v;
    if (e == 0) v = (m == 0) ? 0.f : ((float)m * 0.001953125f);
    else if (e == 15 && m == 7) v = 0.f;
    else {
        v = 1.f + (float)m * 0.125f;
        const int p = e - 7;
        if (p > 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++) if (i < p) v += v;
        } else if (p < 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++) if (i < -p) v *= 0.5f;
        }
    }
    return s ? -v : v;
}

__global__ void nvfp4_dequant_kernel(const unsigned char* __restrict__ packed,
                                     const unsigned char* __restrict__ scales,
                                     float scale2, __nv_bfloat16* __restrict__ out,
                                     int n_out, int n_in) {
    const int ng = n_in >> 4;
    const size_t n_vals = (size_t)n_out * (size_t)n_in;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vals;
         i += (size_t)gridDim.x * blockDim.x) {
        const int row = (int)(i / (size_t)n_in);
        const int col = (int)(i - (size_t)row * (size_t)n_in);
        const unsigned char b = packed[(size_t)row * (n_in >> 1) + (col >> 1)];
        const int nib = (col & 1) ? (b >> 4) : (b & 0xF);
        const float sf = fp8e4m3_f(scales[(size_t)row * ng + (col >> 4)]) * scale2;
        out[i] = __float2bfloat16(e2m1_f(nib) * sf);
    }
}

static constexpr int GEMV_WPB = 8;

__device__ __forceinline__ void gemv_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void gemv_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename OutT>
__global__ void gemv_nvfp4_kernel(const __nv_bfloat16* __restrict__ x,
                                  const unsigned char* __restrict__ packed,
                                  const unsigned char* __restrict__ scales,
                                  float scale2, OutT* __restrict__ y,
                                  int N, int K) {
    extern __shared__ float s_x[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) s_x[i] = __bfloat162float(x[i]);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GEMV_WPB + warp;
    if (n >= N) return;

    const int ng = K >> 4;
    const unsigned char* row_pk = packed + (size_t)n * (K >> 1);
    const unsigned char* row_sc = scales + (size_t)n * ng;
    float acc = 0.f;
    for (int g = lane; g < ng; g += 32) {
        const float sf = fp8e4m3_f(row_sc[g]) * scale2;
        const unsigned char* pk = row_pk + g * 8;
        const int base = g << 4;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const unsigned char b = pk[i >> 1];
            const int nib = (i & 1) ? (b >> 4) : (b & 0xF);
            acc += e2m1_f(nib) * sf * s_x[base + i];
        }
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, m);
    if (lane == 0) gemv_store(y + n, acc);
}

const unsigned char* blob_packed(const Hdr* h) {
    return reinterpret_cast<const unsigned char*>(h + 1);
}
const unsigned char* blob_scales(const Hdr* h) {
    return blob_packed(h) + (size_t)h->n_out * ((size_t)h->n_in / 2);
}

void set_gemv_smem() {
    static int once = 0;
    if (once) return;
    cudaFuncSetAttribute(gemv_nvfp4_kernel<__nv_bfloat16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    cudaFuncSetAttribute(gemv_nvfp4_kernel<float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    once = 1;
}

} // namespace

void launch_nvfp4_dequant(const void* blob, void* out_bf16, int n_out, int n_in,
                          cudaStream_t stream) {
    const Hdr* h = reinterpret_cast<const Hdr*>(blob);
    const size_t n = (size_t)n_out * (size_t)n_in;
    const int blocks = (int)((n + 255) / 256);
    nvfp4_dequant_kernel<<<blocks, 256, 0, stream>>>(
        blob_packed(h), blob_scales(h), h->scale2,
        reinterpret_cast<__nv_bfloat16*>(out_bf16), n_out, n_in);
}

void launch_gemv_nvfp4(const void* x, const void* blob, void* y, int N, int K,
                       cudaStream_t stream) {
    const Hdr* h = reinterpret_cast<const Hdr*>(blob);
    const int n = h->n_out > 0 ? h->n_out : N;
    const int k = h->n_in > 0 ? h->n_in : K;
    set_gemv_smem();
    const int blocks = (n + GEMV_WPB - 1) / GEMV_WPB;
    const size_t smem = (size_t)k * sizeof(float);
    gemv_nvfp4_kernel<__nv_bfloat16><<<blocks, GEMV_WPB * 32, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), blob_packed(h), blob_scales(h),
        h->scale2, reinterpret_cast<__nv_bfloat16*>(y), n, k);
}

void launch_gemv_nvfp4_f32(const void* x, const void* blob, float* y, int N, int K,
                           cudaStream_t stream) {
    const Hdr* h = reinterpret_cast<const Hdr*>(blob);
    const int n = h->n_out > 0 ? h->n_out : N;
    const int k = h->n_in > 0 ? h->n_in : K;
    set_gemv_smem();
    const int blocks = (n + GEMV_WPB - 1) / GEMV_WPB;
    const size_t smem = (size_t)k * sizeof(float);
    gemv_nvfp4_kernel<float><<<blocks, GEMV_WPB * 32, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), blob_packed(h), blob_scales(h),
        h->scale2, y, n, k);
}

} // namespace kernels
} // namespace sparkinfer
