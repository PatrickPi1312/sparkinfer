// top_k / top_p (nucleus) truncation sampling: a full descending sort of the vocab-sized logits
// row (CUB radix sort, keys=logits values=vocab-index) + cumulative-softmax scan (CUB inclusive
// sum) to find the top_p cutoff, then a mask+scatter kernel that writes -infinity back onto the
// original vocab-indexed logits array for everything outside the surviving set. See fused.h for
// the full design rationale (why this always processes the whole vocab regardless of top_k/top_p,
// and why it must always be launched unconditionally on a CUDA-graph-captured decode path).

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "sparkinfer/kernels/fused.h"

namespace sparkinfer {
namespace kernels {

namespace {

__global__ void vocab_iota_kernel(int* vocab_iota, int vocab) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab; v += gridDim.x * blockDim.x)
        vocab_iota[v] = v;
}

// sorted_exp[i] = exp(sorted_logits[i] - sorted_logits[0]) -- sorted_logits[0] is the row max
// (descending sort), so this is a numerically-stable softmax numerator with no separate
// max-reduction pass needed.
__global__ void topk_topp_exp_kernel(const float* __restrict__ sorted_logits,
                                     float* __restrict__ sorted_exp, int vocab) {
    const float row_max = sorted_logits[0];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += gridDim.x * blockDim.x)
        sorted_exp[i] = __expf(sorted_logits[i] - row_max);
}

// One block; grid-stride over sorted rank. Reads the per-call top_k/top_p device scalars and
// scatters -infinity into the ORIGINAL vocab-indexed logits array (via sorted_idx) for every
// rank outside the surviving set. Rank 0 is always kept by construction -- see fused.h.
__global__ void topk_topp_mask_kernel(float* __restrict__ logits,
                                      const int* __restrict__ sorted_idx,
                                      const float* __restrict__ topk_cumsum,
                                      int vocab, const int* __restrict__ top_k_i32,
                                      const float* __restrict__ top_p_f32) {
    int effective_k = *top_k_i32;
    if (effective_k <= 0 || effective_k >= vocab) effective_k = vocab;
    if (effective_k < 1) effective_k = 1;  // defensive floor -- never trust the input blindly

    const float top_p = *top_p_f32;
    const bool top_p_active = top_p > 0.f && top_p < 1.f;
    const float total_topk = topk_cumsum[effective_k - 1];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += gridDim.x * blockDim.x) {
        bool keep;
        if (i >= effective_k) {
            keep = false;
        } else if (i == 0 || !top_p_active) {
            keep = true;
        } else {
            keep = topk_cumsum[i - 1] < top_p * total_topk;
        }
        if (!keep) logits[sorted_idx[i]] = -INFINITY;
    }
}

}  // namespace

size_t topk_sort_temp_storage_bytes(int vocab) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending<float, int>(
        nullptr, bytes, nullptr, nullptr, nullptr, nullptr, vocab);
    return bytes;
}

size_t topk_scan_temp_storage_bytes(int vocab) {
    size_t bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<float*>(nullptr),
                                  static_cast<float*>(nullptr), vocab);
    return bytes;
}

void launch_vocab_iota_init(int* vocab_iota, int vocab, cudaStream_t stream) {
    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    vocab_iota_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(vocab_iota, vocab);
}

void launch_topk_topp_mask(float* logits, int vocab,
                           const int* vocab_iota, float* sorted_logits, int* sorted_idx,
                           float* topk_exp, float* topk_cumsum,
                           void* sort_temp, size_t sort_temp_bytes,
                           void* scan_temp, size_t scan_temp_bytes,
                           const int* top_k_i32, const float* top_p_f32,
                           cudaStream_t stream) {
    size_t sort_bytes = sort_temp_bytes;
    cub::DeviceRadixSort::SortPairsDescending<float, int>(
        sort_temp, sort_bytes, logits, sorted_logits, vocab_iota, sorted_idx, vocab, 0,
        sizeof(float) * 8, stream);

    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    topk_topp_exp_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(sorted_logits, topk_exp, vocab);

    size_t scan_bytes = scan_temp_bytes;
    cub::DeviceScan::InclusiveSum(scan_temp, scan_bytes, topk_exp, topk_cumsum, vocab, stream);

    topk_topp_mask_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(
        logits, sorted_idx, topk_cumsum, vocab, top_k_i32, top_p_f32);
}

}  // namespace kernels
}  // namespace sparkinfer
