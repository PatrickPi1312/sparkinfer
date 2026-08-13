// Gumbel-max temperature sampling: mutates logits in place before argmax, using cuRAND's
// Philox4_32_10 device API (header-only, stateless-per-call -- no persistent curandState needed
// across CUDA graph replays; the caller re-derives the same draw from (seed, row*vocab+v, step)
// every launch). See fused.h for the full design rationale (why temp/seed/step are read from
// device memory on every launch rather than gating the kernel launch itself on temperature).
//
// Sampling from softmax(logits/T) is equivalent to argmax_v(logits[v]/T + G_v), G_v =
// -log(-log(U_v)), U_v ~ Uniform(0,1] i.i.d. per vocab entry (the standard Gumbel-max trick) --
// so this kernel is a thin elementwise transform feeding the existing, unmodified launch_argmax,
// mirroring how launch_logit_softcap already mutates logits in place immediately before argmax.

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "sparkinfer/kernels/fused.h"

namespace sparkinfer {
namespace kernels {

// One block per row, grid-stride over vocab -- same layout as logit_softcap_kernel.
__global__ void temperature_sample_kernel(float* __restrict__ logits, int vocab,
                                          const float* __restrict__ temp_f32,
                                          const unsigned long long* __restrict__ seed_u64,
                                          const unsigned long long* __restrict__ step_u64) {
    const float T = *temp_f32;
    if (T <= 0.f) return;  // greedy: leave logits untouched, byte-identical to plain argmax

    const unsigned long long seed = *seed_u64;
    const unsigned long long step = *step_u64;
    float* L = logits + (size_t)blockIdx.y * vocab;
    const float inv_t = 1.f / T;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab; v += gridDim.x * blockDim.x) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, (unsigned long long)((size_t)blockIdx.y * vocab + v), step, &st);
        const float u = curand_uniform(&st);  // (0, 1]
        const float g = -logf(-logf(u));      // Gumbel(0,1); u==1 gives a well-defined -inf, never wins
        L[v] = L[v] * inv_t + g;
    }
}

void launch_temperature_sample(float* logits, int n_rows, int vocab,
                               const float* temp_f32, const unsigned long long* seed_u64,
                               const unsigned long long* step_u64, cudaStream_t stream) {
    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    temperature_sample_kernel<<<dim3(bx < 1 ? 1 : bx, n_rows), 256, 0, stream>>>(
        logits, vocab, temp_f32, seed_u64, step_u64);
}

}  // namespace kernels
}  // namespace sparkinfer
