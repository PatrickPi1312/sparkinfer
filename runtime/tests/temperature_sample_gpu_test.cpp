// GPU statistical correctness test for launch_temperature_sample (Gumbel-max sampling).
// Calls the kernel directly against a small synthetic logits buffer -- no model needed, since
// the sampler is a pure elementwise transform feeding the existing launch_argmax reduction.
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace {

// Draws N samples for the given (temperature, seed base, logits) and returns per-index counts.
std::vector<int> sample_many(const std::vector<float>& logits, float temperature, unsigned long long seed_base, int n) {
    const int vocab = (int)logits.size();
    float* d_logits = nullptr;
    int* d_out = nullptr;
    float* d_temp = nullptr;
    unsigned long long *d_seed = nullptr, *d_step = nullptr;
    cudaMalloc(&d_logits, vocab * sizeof(float));
    cudaMalloc(&d_out, sizeof(int));
    cudaMalloc(&d_temp, sizeof(float));
    cudaMalloc(&d_seed, sizeof(unsigned long long));
    cudaMalloc(&d_step, sizeof(unsigned long long));
    cudaMemcpy(d_temp, &temperature, sizeof(float), cudaMemcpyHostToDevice);

    std::vector<int> counts(vocab, 0);
    for (int i = 0; i < n; ++i) {
        cudaMemcpy(d_logits, logits.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
        unsigned long long seed = seed_base;
        // Distinct step per draw -- matches real usage (a fresh Philox offset per decode step),
        // not the same offset resampled.
        unsigned long long step = (unsigned long long)i;
        cudaMemcpy(d_seed, &seed, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_step, &step, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        sparkinfer::kernels::launch_temperature_sample(d_logits, 1, vocab, d_temp, d_seed, d_step);
        sparkinfer::kernels::launch_argmax(d_logits, d_out, 1, vocab);
        int out = -1;
        cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
        counts[out]++;
    }
    cudaFree(d_logits); cudaFree(d_out); cudaFree(d_temp); cudaFree(d_seed); cudaFree(d_step);
    return counts;
}

bool test_temperature_zero_is_argmax_noop() {
    const std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.5f};
    const int vocab = (int)logits.size();
    // Reference: plain launch_argmax on untouched logits.
    float* d_ref = nullptr; int* d_ref_out = nullptr;
    cudaMalloc(&d_ref, vocab * sizeof(float));
    cudaMalloc(&d_ref_out, sizeof(int));
    cudaMemcpy(d_ref, logits.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    sparkinfer::kernels::launch_argmax(d_ref, d_ref_out, 1, vocab);
    int ref_out = -1;
    cudaMemcpy(&ref_out, d_ref_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_ref); cudaFree(d_ref_out);

    const auto counts = sample_many(logits, /*temperature=*/0.f, /*seed_base=*/123, /*n=*/2000);
    for (int v = 0; v < vocab; ++v) {
        if (v == ref_out) { if (counts[v] != 2000) { printf("FAIL: T=0 not a no-op\n"); return false; } }
        else if (counts[v] != 0) { printf("FAIL: T=0 not a no-op (spurious index %d)\n", v); return false; }
    }
    printf("[OK] temperature=0 is a byte-identical argmax no-op\n");
    return true;
}

bool test_determinism_given_seed_and_step() {
    const std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.5f, -1.0f};
    const auto a = sample_many(logits, 0.8f, 999, 500);
    const auto b = sample_many(logits, 0.8f, 999, 500);
    if (a != b) { printf("FAIL: same (seed,temperature) gave different outcomes across runs\n"); return false; }
    printf("[OK] same seed/temperature/logits -> identical outcome sequence\n");
    return true;
}

bool test_distribution_matches_softmax() {
    const std::vector<float> logits = {2.0f, 1.0f, 0.0f, -1.0f};
    const float T = 1.0f;
    const int N = 200000;
    const auto counts = sample_many(logits, T, 42, N);

    std::vector<double> expected(logits.size());
    double denom = 0.0;
    for (float l : logits) denom += std::exp((double)l / T);
    for (size_t i = 0; i < logits.size(); ++i) expected[i] = std::exp((double)logits[i] / T) / denom;

    bool ok = true;
    for (size_t i = 0; i < logits.size(); ++i) {
        const double p = expected[i];
        const double empirical = (double)counts[i] / N;
        // Normal-approximation 3-sigma tolerance around the expected binomial proportion.
        const double sigma = std::sqrt(p * (1.0 - p) / N);
        const double tol = std::max(3.0 * sigma, 0.003);  // floor avoids a false fail at tiny p
        if (std::fabs(empirical - p) > tol) {
            printf("FAIL: index %zu expected=%.4f empirical=%.4f tol=%.4f\n", i, p, empirical, tol);
            ok = false;
        }
    }
    if (ok) printf("[OK] empirical distribution matches softmax(logits/T) within tolerance (N=%d)\n", N);
    return ok;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device\n");
        return 0;
    }
    bool ok = true;
    ok = test_temperature_zero_is_argmax_noop() && ok;
    ok = test_determinism_given_seed_and_step() && ok;
    ok = test_distribution_matches_softmax() && ok;
    if (!ok) return 1;
    printf("temperature_sample_gpu_test: OK\n");
    return 0;
}
