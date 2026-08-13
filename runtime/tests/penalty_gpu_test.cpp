// GPU correctness test for the presence_penalty/frequency_penalty kernels: the elementwise
// penalty application and the count-increment accumulator. Mirrors topk_topp_sample_gpu_test.cpp/
// logprob_extract_gpu_test.cpp's Scratch-struct-plus-direct-kernel-call structure, no model needed.
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using sparkinfer::kernels::launch_presence_frequency_penalty;
using sparkinfer::kernels::launch_increment_penalty_count;

struct Scratch {
    int vocab = 0;
    int* counts = nullptr;
    float* d_presence = nullptr;
    float* d_frequency = nullptr;
    int* d_out_id = nullptr;

    explicit Scratch(int v) : vocab(v) {
        cudaMalloc(&counts, vocab * sizeof(int));
        cudaMalloc(&d_presence, sizeof(float));
        cudaMalloc(&d_frequency, sizeof(float));
        cudaMalloc(&d_out_id, sizeof(int));
        cudaDeviceSynchronize();
    }
    ~Scratch() {
        cudaFree(counts); cudaFree(d_presence); cudaFree(d_frequency); cudaFree(d_out_id);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};

void set_counts(Scratch& s, const std::vector<int>& counts) {
    cudaMemcpy(s.counts, counts.data(), counts.size() * sizeof(int), cudaMemcpyHostToDevice);
}

std::vector<int> get_counts(Scratch& s) {
    std::vector<int> out(s.vocab);
    cudaMemcpy(out.data(), s.counts, s.vocab * sizeof(int), cudaMemcpyDeviceToHost);
    return out;
}

std::vector<float> apply_penalty(Scratch& s, std::vector<float> logits, float presence, float frequency) {
    const int vocab = (int)logits.size();
    float* d_logits = nullptr;
    cudaMalloc(&d_logits, vocab * sizeof(float));
    cudaMemcpy(d_logits, logits.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_presence, &presence, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_frequency, &frequency, sizeof(float), cudaMemcpyHostToDevice);
    launch_presence_frequency_penalty(d_logits, s.counts, vocab, s.d_presence, s.d_frequency);
    cudaMemcpy(logits.data(), d_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_logits);
    return logits;
}

void increment(Scratch& s, int chosen_id) {
    cudaMemcpy(s.d_out_id, &chosen_id, sizeof(int), cudaMemcpyHostToDevice);
    launch_increment_penalty_count(s.counts, s.d_out_id);
    cudaDeviceSynchronize();
}

bool nearly_equal(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) < tol; }

bool test_hand_computed_penalty_math() {
    const std::vector<int> counts = {0, 1, 0, 3, 2, 0, 1, 0};
    const std::vector<float> logits = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    Scratch s((int)counts.size());
    set_counts(s, counts);

    struct Case { float presence, frequency; };
    const std::vector<Case> cases = {{0.f, 0.f}, {1.f, 0.5f}, {-1.f, 0.f}, {0.5f, -0.5f}};
    bool ok = true;
    for (const auto& c : cases) {
        const auto out = apply_penalty(s, logits, c.presence, c.frequency);
        for (size_t v = 0; v < counts.size(); v++) {
            const float expect = logits[v] - (c.frequency * (float)counts[v] +
                                              c.presence * (counts[v] > 0 ? 1.f : 0.f));
            if (!nearly_equal(out[v], expect)) {
                printf("FAIL: presence=%.2f frequency=%.2f v=%zu got=%.6f expect=%.6f\n",
                       c.presence, c.frequency, v, out[v], expect);
                ok = false;
            }
        }
    }
    if (ok) printf("[OK] hand-computed penalty math matches for all (presence,frequency) cases\n");
    return ok;
}

bool test_increment_accumulates_across_repeated_launches() {
    const int vocab = 8;
    Scratch s(vocab);
    set_counts(s, std::vector<int>(vocab, 0));
    for (int id : {3, 3, 5, 3}) increment(s, id);
    const auto counts = get_counts(s);
    const std::vector<int> expect = {0, 0, 0, 3, 0, 1, 0, 0};
    const bool ok = counts == expect;
    if (!ok) {
        printf("FAIL: increment accumulation mismatch, got:");
        for (int c : counts) printf(" %d", c);
        printf("\n");
    } else {
        printf("[OK] increment kernel accumulates correctly across repeated launches\n");
    }
    return ok;
}

bool test_combined_increment_then_apply_pipeline() {
    // Interleave increment + apply across several synthetic "decode steps" -- the running-count-
    // across-many-steps property that is the whole point of this feature.
    const int vocab = 4;
    const std::vector<float> logits = {0.f, 0.f, 0.f, 0.f};
    Scratch s(vocab);
    set_counts(s, std::vector<int>(vocab, 0));

    increment(s, /*id_a=*/1);
    auto out1 = apply_penalty(s, logits, /*presence=*/1.f, /*frequency=*/1.f);
    // counts = {0,1,0,0}; id 1 penalized by presence(1)+frequency(1*1)=2, others untouched.
    bool ok = nearly_equal(out1[0], 0.f) && nearly_equal(out1[1], -2.f) &&
              nearly_equal(out1[2], 0.f) && nearly_equal(out1[3], 0.f);

    increment(s, /*id_b=*/2);
    auto out2 = apply_penalty(s, logits, /*presence=*/1.f, /*frequency=*/1.f);
    // counts now = {0,1,1,0}; id 1's count is STILL just 1 (only incremented once total).
    ok = ok && nearly_equal(out2[1], -2.f) && nearly_equal(out2[2], -2.f) && nearly_equal(out2[0], 0.f);

    increment(s, /*id_a again=*/1);
    auto out3 = apply_penalty(s, logits, /*presence=*/1.f, /*frequency=*/1.f);
    // counts now = {0,2,1,0}; id 1's penalty grows (frequency term), id 2's stays the same.
    ok = ok && nearly_equal(out3[1], -3.f) && nearly_equal(out3[2], -2.f);

    if (ok) printf("[OK] combined increment+apply pipeline evolves counts and penalties correctly\n");
    else printf("FAIL: combined pipeline mismatch\n");
    return ok;
}

bool test_reset_genuinely_zeros() {
    const int vocab = 16;
    Scratch s(vocab);
    std::vector<int> garbage(vocab);
    for (int i = 0; i < vocab; i++) garbage[i] = i + 7;
    set_counts(s, garbage);
    // Same cudaMemsetAsync-based reset pattern Qwen35Model::reset_penalty_counts uses.
    cudaMemsetAsync(s.counts, 0, vocab * sizeof(int));
    cudaDeviceSynchronize();
    const auto counts = get_counts(s);
    bool ok = true;
    for (int c : counts) if (c != 0) ok = false;
    if (ok) printf("[OK] memset-based reset genuinely zeros a previously-nonzero buffer\n");
    else printf("FAIL: reset did not fully zero the buffer\n");
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
    ok = test_hand_computed_penalty_math() && ok;
    ok = test_increment_accumulates_across_repeated_launches() && ok;
    ok = test_combined_increment_then_apply_pipeline() && ok;
    ok = test_reset_genuinely_zeros() && ok;
    if (!ok) return 1;
    printf("penalty_gpu_test: OK\n");
    return 0;
}
