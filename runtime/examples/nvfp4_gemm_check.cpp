// Is the SM120 native NVFP4 GEMM correct at the shapes each model actually asks for?
//
// Model-free. Builds random bf16 A[m,k] and W[n,k], runs them through the exact prefill sequence
// (launch_prefill_nvfp4_quant_a + launch_prefill_nvfp4_quant_b + launch_prefill_nvfp4_gemm) and
// compares against a plain bf16 GEMM on the SAME unquantized operands. NVFP4 is a 4-bit format
// with a 16-wide block scale, so a correct kernel lands a few percent off the bf16 reference; a
// broken one is not close at all. The bar below is set from that gap, not from exactness.
//
// Why this exists: batched prefill leaves the Qwen3.8 compressed-tensors state wrong, and the leg
// responsible is this GEMM sequence -- but Muse Glimmer drives the SAME call sites correctly. The
// two differ only in shape (Muse ffn/hidden 19968/6656, Qwen3.8 17408/5120), so the question is
// whether the kernel is shape-dependent. Testing it here rather than through a 27B model removes
// the loader, the KV cache, the GDN recurrence and 64 layers of accumulation from the loop.
//
// Usage: nvfp4_gemm_check [m]        (default: sweeps m = 8, 32, 128, 512)

#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/kernels/gemm.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using sparkinfer::kernels::GemmConfig;
using sparkinfer::kernels::GemmLayout;

struct Shape { const char* name; int n; int k; };

static bool run_case(const Shape& s, int m, double& rel_out) {
    const int n = s.n, k = s.k;
    if (!sparkinfer::kernels::prefill_nvfp4_supported(m, n, k)) {
        printf("  %-22s m=%-5d UNSUPPORTED (m%%8=%d n%%128=%d k%%128=%d)\n",
               s.name, m, m & 7, n & 127, k & 127);
        return true;
    }
    std::mt19937 rng(1234u + m + n);
    std::normal_distribution<float> nd(0.f, 0.02f);   // weight-like magnitudes
    std::vector<__nv_bfloat16> hA((size_t)m * k), hW((size_t)n * k);
    for (auto& v : hA) v = __float2bfloat16(nd(rng));
    for (auto& v : hW) v = __float2bfloat16(nd(rng));

    void *dA = nullptr, *dW = nullptr, *dWt = nullptr, *dRef = nullptr, *dOut = nullptr;
    void *qa = nullptr, *sa = nullptr, *qb = nullptr, *sb = nullptr, *ws = nullptr;
    auto ok = [](cudaError_t e) { return e == cudaSuccess; };
    bool alloc = ok(cudaMalloc(&dA, hA.size() * 2)) && ok(cudaMalloc(&dW, hW.size() * 2)) &&
                 ok(cudaMalloc(&dWt, hW.size() * 2)) &&
                 ok(cudaMalloc(&dRef, (size_t)m * n * 2)) && ok(cudaMalloc(&dOut, (size_t)m * n * 2)) &&
                 ok(cudaMalloc(&qa, sparkinfer::kernels::prefill_nvfp4_data_bytes(m, k))) &&
                 ok(cudaMalloc(&sa, sparkinfer::kernels::prefill_nvfp4_scale_bytes_a(m, k))) &&
                 ok(cudaMalloc(&qb, sparkinfer::kernels::prefill_nvfp4_data_bytes(n, k))) &&
                 ok(cudaMalloc(&sb, sparkinfer::kernels::prefill_nvfp4_scale_bytes_b(n, k)));
    const size_t wsb = sparkinfer::kernels::prefill_nvfp4_workspace_bytes(m, n, k);
    if (alloc && wsb) alloc = ok(cudaMalloc(&ws, wsb));
    if (!alloc) { printf("  %-22s m=%-5d ALLOC FAILED\n", s.name, m); return false; }
    cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice);

    // Reference: C[m,n] = A[m,k] @ W[n,k]^T. launch_gemm's B is [K,N], and COL_MAJOR B is exactly
    // the [n,k] row-major buffer read as its transpose -- the same operand the NVFP4 path takes.
    GemmConfig cfg;
    cfg.layout_a = GemmLayout::ROW_MAJOR;
    cfg.layout_b = GemmLayout::COL_MAJOR;
    sparkinfer::kernels::launch_gemm(dA, dW, dRef, m, n, k, 1.f, 0.f, cfg, nullptr);

    const bool qok = sparkinfer::kernels::launch_prefill_nvfp4_quant_a(dA, qa, sa, m, k, nullptr) &&
                     sparkinfer::kernels::launch_prefill_nvfp4_quant_b(dW, qb, sb, n, k, nullptr) &&
                     sparkinfer::kernels::launch_prefill_nvfp4_gemm(qa, sa, qb, sb, dOut,
                                                                    m, n, k, ws, nullptr, 1.f);
    cudaDeviceSynchronize();
    if (!qok) { printf("  %-22s m=%-5d NVFP4 SEQUENCE RETURNED FALSE\n", s.name, m); return false; }

    std::vector<__nv_bfloat16> hRef((size_t)m * n), hOut((size_t)m * n);
    cudaMemcpy(hRef.data(), dRef, hRef.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOut.data(), dOut, hOut.size() * 2, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t i = 0; i < hRef.size(); i++) {
        const double r = __bfloat162float(hRef[i]), o = __bfloat162float(hOut[i]);
        num += (r - o) * (r - o); den += r * r;
    }
    const double rel = std::sqrt(num / (den > 0 ? den : 1));
    rel_out = rel;
    printf("  %-22s m=%-5d n=%-6d k=%-5d  rel_err=%.4f  %s\n",
           s.name, m, n, k, rel, rel < 0.15 ? "ok" : "*** WRONG ***");
    for (void* p : {dA, dW, dWt, dRef, dOut, qa, sa, qb, sb, ws}) if (p) cudaFree(p);
    return true;
}

// Self-test: at a tiny shape, check BOTH the bf16 reference and the NVFP4 sequence against a CPU
// matmul. Without this the harness cannot tell "the kernel is wrong" from "I drove the reference
// GEMM with the wrong operand layout" -- and those two look identical in the output.
static void cpu_selftest() {
    const int m = 8, n = 128, k = 128;
    std::mt19937 rng(7u);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<__nv_bfloat16> hA((size_t)m * k), hW((size_t)n * k);
    std::vector<float> fA(hA.size()), fW(hW.size());
    for (size_t i = 0; i < hA.size(); i++) { fA[i] = nd(rng); hA[i] = __float2bfloat16(fA[i]); }
    for (size_t i = 0; i < hW.size(); i++) { fW[i] = nd(rng); hW[i] = __float2bfloat16(fW[i]); }
    std::vector<double> cpu((size_t)m * n, 0.0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0;
            for (int t = 0; t < k; t++)
                acc += (double)__bfloat162float(hA[(size_t)i * k + t]) *
                       (double)__bfloat162float(hW[(size_t)j * k + t]);
            cpu[(size_t)i * n + j] = acc;
        }
    void *dA = nullptr, *dW = nullptr, *dRef = nullptr, *dOut = nullptr;
    void *qa = nullptr, *sa = nullptr, *qb = nullptr, *sb = nullptr, *ws = nullptr;
    cudaMalloc(&dA, hA.size() * 2); cudaMalloc(&dW, hW.size() * 2);
    cudaMalloc(&dRef, (size_t)m * n * 2); cudaMalloc(&dOut, (size_t)m * n * 2);
    cudaMalloc(&qa, sparkinfer::kernels::prefill_nvfp4_data_bytes(m, k));
    cudaMalloc(&sa, sparkinfer::kernels::prefill_nvfp4_scale_bytes_a(m, k));
    cudaMalloc(&qb, sparkinfer::kernels::prefill_nvfp4_data_bytes(n, k));
    cudaMalloc(&sb, sparkinfer::kernels::prefill_nvfp4_scale_bytes_b(n, k));
    const size_t wsb = sparkinfer::kernels::prefill_nvfp4_workspace_bytes(m, n, k);
    if (wsb) cudaMalloc(&ws, wsb);
    cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice);
    GemmConfig cfg; cfg.layout_a = GemmLayout::ROW_MAJOR; cfg.layout_b = GemmLayout::COL_MAJOR;
    sparkinfer::kernels::launch_gemm(dA, dW, dRef, m, n, k, 1.f, 0.f, cfg, nullptr);
    sparkinfer::kernels::launch_prefill_nvfp4_quant_a(dA, qa, sa, m, k, nullptr);
    sparkinfer::kernels::launch_prefill_nvfp4_quant_b(dW, qb, sb, n, k, nullptr);
    sparkinfer::kernels::launch_prefill_nvfp4_gemm(qa, sa, qb, sb, dOut, m, n, k, ws, nullptr, 1.f);
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hRef((size_t)m * n), hOut((size_t)m * n);
    cudaMemcpy(hRef.data(), dRef, hRef.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOut.data(), dOut, hOut.size() * 2, cudaMemcpyDeviceToHost);
    auto rel = [&](const std::vector<__nv_bfloat16>& v) {
        double num = 0, den = 0;
        for (size_t i = 0; i < cpu.size(); i++) {
            const double d = cpu[i] - __bfloat162float(v[i]);
            num += d * d; den += cpu[i] * cpu[i];
        }
        return std::sqrt(num / (den > 0 ? den : 1));
    };
    printf("cpu self-test (m=8 n=128 k=128):  bf16_gemm_vs_cpu=%.4f   nvfp4_vs_cpu=%.4f\n",
           rel(hRef), rel(hOut));
    printf("  (bf16 should be ~0.00; if it is ~1.41 the HARNESS drives launch_gemm wrong,\n"
           "   not the NVFP4 kernel -- interpret the sweep below accordingly)\n");
    for (void* p : {dA, dW, dRef, dOut, qa, sa, qb, sb, ws}) if (p) cudaFree(p);
}

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }
    cpu_selftest();
    const Shape shapes[] = {
        {"qwen3.8 gate/up", 17408, 5120},   // the broken model's FFN in-projection
        {"qwen3.8 down",     5120, 17408},
        {"muse gate/up",    19968, 6656},   // the working model's, for contrast
        {"muse down",        6656, 19968},
    };
    std::vector<int> ms;
    if (argc > 1) ms.push_back(atoi(argv[1]));
    else ms = {8, 32, 128, 512};

    double worst = 0;
    for (int m : ms) {
        printf("m = %d\n", m);
        for (const Shape& s : shapes) {
            double rel = 0;
            if (!run_case(s, m, rel)) return 1;
            if (rel > worst) worst = rel;
        }
    }
    printf("WORST rel_err=%.4f  %s\n", worst, worst < 0.15 ? "PASS" : "FAIL");
    return worst < 0.15 ? 0 : 1;
}
