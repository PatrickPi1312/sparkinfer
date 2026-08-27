#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/kernels/compressed_tensors.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/qtype.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

namespace {

bool test_large_row_dequant() {
    using namespace sparkinfer::kernels;
    constexpr int rows = 65536 + 3, cols = 16;
    unsigned char *packed = nullptr, *scale = nullptr;
    __nv_bfloat16* out = nullptr;
    cudaMalloc(&packed, (size_t)rows * cols / 2);
    cudaMalloc(&scale, (size_t)rows * cols / 16);
    cudaMalloc(&out, (size_t)rows * cols * sizeof(*out));
    cudaMemset(packed, 0x22, (size_t)rows * cols / 2); // E2M1 1.0 in both nibbles
    cudaMemset(scale, 0x38, (size_t)rows * cols / 16); // UE4M3 1.0
    launch_ct_dequant_nvfp4(packed, scale, 1.f, out, rows, cols);
    bool ok = cudaDeviceSynchronize() == cudaSuccess;
    __nv_bfloat16 edge[2]{};
    cudaMemcpy(&edge[0], out, sizeof(edge[0]), cudaMemcpyDeviceToHost);
    cudaMemcpy(&edge[1], out + (size_t)(rows - 1) * cols, sizeof(edge[1]),
               cudaMemcpyDeviceToHost);
    ok &= std::fabs(__bfloat162float(edge[0]) - 1.f) < 1e-6f;
    ok &= std::fabs(__bfloat162float(edge[1]) - 1.f) < 1e-6f;
    cudaFree(packed); cudaFree(scale); cudaFree(out);
    std::printf("large-row NVFP4 dequant: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

bool test_vocab_prefix_offset() {
    using namespace sparkinfer::kernels;
    // Use a real projection-sized grid. The production prefix is 65,536 of 248,320 rows; this
    // smaller ratio exercises the identical stored-rows offset without making the unit test big.
    constexpr int stored_rows = 4096, scored_rows = 2048, k = 16;
    const size_t scale_bytes = (size_t)stored_rows * k / 16;
    const size_t packed_bytes = (size_t)stored_rows * k / 2;
    std::vector<unsigned char> payload(SI_NVFP4_HDR + scale_bytes + packed_bytes, 0);
    const float global = 1.f;
    std::memcpy(payload.data(), &global, sizeof(global));
    std::memcpy(payload.data() + 4, &stored_rows, sizeof(stored_rows));
    std::memset(payload.data() + SI_NVFP4_HDR, 0x38, scale_bytes);
    unsigned char* weights = payload.data() + SI_NVFP4_HDR + scale_bytes;
    std::memset(weights, 0x22, k / 2);                    // row 0: all +1
    std::memset(weights + (size_t)scored_rows * k / 2, 0xff,
                packed_bytes - (size_t)scored_rows * k / 2); // poison unused suffix rows

    std::vector<__nv_bfloat16> hx(k, __float2bfloat16(1.f));
    void *x = nullptr, *xq = nullptr, *xs = nullptr, *w = nullptr;
    float *y = nullptr, *y_direct = nullptr;
    cudaMalloc(&x, k * sizeof(__nv_bfloat16));
    cudaMalloc(&xq, k);
    cudaMalloc(&xs, sizeof(float));
    cudaMalloc(&w, payload.size());
    cudaMalloc(&y, scored_rows * sizeof(float));
    cudaMalloc(&y_direct, scored_rows * sizeof(float));
    cudaMemcpy(x, hx.data(), k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(w, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    launch_gemv_nvfp4_quant_x(x, xq, xs, 1, k, nullptr);
    bool ok = launch_gemv_nvfp4_rows_dp4a_f32(xq, xs, w, y, 1, scored_rows, k, nullptr) &&
              cudaDeviceSynchronize() == cudaSuccess;
    float hy[2]{};
    cudaMemcpy(hy, y, sizeof(hy), cudaMemcpyDeviceToHost);
    ok &= std::fabs(hy[0] - 16.f) < 1e-4f;
    ok &= std::fabs(hy[1]) < 1e-6f;
    launch_gemv_q_f32(x, w, SI_QTYPE_NVFP4, y_direct, scored_rows, k, nullptr);
    ok &= cudaDeviceSynchronize() == cudaSuccess;
    float hy_direct[2]{};
    cudaMemcpy(hy_direct, y_direct, sizeof(hy_direct), cudaMemcpyDeviceToHost);
    ok &= std::fabs(hy_direct[0] - 16.f) < 1e-4f;
    ok &= std::fabs(hy_direct[1]) < 1e-6f;
    cudaFree(x); cudaFree(xq); cudaFree(xs); cudaFree(w); cudaFree(y); cudaFree(y_direct);
    std::printf("NVFP4 vocab-prefix offset: %s logits=[%.3f,%.3f]\n",
                ok ? "OK" : "FAIL", hy[0], hy[1]);
    return ok;
}

} // namespace

int main() {
    using namespace sparkinfer::kernels;
    constexpr int M=128, N=128, K=128;
    if (!prefill_nvfp4_supported(M,N,K)) return 77;
    std::vector<__nv_bfloat16> hA(M*K, __float2bfloat16(1.f));
    std::vector<__nv_bfloat16> hB(N*K, __float2bfloat16(1.f)), hD(M*N);
    void *A0=nullptr,*B0=nullptr,*A=nullptr,*B=nullptr,*SA=nullptr,*SB=nullptr,*D=nullptr,*W=nullptr;
    cudaMalloc(&A0,hA.size()*2); cudaMalloc(&B0,hB.size()*2); cudaMalloc(&D,hD.size()*2);
    cudaMalloc(&A,prefill_nvfp4_data_bytes(M,K)); cudaMalloc(&B,prefill_nvfp4_data_bytes(N,K));
    cudaMalloc(&SA,prefill_nvfp4_scale_bytes_a(M,K)); cudaMalloc(&SB,prefill_nvfp4_scale_bytes_b(N,K));
    size_t ws=prefill_nvfp4_workspace_bytes(M,N,K); if(ws) cudaMalloc(&W,ws);
    cudaMemcpy(A0,hA.data(),hA.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(B0,hB.data(),hB.size()*2,cudaMemcpyHostToDevice);
    bool ok=launch_prefill_nvfp4_quant_a(A0,A,SA,M,K) &&
            launch_prefill_nvfp4_quant_b(B0,B,SB,N,K) &&
            launch_prefill_nvfp4_gemm(A,SA,B,SB,D,M,N,K,W) &&
            cudaDeviceSynchronize()==cudaSuccess;
    cudaMemcpy(hD.data(),D,hD.size()*2,cudaMemcpyDeviceToHost);
    float lo=1e9f,hi=-1e9f;
    for(auto v:hD){ float x=__bfloat162float(v); lo=fminf(lo,x); hi=fmaxf(hi,x); ok &= std::isfinite(x) && x>115.f && x<141.f; }
    ok &= test_large_row_dequant();
    ok &= test_vocab_prefix_offset();
    std::printf("prefill_nvfp4_gpu_test: %s range=[%.3f,%.3f]\n",ok?"OK":"FAIL",lo,hi);
    cudaFree(A0);cudaFree(B0);cudaFree(A);cudaFree(B);cudaFree(SA);cudaFree(SB);cudaFree(D);if(W)cudaFree(W);
    return ok?0:1;
}
