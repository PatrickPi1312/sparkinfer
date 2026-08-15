#pragma once
#include <cstddef>
#include <cstdint>

namespace sparkinfer {

// Sentinel ggml-style type id for ModelOpt NVFP4 blobs (not a real ggml type).
constexpr int WTYPE_NVFP4 = 100;
constexpr int NVFP4_GROUP = 16;

// Device blob: header + packed e2m1 [n_out, n_in/2] + fp8 e4m3 scales [n_out, n_in/16].
// Weight layout is GGUF-native [out, in] so decode GEMV is y[N] = x[K] @ W^T.
struct Nvfp4Hdr {
    int32_t n_out = 0;
    int32_t n_in = 0;
    float   scale2 = 1.f;   // ModelOpt weight_scale_2 (multiply, not reciprocal)
    int32_t _pad = 0;
};

inline size_t nvfp4_packed_bytes(int n_out, int n_in) {
    return (size_t)n_out * ((size_t)n_in / 2);
}
inline size_t nvfp4_scale_bytes(int n_out, int n_in) {
    return (size_t)n_out * ((size_t)n_in / NVFP4_GROUP);
}
inline size_t nvfp4_blob_bytes(int n_out, int n_in) {
    return sizeof(Nvfp4Hdr) + nvfp4_packed_bytes(n_out, n_in) + nvfp4_scale_bytes(n_out, n_in);
}

// IEEE-like float4_e2m1fn: 1 sign, 2 exp, 1 mantissa. Index is the 4-bit nibble.
inline float nvfp4_e2m1_lut(int nib) {
    static const float kLut[16] = {
        0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
    };
    return kLut[nib & 15];
}

// FP8 E4M3FN (bias 7). Host reference; the CUDA kernel uses the same math.
inline float nvfp4_fp8e4m3_to_f(unsigned char u) {
    const int s = u >> 7;
    const int e = (u >> 3) & 0xF;
    const int m = u & 7;
    float v;
    if (e == 0) {
        v = (m == 0) ? 0.f : (float)m * 0.001953125f;  // m * 2^-9
    } else if (e == 15 && m == 7) {
        v = 0.f;  // NaN -> 0 (weights should not be NaN)
    } else {
        v = (1.f + (float)m * 0.125f);
        int p = e - 7;
        if (p > 0) { for (int i = 0; i < p; i++) v *= 2.f; }
        else if (p < 0) { for (int i = 0; i < -p; i++) v *= 0.5f; }
    }
    return s ? -v : v;
}

// ModelOpt pack: low nibble = even K, high nibble = odd K.
inline void nvfp4_dequant_row(const unsigned char* packed, const unsigned char* scales,
                              float scale2, int n_in, float* out) {
    const int ng = n_in / NVFP4_GROUP;
    for (int g = 0; g < ng; g++) {
        const float sf = nvfp4_fp8e4m3_to_f(scales[g]) * scale2;
        const unsigned char* pk = packed + g * 8;
        for (int i = 0; i < 16; i++) {
            const unsigned char b = pk[i >> 1];
            const int nib = (i & 1) ? (b >> 4) : (b & 0xF);
            out[g * 16 + i] = nvfp4_e2m1_lut(nib) * sf;
        }
    }
}

} // namespace sparkinfer
