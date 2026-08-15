// CPU-only ModelOpt NVFP4 pack/dequant test (no GPU).

#include "sparkinfer/nvfp4_format.h"

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)

int main() {
    CHECK(std::fabs(sparkinfer::nvfp4_e2m1_lut(0) - 0.f) < 1e-6f);
    CHECK(std::fabs(sparkinfer::nvfp4_e2m1_lut(1) - 0.5f) < 1e-6f);
    CHECK(std::fabs(sparkinfer::nvfp4_e2m1_lut(7) - 6.f) < 1e-6f);
    CHECK(std::fabs(sparkinfer::nvfp4_e2m1_lut(8) - 0.f) < 1e-6f);
    CHECK(std::fabs(sparkinfer::nvfp4_e2m1_lut(15) + 6.f) < 1e-6f);

    // e4m3: 0x38 is 1.0 (exp=7, man=0)
    CHECK(std::fabs(sparkinfer::nvfp4_fp8e4m3_to_f(0x38) - 1.f) < 1e-5f);
    // 0x00 is +0
    CHECK(std::fabs(sparkinfer::nvfp4_fp8e4m3_to_f(0x00)) < 1e-8f);

    const int n_in = 32;
    unsigned char packed[16] = {};
    unsigned char scales[2] = {0x38, 0x38};  // 1.0, 1.0
    // pack 1.0 (nibble 2) then 2.0 (nibble 4) repeating
    for (int i = 0; i < n_in; i++) {
        const int nib = (i & 1) ? 4 : 2;
        packed[i >> 1] |= (unsigned char)(nib << ((i & 1) ? 4 : 0));
    }
    std::vector<float> out(n_in);
    sparkinfer::nvfp4_dequant_row(packed, scales, 2.f, n_in, out.data());
    CHECK(std::fabs(out[0] - 2.f) < 1e-4f);   // 1.0 * 1.0 * 2
    CHECK(std::fabs(out[1] - 4.f) < 1e-4f);   // 2.0 * 1.0 * 2
    CHECK(std::fabs(out[16] - 2.f) < 1e-4f);

    CHECK(sparkinfer::nvfp4_blob_bytes(4, 32) ==
          sizeof(sparkinfer::Nvfp4Hdr) + 4 * 16 + 4 * 2);
    CHECK(sparkinfer::WTYPE_NVFP4 == 100);

    std::printf("nvfp4_cpu_test: OK\n");
    return 0;
}
