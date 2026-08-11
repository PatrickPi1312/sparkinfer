// CPU-only test for the Muse Glimmer DFlash draft's GGUF metadata parsing.
//
// Writes a tiny GGUF with dflash-style scalar/array metadata (no tensors needed --
// museglimmer_dflash_config_from_gguf reads metadata only), then verifies
// museglimmer_dflash_config_from_gguf() derives DFlashDraftConfig correctly. Especially: the
// target_layer_ids -1 adjustment (see dflash_gguf_config.h for why) and sliding_layers being
// all-true for this checkpoint's sliding_window_pattern.

#include "../examples/dflash_gguf_config.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
enum { VT_U32 = 4, VT_I32 = 5, VT_F32 = 6, VT_STR = 8, VT_ARR = 9 };

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}

void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, (uint64_t)s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Meta {
    std::string key;
    int type;
    uint32_t u = 0;
    float f = 0.f;
    std::string s;
    std::vector<int32_t> arr;   // used when type == VT_ARR
};

bool write_tiny_gguf(const std::string& path) {
    std::vector<Meta> meta = {
        {"general.architecture", VT_STR, 0, 0.f, "dflash"},
        {"general.name", VT_STR, 0, 0.f, "Hf_Museglimmer"},
        {"dflash.block_count", VT_U32, 5},
        {"dflash.embedding_length", VT_U32, 6656},
        {"dflash.feed_forward_length", VT_U32, 19968},
        {"dflash.attention.head_count", VT_U32, 32},
        {"dflash.attention.head_count_kv", VT_U32, 8},
        {"dflash.attention.key_length", VT_U32, 128},
        {"dflash.attention.layer_norm_rms_epsilon", VT_F32, 0, 9.999999747378752e-06f},
        {"dflash.attention.sliding_window", VT_U32, 2048},
        {"dflash.rope.freq_base", VT_F32, 0, 500000.f},
        {"dflash.block_size", VT_U32, 16},
        {"tokenizer.ggml.mask_token_id", VT_U32, 201818},
        {"dflash.attention.sliding_window_pattern", VT_ARR, 0, 0.f, "", {1, 1, 1, 1, 1}},
        {"dflash.target_layers", VT_ARR, 0, 0.f, "", {2, 14, 26, 38, 50}},
    };

    std::vector<uint8_t> b;
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 0);           // n_tensors
    put<uint64_t>(b, meta.size()); // n_kv

    for (const Meta& m : meta) {
        put_str(b, m.key);
        put<uint32_t>(b, (uint32_t)m.type);
        if (m.type == VT_STR) {
            put_str(b, m.s);
        } else if (m.type == VT_F32) {
            put<float>(b, m.f);
        } else if (m.type == VT_ARR) {
            put<uint32_t>(b, (uint32_t)VT_I32);
            put<uint64_t>(b, (uint64_t)m.arr.size());
            for (int32_t v : m.arr) put<int32_t>(b, v);
        } else {
            put<uint32_t>(b, m.u);
        }
    }
    // No tensors: museglimmer_dflash_config_from_gguf reads metadata only.

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return out.good();
}

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)
} // namespace

int main() {
    const std::string path = "/tmp/sparkinfer_dflash_gguf_config_cpu_test.gguf";
    CHECK(write_tiny_gguf(path));

    sparkinfer::GGUF g;
    CHECK(g.open(path));
    CHECK(g.meta_str("general.architecture") == "dflash");

    sparkinfer::DFlashDraftConfig cfg;
    museglimmer_dflash_config_from_gguf(g, cfg);

    CHECK(cfg.n_layers == 5);
    CHECK(cfg.hidden == 6656);
    CHECK(cfg.intermediate == 19968);
    CHECK(cfg.n_q_heads == 32);
    CHECK(cfg.n_kv_heads == 8);
    CHECK(cfg.head_dim == 128);
    CHECK(cfg.sliding_window == 2048);
    CHECK(cfg.block_size == 16);
    CHECK(cfg.mask_token_id == 201818);
    CHECK(cfg.rope_theta == 500000.f);
    CHECK(cfg.rms_eps > 9e-6f && cfg.rms_eps < 1.1e-5f);
    CHECK(cfg.rope_normal == true);

    // Every layer of this checkpoint is sliding-window per the metadata array.
    CHECK(cfg.sliding_layers.size() == 5);
    for (bool sw : cfg.sliding_layers) CHECK(sw == true);

    // The critical off-by-one: dflash.target_layers is [2,14,26,38,50] (upstream's "capture
    // layer N's INPUT" convention); this codebase's capture point fires at the END of a layer's
    // processing (layer L's OUTPUT == layer L+1's INPUT), so every raw value must come out -1.
    const std::vector<int> want = {1, 13, 25, 37, 49};
    CHECK(cfg.target_layer_ids.size() == want.size());
    for (size_t i = 0; i < want.size(); i++) CHECK(cfg.target_layer_ids[i] == want[i]);

    std::printf("dflash_gguf_config_cpu_test: OK\n");
    return 0;
}
