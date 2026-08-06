// CPU-only Qwen3.6-27B GGUF metadata test.
//
// The native runtime serves the text decoder from a standard Qwen3.6-27B
// GGUF. Vision tensors live in the optional mmproj GGUF and are intentionally
// not part of this text-only load path.

#include "../examples/qwen3_gguf_config.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {
enum { VT_U32 = 4, VT_F32 = 6, VT_STR = 8 };

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
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0; // F32
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

uint64_t tensor_bytes(const Tensor& t) {
    uint64_t n = 1;
    for (uint64_t d : t.dims) n *= d;
    return n * 4;
}

bool write_tiny_gguf(const std::string& path) {
    std::vector<Meta> meta = {
        {"general.name", VT_STR, 0, 0.f, "Qwen3.6-27B"},
        {"general.architecture", VT_STR, 0, 0.f, "qwen35"},
        {"general.alignment", VT_U32, 32},
        {"qwen35.block_count", VT_U32, 64},
        {"qwen35.embedding_length", VT_U32, 5120},
        {"qwen35.vocab_size", VT_U32, 248320},
        {"qwen35.attention.head_count", VT_U32, 24},
        {"qwen35.attention.head_count_kv", VT_U32, 4},
        {"qwen35.attention.key_length", VT_U32, 256},
        {"qwen35.feed_forward_length", VT_U32, 17408},
        {"qwen35.full_attention_interval", VT_U32, 4},
        {"qwen35.rope.dimension_count", VT_U32, 64},
        {"qwen35.ssm.state_size", VT_U32, 128},
        {"qwen35.ssm.group_count", VT_U32, 16},
        {"qwen35.ssm.conv_kernel", VT_U32, 4},
        {"qwen35.rope.freq_base", VT_F32, 0, 10000000.f},
        {"qwen35.attention.layer_norm_rms_epsilon", VT_F32, 0, 1e-6f},
        {"tokenizer.ggml.eos_token_id", VT_U32, 248044},
    };
    // Dimensions match the real unsloth/ggml-org Qwen3.6-27B-GGUF release: the
    // linear-attention stage uses only 16 Q/K heads (fewer than the 24
    // full-attention heads), so attn_qkv's ne[1] is 2*(16*128) + 48*128, not
    // 2*(24*128) + 48*128. attn_gate mirrors the 48-head value width directly
    // and is what the parser must read to get linear_v_heads right; dense FFN
    // names prove this is not the routed-MoE path.
    std::vector<Tensor> tensors = {
        {"token_embd.weight", {1, 248320}},
        {"blk.0.attn_qkv.weight", {1, 10240}}, // 2*(16*128) + 48*128
        {"blk.0.attn_gate.weight", {1, 6144}}, // 48*128
        {"blk.0.ffn_gate.weight", {1, 17408}},
    };

    uint64_t off = 0;
    for (Tensor& t : tensors) {
        t.offset = off;
        t.bytes = tensor_bytes(t);
        off += t.bytes;
    }

    std::vector<uint8_t> b;
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    put<uint32_t>(b, 3);
    put<uint64_t>(b, tensors.size());
    put<uint64_t>(b, meta.size());
    for (const Meta& m : meta) {
        put_str(b, m.key);
        put<uint32_t>(b, (uint32_t)m.type);
        if (m.type == VT_STR) put_str(b, m.s);
        else if (m.type == VT_F32) put<float>(b, m.f);
        else put<uint32_t>(b, m.u);
    }
    for (const Tensor& t : tensors) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t)t.dims.size());
        for (uint64_t d : t.dims) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, t.offset);
    }
    while (b.size() % 32) b.push_back(0);
    for (const Tensor& t : tensors) b.insert(b.end(), (size_t)t.bytes, 0);

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return out.good();
}

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)
} // namespace

int main() {
    const std::string path = "/tmp/sparkinfer_qwen36_27b_config_cpu_test.gguf";
    CHECK(write_tiny_gguf(path));

    sparkinfer::GGUF g;
    CHECK(g.open(path));
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);

    CHECK(cfg.dense_ffn);
    CHECK(cfg.hybrid);
    CHECK(cfg.n_layers == 64);
    CHECK(cfg.hidden == 5120);
    CHECK(cfg.vocab == 248320);
    CHECK(cfg.n_q_heads == 24);
    CHECK(cfg.n_kv_heads == 4);
    CHECK(cfg.head_dim == 256);
    CHECK(cfg.n_experts == 1);
    CHECK(cfg.top_k == 1);
    CHECK(cfg.n_shared == 0);
    CHECK(cfg.moe_ffn == 17408);
    CHECK(cfg.full_attn_interval == 4);
    CHECK(cfg.rope_dim == 64);
    CHECK(cfg.linear_head_dim == 128);
    CHECK(cfg.linear_q_heads == 16);
    CHECK(cfg.linear_v_heads == 48);
    CHECK(cfg.linear_conv_kernel == 4);
    CHECK(cfg.eos_id == 248044);
    CHECK(std::string(qwen3_model_label(cfg)) == "Qwen3.6-27B dense hybrid (text-only)");

    std::printf("qwen36_27b_config_cpu_test: OK\n");
    return 0;
}
