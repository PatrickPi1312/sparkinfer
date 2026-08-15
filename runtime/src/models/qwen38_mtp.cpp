// Qwen3.8-27B MTP head: weight load. See qwen38_mtp.h for the tensor inventory, the forward
// order, and the three Qwen3.8-specific traps this file has to respect.

#include "sparkinfer/models/qwen38_mtp.h"
#include "sparkinfer/safetensors.h"

#include <cstdio>
#include <vector>

namespace sparkinfer {

Qwen38MtpConfig qwen38_mtp_config_from(const Qwen35Config& cfg) {
    Qwen38MtpConfig m;
    m.hidden       = cfg.hidden;
    m.n_q_heads    = cfg.n_q_heads;
    m.n_kv_heads   = cfg.n_kv_heads;
    m.head_dim     = cfg.head_dim;
    m.intermediate = cfg.moe_ffn;      // dense_ffn model: moe_ffn carries the dense FFN width
    m.vocab        = cfg.vocab;
    m.rms_eps      = cfg.rms_eps;
    m.rope_theta   = cfg.rope_theta;
    m.max_seq      = cfg.max_seq;
    return m;
}

struct Qwen38MtpHead::Impl {
    Qwen38MtpConfig cfg;
    std::vector<void*> owned;

    // Fusion stage.
    const void* pre_fc_norm_embedding = nullptr;   // [hidden]
    const void* pre_fc_norm_hidden    = nullptr;   // [hidden]
    const void* fc                    = nullptr;   // [hidden, 2*hidden]

    // The single decoder layer. Names mirror Qwen35LayerWeights so the shared attention/FFN
    // kernels can be driven from these without a translation step.
    const void* input_layernorm          = nullptr;   // [hidden]
    const void* post_attention_layernorm = nullptr;   // [hidden]
    const void* wq  = nullptr;   // [2*n_q_heads*head_dim, hidden]  -- q|gate fused, see header
    const void* wk  = nullptr;   // [n_kv_heads*head_dim, hidden]
    const void* wv  = nullptr;   // [n_kv_heads*head_dim, hidden]
    const void* wo  = nullptr;   // [hidden, n_q_heads*head_dim]
    const void* q_norm = nullptr;   // [head_dim]
    const void* k_norm = nullptr;   // [head_dim]
    const void* gate_proj = nullptr;   // [intermediate, hidden]
    const void* up_proj   = nullptr;   // [intermediate, hidden]
    const void* down_proj = nullptr;   // [hidden, intermediate]

    const void* norm = nullptr;   // [hidden] final norm before the shared lm_head

    // Borrowed from the target (mtp_use_dedicated_embeddings=false).
    const void* embed   = nullptr;
    const void* lm_head = nullptr;
    int lm_head_type = 0;

    bool loaded = false;
    size_t bytes = 0;
};

Qwen38MtpHead::Qwen38MtpHead(const Qwen38MtpConfig& cfg) : p_(new Impl) { p_->cfg = cfg; }

Qwen38MtpHead::~Qwen38MtpHead() {
    for (void* d : p_->owned) cudaFree(d);
    delete p_;
}

const Qwen38MtpConfig& Qwen38MtpHead::config() const { return p_->cfg; }
bool Qwen38MtpHead::ready() const { return p_->loaded && p_->embed && p_->lm_head; }
size_t Qwen38MtpHead::vram_bytes() const { return p_->bytes; }

void Qwen38MtpHead::set_shared_weights(const void* embed_bf16, const void* lm_head, int lm_head_type) {
    p_->embed = embed_bf16;
    p_->lm_head = lm_head;
    p_->lm_head_type = lm_head_type;
}

bool Qwen38MtpHead::load(SafeTensorsModel& st) {
    Impl& s = *p_;
    const Qwen38MtpConfig& c = s.cfg;
    const long H = c.hidden;
    const long qdim = (long)c.n_q_heads * c.head_dim;
    const long kvdim = (long)c.n_kv_heads * c.head_dim;

    // Every tensor here is plain BF16: quantization_config.ignore ends with "re:^mtp.*", so the
    // whole block is excluded from both the NVFP4 and FP8 groups. Shapes are asserted rather than
    // trusted -- a mis-shaped draft degrades the acceptance rate instead of crashing, which is a
    // much harder failure to attribute later.
    bool ok = true;
    auto bf16 = [&](const char* name, long n_values) -> const void* {
        const STTensor* t = st.tensor(name);
        if (!t) {
            fprintf(stderr, "[mtp] missing %s (is model_mtp.safetensors present?)\n", name);
            ok = false; return nullptr;
        }
        if (t->dtype != STDType::BF16 || t->n_values != n_values) {
            fprintf(stderr, "[mtp] %s: expected BF16[%ld], got dtype=%d n=%ld\n",
                    name, n_values, (int)t->dtype, t->n_values);
            ok = false; return nullptr;
        }
        void* d = nullptr;
        const size_t nb = (size_t)n_values * 2;
        if (cudaMalloc(&d, nb) != cudaSuccess) {
            fprintf(stderr, "[mtp] cudaMalloc failed for %s (%.1f MiB) -- out of VRAM?\n",
                    name, nb / 1048576.0);
            ok = false; return nullptr;
        }
        cudaMemcpy(d, t->data, nb, cudaMemcpyHostToDevice);
        s.owned.push_back(d);
        s.bytes += nb;
        return d;
    };

    s.pre_fc_norm_embedding = bf16("mtp.pre_fc_norm_embedding.weight", H);
    s.pre_fc_norm_hidden    = bf16("mtp.pre_fc_norm_hidden.weight",    H);
    // fc consumes the CONCATENATION of the two normed branches, hence 2*hidden inputs.
    s.fc                    = bf16("mtp.fc.weight",                    H * 2 * H);

    s.input_layernorm          = bf16("mtp.layers.0.input_layernorm.weight",          H);
    s.post_attention_layernorm = bf16("mtp.layers.0.post_attention_layernorm.weight", H);
    // q_proj carries Q **and** the fused output gate: 2*qdim rows, not qdim. Asserting the doubled
    // width here is what pins trap #1 down at load time rather than at the first bad sample.
    s.wq = bf16("mtp.layers.0.self_attn.q_proj.weight", 2 * qdim * H);
    s.wk = bf16("mtp.layers.0.self_attn.k_proj.weight", kvdim * H);
    s.wv = bf16("mtp.layers.0.self_attn.v_proj.weight", kvdim * H);
    s.wo = bf16("mtp.layers.0.self_attn.o_proj.weight", H * qdim);
    s.q_norm = bf16("mtp.layers.0.self_attn.q_norm.weight", c.head_dim);
    s.k_norm = bf16("mtp.layers.0.self_attn.k_norm.weight", c.head_dim);

    s.gate_proj = bf16("mtp.layers.0.mlp.gate_proj.weight", (long)c.intermediate * H);
    s.up_proj   = bf16("mtp.layers.0.mlp.up_proj.weight",   (long)c.intermediate * H);
    s.down_proj = bf16("mtp.layers.0.mlp.down_proj.weight", H * c.intermediate);

    s.norm = bf16("mtp.norm.weight", H);

    if (!ok) {
        for (void* d : s.owned) cudaFree(d);
        s.owned.clear();
        s.bytes = 0;
        return false;
    }
    s.loaded = true;
    fprintf(stderr, "[mtp] loaded 15 bf16 tensors, %.1f MiB (1 decoder layer, %d Q/%d KV heads, "
                    "head_dim %d, ffn %d)\n",
            s.bytes / 1048576.0, c.n_q_heads, c.n_kv_heads, c.head_dim, c.intermediate);
    return true;
}

}   // namespace sparkinfer
