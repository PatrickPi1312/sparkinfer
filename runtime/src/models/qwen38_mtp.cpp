// Qwen3.8-27B MTP head: weight load. See qwen38_mtp.h for the tensor inventory, the forward
// order, and the three Qwen3.8-specific traps this file has to respect.

#include "sparkinfer/models/qwen38_mtp.h"
#include "sparkinfer/safetensors.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/prefill.h"

#include <cmath>
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

    // --- runtime state (init_runtime) ---
    // The MTP layer keeps its OWN KV history. block_table is the identity permutation, so the
    // paged kernels address a plain contiguous pool -- this head serves one sequence, so there is
    // nothing for real paging to do and an identity table keeps the kernel contracts unchanged.
    int  max_seq = 0, max_blocks = 0, kv_len = 0;
    static constexpr int kBlock = 16;
    void *k_pool = nullptr, *v_pool = nullptr;
    int  *d_block_table = nullptr, *d_pos = nullptr, *d_seq_len = nullptr, *d_token = nullptr;
    void *x = nullptr, *cat = nullptr, *normed = nullptr, *resid = nullptr;
    void *qraw = nullptr, *q = nullptr, *qgate = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr;
    void *ffn_gate = nullptr, *ffn_up = nullptr;
    // Flash-decode split scratch. The generic launch_flash_decode is NOT usable here: the target
    // never calls it, and driving head_dim=256 through it faults (illegal memory access on the
    // first MTP step). launch_flash_decode_split is the kernel the target's own full-attention
    // layers use, and it requires these three fp32 partials.
    float *fa_m = nullptr, *fa_l = nullptr, *fa_acc = nullptr;
    int   n_splits = 4;
    float* logits = nullptr;
    int*   d_argmax = nullptr;
    bool   rt_ready = false;

    void* alloc(size_t nb) {
        void* d = nullptr;
        if (cudaMalloc(&d, nb) != cudaSuccess) return nullptr;
        owned.push_back(d);
        bytes += nb;
        return d;
    }
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

bool Qwen38MtpHead::init_runtime(int max_seq) {
    Impl& s = *p_;
    const Qwen38MtpConfig& c = s.cfg;
    if (s.rt_ready) return true;
    if (!s.loaded) { fprintf(stderr, "[mtp] init_runtime before load\n"); return false; }

    s.max_seq = max_seq;
    s.max_blocks = (max_seq + Impl::kBlock - 1) / Impl::kBlock + 1;
    const long qdim = (long)c.n_q_heads * c.head_dim;
    const long kvdim = (long)c.n_kv_heads * c.head_dim;
    const size_t kv_elems = (size_t)s.max_blocks * Impl::kBlock * kvdim;

    s.k_pool = s.alloc(kv_elems * 2);
    s.v_pool = s.alloc(kv_elems * 2);
    s.x        = s.alloc((size_t)c.hidden * 2);
    s.cat      = s.alloc((size_t)c.hidden * 2 * 2);
    s.normed   = s.alloc((size_t)c.hidden * 2);
    s.resid    = s.alloc((size_t)c.hidden * 2);
    s.qraw     = s.alloc((size_t)qdim * 2 * 2);
    s.q        = s.alloc((size_t)qdim * 2);
    s.qgate    = s.alloc((size_t)qdim * 2);
    s.k        = s.alloc((size_t)kvdim * 2);
    s.v        = s.alloc((size_t)kvdim * 2);
    s.attn     = s.alloc((size_t)qdim * 2);
    s.ffn_gate = s.alloc((size_t)c.intermediate * 2);
    s.ffn_up   = s.alloc((size_t)c.intermediate * 2);
    const size_t nparts = (size_t)c.n_q_heads * s.n_splits;
    s.fa_m   = (float*)s.alloc(nparts * sizeof(float));
    s.fa_l   = (float*)s.alloc(nparts * sizeof(float));
    s.fa_acc = (float*)s.alloc(nparts * c.head_dim * sizeof(float));
    s.logits = (float*)s.alloc((size_t)c.vocab * sizeof(float));
    s.d_block_table = (int*)s.alloc((size_t)s.max_blocks * sizeof(int));
    s.d_pos      = (int*)s.alloc(sizeof(int));
    s.d_seq_len  = (int*)s.alloc(sizeof(int));
    s.d_token    = (int*)s.alloc(sizeof(int));
    s.d_argmax   = (int*)s.alloc(sizeof(int));

    if (!s.k_pool || !s.v_pool || !s.x || !s.cat || !s.normed || !s.resid || !s.qraw || !s.q ||
        !s.qgate || !s.k || !s.v || !s.attn || !s.ffn_gate || !s.ffn_up || !s.logits ||
        !s.d_block_table || !s.d_pos || !s.d_seq_len || !s.d_token || !s.d_argmax) {
        fprintf(stderr, "[mtp] init_runtime: out of VRAM\n");
        return false;
    }
    std::vector<int> ident(s.max_blocks);
    for (int i = 0; i < s.max_blocks; i++) ident[i] = i;
    cudaMemcpy(s.d_block_table, ident.data(), ident.size() * sizeof(int), cudaMemcpyHostToDevice);

    s.kv_len = 0;
    s.rt_ready = true;
    fprintf(stderr, "[mtp] runtime ready: max_seq %d, own KV %.1f MiB, total %.1f MiB\n",
            max_seq, (kv_elems * 4) / 1048576.0, s.bytes / 1048576.0);
    return true;
}

bool Qwen38MtpHead::lm_head_argmax(const void* hidden_bf16, int* out_argmax, cudaStream_t stream) {
    Impl& s = *p_;
    if (!ready() || !s.rt_ready || !hidden_bf16 || !out_argmax) return false;
    cudaStream_t st = stream;
    if (s.lm_head_type == 0)
        kernels::launch_gemv_f32(hidden_bf16, s.lm_head, s.logits, s.cfg.vocab, s.cfg.hidden, st);
    else
        kernels::launch_gemv_q_f32(hidden_bf16, s.lm_head, s.lm_head_type, s.logits,
                                   s.cfg.vocab, s.cfg.hidden, st);
    kernels::launch_argmax(s.logits, s.d_argmax, 1, s.cfg.vocab, st);
    cudaMemcpyAsync(out_argmax, s.d_argmax, sizeof(int), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    return cudaGetLastError() == cudaSuccess;
}

void Qwen38MtpHead::reset() { p_->kv_len = 0; }
void Qwen38MtpHead::crop(int keep) { if (keep >= 0 && keep < p_->kv_len) p_->kv_len = keep; }
int  Qwen38MtpHead::seq_len() const { return p_->kv_len; }

bool Qwen38MtpHead::forward(const void* target_hidden, int next_token_id, int pos,
                            int* out_argmax, cudaStream_t stream, bool swap_cat) {
    Impl& s = *p_;
    const Qwen38MtpConfig& c = s.cfg;
    if (!ready() || !s.rt_ready || !target_hidden || !out_argmax) return false;
    if (pos < 0 || pos >= s.max_seq) return false;

    const int H = c.hidden;
    const int qdim = c.n_q_heads * c.head_dim;
    const int kvdim = c.n_kv_heads * c.head_dim;
    cudaStream_t st = stream;

    // SPARKINFER_MTP_DEBUG=1: sync + check after every launch so a fault names the kernel that
    // caused it. Without this an illegal access surfaces at the next sync, which is wherever the
    // caller happens to synchronise -- the first report was "[FAIL] mtp forward", which says only
    // that something in this function went wrong, not what.
    static const bool dbg = [] { const char* e = getenv("SPARKINFER_MTP_DEBUG"); return e && e[0] == '1'; }();

    // SPARKINFER_MTP_DUMP=<dir>: write every intermediate of the FIRST forward to <dir>/<tag>.bin,
    // together with the input that produced them. A NumPy recomputation from the checkpoint bytes,
    // SEEDED WITH THIS EXACT INPUT, then names the first stage that diverges. Seeding matters: it
    // separates "my weights/maths are wrong" from "my input is wrong", which is the distinction
    // that cracked the target's transpose and sigmoid-gate bugs. Black-box agreement sweeps cannot
    // make it -- every remaining suspect here (gate, RoPE, fa scratch, fc operand layout, lm_head
    // path) produces structured, confidently wrong logits and looks identical from outside.
    static const char* dump_dir = getenv("SPARKINFER_MTP_DUMP");
    static bool dumped = false;
    const bool do_dump = dump_dir && !dumped;
    auto dump = [&](const char* tag, const void* dev, size_t nbytes) {
        if (!do_dump) return;
        std::vector<char> host(nbytes);
        cudaStreamSynchronize(st);
        if (cudaMemcpy(host.data(), dev, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) return;
        char pathbuf[512];
        snprintf(pathbuf, sizeof pathbuf, "%s/%s.bin", dump_dir, tag);
        if (FILE* f = fopen(pathbuf, "wb")) { fwrite(host.data(), 1, nbytes, f); fclose(f); }
    };
    auto ck = [&](const char* tag) {
        if (!dbg) return true;
        cudaStreamSynchronize(st);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "[mtp-dbg] FAULT at %s: %s\n", tag, cudaGetErrorString(e)); return false; }
        fprintf(stderr, "[mtp-dbg] ok %s\n", tag);
        return true;
    };

    // --- fusion: x = fc([norm_e(embed[next]) ; norm_h(target_hidden)]) ---
    // Concat order is EMBEDDING first, hidden second -- sglang's qwen3_next_mtp.py does
    // fc(cat((input_embeds, hidden_states))). fc is [hidden, 2*hidden], so the shape cannot
    // disambiguate the halves: swapping them yields plausible-but-degraded proposals that show up
    // only as a poor acceptance rate, never as a crash. Pinned here, and checked empirically by
    // the argmax-agreement test rather than trusted.
    void* emb_slot = swap_cat ? (void*)((char*)s.cat + (size_t)H * 2) : s.cat;
    void* hid_slot = swap_cat ? s.cat : (void*)((char*)s.cat + (size_t)H * 2);
    cudaMemcpyAsync(s.d_token, &next_token_id, sizeof(int), cudaMemcpyHostToDevice, st);
    kernels::launch_embedding(s.d_token, s.embed, emb_slot, 1, H, st);
    if (!ck("embedding")) return false;
    dump("00_in_hidden", target_hidden, (size_t)H * 2);
    kernels::launch_rmsnorm(emb_slot, s.pre_fc_norm_embedding, emb_slot, 1, H, c.rms_eps, st);
    kernels::launch_rmsnorm(target_hidden, s.pre_fc_norm_hidden, hid_slot, 1, H, c.rms_eps, st);
    kernels::launch_gemv(s.cat, s.fc, s.x, H, 2 * H, st);
    if (!ck("fc")) return false;
    dump("01_cat", s.cat, (size_t)H * 2 * 2);
    dump("02_x_fc", s.x, (size_t)H * 2);

    // --- decoder layer 0: pre-attention norm, then gated attention ---
    cudaMemcpyAsync(s.resid, s.x, (size_t)H * 2, cudaMemcpyDeviceToDevice, st);
    kernels::launch_rmsnorm(s.x, s.input_layernorm, s.normed, 1, H, c.rms_eps, st);

    kernels::launch_gemv(s.normed, s.wq, s.qraw, 2 * qdim, H, st);
    kernels::launch_gemv(s.normed, s.wk, s.k, kvdim, H, st);
    kernels::launch_gemv(s.normed, s.wv, s.v, kvdim, H, st);
    // q_proj packs [q|gate] PER HEAD (2*qdim rows); split before anything touches Q.
    kernels::launch_qwen36_split_q_gate(s.qraw, s.q, s.qgate, c.n_q_heads, c.head_dim, st);
    if (!ck("split_q_gate")) return false;
    dump("03_normed_in", s.normed, (size_t)H * 2);
    dump("04_qraw", s.qraw, (size_t)qdim * 2 * 2);

    kernels::launch_rmsnorm_qk(s.q, s.k, s.q_norm, s.k_norm,
                               c.n_q_heads, c.n_kv_heads, c.head_dim, c.rms_eps, st);
    if (!ck("rmsnorm_qk")) return false;
    // NeoX (split-half) RoPE, matching the target's own full-attention layers -- the "normal"
    // consecutive-pair variant is Muse Glimmer only (qwen35.cpp's rope branch).
    cudaMemcpyAsync(s.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, st);
    kernels::launch_rope_kv_append(s.q, s.k, s.v, s.k_pool, s.v_pool, s.d_block_table, s.d_pos,
                                   1, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta,
                                   Impl::kBlock, s.max_blocks, st);
    if (!ck("rope_kv_append")) return false;

    const int seq = pos + 1;
    cudaMemcpyAsync(s.d_seq_len, &seq, sizeof(int), cudaMemcpyHostToDevice, st);
    const float scale = 1.0f / std::sqrt((float)c.head_dim);
    kernels::launch_flash_decode_split(s.q, s.k_pool, s.v_pool, s.d_block_table, s.d_seq_len, s.attn,
                                       s.fa_m, s.fa_l, s.fa_acc,
                                       1, c.n_q_heads, c.n_kv_heads, c.head_dim,
                                       Impl::kBlock, s.max_blocks, s.n_splits, scale, st,
                                       /*out_q8=*/nullptr, /*seqlen=*/seq);
    if (!ck("flash_decode_split")) return false;
    dump("05_attn_pregate", s.attn, (size_t)qdim * 2);
    // SIGMOID, not silu -- despite text_config's output_gate_type: "swish". silu sign-flips this
    // (gate mean ~ -4.5); same trap as the target's full-attention layers.
    kernels::launch_qwen36_mul_sigmoid(s.attn, s.qgate, qdim, st);
    if (!ck("mul_sigmoid")) return false;
    dump("06_attn_postgate", s.attn, (size_t)qdim * 2);
    kernels::launch_gemv(s.attn, s.wo, s.x, H, qdim, st);
    kernels::launch_prefill_add(s.x, s.resid, s.x, H, st);

    // --- FFN ---
    cudaMemcpyAsync(s.resid, s.x, (size_t)H * 2, cudaMemcpyDeviceToDevice, st);
    kernels::launch_rmsnorm(s.x, s.post_attention_layernorm, s.normed, 1, H, c.rms_eps, st);
    kernels::launch_gemv(s.normed, s.gate_proj, s.ffn_gate, c.intermediate, H, st);
    kernels::launch_gemv(s.normed, s.up_proj,   s.ffn_up,   c.intermediate, H, st);
    kernels::launch_prefill_swiglu(s.ffn_gate, s.ffn_up, s.ffn_gate, c.intermediate, st);
    if (!ck("swiglu")) return false;
    dump("07_ffn_h", s.ffn_gate, (size_t)c.intermediate * 2);
    kernels::launch_gemv(s.ffn_gate, s.down_proj, s.x, H, c.intermediate, st);
    kernels::launch_prefill_add(s.x, s.resid, s.x, H, st);

    // --- shared lm_head ---
    kernels::launch_rmsnorm(s.x, s.norm, s.normed, 1, H, c.rms_eps, st);
    if (s.lm_head_type == 0) kernels::launch_gemv_f32(s.normed, s.lm_head, s.logits, c.vocab, H, st);
    else kernels::launch_gemv_q_f32(s.normed, s.lm_head, s.lm_head_type, s.logits, c.vocab, H, st);
    kernels::launch_argmax(s.logits, s.d_argmax, 1, c.vocab, st);
    if (!ck("argmax")) return false;
    dump("08_prelm_normed", s.normed, (size_t)H * 2);
    dump("09_logits", s.logits, (size_t)c.vocab * 4);

    if (do_dump) {
        char pb[512]; snprintf(pb, sizeof pb, "%s/meta.txt", dump_dir);
        if (FILE* f = fopen(pb, "w")) {
            fprintf(f, "next_token_id %d\npos %d\nhidden %d\nqdim %d\nkvdim %d\nffn %d\nvocab %d\nswap_cat %d\n",
                    next_token_id, pos, H, qdim, kvdim, c.intermediate, c.vocab, swap_cat ? 1 : 0);
            fclose(f);
        }
        dumped = true;
    }
    cudaMemcpyAsync(out_argmax, s.d_argmax, sizeof(int), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    if (pos + 1 > s.kv_len) s.kv_len = pos + 1;
    return cudaGetLastError() == cudaSuccess;
}

}   // namespace sparkinfer
