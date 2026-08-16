// Does the Qwen3.8-27B MTP head actually predict the target's next-next token?
//
// This is the FIRST thing to run on the MTP path, before any accept loop exists, because a wrong
// MTP head does not produce garbage. The target verifies every proposal, so output stays correct
// no matter how bad the draft is -- a mis-wired head shows up only as a low acceptance rate, i.e.
// as disappointing throughput, which is far harder to attribute than a crash.
//
// Method: run the target autoregressively over a real prompt. At each step the target commits
// token t+1 and exposes its final hidden at position t. Feed both to the MTP head, which proposes
// token t+2, and compare that against what the target itself commits at the next step. The
// fraction that match IS the acceptance rate a lossless greedy speculative loop would achieve.
//
// Two wiring choices are settled here rather than by argument, because neither is decidable from
// the tensor shapes and both fail silently:
//   --post-norm : feed RMSNorm(x_final) instead of x_final (see Qwen35Model::final_hidden)
//   --swap-cat  : concatenate [hidden ; embedding] instead of [embedding ; hidden]
// Run all four combinations; the correct wiring should stand out by a wide margin. If the best of
// them is still near chance, the head is mis-wired somewhere else entirely and no accept loop
// would rescue it.
//
// Usage: qwen38_mtp_check <checkpoint_dir> <n_steps> <id0> [id1 ...]

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/qwen38_mtp.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/safetensors.h"
#include "qwen3_gguf_config.h"
#include "qwen_checkpoint.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: %s <checkpoint_dir> <n_steps> <id0> [id1 ...] [--post-norm] [--swap-cat]\n",
               argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    bool post_norm = false, swap_cat = false;
    std::vector<int> prompt;
    const std::string path = argv[1];
    const int n_steps = atoi(argv[2]);
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--post-norm")) { post_norm = true; continue; }
        if (!strcmp(argv[i], "--swap-cat"))  { swap_cat = true;  continue; }
        prompt.push_back(atoi(argv[i]));
    }
    if (prompt.empty()) { printf("[FAIL] no prompt tokens\n"); return 2; }

    sparkinfer::GGUF g;
    sparkinfer::Qwen35Config cfg;
    QwenCheckpointKind kind{};
    std::string err;
    if (!qwen_checkpoint_open(path, cfg, g, kind, err)) { printf("[FAIL] %s\n", err.c_str()); return 1; }
    if (kind != QwenCheckpointKind::CompressedTensors) {
        printf("[FAIL] MTP needs the compressed-tensors checkout (model_mtp.safetensors)\n");
        return 1;
    }
    cfg.max_seq = std::max(2048, (int)prompt.size() + n_steps + 16);

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim; kvc.block_size = 16; kvc.int8_kv = false;
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    printf("loading target (%s) ...\n", qwen_checkpoint_kind_label(kind));
    if (!qwen_checkpoint_load(model, path, kind)) { printf("[FAIL] target load\n"); return 1; }

    // The MTP shard lives in the same checkout and the same index; SafeTensorsModel resolves it.
    sparkinfer::SafeTensorsModel st;
    if (!st.open(path)) { printf("[FAIL] reopen checkpoint for MTP\n"); return 1; }
    sparkinfer::Qwen38MtpHead mtp(sparkinfer::qwen38_mtp_config_from(cfg));
    if (!mtp.load(st)) { printf("[FAIL] mtp load\n"); return 1; }
    mtp.set_shared_weights(model.embed_weights(), model.lm_head_weights(), model.lm_head_quant_type());
    if (!mtp.init_runtime(cfg.max_seq)) { printf("[FAIL] mtp init_runtime\n"); return 1; }

    printf("wiring: hidden=%s  concat=%s\n",
           post_norm ? "POST-final-norm" : "pre-final-norm",
           swap_cat ? "[hidden;embedding]" : "[embedding;hidden]");

    // Allocate the target's KV for the default sequence (Impl::active_seq_id == 0) BEFORE any
    // forward_token. Every in-tree caller does this -- bench_decode, generate, cache_prefix all
    // call kv->allocate(active_seq_id, max_seq) first. Skipping it leaves the block table
    // unpopulated, and the target's attention then walks off it: an illegal memory access inside
    // the TARGET's prefill, which surfaces as "[FAIL] mtp forward" at the caller's next sync and
    // reads like an MTP bug. It is not.
    if (!kv.allocate(0, cfg.max_seq)) { printf("[FAIL] kv allocate\n"); return 1; }

    // Prefill: run the prompt through the target, AND run the MTP head in lockstep so its own KV
    // is populated over the same positions.
    //
    // This is not optional bookkeeping. mtp.layers.0 is a full-attention layer that attends over
    // ITS OWN K/V history, and flash_decode_split reads seq_len = pos+1 entries. Writing only the
    // current position leaves every earlier slot uninitialized, so the draft attends over garbage
    // and produces structured-but-wrong proposals -- which is precisely the 0/63 with a live,
    // varying target hidden that this check first measured. In a real speculative loop the head
    // runs at every committed position anyway, so its KV fills naturally; the harness has to do
    // the same or it is not measuring the same thing.
    int tok = prompt[0];
    int pos = 0;
    for (; pos + 1 < (int)prompt.size(); pos++) {
        if (model.forward_token(prompt[pos], pos, false) < 0) { printf("[FAIL] prefill\n"); return 1; }
        int warm = -1;
        if (!mtp.forward(model.final_hidden(!post_norm), prompt[pos + 1], pos, &warm, nullptr, swap_cat)) {
            printf("[FAIL] mtp prefill at pos %d\n", pos);
            return 1;
        }
    }

    int agree = 0, total = 0;
    int prev_proposal = -1;
    tok = prompt.back();
    for (int step = 0; step < n_steps; step++) {
        const int next = model.forward_token(tok, pos, true);   // target commits token at pos+1
        if (next < 0) { printf("[FAIL] target forward at pos %d\n", pos); return 1; }

        // Score the PREVIOUS step's proposal: it predicted this position's token.
        if (prev_proposal >= 0) {
            total++;
            if (prev_proposal == next) agree++;
            // A degenerate head (same id every step, or an id outside the vocab) is a different
            // failure from a merely inaccurate one, and the aggregate rate cannot tell them apart.
            if (getenv("SPARKINFER_MTP_DEBUG") && total <= 12)
                printf("  step %2d: target=%-7d mtp=%-7d %s\n", total, next, prev_proposal,
                       prev_proposal == next ? "HIT" : "");
        }

        // lm_head isolation. Every MTP stage up to prelm_normed matches a NumPy recomputation from
        // the checkpoint bytes, so if the head still disagrees with the target the fault is either
        // the shared lm_head invocation or this harness's own semantics. Drive the SAME lm_head
        // call MTP uses with the TARGET's post-final-norm hidden: the target just produced `next`
        // from exactly that vector, so a correct call must reproduce it. A mismatch indicts the
        // lm_head path; a match clears it and points at the t+1-vs-t+2 semantics instead.
        if (getenv("SPARKINFER_MTP_LMCHECK") && total <= 4) {
            int lm_argmax = -1;
            if (mtp.lm_head_argmax(model.final_hidden(false), &lm_argmax))
                printf("    lm_head(target_xn) -> %d   target committed %d   %s\n",
                       lm_argmax, next, lm_argmax == next ? "MATCH" : "MISMATCH");
        }

        // Propose token pos+2 from (hidden at pos, embedding of the just-committed token).
        const void* h = model.final_hidden(!post_norm);
        // Is the hidden actually live? MTP proposing a pure function of the token id (same id ->
        // same proposal at different positions) is what a zero/stale hidden looks like from the
        // outside, so check the buffer directly rather than inferring.
        if (getenv("SPARKINFER_MTP_DEBUG") && total <= 3) {
            std::vector<uint16_t> probe(cfg.hidden);
            cudaMemcpy(probe.data(), h, probe.size() * 2, cudaMemcpyDeviceToHost);
            double acc = 0; int nz = 0;
            for (uint16_t bits : probe) {
                uint32_t f = (uint32_t)bits << 16;   // bf16 -> f32
                float val; memcpy(&val, &f, 4);
                acc += (double)val * val; nz += (bits != 0);
            }
            printf("    hidden: l2=%.4f nonzero=%d/%d  first=[%u %u %u %u]\n",
                   sqrt(acc), nz, (int)probe.size(), probe[0], probe[1], probe[2], probe[3]);
        }
        int proposal = -1;
        if (!mtp.forward(h, next, pos, &proposal, nullptr, swap_cat)) {
            printf("[FAIL] mtp forward at pos %d\n", pos);
            return 1;
        }
        prev_proposal = proposal;
        tok = next;
        pos++;
    }

    const double rate = total ? (double)agree / total : 0.0;
    printf("MTP_AGREE %d/%d = %.3f   (this is the acceptance rate a greedy spec loop would get)\n",
           agree, total, rate);
    // tau = expected tokens committed per target forward, capped at 2 for a 1-layer MTP head.
    printf("MTP_TAU %.3f   (1 + agreement)\n", 1.0 + rate);
    return 0;
}
