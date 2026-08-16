// What acceptance length (tau) does the DSpark draft achieve against Qwen3.8-27B?
//
// This is the number that decides whether the remaining speculative-decoding work is worth doing.
// DSpark's block_size is 7, so tau can in principle reach 7 -- against MTP's hard ceiling of 2
// (mtp_num_hidden_layers=1). If DSpark lands tau ~4-5 it is clearly the better path; if it lands
// ~2 then MTP is simpler for the same benefit.
//
// It is measured BEFORE the expensive part is built, deliberately. Both speculative paths are
// currently gated behind the same missing piece: dflash_verify_short_run rejects dense_ffn (its
// guards want !c.dense_ffn and n_experts==256, written for Qwen3.6-35B-A3B's MoE), and Qwen3.8 is
// dense_ffn with n_experts==1. So dflash_generate here falls back to the token-loop verify, which
// per DFlash's own measurements never saves a target forward -- this run reports tau and
// LOSSLESSNESS, not throughput. Throughput needs the dense-FFN verify branch, and tau is what
// says whether to write it.
//
// Usage: dspark_tau_check <qwen38_checkpoint_dir> <dspark_draft_dir> <max_new> <id0> [id1 ...]

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"
#include "qwen_checkpoint.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("usage: %s <qwen38_dir> <dspark_dir> <max_new> <id0> [id1 ...]\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string tpath = argv[1], dpath = argv[2];
    const int max_new = atoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; i++) prompt.push_back(atoi(argv[i]));

    sparkinfer::GGUF g;
    sparkinfer::Qwen35Config cfg;
    QwenCheckpointKind kind{};
    std::string err;
    if (!qwen_checkpoint_open(tpath, cfg, g, kind, err)) { printf("[FAIL] %s\n", err.c_str()); return 1; }
    cfg.max_seq = std::max(2048, (int)prompt.size() + max_new + 64);

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
    if (!qwen_checkpoint_load(model, tpath, kind)) { printf("[FAIL] target load\n"); return 1; }

    sparkinfer::DFlashDraftConfig dcfg;
    sparkinfer::DFlashDraftModel draft(dcfg);
    if (!draft.load(dpath)) { printf("[FAIL] draft load\n"); return 1; }
    const sparkinfer::DFlashDraftConfig& dc = draft.config();

    // The draft has no embed_tokens / lm_head of its own -- it borrows the target's, which is what
    // makes its proposals directly comparable to what the target would produce.
    draft.set_shared_weights(model.embed_weights(), model.lm_head_weights(),
                             model.lm_head_quant_type(), cfg.vocab, cfg.hidden);
    draft.ensure_quant();

    // Capture layers come from the CHECKPOINT, not from this runtime's Qwen3.6 default: DSpark
    // projects [4,16,28,40,52] and its fc is sized n_cap*hidden accordingly.
    model.set_dflash_draft(&draft);
    // max_rows: use the API default (16), NOT block_size. dflash_generate writes capture rows
    // 0..kProposalDepth, and kProposalDepth is chosen from the SEQUENCE LENGTH (5 short / 7 long),
    // not from the draft's block_size -- so sizing this buffer from block_size is sizing it from
    // the wrong quantity entirely, and it can be overrun.
    model.set_dflash_capture(true, dc.target_layer_ids);
    printf("draft: layers=%d B=%d n_cap=%zu  target: layers=%d dense_ffn=%d experts=%d\n",
           dc.n_layers, dc.block_size, dc.target_layer_ids.size(),
           cfg.n_layers, (int)cfg.dense_ffn, cfg.n_experts);

    // AR reference FIRST, on clean state. Taking it afterwards produced an EMPTY reference:
    // dflash_generate opens its own session via open_session(), so a following generate() starts
    // against a pool that still holds it, and freeing session 0 (which dflash never used) does not
    // release it. Ordering the runs this way avoids depending on teardown semantics the harness
    // does not control.
    model.set_dflash_draft(nullptr);
    model.set_dflash_capture(false, {}, 0);
    const std::vector<int> ar = model.generate(prompt, max_new);
    if (ar.empty()) { printf("[FAIL] AR reference produced nothing\n"); return 1; }

    model.set_dflash_draft(&draft);
    model.set_dflash_capture(true, dc.target_layer_ids);
    draft.reset();
    sparkinfer::Qwen35Model::DFlashStats stats;
    const std::vector<int> spec = model.dflash_generate(prompt, max_new, &stats);
    if (spec.empty()) { printf("[FAIL] dflash_generate produced nothing\n"); return 1; }

    // An empty AR reference must FAIL, not pass vacuously: min(0, k) == 0 makes "matched 0/0"
    // satisfy same==n and report lossless=YES while comparing nothing at all. Observed for real --
    // generate() returned no tokens and the run still claimed success.
    if (ar.empty() || spec.empty() || ar.size() < spec.size()) {
        printf("DSPARK lossless=UNKNOWN  ar=%zu spec=%zu -- reference unusable, not a pass\n",
               ar.size(), spec.size());
        return 1;
    }
    size_t n = std::min(ar.size(), spec.size()), same = 0;
    while (same < n && ar[same] == spec[same]) same++;
    // A speculative path is lossless BY CONSTRUCTION -- every emitted token is a target argmax --
    // so a mismatch at index 0 does not mean "poor acceptance", it means one of these two runs is
    // not doing what it claims. Print both prefixes so the failing side is identifiable instead of
    // inferred: if AR here disagrees with a fresh AR run, the reference teardown is at fault; if
    // the speculative side is the odd one out, tokens are being committed unverified.
    printf("  AR  [");
    for (size_t i = 0; i < std::min<size_t>(8, ar.size()); i++) printf(" %d", ar[i]);
    printf(" ]\n  SPEC[");
    for (size_t i = 0; i < std::min<size_t>(8, spec.size()); i++) printf(" %d", spec[i]);
    printf(" ]\n");
    printf("DSPARK tau=%.3f  steps=%d  tokens=%zu  decode_s=%.3f\n",
           stats.mean_accept, stats.steps, spec.size(), stats.decode_s);
    printf("DSPARK lossless=%s  matched %zu/%zu\n", same == n ? "YES" : "NO", same, n);
    printf("DSPARK ceiling: block_size=%d (MTP's ceiling is 2)\n", dc.block_size);
    return 0;
}
