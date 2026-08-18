// Can sparkinfer load the RadixArk/Qwen3.8-27B-DSpark draft checkpoint, and what is still missing?
//
// DSpark is a DFlash-family block-diffusion draft for Qwen3.8-27B, not a standalone model: it has
// no embed_tokens and no lm_head (it borrows both from the target), 5 decoder layers, and a
// projector fc [hidden, n_cap*hidden] over the 5 captured target layers [4,16,28,40,52]. Its 62
// tensor names match DFlashDraftModel::load()'s expectations exactly, so the base draft should
// load with existing code once the config is parsed correctly.
//
// Two tensor groups have NO consumer in this runtime yet and are expected to be reported as
// unclaimed rather than silently ignored:
//   confidence_head.proj  [1, 5376]      hidden(5120) + markov_rank(256) -> scalar
//   markov_head.markov_w{1,2} [248320,256]  low-rank bigram over the vocab
//
// Usage: dspark_load_check <draft_dir>

#include "sparkinfer/models/dflash_draft.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <draft_dir>\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    sparkinfer::DFlashDraftConfig cfg;   // load() overwrites from config.json
    sparkinfer::DFlashDraftModel draft(cfg);
    printf("loading DSpark draft from %s ...\n", argv[1]);
    const bool ok = draft.load(argv[1]);
    const sparkinfer::DFlashDraftConfig& c = draft.config();

    printf("parsed config: layers=%d hidden=%d inter=%d q_heads=%d kv_heads=%d head_dim=%d\n",
           c.n_layers, c.hidden, c.intermediate, c.n_q_heads, c.n_kv_heads, c.head_dim);
    printf("               block_size=%d mask_token=%d vocab=%d rope_theta=%.0f\n",
           c.block_size, c.mask_token_id, c.vocab, c.rope_theta);
    printf("               target_layer_ids(%zu) =", c.target_layer_ids.size());
    for (int id : c.target_layer_ids) printf(" %d", id);
    printf("\n");

    // The projector's width is n_cap*hidden, so a wrong capture-list LENGTH is a load failure, not
    // a subtle quality loss -- worth stating explicitly since it is the field the flat config
    // scanner used to miss (it lives nested under "dflash_config").
    printf("               expected fc width = %d (= %zu captures * %d hidden)\n",
           (int)c.target_layer_ids.size() * c.hidden, c.target_layer_ids.size(), c.hidden);

    printf("%s\n", ok ? "LOAD_OK" : "LOAD_FAILED");
    return ok ? 0 : 1;
}
