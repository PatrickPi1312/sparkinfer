#pragma once
// Config derivation for the Muse Glimmer DFlash DRAFT checkpoint (dflash-kquant.gguf),
// mirroring qwen3_gguf_config.h's museglimmer_config_from_gguf for the TARGET model.
//
// general.architecture = "dflash" for this file, with every field under a flat "dflash." prefix
// (no fallback-prefix chain needed -- unlike qwen3_meta_int, there is only ever one namespace).

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/dflash_draft.h"

#include <string>
#include <vector>

static long dflash_meta_int(const sparkinfer::GGUF& g, const std::string& key, long def) {
    return g.meta_int("dflash." + key, def);
}
static double dflash_meta_float(const sparkinfer::GGUF& g, const std::string& key, double def) {
    return g.meta_float("dflash." + key, def);
}

// Populates every DFlashDraftConfig field from a Muse Glimmer DFlash draft GGUF's metadata table.
// Call only after confirming g.meta_str("general.architecture") == "dflash" (DFlashDraftModel::
// load_gguf does this check itself before calling in).
static void museglimmer_dflash_config_from_gguf(const sparkinfer::GGUF& g,
                                                 sparkinfer::DFlashDraftConfig& cfg) {
    cfg.n_layers   = (int)dflash_meta_int(g, "block_count", 5);
    cfg.hidden     = (int)dflash_meta_int(g, "embedding_length", 6656);
    cfg.intermediate = (int)dflash_meta_int(g, "feed_forward_length", 19968);
    cfg.n_q_heads  = (int)dflash_meta_int(g, "attention.head_count", 32);
    cfg.n_kv_heads = (int)dflash_meta_int(g, "attention.head_count_kv", 8);
    cfg.head_dim   = (int)dflash_meta_int(g, "attention.key_length", 128);
    cfg.rms_eps    = (float)dflash_meta_float(g, "attention.layer_norm_rms_epsilon", 1e-5f);
    cfg.rope_theta = (float)dflash_meta_float(g, "rope.freq_base", 500000.f);
    cfg.sliding_window = (int)dflash_meta_int(g, "attention.sliding_window", 2048);
    cfg.block_size = (int)dflash_meta_int(g, "block_size", 16);
    cfg.mask_token_id = (int)g.meta_int("tokenizer.ggml.mask_token_id", 201818);

    // No separate vocab_size key in this file -- the draft shares the target's embedding/lm_head
    // (DFlashDraftModel::set_shared_weights), and that call (not this one) is what actually binds
    // vocab at runtime (Impl::vocab, used by forward_block's V). cfg.vocab only sizes the block's
    // logits SCRATCH buffer at load time (see DFlashDraftModel::load_gguf), so it just needs to be
    // >= the real target vocab, not exact. Use Muse Glimmer's known target vocab (same default
    // museglimmer_config_from_gguf uses in qwen3_gguf_config.h) rather than the struct's
    // Qwen3.6-sized default (248320) -- both are big enough, this is just the closer number.
    cfg.vocab = 202048;

    // Muse Glimmer's own RoPE convention: llama.cpp's llama_model_rope_type() places
    // LLM_ARCH_MUSE_GLIMMER in the LLAMA_ROPE_TYPE_NORM (consecutive-pair) bucket, not the NEOX
    // (split-half) bucket every existing DFlash draft kernel was written for (Qwen3.6). This draft
    // checkpoint is the same model family (same rope_theta, same tokenizer, general.name =
    // "Hf_Museglimmer") and its whole mechanism depends on its hidden states lining up with the
    // target's, so it is very likely -- but UNVERIFIED as of this change, since no DFlash
    // accuracy/SPEC_AGREE evaluation has been run against this draft yet -- that it needs the same
    // fix already made and validated on the target model. See k_rms_heads_rope_normal in
    // dflash_kernels.cu for the kernel and the full reasoning. If Muse Glimmer draft proposals look
    // wrong once evaluation runs, check this flag FIRST.
    cfg.rope_normal = true;

    // Per-layer sliding-window (true) vs full attention (false), read directly from the GGUF array.
    // Unlike the target (whose pattern is roughly "every 4th layer full"), every layer of this
    // draft checkpoint is sliding-window per the metadata -- read faithfully rather than assumed.
    std::vector<long> pattern = g.meta_int_array("dflash.attention.sliding_window_pattern");
    cfg.sliding_layers.assign(cfg.n_layers, true);
    for (int i = 0; i < cfg.n_layers && i < (int)pattern.size(); i++)
        cfg.sliding_layers[i] = pattern[i] != 0;

    // target_layer_ids: dflash.target_layers minus 1.
    //
    // Upstream's dflash.target_layers metadata ([2,14,26,38,50] in the released checkpoint) uses
    // the convention "capture layer N's INPUT" -- confirmed against llama.cpp-adjacent upstream
    // sources, which capture t_layer_inp[il] before layer il runs.
    //
    // This codebase's capture point is different: Qwen35Model::dflash_maybe_capture_layer
    // (qwen35.cpp) does a bare `s.dflash_layer_ids[i] == layer` check with NO adjustment, fired at
    // the END of layer L's processing inside the 0-indexed layer loop -- i.e. it captures layer L's
    // OUTPUT, which is equivalently layer (L+1)'s INPUT. So "capture layer N's INPUT" in upstream's
    // convention corresponds to "layer (N-1)'s OUTPUT" here, i.e. target_layer_ids[i] = raw[i] - 1.
    //
    // Get this right: a silent off-by-one here compiles, runs, and even mostly "works" (plausible
    // but wrong draft proposals, never a crash) -- exactly the failure mode this whole bring-up
    // (target AND draft) has been chasing. Do not "simplify" this away.
    std::vector<long> raw_target_layers = g.meta_int_array("dflash.target_layers");
    cfg.target_layer_ids.clear();
    for (long v : raw_target_layers) cfg.target_layer_ids.push_back((int)v - 1);
}
