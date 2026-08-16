#pragma once

// Qwen3.8-27B Multi-Token Prediction (MTP) head.
//
// Ships in the compressed-tensors checkpoint as a SEPARATE shard, model_mtp.safetensors, referenced
// from model.safetensors.index.json. The main loader already tolerates its absence
// ("[safetensors] shard model_mtp.safetensors not present -- skipping"), so a checkpoint without
// the file keeps working and MTP simply stays unavailable.
//
// Structure (all 15 tensors confirmed by reading the shard's own safetensors header, not assumed
// from the config):
//
//   mtp.pre_fc_norm_embedding.weight   [5120]            RMSNorm on the token-embedding branch
//   mtp.pre_fc_norm_hidden.weight      [5120]            RMSNorm on the target-hidden branch
//   mtp.fc.weight                      [5120, 10240]     [h_norm ; e_norm] (2*hidden) -> hidden
//   mtp.layers.0.*                                       ONE decoder layer (mtp_num_hidden_layers=1)
//   mtp.norm.weight                    [5120]            final norm before the SHARED lm_head
//
// Forward (order matches sglang's qwen3_next_mtp.py, whose parameter names are 1:1 with this
// checkpoint's -- corroboration, not a substitute for the shapes above):
//
//   e = pre_fc_norm_embedding(embed_tokens[tok_{t+1}])     <- embed_tokens SHARED with the target
//   h = pre_fc_norm_hidden(hidden_t)                       <- target's final hidden at position t
//   x = fc([e ; h])                                        <- concat on the feature axis
//   x = layer0(x)                                          <- full attention + SwiGLU MLP
//   logits = lm_head(mtp.norm(x))                          <- lm_head SHARED with the target
//                                                          => predicts tok_{t+2}
//
// mtp_use_dedicated_embeddings=false in text_config, hence the two SHARED weights above; they are
// bound non-owning from the target, exactly as DFlashDraftModel::set_shared_weights already does.
//
// THREE Qwen3.8-specific traps, all of which cost real time during the main model's bring-up and
// every one of which applies identically here:
//
//   1. The attention OUTPUT GATE is fused into q_proj's width. q_proj is [12288, 5120] while
//      o_proj takes 6144 = 24 heads * 256. 12288 = 48 * 256 = 24 Q heads + 24 GATE heads, the same
//      packing the target's full-attention layers use. Split q|gate per head before attention.
//   2. That gate is a plain SIGMOID, despite text_config's output_gate_type: "swish". silu
//      sign-flips the gated output (the gate mean is ~ -4.5). See qwen_config.h's note.
//   3. HF stores Linear weights [out_features, in_features], which is byte-identical to the
//      GGUF-native [out,in] every kernel here wants. Do NOT transpose -- an earlier relayout in
//      load_compressed_tensors silently mis-shaped every projection in the model.
//
// Quantization: NONE. quantization_config.ignore ends with "re:^mtp.*", so all 15 tensors are
// plain BF16 (~810 MiB). They are kept bf16 rather than requantized to Q4_K like the target's own
// projections: the head is the DRAFT in speculative decoding, so its fidelity sets the acceptance
// rate, and 810 MiB fits the measured headroom (~1614 MiB free with the target resident).

#include "sparkinfer/models/qwen_config.h"

#include <cuda_runtime.h>

#include <string>

namespace sparkinfer {

class SafeTensorsModel;

struct Qwen38MtpConfig {
    int   hidden       = 5120;
    int   n_q_heads    = 24;
    int   n_kv_heads   = 4;
    int   head_dim     = 256;
    int   intermediate = 17408;
    int   vocab        = 248320;
    float rms_eps      = 1e-6f;
    float rope_theta   = 10000000.f;
    int   max_seq      = 8192;
};

// Derives the MTP config from the already-parsed target config. Every field is shared with the
// target's full-attention layers (verified against the shard header: q 24*256, kv 4*256,
// ffn 17408, hidden 5120), so there is no independent source of truth to drift from.
Qwen38MtpConfig qwen38_mtp_config_from(const Qwen35Config& cfg);

class Qwen38MtpHead {
public:
    explicit Qwen38MtpHead(const Qwen38MtpConfig& cfg);
    ~Qwen38MtpHead();

    Qwen38MtpHead(const Qwen38MtpHead&) = delete;
    Qwen38MtpHead& operator=(const Qwen38MtpHead&) = delete;

    // Uploads the 15 bf16 tensors from an already-open compressed-tensors checkout. Returns false
    // (with a diagnostic on stderr) if the mtp shard is absent or any tensor is missing or has an
    // unexpected dtype/shape -- never guesses, since a silently-wrong draft shows up as a poor
    // acceptance rate rather than a crash, which is far harder to attribute.
    bool load(SafeTensorsModel& st);

    // Non-owning device pointers borrowed from the target (mtp_use_dedicated_embeddings=false).
    // `lm_head_type` is the target's ggml type id for its head (0 = bf16 dense).
    void set_shared_weights(const void* embed_bf16, const void* lm_head, int lm_head_type);

    const Qwen38MtpConfig& config() const;

    // True once load() and set_shared_weights() have both succeeded.
    bool ready() const;

    size_t vram_bytes() const;

    // Allocates scratch + the head's OWN paged KV cache. Must be called once before forward().
    // The MTP layer is a full-attention layer with its own K/V history, entirely separate from
    // the target's: it attends over the draft's own accepted prefix, so it cannot share the
    // target's pool. ~67 MiB bf16 at max_seq=16384 (4 kv heads * 256 * 2 * 16384 * 2B).
    bool init_runtime(int max_seq);

    // One draft step.
    //   target_hidden : [hidden] bf16, the target's FINAL hidden state at position `pos`
    //                   (post-last-layer, PRE final norm -- the target's own norm is not applied,
    //                   MTP has its own pre_fc_norm_hidden for this branch).
    //   next_token_id : the token the target just committed at position `pos`+1, whose embedding
    //                   is the second fc input. MTP predicts the token AFTER it.
    //   pos           : absolute position for the MTP layer's RoPE and KV write.
    //   out_argmax    : host, [1]. The proposed token id.
    // Greedy only -- speculative decoding needs exact argmax determinism to stay lossless, the
    // same constraint dflash_generate documents.
    bool forward(const void* target_hidden, int next_token_id, int pos,
                 int* out_argmax, cudaStream_t stream = nullptr);

    // Drop the draft's KV back to `keep` tokens after a rejected proposal. Without this the draft
    // desynchronises from the target on the first mismatch and every later proposal is scored
    // against a history the target never had.
    void crop(int keep);
    void reset();
    int  seq_len() const;

private:
    struct Impl;
    Impl* p_;
};

}   // namespace sparkinfer
