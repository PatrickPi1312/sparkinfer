#pragma once

#include <cstdlib>

namespace sparkinfer {

// SPARKINFER_DETERMINISTIC=1 -- opt-in bit-reproducible mode.
//
// WHAT IT IS FOR. External verifiers (and our own lossless checks) need "same prompt, same build,
// same GPU model => same tokens and the same logprobs, every time". By default sparkinfer is not
// that: greedy decode forks run to run, and the logprob of a fixed token at a fixed position moves
// by a few tenths of a nat. Measured on an RTX 5090, Qwen3.6-35B-A3B UD-Q4_K_M, 12 prompts x 3
// repeats at 48 tokens: 1/36 runs bit-identical, 13/36 forked their token sequence, mean drift
// 0.44 nats. With this mode on, all 36 are bit-identical.
//
// WHERE THE NONDETERMINISM ACTUALLY IS. Not in decode -- the decode path contains no float
// atomics at all, its flash-decode split combine is a fixed-order fold, and its MoE expert FFN
// writes each output once. It is in BATCHED PROMPT PREFILL, in two fp32 atomicAdd accumulations
// whose operand order is decided by hardware arbitration:
//
//   1. the routed MoE down-projection combine (prefill_moe.cu / prefill_moe_q.cu, C_SCATTER),
//      where a token's top_k expert contributions race into one accumulator;
//   2. the split-K reduction in the narrow-N prefill GEMM used by the Gated-DeltaNet gate
//      projections (prefill_gemm_skinny.cu).
//
// Both produce differences of a few ULP, which would be irrelevant if they stayed numerical --
// but they feed discrete top-k expert ROUTING and int8 activation requant at the next layer, so
// over 40+ layers one occasionally flips an expert, which moves the logits enough to change the
// argmax. That is why the symptom (early greedy forks, ~0.3 nat swings) looks far larger than the
// cause, and why it is seeded by the prompt pass rather than by decode.
//
// WHAT IT COSTS. Decode is untouched -- unchanged tokens/s. Prefill pays for the deterministic
// combine's extra pass and for the skinny GEMM losing its split-K fan-out.
//
// SCOPE. Bit-exactness holds for a fixed (build, GPU model, batch=1) triple. A few launch
// geometries elsewhere in the tree are chosen from the device's SM count, so two DIFFERENT GPU
// models are not promised to agree with each other even in this mode; verify such a pair
// explicitly rather than assuming it.
inline bool deterministic_mode() {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_DETERMINISTIC");
        return e && e[0] != '0';
    }();
    return on;
}

}  // namespace sparkinfer
