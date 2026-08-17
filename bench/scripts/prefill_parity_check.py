#!/usr/bin/env python3
"""Does batched prefill leave the model in the same state the token loop does?

  prefill_parity_check.py <model_dir_or_gguf> <tokenizer.json> [n1,n2,...]

Generates the SAME continuation from the SAME real-token prompt twice -- once with
SPARKINFER_PREFILL_BATCHED=0 (the token-by-token reference path) and once with it on -- and
reports how long the two agree. A healthy batched path shares a long common prefix; a broken one
answers a different question entirely.

WHY THIS EXISTS, and why the existing gate does not cover it:

The PR accuracy gate (bench/scripts/accuracy_compare_pair.py, top1 >= 0.99 / KL <= 0.01) is fed by
runtime/examples/qwen3_gguf_score.cpp, which teacher-forces the corpus through `forward_token()`
one position at a time. That call never enters prefill_batched_run(). So the gate is structurally
incapable of observing batched-prefill correctness, while the bench it sits next to reports
prefill throughput -- which is measured on exactly the path the gate cannot see. Twelve perf PRs
(#835 through #854) landed against that arrangement.

Use REAL tokens, never synthetic ids. runtime/examples/qwen3_gguf_prefill_check.cpp compares
logits over a prompt of `100 + (i % 20000)`, which is not text: the resulting distribution is
near-uniform, argmax flips on ordinary numerical noise, and the harness reports 0.6-0.9 top-1
agreement at EVERY length -- including lengths where the path is fine. That noise floor is what let
a real divergence sit underneath it. Peaked, in-distribution logits are what make the check sharp.
"""
import json
import os
import subprocess
import sys

from tokenizers import Tokenizer

BIN = os.environ.get("SPARKINFER_GENERATE_BIN", "build/runtime/qwen3_gguf_generate")
NEW = int(os.environ.get("PARITY_NEW_TOKENS", "24"))
# Ordinary prose, so the continuation is strongly determined and a disagreement means something.
# Long enough that n=128 actually runs -- the original ~60-token version made every 128-length
# check print "prompt corpus too short" and silently skip, leaving only n=32 as real coverage
# right as the accuracy bar was loosened to top1 0.90 / KL 0.1 and parity became the harness
# actually catching regressions.
PROSE = (
    "Artificial intelligence is a field of computer science that builds systems able to perform "
    "tasks which normally require human intelligence, such as understanding language, recognising "
    "images, and making decisions under uncertainty. Researchers have pursued this goal since the "
    "middle of the twentieth century, and progress has come in waves rather than steadily. "
    "Early efforts focused on symbolic reasoning and hand-written rules, which worked well in "
    "narrow, well-defined domains but struggled to generalise to the messiness of everyday "
    "perception and language. A shift toward statistical methods in the late twentieth century "
    "let systems learn patterns directly from data rather than from rules an engineer wrote by "
    "hand, and this approach scaled remarkably well as computing power and available data both "
    "grew. The rise of deep neural networks in the following decades pushed performance further "
    "still, first in image recognition and then in natural language processing, where models "
    "trained on vast text corpora learned to produce fluent, contextually appropriate responses. "
    "Despite these advances, fundamental questions remain open: how much of what looks like "
    "understanding is genuine reasoning versus sophisticated pattern matching, how reliably these "
    "systems generalise beyond their training distribution, and how to ensure their behaviour "
    "stays aligned with human intent as their capabilities continue to grow."
)


def gen(model: str, ids, batched: bool):
    env = dict(os.environ, SPARKINFER_PREFILL_BATCHED="1" if batched else "0")
    r = subprocess.run([BIN, model, str(NEW)] + [str(i) for i in ids],
                       capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("OUTPUT_IDS:"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    sys.stderr.write(r.stdout[-500:] + r.stderr[-500:] + "\n")
    return None


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    model, tok_path = sys.argv[1], sys.argv[2]
    lengths = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [16, 32, 64, 128]
    tok = Tokenizer.from_file(tok_path)
    all_ids = tok.encode(PROSE, add_special_tokens=False).ids

    worst, rows = None, []
    for n in lengths:
        if n > len(all_ids):
            print(f"n={n}: prompt corpus too short ({len(all_ids)} tokens), skipped")
            continue
        ids = all_ids[:n]
        ref, got = gen(model, ids, False), gen(model, ids, True)
        if ref is None or got is None:
            print(f"n={n:4d}  RUN FAILED (ref={ref is not None} batched={got is not None})")
            return 1
        agree = 0
        while agree < min(len(ref), len(got)) and ref[agree] == got[agree]:
            agree += 1
        frac = agree / max(1, len(ref))
        worst = frac if worst is None else min(worst, frac)
        rows.append((n, agree, len(ref), frac))
        print(f"n={n:4d}  common-prefix {agree:3d}/{len(ref):<3d} = {frac:.3f}")
        print(f"        token-loop: {tok.decode(ref)[:88]!r}")
        print(f"        batched   : {tok.decode(got)[:88]!r}")

    if not rows:
        print("PARITY no lengths run")
        return 1
    # Deliberately NOT pinning SPARKINFER_PREFILL_I8=0 / SPARKINFER_PREFILL_SKINNY_SPLITK=0 /
    # SPARKINFER_NSPLITS=1 here, unlike dspark_tau_check.cpp's bit-exact lossless check (found
    # 2026-08-17: both prefill_gemm_i8.cu and prefill_gemm_skinny.cu accumulate via atomicAdd,
    # genuinely order-dependent on real hardware). Those pins are right for DSpark's verify, whose
    # correctness guarantee REQUIRES bit-exactness -- they would be wrong here, where the whole
    # point is to reflect production behavior: this repo's own "fine for serving, fatal for
    # verify" split (see item 5 of the 2026-08-17 handoff) means the atomic kernels stay ON in
    # production, so a parity check with them pinned OFF would validate a configuration nothing
    # actually serves. The tolerant PARITY_BAR (0.75, not 1.0) already exists because SOME
    # natural divergence is expected for exactly this reason -- treat that as the intended
    # absorption of split-K noise, not a bar to tighten.
    bar = float(os.environ.get("PARITY_BAR", "0.75"))
    print(f"PARITY worst={worst:.3f} bar={bar} {'OK' if worst >= bar else 'FAIL'}")
    return 0 if worst >= bar else 1


if __name__ == "__main__":
    sys.exit(main())
