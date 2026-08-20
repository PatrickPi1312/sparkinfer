#!/usr/bin/env python3
"""Numerical differential: our CUDA DSpark draft vs a NumPy reference, tensor by tensor.

Every structural hypothesis about the acceptance gap has been eliminated -- block construction,
row->proposal indexing, context injection, injected-KV positions, NVFP4 scale convention, the quant
ignore list, and the aux-layer capture point all match sglang. Acceptance is nonetheless ~1.36
against a reported ~3.8 for the same checkpoint. What is left can only be found by comparing
numbers, so this recomputes one block from the same bytes and reports where the two separate.

Inputs come from SPARKINFER_DSPARK_DUMP=<dir> (see dflash_draft.cpp), which writes one block's
worth of inputs and intermediates on the first forward_block call of a process.

The chain, in the order a divergence would propagate:

    1. target_proj = hidden_norm(fc(target_hidden))     <- the projected target features
    2. per-layer K/V from target_proj                   <- what the draft attends over
    3. draft backbone over the block                    <- 5 full-attention layers
    4. base logits = lm_head(final hidden)
    5. markov bias = w2 @ w1[prev]                      <- the sequential head

Stage 1 and stage 5 need no attention and are checked first: they are cheap, they are where a
convention error (transpose, epsilon, scale) would land, and if either is wrong nothing downstream
is worth comparing.

    python3 bench/scripts/dspark_reference_diff.py <dump_dir> <draft_checkpoint_dir>
"""
import json
import struct
import sys

import numpy as np


def load_safetensors(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    return f, hdr, base


def raw(f, hdr, base, key):
    s, e = hdr[key]["data_offsets"]
    f.seek(base + s)
    return f.read(e - s), hdr[key]


def bf16_to_f32(buf, shape):
    a = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return a.view(np.float32).reshape(shape)


E2M1_MAG = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def decode_e4m3(u):
    """Signed float8_e4m3fn, matching dflash_draft.cpp's decode_e4m3."""
    u = u.astype(np.uint32)
    s, e, m = (u >> 7) & 1, (u >> 3) & 0xF, u & 0x7
    v = np.where(e == 0, (m / 8.0) * 2.0 ** -6, (1.0 + m / 8.0) * np.exp2(e.astype(np.float32) - 7))
    return np.where(s == 1, -v, v).astype(np.float32)


def tensor(f, hdr, base, key):
    """Dequantize to f32 regardless of storage, mirroring the C++ loader's routing by dtype."""
    if key not in hdr:
        return None
    buf, h = raw(f, hdr, base, key)
    dt = h["dtype"]
    if dt == "BF16":
        return bf16_to_f32(buf, h["shape"])
    if dt == "F32":
        return np.frombuffer(buf, dtype=np.float32).reshape(h["shape"])
    if dt == "U8":  # NVFP4: e2m1 nibbles * e4m3 group scale * f32 global scale
        rows, packed_cols = h["shape"]
        cols = packed_cols * 2
        by = np.frombuffer(buf, dtype=np.uint8).reshape(rows, packed_cols)
        nib = np.empty((rows, cols), dtype=np.uint8)
        nib[:, 0::2] = by & 0x0F
        nib[:, 1::2] = by >> 4
        val = np.where(nib & 0x8, -E2M1_MAG[nib & 0x7], E2M1_MAG[nib & 0x7]).astype(np.float32)
        sb, _ = raw(f, hdr, base, key + "_scale")
        gs = decode_e4m3(np.frombuffer(sb, dtype=np.uint8)).reshape(rows, cols // 16)
        g2b, _ = raw(f, hdr, base, key + "_scale_2")
        g2 = np.frombuffer(g2b, dtype=np.float32)[0]
        return val * np.repeat(gs, 16, axis=1) * g2
    raise ValueError("unhandled dtype %s for %s" % (dt, key))


def report(name, ours, ref):
    """One line per compared tensor. Correlation is the headline: a scale or epsilon error keeps
    correlation ~1 while max|d| is large, whereas a transpose or wrong-source error destroys it."""
    ours = ours.astype(np.float32).ravel()
    ref = ref.astype(np.float32).ravel()
    d = np.abs(ours - ref)
    denom = np.abs(ref).max() + 1e-9
    corr = np.corrcoef(ours, ref)[0, 1] if ours.std() > 0 and ref.std() > 0 else float("nan")
    ratio = (np.abs(ours).mean() + 1e-12) / (np.abs(ref).mean() + 1e-12)
    verdict = "MATCH" if corr > 0.999 and d.max() / denom < 0.02 else "*** DIVERGES ***"
    print("  %-16s corr=%.6f  max|d|=%.5f  rel=%.5f  |ours|/|ref|=%.4f   %s"
          % (name, corr, d.max(), d.max() / denom, ratio, verdict))
    return corr


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    dump, ckpt = sys.argv[1], sys.argv[2]
    meta = {}
    for line in open(dump + "/meta.txt"):
        k, v = line.split()
        meta[k] = int(v)
    ctx, H, ncap = meta["ctx_len"], meta["H"], meta["n_cap"]
    V, BW, rank = meta["V"], meta["BW"], meta["markov_rank"]
    print("dump: ctx_len=%d H=%d n_cap=%d BW=%d depth=%d V=%d rank=%d"
          % (ctx, H, ncap, BW, meta["depth"], V, rank))

    f, hdr, base = load_safetensors(ckpt + "/model.safetensors")
    rd = lambda nm, shape, dt=np.float32: (
        bf16_to_f32(open(dump + "/" + nm + ".bin", "rb").read(), shape) if dt is np.float32
        else np.frombuffer(open(dump + "/" + nm + ".bin", "rb").read(), dtype=dt).reshape(shape))

    th = rd("target_hidden", (ctx, ncap * H))
    tp_ours = rd("target_proj", (ctx, H))
    logits_ours = np.frombuffer(open(dump + "/logits.bin", "rb").read(),
                                dtype=np.float32).reshape(BW, V)
    ids = np.frombuffer(open(dump + "/noise_ids.bin", "rb").read(), dtype=np.int32)
    dout = np.frombuffer(open(dump + "/d_out.bin", "rb").read(), dtype=np.int32)
    print("noise_ids=%s  d_out=%s" % (ids.tolist(), dout.tolist()))

    print("\n[1] target_proj = hidden_norm(fc(target_hidden))")
    fc = tensor(f, hdr, base, "fc.weight")
    hn = tensor(f, hdr, base, "hidden_norm.weight")
    if fc is None or hn is None:
        print("  fc/hidden_norm not found; candidates:")
        for k in hdr:
            if k != "__metadata__" and ("fc" in k or "norm" in k) and "layers" not in k:
                print("    ", k, hdr[k]["dtype"], hdr[k]["shape"])
    else:
        print("  fc%s hidden_norm%s" % (fc.shape, hn.shape))
        x = th @ fc.T
        ref = (x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)) * hn
        report("target_proj", tp_ours, ref)

    print("\n[5] markov bias = w2 @ w1[prev]   (independent of attention; BF16 weights)")
    w1 = tensor(f, hdr, base, "markov_head.markov_w1.weight")
    w2 = tensor(f, hdr, base, "markov_head.markov_w2.weight")
    if w1 is None or w2 is None:
        print("  markov head tensors not found; candidates:")
        for k in hdr:
            if "markov" in k:
                print("    ", k, hdr[k]["dtype"], hdr[k]["shape"])
    else:
        print("  w1%s w2%s  (rank=%d)" % (w1.shape, w2.shape, rank))
        prev = int(ids[0])
        bias = w2 @ w1[prev] if w2.shape[1] == w1.shape[1] else w2.T @ w1[prev]
        print("  bias[prev=%d]: std=%.5f  absmax=%.4f  argmax=%d"
              % (prev, bias.std(), np.abs(bias).max(), int(bias.argmax())))
        # Row 1's logits already include the bias in our path; row 0 never does. Their difference
        # isolates it only if the two rows' base logits were equal, which they are not -- so report
        # the bias scale against the logit scale instead, which is what a magnitude error shows up in.
        print("  our logits row0 std=%.4f  row1 std=%.4f  (bias std should be a fraction of these)"
              % (logits_ours[0].std(), logits_ours[1].std()))


if __name__ == "__main__":
    main()
