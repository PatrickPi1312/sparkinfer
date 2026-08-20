#!/usr/bin/env python3
"""Layer-by-layer differential for the DSpark draft backbone.

Reduces "the draft is uniformly ~30% accurate at every position" to "layer N is where our CUDA
forward and a NumPy reference diverge". Everything on either side of the backbone has already been
cleared bit-exact by dspark_reference_diff.py: target_proj (corr 0.999997), the Markov bias
(max|d| 0), the NVFP4 dequant, the injected-KV positions and the aux-layer capture point. The
per-position acceptance profile is flat at ~30%, which is a GLOBAL fault in the forward rather than
anything sequential -- so the five attention layers are the only stage left.

Reproduces one layer exactly as dflash_draft.cpp's forward_block does it:

    xn      = rmsnorm(x, input_layernorm)
    q,k,v   = xn @ {q,k,v}_proj.T          (q/k/v are BF16 in this checkpoint)
    q,k     = rope(headnorm(q)), rope(headnorm(k))     YaRN table, att_scale folded into q
    k_ctx   = rope(headnorm(target_proj @ k_proj.T))   the INJECTED context, positions 0..ctx-1
    ao      = attn(q, [k_ctx; k], [v_ctx; v]) @ o_proj.T
    h       = x + ao
    hn      = rmsnorm(h, post_attention_layernorm)
    out     = h + (silu(hn @ gate.T) * (hn @ up.T)) @ down.T

    python3 bench/scripts/dspark_backbone_diff.py <dump_dir> <draft_dir> [n_layers]
"""
import json
import math
import struct
import sys

import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def open_st(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    return f, json.loads(f.read(n)), 8 + n


def raw(f, hdr, base, k):
    s, e = hdr[k]["data_offsets"]
    f.seek(base + s)
    return f.read(e - s)


def bf16(buf, shape):
    a = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return a.view(np.float32).reshape(shape)


def e4m3(u):
    u = u.astype(np.uint32)
    s, e, m = (u >> 7) & 1, (u >> 3) & 0xF, u & 0x7
    v = np.where(e == 0, (m / 8.0) * 2.0 ** -6, (1.0 + m / 8.0) * np.exp2(e.astype(np.float32) - 7))
    return np.where(s == 1, -v, v).astype(np.float32)


def tensor(f, hdr, base, k):
    h = hdr[k]
    buf = raw(f, hdr, base, k)
    if h["dtype"] == "BF16":
        return bf16(buf, h["shape"])
    if h["dtype"] == "F32":
        return np.frombuffer(buf, dtype=np.float32).reshape(h["shape"])
    rows, pc = h["shape"]
    cols = pc * 2
    by = np.frombuffer(buf, dtype=np.uint8).reshape(rows, pc)
    nib = np.empty((rows, cols), dtype=np.uint8)
    nib[:, 0::2] = by & 0x0F
    nib[:, 1::2] = by >> 4
    val = np.where(nib & 0x8, -E2M1[nib & 0x7], E2M1[nib & 0x7]).astype(np.float32)
    gs = e4m3(np.frombuffer(raw(f, hdr, base, k + "_scale"), dtype=np.uint8)).reshape(rows, cols // 16)
    g2 = np.frombuffer(raw(f, hdr, base, k + "_scale_2"), dtype=np.float32)[0]
    return val * np.repeat(gs, 16, axis=1) * g2


def rms(x, w, eps=1e-6):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def yarn_inv_freq(cfg, d):
    """Mirrors compute_yarn_inv_freq in dflash_draft.cpp, which mirrors HF."""
    rp = cfg.get("rope_parameters", {}) or {}
    base = float(rp.get("rope_theta", cfg.get("rope_theta", 10000.0)))
    factor = float(rp.get("factor", 1.0))
    orig = float(rp.get("original_max_position_embeddings", 8192))
    bf, bs = float(rp.get("beta_fast", 32.0)), float(rp.get("beta_slow", 1.0))
    half = d // 2
    if str(rp.get("rope_type", "")) != "yarn" or factor <= 1.0:
        return 1.0 / (base ** (np.arange(half, dtype=np.float64) * 2.0 / d)), 1.0
    fd = lambda nrot: (d * math.log(orig / (nrot * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(math.floor(fd(bf)), 0.0), min(math.ceil(fd(bs)), d - 1.0)
    if high - low < 1e-3:
        high = low + 1e-3
    i = np.arange(half, dtype=np.float64)
    pos_freq = base ** (2.0 * i / d)
    ramp = np.clip((i - low) / (high - low), 0.0, 1.0)
    extrap_w = 1.0 - ramp
    inv = (1.0 / (factor * pos_freq)) * (1.0 - extrap_w) + (1.0 / pos_freq) * extrap_w
    return inv, 0.1 * math.log(factor) + 1.0


def rope(v, pos, inv, d):
    """v: [T, heads, d]. Rotates the first len(inv)*2 dims, NeoX-style halves."""
    half = len(inv)
    ang = np.outer(np.asarray(pos, dtype=np.float64), inv)
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = v[..., :half].astype(np.float64), v[..., half:half * 2].astype(np.float64)
    out = v.astype(np.float64).copy()
    out[..., :half] = a * cos - b * sin
    out[..., half:half * 2] = b * cos + a * sin
    return out.astype(np.float32)


def report(tag, ours, ref):
    o, r = ours.astype(np.float32).ravel(), ref.astype(np.float32).ravel()
    corr = np.corrcoef(o, r)[0, 1] if o.std() > 0 and r.std() > 0 else float("nan")
    d = np.abs(o - r).max() / (np.abs(r).max() + 1e-9)
    print("  %-12s corr=%.6f  rel_max=%.5f  |ours|/|ref|=%.4f  %s"
          % (tag, corr, d, (np.abs(o).mean() + 1e-12) / (np.abs(r).mean() + 1e-12),
             "MATCH" if corr > 0.999 and d < 0.05 else "*** DIVERGES ***"))
    return corr


def main():
    dump, ck = sys.argv[1], sys.argv[2]
    nlayers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    meta = {k: int(v) for k, v in (l.split() for l in open(dump + "/meta.txt"))}
    ctx, H, BW, pos0 = meta["ctx_len"], meta["H"], meta["BW"], meta["pos0"]
    cfg = json.load(open(ck + "/config.json"))
    nq, nkv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    f, hdr, base = open_st(ck + "/model.safetensors")
    T = lambda k: tensor(f, hdr, base, k)
    rd = lambda nm, shape: bf16(open(dump + "/" + nm + ".bin", "rb").read(), shape)

    x = rd("noise_embed", (BW, H))
    tp = rd("target_proj", (ctx, H))
    inv, att_scale = yarn_inv_freq(cfg, hd)
    print("ctx=%d BW=%d pos0=%d  heads=%d/%d hd=%d  yarn att_scale=%.4f" %
          (ctx, BW, pos0, nq, nkv, hd, att_scale))

    for L in range(nlayers):
        p = "layers.%d." % L
        xn = rms(x, T(p + "input_layernorm.weight"), eps)
        q = (xn @ T(p + "self_attn.q_proj.weight").T).reshape(BW, nq, hd)
        k = (xn @ T(p + "self_attn.k_proj.weight").T).reshape(BW, nkv, hd)
        v = (xn @ T(p + "self_attn.v_proj.weight").T).reshape(BW, nkv, hd)
        qn, kn = T(p + "self_attn.q_norm.weight"), T(p + "self_attn.k_norm.weight")
        # att_scale is folded into cos/sin and applied to BOTH q and k -- matching HF
        # (cos = emb.cos() * attention_scaling, used for q_embed and k_embed alike) and matching
        # k_rms_heads_rope, which multiplies c and sn by att_scale for whichever tensor it is
        # given. Applying it to q alone is the natural reading of YaRN's attention temperature and
        # is NOT what the checkpoint was trained under.
        q = rms(q, qn, eps) * att_scale
        k = rms(k, kn, eps) * att_scale
        q = rope(q, np.arange(pos0, pos0 + BW), inv, hd)
        k = rope(k, np.arange(pos0, pos0 + BW), inv, hd)
        # injected context KV, at absolute positions pos0-ctx .. pos0-1
        kc = rms((tp @ T(p + "self_attn.k_proj.weight").T).reshape(ctx, nkv, hd), kn, eps) * att_scale
        kc = rope(kc, np.arange(pos0 - ctx, pos0), inv, hd)
        vc = (tp @ T(p + "self_attn.v_proj.weight").T).reshape(ctx, nkv, hd)
        K = np.concatenate([kc, k], 0)
        V = np.concatenate([vc, v], 0)
        scale = 1.0 / math.sqrt(hd)
        ao = np.zeros((BW, nq * hd), dtype=np.float32)
        rep = nq // nkv
        for h in range(nq):
            s = (q[:, h, :] @ K[:, h // rep, :].T) * scale
            for t in range(BW):                      # causal within the block
                s[t, ctx + t + 1:] = -np.inf
            s = s - s.max(-1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(-1, keepdims=True)
            ao[:, h * hd:(h + 1) * hd] = w @ V[:, h // rep, :]
        hres = x + ao @ T(p + "self_attn.o_proj.weight").T
        hn = rms(hres, T(p + "post_attention_layernorm.weight"), eps)
        g = hn @ T(p + "mlp.gate_proj.weight").T
        u = hn @ T(p + "mlp.up_proj.weight").T
        out = hres + ((g / (1.0 + np.exp(-g))) * u) @ T(p + "mlp.down_proj.weight").T
        ours = rd("layer%d_x" % L, (BW, H))
        report("layer%d_x" % L, ours, out)
        x = ours          # re-seed from OURS so each layer is scored on its own, not cumulatively


if __name__ == "__main__":
    main()
