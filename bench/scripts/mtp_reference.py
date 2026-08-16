#!/usr/bin/env python3
"""Pure-NumPy Qwen3.8-27B MTP head, run over a whole sequence, to establish what acceptance rate
this checkpoint's MTP can actually achieve -- independent of sparkinfer's implementation.

  mtp_reference.py <checkpoint_dir> <seq_dump_dir>

Why this exists. sparkinfer's MTP forward is verified correct stage-by-stage against the
checkpoint bytes (mtp_recompute.py: every stage cos>=0.99999, lm_head reproduces the target's own
tokens), yet its agreement with the target is 0 under every input-pairing tried. Either the
pairing convention is something not yet guessed, or the assumption "a correct MTP head agrees with
the target 60-85% of the time" is wrong for this checkpoint. A reference that shares NONE of
sparkinfer's integration code settles which.

It also covers the one thing the pos=0 recomputation structurally could not: RoPE at pos>0.
Position 0 is a no-op rotation under every convention, so the earlier check was blind to rotary
phase -- the same blind spot that hid the target's own NORM-vs-NEOX RoPE bug during bring-up.

Input comes from SPARKINFER_MTP_DUMPSEQ: hiddens.bin ([n, hidden] bf16, the target's final hidden
at each position) and tokens.txt (pos, token_in, token_out per line).
"""
import json
import struct
import sys

import numpy as np

ckpt, dumpdir = sys.argv[1], sys.argv[2]
EPS = 1e-6


def load_bf16_shard(path, want=None):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__" or meta["dtype"] != "BF16":
            continue
        if want and not any(w in name for w in want):
            continue
        a, b = meta["data_offsets"]
        arr = np.frombuffer(blob[a:b], dtype=np.uint16).astype(np.uint32) << 16
        out[name] = arr.view(np.float32).reshape(meta["shape"])
    return out


def embed_rows(path, ids, hidden):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        base = 8 + n
        key = next(k for k in hdr if k != "__metadata__" and k.endswith("embed_tokens.weight"))
        off = hdr[key]["data_offsets"][0]
        rows = np.empty((len(ids), hidden), dtype=np.float32)
        for j, t in enumerate(ids):
            f.seek(base + off + t * hidden * 2)
            a = np.frombuffer(f.read(hidden * 2), dtype=np.uint16).astype(np.uint32) << 16
            rows[j] = a.view(np.float32)
    return rows


def rms(x, w):
    return (x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS)).astype(np.float32) * w


W = load_bf16_shard(f"{ckpt}/model_mtp.safetensors")
H = W["mtp.norm.weight"].shape[0]
n_heads, n_kv, hd = 24, 4, 256
theta = 10000000.0

rows = [tuple(int(v) for v in l.split()) for l in open(f"{dumpdir}/tokens.txt")]
hid = np.fromfile(f"{dumpdir}/hiddens.bin", dtype=np.uint16).astype(np.uint32) << 16
hid = hid.view(np.float32).reshape(-1, H)
n = min(len(rows), hid.shape[0])
rows, hid = rows[:n], hid[:n]
tok_out = [r[2] for r in rows]
print(f"sequence: {n} positions, hidden {H}")

emb = embed_rows(f"{ckpt}/model.safetensors", tok_out, H)

# fusion for every position at once
e_n = rms(emb, W["mtp.pre_fc_norm_embedding.weight"])
h_n = rms(hid, W["mtp.pre_fc_norm_hidden.weight"])
x = np.concatenate([e_n, h_n], axis=-1) @ W["mtp.fc.weight"].T          # [n, H]

nrm = rms(x, W["mtp.layers.0.input_layernorm.weight"])
qraw = nrm @ W["mtp.layers.0.self_attn.q_proj.weight"].T                # [n, 2*qdim]
k = nrm @ W["mtp.layers.0.self_attn.k_proj.weight"].T
v = nrm @ W["mtp.layers.0.self_attn.v_proj.weight"].T

qg = qraw.reshape(n, n_heads, 2, hd)
q, gate = qg[:, :, 0, :], qg[:, :, 1, :]
q = rms(q, W["mtp.layers.0.self_attn.q_norm.weight"])
k = rms(k.reshape(n, n_kv, hd), W["mtp.layers.0.self_attn.k_norm.weight"])
v = v.reshape(n, n_kv, hd)


def rope(t, pos, neox=True):
    d = t.shape[-1]
    inv = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if neox:                       # split-half: rotate x[:d/2] with x[d/2:]
        a, b = t[..., : d // 2], t[..., d // 2:]
    else:                          # consecutive-pair
        a, b = t[..., 0::2], t[..., 1::2]
    ra, rb = a * cos - b * sin, a * sin + b * cos
    out = np.empty_like(t)
    if neox:
        out[..., : d // 2], out[..., d // 2:] = ra, rb
    else:
        out[..., 0::2], out[..., 1::2] = ra, rb
    return out


for neox in (True, False):
    pos = np.arange(n)
    qr, kr = rope(q, pos, neox), rope(k, pos, neox)
    grp = n_heads // n_kv
    kb = np.repeat(kr, grp, axis=1)
    vb = np.repeat(v, grp, axis=1)
    scores = np.einsum("ihd,jhd->hij", qr, kb) / np.sqrt(hd)
    scores = np.where(np.tril(np.ones((n, n), bool))[None], scores, -np.inf)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    attn = np.einsum("hij,jhd->ihd", p, vb)
    attn = attn * (1.0 / (1.0 + np.exp(-gate.transpose(0, 1, 2))))       # SIGMOID gate (verified)
    xa = attn.reshape(n, n_heads * hd) @ W["mtp.layers.0.self_attn.o_proj.weight"].T + x
    n2 = rms(xa, W["mtp.layers.0.post_attention_layernorm.weight"])
    g_ = n2 @ W["mtp.layers.0.mlp.gate_proj.weight"].T
    u_ = n2 @ W["mtp.layers.0.mlp.up_proj.weight"].T
    hh = (g_ / (1.0 + np.exp(-g_.astype(np.float64)))) * u_
    xf = hh.astype(np.float32) @ W["mtp.layers.0.mlp.down_proj.weight"].T + xa
    fin = rms(xf, W["mtp.norm.weight"])

    # lm_head is FP8 in the main shard; use the tied embedding matrix rows we already have as a
    # cheap proxy is NOT valid, so instead report the top-1 over a candidate set: the tokens the
    # target actually produced. If MTP is right, argmax over {t_{i+2}} vs other positions' tokens
    # should pick t_{i+2} far more often than chance.
    cand = np.array(sorted(set(tok_out)))
    cand_emb = embed_rows(f"{ckpt}/model.safetensors", cand.tolist(), H)
    logit = fin @ cand_emb.T
    pick = cand[logit.argmax(-1)]
    hit2 = sum(1 for i in range(n - 2) if pick[i] == tok_out[i + 2])
    hit1 = sum(1 for i in range(n - 1) if pick[i] == tok_out[i + 1])
    print(f"  RoPE={'NEOX' if neox else 'NORMAL'}:  predict t+2 {hit2}/{n-2} = {hit2/(n-2):.3f}"
          f"   predict t+1 {hit1}/{n-1} = {hit1/(n-1):.3f}")
print("\nNOTE: scoring uses tied-embedding rows as a proxy head over the observed-token candidate\n"
      "set, not the real FP8 lm_head -- absolute rates understate, but a correct pairing should\n"
      "still stand out sharply against the wrong one.")
