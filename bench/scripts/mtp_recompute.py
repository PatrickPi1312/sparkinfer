#!/usr/bin/env python3
"""Recompute Qwen3.8-27B's MTP head in NumPy from the checkpoint bytes and diff it against
sparkinfer's dumped activations, stage by stage.

  mtp_recompute.py <checkpoint_dir> <dump_dir>

The point is the SEED. Every stage is recomputed from sparkinfer's OWN dumped input
(00_in_hidden.bin + the token id in meta.txt), not from an independent forward pass, so a
divergence at stage N means the bug is in stage N -- not inherited from an earlier difference in
inputs. That is the distinction black-box agreement sweeps cannot make, and it is what localised
the target's [out,in] transpose and its silu-vs-sigmoid gate to a single layer.

Reads model_mtp.safetensors directly (all 15 tensors are plain BF16 -- quantization_config.ignore
ends with "re:^mtp.*"), so nothing here depends on sparkinfer's loader being correct.
"""
import json
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        raw = blob[a:b]
        if meta["dtype"] != "BF16":
            continue
        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        out[name] = arr.view(np.float32).reshape(meta["shape"])
    return out


def bf16_file(path, shape=None):
    a = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    a = a.view(np.float32)
    return a.reshape(shape) if shape else a


def rms_norm(x, w, eps):
    return (x / np.sqrt((x.astype(np.float64) ** 2).mean() + eps)).astype(np.float32) * w


def cmp(tag, mine, theirs):
    mine = mine.astype(np.float64).ravel()
    theirs = theirs.astype(np.float64).ravel()
    n = min(mine.size, theirs.size)
    mine, theirs = mine[:n], theirs[:n]
    denom = np.linalg.norm(mine) * np.linalg.norm(theirs)
    cos = float(mine @ theirs / denom) if denom else 0.0
    rel = float(np.linalg.norm(mine - theirs) / (np.linalg.norm(theirs) + 1e-9))
    flag = "OK  " if cos > 0.99 else ("SIGN" if cos < -0.5 else "BAD ")
    print(f"  [{flag}] {tag:18s} cos={cos:+.5f} rel_err={rel:.4f} "
          f"|mine|={np.linalg.norm(mine):9.3f} |spark|={np.linalg.norm(theirs):9.3f}")
    return cos > 0.99


ckpt, dump = sys.argv[1], sys.argv[2]
meta = {}
for line in open(f"{dump}/meta.txt"):
    k, v = line.split()
    meta[k] = int(v)
H, qdim, kvdim, ffn = meta["hidden"], meta["qdim"], meta["kvdim"], meta["ffn"]
eps = 1e-6

W = load_safetensors(f"{ckpt}/model_mtp.safetensors")


def embed_row(path, token_id, hidden):
    """One row of embed_tokens, by offset. The main shard is 22.5 GB -- reading it whole to get
    5120 values would need more RAM than the box has."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        base = 8 + n
        key = next(k for k in hdr if k != "__metadata__" and k.endswith("embed_tokens.weight"))
        meta_e = hdr[key]
        assert meta_e["dtype"] == "BF16", meta_e["dtype"]
        start = base + meta_e["data_offsets"][0] + token_id * hidden * 2
        f.seek(start)
        raw = f.read(hidden * 2)
    a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return a.view(np.float32)

h_in = bf16_file(f"{dump}/00_in_hidden.bin", (H,))
print(f"seed: token={meta['next_token_id']} pos={meta['pos']} "
      f"swap_cat={meta['swap_cat']} |h_in|={np.linalg.norm(h_in):.3f}")

e = embed_row(f"{ckpt}/model.safetensors", meta["next_token_id"], H)

e_n = rms_norm(e,    W["mtp.pre_fc_norm_embedding.weight"], eps)
h_n = rms_norm(h_in, W["mtp.pre_fc_norm_hidden.weight"],    eps)
cat = np.concatenate([h_n, e_n] if meta["swap_cat"] else [e_n, h_n])
cmp("cat", cat, bf16_file(f"{dump}/01_cat.bin", (2 * H,)))

# HF Linear is [out,in]; y = W @ x.
x = W["mtp.fc.weight"] @ cat
cmp("x_fc", x, bf16_file(f"{dump}/02_x_fc.bin", (H,)))

normed = rms_norm(x, W["mtp.layers.0.input_layernorm.weight"], eps)
cmp("normed_in", normed, bf16_file(f"{dump}/03_normed_in.bin", (H,)))

qraw = W["mtp.layers.0.self_attn.q_proj.weight"] @ normed
cmp("qraw", qraw, bf16_file(f"{dump}/04_qraw.bin", (2 * qdim,)))

# Attention itself is position/KV dependent, so the recomputation stops being a like-for-like
# comparison after qraw for a pos>0 step. What the remaining stages still pin down is the GATE:
# split [q|gate] per head and check which half sparkinfer treated as the gate, and whether the
# activation applied is sigmoid or silu.
hd = qdim // (qdim // 256) if qdim % 256 == 0 else 256
n_heads = qdim // 256
qg = qraw.reshape(n_heads, 2, 256)
gate_first_half = qg[:, 0, :].ravel()
gate_second_half = qg[:, 1, :].ravel()

pre = bf16_file(f"{dump}/05_attn_pregate.bin", (qdim,))
post = bf16_file(f"{dump}/06_attn_postgate.bin", (qdim,))
print("\ngate check (which half is the gate, and sigmoid vs silu):")
for name, g in (("per-head[0]=gate", gate_first_half), ("per-head[1]=gate", gate_second_half)):
    sig = pre * (1.0 / (1.0 + np.exp(-g.astype(np.float64))))
    silu = pre * (g.astype(np.float64) / (1.0 + np.exp(-g.astype(np.float64))))
    cmp(f"{name} sigmoid", sig, post)
    cmp(f"{name} silu", silu, post)
print(f"\ngate stats: half0 mean={gate_first_half.mean():+.3f}  half1 mean={gate_second_half.mean():+.3f}")

# ---- downstream of the gate -------------------------------------------------------------------
# The dump above is pos=0, where attention is exactly checkable: one KV entry means softmax==1, so
# the attention output must equal V itself (broadcast across each GQA group). That turns the whole
# remaining chain -- attention, wo, FFN, final norm -- into a closed-form comparison.
if meta["pos"] == 0:
    n_kv = kvdim // 256
    v = W["mtp.layers.0.self_attn.v_proj.weight"] @ normed          # [kvdim]
    grp = n_heads // n_kv
    v_bcast = np.repeat(v.reshape(n_kv, 256), grp, axis=0).ravel()  # [qdim]
    print("\ndownstream (pos=0, softmax==1 so attn output == V):")
    cmp("attn_pregate", v_bcast, pre)

    gate = qg[:, 1, :].ravel()
    gated = pre * (1.0 / (1.0 + np.exp(-gate.astype(np.float64))))
    x_attn = W["mtp.layers.0.self_attn.o_proj.weight"] @ gated + x

    n2 = rms_norm(x_attn.astype(np.float32),
                  W["mtp.layers.0.post_attention_layernorm.weight"], eps)
    g_ = W["mtp.layers.0.mlp.gate_proj.weight"] @ n2
    u_ = W["mtp.layers.0.mlp.up_proj.weight"] @ n2
    h_ = (g_ / (1.0 + np.exp(-g_.astype(np.float64)))) * u_          # SwiGLU
    cmp("ffn_h", h_, bf16_file(f"{dump}/07_ffn_h.bin", (ffn,)))

    x_ffn = W["mtp.layers.0.mlp.down_proj.weight"] @ h_ + x_attn
    final = rms_norm(x_ffn.astype(np.float32), W["mtp.norm.weight"], eps)
    cmp("prelm_normed", final, bf16_file(f"{dump}/08_prelm_normed.bin", (H,)))

    lg = np.fromfile(f"{dump}/09_logits.bin", dtype=np.float32)
    print(f"\nsparkinfer logits: argmax={int(lg.argmax())} max={lg.max():.3f} "
          f"finite={np.isfinite(lg).all()}")
