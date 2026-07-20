#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Float64 torch-autograd golden for the ViT-block backward gradcheck.

Replicates crates/model/tests/vit_block_gradcheck.rs exactly (same LCG, same
tiny shapes: dim 32, 2 heads, mlp 64, 2 spans x 8 rows) for both configs:
trunk-like (QK-norm + 2D RoPE + LayerScale) and DINOv2-like (LayerScale
only). Finite differences are NOT a usable oracle here — attention softmax
conditioning makes them noise-dominated at f32 — autograd in f64 is exact.

Run from the repo root:  python3 tools/vit_dump_gradcheck.py
"""
import json

import torch

torch.set_default_dtype(torch.float64)

C, HEADS, M, SPAN, ROWS = 32, 2, 64, 8, 16
HD = C // HEADS
EPS = 1e-5


class Lcg:
    def __init__(s, seed):
        s.v = seed

    def next(s):
        s.v = (s.v * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        # replicate the Rust f32 op order exactly:
        # (((v>>33) as f32 / (1<<31) as f32) - 1.0) * 0.5
        import numpy as np
        raw = (np.float32(s.v >> 33) / np.float32(1 << 31) - np.float32(1.0)) * np.float32(0.5)
        return float(raw)

    def vec(s, n):
        return [s.next() for _ in range(n)]


def param_shapes(qk_norm, ls):
    v = [
        ("norm1_w", C), ("norm1_b", C), ("qkv_w", 3 * C * C), ("qkv_b", 3 * C),
        ("proj_w", C * C), ("proj_b", C), ("norm2_w", C), ("norm2_b", C),
        ("fc1_w", M * C), ("fc1_b", M), ("fc2_w", C * M), ("fc2_b", C),
    ]
    if qk_norm:
        v += [("q_norm_w", HD), ("q_norm_b", HD), ("k_norm_w", HD), ("k_norm_b", HD)]
    if ls:
        v += [("ls1", C), ("ls2", C)]
    return v


def setup(qk_norm, ls, seed):
    r = Lcg(seed)
    weights = {}
    for name, n in param_shapes(qk_norm, ls):
        v = r.vec(n)
        if (name.endswith("_w") and "norm" in name) or name.startswith("ls"):
            v = [1.0 + 0.3 * x for x in v]
        elif name.endswith("_w"):
            v = [0.35 * x for x in v]
        weights[name] = v
    x = r.vec(ROWS * C)
    wloss = r.vec(ROWS * C)
    cos = [torch.tensor(v * 2.0).cos().item() for v in r.vec(SPAN * HD // 2)]
    sin = [torch.tensor(v * 2.0).sin().item() for v in r.vec(SPAN * HD // 2)]
    return weights, x, wloss, cos, sin


def ln(x, w, b):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / (var + EPS).sqrt() * w + b


def forward(weights, x, cos, sin, qk_norm, ls):
    W = {k: torch.tensor(v, requires_grad=True) for k, v in weights.items()}
    xt_leaf = torch.tensor(x, requires_grad=True)
    xt = xt_leaf.reshape(ROWS, C)
    h = ln(xt, W["norm1_w"], W["norm1_b"])
    qkv = h @ W["qkv_w"].reshape(3 * C, C).T + W["qkv_b"]  # [R, 3C]
    q, k, v = qkv[:, :C], qkv[:, C:2 * C], qkv[:, 2 * C:]

    def heads(t):
        return t.reshape(ROWS, HEADS, HD)

    q, k, v = heads(q), heads(k), heads(v)
    if qk_norm:
        q = ln(q, W["q_norm_w"], W["q_norm_b"])
        k = ln(k, W["k_norm_w"], W["k_norm_b"])
        # rope: pairs (d, d+half) share angle index d; table row = row % SPAN
        half = HD // 2
        ct = torch.tensor(cos).reshape(SPAN, half)
        st = torch.tensor(sin).reshape(SPAN, half)
        rows = torch.arange(ROWS) % SPAN

        def rope(t):
            c = ct[rows].unsqueeze(1)  # [R,1,half]
            s = st[rows].unsqueeze(1)
            x1, x2 = t[..., :half], t[..., half:]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

        q, k = rope(q), rope(k)
    ctx = torch.zeros(ROWS, HEADS, HD)
    for s0 in range(0, ROWS, SPAN):
        qs, ks, vs = q[s0:s0 + SPAN], k[s0:s0 + SPAN], v[s0:s0 + SPAN]
        scores = torch.einsum("ihd,jhd->hij", qs, ks) / (HD ** 0.5)
        probs = scores.softmax(-1)
        ctx[s0:s0 + SPAN] = torch.einsum("hij,jhd->ihd", probs, vs)
    attn = ctx.reshape(ROWS, C) @ W["proj_w"].reshape(C, C).T + W["proj_b"]
    if ls:
        attn = attn * W["ls1"]
    mid = xt + attn
    h2 = ln(mid, W["norm2_w"], W["norm2_b"])
    hh = h2 @ W["fc1_w"].reshape(M, C).T + W["fc1_b"]
    hh = 0.5 * hh * (1.0 + torch.erf(hh / (2.0 ** 0.5)))
    mlp = hh @ W["fc2_w"].reshape(C, M).T + W["fc2_b"]
    if ls:
        mlp = mlp * W["ls2"]
    y = mid + mlp
    loss = (y.reshape(-1) * torch.tensor(wloss_g)).sum()
    loss.backward()
    out = {k: t.grad.reshape(-1).tolist() for k, t in W.items()}
    out["dx"] = xt_leaf.grad.reshape(-1).tolist()
    return out


golden = {}
for cfg, (qk_norm, ls, seed) in {
    "trunk": (True, True, 0x7A11),
    "dino": (False, True, 0xD1A0),
}.items():
    weights, x, wloss, cos, sin = setup(qk_norm, ls, seed)
    wloss_g = wloss
    golden[cfg] = forward(weights, x, cos, sin, qk_norm, ls)
json.dump(golden, open("crates/model/tests/golden/vit_gradcheck.json", "w"))
print("wrote crates/model/tests/golden/vit_gradcheck.json")
