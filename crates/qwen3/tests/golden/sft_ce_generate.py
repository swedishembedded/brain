#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements fine-tuning whose loss and gradients are checked against an independent reference
# for its clients. If your team needs expertise in verifying training numerics then you can procure our services by
# sending an email to info@swedishembedded.com.
"""Writes `sft_ce.json`: the masked next-token cross-entropy of two chat-shaped records through a tiny Qwen3-shaped
decoder, its per-position log-probabilities and the gradient of every weight, computed in float64 with torch autograd.

The model is written here from the published architecture (RMSNorm, per-head q/k RMSNorm, rotate-half RoPE, grouped
causal attention, SwiGLU MLP, tied embedding head); nothing of brain is imported or run. The weights are drawn here,
rounded to binary32 (what the test loads), and written into the fixture, so the test is independent of brain's own
initialiser.

The loss is the spec, not brain's layout: for each record, every position `j >= 1` whose token is supervised
contributes `-log p(ids[j] | ids[:j])`, and the sum over both records is divided by the number of supervised tokens.
A record's first token is never a target, prompt and tool tokens are not supervised, and padding does not exist here:
each record is evaluated on its own tokens only.

Run only to (re)generate the fixture (`python3 -I sft_ce_generate.py > sft_ce.json`); the test never runs Python.
"""

import json
import math
import sys

import torch

CONFIG = {
    "vocab": 13,
    "d_model": 16,
    "n_layers": 1,
    "n_heads": 3,
    "n_kv_heads": 1,
    "head_dim": 8,
    "d_ff": 40,
    "rope_theta": 10000.0,
    "rms_eps": 1e-6,
    "block": 8,
}

# Two records shaped like chat SFT records: a prompt (unsupervised), an answer, a tool result in the middle of the
# record (unsupervised) and a second answer; the second record is shorter than the block, so its row is padded.
RECORDS = [
    {"ids": [1, 7, 3, 9, 4, 11, 2, 5], "supervised": [False, False, False, True, True, False, True, True]},
    {"ids": [6, 2, 8, 12, 0], "supervised": [False, False, True, True, True]},
]


def param_list(c):
    d, ff, hd = c["d_model"], c["d_ff"], c["head_dim"]
    q, kv = c["n_heads"] * hd, c["n_kv_heads"] * hd
    out = [("tok.weight", (c["vocab"], d))]
    for l in range(c["n_layers"]):
        p = f"blocks.{l}."
        out += [
            (p + "ln1.weight", (d,)),
            (p + "attn.wq.weight", (q, d)),
            (p + "attn.wk.weight", (kv, d)),
            (p + "attn.wv.weight", (kv, d)),
            (p + "attn.q_norm.weight", (hd,)),
            (p + "attn.k_norm.weight", (hd,)),
            (p + "attn.wo.weight", (d, q)),
            (p + "ln2.weight", (d,)),
            (p + "mlp.gate.weight", (ff, d)),
            (p + "mlp.up.weight", (ff, d)),
            (p + "mlp.down.weight", (d, ff)),
        ]
    out.append(("norm.weight", (d,)))
    return out


def draw_weights(c, seed=20261121):
    g = torch.Generator().manual_seed(seed)
    w = {}
    for name, shape in param_list(c):
        z = torch.randn(shape, generator=g, dtype=torch.float64)
        if name.endswith("norm.weight") or name.endswith("ln1.weight") or name.endswith("ln2.weight"):
            v = 1.0 + 0.2 * z  # gains away from 1 so their gradients are exercised
        elif name == "tok.weight":
            v = 0.3 * z
        else:
            v = z / math.sqrt(shape[1])
        w[name] = v.to(torch.float32).to(torch.float64)  # exactly the binary32 values the test loads
    return w


def rms(x, gain, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * gain


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def forward_logp(w, c, ids):
    """log p(. | ids[:t+1]) for every position t of one record, shape [T, vocab]."""
    T, hd, H, KV = len(ids), c["head_dim"], c["n_heads"], c["n_kv_heads"]
    eps = c["rms_eps"]
    x = w["tok.weight"][torch.tensor(ids)]
    inv = c["rope_theta"] ** (-torch.arange(0, hd, 2, dtype=torch.float64) / hd)
    ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None, :]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    for l in range(c["n_layers"]):
        p = f"blocks.{l}."
        h = rms(x, w[p + "ln1.weight"], eps)
        q = (h @ w[p + "attn.wq.weight"].T).view(T, H, hd)
        k = (h @ w[p + "attn.wk.weight"].T).view(T, KV, hd)
        v = (h @ w[p + "attn.wv.weight"].T).view(T, KV, hd)
        q = rms(q, w[p + "attn.q_norm.weight"], eps)
        k = rms(k, w[p + "attn.k_norm.weight"], eps)
        q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
        heads = []
        for hh in range(H):
            kvh = hh // (H // KV)
            s = (q[:, hh, :] @ k[:, kvh, :].T) / math.sqrt(hd)
            s = s.masked_fill(~causal, float("-inf"))
            heads.append(torch.softmax(s, -1) @ v[:, kvh, :])
        x = x + torch.cat(heads, -1) @ w[p + "attn.wo.weight"].T
        h2 = rms(x, w[p + "ln2.weight"], eps)
        gate = h2 @ w[p + "mlp.gate.weight"].T
        up = h2 @ w[p + "mlp.up.weight"].T
        x = x + (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down.weight"].T
    xn = rms(x, w["norm.weight"], eps)
    return torch.log_softmax(xn @ w["tok.weight"].T, -1)


def sig(v):
    """Nine significant digits: a binary32 weight round-trips exactly, and an expected value keeps far more precision
    than the binary32 result it is compared with."""
    return float(f"{float(v.detach() if torch.is_tensor(v) else v):.9g}")


def main():
    c = CONFIG
    w = draw_weights(c)
    for t in w.values():
        t.requires_grad_(True)
    total, count, logprobs = 0.0, 0, []
    for r in RECORDS:
        ids, sup = r["ids"], r["supervised"]
        assert len(ids) <= c["block"] and not sup[0], "a record fits the block and its first token is no target"
        lp = forward_logp(w, c, ids)
        # Position t predicts token t + 1: the log-probability of every token after the first, given its prefix.
        per = [lp[j - 1, ids[j]] for j in range(1, len(ids))]
        logprobs.append([sig(v) for v in per])
        for j in range(1, len(ids)):
            if sup[j]:
                total = total - per[j - 1]
                count += 1
    loss = total / count
    loss.backward()
    fixture = {
        "provenance": {
            "generator": "crates/qwen3/tests/golden/sft_ce_generate.py",
            "torch": torch.__version__,
            "precision": "float64 forward and autograd over binary32-rounded weights",
        },
        "config": c,
        "records": RECORDS,
        "weights": {n: [sig(v) for v in t.detach().flatten()] for n, t in w.items()},
        "expected": {
            "supervised_tokens": count,
            "loss": sig(loss),
            "logprobs": logprobs,
            "grads": {n: [sig(v) for v in t.grad.flatten()] for n, t in w.items()},
        },
    }
    json.dump(fixture, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
