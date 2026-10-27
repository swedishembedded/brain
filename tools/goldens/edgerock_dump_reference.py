#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for long-context inference at a
# fixed memory budget for its clients. If your team needs expertise in
# bounded-memory model architectures then you can procure our services by
# sending an email to info@swedishembedded.com.

"""Dump EdgeRock reference goldens (tiny random weights).

EdgeRock is the Qwen3.5 hybrid decoder (`transformers.models.qwen3_5`) with
two changes on every full-attention layer, both written out here
independently from the published architecture description:

1. **Sliding-window attention.** Query `i` attends keys `j` with
   `i - window < j <= i`; everything else about the gated GQA layer
   (QK-norm, partial M-RoPE, sigmoid output gate) is unchanged.

2. **Anchor memory**, a learned fixed-capacity store added beside the
   attention output. It reads the layer's RESIDUAL input `h` (before
   `input_layernorm`) through its own RMSNorm `u = input_norm(h)` and
   computes, per token:

   * `query = q_norm(q_proj(u))` per head, rotated by the partial RoPE;
   * a per-head read gate `tanh(read_scale) * 2 * sigmoid(<query_pre_rope,
     read_gate_weight> / sqrt(D) + read_gate_bias)`;
   * an admission score per KV group `admission_up(silu(admission_down(u)))`
     in fp32;
   * a written record `z = u + write_down(silu(write_gate(u)) * write_up(u))`,
     `key = rope(k_norm(k_proj(z)))`, `value = v_proj(z)`.

   Tokens are grouped into commit blocks of `C = min(max_commit_chunk,
   window)` positions. The bank a token in block `b` reads holds, per KV
   group, the `capacity` highest-scoring tokens of all blocks `< b` (ties to
   the earlier position); tokens of the token's own block are not visible
   yet. A head reads its group's bank by softmax attention (scale
   `D^-0.5`) with one extra always-present zero logit carrying a zero value
   (an attention sink, so an empty bank reads zero). The read is multiplied
   by the gate and projected by `o_proj`; the result is added to the
   attention output before the residual add.

   The bank therefore depends only on scores and positions, never on how a
   prompt is chunked. The dumper computes it twice - from that closed-form
   definition and by an incremental pending-buffer/commit replay of the
   decode path - and asserts the two reads are bit-identical.

Two variants, each written to its own directory under `--out`:

  tiny_text   random weights: scores are distinct, the bank evicts
  tiny_ties   `admission_up` zeroed: every score is 0.0, so the tie rule
              alone decides which records stay (the earliest ones)

Each directory holds:
  edgerock_tiny.safetensors          stage taps (below)
  edgerock_tiny_weights.safetensors  the model's own weights in brain's
                                     `blocks.{l}.*` naming (raw HF RMSNorm
                                     weights, not `1+w` folded)
  manifest.json                      shapes, sha256, config, versions

Taps: `tokens`, `embed`, `cos`, `sin`, `layer{i}.out` for every layer, and on
memory layers `layer{i}.swa.out` (attention output after `o_proj`),
`layer{i}.mem.{scores,gate,query,key,value,read,out}` and
`layer{i}.mem.refs` (`[blocks, groups, capacity]` token index of every bank
slot, -1 when empty), plus `final_hidden` and `logits` (tied head).

Usage: python3 tools/goldens/edgerock_dump_reference.py --seed 7
"""

import argparse
import hashlib
import json
import math
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5RMSNorm,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
    apply_rotary_pos_emb,
    rotate_half,
)

# Distinct tiny dims (an axis swap cannot pass at cosine 1.0), real flags.
# Two KV groups so per-group admission is exercised; window and commit block
# small enough that a 48-token prompt slides the window and evicts from the
# bank many times over.
TINY = dict(
    vocab_size=29,
    hidden_size=96,
    intermediate_size=112,
    num_hidden_layers=8,
    num_attention_heads=10,
    num_key_value_heads=5,
    head_dim=40,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=20,
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    full_attention_interval=4,
    rms_norm_eps=1e-6,
    attention_bias=False,
    attention_dropout=0.0,
    tie_word_embeddings=True,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [2, 2, 1],
    },
)
WINDOW = 11
MEMORY = dict(capacity=6, max_commit_chunk=4, admission_rank=7, writer_intermediate_size=52)
T = 48


class AnchorMemory(nn.Module):
    """The memory module's parameters under the checkpoint's leaf names."""

    def __init__(self, cfg):
        super().__init__()
        d, h, g, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        r, i = MEMORY["admission_rank"], MEMORY["writer_intermediate_size"]
        self.heads, self.groups, self.dim = h, g, hd
        self.input_norm = Qwen3_5RMSNorm(d, eps=cfg.rms_norm_eps)
        self.q_norm = Qwen3_5RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(hd, eps=cfg.rms_norm_eps)
        for name, n_in, n_out in (
            ("q_proj", d, h * hd), ("k_proj", d, g * hd), ("v_proj", d, g * hd), ("o_proj", h * hd, d),
            ("write_gate_proj", d, i), ("write_up_proj", d, i), ("write_down_proj", i, d),
            ("admission_down", d, r), ("admission_up", r, g),
        ):
            setattr(self, name, nn.Linear(n_in, n_out, bias=False))
        self.read_scale = nn.Parameter(torch.zeros(h))
        self.read_gate_weight = nn.Parameter(torch.zeros(h, hd))
        self.read_gate_bias = nn.Parameter(torch.zeros(h))


def partial_rope(x, cos, sin):
    """Rotate the first `cos.shape[-1]` channels of `x` [T, heads, D]."""
    w = cos.shape[-1]
    c, s = cos[:, None, :], sin[:, None, :]
    rot = x[..., :w]
    return torch.cat((rot * c + rotate_half(rot) * s, x[..., w:]), -1)


def bank_by_definition(scores, chunk, capacity):
    """refs[b, g, :] = the `capacity` best tokens of blocks < b, best first.

    Order key: score descending, then position ascending. Empty slots -1."""
    t, groups = scores.shape
    blocks = math.ceil(t / chunk)
    refs = torch.full((blocks, groups, capacity), -1, dtype=torch.long)
    for b in range(blocks):
        for g in range(groups):
            cands = sorted(range(b * chunk), key=lambda j: (-scores[j, g].item(), j))[:capacity]
            refs[b, g, : len(cands)] = torch.tensor(cands, dtype=torch.long)
    return refs


def bank_by_replay(scores, chunk, capacity):
    """The decode path's bookkeeping: a committed bank plus a pending block,
    merged when the pending block fills. Returns the bank each token read
    (before its own record was pushed), as refs[t, g, :]."""
    t, groups = scores.shape
    bank = [[] for _ in range(groups)]  # list of token indices, best first
    pending = []
    seen = torch.full((t, groups, capacity), -1, dtype=torch.long)
    for i in range(t):
        for g in range(groups):
            seen[i, g, : len(bank[g])] = torch.tensor(bank[g], dtype=torch.long)
        pending.append(i)
        if len(pending) == chunk:
            for g in range(groups):
                merged = bank[g] + pending
                bank[g] = sorted(merged, key=lambda j: (-scores[j, g].item(), j))[:capacity]
            pending = []
    return seen


def memory_forward(mem, h, cos, sin, chunk, taps, prefix):
    t = h.shape[0]
    hh, g, d = mem.heads, mem.groups, mem.dim
    u = mem.input_norm(h)
    query = mem.q_norm(mem.q_proj(u).view(t, hh, d))
    gate = torch.tanh(mem.read_scale) * 2 * torch.sigmoid((query * mem.read_gate_weight).sum(-1) / math.sqrt(d) + mem.read_gate_bias)
    query = partial_rope(query, cos, sin)
    scores = F.linear(F.silu(mem.admission_down(u)).float(), mem.admission_up.weight.float())
    z = u + mem.write_down_proj(F.silu(mem.write_gate_proj(u)) * mem.write_up_proj(u))
    key = partial_rope(mem.k_norm(mem.k_proj(z).view(t, g, d)), cos, sin)
    value = mem.v_proj(z).view(t, g, d)

    refs = bank_by_definition(scores, chunk, MEMORY["capacity"])
    replay = bank_by_replay(scores, chunk, MEMORY["capacity"])
    by_token = refs[torch.arange(t) // chunk]
    assert torch.equal(by_token, replay), f"{prefix}: closed-form bank and decode replay disagree"

    per_group = hh // g
    read = torch.zeros(t, hh, d)
    for i in range(t):
        for head in range(hh):
            grp = head // per_group
            r = by_token[i, grp]
            r = r[r >= 0]
            logits = torch.cat((key[r, grp] @ query[i, head] / math.sqrt(d), torch.zeros(1)))
            p = torch.softmax(logits, 0)[:-1]
            read[i, head] = p @ value[r, grp] if len(r) else torch.zeros(d)
    out = mem.o_proj((read * gate[..., None]).reshape(t, hh * d))
    for name, v in (("scores", scores), ("gate", gate), ("query", query), ("key", key), ("value", value), ("read", read), ("out", out), ("refs", refs)):
        taps[f"{prefix}.{name}"] = v
    return out


def swa_forward(attn, x, cos, sin, window):
    t = x.shape[0]
    q_and_gate = attn.q_proj(x).view(t, -1, attn.head_dim * 2)
    query, gate = torch.chunk(q_and_gate, 2, dim=-1)
    gate = gate.reshape(t, -1)
    query = attn.q_norm(query).transpose(0, 1)[None]
    key = attn.k_norm(attn.k_proj(x).view(t, -1, attn.head_dim)).transpose(0, 1)[None]
    value = attn.v_proj(x).view(t, -1, attn.head_dim).transpose(0, 1)[None]
    query, key = apply_rotary_pos_emb(query, key, cos[None], sin[None])
    i = torch.arange(t)[:, None]
    j = torch.arange(t)[None, :]
    allowed = (j <= i) & (j > i - window)
    ctx = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, scale=attn.scaling, enable_gqa=True)
    ctx = ctx[0].transpose(0, 1).reshape(t, -1)
    return attn.o_proj(ctx * torch.sigmoid(gate))


def build(seed, ties):
    cfg = Qwen3_5TextConfig(**TINY)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    model = Qwen3_5TextModel(cfg).eval()
    memories = {i: AnchorMemory(cfg) for i, ty in enumerate(cfg.layer_types) if ty == "full_attention"}
    with torch.no_grad():
        # Every parameter random: the checkpoint's zero inits (RMSNorm
        # weights, read gate, write_down, admission_up) would make whole
        # paths vanish and pass vacuously.
        for module in [model, *memories.values()]:
            for name, p in module.named_parameters():
                if name.endswith("A_log") or name.endswith("dt_bias"):
                    continue
                scale = 0.1 if p.ndim > 1 else 0.3
                p.copy_(torch.randn(p.shape, generator=gen) * scale)
        if ties:
            for m in memories.values():
                m.admission_up.weight.zero_()
    tokens = torch.randint(0, cfg.vocab_size, (T,), generator=gen)
    return cfg, model, memories, tokens


def forward(cfg, model, memories, tokens):
    taps = {"tokens": tokens.to(torch.int32)}
    chunk = min(MEMORY["max_commit_chunk"], WINDOW)
    h = model.embed_tokens(tokens)
    taps["embed"] = h
    position_ids = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = model.rotary_emb(h[None], position_ids)
    cos, sin = cos[0], sin[0]
    taps["cos"], taps["sin"] = cos, sin
    for i, layer in enumerate(model.layers):
        if cfg.layer_types[i] == "linear_attention":
            h = layer(h[None], position_embeddings=(cos[None], sin[None]), attention_mask=None)[0]
            if isinstance(h, tuple):
                h = h[0]
        else:
            residual = h
            mixed = swa_forward(layer.self_attn, layer.input_layernorm(h), cos, sin, WINDOW)
            taps[f"layer{i}.swa.out"] = mixed
            mixed = mixed + memory_forward(memories[i], residual, cos, sin, chunk, taps, f"layer{i}.mem")
            h = residual + mixed
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        taps[f"layer{i}.out"] = h
    h = model.norm(h)
    taps["final_hidden"] = h
    taps["logits"] = h @ model.embed_tokens.weight.T
    return taps


def collect_weights(cfg, model, memories):
    sd = model.state_dict()
    w = {"tok.weight": sd["embed_tokens.weight"], "norm.weight": sd["norm.weight"]}
    for i, ty in enumerate(cfg.layer_types):
        p, s = f"blocks.{i}", f"layers.{i}"
        w[f"{p}.ln1.weight"] = sd[f"{s}.input_layernorm.weight"]
        w[f"{p}.ln2.weight"] = sd[f"{s}.post_attention_layernorm.weight"]
        if ty == "linear_attention":
            for leaf in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight"):
                w[f"{p}.linear_attn.{leaf}"] = sd[f"{s}.linear_attn.{leaf}"]
            w[f"{p}.linear_attn.conv1d.weight"] = sd[f"{s}.linear_attn.conv1d.weight"].squeeze(1)
        else:
            for leaf in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
                w[f"{p}.self_attn.{leaf}.weight"] = sd[f"{s}.self_attn.{leaf}.weight"]
            for name, v in memories[i].state_dict().items():
                w[f"{p}.memory.{name}"] = v
        w[f"{p}.mlp.gate.weight"] = sd[f"{s}.mlp.gate_proj.weight"]
        w[f"{p}.mlp.up.weight"] = sd[f"{s}.mlp.up_proj.weight"]
        w[f"{p}.mlp.down.weight"] = sd[f"{s}.mlp.down_proj.weight"]
    return w


def save(out_dir, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out_dir, name)
    save_file(tensors, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": digest, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {path}: {len(tensors)} tensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("testdata", "golden", "edgerock"))
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    for variant, ties in (("tiny_text", False), ("tiny_ties", True)):
        cfg, model, memories, tokens = build(args.seed, ties)
        with torch.no_grad():
            taps = forward(cfg, model, memories, tokens)
            again = forward(*build(args.seed, ties))
        assert torch.equal(taps["logits"], again["logits"]), "fresh-construction determinism failed"
        if ties:
            refs = taps["layer3.mem.refs"]
            chunk = min(MEMORY["max_commit_chunk"], WINDOW)
            last = refs[-1, 0].tolist()
            assert last == list(range(MEMORY["capacity"])), f"equal scores must keep the earliest tokens, got {last}"
            assert refs.shape[0] * chunk >= T
        out_dir = os.path.join(args.out, variant)
        os.makedirs(out_dir, exist_ok=True)
        manifest = {}
        save(out_dir, "edgerock_tiny.safetensors", taps, manifest)
        save(out_dir, "edgerock_tiny_weights.safetensors", collect_weights(cfg, model, memories), manifest)
        manifest["_meta"] = {
            "seed": args.seed, "T": T, "variant": variant, "window": WINDOW, "memory": MEMORY, "config": TINY,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        }
        manifest["source"] = source_block(
            checkpoint=None,
            identity={"d_model": TINY["hidden_size"], "n_layers": TINY["num_hidden_layers"], "n_heads": TINY["num_attention_heads"], "n_kv_heads": TINY["num_key_value_heads"], "vocab": TINY["vocab_size"], "capacity": MEMORY["capacity"], "window": WINDOW},
        )
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
