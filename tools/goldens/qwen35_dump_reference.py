#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Qwen3.5/3.8's dense hybrid GDN/GQA text-decoder reference goldens
(tiny random weights).

Unlike `qwen35moe` (ported with no `torch`/`transformers` available - see
`crates/qwen35moe/src/model.rs`'s module doc), `transformers.models.qwen3_5`
(the REAL reference this arch's HF `architectures[0]` names) is installed on
this box, so this dumper drives it directly - no vendored/scratchpad copy.

Runs `Qwen3_5TextModel` (the text tower only - no lm_head, no vision; a
random `lm_head` Linear is added here purely so a `crates/qwen35` logits
parity test has something to check) at TINY, deliberately-DISTINCT dims
(every dimension that could be swapped with another by an axis-order bug is
a different number - see `assert_dims_distinct` below), with every op-
sequence-changing FLAG at its real value (`full_attention_interval=4`,
`partial_rotary_factor=0.25`, `mrope_interleaved=True`) - the same "toy
width, real flags" discipline `ltxv_dit_dump_reference.py` documents.

Outputs (under `--out`, default `testdata/golden/qwen35/tiny_text/`):
  qwen35_tiny_text.safetensors          every stage tap (see below)
  qwen35_tiny_text_weights.safetensors  the tiny model's OWN weights,
                                         renamed to brain's `blocks.{l}.*`
                                         convention - no checkpoint needed to
                                         replay this golden in Rust
  manifest.json                          shapes, sha256, run params, config,
                                         library versions

## What gets tapped, per layer

`layer{i}.input_layernorm` (post-ln1), then per mixer type:

Gated DeltaNet (`layer{i}.gdn.*`): `in_proj_qkv`/`in_proj_z`/`in_proj_b`/
`in_proj_a` (raw projections), `conv_raw` (post-conv1d, PRE-silu/truncate),
`mixed_qkv_silu` (post-silu, the real `query`/`key`/`value` source), `beta`,
`g` (the raw decay gate), `q_l2norm`/`k_l2norm` (informational - see the
chunk-size self-check below for why these are NOT fed back into the real
recurrence here), `core_attn_out`, `gated_norm` (post `Qwen3_5RMSNormGated`,
no `1+w`), `out_proj`.

Gated Attention (`layer{i}.gqa.*`): `query_raw`/`gate_raw` (the two chunks of
the doubled `q_proj`), `q_norm`/`k_norm` (post-QK-norm, pre-RoPE), `q_rope`/
`k_rope` (post partial-RoPE), `attn_ctx` (post softmax-attention), `gated_ctx`
(post `sigmoid(gate)` multiply), `out_proj`.

Then `layer{i}.post_attention_layernorm`, `layer{i}.mlp.{gate_pre,up,down}`,
and `layer{i}.out` (the full decoder layer's output, both residual adds
applied) - what a Rust replay's per-layer output must match.

Top level: `tokens`, `embed`, `cos`/`sin` (the M-RoPE tables), `final_hidden`
(post final norm - `Qwen3_5TextModel.forward` already applies it), `logits`.

## Self-validation inside the dumper (porting.md's "settle with an
## experiment, not a read")

1. **Manual-vs-real decoder-layer replay.** Every tap above comes from a
   HAND-DRIVEN replay of each submodule in the real forward's own order (NOT
   `register_forward_hook`, which only sees a module's own input/output, not
   the values BETWEEN its op-sequence steps) - so each layer's manually
   composed output is cross-checked against the REAL `Qwen3_5DecoderLayer`'s
   own forward (captured via a boundary hook) to prove the replay's op order
   and shapes are right, not just self-consistent.
2. **`(1+w)` RMSNorm fold, both by construction and by direct comparison.**
   `Qwen3_5PreTrainedModel._init_weights` zero-inits every plain
   `Qwen3_5RMSNorm.weight` ("we initialize with 0s to be 1 centered as the
   RMSNorm here does (1 + weight)" - its own comment) - which would make the
   fold's effect INVISIBLE in a golden dumped at default init (multiplying by
   `1+0=1`). This dumper explicitly perturbs every plain RMSNorm weight to a
   small nonzero value after construction (mirroring `ltxv_dit_dump_
   reference.py`'s re-randomization of a same-reason-degenerate
   `scale_shift_table`) specifically so a Rust port that FORGETS the `+1`
   fold produces a visibly wrong tap, not a coincidentally-right one. Also
   directly asserts `norm(x) == x*rsqrt(mean(x^2)+eps)*(1+w)` against the
   real module's own output.
3. **GDN chunk-size invariance.** The real reference always chunks at
   `chunk_size=64` (padding a short sequence to one giant chunk); `model::gdn`
   picks the largest divisor of `t` in `[64,32,16,8,4,2,1]`
   (`gdn_chunk_size`), which for this dumper's `T=24` is 8 (3 real chunks) -
   a DIFFERENT compute partition of the identical recurrence. This dumper
   calls the reference's own `torch_chunk_gated_delta_rule` at BOTH
   `chunk_size=8` and `chunk_size=64` and asserts they agree, before saving
   the `chunk_size=8` result as `core_attn_out` - proving chunk size is a
   pure compute-order choice, not a semantic one, which is the exact
   assumption `crates/model/src/gdn.rs`'s whole chunked design depends on.

## Scope: text only

The vision tower golden (real dims: depth 27, hidden 1152) is deferred to
the M9 vision-splice milestone rather than dumped here unused for several
milestones - `crates/qwen35moe/src/vl.rs` already proves `crates/qwen3vl`'s
ViT/PatchMerger/mRoPE code is reusable unmodified for this model family, so
that golden's real work is straightforward when M9 actually needs it.

Usage:
  python tools/goldens/qwen35_dump_reference.py [--out DIR] [--seed N]
"""

import argparse
import hashlib
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    l2norm,
    torch_chunk_gated_delta_rule,
)

# Every dim that could be confused for another by an axis-order/shape bug is
# a distinct number (lessons.md #4's "at the real config head_dim ==
# linear_key_head_dim == linear_value_head_dim == 128, a head-width/head-
# count swap would pass at cosine 1.0" - this tiny config makes that class of
# bug loud instead of silent). `full_attention_interval`/
# `linear_conv_kernel_dim` are left at their real small-integer values: they
# are schedule/kernel-width KNOBS, never a tensor axis size, so a collision
# between them (both happen to be 4) is not a confusable-axis risk.
TINY_TEXT = dict(
    vocab_size=29,
    hidden_size=96,
    intermediate_size=112,
    num_hidden_layers=4,
    num_attention_heads=3,
    num_key_value_heads=1,
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
    tie_word_embeddings=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        # sums to rotary_dim//2 = int(40*0.25)//2 = 5, the tiny analogue of
        # the real [11,11,10] (which sums to 256*0.25//2 = 32).
        "mrope_section": [2, 2, 1],
    },
)

T = 24  # gdn_chunk_size(24) == 8 -> 3 real chunks, matching qwen35moe's own tiny() convention.
GDN_CHUNK = 8


def assert_dims_distinct(cfg):
    dims = {
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size,
        "linear_num_key_heads": cfg.linear_num_key_heads,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "linear_value_head_dim": cfg.linear_value_head_dim,
        "vocab_size": cfg.vocab_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "T": T,
    }
    seen = {}
    for name, val in dims.items():
        assert val not in seen, f"tiny dims collide: {name}={val} == {seen[val]}={val} (masks axis-order bugs)"
        seen[val] = name
    print(f"dims distinct: {dims}")


def perturb_plain_rmsnorm_weights(model, gen):
    """Every `Qwen3_5RMSNorm.weight` zero-inits by design (see module
    docstring point 2) - overwrite with a small nonzero value so the `1+w`
    fold is actually exercised by this golden. `Qwen3_5RMSNormGated`
    (`linear_attn.norm`) is untouched: its default `ones()` init is already
    a legitimate, non-degenerate value (no `1+w` reparam to hide)."""
    n = 0
    for m in model.modules():
        if isinstance(m, Qwen3_5RMSNorm):
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=gen) * 0.1)
            n += 1
    print(f"perturbed {n} plain RMSNorm weights away from their degenerate zero init")


def save(out_dir, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out_dir, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    total = sum(v.numel() for v in tensors.values()) * 4 / 1e6
    print(f"wrote {name}: {len(tensors)} tensors, {total:.3f} MB")


def manual_gdn_forward(layer, x, taps, prefix):
    b, t, _ = x.shape
    mixed_qkv_raw = layer.in_proj_qkv(x)
    taps[f"{prefix}.in_proj_qkv"] = mixed_qkv_raw[0]
    z_raw = layer.in_proj_z(x)
    taps[f"{prefix}.in_proj_z"] = z_raw[0]
    b_raw = layer.in_proj_b(x)
    taps[f"{prefix}.in_proj_b"] = b_raw[0]
    a_raw = layer.in_proj_a(x)
    taps[f"{prefix}.in_proj_a"] = a_raw[0]

    mixed_qkv_t = mixed_qkv_raw.transpose(1, 2)
    conv_raw = layer.conv1d(mixed_qkv_t)[:, :, :t]
    taps[f"{prefix}.conv_raw"] = conv_raw[0].transpose(0, 1)
    mixed_qkv = F.silu(conv_raw).transpose(1, 2)
    taps[f"{prefix}.mixed_qkv_silu"] = mixed_qkv[0]

    query, key, value = torch.split(mixed_qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    query = query.reshape(b, t, -1, layer.head_k_dim)
    key = key.reshape(b, t, -1, layer.head_k_dim)
    value = value.reshape(b, t, -1, layer.head_v_dim)

    beta = b_raw.sigmoid()
    taps[f"{prefix}.beta"] = beta[0]
    g = -layer.A_log.float().exp() * F.softplus(a_raw.float() + layer.dt_bias)
    taps[f"{prefix}.g"] = g[0]

    if layer.num_v_heads // layer.num_k_heads > 1:
        query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
        key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)

    taps[f"{prefix}.q_l2norm"] = l2norm(query, dim=-1, eps=1e-6)[0]
    taps[f"{prefix}.k_l2norm"] = l2norm(key, dim=-1, eps=1e-6)[0]

    core_small, _ = torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, chunk_size=GDN_CHUNK, use_qk_l2norm_in_kernel=True
    )
    core_real, _ = torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, chunk_size=64, use_qk_l2norm_in_kernel=True
    )
    diff = (core_small - core_real).abs().max().item()
    assert diff < 1e-4, f"{prefix}: chunk-size invariance failed (chunk={GDN_CHUNK} vs chunk=64 diff {diff})"
    print(f"{prefix}: chunk-size invariance (chunk={GDN_CHUNK} vs chunk=64) max abs diff {diff:.2e}")
    core_attn_out = core_small
    taps[f"{prefix}.core_attn_out"] = core_attn_out[0]

    core_flat = core_attn_out.reshape(-1, layer.head_v_dim)
    z_flat = z_raw.reshape(b, t, -1, layer.head_v_dim).reshape(-1, layer.head_v_dim)
    normed = layer.norm(core_flat, z_flat).reshape(b, t, -1)
    taps[f"{prefix}.gated_norm"] = normed[0]
    out = layer.out_proj(normed)
    taps[f"{prefix}.out_proj"] = out[0]
    return out


def manual_gqa_forward(layer, x, cos, sin, taps, prefix):
    b, t, _ = x.shape
    q_and_gate = layer.q_proj(x).view(b, t, -1, layer.head_dim * 2)
    query, gate = torch.chunk(q_and_gate, 2, dim=-1)
    taps[f"{prefix}.query_raw"] = query.reshape(b, t, -1)[0]
    gate = gate.reshape(b, t, -1)
    taps[f"{prefix}.gate_raw"] = gate[0]

    hidden_shape = (b, t, -1, layer.head_dim)
    query = layer.q_norm(query.reshape(hidden_shape))
    taps[f"{prefix}.q_norm"] = query.reshape(b, t, -1)[0]
    query = query.transpose(1, 2)

    key = layer.k_norm(layer.k_proj(x).view(hidden_shape))
    taps[f"{prefix}.k_norm"] = key.reshape(b, t, -1)[0]
    key = key.transpose(1, 2)

    value = layer.v_proj(x).view(hidden_shape).transpose(1, 2)

    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    taps[f"{prefix}.q_rope"] = query.transpose(1, 2).reshape(b, t, -1)[0]
    taps[f"{prefix}.k_rope"] = key.transpose(1, 2).reshape(b, t, -1)[0]

    causal_mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)[None, None, :, :]
    attn_output, _ = eager_attention_forward(layer, query, key, value, attention_mask=causal_mask, scaling=layer.scaling)
    attn_output = attn_output.reshape(b, t, -1).contiguous()
    taps[f"{prefix}.attn_ctx"] = attn_output[0]
    attn_output = attn_output * torch.sigmoid(gate)
    taps[f"{prefix}.gated_ctx"] = attn_output[0]
    out = layer.o_proj(attn_output)
    taps[f"{prefix}.out_proj"] = out[0]
    return out


def collect_weights(model, lm_head, cfg):
    """Rename HF tensors to brain's `blocks.{l}.*` convention
    (`crates/qwen35moe/src/config.rs::param_list`'s naming, extended with
    `mlp.{gate,up,down}` for the dense MLP - `crates/qwen3/src/model.rs`'s
    own dense-MLP leaf names) - no import/rename step needed to replay this
    golden in Rust."""
    w = {}
    sd = model.state_dict()
    w["tok.weight"] = sd["embed_tokens.weight"]
    for i, block_type in enumerate(cfg.layer_types):
        p = f"blocks.{i}"
        w[f"{p}.ln1.weight"] = sd[f"layers.{i}.input_layernorm.weight"]
        w[f"{p}.ln2.weight"] = sd[f"layers.{i}.post_attention_layernorm.weight"]
        if block_type == "linear_attention":
            g = f"layers.{i}.linear_attn"
            w[f"{p}.linear_attn.in_proj_qkv.weight"] = sd[f"{g}.in_proj_qkv.weight"]
            w[f"{p}.linear_attn.in_proj_z.weight"] = sd[f"{g}.in_proj_z.weight"]
            w[f"{p}.linear_attn.in_proj_b.weight"] = sd[f"{g}.in_proj_b.weight"]
            w[f"{p}.linear_attn.in_proj_a.weight"] = sd[f"{g}.in_proj_a.weight"]
            w[f"{p}.linear_attn.conv1d.weight"] = sd[f"{g}.conv1d.weight"].squeeze(1)
            w[f"{p}.linear_attn.A_log"] = sd[f"{g}.A_log"]
            w[f"{p}.linear_attn.dt_bias"] = sd[f"{g}.dt_bias"]
            w[f"{p}.linear_attn.norm.weight"] = sd[f"{g}.norm.weight"]
            w[f"{p}.linear_attn.out_proj.weight"] = sd[f"{g}.out_proj.weight"]
        else:
            g = f"layers.{i}.self_attn"
            w[f"{p}.self_attn.q_proj.weight"] = sd[f"{g}.q_proj.weight"]
            w[f"{p}.self_attn.k_proj.weight"] = sd[f"{g}.k_proj.weight"]
            w[f"{p}.self_attn.v_proj.weight"] = sd[f"{g}.v_proj.weight"]
            w[f"{p}.self_attn.q_norm.weight"] = sd[f"{g}.q_norm.weight"]
            w[f"{p}.self_attn.k_norm.weight"] = sd[f"{g}.k_norm.weight"]
            w[f"{p}.self_attn.o_proj.weight"] = sd[f"{g}.o_proj.weight"]
        w[f"{p}.mlp.gate.weight"] = sd[f"layers.{i}.mlp.gate_proj.weight"]
        w[f"{p}.mlp.up.weight"] = sd[f"layers.{i}.mlp.up_proj.weight"]
        w[f"{p}.mlp.down.weight"] = sd[f"layers.{i}.mlp.down_proj.weight"]
    w["norm.weight"] = sd["norm.weight"]
    w["lm_head.weight"] = lm_head.weight.detach()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("testdata", "golden", "qwen35", "tiny_text"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Qwen3_5TextConfig(**TINY_TEXT)
    assert_dims_distinct(cfg)
    assert cfg.layer_types == ["linear_attention", "linear_attention", "linear_attention", "full_attention"], cfg.layer_types

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1)
    model = Qwen3_5TextModel(cfg).eval()
    perturb_plain_rmsnorm_weights(model, gen)
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(lm_head.weight.shape, generator=gen) * 0.02)

    b = 1
    tokens = torch.randint(0, cfg.vocab_size, (b, T), generator=gen)

    # Boundary hook: capture each REAL decoder layer's own input/output, the
    # ground truth the manual replay below is checked against.
    real_in, real_out = {}, {}

    def mk_hook(i):
        def hook(_module, inputs, output):
            real_in[i] = inputs[0].detach().clone()
            real_out[i] = output.detach().clone()

        return hook

    hooks = [layer.register_forward_hook(mk_hook(i)) for i, layer in enumerate(model.layers)]
    with torch.no_grad():
        real = model(input_ids=tokens)
    for h in hooks:
        h.remove()
    final_hidden = real.last_hidden_state

    # Reproduce two more times from a fresh construction+seed: determinism.
    torch.manual_seed(args.seed)
    gen2 = torch.Generator().manual_seed(args.seed + 1)
    model2 = Qwen3_5TextModel(cfg).eval()
    perturb_plain_rmsnorm_weights(model2, gen2)
    with torch.no_grad():
        real2 = model2(input_ids=tokens)
    assert torch.equal(final_hidden, real2.last_hidden_state), "fresh-construction determinism failed"
    print("fresh-construction determinism: OK (bit-identical)")

    # RoPE tables, the same way Qwen3_5TextModel.forward builds them.
    position_ids = torch.arange(T).view(1, 1, -1).expand(4, b, -1)
    text_position_ids = position_ids[0]
    rope_position_ids = position_ids[1:]
    cos, sin = model.rotary_emb(final_hidden, rope_position_ids)
    # RoPE unit-rotation invariant (same structural check ltxv's dumper uses).
    unit = (cos.float() ** 2 + sin.float() ** 2)
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-5), "cos^2+sin^2 != 1"
    print("RoPE unit-rotation invariant: OK")

    tensors = {"tokens": tokens.to(torch.int32), "embed": model.embed_tokens(tokens)[0], "cos": cos[0], "sin": sin[0]}

    for i, block_type in enumerate(cfg.layer_types):
        layer = model.layers[i]
        x_in = real_in[i]
        residual = x_in
        xn = layer.input_layernorm(x_in)
        tensors[f"layer{i}.input_layernorm"] = xn[0]

        if block_type == "linear_attention":
            mixer_out = manual_gdn_forward(layer.linear_attn, xn, tensors, f"layer{i}.gdn")
        else:
            mixer_out = manual_gqa_forward(layer.self_attn, xn, cos, sin, tensors, f"layer{i}.gqa")

        h1 = residual + mixer_out
        residual2 = h1
        xn2 = layer.post_attention_layernorm(h1)
        tensors[f"layer{i}.post_attention_layernorm"] = xn2[0]

        gate_pre = layer.mlp.gate_proj(xn2)
        up = layer.mlp.up_proj(xn2)
        h_act = F.silu(gate_pre) * up
        down = layer.mlp.down_proj(h_act)
        tensors[f"layer{i}.mlp.gate_pre"] = gate_pre[0]
        tensors[f"layer{i}.mlp.up"] = up[0]
        tensors[f"layer{i}.mlp.down"] = down[0]

        out_manual = residual2 + down
        tensors[f"layer{i}.out"] = out_manual[0]

        diff = (out_manual - real_out[i]).abs().max().item()
        assert diff < 1e-4, f"layer {i} ({block_type}) manual replay diverges from real forward by {diff}"
        print(f"layer {i} ({block_type}): manual replay vs real forward max abs diff {diff:.2e}")

    tensors["final_hidden"] = final_hidden[0]
    logits = lm_head(final_hidden)
    tensors["logits"] = logits[0]

    # `(1+w)` RMSNorm fold, checked directly against the real module (not
    # just by construction of a nonzero weight above).
    probe = torch.randn(4, cfg.hidden_size, generator=gen)
    ln0 = model.layers[0].input_layernorm
    manual_norm = probe * torch.rsqrt(probe.pow(2).mean(-1, keepdim=True) + ln0.eps) * (1.0 + ln0.weight)
    via_module = ln0(probe)
    assert torch.allclose(manual_norm, via_module, atol=1e-5), "RMSNorm (1+w) fold self-check failed"
    print("RMSNorm (1+w) fold self-check: OK")

    save(args.out, "qwen35_tiny_text.safetensors", tensors, manifest := {})
    weights = collect_weights(model, lm_head, cfg)
    save(args.out, "qwen35_tiny_text_weights.safetensors", weights, manifest)

    manifest["_meta"] = {
        "seed": args.seed,
        "T": T,
        "B": b,
        "gdn_chunk": GDN_CHUNK,
        "tiny_text_config": TINY_TEXT,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest.json -> {args.out}")


if __name__ == "__main__":
    main()
