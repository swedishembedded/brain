#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reference-parity gates for sparse mixture-of-
# experts decoders for its clients. If your team needs expertise in proving a
# hybrid linear-attention MoE port matches its published reference then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Dump Qwen3.5-35B-A3B reference goldens on tiny random weights.

Drives `transformers.models.qwen3_5_moe.Qwen3_5MoeForCausalLM` (the reference
the release's `architectures[0]` names, text tower) at a tiny shape with every
op-sequence flag at its released value: `full_attention_interval=4` over 8
layers (`linear_attention` x3, `full_attention`, twice, the last layer full like
the release's), `partial_rotary_factor=0.25`, a doubled gated `q_proj`, a
softmax top-k router over 10 experts with renormalised top-5 weights, the
sigmoid-gated shared expert, untied embeddings (`tie_word_embeddings=false`, as
released). Every dimension a quantized linear contracts over is a multiple of
32, so brain's int8 tier runs on the same weights; the shared expert has the
routed experts' width, as in the release, so the int8 tier carries it as one
more bank block.

Outputs (under `--out`, default `testdata/golden/qwen35moe/tiny/`):

  checkpoint/config.json, checkpoint/model.safetensors
      the tiny model's own weights in the release's directory layout
      (`Qwen3_5MoeForConditionalGeneration` config with a `text_config`,
      `model.language_model.*` tensor names, one tensor per expert projection,
      norm gains stored as `w` for `1 + w`), which brain reads through
      `qwen35moe::hf_source` exactly as it reads the release
  qwen35moe_tiny.safetensors   the reference taps, per prompt `p{i}`:
      tokens [L], embed [L, d], layer{l}.out [L, d] (each decoder layer's
      output), layer{l}.abs_sum [L, d], layer{l}.router_gap [L],
      final_hidden [L, d] (post final norm), logits [L, V], and
      decode_logits [L - m + 1, V]: the logits of the reference's own cached
      decode after a prefill of the first m = decode_from tokens, one row per
      position m - 1 .. L - 1, and int8.layer{l}.out / int8.logits: the
      reference under brain's int8 rounding (see "The int8 tier" below)
  manifest.json                shapes, sha256, prompts, layer types, versions,
                               the int8 statistics, and the `source` identity
                               block

`layer{l}.abs_sum` is the absolute-term sum `S` of every residual element:
`|embed|` plus, for each layer up to `l`, the absolute terms of the mixer's
output projection (`|W_o| |a|`) and of the MoE combine (`sum_e w_e |W_down_e|
|h_e| + sigmoid(g) |W_sd| |h_s|`). A test derives its per-element binary32
agreement bound from it (`brain_testutil::rounding`) instead of choosing one.
`layer{l}.router_gap` is, per token, the router logit of its k-th expert minus
that of its (k+1)-th: the margin within which a rounding difference could
change the routing.

Three prompts: 7 tokens, 29 tokens, and 100 tokens - longer than the 64-row GDN
chunk the reference uses (it pads to two chunks), so the cross-chunk state
hand-off is in the golden.

## The int8 tier

`Int8Emulation` applies brain's W8A8 rounding (`qwen35moe::q8`, `Q8_0`'s
block rule: one scale `max|block| / 127` per 32 elements of the contraction
axis, weight and activation alike) at every site the tier quantizes, so the
tier is compared with the reference's own quantized function rather than with
fp32. `_meta.int8[i]` records how far that function is from fp32
(`loss_vs_fp32`) and how far its binary32 evaluation is from its binary64 one
(`fp32_vs_fp64`): the function is discontinuous at every rounding midpoint, so
rounding alone moves it once a code crosses one.

## Self-validation

1. The MoE block of every layer is replayed by hand (router softmax, top-k,
   renormalisation, each expert, the gated shared expert) and must equal the
   reference module's own output; the decoder layer's output must equal its
   input plus the mixer and MoE outputs. The abs sums come from that replay.
2. The cached decode (recurrent GDN rule, KV cache) must reproduce the full
   forward's logits rows, so the decode golden and the prefill golden are the
   same function.
3. Every plain RMSNorm gain zero-inits by design (`1 + w` with `w = 0`), which
   would hide a missing `+1`; every one, and the gated norm's `ones` init, is
   perturbed. The router zero-inits in the module and is re-drawn wide enough
   that the routing margins sit far above rounding.

Usage:
  python -I tools/goldens/qwen35moe_dump_reference.py [--out DIR] --seed 1234
"""

import argparse
import copy
import hashlib
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeTextConfig,
)

# Every dimension an axis-order bug could confuse with another is a distinct
# number, except the two the release itself keeps equal: the shared expert's
# width is the routed experts' (512 and 512 in the release), which is what lets
# the int8 tier carry it as a bank block. `linear_conv_kernel_dim` is a kernel
# width, not a tensor axis, so its value is not in the check.
TINY = dict(
    vocab_size=53,
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=6,
    num_key_value_heads=3,
    head_dim=16,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=12,
    linear_value_head_dim=24,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=10,
    num_experts_per_tok=5,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
    initializer_range=0.1,
    tie_word_embeddings=False,
    attention_bias=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        # Sums to rotary_dim // 2 = int(16 * 0.25) // 2 = 2, the tiny analogue
        # of the release's [11, 11, 10]. Text positions are equal on all three
        # axes, so the split does not change a text forward.
        "mrope_section": [1, 1, 0],
    },
)
FULL_ATTENTION_INTERVAL = 4
# (length, decode_from): the cached decode prefills `decode_from` tokens, then
# steps through the rest one at a time.
PROMPTS = [(7, 1), (29, 20), (100, 70)]
ROUTER_STD = 0.3
# The dumper's own floor on the routing margin. A test holds the recorded gaps
# to its derived rounding bound; this only refuses a seed whose margins are so
# small that no binary32 evaluation could be expected to route alike.
MIN_ROUTER_GAP = 1e-4


def assert_dims_distinct(cfg):
    dims = {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "linear_value_head_dim": cfg.linear_value_head_dim,
        "linear_num_key_heads": cfg.linear_num_key_heads,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
    }
    seen = {}
    for name, val in dims.items():
        assert val not in seen, f"tiny dims collide: {name}={val} == {seen[val]} (masks axis-order bugs)"
        seen[val] = name
    for k in (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim,
              cfg.linear_num_value_heads * cfg.linear_value_head_dim, cfg.moe_intermediate_size):
        assert k % 32 == 0, f"contraction width {k} is not a whole number of int8 scale groups"
    print(f"dims distinct: {dims}")


def build(seed):
    cfg = Qwen3_5MoeTextConfig(
        **TINY,
        full_attention_interval=FULL_ATTENTION_INTERVAL,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    torch.manual_seed(seed)
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    gen = torch.Generator().manual_seed(seed + 1)
    n_plain = n_gated = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, Qwen3_5MoeRMSNorm):
                m.weight.copy_(torch.randn(m.weight.shape, generator=gen) * 0.1)
                n_plain += 1
            elif isinstance(m, Qwen3_5MoeRMSNormGated):
                m.weight.copy_(1.0 + torch.randn(m.weight.shape, generator=gen) * 0.1)
                n_gated += 1
        for layer in model.model.layers:
            layer.mlp.gate.weight.copy_(torch.randn(layer.mlp.gate.weight.shape, generator=gen) * ROUTER_STD)
    print(f"perturbed {n_plain} plain and {n_gated} gated RMSNorm gains; router weights drawn at std {ROUTER_STD}")
    return cfg, model, gen


def moe_replay(mlp, x, top_k):
    """The sparse MoE block by hand: returns (output, abs_sum, router_gap)."""
    w_gu, w_d = mlp.experts.gate_up_proj, mlp.experts.down_proj
    ff = w_d.shape[-1]
    logits = F.linear(x, mlp.gate.weight)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    top_v, top_i = torch.topk(probs, top_k, dim=-1)
    top_v = top_v / top_v.sum(dim=-1, keepdim=True)
    ordered = torch.sort(logits, dim=-1, descending=True).values
    gap = ordered[:, top_k - 1] - ordered[:, top_k]

    se = mlp.shared_expert
    h_s = F.silu(se.gate_proj(x)) * se.up_proj(x)
    sg = torch.sigmoid(mlp.shared_expert_gate(x))
    out = sg * se.down_proj(h_s)
    abs_sum = sg * (h_s.abs() @ se.down_proj.weight.abs().T)
    for e in range(w_gu.shape[0]):
        wt = (top_v * (top_i == e)).sum(dim=-1, keepdim=True)
        gu = x @ w_gu[e].T
        h = F.silu(gu[:, :ff]) * gu[:, ff:]
        out = out + wt * (h @ w_d[e].T)
        abs_sum = abs_sum + wt * (h.abs() @ w_d[e].abs().T)
    return out, abs_sum, gap, top_i


GROUP = 32


def fake_quant(x, half_away):
    """Round `x` to int8 codes with one scale per 32 elements of its last axis (`max|block| / 127`, floored at 1e-8),
    and back: brain's int8 tier, `Q8_0`'s block rule. Weights round half away from zero (the host packer), activations
    half to even (the device packer); the two differ only on exact ties."""
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // GROUP, GROUP)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    v = blocks / scale
    q = torch.sign(v) * torch.floor(v.abs() + 0.5) if half_away else torch.round(v)
    return (q.clamp(-127, 127) * scale).reshape(shape)


class Int8Emulation:
    """The reference with brain's int8 tier applied at the sites the tier quantizes (`qwen35moe::q8`): every mixer
    projection, every routed expert's gate/up/down, the shared expert when it has their shape, and the untied head;
    the router, the shared-expert gate and the embedding stay fp32. Weight and activation alike are rounded per
    32-element block of the contraction axis. Installed by patching the module forwards; `restore` puts the
    originals back."""

    def __init__(self, model, cfg):
        self.saved = []
        linears = []
        for i, layer in enumerate(model.model.layers):
            if cfg.layer_types[i] == "linear_attention":
                la = layer.linear_attn
                linears += [la.in_proj_qkv, la.in_proj_z, la.in_proj_a, la.in_proj_b, la.out_proj]
            else:
                sa = layer.self_attn
                linears += [sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj]
            if cfg.shared_expert_intermediate_size == cfg.moe_intermediate_size:
                se = layer.mlp.shared_expert
                linears += [se.gate_proj, se.up_proj, se.down_proj]
            self.patch_experts(layer.mlp.experts)
        if not cfg.tie_word_embeddings:
            linears.append(model.lm_head)
        for lin in linears:
            wq = fake_quant(lin.weight.detach(), half_away=True)
            self.patch(lin, lambda x, wq=wq: F.linear(fake_quant(x, half_away=False), wq))

    def patch(self, module, forward):
        self.saved.append((module, module.forward))
        module.forward = forward

    def patch_experts(self, experts):
        w_gu = fake_quant(experts.gate_up_proj.detach(), half_away=True)
        w_d = fake_quant(experts.down_proj.detach(), half_away=True)

        def forward(hidden_states, top_k_index, top_k_weights):
            out = torch.zeros_like(hidden_states)
            xq = fake_quant(hidden_states, half_away=False)
            ff = w_d.shape[-1]
            for e in range(w_gu.shape[0]):
                tok, pos = torch.where(top_k_index == e)
                if tok.numel() == 0:
                    continue
                gu = F.linear(xq[tok], w_gu[e])
                h = F.silu(gu[:, :ff]) * gu[:, ff:]
                y = F.linear(fake_quant(h, half_away=False), w_d[e])
                out.index_add_(0, tok, y * top_k_weights[tok, pos, None])
            return out

        self.patch(experts, forward)

    def restore(self):
        for module, forward in reversed(self.saved):
            module.forward = forward
        self.saved.clear()


def int8_taps(model, cfg, tokens, prefix):
    """The int8-emulated reference's layer outputs and logits for one prompt."""
    emu = Int8Emulation(model, cfg)
    outs = {}
    hooks = [layer.register_forward_hook(lambda _m, _a, o, i=i: outs.__setitem__(i, o[0].detach().clone()))
             for i, layer in enumerate(model.model.layers)]
    try:
        with torch.no_grad():
            logits = model(input_ids=tokens[None, :], use_cache=False).logits[0]
    finally:
        for h in hooks:
            h.remove()
        emu.restore()
    taps = {f"{prefix}.int8.layer{i}.out": outs[i] for i in range(len(model.model.layers))}
    taps[f"{prefix}.int8.logits"] = logits
    return taps


def prompt_taps(model, cfg, tokens, decode_from, prefix):
    """Run one prompt through the reference and return its taps."""
    layers = model.model.layers
    seen = {}
    hooks = []

    def keep(key, fn):
        def hook(*args):
            seen[key] = fn(*args)
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(keep(("in", i), lambda _m, a: a[0][0].detach().clone())))
        hooks.append(layer.register_forward_hook(keep(("out", i), lambda _m, _a, o: o[0].detach().clone())))
        mixer = layer.linear_attn if cfg.layer_types[i] == "linear_attention" else layer.self_attn
        proj = mixer.out_proj if cfg.layer_types[i] == "linear_attention" else mixer.o_proj
        hooks.append(mixer.register_forward_hook(
            keep(("mix", i), lambda _m, _a, o: (o[0] if isinstance(o, tuple) else o)[0].detach().clone())))
        hooks.append(proj.register_forward_pre_hook(
            keep(("mix_abs", i), lambda m, a: (a[0][0].abs() @ m.weight.abs().T).detach().clone())))
        hooks.append(layer.mlp.register_forward_pre_hook(keep(("xn2", i), lambda _m, a: a[0][0].detach().clone())))
        hooks.append(layer.mlp.register_forward_hook(keep(("moe", i), lambda _m, _a, o: o[0].detach().clone())))
    with torch.no_grad():
        full = model(input_ids=tokens[None, :], use_cache=False, output_hidden_states=False)
    for h in hooks:
        h.remove()

    taps = {f"{prefix}.tokens": tokens.to(torch.int32)}
    with torch.no_grad():
        embed = model.model.embed_tokens(tokens)
    taps[f"{prefix}.embed"] = embed
    abs_sum = embed.abs()
    used = set()
    worst = {"moe": 0.0, "layer": 0.0, "gap": float("inf")}
    with torch.no_grad():
        for i, layer in enumerate(layers):
            moe, moe_abs, gap, top_i = moe_replay(layer.mlp, seen[("xn2", i)], cfg.num_experts_per_tok)
            used.update(top_i.flatten().tolist())
            scale = seen[("moe", i)].abs().max().item()
            worst["moe"] = max(worst["moe"], (moe - seen[("moe", i)]).abs().max().item() / max(scale, 1e-30))
            out = seen[("in", i)] + seen[("mix", i)] + seen[("moe", i)]
            worst["layer"] = max(worst["layer"], (out - seen[("out", i)]).abs().max().item())
            worst["gap"] = min(worst["gap"], gap.min().item())
            abs_sum = abs_sum + seen[("mix_abs", i)] + moe_abs
            taps[f"{prefix}.layer{i}.out"] = seen[("out", i)]
            taps[f"{prefix}.layer{i}.abs_sum"] = abs_sum
            taps[f"{prefix}.layer{i}.router_gap"] = gap
    assert worst["moe"] < 1e-5, f"{prefix}: hand-replayed MoE differs from the reference block by {worst['moe']:.2e} (relative)"
    assert worst["layer"] < 1e-5, f"{prefix}: layer output != input + mixer + MoE (max abs {worst['layer']:.2e})"
    assert worst["gap"] > MIN_ROUTER_GAP, f"{prefix}: a routing margin of {worst['gap']:.2e} is below {MIN_ROUTER_GAP}; pick another seed"
    print(f"{prefix}: MoE replay rel {worst['moe']:.2e}, layer identity {worst['layer']:.2e}, "
          f"min router gap {worst['gap']:.3e}, experts used {len(used)}/{cfg.num_experts}")

    logits = full.logits[0]
    with torch.no_grad():
        final_hidden = model.model.norm(seen[("out", len(layers) - 1)])
        assert torch.allclose(model.lm_head(final_hidden), logits, atol=1e-6), "final norm + head do not give the logits"
    taps[f"{prefix}.final_hidden"] = final_hidden
    taps[f"{prefix}.logits"] = logits

    # The reference's own cached decode: prefill, then one token at a time
    # through the recurrent GDN rule and the KV cache.
    rows = []
    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        step = model(input_ids=tokens[None, :decode_from], past_key_values=cache, use_cache=True)
        rows.append(step.logits[0, -1])
        for j in range(decode_from, tokens.shape[0]):
            step = model(input_ids=tokens[None, j:j + 1], past_key_values=cache, use_cache=True)
            rows.append(step.logits[0, -1])
    decode = torch.stack(rows)
    drift = (decode - logits[decode_from - 1:]).abs().max().item()
    assert drift < 1e-4 * max(logits.abs().max().item(), 1.0), f"{prefix}: cached decode drifts from the full forward by {drift:.2e}"
    print(f"{prefix}: cached decode vs full forward max abs {drift:.2e}")
    taps[f"{prefix}.decode_logits"] = decode
    return taps, used


def release_layout(model):
    """The model's state dict under the release's tensor names: the
    `model.language_model.` prefix and one tensor per expert projection."""
    out = {}
    for name, t in model.state_dict().items():
        t = t.detach().to(torch.float32).contiguous()
        if name == "lm_head.weight":
            out[name] = t
            continue
        assert name.startswith("model."), name
        rest = "model.language_model." + name[len("model."):]
        if rest.endswith("mlp.experts.gate_up_proj"):
            base = rest[: -len("gate_up_proj")]
            ff = t.shape[1] // 2
            for e in range(t.shape[0]):
                out[f"{base}{e}.gate_proj.weight"] = t[e, :ff].clone()
                out[f"{base}{e}.up_proj.weight"] = t[e, ff:].clone()
        elif rest.endswith("mlp.experts.down_proj"):
            base = rest[: -len("down_proj")]
            for e in range(t.shape[0]):
                out[f"{base}{e}.down_proj.weight"] = t[e].clone()
        else:
            out[rest] = t
    return out


def release_config(cfg):
    text = json.loads(cfg.to_json_string(use_diff=False))
    text = {k: v for k, v in text.items() if not k.startswith("_")}
    text["full_attention_interval"] = FULL_ATTENTION_INTERVAL
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "text_config": text,
    }


def rel_l2(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def save(path, tensors, manifest, key):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    save_file(tensors, path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    manifest[key] = {"sha256": digest, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {path}: {len(tensors)} tensors, {sum(v.numel() for v in tensors.values()) * 4 / 1e6:.3f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("testdata", "golden", "qwen35moe", "tiny"))
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    ckpt = os.path.join(args.out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)

    cfg, model, gen = build(args.seed)
    assert_dims_distinct(cfg)
    expect = ["linear_attention"] * 3 + ["full_attention"]
    assert cfg.layer_types == expect * 2, cfg.layer_types

    _, again, _ = build(args.seed)
    probe = torch.randint(0, cfg.vocab_size, (1, 9), generator=torch.Generator().manual_seed(args.seed + 2))
    with torch.no_grad():
        assert torch.equal(model(input_ids=probe).logits, again(input_ids=probe).logits), "fresh-construction determinism failed"
    print("fresh-construction determinism: OK (bit-identical)")

    # The same weights in binary64: evaluating the int8 emulation at two precisions shows how far rounding alone
    # moves the W8A8 function (a code that lands on the other side of a rounding midpoint).
    wide = copy.deepcopy(model).double()
    taps, used, int8_stats = {}, set(), []
    for i, (n, decode_from) in enumerate(PROMPTS):
        tokens = torch.randint(0, cfg.vocab_size, (n,), generator=gen)
        t, u = prompt_taps(model, cfg, tokens, decode_from, f"p{i}")
        taps.update(t)
        used |= u
        taps.update(int8_taps(model, cfg, tokens, f"p{i}"))
        q = taps[f"p{i}.int8.logits"]
        stats = {
            "loss_vs_fp32": rel_l2(q, t[f"p{i}.logits"]),
            "fp32_vs_fp64": rel_l2(q, int8_taps(wide, cfg, tokens, "w")["w.int8.logits"]),
        }
        int8_stats.append(stats)
        print(f"p{i}: int8-emulated logits vs fp32 rel-L2 {stats['loss_vs_fp32']:.3e}, "
              f"int8 emulation in binary32 vs binary64 rel-L2 {stats['fp32_vs_fp64']:.3e}")
    with torch.no_grad():
        assert torch.equal(model(input_ids=probe).logits, again(input_ids=probe).logits), "the int8 emulation leaked into the model"
    assert len(used) == cfg.num_experts, f"only experts {sorted(used)} are ever routed to"

    manifest = {}
    save(os.path.join(args.out, "qwen35moe_tiny.safetensors"), taps, manifest, "qwen35moe_tiny.safetensors")
    weights_path = os.path.join(ckpt, "model.safetensors")
    save(weights_path, release_layout(model), manifest, "checkpoint/model.safetensors")
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump(release_config(cfg), f, indent=2)

    manifest["_meta"] = {
        "seed": args.seed,
        "prompts": [{"len": n, "decode_from": m} for n, m in PROMPTS],
        "int8": int8_stats,
        "layer_types": cfg.layer_types,
        "tiny_config": TINY,
        "full_attention_interval": FULL_ATTENTION_INTERVAL,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    # No `checkpoint`: the dumper builds its own weights. The identity is every
    # tensor axis of the tiny model, so the golden cannot be paired with the
    # real 35B-A3B config (or another tiny one) without saying so.
    manifest["source"] = source_block(
        files=[weights_path],
        identity={
            "vocab_size": TINY["vocab_size"],
            "hidden_size": TINY["hidden_size"],
            "num_hidden_layers": TINY["num_hidden_layers"],
            "num_attention_heads": TINY["num_attention_heads"],
            "num_key_value_heads": TINY["num_key_value_heads"],
            "head_dim": TINY["head_dim"],
            "linear_key_head_dim": TINY["linear_key_head_dim"],
            "linear_value_head_dim": TINY["linear_value_head_dim"],
            "linear_num_key_heads": TINY["linear_num_key_heads"],
            "linear_num_value_heads": TINY["linear_num_value_heads"],
            "moe_intermediate_size": TINY["moe_intermediate_size"],
            "shared_expert_intermediate_size": TINY["shared_expert_intermediate_size"],
            "num_experts": TINY["num_experts"],
            "num_experts_per_tok": TINY["num_experts_per_tok"],
        },
    )
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest.json -> {args.out}")


if __name__ == "__main__":
    main()
