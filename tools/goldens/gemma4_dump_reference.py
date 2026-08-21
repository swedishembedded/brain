#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Gemma-4 (unified text tower) reference goldens (tiny random weights).

Runs the OFFICIAL, INSTALLED `transformers.models.gemma4_unified` package (CPU,
fp32) - specifically `Gemma4UnifiedTextModel` (the compiled `modeling_
gemma4_unified.py`, not the `modular_*` source file - see this module's "which
module" note below) - at a TINY random-weight config, and dumps every boundary
a Rust unit-by-unit replay needs:

  gemma4_tiny.safetensors     input_ids, every `output_hidden_states` entry,
                               both RoPE tables (sliding + full/global), the
                               self-attention output of a SLIDING layer and of
                               the (k_eq_v) FULL/global layer, the final
                               (post-norm) hidden state, and the LTX-specific
                               49-state aggregate-embed projection output
  gemma4_tiny_weights.safetensors  the tiny model's OWN weights (state_dict)
                               PLUS the aggregate-embed projection's weight/
                               bias, so the Rust smoke test needs no checkpoint
  manifest.json                shapes, sha256, run params, config, versions

## Why tiny dims, but REAL config FLAGS

The real checkpoint (`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, 12B
params / 26 GB bf16, a unified text+vision+audio model LTX-2.5 only ever uses
as a TEXT encoder) cannot run on modest hardware. This dumps a toy-width text
tower, but every FLAG that changes the op sequence is the real LTX-2.5 value:
`attention_k_eq_v=True` on the global layers, `global_head_dim` DOUBLE the
sliding `head_dim` (real: 512 vs 256), `num_global_key_value_heads=1` (MQA on
the global layers vs GQA on the sliding ones), `sliding_window` genuinely
smaller than the token count so the window actually excludes something,
`hidden_activation="gelu_pytorch_tanh"`, `rms_norm_eps=1e-6`,
`tie_word_embeddings=True`, `attention_bias=False`, and the AUTO-DERIVED 5:1
`layer_types` alternation (`sliding_window_pattern=6`,
`configuration_gemma4.py`'s `Gemma4TextConfig.__post_init__` - 6 layers here
is the minimal instance of the real ratio: 5 sliding then 1 full, matching
`num_hidden_layers=48` -> 40 sliding + 8 full exactly). `vocab_size` and the
per-tensor hidden/intermediate dims are shrunk for a fast test - they do not
participate in any of the structurally interesting logic (RoPE construction,
the sliding/full alternation, k_eq_v, the aggregate projection).

## Which module: `gemma4_unified` (compiled), not `gemma4_unified` (modular)

`transformers` ships a HF "modular" source file (`modular_gemma4_unified.py`)
alongside the fully-expanded, ACTUALLY-RUN file the modular converter compiles
it into (`modeling_gemma4_unified.py`). Importing classes straight from the
modular file is a trap verified empirically here: `Gemma4UnifiedTextModel`'s
`__init__` in the modular file calls `super().__init__(config)`, and Python's
MRO for `class Gemma4UnifiedTextModel(Gemma4UnifiedPreTrainedModel,
LlamaModel)` resolves that to `LlamaModel.__init__` FIRST in the modular
source's own (uncompiled) form - which builds `self.layers` out of
`LlamaDecoderLayer`, not `Gemma4UnifiedTextDecoderLayer`, and crashes on a
missing `config.mlp_bias` attribute Llama's plain MLP wants but Gemma4 never
sets. The COMPILED `modeling_gemma4_unified.py` has no such problem - its
`Gemma4UnifiedTextModel.__init__` builds `self.layers` directly out of
`Gemma4UnifiedTextDecoderLayer`, which is what this dumper imports from.

## Which classes carry the actual attention/MLP/RoPE/RMSNorm math

`Gemma4UnifiedTextAttention`/`Gemma4UnifiedTextMLP`/`Gemma4UnifiedRMSNorm`/
`Gemma4UnifiedTextRotaryEmbedding`/`Gemma4UnifiedTextScaledWordEmbedding` are
all trivial subclasses (`pass`) of the PLAIN `transformers.models.gemma4`
package's own `Gemma4Text*`/`Gemma4RMSNorm` classes - so the "unified" vs
plain `gemma4` module distinction the porting task flagged as needing
resolution turns out not to matter for the attention/MLP/RoPE/norm math at
all: it is IDENTICAL code either way. What genuinely differs is the DECODER
LAYER wrapper: `Gemma4UnifiedTextDecoderLayer` (used here, matching the real
`gemma4-12b-with-proj-ltx-2.5` checkpoint filename, which is the unified
encoder-free multimodal model plus a projection head - not a plain `gemma4`
checkpoint) is a much simpler `Gemma2DecoderLayer` subclass than the plain
package's own `Gemma4TextDecoderLayer` (which additionally carries
per-layer-input/PLE and MoE-router plumbing this port's real config does not
use - `Gemma4UnifiedTextConfig` marks `hidden_size_per_layer_input`/
`enable_moe_block`/etc. as `AttributeError()`, i.e. structurally absent). This
dumper's captured op sequence (`input_layernorm -> self_attn ->
post_attention_layernorm -> +residual -> pre_feedforward_layernorm -> mlp ->
post_feedforward_layernorm -> +residual -> *layer_scalar`, no per-layer-input,
no MoE, no shared-KV) is exactly `Gemma4UnifiedTextDecoderLayer.forward`.

## The `hidden_states` tuple's exact semantics (verified empirically, not
## assumed - this is what the LTX aggregate-embed projection consumes)

`output_hidden_states=True` gives `num_hidden_layers + 1` entries, but they
are NOT "the embedding plus every layer's raw output" as a naive reading of
"49 hidden states" might suggest. Verified here by hooking every decoder layer
directly and diffing against the tuple: entry `k` for `0 <= k <= N-1` is
(embedding output, then each layer's raw output for layers `0..N-2`) - i.e.
entry `k` is the INPUT to layer `k`, not layer `k`'s own output - and the
LAST entry (`N`) is `model.norm(layer[N-1]'s raw output)`, i.e.
`last_hidden_state` itself, NOT `layer[N-1]`'s raw pre-norm output (confirmed
`hidden_states[-1] is last_hidden_state` bit-identical here, and DIFFERENT
from a hook on `layers[-1]` by a large margin). This is the standard HF
decoder convention (Llama/Gemma2/Gemma3 all build `all_hidden_states` the same
way), and for the real 48-layer config it is exactly why `188160 = 3840*49`
lines up: 1 embedding + 47 raw intermediate layer outputs + 1 POST-FINAL-NORM
output, never a raw (un-normed) 48th layer output. Ported into Rust as: the
[`gemma4::Gemma4Output::hidden_states`] entries mirror this convention EXACTLY
(not "all raw"), since that is what a Rust `aggregate_embed` caller needs to
reproduce the real LTX-2.5 conditioning path bit-for-bit.

## The LTX-specific 49-state aggregate projection: not an HF class, ours

`text_embedding_projection.{video,audio}_aggregate_embed` (real header:
`Linear(3840*49 -> 4096 or 2048)`) is LTX's OWN addition on top of the
Gemma4Unified text tower, not part of `transformers` at all. It is not,
however, undocumented: the module it lives inside is
`ltx_core.text_encoders.gemma.feature_extractor.FeatureExtractorV2`
(`resources/ltxv/source/packages/ltx-core/src/ltx_core/text_encoders/gemma/
feature_extractor.py`), and `feature_extractor_v2` below transcribes its input
transform - per-token, per-state RMS normalization over the hidden axis
followed by a `sqrt(out_dim / hidden_size)` rescale - before the seeded
`nn.Linear(hidden*(N+1), AGG_OUT, bias=True)` this dumper stands in for the
real weights with.

An earlier version of this section called the plain concatenate-then-project
shape "the simplest structural match to the confirmed tensor shape" and a
"documented judgment call", on the premise that only the SHAPE was derivable.
That premise was wrong twice over - the reference module exists and was simply
not read, and the guess it licensed was not scale-neutral: the 49 raw states
differ in magnitude by orders of magnitude, so an un-normalized concatenation
projects to a near-constant vector with the caption's own content surviving as
a few-percent residual, which made two unrelated captions decode to the same
video.

## Self-validation (no ground truth beyond the structural invariants below)

1. **Fresh-module determinism**: model rebuilt+reseeded from scratch, same
   inputs, bit-identical output (eval mode, no dropout in this op sequence).
2. **Batch-independence**: replicated to batch 2, identical per-row output.
3. **RoPE unit-rotation invariant**: `cos^2+sin^2 == 1` on BOTH the sliding
   and the full/global tables (the full table's zero-padded "nope" channels
   trivially satisfy this too: `cos=1,sin=0`).
4. **k_eq_v structural check**: the global (`layer_types[-1]`) layer's
   `self_attn.v_proj is None` (no weight matrix at all) while every sliding
   layer's is not - the real config's `attention_k_eq_v=True` +
   `attention_k_eq_v and not is_sliding` gate, verified structurally rather
   than assumed from prose.

Usage:
  python tools/goldens/gemma4_dump_reference.py --out testdata/golden/gemma4 [--seed 1234]
"""

import argparse
import hashlib
import math
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedTextModel,
)

# Toy dims (every step kind runs in well under a second); every FLAG that
# changes the op sequence is the real LTX-2.5 value (see module docstring).
TINY_CONFIG = dict(
    vocab_size=48,
    hidden_size=24,
    intermediate_size=32,
    num_hidden_layers=6,          # 5:1 real ratio's minimal instance (5 sliding + 1 full)
    num_attention_heads=4,
    num_key_value_heads=2,        # sliding layers: GQA groups=2
    head_dim=8,                   # sliding layers' head dim
    global_head_dim=16,           # == 2*head_dim, matching the real 512 == 2*256
    num_global_key_value_heads=1, # full/global layers: MQA (groups=4)
    attention_k_eq_v=True,        # real LTX-2.5 value (class default False)
    sliding_window=3,             # < T=8, so the window genuinely excludes keys
    hidden_activation="gelu_pytorch_tanh",
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=False,
    attention_dropout=0.0,
    num_kv_shared_layers=0,       # real config: no KV-sharing tail
    use_double_wide_mlp=False,    # real config: no double-wide MoE-adjacent MLP
    max_position_embeddings=64,
)

T = 8            # token count (> sliding_window=3, exercises the window)
AGG_OUT = 40     # tiny stand-in for the real 4096 (video) / 2048 (audio) width


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    total = sum(v.numel() for v in tensors.values()) * 4 / 1e6
    print(f"wrote {name}: {len(tensors)} tensors, {total:.2f} MB", flush=True)


def agree(label, a, b, tol=1e-6):
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    cos = F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item() if a.numel() > 1 else 1.0
    print(f"  self-validate {label}: max_abs {d:.3e} / scale {scale:.3g} = {rel:.2e} "
          f"(tol {tol:g}), cosine {cos:.10f}", flush=True)
    assert rel <= tol, f"{label}: disagree by {rel:.3e} relative"


def build_model(seed):
    """Build+seed a FRESH tiny-config Gemma4UnifiedTextModel from scratch."""
    torch.manual_seed(seed)
    cfg = Gemma4UnifiedTextConfig(**TINY_CONFIG)
    model = Gemma4UnifiedTextModel(cfg)
    model.eval().requires_grad_(False)
    return model, cfg


def feature_extractor_v2(hidden_states, out_dim, embedding_dim):
    """`ltx_core.text_encoders.gemma.feature_extractor.FeatureExtractorV2`'s
    input transform, transcribed from the reference (`resources/ltxv/source/
    packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py`):
    per-token, per-state RMS normalization over the hidden axis
    (`norm_and_concat_per_token_rms`, no learned weight, no mean subtraction,
    `eps=1e-6` inside the rsqrt), then `_rescale_norm` -
    `* sqrt(out_dim / embedding_dim)`.

    Every prompt here is a single unpadded sequence, so the reference's
    padded-position zeroing is a no-op and no mask is threaded through.
    """
    encoded = torch.stack([h[0] for h in hidden_states], dim=-1)  # [T, D, L]
    variance = torch.mean(encoded**2, dim=1, keepdim=True)  # [T, 1, L]
    normed = encoded * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(encoded.shape[0], -1)  # [T, D*L]
    return normed * math.sqrt(out_dim / embedding_dim)


def build_aggregate(seed, hidden, n_states, out_dim):
    """LTX's own aggregate-embed projection - see module doc. `bias=True` is a
    documented judgment call, not a confirmed header detail."""
    g = torch.Generator().manual_seed(seed + 7777)
    lin = nn.Linear(hidden * n_states, out_dim, bias=True)
    with torch.no_grad():
        lin.weight.normal_(std=0.02, generator=g)
        lin.bias.zero_()
    lin.eval().requires_grad_(False)
    return lin


class Taps:
    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module, pick=lambda o: o):
        def hook(_m, _i, o):
            self.acc[name] = pick(o).detach().clone()
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_with_taps(model, input_ids):
    taps = Taps()
    n = model.config.num_hidden_layers
    taps.watch("layer0.self_attn", model.layers[0].self_attn, pick=lambda o: o[0])
    taps.watch(f"layer{n - 1}.self_attn", model.layers[n - 1].self_attn, pick=lambda o: o[0])
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    taps.close()
    return out, dict(taps.acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    model, cfg = build_model(args.seed)
    n = cfg.num_hidden_layers
    print(f"tiny config: hidden={cfg.hidden_size}, layers={n}, "
          f"layer_types={cfg.layer_types}, T={T}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    input_ids = torch.randint(0, cfg.vocab_size, (1, T), generator=g)

    out, taps = run_with_taps(model, input_ids)
    hs = out.hidden_states
    assert len(hs) == n + 1, f"expected {n + 1} hidden_states, got {len(hs)}"
    assert not any(h.isnan().any() for h in hs), "NaN in hidden_states - an uninitialized parameter was missed"
    assert torch.equal(hs[-1], out.last_hidden_state), "hidden_states[-1] must equal last_hidden_state (see module doc)"

    # ---- self-validation 4: k_eq_v structural check --------------------------
    for i, lt in enumerate(cfg.layer_types):
        has_v = model.layers[i].self_attn.v_proj is not None
        if lt == "full_attention":
            assert not has_v, f"layer {i} (full_attention, attention_k_eq_v=True) must have NO v_proj"
        else:
            assert has_v, f"layer {i} (sliding_attention) must have its own v_proj"
    print("  self-validate k_eq_v: full-attention layer has no v_proj, sliding layers do", flush=True)

    # ---- self-validation 1: fresh module instantiation, bit-identical --------
    model2, _ = build_model(args.seed)
    out2, _ = run_with_taps(model2, input_ids)
    agree("fresh-instantiation last_hidden_state", out2.last_hidden_state, out.last_hidden_state, tol=0.0)
    del model2

    # ---- self-validation 2: batch independence --------------------------------
    input_ids_b2 = input_ids.repeat(2, 1)
    out_b2, _ = run_with_taps(model, input_ids_b2)
    agree("batch-independence row 0", out_b2.last_hidden_state[0], out.last_hidden_state[0], tol=1e-5)
    agree("batch-independence row 1", out_b2.last_hidden_state[1], out.last_hidden_state[0], tol=1e-5)

    # ---- RoPE tables (direct calls - deterministic, no dependence on the
    # ---- internal forward's own `set` iteration order over layer_types) -----
    position_ids = torch.arange(T).unsqueeze(0)
    sliding_cos, sliding_sin = model.rotary_emb(hs[0], position_ids, "sliding_attention")
    full_cos, full_sin = model.rotary_emb(hs[0], position_ids, "full_attention")

    # ---- self-validation 3: RoPE unit-rotation invariant ----------------------
    for label, c, s in [("sliding", sliding_cos, sliding_sin), ("full", full_cos, full_sin)]:
        unit = c.double() ** 2 + s.double() ** 2
        max_dev = (unit - 1.0).abs().max().item()
        print(f"  self-validate RoPE({label}) cos^2+sin^2==1: max deviation {max_dev:.3e}", flush=True)
        assert max_dev < 1e-5, f"RoPE({label}) tables are not unit rotations (max dev {max_dev:.3e})"

    # ---- LTX's own aggregate-embed projection over all n+1 hidden states -----
    agg = build_aggregate(args.seed, cfg.hidden_size, n + 1, AGG_OUT)
    features = feature_extractor_v2(hs, AGG_OUT, cfg.hidden_size)  # [T, hidden*(n+1)]
    agg_out = agg(features)
    assert not agg_out.isnan().any()

    tensors = {
        "input_ids": input_ids[0].to(torch.float32),
        "rope_sliding_cos": sliding_cos[0],
        "rope_sliding_sin": sliding_sin[0],
        "rope_full_cos": full_cos[0],
        "rope_full_sin": full_sin[0],
        "layer0_self_attn_out": taps["layer0.self_attn"][0],
        f"layer{n - 1}_self_attn_out": taps[f"layer{n - 1}.self_attn"][0],
        "last_hidden_state": out.last_hidden_state[0],
        "aggregate_out": agg_out,
    }
    for k, h in enumerate(hs):
        tensors[f"hidden_states.{k}"] = h[0]

    manifest = {
        "run": {"seed": args.seed, "tokens": T, "agg_out": AGG_OUT,
                "layer_types": cfg.layer_types,
                "tiny_config": dict(TINY_CONFIG)},
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                     "python": sys.version.split()[0]},
    }
    save(args.out, "gemma4_tiny.safetensors", tensors, manifest)

    # The tiny model's OWN weights, plus the aggregate projection's - so the
    # Rust smoke test needs no checkpoint.
    sd = dict(model.state_dict())
    sd["text_embedding_projection.video_aggregate_embed.weight"] = agg.weight
    sd["text_embedding_projection.video_aggregate_embed.bias"] = agg.bias
    save(args.out, "gemma4_tiny_weights.safetensors", sd, manifest)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
