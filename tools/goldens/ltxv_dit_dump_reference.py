#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 video-ONLY DiT stream reference goldens (tiny random weights).

Runs the OFFICIAL `ltx_core.model.transformer.model.LTXModel` (CPU, fp32),
built with `model_type=LTXModelType.VideoOnly` (the same construction
`LTXVideoOnlyModelConfigurator` uses for the real 22B checkpoint - see that
class in `ltx_core/model/transformer/model_configurator.py`), at a TINY
random-weight config (2 layers, inner_dim 64), and dumps every boundary a
Rust unit-by-unit replay needs:

  dit_tiny.safetensors   RoPE tables, adaLN-single raw table, per-block
                         modulation tables, block-0 internal taps
                         (self-attn/cross-attn/FF), every block's output,
                         the final model output, and the tiny model's OWN
                         weights (so the Rust smoke test needs no checkpoint)
  manifest.json           shapes, sha256, run params, config, versions

## Why VideoOnly, tiny dims, but REAL config VALUES

The real 22B transformer (42 GB bf16) and its 12B text encoder cannot run on
modest hardware at all, so this dumps a TOY-WIDTH model - but every FLAG that
changes the op sequence is set to the real LTX-2.5 value (confirmed against
`LTXVideoOnlyModelConfigurator` and the real checkpoint's own metadata, not the
class defaults, which describe the superseded ~19B checkpoints):
`cross_attention_adaln=True`, `use_prompt_adaln_single=False`, `ff_bias=False`,
`use_keyframes_abs_pos_embedding=True`. This is what makes a tiny-config parity
test meaningful even though it can never run the real weights: it proves the OP
SEQUENCE and every CONVENTION (adaLN row order, RoPE layout, gate/shift/scale
indexing) at a size that runs everywhere - the same convention this port uses
for every component too large to validate on real weights on ordinary hardware.

## One judgment call: `caption_projection=None`, fake context already at inner_dim

The real LTX-2.5 checkpoint config sets `caption_proj_before_connector: true`
(see `_build_caption_projections` in `model_configurator.py`), which means NO
`caption_projection` module is built into the transformer at all for 22B/2.5 -
the raw Gemma-4 caption is projected to `inner_dim` upstream by the (out-of-
scope) "embeddings connector". `TransformerArgsPreprocessor._prepare_context`
reshapes `context` to `(B, -1, x.shape[-1]=inner_dim)`, which is only a
semantically correct no-op when the incoming context's last dim already IS
`inner_dim` - true for the real config, and NOT what a literal
`cross_attention_dim=32` (a value smaller than `inner_dim=64`) would give
without a projection module. So this dumper sets `cross_attention_dim=64`
(== `inner_dim`, matching the real invariant empirically verified from
`LTXVideoOnlyModelConfigurator`: `cross_attention_dim` there is `4096`, exactly
`num_attention_heads(32) * attention_head_dim(128)`) and builds the fake text
context tensor directly at that width, with `caption_projection=None` - the
same configuration the real checkpoint uses, just at toy dims.

## Self-validation inside the dumper (no ground truth, so structural invariants
## stand in)

1. **Fresh-module determinism**: model built+seeded a SECOND time from
   scratch, run on the SAME inputs; asserted bit-identical (eval mode, no
   dropout anywhere in this op sequence).
2. **Batch-independence**: the same single sample, replicated to batch 2,
   produces IDENTICAL per-sample output for both batch rows - proves no
   cross-batch leakage (a real bug class: an accidentally-batched norm or a
   RoPE table built from the wrong batch dim).
3. **RoPE unit-rotation invariant**: `cos^2 + sin^2 == 1` elementwise on the
   captured RoPE tables (a property that only holds if the table is genuinely
   `(cos(theta), sin(theta))` for some real `theta`, not corrupted data).

## Judgment call: uninitialized `torch.empty(...)` parameters

`LTXModel`/`BasicAVTransformerBlock` allocate several `scale_shift_table`
parameters as `torch.nn.Parameter(torch.empty(...))` with NO init call
anywhere in the class (real checkpoints always overwrite them on load, so the
class never needed one) - verified empirically: a freshly-constructed model's
`scale_shift_table` contains NaN. This dumper explicitly re-initializes every
`*scale_shift_table` and `keyframes_abs_pos_embedding` parameter with a seeded
`normal_(std=0.02)` after construction, mirroring
`wan_dit_dump_reference.py`'s `build_tiny`'s re-randomization of a
zero-initialized `head.head.weight` for the same reason (an uninitialized or
identically-zero parameter would either crash or make part of the op graph
numerically inert, hiding bugs in a parity test).

Usage:
  python tools/goldens/ltxv_dit_dump_reference.py --out testdata/golden/ltxv/dit [--seed 1234]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

# `LTXV_REFERENCE_ROOT` overrides for a checkout elsewhere; the default is
# repo-relative (`scratchpad/reference/ltxv/`, gitignored), never a
# machine-specific absolute path.
_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

import ltx_core.model.transformer.transformer_args as transformer_args_mod  # noqa: E402
from ltx_core.model.transformer.model import LTXModel, LTXModelType  # noqa: E402
from ltx_core.model.transformer.modality import Modality  # noqa: E402
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

# Toy dims (every step kind runs in well under a second); every FLAG is the
# real LTX-2.5 value (see module docstring). `dim` is a multiple of 64 for
# the same storage-binding-alignment reason `wan_dit_dump_reference.py`'s
# CFG_TINY documents.
TINY_CONFIG = dict(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=4,
    attention_head_dim=16,           # inner_dim = 64
    in_channels=128,                 # real VAE latent channel count
    out_channels=128,
    num_layers=2,
    cross_attention_dim=64,          # == inner_dim, see module docstring
    norm_eps=1e-6,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],   # class default, video
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    apply_gated_attention=False,
    caption_projection=None,          # real 22B config: caption_proj_before_connector=true
    cross_attention_adaln=True,       # real LTX-2.5 value (class default False)
    use_prompt_adaln_single=False,    # real LTX-2.5 value (class default True)
    ff_bias=False,                    # real LTX-2.5 value (class default True)
    use_keyframes_abs_pos_embedding=True,  # real LTX-2.5 value (class default False)
)

GRID = (2, 2, 2)   # (frames, height, width) latent-token grid -> T = 8 tokens
CONTEXT_LEN = 6    # fake text token count

# ---------------------------------------------------------------------------
# Gated-attention + embeddings-connector tiny config (Phase 2) - a SECOND,
# independent dump alongside TINY_CONFIG's (that run/its output files are
# UNCHANGED by this addition - `crates/ltxv/tests/dit_parity.rs`'s existing
# no-gating test keeps replaying `dit_tiny.safetensors` untouched). Every
# dimension differs from TINY_CONFIG's AND from every other dimension in
# THIS config - a degenerate or repeated dim would let a transpose/off-by-one
# bug hide behind a shape that still happens to round-trip - mirrors
# `crates/ltxv/src/config.rs::
# LtxDitConfig::tiny_gated`'s Rust literal exactly, field by field, which is
# what makes this dumper's output a valid golden for that Rust config.
#
# The `Embeddings1DConnector` is NOT part of `LTXModel` in the reference
# (confirmed against `model_configurator.py`: neither configurator ever
# passes a connector into `LTXModel`) - it is a standalone module the real
# pipeline runs on the text encoder's output BEFORE calling `LTXModel.
# forward` at all (`caption_proj_before_connector=True`'s meaning). This
# dumper reproduces that: build the connector, run it on a fake RAW
# pre-connector context, feed its OUTPUT as `Modality.context` to `LTXModel`
# - and separately saves the RAW input + the connector's own weights so the
# Rust test can run ITS OWN connector forward and check it reproduces the
# same output, independent of the surrounding model.
TINY_GATED_CONFIG = dict(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=3,
    attention_head_dim=8,             # inner_dim = 24
    in_channels=128,
    out_channels=128,
    num_layers=2,
    cross_attention_dim=24,           # == inner_dim
    norm_eps=1e-6,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 64, 96],   # every axis distinct
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    apply_gated_attention=True,       # THE flag this dump exists to pin
    caption_projection=None,
    cross_attention_adaln=True,
    use_prompt_adaln_single=False,
    ff_bias=False,
    use_keyframes_abs_pos_embedding=True,
)

GRID_GATED = (2, 3, 5)     # every axis distinct -> T = 30 tokens
CONTEXT_LEN_GATED = 9      # multiple of CONNECTOR_NUM_REGISTERS (3 tiles)

# `video_embeddings_connector`'s own config - matches `LtxDitConfig::
# tiny_gated`'s `connector_*` fields. Head count/dim REVERSED vs. the main
# attention's `3 heads x 8` (`4 heads x 6` here, same product 24) so a
# heads/head_dim transpose between the two attention geometries cannot hide.
CONNECTOR_NUM_ATTENTION_HEADS = 4
CONNECTOR_ATTENTION_HEAD_DIM = 6      # connector inner_dim = 24 (== main inner_dim)
CONNECTOR_NUM_LAYERS = 2
CONNECTOR_NUM_REGISTERS = 3
CONNECTOR_MAX_POS = [50]
CONNECTOR_APPLY_GATED_ATTENTION = True
CONNECTOR_FF_BIAS = True              # class default - real LTX-2.5 value too


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


def _clone(x):
    """Detach+clone a tensor, or every tensor inside a tuple (e.g. a module
    that returns `(a, b)`, like `AdaLayerNormSingle`)."""
    if isinstance(x, tuple):
        return tuple(_clone(v) for v in x)
    return x.detach().clone()


class Taps:
    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module, pick=lambda o: o):
        def hook(_m, _i, o):
            self.acc[name] = _clone(pick(o))
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_model(seed, config=None):
    """Build+seed a FRESH tiny-config VideoOnly LTXModel from scratch.
    `config` defaults to `TINY_CONFIG` - callers dumping a DIFFERENT tiny
    config (e.g. `TINY_GATED_CONFIG` below) pass it explicitly; every
    existing call site is unchanged (positional-only, no second arg), so
    this file's ORIGINAL `dit_tiny.safetensors`/`dit_tiny_weights.safetensors`
    output is byte-identical to before this parameter existed."""
    config = TINY_CONFIG if config is None else config
    torch.manual_seed(seed)
    model = LTXModel(**config)
    g = torch.Generator().manual_seed(seed + 999)
    reinit = 0
    for name, p in model.named_parameters():
        if "scale_shift_table" in name or name == "keyframes_abs_pos_embedding":
            torch.nn.init.normal_(p, std=0.02, generator=g)
            reinit += 1
    assert reinit > 0, "no torch.empty(...)-sourced parameters found to re-initialize - has the class changed?"
    model.eval().requires_grad_(False)
    return model


def det_video_modality(seed, grid, context_len, inner_dim, in_channels, keyframe_token=0):
    """Deterministic video Modality: latent, per-token sigma*mask timesteps,
    RoPE position bounds, and a fake (already-inner_dim, see module docstring)
    text context - all pipeline-shaped, per porting.md section 1 ("do not
    hand-assemble model inputs" - these ARE the inputs a pipeline builds; only
    the surrounding pipeline machinery that derives them from a prompt/video is
    out of scope here).
    """
    g = torch.Generator().manual_seed(seed)
    f, h, w = grid
    t = f * h * w
    b = 1

    latent = torch.randn(b, t, in_channels, generator=g)

    # Per-token timestep = denoise_mask * sigma, shape (B, T, 1) - exactly
    # `ltx_pipelines.utils.helpers.timesteps_from_mask`'s formula (diffusion
    # forcing convention, see module docstring / roadmap doc).
    sigma = torch.tensor([0.7])
    denoise_mask = torch.ones(b, t, 1)
    timesteps = denoise_mask * sigma.view(-1, 1, 1)

    # RoPE position bounds (B, 3, T, 2): [start, end) per patch, patch_size=1
    # in latent-token space - the exact construction
    # `ltx_core.components.patchifiers.VideoLatentPatchifier.get_patch_grid_bounds`
    # uses (meshgrid, indexing="ij", flatten (f h w)).
    grid_coords = torch.meshgrid(torch.arange(f), torch.arange(h), torch.arange(w), indexing="ij")
    starts = torch.stack(grid_coords, dim=0).to(torch.float32)     # (3, f, h, w)
    ends = starts + 1.0
    bounds = torch.stack([starts, ends], dim=-1)                    # (3, f, h, w, 2)
    positions = einops.repeat(bounds, "c f h w bounds -> bs c (f h w) bounds", bs=b)

    context = 0.5 * torch.randn(b, context_len, inner_dim, generator=g)

    keyframes_mask = torch.zeros(b, t, 1)
    keyframes_mask[:, keyframe_token, :] = 1.0

    return Modality(latent=latent, sigma=sigma, timesteps=timesteps, positions=positions,
                    context=context, keyframes_mask=keyframes_mask)


def run_with_taps(model, video):
    """One forward, with every boundary tapped via real hooks (+ one light
    monkeypatch of the plain `precompute_freqs_cis` function call, which is
    not an nn.Module and so cannot be forward-hooked - restored immediately
    after, and is the SAME function the model calls, not a reimplementation).
    """
    captured_rope = {}
    orig_precompute = transformer_args_mod.precompute_freqs_cis

    def _capture_rope(*args, **kwargs):
        result = orig_precompute(*args, **kwargs)
        captured_rope["pe"] = result
        return result

    transformer_args_mod.precompute_freqs_cis = _capture_rope
    try:
        taps = Taps()
        taps.watch("adaln_single", model.adaln_single)
        for i, block in enumerate(model.transformer_blocks):
            taps.watch(f"block.{i}", block, pick=lambda o: o[0].x)
        b0 = model.transformer_blocks[0]
        taps.watch("b0.attn1", b0.attn1)
        taps.watch("b0.attn2", b0.attn2)
        taps.watch("b0.ff", b0.ff)
        taps.watch("norm_out", model.norm_out)
        taps.watch("proj_out", model.proj_out)

        out_v, out_a = model(video=video, audio=None, perturbations=None)
        assert out_a is None
    finally:
        transformer_args_mod.precompute_freqs_cis = orig_precompute
    taps.close()

    return out_v, dict(taps.acc), captured_rope["pe"]


def build_gated_model(seed):
    """Build+seed a FRESH `TINY_GATED_CONFIG` `LTXModel` - same re-init
    dance as [`build_model`], `apply_gated_attention=True` also means every
    attention now carries a freshly-initialized `to_gate_logits` (a plain
    `nn.Linear`, already initialized by its own constructor - no `torch.
    empty(...)` re-init needed for that one)."""
    torch.manual_seed(seed)
    model = LTXModel(**TINY_GATED_CONFIG)
    g = torch.Generator().manual_seed(seed + 999)
    reinit = 0
    for name, p in model.named_parameters():
        if "scale_shift_table" in name or name == "keyframes_abs_pos_embedding":
            torch.nn.init.normal_(p, std=0.02, generator=g)
            reinit += 1
    assert reinit > 0, "no torch.empty(...)-sourced parameters found to re-initialize - has the class changed?"
    model.eval().requires_grad_(False)
    return model


def build_connector(seed):
    """Build+seed a fresh `video_embeddings_connector`-shaped
    `Embeddings1DConnector` - a STANDALONE module, not part of `LTXModel`
    (this file's module-level doc / `TINY_GATED_CONFIG`'s doc)."""
    torch.manual_seed(seed)
    connector = Embeddings1DConnector(
        attention_head_dim=CONNECTOR_ATTENTION_HEAD_DIM,
        num_attention_heads=CONNECTOR_NUM_ATTENTION_HEADS,
        num_layers=CONNECTOR_NUM_LAYERS,
        positional_embedding_max_pos=CONNECTOR_MAX_POS,
        num_learnable_registers=CONNECTOR_NUM_REGISTERS,
        apply_gated_attention=CONNECTOR_APPLY_GATED_ATTENTION,
        ff_bias=CONNECTOR_FF_BIAS,
    )
    connector.eval().requires_grad_(False)
    return connector


def det_video_modality_with_context(seed, grid, context, in_channels, keyframe_token=0):
    """Same construction as [`det_video_modality`], but `context` is
    supplied by the caller (the connector's OWN output, for the gated dump)
    instead of generated here."""
    g = torch.Generator().manual_seed(seed)
    f, h, w = grid
    t = f * h * w
    b = 1

    latent = torch.randn(b, t, in_channels, generator=g)

    sigma = torch.tensor([0.7])
    denoise_mask = torch.ones(b, t, 1)
    timesteps = denoise_mask * sigma.view(-1, 1, 1)

    grid_coords = torch.meshgrid(torch.arange(f), torch.arange(h), torch.arange(w), indexing="ij")
    starts = torch.stack(grid_coords, dim=0).to(torch.float32)
    ends = starts + 1.0
    bounds = torch.stack([starts, ends], dim=-1)
    positions = einops.repeat(bounds, "c f h w bounds -> bs c (f h w) bounds", bs=b)

    keyframes_mask = torch.zeros(b, t, 1)
    keyframes_mask[:, keyframe_token, :] = 1.0

    return Modality(latent=latent, sigma=sigma, timesteps=timesteps, positions=positions,
                    context=context, keyframes_mask=keyframes_mask)


def _additive_mask_from_valid(valid):
    """`valid`: `(S,)` float, `1.0`=keep/`0.0`=padded -> `(1,1,1,S)`
    additive mask `_replace_padded_with_learnable_registers` expects
    (`embeddings_connector.py:139-152`: `binary_mask = mask[:,0,0,:] >= 0`,
    so `0.0` is "valid", any negative value is "padded")."""
    add = torch.where(valid > 0.5, torch.zeros_like(valid), torch.full_like(valid, -1e9))
    return add.view(1, 1, 1, -1)


def dump_gated(args):
    """The gated-attention + embeddings-connector dump - a SECOND,
    independent run alongside [`main`]'s original TINY_CONFIG one (see
    `TINY_GATED_CONFIG`'s module-level doc)."""
    seed = args.seed + 5000  # distinct seed stream from the ungated run

    connector = build_connector(seed)
    f, h, w = GRID_GATED
    tokens = f * h * w
    inner_dim = CONNECTOR_NUM_ATTENTION_HEADS * CONNECTOR_ATTENTION_HEAD_DIM
    assert inner_dim == TINY_GATED_CONFIG["num_attention_heads"] * TINY_GATED_CONFIG["attention_head_dim"], \
        "connector inner_dim must equal the main DiT's inner_dim (this milestone's structural invariant)"

    # RAW pre-connector context - what a (Phase-3, out of scope) real text
    # encoder's caption_projection output would look like. Some positions
    # marked padded (context_valid==0) so the register-substitution path is
    # ACTUALLY exercised, not vacuously true (a real config where every
    # padded pattern happens to land on a multiple of the register count).
    g = torch.Generator().manual_seed(seed + 1)
    raw_context = 0.5 * torch.randn(1, CONTEXT_LEN_GATED, inner_dim, generator=g)
    context_valid = torch.ones(CONTEXT_LEN_GATED)
    for i in range(CONTEXT_LEN_GATED):
        if i % 3 == 2:  # every third position padded - exercises every register row
            context_valid[i] = 0.0
    additive_mask = _additive_mask_from_valid(context_valid)

    connector_out, _ = connector(raw_context, additive_attention_mask=additive_mask)
    assert not connector_out.isnan().any(), "connector output contains NaN"

    # ---- self-validation: fresh connector instantiation, bit-identical -----
    connector2 = build_connector(seed)
    connector_out2, _ = connector2(raw_context, additive_attention_mask=additive_mask)
    agree("connector fresh-instantiation output", connector_out2, connector_out, tol=0.0)
    del connector2

    model = build_gated_model(seed)
    video = det_video_modality_with_context(seed, GRID_GATED, connector_out, TINY_GATED_CONFIG["in_channels"])
    print(f"tiny GATED config: inner_dim={model.inner_dim}, layers={TINY_GATED_CONFIG['num_layers']}, "
          f"grid={GRID_GATED} -> {tokens} tokens, context_len={CONTEXT_LEN_GATED}, "
          f"connector layers={CONNECTOR_NUM_LAYERS} registers={CONNECTOR_NUM_REGISTERS}", flush=True)

    out_v, taps, (rope_cos, rope_sin) = run_with_taps(model, video)
    assert not out_v.isnan().any(), "output contains NaN - an uninitialized parameter was missed"

    # ---- self-validation 1: fresh module instantiation, bit-identical ------
    model2 = build_gated_model(seed)
    out_v2, _, _ = run_with_taps(model2, video)
    agree("gated fresh-instantiation output", out_v2, out_v, tol=0.0)
    del model2

    # ---- self-validation 2: batch independence ------------------------------
    video_b2 = Modality(
        latent=video.latent.repeat(2, 1, 1), sigma=video.sigma.repeat(2),
        timesteps=video.timesteps.repeat(2, 1, 1), positions=video.positions.repeat(2, 1, 1, 1),
        context=video.context.repeat(2, 1, 1), keyframes_mask=video.keyframes_mask.repeat(2, 1, 1))
    out_v_b2, _, _ = run_with_taps(model, video_b2)
    agree("gated batch-independence row 0", out_v_b2[0], out_v[0], tol=1e-5)
    agree("gated batch-independence row 1", out_v_b2[1], out_v[0], tol=1e-5)

    # ---- self-validation 3: RoPE unit-rotation invariant --------------------
    unit = rope_cos.double() ** 2 + rope_sin.double() ** 2
    max_dev = (unit - 1.0).abs().max().item()
    print(f"  self-validate gated RoPE cos^2+sin^2==1: max deviation {max_dev:.3e}", flush=True)
    assert max_dev < 1e-5, f"RoPE tables are not unit rotations (max dev {max_dev:.3e})"

    tensors = {
        "latent": video.latent[0],
        "raw_context": raw_context[0],
        "context_valid": context_valid,
        "connector_out": connector_out[0],
        "timesteps": video.timesteps[0],
        "positions": video.positions[0],
        "keyframes_mask": video.keyframes_mask[0],
        "sigma": video.sigma,
        "rope_cos": rope_cos[0],
        "rope_sin": rope_sin[0],
        "adaln_table": taps["adaln_single"][0],
        "embedded_timestep": taps["adaln_single"][1],
        "b0_attn1_out": taps["b0.attn1"][0],
        "b0_attn2_out": taps["b0.attn2"][0],
        "b0_ff_out": taps["b0.ff"][0],
        "out": out_v[0],
    }
    for i in range(TINY_GATED_CONFIG["num_layers"]):
        tensors[f"block.{i}.out"] = taps[f"block.{i}"][0]

    manifest = {
        "run": {"seed": seed, "grid": list(GRID_GATED), "tokens": tokens, "context_len": CONTEXT_LEN_GATED,
                "connector_num_layers": CONNECTOR_NUM_LAYERS, "connector_num_registers": CONNECTOR_NUM_REGISTERS,
                "connector_num_attention_heads": CONNECTOR_NUM_ATTENTION_HEADS,
                "connector_attention_head_dim": CONNECTOR_ATTENTION_HEAD_DIM,
                "tiny_gated_config": {k: (v.value if hasattr(v, "value") else v) for k, v in TINY_GATED_CONFIG.items()
                                      if k != "caption_projection"}},
        "versions": {"torch": torch.__version__, "einops": einops.__version__,
                     "python": sys.version.split()[0]},
    }
    save(args.out, "dit_tiny_gated.safetensors", tensors, manifest)

    # Model weights + the connector's own weights, prefixed
    # `video_embeddings_connector.` - the SAME canonical name space
    # `crate::dit::push_connector` reads (`crate::dit::av_dit_tensor_
    # manifest`'s doc), so the Rust smoke test needs no checkpoint.
    sd = dict(model.state_dict())
    sd.update({f"video_embeddings_connector.{k}": v for k, v in connector.state_dict().items()})
    save(args.out, "dit_tiny_gated_weights.safetensors", sd, manifest)

    # Tiny random weights, so no upstream artifact to name. `apply_gated_
    # attention` is folded into the identity as 0/1 (source_block enforces
    # ints) because it changes the OP SEQUENCE, not just a shape - it is the
    # entire reason the gated variant is dumped separately, so leaving it out
    # would let the two goldens be swapped without anything noticing.
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        identity={
            "num_attention_heads": TINY_GATED_CONFIG["num_attention_heads"],
            "attention_head_dim": TINY_GATED_CONFIG["attention_head_dim"],
            "in_channels": TINY_GATED_CONFIG["in_channels"],
            "out_channels": TINY_GATED_CONFIG["out_channels"],
            "num_layers": TINY_GATED_CONFIG["num_layers"],
            "cross_attention_dim": TINY_GATED_CONFIG["cross_attention_dim"],
            "apply_gated_attention": int(TINY_GATED_CONFIG["apply_gated_attention"]),
        },
    )

    with open(os.path.join(args.out, "manifest_gated.json"), "w") as mf:
        json.dump(manifest, mf, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest_gated.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    model = build_model(args.seed)
    video = det_video_modality(args.seed, GRID, CONTEXT_LEN, model.inner_dim, TINY_CONFIG["in_channels"])
    f, h, w = GRID
    tokens = f * h * w
    print(f"tiny config: inner_dim={model.inner_dim}, layers={TINY_CONFIG['num_layers']}, "
          f"grid={GRID} -> {tokens} tokens, context_len={CONTEXT_LEN}", flush=True)

    out_v, taps, (rope_cos, rope_sin) = run_with_taps(model, video)
    assert not out_v.isnan().any(), "output contains NaN - an uninitialized parameter was missed"

    # ---- self-validation 1: fresh module instantiation, bit-identical ------
    model2 = build_model(args.seed)
    out_v2, _, _ = run_with_taps(model2, video)
    agree("fresh-instantiation output", out_v2, out_v, tol=0.0)
    del model2

    # ---- self-validation 2: batch independence ------------------------------
    video_b2 = Modality(
        latent=video.latent.repeat(2, 1, 1), sigma=video.sigma.repeat(2),
        timesteps=video.timesteps.repeat(2, 1, 1), positions=video.positions.repeat(2, 1, 1, 1),
        context=video.context.repeat(2, 1, 1), keyframes_mask=video.keyframes_mask.repeat(2, 1, 1))
    # tol is fp32 batched-matmul reassociation (SDPA blocks batch=2 differently
    # than batch=1), not semantics - see `wan_vae_dump_reference.py`'s `agree`
    # docstring for the same phenomenon on a different op.
    out_v_b2, _, _ = run_with_taps(model, video_b2)
    agree("batch-independence row 0", out_v_b2[0], out_v[0], tol=1e-5)
    agree("batch-independence row 1", out_v_b2[1], out_v[0], tol=1e-5)

    # ---- self-validation 3: RoPE unit-rotation invariant --------------------
    unit = rope_cos.double() ** 2 + rope_sin.double() ** 2
    max_dev = (unit - 1.0).abs().max().item()
    print(f"  self-validate RoPE cos^2+sin^2==1: max deviation {max_dev:.3e}", flush=True)
    assert max_dev < 1e-5, f"RoPE tables are not unit rotations (max dev {max_dev:.3e})"

    tensors = {
        "latent": video.latent[0],
        "context": video.context[0],
        "timesteps": video.timesteps[0],
        "positions": video.positions[0],
        "keyframes_mask": video.keyframes_mask[0],
        "sigma": video.sigma,
        "rope_cos": rope_cos[0],
        "rope_sin": rope_sin[0],
        # `adaln_single` is called on the FLATTENED (B*T,) timestep, before the
        # caller's `.view(batch_size, -1, dim)` reshape - since B=1 here, its
        # raw output is already (T, 9*dim) / (T, dim), one row per token.
        "adaln_table": taps["adaln_single"][0],          # (T, 9*dim) raw linear output
        "embedded_timestep": taps["adaln_single"][1],     # (T, dim)
        "scale_shift_table": model.scale_shift_table,          # (2, dim) output-stage table
        "keyframes_abs_pos_embedding": model.keyframes_abs_pos_embedding,
        "b0_attn1_out": taps["b0.attn1"][0],
        "b0_attn2_out": taps["b0.attn2"][0],
        "b0_ff_out": taps["b0.ff"][0],
        "out": out_v[0],
    }
    for i in range(TINY_CONFIG["num_layers"]):
        tensors[f"block.{i}.out"] = taps[f"block.{i}"][0]
        block = model.transformer_blocks[i]
        tensors[f"block.{i}.scale_shift_table"] = block.scale_shift_table
        tensors[f"block.{i}.prompt_scale_shift_table"] = block.prompt_scale_shift_table

    manifest = {
        "run": {"seed": args.seed, "grid": list(GRID), "tokens": tokens, "context_len": CONTEXT_LEN,
                "tiny_config": {k: (v.value if hasattr(v, "value") else v) for k, v in TINY_CONFIG.items()
                               if k != "caption_projection"}},
        "versions": {"torch": torch.__version__, "einops": einops.__version__,
                     "python": sys.version.split()[0]},
    }
    save(args.out, "dit_tiny.safetensors", tensors, manifest)

    # The tiny model's OWN weights, so the Rust smoke test needs no checkpoint.
    sd = dict(model.state_dict())
    save(args.out, "dit_tiny_weights.safetensors", sd, manifest)

    # Tiny random weights, so no upstream artifact to name. `apply_gated_
    # attention` is folded into the identity as 0/1 (source_block enforces
    # ints) because it changes the OP SEQUENCE, not just a shape - it is the
    # entire reason the gated variant is dumped separately, so leaving it out
    # would let the two goldens be swapped without anything noticing.
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        identity={
            "num_attention_heads": TINY_CONFIG["num_attention_heads"],
            "attention_head_dim": TINY_CONFIG["attention_head_dim"],
            "in_channels": TINY_CONFIG["in_channels"],
            "out_channels": TINY_CONFIG["out_channels"],
            "num_layers": TINY_CONFIG["num_layers"],
            "cross_attention_dim": TINY_CONFIG["cross_attention_dim"],
            "apply_gated_attention": int(TINY_CONFIG["apply_gated_attention"]),
        },
    )

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)

    dump_gated(args)


if __name__ == "__main__":
    main()
