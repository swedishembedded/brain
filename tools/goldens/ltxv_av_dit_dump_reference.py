#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 AUDIO+VIDEO DiT stream reference goldens (tiny random weights).

Sibling of `ltxv_dit_dump_reference.py` (that script and its golden output
are NOT modified - `crates/ltxv/tests/dit_parity.rs` depends on them
unchanged). Runs the OFFICIAL `ltx_core.model.transformer.model.LTXModel`
(CPU, fp32), built with `model_type=LTXModelType.AudioVideo` (the same
construction `LTXModelConfigurator` uses for the real 22B+audio checkpoint -
see that class in `ltx_core/model/transformer/model_configurator.py`), at a
TINY random-weight config, and dumps every boundary a Rust unit-by-unit
replay needs: both streams' RoPE tables (self-attention AND the shared
cross-modal table), both streams' adaLN-single raw tables, the four new
per-block AV adaLN tables, the A2V/V2A attention outputs, both streams'
block-0 internal taps, and both final outputs.

## Why AudioVideo, tiny dims, but REAL config VALUES

Same policy as the video-only dumper: every FLAG that changes the op
sequence is set to the real LTX-2.5 value - `cross_attention_adaln=True`,
`use_prompt_adaln_single=False`, `ff_bias=False`/`audio_ff_bias=False`,
`use_keyframes_abs_pos_embedding=True`. The reference `LTXModel` class does
not expose `use_audio_video_cross_attention`/`av_cross_ada_norm` as
constructor knobs at all (only `LTXModelConfigurator`'s `check_config_value`
asserts every real checkpoint satisfies them) - building with
`model_type=AudioVideo` always runs the full bidirectional cross-attention
this milestone implements, there is no "off" path.

`audio_ff_bias` has no independently-verified real-checkpoint value on this
port's roadmap ledger (only video's own `ff_bias=false` is confirmed) - set
to `False` here as the consistent assumption, same judgment-call category as
every other "not independently checkpoint-verified, but the analogous real
value" choice this port's goldens make; flagged here rather than silently
assumed.

## Audio stream tiny dims

Audio's `inner_dim` is HALF video's tiny `inner_dim` (32 vs 64) - "audio
proportionally narrower than video", matching the real checkpoint's own
64-vs-128-per-head-dim ratio - but the SAME head COUNT (4) as video, which
is what the real config does too (32 heads both streams) and what keeps the
shared cross-modal RoPE table's per-head split consistent regardless of
which stream's preprocessor built it (see `rope.py`'s `split_freqs_cis` -
`num_attention_heads` differs between the video and audio preprocessors in
general, but must agree with whichever stream's geometry the actual
`Attention` module was built at; equal head counts sidesteps that
entirely - see `crates/ltxv/src/config.rs::LtxAvDitConfig::assert_supported`
for the same invariant on the Rust side).

`av_ca_timestep_scale_multiplier` is set to a NON-1 value (not the class
default `1`, and not the unverified-on-this-port's-roadmap real value
`1000.0`) so a Rust implementation that hardcodes either number instead of
reading the config would fail the parity test loudly.

## Self-validation inside the dumper

Same three checks as the video-only dumper (fresh-module determinism,
batch-independence, RoPE unit-rotation) - the last one now covers FOUR
captured `(cos,sin)` table pairs, not one: each stream's own self-attention
RoPE and the shared cross-modal RoPE (built independently per stream, at
`audio_cross_attention_dim`, from each stream's own time-axis positions).

Usage:
  python tools/goldens/ltxv_av_dit_dump_reference.py --out testdata/golden/ltxv/av_dit [--seed 1234]
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

# Toy dims (every step kind runs in well under a second); every FLAG is the
# real LTX-2.5 value (see module docstring). Video half matches the
# video-only dumper's TINY_CONFIG exactly (so both goldens describe the same
# video geometry); audio half is proportionally narrower, same head count.
TINY_CONFIG = dict(
    model_type=LTXModelType.AudioVideo,
    num_attention_heads=4,
    attention_head_dim=16,           # video inner_dim = 64
    in_channels=128,
    out_channels=128,
    num_layers=2,
    cross_attention_dim=64,          # == video inner_dim, see video-only dumper's docstring
    norm_eps=1e-6,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    apply_gated_attention=False,
    caption_projection=None,
    cross_attention_adaln=True,
    use_prompt_adaln_single=False,
    ff_bias=False,
    use_keyframes_abs_pos_embedding=True,
    # ---- audio half ----
    audio_num_attention_heads=4,     # SAME head count as video - see module docstring
    audio_attention_head_dim=8,      # audio inner_dim = 32 (half video's 64)
    audio_in_channels=128,
    audio_out_channels=128,
    audio_cross_attention_dim=32,    # == audio inner_dim, same judgment call as video's own
    audio_positional_embedding_max_pos=[20],
    audio_ff_bias=False,             # assumed symmetric with video's real value - see module docstring
    av_ca_timestep_scale_multiplier=3,  # non-1 on purpose, see module docstring
    audio_caption_projection=None,
)

GRID = (2, 2, 2)      # (frames, height, width) video latent-token grid -> Tv = 8 tokens
CONTEXT_LEN = 6        # video fake text token count
T_AUDIO = 5             # audio latent-token count (distinct from Tv, catches T mixups)
AUDIO_CONTEXT_LEN = 4   # audio fake text token count (distinct from CONTEXT_LEN)
VIDEO_SIGMA = 0.7
AUDIO_SIGMA = 0.35      # deliberately DIFFERENT from VIDEO_SIGMA - see module docstring


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


def build_model(seed):
    """Build+seed a FRESH tiny-config AudioVideo LTXModel from scratch."""
    torch.manual_seed(seed)
    model = LTXModel(**TINY_CONFIG)
    g = torch.Generator().manual_seed(seed + 999)
    reinit = 0
    for name, p in model.named_parameters():
        if "scale_shift_table" in name or name == "keyframes_abs_pos_embedding":
            torch.nn.init.normal_(p, std=0.02, generator=g)
            reinit += 1
    assert reinit > 0, "no torch.empty(...)-sourced parameters found to re-initialize - has the class changed?"
    model.eval().requires_grad_(False)
    return model


def det_video_modality(seed, grid, context_len, inner_dim, in_channels, sigma_value, keyframe_token=0):
    """Deterministic video Modality - same construction as the video-only
    dumper's `det_video_modality` (see that module's docstring for why this
    IS pipeline-shaped input, not hand-assembled)."""
    g = torch.Generator().manual_seed(seed)
    f, h, w = grid
    t = f * h * w
    b = 1

    latent = torch.randn(b, t, in_channels, generator=g)

    sigma = torch.tensor([sigma_value])
    denoise_mask = torch.ones(b, t, 1)
    timesteps = denoise_mask * sigma.view(-1, 1, 1)

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


def det_audio_modality(seed, t_audio, context_len, inner_dim, in_channels, sigma_value):
    """Deterministic audio Modality - single (time-only) position axis,
    n_pos_dims=1, `positions` shape `(B, 1, T, 2)` - see `Modality`'s own
    docstring."""
    g = torch.Generator().manual_seed(seed)
    b = 1

    latent = torch.randn(b, t_audio, in_channels, generator=g)

    sigma = torch.tensor([sigma_value])
    denoise_mask = torch.ones(b, t_audio, 1)
    timesteps = denoise_mask * sigma.view(-1, 1, 1)

    idx = torch.arange(t_audio, dtype=torch.float32)
    starts = idx.view(1, t_audio)         # (1, T) - one axis
    ends = starts + 1.0
    bounds = torch.stack([starts, ends], dim=-1)   # (1, T, 2)
    positions = einops.repeat(bounds, "c t bounds -> bs c t bounds", bs=b)

    context = 0.5 * torch.randn(b, context_len, inner_dim, generator=g)

    return Modality(latent=latent, sigma=sigma, timesteps=timesteps, positions=positions,
                    context=context, keyframes_mask=None)


def run_with_taps(model, video, audio):
    """One forward, with every boundary tapped via real hooks (+ monkeypatch
    of the plain `precompute_freqs_cis` function, which is not an nn.Module
    and so cannot be forward-hooked - restored immediately after, and is the
    SAME function the model calls, not a reimplementation).

    Four `precompute_freqs_cis` calls happen per AV forward, in this fixed
    order (`LTXModel.forward` -> `video_args_preprocessor.prepare(video,
    audio)` then `audio_args_preprocessor.prepare(audio, video)`, each of
    which computes its OWN self-attention pe first, then its cross pe -
    `MultiModalTransformerArgsPreprocessor.prepare`): [video_self,
    video_cross, audio_self, audio_cross].
    """
    captured_rope = []
    orig_precompute = transformer_args_mod.precompute_freqs_cis

    def _capture_rope(*args, **kwargs):
        result = orig_precompute(*args, **kwargs)
        captured_rope.append(result)
        return result

    transformer_args_mod.precompute_freqs_cis = _capture_rope
    try:
        taps = Taps()
        taps.watch("adaln_single", model.adaln_single)
        taps.watch("audio_adaln_single", model.audio_adaln_single)
        taps.watch("av_ca_video_scale_shift_adaln_single", model.av_ca_video_scale_shift_adaln_single)
        taps.watch("av_ca_audio_scale_shift_adaln_single", model.av_ca_audio_scale_shift_adaln_single)
        taps.watch("av_ca_a2v_gate_adaln_single", model.av_ca_a2v_gate_adaln_single)
        taps.watch("av_ca_v2a_gate_adaln_single", model.av_ca_v2a_gate_adaln_single)
        for i, block in enumerate(model.transformer_blocks):
            taps.watch(f"block.{i}", block, pick=lambda o: (o[0].x, o[1].x))
        b0 = model.transformer_blocks[0]
        taps.watch("b0.attn1", b0.attn1)
        taps.watch("b0.attn2", b0.attn2)
        taps.watch("b0.ff", b0.ff)
        taps.watch("b0.audio_attn1", b0.audio_attn1)
        taps.watch("b0.audio_attn2", b0.audio_attn2)
        taps.watch("b0.audio_ff", b0.audio_ff)
        taps.watch("b0.audio_to_video_attn", b0.audio_to_video_attn)
        taps.watch("b0.video_to_audio_attn", b0.video_to_audio_attn)

        out_v, out_a = model(video=video, audio=audio, perturbations=None)
    finally:
        transformer_args_mod.precompute_freqs_cis = orig_precompute
    taps.close()

    assert len(captured_rope) == 4, f"expected 4 precompute_freqs_cis calls, got {len(captured_rope)}"
    return out_v, out_a, dict(taps.acc), captured_rope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    model = build_model(args.seed)
    video = det_video_modality(args.seed, GRID, CONTEXT_LEN, model.inner_dim, TINY_CONFIG["in_channels"], VIDEO_SIGMA)
    audio = det_audio_modality(args.seed + 1, T_AUDIO, AUDIO_CONTEXT_LEN, model.audio_inner_dim, TINY_CONFIG["audio_in_channels"], AUDIO_SIGMA)
    f, h, w = GRID
    tokens = f * h * w
    print(f"tiny AV config: video inner_dim={model.inner_dim}, audio inner_dim={model.audio_inner_dim}, "
          f"layers={TINY_CONFIG['num_layers']}, video grid={GRID} -> {tokens} tokens, audio tokens={T_AUDIO}", flush=True)

    out_v, out_a, taps, rope_calls = run_with_taps(model, video, audio)
    (v_rope_cos, v_rope_sin), (v_cross_cos, v_cross_sin), (a_rope_cos, a_rope_sin), (a_cross_cos, a_cross_sin) = rope_calls
    assert not out_v.isnan().any(), "video output contains NaN - an uninitialized parameter was missed"
    assert not out_a.isnan().any(), "audio output contains NaN - an uninitialized parameter was missed"

    # ---- self-validation 1: fresh module instantiation, bit-identical ------
    model2 = build_model(args.seed)
    out_v2, out_a2, _, _ = run_with_taps(model2, video, audio)
    agree("fresh-instantiation video output", out_v2, out_v, tol=0.0)
    agree("fresh-instantiation audio output", out_a2, out_a, tol=0.0)
    del model2

    # ---- self-validation 2: batch independence ------------------------------
    video_b2 = Modality(
        latent=video.latent.repeat(2, 1, 1), sigma=video.sigma.repeat(2),
        timesteps=video.timesteps.repeat(2, 1, 1), positions=video.positions.repeat(2, 1, 1, 1),
        context=video.context.repeat(2, 1, 1), keyframes_mask=video.keyframes_mask.repeat(2, 1, 1))
    audio_b2 = Modality(
        latent=audio.latent.repeat(2, 1, 1), sigma=audio.sigma.repeat(2),
        timesteps=audio.timesteps.repeat(2, 1, 1), positions=audio.positions.repeat(2, 1, 1, 1),
        context=audio.context.repeat(2, 1, 1), keyframes_mask=None)
    # tol is fp32 batched-matmul reassociation, not semantics - see the
    # video-only dumper's `agree` docstring for the same phenomenon.
    out_v_b2, out_a_b2, _, _ = run_with_taps(model, video_b2, audio_b2)
    agree("batch-independence video row 0", out_v_b2[0], out_v[0], tol=1e-5)
    agree("batch-independence video row 1", out_v_b2[1], out_v[0], tol=1e-5)
    agree("batch-independence audio row 0", out_a_b2[0], out_a[0], tol=1e-5)
    agree("batch-independence audio row 1", out_a_b2[1], out_a[0], tol=1e-5)

    # ---- self-validation 3: RoPE unit-rotation invariant (all 4 tables) ----
    for label, (c, s) in (("video self", (v_rope_cos, v_rope_sin)), ("video cross", (v_cross_cos, v_cross_sin)),
                          ("audio self", (a_rope_cos, a_rope_sin)), ("audio cross", (a_cross_cos, a_cross_sin))):
        unit = c.double() ** 2 + s.double() ** 2
        max_dev = (unit - 1.0).abs().max().item()
        print(f"  self-validate RoPE cos^2+sin^2==1 [{label}]: max deviation {max_dev:.3e}", flush=True)
        assert max_dev < 1e-5, f"RoPE table [{label}] is not a unit rotation (max dev {max_dev:.3e})"

    tensors = {
        "video.latent": video.latent[0],
        "video.context": video.context[0],
        "video.timesteps": video.timesteps[0],
        "video.positions": video.positions[0],
        "video.keyframes_mask": video.keyframes_mask[0],
        "video.sigma": video.sigma,
        "audio.latent": audio.latent[0],
        "audio.context": audio.context[0],
        "audio.timesteps": audio.timesteps[0],
        "audio.positions": audio.positions[0],
        "audio.sigma": audio.sigma,
        "video.rope_cos": v_rope_cos[0],
        "video.rope_sin": v_rope_sin[0],
        "video.cross_rope_cos": v_cross_cos[0],
        "video.cross_rope_sin": v_cross_sin[0],
        "audio.rope_cos": a_rope_cos[0],
        "audio.rope_sin": a_rope_sin[0],
        "audio.cross_rope_cos": a_cross_cos[0],
        "audio.cross_rope_sin": a_cross_sin[0],
        "video.adaln_table": taps["adaln_single"][0],
        "video.embedded_timestep": taps["adaln_single"][1],
        "audio.adaln_table": taps["audio_adaln_single"][0],
        "audio.embedded_timestep": taps["audio_adaln_single"][1],
        "av.video_ss_table": taps["av_ca_video_scale_shift_adaln_single"][0],
        "av.audio_ss_table": taps["av_ca_audio_scale_shift_adaln_single"][0],
        "av.a2v_gate_table": taps["av_ca_a2v_gate_adaln_single"][0],
        "av.v2a_gate_table": taps["av_ca_v2a_gate_adaln_single"][0],
        "video.b0_attn1_out": taps["b0.attn1"][0],
        "video.b0_attn2_out": taps["b0.attn2"][0],
        "video.b0_ff_out": taps["b0.ff"][0],
        "audio.b0_attn1_out": taps["b0.audio_attn1"][0],
        "audio.b0_attn2_out": taps["b0.audio_attn2"][0],
        "audio.b0_ff_out": taps["b0.audio_ff"][0],
        "av.b0_a2v_out": taps["b0.audio_to_video_attn"][0],
        "av.b0_v2a_out": taps["b0.video_to_audio_attn"][0],
        "video.out": out_v[0],
        "audio.out": out_a[0],
    }
    for i in range(TINY_CONFIG["num_layers"]):
        vx_i, ax_i = taps[f"block.{i}"]
        tensors[f"video.block.{i}.out"] = vx_i[0]
        tensors[f"audio.block.{i}.out"] = ax_i[0]
        block = model.transformer_blocks[i]
        tensors[f"video.block.{i}.scale_shift_table"] = block.scale_shift_table
        tensors[f"video.block.{i}.prompt_scale_shift_table"] = block.prompt_scale_shift_table
        tensors[f"audio.block.{i}.scale_shift_table"] = block.audio_scale_shift_table
        tensors[f"audio.block.{i}.prompt_scale_shift_table"] = block.audio_prompt_scale_shift_table
        tensors[f"av.block.{i}.scale_shift_table_a2v_ca_video"] = block.scale_shift_table_a2v_ca_video
        tensors[f"av.block.{i}.scale_shift_table_a2v_ca_audio"] = block.scale_shift_table_a2v_ca_audio

    manifest = {
        "run": {"seed": args.seed, "grid": list(GRID), "video_tokens": tokens, "context_len": CONTEXT_LEN,
                "audio_tokens": T_AUDIO, "audio_context_len": AUDIO_CONTEXT_LEN,
                "video_sigma": VIDEO_SIGMA, "audio_sigma": AUDIO_SIGMA,
                "tiny_config": {k: (v.value if hasattr(v, "value") else v) for k, v in TINY_CONFIG.items()
                               if k not in ("caption_projection", "audio_caption_projection")}},
        "versions": {"torch": torch.__version__, "einops": einops.__version__,
                     "python": sys.version.split()[0]},
    }
    save(args.out, "av_dit_tiny.safetensors", tensors, manifest)

    # The tiny model's OWN weights, so the Rust smoke test needs no checkpoint.
    sd = dict(model.state_dict())
    save(args.out, "av_dit_tiny_weights.safetensors", sd, manifest)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
