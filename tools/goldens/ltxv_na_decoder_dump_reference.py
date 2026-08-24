#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 NA ("diffusion") video VAE decoder reference goldens.

Runs the OFFICIAL `ltx_core.model.video_vae.diffusion_video_decoder.
DiffusionVideoDecoder` (CPU, fp32) with the REAL
`ltx-2.5-video-vae-bf16.safetensors` weights (the file paired with the NA
decoder - `config.vae._class_name == "CausalDiffusionVAE"`, NOT the
`-conv-` file the M2 conv-decoder milestone uses) and dumps two independent
taps:

  na_context.safetensors   a small NORMALIZED latent -> stages 1-3 (stage-4
                            INPUT feature) -> stage 4 (the final context) -
                            real weights, real op sequence, at the smallest
                            (T,H,W) stage-0's own kernel allows.
  na_diff.safetensors      ONE stage-5 forward (`forward_diff_step` at
                            t=1.0) over a DECOUPLED synthetic context + noised
                            pixels (not chained from na_context - see below),
                            at the checkpoint's real `default_num_inference_
                            steps=1` / `model_output_type="x0"`, which
                            collapses the usual multi-step Euler sampling
                            loop to exactly this one forward (see the
                            `--diff-only` branch's comment).
  manifest.json             shapes, sha256, run params, self-validation notes.

## Why the context tap and the diffusion tap use DIFFERENT (T,H,W)

Stage 0's NA kernel is `(3,7,7)`, so ANY latent this decoder can run at all
needs `H,W>=7` - and since every upsample stage multiplies H,W by a FIXED
total factor of 8 (independent of the starting size), the real context this
produces is already `(17,56,56)` at `head_dim=64`/4 heads before stage 5
even starts (measured: `eager_na3d` at that exact shape/kernel(11,11,11)
runs in ~4s in pure Python - not slow to dump). But `crates/kernels/wgsl/
na3d_scores.wgsl`'s naive gather-then-dense kernel dispatches
`heads*nq*window` THREADS (`4*53312*1331` ~= 284M for that one tap) - fine
for a one-off Python reference run, excessive for a Rust test fixture this
port's own test suite re-runs routinely. So the diffusion-stage tap uses a
SEPARATE, DECOUPLED, smaller synthetic context (`T=H=W=13`, chosen `>
stage5_kernel=11` so the NATTEN border-shift is actually exercised, not the
trivial `window==volume` case) rather than literally chaining na_context's
own (17,56,56) output - real weights either way, per this port's own
established "real weights + a fresh deterministic synthetic input" pattern
(`ltxv_dit_dump_reference.py`'s tiny-config taps, `ltxv_vae_dump_reference.
py`'s `det_video`). `na_context`'s own real-weight parity is a SEPARATE,
independent gate.

## Weight loading

`ltx_core.model.video_vae.model_configurator.video_decoder_sd_ops_for_
checkpoint(weights_path)` is the production loader for this checkpoint
family (conv OR diffusion, auto-detected from `config.vae._class_name`) -
for a diffusion checkpoint it pre-reads any `*.gate_msa`/`*.gate_mlp`/
`*.gate_ctx` sibling tensors and folds them into `attn.proj`/`mlp.w_down`/
`context_proj` (`_fold_gate_into_linear`), passing every OTHER tensor
through unchanged. **This checkpoint has zero such gate tensors** (verified
directly against the raw header below, not assumed) - self-validation 3
below confirms `_read_diff_vae_gates` returns an empty dict for this exact
file, which means the fold path never fires: this checkpoint's
`scale_shift_table` is the ONLY per-block modulation source, and
`DiffusionNABlock._modulation` already only reads 4 of its 7 rows (scale_
msa, shift_msa, scale_mlp, shift_mlp - see `crates/ltxv/src/na_decoder.rs`'s
module doc for the full "gate chunks are computed-then-discarded, not
folded" chain of evidence). So brain's own importer needs no analogous fold
step - this checkpoint simply never exercises it.

Usage:
  python tools/goldens/ltxv_na_decoder_dump_reference.py \\
      --weights /path/to/ltx-2.5-video-vae-bf16.safetensors \\
      --out testdata/golden/ltxv/na_decoder [--seed 42]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# `LTXV_REFERENCE_ROOT` overrides for a checkout elsewhere; the default is
# repo-relative (`scratchpad/reference/ltxv/`, gitignored), never a
# machine-specific absolute path.
_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
from ltx_core.model.video_vae.model_configurator import (  # noqa: E402
    VideoDecoderConfigurator,
    _read_diff_vae_gates,
    video_decoder_sd_ops_for_checkpoint,
)
from ltx_core.model.video_vae.transformer import CombinedDiffusionNABlock  # noqa: E402
from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D  # noqa: E402
from ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock  # noqa: E402
from ltx_core.model.video_vae.transformer.fallback_na import EagerSdpaAttention  # noqa: E402
from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d as eager_na3d  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

_DIFF_CLASS_NAME = "CausalDiffusionVAE"


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                       "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}"
                                         for k, v in sorted(tensors.items())), flush=True)


def agree(name, a, b, tol=1e-6):
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    print(f"  self-validate {name}: max abs {d:.3e} / scale {scale:.3g} "
          f"= {rel:.2e} (tol {tol:g})", flush=True)
    assert rel <= tol, f"{name}: disagree by {rel:.3e} relative"
    return d


def build_decoder(weights_path):
    """Build+load a FRESH decoder from the real checkpoint, installed as the
    COMBINED pathway (`CombinedDiffusionNABlock`, full-volume attention,
    `w_chunks=1` - what `DiffVAEMode.COMBINED_COMPILE` selects in
    production, minus `torch.compile`) with the eager tiled-SDPA NA backend
    (no `natten`/Triton dependency needed - this IS this port's
    correctness oracle, not a fallback being tolerated).
    """
    loader = SafetensorsModelStateDictLoader()
    metadata = loader.metadata(weights_path)
    class_name = metadata.get("config", {}).get("vae", {}).get("_class_name")
    assert class_name == _DIFF_CLASS_NAME, (
        f"expected the diffusion-decoder-bundled checkpoint (_class_name={_DIFF_CLASS_NAME!r}), "
        f"got {class_name!r} - use ltx-2.5-video-vae-bf16.safetensors, not the "
        f"conv-decoder-paired file, for this milestone"
    )

    decoder = VideoDecoderConfigurator.from_metadata(metadata)
    assert decoder.__class__.__name__ == "DiffusionVideoDecoder", decoder.__class__.__name__
    assert decoder.stage5_kernel == (11, 11, 11), decoder.stage5_kernel
    assert tuple(decoder.stage_channels) == (2048, 1024, 512, 512, 256), decoder.stage_channels
    assert tuple(decoder.stage_depths) == (4, 6, 4, 2, 8), decoder.stage_depths
    assert decoder.model_output_type == "x0", decoder.model_output_type
    assert decoder.default_inference_timesteps.numel() == 1, decoder.default_inference_timesteps
    assert float(decoder.default_inference_timesteps[0]) == 1.0, decoder.default_inference_timesteps
    assert decoder.timestep_scale_multiplier == 1000.0, decoder.timestep_scale_multiplier

    sd_ops = video_decoder_sd_ops_for_checkpoint(weights_path, diffusion_vae=True)
    sd = loader.load(weights_path, sd_ops)
    # "type_emb" is a real tensor in the checkpoint with ZERO references
    # anywhere in the reference source tree (grepped, not assumed) - dead
    # weight `DiffusionVideoDecoder` never registers a matching parameter/
    # buffer for, confirmed empirically: `load_state_dict(strict=True)`
    # raises "Unexpected key(s): type_emb" without this filter. Same
    # treatment brain's own importer gives it (`na_decoder.rs`'s doc).
    state = {k: v.to(torch.float32) for k, v in sd.sd.items() if k != "type_emb"}
    decoder.load_state_dict(state, strict=True)

    for block in decoder.diff_blocks:
        assert isinstance(block, DiffusionNABlock)
        block.__class__ = CombinedDiffusionNABlock
    for module in decoder.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.attention_function = EagerSdpaAttention()
            module.natten_backend = None

    decoder.eval().requires_grad_(False)
    return decoder


def run_context(decoder, t, h, w, seed):
    """Stages 1-4 on a small deterministic NORMALIZED latent."""
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(1, 128, t, h, w, generator=g)

    stage4_input = decoder.forward_stages_1_to_3(latent, drop_leading_frame=True)
    context = decoder.forward_stage_4(stage4_input, drop_leading_frame=True, pad_trailing=False)
    return latent, stage4_input, context


def run_diff(decoder, t, h, w, seed):
    """ONE stage-5 forward on a decoupled synthetic context + noised pixels."""
    g = torch.Generator().manual_seed(seed)
    context = torch.randn(1, t, h, w, decoder.context_channels, generator=g) * 0.5
    p = decoder.patch_size
    x_t = torch.randn(1, 3, t, h * p, w * p, generator=g)

    context_and_x = decoder._context_and_x_for_diff_step(context, x_t)  # noqa: SLF001
    t_now = decoder.default_inference_timesteps[-1:].expand(1)
    model_out = decoder.forward_diff_step(context_and_x, t_now)
    assert decoder.model_output_type == "x0"
    x0_pred = model_out  # single-step x0: the model output IS the final prediction
    return context, x_t, t_now, x0_pred


def bruteforce_na3d(q, k, v, kernel_size):
    """Second, independent derivation of NATTEN windowed attention: plain
    per-query Python loops (no tiling/grouping/batching machinery at all),
    used only to self-validate `eager_na3d` on a small volume - NOT called
    on any of the real dumped taps (too slow at real sizes by design, which
    is exactly why it is trustworthy: it cannot share a bug with the tiled/
    grouped/batched production fallback it is checking).
    """
    b, t, h, w, nh, hd = q.shape
    assert b == 1
    kt, kh, kw = kernel_size
    out = torch.zeros_like(q)

    def bounds(length, kernel, i):
        half = kernel // 2
        lo = length - kernel
        start = min(max(i - half, 0), lo)
        return start, start + kernel

    for qt in range(t):
        st0, en0 = bounds(t, kt, qt)
        for qh in range(h):
            st1, en1 = bounds(h, kh, qh)
            for qw in range(w):
                st2, en2 = bounds(w, kw, qw)
                qq = q[0, qt, qh, qw]  # [nh, hd]
                kk = k[0, st0:en0, st1:en1, st2:en2].reshape(-1, nh, hd)  # [win, nh, hd]
                vv = v[0, st0:en0, st1:en1, st2:en2].reshape(-1, nh, hd)
                # qq: [nh, hd], kk: [win, nh, hd] -> scores [nh, win]
                scores = torch.einsum("hd,whd->hw", qq, kk)
                probs = torch.softmax(scores, dim=-1)
                ctx = torch.einsum("hw,whd->hd", probs, vv)
                out[0, qt, qh, qw] = ctx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ltx-2.5-video-vae-bf16.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    decoder = build_decoder(args.weights)
    print(f"built decoder ({sum(p.numel() for p in decoder.parameters())} params)", flush=True)

    manifest = {
        "run": {"seed": args.seed, "weights": os.path.abspath(args.weights),
                "vae_class": _DIFF_CLASS_NAME,
                "stage_channels": list(decoder.stage_channels),
                "stage_depths": list(decoder.stage_depths),
                "stage5_kernel": list(decoder.stage5_kernel),
                "context_channels": decoder.context_channels,
                "default_num_inference_steps": int(decoder.default_inference_timesteps.numel()),
                "timestep_scale_multiplier": decoder.timestep_scale_multiplier,
                "model_output_type": decoder.model_output_type},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }

    # ---- self-validation 1: this checkpoint has NO gate_* tensors, so the
    # production loader's gate-fold path is empirically a no-op here. -------
    gates = _read_diff_vae_gates(args.weights)
    print(f"  self-validate: _read_diff_vae_gates found {len(gates)} gate tensors "
          f"(expected 0 - this checkpoint's scale_shift_table is the only "
          f"modulation source, no separate gate_msa/gate_mlp/gate_ctx to fold)", flush=True)
    assert len(gates) == 0, f"unexpected gate tensors in this checkpoint: {sorted(gates)}"
    manifest["run"]["gate_tensors_found"] = len(gates)
    # The NA decoder's stage geometry is what fixes every dumped tensor. The
    # per-stage channel/depth lists cannot go in as lists (source_block enforces
    # ints, so the Rust side can compare field by field), so they go in as a
    # count plus their first and last entries - enough that a decoder with a
    # different stage ladder cannot match, without inventing a list comparison
    # the enforcing side does not have.
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        files=[args.weights],
        hash_files=False,
        identity={
            "num_stages": len(decoder.stage_channels),
            "stage_channels_first": int(decoder.stage_channels[0]),
            "stage_channels_last": int(decoder.stage_channels[-1]),
            "stage_depths_first": int(decoder.stage_depths[0]),
            "stage_depths_last": int(decoder.stage_depths[-1]),
            "context_channels": int(decoder.context_channels),
            "default_num_inference_steps": int(decoder.default_inference_timesteps.numel()),
        },
    )

    # ---- self-validation 2: eager_na3d vs an independent brute-force loop,
    # small synthetic volume, nothing to do with the real weights. ----------
    torch.manual_seed(args.seed)
    qs = torch.randn(1, 5, 9, 9, 2, 8)
    ks = torch.randn(1, 5, 9, 9, 2, 8)
    vs = torch.randn(1, 5, 9, 9, 2, 8)
    kernel = (3, 7, 7)
    fast = eager_na3d(qs, ks, vs, kernel_size=kernel, is_causal=None, scale=1.0)
    slow = bruteforce_na3d(qs, ks, vs, kernel)
    agree("eager_na3d vs bruteforce_na3d (NATTEN border semantics)", fast, slow, tol=1e-5)

    # ---- context tap: stages 1-4 -------------------------------------------
    print("\n=== context (stages 1-4) ===", flush=True)
    t0, h0, w0 = 3, 7, 7  # stage-0 kernel (3,7,7) is the hard floor
    latent, stage4_input, context = run_context(decoder, t0, h0, w0, args.seed)
    print(f"  latent {tuple(latent.shape)} -> stage4_input {tuple(stage4_input.shape)} "
          f"-> context {tuple(context.shape)}", flush=True)

    # fresh-instantiation determinism (same convention as the conv-decoder dumper)
    decoder2 = build_decoder(args.weights)
    _, stage4_input2, context2 = run_context(decoder2, t0, h0, w0, args.seed)
    agree("fresh-instantiation stage4_input", stage4_input2, stage4_input, tol=0.0)
    agree("fresh-instantiation context", context2, context, tol=0.0)
    del decoder2

    save(args.out, "na_context.safetensors", {
        "latent": latent[0],
        "stage4_input": stage4_input[0],
        "context": context[0],
    }, manifest)

    # ---- diffusion tap: one stage-5 forward, decoupled synthetic context --
    print("\n=== diffusion (stage 5, single forward, t=1.0, x0) ===", flush=True)
    td, hd_, wd = 13, 13, 13  # > stage5_kernel=11, so border-shift is exercised
    ctx_synth, x_t, t_now, x0_pred = run_diff(decoder, td, hd_, wd, args.seed)
    print(f"  context {tuple(ctx_synth.shape)}, x_t {tuple(x_t.shape)}, t={float(t_now[0])} "
          f"-> x0_pred {tuple(x0_pred.shape)}", flush=True)

    decoder3 = build_decoder(args.weights)
    _, _, _, x0_pred2 = run_diff(decoder3, td, hd_, wd, args.seed)
    agree("fresh-instantiation x0_pred", x0_pred2, x0_pred, tol=0.0)
    del decoder3

    save(args.out, "na_diff.safetensors", {
        "context": ctx_synth[0],
        "x_t": x_t[0],
        "x0_pred": x0_pred[0],
    }, manifest)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
