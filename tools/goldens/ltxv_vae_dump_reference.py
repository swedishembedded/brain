#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 video VAE (encoder + CONV decoder only) reference goldens.

Runs the OFFICIAL `ltx_core.model.video_vae.{video_vae,conv_video_decoder}`
(CPU, fp32) with the REAL `ltx-2.5-video-vae-conv-bf16.safetensors` weights on
a deterministic synthetic clip, and dumps every boundary of the encode/decode
pipeline:

  vae_t{T}.safetensors   video -> per-down-block encoder taps -> moments
                         (mean/logvar) -> normalized latent -> denormalized
                         latent -> per-up-block conv-decoder taps -> recon
  manifest.json          shapes, sha256 per file, run params, versions

This is the M1 (video-first) milestone: the NA "diffusion decoder"
(`DiffusionVideoDecoder`, needs `natten`/a windowed-attention eager fallback)
is explicitly OUT OF SCOPE here and deferred to the DFR milestone. The
checkpoint's embedded metadata confirms this file is the right one to use:
`config.vae._class_name == "CausalVideoAutoencoder"` (the conv-decoder-paired
class), never `CausalDiffusionVAE`.

## Weight loading: the checkpoint's own loader, no hand-rolled remap needed

`ltx_core.model.video_vae.model_configurator` ships `VideoEncoderConfigurator`
/ `VideoDecoderConfigurator` (build the module from the checkpoint's embedded
`config.vae` JSON) and `VAE_ENCODER_COMFY_KEYS_FILTER` /
`VAE_DECODER_COMFY_KEYS_FILTER` (`SDOps` key-rename tables that strip the
checkpoint's bare `encoder.` / `decoder.` prefixes). Loading the real
checkpoint through these two pairs is a `state_dict(strict=True)` load with
ZERO missing/unexpected keys - verified empirically before writing this
dumper, so no manual remap was needed (unlike Wan's diffusers<->native rename).

## Self-validation inside the dumper (no ground truth to compare against, so
## two independent internal checks stand in for it)

1. **Fresh-module determinism**: the encoder and decoder are built and loaded
   a SECOND time from scratch, and the whole encode/decode run is repeated;
   asserted bit-identical to the first run (no dropout/noise is active at
   `timestep_conditioning=False`, so this is a real determinism gate, not a
   near-miss tolerance).
2. **Per-channel-statistics round trip, two independent derivations**: the
   encoder's raw (pre-normalize) mean is available two ways - (a) sliced
   directly off the `conv_out` tap (channels `[:128]`, which is exactly what
   `VideoEncoder.forward`'s uniform-variance expand+chunk dance reduces to),
   and (b) `per_channel_statistics.un_normalize(latent)` applied to the
   dumped normalized latent. These are asserted to agree, and separately
   `normalize(un_normalize(latent)) == latent` is asserted exactly.

Usage:
  python tools/goldens/ltxv_vae_dump_reference.py \\
      --weights /path/to/ltx-2.5-video-vae-conv-bf16.safetensors \\
      --out testdata/golden/ltxv/vae [--frames 9,17 --size 64 --seed 42]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

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

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
from ltx_core.model.video_vae.model_configurator import (  # noqa: E402
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

_CONV_CLASS_NAME = "CausalVideoAutoencoder"


def det_video(t, h, w, seed):
    """Deterministic RGB clip in [-1, 1], shape (1, 3, t, h, w). Mirrors
    `wan_vae_dump_reference.py`'s `det_video` (same construction, no cross-
    architecture meaning intended - just a convenient smooth deterministic
    signal)."""
    g = torch.Generator().manual_seed(seed)
    ts = torch.linspace(0.0, 1.0, t).view(t, 1, 1)
    ys = torch.linspace(0.0, 3.14159, h).view(1, h, 1)
    xs = torch.linspace(0.0, 6.28318, w).view(1, 1, w)
    r = torch.sin(xs + ys + 3.0 * ts)
    gg = torch.cos(2.0 * xs) * torch.sin(0.5 * ys + ts)
    b = 2.0 * (ys / 3.14159) - 1.0 + 0.3 * torch.cos(5.0 * ts)
    v = torch.stack([r, gg.expand(t, h, w), b.expand(t, h, w)], 0)
    v = v + 0.02 * torch.randn(v.shape, generator=g)
    return v.clamp(-1.0, 1.0).unsqueeze(0).contiguous()


def save(out, name, tensors, manifest):
    # everything as f32 - brain's safetensors reader is F32/F16/BF16-only
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
    """Two independently-derived quantities, one assert - relative to scale."""
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    print(f"  self-validate {name}: max abs {d:.3e} / scale {scale:.3g} "
          f"= {rel:.2e} (tol {tol:g})", flush=True)
    assert rel <= tol, f"{name}: disagree by {rel:.3e} relative"
    return d


class Taps:
    """Forward hooks capturing module outputs exactly as the forward made them."""

    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module):
        def hook(_m, _i, o):
            self.acc[name] = o.detach().clone()
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_models(weights_path):
    """Build+load a FRESH encoder/decoder pair from the real checkpoint."""
    loader = SafetensorsModelStateDictLoader()
    metadata = loader.metadata(weights_path)
    class_name = metadata.get("config", {}).get("vae", {}).get("_class_name")
    assert class_name == _CONV_CLASS_NAME, (
        f"expected the conv-decoder-bundled checkpoint (_class_name={_CONV_CLASS_NAME!r}), "
        f"got {class_name!r} - use ltx-2.5-video-vae-conv-bf16.safetensors, not the "
        f"diffusion-decoder-paired file, for the M1 milestone"
    )

    encoder = VideoEncoderConfigurator.from_metadata(metadata)
    decoder = VideoDecoderConfigurator.from_metadata(metadata)
    assert decoder.timestep_conditioning is False, (
        "conv decoder unexpectedly has timestep_conditioning=True - decode() would need a "
        "timestep and the round trip would no longer be deterministic-by-construction"
    )

    sd_enc = loader.load(weights_path, VAE_ENCODER_COMFY_KEYS_FILTER)
    sd_dec = loader.load(weights_path, VAE_DECODER_COMFY_KEYS_FILTER)
    encoder.load_state_dict({k: v.to(torch.float32) for k, v in sd_enc.sd.items()}, strict=True)
    decoder.load_state_dict({k: v.to(torch.float32) for k, v in sd_dec.sd.items()}, strict=True)
    encoder.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    return encoder, decoder


def run_round_trip(encoder, decoder, video):
    """One encode/decode pass with every boundary tapped."""
    etaps = Taps()
    etaps.watch("enc.conv_in", encoder.conv_in)
    for i, block in enumerate(encoder.down_blocks):
        etaps.watch(f"enc.down_blocks.{i}", block)
    etaps.watch("enc.conv_norm_out", encoder.conv_norm_out)
    etaps.watch("enc.conv_out", encoder.conv_out)

    latent = encoder(video)
    etaps.close()
    eacc = dict(etaps.acc)

    # `VideoEncoder.forward`'s uniform-variance expand+chunk always reduces to
    # slicing the raw conv_out output: means = out[:, :latent_channels],
    # logvar = out[:, latent_channels:] (broadcast). See module docstring.
    moments = eacc["enc.conv_out"]
    latent_channels = encoder.latent_channels
    mean_raw = moments[:, :latent_channels]
    logvar_raw = moments[:, latent_channels:]

    dtaps = Taps()
    dtaps.watch("dec.conv_in", decoder.conv_in)
    for i, block in enumerate(decoder.up_blocks):
        dtaps.watch(f"dec.up_blocks.{i}", block)
    dtaps.watch("dec.conv_norm_out", decoder.conv_norm_out)
    dtaps.watch("dec.conv_out", decoder.conv_out)

    recon = decoder(latent)
    dtaps.close()
    dacc = dict(dtaps.acc)

    z_denorm = decoder.per_channel_statistics.un_normalize(latent)

    return {
        "latent": latent, "mean_raw": mean_raw, "logvar_raw": logvar_raw,
        "moments": moments, "z_denorm": z_denorm, "recon": recon,
        "enc_taps": eacc, "dec_taps": dacc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ltx-2.5-video-vae-conv-bf16.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="9,17", help="comma-separated clip lengths (1+8k)")
    ap.add_argument("--size", type=int, default=64, help="H=W, divisible by 32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--taps", action="store_true",
                    help="also dump every per-block activation (self-validation always "
                         "compares the taps in memory either way)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    assert args.size % 32 == 0, f"--size {args.size} must be divisible by 32"

    encoder, decoder = build_models(args.weights)
    print(f"built encoder ({sum(p.numel() for p in encoder.parameters())} params), "
          f"decoder ({sum(p.numel() for p in decoder.parameters())} params)", flush=True)

    manifest = {
        "run": {"seed": args.seed, "size": args.size, "frames": args.frames,
                "weights": os.path.abspath(args.weights),
                "vae_class": _CONV_CLASS_NAME,
                "latent_channels": encoder.latent_channels,
                "video_scale_factors": list(encoder.video_scale_factors),
                "norm_layer": encoder.norm_layer.value,
                "latent_log_var": encoder.latent_log_var.value,
                "decoder_causal": decoder.causal,
                "decoder_timestep_conditioning": decoder.timestep_conditioning},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    # `hash_files=False`: the VAE checkpoint is large and hashing it would
    # dominate a dump that is otherwise a few round trips. The scale factors go
    # in one per axis rather than as a list, because `source_block` enforces
    # ints (a list cannot be compared field-by-field on the Rust side) and it
    # is the (t, h, w) stride that fixes every latent shape here.
    tsf, hsf, wsf = (int(v) for v in encoder.video_scale_factors)
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        files=[args.weights],
        hash_files=False,
        identity={
            "latent_channels": int(encoder.latent_channels),
            "video_scale_factor_t": tsf,
            "video_scale_factor_h": hsf,
            "video_scale_factor_w": wsf,
        },
    )

    for t in [int(v) for v in args.frames.split(",")]:
        assert (t - 1) % 8 == 0, f"{t} frames is not 1+8k"
        print(f"\n=== {t} frames at {args.size}x{args.size} ===", flush=True)
        video = det_video(t, args.size, args.size, args.seed)

        r1 = run_round_trip(encoder, decoder, video)

        # ---- self-validation 1: per-channel-statistics round trip ----------
        un_norm = decoder.per_channel_statistics.un_normalize(r1["latent"])
        agree(f"un_normalize(latent) vs raw mean tap t={t}", un_norm, r1["mean_raw"])
        renorm = decoder.per_channel_statistics.normalize(un_norm)
        agree(f"normalize(un_normalize(latent)) vs latent t={t}", renorm, r1["latent"])

        # ---- self-validation 2: fresh module instantiation, bit-identical --
        encoder2, decoder2 = build_models(args.weights)
        r2 = run_round_trip(encoder2, decoder2, video)
        agree(f"fresh-instantiation latent t={t}", r2["latent"], r1["latent"], tol=0.0)
        agree(f"fresh-instantiation recon t={t}", r2["recon"], r1["recon"], tol=0.0)
        del encoder2, decoder2

        cos = F.cosine_similarity(r1["recon"].flatten().double(),
                                  video.flatten().double(), dim=0).item()
        print(f"  round-trip cosine(recon, video) t={t}: {cos:.6f}", flush=True)

        tensors = {
            "video": video[0],
            "moments": r1["moments"][0],
            "mean": r1["mean_raw"][0],
            "log_var": r1["logvar_raw"][0],
            "latent": r1["latent"][0],
            "z_denorm": r1["z_denorm"][0],
            "recon": r1["recon"][0],
            "recon_clamped": r1["recon"][0].clamp(-1.0, 1.0),
        }
        if args.taps:
            for k, v in r1["enc_taps"].items():
                tensors[f"tap_{k}"] = v[0]
            for k, v in r1["dec_taps"].items():
                tensors[f"tap_{k}"] = v[0]
        save(args.out, f"vae_t{t}.safetensors", tensors, manifest)
        manifest["run"].setdefault("round_trip_cosine", {})[str(t)] = cos

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
