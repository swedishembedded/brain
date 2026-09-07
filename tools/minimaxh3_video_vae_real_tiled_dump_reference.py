#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a REAL-WEIGHT tiled-decode parity golden for MiniMax-H3's video VAE.

`minimaxh3_video_vae_dump_reference.py` gates the spatial tiling algorithm
itself (`_split_tiles` / `_blend` / `_stitch_tiles` and the tiled halves of
`_decode_clip` / `_encode_clip`) at a tiny config with random weights, which
is where a layout or blend-formula error shows up most sharply. What it
cannot see is the interaction of that algorithm with the REAL 36-layer ViT
decoder at its real widths - in particular that a real tile really is a
16x16 latent grid and that the decoder's `[-1, 1)`-normalized rotary
coordinates are therefore the ones it was trained on.

So this dumper loads the actual `MiniMaxAI/MiniMax-H3` video VAE and runs
ONE tiled `_decode_clip` over a canvas large enough to need a multi-tile
grid, dumping the input latent and the decoded pixels. It deliberately does
NOT dump weights: the Rust side imports those from the same checkout through
`crate::import::import_video_vae`, so this golden stays small and cannot
drift from the checkpoint the port actually reads.

The latent is derived from the CHECKPOINT's own `latents_mean`/`latents_std`
rather than being unit-normal noise, so the decoder is exercised in the
region of latent space it actually sees at inference. A latent far outside
that region would push the whole network into a regime where any two
implementations agree trivially (saturated) or disagree meaninglessly.

Usage:
  BRAIN_MINIMAXH3_DIR=/path/to/MiniMax-H3 \\
  python3 \\
      tools/minimaxh3_video_vae_real_tiled_dump_reference.py \\
      --out testdata/golden/minimaxh3/video_vae_real_tiled
"""

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/goldens")
from golden_source import source_block  # noqa: E402

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

# A 384x384 canvas is a 24x24 latent grid and a 2x2 tile grid: the smallest
# geometry that exercises every branch of `_stitch_tiles` (a vertical blend,
# a horizontal blend, the corner tile that takes both, and all four trims).
# Bigger would cost real decoder time per tile without reaching a branch this
# does not already reach - the uneven-overlap and >2-tile layouts are gated
# on the tiny config, where a tile is cheap.
HEIGHT = 384
WIDTH = 384
LATENT_FRAMES = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--root", default=os.environ.get("BRAIN_MINIMAXH3_DIR"))
    args = ap.parse_args()
    if not args.root:
        sys.exit("set BRAIN_MINIMAXH3_DIR (or pass --root) to the MiniMax-H3 checkout")
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    vae_dir = os.path.join(args.root, "vae")
    print(f"loading the real video VAE from {vae_dir} (fp32) ...", flush=True)
    model = AutoencoderKLMiniMaxH3.from_pretrained(vae_dir, torch_dtype=torch.float32)
    model.eval()
    print(f"loaded ({sum(p.numel() for p in model.parameters())} params), use_tiling={model.use_tiling}", flush=True)

    ratio = int(model.spatial_compression_ratio)
    assert HEIGHT % ratio == 0 and WIDTH % ratio == 0

    # The trained LayerScale gates must be non-zero, or the whole 36-layer
    # transformer stack is an identity and this golden proves nothing about
    # tiling (see the tiny dumper's own note - they are ZERO at init).
    gate_max = max(float(p.abs().max()) for n, p in model.named_parameters() if n.endswith(("scale1", "scale2")))
    print(f"  trained LayerScale gates: max |scale| {gate_max:.4e}", flush=True)
    assert gate_max > 1e-6, "the checkpoint's LayerScale gates are zero - the ViT decoder would be an identity"

    y_idx, _, y_ov = model._split_tiles(HEIGHT, model.tile_sample_min_height, model.tile_sample_min_overlap_height)
    x_idx, _, x_ov = model._split_tiles(WIDTH, model.tile_sample_min_width, model.tile_sample_min_overlap_width)
    print(f"  tile grid {len(y_idx)}x{len(x_idx)}, height overlaps {y_ov}, width overlaps {x_ov}", flush=True)
    assert len(y_idx) > 1 and len(x_idx) > 1, f"{HEIGHT}x{WIDTH} is not a multi-tile canvas"

    # A latent in the checkpoint's own normalized region, not unit noise.
    mean = torch.tensor(model.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(model.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    g = torch.Generator().manual_seed(args.seed)
    shape = (1, model.config.latent_channels, LATENT_FRAMES, HEIGHT // ratio, WIDTH // ratio)
    z = torch.randn(shape, generator=g, dtype=torch.float32) * std + mean

    model.use_tiling = True
    print("decoding tiled ...", flush=True)
    dec_tiled = model._decode_clip(z)
    print(f"  tap_real_tiled_pixels: {tuple(dec_tiled.shape)}", flush=True)

    # Self-validation: at the REAL weights the tiled and untiled paths must
    # be genuinely different computations, or the Rust rung fed by this
    # golden could pass without tiling being implemented at all.
    model.use_tiling = False
    print("decoding untiled (self-validation only) ...", flush=True)
    dec_untiled = model._decode_clip(z)
    model.use_tiling = True
    d = (dec_tiled.double() - dec_untiled.double()).abs().max().item()
    rel = ((dec_tiled.double() - dec_untiled.double()).norm() / dec_tiled.double().norm()).item()
    print(f"  self-validate tiled vs untiled at real weights: max abs {d:.4e}, rel_l2 {rel:.4e}", flush=True)
    assert d > 1e-2, f"tiled and untiled agree to {d:.3e} at real weights - the golden would be vacuous"

    tensors = {
        "input_z_real_tiled": z[0].contiguous(),
        "tap_real_tiled_pixels": dec_tiled[0].contiguous(),
        "tap_real_untiled_pixels": dec_untiled[0].contiguous(),
    }
    path = os.path.join(args.out, "minimaxh3_video_vae_real_tiled.safetensors")
    save_file(tensors, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)

    manifest = {
        "run": {
            "seed": args.seed,
            "height": HEIGHT,
            "width": WIDTH,
            "latent_frames": LATENT_FRAMES,
            "tile_grid": [len(y_idx), len(x_idx)],
            "height_overlaps": [int(v) for v in y_ov],
            "width_overlaps": [int(v) for v in x_ov],
            "tiled_vs_untiled_max_abs": d,
            "tiled_vs_untiled_rel_l2": rel,
        },
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "geometry": {
            "spatial_compression_ratio": ratio,
            "temporal_compression_ratio": int(model.temporal_compression_ratio),
            "tile_sample_min_height": int(model.tile_sample_min_height),
            "tile_sample_min_width": int(model.tile_sample_min_width),
            "tile_sample_min_overlap_height": int(model.tile_sample_min_overlap_height),
            "tile_sample_min_overlap_width": int(model.tile_sample_min_overlap_width),
        },
        "sha256": {os.path.basename(path): sha},
    }
    manifest["source"] = source_block(
        checkpoint=vae_dir,
        files=("config.json",),
        hash_files=False,
        identity={
            "latent_channels": int(model.config.latent_channels),
            "decoder_num_layers": int(model.config.decoder_num_layers),
            "decoder_num_attention_heads": int(model.config.decoder_num_attention_heads),
            "decoder_attention_head_dim": int(model.config.decoder_attention_head_dim),
            "spatial_compression_ratio": ratio,
            "height": HEIGHT,
            "width": WIDTH,
        },
    )
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
