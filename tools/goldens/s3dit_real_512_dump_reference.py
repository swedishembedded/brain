#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake a REAL Z-Image-Turbo (6B) DiT forward golden at the actual 512x512
generation scale (1024 image tokens + 64 caption tokens = 1088 joint tokens,
the real `layers.*` shape brain's text2image CLI runs).

`tools/goldens/s3dit_real_dump_reference.py`'s existing golden uses a tiny
16x16 latent (64 image tokens) - cheap, but too small to expose corruption
that only shows up once attention runs over the realistic joint sequence
length (see `crates/s3dit/tests/real_parity.rs`'s
`zimage_real_dit_matches_diffusers_at_512` for the concrete regression this
caught: a `flip_sin_to_cos` sign/order bug in the timestep embedding that a
64-token forward could not distinguish from noise, but a 1088-token one
catches at cosine ~0.80).

Loads the HF-native `transformer/` directory directly (already diffusers-
named - `all_x_embedder.*`, unfused `to_q`/`to_k`/`to_v` - so no Comfy
remap is needed, unlike the fused-qkv Comfy checkpoint the sibling script
targets). Dev-time only; the output is a small (~1.5 MB) fixture of just the
inputs and final output - not the ~24 GB checkpoint.
"""
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

if len(sys.argv) < 2:
    sys.exit("usage: s3dit_real_512_dump_reference.py <z_image_turbo transformer/ dir> [out.safetensors]")
TRANSFORMER_DIR = sys.argv[1]
_TESTDATA = os.environ.get("BRAIN_TESTDATA") or str(Path(__file__).resolve().parents[2] / "testdata")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(Path(_TESTDATA) / "golden" / "zimage" / "zimage_real_512.safetensors")


def main():
    model = ZImageTransformer2DModel.from_pretrained(TRANSFORMER_DIR, torch_dtype=torch.float32)
    model = model.eval()

    torch.manual_seed(0)
    # 512x512 image -> VAE latent 64x64 -> patch_size=2 -> 32x32 = 1024 image
    # tokens; cap_len=64 (the CLI's fixed caption length, `s3dit::caps`'s
    # `text2image` hot-pipeline build). Joint sequence: 1024 + 64 = 1088,
    # matching the CLI's real 512x512 shape exactly.
    latent = torch.randn(16, 1, 64, 64)
    cap = torch.randn(64, 2560)
    t = torch.tensor([0.5])
    with torch.no_grad():
        out = model([latent], t, [cap], patch_size=2, f_patch_size=1).sample[0]

    save_file(
        {"_latent": latent.contiguous(), "_cap": cap.contiguous(), "_t": t.contiguous(), "_out": out.contiguous()},
        OUT, metadata={"model": "Z-Image-Turbo 6B real, 512x512 scale"},
    )
    print(f"wrote {OUT}  out {tuple(out.shape)} [min,max,mean]=[{out.min():.4f},{out.max():.4f},{out.mean():.4f}]")


if __name__ == "__main__":
    sys.exit(main())
