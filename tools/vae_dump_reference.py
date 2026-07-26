#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake a Z-Image VAE (FLUX AutoencoderKL) decode golden for brain's parity test.

Dev-time only: brain's build/test path never imports torch. This loads the
reference decoder, decodes a fixed small latent on CPU, and writes the input
latent + expected image to a committed safetensors fixture that the Rust test
reads back. Small spatial size (8x8 latent -> 64x64 image) keeps the fixture
tiny while exercising every decoder stage (conv_in, mid attention, 4 up-blocks).
"""
import json, os, sys
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file, save_file

if len(sys.argv) < 2:
    sys.exit("usage: vae_dump_reference.py <vae_dir> [out.safetensors]\n"
             "  <vae_dir>: diffusers vae/ dir (config.json + diffusion_pytorch_model.safetensors)")
VAE_DIR = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(__file__), "..", "crates", "vae", "tests", "golden", "zimage_vae_decode.safetensors")

def main():
    cfg = json.load(open(os.path.join(VAE_DIR, "config.json")))
    vae = AutoencoderKL.from_config(cfg)
    sd = load_file(os.path.join(VAE_DIR, "diffusion_pytorch_model.safetensors"))
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not missing, f"missing VAE weights: {missing[:8]}"
    assert not unexpected, f"unexpected VAE weights: {unexpected[:8]}"
    vae = vae.to(torch.float32).eval()

    torch.manual_seed(0)
    z = torch.randn(1, cfg["latent_channels"], 8, 8, dtype=torch.float32)
    with torch.no_grad():
        img = vae.decode(z).sample  # [1, 3, 64, 64]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        {"latent": z.contiguous(), "image": img.contiguous()},
        OUT,
        metadata={"src": "AutoencoderKL flux-dev / Z-Image", "note": "vae.decode(z).sample, fp32 CPU"},
    )
    print(f"wrote {OUT}")
    print(f"latent {tuple(z.shape)}  image {tuple(img.shape)}  "
          f"img[min,max,mean]=[{img.min():.4f},{img.max():.4f},{img.mean():.4f}]")

if __name__ == "__main__":
    sys.exit(main())
