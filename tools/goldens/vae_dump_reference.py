#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake a Z-Image VAE (FLUX AutoencoderKL) decode golden for brain's parity test.

Dev-time only: brain's build/test path never imports torch. This loads the
reference decoder, decodes a fixed small latent on CPU, and writes the input
latent + expected image to a committed safetensors fixture that the Rust test
reads back. Small spatial size (8x8 latent -> 64x64 image) keeps the fixture
tiny while exercising every decoder stage (conv_in, mid attention, 4 up-blocks).

usage: vae_dump_reference.py <vae_dir> [out.safetensors] --seed 0
"""
import argparse, json, os, sys
from pathlib import Path
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file, save_file

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("vae_dir", help="diffusers vae/ dir (config.json + diffusion_pytorch_model.safetensors)")
_ap.add_argument("out", nargs="?", help="output safetensors (default: testdata/golden/vae/zimage_vae_decode.safetensors)")
_ap.add_argument("--seed", type=int, required=True, help="seed of the random input; recorded in the output metadata")
_ARGS = _ap.parse_args()
VAE_DIR = _ARGS.vae_dir
# testdata/golden/vae/... by default -- where
# crates/vae/tests/decode_parity.rs's testdata("golden/vae/zimage_vae_decode.safetensors")
# actually looks.
_TESTDATA = os.environ.get("BRAIN_TESTDATA") or str(Path(__file__).resolve().parents[2] / "testdata")
OUT = _ARGS.out or str(Path(_TESTDATA) / "golden" / "vae" / "zimage_vae_decode.safetensors")

def main():
    cfg = json.load(open(os.path.join(VAE_DIR, "config.json")))
    vae = AutoencoderKL.from_config(cfg)
    sd = load_file(os.path.join(VAE_DIR, "diffusion_pytorch_model.safetensors"))
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not missing, f"missing VAE weights: {missing[:8]}"
    assert not unexpected, f"unexpected VAE weights: {unexpected[:8]}"
    vae = vae.to(torch.float32).eval()

    torch.manual_seed(_ARGS.seed)
    z = torch.randn(1, cfg["latent_channels"], 8, 8, dtype=torch.float32)
    with torch.no_grad():
        img = vae.decode(z).sample  # [1, 3, 64, 64]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        {"latent": z.contiguous(), "image": img.contiguous()},
        OUT,
        metadata={"src": "AutoencoderKL flux-dev / Z-Image", "note": "vae.decode(z).sample, fp32 CPU", "seed": str(_ARGS.seed)},
    )
    print(f"wrote {OUT}")
    print(f"latent {tuple(z.shape)}  image {tuple(img.shape)}  "
          f"img[min,max,mean]=[{img.min():.4f},{img.max():.4f},{img.mean():.4f}]")

if __name__ == "__main__":
    sys.exit(main())
