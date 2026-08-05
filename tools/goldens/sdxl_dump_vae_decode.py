#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake the SDXL VAE decode golden for `crates/vae/tests/sdxl_decode_parity.rs`.

Writes `<mirror>/vae/sdxl_vae_decode.safetensors`, which `make fetch/testdata`
links to `testdata/golden/vae/`: a fixed `[1,4,16,16]` latent and
diffusers' `vae.decode(z).sample` `[1,3,128,128]`.

Why a hand-rolled load instead of `AutoencoderKL.from_pretrained`: SDXL base
ships only `diffusion_pytorch_model.fp16.safetensors`, and `from_pretrained`
looks for the un-suffixed name. brain reads that same fp16 file and upcasts to
f32, so the golden is baked from exactly the tensors brain consumes.

Usage:
  python3 tools/goldens/sdxl_dump_vae_decode.py \
      --sdxl /path/to/stable-diffusion-xl-base-1.0 [--out <golden-mirror>/vae]
"""

import argparse
import json
import os
import pathlib

import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file, save_file


# Overridable machine path (scripts/gates/check-scripts.sh 3/3); the same
# BRAIN_GOLDEN_MIRROR that scripts/data/fetch-testdata.sh links goldens from.
GOLDEN_MIRROR = os.environ.get("BRAIN_GOLDEN_MIRROR", "/data/workspace/resources/brain-goldens")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdxl", required=True, help="stable-diffusion-xl-base-1.0 root")
    ap.add_argument(
        "--out",
        default=f"{GOLDEN_MIRROR}/vae",
        help="golden mirror dir that make fetch/testdata links from",
    )
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    vdir = pathlib.Path(a.sdxl) / "vae"
    cfg = json.loads((vdir / "config.json").read_text())
    # The reference's own defaults matter here: `use_quant_conv` and
    # `use_post_quant_conv` are True in `AutoencoderKL.__init__` and SDXL's
    # config.json omits both. Constructing from_config preserves that, which is
    # the behaviour brain's VaeConfig must match (docs/lessons.md #18).
    vae = AutoencoderKL.from_config(cfg)
    assert vae.config.use_post_quant_conv, "reference defaults changed — re-check VaeConfig"

    weights = next(vdir.glob("*.safetensors"))
    sd = {k: v.float() for k, v in load_file(str(weights)).items()}
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    vae = vae.float().eval()

    g = torch.Generator().manual_seed(a.seed)
    z = torch.randn(1, 4, 16, 16, generator=g)
    with torch.no_grad():
        decoded = vae.decode(z).sample

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {"z": z.contiguous(), "decoded": decoded.contiguous()},
        str(out / "sdxl_vae_decode.safetensors"),
    )
    print(f"z {tuple(z.shape)} -> decoded {tuple(decoded.shape)} "
          f"range [{decoded.min():.4f}, {decoded.max():.4f}]")
    print(f"wrote {out / 'sdxl_vae_decode.safetensors'} (from {weights.name})")


if __name__ == "__main__":
    main()
