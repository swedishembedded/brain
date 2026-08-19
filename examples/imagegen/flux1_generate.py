#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""FLUX.1 text-to-image over brain's D-Bus interface.

Like `sdxl_generate.py`, `text2image` here is a plain `Run` call (no
per-step progress hook yet), so this is a single blocking request that
returns the finished image. See `crates/flux1/src/pipeline.rs`'s module
docs for what this pipeline does and does not cover yet (no Kontext
reference-image editing, no img2img, no LoRA) and its honest note on
verification -- this has not been run against real FLUX.1 weights in the
environment that wrote it.

Run under a private session bus (weights via env):

    BRAIN_FLUX1_DIR=/path/to/FLUX.1-dev dbus-run-session -- bash -c '
      brain serve --dbus & sleep 2
      python3 examples/imagegen/flux1_generate.py --prompt "a red fox in the snow"'

`BRAIN_FLUX1_DIR` is a released diffusers FLUX.1 checkpoint root
(`transformer/`, `vae/`, `text_encoder/`+`tokenizer/` for CLIP-L,
`text_encoder_2/`+`tokenizer_2/` for T5-XXL).

Requires: jeepney - `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainError, skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import save_ppm  # noqa: E402

MODEL = "brain/flux1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="text description of the desired image")
    ap.add_argument("--variant", default="dev", choices=["dev", "kontext-dev", "schnell"])
    ap.add_argument("--out", default="flux1.ppm", help="output PPM path")
    ap.add_argument("--width", type=int, default=1024, help="output width (multiple of 16)")
    ap.add_argument("--height", type=int, default=1024, help="output height (multiple of 16)")
    ap.add_argument("--steps", type=int, default=0, help="denoise steps (0 = variant default)")
    ap.add_argument("--guidance", type=float, default=3.5, help="guidance_in scalar (dev/kontext-dev only)")
    ap.add_argument("--max-len", type=int, default=512, dest="max_len", help="T5-XXL context length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params = {
        "prompt": args.prompt,
        "variant": args.variant,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
        "max_len": args.max_len,
        "seed": args.seed,
    }

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            skip(f"{MODEL!r} not served (set BRAIN_FLUX1_DIR)")

        print(f"text2image {args.width}x{args.height} ({args.variant}):")
        t0 = time.monotonic()
        try:
            out = brain.run(MODEL, "text2image", params)
        except BrainError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        dt = time.monotonic() - t0

        data = out.blobs.get("image")
        if data is None:
            print("  no image blob arrived", file=sys.stderr)
            return 1
        meta = (out.meta.get("image") or {}).get("meta") or {}
        w, h, c = int(meta.get("w", args.width)), int(meta.get("h", args.height)), int(meta.get("c", 3))
        save_ppm(args.out, data, w, h, c)
        print(f"wrote {args.out} ({w}x{h}) in {dt:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
