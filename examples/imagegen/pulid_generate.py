#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""PuLID identity-conditioned FLUX.1 over brain's D-Bus interface.

Adds a `face_image` blob to `flux1_generate.py`'s params: the photo becomes
32 ID tokens (ArcFace + EVA-CLIP -> IDFormer) cross-attended into the FLUX.1
image stream at `id_weight` strength. See
`crates/pulid/src/caps.rs`'s module docs for exactly what this composes and
the one real, documented preprocessing gap (a plain resize stands in for the
reference's RetinaFace+BiSeNet face crop ahead of EVA-CLIP).

Run under a private session bus (weights via env):

    BRAIN_FLUX1_DIR=/path/to/FLUX.1-dev \\
    BRAIN_PULID_DIR=/path/to/pulid_flux_v0.9.1.safetensors \\
    BRAIN_ARCFACE_DIR=/path/to/antelopev2 \\
    BRAIN_CLIP_DIR=/path/to/eva-clip-dir \\
    dbus-run-session -- bash -c '
      brain serve --dbus & sleep 2
      python3 examples/imagegen/pulid_generate.py \\
        --prompt "a photo of a person hiking in the mountains" --face portrait.ppm'

`BRAIN_CLIP_DIR` only needs to hold the EVA-CLIP-L/336 file
(`EVA02_CLIP_L_336_psz14_s6B.pt`) at its root - the same convention
`clip::caps` uses - not the CLIP-L/OpenCLIP-bigG text towers, which this
action does not need.

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
from brain_py.image import load_ppm, save_ppm  # noqa: E402

MODEL = "brain/flux1-pulid"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="text description of the desired image")
    ap.add_argument("--face", required=True, help="binary PPM (P6) photo of the identity to condition on")
    ap.add_argument("--variant", default="dev", choices=["dev", "kontext-dev", "schnell"],
                     help="only dev is validated against a PuLID reference")
    ap.add_argument("--out", default="pulid.ppm", help="output PPM path")
    ap.add_argument("--width", type=int, default=1024, help="output width (multiple of 16)")
    ap.add_argument("--height", type=int, default=1024, help="output height (multiple of 16)")
    ap.add_argument("--steps", type=int, default=0, help="denoise steps (0 = variant default)")
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--id-weight", type=float, default=0.8, dest="id_weight", help="identity conditioning strength")
    ap.add_argument("--max-len", type=int, default=512, dest="max_len")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    face, fw, fh = load_ppm(args.face)
    params = {
        "prompt": args.prompt,
        "variant": args.variant,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
        "id_weight": args.id_weight,
        "max_len": args.max_len,
        "seed": args.seed,
    }

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            skip(f"{MODEL!r} not served (set BRAIN_FLUX1_DIR + BRAIN_PULID_DIR + BRAIN_ARCFACE_DIR + BRAIN_CLIP_DIR)")

        print(f"text2image {args.width}x{args.height} ({args.variant}), face {fw}x{fh}, id_weight={args.id_weight}:")
        t0 = time.monotonic()
        try:
            out = brain.run(
                MODEL, "text2image", params,
                blobs={"face_image": face}, meta={"face_image": {"media": "image", "w": fw, "h": fh, "c": 3}},
            )
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
