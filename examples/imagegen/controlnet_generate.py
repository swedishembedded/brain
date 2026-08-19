#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""SDXL + ControlNet text-to-image over brain's D-Bus interface.

The one addition over `sdxl_generate.py`: a conditioning image (edge map,
depth map, pose, ...) rides in as a second input blob, `control_image`.
`crates/controlnet/src/caps.rs` resizes it on the device to match the output
size, so it need not be pre-sized to match `--width`/`--height`.

Run under a private session bus (weights via env):

    BRAIN_SDXL_DIR=/path/to/stable-diffusion-xl-base-1.0 \\
    BRAIN_CONTROLNET_DIR=/path/to/controlnet-canny-sdxl-1.0 \\
    dbus-run-session -- bash -c '
      brain serve --dbus & sleep 2
      python3 examples/imagegen/controlnet_generate.py \\
        --prompt "a red fox in the snow" --control canny_edges.ppm'

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

MODEL = "brain/sdxl-controlnet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="text description of the desired image")
    ap.add_argument("--control", required=True, help="binary PPM (P6) conditioning image (edges/depth/pose/...)")
    ap.add_argument("--negative", default="")
    ap.add_argument("--out", default="controlnet.ppm", help="output PPM path")
    ap.add_argument("--width", type=int, default=1024, help="output width (multiple of 8)")
    ap.add_argument("--height", type=int, default=1024, help="output height (multiple of 8)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--conditioning-scale", type=float, default=1.0, dest="conditioning_scale")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cond, cw, ch = load_ppm(args.control)
    params = {
        "prompt": args.prompt,
        "negative": args.negative,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
        "conditioning_scale": args.conditioning_scale,
        "seed": args.seed,
    }

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            skip(f"{MODEL!r} not served (set BRAIN_SDXL_DIR + BRAIN_CONTROLNET_DIR)")

        print(f"text2image {args.width}x{args.height}, control {cw}x{ch}, {args.steps} steps:")
        t0 = time.monotonic()
        try:
            out = brain.run(
                MODEL, "text2image", params,
                blobs={"control_image": cond}, meta={"control_image": {"media": "image", "w": cw, "h": ch, "c": 3}},
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
