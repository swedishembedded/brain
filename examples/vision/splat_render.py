#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""3D Gaussian Splatting: one-shot `render` over brain's D-Bus surface.

Sends a scene (Inria-layout binary PLY) and a camera pose, gets back one
rendered image. `render` is NOT streaming (unlike `fit` below): there is
nothing to report progress about for a single rasterizer pass.

With neither `--eye` nor `--target` given, the server auto-frames the scene
from its own bounds - the same default `brain splat render` uses.

    dbus-run-session -- bash -c '
      brain serve --dbus & sleep 2
      python3 examples/vision/splat_render.py --scene scene.ply --out render.ppm'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import save_ppm  # noqa: E402

MODEL = "brain/splat"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="Inria-layout binary PLY to render")
    ap.add_argument("--out", default="render.ppm", help="output PPM path")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=60.0, help="vertical field of view, degrees")
    ap.add_argument("--eye", default="", help="camera position 'x,y,z' (pairs with --target)")
    ap.add_argument("--target", default="", help="camera look-at point 'x,y,z' (pairs with --eye)")
    ap.add_argument("--bg", default="0,0,0", help="background color 'r,g,b' in [0,1]")
    ap.add_argument("--depth", action="store_true", help="render expected depth instead of color")
    args = ap.parse_args()

    scene = Path(args.scene).read_bytes()
    params = {"width": args.width, "height": args.height, "fov": args.fov, "bg": args.bg, "depth": args.depth}
    if args.eye and args.target:
        params["eye"] = args.eye
        params["target"] = args.target

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            print(f"FATAL: '{MODEL}' not served", file=sys.stderr)
            return 2

        # `{"media": "bytes"}` is the whole story for a self-describing PLY
        # blob - no w/h/c needed, unlike an image blob (see brain_py.image's
        # own doc on that convention, which `render`'s output below follows).
        out = brain.run(MODEL, "render", params, blobs={"scene": scene}, meta={"scene": {"media": "bytes"}})
        image = out.blobs["image"]
        w, h = out.outputs["width"], out.outputs["height"]
        save_ppm(args.out, image, w, h, 3)
        print(f"{args.scene} -> {args.out} ({w}x{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
