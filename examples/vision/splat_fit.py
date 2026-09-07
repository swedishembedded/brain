#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""3D Gaussian Splatting: streaming `fit` (scene optimization) over brain's
D-Bus surface.

Takes an initial scene plus N posed target views (a `cameras.json` in the same
shape `brain mirror infer`/`brain splat fit` write, and one binary PPM per
camera, in order) and optimizes the scene to match them - the rasterizer
backward pass, driven remotely. `fit` is streaming: `on_progress` prints the
per-iteration MSE as it improves, exactly like `brain splat fit`'s own
`println!`.

    dbus-run-session -- bash -c '
      brain serve --dbus & sleep 2
      python3 examples/vision/splat_fit.py --scene init.ply \\
          --cameras out/mirror/cameras.json --images out/mirror \\
          --out fitted.ply --iters 200'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import load_ppm  # noqa: E402

MODEL = "brain/splat"


def load_video(images: str, n_expected: int) -> tuple[bytes, int, int]:
    """N posed-view PPMs -> ONE concatenated HWC-f32 `video` blob payload plus
    the shared `(w, h)` every frame must match - the same convention every
    other video input in this repo uses (`capability::blob::video_blob`)."""
    p = Path(images)
    paths = sorted(p.glob("*.ppm")) if p.is_dir() else [Path(x.strip()) for x in images.split(",")]
    if len(paths) != n_expected:
        raise SystemExit(f"'views' has {n_expected} cameras but {len(paths)} images were given")
    frames = []
    w = h = 0
    for i, path in enumerate(paths):
        data, fw, fh = load_ppm(path)
        if i == 0:
            w, h = fw, fh
        elif (fw, fh) != (w, h):
            raise SystemExit(f"{path}: {fw}x{fh}, expected {w}x{h} (every frame must share dims)")
        frames.append(data)
    return b"".join(frames), w, h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="initial scene: Inria-layout binary PLY")
    ap.add_argument("--cameras", required=True, help="cameras.json (brain mirror infer's format)")
    ap.add_argument("--images", required=True, help="directory of *.ppm (sorted) or a comma-separated list, one per camera")
    ap.add_argument("--out", default="fitted.ply", help="output PLY path")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--min-scale", type=float, default=1e-4)
    args = ap.parse_args()

    scene = Path(args.scene).read_bytes()
    cameras = json.loads(Path(args.cameras).read_text())
    video, w, h = load_video(args.images, len(cameras))

    params = {"views": json.dumps(cameras), "iters": args.iters, "lr": args.lr, "min_scale": args.min_scale}

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            print(f"FATAL: '{MODEL}' not served", file=sys.stderr)
            return 2

        print(f"fitting against {len(cameras)} views ({w}x{h}, {args.iters} iters, lr {args.lr}) ...")

        def on_progress(step: int, total: int, message: str) -> None:
            print(f"  [{step}/{total}] {message}", flush=True)

        out = brain.subscribe(
            MODEL,
            "fit",
            params,
            blobs={"scene": scene, "video": video},
            meta={"scene": {"media": "bytes"}, "video": {"media": "video", "frames": len(cameras), "w": w, "h": h, "c": 3}},
            on_progress=on_progress,
            timeout=3600.0,
        )
        fitted = out.blobs["scene"]
        Path(args.out).write_bytes(fitted)
        print(f"final mse {out.outputs['mse']:.6f} -> {args.out} ({len(fitted)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
