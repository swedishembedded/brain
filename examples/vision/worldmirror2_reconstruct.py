#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""WorldMirror-2 multi-view 3D reconstruction over brain's D-Bus surface.

Packs N local images into one `video` blob (the same convention every other
video input in this repo uses - `capability::blob::video_blob`, one shared
`(w, h)` for every frame) and sends them to the one-shot `reconstruct` action:
back comes a Gaussian-splat scene (Inria-layout PLY) plus the per-frame
cameras WorldMirror-2 predicted (no poses go IN - this model estimates them).
`reconstruct` is NOT streaming: it is a single feed-forward pass, nothing to
report progress about.

    BRAIN_WORLDMIRROR2_WEIGHTS=<mirror.safetensors> \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 5
        python3 examples/vision/worldmirror2_reconstruct.py \\
            --images a.ppm,b.ppm,c.ppm --out scene.ply'

    brain splat view scene.ply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import load_ppm, save_ppm  # noqa: E402

MODEL = "brain/worldmirror2"


def load_video(images: str) -> tuple[bytes, int, int, int]:
    """N local PPMs -> ONE concatenated HWC-f32 `video` blob payload plus
    `(frames, w, h)` - `splat_fit.py::load_video`'s own shape, minus the
    camera-count cross-check `fit` needs and `reconstruct` does not (there are
    no input poses here for a frame count to agree with)."""
    p = Path(images)
    paths = sorted(p.glob("*.ppm")) if p.is_dir() else [Path(x.strip()) for x in images.split(",")]
    if not paths:
        raise SystemExit(f"no images found in {images!r}")
    frames = []
    w = h = 0
    for i, path in enumerate(paths):
        data, fw, fh = load_ppm(path)
        if i == 0:
            w, h = fw, fh
        elif (fw, fh) != (w, h):
            raise SystemExit(f"{path}: {fw}x{fh}, expected {w}x{h} (every frame must share one size)")
        frames.append(data)
    return b"".join(frames), len(frames), w, h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of *.ppm (sorted) or a comma-separated list, in order")
    ap.add_argument("--out", default="scene.ply", help="output PLY path")
    ap.add_argument("--cameras", default=None, help="output cameras.json path (default: <out> with .ply -> _cameras.json)")
    ap.add_argument("--min-opacity", type=float, default=0.01, help="drop gaussians below this opacity")
    ap.add_argument("--max-depth", type=float, default=0.0, help="clip gaussians past this depth (0 = off)")
    ap.add_argument("--prune-voxel", type=float, default=0.0, help="voxel-merge duplicate gaussians at this edge length (0 = off)")
    ap.add_argument("--maps", action="store_true", help="also fetch a per-frame depth-map video, written as maps_NN.ppm")
    args = ap.parse_args()

    video, n_frames, w, h = load_video(args.images)
    cameras_path = args.cameras or (Path(args.out).with_suffix("").as_posix() + "_cameras.json")

    params = {
        "min_opacity": args.min_opacity,
        "max_depth": args.max_depth,
        "prune_voxel": args.prune_voxel,
        "maps": args.maps,
    }

    with BrainDBus() as brain:
        if MODEL not in brain.models():
            print(f"FATAL: '{MODEL}' not served (set BRAIN_WORLDMIRROR2_WEIGHTS)", file=sys.stderr)
            return 2

        print(f"reconstructing {n_frames} frame(s) at {w}x{h} ...", flush=True)
        out = brain.run(
            MODEL,
            "reconstruct",
            params,
            blobs={"images": video},
            meta={"images": {"media": "video", "frames": n_frames, "w": w, "h": h, "c": 3}},
        )

        scene = out.blobs["scene"]
        Path(args.out).write_bytes(scene)
        cameras = out.outputs["cameras"]
        Path(cameras_path).write_text(json.dumps(cameras, indent=2))
        print(f"wrote {args.out} ({len(scene)} bytes) + {cameras_path} ({len(cameras)} camera(s))")

        if args.maps and "maps" in out.blobs:
            maps_meta = out.meta.get("maps", {}).get("meta", {})
            mw, mh = maps_meta.get("w", w), maps_meta.get("h", h)
            data = out.blobs["maps"]
            per_frame = mw * mh * 3 * 4
            for i in range(len(data) // per_frame):
                chunk = data[i * per_frame : (i + 1) * per_frame]
                path = f"{Path(args.out).with_suffix('').as_posix()}_depth_{i:02d}.ppm"
                save_ppm(path, chunk, mw, mh)
                print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
