#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Face detection + identity matching over brain's D-Bus surface.

Two models of the insightface antelopev2 stack, one action each:

  * `brain/scrfd` `detect` - boxes, scores and the 5 landmarks, in SOURCE-image
    pixels;
  * `brain/arcface` `embed` - one L2-normalised 512-d ArcFace vector for the
    primary face (it runs the detector itself unless `align=false`).

Given several photos it prints the detections, writes a box overlay per photo,
and then prints the full cosine similarity matrix over the embeddings. Because
the vectors are already unit-norm, the cosine is a plain dot product - same
faces land near 1.0, different faces near 0.

    BRAIN_SCRFD_DIR=$BRAIN_TESTDATA/face/antelopev2 \\
    BRAIN_ARCFACE_DIR=$BRAIN_TESTDATA/face/antelopev2 \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 3
        python3 examples/vision/face_id.py a.ppm b.ppm c.ppm'
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd, sealed_memfd  # noqa: E402
from brain_py.image import draw_boxes, load_ppm, save_ppm  # noqa: E402


def image_args(img: bytes, w: int, h: int) -> dict:
    return {
        "in_fds": {"image": sealed_memfd(img)},
        "in_meta": {"image": {"media": "image", "w": w, "h": h, "c": 3}},
    }


def embedding_of(brain: BrainDBus, img: bytes, w: int, h: int) -> tuple[list[float], dict]:
    """`embed` returns the vector as a raw `Media::Bytes` blob (512 f32 LE)."""
    r = brain.run("brain/arcface", "embed", {}, **image_args(img, w, h))
    raw = read_fd(r.fds["embedding"])
    vec = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    return vec, r.result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="binary PPMs (P6), one or more faces each")
    ap.add_argument("--out", default="/tmp", help="directory for the box overlays")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with BrainDBus() as brain:
        served = brain.models()
        missing = [m for m in ("brain/scrfd", "brain/arcface") if m not in served]
        if missing:
            print(f"FATAL: {missing} not served (set BRAIN_SCRFD_DIR / BRAIN_ARCFACE_DIR)", file=sys.stderr)
            return 2

        embeddings: list[list[float]] = []
        for i, path in enumerate(args.photos):
            img, w, h = load_ppm(path)
            det = brain.run("brain/scrfd", "detect", {}, **image_args(img, w, h)).result
            print(f"{path}: {w}x{h}, {det['count']} face(s)")
            for f in det["faces"]:
                x1, y1, x2, y2 = f["bbox"]
                print(f"    score {f['score']:.4f}  box [{x1:7.1f} {y1:7.1f} {x2:7.1f} {y2:7.1f}]")
            if det["count"] == 0:
                print("    (no face - skipping the embedding)")
                embeddings.append([])
                continue
            boxed = draw_boxes(img, w, h, [tuple(f["bbox"]) for f in det["faces"]])
            save_ppm(out / f"faces{i}.ppm", boxed, w, h, 3)

            vec, meta = embedding_of(brain, img, w, h)
            norm = sum(v * v for v in vec) ** 0.5
            print(f"    embedding: {meta['dim']}-d, |v| = {norm:.6f} (the action L2-normalises)")
            embeddings.append(vec)

        print("\ncosine similarity (unit vectors -> a plain dot product):")
        names = [Path(p).name for p in args.photos]
        width = max(len(n) for n in names) + 2
        print(" " * width + "".join(f"{n:>10}" for n in names))
        for i, a in enumerate(embeddings):
            row = ""
            for b in embeddings:
                row += "         -" if not a or not b else f"{sum(x * y for x, y in zip(a, b)):>10.4f}"
            print(f"{names[i]:<{width}}{row}")

    print(f"\nwrote box overlays to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
