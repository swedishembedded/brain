# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""End-to-end example: image in -> annotated image out, via ``brain run``.

    python -m brain_py.examples.detect_image \
        --weights /tmp/bp.weights --image in.png --out annotated.png [--conf 0.25]

Reads the input image with Pillow (any format), runs detection through a
``BrainClient`` (which drives ``brain run --yolo <weights>`` as a subprocess),
draws the returned boxes, and saves the annotated image. If ``--image`` is
omitted, a synthetic in-distribution image is generated so the demo runs out of
the box.
"""

from __future__ import annotations

import argparse
import sys

from PIL import Image

from ..annotate import annotate
from ..client import BrainClient
from ..coco import COCO_NAMES
from .make_test_image import make_test_image


def _load_names(spec):
    """Resolve --names into a class-id -> name list, or None to keep numeric ids."""
    if not spec:
        return None
    if spec.lower() == "coco":
        return COCO_NAMES
    with open(spec, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="detect objects in an image via brain run")
    ap.add_argument("--weights", help="YOLO weights for `brain run --yolo`. "
                    "If omitted, brain's built-in fake detector is used.")
    ap.add_argument("--image", help="input image (any Pillow format). "
                    "If omitted, a synthetic gen_detect image is generated.")
    ap.add_argument("--out", default="annotated.png", help="output annotated PNG")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence threshold (passed to brain run --conf)")
    ap.add_argument("--names", help="class names: 'coco' for the built-in COCO-80 "
                    "list, or a path to a newline-separated names file. Default: "
                    "numeric class ids.")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds to wait for a detection (real yolov8n at 640px on "
                    "the CPU JIT takes ~10s/frame; raise for slower hosts).")
    ap.add_argument("--brain-bin", help="path to the brain executable")
    args = ap.parse_args()

    names = _load_names(args.names)

    if args.image:
        image = Image.open(args.image)
        print(f"loaded image {args.image} ({image.size[0]}x{image.size[1]})")
    else:
        image = make_test_image()
        print(f"generated synthetic gen_detect image ({image.size[0]}x{image.size[1]})")

    with BrainClient(yolo=args.weights, conf=args.conf, brain_bin=args.brain_bin) as client:
        dets = client.detect(image, conf=args.conf, timeout=args.timeout)

    # Relabel with human-readable class names when provided.
    if names is not None:
        for d in dets:
            if 0 <= int(d.cls) < len(names):
                d.label = names[int(d.cls)]

    print(f"{len(dets)} detection(s):")
    for d in dets:
        print(f"  class={d.cls} label={d.label!r} conf={d.conf:.3f} "
              f"box=[{d.x1:.1f},{d.y1:.1f},{d.x2:.1f},{d.y2:.1f}]")

    out = annotate(image, dets)
    out.save(args.out)
    print(f"wrote annotated image: {args.out}")
    return 0 if dets else 0  # not an error to have zero boxes; caller decides


if __name__ == "__main__":
    sys.exit(main())
