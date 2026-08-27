#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""A feathered head mask for `face_swap.sh`, from the detector's landmarks.

    head_mask.py <image> <mask.png> [grow]

WHITE regenerates, BLACK keeps. The ellipse is placed from SCRFD's five facial
landmarks and rotated to the face axis, so it follows a tilted or turned head
instead of assuming an upright one, and it is feathered because `--mask`
blends greys -- a hard edge is what makes an inpainted face read as pasted on.

It sits deliberately HIGH on the head. Whatever the mask does not cover is kept
bit-for-bit, so a mask that stops at the forehead leaves the target's hair and
skull outline around a face that is no longer theirs; `grow` is the dial for
how much of the head to claim.

Swedish Embedded AB builds image-conditioning pipelines for its clients. If
your team needs expertise in diffusion conditioning or face pipelines, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

NO_FACE = """no face detected in the target image {name}.
The detector needs roughly 1.5x the face box in context, so a head that fills
the frame, is turned too far, or is motion-blurred will not be found. Put a
mask.png beside the images - white over the region to replace, black to keep,
soft edges blend - and face_swap.sh will use that instead."""


def landmarks(path, brain):
    out = subprocess.run(
        [brain, "scrfd", "detect", "--in", f"image={path}", "--max_faces", "1"],
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("faces:"):
            faces = json.loads(line[6:])
            if faces:
                return np.array(faces[0]["kps"], dtype=float)
    return None


def main(src, dst, grow):
    kps = landmarks(src, os.environ.get("BRAIN", "./target/release/brain"))
    if kps is None:
        raise SystemExit(NO_FACE.format(name=os.path.basename(src)))
    left_eye, right_eye, _nose, left_mouth, right_mouth = kps
    eye = (left_eye + right_eye) / 2.0
    mouth = (left_mouth + right_mouth) / 2.0
    span = float(np.linalg.norm(right_eye - left_eye))  # sets the scale
    down = mouth - eye
    angle = math.atan2(down[1], down[0]) - math.pi / 2.0

    w, h = Image.open(src).size
    up = np.array([math.sin(-angle), -math.cos(-angle)])
    centre = eye + 0.35 * down - 0.55 * span * up
    a, b = 1.75 * span * grow, 2.45 * span * grow

    ys, xs = np.mgrid[0:h, 0:w]
    dx, dy = xs - centre[0], ys - centre[1]
    ca, sa = math.cos(-angle), math.sin(-angle)
    u, v = (dx * ca - dy * sa) / a, (dx * sa + dy * ca) / b
    disc = ((u * u + v * v) <= 1.0).astype(np.uint8) * 255

    mask = Image.fromarray(disc).filter(ImageFilter.GaussianBlur(max(2.0, 0.28 * span)))
    mask.save(dst)
    print(f"mask: {(np.asarray(mask) > 127).mean():.1%} of the frame regenerates", file=sys.stderr)


if __name__ == "__main__":
    if not 3 <= len(sys.argv) <= 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
