# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Generate a synthetic in-distribution image for the tiny ``gen detect`` model.

The brain synthetic detection dataset (``brain data gen detect``) renders solid
shapes on a constant DARK-GREY background, one saturated fill color per class.
These constants are copied verbatim from ``crates/data/src/gen_detect.rs`` so an
image we produce here lands in the model's training distribution.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

# crates/data/src/gen_detect.rs : const BG
BG = (32, 32, 32)

# crates/data/src/gen_detect.rs : const PALETTE (class id -> fill color)
PALETTE = [
    (220, 40, 40),    # 0: red
    (40, 200, 40),    # 1: green
    (60, 80, 230),    # 2: blue
    (230, 200, 40),   # 3: yellow
    (210, 60, 210),   # 4: magenta
    (40, 210, 210),   # 5: cyan
]


def make_test_image(w: int = 128, h: int = 128, nc: int = 3) -> Image.Image:
    """A 128x128 dark-grey scene with a few solid shapes in class colors.

    Default geometry matches the tiny model (128px input, MultiObject preset,
    3 classes). Shapes are non-overlapping and sized like the generator's
    0.10-0.28 fraction range so the model sees familiar objects.
    """
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    # A red rectangle (class 0), a green circle (class 1), a blue rectangle (2).
    if nc >= 1:
        draw.rectangle([14, 16, 46, 52], fill=PALETTE[0])          # class 0 rect
    if nc >= 2:
        draw.ellipse([70, 18, 104, 52], fill=PALETTE[1])           # class 1 circle
    if nc >= 3:
        draw.rectangle([40, 78, 86, 112], fill=PALETTE[2])         # class 2 rect
    return img


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="generate a synthetic gen_detect image")
    ap.add_argument("--out", default="test_image.png")
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--nc", type=int, default=3)
    args = ap.parse_args()
    img = make_test_image(args.w, args.h, args.nc)
    img.save(args.out)
    print(f"wrote {args.out} ({args.w}x{args.h}, {args.nc} shapes)")


if __name__ == "__main__":
    main()
