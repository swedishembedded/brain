#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Print the reference resizes `crates/imaging/tests/resize_reference.rs`
embeds: PIL's fixed-point bicubic on uint8 RGB (what torchvision's resize does
to a PIL image) and torch's antialiased bilinear and bicubic on a float
tensor (`F.interpolate(..., antialias=True, align_corners=False)`), at odd
sizes, downscales and upscales.

Run in the DeepSeek-VL reference environment (Pillow, torch):

    python3 tools/goldens/imaging_resize_reference.py

and paste the printed Rust constants into the test.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

W, H = 7, 5


def rgb() -> np.ndarray:
    """The fixture image: `(37x + 11y + 53c) mod 256`, HWC uint8."""
    y, x, c = np.meshgrid(np.arange(H), np.arange(W), np.arange(3), indexing="ij")
    return ((37 * x + 11 * y + 53 * c) % 256).astype(np.uint8)


def rust(name: str, values, ty: str) -> None:
    flat = np.asarray(values).reshape(-1)
    body = ", ".join(f"{v:.9e}" if ty == "f32" else str(int(v)) for v in flat)
    print(f"const {name}: [{ty}; {flat.size}] = [{body}];")


def main() -> None:
    img = rgb()
    for (w, h) in [(3, 4), (11, 9)]:
        out = np.asarray(Image.fromarray(img).resize((w, h), Image.BICUBIC))
        rust(f"PIL_BICUBIC_{w}X{h}", out, "u8")

    # Through lists: the reference environment's torch predates its NumPy.
    planar = (torch.tensor(img.transpose(2, 0, 1).tolist(), dtype=torch.float32) / 255.0)[None]
    for mode in ["bilinear", "bicubic"]:
        for (w, h) in [(3, 4), (11, 9)]:
            out = F.interpolate(planar, size=(h, w), mode=mode, align_corners=False, antialias=True)[0]
            rust(f"TORCH_{mode.upper()}_AA_{w}X{h}", np.array(out.tolist(), dtype=np.float32), "f32")


if __name__ == "__main__":
    main()
