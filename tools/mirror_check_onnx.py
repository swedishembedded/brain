#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Verify the exported WorldMirror-2 DINOv2 ONNX under OpenVINO.

Runs the graph on the deterministic synthetic T2 input (same as the Rust
parity tests) and compares the patch tokens against the committed golden
samples in crates/mirror/tests/golden/golden_meta.json.

  python3 tools/mirror_check_onnx.py out/mirror-dino.onnx [CPU|NPU]
"""
import json
import sys

import numpy as np
import openvino as ov
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def synth_image(w=600, h=400):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x, 0] = (x * 255) // max(w - 1, 1)
            img[y, x, 1] = (y * 255) // max(h - 1, 1)
            img[y, x, 2] = ((x * 7 + y * 13) // 4) % 256
    yy, xx = np.mgrid[0:h, 0:w]
    for (cx, cy, r, col) in [(150, 100, 60, (255, 40, 40)), (420, 260, 90, (30, 220, 90)), (300, 180, 30, (250, 250, 250))]:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[m] = col
    return img


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out/mirror-dino.onnx"
    device = sys.argv[2] if len(sys.argv) > 2 else "CPU"
    pil = Image.fromarray(synth_image(), "RGB")
    sq = pil.crop((41, 0, 441, 400)).resize((518, 518), Image.Resampling.BICUBIC)
    sn = ((np.asarray(sq).astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    x = sn.transpose(2, 0, 1)[None]  # [1,3,518,518]

    core = ov.Core()
    model = core.compile_model(path, device)
    out = model({"frame": x})["patch_tokens"][0]  # [1369,1024]

    meta = json.load(open("crates/mirror/tests/golden/golden_meta.json"))
    s = meta["t2_patch_tokens"]
    flat = out.reshape(-1)
    rms = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))
    ok = abs(rms - s["rms"]) < 0.001 * abs(s["rms"])
    worst = 0.0
    for i, v in zip(s["indices"], s["values"]):
        worst = max(worst, abs(float(flat[i]) - v))
    print(f"device {device}: rms {rms:.6f} (golden {s['rms']:.6f}), worst sampled abs diff {worst:.2e}")
    if not ok or worst > 5e-4:
        print("MISMATCH")
        sys.exit(1)
    print("OK — ONNX export matches the reference within tolerance")


if __name__ == "__main__":
    main()
