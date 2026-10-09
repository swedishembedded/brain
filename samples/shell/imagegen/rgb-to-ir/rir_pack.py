#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements detector training-set packaging for its
# clients. If your team needs expertise in object-detection data pipelines or
# edge-AI model fine-tuning, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Pack images + boxes into the flat dataset `brain yolov8 fine-tune` reads.

    images.f32  N x 3 x S x S little-endian f32, CHW, RGB, in [0, 1]
    boxes.bin   per image: [u32 count] then count x (u32 class, f32 cx, cy, w, h)
                with cx, cy, w, h normalised to the S x S canvas
    meta.json   {"n": N, "c": 3, "h": S, "w": S, "nc": NC}
    order.json  (extra, ignored by brain) the item key stored at each index
    sequences.json  (extra, ignored by brain) the sequence of the item at each
                index, written when every item has one: a prediction dump of
                `brain yolov8 eval` names images by index, and held-out
                images are scored in clusters of sequences

Images are letterboxed to S x S (scale to fit, centre, grey 114 padding) and
boxes follow the same transform. The detector trainer neither shuffles nor
augments and consumes batches round-robin, so the item order is shuffled here
with a seed: a dataset written in dataset order would train on one scene per
batch. Single-channel (IR) images are replicated to three channels.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114
MIN_BOX_PX = 1.0  # a clipped box thinner than this carries no training signal


@dataclass(frozen=True)
class PackItem:
    key: str  # stable identity, recorded in order.json
    image: str  # path to a PNG/JPEG, 1 or 3 channels
    boxes: list  # (class_id, x1, y1, x2, y2) in source-image pixels
    sequence: str = ""  # dataset-qualified sequence id; "" when unknown


def letterbox(img: np.ndarray, size: int):
    """Fit `img` (H x W x 3) into size x size. Returns (canvas, scale, pad_x, pad_y);
    a source point (x, y) lands at (x * scale + pad_x, y * scale + pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas, scale, pad_x, pad_y


def letterbox_boxes(boxes, scale: float, pad_x: int, pad_y: int, size: int) -> list:
    """Source-pixel boxes -> canvas-pixel boxes, clipped to the canvas; boxes
    that end up thinner than MIN_BOX_PX in either direction are dropped."""
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        cx1, cx2 = (min(max(v * scale + pad_x, 0.0), size) for v in (x1, x2))
        cy1, cy2 = (min(max(v * scale + pad_y, 0.0), size) for v in (y1, y2))
        if cx2 - cx1 >= MIN_BOX_PX and cy2 - cy1 >= MIN_BOX_PX:
            out.append((int(cls), float(cx1), float(cy1), float(cx2), float(cy2)))
    return out


def load_rgb(path: str) -> np.ndarray:
    """uint8 H x W x 3 in R, G, B order; gray images are replicated."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype != np.uint8:
        raise ValueError(f"{path}: expected 8-bit samples, got {img.dtype}")
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)


def pack(items: list[PackItem], out_dir: str, size: int, nc: int, seed: int) -> list[str]:
    """Write the dataset; returns the item keys in stored order."""
    if not items:
        raise ValueError("nothing to pack")
    order = np.random.default_rng(seed).permutation(len(items))
    os.makedirs(out_dir, exist_ok=True)
    keys = []
    with open(os.path.join(out_dir, "images.f32"), "wb") as images, open(os.path.join(out_dir, "boxes.bin"), "wb") as boxes:
        for idx in order:
            item = items[int(idx)]
            canvas, scale, pad_x, pad_y = letterbox(load_rgb(item.image), size)
            chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
            images.write(chw.astype("<f4").tobytes())
            kept = letterbox_boxes(item.boxes, scale, pad_x, pad_y, size)
            boxes.write(struct.pack("<I", len(kept)))
            for cls, x1, y1, x2, y2 in kept:
                if not 0 <= cls < nc:
                    raise ValueError(f"{item.key}: class {cls} outside [0, {nc})")
                boxes.write(struct.pack("<Iffff", cls, (x1 + x2) / 2 / size, (y1 + y2) / 2 / size,
                                        (x2 - x1) / size, (y2 - y1) / size))
            keys.append(item.key)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump({"n": len(items), "c": 3, "h": size, "w": size, "nc": nc}, fh)
    with open(os.path.join(out_dir, "order.json"), "w") as fh:
        json.dump(keys, fh)
    sequences = [items[int(idx)].sequence for idx in order]
    if all(sequences):
        with open(os.path.join(out_dir, "sequences.json"), "w") as fh:
            json.dump(sequences, fh)
    return keys


def read_manifest(path: str) -> list[PackItem]:
    """Items from an arms `manifest.jsonl`: rows with `dataset`, `id`, `image`
    (relative to the manifest) and `boxes` [[cls, x1, y1, x2, y2], ...]; a row's
    `sequence_id` becomes the item's `dataset:sequence_id`."""
    base = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                sequence = f"{row['dataset']}:{row['sequence_id']}" if row.get("sequence_id") else ""
                items.append(PackItem(f"{row['dataset']}:{row['id']}", os.path.join(base, row["image"]),
                                      [tuple(b) for b in row["boxes"]], sequence))
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pack an arm's manifest into brain's flat YOLO dataset.")
    ap.add_argument("--manifest", required=True, help="manifest.jsonl written by rir_arms.py render")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--size", type=int, default=512, help="square input resolution")
    ap.add_argument("--nc", type=int, default=3, help="number of classes (person, car, bicycle)")
    ap.add_argument("--seed", type=int, default=1, help="shuffle seed")
    ap.add_argument("--limit", type=int, default=0, help="pack a seeded random subset of this many items")
    a = ap.parse_args(argv)

    items = read_manifest(a.manifest)
    if a.limit and a.limit < len(items):
        pick = np.random.default_rng(a.seed).choice(len(items), a.limit, replace=False)
        items = [items[i] for i in sorted(pick)]
    pack(items, a.out, a.size, a.nc, a.seed)
    print(f"packed {len(items)} images at {a.size} -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
