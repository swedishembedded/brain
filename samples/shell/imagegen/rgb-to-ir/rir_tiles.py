#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements paired-image training-set preparation for
# image-editing LoRA fine-tuning for its clients. If your team needs
# expertise in reference-to-target dataset pipelines for diffusion
# fine-tuning, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Aligned square tiles of split-T pairs, in the folder format
`brain flux2 finetune <dir>` reads for PAIRED reference-to-target training.

The translator maps RGB (the reference) to IR (the target). A training folder
holds, per accepted tile:

    <name>_rgb.png   the reference, 3 channels
    <name>_ir.png    the target, the single IR channel replicated to 3 channels
    pairs.yaml       `<name>_ir.png: "<name>_rgb.png"`   (target: reference)
    captions.yaml    `<name>_ir.png: "<instruction>"`    (targets only: a
                     reference is an input, never a sample, so it gets no caption)
    tiles.jsonl      the audit index: every candidate tile, accepted or not,
                     with its crop box, rejection reason and boxes in tile pixels

The trainer center-crops each image to a square and resizes it to --size
identically for target and reference, so tiles are cut square here and the
crop box is the SAME for the RGB and the IR of a pair: alignment survives.

Tiling rule, any aspect ratio: along each axis, n = ceil(extent / size) tiles
whose origins are spread evenly over [0, extent - size], so neighbouring tiles
overlap and the whole frame is covered (a 640x512 frame at size 512 gives a
left and a right tile). A frame smaller than the tile on either axis is an
error: resizing would silently change the scale the trainer sees.

Only split-T pairs are tiled. A tile is rejected, with the reason recorded and
nothing written, when its RGB has the black-padding artifact (the RGB occupies
under `min_valid_fraction` of the tile once black is ignored) or when the
RGB / IR edge correlation shows the pair is misregistered or unrelated
(`rir_data.edge_alignment`, the same frame check the splits use).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

import rir_arms as A
import rir_data as D

NEUTRAL_CAPTION = "Convert to thermal infrared, white-hot."


@dataclass(frozen=True)
class TileChecks:
    """Thresholds of the per-tile checks. Cross-modal edge correlation is
    modest by nature (see `rir_data.edge_alignment`): the floor only rejects
    tiles with no relation at all, the shift gain rejects a visible offset."""

    min_valid_fraction: float = 0.98
    min_edge_corr: float = 0.05
    max_shift_gain: float = 0.08
    min_box_visible: float = 0.5  # share of a box's area that must lie in the tile to keep it


def tile_origins(extent: int, size: int) -> list[int]:
    """Origins along one axis, evenly spread so the tiles cover [0, extent)."""
    if extent < size:
        raise ValueError(f"frame extent {extent} is smaller than the tile size {size}")
    n = math.ceil(extent / size)
    if n == 1:
        return [0]
    return [round(i * (extent - size) / (n - 1)) for i in range(n)]


def tile_grid(width: int, height: int, size: int) -> list[tuple[int, int]]:
    """(x0, y0) of every tile, row-major."""
    return [(x, y) for y in tile_origins(height, size) for x in tile_origins(width, size)]


def tile_boxes(boxes: list[dict], x0: int, y0: int, size: int, min_visible: float) -> list[dict]:
    """Boxes clipped into tile pixels; a box with less than `min_visible` of its area in the tile is dropped."""
    out = []
    for b in boxes:
        cx1, cy1 = max(b["x1"], x0), max(b["y1"], y0)
        cx2, cy2 = min(b["x2"], x0 + size), min(b["y2"], y0 + size)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if (cx2 - cx1) * (cy2 - cy1) < min_visible * (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]):
            continue
        out.append({"class": b["class"], "x1": round(cx1 - x0, 1), "y1": round(cy1 - y0, 1),
                    "x2": round(cx2 - x0, 1), "y2": round(cy2 - y0, 1)})
    return out


def check_tile(rgb: np.ndarray, ir: np.ndarray, checks: TileChecks) -> tuple[str | None, dict]:
    """(rejection reason or None, measurements) for one aligned tile pair."""
    valid = D.valid_fraction(D.to_gray(rgb))
    zero, best, shift = D.edge_alignment(rgb, ir)
    measured = {"valid_fraction": round(valid, 4), "edge_corr": round(zero, 4), "edge_corr_best": round(best, 4),
                "best_shift": list(shift)}
    if valid < checks.min_valid_fraction:
        return "rgb_black_padding", measured
    if best - zero > checks.max_shift_gain:
        return "edge_misaligned", measured
    if zero < checks.min_edge_corr:
        return "edge_uncorrelated", measured
    return None, measured


def tile_name(row: dict, x0: int, y0: int) -> str:
    return f"{A.safe_name(row['dataset'])}_{A.safe_name(row['id'])}_x{x0}y{y0}"


def _write_png(path: str, img: np.ndarray) -> None:
    if not cv2.imwrite(path, img):
        raise OSError(f"could not write {path}")


def _cut_frame(job: tuple[dict, str, int, TileChecks]) -> list[dict]:
    """Every candidate tile of one frame; accepted ones are written to `out_dir`."""
    row, out_dir, size, checks = job
    rgb, ir = D.read_rgb(row), D.read_ir(row)
    h, w = rgb.shape[:2]
    base = {"dataset": row["dataset"], "id": row["id"], "sequence_id": row["sequence_id"]}
    if ir.shape != rgb.shape[:2]:
        return [{**base, "accepted": False, "reason": "rgb_ir_size_mismatch", "x0": 0, "y0": 0, "size": size}]
    index = []
    for x0, y0 in tile_grid(w, h, size):
        name = tile_name(row, x0, y0)
        rgb_tile = np.ascontiguousarray(rgb[y0:y0 + size, x0:x0 + size])
        ir_tile = np.ascontiguousarray(ir[y0:y0 + size, x0:x0 + size])
        reason, measured = check_tile(rgb_tile, ir_tile, checks)
        entry = {**base, "name": name, "x0": x0, "y0": y0, "size": size, "accepted": reason is None,
                 "reason": reason, **measured, "boxes": tile_boxes(row["boxes"], x0, y0, size, checks.min_box_visible)}
        if reason is None:
            entry.update(rgb=f"{name}_rgb.png", ir=f"{name}_ir.png")
            _write_png(os.path.join(out_dir, entry["rgb"]), rgb_tile)
            _write_png(os.path.join(out_dir, entry["ir"]), np.dstack([ir_tile] * 3))
        index.append(entry)
    return index


def write_flat_yaml(path: str, mapping: dict[str, str]) -> None:
    """A flat `key: "value"` mapping sorted by key, the shape of both pairs.yaml
    and captions.yaml. JSON string syntax is valid YAML double-quoted syntax,
    so any caption text round-trips through the trainer's YAML parser."""
    with open(path, "w") as fh:
        for key in sorted(mapping):
            fh.write(f"{key}: {json.dumps(mapping[key])}\n")


def write_index(out_dir: str, index: list[dict]) -> None:
    with open(os.path.join(out_dir, "tiles.jsonl"), "w") as fh:
        for entry in index:
            fh.write(json.dumps(entry) + "\n")


def read_index(out_dir: str) -> list[dict]:
    with open(os.path.join(out_dir, "tiles.jsonl")) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cut_tiles(doc: dict, out_dir: str, size: int = 512, seed: int = 1, limit: int = 0, datasets=None,
              checks: TileChecks = TileChecks(), workers: int = 1) -> dict:
    """Tile the usable split-T pairs of `doc` (a splits.json document) into `out_dir`.

    `limit` takes a seeded random subset of that many frames. Writes the
    tiles, `pairs.yaml`, a `captions.yaml` holding the neutral caption for
    every target (rir_captions rewrites it with instruction captions) and
    `tiles.jsonl`. Returns the counts: frames, tiles, accepted, rejected by reason.
    """
    datasets = tuple(datasets) if datasets else tuple(sorted({r["dataset"] for r in doc["frames"]}))
    rows = A.select_rows(doc, "T", datasets, limit, seed)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(row, out_dir, size, checks) for row in rows]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            per_frame = list(pool.map(_cut_frame, jobs, chunksize=8))
    else:
        per_frame = [_cut_frame(job) for job in jobs]
    index = [entry for frame in per_frame for entry in frame]
    accepted = [e for e in index if e["accepted"]]
    write_index(out_dir, index)
    write_flat_yaml(os.path.join(out_dir, "pairs.yaml"), {e["ir"]: e["rgb"] for e in accepted})
    write_flat_yaml(os.path.join(out_dir, "captions.yaml"), {e["ir"]: NEUTRAL_CAPTION for e in accepted})
    rejected: dict[str, int] = {}
    for e in index:
        if not e["accepted"]:
            rejected[e["reason"]] = rejected.get(e["reason"], 0) + 1
    return {"frames": len(rows), "tiles": len(index), "accepted": len(accepted), "rejected": rejected}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cut aligned square tiles of split-T RGB / IR pairs into a "
                                             "`brain flux2 finetune` paired training folder.")
    ap.add_argument("--splits", required=True, help="splits.json from rir_splits.py")
    ap.add_argument("--out", required=True, help="training folder to write")
    ap.add_argument("--size", type=int, default=512, help="tile side in pixels (default 512)")
    ap.add_argument("--datasets", default="", help="comma-separated dataset ids; default: all in splits.json")
    ap.add_argument("--limit", type=int, default=0, help="seeded random subset of this many frames")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--min-edge-corr", type=float, default=TileChecks.min_edge_corr)
    ap.add_argument("--max-shift-gain", type=float, default=TileChecks.max_shift_gain)
    a = ap.parse_args(argv)
    with open(a.splits) as fh:
        doc = json.load(fh)
    checks = TileChecks(min_edge_corr=a.min_edge_corr, max_shift_gain=a.max_shift_gain)
    counts = cut_tiles(doc, a.out, a.size, a.seed, a.limit, [d for d in a.datasets.split(",") if d], checks, a.workers)
    print(json.dumps(counts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
