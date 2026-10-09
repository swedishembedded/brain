#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset inspection tooling for paired
# RGB / thermal-IR training sets for its clients. If your team needs
# expertise in auditing instruction-conditioned image-editing data, you can
# procure our services by sending an email to info@swedishembedded.com.

"""A contact sheet of random tiles: RGB | IR side by side, the instruction
caption under each pair, and the object masks outlined on both images in the
colour of the measured polarity (red warmer, blue cooler, grey same, white not
stated). The masks are segmented again for the sheet, since the per-tile JSON
stores their statistics and not their pixels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import cv2
import numpy as np

import rir_regions as R
import rir_tiles as T

COLOURS = {"warmer": (60, 60, 255), "cooler": (255, 140, 40), "same": (170, 170, 170), None: (255, 255, 255)}  # BGR


def outline(img: np.ndarray, mask: np.ndarray, colour) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, colour, 2)


def render_cell(tiles_dir: str, entry: dict, caption: str, segmenter, panel: int) -> np.ndarray:
    rgb = cv2.imread(os.path.join(tiles_dir, entry["rgb"]), cv2.IMREAD_COLOR)
    ir = cv2.imread(os.path.join(tiles_dir, entry["ir"]), cv2.IMREAD_COLOR)
    with open(os.path.join(tiles_dir, "regions", f"{entry['name']}.json")) as fh:
        objects = json.load(fh)["objects"]
    boxes = [o["box"] for o in objects]
    masks = segmenter.segment(rgb, boxes) if boxes else []
    for o, mask in zip(objects, masks):
        fitted = R._fit_mask(mask, o["box"], R.RegionParams())
        if fitted is not None:
            for panel_img in (rgb, ir):
                outline(panel_img, fitted, COLOURS[o["polarity"]])
    pair = np.hstack([cv2.resize(rgb, (panel, panel), interpolation=cv2.INTER_AREA),
                      cv2.resize(ir, (panel, panel), interpolation=cv2.INTER_AREA)])
    text = np.zeros((70, pair.shape[1], 3), np.uint8)
    for k, line in enumerate(textwrap.wrap(caption, width=max(20, pair.shape[1] // 7))[:5]):
        cv2.putText(text, line, (6, 14 + 13 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([pair, text])


def contact_sheet(tiles_dir: str, out_path: str, segmenter, count: int = 12, columns: int = 3, panel: int = 256,
                  seed: int = 1) -> list[str]:
    """Write the sheet; returns the names of the tiles on it."""
    captions = {}
    with open(os.path.join(tiles_dir, "captions.yaml")) as fh:
        for line in fh:
            key, value = line.rstrip("\n").split(": ", 1)
            captions[key] = json.loads(value)
    candidates = [e for e in T.read_index(tiles_dir)
                  if e["accepted"] and os.path.isfile(os.path.join(tiles_dir, "regions", f"{e['name']}.json"))]
    pick = np.random.default_rng(seed).choice(len(candidates), min(count, len(candidates)), replace=False)
    chosen = [candidates[i] for i in sorted(pick)]
    cells = [render_cell(tiles_dir, e, captions[e["ir"]], segmenter, panel) for e in chosen]
    while len(cells) % columns:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)]
    if not cv2.imwrite(out_path, np.vstack(rows)):
        raise OSError(f"could not write {out_path}")
    return [e["name"] for e in chosen]


def main(argv=None) -> int:
    import rir_sam2 as S

    ap = argparse.ArgumentParser(description="Contact sheet of random tiles with captions and mask overlays.")
    ap.add_argument("tiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dbus-address", default=os.environ.get("DBUS_SESSION_BUS_ADDRESS", "SESSION"))
    a = ap.parse_args(argv)
    names = contact_sheet(a.tiles, a.out, S.DbusSegmenter(a.dbus_address), a.count, seed=a.seed)
    print(json.dumps(names), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
