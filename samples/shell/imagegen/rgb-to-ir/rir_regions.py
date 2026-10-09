#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements region-level thermal measurement for
# paired RGB / thermal-IR datasets for its clients. If your team needs
# expertise in robust radiometric contrast statistics or promptable
# segmentation pipelines, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Region contrast measured on the REAL IR of a tile.

For every ground-truth box of a tile the RGB tile is segmented by prompting a
segmenter with the box (rir_sam2.py: SAM 2 through the brain CLI or D-Bus).
The object's contrast is then read off the real IR tile, within the frame:

    c = (median IR inside the eroded mask - median IR in a ring around it) / MAD of the IR frame

* The mask is eroded by `erode_frac` of the object's equivalent diameter so the
  thermal point-spread skirt at the boundary does not count as the object.
* The ring spans `ring_frac` of the diameter, starts `gap_frac` of it outside
  the mask (the same skirt, from outside) and excludes every other object's
  mask, so a hot neighbour does not warm the reference.
* MAD is the frame's median absolute deviation, scaled by 1.4826 to a standard
  deviation for Gaussian data and floored at one grey level; c is therefore in
  units of the frame's own robust spread and needs no radiometric calibration.
* 8-bit IR medians lie on a half-grey-level grid, which would make the null
  distribution below lumpy, so the IR is first dequantised with a seeded
  uniform +-0.5 dither (unbiased, 0.29 grey levels of noise).

The noise floor tau is measured, not assumed. For an object of n eroded pixels,
tau is the 75th percentile of |c'| over random PAIRS of background patches of
the same area in the same frame, c' = (median(patch 1) - median(patch 2)) / MAD,
patches lying entirely outside every object (and its gap) and at most
`pair_reach` patch sides apart: the ring comparison is local, so its null is
too (pairs drawn from anywhere in the frame compare sky with ground and set a
floor that no local object can reach; `pair_reach=None` gives that variant).
It grows with local scene structure and shrinks with patch size, as it should. A statement is made only when |c| >= tau
(`polarity`); a frame with no room for background patches has no tau and makes
no statement at all.

Parts. A part grounder (`Grounder.ground`) boxes named parts of an object
(bonnet, windows); each part box is segmented in turn and measured exactly like
an object, against a ring that excludes other objects and sibling parts but not
its own parent. Parts are measured only when both a grounder and a part list
for the class are given; asking for parts without a grounder is an error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

import rir_tiles as T


class Segmenter(Protocol):
    def segment(self, bgr: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
        """One boolean mask (image-sized) per prompt box."""


class Grounder(Protocol):
    def ground(self, bgr: np.ndarray, box: tuple[float, float, float, float], parts: list[str]) -> dict:
        """{part name: (x1, y1, x2, y2) in image pixels, or None when not found} for one object."""


@dataclass(frozen=True)
class RegionParams:
    erode_frac: float = 0.08  # of the equivalent diameter sqrt(mask area)
    gap_frac: float = 0.08
    ring_frac: float = 0.5
    min_inside_px: int = 16
    min_ring_px: int = 48
    min_box_side: float = 10.0
    box_pad: float = 0.05  # a mask must stay within its box grown by this share of its size
    min_box_fill: float = 0.1  # ... and cover at least this share of the box
    n_pairs: int = 64
    pair_percentile: float = 75.0
    pair_reach: float | None = 2.0  # max distance between the two patches of a pair, in patch sides; None = anywhere
    mad_floor: float = 1.0  # grey levels


def dequantize(ir: np.ndarray, seed: int) -> np.ndarray:
    """float32 IR with a seeded uniform +-0.5 dither, see the module docstring."""
    rng = np.random.default_rng([seed, 0x1D17])
    return ir.astype(np.float32) + rng.uniform(-0.5, 0.5, ir.shape).astype(np.float32)


def frame_mad(ir: np.ndarray, params: RegionParams) -> float:
    return max(float(1.4826 * np.median(np.abs(ir - np.median(ir)))), params.mad_floor)


def _grow(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _shrink(mask: np.ndarray, radius: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def object_contrast(ir: np.ndarray, mask: np.ndarray, other_masks: list[np.ndarray], params: RegionParams) -> dict | None:
    """Contrast of one mask against its ring; None when the eroded mask or the ring is too small to be stable."""
    diameter = float(np.sqrt(mask.sum()))
    inside = _shrink(mask, max(1, round(params.erode_frac * diameter)))
    if inside.sum() < params.min_inside_px:
        return None
    gap = max(1, round(params.gap_frac * diameter))
    ring = _grow(mask, gap + max(3, round(params.ring_frac * diameter))) & ~_grow(mask, gap)
    for other in other_masks:
        ring &= ~_grow(other, gap)
    if ring.sum() < params.min_ring_px:
        return None
    mad = frame_mad(ir, params)
    inside_median, ring_median = float(np.median(ir[inside])), float(np.median(ir[ring]))
    return {"contrast": (inside_median - ring_median) / mad, "mad": mad, "inside_median": inside_median,
            "ring_median": ring_median, "inside_px": int(inside.sum()), "ring_px": int(ring.sum())}


def noise_floor(ir: np.ndarray, object_masks: list[np.ndarray], n_pixels: int, rng: np.random.Generator,
                params: RegionParams) -> float | None:
    """tau: the 75th percentile of |c'| between random background patch pairs of `n_pixels` pixels."""
    h, w = ir.shape
    side = max(3, round(np.sqrt(n_pixels)))
    if side >= min(h, w):
        return None
    background = np.ones((h, w), bool)
    for m in object_masks:
        background &= ~_grow(m, max(1, round(params.gap_frac * side)))
    integral = cv2.integral(background.astype(np.uint8))
    free = np.zeros((h, w), bool)  # free[y, x]: the side x side window with top-left (y, x) is all background
    free[: h - side + 1, : w - side + 1] = (
        integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side] + integral[:-side, :-side]
    ) == side * side
    ys, xs = np.nonzero(free)
    if len(ys) < 2:
        return None
    n = params.n_pairs * 8
    first = rng.integers(0, len(ys), n)
    py, px = ys[first], xs[first]
    if params.pair_reach is None:
        second = rng.integers(0, len(ys), n)
        qy, qx = ys[second], xs[second]
    else:
        # The ring comparison is local, so its null is too: the partner lies within `pair_reach` sides of the first.
        reach = max(1, round(params.pair_reach * side))
        qy, qx = py + rng.integers(-reach, reach + 1, n), px + rng.integers(-reach, reach + 1, n)
    inside = (qy >= 0) & (qy < h) & (qx >= 0) & (qx < w)
    ok = inside & (np.abs(qy - py) >= side) | inside & (np.abs(qx - px) >= side)
    ok[inside] &= free[qy[inside], qx[inside]]
    py, px, qy, qx = py[ok][: params.n_pairs], px[ok][: params.n_pairs], qy[ok][: params.n_pairs], qx[ok][: params.n_pairs]
    if len(py) < max(8, params.n_pairs // 4):
        return None
    mad = frame_mad(ir, params)
    diffs = [abs(float(np.median(ir[y1:y1 + side, x1:x1 + side]) - np.median(ir[y2:y2 + side, x2:x2 + side]))) / mad
             for y1, x1, y2, x2 in zip(py, px, qy, qx)]
    return float(np.percentile(diffs, params.pair_percentile))


def polarity(contrast: float | None, tau: float | None) -> str | None:
    """warmer / cooler when |c| reaches tau, same below it, None when either is unknown."""
    if contrast is None or tau is None:
        return None
    if contrast >= tau:
        return "warmer"
    if contrast <= -tau:
        return "cooler"
    return "same"


def _fit_mask(mask: np.ndarray, box, params: RegionParams) -> np.ndarray | None:
    """The mask clipped to its prompt box grown by `box_pad`, or None when the segmenter left the box."""
    x1, y1, x2, y2 = box
    px, py = params.box_pad * (x2 - x1), params.box_pad * (y2 - y1)
    h, w = mask.shape
    window = np.zeros_like(mask)
    window[max(0, int(y1 - py)):min(h, int(np.ceil(y2 + py))), max(0, int(x1 - px)):min(w, int(np.ceil(x2 + px)))] = True
    clipped = mask & window
    if mask.sum() == 0 or clipped.sum() < 0.9 * mask.sum() or clipped.sum() < params.min_box_fill * (x2 - x1) * (y2 - y1):
        return None
    return clipped


def _skipped(cls: str, box, reason: str) -> dict:
    return {"class": cls, "box": [round(float(v), 1) for v in box], "skipped": reason, "contrast": None, "tau": None,
            "polarity": None}


def _measured(cls: str, box, mask: np.ndarray, c: dict, tau: float | None) -> dict:
    return {"class": cls, "box": [round(float(v), 1) for v in box], "mask_px": int(mask.sum()),
            "contrast": round(c["contrast"], 4), "tau": None if tau is None else round(tau, 4),
            "polarity": polarity(c["contrast"], tau), "inside_median": round(c["inside_median"], 2),
            "ring_median": round(c["ring_median"], 2), "mad": round(c["mad"], 3), "inside_px": c["inside_px"],
            "ring_px": c["ring_px"]}


def _measure_mask(ir, cls, box, mask, others, all_objects, rng, params) -> dict:
    c = object_contrast(ir, mask, others, params)
    if c is None:
        return _skipped(cls, box, "mask_or_ring_too_small")
    return _measured(cls, box, mask, c, noise_floor(ir, all_objects, c["inside_px"], rng, params))


def _measure_parts(rgb, ir, cls, box, obj_mask, other_objects, all_objects, parts, segmenter, grounder, rng, params):
    found = {p: b for p, b in grounder.ground(rgb, tuple(box), parts).items() if b is not None}
    names = sorted(found)
    masks = segmenter.segment(rgb, [found[p] for p in names]) if names else []
    fitted = [(p, _fit_mask(m, found[p], params)) for p, m in zip(names, masks)]
    fitted = [(p, m & obj_mask) for p, m in fitted if m is not None]
    out = []
    for p, m in fitted:
        siblings = [s for q, s in fitted if q != p]
        entry = _measure_mask(ir, cls, found[p], m, other_objects + siblings, all_objects, rng, params)
        out.append({**entry, "part": p})
    return out


def measure_tile(rgb: np.ndarray, ir: np.ndarray, boxes: list[dict], segmenter: Segmenter, seed: int,
                 params: RegionParams = RegionParams(), grounder: Grounder | None = None,
                 parts_by_class: dict[str, list[str]] | None = None) -> dict:
    """Contrast, tau and polarity of every box of a tile (and of its parts, when asked for).

    `rgb` is the BGR tile the segmenter sees, `ir` the single-channel real IR
    tile. `seed` fixes the dither and the patch sampling.
    """
    if parts_by_class and grounder is None:
        raise ValueError("parts were requested but no part grounder was given; part-level regions need one")
    ir = dequantize(ir, seed)
    prompts = [(b, (b["x1"], b["y1"], b["x2"], b["y2"])) for b in boxes]
    usable = [(b, bx) for b, bx in prompts if min(bx[2] - bx[0], bx[3] - bx[1]) >= params.min_box_side]
    masks = segmenter.segment(rgb, [bx for _, bx in usable]) if usable else []
    fitted = {i: _fit_mask(m, bx, params) for i, ((_, bx), m) in enumerate(zip(usable, masks))}
    all_objects = [m for m in fitted.values() if m is not None]
    rng = np.random.default_rng([seed, 0x70A5])
    objects, slot = [], 0
    for b, bx in prompts:
        if (b, bx) not in usable:
            objects.append(_skipped(b["class"], bx, "box_too_small"))
            continue
        i, slot = slot, slot + 1
        mask = fitted[i]
        if mask is None:
            objects.append(_skipped(b["class"], bx, "mask_does_not_fit_box"))
            continue
        others = [m for j, m in enumerate(fitted.values()) if j != i and m is not None]
        entry = _measure_mask(ir, b["class"], bx, mask, others, all_objects, rng, params)
        part_names = (parts_by_class or {}).get(b["class"])
        if part_names and "skipped" not in entry:
            entry["parts"] = _measure_parts(rgb, ir, b["class"], bx, mask, others, all_objects, part_names, segmenter,
                                            grounder, rng, params)
        objects.append(entry)
    return {"objects": objects}


def tile_seed(seed: int, name: str) -> int:
    return zlib.crc32(f"{seed}/{name}".encode())


def measure_set(tiles_dir: str, segmenter: Segmenter, seed: int = 1, params: RegionParams = RegionParams(),
                limit: int = 0, grounder: Grounder | None = None, parts_by_class: dict[str, list[str]] | None = None,
                log=lambda msg: None) -> dict:
    """Measure the accepted tiles of a tile folder that have no `regions/<name>.json` yet.

    Resumable: a tile whose JSON exists is left alone, so an interrupted run
    continues where it stopped. `regions/summary.json` accumulates the tile and
    box counts and the seconds spent, i.e. the segmentation throughput. Returns
    this run's numbers.
    """
    out_dir = os.path.join(tiles_dir, "regions")
    os.makedirs(out_dir, exist_ok=True)
    todo = [e for e in T.read_index(tiles_dir) if e["accepted"] and e["boxes"]
            and not os.path.isfile(os.path.join(out_dir, f"{e['name']}.json"))]
    if limit:
        todo = todo[:limit]
    run = {"tiles": 0, "boxes": 0, "seconds": 0.0}
    for entry in todo:
        rgb = cv2.imread(os.path.join(tiles_dir, entry["rgb"]), cv2.IMREAD_COLOR)
        ir = cv2.imread(os.path.join(tiles_dir, entry["ir"]), cv2.IMREAD_UNCHANGED)[..., 0]
        start = time.perf_counter()
        doc = measure_tile(rgb, ir, entry["boxes"], segmenter, tile_seed(seed, entry["name"]), params, grounder,
                           parts_by_class)
        run["seconds"] += time.perf_counter() - start
        run["tiles"] += 1
        run["boxes"] += len(entry["boxes"])
        with open(os.path.join(out_dir, f"{entry['name']}.json"), "w") as fh:
            json.dump({"tile": entry["name"], **doc}, fh)
        if run["tiles"] % 25 == 0:
            log(f"measured {run['tiles']}/{len(todo)} tiles, {run['seconds'] / run['boxes']:.2f} s per box")
    path = os.path.join(out_dir, "summary.json")
    total = {"tiles": 0, "boxes": 0, "seconds": 0.0}
    if os.path.isfile(path):
        with open(path) as fh:
            total = json.load(fh)
    total = {k: total[k] + run[k] for k in run}
    if run["tiles"]:
        total["seconds_per_box"] = total["seconds"] / total["boxes"]
        with open(path, "w") as fh:
            json.dump(total, fh)
    return {**run, "seconds_per_box": run["seconds"] / run["boxes"] if run["boxes"] else None}


def main(argv=None) -> int:
    import rir_sam2 as S

    ap = argparse.ArgumentParser(description="Measure per-object IR contrast of a tile folder (tiles.jsonl) "
                                             "with SAM 2 masks prompted by the ground-truth boxes.")
    ap.add_argument("tiles", help="tile folder written by rir_tiles.py")
    ap.add_argument("--backend", choices=("cli", "dbus"), default="dbus",
                    help="dbus: one resident SAM 2 (needs a running `brain serve --dbus`, see --dbus-address); "
                         "cli: one `brain sam2 segment` process per box (slow: every process loads the model)")
    ap.add_argument("--dbus-address", default=os.environ.get("DBUS_SESSION_BUS_ADDRESS", "SESSION"))
    ap.add_argument("--brain", default="brain", help="the brain binary, for --backend cli")
    ap.add_argument("--brain-args", default="", help="global flags before the verb, e.g. '--device gpu1 --backend cuda'")
    ap.add_argument("--variant", default="tiny", choices=("tiny", "large"))
    ap.add_argument("--parts", default="", help="class=part,part;class=part ... part-level regions; needs --grounder")
    ap.add_argument("--grounder", choices=("florence2",), help="part grounder (Florence-2 weights through the brain CLI)")
    ap.add_argument("--limit", type=int, default=0, help="measure at most this many not yet measured tiles")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.backend == "dbus":
        segmenter = S.DbusSegmenter(a.dbus_address, a.variant)
    else:
        segmenter = S.CliSegmenter(a.brain, a.brain_args.split(), a.variant)
    grounder = S.CliGrounder(a.brain, a.brain_args.split()) if a.grounder else None
    parts = {k: v.split(",") for k, v in (item.split("=") for item in a.parts.split(";") if item)} if a.parts else None
    run = measure_set(a.tiles, segmenter, a.seed, RegionParams(), a.limit, grounder, parts,
                      log=lambda m: print(m, file=sys.stderr, flush=True))
    print(json.dumps(run), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
