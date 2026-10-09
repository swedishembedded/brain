# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements label-preservation checks for synthetic
# training images for its clients. If your team needs expertise in
# detector-training data validation or image-to-image translation quality
# control, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Label-preservation gates for a synthetic IR image made from a labelled RGB image.

A translator may move, drop or invent things, and the RGB image's boxes then
no longer describe the IR image. Each gate asks one question; an image that
trips any of them is rejected (`check_image`), and `RejectionStats` counts the
rejections per reason in the JSON that `rir_decide.py` reads for K4.

    global_shift         the whole image is within `max_shift_px` (2) of the RGB. Phase correlation
                         of the two edge maps (edges, because the grey levels of the modalities differ
                         and may be inverted). A featureless pair has no measurable shift: it is
                         reported as such and does not fail this gate (the edge gate sees it).
    box_edge_correlation inside each ground-truth box the IR edges follow the RGB edges at least as
                         well as the 10th percentile of REAL pairs do (`reference_distribution`,
                         computed on the real pairs of a manifest with `manifest_pairs`). Boxes
                         thinner than `min_box_px` are not evaluated.
    box_mask_iou         pluggable: a callable `mask_iou(rgb, ir, box) -> float | None` returning the
                         IoU of the object's mask segmented on the RGB image and on the synthetic IR
                         image (the pipeline wires a SAM 2 segmenter prompted with `box`); `rgb` is the
                         BGR uint8 image, `ir` the single-channel uint8 synthetic IR, `box` the integer
                         (x1, y1, x2, y2). None means "could not be computed": not a failure, kept as
                         None in the box record. Below `mask_iou_min` the box fails.
    hallucination        detections of a reference detector run on the synthetic IR image
                         (`Detection`, e.g. from `brain yolov8 detect`) that have no ground truth of
                         their class at `halluc_iou` and at least `halluc_conf` confidence.

Gates whose input is not given (no detections, no mask callable) are absent, not passed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np

import rir_data as D

REASONS = ("global_shift", "box_edge_correlation", "box_mask_iou", "hallucination")
MIN_PHASE_RESPONSE = 0.02  # below this phase-correlation peak there is nothing to measure a shift from

Box = tuple  # (class, x1, y1, x2, y2), pixels
MaskIou = Callable[[np.ndarray, np.ndarray, tuple], "float | None"]


@dataclass(frozen=True)
class GateConfig:
    edge_threshold: float  # 10th percentile of real pairs: see reference_distribution
    max_shift_px: float = 2.0
    min_box_px: int = 8
    halluc_conf: float = 0.5
    halluc_iou: float = 0.5
    mask_iou_min: float = 0.5


# ------------------------------------------------------------------- edge gate


def _same_size_gray(rgb: np.ndarray, ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g, i = D.to_gray(rgb), D.to_gray(ir)
    if g.shape != i.shape:
        i = cv2.resize(i, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
    return g, i


def box_edge_correlations(rgb: np.ndarray, ir: np.ndarray, boxes: list[Box], min_box_px: int = 8) -> list[float | None]:
    """Edge-map correlation of RGB luma and IR inside each box; None for a box thinner than `min_box_px`."""
    g, i = _same_size_gray(rgb, ir)
    eg, ei = D.edge_magnitude(g), D.edge_magnitude(i)
    h, w = eg.shape
    out = []
    for _, x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        out.append(None if min(x2 - x1, y2 - y1) < min_box_px else D.ncc(eg[y1:y2, x1:x2], ei[y1:y2, x1:x2]))
    return out


@dataclass(frozen=True)
class Reference:
    """Edge correlations of real RGB / IR pairs, and the threshold read off them."""
    values: tuple
    percentile: float
    threshold: float

    @property
    def n(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict:
        return {"values": list(self.values), "percentile": self.percentile, "threshold": self.threshold}

    @staticmethod
    def from_dict(d: dict) -> "Reference":
        return Reference(tuple(d["values"]), float(d["percentile"]), float(d["threshold"]))


def reference_distribution(pairs: Iterable, percentile: float = 10.0, min_box_px: int = 8) -> Reference:
    """Per-box edge correlations over real pairs `(rgb, ir, boxes)`; `threshold` is their `percentile`-th percentile."""
    values = [c for rgb, ir, boxes in pairs for c in box_edge_correlations(rgb, ir, boxes, min_box_px) if c is not None]
    if not values:
        raise ValueError("no boxes to build the reference distribution from: the real pairs have no box of at least "
                         f"{min_box_px} px")
    return Reference(tuple(values), percentile, float(np.percentile(values, percentile)))


def manifest_pairs(records: list[dict], classes: tuple[str, ...], limit: int = 0, seed: int = 1):
    """Real pairs `(rgb, ir, boxes)` of manifest records (seeded subset of `limit` when given), boxes of `classes` only."""
    rows = sorted(records, key=lambda r: (r["dataset"], r["id"]))
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    for row in rows:
        boxes = [(c, *xyxy) for c, *xyxy in D.class_boxes(row, classes)]
        if boxes:
            yield D.read_rgb(row), D.read_ir(row), boxes


# ----------------------------------------------------------------- shift gate


def global_shift(rgb: np.ndarray, ir: np.ndarray) -> tuple[float, float] | None:
    """(dx, dy) in pixels between the edge maps of the RGB luma and the IR, or None when they carry no common structure."""
    g, i = _same_size_gray(rgb, ir)
    eg, ei = D.edge_magnitude(g), D.edge_magnitude(i)
    if eg.std() < 1e-3 or ei.std() < 1e-3:
        return None
    window = cv2.createHanningWindow((eg.shape[1], eg.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(eg.astype(np.float32), ei.astype(np.float32), window)
    return None if response < MIN_PHASE_RESPONSE else (float(dx), float(dy))


# ------------------------------------------------------------- hallucinations


@dataclass(frozen=True)
class Detection:
    cls: int
    score: float
    box: tuple  # (x1, y1, x2, y2)

    @staticmethod
    def from_row(row) -> "Detection":
        """A row `[x1, y1, x2, y2, conf, class]` as `brain yolov8 detect` prints it."""
        x1, y1, x2, y2, conf, cls = row
        return Detection(int(cls), float(conf), (x1, y1, x2, y2))


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def find_hallucinations(detections: list[Detection], boxes: list[Box], conf_min: float = 0.5,
                        iou_min: float = 0.5) -> list[Detection]:
    """Detections of at least `conf_min` confidence with no ground truth of their class at IoU `iou_min`."""
    return [d for d in detections
            if d.score >= conf_min and not any(c == d.cls and _iou(d.box, (x1, y1, x2, y2)) >= iou_min
                                               for c, x1, y1, x2, y2 in boxes)]


# ----------------------------------------------------------------- whole image


@dataclass(frozen=True)
class BoxGate:
    box: tuple
    edge_correlation: float | None
    mask_iou: float | None
    reasons: tuple

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class ImageGate:
    shift: tuple | None
    boxes: list
    hallucinations: list
    hallucinations_checked: bool
    reasons: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.reasons


def check_image(rgb: np.ndarray, ir: np.ndarray, boxes: list[Box], config: GateConfig,
                detections: list[Detection] | None = None, mask_iou: MaskIou | None = None) -> ImageGate:
    """Run the gates on one pair. `detections=None` / `mask_iou=None` leave those gates out."""
    shift = global_shift(rgb, ir)
    reasons = []
    if shift is not None and float(np.hypot(*shift)) > config.max_shift_px:
        reasons.append("global_shift")
    correlations = box_edge_correlations(rgb, ir, boxes, config.min_box_px)
    gates = []
    for box, corr in zip(boxes, correlations):
        failed = []
        if corr is not None and corr < config.edge_threshold:
            failed.append("box_edge_correlation")
        iou = None if mask_iou is None else mask_iou(rgb, ir, tuple(int(v) for v in box[1:]))
        if iou is not None and iou < config.mask_iou_min:
            failed.append("box_mask_iou")
        gates.append(BoxGate(tuple(box), corr, iou, tuple(failed)))
    for reason in ("box_edge_correlation", "box_mask_iou"):
        if any(reason in g.reasons for g in gates):
            reasons.append(reason)
    found = [] if detections is None else find_hallucinations(detections, boxes, config.halluc_conf, config.halluc_iou)
    if found:
        reasons.append("hallucination")
    return ImageGate(shift, gates, found, detections is not None, reasons)


class RejectionStats:
    """Rejections of one arm's synthetic images, per reason. `to_dict` is the gate-statistics JSON of rir_decide."""

    def __init__(self, arm: str):
        self.arm = arm
        self.n_images = self.n_failed = self.n_boxes = self.n_boxes_failed = 0
        self.by_reason = {r: 0 for r in REASONS}
        self.shift_undetermined = self.boxes_not_evaluated = 0

    def add(self, gate: ImageGate) -> None:
        self.n_images += 1
        self.n_failed += not gate.passed
        for reason in gate.reasons:
            self.by_reason[reason] += 1
        self.shift_undetermined += gate.shift is None
        self.n_boxes += len(gate.boxes)
        self.n_boxes_failed += sum(not b.passed for b in gate.boxes)
        self.boxes_not_evaluated += sum(b.edge_correlation is None for b in gate.boxes)

    def to_dict(self) -> dict:
        return {"arm": self.arm, "n_images": self.n_images, "n_failed": self.n_failed,
                "fail_fraction": self.n_failed / self.n_images if self.n_images else None,
                "by_reason": dict(self.by_reason), "n_boxes": self.n_boxes, "n_boxes_failed": self.n_boxes_failed,
                "boxes_not_evaluated": self.boxes_not_evaluated, "shift_undetermined": self.shift_undetermined}
