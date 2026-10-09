#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements object-detection evaluation for its clients.
# If your team needs expertise in detector benchmarking and clustered
# significance testing, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Offline COCO-style detection scoring of a `brain yolov8 eval --dump-preds` file.

    brain yolov8 eval --weights W --data PACKED --split all --conf 0.001 --dump-preds preds.jsonl
    rir_eval.py preds.jsonl [--sequences PACKED/sequences.json]

The dump has one line per image, `{"image": idx, "gts": [{"class", "xyxy"}],
"preds": [{"class", "score", "xyxy"}]}`, boxes in the image's own pixels. This
module reproduces `eval::detection_report` (crates/eval) in numpy so the same
numbers can be recomputed on subsets of images, which is what a cluster
bootstrap needs:

* matching: predictions in descending score (ties by position); each takes the
  unmatched same-class ground truth of the SAME image with the highest IoU
  that reaches the threshold (a tie goes to the later ground truth); a ground
  truth is matched once. Nothing is ever matched across images.
* AP of a class with ground truth: precision-recall curve over the class's
  predictions ranked across all images, precision replaced by its right-to-left
  maximum, area under it (all-points). A class without ground truth is left out
  of the mean; one with ground truth and no hit counts 0.
* mAP@0.5:0.95 averages ten thresholds 0.50, 0.55, ..., 0.95.

Boxes, scores and the IoU are float32 as in the Rust code, so the matching is
the same decision for decision; the area under the curve is summed in float64
where Rust sums in float32, which differs by float32 rounding (about 1e-7).
tests/test_eval.py compares every number of the real binary's table, printed
to four decimals, when a binary is built.

Matching depends only on one image, so it is done once per run (`prepare`) and
the cluster bootstrap then re-weights images (`Prepared.maps`): an image with
weight 2 counts exactly like two copies of it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

IOU_STEPS = 10
IOU_THRESHOLDS = (np.float32(0.5) + np.float32(0.05) * np.arange(IOU_STEPS, dtype=np.float32)).astype(np.float32)


@dataclass(frozen=True)
class Image:
    """Ground truth and predictions of one image (float32 boxes, xyxy)."""
    index: int
    gt_class: np.ndarray
    gt_box: np.ndarray
    pred_class: np.ndarray
    pred_score: np.ndarray
    pred_box: np.ndarray

    @staticmethod
    def empty(index: int) -> "Image":
        z = np.zeros((0, 4), np.float32)
        return Image(index, np.zeros(0, np.int64), z, np.zeros(0, np.int64), np.zeros(0, np.float32), z)


@dataclass(frozen=True)
class ClassAP:
    ap50: float
    ap50_95: float


@dataclass(frozen=True)
class Report:
    map50: float
    map50_95: float
    precision50: float
    recall50: float
    per_class: dict  # class id -> ClassAP, only classes with ground truth
    n_images: int
    n_preds: int
    n_gts: int


# ------------------------------------------------------------------- reading


def _boxes(items, line_no: int, key: str) -> np.ndarray:
    boxes = np.zeros((len(items), 4), np.float32)
    for i, item in enumerate(items):
        xyxy = item.get("xyxy")
        if not (isinstance(xyxy, list) and len(xyxy) == 4 and all(isinstance(v, (int, float)) for v in xyxy)):
            raise ValueError(f"line {line_no}: {key}[{i}].xyxy must be 4 numbers")
        boxes[i] = xyxy
    return boxes


def _parse_image(row: dict, line_no: int) -> Image:
    try:
        index, gts, preds = row["image"], row["gts"], row["preds"]
        if not (isinstance(index, int) and isinstance(gts, list) and isinstance(preds, list)):
            raise TypeError("image must be an integer and gts, preds lists")
        gt_class = np.array([g["class"] for g in gts], np.int64)
        pred_class = np.array([p["class"] for p in preds], np.int64)
        pred_score = np.array([p["score"] for p in preds], np.float32)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"line {line_no}: malformed image record: {e!r}") from e
    return Image(index, gt_class, _boxes(gts, line_no, "gts"), pred_class, pred_score, _boxes(preds, line_no, "preds"))


def read_jsonl(path: str) -> list[Image]:
    """Images of a dump, in file order. A malformed line is an error naming it."""
    images = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {n}: not valid JSON: {e}") from e
            images.append(_parse_image(row, n))
    return images


def load_sequences(path: str) -> list[str]:
    """The sequence of each packed image index (`sequences.json` next to the packed dataset)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: not found; re-pack the evaluation set with rir_pack.py (it writes sequences.json)")
    with open(path) as fh:
        return json.load(fh)


def sequence_ids(images: list[Image], sequences: list[str]) -> list[str]:
    """The sequence of each image of a dump, by its dataset index."""
    out = []
    for im in images:
        if not 0 <= im.index < len(sequences):
            raise ValueError(f"image index {im.index} is outside the packed set of {len(sequences)} images")
        out.append(sequences[im.index])
    return out


# ------------------------------------------------------------------ matching


def _iou(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """float32 IoU matrix, operation for operation as yolov8::boxmath::iou."""
    a, b = preds[:, None, :], gts[None, :, :]
    iw = np.maximum(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), np.float32(0))
    ih = np.maximum(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), np.float32(0))
    inter = iw * ih
    area_a = np.maximum(a[..., 2] - a[..., 0], np.float32(0)) * np.maximum(a[..., 3] - a[..., 1], np.float32(0))
    area_b = np.maximum(b[..., 2] - b[..., 0], np.float32(0)) * np.maximum(b[..., 3] - b[..., 1], np.float32(0))
    return inter / np.maximum(area_a + area_b - inter, np.float32(1e-9))


def match_image(im: Image, thresholds: np.ndarray = IOU_THRESHOLDS) -> np.ndarray:
    """Boolean (n_preds, n_thresholds): is the prediction a true positive at each IoU threshold."""
    n_pred, n_gt = len(im.pred_score), len(im.gt_class)
    hit = np.zeros((n_pred, len(thresholds)), bool)
    if n_pred == 0 or n_gt == 0:
        return hit
    iou = np.where(im.pred_class[:, None] == im.gt_class[None, :], _iou(im.pred_box, im.gt_box), np.float32(-1))
    used = np.zeros((len(thresholds), n_gt), bool)
    for p in np.lexsort((np.arange(n_pred), -im.pred_score)):
        row = iou[p]
        if row.max() < thresholds[0]:
            continue
        for t, thr in enumerate(thresholds):
            free = ~used[t] & (row >= thr)
            if free.any():
                g = n_gt - 1 - int(np.argmax(np.where(free, row, np.float32(-1))[::-1]))  # best IoU, later one on a tie
                used[t, g] = True
                hit[p, t] = True
    return hit


# ------------------------------------------------------------------- scoring


def _class_ap(w: np.ndarray, tp_positions: list, n_gt: float) -> np.ndarray:
    """AP at every threshold of one class from its weights `w` in rank order.

    Recall only moves on a true positive, and precision after any rank is
    tp / (all predictions so far), so the area under the right-to-left
    maximum of precision needs only the true-positive ranks: with the running
    weight W of ALL predictions, AP = sum_k (w_k / n_gt) * max_{j >= k} tp_j / W_j
    over the true positives k. A weight is a multiplicity: w = 2 counts the
    prediction as two identical, adjacent ones.
    """
    total = np.cumsum(w)
    out = np.zeros(len(tp_positions))
    for t, pos in enumerate(tp_positions):
        if len(pos) == 0:
            continue
        wt = w[pos]
        denom = total[pos]
        precision = np.divide(np.cumsum(wt), denom, out=np.zeros(len(pos)), where=denom > 0)
        out[t] = (wt / n_gt) @ np.maximum.accumulate(precision[::-1])[::-1]
    return out


@dataclass
class Prepared:
    """One run's matches, ready to be scored under any image weights."""
    n_images: int
    gt_counts: np.ndarray  # (n_images, n_classes) ground truths per image and class
    n_preds: np.ndarray  # (n_classes,) predictions per class
    rank_image: dict  # class id -> image position of each prediction, in rank order (score desc, then file order)
    tp_positions: dict  # class id -> per threshold, the ranks that are true positives
    n_hits50: int  # true positives at IoU 0.5 over all classes

    def class_aps(self, weights: np.ndarray | None = None, nc: int | None = None) -> dict:
        """class id -> AP (T,) for every class with ground truth under `weights` (image multiplicities)."""
        w_img = np.ones(self.n_images) if weights is None else np.asarray(weights, np.float64)
        n_gt = w_img @ self.gt_counts
        out = {}
        for c in range(self.gt_counts.shape[1] if nc is None else min(nc, self.gt_counts.shape[1])):
            if n_gt[c] > 0:
                out[c] = _class_ap(w_img[self.rank_image[c]], self.tp_positions[c], n_gt[c])
        return out

    def maps(self, weights: np.ndarray | None = None, nc: int | None = None) -> tuple[float, float]:
        """(mAP@0.5, mAP@0.5:0.95) of the images counted `weights` times each; 0 when no class has ground truth."""
        aps = list(self.class_aps(weights, nc).values())
        if not aps:
            return 0.0, 0.0
        per_thr = np.mean(aps, axis=0)
        return float(per_thr[0]), float(per_thr.mean())


def prepare(images: list[Image]) -> Prepared:
    classes = [int(c) for im in images for c in (*im.gt_class, *im.pred_class)]
    n_classes = 1 + max(classes, default=-1)
    gt_counts = np.zeros((len(images), n_classes))
    for i, im in enumerate(images):
        np.add.at(gt_counts[i], im.gt_class, 1)
    hit = np.concatenate([match_image(im) for im in images] or [np.zeros((0, IOU_STEPS), bool)])
    image_of_pred = np.concatenate([np.full(len(im.pred_score), i, np.int64) for i, im in enumerate(images)]
                                   or [np.zeros(0, np.int64)])
    pred_class = np.concatenate([im.pred_class for im in images] or [np.zeros(0, np.int64)])
    score = np.concatenate([im.pred_score for im in images] or [np.zeros(0, np.float32)])
    order = np.lexsort((np.arange(len(score)), -score))
    ranked = {c: order[pred_class[order] == c] for c in range(n_classes)}
    return Prepared(len(images), gt_counts, np.bincount(pred_class, minlength=n_classes),
                    {c: image_of_pred[r] for c, r in ranked.items()},
                    {c: [np.nonzero(hit[r, t])[0] for t in range(IOU_STEPS)] for c, r in ranked.items()},
                    int(hit[:, 0].sum()))


def score(images: list[Image], nc: int | None = None) -> Report:
    """The Rust `detection_report::score`. `nc` limits class ids to 0..nc as the model's head does."""
    prep = prepare(images)
    aps = prep.class_aps(None, nc)
    classes = sorted(aps)
    heads = slice(None, nc)
    n_preds, n_gts = int(prep.n_preds[heads].sum()), int(prep.gt_counts[:, heads].sum())
    tp50 = sum(len(prep.tp_positions[c][0]) for c in range(len(prep.n_preds))[heads])
    per_class = {c: ClassAP(float(aps[c][0]), float(aps[c].mean())) for c in classes}
    map50, map50_95 = prep.maps(None, nc)
    return Report(map50, map50_95, tp50 / n_preds if n_preds else 0.0, tp50 / n_gts if n_gts else 0.0, per_class,
                  len(images), n_preds, n_gts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a `brain yolov8 eval --dump-preds` file offline.")
    ap.add_argument("preds", help="predictions jsonl")
    ap.add_argument("--nc", type=int, default=None, help="number of classes of the detector head (class ids 0..nc)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    r = score(read_jsonl(a.preds), a.nc)
    if a.json:
        print(json.dumps({"map50": r.map50, "map50_95": r.map50_95, "precision50": r.precision50, "recall50": r.recall50,
                          "per_class": {c: {"ap50": v.ap50, "ap50_95": v.ap50_95} for c, v in r.per_class.items()},
                          "n_images": r.n_images, "n_preds": r.n_preds, "n_gts": r.n_gts}))
        return 0
    print(f"mAP@0.5        {r.map50:.4f}\nmAP@0.5:0.95   {r.map50_95:.4f}\n"
          f"precision@0.5  {r.precision50:.4f}\nrecall@0.5     {r.recall50:.4f}\nclass  AP@0.5  AP@0.5:0.95")
    for c, v in sorted(r.per_class.items()):
        print(f"{c:<5}  {v.ap50:.4f}   {v.ap50_95:.4f}")
    print(f"preds {r.n_preds}  gts {r.n_gts}  (images {r.n_images})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
