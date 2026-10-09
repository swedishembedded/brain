# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements object-detection evaluation for its clients.
# If your team needs expertise in detector benchmarking and clustered
# significance testing, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Synthetic detector outputs with a planted quality, for the statistics tests.

A dataset is a list of sequences, each a few frames with the same two
ground-truth boxes (neighbouring video frames are near-duplicates). A
sequence carries a difficulty shared by all its frames and by every arm. An
arm detects a ground truth with probability sigmoid(skill - difficulty) and
adds a few random false positives; a seed redraws only the arm's own luck.
So frames of one sequence are strongly dependent and arms are paired, which is
what a frame-level bootstrap gets wrong and a sequence-level one gets right.
"""
from __future__ import annotations

import json

import numpy as np

import rir_eval as E

GT_BOXES = (np.array([10, 10, 50, 50], np.float32), np.array([70, 20, 110, 70], np.float32))


class Dataset:
    def __init__(self, n_sequences: int, frames: int, seed: int, spread: float = 1.0):
        rng = np.random.default_rng(seed)
        self.difficulty = rng.normal(0, spread, n_sequences)
        self.frames = frames
        self.sequence_of_image = [f"s{s:03d}" for s in range(n_sequences) for _ in range(frames)]

    @property
    def n_images(self) -> int:
        return len(self.sequence_of_image)


def detect(ds: Dataset, skill: float, seed: int, false_positives: int = 2) -> list[E.Image]:
    """One detector's output on `ds`."""
    rng = np.random.default_rng(seed)
    images = []
    for i in range(ds.n_images):
        p_detect = 1 / (1 + np.exp(-(skill - ds.difficulty[i // ds.frames])))
        boxes, classes, scores = [], [], []
        for gt in GT_BOXES:
            if rng.random() < p_detect:
                boxes.append(gt + rng.normal(0, 0.5, 4).astype(np.float32))
                classes.append(0)
                scores.append(rng.uniform(0.4, 1.0))
        for _ in range(false_positives):
            xy = rng.uniform(0, 80, 2).astype(np.float32)
            boxes.append(np.array([xy[0], xy[1], xy[0] + 30, xy[1] + 30], np.float32) + 200)  # far from every ground truth
            classes.append(0)
            scores.append(rng.uniform(0.0, 0.7))
        images.append(E.Image(i, np.zeros(len(GT_BOXES), np.int64), np.stack(GT_BOXES),
                              np.array(classes, np.int64), np.array(scores, np.float32),
                              np.stack(boxes) if boxes else np.zeros((0, 4), np.float32)))
    return images


def arm(ds: Dataset, skill: float, seeds=(1, 2, 3), salt: int = 0) -> list[E.Prepared]:
    """One prepared run per training seed."""
    return [E.prepare(detect(ds, skill, 1000 * salt + s)) for s in seeds]


def write_dump(path: str, images: list[E.Image]) -> None:
    """The jsonl `brain yolov8 eval --dump-preds` writes."""
    with open(path, "w") as fh:
        for im in images:
            fh.write(json.dumps({
                "image": im.index,
                "gts": [{"class": int(c), "xyxy": [float(v) for v in b]} for c, b in zip(im.gt_class, im.gt_box)],
                "preds": [{"class": int(c), "score": float(s), "xyxy": [float(v) for v in b]}
                          for c, s, b in zip(im.pred_class, im.pred_score, im.pred_box)]}) + "\n")
