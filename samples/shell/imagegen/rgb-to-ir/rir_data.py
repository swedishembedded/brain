# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset adapters and leakage-safe evaluation
# protocols for multimodal perception pipelines for its clients. If your team
# needs expertise in paired RGB / thermal-IR data, detector training sets or
# sensor-domain adaptation, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Image access and dataset-independent frame checks, driven by manifest records.

Reading honours the record's `ir_read` knobs: IR stored as a gray image, a
3-channel replicated gray image, or the 4th (alpha) channel of an RGBA file
(read with IMREAD_UNCHANGED, never through an alpha-premultiplying loader);
16-bit samples reduced to 8 bits by >> 8; black-hot sources inverted so every
IR image is white-hot (hot is bright).

The checks are properties of one frame or one pair, not of any dataset:

* `valid_fraction`: share of the frame the RGB occupies once black padding is
  ignored. An RGB shrunk into a sub-rectangle of a black frame (a field-of-view
  mismatch with the IR) scores low.
* `edge_alignment`: Sobel-magnitude correlation of the RGB and IR at zero
  shift and the best correlation over small shifts, a misregistration measure.
  Cross-modal edge correlation is inherently modest: read it relatively.
* `thumb` / `dhash`: frame similarity for sequence breaks and leak detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    raise ValueError(f"unsupported sample type {img.dtype}; expected 8- or 16-bit")


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return _to_uint8(img)


def read_rgb(rec: dict, reduce: int = 1) -> np.ndarray:
    """BGR uint8. A gray source is replicated; an alpha channel is dropped."""
    img = _read(rec["rgb"])
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else np.ascontiguousarray(img[..., :3])
    if reduce > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, w // reduce), max(1, h // reduce)), interpolation=cv2.INTER_AREA)
    return img


def read_ir(rec: dict) -> np.ndarray:
    """Single-channel uint8, white-hot."""
    spec = rec.get("ir_read") or {}
    img = _read(rec["ir"])
    if spec.get("channel", "gray") == "alpha":
        if img.ndim != 3 or img.shape[2] < 4:
            raise ValueError(f"{rec['ir']}: ir_read.channel is 'alpha' but the image has no alpha channel")
        img = np.ascontiguousarray(img[..., 3])
    elif img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return 255 - img if spec.get("invert") else img


def image_size(rec: dict) -> tuple[int, int]:
    """(width, height), from the record when present."""
    if rec.get("width") and rec.get("height"):
        return rec["width"], rec["height"]
    h, w = read_rgb(rec).shape[:2]
    return w, h


def class_boxes(rec: dict, classes: tuple[str, ...]) -> list[tuple[int, float, float, float, float]]:
    """(class_index, x1, y1, x2, y2) for boxes whose class is in `classes`; others are dropped."""
    index = {c: i for i, c in enumerate(classes)}
    return [(index[b["class"]], b["x1"], b["y1"], b["x2"], b["y2"]) for b in rec["boxes"] if b["class"] in index]


# ------------------------------------------------------------------ probes


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)


def valid_bbox(gray: np.ndarray, thr: int = 8) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the non-black content; black padding is excluded."""
    cols = np.where(gray.max(0) > thr)[0]
    rows = np.where(gray.max(1) > thr)[0]
    if len(cols) == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def valid_fraction(gray: np.ndarray) -> float:
    x0, y0, x1, y1 = valid_bbox(gray)
    return (x1 - x0) * (y1 - y0) / gray.size


def thumb(gray: np.ndarray, size: int = 16) -> np.ndarray:
    """Mean-removed, unit-norm size x size thumbnail: the dot product of two is
    their correlation."""
    t = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    t -= t.mean()
    n = float(np.linalg.norm(t))
    return (t / n).ravel() if n > 0 else t.ravel()


def dhash(gray: np.ndarray, size: int = 8) -> np.ndarray:
    t = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (t[:, 1:] > t[:, :-1]).ravel()


@dataclass(frozen=True)
class Probe:
    thumb: np.ndarray
    dhash: np.ndarray
    luminance: float  # mean gray over the non-black content box
    valid_fraction: float


def probe_rgb(rgb: np.ndarray) -> Probe:
    gray = to_gray(rgb)
    x0, y0, x1, y1 = valid_bbox(gray)
    return Probe(thumb(gray), dhash(gray), float(gray[y0:y1, x0:x1].mean()), valid_fraction(gray))


def _edges(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def edge_alignment(rgb: np.ndarray, ir: np.ndarray, search: int = 8, margin: int = 12) -> tuple[float, float, tuple[int, int]]:
    """(correlation at zero shift, best correlation, (dx, dy) of the best) over
    integer shifts of the IR within +-`search` px, on the RGB's non-black area."""
    g, i = to_gray(rgb), to_gray(ir)
    if g.shape != i.shape:
        i = cv2.resize(i, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
    eg, ei = _edges(g), _edges(i)
    x0, y0, x1, y1 = valid_bbox(g)
    h, w = eg.shape
    x0, y0 = max(x0 + margin, search), max(y0 + margin, search)
    x1, y1 = min(x1 - margin, w - search), min(y1 - margin, h - search)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0, 0.0, (0, 0)
    core = eg[y0:y1, x0:x1]
    zero = _ncc(core, ei[y0:y1, x0:x1])
    best, shift = zero, (0, 0)
    for dy in range(-search, search + 1, 2):
        for dx in range(-search, search + 1, 2):
            v = _ncc(core, ei[y0 + dy:y1 + dy, x0 + dx:x1 + dx])
            if v > best:
                best, shift = v, (dx, dy)
    return zero, best, shift
