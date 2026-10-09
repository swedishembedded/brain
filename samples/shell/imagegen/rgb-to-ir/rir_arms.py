#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements synthetic thermal-image rendering and sensor
# modelling for its clients. If your team needs expertise in thermal-IR
# simulation or detector training-data synthesis, you can procure our
# services by sending an email to info@swedishembedded.com.

"""The training-image arms that need no generative model.

Detectors are fine-tuned on different renderings of the SAME detector-training
frames (split S) and all scored on real held-out IR. Arms rendered here:

    a1  RGB as is (3-channel; the packer feeds it to the detector unchanged)
    a2  real IR twin of the frame - the upper bound
    b1  grayscale of the RGB
    b2  inverted grayscale (black-hot rendering of b1)
    b3  b1 -> CLAHE -> sensor model (blur, contrast match, noise, stripes)
    b4  semantic renderer: class-prior intensities painted from the GT boxes
        on a cold-sky gradient, then the sensor model

IR output is 8-bit, single channel, white-hot (hot is bright).

Sensor model. Fitted per dataset on real IR of split T ONLY (the fit refuses
any other split), from single frames, with estimators that do not need a
clean reference:

* noise_sigma: Immerkaer's 3x3 Laplacian-difference estimator, as a robust
  (MAD) spread. Its kernel has zero response to column-constant patterns, so
  stripes do not inflate it.
* stripe_amplitude: spread of the column means after removing their 9-column
  running mean, less the noise share, i.e. the std of per-column offsets.
  Scene content that varies column to column inflates it, so on real frames
  it is an upper bound.
* blur_sigma: edge-spread. For an edge blurred by sigma the peak response of
  a Gaussian derivative at scale s falls as 1/sqrt(sigma^2 + s^2); the ratio
  of the responses at two scales over the strongest edges yields sigma. The
  rendered gray already has its own blur, so the model stores the EXTRA blur
  sqrt(max(sigma_ir^2 - sigma_rgb^2, 0)) measured on the T pairs.
* mean_mu/mean_sd, std_mu/std_sd: spread of the per-frame mean and standard
  deviation of the real IR; B3 draws its target contrast from them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from dataclasses import asdict, dataclass

import cv2
import numpy as np

import rir_data as D

DEFAULT_CLASSES = ("person", "car", "bicycle")
ARMS = ("a1", "a2", "b1", "b2", "b3", "b4")
MODEL_FIELDS = ("blur_sigma", "noise_sigma", "stripe_amplitude", "mean_mu", "mean_sd", "std_mu", "std_sd")

CLAHE_CLIP, CLAHE_TILE = 2.0, 8  # fixed, not fitted

# B4 class priors (8-bit white-hot levels) and background. People are the
# hottest, a car's body sits between a warm engine and a cool shell, a
# bicycle is a thin, mostly cool frame. The sky is colder than the ground
# everywhere, as in typical outdoor thermal scenes. A class with no entry is
# not painted.
CLASS_LEVEL = {"person": 215.0, "car": 165.0, "bicycle": 150.0}
SKY_TOP, SKY_HORIZON, GROUND_HORIZON, GROUND_BOTTOM = 30.0, 60.0, 95.0, 125.0
HORIZON_FRACTION = 0.45
CAR_SHAPE_INSET = 0.92


# ------------------------------------------------------------------ gray arms


def render_b1(rgb: np.ndarray) -> np.ndarray:
    return D.to_gray(rgb)


def render_b2(rgb: np.ndarray) -> np.ndarray:
    return 255 - render_b1(rgb)


# --------------------------------------------------------------- sensor model


@dataclass(frozen=True)
class SensorModel:
    blur_sigma: float
    noise_sigma: float
    stripe_amplitude: float
    mean_mu: float
    mean_sd: float
    std_mu: float
    std_sd: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SensorModel":
        return SensorModel(**{k: float(d[k]) for k in MODEL_FIELDS})


def _robust_sd(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def estimate_noise_sigma(gray: np.ndarray) -> float:
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)  # |k|_2 = 6
    resp = cv2.filter2D(gray.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return _robust_sd(resp) / 6.0


def estimate_stripe_amplitude(gray: np.ndarray, noise_sigma: float) -> float:
    cols = gray.astype(np.float32).mean(axis=0)
    resid = cols - cv2.blur(cols.reshape(1, -1), (9, 1), borderType=cv2.BORDER_REFLECT).ravel()
    # iid offsets of std a leave a residual of variance a^2 * 8/9 after the 9-tap mean is removed.
    var = _robust_sd(resid) ** 2 * 9 / 8 - noise_sigma ** 2 / gray.shape[0]
    return float(np.sqrt(max(var, 0.0)))


def _gradient_magnitude(g: np.ndarray, sigma: float) -> np.ndarray:
    k = np.array([[-0.5, 0.0, 0.5]], np.float32)
    b = cv2.GaussianBlur(g, (0, 0), sigma)
    return np.hypot(cv2.filter2D(b, -1, k), cv2.filter2D(b, -1, k.T))


def estimate_blur_sigma(gray: np.ndarray, s1: float = 1.0, s2: float = 2.5, edge_quantile: float = 0.98) -> float:
    """Effective edge-spread sigma (includes about 0.7 px of pixel-grid blur
    that even a perfectly sharp edge shows; callers compare in quadrature)."""
    g = gray.astype(np.float32)
    m1, m2 = _gradient_magnitude(g, s1), _gradient_magnitude(g, s2)
    edges = m1 >= max(np.quantile(m1, edge_quantile), 1.0)
    if not edges.any():
        return 0.0
    r2 = float(np.median(m1[edges] / np.maximum(m2[edges], 1e-6))) ** 2
    if r2 <= 1.0:
        return float(s2)  # no sharper at the fine scale: at least as blurred as the coarse one
    return float(np.sqrt(max((s2 ** 2 - r2 * s1 ** 2) / (r2 - 1.0), 0.0)))


def fit_sensor_model(pairs) -> SensorModel:
    """Fit from an iterable of (rgb_bgr, ir_gray) pairs of the SAME size."""
    blur_ir, blur_rgb, noise, stripe, mean, std = [], [], [], [], [], []
    for rgb, ir in pairs:
        gray = D.to_gray(rgb)
        n = estimate_noise_sigma(ir)
        noise.append(n)
        stripe.append(estimate_stripe_amplitude(ir, n))
        blur_ir.append(estimate_blur_sigma(ir))
        blur_rgb.append(estimate_blur_sigma(gray))
        mean.append(float(ir.mean()))
        std.append(float(ir.std()))
    if not noise:
        raise ValueError("no frame pairs to fit the sensor model on")
    extra_blur = float(np.sqrt(max(np.median(blur_ir) ** 2 - np.median(blur_rgb) ** 2, 0.0)))
    sd = lambda v: float(np.std(v)) if len(v) > 1 else 0.0
    return SensorModel(extra_blur, float(np.median(noise)), float(np.median(stripe)),
                       float(np.median(mean)), sd(mean), float(np.median(std)), sd(std))


def fit_from_splits(doc: dict, dataset: str, read_pair, max_frames: int, seed: int) -> SensorModel:
    """Fit on up to `max_frames` seeded-random usable split-T frames of
    `dataset`. `read_pair(row) -> (rgb, ir)` is called for those rows only."""
    t_rows = [r for r in doc["frames"] if r["dataset"] == dataset and r["usable"] and r["split"] == "T"]
    if not t_rows:
        raise ValueError(f"no usable split-T frames for {dataset}: refusing to fit the sensor model elsewhere")
    t_rows.sort(key=lambda r: r["id"])
    pick = np.random.default_rng([seed, zlib.crc32(dataset.encode())]).permutation(len(t_rows))[:max_frames]
    return fit_sensor_model(read_pair(t_rows[i]) for i in sorted(pick))


def fit_dataset_models(doc: dict, datasets, max_frames: int, seed: int) -> dict[str, SensorModel]:
    def read_pair(row):
        return D.read_rgb(row), D.read_ir(row)

    return {ds: fit_from_splits(doc, ds, read_pair, max_frames, seed) for ds in datasets}


def apply_sensor(gray: np.ndarray, model: SensorModel, rng: np.random.Generator) -> np.ndarray:
    """Blur, then additive noise, then per-column stripe offsets; uint8."""
    img = gray.astype(np.float32)
    if model.blur_sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), model.blur_sigma)
    img = img + rng.normal(0.0, model.noise_sigma, img.shape).astype(np.float32)
    img = img + rng.normal(0.0, model.stripe_amplitude, (1, img.shape[1])).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------- B3/B4


def render_b3(rgb: np.ndarray, model: SensorModel, rng: np.random.Generator) -> np.ndarray:
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE)).apply(render_b1(rgb))
    img = gray.astype(np.float32)
    if model.blur_sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), model.blur_sigma)
    target_mean = rng.normal(model.mean_mu, model.mean_sd)
    target_std = max(rng.normal(model.std_mu, model.std_sd), 1.0)
    img = (img - img.mean()) / max(float(img.std()), 1e-3) * target_std + target_mean
    img = img + rng.normal(0.0, model.noise_sigma, img.shape).astype(np.float32)
    img = img + rng.normal(0.0, model.stripe_amplitude, (1, img.shape[1])).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _background(height: int, width: int) -> np.ndarray:
    horizon = max(1, int(round(height * HORIZON_FRACTION)))
    col = np.empty(height, np.float32)
    col[:horizon] = np.linspace(SKY_TOP, SKY_HORIZON, horizon)
    col[horizon:] = np.linspace(GROUND_HORIZON, GROUND_BOTTOM, height - horizon)
    return np.repeat(col[:, None], width, axis=1)


def render_b4(height: int, width: int, boxes, model: SensorModel, rng: np.random.Generator,
              classes=DEFAULT_CLASSES) -> np.ndarray:
    """Cold-sky gradient + class-prior regions from GT boxes + sensor model.

    Persons and bicycles are painted as the ellipse inscribed in the box,
    cars as the box inset by CAR_SHAPE_INSET. Larger boxes are painted first
    so a small object in front of a large one stays visible."""
    img = _background(height, width)
    for cls, x1, y1, x2, y2 in sorted(boxes, key=lambda b: -(b[3] - b[1]) * (b[4] - b[2])):
        name = classes[int(cls)]
        level = CLASS_LEVEL.get(name)
        if level is None:
            continue
        cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
        mask = np.zeros((height, width), np.uint8)
        if name == "car":
            p1 = (int(round(cx - hw * CAR_SHAPE_INSET)), int(round(cy - hh * CAR_SHAPE_INSET)))
            p2 = (int(round(cx + hw * CAR_SHAPE_INSET)), int(round(cy + hh * CAR_SHAPE_INSET)))
            cv2.rectangle(mask, p1, p2, 1, thickness=-1)
        else:
            cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (max(1, int(round(hw))), max(1, int(round(hh)))),
                        0, 0, 360, 1, thickness=-1)
        img[mask > 0] = level
    return apply_sensor(np.clip(img, 0, 255).astype(np.uint8), model, rng)


# ------------------------------------------------------------------- rendering


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _frame_rng(seed: int, dataset: str, frame_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{dataset}/{frame_id}".encode())])


def select_rows(doc: dict, split: str, datasets, limit: int, seed: int) -> list[dict]:
    rows = sorted((r for r in doc["frames"] if r["usable"] and r["split"] == split and r["dataset"] in datasets),
                  key=lambda r: (r["dataset"], r["id"]))
    if limit and limit < len(rows):
        pick = np.random.default_rng(seed).choice(len(rows), limit, replace=False)
        rows = [rows[i] for i in sorted(pick)]
    return rows


def render_arms(doc: dict, models: dict[str, SensorModel], out_dir: str, arms, split: str,
                datasets, limit: int, seed: int, classes=DEFAULT_CLASSES) -> dict[str, int]:
    """Render `arms` for the same selected frames of `split`; writes
    `<out>/<arm>/<dataset>_<frame>.png` and `<out>/<arm>/manifest.jsonl`."""
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; expected a subset of {ARMS}")
    rows = select_rows(doc, split, datasets, limit, seed)
    for arm in arms:
        os.makedirs(os.path.join(out_dir, arm), exist_ok=True)
    manifests = {arm: open(os.path.join(out_dir, arm, "manifest.jsonl"), "w") for arm in arms}
    try:
        for row in rows:
            rgb = D.read_rgb(row)
            ir = D.read_ir(row) if "a2" in arms else None
            boxes = D.class_boxes(row, classes)
            width, height = D.image_size(row)
            model = models.get(row["dataset"])
            if model is None and any(a in arms for a in ("b3", "b4")):
                raise ValueError(f"no sensor model for dataset {row['dataset']!r}")
            name = f"{_safe(row['dataset'])}_{_safe(row['id'])}.png"
            for arm in arms:
                rng = _frame_rng(seed, row["dataset"], row["id"])
                img = {"a1": lambda: rgb, "a2": lambda: ir, "b1": lambda: render_b1(rgb), "b2": lambda: render_b2(rgb),
                       "b3": lambda: render_b3(rgb, model, rng),
                       "b4": lambda: render_b4(height, width, boxes, model, rng, classes)}[arm]()
                if not cv2.imwrite(os.path.join(out_dir, arm, name), img):
                    raise OSError(f"could not write {os.path.join(out_dir, arm, name)}")
                manifests[arm].write(json.dumps({
                    "arm": arm, "dataset": row["dataset"], "id": row["id"], "sequence_id": row["sequence_id"],
                    "split": split, "day_night": row["day_night"], "image": name, "width": width,
                    "height": height, "boxes": [[b[0], *(round(v, 1) for v in b[1:])] for b in boxes]}) + "\n")
    finally:
        for fh in manifests.values():
            fh.close()
    return {arm: len(rows) for arm in arms}


# ------------------------------------------------------------------------- CLI


def _load_doc(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v for v in value.split(",") if v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit the B3/B4 sensor model and render the generative-model-free arms.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--splits", required=True, help="splits.json from rir_splits.py")
    common.add_argument("--datasets", default="", help="comma-separated dataset ids; default: every dataset in splits.json")
    common.add_argument("--seed", type=int, default=1)

    fit = sub.add_parser("fit-sensor", parents=[common], help="fit the sensor model on split T")
    fit.add_argument("--out", required=True, help="sensor-model.json to write")
    fit.add_argument("--max-frames", type=int, default=200, help="T frames used per dataset")

    render = sub.add_parser("render", parents=[common], help="render arms for a split's frames")
    render.add_argument("--sensor-model", help="sensor-model.json (needed for b3 and b4)")
    render.add_argument("--out", required=True, help="output directory (one subdirectory per arm)")
    render.add_argument("--arms", default=",".join(ARMS))
    render.add_argument("--split", default="S", choices=("T", "S", "V", "Test"))
    render.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="class names, in class-index order")
    render.add_argument("--limit", type=int, default=0, help="render a seeded random subset of this many frames")
    a = ap.parse_args(argv)

    doc = _load_doc(a.splits)
    datasets = _csv(a.datasets) or tuple(sorted({r["dataset"] for r in doc["frames"]}))
    if a.cmd == "fit-sensor":
        models = fit_dataset_models(doc, datasets, a.max_frames, a.seed)
        with open(a.out, "w") as fh:
            json.dump({"fitted_on_split": "T", "seed": a.seed, "max_frames": a.max_frames,
                       "datasets": {k: m.to_dict() for k, m in models.items()}}, fh, indent=1)
        print(json.dumps({k: m.to_dict() for k, m in models.items()}, indent=1), file=sys.stderr)
        return 0
    models = {}
    if a.sensor_model:
        models = {k: SensorModel.from_dict(v) for k, v in _load_doc(a.sensor_model)["datasets"].items()}
    counts = render_arms(doc, models, a.out, _csv(a.arms), a.split, datasets, a.limit, a.seed, _csv(a.classes))
    print(f"rendered {counts} -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
