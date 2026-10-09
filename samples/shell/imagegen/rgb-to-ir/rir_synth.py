#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements synthetic thermal-image generation, ingestion
# and label-preservation gating for its clients. If your team needs expertise
# in turning translator outputs into leakage-safe detector training arms, you
# can procure our services by sending an email to info@swedishembedded.com.

"""Synthetic IR for the frames of a split: generate, ingest, gate.

    rir_synth.py generate --splits S.json --config synth.json --brain BRAIN --out DIR --modes neutral,prior

`generate` runs `brain flux2 generate` once per frame and mode (the CLI takes
one prompt and one output per process, so the model is loaded for every image)
and is built to be interrupted and restarted:

* Resumable. A frame is finished when its output is a valid PPM of the size its
  record promises; a rerun skips it. An existing record made with another
  instruction, seed, adapter (by sha256), strength or crop is a `ConfigMismatch`
  and aborts: one output directory holds one configuration.
* Bounded retry. A run that fails because the card has no room (the placement
  or out-of-memory messages) is retried after `Retry.sleep_s`, at most
  `Retry.max_retries` times, then fails with that reason. Any other failure, a
  timeout, a missing binary or a "success" without a valid output aborts at
  once with the stderr; nothing is retried that is not plausibly transient.
* Sharded and limited. `--shard i/n` takes every n-th frame of the selected
  set, offset i, so n processes (one per card) cover it exactly once;
  `--limit` selects a seeded subset BEFORE sharding.

Frames are cut to the largest centred window whose sides are multiples of 16
(`crop_for`), because that is what the model's reference path would crop to
anyway; the crop is recorded with every output so that boxes and pixels map
back to the source exactly (`crop_boxes`). A frame under 16 px on a side is an
error. Nothing is padded or resampled.

`ingest` turns a directory of outputs into an ARM in the format `rir_arms.py render` writes
(`<arm>/manifest.jsonl` and single-channel 8-bit white-hot PNGs), so `rir_pack.py` packs it unchanged: the RGB output
becomes luma, the boxes of the source frame go through the recorded crop, and `--sensor-model` applies the fitted
sensor model with the function B3 and B4 use (`rir_arms.apply_sensor`, the same per-frame noise stream); `--no-sensor`
leaves the luma alone. One of the two is required. Outputs without a record (made by other means) are mapped by the
default crop or taken as the whole frame, by their size. Frames without an output are counted, never invented.

`gate` runs the label-preservation gates of rir_gates over an ingested arm: for every frame the edge correlation
inside each ground-truth box against the 10th percentile of the same quantity over the real pairs of split T, and the
global alignment shift (phase correlation, 2 px); with `--sam2` also the SAM 2 mask IoU of the box on the RGB and on
the synthetic IR (the resident D-Bus segmenter of rir_sam2, made from `--dbus-address` and `--brain-py`), for the
boxes that passed the model-free gates. Failing frames are written to `rejects.jsonl` with their reasons and leave
`manifest.jsonl` (the file `rir_pack.py` packs) unless `--keep-rejected`; `manifest.all.jsonl` keeps the whole arm and
`gate-stats.json` is the per-arm rejection statistics rir_decide reads for K4.

Modes (the instruction and the source image):

    neutral   the neutral caption; the RGB frame
    prior     the neutral caption plus one clause per class of the frame that has a
              class prior (`prior_polarities`); the RGB frame
    vae-roundtrip
              the frame's REAL IR (one channel replicated to three) as the reference at
              `--strength 0`, which `brain flux2 generate` documents as the source itself
              through the autoencoder with no denoising step: a control arm that carries the
              autoencoder's artefacts and nothing a translator learned, to tell whether a
              detector learns those artefacts. The adapter is not applied.

Class priors come from the caption report of split T. A polarity is a prior for
a class only if it is warmer or cooler (never the held-out wording: the clauses
are the training templates of rir_captions) and at least `FLAG_BELOW` of that
class's statements state it, i.e. the adapter has seen it enough to follow it;
of the polarities that qualify the commonest one is the prior. No IR is
measured for a prior instruction: it uses the class and nothing else about the
frame. A frame without such a class gets the neutral caption.

The config is a JSON file (paths relative to it): `dit`, `vae`, `text_encoder`,
`tokenizer`, `strength`, `seed` (the generation seed, the same for every frame
and mode so that modes are paired) and optionally `adapter` (omitted: zero-shot
edit), `device`, `backend`, `variant` (klein-4b) and `extra_args`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Callable

import cv2
import numpy as np

import rir_arms as A
import rir_captions as C
import rir_data as D
import rir_gates as G
import rir_sam2 as SAM2

OUT_MULTIPLE = 16
OOM_PATTERN = re.compile(r"no GPU placement fits|out of memory|\bOOM\b|OUT_OF_MEMORY", re.IGNORECASE)
STDERR_TAIL = 600
MIN_BOX_VISIBLE = 0.5  # share of a box's area that must lie in the crop to keep it


class GenerationError(RuntimeError):
    """The brain process could not produce an image; carries the reason."""


class ConfigMismatch(RuntimeError):
    """An output directory already holds frames made with another configuration."""


# ------------------------------------------------------------------------ crop


@dataclass(frozen=True)
class Crop:
    """The window of the source frame that was generated: (x0, y0), size, and the source size."""
    x0: int
    y0: int
    width: int
    height: int
    src_width: int
    src_height: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Crop":
        return Crop(**{k: int(d[k]) for k in ("x0", "y0", "width", "height", "src_width", "src_height")})


def crop_for(width: int, height: int) -> Crop:
    """The centred window with both sides rounded down to a multiple of 16."""
    w, h = width // OUT_MULTIPLE * OUT_MULTIPLE, height // OUT_MULTIPLE * OUT_MULTIPLE
    if w == 0 or h == 0:
        raise ValueError(f"a {width}x{height} frame is under {OUT_MULTIPLE} px on a side: it cannot be generated")
    return Crop((width - w) // 2, (height - h) // 2, w, h, width, height)


def crop_image(img: np.ndarray, crop: Crop) -> np.ndarray:
    return np.ascontiguousarray(img[crop.y0:crop.y0 + crop.height, crop.x0:crop.x0 + crop.width])


def crop_boxes(boxes, crop: Crop, min_visible: float = MIN_BOX_VISIBLE) -> list[tuple]:
    """(class, x1, y1, x2, y2) of the source frame -> crop pixels, clipped to the crop. A box with less than
    `min_visible` of its area inside the crop is dropped."""
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        cx1, cy1 = max(x1, crop.x0), max(y1, crop.y0)
        cx2, cy2 = min(x2, crop.x0 + crop.width), min(y2, crop.y0 + crop.height)
        if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < min_visible * (x2 - x1) * (y2 - y1):
            continue
        out.append((int(cls), *(round(v, 1) for v in (cx1 - crop.x0, cy1 - crop.y0, cx2 - crop.x0, cy2 - crop.y0))))
    return out


# ---------------------------------------------------------------------- config


@dataclass(frozen=True)
class SynthConfig:
    dit: str
    vae: str
    text_encoder: str
    tokenizer: str
    strength: float
    seed: int
    adapter: str | None = None
    device: str | None = None
    backend: str | None = None
    variant: str = "klein-4b"
    extra_args: tuple = ()


_PATHS = ("dit", "vae", "text_encoder", "tokenizer", "adapter")
_REQUIRED = ("dit", "vae", "text_encoder", "tokenizer", "strength", "seed")


def load_config(path: str) -> SynthConfig:
    with open(path) as fh:
        doc = json.load(fh)
    known = {f for f in SynthConfig.__dataclass_fields__}
    unknown = sorted(set(doc) - known)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; expected a subset of {sorted(known)}")
    missing = [k for k in _REQUIRED if k not in doc]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    base = os.path.dirname(os.path.abspath(path))
    for key in _PATHS:
        if doc.get(key) is not None and not os.path.isabs(doc[key]):
            doc[key] = os.path.join(base, doc[key])
    doc["extra_args"] = tuple(str(a) for a in doc.get("extra_args", ()))
    return SynthConfig(**doc)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- instructions


def prior_polarities(report: dict, min_share: float = C.FLAG_BELOW) -> dict[str, str]:
    """Class -> its prior polarity from a captions-report.json: the commonest of warmer / cooler among those
    stated by at least `min_share` of the class's statements (ties: warmer). A class with none has no prior."""
    priors = {}
    for name, c in report["classes"].items():
        n = c.get("n", 0)
        usable = [p for p in ("warmer", "cooler") if n and c.get(p, 0) / n >= min_share]
        if usable:
            priors[name] = max(usable, key=lambda p: (c[p], p == "warmer"))
    return priors


def prior_instruction(class_names, priors: dict[str, str], rng: np.random.Generator) -> str:
    """The neutral caption plus one training-template clause per class with a prior (at most MAX_OBJECTS)."""
    named = [n for n in sorted(set(class_names)) if n in priors][:C.MAX_OBJECTS]
    if not named:
        return C.NEUTRAL_CAPTION
    return " ".join([C.NEUTRAL_CAPTION, *(C.clause(n, priors[n], rng) for n in named)])


# ----------------------------------------------------------------------- modes


@dataclass(frozen=True)
class ModeSpec:
    source: str  # "rgb": the frame's RGB; "real-ir": its real IR, replicated to three channels
    adapter: bool  # whether the config's adapter is applied
    strength: float | None  # None: the config's


MODES = {"neutral": ModeSpec("rgb", True, None), "prior": ModeSpec("rgb", True, None),
         "vae-roundtrip": ModeSpec("real-ir", False, 0.0)}


def mode_instruction(mode: str, row: dict, classes, priors: dict[str, str] | None, seed: int) -> str:
    if mode == "prior":
        rng = np.random.default_rng([seed, zlib.crc32(f"{row['dataset']}/{row['id']}".encode())])
        return prior_instruction([classes[b[0]] for b in D.class_boxes(row, classes)], priors or {}, rng)
    return C.NEUTRAL_CAPTION


# ------------------------------------------------------------------------- ppm


_PPM_HEADER = re.compile(rb"P6\s+(?:#[^\n]*\n\s*)*(\d+)\s+(\d+)\s+(\d+)\s")


def ppm_size(path: str) -> tuple[int, int] | None:
    """(width, height) of a complete binary PPM, None for a missing, truncated or padded one."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    m = _PPM_HEADER.match(data)
    if not m:
        return None
    w, h, maxval = (int(g) for g in m.groups())
    if not 0 < maxval < 65536 or len(data) - m.end() != w * h * 3 * (1 if maxval < 256 else 2):
        return None
    return w, h


def write_ppm(path: str, img: np.ndarray) -> None:
    """A BGR or single-channel image as a binary RGB PPM (a single channel is replicated)."""
    if not cv2.imwrite(path, img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)):
        raise OSError(f"could not write {path}")


# ------------------------------------------------------------------------ brain


@dataclass(frozen=True)
class Retry:
    max_retries: int = 20  # retries after the first attempt, memory pressure only
    sleep_s: float = 30.0
    timeout_s: float = 1800.0  # per attempt


class BrainRunner:
    """Builds and runs the `brain flux2 generate` command of a config."""

    def __init__(self, brain: str, config: SynthConfig, retry: Retry = Retry(),
                 sleep: Callable[[float], None] = time.sleep):
        self.brain, self.config, self.retry, self._sleep = brain, config, retry, sleep
        self._adapter_sha256: str | None = None

    @property
    def adapter_sha256(self) -> str | None:
        if self.config.adapter is None:
            return None
        if self._adapter_sha256 is None:
            self._adapter_sha256 = sha256_file(self.config.adapter)
        return self._adapter_sha256

    def command(self, prompt: str, ref: str, out: str, size: tuple[int, int], strength: float, adapter: bool) -> list[str]:
        c = self.config
        global_args = [a for flag, v in (("--device", c.device), ("--backend", c.backend)) if v for a in (flag, v)]
        cmd = [self.brain, *global_args, "flux2", "generate", "--variant", c.variant,
               "--dit", c.dit, "--vae", c.vae, "--text-encoder", c.text_encoder, "--tokenizer", c.tokenizer,
               "--prompt", prompt, "--ref", ref, "--strength", str(strength),
               "--width", str(size[0]), "--height", str(size[1]), "--seed", str(c.seed)]
        if adapter and c.adapter:
            cmd += ["--adapter", c.adapter]
        return [*cmd, *c.extra_args, "--out", out]

    def run(self, cmd: list[str]) -> int:
        """Run until success; returns the number of attempts."""
        attempts = 1 + self.retry.max_retries
        for attempt in range(1, attempts + 1):
            try:
                done = subprocess.run(cmd, capture_output=True, text=True, timeout=self.retry.timeout_s)
            except OSError as e:
                raise GenerationError(f"cannot run {self.brain}: {e}") from e
            except subprocess.TimeoutExpired as e:
                raise GenerationError(f"{self.brain} did not finish within {self.retry.timeout_s:g} s") from e
            if done.returncode == 0:
                return attempt
            tail = (done.stderr or done.stdout).strip()[-STDERR_TAIL:]
            if not OOM_PATTERN.search(done.stderr + done.stdout):
                raise GenerationError(f"{self.brain} failed (exit {done.returncode}): {tail}")
            if attempt == attempts:
                raise GenerationError(f"{self.brain} ran out of memory on all {attempts} attempts: {tail}")
            self._sleep(self.retry.sleep_s)
        raise AssertionError("unreachable")


# ------------------------------------------------------------------- generation


@dataclass(frozen=True)
class Selection:
    """Which frames: a seeded `limit` first, then shard `(i, n)`."""
    split: str = "S"
    datasets: tuple = ()
    limit: int = 0
    shard: tuple = (0, 1)
    seed: int = 1


@dataclass(frozen=True)
class Job:
    doc: dict  # a splits.json document
    selection: Selection
    modes: tuple
    classes: tuple
    priors: dict | None  # class -> polarity, needed by the prior mode
    out_dir: str


def parse_shard(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)/(\d+)", text)
    if not m or not int(m[1]) < int(m[2]):
        raise ValueError(f"--shard wants i/n with 0 <= i < n, got {text!r}")
    return int(m[1]), int(m[2])


def select_frames(doc: dict, sel: Selection) -> list[dict]:
    datasets = sel.datasets or tuple(sorted({r["dataset"] for r in doc["frames"]}))
    rows = A.select_rows(doc, sel.split, datasets, sel.limit, sel.seed)
    return rows[sel.shard[0]::sel.shard[1]]


def frame_name(row: dict) -> str:
    return f"{A.safe_name(row['dataset'])}_{A.safe_name(row['id'])}"


def _identity_mismatch(record: dict, identity: dict) -> str | None:
    return next((k for k, v in identity.items() if record.get(k) != v), None)


class ProgressLog:
    def __init__(self, path: str):
        self.path = path

    def add(self, event: str, **fields) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields}) + "\n")


def _source_image(row: dict, spec: ModeSpec) -> np.ndarray:
    return D.read_rgb(row) if spec.source == "rgb" else D.read_ir(row)


def _write_json_atomic(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def _generate_one(job: Job, runner: BrainRunner, row: dict, mode: str, log: ProgressLog) -> str:
    """One frame in one mode; returns "generated" or "skipped"."""
    spec = MODES[mode]
    name = frame_name(row)
    mode_dir = os.path.join(job.out_dir, mode)
    out, record_path = os.path.join(mode_dir, f"{name}.ppm"), os.path.join(mode_dir, f"{name}.json")
    width, height = D.image_size(row)
    crop = crop_for(width, height)
    strength = runner.config.strength if spec.strength is None else spec.strength
    identity = {"mode": mode, "instruction": mode_instruction(mode, row, job.classes, job.priors, job.selection.seed),
                "seed": runner.config.seed, "strength": strength, "crop": crop.to_dict(), "source": spec.source,
                "adapter_sha256": runner.adapter_sha256 if spec.adapter else None}
    if os.path.isfile(record_path):
        with open(record_path) as fh:
            record = json.load(fh)
        field_name = _identity_mismatch(record, identity)
        if field_name:
            raise ConfigMismatch(f"{record_path} was made with a different {field_name} "
                                 f"({record.get(field_name)!r}, now {identity[field_name]!r}): use a new output directory")
        if ppm_size(out) == (crop.width, crop.height):
            log.add("skipped", mode=mode, id=row["id"])
            return "skipped"
    src, partial = os.path.join(mode_dir, f".{name}.src.ppm"), os.path.join(mode_dir, f"{name}.partial.ppm")
    try:
        write_ppm(src, crop_image(_source_image(row, spec), crop))
        started = time.monotonic()
        cmd = runner.command(identity["instruction"], src, partial, (crop.width, crop.height), strength, spec.adapter)
        attempts = runner.run(cmd)
        if ppm_size(partial) != (crop.width, crop.height):
            raise GenerationError(f"{runner.brain} exited 0 but wrote no valid output of {crop.width}x{crop.height} "
                                  f"for {row['dataset']}/{row['id']}")
        os.replace(partial, out)
    finally:
        for leftover in (src, partial):
            if os.path.exists(leftover):
                os.remove(leftover)
    duration = round(time.monotonic() - started, 3)
    _write_json_atomic(record_path, {"dataset": row["dataset"], "id": row["id"], **identity, "adapter": runner.config.adapter,
                                     "output": os.path.basename(out), "command": cmd, "duration_s": duration,
                                     "attempts": attempts})
    log.add("generated", mode=mode, id=row["id"], duration_s=duration, attempts=attempts)
    return "generated"


def generate(job: Job, runner: BrainRunner) -> dict:
    """Generate every selected frame in every mode; returns the counts and timings."""
    unknown = sorted(set(job.modes) - set(MODES))
    if unknown:
        raise ValueError(f"unknown mode {unknown}; expected a subset of {sorted(MODES)}")
    if "prior" in job.modes and job.priors is None:
        raise ValueError("the prior mode needs the class priors (a captions report)")
    rows = select_frames(job.doc, job.selection)
    for mode in job.modes:
        os.makedirs(os.path.join(job.out_dir, mode), exist_ok=True)
    log = ProgressLog(os.path.join(job.out_dir, "progress.log"))
    counts = {"generated": 0, "skipped": 0}
    started = time.monotonic()
    for row in rows:
        for mode in job.modes:
            counts[_generate_one(job, runner, row, mode, log)] += 1
    seconds = round(time.monotonic() - started, 2)
    return {"frames": len(rows), "modes": list(job.modes), **counts, "seconds": seconds,
            "seconds_per_image": round(seconds / counts["generated"], 2) if counts["generated"] else None}


# ----------------------------------------------------------------------- ingest


@dataclass(frozen=True)
class IngestRequest:
    doc: dict  # a splits.json document
    generated_dir: str  # the outputs of one mode: `generate`'s <out>/<mode>, or any directory of <frame>.ppm / .png
    arm_dir: str  # <arms>/<arm>: manifest.jsonl, the PNGs and ingest-stats.json go here
    arm: str
    models: dict | None  # dataset -> SensorModel, or None for no sensor model
    selection: Selection
    classes: tuple
    seed: int = 1  # the per-frame sensor noise


def _find_output(generated_dir: str, name: str) -> tuple[str, Crop | None] | None:
    """(image path, crop of its record or None) of a frame's output, None when it has none."""
    record_path = os.path.join(generated_dir, f"{name}.json")
    if os.path.isfile(record_path):
        record = _load_json(record_path)
        path = os.path.join(generated_dir, record["output"])
        return (path, Crop.from_dict(record["crop"])) if os.path.isfile(path) else None
    for ext in (".ppm", ".png"):
        if os.path.isfile(os.path.join(generated_dir, name + ext)):
            return os.path.join(generated_dir, name + ext), None
    return None


def _resolve_crop(row: dict, image: np.ndarray, recorded: Crop | None) -> Crop:
    """The crop the output was made with: its record's, else (no record) the default crop or the whole frame,
    whichever has the output's size."""
    h, w = image.shape[:2]
    width, height = D.image_size(row)
    candidates = [recorded] if recorded else [crop_for(width, height), Crop(0, 0, width, height, width, height)]
    for crop in candidates:
        if (crop.width, crop.height) == (w, h):
            return crop
    raise ValueError(f"{row['dataset']}/{row['id']}: the output is {w}x{h}, not the {candidates[0].width}x{candidates[0].height} "
                     f"{'its record says' if recorded else 'of the default crop or the whole'} frame")


def ingest(req: IngestRequest) -> dict:
    """Turn the outputs into an arm in the format rir_arms writes (single-channel 8-bit white-hot PNGs and a
    manifest.jsonl with boxes in output pixels); returns, and writes as ingest-stats.json, the statistics."""
    os.makedirs(req.arm_dir, exist_ok=True)
    rows = select_frames(req.doc, req.selection)
    stats = {"arm": req.arm, "frames": len(rows), "ingested": 0, "missing_output": 0, "boxes_in": 0, "boxes_kept": 0,
             "boxes_dropped": 0, "sensor": {"applied": req.models is not None,
                                           "models": {k: m.to_dict() for k, m in (req.models or {}).items()}}}
    means, stds = [], []
    with open(os.path.join(req.arm_dir, "manifest.jsonl"), "w") as manifest:
        for row in rows:
            name = frame_name(row)
            found = _find_output(req.generated_dir, name)
            if found is None:
                stats["missing_output"] += 1
                continue
            image = cv2.imread(found[0], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read {found[0]}")
            crop = _resolve_crop(row, image, found[1])
            luma = D.to_gray(image)
            if req.models is not None:
                model = req.models.get(row["dataset"])
                if model is None:
                    raise ValueError(f"no sensor model for dataset {row['dataset']!r}")
                luma = A.apply_sensor(luma, model, A.frame_rng(req.seed, row["dataset"], row["id"]))
            if not cv2.imwrite(os.path.join(req.arm_dir, f"{name}.png"), luma):
                raise OSError(f"could not write {os.path.join(req.arm_dir, name)}.png")
            source_boxes = D.class_boxes(row, req.classes)
            boxes = crop_boxes(source_boxes, crop)
            manifest.write(json.dumps({
                "arm": req.arm, "dataset": row["dataset"], "id": row["id"], "sequence_id": row["sequence_id"],
                "split": row["split"], "day_night": row["day_night"], "image": f"{name}.png", "width": crop.width,
                "height": crop.height, "boxes": [list(b) for b in boxes], "crop": crop.to_dict()}) + "\n")
            stats["ingested"] += 1
            stats["boxes_in"] += len(source_boxes)
            stats["boxes_kept"] += len(boxes)
            means.append(float(luma.mean()))
            stds.append(float(luma.std()))
    stats["boxes_dropped"] = stats["boxes_in"] - stats["boxes_kept"]
    stats["luma_mean"] = float(np.mean(means)) if means else None
    stats["luma_std"] = float(np.mean(stds)) if stds else None
    _write_json_atomic(os.path.join(req.arm_dir, "ingest-stats.json"), stats)
    return stats


# ------------------------------------------------------------------------ gate


@dataclass(frozen=True)
class GateRequest:
    doc: dict  # a splits.json document: the frames' RGB and the real pairs of split T
    arm_dir: str  # <arms>/<arm>, as ingest wrote it
    arm: str
    classes: tuple
    mask_iou: G.MaskIou | None = None  # the SAM 2 gate (G.segmenter_mask_iou); None: that gate is absent
    keep_rejected: bool = False
    reference_limit: int = 300  # real split-T frames the edge threshold is read off
    max_shift_px: float = 2.0
    seed: int = 1


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def _reference(req: GateRequest, datasets: set[str]) -> G.Reference:
    pairs = [r for r in req.doc["frames"] if r["split"] == "T" and r["usable"] and r["dataset"] in datasets]
    if not pairs:
        raise ValueError(f"no usable split T frames of {sorted(datasets)}: the edge threshold is the 10th percentile "
                         "of the real pairs of split T")
    return G.reference_distribution(G.manifest_pairs(pairs, req.classes, req.reference_limit, req.seed))


def _gate_record(entry: dict, gate: G.ImageGate) -> dict:
    return {"dataset": entry["dataset"], "id": entry["id"], "image": entry["image"], "reasons": list(gate.reasons),
            "shift": None if gate.shift is None else list(gate.shift),
            "boxes": [{"box": list(b.box), "edge_correlation": b.edge_correlation, "mask_iou": b.mask_iou,
                       "reasons": list(b.reasons)} for b in gate.boxes]}


def gate(req: GateRequest) -> dict:
    """Run the label-preservation gates over an ingested arm.

    Every frame gets the model-free gates (box edge correlation against the 10th percentile of the real pairs of
    split T, global shift by phase correlation) and, with `mask_iou`, the mask gate on the boxes that passed them.
    Writes `rejects.jsonl` (reason per failing frame), `gate-stats.json` (rir_gates.RejectionStats plus the
    threshold and the counts; the file rir_decide reads for K4) and prunes `manifest.jsonl` to the passing frames
    unless `keep_rejected`. The whole arm is kept in `manifest.all.jsonl`, which a rerun starts from."""
    manifest, everything = (os.path.join(req.arm_dir, n) for n in ("manifest.jsonl", "manifest.all.jsonl"))
    if not os.path.isfile(everything):
        shutil.copyfile(manifest, everything)
    entries = _read_jsonl(everything)
    frames = {(r["dataset"], r["id"]): r for r in req.doc["frames"]}
    reference = _reference(req, {e["dataset"] for e in entries})
    config = G.GateConfig(edge_threshold=reference.threshold, max_shift_px=req.max_shift_px)
    stats, rejects, kept = G.RejectionStats(req.arm), [], []
    for entry in entries:
        row = frames.get((entry["dataset"], entry["id"]))
        if row is None:
            raise ValueError(f"{entry['dataset']}/{entry['id']} of the arm is not a frame of the splits")
        rgb = D.read_rgb(row)
        rgb = crop_image(rgb, Crop.from_dict(entry["crop"])) if "crop" in entry else rgb
        ir = cv2.imread(os.path.join(req.arm_dir, entry["image"]), cv2.IMREAD_GRAYSCALE)
        if ir is None:
            raise FileNotFoundError(os.path.join(req.arm_dir, entry["image"]))
        result = G.check_image(rgb, ir, [tuple(b) for b in entry["boxes"]], config, mask_iou=req.mask_iou)
        stats.add(result)
        if result.passed:
            kept.append(entry)
        else:
            rejects.append(_gate_record(entry, result))
    _write_jsonl(os.path.join(req.arm_dir, "rejects.jsonl"), rejects)
    _write_jsonl(manifest, entries if req.keep_rejected else kept)
    report = {**stats.to_dict(), "edge_threshold": reference.threshold, "reference_boxes": reference.n,
              "reference_percentile": reference.percentile, "max_shift_px": req.max_shift_px,
              "mask_gate": req.mask_iou is not None, "keep_rejected": req.keep_rejected, "kept": len(kept),
              "excluded": 0 if req.keep_rejected else len(rejects)}
    _write_json_atomic(os.path.join(req.arm_dir, "gate-stats.json"), report)
    return report


# ------------------------------------------------------------------------- CLI


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v for v in value.split(",") if v)


def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _add_generate(sub) -> None:
    g = sub.add_parser("generate", help="synthetic IR for the frames of a split, one brain process per image")
    g.add_argument("--splits", required=True, help="splits.json from rir_splits.py")
    g.add_argument("--config", required=True, help="synth config JSON (components, adapter, strength, seed, device)")
    g.add_argument("--brain", required=True, help="the brain binary")
    g.add_argument("--out", required=True, help="output directory: <mode>/<frame>.ppm and .json, progress.log")
    g.add_argument("--modes", default="neutral", help=f"comma-separated, from {sorted(MODES)}")
    g.add_argument("--captions-report", help="captions-report.json of split T (the prior mode's class priors)")
    g.add_argument("--split", default="S", choices=("T", "S", "V", "Test"))
    g.add_argument("--datasets", default="", help="comma-separated dataset ids; default: all in splits.json")
    g.add_argument("--classes", default=",".join(A.DEFAULT_CLASSES), help="class names, in class-index order")
    g.add_argument("--limit", type=int, default=0, help="a seeded random subset of this many frames, taken before sharding")
    g.add_argument("--shard", default="0/1", help="i/n: every n-th selected frame from offset i")
    g.add_argument("--seed", type=int, default=1, help="frame selection and prior wording (not the generation seed)")
    g.add_argument("--device", help="overrides the config's device")
    g.add_argument("--backend", help="overrides the config's backend")
    g.add_argument("--max-retries", type=int, default=Retry.max_retries, help="retries after memory pressure, per image")
    g.add_argument("--retry-sleep", type=float, default=Retry.sleep_s, help="seconds between those retries")
    g.add_argument("--timeout", type=float, default=Retry.timeout_s, help="seconds per brain process")


def _run_generate(a, ap) -> int:
    modes = _csv(a.modes)
    if "prior" in modes and not a.captions_report:
        ap.error("--modes prior needs --captions-report")
    config = load_config(a.config)
    config = SynthConfig(**{**asdict(config), **{k: v for k, v in (("device", a.device), ("backend", a.backend)) if v}})
    job = Job(_load_json(a.splits), Selection(a.split, _csv(a.datasets), a.limit, parse_shard(a.shard), a.seed), modes,
              _csv(a.classes), prior_polarities(_load_json(a.captions_report)) if a.captions_report else None, a.out)
    summary = generate(job, BrainRunner(a.brain, config, Retry(a.max_retries, a.retry_sleep, a.timeout)))
    print(json.dumps(summary), file=sys.stderr)
    return 0


def _add_ingest(sub) -> None:
    g = sub.add_parser("ingest", help="turn generated outputs into an arm rir_pack.py packs")
    g.add_argument("--splits", required=True, help="splits.json from rir_splits.py")
    g.add_argument("--generated", required=True, help="directory of the outputs of one mode (<generate --out>/<mode>)")
    g.add_argument("--arm", required=True, help="arm name, e.g. c2; the arm is written to <out>/<arm>")
    g.add_argument("--out", required=True, help="the arms directory")
    sensor = g.add_mutually_exclusive_group(required=True)
    sensor.add_argument("--sensor-model", help="sensor-model.json: apply the fitted sensor model (rir_arms.apply_sensor)")
    sensor.add_argument("--no-sensor", action="store_true", help="leave the generated luma as it is")
    g.add_argument("--split", default="S", choices=("T", "S", "V", "Test"))
    g.add_argument("--datasets", default="", help="comma-separated dataset ids; default: all in splits.json")
    g.add_argument("--classes", default=",".join(A.DEFAULT_CLASSES), help="class names, in class-index order")
    g.add_argument("--seed", type=int, default=1, help="seeds the sensor noise, per frame")


def _run_ingest(a, ap) -> int:
    models = None if a.no_sensor else {k: A.SensorModel.from_dict(v) for k, v in _load_json(a.sensor_model)["datasets"].items()}
    request = IngestRequest(_load_json(a.splits), a.generated, os.path.join(a.out, a.arm), a.arm, models,
                            Selection(a.split, _csv(a.datasets)), _csv(a.classes), a.seed)
    stats = ingest(request)
    print(json.dumps({k: v for k, v in stats.items() if k != "sensor"}), file=sys.stderr)
    return 0


def _add_gate(sub) -> None:
    g = sub.add_parser("gate", help="label-preservation gates over an ingested arm; rejected frames leave the packed arm")
    g.add_argument("--splits", required=True, help="splits.json: the frames' RGB and the real pairs of split T")
    g.add_argument("--arm", required=True, help="arm name; the arm is read from <out>/<arm>")
    g.add_argument("--out", required=True, help="the arms directory")
    g.add_argument("--keep-rejected", action="store_true", help="keep the rejected frames in manifest.jsonl (they are still listed)")
    g.add_argument("--classes", default=",".join(A.DEFAULT_CLASSES), help="class names, in class-index order")
    g.add_argument("--reference-limit", type=int, default=GateRequest.reference_limit,
                   help="real split-T frames the edge threshold is read off")
    g.add_argument("--max-shift", type=float, default=GateRequest.max_shift_px, help="global shift limit in pixels")
    g.add_argument("--seed", type=int, default=1, help="which split-T frames form the reference")
    g.add_argument("--sam2", action="store_true", help="also gate the mask IoU of SAM 2 on RGB and IR (resident D-Bus segmenter)")
    g.add_argument("--dbus-address", help="bus address of the resident `brain serve --dbus` (with --sam2)")
    g.add_argument("--brain-py", help="location of brain-py (with --sam2; default: the repository's)")
    g.add_argument("--sam2-variant", default="tiny", help="SAM 2 variant the server loaded")


def _run_gate(a, ap) -> int:
    if a.sam2 and not a.dbus_address:
        ap.error("--sam2 needs --dbus-address (the bus of the resident segmenter)")
    segmenter = SAM2.DbusSegmenter(a.dbus_address, a.sam2_variant, a.brain_py) if a.sam2 else None
    try:
        request = GateRequest(_load_json(a.splits), os.path.join(a.out, a.arm), a.arm, _csv(a.classes),
                              None if segmenter is None else G.segmenter_mask_iou(segmenter), a.keep_rejected,
                              a.reference_limit, a.max_shift, a.seed)
        stats = gate(request)
    finally:
        if segmenter is not None:
            segmenter.close()
    print(json.dumps({k: stats[k] for k in ("arm", "n_images", "n_failed", "fail_fraction", "by_reason", "kept", "excluded")}),
          file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rir_synth.py", description="Generate, ingest and gate synthetic IR for the frames of a split.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_generate(sub)
    _add_ingest(sub)
    _add_gate(sub)
    a = ap.parse_args(argv)
    try:
        return {"generate": _run_generate, "ingest": _run_ingest, "gate": _run_gate}[a.cmd](a, ap)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"rir_synth {a.cmd}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
