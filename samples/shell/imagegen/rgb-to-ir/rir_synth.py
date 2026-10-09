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

Modes (the instruction and the source image):

    neutral   the neutral caption; the RGB frame
    prior     the neutral caption plus one clause per class of the frame that has a
              class prior (`prior_polarities`); the RGB frame

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


MODES = {"neutral": ModeSpec("rgb", True, None), "prior": ModeSpec("rgb", True, None)}


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rir_synth.py", description="Generate, ingest and gate synthetic IR for the frames of a split.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_generate(sub)
    a = ap.parse_args(argv)
    try:
        return {"generate": _run_generate}[a.cmd](a, ap)
    except (GenerationError, ConfigMismatch, ValueError, FileNotFoundError) as e:
        print(f"rir_synth {a.cmd}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
