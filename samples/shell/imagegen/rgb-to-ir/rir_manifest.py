#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset contracts and validators for
# multimodal perception studies for its clients. If your team needs expertise
# in paired RGB / thermal-IR data pipelines, you can procure our services by
# sending an email to info@swedishembedded.com.

"""The pairs manifest: the one input contract of every rgb-to-ir stage.

A pairs manifest is a JSON-lines file, one record per aligned RGB / IR frame.
Paths are absolute or relative to the manifest's directory.

    required
      id            str    unique within `dataset`
      dataset       str    user-chosen dataset id (sensor models and splits are per dataset)
      rgb           str    path of the RGB image (8-bit; 16-bit is reduced by >> 8)
      ir            str    path of the thermal image; may be the same file as `rgb`
      boxes         list   [{"class": str, "x1": n, "y1": n, "x2": n, "y2": n}, ...]
                           pixel coordinates in the image, x1 < x2, y1 < y2; may be []
      sequence_id   str    frames that may be near-duplicates share one; splits never separate them

    optional
      official_split  "train" | "test"   the dataset's own held-out hint; whole sequences must agree
      capture_time    str                free-form timestamp, kept for analysis
      day_night       "day" | "night"    measured by the dataset or the reader
      tags            [str]              free-form
      width, height   int                image size in pixels
      ir_read         {"channel": "gray" | "alpha",   gray: any 1- or 3-channel image read as gray;
                                                     alpha: IR stored in the 4th channel
                       "invert": bool}               true for black-hot sources, so that the
                                                     frame is read white-hot (hot is bright)
      unusable        str                a reason this frame must not be used (kept in splits.json)

`python3 rir_manifest.py validate FILE [--check-files]` reports every
violation with its line number and field.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REQUIRED = ("id", "dataset", "rgb", "ir", "boxes", "sequence_id")
OPTIONAL = ("official_split", "capture_time", "day_night", "tags", "width", "height", "ir_read", "unusable")
IR_CHANNELS = ("gray", "alpha")


class ManifestError(ValueError):
    """Carries every violation found, one per line."""


def validate_record(rec, check_files_in: str | None = None) -> list[str]:
    """Violations of the contract in one record, as 'field: message' strings."""
    if not isinstance(rec, dict):
        return ["record: must be a JSON object"]
    errs = []
    for key in REQUIRED:
        if key not in rec:
            errs.append(f"{key}: required field is missing")
    for key in rec:
        if key not in REQUIRED + OPTIONAL:
            errs.append(f"{key}: unknown field")
    for key in ("id", "dataset", "rgb", "ir", "sequence_id"):
        if key in rec and not (isinstance(rec[key], str) and rec[key]):
            errs.append(f"{key}: must be a non-empty string")
    errs += _validate_boxes(rec.get("boxes")) if "boxes" in rec else []
    if rec.get("official_split") not in (None, "train", "test"):
        errs.append(f"official_split: must be 'train' or 'test', got {rec['official_split']!r}")
    if rec.get("day_night") not in (None, "day", "night"):
        errs.append(f"day_night: must be 'day' or 'night', got {rec['day_night']!r}")
    if "tags" in rec and not (isinstance(rec["tags"], list) and all(isinstance(t, str) for t in rec["tags"])):
        errs.append("tags: must be a list of strings")
    for key in ("width", "height"):
        if key in rec and not (isinstance(rec[key], int) and not isinstance(rec[key], bool) and rec[key] > 0):
            errs.append(f"{key}: must be a positive integer")
    if "ir_read" in rec:
        errs += _validate_ir_read(rec["ir_read"])
    if check_files_in is not None:
        for key in ("rgb", "ir"):
            if isinstance(rec.get(key), str) and not os.path.isfile(resolve(check_files_in, rec[key])):
                errs.append(f"{key}: file not found: {resolve(check_files_in, rec[key])}")
    return errs


def _validate_ir_read(spec) -> list[str]:
    if not isinstance(spec, dict):
        return ["ir_read: must be an object"]
    errs = [f"ir_read.{k}: unknown field" for k in spec if k not in ("channel", "invert")]
    if spec.get("channel", "gray") not in IR_CHANNELS:
        errs.append(f"ir_read.channel: must be one of {IR_CHANNELS}, got {spec['channel']!r}")
    if not isinstance(spec.get("invert", False), bool):
        errs.append("ir_read.invert: must be a boolean")
    return errs


def _validate_boxes(boxes) -> list[str]:
    if not isinstance(boxes, list):
        return ["boxes: must be a list"]
    errs = []
    for i, b in enumerate(boxes):
        if not isinstance(b, dict):
            errs.append(f"boxes[{i}]: must be an object")
            continue
        if not (isinstance(b.get("class"), str) and b["class"]):
            errs.append(f"boxes[{i}].class: must be a non-empty class name")
        coords = [b.get(k) for k in ("x1", "y1", "x2", "y2")]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in coords):
            errs.append(f"boxes[{i}]: x1, y1, x2, y2 must be numbers")
        elif not (coords[0] < coords[2] and coords[1] < coords[3]):
            errs.append(f"boxes[{i}]: need x1 < x2 and y1 < y2, got {coords}")
    return errs


def resolve(base: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def read_records(path: str, check_files: bool = False) -> tuple[list[dict], list[str]]:
    """(records, errors); errors are 'line N: field: message' and include
    duplicate (dataset, id) pairs and sequences that straddle official splits."""
    base = os.path.dirname(os.path.abspath(path))
    records, errs, seen, seq_official = [], [], {}, {}
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"line {n}: not valid JSON: {e}")
                continue
            bad = validate_record(rec, base if check_files else None)
            errs += [f"line {n}: {m}" for m in bad]
            if bad:
                continue
            key = (rec["dataset"], rec["id"])
            if key in seen:
                errs.append(f"line {n}: id: duplicate of line {seen[key]} in dataset {rec['dataset']!r}")
            seen[key] = n
            seq = (rec["dataset"], rec["sequence_id"])
            off = rec.get("official_split")
            if off and seq_official.setdefault(seq, (off, n))[0] != off:
                errs.append(f"line {n}: official_split: sequence {rec['sequence_id']!r} is "
                            f"{seq_official[seq][0]!r} on line {seq_official[seq][1]} but {off!r} here")
            records.append({**rec, "rgb": resolve(base, rec["rgb"]), "ir": resolve(base, rec["ir"])})
    if not records and not errs:
        errs.append("manifest has no records")
    return records, errs


def load_manifests(paths: list[str], check_files: bool = False) -> list[dict]:
    """Records of all manifests (paths made absolute). Raises ManifestError listing every violation."""
    records, errs = [], []
    for p in paths:
        r, e = read_records(p, check_files)
        records += r
        errs += [f"{p}: {m}" for m in e]
    if errs:
        raise ManifestError("\n".join(errs))
    return records


def write_records(path: str, records: list[dict]) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a pairs manifest against the contract.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="report every contract violation, or print a summary")
    v.add_argument("manifest", nargs="+")
    v.add_argument("--check-files", action="store_true", help="also require every rgb / ir file to exist")
    a = ap.parse_args(argv)
    bad = False
    for path in a.manifest:
        records, errs = read_records(path, a.check_files)
        for e in errs:
            print(f"{path}: {e}", file=sys.stderr)
        bad |= bool(errs)
        if not errs:
            seqs = {(r["dataset"], r["sequence_id"]) for r in records}
            print(f"{path}: ok, {len(records)} frames, {len(seqs)} sequences, "
                  f"datasets {sorted({r['dataset'] for r in records})}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
