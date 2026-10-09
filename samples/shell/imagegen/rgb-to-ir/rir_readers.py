#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset adapters for paired RGB / thermal-IR
# detection data for its clients. If your team needs expertise in turning
# heterogeneous label layouts into one clean training contract, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Build a pairs manifest (see rir_manifest.py) from a common on-disk layout.

    rir_readers.py --dataset ID --out pairs.jsonl \\
        --rgb-glob 'data/rgb/*.jpg' --ir-glob 'data/ir/*.jpg' \\
        --labels-dir data/labels --label-format voc|yolo|coco \\
        --sequence-rule dir|regex:PATTERN|similarity|block:WIDTH

Pairing. RGB and IR files (and label files) are matched by a KEY: group 1 of
`--pair-regex` applied to the base name (default: the name without its
extension). Use a regex such as `^(.+?)(?:_RGB|_IR)?\\.[^.]+$` when the
modalities carry suffixes. With no `--ir-glob` the IR is read from the RGB
file itself (`--ir-channel alpha` for IR stored in an RGBA file's alpha
channel). Frames that are missing a partner are counted and skipped.

Labels. `voc` reads Pascal VOC xml files from `--labels-dir`, `yolo` reads
normalised `class cx cy w h` txt files (class names from `--yolo-names`),
`coco` reads one json (`--coco-json`). Class names can be renamed with
`--class-map src=dst,...`; consumers pick the classes they train on.

Sequences (`--sequence-rule`; the id is stored as is):
  dir             the name of the RGB file's directory
  regex:PATTERN   group 1 of PATTERN searched in the RGB path
  similarity      frames carry a number (`--numeric-id-regex`, group 1) in
                  temporal order; a break starts a new sequence when the number
                  jumps by more than `--max-gap` or the correlation of the
                  16x16 grayscale thumbnails of neighbours falls below
                  `--min-corr`. A pair that touches an RGB shrunk into a black
                  sub-rectangle is judged on the IR thumbnails instead, so a
                  padded frame inside a video does not cut it.
  block:WIDTH     numbers (`--numeric-id-regex`) are cut into contiguous blocks
                  of WIDTH; `--block-guard G` marks frames within G of a block
                  edge unusable, so blocks never touch.

Official split hint (optional). `--official-split-regex` takes a word from the
RGB path (group 1; words in `--test-words`, default test,val, mean test, others
train); `--test-list FILE` names pair keys, one per line, that are the test.

IR. `--ir-channel gray|alpha`; 16-bit samples are read as 8-bit. `--ir-polarity
auto|white-hot|black-hot`: auto compares the mean IR inside boxes of
`--hot-classes` (default person) with the frame median over up to 200 frames
and inverts the source when those boxes are mostly darker.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

import rir_data as D
import rir_manifest as M

DEFAULT_PAIR_REGEX = r"^(.+?)\.[^.]+$"
PADDED_BELOW = 0.98


class ReaderError(ValueError):
    pass


# ---------------------------------------------------------------- sequences


def thumb_similarity(thumbs):
    """`similarity(i)` for `segment_by_similarity`: correlation of unit thumbnails i-1 and i."""
    return lambda i: float(np.dot(thumbs[i], thumbs[i - 1]))


def segment_by_similarity(nums, similarity, min_corr: float, max_gap: int) -> list[int]:
    """Sequence index per frame, frames in temporal order: a new sequence starts
    between frames i-1 and i when `nums[i] - nums[i-1] > max_gap` or
    `similarity(i) < min_corr`."""
    out, seq = [], 0
    for i in range(len(nums)):
        if i and (nums[i] - nums[i - 1] > max_gap or similarity(i) < min_corr):
            seq += 1
        out.append(seq)
    return out


def _probe_job(rec):
    rgb = D.read_rgb(rec, 2)
    gray = D.to_gray(rgb)
    return D.thumb(gray), D.valid_fraction(gray), D.thumb(D.read_ir(rec))


def _similarity_sequences(recs, nums, groups, args, workers) -> list[str]:
    if workers <= 1:
        probes = [_probe_job(r) for r in recs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            probes = list(pool.map(_probe_job, recs, chunksize=32))
    padded = [p[1] < PADDED_BELOW for p in probes]
    seqs = [None] * len(recs)
    for group in sorted(set(groups)):
        idx = sorted((i for i in range(len(recs)) if groups[i] == group), key=lambda i: nums[i])

        def similarity(k, idx=idx):
            a, b = idx[k - 1], idx[k]
            which = 2 if padded[a] or padded[b] else 0
            return float(np.dot(probes[a][which], probes[b][which]))

        labels = segment_by_similarity([nums[i] for i in idx], similarity, args.get("min_corr", 0.5), args.get("max_gap", 3))
        first = {}
        for i, lab in zip(idx, labels):
            first.setdefault(lab, nums[i])
            seqs[i] = f"{group}-{first[lab]:06d}" if group else f"{first[lab]:06d}"
    return seqs


# ------------------------------------------------------------------- labels


def _voc_boxes(path: str) -> list[dict]:
    out = []
    for obj in ET.parse(path).getroot().findall("object"):
        bb = obj.find("bndbox")
        out.append({"class": (obj.findtext("name") or "").strip(),
                    **{k: float(bb.findtext(k2)) for k, k2 in (("x1", "xmin"), ("y1", "ymin"), ("x2", "xmax"), ("y2", "ymax"))}})
    return out


def _yolo_boxes(path: str, names: list[str], width: int, height: int) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if len(tok) != 5:
                continue
            cls, cx, cy, w, h = int(tok[0]), *map(float, tok[1:])
            if not 0 <= cls < len(names):
                raise ReaderError(f"{path}: class index {cls} outside --yolo-names ({len(names)} names)")
            out.append({"class": names[cls], "x1": (cx - w / 2) * width, "y1": (cy - h / 2) * height,
                        "x2": (cx + w / 2) * width, "y2": (cy + h / 2) * height})
    return out


def _coco_index(path: str, key_of) -> dict[str, list[dict]]:
    with open(path) as fh:
        doc = json.load(fh)
    names = {c["id"]: c["name"] for c in doc["categories"]}
    by_image = {im["id"]: key_of(os.path.basename(im["file_name"])) for im in doc["images"]}
    out: dict[str, list[dict]] = {}
    for a in doc["annotations"]:
        x, y, w, h = a["bbox"]
        out.setdefault(by_image[a["image_id"]], []).append(
            {"class": names[a["category_id"]], "x1": x, "y1": y, "x2": x + w, "y2": y + h})
    return out


def _clean(boxes: list[dict], class_map: dict[str, str]) -> list[dict]:
    out = []
    for b in boxes:
        if b["x2"] > b["x1"] and b["y2"] > b["y1"]:
            out.append({**b, "class": class_map.get(b["class"], b["class"])})
    return out


# ---------------------------------------------------------------- polarity


def detect_polarity(records: list[dict], hot_classes, limit: int = 200) -> tuple[bool, dict]:
    """(invert, evidence): white-hot sources have hot-class boxes brighter than
    the frame median in most frames."""
    cands = [r for r in records if any(b["class"] in hot_classes for b in r["boxes"])]
    if not cands:
        return False, {"frames": 0}
    step = max(1, len(cands) // limit)
    brighter, total = 0, 0
    for rec in cands[::step][:limit]:
        ir = D.read_ir({**rec, "ir_read": {**rec["ir_read"], "invert": False}})
        med = float(np.median(ir))
        vals = [float(ir[int(b["y1"]):int(b["y2"]), int(b["x1"]):int(b["x2"])].mean())
                for b in rec["boxes"] if b["class"] in hot_classes
                and int(b["y2"]) - int(b["y1"]) >= 2 and int(b["x2"]) - int(b["x1"]) >= 2]
        if vals:
            total += 1
            brighter += float(np.mean(vals)) > med
    frac = brighter / total if total else 1.0
    return frac < 0.5, {"frames": total, "fraction_brighter_than_median": round(frac, 3)}


# -------------------------------------------------------------------- build


def build_records(args: dict, workers: int = 1) -> tuple[list[dict], dict]:
    """`args` keys: see the CLI. Returns (records, report)."""
    pair_re = re.compile(args.get("pair_regex") or DEFAULT_PAIR_REGEX)

    def key_of(name: str) -> str:
        m = pair_re.search(name)
        if not m or m.lastindex is None:
            raise ReaderError(f"--pair-regex {pair_re.pattern!r} has no group match in {name!r}")
        return m.group(1)

    rgb_paths = sorted(glob.glob(args["rgb_glob"], recursive=True))
    if not rgb_paths:
        raise ReaderError(f"--rgb-glob {args['rgb_glob']!r} matched no files")
    keyed = {}
    for p in rgb_paths:
        k = key_of(os.path.basename(p))
        if k in keyed:
            raise ReaderError(f"two RGB files share the pair key {k!r}: {keyed[k]} and {p}; "
                              "make --pair-regex capture the distinguishing part")
        keyed[k] = p
    ir_paths = {key_of(os.path.basename(p)): p for p in glob.glob(args["ir_glob"], recursive=True)} if args.get("ir_glob") else None

    label_fmt = args.get("label_format")
    voc = yolo = coco = None
    if label_fmt in ("voc", "yolo"):
        ext = "xml" if label_fmt == "voc" else "txt"
        files = glob.glob(os.path.join(args["labels_dir"], "**", f"*.{ext}"), recursive=True)
        table = {key_of(os.path.basename(p)): p for p in files}
        voc, yolo = (table, None) if label_fmt == "voc" else (None, table)
    elif label_fmt == "coco":
        coco = _coco_index(args["coco_json"], key_of)
    elif label_fmt is not None:
        raise ReaderError(f"unknown --label-format {label_fmt!r}; expected voc, yolo or coco")
    names = [n for n in (args.get("yolo_names") or "").split(",") if n]
    if label_fmt == "yolo" and not names:
        raise ReaderError("--label-format yolo needs --yolo-names")
    class_map = dict(kv.split("=", 1) for kv in (args.get("class_map") or "").split(",") if kv)
    test_keys = None
    if args.get("test_list"):
        with open(args["test_list"]) as fh:
            test_keys = set(fh.read().split())
    off_re = re.compile(args["official_split_regex"]) if args.get("official_split_regex") else None
    test_words = set((args.get("test_words") or "test,val").split(","))
    channel = args.get("ir_channel", "gray")

    report = {"unpaired_rgb": 0, "no_labels": 0}
    records = []
    for key, rgb in keyed.items():
        ir = rgb if ir_paths is None else ir_paths.get(key)
        if ir is None:
            report["unpaired_rgb"] += 1
            continue
        with Image.open(rgb) as im:
            width, height = im.size
        if label_fmt is None:
            boxes = []
        elif label_fmt == "coco":
            boxes = coco.get(key)
        else:
            table = voc or yolo
            boxes = None
            if key in table:
                boxes = _voc_boxes(table[key]) if voc else _yolo_boxes(table[key], names, width, height)
        if boxes is None:
            report["no_labels"] += 1
            boxes = []
        rec = {"id": key, "dataset": args["dataset"], "rgb": os.path.abspath(rgb), "ir": os.path.abspath(ir),
               "boxes": _clean(boxes, class_map), "sequence_id": "", "width": width, "height": height,
               "ir_read": {"channel": channel, "invert": False}}
        if off_re is not None:
            m = off_re.search(rgb)
            if not m:
                raise ReaderError(f"--official-split-regex {off_re.pattern!r} does not match {rgb}")
            rec["official_split"] = "test" if m.group(1) in test_words else "train"
        elif test_keys is not None:
            rec["official_split"] = "test" if key in test_keys else "train"
        records.append(rec)
    if not records:
        raise ReaderError("no frame has both an RGB and an IR image")

    _assign_sequences(records, args, workers)
    polarity = args.get("ir_polarity", "auto")
    if polarity == "auto":
        invert, evidence = detect_polarity(records, set((args.get("hot_classes") or "person").split(",")))
        report["polarity"] = {"decision": "black-hot" if invert else "white-hot", **evidence}
    else:
        invert = polarity == "black-hot"
    if invert:
        for r in records:
            r["ir_read"]["invert"] = True
    return records, report


def _numbers(records, args) -> list[int]:
    if not args.get("numeric_id_regex"):
        raise ReaderError("this --sequence-rule needs --numeric-id-regex")
    rx = re.compile(args["numeric_id_regex"])
    out = []
    for r in records:
        m = rx.search(r["id"])
        if not m or m.lastindex is None:
            raise ReaderError(f"--numeric-id-regex {rx.pattern!r} has no group match in id {r['id']!r}")
        out.append(int(m.group(1)))
    return out


def _assign_sequences(records: list[dict], args: dict, workers: int) -> None:
    rule = args.get("sequence_rule") or "dir"
    if rule == "dir":
        seqs = [os.path.basename(os.path.dirname(r["rgb"])) for r in records]
    elif rule.startswith("regex:"):
        rx = re.compile(rule[len("regex:"):])
        seqs = []
        for r in records:
            m = rx.search(r["rgb"])
            if not m or m.lastindex is None:
                raise ReaderError(f"sequence regex {rx.pattern!r} has no group match in {r['rgb']}")
            seqs.append(m.group(1))
    elif rule == "similarity":
        nums = _numbers(records, args)
        groups = [r.get("official_split", "") for r in records]
        seqs = _similarity_sequences(records, nums, groups, args, workers)
    elif rule.startswith("block:"):
        width = int(rule[len("block:"):])
        guard = int(args.get("block_guard") or 0)
        nums = _numbers(records, args)
        seqs = [f"block{n // width:05d}" for n in nums]
        for r, n in zip(records, nums):
            if n % width < guard or n % width >= width - guard:
                r["unusable"] = "block_gap"
    else:
        raise ReaderError(f"unknown --sequence-rule {rule!r}")
    for r, s in zip(records, seqs):
        r["sequence_id"] = s


# ---------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a pairs manifest from a common label layout.",
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset id stored in every record")
    ap.add_argument("--out", required=True, help="pairs manifest (jsonl) to write")
    ap.add_argument("--rgb-glob", required=True)
    ap.add_argument("--ir-glob")
    ap.add_argument("--ir-channel", choices=M.IR_CHANNELS, default="gray")
    ap.add_argument("--ir-polarity", choices=("auto", "white-hot", "black-hot"), default="auto")
    ap.add_argument("--hot-classes", default="person")
    ap.add_argument("--pair-regex", default=DEFAULT_PAIR_REGEX)
    ap.add_argument("--labels-dir")
    ap.add_argument("--label-format", choices=("voc", "yolo", "coco"))
    ap.add_argument("--coco-json")
    ap.add_argument("--yolo-names", help="comma-separated class names in index order")
    ap.add_argument("--class-map", help="src=dst,... renames class names")
    ap.add_argument("--sequence-rule", default="dir")
    ap.add_argument("--numeric-id-regex")
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--min-corr", type=float, default=0.5)
    ap.add_argument("--block-guard", type=int, default=0)
    ap.add_argument("--official-split-regex")
    ap.add_argument("--test-words", default="test,val")
    ap.add_argument("--test-list")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    if a.label_format in ("voc", "yolo") and not a.labels_dir:
        ap.error("--label-format voc/yolo needs --labels-dir")
    if a.label_format == "coco" and not a.coco_json:
        ap.error("--label-format coco needs --coco-json")
    try:
        records, report = build_records(vars(a), a.workers)
    except ReaderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    M.write_records(a.out, records)
    seqs = {r["sequence_id"] for r in records}
    print(f"wrote {len(records)} frames in {len(seqs)} sequences -> {a.out}; {json.dumps(report)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
