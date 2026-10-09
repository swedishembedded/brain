#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements leakage-safe dataset splitting for detector
# and domain-translation studies for its clients. If your team needs
# expertise in evaluation protocols for video-derived datasets, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Sequence-only train / validation / test splits for the rgb-to-ir study.

Consecutive video frames are near-duplicates, so a split by FRAME leaks: the
detector would be scored on frames it has effectively seen. Every split here
is a split of SEQUENCES, taken from the `sequence_id` of a pairs manifest (see
rir_manifest.py). Four splits, never overlapping:

    T     translator-training pairs (RGB + real IR)
    S     detector-training frames (RGB + GT boxes; its real IR twin is used
          only by the upper-bound arm)
    V     validation, where all tuning happens
    Test  held out, touched once at the end

Per dataset id: if the manifest carries an official hint (`official_split` on
the records, or `--test-sequences` naming sequences), the hinted sequences are
Test and nothing else is. Without a hint, a seeded `holdout_fraction` of the
sequences is held out as Test. Of the remaining sequences V takes
`v_fraction`, S takes sequences while they fit under a frame cap, T gets the
rest.

A frame is unusable (kept in splits.json with `split: null` and a reason) when
the record says so (`unusable`) or its RGB sits in a black-padded
sub-rectangle (`rgb_inset_padding`).

Day/night comes from the record when given; otherwise it is Otsu on the median
RGB luminance of each sequence, and left unset when the sequences' luminance
spread is below `day_night_min_spread`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

import rir_data as D
import rir_manifest as M

HELD_OUT = ("V", "Test")
TRAINING = ("T", "S")


@dataclass(frozen=True)
class SplitParams:
    v_fraction: float = 0.10
    holdout_fraction: float = 0.20  # Test share of sequences when a dataset has no official hint
    default_s_cap: int = 800  # detector-training frames
    s_cap: dict = field(default_factory=dict)  # per-dataset overrides
    padded_below: float = 0.98  # RGB valid-content fraction under which a frame is unusable
    day_night_min_spread: float = 20.0
    test_sequences: tuple = ()  # sequence ids that are Test whatever the records say
    probe_reduce: int = 2  # decode at 1/2 size for probing; thresholds are scale-free


# --------------------------------------------------------------- primitives


def assign_splits(sizes: dict[str, int], seed_key: str, test_fraction: float, v_fraction: float, s_cap: int) -> dict[str, str]:
    """Partition sequences (id -> usable frame count) into Test / V / S / T.

    Seeded shuffle, then Test and V take a ceil'd share of the sequences, S
    takes sequences that still fit under `s_cap` frames (and the smallest one
    if none does), and T gets the rest and is never left empty."""
    ids = sorted(sizes)
    random.Random(seed_key).shuffle(ids)
    n_test = math.ceil(test_fraction * len(ids)) if test_fraction > 0 else 0
    n_v = math.ceil(v_fraction * (len(ids) - n_test)) if len(ids) - n_test > 1 else 0
    assign = {i: "Test" for i in ids[:n_test]}
    assign.update({i: "V" for i in ids[n_test:n_test + n_v]})
    rest = ids[n_test + n_v:]
    s_frames, s_ids = 0, []
    for sid in rest:
        keeps_t_nonempty = len(s_ids) + 1 < len(rest)
        if s_frames + sizes[sid] <= s_cap and keeps_t_nonempty:
            s_ids.append(sid)
            s_frames += sizes[sid]
    if not s_ids and len(rest) > 1:
        s_ids = [min(rest, key=lambda i: (sizes[i], i))]
    assign.update({i: "S" for i in s_ids})
    assign.update({i: "T" for i in rest if i not in assign})
    return assign


def label_day_night(luminance: dict, min_spread: float) -> dict:
    """key -> "day" / "night" by Otsu on luminance, or None for every key when
    the values do not spread enough for the threshold to mean anything."""
    vals = np.array(list(luminance.values()), dtype=np.float32)
    if len(vals) < 2 or float(vals.max() - vals.min()) < min_spread:
        return {k: None for k in luminance}
    levels = np.clip(vals, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(levels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Compare the quantised level Otsu saw: a value just above the integer
    # threshold would otherwise land in the class the threshold cut it from.
    return {k: ("day" if lv > thr else "night") for k, lv in zip(luminance, levels)}


def check_invariants(rows: list[dict]) -> None:
    """Raise if any sequence spans two splits or a row's flags contradict."""
    seen = defaultdict(set)
    for r in rows:
        if r["usable"] != (r["split"] is not None):
            raise AssertionError(f"usable/split disagree: {r['dataset']} {r['id']}")
        if not r["usable"] and not r["reason"]:
            raise AssertionError(f"unusable without a reason: {r['dataset']} {r['id']}")
        if r["usable"]:
            seen[(r["dataset"], r["sequence_id"])].add(r["split"])
    bad = {k: v for k, v in seen.items() if len(v) > 1}
    if bad:
        raise AssertionError(f"sequences in more than one split: {bad}")


# ------------------------------------------------------------------- probing


# ------------------------------------------------------------------- probing


def _probe_one(args):
    rec, reduce = args
    return D.probe_rgb(D.read_rgb(rec, reduce))


def _probe_all(records: list[dict], reduce: int, workers: int) -> list[D.Probe]:
    jobs = [(r, reduce) for r in records]
    if workers <= 1:
        return [_probe_one(j) for j in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_probe_one, jobs, chunksize=32))


# ---------------------------------------------------------------- per dataset


def _row(rec: dict, probe: D.Probe, params: SplitParams) -> dict:
    row = {k: v for k, v in rec.items() if k != "unusable"}
    row.update(split=None, usable=True, reason=None, luminance=round(probe.luminance, 2))
    row.setdefault("day_night", None)
    if rec.get("unusable"):
        row["usable"], row["reason"] = False, rec["unusable"]
    elif probe.valid_fraction < params.padded_below:
        row["usable"], row["reason"] = False, "rgb_inset_padding"
    return row


def _label_day_night(rows: list[dict], params: SplitParams) -> None:
    todo = [r for r in rows if r["usable"] and r["day_night"] is None]
    if not todo:
        return
    by_seq = defaultdict(list)
    for r in todo:
        by_seq[r["sequence_id"]].append(r["luminance"])
    labels = label_day_night({s: float(np.median(v)) for s, v in by_seq.items()}, params.day_night_min_spread)
    for r in todo:
        r["day_night"] = labels[r["sequence_id"]]


def _assign(rows: list[dict], dataset: str, seed: int, params: SplitParams) -> None:
    usable = [r for r in rows if r["usable"]]
    sizes = Counter(r["sequence_id"] for r in usable)
    forced = set(params.test_sequences)
    hinted = {r["sequence_id"] for r in rows if r.get("official_split") == "test"} | (forced & set(sizes))
    has_hint = bool(hinted) or any(r.get("official_split") for r in rows)
    plan = {s: "Test" for s in hinted if s in sizes}
    rest = {s: n for s, n in sizes.items() if s not in plan}
    cap = params.s_cap.get(dataset, params.default_s_cap)
    plan.update(assign_splits(rest, f"{seed}:{dataset}", 0.0 if has_hint else params.holdout_fraction,
                              params.v_fraction, cap))
    for r in usable:
        r["split"] = plan[r["sequence_id"]]


def build_splits(records: list[dict], seed: int = 1, params: SplitParams = SplitParams(), workers: int = 1):
    """Returns (splits document, probes keyed by (dataset, id)). `records` are
    manifest records as loaded by rir_manifest.load_manifests."""
    probes_list = _probe_all(records, params.probe_reduce, workers)
    rows, probes = [], {}
    for rec, p in zip(records, probes_list):
        rows.append(_row(rec, p, params))
        probes[(rec["dataset"], rec["id"])] = p
    for ds in sorted({r["dataset"] for r in rows}):
        mine = [r for r in rows if r["dataset"] == ds]
        _label_day_night(mine, params)
        _assign(mine, ds, seed, params)
    check_invariants(rows)
    doc = {"version": 1, "seed": seed, "params": asdict(params), "frames": rows, "summary": summarize(rows)}
    return doc, probes


def summarize(rows: list[dict]) -> dict:
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        mine = [r for r in rows if r["dataset"] == ds]
        splits = {}
        for sp in ("T", "S", "V", "Test"):
            m = [r for r in mine if r["split"] == sp]
            splits[sp] = {"frames": len(m), "sequences": len({r["sequence_id"] for r in m}),
                          "day": sum(r["day_night"] == "day" for r in m),
                          "night": sum(r["day_night"] == "night" for r in m)}
        segments = defaultdict(set)
        for r in mine:
            segments[r.get("official_split")].add(r["sequence_id"])
        out[ds] = {"frames": len(mine), "unusable": dict(Counter(r["reason"] for r in mine if not r["usable"])),
                   "splits": splits, "sequences_by_official_split": {str(k): len(v) for k, v in segments.items()}}
    return out


def find_split_leaks(doc: dict, probes: dict, min_corr: float = 0.95, max_hamming: int = 6,
                     hash_corr_floor: float = 0.8) -> list[dict]:
    """Held-out (V, Test) frames that have a near-duplicate in T or S.

    Within each dataset, a held-out frame is flagged when its most correlated
    training-side frame has thumbnail correlation >= `min_corr`, or its
    closest dHash neighbour is within `max_hamming` bits AND its thumbnail
    correlation is at least `hash_corr_floor`. (A hash match alone also fires
    on flat or very dark frames whose hash is noise.) A sequence-only split
    should report none inside a dataset; this checks that it held, and it
    also surfaces repeated scenery across different scenes, which no split by
    sequence can remove."""
    leaks = []
    for ds in sorted({r["dataset"] for r in doc["frames"]}):
        usable = [r for r in doc["frames"] if r["dataset"] == ds and r["usable"]]
        held = [r for r in usable if r["split"] in HELD_OUT]
        train = [r for r in usable if r["split"] in TRAINING]
        if not held or not train:
            continue
        pick = lambda rows, attr: np.stack([getattr(probes[(ds, r["id"])], attr) for r in rows]).astype(np.float32)
        ht, tt = pick(held, "thumb"), pick(train, "thumb")
        hh, th = pick(held, "dhash"), pick(train, "dhash")
        for start in range(0, len(held), 2048):
            sl = slice(start, start + 2048)
            corr = ht[sl] @ tt.T
            ham = hh[sl] @ (1 - th).T + (1 - hh[sl]) @ th.T
            for k in range(corr.shape[0]):
                jc, jh = int(corr[k].argmax()), int(ham[k].argmin())
                if corr[k, jc] >= min_corr:
                    j = jc
                elif ham[k, jh] <= max_hamming and corr[k, jh] >= hash_corr_floor:
                    j = jh
                else:
                    continue
                h = held[start + k]
                leaks.append({"dataset": ds, "held_out": h["id"], "held_out_split": h["split"],
                              "train": train[j]["id"], "train_split": train[j]["split"],
                              "corr": round(float(corr[k, j]), 4), "hamming": int(ham[k, j])})
    return leaks


def _parse_caps(items: list[str]) -> dict:
    caps = {}
    for item in items:
        name, _, val = item.partition("=")
        if not name or not val.isdigit():
            raise SystemExit(f"--s-cap expects dataset=frames, got {item!r}")
        caps[name] = int(val)
    return caps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write splits.json: sequence-only T/S/V/Test splits of pairs manifests.")
    ap.add_argument("--manifest", required=True, nargs="+", help="pairs manifest(s) (see rir_manifest.py)")
    ap.add_argument("--out", required=True, help="splits.json to write")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--default-s-cap", type=int, default=SplitParams().default_s_cap,
                    help="cap on detector-training (S) frames per dataset")
    ap.add_argument("--s-cap", action="append", default=[], metavar="DATASET=FRAMES", help="per-dataset cap")
    ap.add_argument("--holdout-fraction", type=float, default=SplitParams().holdout_fraction,
                    help="Test share of sequences for a dataset without an official hint")
    ap.add_argument("--test-sequences", metavar="FILE", help="file naming sequence ids (one per line) that are Test")
    ap.add_argument("--workers", type=int, default=4, help="processes used to probe images")
    ap.add_argument("--leak-audit", metavar="FILE", help="also write near-duplicates between held-out and training splits")
    a = ap.parse_args(argv)

    forced = ()
    if a.test_sequences:
        with open(a.test_sequences) as fh:
            forced = tuple(fh.read().split())
    params = SplitParams(default_s_cap=a.default_s_cap, s_cap=_parse_caps(a.s_cap),
                         holdout_fraction=a.holdout_fraction, test_sequences=forced)
    try:
        records = M.load_manifests(a.manifest)
    except M.ManifestError as e:
        print(f"invalid manifest:\n{e}", file=sys.stderr)
        return 2
    doc, probes = build_splits(records, a.seed, params, a.workers)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(json.dumps(doc["summary"], indent=1), file=sys.stderr)
    print(f"wrote {len(doc['frames'])} frames -> {a.out}", file=sys.stderr)
    if a.leak_audit:
        leaks = find_split_leaks(doc, probes)
        with open(a.leak_audit, "w") as fh:
            json.dump({"count": len(leaks), "leaks": leaks}, fh, indent=1)
        print(f"leak audit: {len(leaks)} held-out frame(s) with a near-duplicate in T/S -> {a.leak_audit}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
