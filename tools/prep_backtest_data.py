#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Prepare a leak-free walk-forward split from trademiner's stocks.db:

  - FT (fine-tune): HALF the universe, liquidity-stratified random split
    (seeded), recent history ENDING at T0 (`embargo` bars before the latest) —
    the model sees nothing after T0.
  - HOLDOUT: the other half, same ≤T0 window — never fine-tuned, so its
    RankIC measures generalization, not memorization.
  - BT (backtest): ALL names, the full recent window — the ranking universe,
    whose graded weeks fall AFTER T0, so the backtest is genuinely
    out-of-sample.

The ft/holdout split is stratified by liquidity decile (20-bar median dollar
volume as of T0) so both halves span the size/liquidity spectrum — an
alphabetical split confounds the comparison with whatever sector/size
structure the alphabet happens to carry. A `split_manifest.json` (seed, T0,
per-name decile + assignment) makes the split reproducible and lets the
report score ft-names and holdout-names separately.

Defaults are FULL-UNIVERSE, quality-first (--names 0 = everything fresh with
enough history); windows sized for the pre-registered 52-weekly-origin
protocol at CTX=120 (bt = 120 + 52*5 + margin, embargo = 2x horizon).

Usage: prep_backtest_data.py --db <stocks.db> --out <dir> [--names 0]
         [--bt-bars 400] [--ft-bars 400] [--embargo 10] [--seed 7]
"""
import argparse
import glob
import json
import os
import random
import statistics
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--out", required=True, help="output root (creates ft/ holdout/ bt/)")
ap.add_argument("--names", type=int, default=0, help="cap universe (0 = full fresh universe)")
ap.add_argument("--bt-bars", type=int, default=400)
ap.add_argument("--ft-bars", type=int, default=400)
ap.add_argument("--embargo", type=int, default=10)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
allroot = os.path.join(args.out, "_all")
for x in glob.glob(f"{allroot}/*.csv"):
    os.remove(x)
subprocess.run([sys.executable, os.path.join(HERE, "export_ohlcv.py"),
                "--db", args.db, "--out", allroot, "--max", str(args.names),
                "--fresh-only", "--min-history",
                str(max(args.bt_bars, args.ft_bars + args.embargo) + 40)], check=True)

files = sorted(glob.glob(f"{allroot}/*.csv"))
if args.names > 0:
    files = files[: args.names]
names = [os.path.basename(f)[:-4] for f in files]

# --- liquidity as of T0: median close*volume over the last 20 bars <= T0 ----
def liquidity(path: str) -> float:
    rows = open(path).read().splitlines()[1:]
    upto = rows[: -args.embargo] if args.embargo else rows
    tail = upto[-20:]
    dollars = []
    for ln in tail:
        c = ln.split(",")
        try:
            dollars.append(float(c[4]) * float(c[5]))
        except (ValueError, IndexError):
            continue
    return statistics.median(dollars) if dollars else 0.0

liq = {n: liquidity(f) for n, f in zip(names, files)}
ranked = sorted(names, key=lambda n: -liq[n])

# --- stratified split: 10 liquidity deciles, half of each to ft (seeded) ----
rng = random.Random(args.seed)
n_dec = 10 if len(ranked) >= 20 else max(1, len(ranked) // 4)
decile = {}
ft_set, hold_set = set(), set()
for d in range(n_dec):
    bucket = ranked[d * len(ranked) // n_dec:(d + 1) * len(ranked) // n_dec]
    for n in bucket:
        decile[n] = d
    picks = rng.sample(bucket, len(bucket) // 2)
    ft_set.update(picks)
    hold_set.update(set(bucket) - set(picks))

# --- write the three window sets --------------------------------------------
for sub in ("ft", "holdout", "bt"):
    d = os.path.join(args.out, sub)
    os.makedirs(d, exist_ok=True)
    for x in glob.glob(f"{d}/*.csv"):
        os.remove(x)
t0_date = None
for f in files:
    n = os.path.basename(f)[:-4]
    ls = open(f).read().splitlines()
    hdr, data = ls[0], ls[1:]
    open(os.path.join(args.out, "bt", f"{n}.csv"), "w").write(
        "\n".join([hdr] + data[-args.bt_bars:]) + "\n")
    ft = (data[: -args.embargo] if args.embargo else data)[-args.ft_bars:]
    if ft and t0_date is None:
        t0_date = ft[-1].split(",")[0]
    dest = "ft" if n in ft_set else "holdout"
    open(os.path.join(args.out, dest, f"{n}.csv"), "w").write("\n".join([hdr] + ft) + "\n")

manifest = {
    "seed": args.seed, "t0": t0_date, "embargo_bars": args.embargo,
    "bt_bars": args.bt_bars, "ft_bars": args.ft_bars, "n_deciles": n_dec,
    "ft": sorted(ft_set), "holdout": sorted(hold_set),
    "decile": decile, "liquidity_usd": {n: round(v) for n, v in liq.items()},
}
with open(os.path.join(args.out, "split_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"ft={len(ft_set)} names  holdout={len(hold_set)} names  bt={len(files)} names  "
      f"(T0={t0_date}, {args.embargo} bars before latest; liquidity-stratified seed={args.seed})")
print(f"wrote {os.path.join(args.out, 'split_manifest.json')}")
