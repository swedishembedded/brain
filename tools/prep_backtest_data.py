#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Prepare a leak-free walk-forward split from trademiner's stocks.db:
  - FT (fine-tune): the first N/2 liquid names, recent history ENDING at T0
    (`embargo` bars before the latest) — the model sees nothing after T0.
  - HOLDOUT: the next N/2 names, same ≤T0 window — never fine-tuned (generalization).
  - BT (backtest): all N names, the full recent window — the ranking universe, whose
    graded weeks all fall AFTER T0, so the backtest is genuinely out-of-sample.

Usage: prep_backtest_data.py --db <stocks.db> --out <dir> [--names 32] [--bt-bars 300]
                             [--ft-bars 180] [--embargo 60]
"""
import argparse, os, glob, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--out", required=True, help="output root (creates ft/ holdout/ bt/)")
ap.add_argument("--names", type=int, default=32)
ap.add_argument("--bt-bars", type=int, default=300)
ap.add_argument("--ft-bars", type=int, default=180)
ap.add_argument("--embargo", type=int, default=60)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
allroot = os.path.join(args.out, "_all")
subprocess.run([sys.executable, os.path.join(HERE, "export_ohlcv.py"),
                "--db", args.db, "--out", allroot, "--max", str(args.names),
                "--fresh-only", "--min-history", str(args.bt_bars + args.embargo + 40)], check=True)

files = sorted(glob.glob(f"{allroot}/*.csv"))[: args.names]
names = [os.path.basename(f)[:-4] for f in files]
half = len(names) // 2
incl, hold = set(names[:half]), set(names[half:])
for sub in ("ft", "holdout", "bt"):
    d = os.path.join(args.out, sub)
    os.makedirs(d, exist_ok=True)
    for x in glob.glob(f"{d}/*.csv"):
        os.remove(x)
for f in files:
    n = os.path.basename(f)[:-4]
    ls = open(f).read().splitlines()
    hdr, data = ls[0], ls[1:]
    open(os.path.join(args.out, "bt", f"{n}.csv"), "w").write("\n".join([hdr] + data[-args.bt_bars:]) + "\n")
    ft = data[: -args.embargo][-args.ft_bars:]  # ≤ T0
    if n in incl:
        open(os.path.join(args.out, "ft", f"{n}.csv"), "w").write("\n".join([hdr] + ft) + "\n")
    if n in hold:
        open(os.path.join(args.out, "holdout", f"{n}.csv"), "w").write("\n".join([hdr] + ft) + "\n")
print(f"ft={len(incl)} names  holdout={len(hold)} names  bt={len(files)} names  "
      f"(T0 = {args.embargo} bars before latest; backtest origins start at bar {args.bt_bars - args.embargo})")
