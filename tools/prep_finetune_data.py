#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Prepare a fine-tune training set with an explicit held-out-NAMES split.

Two separate decisions that were previously conflated:

  WHICH INSTRUMENTS to train on — deliberately broader than the universe you
  trade. More instruments means more training windows and a better-conditioned
  adapter, and it costs nothing at inference. `--universe db` uses every
  liquid name in the database (Nordic/European/Japanese equities, ETFs,
  crypto, non-index US names…), not just the SP500.

  WHICH NAMES ARE HELD OUT — a liquidity-stratified random slice
  (`--holdout-frac`) that the trainer never sees. The fine-tune's promotion
  gate then measures generalization to *instruments it never trained on*,
  which is the property we actually need; a temporal holdout on the same
  names can be passed by a model that merely adapted to those tickers.

Bars per name default to FULL history: windows are the scarce resource
(a 52-week window at context 120 leaves only ~50 usable windows per name),
so truncating the history starves the fine-tune for no benefit — the recency
of the *evaluation* window is what has to be controlled, not the training one.

Writes `<out>/train/*.csv`, `<out>/holdout/*.csv` and `<out>/manifest.json`.

Usage:
  prep_finetune_data.py --db stocks.db --out out/ft-data [--universe db|sp500|@file]
      [--max 0] [--bars 0] [--min-history 400] [--holdout-frac 0.25] [--seed 7]
"""
import argparse
import json
import os
import random
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--universe", default="db", help="db (everything liquid) | sp500 | @file")
ap.add_argument("--max", type=int, default=0, help="cap by liquidity (0 = no cap)")
ap.add_argument("--bars", type=int, default=0, help="bars per name (0 = full history)")
ap.add_argument("--min-history", type=int, default=400)
ap.add_argument("--holdout-frac", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--context", type=int, default=120, help="for the training-budget estimate")
ap.add_argument("--horizon", type=int, default=5)
ap.add_argument("--windows-per-s", type=float, default=1.38,
                help="measured trainer throughput (trademiner bench finetune)")
args = ap.parse_args()

con = sqlite3.connect(args.db)
cur = con.cursor()
cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stock_data(Ticker)")
con.commit()

if args.universe.startswith("@"):
    with open(args.universe[1:]) as f:
        cand = [ln.strip().lower() for ln in f if ln.strip()]
elif args.universe == "sp500":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "..", "..", "trademiner", "src"))
    from trademiner.universe.sp500 import stocks  # noqa: PLC0415
    cand = [t.lower() for t in stocks]
else:
    cand = [r[0] for r in cur.execute("SELECT DISTINCT Ticker FROM stock_data")]
# Indices are not tradeable instruments and their dynamics differ; crypto IS
# kept (it is a legitimate extra regime for a price model to learn from).
cand = [t for t in cand if not t.startswith("^")]

rows = []
for t in cand:
    n = cur.execute("SELECT COUNT(*) FROM stock_data WHERE Ticker=? AND Close IS NOT NULL",
                    (t,)).fetchone()[0]
    if n < args.min_history:
        continue
    dv = cur.execute(
        "SELECT AVG(Close*Volume) FROM (SELECT Close, Volume FROM stock_data "
        "WHERE Ticker=? AND Close IS NOT NULL ORDER BY Date DESC LIMIT 20)", (t,)).fetchone()[0]
    if dv:
        rows.append((t, dv, n))
rows.sort(key=lambda r: -r[1])
if args.max > 0:
    rows = rows[: args.max]
names = [r[0] for r in rows]
if len(names) < 8:
    raise SystemExit(f"only {len(names)} eligible names — need at least 8")

# Liquidity-stratified holdout: take the same fraction from each decile so the
# held-out set spans the size spectrum instead of being all small names.
rng = random.Random(args.seed)
n_dec = 10 if len(names) >= 40 else max(1, len(names) // 4)
holdout = set()
for d in range(n_dec):
    bucket = names[d * len(names) // n_dec:(d + 1) * len(names) // n_dec]
    k = max(1, round(len(bucket) * args.holdout_frac)) if bucket else 0
    holdout.update(rng.sample(bucket, min(k, len(bucket))))
train = [t for t in names if t not in holdout]

for sub in ("train", "holdout"):
    d = os.path.join(args.out, sub)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))

written = {"train": 0, "holdout": 0}
bars_written = []
for t in names:
    recs = cur.execute(
        "SELECT Date, Open, High, Low, Close, Volume FROM stock_data "
        "WHERE Ticker=? AND Close IS NOT NULL ORDER BY Date ASC", (t,)).fetchall()
    recs = [r for r in recs if None not in r]
    if args.bars > 0:
        recs = recs[-args.bars:]
    # Eligibility (>= min_history) was already decided on the FULL history
    # above; after truncation a name only needs enough bars for one window.
    if len(recs) < args.context + args.horizon + 1:
        continue
    sub = "holdout" if t in holdout else "train"
    with open(os.path.join(args.out, sub, f"{t.upper()}.csv"), "w") as f:
        f.write("Date,open,high,low,close,volume\n")
        for d_, o, h, l, c, v in recs:
            f.write(f"{str(d_)[:10]},{o:.4f},{h:.4f},{l:.4f},{c:.4f},{int(v or 0)}\n")
    written[sub] += 1
    bars_written.append(len(recs))

manifest = {
    "universe_spec": args.universe, "seed": args.seed,
    "bars_per_name": args.bars or "full", "min_history": args.min_history,
    "holdout_frac": args.holdout_frac, "n_train": written["train"],
    "n_holdout": written["holdout"],
    "median_bars": int(statistics.median(bars_written)) if bars_written else 0,
    "train_names": sorted(t for t in train), "holdout_names": sorted(holdout),
}
with open(os.path.join(args.out, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# The training BUDGET is set here, not inside the trainer: it enumerates every
# window origin (stride 1), so cost = names x (bars - context - horizon + 1).
per_name = max(0, (statistics.median(bars_written) if bars_written else 0) - args.context - args.horizon + 1)
windows = int(per_name * written["train"])
hours = windows / args.windows_per_s / 3600
manifest["est_train_windows"] = windows
manifest["est_hours_per_epoch"] = round(hours, 2)
with open(os.path.join(args.out, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"train {written['train']} names · holdout {written['holdout']} names "
      f"(never trained on) · median {manifest['median_bars']} bars/name "
      f"· universe '{args.universe}' seed {args.seed}")
print(f"BUDGET: ~{windows:,} training windows "
      f"({written['train']} names x {int(per_name)} each) "
      f"=> ~{hours:.1f} h/epoch at {args.windows_per_s} windows/s")
if hours > 12:
    print("        ^ that is a multi-day run: lower --bars or --max to fit your budget")
print(f"wrote {os.path.join(args.out, 'manifest.json')}")
