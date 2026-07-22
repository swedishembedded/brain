#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Export a training universe of daily OHLCV from trademiner's `stocks.db` (or any
SQLite with a `stock_data(Ticker,Date,Open,High,Low,Close,Volume)` table) into the
per-ticker CSV directory `brain forecast finetune --data <dir>` expects
(`Date,open,high,low,close,volume`, one file per name).

This is the bridge between the data fetcher (trademiner `make update`, which refreshes
the DB from Yahoo Finance) and the fine-tuner. Selects the most-liquid, fresh, long-
enough names so the cross-sectional fine-tune has breadth.

Usage:
  python3 tools/export_ohlcv.py --db <stocks.db> --out <csv-dir> [--max 150] [--min-history 400]
"""
import argparse
import os
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to trademiner stocks.db")
    ap.add_argument("--out", required=True, help="output CSV directory")
    ap.add_argument("--max", type=int, default=150, help="cap universe to the N most liquid (0 = all)")
    ap.add_argument("--min-history", type=int, default=400, help="require >= this many bars")
    ap.add_argument("--fresh-only", action="store_true", help="only names updated to the DB's latest date")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()
    # A Ticker index makes per-name queries fast (the composite (Date,Ticker) unique
    # constraint's index is Date-leftmost, so it doesn't help WHERE Ticker=?). Safe,
    # additive, one-time; also speeds up trademiner's own queries.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stock_data(Ticker)")
    con.commit()

    # One grouped scan: per-ticker bar count + last date.
    meta = {t: (c, m) for t, c, m in cur.execute("SELECT Ticker, COUNT(*), MAX(Date) FROM stock_data GROUP BY Ticker")}
    latest = max(m for _, m in meta.values())
    cand = [
        t for t, (c, m) in meta.items()
        if not t.startswith("^") and not t.endswith("-usd") and c >= args.min_history and (not args.fresh_only or m == latest)
    ]
    # Recent dollar-volume per candidate (indexed → fast) for the liquidity ranking.
    rows = []
    for t in cand:
        r = cur.execute(
            "SELECT AVG(Close*Volume) FROM (SELECT Close, Volume FROM stock_data WHERE Ticker=? ORDER BY Date DESC LIMIT 20)",
            (t,),
        ).fetchone()[0]
        if r:
            rows.append((t, r))
    rows.sort(key=lambda x: x[1], reverse=True)
    if args.max > 0:
        rows = rows[: args.max]

    os.makedirs(args.out, exist_ok=True)
    for t, _ in rows:
        recs = cur.execute(
            "SELECT Date, Open, High, Low, Close, Volume FROM stock_data WHERE Ticker=? ORDER BY Date ASC",
            (t,),
        ).fetchall()
        with open(os.path.join(args.out, f"{t.upper()}.csv"), "w") as f:
            f.write("Date,open,high,low,close,volume\n")
            for d, o, h, l, c, v in recs:
                if None in (o, h, l, c, v):
                    continue
                f.write(f"{str(d)[:10]},{o:.4f},{h:.4f},{l:.4f},{c:.4f},{int(v or 0)}\n")
    print(f"exported {len(rows)} names (latest {str(latest)[:10]}) to {args.out}")


if __name__ == "__main__":
    main()
