#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Merge per-model OOS record dumps (each {meta, models:{name:...}}) into one
combined dump, intersecting on the weeks (origin dates) all models share so the
cross-sectional comparison is apples-to-apples.

Usage: merge_records.py <out.json> <rec_a.json> <rec_b.json> ...
"""
import sys, json

OUT = sys.argv[1]
INS = sys.argv[2:]

models = {}
meta = None
tickersets = {}
for p in INS:
    d = json.load(open(p))
    meta = d["meta"] if meta is None else meta
    for name, m in d["models"].items():
        models[name] = m
        tickersets[name] = {r["ticker"] for r in m["records"]}

# Restrict every model to the COMMON universe of tickers (so each ranks the same
# cross-section), but keep each model's own weeks — per-week RankIC is independent,
# so this is a fair same-universe comparison that preserves each model's full
# statistical power (its own n weeks, reported per model).
common_tickers = set.intersection(*tickersets.values()) if tickersets else set()
maxw = 0
for name, m in models.items():
    m["records"] = [r for r in m["records"] if r["ticker"] in common_tickers]
    maxw = max(maxw, len({r["date"] for r in m["records"]}))

meta = dict(meta)
meta["n_names"] = len(common_tickers)
meta["n_origins"] = maxw
out = {"meta": meta, "models": models}
json.dump(out, open(OUT, "w"), indent=2)
print("merged", list(models.keys()), "common universe:", len(common_tickers), "names")
for name in models:
    wk = len({r["date"] for r in models[name]["records"]})
    print(f"  {name}: {len(models[name]['records'])} recs over {wk} weeks")
