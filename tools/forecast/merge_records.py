#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Merge OOS record dumps (each {meta, models:{name:...}}).

Two modes:

DEFAULT (cross-model): different models from different runs are combined and
restricted to the COMMON universe of tickers, so each ranks the same
cross-section. Each model keeps its own weeks (per-week RankIC is independent).

--concat (shards): the SAME model name across inputs means name-shards of one
sweep — records are APPENDED (dates key the cross-sections back together),
mase is recombined weighted by record count, latency stats weighted by n.

Usage: merge_records.py [--concat] <out.json> <rec_a.json> <rec_b.json> ...
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--concat", action="store_true",
                help="same-name models are shards: append records instead of replacing")
ap.add_argument("out_json")
ap.add_argument("inputs", nargs="+")
args = ap.parse_args()

models = {}
meta = None
for p in args.inputs:
    d = json.load(open(p))
    meta = d["meta"] if meta is None else meta
    for name, m in d["models"].items():
        if name not in models:
            models[name] = m
            continue
        if not args.concat:
            models[name] = m  # cross-model mode: last input wins for a duplicate name
            continue
        # shard concatenation
        a, b = models[name], m
        na, nb = len(a["records"]), len(b["records"])
        a["records"] = a["records"] + b["records"]
        ma, mb = a.get("mase_mean"), b.get("mase_mean")
        if ma is not None and mb is not None and na + nb:
            a["mase_mean"] = (ma * na + mb * nb) / (na + nb)
        elif ma is None:
            a["mase_mean"] = mb
        la, lb = a.get("latency_ms") or {}, b.get("latency_ms") or {}
        if la.get("n") and lb.get("n"):
            n = la["n"] + lb["n"]
            a["latency_ms"] = {
                "mean": (la["mean"] * la["n"] + lb["mean"] * lb["n"]) / n,
                # medians/percentiles don't combine exactly; the n-weighted mean
                # of shard medians is recorded as an approximation.
                "median": (la["median"] * la["n"] + lb["median"] * lb["n"]) / n,
                "min": min(la["min"], lb["min"]),
                "p90": max(la["p90"], lb["p90"]),
                "n": n,
            }

if not args.concat:
    # Restrict every model to the COMMON universe of tickers.
    tickersets = {name: {r["ticker"] for r in m["records"]} for name, m in models.items()}
    common = set.intersection(*tickersets.values()) if tickersets else set()
    for m in models.values():
        m["records"] = [r for r in m["records"] if r["ticker"] in common]

meta = dict(meta)
all_tickers = {r["ticker"] for m in models.values() for r in m["records"]}
meta["n_names"] = len(all_tickers)
meta["n_origins"] = max(
    (len({r["date"] for r in m["records"]}) for m in models.values()), default=0)
json.dump({"meta": meta, "models": models}, open(args.out_json, "w"), indent=2)
print(f"merged {list(models)} ({'concat' if args.concat else 'common-universe'}): "
      f"{len(all_tickers)} names")
for name, m in models.items():
    wk = len({r["date"] for r in m["records"]})
    print(f"  {name}: {len(m['records'])} recs over {wk} weeks")
