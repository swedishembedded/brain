#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Compute honest out-of-sample cross-sectional skill metrics from the model-agnostic
eval dump written by `crates/cli/tests/oos_skill_eval.rs`.

Per model (+ a shuffled-prediction NEGATIVE CONTROL and a naive last-value baseline):
  - per-week cross-sectional RankIC (Spearman) -> mean +/- stderr + t-stat
  - directional accuracy (50% = coin flip)
  - market-neutral top-K/bottom-K long/short basket, net of realistic cost,
    compounded, vs ^gspc buy-hold over the SAME weeks, and % weeks beating the index
  - point MASE vs naive (read from the Rust dump, which has the context)

Leak-safety is the harness's job (2026 origins, past-only normalization). This step
only ranks and scores what it produced. Small samples are noise-dominated: the honest
read is RankIC significance (t-stat), not a single cumulative-return line.

Usage: oos_skill_report.py <eval.json> <gspc.csv> <out_metrics.json> [--k 6 --cost-bps 5]
"""
import sys, json, argparse
import numpy as np
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("eval_json")
ap.add_argument("gspc_csv")
ap.add_argument("out_json")
ap.add_argument("--k", type=int, default=6, help="names per L/S side")
ap.add_argument("--cost-bps", type=float, default=5.0, help="per side per rebalance")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

D = json.load(open(args.eval_json))
META = D["meta"]
H = META["horizon"]
rng = np.random.default_rng(args.seed)

# ^gspc H-day-ahead return keyed by last-observed date.
gdates, gclose = [], []
for i, line in enumerate(open(args.gspc_csv)):
    if i == 0:
        continue
    c = line.strip().split(",")
    if len(c) < 5:
        continue
    gdates.append(c[0][:10])
    gclose.append(float(c[4]))
gidx = {d: i for i, d in enumerate(gdates)}


def gspc_ret(date):
    i = gidx.get(date)
    if i is None:
        cand = [j for j, d in enumerate(gdates) if d <= date]
        if not cand:
            return None
        i = cand[-1]
    if i + H < len(gclose) and gclose[i] and gclose[i + H]:
        return gclose[i + H] / gclose[i] - 1.0
    return None


def spearman(pred, real):
    """Spearman rank correlation across a cross-section (ties broken by average rank)."""
    if len(pred) < 3:
        return np.nan
    pr = _rank(pred)
    rr = _rank(real)
    if np.std(pr) == 0 or np.std(rr) == 0:
        return np.nan
    return float(np.corrcoef(pr, rr)[0, 1])


def _rank(x):
    x = np.asarray(x, float)
    order = x.argsort()
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


COST = 4.0 * args.cost_bps / 1e4  # both legs, round trip on the weekly rebalance


def by_origin(recs):
    o = defaultdict(dict)  # date -> {ticker: (pred, real)}
    for r in recs:
        o[r["date"]][r["ticker"]] = (r["pred"], r["real"])
    return o


def ls_week(row, preds=None):
    """Top-K long minus bottom-K short realized return (equal weight)."""
    items = list(row.items())
    if preds is not None:  # negative control: use shuffled preds, real from row
        p = list(preds)
    else:
        p = [v[0] for _, v in items]
    real = [v[1] for _, v in items]
    order = np.argsort(p)[::-1]
    k = min(args.k, len(items) // 2)
    if k < 1:
        return None
    longs = [real[order[i]] for i in range(k)]
    shorts = [real[order[-1 - i]] for i in range(k)]
    return float(np.mean(longs) - np.mean(shorts))


def eval_model(recs, shuffle=False, naive=False):
    o = by_origin(recs)
    weeks = sorted(o.keys())
    ric, dacc_hit, dacc_n = [], 0, 0
    ls, gspc, mkt = [], [], []
    used_weeks = []
    for w in weeks:
        row = o[w]
        tickers = list(row.keys())
        pred = np.array([row[t][0] for t in tickers])
        real = np.array([row[t][1] for t in tickers])
        if naive:
            pred = np.zeros_like(pred)  # last-value: zero predicted change
        if shuffle:
            pred = pred.copy()
            rng.shuffle(pred)
        # RankIC
        r = spearman(pred, real)
        # directional accuracy
        if not naive:  # sign(0) is uninformative for naive
            dacc_hit += int(np.sum(np.sign(pred) == np.sign(real)))
            dacc_n += len(pred)
        # basket
        g = gspc_ret(w)
        if g is None:
            continue
        lsw = ls_week({t: (pred[i], real[i]) for i, t in enumerate(tickers)})
        if lsw is None:
            continue
        used_weeks.append(w)
        if not np.isnan(r):
            ric.append(r)
        ls.append(lsw - COST)
        gspc.append(g)
        mkt.append(float(np.mean(real)))
    ric = np.array(ric)
    n = len(ric)
    mean = float(np.mean(ric)) if n else float("nan")
    se = float(np.std(ric, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = mean / se if se and not np.isnan(se) else float("nan")
    # compounded cumulative curves
    ls_cum = np.cumprod(1 + np.array(ls)) - 1 if ls else np.array([])
    gs_cum = np.cumprod(1 + np.array(gspc)) - 1 if gspc else np.array([])
    beat = float(np.mean(np.array(ls) > np.array(gspc))) if ls else float("nan")
    return {
        "rankic_mean": mean,
        "rankic_se": se,
        "rankic_t": t,
        "n_weeks": n,
        "rankic_by_week": [None if np.isnan(x) else round(float(x), 4) for x in ric],
        "dir_acc": (dacc_hit / dacc_n) if dacc_n else None,
        "ls_weekly": [round(float(x), 6) for x in ls],
        "ls_cum": [round(float(x), 6) for x in ls_cum],
        "ls_total": float(ls_cum[-1]) if len(ls_cum) else float("nan"),
        "gspc_cum": [round(float(x), 6) for x in gs_cum],
        "gspc_total": float(gs_cum[-1]) if len(gs_cum) else float("nan"),
        "pct_weeks_beat_index": beat,
        "weeks": used_weeks,
    }


out = {"meta": META, "k": args.k, "cost_bps": args.cost_bps, "models": {}}
for name, m in D["models"].items():
    recs = m["records"]
    res = eval_model(recs)
    res["mase"] = m.get("mase_mean")
    res["latency_ms"] = m.get("latency_ms")
    # Proper negative control: the null distribution of the skill metric under NO
    # signal, estimated by averaging many independent within-week permutations of
    # the predictions. The null mean must sit at ≈ 0 (the pipeline invents no
    # skill); its spread is the noise band the real RankIC must clear.
    ncs = [eval_model(recs, shuffle=True) for _ in range(40)]
    nc_means = [x["rankic_mean"] for x in ncs if not np.isnan(x["rankic_mean"])]
    nc = ncs[0]  # keep one realization for the L/S control curve
    nc["rankic_mean"] = float(np.mean(nc_means)) if nc_means else float("nan")
    nc["rankic_se"] = float(np.std(nc_means, ddof=1)) if len(nc_means) > 1 else float("nan")
    nc["rankic_t"] = (nc["rankic_mean"] / nc["rankic_se"]) if nc["rankic_se"] else float("nan")
    nc["n_perms"] = len(nc_means)
    res["neg_control"] = nc
    res["naive"] = eval_model(recs, naive=True)
    out["models"][name] = res
    print(f"{name}: RankIC {res['rankic_mean']:+.4f} +/- {res['rankic_se']:.4f} "
          f"(t={res['rankic_t']:+.2f}, n={res['n_weeks']})  dir-acc {res['dir_acc']:.3f}  "
          f"L/S {res['ls_total']:+.2%} vs gspc {res['gspc_total']:+.2%}  "
          f"beat {res['pct_weeks_beat_index']:.0%}  MASE {res['mase']}")
    nc = res["neg_control"]
    print(f"    neg-control (shuffled): RankIC {nc['rankic_mean']:+.4f} +/- {nc['rankic_se']:.4f} (t={nc['rankic_t']:+.2f})")

json.dump(out, open(args.out_json, "w"), indent=2)
print("wrote", args.out_json)
