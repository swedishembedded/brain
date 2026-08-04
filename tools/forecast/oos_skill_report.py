#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Compute honest out-of-sample cross-sectional skill metrics from the model-agnostic
eval dump written by `crates/cli/tests/oos_skill_eval.rs`.

Per model (+ a shuffled-prediction NEGATIVE CONTROL and a naive last-value baseline):
  - per-week cross-sectional RankIC (Spearman) -> mean +/- stderr + t-stat
    (Newey-West stderr when step < horizon makes weekly labels overlap)
  - directional accuracy (50% = coin flip)
  - market-neutral top-K/bottom-K long/short basket (fixed --k or --k-frac of the
    week's cross-section), net of cost at --cost-bps AND a 2x stress cost,
    compounded, vs ^gspc buy-hold over the SAME weeks
  - point MASE vs naive (read from the Rust dump, which has the context)
  - with --split-manifest: RankIC per subset (ft-names vs holdout-names) so a
    fine-tune is graded on names it never trained on
  - with --ft-model/--base-model: the paired per-week IC difference (promotion
    gate) and a mechanical VERDICT block against the pre-registered criteria
    (docs/validation-criteria.md in trademiner)

Leak-safety is the harness's job (post-cutoff origins, past-only normalization).
This step only ranks and scores what it produced. Small samples are noise-dominated:
the honest read is RankIC significance (t-stat), not a single cumulative-return line.

Usage: oos_skill_report.py <eval.json> <gspc.csv> <out_metrics.json>
         [--k 6 | --k-frac 0.10] [--cost-bps 5] [--split-manifest m.json]
         [--ft-model kronos_ft --base-model kronos] [--summary-out backtest_summary.json]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("eval_json")
ap.add_argument("gspc_csv")
ap.add_argument("out_json")
ap.add_argument("--k", type=int, default=6, help="names per L/S side (fixed)")
ap.add_argument("--k-frac", type=float, default=None,
                help="names per side as a fraction of the week's cross-section (overrides --k)")
ap.add_argument("--cost-bps", type=float, default=5.0, help="per side per rebalance")
ap.add_argument("--stress-bps", type=float, default=10.0, help="stress cost, reported not gated")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--split-manifest", default=None, help="split_manifest.json from prep_backtest_data")
ap.add_argument("--ft-model", default=None, help="model name of the fine-tuned entry (e.g. kronos_ft)")
ap.add_argument("--base-model", default=None, help="model name of the base entry (e.g. kronos)")
ap.add_argument("--summary-out", default=None,
                help="also write a trademiner-compatible backtest_summary.json here")
ap.add_argument("--min-weeks", type=int, default=40, help="pre-registered sample-size criterion")
args = ap.parse_args()

D = json.load(open(args.eval_json))
META = D["meta"]
H = META["horizon"]
STEP = META.get("step", H)
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


def _rank(x):
    x = np.asarray(x, float)
    order = x.argsort()
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(pred, real):
    if len(pred) < 3:
        return np.nan
    pr, rr = _rank(pred), _rank(real)
    if np.std(pr) == 0 or np.std(rr) == 0:
        return np.nan
    return float(np.corrcoef(pr, rr)[0, 1])


def series_stats(vals):
    """mean, stderr, t of a weekly series. Plain iid stderr when weekly labels
    don't overlap (STEP >= H); Newey-West with L = ceil(H/STEP)-1 otherwise."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return (float(np.mean(v)) if n else float("nan")), float("nan"), float("nan"), n
    mean = float(np.mean(v))
    if STEP >= H:
        se = float(np.std(v, ddof=1) / np.sqrt(n))
    else:
        lag = int(np.ceil(H / STEP)) - 1
        d = v - mean
        gamma0 = float(np.mean(d * d))
        acc = gamma0
        for l in range(1, lag + 1):
            gl = float(np.mean(d[l:] * d[:-l]))
            acc += 2.0 * (1.0 - l / (lag + 1.0)) * gl
        se = float(np.sqrt(max(acc, 0.0) / n))
    t = mean / se if se else float("nan")
    return mean, se, t, n


def by_origin(recs):
    o = defaultdict(dict)
    for r in recs:
        o[r["date"]][r["ticker"]] = (r["pred"], r["real"])
    return o


def k_for(n_items):
    if args.k_frac is not None:
        return max(1, min(int(args.k_frac * n_items), n_items // 2))
    return min(args.k, n_items // 2)


def ls_week(pred, real):
    order = np.argsort(pred)[::-1]
    k = k_for(len(pred))
    if k < 1:
        return None
    longs = [real[order[i]] for i in range(k)]
    shorts = [real[order[-1 - i]] for i in range(k)]
    return float(np.mean(longs) - np.mean(shorts)), k


def eval_model(recs, shuffle=False, naive=False, tickers_subset=None):
    o = by_origin(recs)
    weeks = sorted(o.keys())
    ric, dacc_hit, dacc_n = [], 0, 0
    ls_gross, gspc, mkt, ks = [], [], [], []
    used_weeks = []
    for w in weeks:
        row = o[w]
        tickers = [t for t in row if tickers_subset is None or t in tickers_subset]
        if len(tickers) < 3:
            continue
        pred = np.array([row[t][0] for t in tickers])
        real = np.array([row[t][1] for t in tickers])
        if naive:
            pred = np.zeros_like(pred)
        if shuffle:
            pred = pred.copy()
            rng.shuffle(pred)
        r = spearman(pred, real)
        if not naive:
            dacc_hit += int(np.sum(np.sign(pred) == np.sign(real)))
            dacc_n += len(pred)
        g = gspc_ret(w)
        if g is None:
            continue
        lsw = ls_week(pred, real)
        if lsw is None:
            continue
        used_weeks.append(w)
        ric.append(r)
        ls_gross.append(lsw[0])
        ks.append(lsw[1])
        gspc.append(g)
        mkt.append(float(np.mean(real)))
    mean, se, t, n = series_stats(ric)

    def net_curve(bps):
        cost = 4.0 * bps / 1e4
        net = np.array(ls_gross) - cost
        cum = np.cumprod(1 + net) - 1 if len(net) else np.array([])
        return net, cum

    net, ls_cum = net_curve(args.cost_bps)
    net_stress, ls_cum_stress = net_curve(args.stress_bps)
    gs_cum = np.cumprod(1 + np.array(gspc)) - 1 if gspc else np.array([])
    beat = float(np.mean(net > np.array(gspc))) if len(net) else float("nan")
    return {
        "rankic_mean": mean, "rankic_se": se, "rankic_t": t, "n_weeks": n,
        "rankic_by_week": [None if np.isnan(x) else round(float(x), 4) for x in ric],
        "dir_acc": (dacc_hit / dacc_n) if dacc_n else None,
        "k_per_side_mean": float(np.mean(ks)) if ks else None,
        "ls_weekly": [round(float(x), 6) for x in net],
        "ls_cum": [round(float(x), 6) for x in ls_cum],
        "ls_total": float(ls_cum[-1]) if len(ls_cum) else float("nan"),
        "ls_total_stress": float(ls_cum_stress[-1]) if len(ls_cum_stress) else float("nan"),
        "gspc_cum": [round(float(x), 6) for x in gs_cum],
        "gspc_total": float(gs_cum[-1]) if len(gs_cum) else float("nan"),
        "pct_weeks_beat_index": beat,
        "weeks": used_weeks,
    }


manifest = json.load(open(args.split_manifest)) if args.split_manifest else None
ft_names = set(manifest["ft"]) if manifest else None
hold_names = set(manifest["holdout"]) if manifest else None

out = {"meta": META, "k": args.k, "k_frac": args.k_frac,
       "cost_bps": args.cost_bps, "stress_bps": args.stress_bps,
       "stderr": "newey-west" if STEP < H else "iid",
       "survivorship_note": ("universe = current constituents applied retroactively; "
                             "market-neutral IC vs the shuffled control is the primary "
                             "read, absolute L/S levels are optimistic"),
       "models": {}}

for name, m in D["models"].items():
    recs = m["records"]
    if not recs:
        continue
    res = eval_model(recs)
    res["mase"] = m.get("mase_mean")
    res["latency_ms"] = m.get("latency_ms")
    # Negative control: the null distribution of the skill metric under NO
    # signal — many independent within-week permutations of the predictions.
    ncs = [eval_model(recs, shuffle=True) for _ in range(40)]
    nc_means = [x["rankic_mean"] for x in ncs if not np.isnan(x["rankic_mean"])]
    nc = ncs[0]
    nc["rankic_mean"] = float(np.mean(nc_means)) if nc_means else float("nan")
    nc["rankic_se"] = float(np.std(nc_means, ddof=1)) if len(nc_means) > 1 else float("nan")
    nc["rankic_t"] = (nc["rankic_mean"] / nc["rankic_se"]) if nc["rankic_se"] else float("nan")
    nc["n_perms"] = len(nc_means)
    res["neg_control"] = {k: nc[k] for k in
                          ("rankic_mean", "rankic_se", "rankic_t", "n_perms", "ls_total")}
    res["naive"] = {k: eval_model(recs, naive=True)[k]
                    for k in ("rankic_mean", "rankic_se", "n_weeks", "ls_total")}
    if ft_names:
        res["subset_ft_names"] = {k: eval_model(recs, tickers_subset=ft_names)[k]
                                  for k in ("rankic_mean", "rankic_se", "rankic_t", "n_weeks")}
        res["subset_holdout_names"] = {k: eval_model(recs, tickers_subset=hold_names)[k]
                                       for k in ("rankic_mean", "rankic_se", "rankic_t", "n_weeks")}
    out["models"][name] = res
    print(f"{name}: RankIC {res['rankic_mean']:+.4f} +/- {res['rankic_se']:.4f} "
          f"(t={res['rankic_t']:+.2f}, n={res['n_weeks']})  dir-acc {res['dir_acc']:.3f}  "
          f"L/S {res['ls_total']:+.2%} (stress {res['ls_total_stress']:+.2%}) "
          f"vs gspc {res['gspc_total']:+.2%}  beat {res['pct_weeks_beat_index']:.0%}  "
          f"MASE {res['mase']}")
    print(f"    neg-control (shuffled): RankIC {res['neg_control']['rankic_mean']:+.4f} "
          f"+/- {res['neg_control']['rankic_se']:.4f}")
    if ft_names:
        sh = res["subset_holdout_names"]
        print(f"    holdout-names subset: RankIC {sh['rankic_mean']:+.4f} +/- {sh['rankic_se']:.4f}")

# ---- paired ft-vs-base promotion gate + mechanical verdict ------------------
verdict = None
if args.ft_model and args.base_model and \
        args.ft_model in out["models"] and args.base_model in out["models"]:
    ft, base = out["models"][args.ft_model], out["models"][args.base_model]
    ow_ft = dict(zip(ft["weeks"], ft["rankic_by_week"]))
    ow_b = dict(zip(base["weeks"], base["rankic_by_week"]))
    common = [w for w in ft["weeks"] if w in ow_b
              and ow_ft[w] is not None and ow_b[w] is not None]
    diffs = [ow_ft[w] - ow_b[w] for w in common]
    pmean, pse, pt, pn = series_stats(diffs)
    out["paired_ft_minus_base"] = {"mean": pmean, "se": pse, "t": pt, "n_weeks": pn}

    hold = ft.get("subset_holdout_names") or {"rankic_mean": float("nan"),
                                              "rankic_se": float("nan")}
    nc = ft["neg_control"]
    crit = {
        "n_weeks_ge_min": ft["n_weeks"] >= args.min_weeks,
        "holdout_ic_minus_2se_gt_0":
            bool(hold["rankic_mean"] == hold["rankic_mean"]
                 and hold["rankic_mean"] - 2 * hold["rankic_se"] > 0),
        "beats_shuffled_null":
            bool(ft["rankic_mean"] == ft["rankic_mean"]
                 and ft["rankic_mean"] > nc["rankic_mean"] + 2 * nc["rankic_se"]),
        "net_ls_positive": bool(ft["ls_total"] == ft["ls_total"] and ft["ls_total"] > 0),
        "mase_sane": bool(ft["mase"] is None or ft["mase"] <= 1.05),
    }
    promote = bool(pmean == pmean and pmean > 0 and pt == pt and pt >= 1.5)
    verdict = {
        "criteria": crit,
        "edge_shown": all(crit.values()),
        "promote_finetune": promote,
        "note": ("all pre-registered criteria hold" if all(crit.values()) else
                 "pre-registered criteria NOT all met — no reliable edge demonstrated"),
    }
    out["verdict"] = verdict
    print(f"paired ft-base RankIC: {pmean:+.4f} +/- {pse:.4f} (t={pt:+.2f}, n={pn})  "
          f"promote_finetune={promote}")
    print(f"VERDICT: edge_shown={verdict['edge_shown']}  {json.dumps(crit)}")

json.dump(out, open(args.out_json, "w"), indent=2)
print("wrote", args.out_json)

# ---- trademiner-compatible superset summary ---------------------------------
if args.summary_out and args.ft_model and args.base_model:
    ft = out["models"].get(args.ft_model, {})
    base = out["models"].get(args.base_model, {})
    hold = ft.get("subset_holdout_names", {})
    summary = {
        "weeks": ft.get("n_weeks"), "horizon": H,
        "k_per_side": ft.get("k_per_side_mean"), "cost_bps": args.cost_bps,
        "ft_total": ft.get("ls_total"), "base_total": base.get("ls_total"),
        "sp500_total": ft.get("gspc_total"),
        "ft_vs_sp500": (ft.get("ls_total", float("nan")) -
                        ft.get("gspc_total", float("nan"))),
        "ft_weeks_beating_sp500": ft.get("pct_weeks_beat_index"),
        "base_weeks_beating_sp500": base.get("pct_weeks_beat_index"),
        "ft_rankic": ft.get("rankic_mean"), "ft_rankic_se": ft.get("rankic_se"),
        "base_rankic": base.get("rankic_mean"), "base_rankic_se": base.get("rankic_se"),
        "ft_dir_acc": ft.get("dir_acc"), "base_dir_acc": base.get("dir_acc"),
        "edge_shown": bool(verdict and verdict["edge_shown"]),
        "verdict": verdict,
        "holdout_rankic": hold.get("rankic_mean"),
        "holdout_rankic_se": hold.get("rankic_se"),
        "paired_ft_minus_base": out.get("paired_ft_minus_base"),
        "ls_total_stress": ft.get("ls_total_stress"),
        "survivorship_note": out["survivorship_note"],
    }
    json.dump(summary, open(args.summary_out, "w"), indent=2)
    print("wrote", args.summary_out)
