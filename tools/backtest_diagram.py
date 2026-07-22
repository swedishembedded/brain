#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Walk-forward proof: render cumulative performance of the base Kronos strategy vs.
the fine-tuned strategy vs. the SP500 index vs. an equal-weight market — from two
forecast dumps (base + fine-tuned) produced by the in-process rankic harness over
the SAME held-out weeks (fine-tune data ended before these weeks → no look-ahead).

Each week: rank the universe by predicted return, long the top-K / short the
bottom-K (equal weight, market-neutral), realize the next-week return; compound.
The SP500 line is ^gspc over the same weeks; the "weekly trader" is the fine-tuned
L/S book net of transaction cost — what a user actually gets running the strategy.

Usage: backtest_diagram.py <base.json> <ft.json> <stocks.db> <out.html>
"""
import sys, json, math, statistics, html
from collections import defaultdict

BASE_J, FT_J, DB, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
K = 6              # names per side
COST_BPS = 5.0     # per side, per weekly rebalance

def load(p):
    d = json.load(open(p))
    by_o = defaultdict(dict)   # origin -> {ticker: (pred, real)}
    dates = {}
    for r in d["records"]:
        by_o[r["o"]][r["ticker"]] = (r["pred"], r["real"])
        dates[r["o"]] = r["date"]
    return d["meta"], by_o, dates

meta, base, dates = load(BASE_J)
_, ft, _ = load(FT_J)
H = meta["horizon"]
# Only weeks present in BOTH dumps — the fine-tuned run may be partial (its
# harness checkpoints per-origin, so a timeout on a contended box still yields
# usable weeks); the comparison stays apples-to-apples on the shared origins.
origins = sorted(set(base) & set(ft))
if len(origins) < len(base):
    print(f"note: aligning on {len(origins)} weeks shared by base({len(base)}) and ft({len(ft)})")

def ls_return(row):
    """row: {ticker:(pred,real)} -> equal-weight top-K long minus bottom-K short realized."""
    items = sorted(row.items(), key=lambda kv: kv[1][0], reverse=True)  # by pred desc
    k = min(K, len(items) // 2)
    if k < 1:
        return None
    longs = [items[i][1][1] for i in range(k)]
    shorts = [items[-1 - i][1][1] for i in range(k)]
    return statistics.fmean(longs) - statistics.fmean(shorts)

def market_return(row):
    return statistics.fmean(v[1] for v in row.values())

# ^gspc over each origin's horizon, aligned by date
import sqlite3
con = sqlite3.connect(DB); cur = con.cursor()
gs = cur.execute("SELECT Date, Close FROM stock_data WHERE Ticker='^gspc' ORDER BY Date ASC").fetchall()
gdates = [d[:10] for d, _ in gs]; gclose = [c for _, c in gs]
gidx = {d: i for i, d in enumerate(gdates)}
def gspc_return(date):
    # find nearest index <= date, take H trading days ahead
    i = gidx.get(date)
    if i is None:
        # nearest prior
        cand = [j for j, d in enumerate(gdates) if d <= date]
        if not cand:
            return None
        i = cand[-1]
    if i + H < len(gclose) and gclose[i] and gclose[i + H]:
        return gclose[i + H] / gclose[i] - 1.0
    return None

cost = 4.0 * COST_BPS / 1e4   # both legs, round trip, on the spread
series = {"Base Kronos (L/S)": [], "Fine-tuned (L/S)": [], "SP500 (^gspc)": [], "Equal-weight market": []}
labels = []
for o in origins:
    b = ls_return(base[o]); f = ls_return(ft[o]); m = market_return(base[o]); g = gspc_return(dates[o])
    if b is None or f is None or g is None:
        continue
    labels.append(dates[o])
    series["Base Kronos (L/S)"].append(b - cost)
    series["Fine-tuned (L/S)"].append(f - cost)
    series["SP500 (^gspc)"].append(g)
    series["Equal-weight market"].append(m)

def cumulative(rets):
    out, acc = [], 1.0
    for r in rets:
        acc *= (1.0 + r); out.append(acc - 1.0)
    return out

cum = {k: cumulative(v) for k, v in series.items()}
n = len(labels)

# --- skill metrics (the durable read; 11-week cumulative return is noise-dominated) ---
def rank_ic(byo):
    """Mean per-week cross-sectional IC (corr of predicted vs realized return) ± stderr."""
    ics = []
    for o in origins:
        rows = list(byo[o].values())
        ps = [p for p, _ in rows]; rs = [r for _, r in rows]
        mp, mr = statistics.fmean(ps), statistics.fmean(rs)
        cv = sum((p - mp) * (r - mr) for p, r in rows)
        sp = math.sqrt(sum((p - mp) ** 2 for p in ps)); sr = math.sqrt(sum((r - mr) ** 2 for r in rs))
        if sp > 0 and sr > 0:
            ics.append(cv / (sp * sr))
    m = statistics.fmean(ics) if ics else float("nan")
    se = statistics.pstdev(ics) / math.sqrt(len(ics)) if len(ics) > 1 else float("nan")
    return m, se

def dir_acc(byo):
    tot = ok = 0
    for o in origins:
        for p, r in byo[o].values():
            tot += 1; ok += (p > 0) == (r > 0)
    return ok / tot if tot else float("nan")

ft_ic, base_ic = rank_ic(ft), rank_ic(base)
ft_da, base_da = dir_acc(ft), dir_acc(base)
# "edge" is demonstrated only if fine-tuned IC is positive AND at least a stderr above zero.
edge_shown = ft_ic[0] == ft_ic[0] and (ft_ic[0] - ft_ic[1]) > 0

def stats(rets):
    if not rets:
        return (float("nan"),) * 3
    tot = 1.0
    for r in rets: tot *= (1 + r)
    tot -= 1
    sd = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    sharpe = (statistics.fmean(rets) / sd * math.sqrt(52)) if sd > 0 else float("nan")  # weekly → annualized
    win = sum(1 for r in rets if r > 0) / len(rets)
    return tot, sharpe, win

# beat-the-index: fine-tuned L/S weekly excess over SP500
ft_ex = [series["Fine-tuned (L/S)"][i] - series["SP500 (^gspc)"][i] for i in range(n)]
base_ex = [series["Base Kronos (L/S)"][i] - series["SP500 (^gspc)"][i] for i in range(n)]
ft_beat = sum(1 for x in ft_ex if x > 0) / n if n else float("nan")
base_beat = sum(1 for x in base_ex if x > 0) / n if n else float("nan")

def pc(x, dp=1):
    return "—" if x != x else f"{x*100:+.{dp}f}%"

# ---- render ----
COL = {"Base Kronos (L/S)": "#c07d29", "Fine-tuned (L/S)": "#0d6e88", "SP500 (^gspc)": "#6a747e", "Equal-weight market": "#b0b4b8"}
def line_svg(w=940, h=340, pad=48):
    allv = [x for v in cum.values() for x in v] + [0.0]
    lo, hi = min(allv), max(allv); rg = (hi - lo) or 1
    def X(i): return pad + i * (w - 2 * pad) / max(1, n - 1)
    def Y(v): return h - pad - (v - lo) / rg * (h - 2 * pad)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    out.append(f'<line x1="{pad}" y1="{Y(0):.1f}" x2="{w-pad}" y2="{Y(0):.1f}" class="axis"/>')
    for name, v in cum.items():
        pts = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(v))
        dash = ' stroke-dasharray="5 4"' if "SP500" in name or "Equal" in name else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COL[name]}" stroke-width="{3 if "Fine" in name else 2}"{dash}/>')
        out.append(f'<circle cx="{X(n-1):.1f}" cy="{Y(v[-1]):.1f}" r="3.5" fill="{COL[name]}"/>')
        out.append(f'<text x="{X(n-1)+6:.1f}" y="{Y(v[-1])+3:.1f}" class="lbl" fill="{COL[name]}">{pc(v[-1])}</text>')
    for i in range(0, n, max(1, n // 8)):
        out.append(f'<text x="{X(i):.1f}" y="{h-14}" class="axl" text-anchor="middle">{html.escape(labels[i][2:])}</text>')
    out.append("</svg>")
    return "".join(out)

IC = {"Fine-tuned (L/S)": ft_ic, "Base Kronos (L/S)": base_ic}
DA = {"Fine-tuned (L/S)": ft_da, "Base Kronos (L/S)": base_da}
rows = ""
for name in ["Fine-tuned (L/S)", "Base Kronos (L/S)", "SP500 (^gspc)", "Equal-weight market"]:
    tot, sh, win = stats(series[name])
    ic = IC.get(name); da = DA.get(name)
    ic_s = f"{ic[0]:+.3f}<span class='se'>±{ic[1]:.3f}</span>" if ic else "—"
    da_s = f"{da*100:.0f}%" if da is not None else "—"
    beat = pc(ft_beat, 0) if name.startswith("Fine") else (pc(base_beat, 0) if name.startswith("Base") else "—")
    rows += (f'<tr><td><span class="sw" style="background:{COL[name]}"></span>{name}</td>'
             f'<td class="num">{ic_s}</td><td class="num">{da_s}</td>'
             f'<td class="num" style="color:{COL[name]};font-weight:700">{pc(tot)}</td>'
             f'<td class="num">{sh:.2f}</td><td class="num">{beat}</td></tr>')

ft_tot = stats(series["Fine-tuned (L/S)"])[0]
sp_tot = stats(series["SP500 (^gspc)"])[0]
edge = ft_tot - sp_tot

HTML = f"""<title>Fine-tuned Kronos — walk-forward vs. SP500</title>
<style>
:root{{--bg:#f5f4f1;--surface:#fff;--ink:#181a1d;--muted:#6a747e;--line:#e5e3dd;--acc:#0d6e88;--mono:ui-monospace,Menlo,Consolas,monospace}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1316;--surface:#171c20;--ink:#e9edf0;--muted:#93a0aa;--line:#262d33;--acc:#42b2ce}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:40px 24px 72px}}
h1{{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}}.sub{{color:var(--muted);margin:0 0 24px;max-width:70ch}}
.tiles{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}
@media(max-width:640px){{.tiles{{grid-template-columns:1fr}}}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px}}
.tl{{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
.tv{{font-size:26px;font-weight:800;margin:6px 0 2px;font-variant-numeric:tabular-nums}}.ts{{font-size:12.5px;color:var(--muted)}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px;margin:16px 0}}
.chart{{width:100%;height:auto}}.axis{{stroke:var(--line);stroke-width:1.5}}.axl{{fill:var(--muted);font-family:var(--mono);font-size:10px}}
.lbl{{font-family:var(--mono);font-size:11px;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:14.5px}}td,th{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}}
th{{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:middle}}
.note{{color:var(--muted);font-size:13px}}.good{{color:var(--acc)}}
.se{{font-size:11px;color:var(--muted);font-weight:400;margin-left:2px}}
.verdict{{background:var(--surface);border:1px solid var(--line);border-left:4px solid {'#3f9d5a' if edge_shown else '#c48a2a'};border-radius:12px;padding:16px 18px;margin:16px 0}}
.verdict b{{color:{'#3f9d5a' if edge_shown else '#c48a2a'}}}
</style>
<div class="wrap">
<h1>Fine-tuned Kronos vs. the SP500 — walk-forward evaluation</h1>
<p class="sub">{n} weekly rebalances on held-out weeks (the fine-tune's training data ended before this period — no look-ahead). Each week: rank the {meta['n_tickers']}-name universe, long the top {K} / short the bottom {K}, hold {H} days, net of {COST_BPS:.0f} bps/side. The honest read is <b>RankIC</b> (does predicted rank track realized rank?), not the noise-dominated {n}-week return.</p>
<div class="verdict"><b>{'EDGE SHOWN' if edge_shown else 'NO EDGE DEMONSTRATED ON THIS SAMPLE'}.</b>
 Fine-tuned RankIC {ft_ic[0]:+.3f}±{ft_ic[1]:.3f}, base {base_ic[0]:+.3f}±{base_ic[1]:.3f} — {'the fine-tune shows positive, above-noise ranking skill.' if edge_shown else 'both are statistically indistinguishable from zero over ' + str(n) + ' weeks, so neither model shows reliable directional skill here.'}
 The fine-tune lowered held-out next-token loss (its training objective) and generalized to unseen names, but on this window that did <b>not</b> convert into trading edge. A trustworthy verdict needs many more weeks of live RankIC.</div>
<div class="tiles">
 <div class="tile"><div class="tl">Fine-tuned RankIC</div><div class="tv">{ft_ic[0]:+.3f}</div><div class="ts">±{ft_ic[1]:.3f} se · base {base_ic[0]:+.3f}</div></div>
 <div class="tile"><div class="tl">Directional acc.</div><div class="tv">{ft_da*100:.0f}%</div><div class="ts">ft vs base {base_da*100:.0f}% · 50% = coin-flip</div></div>
 <div class="tile"><div class="tl">Fine-tuned total (net)</div><div class="tv">{pc(ft_tot)}</div><div class="ts">vs SP500 {pc(sp_tot)} · base {pc(stats(series['Base Kronos (L/S)'])[0])}</div></div>
</div>
<div class="panel">{line_svg()}</div>
<div class="panel"><table>
<thead><tr><th>Strategy</th><th class="num">RankIC ±se</th><th class="num">Dir. acc</th><th class="num">Total</th><th class="num">Sharpe (ann.)</th><th class="num">Beat SP500</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="note">L/S = market-neutral long/short book (the "weekly trader" is the fine-tuned line). RankIC = mean per-week cross-sectional corr of predicted vs realized {H}-day return (the skill metric; ±se over {n} weeks). SP500 = ^gspc buy-hold over the same weeks. Sharpe annualized from weekly (×√52). {n}-week cumulative return is dominated by a few idiosyncratic moves — read RankIC, not the last point.</p></div>
<p class="note">Not financial advice. Fine-tune adapted on data strictly before this window; the comparison is genuinely out-of-sample. Small sample — this proves the <i>machinery</i> (leak-safe fine-tune → gated promotion → walk-forward eval), not a market edge.</p>
</div>"""
open(OUT, "w").write(HTML)
# a small machine-readable summary the weekly strategy script cites when it runs.
summary = {
    "weeks": n, "horizon": H, "k_per_side": K, "cost_bps": COST_BPS,
    "ft_total": stats(series["Fine-tuned (L/S)"])[0],
    "base_total": stats(series["Base Kronos (L/S)"])[0],
    "sp500_total": sp_tot, "ft_vs_sp500": edge,
    "ft_weeks_beating_sp500": ft_beat, "base_weeks_beating_sp500": base_beat,
    "ft_rankic": ft_ic[0], "ft_rankic_se": ft_ic[1],
    "base_rankic": base_ic[0], "base_rankic_se": base_ic[1],
    "ft_dir_acc": ft_da, "base_dir_acc": base_da,
    "edge_shown": bool(edge_shown),
}
import os
json.dump(summary, open(os.path.join(os.path.dirname(OUT) or ".", "backtest_summary.json"), "w"), indent=2)
print(f"wrote {OUT}")
print(f"weeks={n}  ft_total={pc(ft_tot)}  base_total={pc(stats(series['Base Kronos (L/S)'])[0])}  sp500={pc(sp_tot)}  ft_vs_sp={pc(edge)}  ft_beat={pc(ft_beat,0)}")
