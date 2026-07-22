#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Render the self-contained HTML comparison report (+ machine-readable JSON) for
brain's three from-scratch forecasting models on real out-of-sample market data.

Inputs:
  skill_metrics.json  — from tools/oos_skill_report.py (RankIC, dir-acc, L/S vs ^gspc,
                        neg control, naive, MASE, per-model)
  report_inputs.json  — hand-assembled: latency table (cpu/gpu/npu), model sizes,
                        capability matrix, optimization before/after, eval scope notes

Output:
  out/model_comparison.html  — self-contained, theme-aware, responsive
  out/model_comparison.json  — every number

Usage: render_model_report.py skill_metrics.json report_inputs.json out.html out.json
"""
import sys, json, html, math

SKILL, EXTRA, OUT_HTML, OUT_JSON = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
S = json.load(open(SKILL))
X = json.load(open(EXTRA))

# categorical palette (validated dataviz default): blue / aqua / amber
COLORS = {"chronos2": "var(--c1)", "kronos": "var(--c2)", "fincast": "var(--c3)"}
MODEL_ORDER = [m for m in ["chronos2", "kronos", "fincast"] if m in S["models"]]


def esc(x):
    return html.escape(str(x))


def fmt(v, pct=False, sign=False, nd=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "&mdash;"
    if pct:
        s = f"{v*100:+.{nd}f}%" if sign else f"{v*100:.{nd}f}%"
        return s
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


# ---------- SVG chart helpers ----------
def line_chart(series, width=680, height=300, ylab="cumulative return", zero=True):
    """series: list of dicts {name, color, points:[y...], dash?}. x is index 0..n-1.
    Draws a shared x (week index), y auto-scaled, with a zero baseline."""
    pad_l, pad_r, pad_t, pad_b = 52, 14, 14, 34
    ally = [y for s in series for y in s["points"] if y is not None]
    if not ally:
        return "<p class='muted'>no data</p>"
    ymin, ymax = min(ally), max(ally)
    if zero:
        ymin, ymax = min(ymin, 0.0), max(ymax, 0.0)
    if ymax == ymin:
        ymax += 1e-6
    nmax = max(len(s["points"]) for s in series)
    def X(i): return pad_l + (width - pad_l - pad_r) * (i / max(1, nmax - 1))
    def Y(v): return pad_t + (height - pad_t - pad_b) * (1 - (v - ymin) / (ymax - ymin))
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" preserveAspectRatio="xMidYMid meet">']
    # y gridlines
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        y = Y(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" class="axlab" text-anchor="end">{v*100:.1f}%</text>')
    if zero:
        y0 = Y(0.0)
        parts.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{width-pad_r}" y2="{y0:.1f}" class="zero"/>')
    for s in series:
        pts = s["points"]
        d = " ".join(f'{"M" if i==0 else "L"}{X(i):.1f} {Y(v):.1f}' for i, v in enumerate(pts) if v is not None)
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2"{dash}/>')
        # endpoint marker + label
        if pts and pts[-1] is not None:
            parts.append(f'<circle cx="{X(len(pts)-1):.1f}" cy="{Y(pts[-1]):.1f}" r="3.2" fill="{s["color"]}"/>')
    parts.append(f'<text x="{pad_l}" y="{height-6}" class="axlab">week 1</text>')
    parts.append(f'<text x="{width-pad_r}" y="{height-6}" class="axlab" text-anchor="end">week {nmax}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_rankic(width=680, height=320):
    """RankIC +/- se per model + shuffled negative control, with a zero line and t labels."""
    items = []
    for m in MODEL_ORDER:
        d = S["models"][m]
        items.append((m, d["rankic_mean"], d["rankic_se"], d.get("rankic_t"), COLORS[m], False))
        nc = d["neg_control"]
        items.append((m + " (shuffled)", nc["rankic_mean"], nc["rankic_se"], nc.get("rankic_t"), "var(--muted-mark)", True))
    vals = [v for _, v, se, *_ in items] + [v + (se or 0) for _, v, se, *_ in items] + [v - (se or 0) for _, v, se, *_ in items]
    ymin, ymax = min(vals + [0]), max(vals + [0])
    span = (ymax - ymin) or 1e-6
    ymin -= span * 0.12; ymax += span * 0.18
    pad_l, pad_r, pad_t, pad_b = 52, 14, 14, 66
    n = len(items)
    bw = (width - pad_l - pad_r) / n * 0.6
    step = (width - pad_l - pad_r) / n
    def Y(v): return pad_t + (height - pad_t - pad_b) * (1 - (v - ymin) / (ymax - ymin))
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        y = Y(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" class="axlab" text-anchor="end">{v:+.02f}</text>')
    y0 = Y(0.0)
    parts.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{width-pad_r}" y2="{y0:.1f}" class="zero"/>')
    for i, (name, v, se, t, color, hatch) in enumerate(items):
        cx = pad_l + step * (i + 0.5)
        x = cx - bw / 2
        y = Y(max(v, 0)); h = abs(Y(v) - Y(0))
        fill = color if not hatch else "url(#hatch)"
        op = "0.9" if not hatch else "0.7"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{fill}" opacity="{op}"/>')
        # error bar
        if se and not math.isnan(se):
            parts.append(f'<line x1="{cx:.1f}" y1="{Y(v-se):.1f}" x2="{cx:.1f}" y2="{Y(v+se):.1f}" stroke="var(--ink)" stroke-width="1.4"/>')
            parts.append(f'<line x1="{cx-4:.1f}" y1="{Y(v+se):.1f}" x2="{cx+4:.1f}" y2="{Y(v+se):.1f}" stroke="var(--ink)" stroke-width="1.4"/>')
            parts.append(f'<line x1="{cx-4:.1f}" y1="{Y(v-se):.1f}" x2="{cx+4:.1f}" y2="{Y(v-se):.1f}" stroke="var(--ink)" stroke-width="1.4"/>')
        tl = f"t={t:+.2f}" if t is not None and not (isinstance(t, float) and math.isnan(t)) else ""
        parts.append(f'<text x="{cx:.1f}" y="{Y(v+se if se else v)-6:.1f}" class="axlab" text-anchor="middle">{tl}</text>')
        # rotated label
        short = name.replace(" (shuffled)", " shuf")
        parts.append(f'<text x="{cx:.1f}" y="{height-pad_b+14:.1f}" class="axlab" text-anchor="end" transform="rotate(-32 {cx:.1f} {height-pad_b+14:.1f})">{esc(short)}</text>')
    parts.append('<defs><pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--muted-mark)" opacity="0.28"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--muted-mark)" stroke-width="2"/></pattern></defs>')
    parts.append("</svg>")
    return "".join(parts)


def grouped_latency(width=680, height=330):
    """Grouped bars: per model, one bar per device (cpu/gpu/npu), log-ish linear ms."""
    lat = X["latency_ms"]
    devices = ["cpu", "gpu", "npu"]
    devcol = {"cpu": "var(--c1)", "gpu": "var(--c2)", "npu": "var(--c3)"}
    allv = [lat[m][dv] for m in MODEL_ORDER for dv in devices if lat.get(m, {}).get(dv) is not None]
    vmax = max(allv) * 1.15 if allv else 1
    pad_l, pad_r, pad_t, pad_b = 56, 14, 14, 46
    ng = len(MODEL_ORDER)
    gstep = (width - pad_l - pad_r) / ng
    bw = gstep / (len(devices) + 1)
    def Y(v): return pad_t + (height - pad_t - pad_b) * (1 - v / vmax)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for k in range(5):
        v = vmax * k / 4; y = Y(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" class="axlab" text-anchor="end">{v:.0f}</text>')
    parts.append(f'<text x="14" y="{pad_t+8}" class="axlab">ms</text>')
    for gi, m in enumerate(MODEL_ORDER):
        gx = pad_l + gstep * gi
        for di, dv in enumerate(devices):
            v = lat.get(m, {}).get(dv)
            x = gx + bw * (di + 0.5)
            if v is None:
                parts.append(f'<text x="{x+bw/2:.1f}" y="{Y(0)-4:.1f}" class="axlab" text-anchor="middle">n/a</text>')
                continue
            y = Y(v); h = Y(0) - y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.86:.1f}" height="{h:.1f}" rx="3" fill="{devcol[dv]}" opacity="0.9"/>')
            parts.append(f'<text x="{x+bw*0.43:.1f}" y="{y-4:.1f}" class="axlab tick" text-anchor="middle">{v:.0f}</text>')
        parts.append(f'<text x="{gx+gstep/2:.1f}" y="{height-pad_b+22:.1f}" class="axlab strong" text-anchor="middle">{esc(m)}</text>')
    parts.append("</svg>")
    legend = '<div class="legend">' + "".join(
        f'<span class="lg"><i style="background:{devcol[d]}"></i>{d.upper()}</span>' for d in devices) + "</div>"
    return "".join(parts) + legend


def opt_chart(width=520, height=210):
    o = X["optimization"]
    rows = o["measurements"]  # list of {label, before, after}
    vmax = max(max(r["before"], r["after"]) for r in rows) * 1.18
    pad_l, pad_r, pad_t, pad_b = 56, 60, 14, 40
    n = len(rows); gstep = (width - pad_l - pad_r) / n
    def Y(v): return pad_t + (height - pad_t - pad_b) * (1 - v / vmax)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    parts.append(f'<text x="14" y="{pad_t+8}" class="axlab">ms</text>')
    for gi, r in enumerate(rows):
        gx = pad_l + gstep * gi
        bw = gstep / 3
        for j, (lab, v, col) in enumerate([("before", r["before"], "var(--muted-mark)"), ("after", r["after"], "var(--c2)")]):
            x = gx + bw * (j + 0.4)
            y = Y(v); h = Y(0) - y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.8:.1f}" height="{h:.1f}" rx="3" fill="{col}" opacity="0.9"/>')
            parts.append(f'<text x="{x+bw*0.4:.1f}" y="{y-4:.1f}" class="axlab tick" text-anchor="middle">{v:.0f}</text>')
        spd = r["before"] / r["after"] if r["after"] else 0
        parts.append(f'<text x="{gx+gstep/2:.1f}" y="{height-pad_b+18:.1f}" class="axlab strong" text-anchor="middle">{esc(r["label"])}</text>')
        parts.append(f'<text x="{gx+gstep/2:.1f}" y="{height-pad_b+32:.1f}" class="axlab good" text-anchor="middle">{spd:.2f}× faster</text>')
    parts.append("</svg>")
    legend = '<div class="legend"><span class="lg"><i style="background:var(--muted-mark)"></i>dense MoE (before)</span><span class="lg"><i style="background:var(--c2)"></i>gather/scatter (after)</span></div>'
    return "".join(parts) + legend


# ---------- verdict logic ----------
def verdict(d):
    t = d.get("rankic_t")
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return ("no read", "warn")
    if abs(t) < 2.0:
        return ("no significant edge", "warn")
    return ("significant edge" if t > 0 else "significant NEGATIVE edge", "good" if t > 0 else "critical")


# ---------- assemble machine-readable JSON ----------
machine = {"meta": S["meta"], "scope": X.get("scope"), "models": {}, "latency_ms": X["latency_ms"],
           "model_sizes": X.get("model_sizes"), "optimization": X["optimization"], "capabilities": X["capabilities"]}
for m in MODEL_ORDER:
    d = S["models"][m]
    machine["models"][m] = {
        "rankic_mean": d["rankic_mean"], "rankic_se": d["rankic_se"], "rankic_t": d["rankic_t"],
        "n_weeks": d["n_weeks"], "dir_acc": d["dir_acc"], "mase": d.get("mase"),
        "ls_total": d["ls_total"], "gspc_total": d["gspc_total"], "pct_weeks_beat_index": d["pct_weeks_beat_index"],
        "neg_control_rankic_mean": d["neg_control"]["rankic_mean"], "neg_control_rankic_t": d["neg_control"]["rankic_t"],
        "verdict": verdict(d)[0],
    }
json.dump(machine, open(OUT_JSON, "w"), indent=2)

# ---------- build HTML ----------
scope = X.get("scope", {})
weeks_n = max((S["models"][m]["n_weeks"] for m in MODEL_ORDER), default=0)


def summary_table():
    head = "<tr><th>model</th><th>RankIC</th><th>&plusmn;se</th><th>t</th><th>dir&nbsp;acc</th><th>L/S&nbsp;total</th><th>^gspc</th><th>%wks&nbsp;beat</th><th>MASE</th><th>verdict</th></tr>"
    rows = [head]
    for m in MODEL_ORDER:
        d = S["models"][m]
        vtext, vcls = verdict(d)
        rows.append(
            f'<tr><td class="mono strong" style="color:{COLORS[m]}">{esc(m)}</td>'
            f'<td class="num">{fmt(d["rankic_mean"],sign=True,nd=3)}</td>'
            f'<td class="num">{fmt(d["rankic_se"],nd=3)}</td>'
            f'<td class="num">{fmt(d["rankic_t"],sign=True,nd=2)}</td>'
            f'<td class="num">{fmt(d["dir_acc"],pct=True,nd=1)}</td>'
            f'<td class="num">{fmt(d["ls_total"],pct=True,sign=True,nd=1)}</td>'
            f'<td class="num">{fmt(d["gspc_total"],pct=True,sign=True,nd=1)}</td>'
            f'<td class="num">{fmt(d["pct_weeks_beat_index"],pct=True,nd=0)}</td>'
            f'<td class="num">{fmt(d.get("mase"),nd=3)}</td>'
            f'<td><span class="pill {vcls}">{esc(vtext)}</span></td></tr>')
    # naive + shuffled reference rows
    d0 = S["models"][MODEL_ORDER[0]]
    nc = d0["neg_control"]
    rows.append(
        f'<tr class="ref"><td class="mono">shuffled control</td>'
        f'<td class="num">{fmt(nc["rankic_mean"],sign=True,nd=3)}</td><td class="num">{fmt(nc["rankic_se"],nd=3)}</td>'
        f'<td class="num">{fmt(nc["rankic_t"],sign=True,nd=2)}</td><td class="num">&mdash;</td>'
        f'<td class="num">{fmt(nc["ls_total"],pct=True,sign=True,nd=1)}</td><td class="num">&mdash;</td>'
        f'<td class="num">{fmt(nc["pct_weeks_beat_index"],pct=True,nd=0)}</td><td class="num">&mdash;</td>'
        f'<td><span class="pill warn">no skill (by design)</span></td></tr>')
    nv = d0["naive"]
    rows.append(
        f'<tr class="ref"><td class="mono">naive last-value</td><td class="num">&mdash;</td><td class="num">&mdash;</td>'
        f'<td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">&mdash;</td>'
        f'<td class="num">&mdash;</td><td class="num">1.000</td><td><span class="pill warn">MASE baseline</span></td></tr>')
    return '<div class="tablewrap"><table class="data">' + "".join(rows) + "</table></div>"


def cap_table():
    caps = X["capabilities"]
    fields = caps["fields"]
    head = "<tr><th>capability</th>" + "".join(f'<th class="mono" style="color:{COLORS[m]}">{esc(m)}</th>' for m in MODEL_ORDER) + "</tr>"
    rows = [head]
    for f in fields:
        cells = "".join(f'<td>{caps["models"][m].get(f["key"],"&mdash;")}</td>' for m in MODEL_ORDER)
        rows.append(f'<tr><td class="rowlab">{esc(f["label"])}</td>{cells}</tr>')
    return '<div class="tablewrap"><table class="data caps">' + "".join(rows) + "</table></div>"


def verdict_cards():
    cards = []
    for m in MODEL_ORDER:
        d = S["models"][m]
        vtext, vcls = verdict(d)
        cards.append(f'''<div class="card">
  <div class="card-h"><span class="dot" style="background:{COLORS[m]}"></span><span class="mono strong">{esc(m)}</span>
    <span class="pill {vcls}">{esc(vtext)}</span></div>
  <div class="kv"><span>RankIC</span><b class="num">{fmt(d["rankic_mean"],sign=True,nd=3)} &plusmn; {fmt(d["rankic_se"],nd=3)}</b></div>
  <div class="kv"><span>t-stat ({d["n_weeks"]} wks)</span><b class="num">{fmt(d["rankic_t"],sign=True,nd=2)}</b></div>
  <div class="kv"><span>directional acc</span><b class="num">{fmt(d["dir_acc"],pct=True,nd=1)}</b></div>
  <div class="kv"><span>L/S net vs ^gspc</span><b class="num">{fmt(d["ls_total"],pct=True,sign=True,nd=1)} vs {fmt(d["gspc_total"],pct=True,sign=True,nd=1)}</b></div>
  <div class="kv"><span>point MASE</span><b class="num">{fmt(d.get("mase"),nd=3)}</b></div>
</div>''')
    return "".join(cards)


# per-model L/S vs gspc small multiples (cumulative)
def ls_charts():
    blocks = []
    for m in MODEL_ORDER:
        d = S["models"][m]
        series = [
            {"name": f"{m} L/S", "color": COLORS[m], "points": d["ls_cum"]},
            {"name": "^gspc", "color": "var(--muted-mark)", "points": d["gspc_cum"], "dash": True},
            {"name": "shuffled", "color": "var(--faint-mark)", "points": d["neg_control"]["ls_cum"], "dash": True},
        ]
        blocks.append(f'''<figure class="fig">
  <figcaption><span class="mono strong" style="color:{COLORS[m]}">{esc(m)}</span> &mdash; top-{S["k"]}/bottom-{S["k"]} L/S, net {S["cost_bps"]}bps/side, vs ^gspc
    <span class="fnote">total {fmt(d["ls_total"],pct=True,sign=True,nd=1)} &middot; index {fmt(d["gspc_total"],pct=True,sign=True,nd=1)} &middot; beat {fmt(d["pct_weeks_beat_index"],pct=True,nd=0)} of weeks</span></figcaption>
  {line_chart(series)}
  <div class="legend"><span class="lg"><i style="background:{COLORS[m]}"></i>{esc(m)} L/S</span><span class="lg"><i style="background:var(--muted-mark)"></i>^gspc buy-hold</span><span class="lg"><i style="background:var(--faint-mark)"></i>shuffled control</span></div>
</figure>''')
    return "".join(blocks)


meta = S["meta"]
scope_line = (f'{scope.get("n_names","?")} liquid SP500 names &times; {weeks_n} weekly origins in 2026 '
              f'(horizon {meta["horizon"]} trading days, {meta["ctx"]}-bar context, {meta.get("device","?")} backend). '
              f'Every 2026 origin post-dates each checkpoint’s training cutoff — genuinely out-of-sample.')

CSS = """
:root{
  --bg:#eef1f4; --surface-1:#f8fafb; --surface-2:#ffffff; --ink:#12161c; --text-2:#4a5563; --muted:#6b7480;
  --line:#d8dee6; --c1:#2a78d6; --c2:#1baf7a; --c3:#eda100; --muted-mark:#8a94a1; --faint-mark:#c2c9d2;
  --good:#0ca30c; --warn:#b7791f; --crit:#d03b3b; --grid:#e4e9ef; --zero:#9aa4b1;
  --accent:#1c5cab;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1216; --surface-1:#171b21; --surface-2:#1b2028; --ink:#eef1f5; --text-2:#b3bcc7; --muted:#8993a0;
  --line:#2a313b; --c1:#3987e5; --c2:#199e70; --c3:#c98500; --muted-mark:#7a8592; --faint-mark:#3b434d;
  --good:#3ec13e; --warn:#e0a53a; --crit:#e66767; --grid:#232a33; --zero:#4a545f; --accent:#6ea8f0;
}}
:root[data-theme="light"]{--bg:#eef1f4;--surface-1:#f8fafb;--surface-2:#fff;--ink:#12161c;--text-2:#4a5563;--muted:#6b7480;--line:#d8dee6;--c1:#2a78d6;--c2:#1baf7a;--c3:#eda100;--muted-mark:#8a94a1;--faint-mark:#c2c9d2;--good:#0ca30c;--warn:#b7791f;--crit:#d03b3b;--grid:#e4e9ef;--zero:#9aa4b1;--accent:#1c5cab;}
:root[data-theme="dark"]{--bg:#0f1216;--surface-1:#171b21;--surface-2:#1b2028;--ink:#eef1f5;--text-2:#b3bcc7;--muted:#8993a0;--line:#2a313b;--c1:#3987e5;--c2:#199e70;--c3:#c98500;--muted-mark:#7a8592;--faint-mark:#3b434d;--good:#3ec13e;--warn:#e0a53a;--crit:#e66767;--grid:#232a33;--zero:#4a545f;--accent:#6ea8f0;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 80px}
.mono{font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace}
.num{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}
h1,h2,h3{font-family:Georgia,"Iowan Old Style","Times New Roman",serif;font-weight:600;text-wrap:balance;line-height:1.2}
h1{font-size:2.15rem;margin:0 0 .3em;letter-spacing:-.01em}
.eyebrow{font:600 12px/1 ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:14px}
.lede{font-size:1.06rem;color:var(--text-2);max-width:66ch;margin:0 0 6px}
h2{font-size:1.5rem;margin:56px 0 6px;padding-top:18px;border-top:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 20px;font-size:.95rem;max-width:70ch}
section{scroll-margin-top:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin:22px 0}
.card{background:var(--surface-2);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card-h{display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.dot{width:11px;height:11px;border-radius:50%;flex:none}
.kv{display:flex;justify-content:space-between;gap:10px;font-size:.9rem;padding:3px 0;color:var(--text-2)}
.kv b{color:var(--ink);font-weight:600}
.pill{font:600 11px/1 ui-monospace,monospace;padding:4px 8px;border-radius:20px;letter-spacing:.02em;white-space:nowrap}
.pill.good{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.pill.warn{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.pill.crit{background:color-mix(in srgb,var(--crit) 16%,transparent);color:var(--crit)}
.strong{font-weight:600}
.chart{width:100%;height:auto;display:block;background:var(--surface-1);border-radius:10px;margin-top:8px}
.grid{stroke:var(--grid);stroke-width:1}
.zero{stroke:var(--zero);stroke-width:1.3;stroke-dasharray:2 3}
.axlab{fill:var(--muted);font:11px ui-monospace,monospace}
.axlab.strong{fill:var(--text-2);font-weight:600;font-size:12px}
.axlab.good{fill:var(--good)}
.axlab.tick{font-size:10px}
.fig{margin:20px 0 8px}
figcaption{font-size:.92rem;color:var(--text-2);margin-bottom:2px}
.fnote{display:block;color:var(--muted);font-size:.82rem;margin-top:3px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:8px 2px 0;font:12px ui-monospace,monospace;color:var(--text-2)}
.lg{display:inline-flex;align-items:center;gap:6px}
.lg i{width:11px;height:11px;border-radius:3px;display:inline-block}
.tablewrap{overflow-x:auto;margin:16px 0}
table.data{border-collapse:collapse;width:100%;font-size:.9rem}
table.data th{font:600 11px/1.3 ui-monospace,monospace;letter-spacing:.03em;text-transform:uppercase;color:var(--muted);text-align:right;padding:8px 10px;border-bottom:1px solid var(--line)}
table.data th:first-child{text-align:left}
table.data td{padding:8px 10px;border-bottom:1px solid var(--grid);text-align:right}
table.data td:first-child{text-align:left}
table.data td.num{font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums}
table.data tr.ref td{color:var(--muted)}
table.caps td{text-align:left}
.rowlab{color:var(--text-2);font-weight:500}
.note{background:var(--surface-2);border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:8px;padding:14px 18px;margin:14px 0;font-size:.92rem;color:var(--text-2)}
.note b{color:var(--ink)}
.note.crit{border-left-color:var(--crit)}
.note.good{border-left-color:var(--good)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}
@media(max-width:720px){.grid2{grid-template-columns:1fr}}
footer{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:.82rem}
code{font-family:ui-monospace,monospace;font-size:.86em;background:color-mix(in srgb,var(--muted) 14%,transparent);padding:1px 5px;border-radius:4px}
a{color:var(--accent)}
"""

opt = X["optimization"]
html_doc = f"""<div class="wrap">
<div class="eyebrow">brain · forecasting · out-of-sample evaluation</div>
<h1>Can three from-scratch foundation models beat the index?</h1>
<p class="lede">Chronos-2, Kronos and FinCast — each reimplemented from scratch in brain (pure Rust + WGSL kernels, no PyTorch) — evaluated on <b>real 2026 out-of-sample</b> SP500 data through one model-agnostic harness. Cross-sectional skill, a market-neutral basket net of cost, per-device latency, and a parity-gated speedup.</p>
<p class="sub">{scope_line}</p>

<section>
<h2>Verdict</h2>
<p class="sub">The honest read is <b>RankIC significance</b> (does the cross-sectional ranking correlate with next-week returns, beyond noise), not a single cumulative-return line. A shuffled-prediction control sits beside every model to prove the pipeline invents no false skill.</p>
<div class="cards">{verdict_cards()}</div>
{summary_table()}
</section>

<section>
<h2>Cross-sectional skill &mdash; RankIC &plusmn; stderr</h2>
<p class="sub">Mean per-week Spearman rank correlation between predicted and realized {meta["horizon"]}-day returns, with standard error over {weeks_n} weeks and a t-stat. |t| &lt; 2 means no edge distinguishable from zero at this sample size. Each model's <b>shuffled negative control</b> (hatched) collapses to ≈ 0 — as it must.</p>
{bar_rankic()}
</section>

<section>
<h2>Market-neutral basket vs the index</h2>
<p class="sub">Each week: rank the universe by predicted return, long the top {S["k"]} / short the bottom {S["k"]} (equal-weight, market-neutral), realize next week's return net of {S["cost_bps"]} bps/side, compound. Plotted against ^gspc buy-hold and the shuffled control over the same weeks. A single line is noise-dominated at this sample size — read it beside the RankIC t-stat above.</p>
{ls_charts()}
</section>

<section>
<h2>Capabilities</h2>
<p class="sub">What each model natively emits and consumes — read from each model's <code>capabilities()</code> and what it actually produces through the shared <code>ForecastModel</code> seam.</p>
{cap_table()}
</section>

<section>
<h2>Latency by device</h2>
<p class="sub">Real per-forecast latency (warm, median) driving each model through the same <code>forecast()</code> seam on the CPU (wgsl-cpu Cranelift-JIT + AVX2) and GPU (wgpu/Vulkan, Intel Arc), and via the dedicated <code>brain npu</code> driver (OpenVINO, Intel NPU / <code>/dev/accel/accel0</code>). Kronos draws {meta.get("nsamples","?")} stochastic sample paths per forecast; Chronos-2 and FinCast are deterministic.</p>
{grouped_latency()}
<div class="tablewrap"><table class="data"><tr><th>model</th><th>params</th><th>on disk</th><th>CPU ms</th><th>GPU ms</th><th>NPU ms</th></tr>
{"".join(f'<tr><td class="mono strong" style="color:{COLORS[m]}">{esc(m)}</td><td class="num">{esc(X["model_sizes"][m]["params"])}</td><td class="num">{esc(X["model_sizes"][m]["disk"])}</td><td class="num">{fmt(X["latency_ms"][m].get("cpu"),nd=0)}</td><td class="num">{fmt(X["latency_ms"][m].get("gpu"),nd=0)}</td><td class="num">{fmt(X["latency_ms"][m].get("npu"),nd=0) if X["latency_ms"][m].get("npu") else "&mdash;"}</td></tr>' for m in MODEL_ORDER)}
</table></div>
</section>

<section>
<h2>Optimization &mdash; FinCast MoE gather/scatter</h2>
<div class="grid2">
<div>
<p class="sub">{opt["summary"]}</p>
<div class="note good"><b>Parity:</b> {opt["parity"]}</div>
</div>
<div>{opt_chart()}</div>
</div>
</section>

<section>
<h2>How to read this &mdash; honesty notes</h2>
<div class="note"><b>Small samples are noise-dominated.</b> {weeks_n} weekly origins is a small sample for a RankIC of order 0.0–0.1. A single cumulative-return line can look impressive or dismal purely by chance; the t-stat is the load-bearing number, and at this n most are indistinguishable from zero.</div>
<div class="note"><b>Negative control.</b> Shuffling each model's predictions within every week destroys any real signal and should give RankIC ≈ 0. It does — demonstrating the harness manufactures no false skill. A model whose real RankIC is not clearly separated from its own shuffled control has shown no edge.</div>
<div class="note"><b>Contamination.</b> Checkpoints predate 2026, so 2026 bars are genuinely unseen — but these names and their regimes resemble the pretraining distribution. Out-of-time is not out-of-distribution.</div>
<div class="note"><b>Cost is modelled, and it dominates.</b> The L/S curves are net of {S["cost_bps"]} bps/side per weekly rebalance. Zero-cost backtests (e.g. StockMixer) are an upper bound, not a comparison point.</div>
<div class="note crit"><b>Not financial advice.</b> This is an engineering evaluation of model implementations, not an investment recommendation. Nothing here is a signal to trade.</div>
</section>

<footer>
Generated by brain · <span class="mono">tools/oos_skill_report.py</span> + <span class="mono">tools/render_model_report.py</span> over <span class="mono">crates/cli/tests/oos_skill_eval.rs</span>.
Data: trademiner <span class="mono">stocks.db</span> (SP500 daily OHLCV + ^gspc, to 2026-07-21). Checkpoints: Chronos-2 (Apache-2.0), Kronos (MIT), FinCast (Apache-2.0, research/educational-use-only).
</footer>
</div>
<script>
// theme toggle stamp respected via CSS; add simple crosshair tooltips could go here.
</script>
"""

open(OUT_HTML, "w").write(f"<style>{CSS}</style>\n{html_doc}")
print("wrote", OUT_HTML, "and", OUT_JSON)
