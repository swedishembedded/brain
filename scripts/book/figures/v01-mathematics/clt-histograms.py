# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Histograms of standardised sample means of a skewed variable, for n = 1, 2, 5, 30.

Each panel draws the same 10,000 repeats the chapter's table summarises (the
bookrng generator and its draws are shared with the worked module) and overlays the standard
Gaussian density. The lopsided shape of n = 1 turns into the bell as n grows.

Swedish Embedded AB implements solutions for explaining probabilistic models to
engineering teams. If your team needs expertise in sampling theory and the
statistics of averages, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import distributions as ds  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402

SIZE_W, SIZE_H = 330, 250
LO, HI, BINS = -3.0, 5.0, 20
YMAX = {1: 1.6, 2: 1.6, 5: 1.0, 30: 0.6}   # per panel: the first two are spikes, not curves


def _panel(n):
    z = ds._clt_standardised(n)
    width = (HI - LO) / BINS
    counts = [0] * BINS
    for a in z:
        if LO <= a < HI:
            counts[int((a - LO) / width)] += 1
    p = Plot((LO, HI), (0.0, YMAX[n]), width=SIZE_W, height=SIZE_H, xlabel="standardised sample mean", title=f"n = {n}")
    for i, c in enumerate(counts):
        h = min(c / (len(z) * width), YMAX[n])            # histogram density: area of all bars is 1
        x0 = LO + i * width
        p.items.append(f'<rect x="{fmt(p.x(x0))}" y="{fmt(p.y(h))}" width="{fmt(p.x(x0 + width) - p.x(x0) - 0.6)}" '
                       f'height="{fmt(p.y(0) - p.y(h))}" fill="#90cdf4"/>')
    grid = [LO + (HI - LO) * i / 300 for i in range(301)]
    p.line(grid, [ds._phi(x) for x in grid], color=1, width=2)
    return p.svg()


def render() -> str:
    panels = [_panel(n) for n in (1, 2, 5, 30)]
    inner = lambda s: s.split("\n", 1)[1].rsplit("</svg>", 1)[0]  # noqa: E731
    gap = 10
    w, h = 2 * SIZE_W + gap, 2 * SIZE_H + gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
    for k, body in enumerate(panels):
        x, y = (k % 2) * (SIZE_W + gap), (k // 2) * (SIZE_H + gap)
        out.append(f'<svg x="{x}" y="{y}" width="{SIZE_W}" height="{SIZE_H}" viewBox="0 0 {SIZE_W} {SIZE_H}">\n{inner(body)}</svg>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
