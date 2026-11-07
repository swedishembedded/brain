# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Sampling distribution of an accuracy for 25, 100, 400 and 1600 questions.

Each panel is a histogram of the 1000 accuracies the chapter's table
summarises (a model that answers each question right with probability 0.7),
with the Gaussian of standard deviation sqrt(p(1-p)/n) drawn over it. The
horizontal range is the same in every panel, so the narrowing is visible.

Swedish Embedded AB implements solutions for deciding, from finite test sets,
whether one model is really better than another. If your team needs expertise
in statistical evaluation of machine learning systems, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import statistics_ch as st  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402

W, H = 330, 250
LO, HI = 0.5, 0.9
P = 0.7


def _panel(n):
    vals = st.sample_means(n)
    width = max(1 / n, 0.01)          # an accuracy over n questions is a multiple of 1/n: one bar per value when n is small
    counts = {}
    for v in vals:
        i = int(math.floor(v / width + 0.5))     # bar i is centred on i * width
        counts[i] = counts.get(i, 0) + 1
    sd = math.sqrt(P * (1 - P) / n)
    peak = 1 / (sd * math.sqrt(2 * math.pi))
    top = max(1.25 * peak, 1.1 * max(counts.values()) / (len(vals) * width))
    p = Plot((LO, HI), (0.0, top), width=W, height=H, xlabel="accuracy on n questions", title=f"n = {n}, SE = {sd:.4f}")
    for i, c in counts.items():
        h = c / (len(vals) * width)
        x0 = (i - 0.5) * width
        p.items.append(f'<rect x="{fmt(p.x(x0))}" y="{fmt(p.y(h))}" width="{fmt(p.x(x0 + width) - p.x(x0) - 0.6)}" '
                       f'height="{fmt(p.y(0) - p.y(h))}" fill="#90cdf4"/>')
    grid = [LO + (HI - LO) * i / 300 for i in range(301)]
    p.line(grid, [math.exp(-((x - P) / sd) ** 2 / 2) / (sd * math.sqrt(2 * math.pi)) for x in grid], color=1, width=2)
    p.vline(P)
    return p.svg()


def render() -> str:
    panels = [_panel(n) for n in st.SAMPLE_SIZES]
    inner = lambda s: s.split("\n", 1)[1].rsplit("</svg>", 1)[0]  # noqa: E731
    gap = 10
    w, h = 2 * W + gap, 2 * H + gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
    for k, body in enumerate(panels):
        x, y = (k % 2) * (W + gap), (k // 2) * (H + gap)
        out.append(f'<svg x="{x}" y="{y}" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n{inner(body)}</svg>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
