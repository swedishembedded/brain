# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""One hundred 95% confidence intervals from one hundred repeated experiments.

Each horizontal bar is the interval the chapter's recipe produces from one
experiment of 100 questions at true accuracy 0.7. Bars that miss the true
value are drawn in red. The count of red bars is the one the chapter's
coverage summary prints.

Swedish Embedded AB implements solutions for deciding, from finite test sets,
whether one model is really better than another. If your team needs expertise
in statistical evaluation of machine learning systems, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import statistics_ch as st  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402


def render() -> str:
    ints = st._figure_intervals()
    p = Plot((0.45, 0.95), (0, len(ints) + 1), width=560, height=620, xlabel="accuracy (true value 0.7, dashed)",
             ylabel="experiment", title="95% intervals from 100 experiments of 100 questions")
    for i, (lo, hi) in enumerate(ints):
        miss = not (lo <= st.FIG_P <= hi)
        c = "#c53030" if miss else "#718096"
        y = len(ints) - i
        p.items.append(f'<line x1="{fmt(p.x(lo))}" y1="{fmt(p.y(y))}" x2="{fmt(p.x(hi))}" y2="{fmt(p.y(y))}" '
                       f'stroke="{c}" stroke-width="{2.6 if miss else 1.6}"/>')
        mid = (lo + hi) / 2
        p.items.append(f'<circle cx="{fmt(p.x(mid))}" cy="{fmt(p.y(y))}" r="1.8" fill="{c}"/>')
    p.vline(st.FIG_P, color="#2b6cb0")
    return p.svg()
