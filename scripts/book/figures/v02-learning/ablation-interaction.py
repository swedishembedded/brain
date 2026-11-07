# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The 2 x 2 ablation as an interaction plot: mean test accuracy for each of the four pipelines.

The horizontal axis is whether the inputs are centred, the two lines are whether the interaction feature is added.
If the two parts acted independently the lines would be parallel. Here the interaction feature lowers accuracy when
the inputs are not centred and raises it when they are, so the lines cross the gap in opposite directions. Bars are
95 percent t intervals over 100 seeds, from the chapter's worked module.

Swedish Embedded AB implements solutions for experiments whose results a customer can believe and repeat. If your
team needs expertise in experiment design for machine learning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments as e  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    cell = {k: e._interval([100 * a for a in e._seed_accuracies(k)]) for k in e.PIPELINES}
    p = Plot((-0.4, 1.4), (67, 76), width=560, height=380, ylabel="mean test accuracy (%) over 100 seeds",
             title="Ablation 2 x 2: the interaction feature helps only after centring")
    rows = (("without the interaction feature", ("plain", "centring"), 0), ("with the interaction feature", ("interaction", "both"), 1))
    for label, (off, on), color in rows:
        ys = [cell[off][0], cell[on][0]]
        p.line([0, 1], ys, color=color, width=2.4, label=label)
        for x, key in ((0, off), (1, on)):
            m, s, se, lo, hi = cell[key]
            p.line([x, x], [lo, hi], color=color, width=2.0)
            p.points([x], [m], color=color, r=4.5)
            p.text(x - 0.06 if x == 0 else x + 0.06, m - 0.1, f"{m:.1f}", anchor="end" if x == 0 else "start", size=11, color="#1a202c")
    svg = p.svg()
    labels = {"0": "inputs not centred", "1": "inputs centred"}
    return re.sub(r'(font-size="11" fill="#4a5568" text-anchor="middle">)([0-9.]+)</text>',
                  lambda m: m.group(1) + labels.get(m.group(2), "") + "</text>", svg)
