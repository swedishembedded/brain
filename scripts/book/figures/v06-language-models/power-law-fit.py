# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A straight line on log-log axes is a power law. Blue: eight noisy synthetic points of loss = 15 P^-0.08 and the least-squares line
# through them (solid over the data, dashed to ten times the largest size). Orange: eight noise-free synthetic points of
# loss = 1.7 + 40 P^-0.3, with the straight lines fitted to the first four and to the last four; the lines disagree because the
# curve bends towards its floor. Numbers from `model_cost.power_law_figure_data`; none of this is the loss of a real model.
#
# Swedish Embedded AB implements solutions for planning training runs from small experiments. If your team needs expertise in
# scaling-law fitting and its uncertainty, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import model_cost as mc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    d = mc.power_law_figure_data()
    lo, hi = 5e5, mc.FIT_SIZES[-1] * 10
    p = Plot((lo, hi), (1.4, 6), width=620, height=420, xlabel="parameters P (log scale)", ylabel="loss (log scale)",
             title="Fitting a power law to synthetic points", legend_loc="below", xlog=True, ylog=True)
    a, b = d["fit"]
    last = mc.FIT_SIZES[-1]
    p.line([lo, last], [math.exp(a + b * math.log(lo)), math.exp(a + b * math.log(last))], color=0, label="least-squares line through the blue points")
    p.line([last, hi], [math.exp(a + b * math.log(last)), math.exp(a + b * math.log(hi))], color=0, dash=True)
    p.points([x for x, _ in d["points"]], [y for _, y in d["points"]], color=0, label="loss = 15 P^-0.08 with 1% noise")
    p.hline(1.7, color="#a0aec0")
    p.points([x for x, _ in d["floor"]], [y for _, y in d["floor"]], color=1, label="loss = 1.7 + 40 P^-0.3, no noise", r=3.5)
    for key, color, label in (("low", 2, "line fitted to the first four orange points"), ("high", 3, "line fitted to the last four orange points")):
        a2, b2 = d[key]
        p.line([lo, hi], [math.exp(a2 + b2 * math.log(lo)), math.exp(a2 + b2 * math.log(hi))], color=color, dash=True, label=label)
    p.text(2.2e6, 1.6, "floor 1.7", size=11, color="#4a5568")
    return p.svg()
