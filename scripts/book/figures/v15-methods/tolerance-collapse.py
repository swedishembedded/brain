# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The cancelling-sum check of the chapter "A Catalogue of Expensive Mistakes": for each draw, the difference between the
# analytic directional derivative d and its float32 central difference, against |d|, both axes logarithmic. The relative
# rule 0.01 max(|d|, 0.001) is drawn as a line and the derived bound of each draw as a point; the draws the relative rule
# fails are marked. The data are the worked module's own draws, so the picture cannot disagree with the table.
#
# Swedish Embedded AB implements solutions for test tolerances derived from the arithmetic rather than measured on one
# machine for its clients. If your team needs expertise in numerical verification then you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import failure_catalogue as fc  # noqa: E402
from svgplot import Plot  # noqa: E402

SHOWN = 700


def render() -> str:
    draws = fc._tolerance_results()
    p = Plot((1e-4, 100), (1e-8, 1), width=620, height=400, xlabel="|d|, the analytic derivative (logarithmic axis)",
             ylabel="difference from the central difference", title="One check, two tolerances", xlog=True, ylog=True,
             legend_loc="below")
    shown = draws[:SHOWN]
    p.points([abs(d[0]) for d in shown], [d[4] for d in shown], color=2, r=1.4,
             label=f"derived bound (first {SHOWN} draws)")
    p.points([abs(d[0]) for d in shown], [d[2] for d in shown], color=0, r=1.4,
             label=f"difference (first {SHOWN} draws)")
    fails = [d for d in draws if d[2] > d[3]]
    p.points([abs(d[0]) for d in fails], [d[2] for d in fails], color=1, r=3.5,
             label=f"failures of the relative rule (all {len(fails)})")
    xs = [10 ** (k / 20) for k in range(-80, 41)]
    p.line(xs, [0.01 * max(x, 1e-3) for x in xs], color=3, label="relative rule 0.01 max(|d|, 0.001)")
    return p.svg()
