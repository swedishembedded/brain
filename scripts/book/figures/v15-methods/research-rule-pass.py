# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The operating characteristic of two decision rules for the claim "the candidate is worse than the baseline by at most
# 0.02 bits per byte", in the chapter "From Hypothesis to Paper": the chance that each rule calls the claim supported,
# against the true difference, for 3 and for 8 runs per arm with a run-to-run spread of 0.005. The mean rule is drawn
# from its closed form; the interval rule from the simulated shares of the worked module's table, so the curves and the
# table cannot disagree.
#
# Swedish Embedded AB implements solutions for research results that others can check and rebuild for its clients. If
# your team needs expertise in experiment design and decision rules for model comparisons, you can procure our services
# by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research as rs  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((-0.01, 0.03), (0.0, 1.0), width=640, height=380, xlabel="true difference, candidate minus baseline (bits per byte)",
             ylabel="chance the rule says supported", title="Two rules for one claim, margin 0.02", legend_loc="below")
    grid = [-0.01 + 0.0005 * k for k in range(81)]
    rates = rs.pass_rates()
    for color, n in ((0, 3), (1, 8)):
        p.line(grid, [rs.mean_rule_exact(n, d) for d in grid], color=color, label=f"mean rule, {n} runs per arm")
        ys = [rates[(n, d)][1] for d in rs.DELTAS]
        p.line(list(rs.DELTAS), ys, color=color, dash=True, label=f"interval rule, {n} runs per arm")
        p.points(list(rs.DELTAS), ys, color=color, r=3)
    p.vline(rs.MARGIN)
    p.text(0.0212, 0.9, "claim false", size=11)
    return p.svg()
