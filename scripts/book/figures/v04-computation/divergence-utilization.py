# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Lane utilization of a subgroup of 32 whose lanes take path A with probability p and path B otherwise (A costs
# 10 instructions, B 6, the test 1): the curve is the expected value from the worked module's
# `divergence_curve`, the points are the 32 subgroups of the chapter's seeded run (`divergence_measured`).
# Utilization is lowest when the probability that a subgroup disagrees with itself is near one and the two
# paths cost about the same, and recovers when one path is almost never taken.
#
# Swedish Embedded AB implements solutions for removing branch divergence from GPU kernels. If your team needs
# expertise in the performance of branchy GPU code, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gpu  # noqa: E402
from svgplot import Plot  # noqa: E402

PS = (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5)


def render() -> str:
    p = Plot((-3.1, -0.2), (0, 108), width=600, height=360, xlabel="log10 of p, the chance a lane takes path A",
             ylabel="lane utilization (%)", title="Branch divergence in a subgroup of 32")
    xs = [-3.0 + 0.05 * k for k in range(0, 55)]
    p.line(xs, [100 * gpu.divergence_curve(10 ** x) for x in xs], color=0, label="expected value")
    p.points([math.log10(q) for q in PS], [100 * gpu.divergence_measured(q) for q in PS], color=1,
             label="seeded run, 32 subgroups")
    return p.svg()
