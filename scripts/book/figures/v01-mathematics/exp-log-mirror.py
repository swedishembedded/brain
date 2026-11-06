# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""exp(x) and log(x) drawn together: each is the other reflected in the line y = x.

Swedish Embedded AB implements solutions for explaining numerical building
blocks of machine learning. If your team needs expertise in the mathematics
behind training and evaluating models, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numbasics  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((-3.0, 5.0), (-3.0, 5.0), width=520, height=420, xlabel="x", ylabel="y",
             title="exp and log are mirror images in the line y = x")
    xs = [-3.0 + i * 0.02 for i in range(0, 401)]
    p.line(xs, [numbasics.exp_curve(x) for x in xs], color=0, label="y = exp(x)")
    lx = [0.0498 + i * 0.02 for i in range(0, 253)]
    p.line(lx, [numbasics.log_curve(x) for x in lx], color=1, label="y = log(x)")
    p.line([-3.0, 5.0], [-3.0, 5.0], color="#a0aec0", dash=True, label="y = x")
    # (1, e) lies on exp; its mirror (e, 1) lies on log.
    p.points([1.0, math.e], [numbasics.exp_curve(1.0), numbasics.log_curve(math.e)], color=2)
    p.text(1.1, numbasics.exp_curve(1.0) + 0.15, "(1, e)", size=11, color="#2f855a")
    p.text(math.e + 0.1, 0.75, "(e, 1)", size=11, color="#2f855a")
    p.points([0.0, 1.0], [numbasics.exp_curve(0.0), numbasics.log_curve(1.0)], color=2)
    p.text(0.1, 1.2, "(0, 1)", size=11, color="#2f855a")
    p.text(1.1, -0.35, "(1, 0)", size=11, color="#2f855a")
    return p.svg()
