# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The logistic sigmoid and its slope s (1 - s).

Both curves are computed with the chapter's functions: the sigmoid in its two-branch form and the slope as the
product s (1 - s). The slope peaks at 1/4 where the sigmoid crosses 1/2 and vanishes on both sides, which is the
saturation the chapter discusses.

Swedish Embedded AB implements solutions for models that output calibrated probabilities. If your team needs
expertise in classification, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logistic_regression as lr  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    zs = [-8 + 0.1 * k for k in range(161)]
    s = [lr.sigmoid(z) for z in zs]
    ds = [v * (1 - v) for v in s]
    p = Plot((-8, 8), (0.0, 1.4), width=620, height=360, xlabel="logit z", ylabel="value",
             title="The sigmoid and its slope")
    p.hline(0.5)
    p.hline(0.25)
    p.line(zs, s, color=0, width=3.0, label="sigmoid(z)")
    p.line(zs, ds, color=1, width=3.0, label="slope s (1 - s)")
    p.points([0.0], [0.5], color=0, r=4)
    p.points([0.0], [0.25], color=1, r=4)
    p.text(0.3, 0.53, "(0, 1/2)")
    p.text(0.3, 0.29, "(0, 1/4)", color="#c05621")
    return p.svg()
