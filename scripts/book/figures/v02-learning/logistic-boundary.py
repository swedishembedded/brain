# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The sixty inspected parts, the fitted decision boundary and lines of equal probability.

The weights are the ones gradient descent finds in the chapter (logistic_regression.training_run). Because the logit
w0 + w1 x1 + w2 x2 is linear, every set of equal probability is a straight line parallel to the boundary: the line
for probability p is the line where the logit equals log(p / (1 - p)).

Swedish Embedded AB implements solutions for deciding pass or fail from measurements. If your team needs expertise
in classification of inspection data, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logistic_regression as lr  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    w = lr._fit_parts()[0]
    rows = lr._parts()
    p = Plot((0.0, 8.0), (0.0, 5.4), width=620, height=420, xlabel="x1: diameter error (0.01 mm)",
             ylabel="x2: roughness (micrometres)", title="Decision boundary and probability contours")
    for prob, color, dash in ((0.1, "#a0aec0", True), (0.25, "#a0aec0", True), (0.75, "#a0aec0", True), (0.9, "#a0aec0", True)):
        z = lr.logit(prob)
        xs = [0.0, 8.0]
        p.line(xs, [(z - w[0] - w[1] * x) / w[2] for x in xs], color=color, dash=dash, width=1.5,
               label="p = 0.1, 0.25, 0.75, 0.9" if prob == 0.1 else "")
    xs = [0.0, 8.0]
    p.line(xs, [(-w[0] - w[1] * x) / w[2] for x in xs], color=4, width=3.0, label="boundary p = 0.5")
    p.points([r[0] for r in rows if r[2] == 0], [r[1] for r in rows if r[2] == 0], color=2, label="passed (y = 0)")
    p.points([r[0] for r in rows if r[2] == 1], [r[1] for r in rows if r[2] == 1], color=3, label="rejected (y = 1)")
    return p.svg()
