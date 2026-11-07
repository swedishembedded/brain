# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A least-squares line fitted to yes-or-no labels, with and without one extreme part, beside the logistic curve.

The twelve parts and the extreme part (x = 15, rejected) are those of logistic_regression.linear_fit_on_labels; the
logistic curve is the gradient-descent fit of logistic_regression.logistic_fit_one_feature on the twelve parts.

Swedish Embedded AB implements solutions for classifying parts from a single gauge reading. If your team needs
expertise in classification, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import logistic_regression as lr  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = lr._one_feature()
    far = rows + (lr.FAR_PART,)
    a0, a1 = lr._ols(rows)
    b0, b1 = lr._ols(far)
    w = lr._fit_one_feature(rows)
    xs = [0.0, 16.0]
    curve_x = [0.1 * k for k in range(161)]
    p = Plot((0.0, 16.0), (-0.5, 1.9), width=620, height=380, xlabel="x: diameter error (0.01 mm)",
             ylabel="label y, or fitted value", title="A line fitted to yes-or-no labels")
    p.hline(0.0)
    p.hline(1.0)
    p.line(xs, [a0 + a1 * x for x in xs], color=0, width=2.5, label="least squares, 12 parts")
    p.line(xs, [b0 + b1 * x for x in xs], color=1, width=2.5, label="least squares, 13 parts")
    p.line(curve_x, [lr.sigmoid(w[0] + w[1] * x) for x in curve_x], color=2, width=2.5, label="logistic, 12 parts")
    p.points([r[0] for r in far], [r[1] for r in far], color=5, r=4, label="parts (y = 0 or 1)")
    return p.svg()
