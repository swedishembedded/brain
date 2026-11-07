# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Contours of the sensor line's mean squared error with gradient descent at three learning rates.

The contours are exact level sets of the quadratic bowl (ellipses built from the eigen-decomposition of the
Hessian) and every path is produced by the chapter's update rule from (w, b) = (0, 0).

Swedish Embedded AB implements solutions for tuning optimisation procedures so that training converges. If your
team needs expertise in learning rates and conditioning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import linear_regression as lr  # noqa: E402
from svgplot import Plot  # noqa: E402

STEPS = {0.01: 200, 0.20: 60, 0.23: 7}
COLORS = {0.01: 1, 0.20: 2, 0.23: 3}


def _contour(q, level):
    """Points (w, b) where the loss exceeds its minimum by `level`; parameters are ordered (b, w)."""
    (l1, l2), vecs = q.lams, q.vecs
    pts = []
    for i in range(121):
        th = 2 * math.pi * i / 120
        a = math.sqrt(2 * level / l1) * math.cos(th)  # the loss is gap = (1/2) e^T H e
        c = math.sqrt(2 * level / l2) * math.sin(th)
        e = [vecs[k][0] * a + vecs[k][1] * c for k in range(2)]
        pts.append((q.w_star[1] + e[1], q.w_star[0] + e[0]))
    return pts


def render() -> str:
    x, ys, q = lr._raw()
    p = Plot((-15.0, 60.0), (-15.0, 50.0), width=620, height=420, xlabel="slope w", ylabel="intercept b",
             title="Gradient descent on the sensor line, three learning rates")
    for level in (1.0, 4.0, 16.0, 64.0, 256.0):
        pts = _contour(q, level)
        p.line([a for a, _ in pts], [b for _, b in pts], color=5, width=1.0)
    for eta, steps in STEPS.items():
        path = lr._descend(x, ys, lr.START, eta, steps)
        label = f"eta = {eta} ({steps} steps)"
        p.line([w[1] for w in path], [w[0] for w in path], color=COLORS[eta], width=1.6, label=label)
        p.points([w[1] for w in path], [w[0] for w in path], color=COLORS[eta], r=1.8)
    p.points([lr.START[1]], [lr.START[0]], color=4, r=5, label="start (w, b) = (0, 30)")
    p.points([q.w_star[1]], [q.w_star[0]], color=0, r=5, label="minimum")
    return p.svg()
