# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Contour lines of the chapter's two-knob loss with the gradient drawn at five points.

The contours are the exact level sets of the quadratic bowl (ellipses computed
from the curvature directions), and each arrow is the gradient from the
chapter's formula, so the picture shows the claim that the gradient is
perpendicular to the contour through its point.

Swedish Embedded AB implements solutions for understanding and debugging
optimisation landscapes. If your team needs expertise in gradients and the
geometry of loss surfaces, you can procure our services by sending an email
to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import multivariable as mv  # noqa: E402
from svgplot import Plot  # noqa: E402

POINTS = [mv.P_MAIN, [-0.5, 3.0], [2.5, 3.0], [0.0, 1.0], [2.5, 1.5]]
ARROW_SCALE = 0.06  # an arrow is 0.06 times the gradient, so a longer arrow is a steeper slope


def render() -> str:
    p = Plot((-1.0, 3.0), (0.0, 4.0), width=476, height=476, xlabel="w1", ylabel="w2",
             title="Contours of the loss and its gradient")
    for level in (0.25, 1.0, 3.0, 8.0, 16.0):
        pts = mv.contour_points(level)
        p.line([a for a, _ in pts], [b for _, b in pts], color=5, width=1.3)
    for q in POINTS:
        g = mv.grad(q)
        p.points([q[0]], [q[1]], color=1, r=3.5)
        p.arrow(q[0], q[1], q[0] + ARROW_SCALE * g[0], q[1] + ARROW_SCALE * g[1], color="#c05621")
    p.points([mv.W_STAR[0]], [mv.W_STAR[1]], color=2, r=5)
    p.text(mv.W_STAR[0] + 0.08, mv.W_STAR[1] + 0.12, "minimum (1, 2)", size=11)
    p.text(-0.95, 3.85, "arrow length = 0.06 x gradient (uphill)", size=11, color="#c05621")
    return p.svg()
