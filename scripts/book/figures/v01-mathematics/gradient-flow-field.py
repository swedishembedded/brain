# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The vector field -grad L of the chapter-8 loss, with the gradient flow and a gradient-descent path.

Arrows have a fixed length, so they show direction only. The smooth curve is
the exact solution of dw/dt = -grad L(w); the dots are gradient descent with
step 0.15 from the same start, which is Euler's method with h = 0.15.

Swedish Embedded AB implements solutions for understanding and debugging
optimisation dynamics. If your team needs expertise in gradient flows and step
size selection, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import integrals as ig  # noqa: E402
from svgplot import Plot  # noqa: E402

START = [0.0, 3.5]
ETA = 0.15


def _contour(level, n=181):
    """Points of the ellipse L(w) = level: w* plus the curvature directions scaled by sqrt(2 level / lambda)."""
    (l1, v1), (l2, v2) = ig._eigen_sym2(ig._H)
    pts = []
    for k in range(n):
        a = 2.0 * math.pi * k / (n - 1)
        r1 = math.sqrt(2.0 * level / l1) * math.cos(a)
        r2 = math.sqrt(2.0 * level / l2) * math.sin(a)
        pts.append((ig._WSTAR[0] + r1 * v1[0] + r2 * v2[0], ig._WSTAR[1] + r1 * v1[1] + r2 * v2[1]))
    return pts


def _arrow(p, centre, d, length=0.3, head=0.07):
    """A thin arrow of fixed length centred on `centre`, pointing along the unit vector `d`."""
    tail = [centre[0] - d[0] * length / 2, centre[1] - d[1] * length / 2]
    tip = [centre[0] + d[0] * length / 2, centre[1] + d[1] * length / 2]
    p.line([tail[0], tip[0]], [tail[1], tip[1]], color="#a0aec0", width=1.3)
    for sign in (1.0, -1.0):
        c, s = math.cos(math.radians(155)), sign * math.sin(math.radians(155))
        back = [d[0] * c - d[1] * s, d[0] * s + d[1] * c]
        p.line([tip[0], tip[0] + head * back[0]], [tip[1], tip[1] + head * back[1]], color="#a0aec0", width=1.3)


def render() -> str:
    p = Plot((-1.0, 3.0), (0.0, 4.0), width=476, height=476, xlabel="w1", ylabel="w2",
             title="Gradient flow and gradient descent")
    for level in (0.25, 1.0, 3.0, 8.0, 16.0):
        pts = _contour(level)
        p.line([a for a, _ in pts], [b for _, b in pts], color=5, width=1.0)
    for i in range(-2, 7):
        for j in range(0, 8):
            w = [i * 0.5, j * 0.5]
            g = ig._grad(w)
            n = math.hypot(*g)
            if n < 1e-9:
                continue
            _arrow(p, w, [-g[0] / n, -g[1] / n])
    ts = [3.0 * k / 300 for k in range(301)]
    flow = [ig.flow_at(START, t) for t in ts]
    p.line([w[0] for w in flow], [w[1] for w in flow], color=0, label="gradient flow", width=2.4)
    path = ig.descend(START, ETA, 40)
    p.line([w[0] for w in path], [w[1] for w in path], color=1, label="descent, eta = 0.15", width=1.2)
    p.points([w[0] for w in path], [w[1] for w in path], color=1, r=2.5)
    p.points([ig._WSTAR[0]], [ig._WSTAR[1]], color=2, r=5)
    p.text(ig._WSTAR[0] + 0.08, ig._WSTAR[1] - 0.2, "minimum (1, 2)", size=11, plate=True)
    p.text(-0.95, 0.15, "grey arrows: direction of -grad L (fixed length)", size=11, color="#4a5568", plate=True)
    return p.svg()
