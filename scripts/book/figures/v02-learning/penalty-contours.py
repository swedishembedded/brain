# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Why an L1 penalty lands on a corner: a loss with elliptical contours meets a diamond and a circle.

The loss is 1/2 (w - m)^T A (w - m), minimal at m = (1.2, 1.9). Left: the best point on the L1 ball |w1| + |w2| = r is
the lasso solution of the worked module (likelihood.penalty_corners), and it sits at the corner on the w2 axis, where
w1 is exactly zero. Right: the circle with the same length ||w||_2 touches the loss contour through the ridge solution,
a point where both weights are nonzero. The contour through each solution is drawn; ellipses are computed from the
eigen-decomposition of A.

Swedish Embedded AB implements solutions for models that keep only the inputs that matter. If your team needs expertise
in sparse and regularized estimation, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import likelihood as lk  # noqa: E402
from svgplot import Plot  # noqa: E402

SIZE = 400


def _ellipse(level, c, a):
    """Points of the contour 1/2 (w - c)^T A (w - c) = level for the symmetric 2 x 2 matrix A."""
    tr, det = a[0][0] + a[1][1], a[0][0] * a[1][1] - a[0][1] ** 2
    disc = math.sqrt(tr * tr / 4 - det)
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    v1 = (a[0][1], l1 - a[0][0])
    n1 = math.hypot(*v1)
    v1 = (v1[0] / n1, v1[1] / n1)
    v2 = (-v1[1], v1[0])
    pts = []
    for i in range(121):
        t = 2 * math.pi * i / 120
        s1, s2 = math.sqrt(2 * level / l1) * math.cos(t), math.sqrt(2 * level / l2) * math.sin(t)
        pts.append((c[0] + s1 * v1[0] + s2 * v2[0], c[1] + s1 * v1[1] + s2 * v2[1]))
    return pts


def _panel(title, prob, w_star, ball):
    p = Plot((-1.0, 2.2), (-1.0, 2.2), width=SIZE, height=SIZE, xlabel="w1", ylabel="w2", title=title)
    p.hline(0.0, dash=False)
    p.vline(0.0, color="#a0aec0", dash=False)
    c, a = prob["c"], prob["a"]
    star = prob["loss"](w_star)
    for k, level in enumerate((0.3, 0.9, 1.8, star, 3.6)):
        pts = _ellipse(level, c, a)
        p.line([q[0] for q in pts], [q[1] for q in pts], color=1 if level == star else "#cbd5e0",
               width=2.4 if level == star else 1.2)
    p.points([c[0]], [c[1]], color=5, r=3.0)
    p.text(c[0] + 0.05, c[1] + 0.08, "unpenalised", size=11)
    p.line([q[0] for q in ball], [q[1] for q in ball], color=0, width=2.8)
    p.points([w_star[0]], [w_star[1]], color=3, r=5.0)
    return p.svg()


def render() -> str:
    prob = lk.contour_problem()
    r1 = prob["r1"]
    diamond = [(r1, 0), (0, r1), (-r1, 0), (0, -r1), (r1, 0)]
    r2 = math.hypot(*prob["w_l2"])
    circle = [(r2 * math.cos(2 * math.pi * i / 120), r2 * math.sin(2 * math.pi * i / 120)) for i in range(121)]
    left = _panel("L1: |w1| + |w2| = r", prob, prob["w_l1"], diamond)
    right = _panel("L2: w1^2 + w2^2 = r^2", prob, prob["w_l2"], circle)
    h = SIZE
    inner = []
    for k, doc in enumerate((left, right)):
        body = doc.split(">", 1)[1].rsplit("</svg>", 1)[0]
        inner.append(f'<svg x="{k * SIZE}" width="{SIZE}" height="{h}" viewBox="0 0 {SIZE} {h}">{body}</svg>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2 * SIZE} {h}" width="{2 * SIZE}" height="{h}">'
            + "".join(inner) + "</svg>\n")
