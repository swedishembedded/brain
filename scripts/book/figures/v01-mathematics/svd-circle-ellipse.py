# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The SVD as rotate, stretch, rotate: the unit circle becomes an ellipse in three steps.

The matrix is A = [[3, 0], [4, 5]]. Panel 1 is the unit circle with the right
singular vectors v1, v2; panel 2 is V^T x (a rotation, so the circle is
unchanged and v1, v2 land on the axes); panel 3 is Sigma V^T x (the circle
stretched by sigma_1 and sigma_2 along the axes); panel 4 is U Sigma V^T x = A x
(a second rotation that places the ellipse, with its axes sigma_i u_i drawn).
The black dot follows the point x = (1, 0) through all four stages. Every
point is computed with the chapter's SVD arithmetic.

Swedish Embedded AB implements solutions for understanding what weight
matrices do to data. If your team needs expertise in matrix decompositions
for model analysis and compression, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import decompositions as dc  # noqa: E402
from svgplot import Plot  # noqa: E402

SIZE = 330
ORANGE, GREEN, BLUE = "#c05621", "#2f855a", "#2b6cb0"


def _spoke(p, tip, color, label, dx=0.06, dy=0.06):
    p.line([0, tip[0]], [0, tip[1]], color=color, width=3)
    p.points([tip[0]], [tip[1]], color=color, r=4)
    p.text(tip[0] + dx, tip[1] + dy, label, color=color, size=12)


def _panel(title, lim, curve, spokes, tracked):
    p = Plot((-lim, lim), (-lim, lim), width=SIZE, height=SIZE, title=title)
    p.line([c[0] for c in curve], [c[1] for c in curve], color=BLUE, width=2)
    for tip, color, label, dx, dy in spokes:
        _spoke(p, tip, color, label, dx, dy)
    p.points([tracked[0]], [tracked[1]], color="#1a202c", r=4.5)
    return p.svg()


def render() -> str:
    sig, vecs, U, stages = dc._svd_circle_stages(n=72)
    s1, s2 = sig
    x_of = lambda k: [st[k] for st in stages]  # noqa: E731  the k-th stage of every circle point
    tr = stages[0]  # the point x = (1, 0)
    v1, v2 = vecs
    u1, u2 = U
    panels = [
        _panel("1. unit circle, right singular vectors", 1.6, x_of(0),
               [(v1, ORANGE, "v1", 0.06, 0.06), (v2, GREEN, "v2", 0.06, -0.16)], tr[0]),
        _panel("2. V^T: rotate (v1, v2 to the axes)", 1.6, x_of(1),
               [([1, 0], ORANGE, "e1", 0.06, 0.06), ([0, 1], GREEN, "e2", 0.06, 0.06)], tr[1]),
        _panel(f"3. Sigma: stretch by {s1:.2f} and {s2:.2f}", 7.5, x_of(2),
               [([s1, 0], ORANGE, "sigma1 e1", -4.4, 0.5), ([0, s2], GREEN, "sigma2 e2", 0.4, 0.1)], tr[2]),
        _panel("4. U: rotate again; this is A x", 7.5, x_of(3),
               [([s1 * u1[0], s1 * u1[1]], ORANGE, "sigma1 u1", -4.2, 0.1),
                ([s2 * u2[0], s2 * u2[1]], GREEN, "sigma2 u2", 0.3, -0.5)], tr[3]),
    ]
    inner = lambda s: s.split("\n", 1)[1].rsplit("</svg>", 1)[0]  # noqa: E731  body without the outer <svg>
    gap = 10
    w = 2 * SIZE + gap
    h = 2 * SIZE + gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
    for k, body in enumerate(panels):
        x, y = (k % 2) * (SIZE + gap), (k // 2) * (SIZE + gap)
        out.append(f'<svg x="{x}" y="{y}" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">\n{inner(body)}</svg>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
