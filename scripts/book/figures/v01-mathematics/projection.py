# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Projection of v onto u and the residual, which meets u at a right angle.

The numbers are those of the chapter's worked projection, computed with the
formula t = v.u / u.u rather than drawn by hand.

Swedish Embedded AB implements solutions for explaining vector geometry behind
embeddings and similarity search. If your team needs expertise in the linear
algebra of machine learning, you can procure our services by sending an email
to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vectors as vc  # noqa: E402
from svgplot import Plot  # noqa: E402

SCALE = 90


def arrow(p, tail, head, color, width=2.2, dash=False):
    """A line with a filled triangular head in the line's own colour (pixel geometry)."""
    x0, y0, x1, y1 = p.x(tail[0]), p.y(tail[1]), p.x(head[0]), p.y(head[1])
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    bx, by = x1 - 11 * ux, y1 - 11 * uy
    d = ' stroke-dasharray="5 4"' if dash else ""
    p.items.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" '
                   f'stroke-width="{width}"{d}/>')
    p.items.append(f'<polygon fill="{color}" points="{x1:.1f},{y1:.1f} {bx - 4.5 * uy:.1f},{by + 4.5 * ux:.1f} '
                   f'{bx + 4.5 * uy:.1f},{by - 4.5 * ux:.1f}"/>')


def render() -> str:
    v, u = [3.0, 4.0], [2.0, 1.0]
    t, proj, res = vc.project(v, u)
    assert abs(vc.dot(res, u)) < 1e-12
    xlim, ylim = (-0.5, 5.0), (-0.5, 4.5)
    p = Plot(xlim, ylim, width=76 + SCALE * 5.5, height=60 + SCALE * 5.0, xlabel="first entry", ylabel="second entry")
    # the line through u, extended so the projection visibly lies on it
    p.line([-0.5, 5.0], [-0.25, 2.5], color="#a0aec0", dash=True, width=1.2)
    # right-angle mark at the foot of the residual, along u and along the residual
    ul = [c / vc.norm(u) for c in u]
    rl = [c / vc.norm(res) for c in res]
    k = 0.28
    corner = proj
    q1 = [corner[0] - k * ul[0], corner[1] - k * ul[1]]
    q3 = [corner[0] + k * rl[0], corner[1] + k * rl[1]]
    q2 = [q1[0] + k * rl[0], q1[1] + k * rl[1]]
    p.line([q1[0], q2[0], q3[0]], [q1[1], q2[1], q3[1]], color="#4a5568", width=1.2)
    arrow(p, (0, 0), v, "#2b6cb0")
    arrow(p, (0, 0), proj, "#2f855a", 3.4)
    arrow(p, (0, 0), u, "#c05621")
    arrow(p, proj, v, "#9b2c2c", 2.2)
    p.text(v[0] - 0.1, v[1] + 0.12, "v = (3, 4)", anchor="end", color="#2b6cb0")
    p.text(u[0] + 0.1, u[1] - 0.28, "u = (2, 1)", color="#c05621")
    p.text(4.95, proj[1] - 0.6, f"t u = ({proj[0]:g}, {proj[1]:g}), t = {t:g}", anchor="end", color="#2f855a")
    p.text((proj[0] + v[0]) / 2 + 0.25, (proj[1] + v[1]) / 2, f"r = ({res[0]:g}, {res[1]:g})", anchor="start", color="#9b2c2c")
    return p.svg()
