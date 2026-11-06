# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Two 2-D linear maps applied to the unit square and a grid drawn on it.

Left: a map whose matrix has rank 2; the grid becomes a tilted grid of
parallelograms and the square keeps a positive area. Right: a rank-1 map whose
second column is twice its first; the whole grid lands on one line. The image
of every grid point is computed with the chapter's matrix-vector product, and
the columns of the matrix are drawn as the images of the unit vectors.

Swedish Embedded AB implements solutions for explaining how layers transform
data. If your team needs expertise in the geometry of linear maps in machine
learning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matrices as mx  # noqa: E402
from svgplot import Plot  # noqa: E402

UNIT = 62  # pixels per unit on both axes, so a unit square stays square
XLIM, YLIM = (-0.5, 3.5), (-0.5, 2.0)
GRID = [i / 4 for i in range(5)]


def _arrow(p, tail, head, color, width=2.4):
    x0, y0, x1, y1 = p.x(tail[0]), p.y(tail[1]), p.x(head[0]), p.y(head[1])
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    bx, by = x1 - 10 * ux, y1 - 10 * uy
    p.items.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{width}"/>')
    p.items.append(f'<polygon fill="{color}" points="{x1:.1f},{y1:.1f} {bx - 4 * uy:.1f},{by + 4 * ux:.1f} {bx + 4 * uy:.1f},{by - 4 * ux:.1f}"/>')


def _panel(A, title, note):
    f = [[float(c) for c in row] for row in A]
    w = round(76 + UNIT * (XLIM[1] - XLIM[0]))
    h = round(74 + UNIT * (YLIM[1] - YLIM[0]))
    p = Plot(XLIM, YLIM, width=w, height=h, title=title)
    # the unit square and its grid, before the map
    for t in GRID:
        p.line([t, t], [0, 1], color="#cbd5e0", width=1)
        p.line([0, 1], [t, t], color="#cbd5e0", width=1)
    # the image of every grid line: lines of constant x[0] and of constant x[1], mapped point by point
    corners = [mx._matvec(f, c) for c in ([0, 0], [1, 0], [1, 1], [0, 1])]
    poly = " ".join(f"{p.x(a):.1f},{p.y(b):.1f}" for a, b in corners)
    p.items.append(f'<polygon points="{poly}" fill="#bee3f8" fill-opacity="0.55" stroke="none"/>')
    for t in GRID:
        a0, a1 = mx._matvec(f, [t, 0]), mx._matvec(f, [t, 1])
        b0, b1 = mx._matvec(f, [0, t]), mx._matvec(f, [1, t])
        p.line([a0[0], a1[0]], [a0[1], a1[1]], color="#2b6cb0", width=1.2)
        p.line([b0[0], b1[0]], [b0[1], b1[1]], color="#2b6cb0", width=1.2)
    _arrow(p, (0, 0), (1, 0), "#718096", 2)
    _arrow(p, (0, 0), (0, 1), "#718096", 2)
    c1, c2 = mx._matvec(f, [1, 0]), mx._matvec(f, [0, 1])
    _arrow(p, (0, 0), c1, "#c05621")
    _arrow(p, (0, 0), c2, "#2f855a")
    p.text(1.02, -0.1, "e1", color="#4a5568")
    p.text(-0.12, 1.02, "e2", anchor="end", color="#4a5568")
    p.text(c1[0] + 0.05, c1[1] - 0.17, f"A e1 = ({c1[0]:g}, {c1[1]:g})", color="#c05621", size=11)
    p.text(c2[0] + 0.05, c2[1] + 0.1, f"A e2 = ({c2[0]:g}, {c2[1]:g})", color="#2f855a", size=11)
    p.text(3.45, -0.35, note, anchor="end", size=11, color="#1a202c")
    return p.svg(), w, h


def render() -> str:
    left, w, h = _panel(mx.SQUARE_A, "rank 2: the square becomes a parallelogram", "area 5/2")
    right, _, _ = _panel(mx.SQUASH_A, "rank 1: the square collapses to a line", "area 0")
    inner = lambda s: s.split("\n", 1)[1].rsplit("</svg>", 1)[0]  # noqa: E731  body without the outer <svg>
    gap = 16
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2 * w + gap} {h}" width="{2 * w + gap}" height="{h}">\n'
            f'<rect width="{2 * w + gap}" height="{h}" fill="#ffffff"/>\n'
            f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n{inner(left)}</svg>\n'
            f'<svg x="{w + gap}" y="0" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n{inner(right)}</svg>\n</svg>\n')
