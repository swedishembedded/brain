# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Vector addition as a parallelogram: a + b, with a - b as the arrow from b to a.

The coordinates come from the chapter's worked module, so the picture and the
table of sums cannot disagree.

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

SCALE = 90  # pixels per unit, equal on both axes so angles look right


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
    a, b = [3.0, 1.0], [1.0, 2.0]
    s = vc.add(a, b)
    d = vc.sub(a, b)
    xlim, ylim = (-0.5, 5.5), (-0.5, 3.5)
    p = Plot(xlim, ylim, width=76 + SCALE * 6.0, height=60 + SCALE * 4.0, xlabel="first entry", ylabel="second entry")
    arrow(p, b, s, "#2b6cb0", 1.5, dash=True)   # b moved to the tip of a
    arrow(p, a, s, "#c05621", 1.5, dash=True)   # a moved to the tip of b
    arrow(p, (0, 0), a, "#2b6cb0")
    arrow(p, (0, 0), b, "#c05621")
    arrow(p, (0, 0), s, "#2f855a", 2.6)
    arrow(p, b, a, "#6b46c1", 1.8)              # a - b runs from the tip of b to the tip of a
    p.text(a[0] / 2, a[1] / 2 - 0.22, "a = (3, 1)", anchor="middle", color="#2b6cb0")
    p.text(b[0] / 2 - 0.1, b[1] / 2 + 0.12, "b = (1, 2)", anchor="end", color="#c05621")
    p.text(s[0] + 0.1, s[1] - 0.05, "a + b = (4, 3)", anchor="start", color="#2f855a")
    p.text(a[0] + 0.15, a[1] - 0.35, "a - b = (2, -1)", anchor="start", color="#6b46c1")
    assert d == [2.0, -1.0] and s == [4.0, 3.0]
    return p.svg()
