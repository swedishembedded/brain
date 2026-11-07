# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Multiplying by a unit complex number turns the plane.

z = 2 + i is multiplied by w = 0.6 + 0.8i (length 1, angle 53.13 degrees). The
product is computed with the chapter's pair rule (ac - bd) + (ad + bc)i; the
dashed arc around the origin shows that z w is z turned by the angle of w
with its length unchanged.

Swedish Embedded AB implements solutions for rotation-based position encodings
and frequency-domain signal processing. If your team needs expertise in
complex-valued arithmetic for machine learning, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fourier as fo  # noqa: E402
from svgplot import Plot  # noqa: E402

SCALE = 100
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#c05621", "#2f855a", "#718096"


def arrow(p, tip, color, width=2.4):
    """A line from the origin with a filled head in the line's own colour."""
    x0, y0, x1, y1 = p.x(0), p.y(0), p.x(tip[0]), p.y(tip[1])
    length = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    bx, by = x1 - 11 * ux, y1 - 11 * uy
    p.items.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{width}"/>')
    p.items.append(f'<polygon fill="{color}" points="{x1:.1f},{y1:.1f} {bx - 4.5 * uy:.1f},{by + 4.5 * ux:.1f} '
                   f'{bx + 4.5 * uy:.1f},{by - 4.5 * ux:.1f}"/>')


def render() -> str:
    z, w = (2.0, 1.0), (0.6, 0.8)
    zw = fo._cmul_pairs(z, w)
    r = math.hypot(*z)
    assert abs(math.hypot(*zw) - r) < 1e-12
    a0, a1 = math.atan2(z[1], z[0]), math.atan2(zw[1], zw[0])
    xlim, ylim = (-1.3, 3.0), (-0.5, 2.8)
    p = Plot(xlim, ylim, width=76 + SCALE * 4.3, height=60 + SCALE * 3.3, xlabel="real part", ylabel="imaginary part")
    circle = [i * 2 * math.pi / 120 for i in range(121)]
    p.line([math.cos(t) for t in circle], [math.sin(t) for t in circle], color="#a0aec0", dash=True, width=1.2)
    arc = [a0 + (a1 - a0) * i / 40 for i in range(41)]
    p.line([r * math.cos(t) for t in arc], [r * math.sin(t) for t in arc], color=GREY, dash=True, width=1.6)
    arrow(p, w, ORANGE)
    arrow(p, z, BLUE)
    arrow(p, zw, GREEN, 3.2)
    p.text(z[0] + 0.08, z[1] - 0.2, "z = 2 + i", color=BLUE)
    p.text(w[0] + 0.08, w[1] + 0.02, "w = 0.6 + 0.8i", color=ORANGE)
    p.text(zw[0] + 0.1, zw[1], f"z w = {zw[0]:g} + {zw[1]:g}i", color=GREEN)
    mid = (a0 + a1) / 2
    p.text(r * 1.04 * math.cos(mid) + 0.05, r * 1.04 * math.sin(mid) + 0.05,
           f"turn by {math.degrees(a1 - a0):.1f} degrees", anchor="start", color=GREY)
    return p.svg()
