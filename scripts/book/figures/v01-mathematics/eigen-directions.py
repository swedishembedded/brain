# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The directions a matrix only stretches: eigenvectors of [[2, 1], [1, 2]].

Sixteen unit vectors (grey dots) are joined by a thin line to their images
under A (blue dots). Every image is pulled towards the diagonal, except on the
two eigenvector lines, where the image stays on the same line: stretched by 3
along (1, 1) and by 1 (unchanged) along (1, -1). The blue curve is the image of
the whole unit circle.

Swedish Embedded AB implements solutions for explaining linear structure in
learned weights. If your team needs expertise in spectral analysis of neural
networks, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import decompositions as dc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    A = dc.SYM2
    vals, vecs = dc._eigh(A)
    p = Plot((-3.6, 3.6), (-3.6, 3.6), width=420, height=420, title="A = [[2, 1], [1, 2]]: what it does to directions")
    for (lam, v), color in zip(zip(vals, vecs), ("#c05621", "#2f855a")):
        p.line([-3.4 * v[0], 3.4 * v[0]], [-3.4 * v[1], 3.4 * v[1]], color=color, width=1.6, dash=True)
    circle = [[math.cos(2 * math.pi * k / 120), math.sin(2 * math.pi * k / 120)] for k in range(121)]
    p.line([c[0] for c in circle], [c[1] for c in circle], color="#a0aec0", width=1.5)
    image = [dc._matvec(A, c) for c in circle]
    p.line([c[0] for c in image], [c[1] for c in image], color=0, width=2)
    for k in range(16):
        t = 2 * math.pi * k / 16 + math.radians(7.5)
        x = [math.cos(t), math.sin(t)]
        y = dc._matvec(A, x)
        p.line([x[0], y[0]], [x[1], y[1]], color="#cbd5e0", width=1)
        p.points([x[0]], [x[1]], color="#718096", r=3)
        p.points([y[0]], [y[1]], color=0, r=3)
    p.text(0.2, 3.1, f"lambda = {vals[0]:g}, v = (1, 1)", color="#c05621", size=12)
    p.text(0.55, -3.3, f"lambda = {vals[1]:g}, v = (1, -1)", color="#2f855a", size=12)
    p.text(-3.5, -3.45, "grey: x on the unit circle; blue: A x", size=11, color="#4a5568")
    return p.svg()
