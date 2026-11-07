# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A gradient (4, 1) in two dimensions, clipped to a threshold of 1 per element and by its global norm.

The per-element clip gives (1, 1), which points 45 degrees from the axis; the global clip gives (0.9701, 0.2425), which
points where the gradient did (14.04 degrees). Values from `schedules.clip_elementwise` and `schedules.clip_global`.

Swedish Embedded AB implements solutions for stable training of neural networks. If your team needs expertise in
gradient clipping and optimiser diagnosis, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import schedules as sch  # noqa: E402
from svgplot import Plot  # noqa: E402

G = (4.0, 1.0)
UNITS = (6.5, 3.5)  # x and y extent at the same pixels per unit


def render() -> str:
    p = Plot((-1.5, 5.0), (-1.5, 2.0), width=700, height=420, xlabel="g1", ylabel="g2",
             title="Per-element and global clipping of g = (4, 1) at threshold 1", legend_loc="right")
    p.w, p.h = round(100 * UNITS[0] + p.ml + p.mr), round(100 * UNITS[1] + p.mt + p.mb)
    circle = [(math.cos(a * math.pi / 90), math.sin(a * math.pi / 90)) for a in range(181)]
    p.line([c[0] for c in circle], [c[1] for c in circle], color=2, width=1.5, dash=True, label="norm 1 (global bound)")
    p.line([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=1, width=1.5, dash=True, label="each |g_i| <= 1 (per-element bound)")
    e, k = sch.clip_elementwise(list(G), 1.0), sch.clip_global(list(G), 1.0)
    p.line([0, G[0]], [0, G[1]], color=5, width=2.0, label="g")
    p.arrow(0, 0, G[0], G[1], color="#4a5568")
    p.line([0, e[0]], [0, e[1]], color=1, width=2.4, label="per-element clip")
    p.arrow(0, 0, e[0], e[1], color="#c05621")
    p.line([0, k[0]], [0, k[1]], color=2, width=2.4, label="global clip")
    p.arrow(0, 0, k[0], k[1], color="#2f855a")
    p.points([0], [0], color=5, r=3)
    p.text(G[0], G[1] + 0.2, "g = (4, 1)", anchor="end")
    p.text(e[0] - 0.1, e[1] + 0.22, "(1, 1)", anchor="end")
    p.text(k[0] + 0.1, k[1] - 0.3, "(0.970, 0.243)", anchor="start")
    return p.svg()
