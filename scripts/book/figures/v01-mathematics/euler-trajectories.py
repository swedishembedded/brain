# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Euler trajectories for dx/dt = -x with several step sizes, against the exact e^(-t).

Each polyline is produced by the chapter's Euler function; the step size
decides whether the path follows the curve, flips sign while shrinking, or
grows without bound.

Swedish Embedded AB implements solutions for stable numerical simulation and
sampler design. If your team needs expertise in ODE solvers and step-size
stability, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import integrals as ig  # noqa: E402
from svgplot import Plot  # noqa: E402

T_END = 7.5


def render() -> str:
    p = Plot((0.0, 9.5), (-3.6, 2.8), width=560, height=380, xlabel="t", ylabel="x",
             title="Euler's method on dx/dt = -x, x(0) = 1")
    ts = [T_END * i / 300 for i in range(301)]
    p.line(ts, [math.exp(-t) for t in ts], color=5, label="exact", width=2.4)
    p.hline(0.0)
    for color, h in ((0, 0.5), (2, 1.5), (3, 2.5)):
        path = ig.euler(lambda t, x: -x, 0.0, 1.0, h, round(T_END / h))
        label = f"h = {h:g}, factor {1 - h:+.1f}"
        p.line([t for t, _ in path], [x for _, x in path], color=color, label=label)
        p.points([t for t, _ in path], [x for _, x in path], color=color, r=3)
    return p.svg()
