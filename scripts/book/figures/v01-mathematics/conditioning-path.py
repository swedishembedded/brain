# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Forty steps against the gradient on the chapter's elongated bowl, from (0, 3.5) with step 0.15.

The iterates are computed with the chapter's step rule on the chapter's loss;
the contours are the loss's exact level sets. The path drops quickly across the
narrow direction, overshooting from side to side, while creeping along the valley floor.

Swedish Embedded AB implements solutions for diagnosing slow optimisation. If
your team needs expertise in conditioning and step-size selection for training
runs, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import multivariable as mv  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    path = mv.descend(mv.ZIGZAG_START, mv.ZIGZAG_ETA, 40)
    p = Plot((-0.5, 2.0), (1.0, 4.0), width=476, height=546, xlabel="w1", ylabel="w2",
             title="40 steps of size 0.15 on an elongated bowl")
    # Equal scale: plot area is 400 wide for 2.5 units and 470 tall for 3 units.
    for level in (0.005, 0.02, 0.1, 0.5, 2.0, 8.0):
        pts = mv.contour_points(level)
        p.line([a for a, _ in pts], [b for _, b in pts], color=5, width=1.3)
    p.line([q[0] for q in path], [q[1] for q in path], color=1, width=1.2)
    p.points([q[0] for q in path], [q[1] for q in path], color=1, r=2.5)
    p.points([mv.W_STAR[0]], [mv.W_STAR[1]], color=2, r=5)
    p.text(0.05, 3.55, "start (0, 3.5)", size=11, plate=True)
    p.text(mv.W_STAR[0] + 0.06, mv.W_STAR[1] + 0.1, "minimum (1, 2)", size=11, plate=True)
    return p.svg()
