# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Left-reading rectangles under the speed curve v(t) = t (4 - t) on [0, 3].

The rectangles and the sum printed on the figure come from the chapter's own
left-sum function, so the picture and the table beside it cannot disagree.

Swedish Embedded AB implements solutions for explaining numerical methods to
engineering teams. If your team needs expertise in numerical integration, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import integrals as ig  # noqa: E402
from svgplot import Plot  # noqa: E402

N = 6


def render() -> str:
    h = 3.0 / N
    p = Plot((0.0, 3.0), (0.0, 4.5), width=520, height=340, xlabel="time t (s)", ylabel="speed (m/s)",
             title="Six speed readings as rectangles")
    xs = [3.0 * i / 300 for i in range(301)]
    p.line(xs, [ig.speed(t) for t in xs], color=0, width=2.4)
    stair_x, stair_y = [0.0], [0.0]
    for i in range(N):
        t = i * h
        v = ig.speed(t)
        stair_x += [t, t + h, t + h]
        stair_y += [v, v, 0.0]
        p.line([t, t], [0.0, v], color=1, width=1.2)
    p.line(stair_x[1:], stair_y[1:], color=1, width=1.6)
    left = ig.rule_sum(ig.speed, 0.0, 3.0, N, "left")
    p.text(0.06, 4.3, f"orange rectangles (left readings): {left:.3f} m", size=12, color="#c05621")
    p.text(0.06, 4.0, f"blue curve, exact area: {ig.SPEED_AREA_0_3:.3f} m", size=12, color="#2b6cb0")
    return p.svg()
