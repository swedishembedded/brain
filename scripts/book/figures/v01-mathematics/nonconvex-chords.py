# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A non-convex curve (x^4 - 3x^2 + x) with a chord that dips below it.

Swedish Embedded AB implements solutions for explaining optimisation. If your
team needs expertise in non-convex training problems and their local minima,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import derivatives as dv  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    f = dv.cubic_wiggle
    p = Plot((-2.0, 2.1), (-4.5, 9.0), width=520, height=360, xlabel="x", ylabel="f(x)",
             title="Not convex: f(x) = x^4 - 3x^2 + x has two valleys")
    xs = [-2 + 4.1 * i / 160 for i in range(161)]
    p.line(xs, [f(x) for x in xs], color=5, width=2.5)
    # the two valley floors, found by stepping against the slope
    floors = []
    for start in (-1.0, 1.0):
        x = start
        for _ in range(2000):
            x -= 0.01 * dv.cubic_wiggle_slope(x)
        floors.append(x)
    a, b = floors
    p.line([a, b], [f(a), f(b)], color=3, label="chord between the valleys")
    p.points([a, b], [f(a), f(b)], color=3)
    m = (a + b) / 2
    chord_m = (f(a) + f(b)) / 2
    p.points([m], [f(m)], color=5, r=4)
    p.points([m], [chord_m], color=3, r=4)
    c, d = 1.2, 1.9
    p.line([c, d], [f(c), f(d)], color=0, label="a chord on the right branch")
    p.points([c, d], [f(c), f(d)], color=0)
    return p.svg()
