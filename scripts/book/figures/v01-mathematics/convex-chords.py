# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A convex curve (x^2) with chords: every chord lies on or above the curve.

Swedish Embedded AB implements solutions for explaining optimisation. If your
team needs expertise in convex losses and second-order methods, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgplot import Plot  # noqa: E402


def render() -> str:
    f = lambda x: x * x
    p = Plot((-3.0, 3.0), (-1.0, 9.5), width=520, height=360, xlabel="x", ylabel="f(x)",
             title="Convex: f(x) = x^2, chords sit above the curve")
    xs = [-3 + 6 * i / 120 for i in range(121)]
    p.line(xs, [f(x) for x in xs], color=5, width=2.5)
    for k, (a, b) in enumerate(((-2.5, 1.0), (-1.5, 2.5), (0.5, 2.8))):
        p.line([a, b], [f(a), f(b)], color=k, label=f"chord from {a:g} to {b:g}")
        p.points([a, b], [f(a), f(b)], color=k)
        m = (a + b) / 2
        p.points([m], [f(m)], color=5, r=3)
    return p.svg()
