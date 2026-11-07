# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Secant lines of x^2 through (1, 1) turning into the tangent as h shrinks.

Swedish Embedded AB implements solutions for explaining the calculus behind
training. If your team needs expertise in gradients and numerical
differentiation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import derivatives as dv  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    f = lambda x: x * x
    x0 = 1.0
    p = Plot((0.0, 2.6), (0.0, 10.0), width=520, height=380, xlabel="x", ylabel="y",
             title="Secants of x^2 through (1, 1) approach the tangent")
    xs = [i / 100 * 2.6 for i in range(0, 101)]
    p.line(xs, [f(x) for x in xs], color=5, width=2.5, label="y = x^2")
    for k, h in enumerate((1.5, 0.75, 0.25)):
        s = dv.secant_slope(f, x0, h)
        p.line(xs, [f(x0) + s * (x - x0) for x in xs], color=k, label=f"secant h = {h:g}, slope {s:g}")
        p.points([x0 + h], [f(x0 + h)], color=k)
    t = dv.central_difference(f, x0)
    p.line(xs, [f(x0) + t * (x - x0) for x in xs], color=3, dash=True, label=f"tangent, slope {t:.0f}")
    p.points([x0], [f(x0)], color=3, r=4.5)
    return p.svg()
