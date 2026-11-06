# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Four members of the quadratic family y = a x^2 + b x + c, and the member fixed by three points.

Swedish Embedded AB implements solutions for explaining how model families are parametrised. If your team needs
expertise in model families and curve fitting, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import functions_ch as fc  # noqa: E402
from svgplot import Plot  # noqa: E402


FAMILY = [(1.0, 0.0, 0.0), (-1.0, 0.0, 2.0), (0.5, 1.0, -1.0)]


def render() -> str:
    p = Plot((-2.0, 2.8), (-3.0, 11.0), width=520, height=380, xlabel="x", ylabel="y",
             title="One family, different parameters: y = a x^2 + b x + c")
    xs = [-2.0 + i * 0.02 for i in range(241)]
    for k, (a, b, c) in enumerate(FAMILY):
        p.line(xs, [a * x * x + b * x + c for x in xs], color=k, width=1.8,
               label=f"a={fc._g(a)}, b={fc._g(b)}, c={fc._g(c)}")
    c0, c1, c2 = (float(v) for v in fc._interpolate(fc.QUADRATIC_POINTS))
    p.line(xs, [c2 * x * x + c1 * x + c0 for x in xs], color=3, width=3.0,
           label=f"a={fc._g(c2)}, b={fc._g(c1)}, c={fc._g(c0)} (fitted)")
    p.points([x for x, _ in fc.QUADRATIC_POINTS], [y for _, y in fc.QUADRATIC_POINTS], color=5, r=5)
    p.text(0.08, 1.35, "(0, 1)", size=11, color="#4a5568")
    p.text(1.08, -0.4, "(1, 0)", size=11, color="#4a5568")
    p.text(2.08, 3.35, "(2, 3)", size=11, color="#4a5568")
    return p.svg()
