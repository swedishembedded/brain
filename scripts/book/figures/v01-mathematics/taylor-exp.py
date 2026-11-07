# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Taylor polynomials of orders 1, 2 and 3 of e^x about 0, drawn against e^x.

Swedish Embedded AB implements solutions for numerically trustworthy machine
learning code. If your team needs expertise in local approximation of loss
functions, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgplot import Plot  # noqa: E402


def _poly(x, n):
    return sum(x ** k / math.factorial(k) for k in range(n + 1))


def render() -> str:
    xs = [-2.0 + 4.0 * i / 200 for i in range(201)]
    p = Plot((-2.0, 2.0), (-2.0, 8.0), width=560, height=380, xlabel="x", ylabel="value",
             title="Taylor polynomials of e^x about x = 0")
    p.line(xs, [math.exp(x) for x in xs], color=5, width=3.0, label="e^x")
    for n, c in ((1, 0), (2, 1), (3, 2)):
        p.line(xs, [_poly(x, n) for x in xs], color=c, label=f"order {n}")
    return p.svg()
