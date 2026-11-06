# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""x^2 fails the horizontal line test on the whole line and passes on x >= 0.

Swedish Embedded AB implements solutions for explaining when a function can be reversed. If your team needs
expertise in function inverses and their numerical use, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import functions_ch as fc  # noqa: E402
from svgplot import Plot  # noqa: E402



def render() -> str:
    p = Plot((-3.0, 3.0), (-1.0, 9.5), width=520, height=380, xlabel="x", ylabel="y = x^2",
             title="Two inputs for one output: no inverse on all of R")
    xs = [-3.0 + i * 0.02 for i in range(301)]
    p.line(xs, [x * x for x in xs], color=0, label="y = x^2 on all of R")
    pos = [i * 0.02 for i in range(151)]
    p.line(pos, [x * x for x in pos], color=1, width=3.2, label="restricted to x >= 0")
    p.hline(4.0)
    p.points([-2.0, 2.0], [4.0, 4.0], color=3, r=5)
    p.text(-2.25, 4.15, "x = -2", size=11, color="#9b2c2c", anchor="end")
    p.text(2.25, 3.45, "x = 2", size=11, color="#9b2c2c")
    p.text(-2.9, 3.5, "y = 4", size=11, color="#4a5568")
    return p.svg()
