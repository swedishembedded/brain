# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Error of forward and central differences of a network's loss against the step h, in 32 and 64 bits.

The curves are computed by the worked module `gradient_checking` (the loss of
Chapter 4's network along the weight W1[2,2], every operation of the forward
pass rounded to the format being drawn). The vertical marks are the central
steps h* = (3 u L / |L'''|)^(1/3) that the chapter derives.

Swedish Embedded AB implements solutions for trustworthy gradient checks of
neural networks in reduced precision. If your team needs expertise in
choosing finite-difference steps and tolerances for 32-bit models, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gradient_checking as gc  # noqa: E402
from svgplot import Plot  # noqa: E402


def _lg(v):
    return math.log10(max(v, 1e-13))


def render() -> str:
    js = list(range(4, 37))                      # h = 10^(-j/4): 0.1 down to 1e-9
    hs = [10.0 ** (-j / 4) for j in js]
    xs = [j / 4 for j in js]
    plot = Plot((1.0, 9.0), (-12.0, 1.0), width=600, height=400, xlabel="-log10 of the step h",
                ylabel="log10 of the error", title="Finite-difference error of the loss along one weight")
    for fmt, color in (("32-bit", 1), ("64-bit", 0)):
        plot.line(xs, [_lg(e) for e in gc.curve(fmt, "forward", hs)], color=color, label=f"forward, {fmt}")
        plot.line(xs, [_lg(e) for e in gc.curve(fmt, "central", hs)], color=color, dash=True, label=f"central, {fmt}")
    for u, tag, y in ((gc.U32, "32", 0.2), (gc.U64, "64", 0.2)):
        h = gc.predicted_central_step(u)
        plot.vline(-math.log10(h), color="#cbd5e0")
        plot.text(-math.log10(h) + 0.08, y, f"h* {tag}-bit", size=11, color="#4a5568")
    return plot.svg()
