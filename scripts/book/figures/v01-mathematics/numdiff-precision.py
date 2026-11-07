# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Error of forward and central differences of e^x at x = 1 against the step h, in 32 and 64 bits.

The curves come from the worked module `floating` (the function and the
arithmetic are rounded to the format being drawn); the vertical marks are the
steps h* = 2 sqrt(u) and (3u)^(1/3) that the chapter derives for the
forward and the central difference.

Swedish Embedded AB implements solutions for trustworthy numerical checks of
gradients. If your team needs expertise in finite-difference gradient checking
in reduced precision, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import floating  # noqa: E402
from svgplot import Plot  # noqa: E402


def _lg(v):
    return math.log10(max(v, 1e-13))


def render() -> str:
    js = list(range(4, 61))                              # h = 10^(-j/4), 10^-1 down to 10^-15
    hs = [10.0 ** (-j / 4) for j in js]
    xs = [j / 4 for j in js]
    plot = Plot((1.0, 15.0), (-12.0, 3.2), width=600, height=400, xlabel="-log10 of the step h",
                ylabel="log10 of the error", title="Finite-difference error of e^x at x = 1")
    plot.line(xs, [_lg(e) for e in floating.numdiff_curve("32-bit", "forward", hs)], color=1, label="forward, 32-bit")
    plot.line(xs, [_lg(e) for e in floating.numdiff_curve("32-bit", "central", hs)], color=1, dash=True, label="central, 32-bit")
    plot.line(xs, [_lg(e) for e in floating.numdiff_curve("64-bit", "forward", hs)], color=0, label="forward, 64-bit")
    plot.line(xs, [_lg(e) for e in floating.numdiff_curve("64-bit", "central", hs)], color=0, dash=True, label="central, 64-bit")
    for u, tag in ((floating.U32, "32"), (floating.U64, "64")):
        for h, kind in ((2 * math.sqrt(u), "f"), ((3 * u) ** (1 / 3), "c")):
            plot.vline(-math.log10(h), color="#cbd5e0")
            # Top of the panel is empty; stagger so neighbouring labels cannot touch.
            plot.text(-math.log10(h) + 0.1, 2.4 if kind == "c" else 1.8, f"h* {kind}{tag}", size=11, color="#4a5568")
    return plot.svg()
