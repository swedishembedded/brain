# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The numbers of a toy floating-point format and the gap to the next one.

The format has a 3-bit significand (1.xx) and exponents -2 to 1. Each point is
one representable number, plotted at the gap between it and its successor: the
gap doubles in every binade, and below the smallest normal number the
subnormals keep the gap of the lowest binade instead of shrinking further.

Swedish Embedded AB implements solutions for numerically trustworthy machine
learning code. If your team needs expertise in number formats and rounding
behaviour, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import floating  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p, emin = floating.TOY["p"], floating.TOY["emin"]
    normals, subs = floating.toy_numbers(**floating.TOY)
    tiny_gap = Fraction(2) ** (emin - (p - 1))
    pts = [(Fraction(0), tiny_gap)] + [(v, tiny_gap) for v in subs]
    pts_n = [(v, Fraction(2) ** (e - (p - 1))) for e, v in normals]
    plot = Plot((-0.1, 3.9), (-0.08, 0.62), width=600, height=360, xlabel="the number x",
                ylabel="gap to the next representable number", title="A format with a 3-bit significand")
    plot.points([float(a) for a, _ in pts_n], [float(g) for _, g in pts_n], color=0, label="normal numbers")
    plot.points([float(a) for a, _ in pts], [float(g) for _, g in pts], color=1, label="zero and subnormals")
    for a, _ in pts + pts_n:                      # the rug: where the numbers sit on the real line
        plot.line([float(a)] * 2, [-0.06, -0.02], color="#4a5568", width=1.2)
    for e in range(emin, floating.TOY["emax"] + 1):
        x = float(Fraction(2) ** e)
        plot.vline(x)
        plot.text(x + 0.03, 0.57 if e >= 0 else 0.52, f"2^{e}" if e != 0 else "1", size=11, color="#4a5568")
    plot.text(1.9, 0.12, "relative gap 2^(1-p) down to 2^-p", size=11, color="#4a5568")
    return plot.svg()
