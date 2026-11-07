# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Relative error of a 32-bit running sum of copies of 0.1: left to right, pairwise and compensated.

Every curve is computed by emulating 32-bit rounding (worked module
`floating`); the worst-case line is the bound (n - 1) u of the chapter.

Swedish Embedded AB implements solutions for accurate large reductions in
machine learning code. If your team needs expertise in summation order,
compensated accumulation and mixed-precision reductions, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import floating  # noqa: E402
from svgplot import Plot  # noqa: E402


FLOOR = -8.8   # an error of exactly zero (or below 10^-8.8) is drawn here


def _lg(v):
    return math.log10(max(float(v), 10 ** FLOOR))


def render() -> str:
    ns = sorted({round(10 ** (k / 4)) for k in range(4, 25)})
    rows = floating.sum_error_curves(ns)
    xs = [math.log10(r[0]) for r in rows]
    plot = Plot((1.0, 6.0), (-9.0, 1.6), width=600, height=380, xlabel="log10 of the number of terms n",
                ylabel="log10 of the relative error", title="32-bit sum of n copies of 0.1 (zero error drawn at -8.8)")
    plot.line(xs, [_lg((r[0] - 1) * floating.U32) for r in rows], color=5, dash=True, width=1.4,
              label="worst-case bound (n-1)u")
    plot.line(xs, [_lg(r[1]) for r in rows], color=1, label="left to right")
    plot.line(xs, [_lg(r[2]) for r in rows], color=0, label="pairwise")
    plot.line(xs, [_lg(r[3]) for r in rows], color=2, dash=True, width=2.2, label="Kahan (same as pairwise)")
    plot.hline(_lg(floating.U32), color="#a0aec0")
    plot.text(1.1, _lg(floating.U32) + 0.15, "u = 2^-24", size=11, color="#4a5568")
    return plot.svg()
