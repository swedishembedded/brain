# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The relative error of a float32 sum against the number of terms, in units of the unit roundoff u, for the left fold, the
# adjacent-pair tree and a 256-lane workgroup arrangement on uniform numbers (mean of six seeded vectors), and for the left
# fold on a constant number (0.1) where the rounding errors do not cancel. Dashed: the worst-case bounds (n - 1) u and
# ceil(log2 n) u of Volume I, Chapter 15. The data come from the worked module's `error_series`; the tree on the constant
# data is exactly 0 and is not drawn on a logarithmic axis.
#
# Swedish Embedded AB implements solutions for numerically sound parallel summation. If your team needs expertise in
# floating-point reductions for machine learning, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import reductions as rd  # noqa: E402
from svgplot import Plot  # noqa: E402

LG = math.log10


def render() -> str:
    rows = rd.error_series(3, 18)
    xs = [LG(r[0]) for r in rows]
    p = Plot((0.7, 5.6), (-1.0, 5.7), width=640, height=400, xlabel="log10 of the number of terms n",
             ylabel="log10 of the error in units of u", title="Error of a float32 sum against its length")
    p.line(xs, [LG(r[0] - 1) for r in rows], color=5, dash=True, label="bound of the left fold, (n - 1) u")
    p.line(xs, [LG(math.ceil(math.log2(r[0]))) for r in rows], color=4, dash=True, label="bound of the tree, ceil(log2 n) u")
    p.line(xs, [LG(r[4]) for r in rows], color=3, label="left fold, constant 0.1")
    p.line(xs, [LG(r[1]) for r in rows], color=1, label="left fold, uniform numbers")
    p.line(xs, [LG(r[2]) for r in rows], color=0, label="tree, uniform numbers")
    p.line(xs, [LG(r[3]) for r in rows], color=2, label="256 lanes then tree, uniform numbers")
    return p.svg()
