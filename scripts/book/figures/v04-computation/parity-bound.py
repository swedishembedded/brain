# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The error of a float32 dot product against its length n, in units of u S (u = 2^-24, S the sum of the absolute values of the
# products), for the left fold and the adjacent-pair tree, with the bound k u S / (1 - k u) of each (k = n for the fold,
# 1 + ceil(log2 n) for the tree) and the largest difference between the two programs, from the worked module's `parity_series`.
# The bounds are what a parity tolerance is derived from; the observed errors lie far below them, the left fold's bound far
# more than the tree's.
#
# Swedish Embedded AB implements solutions for numerical tests of compute backends that neither flag correct results nor miss
# defects. If your team needs expertise in cross-backend validation, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import backends as bk  # noqa: E402
from svgplot import Plot  # noqa: E402

LG = math.log10


def render() -> str:
    rows = bk.parity_series()
    xs = [LG(r["n"]) for r in rows]
    p = Plot((1.0, 3.8), (-1.6, 3.9), width=640, height=400, xlabel="log10 of the number of terms n",
             ylabel="log10 of the error in units of u S", title="Parity tolerance of a float32 dot product")
    p.line(xs, [LG(r["bL"]) for r in rows], color=5, dash=True, label="bound of the left fold, k = n")
    p.line(xs, [LG(r["bT"]) for r in rows], color=4, dash=True, label="bound of the tree, k = 1 + ceil(log2 n)")
    p.line(xs, [LG(r["L"]) for r in rows], color=1, label="left fold, largest error of 40")
    p.line(xs, [LG(r["T"]) for r in rows], color=0, label="tree, largest error of 40")
    p.line(xs, [LG(r["diff"]) for r in rows], color=2, label="left fold against tree, largest difference")
    return p.svg()
