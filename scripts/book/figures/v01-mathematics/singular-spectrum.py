# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Singular value spectrum of an 8 x 6 table, and the error of its best rank-r approximations.

Left: the six singular values of the chapter's test table (three smooth
patterns plus noise), a clear cliff after the third. Right: the relative
Frobenius error |M - M_r|_F / |M|_F of the truncated SVD for r = 0..5 on a
log scale (solid), together with sqrt(sum of the dropped sigma_i^2) / |M|_F
(circles), which Eckart and Young's theorem says is the same number.

Swedish Embedded AB implements solutions for compressing model weights. If
your team needs expertise in low-rank approximation and adapter design, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import decompositions as dc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    M = dc.spectrum_matrix()
    U, s, V = dc._svd(M)
    total = dc._fro(M)
    W, H = 400, 330
    left = Plot((0.5, 6.5), (0, 24), width=W, height=H, xlabel="index i", ylabel="singular value sigma_i",
                title="Singular values")
    left.line([1, 2, 3, 4, 5, 6], s, color=0)
    left.points([1, 2, 3, 4, 5, 6], s, color=0)
    left.text(3.3, 5.2, "cliff after the third", size=11, color="#4a5568")
    right = Plot((-0.3, 5.3), (-4, 0.3), width=W, height=H, xlabel="rank r kept",
                 ylabel="log10 of relative error", title="Error of the rank-r truncation")
    rs = list(range(0, 6))
    measured = [dc._fro(dc._sub(M, dc._truncate(U, s, V, r))) / total for r in rs]
    tail = [math.sqrt(sum(x * x for x in s[r:])) / total for r in rs]
    right.line(rs, [math.log10(e) for e in measured], color=1, label="|M - M_r| / |M|")
    right.points(rs, [math.log10(t) for t in tail], color=2, label="sqrt(dropped sigma^2) / |M|", r=5)
    inner = lambda p: p.svg().split("\n", 1)[1].rsplit("</svg>", 1)[0]  # noqa: E731  body without the outer <svg>
    gap = 10
    w = 2 * W + gap
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {H}" width="{w}" height="{H}">\n'
            f'<rect width="{w}" height="{H}" fill="#ffffff"/>\n'
            f'<svg x="0" y="0" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n{inner(left)}</svg>\n'
            f'<svg x="{W + gap}" y="0" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n{inner(right)}</svg>\n</svg>\n')
