# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Global loads of a 1024 by 1024 product against the side of the workgroup tile.

Both axes are log10 of the quantity. Loads are 2 n^3 / T for a workgroup tile of side T, from the worked module's
`loads_curve`. A workgroup of one output per invocation cannot have a tile beyond 16 (256 invocations, the portable
limit of Chapter 4); a workgroup of 256 invocations that each hold an r x r block reaches T = 16 r.

Swedish Embedded AB implements solutions for cutting the memory traffic of dense linear algebra. If your team
needs expertise in GEMM kernel design for machine learning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matmul_fast as mf  # noqa: E402
from svgplot import Plot  # noqa: E402

LG = math.log10


def render() -> str:
    n = mf.N_BIG
    curve = mf.loads_curve(n)
    p = Plot((-0.15, 3.15), (6.0, 9.6), width=620, height=380, xlabel="log10 of the side T of the workgroup tile",
             ylabel="log10 of global loads (numbers)", title=f"Global loads of a {n} by {n} product, 2 n^3 / T")
    p.line([LG(b) for b, _ in curve], [LG(v) for _, v in curve], color=0, label="loads = 2 n^3 / T")
    one = [(b, v) for b, v in curve if b <= 16]
    blocked = [(b, v) for b, v in curve if b in (32, 64, 128)]
    p.points([LG(b) for b, _ in one], [LG(v) for _, v in one], color=2, label="one output per invocation, T <= 16")
    p.points([LG(b) for b, _ in blocked], [LG(v) for _, v in blocked], color=3, label="256 invocations of r x r blocks, T = 16 r")
    p.vline(LG(16))
    p.hline(LG(2 * n * n))
    p.text(LG(16) + 0.04, 9.35, "256 invocations", size=11)
    p.text(0.05, LG(2 * n * n) + 0.12, "T = n: each number loaded once", size=11)
    for b, v in blocked:
        p.text(LG(b) + 0.05, LG(v) + 0.14, f"r = {b // 16}", size=10)
    return p.svg()
