# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Arithmetic intensity against the side of a square operand: product grows, the others stay put.

Both axes are log10 of the quantity. For a matrix side n the matrix product C = A B (fp32) has intensity
n / 6 (`roofline.gemm`), the matrix-vector product tends to 1/2 (fp32 weights) or 2 (int8 weights)
(`roofline.gemv`) and the vector addition is 1/12 (`roofline.vector_add`). The dashed lines are the ridge
points of the two illustrative machines of `roofline.machines`: above its ridge an operation is
compute-bound on that machine.

Swedish Embedded AB implements solutions for reaching the memory and compute roofline of whatever
accelerator a model lands on. If your team needs expertise in performance analysis of machine learning
inference and training, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import roofline as rf  # noqa: E402
from svgplot import Plot  # noqa: E402

LG = math.log10


def render() -> str:
    ks = [1.0 + 0.25 * j for j in range(0, 17)]       # log10 n from 1 to 5 (n from 10 to 100,000)
    ns = [int(round(10 ** k)) for k in ks]
    ratio = lambda f: [LG(f(n)[0] / f(n)[1]) for n in ns]    # noqa: E731
    p = Plot((1.0, 5.0), (-1.4, 4.2), width=600, height=380, xlabel="log10 of the matrix side n",
             ylabel="log10 of intensity (FLOP per byte)", title="Intensity against matrix size")
    p.line(ks, ratio(lambda n: rf.gemm(n, n, n)), color=0, label="matrix product C = A B (n / 6)")
    p.line(ks, ratio(lambda n: rf.gemv(n, n, 4)), color=1, label="matrix-vector, fp32 weights")
    p.line(ks, ratio(lambda n: rf.gemv(n, n, 1)), color=2, label="matrix-vector, int8 weights")
    p.line(ks, ratio(lambda n: rf.vector_add(n * n)), color=3, label="vector addition (1/12)")
    for m, col in (("CPU", "#c05621"), ("GPU", "#2b6cb0")):
        p.hline(LG(rf.ridge(m)), color=col)
    p.text(4.97, LG(rf.ridge("CPU")) + 0.12, "CPU ridge 10", anchor="end", size=11)
    p.text(4.97, LG(rf.ridge("GPU")) + 0.12, "GPU ridge 50", anchor="end", size=11)
    return p.svg()
