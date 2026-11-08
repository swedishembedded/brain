# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The roofline climb of a matrix product: naive, tiled, register-blocked and matrix-unit kernels on the model GPU.

Both axes are log10 of the quantity (the book's convention for log plots). The two roofs are the model GPU's vector
roof (40,000 and 800, in GFLOP/s and GB/s) and the matrix unit's roof (8 times the vector peak, a parameter of the chapter);
the points are the rows of the worked module's `climb_rows`, so the picture and `matmul_fast.roofline_climb`
cannot disagree.

Swedish Embedded AB implements solutions for reaching the roofline of the accelerator a model lands on. If your
team needs expertise in GEMM performance analysis for machine learning inference and training, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matmul_fast as mf  # noqa: E402
from svgplot import Plot  # noqa: E402

LG = math.log10
SHORT = ["naive", "T = 16", "T = 32", "T = 64", "T = 128", "fp16, T = 128", "fp16, T = 256"]
# label offsets (dx, dy, anchor) in plot units, one per point
OFFS = [(0.06, -0.20, "start"), (0.06, -0.20, "start"), (0.06, -0.20, "start"), (0.06, -0.20, "start"),
        (0.06, -0.20, "start"), (-0.07, 0.17, "end"), (-0.07, 0.17, "end")]


def _roof(peak, lo=-1.0, hi=3.0):
    knee = LG(peak / mf.BW)
    xs = [lo, knee, hi]
    ys = [LG(min(peak, 10 ** x * mf.BW)) for x in xs]
    return xs, ys


def render() -> str:
    p = Plot((-1.0, 3.0), (1.8, 5.8), width=640, height=420, xlabel="log10 of arithmetic intensity (FLOP per byte)",
             ylabel="log10 of attainable GFLOP/s", title="A matrix product climbing the roofline of the model GPU")
    xs, ys = _roof(mf.F_PK)
    p.line(xs, ys, color=0, label="vector roof, 40,000 (GFLOP/s)")
    xs, ys = _roof(mf.F_TC)
    p.line(xs, ys, color=1, label="matrix-unit roof, 320,000 (GFLOP/s)", dash=True)
    rows = mf.climb_rows()
    vec = [(LG(i), LG(a)) for (_, i, a, pk) in rows if pk == mf.F_PK]
    mat = [(LG(i), LG(a)) for (_, i, a, pk) in rows if pk == mf.F_TC]
    p.points([a for a, _ in vec], [b for _, b in vec], color=2, label="vector units, fp32 operands")
    p.points([a for a, _ in mat], [b for _, b in mat], color=3, label="matrix unit, fp16 operands")
    for (x, y), nm, (dx, dy, anc) in zip(vec + mat, SHORT, OFFS):
        p.text(x + dx, y + dy, nm, anchor=anc, size=11)
    p.vline(LG(mf.F_PK / mf.BW))
    p.vline(LG(mf.F_TC / mf.BW))
    return p.svg()
