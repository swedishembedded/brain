# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The roofline of the illustrative GPU, with the chapter's operations and LLM decode marked.

Both axes are log10 of the quantity (the book's convention for log plots). The roof is
min(F_pk, intensity x BW) for the GPU of `roofline.machines` (40,000 GFLOP/s, 800 GB/s) and for the
CPU (800 GFLOP/s, 80 GB/s); the points are the operations of `roofline.three_operations` and a
7-billion-weight model in fp16 at batch sizes 1, 8, 64 and 512 (intensity = batch size), all placed on
the GPU roof by the worked module's own `attainable`.

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


def _roof(machine, lo=-2.0, hi=3.4):
    peak, bw = rf.MACHINES[machine]
    knee = LG(rf.ridge(machine))
    xs = [lo, knee, hi]
    ys = [LG(rf.attainable(machine, 10 ** x)) for x in xs]
    return xs, ys


def render() -> str:
    p = Plot((-2.0, 3.4), (0.5, 4.9), width=620, height=400, xlabel="log10 of arithmetic intensity (FLOP per byte)",
             ylabel="log10 of attainable GFLOP/s", title="The roofline of two illustrative machines")
    xs, ys = _roof("GPU")
    p.line(xs, ys, color=0, label="GPU roof (40,000 GFLOP/s, 800 GB/s)")
    xs, ys = _roof("CPU")
    p.line(xs, ys, color=1, label="CPU roof (800 GFLOP/s, 80 GB/s)")
    ops = rf._operations()
    marks = [(f / b, name) for name, f, b in ops]
    pts = [(LG(i), LG(rf.attainable("GPU", i))) for i, _ in marks]
    p.points([a for a, _ in pts], [b for _, b in pts], color=2, label="operations on the GPU roof")
    names = ["vector add", "GEMV fp32", "GEMV int8", "GEMM 4096"]
    offs = [(0.07, -0.22, "start"), (0.07, -0.22, "start"), (0.07, -0.22, "start"), (-0.07, -0.30, "end")]
    for (x, y), nm, (dx, dy, anc) in zip(pts, names, offs):
        p.text(x + dx, y + dy, nm, anchor=anc, size=11)
    dec = [1, 8, 64, 512]
    dp = [(LG(b), LG(rf.attainable("GPU", float(b)))) for b in dec]
    p.points([a for a, _ in dp], [b for _, b in dp], color=3, label="fp16 decode, batch 1, 8, 64, 512")
    for (x, y), b in zip(dp, dec):
        p.text(x - 0.07, y + 0.18, f"B={b}", anchor="end", size=10)
    p.vline(LG(rf.ridge("GPU")))
    p.vline(LG(rf.ridge("CPU")))
    return p.svg()
