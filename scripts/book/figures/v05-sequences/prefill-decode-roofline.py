# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Decode steps and prefill passes on the roofline of the illustrative GPU.

Both axes are log10 of the quantity (the book's convention for log plots). The roof is min(peak, intensity x bandwidth) for the GPU of
`roofline.machines` (40,000 GFLOP/s, 800 GB/s). The points are the FLOPs divided by the bytes of `kv_cache.roofline_points` for the
8-billion decoder of the chapter: one decode step at context 4,096 for 1, 8 and 64 sequences, and prefill passes of 16 to 8,192 tokens.

Swedish Embedded AB implements solutions for serving language models without recomputing the past. If your team needs expertise
in key-value caches and decode performance, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kv_cache as kc  # noqa: E402
import roofline as rf  # noqa: E402
from svgplot import Plot  # noqa: E402

LG = math.log10


def render() -> str:
    p = Plot((-0.2, 4.3), (1.2, 5.0), width=620, height=400, xlabel="log10 of arithmetic intensity (FLOP per byte)",
             ylabel="log10 of attainable GFLOP/s", title="Decode and prefill on the roofline")
    knee = LG(rf.ridge("GPU"))
    xs = [-0.2, knee, 4.3]
    # perf-number: illustrative machine of the figure, not a measurement
    p.line(xs, [LG(rf.attainable("GPU", 10 ** x)) for x in xs], color=5, label="GPU roof (40,000 GFLOP/s, 800 GB/s)")
    pts = kc.roofline_points()
    dec = [(LG(i), LG(rf.attainable("GPU", i))) for name, i in pts if name.startswith("decode")]
    pre = [(LG(i), LG(rf.attainable("GPU", i))) for name, i in pts if name.startswith("prefill")]
    p.points([a for a, _ in dec], [b for _, b in dec], color=3, label="decode step, context 4,096", r=4.5)
    p.points([a for a, _ in pre], [b for _, b in pre], color=2, label="prefill pass", r=4.5)
    for (x, y), (name, _) in zip(dec, [q for q in pts if q[0].startswith("decode")]):
        p.text(x + 0.06, y - 0.2, name.replace("decode, ", ""), size=10)
    for (x, y), (name, _) in zip(pre, [q for q in pts if q[0].startswith("prefill")]):
        small = name.startswith("prefill, 16 ")
        p.text(x - 0.06, y + (0.17 if small else -0.22), name.replace("prefill, ", ""), anchor="end" if small else "middle", size=10)
    p.vline(knee)
    return p.svg()
