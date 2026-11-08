# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bytes read per generated token against the storage size of a weight, with the four formats of chapter 1.

A dense model of 7,000,000,000 weights reads every weight once per token at batch size 1, so the bytes per
token are P times the bytes per weight (`roofline.FORMATS`: fp32 4, fp16 2, int8 with an f32 scale per 32
values 1.125, int4 with the same scale 0.625). The second curve is the same bytes expressed as the seconds the
illustrative GPU of `roofline.machines` (800 GB/s) needs to deliver them, which is the roofline's lower bound
on the time of one token.

Swedish Embedded AB implements solutions for reaching the memory and compute roofline of whatever
accelerator a model lands on. If your team needs expertise in performance analysis of machine learning
inference and training, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import roofline as rf  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0.0, 4.6), (0.0, 34.0), width=600, height=360, xlabel="bytes per weight",
             ylabel="GB read per token (7 billion weights)", title="Bytes per token against weight format")
    xs = [0.0, 4.0]
    p.line(xs, [rf.DECODE_PARAMS * x / 1e9 for x in xs], color=0, dash=True, width=1.2, label="bytes = 7e9 x bytes per weight")
    pts = [(b, rf.DECODE_PARAMS * b / 1e9) for _, b in rf.FORMATS]
    p.points([a for a, _ in pts], [b for _, b in pts], color=0, label="stored formats", r=4.5)
    short = ["fp32", "fp16", "int8 + scales", "int4 + scales"]
    for (x, y), nm in zip(pts, short):
        if x >= 3.5:
            p.text(x + 0.3, y - 8.0, f"{nm}: {y:.2f} GB", anchor="end", size=11)
        else:
            p.text(x + 0.15, y - 1.8, f"{nm}: {y:.2f} GB", size=11)
    return p.svg()
