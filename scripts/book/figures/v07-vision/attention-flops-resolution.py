# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# FLOPs of the score and average products of attention over a whole network against the side of the picture, for a plain ViT with
# global attention, with windows, with windows and four global blocks, for Swin-T and for the hybrid encoder of the chapter
# (`vision_transformers.pyramid_series`).
#
# Swedish Embedded AB implements solutions for vision encoders whose cost is counted before they are trained. If your team needs expertise
# in the cost of attention at high resolution, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vision_transformers as vt  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    sides, names, _, flops = vt.pyramid_series()
    p = Plot((300, 2400), (0.1, 1e4), width=640, height=400, xlabel="side of the square picture (pixels)",
             ylabel="GFLOPs of the score and average products", title="Attention cost of the whole network against resolution",
             xlog=True, ylog=True, legend_loc="below")
    for i, n in enumerate(names):
        g = [f / 1e9 for f in flops[n]]
        p.line(list(sides), g, color=i, label=n)
        p.points(list(sides), g, color=i, r=3)
    return p.svg()
