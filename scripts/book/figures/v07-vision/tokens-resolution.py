# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Tokens against the side of the picture: the plain ViT with 16-pixel patches, the first and last stage of Swin-T and the output of the
# hybrid encoder (`vision_transformers.pyramid_series`).
#
# Swedish Embedded AB implements solutions for vision encoders that hand a language model few tokens. If your team needs expertise in
# token budgets for vision-language models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vision_transformers as vt  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    sides, names, tok, _ = vt.pyramid_series()
    series = [("plain ViT, 16-pixel patches", tok[names[0]]),
              ("Swin-T, first stage (stride 4)", tok[names[3]]),
              ("Swin-T, last stage (stride 32)", [(s // 32) ** 2 for s in sides]),
              ("hybrid encoder, output (stride 64)", tok[names[4]])]
    p = Plot((300, 2400), (10, 1e6), width=640, height=380, xlabel="side of the square picture (pixels)", ylabel="tokens",
             title="Tokens against resolution", xlog=True, ylog=True, legend_loc="below")
    for i, (label, ys) in enumerate(series):
        p.line(list(sides), ys, color=i, label=label)
        p.points(list(sides), ys, color=i, r=3)
    return p.svg()
