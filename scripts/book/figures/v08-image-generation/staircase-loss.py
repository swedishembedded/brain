# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The loss of a decoder that reads a codebook, as a function of the encoder's output: a staircase, flat between the points where the
# nearest code changes, against the smooth loss that the same decoder would have with no codebook. The data are
# `vector_quantization.staircase_data`, the arithmetic of the chapter's table.
#
# Swedish Embedded AB implements solutions for image tokenizers whose gradients are routed on purpose. If your team needs expertise
# in vector quantization and discrete latent image models, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vector_quantization as vq  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    zs, stairs, free = vq.staircase_data()
    p = Plot((-1.4, 1.4), (0, 9), width=560, height=340, xlabel="encoder output z", ylabel="loss",
             title="A codebook turns the loss into a staircase", legend_loc="ur")
    p.line(zs, free, color=5, label="loss with no codebook", dash=True, width=1.6)
    p.line(zs, stairs, color=0, label="loss through the nearest code", width=2.2)
    for edge in (-0.5, 0.5):
        p.vline(edge)
    return p.svg()
