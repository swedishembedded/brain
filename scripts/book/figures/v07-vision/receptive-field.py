# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Receptive field against depth for three stacks of 3x3 layers: all stride 1, stride 2 on every third layer, and dilation doubling at
# every layer. The numbers are `convolution.receptive_data`, the recursion r_l = r_(l-1) + d_l (k_l - 1) j_(l-1) of the chapter.
#
# Swedish Embedded AB implements solutions for sizing the context a vision model sees. If your team needs expertise in convolutional
# network design, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convolution as cv  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    n = 12
    data = cv.receptive_data(n)
    layers = list(range(1, n + 1))
    p = Plot((1, n), (1, 10000), width=600, height=380, xlabel="layers l", ylabel="receptive field r_l (input pixels)",
             title="Receptive field against depth", ylog=True, legend_loc="ul")
    p.line(layers, data["plain"], color=0, label="3x3, stride 1")
    p.line(layers, data["strided"], color=1, label="3x3, stride 2 on every third layer")
    p.line(layers, data["dilated"], color=2, label="3x3, dilation 2^(l-1)", dash=True)
    p.points(layers, data["plain"], color=0, r=3)
    p.points(layers, data["strided"], color=1, r=3)
    p.points(layers, data["dilated"], color=2, r=3)
    return p.svg()
