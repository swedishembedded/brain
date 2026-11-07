# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Histogram of the pre-activations z_10 of the ten-layer tanh stack, Xavier start against weights of standard deviation 1.

The values are the 128 x 64 pre-activations of layer 10 measured by worked module `initialization` (function
`layer_statistics`). Bars show the share of the values in bins of width 1; the dashed lines mark |z| = 3, beyond which the
slope of tanh is below 0.01.

Swedish Embedded AB implements solutions for making deep networks train from their first step. If your team needs
expertise in weight initialization and activation statistics, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import initialization as ini  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402

LO, HI, WIDTH = -20, 20, 1.0
EDGE = 3.0  # tanh'(3) = 0.0099


def shares(zs):
    bins = int((HI - LO) / WIDTH)
    counts = [0] * bins
    for z in zs:
        if LO <= z < HI:
            counts[int((z - LO) / WIDTH)] += 1
    return [c / len(zs) for c in counts]


def render() -> str:
    p = Plot((LO, HI), (0.0, 0.7), width=620, height=360, xlabel="pre-activation z_10", ylabel="share of the 8192 values per bin",
             title="Layer 10 of the tanh stack: two starts")
    for scheme, color, label in (("large", "#c05621", "weights of std 1"), ("xavier", "#2f855a", "Xavier")):
        for i, h in enumerate(shares(ini.figure_histogram("tanh", scheme))):
            if h == 0.0:
                continue
            x0 = LO + i * WIDTH
            p.items.append(f'<rect x="{fmt(p.x(x0))}" y="{fmt(p.y(h))}" width="{fmt(p.x(x0 + WIDTH) - p.x(x0) - 0.6)}" '
                           f'height="{fmt(p.y(0) - p.y(h))}" fill="{color}" fill-opacity="0.65"/>')
        p.legend.append((label, color, "point"))
    for s in (-EDGE, EDGE):
        p.vline(s)
    p.text(EDGE + 0.5, 0.3, "tanh slope < 0.01 beyond |z| = 3")
    return p.svg()
