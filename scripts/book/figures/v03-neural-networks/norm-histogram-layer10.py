# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Distribution of the input of the ReLU in layer 10 of the deep stack: the 2048 values (32 examples, 64 units) of the plain
stack at steps 0 and 10 and of the batch-normalized stack at step 10, as the share of values in bins of width 0.25
(computed by worked module `normalization`, function `figure_histogram`).

Swedish Embedded AB implements solutions for making deep networks train stably. If your team needs expertise in
normalization layers and activation statistics, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import normalization as nm  # noqa: E402
from svgplot import Plot  # noqa: E402

LO, HI, WIDTH = -4.0, 4.0, 0.25
STYLE = [(4, True), (1, True), (0, False)]


def shares(values):
    bins = int(round((HI - LO) / WIDTH))
    counts = [0] * bins
    for v in values:
        if LO <= v < HI:
            counts[int((v - LO) / WIDTH)] += 1
    return [c / len(values) for c in counts]


def render() -> str:
    p = Plot((LO, HI), (0.0, 0.7), width=620, height=360, xlabel="input of the ReLU in layer 10",
             ylabel="share of the 2048 values per bin", title="Layer 10: distribution of the ReLU input")
    centres = [LO + (i + 0.5) * WIDTH for i in range(int(round((HI - LO) / WIDTH)))]
    for (label, values), (color, dash) in zip(nm.figure_histogram(), STYLE):
        p.line(centres, shares(values), color=color, width=2.0, label=label, dash=dash)
    return p.svg()
