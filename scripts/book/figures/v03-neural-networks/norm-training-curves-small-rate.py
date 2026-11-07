# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training loss of the deep stack with no normalization, batch norm, layer norm and RMS norm at the rate 0.01
(computed by worked module `normalization`, function `figure_losses`). The vertical axis is the base-ten logarithm of
the loss; a run whose loss exceeds 1e8 is cut at that step and marked with a point on the top edge.

Swedish Embedded AB implements solutions for making deep networks train stably. If your team needs expertise in
normalization layers and training dynamics, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import normalization as nm  # noqa: E402
from svgplot import Plot  # noqa: E402

RATE, LOW, TOP = 0.01, -14, 8.0
COLORS = {"none": 4, "batch": 0, "layer": 2, "rms": 1}


def render() -> str:
    p = Plot((0, nm.STACK_STEPS), (LOW, 9), width=620, height=380, xlabel="gradient step", ylabel="log10 of the loss",
             title="Deep stack, learning rate 0.01")
    for kind, losses in nm.figure_losses(RATE).items():
        xs, ys = [], []
        for i, v in enumerate(losses):
            if not math.isfinite(v) or v > 10 ** TOP:
                xs.append(i)
                ys.append(TOP)
                p.points([i], [TOP], color=COLORS[kind], r=4.5)
                break
            xs.append(i)
            ys.append(math.log10(v))
        p.line(xs, ys, color=COLORS[kind], width=2.0, label=nm.KIND_LABEL[kind])
    return p.svg()
