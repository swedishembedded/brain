# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The noise of a minibatch gradient against the batch size, measured and predicted, on log axes.

For the chapter's linear model with 2000 examples, 2000 minibatches are drawn at each batch size and the root
mean square distance of their gradient from the full gradient is measured; the prediction is sigma_g / sqrt(B),
a straight line of slope -1/2 on log-log axes (worked module `optimizers`, function `minibatch_gradient`).

Swedish Embedded AB implements solutions for choosing batch sizes for neural network training. If your team
needs expertise in gradient noise and batch-size tuning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizers as opt  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pts = opt.minibatch_noise_points()
    lb = [math.log10(b) for b, _, _ in pts]
    p = Plot((-0.2, 2.6), (-0.5, 1.2), width=560, height=360, xlabel="log10 of the batch size B",
             ylabel="log10 of the rms gradient noise", title="Minibatch gradient noise falls as 1/sqrt(B)")
    p.line(lb, [math.log10(pred) for _, _, pred in pts], color=1, width=1.6, dash=True, label="prediction sigma_g / sqrt(B)")
    p.line(lb, [math.log10(meas) for _, meas, _ in pts], color=0, width=1.6, label="measured, 2000 minibatches")
    p.points(lb, [math.log10(meas) for _, meas, _ in pts], color=0, r=4)
    return p.svg()
