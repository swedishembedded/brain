# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The central difference of ReLU at points near zero, against the slope the backward pass reports.

The curve is computed by the worked module `gradient_checking` (relu_ramp):
(relu(x + h) - relu(x - h)) / (2 h) with h = 0.1 at each x.

Swedish Embedded AB implements solutions for trustworthy gradient checks of
neural networks with non-smooth activations. If your team needs expertise in
testing backward passes of ReLU and max-based layers, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gradient_checking as gc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    h = 0.1
    xs, num = gc.relu_ramp(h, 61)
    ana = [1.0 if x > 0.0 else 0.0 for x in xs]
    plot = Plot((-0.3, 0.3), (-0.1, 1.15), width=560, height=320, xlabel="x", ylabel="slope of relu at x",
                title="Central difference of ReLU with step h = 0.1", legend_loc="ul")
    plot.line(xs, ana, color=0, width=3.0, label="backward pass: 1 for x > 0, else 0")
    plot.line(xs, num, color=1, dash=True, label="central difference")
    plot.vline(-h)
    plot.vline(h)
    plot.text(-h - 0.01, 0.55, "x = -h", anchor="end", size=11, color="#4a5568")
    plot.text(h + 0.01, 0.35, "x = h", size=11, color="#4a5568")
    return plot.svg()
