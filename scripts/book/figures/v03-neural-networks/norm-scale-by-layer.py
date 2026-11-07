# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Root mean square of the input of the ReLU in each of the ten layers of the deep stack, with and without a
normalization layer (computed by worked module `normalization`, function `figure_scales`). Without normalization the
scale of the deep layers falls away from the designed value within ten steps of training; with a normalization layer it
is held near 1 by construction.

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

STYLE = [(4, True), (1, True), (3, False), (0, False)]


def render() -> str:
    series = nm.figure_scales()[:4]
    series[3] = ("with any of the three normalization layers, step 10", series[3][1])
    p = Plot((1, 10), (0, 2.4), width=620, height=380, xlabel="layer l", ylabel="rms of the input of the ReLU",
             title="Scale of the activations by layer (rate 0.01)")
    p.hline(1.0)
    for (label, vals), (color, dash) in zip(series, STYLE):
        p.line(list(range(1, 11)), vals, color=color, width=2.0, label=label, dash=dash)
    return p.svg()
