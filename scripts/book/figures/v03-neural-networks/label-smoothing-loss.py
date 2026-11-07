# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Cross-entropy against the logit of the true class, with and without label smoothing, four classes.

Swedish Embedded AB implements solutions for training classifiers whose confidence can be trusted. If your team
needs expertise in classification losses and calibration, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import softmax  # noqa: E402
from svgplot import Plot  # noqa: E402

EPSILONS = [0.0, 0.1, 0.2]


def render() -> str:
    gs = [-2.0 + 14.0 * i / 560 for i in range(561)]
    p = Plot((-2.0, 12.0), (0.0, 2.0), width=560, height=340, xlabel="logit of the true class (the other three are 0)",
             ylabel="cross-entropy loss", title="Loss against the true logit, with label smoothing", legend_loc="ur")
    for color, eps in enumerate(EPSILONS):
        label = "one-hot target" if eps == 0 else f"smoothed, eps = {eps:g}"
        p.line(gs, [softmax._smoothed_loss_gap(g, eps) for g in gs], color=color, label=label)
        if eps > 0:
            g = softmax._optimal_gap(eps)
            p.points([g], [softmax._smoothed_loss_gap(g, eps)], color=color, r=4.5)
            p.text(g, softmax._smoothed_loss_gap(g, eps) - 0.14, f"{g:.2f}", anchor="middle", size=11)
    return p.svg()
