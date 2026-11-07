# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training loss of the ten-layer tanh stack from five starts: 100 steps of gradient descent.

The curves are the runs of worked module `initialization` (function `train_run`): 32 examples, mean squared error, rate 0.01.
A curve that leaves the plot at the top is off the scale (the start with weights of standard deviation 1 begins at loss 82.8). The dashed
horizontal line is the loss of always predicting the mean of the targets; the dashed curve is the all-zero start.

Swedish Embedded AB implements solutions for making deep networks train from their first step. If your team needs
expertise in weight initialization and training diagnostics, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import initialization as ini  # noqa: E402
from svgplot import Plot  # noqa: E402

LABELS = {"zero": "zeros", "small": "small (0.01)", "large": "large (1)", "xavier": "Xavier", "he": "He"}
COLORS = {"zero": 5, "small": 0, "large": 1, "xavier": 2, "he": 3}


def render() -> str:
    _, var_t = ini._target_stats()
    p = Plot((0, ini.TRAIN_STEPS), (0.0, 2.5), width=620, height=360, xlabel="step",
             ylabel="mean squared error on the 32 examples", title="Ten tanh layers, five starts")
    p.hline(var_t)
    for scheme, losses in ini.figure_training_curves("tanh").items():
        finite = [(s, v) for s, v in enumerate(losses) if v < 1e6]
        p.line([s for s, _ in finite], [v for _, v in finite], color=COLORS[scheme], width=2.0, dash=(scheme == "zero"),
               label=LABELS[scheme])
    return p.svg()
