# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Share of dead hidden units over training, for ReLU at two rates and for a leaky ReLU at the high rate.

Computed by worked module `training_dynamics` (function `dead_runs`): the fraction of the 64 hidden units that are not positive on any of 256 fixed
training inputs, after every 10 updates of 200.

Swedish Embedded AB implements solutions for finding out why a training run misbehaves. If your team needs expertise in
training diagnostics for neural networks, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_dynamics as td  # noqa: E402
from svgplot import Plot  # noqa: E402

STYLE = {"ReLU, rate 0.1": (2, False), "ReLU, rate 2": (3, False), "leaky ReLU, rate 2": (0, True)}


def render() -> str:
    p = Plot((0, td.DEAD_STEPS), (0.0, 0.4), width=620, height=360, xlabel="gradient step",
             ylabel="fraction of hidden units that are dead", title="Dead units during training")
    for label, series in td.dead_curves():
        color, dash = STYLE[label]
        p.line([t for t, _ in series], [f for _, f in series], color=color, dash=dash, label=label)
    return p.svg()
