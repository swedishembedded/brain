# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training loss of four 24-matrix stacks: plain and residual, each with the same weights.

Computed by worked module `training_dynamics` (function `deep_runs`): the loss of the 512 training examples after every 25 updates of
300, rate 0.1, batch 32. The dashed line is the entropy of the training labels. A curve that leaves the plot at the top starts above 1.6.

Swedish Embedded AB implements solutions for finding out why a training run misbehaves. If your team needs expertise in
training diagnostics for neural networks, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_dynamics as td  # noqa: E402
from svgplot import Plot  # noqa: E402

STYLE = {"plain, tanh, gain 0.5": (1, True), "residual, tanh, gain 0.5": (1, False),
         "plain, ReLU, He": (3, True), "residual, ReLU, He": (3, False)}


def render() -> str:
    _, yt, _, _ = td.gallery_data()
    p = Plot((0, 300), (0.2, 1.6), width=640, height=380, xlabel="gradient step", ylabel="training loss, 512 examples",
             title="Deep stacks, plain and residual")
    p.hline(td._entropy(yt))
    for label, r in td.deep_runs().items():
        color, dash = STYLE[label]
        p.line([e[0] for e in r["evals"]], [min(1.6, e[1]) for e in r["evals"]], color=color, dash=dash, label=label)
    return p.svg()
