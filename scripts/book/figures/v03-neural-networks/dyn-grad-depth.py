# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Gradient norm of each weight matrix of a 24-matrix stack at the start of training, for six stacks.

Computed by worked module `training_dynamics` (function `stack_norms`): the Frobenius norm of the gradient of each of the 24 hidden weight matrices,
base-10 logarithm, matrix 1 nearest the input and matrix 24 nearest the read-out, on the first 32 training examples. The stacks share their
normal() draws and differ in activation, weight scale (gain) and skip connections.

Swedish Embedded AB implements solutions for finding out why a training run misbehaves. If your team needs expertise in
training diagnostics for neural networks, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_dynamics as td  # noqa: E402
from svgplot import Plot  # noqa: E402

STYLE = {"plain, sigmoid, gain 1": (0, False), "plain, tanh, gain 0.5": (1, False), "plain, tanh, gain 1": (5, True),
         "plain, tanh, gain 2.5": (3, False), "plain, ReLU, He": (4, True), "residual, ReLU, He": (2, False)}


def render() -> str:
    p = Plot((1, td.STACK_MATRICES), (-17, 3), width=640, height=400, xlabel="weight matrix, counted from the input",
             ylabel="log10 of the gradient norm of the matrix", title="Gradient norm by depth at the start")
    for label, vals in td.stack_grad_curves():
        color, dash = STYLE[label]
        p.line(list(range(1, td.STACK_MATRICES + 1)), vals, color=color, dash=dash, label=label)
    return p.svg()
