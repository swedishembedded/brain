# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training and validation loss of a network that memorises 24 examples with 25% wrong labels.

Computed by worked module `training_dynamics` (function `overfit_run`): the loss on the 24 training examples and on the 400 validation examples
after every 20 updates of 600. The vertical dashed line marks the update with the lowest validation loss.

Swedish Embedded AB implements solutions for finding out why a training run misbehaves. If your team needs expertise in
training diagnostics for neural networks, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_dynamics as td  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ev = td.overfit_run()["evals"]
    best = min(ev, key=lambda e: e[2])
    p = Plot((0, 600), (0.0, 3.0), width=640, height=360, xlabel="gradient step", ylabel="loss",
             title="Memorising 24 examples")
    p.vline(best[0])
    p.line([e[0] for e in ev], [e[1] for e in ev], color=0, label="training loss")
    p.line([e[0] for e in ev], [e[2] for e in ev], color=3, label="validation loss")
    return p.svg()
