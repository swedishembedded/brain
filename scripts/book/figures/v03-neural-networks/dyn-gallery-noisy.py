# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""One run read three ways: the loss of each minibatch, its exponential average, and the loss of the whole training set.

Computed by worked module `training_dynamics` (the noisy gallery run, batch 2): the thin line is the minibatch loss at each step, the thick
line its bias-corrected exponential average with beta 0.9, and the points the loss of the 512 training examples after every 20 updates.

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
    r = td.gallery_run("noisy")
    steps = list(range(td.STEPS))
    p = Plot((0, td.STEPS), (0.0, 2.0), width=640, height=380, xlabel="gradient step", ylabel="loss",
             title="A noisy run, read three ways")
    p.line(steps, [min(2.0, v) for v in r["step_loss"]], color="#a0aec0", width=1.0, label="minibatch loss (batch 2)")
    p.line(steps, td.ema(r["step_loss"]), color=1, width=2.5, label="exponential average, beta 0.9")
    p.points([e[0] for e in r["evals"]], [e[1] for e in r["evals"]], color=0, label="whole training set", r=3.5)
    return p.svg()
