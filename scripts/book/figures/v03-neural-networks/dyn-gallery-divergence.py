# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The largest logit of the diverging run, against the point at which the textbook softmax overflows.

Computed by worked module `training_dynamics` (the diverging gallery run, rate 10): the base-10 logarithm of the largest absolute logit of the
minibatch at each of the first 7 steps. The dashed horizontal line is log10 of 709.78, the largest argument for which exp is finite
in double precision; the textbook softmax of the run first fails at step 3, where the largest logit is 169436.

Swedish Embedded AB implements solutions for finding out why a training run misbehaves. If your team needs expertise in
training diagnostics for neural networks, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_dynamics as td  # noqa: E402
from svgplot import Plot  # noqa: E402

LAST = 7


def render() -> str:
    r = td.gallery_run("diverging")
    steps = list(range(LAST))
    p = Plot((0, LAST - 1), (0, 24), width=620, height=360, xlabel="gradient step",
             ylabel="log10 of the largest |logit| in the batch", title="The diverging run (rate 10)")
    p.hline(math.log10(709.78))
    p.line(steps, [math.log10(r["max_logit"][t]) for t in steps], color=1, label="largest |logit|")
    p.line(steps, [math.log10(r["step_loss"][t]) for t in steps], color=0, label="minibatch loss", dash=True)
    p.text(0.1, 3.7, "exp overflows above 709.78", size=12)
    return p.svg()
