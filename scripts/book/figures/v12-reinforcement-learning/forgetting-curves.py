# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The accuracy on task A of the network trained on A while it is trained on task B alone, against the number of steps, for three step
# sizes (every 5 steps of `continual_learning.forgetting_grid`'s runs), and the accuracy on task B beside it.
#
# Swedish Embedded AB implements solutions for models that keep learning without losing what they knew. If your team needs expertise in
# continual learning, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import continual_learning  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curves = continual_learning._curves()
    p = Plot((0, 200), (0.0, 1.0), width=620, height=360, xlabel="steps of gradient descent on task B alone",
             ylabel="share of 400 test rows classified right", title="Training on B takes accuracy on A away")
    for i, lr in enumerate(continual_learning.RATES):
        pts = curves[lr]
        p.line([s for s, _, _ in pts], [a / continual_learning.N_TEST for _, a, _ in pts], color=i, label=f"task A, step size {lr}")
        p.line([s for s, _, _ in pts], [b / continual_learning.N_TEST for _, _, b in pts], color=i, dash=True, label=f"task B, step size {lr}")
    return p.svg()
