# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Full-data loss against step for six optimisers on the chapter's small network, each at its best learning rate.

The network is 2 -> 8 tanh -> 1 sigmoid on 200 points labelled by the sign of x1 x2, trained for 300 steps of batch
size 20 with the same batches for every rule. The learning rate of each rule is the best of the chapter's grid
(worked module `optimizers`, functions `mlp_sweep` and `mlp_curve_table`); the curves are the sweep's own runs.

Swedish Embedded AB implements solutions for tuning the training of neural networks. If your team needs expertise
in optimiser choice and loss-curve diagnosis, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizers as opt  # noqa: E402
from svgplot import Plot  # noqa: E402

DASHED = {"AdamW 0.05"}


def render() -> str:
    curves = opt.mlp_curves()
    p = Plot((0, opt.MLP_STEPS), (0.0, 0.8), width=620, height=380, xlabel="step (batch of 20)",
             ylabel="mean loss over all 200 examples", title="Six optimisers at their best learning rate", legend_loc="right")
    for i, (name, (lr, curve)) in enumerate(curves.items()):
        p.line([s for s, _ in curve], [v for _, v in curve], color=i, width=1.8, dash=name in DASHED, label=f"{name} (eta {lr:g})")
    return p.svg()
