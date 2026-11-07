# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Full-data loss over the first 60 steps of AdamW at a peak rate of 3, with no warmup and with warmups of 10, 30 and 60 steps.

The runs are those of `schedules.warmup_lengths` (the Chapter 7 network, 33 parameters, batches of 20); the starting
loss is 0.7471.

Swedish Embedded AB implements solutions for tuning the training of neural networks. If your team needs expertise
in learning-rate warmup and loss-curve diagnosis, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import schedules as sch  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curves = sch.warmup_curves()
    p = Plot((0, 60), (0.0, 7.5), width=620, height=380, xlabel="step", ylabel="mean loss over all 200 examples",
             title="AdamW at a peak rate of 3: the first 60 steps", legend_loc="right")
    p.hline(curves[0][0][1])
    for i, (W, curve) in enumerate(curves.items()):
        p.line([s for s, _ in curve], [v for _, v in curve], color=i, width=1.8, label="no warmup" if W == 0 else f"warmup {W}")
    return p.svg()
