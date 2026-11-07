# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The learning rate against step for six schedules of one run: peak 0.3, floor 0.003, warmup 30, T = 600.

Constant, step decay, exponential decay, inverse square root, cosine and warmup-hold-cosine, each computed by the
chapter's formulas (worked module `schedules`, function `shape_functions`; the table `schedules.shape_table` lists the same values).

Swedish Embedded AB implements solutions for tuning the training of neural networks. If your team needs expertise
in learning-rate schedules and loss-curve diagnosis, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import schedules as sch  # noqa: E402
from svgplot import Plot  # noqa: E402

DASHED = {"constant"}


def render() -> str:
    p = Plot((0, sch.SHAPE_T), (0.0, 0.32), width=620, height=380, xlabel="step t", ylabel="learning rate",
             title="Six schedules with the same peak, warmup and length", legend_loc="right")
    ts = list(range(sch.SHAPE_T))
    for i, (name, fn) in enumerate(sch.shape_functions()):
        p.line(ts, [fn(t) for t in ts], color=i, width=1.8, dash=name in DASHED, label=name)
    return p.svg()
