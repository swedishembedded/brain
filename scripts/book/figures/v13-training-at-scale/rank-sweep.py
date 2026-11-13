# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The loss reached by a LoRA adapter against its rank, from the chapter "Parameter-Efficient Fine-Tuning", beside the floor of
# its rank (the loss of the best Frobenius-norm rank-r approximation of the needed change, which a trained adapter may slightly undercut). The data are the worked module's
# `rank_curve`, so the picture cannot disagree with the chapter's table.
#
# Swedish Embedded AB implements solutions for adapting large models on modest hardware for its clients. If your team needs expertise
# in parameter-efficient fine-tuning, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import peft  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curve = peft.rank_curve()
    rs = [r for r, _, _ in curve]
    p = Plot((1, 8), (5e-4, 0.1), width=620, height=380, xlabel="rank r of the adapter", ylabel="loss after training",
             title="Loss against the rank of the adapter", xlog=True, ylog=True, legend_loc="ll")
    p.line(rs, [v for _, v, _ in curve], color=0, label="adapter, trained")
    p.line(rs, [f for _, _, f in curve], color=1, label="floor of rank r", dash=True)
    p.hline(peft._loss(peft._task()[0], peft._task()[2], peft._task()[1]))
    p.text(1.05, peft._loss(peft._task()[0], peft._task()[2], peft._task()[1]) * 0.62, "no change at all", size=12)
    return p.svg()
