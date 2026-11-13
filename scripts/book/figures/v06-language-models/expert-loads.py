# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Slots per expert after training the tiny mixture of the chapter (8 experts, 2 per token, 128 training tokens, initialisation
# seed 0) with three strengths of the balancing loss. The dashed line is the balanced load, 32 slots. Numbers from
# `mixture_of_experts.balance_data`, the function behind the table of the chapter.
#
# Swedish Embedded AB implements solutions for training sparse language models. If your team needs expertise in expert load
# balancing, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mixture_of_experts as moe  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = moe.balance_data()
    ideal = moe.TR_N * moe.TR_K // moe.TR_E
    p = Plot((-0.4, moe.TR_E - 0.6), (0, 80), width=620, height=360, xlabel="expert", ylabel="slots routed to the expert",
             title="Load of each expert after training (initialization 0)", legend_loc="ur")
    p.hline(ideal)
    for color, a in ((3, 0.0), (1, 0.05), (0, 0.5)):
        loads = data[a][0]
        label = "no balancing loss" if a == 0.0 else f"balancing coefficient {a:g}"
        p.line(list(range(moe.TR_E)), loads, color=color, label=label, width=1.6)
        p.points(list(range(moe.TR_E)), loads, color=color, r=3.5)
    p.text(moe.TR_E - 0.7, ideal + 2.5, f"balanced: {ideal}", anchor="end", size=11)
    return p.svg()
