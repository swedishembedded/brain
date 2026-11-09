# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Parameters of one feed-forward layer against the FLOPs a token spends in it (D = 2,048, expert hidden width 512, SwiGLU). A dense
# layer moves along the diagonal: every parameter is used. A mixture with k experts used per token stands at the same x as a dense
# layer of k experts' width and rises with the number of experts E. Numbers from `mixture_of_experts.params_flops_data`, the
# function behind the table of the chapter.
#
# Swedish Embedded AB implements solutions for sizing sparse language models. If your team needs expertise in mixture-of-experts
# capacity planning, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mixture_of_experts as moe  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    series = moe.params_flops_data()
    p = Plot((0, 90), (0, 900), width=620, height=400, xlabel="MFLOPs per token in the layer", ylabel="parameters of the layer (millions)",
             title="Capacity against compute, one feed-forward layer", legend_loc="below")
    dense = series[0][1]
    p.line([x for x, _ in dense], [y for _, y in dense], color=3, label=series[0][0], width=1.6)
    p.points([x for x, _ in dense], [y for _, y in dense], color=3, r=3)
    for color, (label, pts) in zip((1, 0), series[1:]):
        p.line([x for x, _ in pts], [y for _, y in pts], color=color, label=label, width=1.6)
        p.points([x for x, _ in pts], [y for _, y in pts], color=color, r=3.5)
    top = series[2][1][-1]
    p.text(top[0] + 1.5, top[1] - 4, f"E = 256, k = 8:", size=11)
    p.text(top[0] + 1.5, top[1] - 38, f"{top[1]:.0f} M parameters,", size=11)
    p.text(top[0] + 1.5, top[1] - 72, f"{top[0]:.1f} MFLOPs", size=11)
    return p.svg()
