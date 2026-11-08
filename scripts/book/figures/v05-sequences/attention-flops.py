# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The FLOPs of one attention layer (width 1024, 16 heads) as the sequence grows: the four projections grow in proportion to T,
# the products Q K^T and P V grow as T squared, and the two are equal at T = 2 D. Numbers from `self_attention.flops_data`,
# the same closed forms as the table of the chapter.
#
# Swedish Embedded AB implements solutions for sizing the compute and memory of long-context models. If your team needs
# expertise in attention cost and long-context inference, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import self_attention as sa  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ts, proj, attn = sa.flops_data()
    p = Plot((0, 8448), (0, 300), width=640, height=380, xlabel="sequence length T", ylabel="GFLOPs in one layer",
             title=f"Projections against the T x T products, D = {sa.COST_D}")
    p.line(ts, proj, color=1, label="four projections, 8 T D^2")
    p.line(ts, attn, color=3, label="Q K^T and P V, 4 T^2 D")
    cross = 2 * sa.COST_D
    p.vline(cross)
    p.text(cross + 150, 270, "equal at T = 2 D", size=12, color="#4a5568")
    return p.svg()
