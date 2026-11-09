# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Bytes moved to and from global memory by one head (width 64, float32) as the sequence length grows, for the three-kernel chain that
# materialises the score matrix and for tiled attention with 16, 64 and 256 query rows per block. Numbers from
# `flash_attention.bytes_curve`, the formulas of the chapter's table.
#
# Swedish Embedded AB implements solutions for counting the memory traffic of attention before building the kernel. If your team needs
# expertise in fused attention and the roofline, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flash_attention as fa  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ts, chain, tiled = fa.bytes_curve()
    p = Plot((0, 16896), (0, 8.8), width=620, height=380, xlabel="sequence length T", ylabel="GiB moved by one head",
             title=f"Global-memory traffic of one head, d = {fa.D_HEAD}, float32")
    p.line(ts, chain, color=3, label="chain that materialises S and P")
    p.line(ts, tiled[16], color=1, label="tiled, B_q = 16", dash=True)
    p.line(ts, tiled[64], color=0, label="tiled, B_q = 64")
    p.line(ts, tiled[256], color=2, label="tiled, B_q = 256")
    return p.svg()
