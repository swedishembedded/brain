# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The root mean square of the stream after k blocks of a freshly initialized stack of 32 blocks, for the wirings of the chapter: in a
# pre-norm stack every sublayer adds to the stream and the scale grows like sqrt(1 + k); a post-norm block ends in a norm, so the scale
# is pinned at 1. Numbers from `transformer_block.depth_data`, the same function as the table of the chapter.
#
# Swedish Embedded AB implements solutions for training deep transformer stacks that start stable. If your team needs expertise in
# initialization and normalization of transformers, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import transformer_block as tb  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    runs = tb.depth_data()
    ks = list(range(tb.DEPTH_L + 1))
    p = Plot((0, tb.DEPTH_L), (0, 6.5), width=640, height=380, xlabel="blocks k from the input", ylabel="rms of the stream",
             title=f"Scale of the stream along a stack of {tb.DEPTH_L} blocks at initialization")
    p.line(ks, [math.sqrt(1 + k) for k in ks], color=5, dash=True, label="sqrt(1 + k)", width=1.4)
    for color, (_, _, label) in zip((1, 0, 2, 3), tb.DEPTH_VARIANTS):
        p.line(ks, runs[label]["rms"], color=color, label=label)
    return p.svg()
