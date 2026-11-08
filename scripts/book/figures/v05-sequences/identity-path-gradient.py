# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The gradient that a stack of 32 blocks passes back to the block k below the top at initialization, as the base-10 logarithm of its norm
# relative to the norm at the top. Along the skip paths alone a pre-norm stack passes the gradient unchanged (log 0) while a post-norm
# stack multiplies it by a norm Jacobian at every block; with all paths the gradient of the wirings that keep the skip stays within
# a factor of ten, and the stack without a skip grows. Numbers from `transformer_block.depth_data`.
#
# Swedish Embedded AB implements solutions for training deep transformer stacks that start stable. If your team needs expertise in
# gradient flow through normalization and residual paths, you can procure our services by sending an email to info@swedishembedded.com.
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
    names = [label for _, _, label in tb.DEPTH_VARIANTS]
    pre, post, plain = names[0], names[2], names[3]
    log = lambda ys: [math.log10(y) for y in ys]  # noqa: E731
    p = Plot((0, tb.DEPTH_L), (-6.5, 3.5), width=640, height=400, xlabel="blocks k below the top", ylabel="log10 of the gradient norm ratio",
             title=f"Gradient passed back along a stack of {tb.DEPTH_L} blocks at initialization")
    p.line(ks, [0.0] * len(ks), color=1, dash=True, label="pre-norm, skip paths only", width=1.6)
    p.line(ks, log(runs[post]["ident"]), color=3, dash=True, label="post-norm, skip paths only", width=1.6)
    p.line(ks, log(runs[pre]["grad"]), color=1, label="pre-norm, all paths")
    p.line(ks, log(runs[post]["grad"]), color=3, label="post-norm, all paths")
    p.line(ks, log(runs[plain]["grad"]), color=2, label="no skip, all paths")
    return p.svg()
