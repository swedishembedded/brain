# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One pre-norm transformer block drawn around its residual stream: the stream runs down the left edge and is never rewritten, only
# added to; each of the two sublayers (attention, feed-forward network) reads a normalized copy off the stream and adds its result back.
# The shapes are those of the chapter's example (T = 3 tokens, D = 4).
#
# Swedish Embedded AB implements solutions for transformer models, from the block layout to the kernels. If your team needs expertise
# in the architecture of transformer blocks, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import transformer_block as tb  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 860, 520
XS = 90                    # the stream
Y_IN, Y_ADD1, Y_MID, Y_ADD2, Y_OUT = 40, 160, 262, 382, 490
Y_B1, Y_B2 = 88, 308       # the branch rows
NORM_X, SUB_X = 270, 470


def render() -> str:
    T, D = tb.TINY_T, tb.TINY_D
    d = Diagram(W, H)
    x_in = d.oval(XS, Y_IN, 100, 44, ["x", f"{T} x {D}"], "accent")
    add1 = d.oval(XS, Y_ADD1, 40, 40, ["+"], "plain")
    x_mid = d.oval(XS, Y_MID, 100, 44, ["x1", f"{T} x {D}"], "accent")
    add2 = d.oval(XS, Y_ADD2, 40, 40, ["+"], "plain")
    x_out = d.oval(XS, Y_OUT, 100, 44, ["x2", f"{T} x {D}"], "result")
    d.arrow(x_in["b"], add1["t"])
    d.arrow(add1["b"], x_mid["t"])
    d.arrow(x_mid["b"], add2["t"])
    d.arrow(add2["b"], x_out["t"])
    for y_branch, add, name, sub in ((Y_B1, add1, "norm 1", ["multi-head", "attention"]), (Y_B2, add2, "norm 2", ["SwiGLU", "feed-forward"])):
        norm = d.box(NORM_X, y_branch, 120, 50, ["RMS norm", name.split()[1] and f"({name})"], "model")
        res = d.box(SUB_X, y_branch, 150, 56, sub, "model")
        d.path([(XS, y_branch), norm["l"]])
        d.arrow(norm["r"], res["l"])
        d.path([res["b"], (SUB_X, add["r"][1]), add["r"]])
    d.text(XS + 30, (Y_IN + Y_ADD1) / 2 + 18, ["the stream"], size=12, color=MUTED, anchor="start")
    d.text(555, Y_B1 + 70, ["sublayer 1 reads a normalized copy", "and adds what it computes"], size=12, color=MUTED, anchor="start")
    d.text(555, Y_B2 + 70, ["sublayer 2 does the same on a stream", "that already holds the result of sublayer 1"], size=12, color=MUTED, anchor="start")
    return d.svg()
