# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The backward pass of one batched linear layer as a shapes diagram. Top: the
# forward pass Y = X W^T + b. Bottom: the gradient dY arriving at Y is turned
# into dX, dW and db by three products whose shapes are checked on the arrows'
# boxes: the inner dimensions must agree, and each result has the shape of the
# array it is the gradient of. The sizes (B = 3, D = 2, O = 4) are those of the
# chapter's worked example and are read from the worked module.
#
# Swedish Embedded AB implements solutions for verifiable gradient computation
# in neural network training. If your team needs expertise in the backward
# pass of neural network layers, you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import layer_gradients as lg  # noqa: E402
from svgdiagram import RED, Diagram  # noqa: E402

W, H = 860, 400
OVAL_W, OVAL_H = 150, 56
BOX_W, BOX_H = 270, 62


def render() -> str:
    b, d, o = lg.LIN_SIZES
    dg = Diagram(W, H)

    dg.panel(16, 14, W - 32, 112, "forward")
    cy = 76
    x_in = dg.oval(110, cy, OVAL_W, OVAL_H, ["X", f"[B, D] = [{b}, {d}]"], "result")
    lin = dg.box(430, cy, BOX_W, BOX_H, ["linear: Y = X W^T + b", f"W [O, D] = [{o}, {d}],  b [O] = [{o}]"], "model")
    y_out = dg.oval(750, cy, OVAL_W, OVAL_H, ["Y", f"[B, O] = [{b}, {o}]"], "plain")
    dg.arrow(x_in["r"], lin["l"])
    dg.arrow(lin["r"], y_out["l"])

    dg.panel(16, 142, W - 32, 246, "backward: the gradient arriving at Y is split into three")
    rows = (206, 272, 338)
    dy = dg.oval(110, 272, OVAL_W, OVAL_H, ["dY", f"[B, O] = [{b}, {o}]"], "plain")
    specs = [
        (["dX = dY W", "[B, O] x [O, D] = [B, D]", "reads the weight W"], ["dX", f"[B, D] = [{b}, {d}]"], "result"),
        (["dW = dY^T X", "[O, B] x [B, D] = [O, D]", "reads the input X"], ["dW", f"[O, D] = [{o}, {d}]"], "result"),
        (["db = sum over b of dY", "[B, O] summed over B = [O]", "reads nothing"], ["db", f"[O] = [{o}]"], "result"),
    ]
    for cy_row, (box_lines, out_lines, kind) in zip(rows, specs):
        op = dg.box(430, cy_row, BOX_W, BOX_H, box_lines, "accent")
        res = dg.oval(750, cy_row, OVAL_W, OVAL_H, out_lines, kind)
        if cy_row == dy["l"][1]:
            dg.arrow(dy["r"], op["l"], color=RED, dash="5 3")
        else:
            dg.path([dy["r"], (250, dy["r"][1]), (250, cy_row), op["l"]], color=RED, dash="5 3")
        dg.arrow(op["r"], res["l"], color=RED, dash="5 3")
    return dg.svg()
