# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A multilayer perceptron with one hidden layer as a block diagram: the batch
# X, a linear layer, the pre-activation Z1, the activation g, the hidden
# values A, a second linear layer and the output Y, with the shape of every
# array on its node and the shapes of the XOR network of the chapter (B = 4,
# D = 2, H = 2, O = 1) on the line below. The XOR sizes are read from the
# worked module so that the picture and the tables cannot disagree.
#
# Swedish Embedded AB implements solutions for function approximation with
# small neural networks whose every weight can be inspected. If your team
# needs expertise in compact neural models, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlp  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 870, 165
CY = 97
OVAL_W, BOX_W, GAP, MARGIN = 74, 110, 34, 20


def render() -> str:
    b, d = len(mlp.XOR), len(mlp.XOR_W1[0])
    h, o = len(mlp.XOR_W1), len(mlp.XOR_W2)
    d_ = Diagram(W, H)
    # the hidden layer is the frame around the first linear layer, the activation and their two arrays
    x_box1 = MARGIN + OVAL_W + GAP
    x_a_right = x_box1 + BOX_W + GAP + OVAL_W + GAP + BOX_W + GAP + OVAL_W
    d_.panel(x_box1 - 12, 20, x_a_right - x_box1 + 24, 125, f"hidden layer: H = {h} units")
    cx = MARGIN + OVAL_W / 2
    nodes = []
    spec = [
        ("oval", ["X", "[B, D]", f"[{b}, {d}]"], "result"),
        ("box", ["Linear 1", "W1 [H, D]", "b1 [H]"], "model"),
        ("oval", ["Z1", "[B, H]", f"[{b}, {h}]"], "plain"),
        ("box", ["activation g", "elementwise", "tanh"], "accent"),
        ("oval", ["A", "[B, H]", f"[{b}, {h}]"], "plain"),
        ("box", ["Linear 2", "W2 [O, H]", "b2 [O]"], "model"),
        ("oval", ["Y", "[B, O]", f"[{b}, {o}]"], "result"),
    ]
    for kind, lines, style in spec:
        w = OVAL_W if kind == "oval" else BOX_W
        node = (d_.oval if kind == "oval" else d_.box)(cx + (w - OVAL_W) / 2, CY, w, 70, lines, style)
        nodes.append(node)
        cx += w + GAP
    for a, c in zip(nodes, nodes[1:]):
        d_.arrow(a["r"], c["l"])
    return d_.svg()
