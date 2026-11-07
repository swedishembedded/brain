# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The computational graph of the chapter's two-layer network. Black arrows are
# the forward pass, red dashed arrows are the backward pass; each red arrow is
# labelled with the gradient it carries. Yellow values are the ones the
# backward pass reads again, so the forward pass must save them.
#
# The forward pass is one spine running down the middle (X, linear, Z1, tanh,
# A, linear, Y, subtract, E, mean of squares, L). Each operation's own
# parameters sit beside it: weights on the left, biases and the targets on
# the right. A backward arrow runs next to the forward arrow it reverses, from
# the operation to the value it differentiates, with its formula to the right;
# the gradients of the parameters leave sideways under the forward arrows.
#
# Swedish Embedded AB implements solutions for explaining and verifying
# gradient computation in neural network training. If your team needs
# expertise in backpropagation, you can procure our services by sending an
# email to info@swedishembedded.com.
from svgdiagram import RED, Diagram, ellipse_x, ellipse_y

W = 780
SPINE_X, LEFT_X, RIGHT_X = 390, 110, 670
TOP, GAP = 40, 42
VALUE_W, VALUE_H, OP_W, OP_H = 92, 40, 176, 46
BACK_DX = 24                    # backward arrows run this far right of the spine
LANE = 9                        # forward / backward arrows to a parameter sit this far above / below the row

# (name, lines, kind); values are ovals ("saved" ones highlighted), operations boxes.
SPINE = [
    ("X", ["X", "[B,D]"], "saved"),
    ("lin1", ["linear", "Z1 = X W1^T + b1"], "op"),
    ("Z1", ["Z1", "[B,H]"], "value"),
    ("act", ["tanh"], "op"),
    ("A", ["A", "[B,H]"], "saved"),
    ("lin2", ["linear", "Y = A W2^T + b2"], "op"),
    ("Y", ["Y", "[B,O]"], "value"),
    ("sub", ["subtract", "E = Y − T"], "op"),
    ("E", ["E", "[B,O]"], "saved"),
    ("mse", ["mean of squares", "L = ΣE² / (B·O)"], "op"),
    ("L", ["L", "scalar"], "value"),
]
# parameters: operation -> (side, name, lines, kind)
PARAMS = [
    ("lin1", "left", "W1", ["W1", "[H,D]"], "value"),
    ("lin1", "right", "b1", ["b1", "[H]"], "value"),
    ("lin2", "left", "W2", ["W2", "[O,H]"], "saved"),
    ("lin2", "right", "b2", ["b2", "[O]"], "value"),
    ("sub", "right", "T", ["T", "[B,O]"], "value"),
]
# backward arrows along the spine: operation -> the value it sends a gradient to
BACK_SPINE = {
    "mse": ("E", "dE = 2E / (B·O)"),
    "sub": ("Y", "dY = dE"),
    "lin2": ("A", "dA = dY W2"),
    "act": ("Z1", "dZ1 = dA ⊙ (1 − A²)"),
}
# backward arrows to parameters: (operation, parameter) -> label
BACK_PARAM = {
    ("lin2", "W2"): "dW2 = dY^T A",
    ("lin2", "b2"): "db2 = Σ_b dY",
    ("lin1", "W1"): "dW1 = dZ1^T X",
    ("lin1", "b1"): "db1 = Σ_b dZ1",
}
KIND = {"saved": "result", "value": "plain", "op": "model"}


def render() -> str:
    ys, y = {}, TOP
    for i, (name, _, kind) in enumerate(SPINE):
        h = OP_H if kind == "op" else VALUE_H
        if i:
            y += GAP
        ys[name] = y + h / 2
        y += h
    d = Diagram(W, int(y + TOP))

    node = {}
    for name, lines, kind in SPINE:
        node[name] = (d.box(SPINE_X, ys[name], OP_W, OP_H, lines, "model") if kind == "op"
                      else d.oval(SPINE_X, ys[name], VALUE_W, VALUE_H, lines, KIND[kind]))
    for op, side, name, lines, kind in PARAMS:
        x = LEFT_X if side == "left" else RIGHT_X
        node[name] = d.oval(x, ys[op], VALUE_W, VALUE_H, lines, KIND[kind])

    # Forward pass: down the spine, then each parameter into its operation.
    for (a, _, _), (b, _, _) in zip(SPINE, SPINE[1:]):
        d.arrow(node[a]["b"], node[b]["t"])
    for op, side, name, _, _ in PARAMS:
        ell, row = node[name]["ell"], ys[op]
        s, e = ("r", "l") if side == "left" else ("l", "r")
        d.arrow((ellipse_x(ell, s, -LANE), row - LANE), (node[op][e][0], row - LANE))

    # Backward pass: dashed red, next to the forward arrow it reverses.
    for op, (value, label) in BACK_SPINE.items():
        x = SPINE_X + BACK_DX
        d.arrow((x, node[op]["t"][1]), (x, ellipse_y(node[value]["ell"], "b", BACK_DX)), label,
                color=RED, dash="5 3", side="right")
    for (op, name), label in BACK_PARAM.items():
        side = next(s for o, s, n, _, _ in PARAMS if n == name)
        row = ys[op]
        s, e = ("r", "l") if side == "left" else ("l", "r")
        d.arrow((node[op][e][0], row + LANE), (ellipse_x(node[name]["ell"], s, LANE), row + LANE), label,
                color=RED, dash="5 3", side="below")

    # Legend, in the free corner beside the first value.
    lx, ly = 548, TOP - 10
    d.arrow((lx, ly), (lx + 34, ly), gap=0)
    d.text(lx + 44, ly, ["forward"], size=12, color="#4a5568", anchor="start")
    d.arrow((lx, ly + 22), (lx + 34, ly + 22), gap=0, color=RED, dash="5 3")
    d.text(lx + 44, ly + 22, ["gradient sent back"], size=12, color=RED, anchor="start")
    d.swatch(lx, ly + 37, 34, 14, "result")
    d.text(lx + 44, ly + 44, ["saved for the backward pass"], size=12, color="#4a5568", anchor="start")
    return d.svg()
