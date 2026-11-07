# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The shapes in the chain rule for a composition x -> u -> h -> L. Each arrow
# carries the Jacobian of its step; the Jacobian of the whole map is their
# product, and the inner sizes cancel exactly as in a matrix product. The
# chain runs down the left, each Jacobian beside its arrow, and the note that
# multiplies them (forward) and transposes them (backward) sits beside it.
#
# Swedish Embedded AB implements solutions for getting tensor shapes right
# before a backward pass is written. If your team needs expertise in
# derivatives of composed models, you can procure our services by sending an
# email to info@swedishembedded.com.
from svgdiagram import Diagram

W, TOP, PITCH, BOX_W, BOX_H = 760, 44, 112, 190, 54
CHAIN_X = 150
NOTE_X, NOTE_W, NOTE_H = 585, 310, 112

NODES = [
    (["x", "[2]"], "plain"),
    (["u = W x", "[3]"], "model"),
    (["h = tanh(u)", "[3]"], "model"),
    (["L = 1/2 |h - t|^2", "[1]"], "result"),
]
JACOBIANS = [
    ["J1 = W", "[3, 2]"],
    ["J2 = diag(1 - h^2)", "[3, 3]"],
    ["J3 = (h - t)^T", "[1, 3]"],
]
NOTE = [
    "dL/dx = J3 J2 J1",
    "[1,3] [3,3] [3,2] = [1,2]",
    "",
    "backward: grad_x = J1^T J2^T grad_h",
    "[2,3] [3,3] [3] = [2]",
]


def render() -> str:
    d = Diagram(W, 2 * TOP + (len(NODES) - 1) * PITCH)
    boxes = [d.box(CHAIN_X, TOP + i * PITCH, BOX_W, BOX_H, lines, kind) for i, (lines, kind) in enumerate(NODES)]
    for a, b, label in zip(boxes, boxes[1:], JACOBIANS):
        d.arrow(a["b"], b["t"], label, side="right")
    d.note(NOTE_X, (boxes[0]["t"][1] + boxes[-1]["b"][1]) / 2, NOTE_W, NOTE_H, NOTE, "plain")
    return d.svg()
