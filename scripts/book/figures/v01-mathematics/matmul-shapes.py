# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The shapes in a matrix product [m,k] x [k,n] = [m,n]: the inner size k is
# shared and disappears, the outer sizes survive, and each output entry is one
# dot product of length k. The chapter's cost count (m*n*k multiply-adds) is
# read straight off this picture.
#
# Swedish Embedded AB implements solutions for sizing and costing matrix
# workloads before they run. If your team needs expertise in the cost of
# linear algebra in machine learning, you can procure our services by sending
# an email to info@swedishembedded.com.
from svgdiagram import Diagram, ellipse_y

W, H = 760, 380
CENTRE_X = 370
OPERAND_Y, DOT_Y, RESULT_Y = 56, 192, 322
OPERAND_DX = 150                      # operands sit this far either side of the centre
OPERAND_W, OPERAND_H = 230, 76
DOT_W, DOT_H = 340, 92
RESULT_W, RESULT_H = 230, 76


def render() -> str:
    d = Diagram(W, H)
    a = d.box(CENTRE_X - OPERAND_DX, OPERAND_Y, OPERAND_W, OPERAND_H,
              ["A", "[m, k]", "m rows, each of length k"], "model")
    b = d.box(CENTRE_X + OPERAND_DX, OPERAND_Y, OPERAND_W, OPERAND_H,
              ["B", "[k, n]", "n columns, each of length k"], "model")
    dot = d.oval(CENTRE_X, DOT_Y, DOT_W, DOT_H,
                 ["C[i][j] =", "row i of A · column j of B", "(k multiply-adds)"], "result")
    c = d.box(CENTRE_X, RESULT_Y, RESULT_W, RESULT_H, ["C = A B", "[m, n]", "m*n entries"], "accent")

    # Each operand drops straight onto the dot-product ellipse; the label
    # sits on the inner side, where there is room between the two arrows.
    for operand, side in ((a, "right"), (b, "left")):
        x = operand["b"][0]
        d.arrow(operand["b"], (x, ellipse_y(dot["ell"], "t", x - CENTRE_X)), "k shared", side=side)
    d.arrow(dot["b"], c["t"], "k disappears", side="right")

    note = d.note(CENTRE_X + 270, RESULT_Y, 220, 56, ["cost: m*n entries x k", "= m*n*k multiply-adds"])
    d.line(c["r"], note["l"])
    return d.svg()
