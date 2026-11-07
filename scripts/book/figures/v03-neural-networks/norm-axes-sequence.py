# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Which values each normalization averages together, for a sequence tensor
# [B, T, D]. The tensor is drawn as its B*T rows (B = 2 sequences of T = 3
# positions) by D = 5 features; a cell is one number. In each panel the shaded
# cells are ONE group: the numbers whose mean and variance (RMS norm: whose mean
# square) are computed together. Counts match `normalization.norm_axes`.
#
# Swedish Embedded AB implements solutions for stable training of sequence
# models. If your team needs expertise in normalization layers for
# transformers, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

B, T, D = 2, 3, 5
ROWS = B * T
PITCH_X, PITCH_Y, CELL_W, CELL_H = 40, 30, 34, 24
PANEL_W, PANEL_H, GAP, TOP = 300, 296, 16, 16
W = 3 * PANEL_W + 4 * GAP
H = PANEL_H + 2 * TOP


def panels():
    col = [(r, 2) for r in range(ROWS)]
    row = [(4, c) for c in range(D)]
    return [
        ("batch norm", col, ["one group per feature:", "D groups of B*T values"]),
        ("layer norm", row, ["one group per position:", "B*T groups of D values,", "mean and variance"]),
        ("RMS norm", row, ["one group per position:", "B*T groups of D values,", "mean square only"]),
    ]


def render() -> str:
    d = Diagram(W, H)
    for k, (title, group, caption) in enumerate(panels()):
        x = GAP + k * (PANEL_W + GAP)
        d.panel(x, TOP, PANEL_W, PANEL_H, title)
        grid_w = (D - 1) * PITCH_X + CELL_W
        x0 = x + (PANEL_W - grid_w) / 2 + CELL_W / 2 + 16
        y0 = TOP + 72
        for c in range(D):
            d.text(x0 + c * PITCH_X, y0 - 28, [f"d{c}"], size=11, color="#4a5568")
        for r in range(ROWS):
            seq, pos = divmod(r, T)
            d.text(x0 - 24, y0 + r * PITCH_Y, [f"b{seq} t{pos}"], size=11, color="#4a5568", anchor="end")
            for c in range(D):
                d.box(x0 + c * PITCH_X, y0 + r * PITCH_Y, CELL_W, CELL_H, [""], "result" if (r, c) in group else "plain")
        d.text(x + PANEL_W / 2, TOP + PANEL_H - 36, caption, size=12, color="#4a5568")
    return d.svg()
