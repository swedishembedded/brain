# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Which values each normalization averages together, for an image tensor
# [B, C, H, W]. The tensor is drawn as a grid of B = 4 examples (rows) by C = 4
# channels (columns); every cell stands for the whole H x W plane of one channel
# of one example. In each panel the shaded cells are ONE group: the cells whose
# mean and variance are computed together. The captions give the number of
# groups and the values per group, with the same counts as the worked table
# `normalization.norm_axes`.
#
# Swedish Embedded AB implements solutions for stable training of vision
# models. If your team needs expertise in normalization layers for
# convolutional networks, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

B, C = 4, 4
PITCH_X, PITCH_Y, CELL_W, CELL_H = 40, 32, 34, 26
PANEL_W, PANEL_H, GAP, TOP = 196, 252, 14, 16
W = 4 * PANEL_W + 5 * GAP
H = PANEL_H + 2 * TOP

# (title, cells in the shown group as (row, column) pairs, other group, caption lines)
def panels():
    col1 = [(r, 1) for r in range(B)]
    row1 = [(1, c) for c in range(C)]
    pair = [(1, 0), (1, 1)]
    other_pair = [(1, 2), (1, 3)]
    one = [(1, 1)]
    return [
        ("batch norm", col1, [], ["one group per channel:", "C groups of B*H*W"]),
        ("layer norm", row1, [], ["one group per example:", "B groups of C*H*W"]),
        ("group norm, G = 2", pair, other_pair, ["B*G groups of", "(C/G)*H*W values"]),
        ("instance norm", one, [], ["one group per cell:", "B*C groups of H*W"]),
    ]


def render() -> str:
    d = Diagram(W, H)
    for k, (title, group, other, caption) in enumerate(panels()):
        x = GAP + k * (PANEL_W + GAP)
        d.panel(x, TOP, PANEL_W, PANEL_H, title)
        grid_w = (C - 1) * PITCH_X + CELL_W
        x0 = x + (PANEL_W - grid_w) / 2 + CELL_W / 2 + 8
        y0 = TOP + 78
        for c in range(C):
            d.text(x0 + c * PITCH_X, y0 - 30, [f"c{c}"], size=11, color="#4a5568")
        for r in range(B):
            d.text(x0 - 24, y0 + r * PITCH_Y, [f"n{r}"], size=11, color="#4a5568", anchor="end")
            for c in range(C):
                kind = "result" if (r, c) in group else ("model" if (r, c) in other else "plain")
                d.box(x0 + c * PITCH_X, y0 + r * PITCH_Y, CELL_W, CELL_H, [""], kind)
        d.text(x + PANEL_W / 2, TOP + PANEL_H - 40, caption, size=12, color="#4a5568")
    return d.svg()
