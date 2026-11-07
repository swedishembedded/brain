# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The forward and backward pass of a broadcast, side by side on the same
# cells. Left: the bias b [O] is copied to every one of the B rows of the
# result. Right: the gradient dY [B, O] arriving at the result is summed down
# every column to give db [O]. The numbers are those of the chapter's worked
# example (read from the worked module), so the picture can be checked
# against the table beside it.
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

CELL_W, CELL_H, PITCH_X, PITCH_Y = 56, 34, 72, 42
PANEL_W, PANEL_H, TOP = 320, 296, 16
W, H = 2 * PANEL_W + 3 * 20, PANEL_H + 2 * TOP


def num(v):
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def render() -> str:
    b, _, o = lg.LIN_SIZES
    d = Diagram(W, H)
    left, right = 20, 20 + PANEL_W + 20
    d.panel(left, TOP, PANEL_W, PANEL_H, "forward: copy b to every row")
    d.panel(right, TOP, PANEL_W, PANEL_H, "backward: sum dY down every column")

    def col_x(panel_x, j):
        return panel_x + (PANEL_W - ((o - 1) * PITCH_X + CELL_W)) / 2 + CELL_W / 2 + j * PITCH_X

    bias_y = 92
    grid_y = [170 + i * PITCH_Y for i in range(b)]
    for j in range(o):
        x = col_x(left, j)
        top = d.box(x, bias_y, CELL_W, CELL_H, [num(lg.LIN_B[j])], "result")
        for y in grid_y:
            d.box(x, y, CELL_W, CELL_H, [num(lg.LIN_B[j])], "plain")
        # The whole column receives the copy, so the arrow ends on its outline.
        col = d.group(x - CELL_W / 2 - 4, grid_y[0] - CELL_H / 2 - 4, CELL_W + 8,
                      grid_y[-1] - grid_y[0] + CELL_H + 8)
        d.arrow(top["b"], col["t"], "copy" if j == 0 else None, side="right")
    d.text(left + 14, 62, ["b [O]"], size=12, color="#4a5568", anchor="start")
    d.text(left + 14, TOP + PANEL_H - 18, [f"the same value in all {b} rows of the result"], size=12,
           color="#4a5568", anchor="start")

    g_y = [bias_y + i * PITCH_Y for i in range(b)]
    sum_y = grid_y[-1]
    sums = lg.unbroadcast(lg.LIN_G, (o,))
    for j in range(o):
        x = col_x(right, j)
        for i, y in enumerate(g_y):
            d.box(x, y, CELL_W, CELL_H, [num(lg.LIN_G[i][j])], "plain")
        # Every entry of the column is summed, so the arrow leaves its outline.
        col = d.group(x - CELL_W / 2 - 4, g_y[0] - CELL_H / 2 - 4, CELL_W + 8,
                      g_y[-1] - g_y[0] + CELL_H + 8, color=RED)
        total = d.box(x, sum_y, CELL_W, CELL_H, [num(sums[j])], "result")
        d.arrow(col["b"], total["t"], "sum" if j == 0 else None, color=RED, dash="5 3", side="right")
    d.text(right + 14, 62, ["dY [B, O]"], size=12, color="#4a5568", anchor="start")
    d.text(right + 14, TOP + PANEL_H - 18, [f"db [O]: each entry is the sum of {b} entries of dY"], size=12, color="#4a5568", anchor="start")
    return d.svg()
