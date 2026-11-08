# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The attention weights of the four-token example of the chapter as two grids: without a mask every position looks at all four,
# with the causal mask the cells above the diagonal are zero and each row still adds to 1. Numbers from
# `self_attention.attention_data`, the same as the tables of the chapter.
#
# Swedish Embedded AB implements solutions for models that read long sequences. If your team needs expertise in attention
# masks and their gradients, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import self_attention as sa  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402
from svgplot import fmt  # noqa: E402

CELL_W, CELL_H = 66, 42
PANEL_W, PANEL_H = 440, 300
GRID_X, GRID_Y = 118, 112          # top-left of the grid inside a panel


def _shade(w):
    """White at 0 to the book's blue at 1."""
    lo, hi = (255, 255, 255), (43, 108, 176)
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * w) for a, b in zip(lo, hi))


def _grid(d, left, title, tokens, weights, masked):
    d.panel(left, 10, PANEL_W, PANEL_H, title)
    gx, gy = left + GRID_X - 70, 10 + GRID_Y - 20
    n = len(tokens)
    d.text(gx + 48 + n * CELL_W / 2, 10 + 52, ["key position j (what is looked at)"], size=12, color=MUTED)
    for j in range(n):
        d.text(gx + 48 + (j + 0.5) * CELL_W, gy - 2, [f"{j + 1} {tokens[j]}"], size=12)
    for i in range(n):
        cy = gy + 12 + (i + 0.5) * CELL_H
        d.text(gx + 40, cy, [f"{i + 1} {tokens[i]}"], size=12, anchor="end")
        for j in range(n):
            x, y, w = gx + 48 + j * CELL_W, gy + 12 + i * CELL_H, weights[i][j]
            hidden = masked and j > i
            fill = "#edf2f7" if hidden else _shade(w)
            d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{CELL_W}" height="{CELL_H}" fill="{fill}" '
                           f'stroke="#ffffff" stroke-width="1"/>')
            label = "-inf" if hidden else f"{w:.2f}"
            d.text(x + CELL_W / 2, cy, [label], size=12, color="#718096" if hidden else ("#ffffff" if w > 0.45 else INK))
    d.text(gx + 6, gy + 12 + n * CELL_H / 2, ["query i"], size=12, color=MUTED, anchor="end")


def render() -> str:
    tokens, full, causal = sa.attention_data()
    d = Diagram(2 * PANEL_W + 40, PANEL_H + 56)
    _grid(d, 10, "No mask: every position sees all four", tokens, full, False)
    _grid(d, 30 + PANEL_W, "Causal mask: position i sees j <= i", tokens, causal, True)
    d.text(10 + PANEL_W + 10, PANEL_H + 36,
           ["darker = larger weight; each row adds to 1; -inf marks a score replaced before the softmax"], size=12, color=MUTED)
    return d.svg()
