# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The tiles of a causal score matrix at T = 24, drawn for two blockings: 4 query rows against 4 keys, and 8 query rows against 4
# keys. A tile is skipped when every key comes after every row of its block, full when every key is at or before the first row, and
# diagonal when the mask cuts it. Classes and counts come from `flash_attention.tile_classes` and `flash_attention.causal_tile_rows`.
#
# Swedish Embedded AB implements solutions for skipping the masked half of causal attention without losing the diagonal. If your team
# needs expertise in tiled causal attention, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flash_attention as fa  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

W, H = 1000, 500
T = 24
CELL = 44
FILL = {"skipped": "#ffffff", "full": "#9ec5e8", "diagonal": "#fbd38d"}
PANEL_W, PANEL_H, PANEL_Y = 484, 400, 46


def grid(d, x0, bq, bk):
    cls = fa.tile_classes(T, bq, bk)
    nq, nk = len(cls), len(cls[0])
    ch = CELL * bq // 4                       # a block of 8 rows is drawn twice as tall as one of 4
    left = x0 + (PANEL_W - nk * CELL) / 2
    top = PANEL_Y + 62
    d.panel(x0, PANEL_Y, PANEL_W, PANEL_H, f"{bq} query rows x {bk} keys per tile")
    for i, row in enumerate(cls):
        for j, c in enumerate(row):
            x, y = left + j * CELL, top + i * ch
            d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{CELL}" height="{fmt(ch)}" fill="{FILL[c]}" stroke="{FRAME}" stroke-width="1"/>')
    d.text(left + nk * CELL / 2, top - 12, ["key tiles, left to right"], size=12, color=MUTED)
    flat = [c for r in cls for c in r]
    full, diag, skip = flat.count("full"), flat.count("diagonal"), flat.count("skipped")
    rows = fa.causal_tile_rows([(T, bq, bk)])[0]
    d.text(x0 + PANEL_W / 2, top + nq * ch + 26, [f"{len(flat)} tiles: {full} full, {diag} diagonal, {skip} skipped"], size=13, weight="bold")
    d.text(x0 + PANEL_W / 2, top + nq * ch + 46, [f"scores computed: {rows[6]} of T^2 (the mask allows {(T + 1) / (2 * T):.4f})"], size=12, color=MUTED)


def render() -> str:
    d = Diagram(W, H)
    d.text(W / 2, 24, [f"Tiles of a causal score matrix, T = {T}: query blocks down the side, key tiles across"], size=14, weight="bold")
    grid(d, 12, 4, 4)
    grid(d, 504, 8, 4)
    y = H - 24
    for k, (name, label) in enumerate((("full", "full: computed without a mask"), ("diagonal", "diagonal: computed, keys past the row masked"),
                                       ("skipped", "skipped: never loaded"))):
        x = 60 + k * 300
        d.items.append(f'<rect x="{x}" y="{y - 12}" width="22" height="14" rx="3" fill="{FILL[name]}" stroke="{MUTED}" stroke-width="1.2"/>')
        d.text(x + 30, y - 5, [label], size=12, color=MUTED, anchor="start")
    return d.svg()
