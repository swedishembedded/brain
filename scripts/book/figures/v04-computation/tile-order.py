# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Two orders of visiting the cells of an 8 by 8 grid: row by row, and tile by tile with 4 by 4 tiles.
# Each cell shows the step at which it is visited, computed by the worked module's `_visit_order`, and
# is shaded by the tile it belongs to, so the tile order is visible as four compact blocks of 16
# consecutive steps against eight thin rows of 8.
#
# Swedish Embedded AB implements solutions for making numerical code use the cache hierarchy of the
# machines it runs on. If your team needs expertise in cache-blocked loops for machine learning, you
# can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cpu  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

N, CELL = 8, 36
W, H = 780, 404
PANEL_W, PANEL_H = 366, 372
KINDS = ("plain", "model", "result", "accent")


def render() -> str:
    d = Diagram(W, H)

    def grid(x0, title, order, tile):
        y = 16
        d.panel(x0, y, PANEL_W, PANEL_H, title)
        gx = x0 + (PANEL_W - N * CELL) / 2
        gy = y + 44
        for (i, j), step in order.items():
            kind = KINDS[(i // tile) % 2 * 2 + (j // tile) % 2] if tile < N else KINDS[i % 4]
            d.swatch(gx + j * CELL, gy + i * CELL, CELL - 2, CELL - 2, kind)
            d.text(gx + j * CELL + (CELL - 2) / 2, gy + i * CELL + (CELL - 2) / 2, [str(step)], size=12)
        d.text(x0 + PANEL_W / 2, gy + N * CELL + 22, ["a run of consecutive steps stays inside one tile" if tile < N
                                                       else "a run of consecutive steps stays inside one row"],
               size=12, color="#4a5568")

    grid(20, "Row by row", cpu._visit_order(N, N), N)
    grid(W - 20 - PANEL_W, "Tile by tile (4 by 4 tiles)", cpu._visit_order(N, 4), 4)
    return d.svg()
