# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Which keys each query reads under four sparse patterns on 16 positions: a window, a window with a stride, blocks of 2 with a global
# first block, and the 4 keys of largest true weight on a small toy with 4 topics. Row i is the query at position i, column j the
# key at position j; a yellow cell is a key the query reads, a white cell one it skips, an empty cell the future. The masks are
# `sparse_attention.mask_data`, the function behind the worked examples of the chapter.
#
# Swedish Embedded AB implements solutions for serving long-context language models whose attention cost is set by what a token
# needs. If your team needs expertise in sparse attention, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sparse_attention as sa  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

CELL = 15
PANEL_W, PANEL_H = 330, 350
TOP = 66  # distance from the panel top to the first row of the grid


def render() -> str:
    n, masks = sa.mask_data()
    d = Diagram(2 * PANEL_W + 30, 2 * PANEL_H + 30)
    for idx, (title, mask) in enumerate(masks.items()):
        px, py = 10 + (idx % 2) * (PANEL_W + 10), 10 + (idx // 2) * (PANEL_H + 10)
        d.panel(px, py, PANEL_W, PANEL_H, title)
        gx, gy = px + (PANEL_W - n * CELL) / 2 + 8, py + TOP
        d.text(gx + n * CELL / 2, py + 40, ["key position j"], size=11, color=MUTED)
        reads = sum(len(row) for row in mask)
        d.text(gx + n * CELL / 2, gy + n * CELL + 22, [f"{reads} of {n * (n + 1) // 2} causal cells read"], size=12, color=MUTED)
        for i in range(n):
            if i % 3 == 0:
                d.text(gx - 7, gy + i * CELL + CELL / 2, [str(i)], size=10, color=MUTED, anchor="end")
                d.text(gx + i * CELL + CELL / 2, gy - 8, [str(i)], size=10, color=MUTED)
            for j in range(i + 1):
                d.swatch(gx + j * CELL + 1, gy + i * CELL + 1, CELL - 2, CELL - 2, "result" if j in mask[i] else "plain")
    return d.svg()
