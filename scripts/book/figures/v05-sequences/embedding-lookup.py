# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The same three rows obtained two ways from the six-row table of the chapter: a lookup, in which each id picks one
# row, and a product of a one-hot matrix with the whole table. The numbers are those of `embeddings.lookup`.
#
# Swedish Embedded AB implements solutions for fast token tables in language models. If your team needs expertise in
# embedding lookups and the memory traffic they cost, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import embeddings  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 780, 560
ROW_H = 28


def _vals(row):
    return "  ".join(f"{x:5.2f}" for x in row)


def render() -> str:
    data = embeddings.lookup_figure_data()
    toks, table, ids, picked = data["tokens"], data["table"], data["ids"], data["rows"]
    V, T = len(table), len(ids)
    d = Diagram(W, H)

    # top panel: lookup
    d.panel(10, 10, W - 20, 276, "Lookup: each id picks one row of the table")
    top0 = 92
    row_y = [top0 + ROW_H * i for i in range(V)]
    tx, tw = 330, 250
    for i in range(V):
        sel = i in ids
        d.box(tx, row_y[i], tw, ROW_H - 4, [f"{i}  {toks[i]:<4}  {_vals(table[i])}"], "result" if sel else "plain")
    for t, i in enumerate(ids):
        e = d.box(88, row_y[i], 120, ROW_H - 4, [f"t = {t}, id {i}"], "accent")
        d.arrow(e["r"], (tx - tw / 2, row_y[i]), gap=2)
        o = d.box(690, row_y[i], 150, ROW_H - 4, [_vals(picked[t])], "accent")
        d.arrow((tx + tw / 2, row_y[i]), o["l"], gap=2)
    d.text(88, top0 - 30, ["ids"], size=12, color="#4a5568")
    d.text(tx, top0 - 30, ["table (6 x 3)"], size=12, color="#4a5568")
    d.text(690, top0 - 30, ["output X (3 x 3)"], size=12, color="#4a5568")
    d.text(W / 2, 264, ["reads 3 rows of 3 numbers = 9 numbers; multiplies nothing"], size=12, color="#2d3748")

    # bottom panel: one-hot product
    y0 = 300
    d.panel(10, y0, W - 20, 244, "The same rows as a product: one-hot matrix N times the whole table")
    cell, gap = 24, 3
    nx0 = 60
    ny = [y0 + 84 + 34 * t for t in range(T)]
    d.text(nx0 + 3 * (cell + gap) + 12, y0 + 48, ["N (3 x 6)"], size=12, color="#4a5568")
    for t, i in enumerate(ids):
        for j in range(V):
            d.box(nx0 + j * (cell + gap) + cell / 2, ny[t], cell, cell, ["1" if j == i else "0"], "result" if j == i else "plain")
    mid = nx0 + V * (cell + gap) + 36
    d.text(mid, y0 + 118, ["x"], size=20)
    bx, bw, bh = mid + 170, 270, 150
    d.box(bx, y0 + 118, bw, bh, ["the whole table", "6 x 3 numbers, all read"], "model")
    eq = bx + bw / 2 + 30
    d.text(eq, y0 + 118, ["="], size=20)
    d.box(eq + 100, y0 + 118, 150, bh - 30, ["X (3 x 3)", "the same rows"], "accent")
    d.text(W / 2, y0 + 224, ["T V D = 3 * 6 * 3 = 54 multiply-adds, 45 of them by zero"], size=12, color="#2d3748")
    return d.svg()
