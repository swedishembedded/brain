# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Four rows of the packed training stream of the chapter, each cell shaded by the document its token belongs to, and the two masks
# that attention may use on row 1: the causal mask of the row (a position reads every earlier position, whichever document it
# belongs to) and the block-diagonal mask (a position reads only the earlier positions of its own document). Tokens, document ids
# and allowed pairs come from `training_gpt.data` and `training_gpt.same_document`, the functions the chapter's tables use.
#
# Swedish Embedded AB implements solutions for packing and masking training data for language models. If your team needs expertise in
# sequence packing and attention masks, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_gpt as tg  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

W, H = 1000, 620
SHADES = ("#ebf4ff", "#fefcbf", "#e2e8f0")   # a document's cells; neighbouring documents differ
CELL_W, CELL_H = 52, 28
X0, Y0 = 118, 62
MASK = 17            # one cell of a mask grid
MASK_Y = 262


def cell(d, x, y, w, h, fill, label="", size=11, stroke=FRAME):
    d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
    if label:
        d.items.append(f'<text x="{fmt(x + w / 2)}" y="{fmt(y + h / 2 + size * 0.35)}" {FONT} font-size="{size}" fill="{INK}" '
                       f'text-anchor="middle">{escape(label)}</text>')


def mask_grid(d, x, y, allowed, title, boundary, count):
    n = len(allowed)
    for i in range(n):
        for j in range(n):
            on = allowed[i][j]
            cell(d, x + j * MASK, y + i * MASK, MASK, MASK, "#4a5568" if on else "#ffffff", stroke="#cbd5e0")
    bx = x + boundary * MASK
    by = y + boundary * MASK
    d.items.append(f'<line x1="{fmt(bx)}" y1="{fmt(y - 4)}" x2="{fmt(bx)}" y2="{fmt(y + n * MASK + 4)}" stroke="{tg_red()}" stroke-width="1.4" stroke-dasharray="4 3"/>')
    d.items.append(f'<line x1="{fmt(x - 4)}" y1="{fmt(by)}" x2="{fmt(x + n * MASK + 4)}" y2="{fmt(by)}" stroke="{tg_red()}" stroke-width="1.4" stroke-dasharray="4 3"/>')
    d.text(x + n * MASK / 2, y - 22, [title], size=13, color=INK, weight="bold")
    d.text(x + n * MASK / 2, y + n * MASK + 24, [f"{count} allowed pairs of {n * n}"], size=12, color=MUTED)
    d.text(x - 10, y + n * MASK / 2, ["query i"], size=12, color=MUTED, anchor="end")
    d.text(x + n * MASK / 2, y + n * MASK + 42, ["key j"], size=12, color=MUTED)


def tg_red():
    return "#c53030"


def render() -> str:
    tk = tg.tokenizer()
    rows = tg.data()["train"]["rows"]
    d = Diagram(W, H)
    d.text(W / 2, 22, ["Four packed rows of the training stream; shade = document"], size=14, weight="bold")
    d.text(X0 + 8 * CELL_W, 44, ["input at position i (the target is the next token)"], size=12, color=MUTED)
    for r in range(4):
        y = Y0 + r * (CELL_H + 6)
        d.text(X0 - 12, y + CELL_H / 2, [f"row {r}"], size=12, color=MUTED, anchor="end")
        for i in range(tg.ROW):
            doc = rows[r]["doc"][i]
            cell(d, X0 + i * CELL_W, y, CELL_W, CELL_H, SHADES[doc % 3], tk.name(rows[r]["x"][i]))
    r1 = rows[1]
    causal = [[tg.causal(i, j) for j in range(tg.ROW)] for i in range(tg.ROW)]
    allowed = tg.same_document(r1["doc"])
    block = [[allowed(i, j) for j in range(tg.ROW)] for i in range(tg.ROW)]
    boundary = r1["doc"].index(max(r1["doc"]))
    d.text(W / 2, MASK_Y - 52, ["Attention masks for row 1 (dashed: where document 2 begins)"], size=14, weight="bold")
    gx = 150
    mask_grid(d, gx, MASK_Y, causal, "causal mask of the row", boundary, sum(map(sum, causal)))
    mask_grid(d, gx + 16 * MASK + 120, MASK_Y, block, "block-diagonal mask", boundary, sum(map(sum, block)))
    return d.svg()
