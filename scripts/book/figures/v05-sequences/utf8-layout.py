# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The four UTF-8 sequence lengths as rows of bit cells on a common grid.

The patterns come from `text.layout_rows` and the ranges from the encoder's own arithmetic (payload bits n
give the code points below 2^n, the first of each length is the bound of the shorter one), so the picture and
the chapter's table of capacities cannot disagree.

Swedish Embedded AB implements solutions for getting text into models without silent corruption. If your
team needs expertise in text encodings and tokenizer pipelines, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import text  # noqa: E402
from svgdiagram import FILL, MUTED, Diagram  # noqa: E402
from svgplot import fmt  # noqa: E402

CELL, CELL_H, BYTE_GAP = 19, 26, 9
LABEL_W, LEFT, TOP, ROW_H = 150, 12, 54, 52
PAYLOAD = [7, 11, 16, 21]
FIRST = [0, 0x80, 0x800, 0x10000]


def render() -> str:
    rows = text.layout_rows()
    byte_w = 8 * CELL
    width = LEFT + LABEL_W + 4 * byte_w + 3 * BYTE_GAP + 14
    d = Diagram(width, TOP + len(rows) * ROW_H + 6)
    d.swatch(LEFT + LABEL_W, 14, 22, 16, "result")
    d.text(LEFT + LABEL_W + 30, 22, ["prefix bits (fixed, 0 or 1)"], anchor="start", size=12)
    d.swatch(LEFT + LABEL_W + 230, 14, 22, 16, "model")
    d.text(LEFT + LABEL_W + 260, 22, ["payload bits (x, the code point)"], anchor="start", size=12)
    for r, bytes_ in enumerate(rows):
        cy = TOP + r * ROW_H + ROW_H / 2 - 4
        n = r + 1
        last = min((1 << PAYLOAD[r]) - 1, text.MAX_CODE_POINT)
        d.text(LEFT, cy - 8, [f"{n} byte{'s' if n > 1 else ''}, {PAYLOAD[r]} payload bits"], anchor="start", size=12, weight="bold")
        d.text(LEFT, cy + 10, [f"{text.cp_hex(FIRST[r])} to {text.cp_hex(last)}"], anchor="start", size=12, color=MUTED)
        x = LEFT + LABEL_W
        for bits in bytes_:
            for ch in bits:
                kind = "model" if ch == "x" else "result"
                d.items.append(f'<rect x="{fmt(x)}" y="{fmt(cy - CELL_H / 2)}" width="{CELL - 2}" height="{CELL_H}" rx="2" '
                               f'fill="{FILL[kind]}" stroke="{MUTED}" stroke-width="1"/>')
                d.text(x + (CELL - 2) / 2, cy, [ch], size=12)
                x += CELL
            x += BYTE_GAP
    return d.svg()
