# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The order of the rows of the image block (`ocr.toy_plan`): the tokens of a mosaic of two tiles read row by row across both tiles,
# a learned newline after every mosaic row, then the global view row by row, then the separator.
#
# Swedish Embedded AB implements solutions for document-reading models whose image rows stand where the checkpoint expects them. If
# your team needs expertise in the layout of vision tokens for OCR, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ocr  # noqa: E402
from svgdiagram import FILL, INK, MUTED, Diagram  # noqa: E402
from svgplot import fmt  # noqa: E402

CELL = 38
W, H = 960, 330


def _cell(d, x, y, kind, lines):
    fill = {"tile0": "#bee3f8", "tile1": "#c6f6d5", "global": "#fefcbf", "N": "#e2e8f0", "S": "#fed7d7"}[kind]
    d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{CELL}" height="{CELL}" rx="4" fill="{fill}" stroke="{MUTED}" stroke-width="1"/>')
    d.text(x + CELL / 2, y + CELL / 2, lines, size=12)


def render() -> str:
    plan = ocr.toy_plan()
    d = Diagram(W, H)
    per_line = 12
    d.text(W / 2, 22, ["image block of the toy page: 2 tiles of 2 x 2 tokens, then a global view of 3 x 3 tokens"], size=13, weight="bold")
    x0 = (W - per_line * (CELL + 6)) / 2 + 3
    for i, e in enumerate(plan):
        line, col = divmod(i, per_line)
        x, y = x0 + col * (CELL + 6), 50 + line * (CELL + 28)
        if isinstance(e, tuple):
            _, view, token = e
            kind = "global" if view == 0 else f"tile{view - 1}"
            lab = [f"P{ocr.projector_index(e, 3, 2)}"]
        else:
            kind = e
            lab = ["nl" if e == ocr.NEWLINE else "sep"]
        _cell(d, x, y, kind, lab)
        d.text(x + CELL / 2, y + CELL + 12, [str(i)], size=11, color=MUTED)
    ly = 190
    for k, (kind, text) in enumerate([("tile0", "tile 0 token"), ("tile1", "tile 1 token"), ("global", "global view token"),
                                      ("N", "learned newline"), ("S", "learned separator")]):
        x = 60 + k * 180
        _cell(d, x, ly, kind, [""])
        d.text(x + CELL + 8, ly + CELL / 2, [text], size=12, anchor="start")
    d.text(W / 2, 262, ["P<k> is projector row k (the encoder numbers the global view first, then the tiles); the small number below a cell is its row of the block.",
                        "Rows 0 to 3 read tile 0 and tile 1 side by side along the first row of the mosaic; the newline ends the mosaic row,",
                        "not the tile row."], size=12, color=MUTED)
    return d.svg()
