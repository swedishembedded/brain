# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The bit layouts of fp32, fp16, bf16, E4M3, E5M2 and an int8 code, one box per bit on a common grid.

The splits are read from the worked module `number_formats` (FORMATS), so the picture and the chapter's
tables cannot disagree.

Swedish Embedded AB implements solutions for compact numeric formats in machine learning engines. If your
team needs expertise in choosing and implementing low-precision number formats, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import number_formats as nf  # noqa: E402
from svgdiagram import FILL, MUTED, Diagram  # noqa: E402
from svgplot import fmt  # noqa: E402

PITCH, BOX_H, ROW_H = 15, 24, 44
LABEL_W, LEFT, TOP = 70, 14, 56
SPLIT_X = LEFT + LABEL_W + 32 * PITCH + 18


def _bit(d, x, cy, kind):
    d.items.append(f'<rect x="{fmt(x)}" y="{fmt(cy - BOX_H / 2)}" width="{PITCH - 2}" height="{BOX_H}" rx="2" '
                   f'fill="{FILL[kind]}" stroke="{MUTED}" stroke-width="1"/>')


def render() -> str:
    rows = []
    for f in ["fp32", "fp16", "bf16", "E4M3", "E5M2"]:
        ne, nm, _ = nf.FORMATS[f]
        rows.append((f, [("result", 1), ("accent", ne), ("model", nm)],
                     [f"1 + {ne} + {nm} bits", f"bias {nf.bias(f)}"]))
    rows.append(("int8", [("result", 1), ("model", 7)], ["two's complement", "weights -128, 64, ..., 1"]))
    width = SPLIT_X + 175
    d = Diagram(width, TOP + len(rows) * ROW_H + 10)
    for i, (kind, label) in enumerate([("result", "sign"), ("accent", "exponent"), ("model", "fraction (int8: magnitude bits)")]):
        x = LEFT + LABEL_W + i * 100 if i < 2 else LEFT + LABEL_W + 200
        d.swatch(x, 14, 22, 16, kind)
        d.text(x + 30, 22, [label], anchor="start", size=12)
    for r, (name, fields, note) in enumerate(rows):
        cy = TOP + r * ROW_H + ROW_H / 2 - 4
        d.text(LEFT, cy, [name], anchor="start", size=13, weight="bold")
        x = LEFT + LABEL_W
        for kind, count in fields:
            for _ in range(count):
                _bit(d, x, cy, kind)
                x += PITCH
        d.text(SPLIT_X, cy, note, anchor="start", size=12)
    return d.svg()
