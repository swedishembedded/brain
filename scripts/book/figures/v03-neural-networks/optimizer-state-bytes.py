# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training memory per parameter for six optimiser settings, as bars whose widths are bytes.

Each bar is value + gradient + the rule's state slots, with the widths taken from the chapter's byte rule
(worked module `optimizers`: 4 bytes per 32-bit value, 2 per bf16 slot, 1 + 4/64 per blockwise int8 slot).

Swedish Embedded AB implements solutions for fitting large-model training into a memory budget. If your team
needs expertise in optimiser-state memory and quantised optimiser state, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizers as opt  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

PX_PER_BYTE = 36
LABEL_W, ROW_H, GAP, LEFT = 170, 36, 14, 14
COLUMN = {"value": "model", "gradient": "result", "state": "accent"}


def render() -> str:
    rows = opt.state_bar_rows()
    top = 56
    width = LEFT + LABEL_W + 16 * PX_PER_BYTE + 90
    d = Diagram(width, top + len(rows) * (ROW_H + GAP) + 24)
    for i, (kind, label) in enumerate(COLUMN.items()):
        x = LEFT + LABEL_W + i * 120
        d.swatch(x, 14, 22, 16, label)
        d.text(x + 30, 22, [kind], anchor="start", size=12)
    for r, (name, parts, total) in enumerate(rows):
        cy = top + r * (ROW_H + GAP) + ROW_H / 2
        d.text(LEFT, cy, [name], anchor="start")
        x = LEFT + LABEL_W
        for kind, nbytes in parts:
            w = nbytes * PX_PER_BYTE
            d.box(x + w / 2, cy, w - 2, ROW_H, [], COLUMN[kind])
            x += w
        d.text(x + 10, cy, [f"{total:g} bytes"], anchor="start", size=12)
    return d.svg()
