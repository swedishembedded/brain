# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two attention masks of the chapter on the sentence `the [MASK] sat on the mat`: the causal mask of a decoder and the full mask of an
# encoder. A dark cell is a pair (query i, key j) that attention may use. The row of the blank is outlined. The allowed pairs come from
# `encoders.masks_matrices`, the function the chapter's table uses.
#
# Swedish Embedded AB implements solutions for choosing and verifying attention masks in language models. If your team needs expertise in
# bidirectional and causal attention, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import encoders  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, RED, Diagram  # noqa: E402
from svgplot import fmt  # noqa: E402

W, H = 1000, 500
PANEL_W, PANEL_H, PANEL_Y = 484, 450, 24
CELL = 50
BLANK = 1


def panel(d, x0, title, allowed, caption):
    n = len(allowed)
    left = x0 + (PANEL_W - n * CELL) / 2 + 20
    top = PANEL_Y + 92
    d.panel(x0, PANEL_Y, PANEL_W, PANEL_H, title)
    d.text(left + n * CELL / 2, top - 52, ["key j (read)"], size=12, color=MUTED)
    for j, tok in enumerate(encoders.BLANK_EXAMPLE):
        d.text(left + j * CELL + CELL / 2, top - 20, [tok], size=12, color=INK)
    d.text(left - 8, top - 20, ["query i"], size=12, color=MUTED, anchor="end")
    for i in range(n):
        d.text(left - 8, top + i * CELL + CELL / 2, [encoders.BLANK_EXAMPLE[i]], size=12, color=INK, anchor="end")
        for j in range(n):
            on = allowed[i][j]
            d.items.append(f'<rect x="{fmt(left + j * CELL)}" y="{fmt(top + i * CELL)}" width="{CELL}" height="{CELL}" '
                           f'fill="{"#4a5568" if on else "#ffffff"}" stroke="{FRAME}" stroke-width="1"/>')
    d.items.append(f'<rect x="{fmt(left)}" y="{fmt(top + BLANK * CELL)}" width="{n * CELL}" height="{CELL}" fill="none" '
                   f'stroke="{RED}" stroke-width="2.4"/>')
    d.text(x0 + PANEL_W / 2, top + n * CELL + 28, [caption], size=13, weight="bold")
    reads = sum(allowed[BLANK])
    d.text(x0 + PANEL_W / 2, top + n * CELL + 50, [f"the blank (outlined) reads {reads} of {n} positions"], size=12, color=MUTED)


def render() -> str:
    causal, full = encoders.masks_matrices()
    d = Diagram(W, H)
    panel(d, 12, "decoder: causal mask", causal, f"{sum(map(sum, causal))} allowed pairs of {len(causal) ** 2}")
    panel(d, 504, "encoder: full mask", full, f"{sum(map(sum, full))} allowed pairs of {len(full) ** 2}")
    return d.svg()
