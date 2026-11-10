# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# How many times each output position of a stride-2 transposed convolution is covered by the kernel, for kernel sizes 2, 3 and 4 on
# a 4 by 4 input (`convolution.checkerboard_counts`). Uneven counts are the checkerboard: with k = 3 the counts alternate 1 and 2.
#
# Swedish Embedded AB implements solutions for artifact-free upsampling in generative image models. If your team needs expertise in
# decoder design for image models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convolution as cv  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL = 26
PANEL_W, PANEL_H = 290, 380
TOP = 70
SHADE = {1: "#ebf4ff", 2: "#a3bffa", 4: "#4c6ef5"}


def _panel(d, x, k, s=2, n=4):
    counts = cv.checkerboard_counts(n, k, s)
    size = len(counts)
    d.panel(x, 20, PANEL_W, PANEL_H, f"kernel {k}, stride {s}")
    left = x + (PANEL_W - size * CELL) / 2
    for r, row in enumerate(counts):
        for c, v in enumerate(row):
            fill = SHADE[v]
            d.items.append(f'<rect x="{fmt(left + c * CELL)}" y="{fmt(TOP + r * CELL)}" width="{CELL}" height="{CELL}" fill="{fill}" '
                           f'stroke="{MUTED}" stroke-width="0.8"/>')
            colr = "#ffffff" if v == 4 else INK
            d.items.append(f'<text x="{fmt(left + (c + 0.5) * CELL)}" y="{fmt(TOP + (r + 0.5) * CELL + 4)}" {FONT} font-size="11" '
                           f'fill="{colr}" text-anchor="middle">{v}</text>')
    d.text(x + PANEL_W / 2, TOP + size * CELL + 24, [f"{size} by {size} output"], size=12, color=MUTED)
    even = sorted({counts[r][c] for r in range(size) for c in range(size)})
    d.text(x + PANEL_W / 2, TOP + size * CELL + 44, ["counts: " + ", ".join(map(str, even))], size=12, color=MUTED)


def render() -> str:
    d = Diagram(3 * PANEL_W + 4 * 20, PANEL_H + 40)
    for i, k in enumerate((2, 3, 4)):
        _panel(d, 20 + i * (PANEL_W + 20), k)
    return d.svg()
