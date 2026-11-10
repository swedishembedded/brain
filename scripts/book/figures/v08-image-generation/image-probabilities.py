# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 16 two-by-two images of the chapter's world, most probable first, each with three bars: its probability in the world, under
# the independent-pixel model and under the latent model, both fitted on 1000 training images. The numbers are
# `generative_modeling.probability_rows`, the table of the chapter.
#
# Swedish Embedded AB implements solutions for generative models whose probabilities can be checked image by image. If your team
# needs expertise in density estimation for images, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import generative_modeling as gm  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, PALETTE, fmt  # noqa: E402

CELL = 14                      # one pixel of a drawn image
ROW = 3 * 7 + 11               # row height
X_IMG = 70                     # left edge of the drawn images
X_BAR = 130                    # left edge of the bars
SCALE = 600                    # px per unit of probability (the largest is 0.2649)
W = X_BAR + int(0.27 * SCALE) + 60
TOP = 58
H = TOP + 16 * ROW + 12
NAMES = ("world", "independent pixels", "latent model")


def render() -> str:
    d = Diagram(W, H)
    rows = gm.probability_rows()
    for k, name in enumerate(NAMES):
        y = 16 + 14 * k
        d.items.append(f'<rect x="{X_BAR}" y="{y - 8}" width="14" height="9" fill="{PALETTE[k]}"/>')
        d.items.append(f'<text x="{X_BAR + 20}" y="{y}" {FONT} font-size="12" fill="{INK}">{name}</text>')
    for r, (i, x, *probs) in enumerate(rows):
        y = TOP + r * ROW
        d.items.append(f'<text x="{X_IMG - 8}" y="{fmt(y + ROW / 2 + 1)}" {FONT} font-size="11" fill="{MUTED}" '
                       f'text-anchor="end">{i}</text>')
        for j in range(4):
            cx, cy = X_IMG + (j % 2) * CELL, y + 2 + (j // 2) * CELL
            shade = "#1a202c" if x[j] else "#ffffff"
            d.items.append(f'<rect x="{cx}" y="{cy}" width="{CELL}" height="{CELL}" fill="{shade}" stroke="{FRAME}" '
                           f'stroke-width="0.8"/>')
        for k, p in enumerate(probs):
            by = y + 2 + k * 8
            d.items.append(f'<rect x="{X_BAR}" y="{by}" width="{fmt(p * SCALE)}" height="7" fill="{PALETTE[k]}"/>')
            d.items.append(f'<text x="{fmt(X_BAR + p * SCALE + 4)}" y="{by + 6.5}" {FONT} font-size="9" fill="{MUTED}">'
                           f'{p:.4f}</text>')
    return d.svg()
