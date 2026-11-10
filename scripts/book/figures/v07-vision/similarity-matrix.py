# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 4 by 4 similarity matrix of a batch of four picture-caption pairs (`clip.hand_batch`) in three panels: the cosines, the softmax along
# each row (picture to caption) and the softmax down each column (caption to picture). The diagonal, where the true pairs are, is outlined;
# the loss of a direction is the negative log of its diagonal entries.
#
# Swedish Embedded AB implements solutions for joint image and text embeddings whose behaviour can be read off a matrix. If your team needs
# expertise in contrastive vision-language models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import clip  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402

CELL = 58
W, H = 960, 400
PANEL_W, PANEL_H, PANEL_Y = 290, 375, 14
LOW, HIGH = (255, 255, 255), (43, 108, 176)


def _shade(t):
    t = min(max(t, 0.0), 1.0)
    return "#{:02x}{:02x}{:02x}".format(*[round(a + (b - a) * t) for a, b in zip(LOW, HIGH)])


def _panel(d, x, title, mat, lo, hi, digits, caption):
    d.panel(x, PANEL_Y, PANEL_W, PANEL_H, title)
    x0, y0 = x + 46, 96
    for j in range(4):
        d.text(x0 + j * CELL + CELL / 2, y0 - 14, [f"caption {j}"], size=11, color=MUTED)
    for i in range(4):
        d.text(x0 - 6, y0 + i * CELL + CELL / 2, [f"image {i}"], size=11, color=MUTED, anchor="end")
        for j in range(4):
            t = (mat[i][j] - lo) / (hi - lo)
            d.items.append(f'<rect x="{x0 + j * CELL}" y="{y0 + i * CELL}" width="{CELL}" height="{CELL}" fill="{_shade(t)}" '
                           f'stroke="{FRAME}" stroke-width="1"/>')
            d.text(x0 + j * CELL + CELL / 2, y0 + i * CELL + CELL / 2, [f"{mat[i][j]:.{digits}f}"], size=12,
                   color="#ffffff" if t > 0.55 else INK)
    for k in range(4):
        d.items.append(f'<rect x="{x0 + k * CELL}" y="{y0 + k * CELL}" width="{CELL}" height="{CELL}" fill="none" stroke="{INK}" '
                       f'stroke-width="2.4"/>')
    d.text(x + PANEL_W / 2, y0 + 4 * CELL + 30, caption, size=12, color=MUTED)


def render() -> str:
    cs, p, q = clip.hand_batch_data()
    d = Diagram(W, H)
    _panel(d, 20, "cosines cs_ij", cs, -0.4, 1.0, 3, ["the diagonal holds the true pairs;", "image 2 is closer to caption 3"])
    _panel(d, 335, "softmax along rows (a = 10)", p, 0.0, 1.0, 3, ["image to caption: each row sums to 1;", "loss = -ln of the outlined entry"])
    _panel(d, 650, "softmax down columns (a = 10)", q, 0.0, 1.0, 3, ["caption to image: each column sums to 1;", "loss = -ln of the outlined entry"])
    return d.svg()
