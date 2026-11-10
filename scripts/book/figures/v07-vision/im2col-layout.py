# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 5 by 5 image of the chapter unrolled by im2col for a 3 by 3 kernel (stride 1, no padding): each window becomes one row of a 9 by 9
# matrix, and the convolution is that matrix times the kernel written as a vector (`convolution.im2col`, `convolution.X5`,
# `convolution.SOBEL`). Windows 0 and 1 and their rows are coloured.
#
# Swedish Embedded AB implements solutions for turning convolutions into matrix products that run at the speed of the hardware. If your
# team needs expertise in GEMM-based convolution kernels, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convolution as cv  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

BLUE, GREEN = "#2b6cb0", "#2f855a"
TINT = {0: "#bee3f8", 1: "#c6f6d5"}
CELL = 30
W, H = 960, 450


def _num(d, x, y, v, size=12, color=INK):
    d.items.append(f'<text x="{fmt(x)}" y="{fmt(y + 4)}" {FONT} font-size="{size}" fill="{color}" text-anchor="middle">{v}</text>')


def _grid(d, x0, y0, rows, tint=None, cell=CELL):
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            fill = (tint or {}).get(r, "#ffffff")
            d.items.append(f'<rect x="{fmt(x0 + c * cell)}" y="{fmt(y0 + r * cell)}" width="{cell}" height="{cell}" fill="{fill}" '
                           f'stroke="{FRAME}" stroke-width="1"/>')
            _num(d, x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell, v)


def _outline(d, x, y, w, h, color, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" fill="none" stroke="{color}" stroke-width="3"{extra}/>')


def render() -> str:
    col = cv.im2col([cv.X5], 3)
    wvec = [[v] for r in cv.SOBEL for v in r]
    y = [[int(sum(col[hw][t] * wvec[t][0] for t in range(9)))] for hw in range(9)]
    d = Diagram(W, H)
    # image
    ix, iy = 40, 100
    d.panel(20, 20, 5 * CELL + 40, 5 * CELL + 140, "image X")
    _grid(d, ix, iy, cv.X5)
    _outline(d, ix, iy, 3 * CELL, 3 * CELL, BLUE)
    _outline(d, ix + CELL, iy, 3 * CELL, 3 * CELL, GREEN, "6 3")
    d.text(ix + 2.5 * CELL, iy + 5 * CELL + 28, ["window 0 (solid) and", "window 1 (dashed)"], size=12, color=MUTED)
    # unrolled matrix
    cx, cy = 280, 100
    d.panel(250, 20, 9 * CELL + 70, 9 * CELL + 140, "col: one row per window (9 by 9)")
    for t in range(9):
        d.text(cx + (t + 0.5) * CELL, cy - 12, [f"{t // 3}{t % 3}"], size=11, color=MUTED)
    for hw in range(9):
        d.text(cx - 14, cy + (hw + 0.5) * CELL, [str(hw)], size=11, color=MUTED, anchor="end")
    _grid(d, cx, cy, col, TINT)
    _outline(d, cx, cy, 9 * CELL, CELL, BLUE)
    _outline(d, cx, cy + CELL, 9 * CELL, CELL, GREEN, "6 3")
    d.text(cx + 4.5 * CELL, cy + 9 * CELL + 28, ["columns: tap (m, n);", "rows: output position hw = 3i + j"], size=12, color=MUTED)
    # times kernel vector equals output vector
    kx = cx + 9 * CELL + 70
    d.items.append(f'<text x="{fmt(kx - 6)}" y="{fmt(cy + 4.5 * CELL + 6)}" {FONT} font-size="22" fill="{INK}" text-anchor="middle">x</text>')
    d.panel(kx + 14, 20, 2 * CELL + 120, 9 * CELL + 140, "K, then Y")
    _grid(d, kx + 40, cy, wvec)
    d.items.append(f'<text x="{fmt(kx + 40 + CELL + 26)}" y="{fmt(cy + 4.5 * CELL + 6)}" {FONT} font-size="22" fill="{INK}" text-anchor="middle">=</text>')
    _grid(d, kx + 40 + CELL + 52, cy, y, TINT)
    d.text(kx + 40 + CELL, cy + 9 * CELL + 28, ["kernel as a vector", "and the 9 outputs"], size=12, color=MUTED)
    return d.svg()
