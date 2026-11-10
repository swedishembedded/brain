# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Windows on an 8 by 8 grid of tokens: the regular partition into windows of 4 by 4, the partition after a shift of 2 (ragged windows),
# and the attention mask of the last window of the rolled map, derived by the region labels of the chapter
# (`vision_transformers.windows_figure_data`, `vision_transformers.window_pair_data`).
#
# Swedish Embedded AB implements solutions for attention whose cost grows with the picture only as fast as the work requires. If your
# team needs expertise in windowed attention for images, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vision_transformers as vt  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402

CELL = 30
W, H = 960, 400
PANEL_W, PANEL_H, PANEL_Y = 290, 375, 14
TINTS = ("#ebf4ff", "#fefcbf", "#e6fffa", "#fed7e2", "#e9d8fd")


def _grid(d, x0, y0, n, fills, cell=CELL):
    for r in range(n):
        for c in range(n):
            d.items.append(f'<rect x="{x0 + c * cell}" y="{y0 + r * cell}" width="{cell}" height="{cell}" fill="{fills(r, c)}" '
                           f'stroke="{FRAME}" stroke-width="1"/>')


def _outline(d, x, y, w, h, width=2.4):
    d.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{INK}" stroke-width="{width}"/>')


def render() -> str:
    data = vt.windows_figure_data()
    n, m, s, cuts = data["n"], data["m"], data["s"], data["cuts"]
    d = Diagram(W, H)
    px = [20, 335, 650]
    titles = [f"regular windows of {m} x {m}", f"after a shift of {s}: ragged windows", "mask of the last rolled window"]
    for x, t in zip(px, titles):
        d.panel(x, PANEL_Y, PANEL_W, PANEL_H, t)
    # panel A: regular windows
    x0, y0 = px[0] + 25, 60
    _grid(d, x0, y0, n, lambda r, c: TINTS[(r // m) * (n // m) + c // m])
    for a in range(n // m):
        for b in range(n // m):
            _outline(d, x0 + b * m * CELL, y0 + a * m * CELL, m * CELL, m * CELL)
    # panel B: ragged windows from the cuts
    x1 = px[1] + 25

    def group(idx):
        for k, (start, ln) in enumerate(cuts):
            if start <= idx < start + ln:
                return k
        raise ValueError(idx)
    _grid(d, x1, y0, n, lambda r, c: TINTS[(len(cuts) * group(r) + group(c)) % 5])
    for rs, rl in cuts:
        for cs, cl in cuts:
            _outline(d, x1 + cs * CELL, y0 + rs * CELL, cl * CELL, rl * CELL)
    # panel C: the mask of the last window, 16 x 16 positions
    win = data["last"]
    k = m * m
    mc = 15
    x2 = px[2] + 25
    for u in range(k):
        for v in range(k):
            on = win["mask"][u][v]
            d.items.append(f'<rect x="{x2 + v * mc}" y="{y0 + u * mc}" width="{mc}" height="{mc}" fill="{"#2d3748" if on else "#ffffff"}" '
                           f'stroke="{FRAME}" stroke-width="0.8"/>')
    _outline(d, x2, y0, k * mc, k * mc, 1.6)
    allowed = sum(1 for r in win["mask"] for v in r if v)
    counts = vt.window_pair_data()
    cap = lambda x, lines: d.text(x + PANEL_W / 2 - 14, y0 + n * CELL + 28, lines, size=12, color=MUTED)
    cap(px[0], ["4 windows of 16 tokens,", f"{counts['regular']:,} pairs per head"])
    cap(px[1], ["9 groups (2, 4 and 2 per axis),", f"{counts['ragged']:,} pairs per head"])
    d.text(x2 + k * mc / 2, y0 + k * mc + 26, ["row u reads column v when the two positions", f"have the same region: {allowed} of {k * k} pairs", "(position = row-major in the 4 x 4 window)"],
           size=12, color=MUTED)
    return d.svg()
