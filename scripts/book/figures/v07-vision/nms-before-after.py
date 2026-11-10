# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The eight candidate boxes of the non-maximum suppression example and the five that class-aware suppression at 0.5 keeps
# (`detection.CANDIDATES`, `detection.nms_data`).
#
# Swedish Embedded AB implements solutions for detector post-processing whose output is checked candidate by candidate. If your team
# needs expertise in non-maximum suppression and evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import detection as dt  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL = 17
Y0 = 58
XS = (34, 34 + 16 * CELL + 70)
W = XS[1] + 16 * CELL + 34
H = Y0 + 16 * CELL + 52
COLORS = {0: "#2b6cb0", 1: "#c05621"}


def draw_panel(d, x0, title, ids, dets):
    d.text(x0 + 8 * CELL, 28, [title], size=14, weight="bold")
    d.items.append(f'<rect x="{x0}" y="{Y0}" width="{16 * CELL}" height="{16 * CELL}" fill="#ffffff" stroke="{MUTED}" stroke-width="1.2"/>')
    for k in range(1, 16):
        d.items.append(f'<line x1="{x0 + k * CELL}" y1="{Y0}" x2="{x0 + k * CELL}" y2="{Y0 + 16 * CELL}" stroke="#f1f4f8" stroke-width="1"/>')
        d.items.append(f'<line x1="{x0}" y1="{Y0 + k * CELL}" x2="{x0 + 16 * CELL}" y2="{Y0 + k * CELL}" stroke="#f1f4f8" stroke-width="1"/>')
    for cid, b, conf, cls in dets:
        if cid not in ids:
            continue
        x, y = x0 + b[0] * CELL, Y0 + b[1] * CELL
        dash = "" if cls == 0 else ' stroke-dasharray="7 3"'
        d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt((b[2] - b[0]) * CELL)}" height="{fmt((b[3] - b[1]) * CELL)}" fill="none" '
                       f'stroke="{COLORS[cls]}" stroke-width="2"{dash}/>')
    for cid, b, conf, cls in dets:
        if cid not in ids:
            continue
        label = f"{cid}: {conf:.2f}"
        # class 0 labels sit inside the bottom-left corner, class 1 labels inside the bottom-right corner
        x = x0 + (b[0] * CELL + 3 if cls == 0 else b[2] * CELL - 3)
        y = Y0 + b[3] * CELL - 4
        anchor = "start" if cls == 0 else "end"
        d.items.append(f'<text x="{fmt(x)}" y="{fmt(y)}" {FONT} font-size="11" fill="{COLORS[cls]}" stroke="#ffffff" stroke-width="3" '
                       f'stroke-linejoin="round" paint-order="stroke" text-anchor="{anchor}">{label}</text>')


def render() -> str:
    dets, kept = dt.nms_data()
    d = Diagram(W, H)
    draw_panel(d, XS[0], "eight candidates", {c[0] for c in dets}, dets)
    draw_panel(d, XS[1], "after class-aware NMS at 0.5", set(kept), dets)
    d.text(W / 2, H - 16, ["Solid: class 0. Dashed: class 1. Labels are candidate id: confidence; box coordinates in pixels."], size=12, color=MUTED)
    return d.svg()
