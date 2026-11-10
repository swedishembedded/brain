# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Why one picture cannot fix the size of the world: a box 0.75 m tall at a depth of 1 m and a box twice as tall at twice the depth
# subtend the same rays and fill the same rows of the image (`dense_prediction.pinhole_ambiguity`).
#
# Swedish Embedded AB implements solutions for depth estimation whose scale is pinned down by a stated reference. If your team needs
# expertise in monocular depth estimation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dense_prediction as dp  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, RED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

W, H = 760, 320
APEX = (70, 250)
M = 190  # drawing pixels per metre of depth
PLANE_DX = 70  # drawing distance of the image plane from the apex
ZOOM = 100  # drawing pixels per metre of height


def _bar(d, z, h, color, label):
    x = APEX[0] + z * M
    d.items.append(f'<line x1="{fmt(x)}" y1="{APEX[1]}" x2="{fmt(x)}" y2="{fmt(APEX[1] - h * ZOOM)}" stroke="{color}" stroke-width="5"/>')
    d.text(x, APEX[1] + 24, label, size=12, color=color)
    return x, APEX[1] - h * ZOOM


def render() -> str:
    box_h = 0.75  # the box of the first example: from Y = 0.25 down to the ground at Y = 1.0
    d = Diagram(W, H)
    d.text(W / 2, 26, ["Two boxes, one picture"], size=14, weight="bold")
    d.items.append(f'<line x1="{APEX[0]}" y1="{APEX[1]}" x2="{W - 30}" y2="{APEX[1]}" stroke="{FRAME}" stroke-width="1.5"/>')
    d.text(W - 30, APEX[1] + 22, ["ground"], size=12, color=MUTED, anchor="end")
    px = APEX[0] + PLANE_DX
    d.items.append(f'<line x1="{px}" y1="{APEX[1] - 130}" x2="{px}" y2="{APEX[1]}" stroke="{INK}" stroke-width="2"/>')
    d.text(px, APEX[1] - 144, ["image plane"], size=12)
    d.items.append(f'<circle cx="{APEX[0]}" cy="{APEX[1]}" r="4" fill="{INK}"/>')
    d.text(APEX[0], APEX[1] + 24, ["camera"], size=12)
    (xa, ya) = _bar(d, 1.0, box_h, "#2b6cb0", ["Z = 1 m", f"{box_h:g} m tall"])
    (xb, yb) = _bar(d, 2.0, 2 * box_h, "#c05621", ["Z = 2 m", f"{2 * box_h:g} m tall"])
    d.items.append(f'<line x1="{APEX[0]}" y1="{APEX[1]}" x2="{fmt(xb)}" y2="{fmt(yb)}" stroke="{RED}" stroke-width="1.5" stroke-dasharray="6 4"/>')
    ypl = APEX[1] - (APEX[1] - ya) * PLANE_DX / (xa - APEX[0])
    d.items.append(f'<circle cx="{px}" cy="{fmt(ypl)}" r="4" fill="{RED}"/>')
    d.text(px - 10, ypl - 18, ["the tops of both", "boxes land here"], size=12, color=RED, anchor="end")
    rows_a = dp.F_PX * box_h / 1.0
    rows_b = dp.F_PX * 2 * box_h / 2.0
    tx = 470
    lines = ["Pinhole projection, focal length f = 4 pixels:", "image height = f x height / depth",
             f"near box:  4 x {box_h:g} / 1 = {rows_a:g} pixels", f"far box:   4 x {2 * box_h:g} / 2 = {rows_b:g} pixels",
             "Both tops lie on the dashed ray, so the", "image cannot tell the two worlds apart."]
    for k, line in enumerate(lines):
        d.text(tx, 90 + k * 24, [line], size=13, anchor="start", color=INK if k < 4 else MUTED, weight="bold" if k == 0 else "normal")
    return d.svg()
