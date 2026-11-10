# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 16 by 16 picture of the detection chapter with the boxes of its two rectangles and one predicted box, and the overlap that gives
# the IoU of the predicted box with rectangle a (`detection.toy_scene`, `detection.iou_pairs`).
#
# Swedish Embedded AB implements solutions for detectors whose box conventions are checked against a hand computation. If your team needs
# expertise in object-detection geometry and evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import detection as dt  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL = 20
X0, Y0 = 44, 44
W, H = 700, 400
PRED = (3, 4, 11, 12)
COLORS = {"a": "#2b6cb0", "b": "#c05621"}
TINT = {"a": "#bee3f8", "b": "#feebc8"}


def px(x):
    return X0 + x * CELL


def py(y):
    return Y0 + y * CELL


def rect(d, b, stroke, width=2.2, dash=None, fill="none", opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    d.items.append(f'<rect x="{fmt(px(b[0]))}" y="{fmt(py(b[1]))}" width="{fmt((b[2] - b[0]) * CELL)}" height="{fmt((b[3] - b[1]) * CELL)}" '
                   f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{width}"{extra}/>')


def render() -> str:
    d = Diagram(W, H)
    img = dt.scene_pixels()
    d.items.append(f'<rect x="{X0}" y="{Y0}" width="{dt.SIZE * CELL}" height="{dt.SIZE * CELL}" fill="#ffffff" stroke="{MUTED}" stroke-width="1.2"/>')
    for y, row in enumerate(img):
        for x, ch in enumerate(row):
            if ch != ".":
                d.items.append(f'<rect x="{px(x)}" y="{py(y)}" width="{CELL}" height="{CELL}" fill="{TINT[ch]}"/>')
    for k in range(dt.SIZE + 1):
        d.items.append(f'<line x1="{px(k)}" y1="{Y0}" x2="{px(k)}" y2="{py(dt.SIZE)}" stroke="#edf2f7" stroke-width="1"/>')
        d.items.append(f'<line x1="{X0}" y1="{py(k)}" x2="{px(dt.SIZE)}" y2="{py(k)}" stroke="#edf2f7" stroke-width="1"/>')
    for k in range(0, dt.SIZE + 1, 4):
        d.text(px(k), Y0 - 12, [str(k)], size=11, color=MUTED)
        d.text(X0 - 16, py(k), [str(k)], size=11, color=MUTED)
    for _, name, b in (dt.RECT_A, dt.RECT_B):
        rect(d, b, COLORS[name])
    rect(d, PRED, INK, width=2.0, dash="6 4")
    d.text(px(5.0), py(2.35), ["a (2, 3, 10, 11)"], size=12, color=COLORS["a"])
    d.text(px(11.3), py(15.2), ["b (7, 6, 14, 14)"], size=12, color=COLORS["b"])
    d.text(px(14.2), py(3.6), ["prediction", "(3, 4, 11, 12)"], size=12, color=INK)
    # the overlap of the prediction with a, hatched
    ov = (max(PRED[0], dt.RECT_A[2][0]), max(PRED[1], dt.RECT_A[2][1]), min(PRED[2], dt.RECT_A[2][2]), min(PRED[3], dt.RECT_A[2][3]))
    rect(d, ov, "none", width=0, fill=INK, opacity=0.12)
    # right-hand side: the arithmetic
    i, u, fr = dt.iou_fraction(dt.RECT_A[2], PRED)
    w, h = dt.inter_wh(dt.RECT_A[2], PRED)
    tx = 420
    d.text(tx, 66, ["IoU of the prediction with a"], size=14, weight="bold", anchor="start")
    lines = [
        f"overlap width  min(10, 11) - max(2, 3) = {w}",
        f"overlap height min(11, 12) - max(3, 4) = {h}",
        f"intersection   {w} x {h} = {i}",
        f"area of a      8 x 8 = {dt.area(dt.RECT_A[2])}",
        f"area of the prediction  8 x 8 = {dt.area(PRED)}",
        f"union          {dt.area(dt.RECT_A[2])} + {dt.area(PRED)} - {i} = {u}",
        f"IoU            {i} / {u} = {float(fr):.4f}",
    ]
    for k, line in enumerate(lines):
        d.text(tx, 100 + k * 26, [line], size=13, anchor="start", color=INK if k == len(lines) - 1 else MUTED,
               weight="bold" if k == len(lines) - 1 else "normal")
    d.text(tx, 100 + len(lines) * 26 + 20, ["Shaded: the overlap. Solid outlines: the", "ground-truth boxes. Dashed: the prediction."],
           size=12, color=MUTED, anchor="start")
    return d.svg()
