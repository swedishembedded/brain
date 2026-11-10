# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The landmarks of the three photographs of the face chapter before and after the least-squares similarity fit
# (`faces.photos`, `faces.aligned_data`): in the pixels of their own photographs they are far apart whoever is in them, in the frame of
# the template the two photographs of person A fall on top of each other.
#
# Swedish Embedded AB implements solutions for face pipelines whose alignment is verified before the recognition network sees a pixel.
# If your team needs expertise in face alignment, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import faces  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

W, H = 900, 470
PW, PH = 400, 340          # the two panels are equal
PX = (30, 470)
PY = 40
COLORS = {"A1": "#2b6cb0", "A2": "#c05621", "B": "#2f855a"}


def marker(d, name, x, y):
    c = COLORS[name]
    if name == "A1":
        d.items.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="5.5" fill="{c}" fill-opacity="0.85"/>')
    elif name == "A2":
        d.items.append(f'<rect x="{fmt(x - 5.5)}" y="{fmt(y - 5.5)}" width="11" height="11" fill="none" stroke="{c}" stroke-width="2.4"/>')
    else:
        d.items.append(f'<path d="M{fmt(x)},{fmt(y - 6.5)} L{fmt(x + 6)},{fmt(y + 5)} L{fmt(x - 6)},{fmt(y + 5)} Z" fill="{c}" fill-opacity="0.85"/>')


def cross(d, x, y):
    d.items.append(f'<path d="M{fmt(x - 6)},{fmt(y - 6)} L{fmt(x + 6)},{fmt(y + 6)} M{fmt(x - 6)},{fmt(y + 6)} L{fmt(x + 6)},{fmt(y - 6)}" '
                   f'stroke="{INK}" stroke-width="1.6" fill="none"/>')


def panel(d, i, title, lo, hi, step, points, template):
    x0, y0 = PX[i], PY
    d.panel(x0, y0, PW, PH, title)
    inner = (x0 + 28, y0 + 40, PW - 56, PH - 62)
    span_x, span_y = hi[0] - lo[0], hi[1] - lo[1]
    k = min(inner[2] / span_x, inner[3] / span_y)

    def sx(v):
        return inner[0] + (v - lo[0]) * k

    def sy(v):
        return inner[1] + (v - lo[1]) * k

    d.items.append(f'<rect x="{fmt(sx(lo[0]))}" y="{fmt(sy(lo[1]))}" width="{fmt(span_x * k)}" height="{fmt(span_y * k)}" fill="#ffffff" '
                   f'stroke="{FRAME}" stroke-width="1"/>')
    v = (int(lo[0] / step) + 1) * step
    while v < hi[0]:
        d.items.append(f'<line x1="{fmt(sx(v))}" y1="{fmt(sy(lo[1]))}" x2="{fmt(sx(v))}" y2="{fmt(sy(hi[1]))}" stroke="#edf2f7" stroke-width="1"/>')
        d.text(sx(v), sy(hi[1]) + 12, [f"{v:g}"], size=10, color=MUTED)
        v += step
    v = (int(lo[1] / step) + 1) * step
    while v < hi[1]:
        d.items.append(f'<line x1="{fmt(sx(lo[0]))}" y1="{fmt(sy(v))}" x2="{fmt(sx(hi[0]))}" y2="{fmt(sy(v))}" stroke="#edf2f7" stroke-width="1"/>')
        d.text(sx(lo[0]) - 14, sy(v), [f"{v:g}"], size=10, color=MUTED)
        v += step
    if template:
        for p in faces.TEMPLATE_112:
            cross(d, sx(p[0]), sy(p[1]))
    for name, pts in points.items():
        for p in pts:
            marker(d, name, sx(p[0]), sy(p[1]))
    return sx, sy


def render() -> str:
    lm = faces.photo_landmarks()
    al = faces.aligned_data()
    d = Diagram(W, H)
    panel(d, 0, "in the pixels of each photograph", (80, 190), (440, 400), 50, lm, False)
    sx, sy = panel(d, 1, "in the frame of the template", (28, 44), (84, 100), 10, al, True)
    # names of the landmarks beside the first photograph's aligned points
    for i, name in enumerate(faces.POINTS):
        p = faces.TEMPLATE_112[i]
        dx = -10 if i in (0, 3) else 10
        dy = 24 if i in (3, 4) else -13
        d.text(sx(p[0]) + dx, sy(p[1]) + dy, [name], size=10, color=MUTED, anchor="end" if dx < 0 else "start")
    ly = PY + PH + 30
    lx = 120
    for name, label in (("A1", "person A, photo 1"), ("A2", "person A, photo 2"), ("B", "person B")):
        marker(d, name, lx, ly)
        d.text(lx + 12, ly, [label], size=12, anchor="start")
        lx += 190
    cross(d, lx, ly)
    d.text(lx + 12, ly, ["template"], size=12, anchor="start")
    d.text(W / 2, H - 20, ["Axes are pixels. Left: photo 2 was taken closer and turned the other way, so its landmarks are far from photo 1's; person B,",
                           "photographed like photo 1, lands near it. Right: after the fit, the two photographs of A coincide and B is the one apart."],
           size=11, color=MUTED)
    return d.svg()
