# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A small, dependency-free SVG plotting helper for the book's figures.

Book figures that show numbers (curves, histograms, heat maps) are computed by
a Python script from the same arithmetic the chapter describes, then drawn
with this module, so a figure can never disagree with its own equations. The
output is plain SVG with a viewBox, no scripts and no embedded fonts: small,
crisp at any zoom, and accepted by both the web publisher and the PDF build.

Swedish Embedded AB implements solutions for clear, verifiable technical
communication in engineering teams. If your team needs expertise in machine
learning education material, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
import re
import sys
from html import escape

FONT = "font-family=\"Helvetica, Arial, sans-serif\""
PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#9b2c2c", "#6b46c1", "#4a5568"]


def fmt(v: float) -> str:
    """Shortest exact-enough decimal for an SVG coordinate."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    ticks, t = [], first
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


_NARROW = set("iljtf.,:;!|'` ()[]-/I1")
_WIDE = set("mwMW%@")


def text_width(s: str, size: float = 11, bold: bool = False) -> float:
    """Conservative width estimate of Helvetica-like text, in px.

    The helper has no font metrics (no dependencies, no embedded fonts), so a
    per-class advance table stands in. It errs wide, which is the safe side
    for collision checks and legend sizing."""
    em = 0.0
    for ch in s:
        if ch in _NARROW:
            em += 0.30
        elif ch in _WIDE:
            em += 0.85
        elif ch.isupper():
            em += 0.68
        else:
            em += 0.56
    return em * size * (1.07 if bold else 1.0)


def _fit_ticks(lo, hi, extent_px, along_x: bool, label_px=lambda t: 0.0) -> list[float]:
    """Nice ticks, thinned until neighbouring labels cannot collide."""
    for n in (6, 5, 4, 3, 2):
        ticks = nice_ticks(lo, hi, n)
        if len(ticks) < 2:
            return ticks
        pos = [(t - lo) / (hi - lo) * extent_px for t in ticks]
        gaps = [b - a for a, b in zip(pos, pos[1:])]
        need = max(label_px(t) for t in ticks) + 10 if along_x else 18
        if min(gaps) >= need:
            return ticks
    return nice_ticks(lo, hi, 2)


class Plot:
    """A single x/y chart: axes, ticks, lines, points, text.

    Items are appended as SVG strings in pixel space (figures may append their
    own through `items`), so the plot mapping is fixed at construction. The
    legend, tick thinning and layout checks run at `svg()` time by reading the
    drawn items back, which is what lets the legend avoid data it was never
    told about. `legend_loc` is "auto" (best corner, else below the plot), one
    of "ur" "ul" "lr" "ll" (inside corner), "right"/"below" (outside, the
    canvas grows) or None (no legend)."""

    context = ""  # set by figures.py to the figure being rendered, for messages

    def __init__(self, xlim, ylim, width=560, height=340, xlabel="", ylabel="", title="", legend_loc="auto"):
        self.xlim, self.ylim = xlim, ylim
        self.w, self.h = width, height
        self.ml, self.mr, self.mt, self.mb = 58, 18, 30 if title else 14, 46
        yw = max((text_width(fmt(t)) for t in nice_ticks(*ylim)), default=0)
        self.ml = max(self.ml, math.ceil(yw + (28 if ylabel else 14)))
        xw = max((text_width(fmt(t)) for t in nice_ticks(*xlim)), default=0) / 2
        self.mr = max(self.mr, math.ceil(xw + 4))
        self.items: list[str] = []
        self.legend: list[tuple[str, str, str]] = []
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
        self.legend_loc = legend_loc

    def x(self, v):
        x0, x1 = self.xlim
        return self.ml + (v - x0) / (x1 - x0) * (self.w - self.ml - self.mr)

    def y(self, v):
        y0, y1 = self.ylim
        return self.h - self.mb - (v - y0) / (y1 - y0) * (self.h - self.mt - self.mb)

    def line(self, xs, ys, color=0, label="", dash=False, width=2.0):
        c = PALETTE[color] if isinstance(color, int) else color
        pts = " ".join(f"{fmt(self.x(a))},{fmt(self.y(b))}" for a, b in zip(xs, ys))
        d = ' stroke-dasharray="6 4"' if dash else ""
        self.items.append(f'<polyline fill="none" stroke="{c}" stroke-width="{width}"{d} points="{pts}"/>')
        if label:
            self.legend.append((label, c, "line-dash" if dash else "line"))

    def points(self, xs, ys, color=0, label="", r=3.5):
        c = PALETTE[color] if isinstance(color, int) else color
        for a, b in zip(xs, ys):
            self.items.append(f'<circle cx="{fmt(self.x(a))}" cy="{fmt(self.y(b))}" r="{r}" fill="{c}"/>')
        if label:
            self.legend.append((label, c, "point"))

    def arrow(self, x0, y0, x1, y1, color="#4a5568"):
        self.items.append(
            f'<line x1="{fmt(self.x(x0))}" y1="{fmt(self.y(y0))}" x2="{fmt(self.x(x1))}" '
            f'y2="{fmt(self.y(y1))}" stroke="{color}" stroke-width="1.5" marker-end="url(#arrow)"/>')

    def text(self, xv, yv, s, anchor="start", size=12, color="#1a202c", plate=False):
        """A label. The white halo keeps it legible over a grid line or faint
        guide; `plate=True` adds an opaque backing for a label that must sit
        across several lines (contours). Neither replaces moving a label off a
        curve it would otherwise hide."""
        x, y = self.x(xv), self.y(yv)
        if plate:
            wd = text_width(s, size)
            bx = x - wd / 2 if anchor == "middle" else x - wd if anchor == "end" else x
            self.items.append(f'<rect x="{fmt(bx - 3)}" y="{fmt(y - size * 0.95)}" width="{fmt(wd + 6)}" '
                              f'height="{fmt(size * 1.3)}" rx="2" fill="#ffffff" fill-opacity="0.92"/>')
        self.items.append(f'<text x="{fmt(x)}" y="{fmt(y)}" {FONT} font-size="{size}" '
                          f'fill="{color}" stroke="#ffffff" stroke-width="3" stroke-linejoin="round" '
                          f'paint-order="stroke" text-anchor="{anchor}">{escape(s)}</text>')

    def vline(self, xv, color="#a0aec0", dash=True):
        d = ' stroke-dasharray="4 4"' if dash else ""
        self.items.append(f'<line x1="{fmt(self.x(xv))}" y1="{fmt(self.y(self.ylim[0]))}" x2="{fmt(self.x(xv))}" '
                          f'y2="{fmt(self.y(self.ylim[1]))}" stroke="{color}"{d}/>')

    def hline(self, yv, color="#a0aec0", dash=True):
        d = ' stroke-dasharray="4 4"' if dash else ""
        self.items.append(f'<line x1="{fmt(self.x(self.xlim[0]))}" y1="{fmt(self.y(yv))}" x2="{fmt(self.x(self.xlim[1]))}" '
                          f'y2="{fmt(self.y(yv))}" stroke="{color}"{d}/>')

    # ---- reading the drawn items back -------------------------------------

    def _scene(self):
        """(weighted sample points, filled boxes, text boxes) in pixels."""
        pts: list[tuple[float, float, float]] = []
        boxes: list[tuple[float, float, float, float]] = []
        texts: list[tuple[float, float, float, float, str]] = []

        def seg(a, b, wt):
            n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 3))
            for k in range(n + 1):
                pts.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, wt))

        def attr(tag, name, default=None):
            m = re.search(rf'\b{name}="([^"]*)"', tag)
            return m.group(1) if m else default

        for it in self.items:
            for tag in re.findall(r"<(?:polyline|polygon|circle|line|rect|text)\b[^>]*>(?:[^<]*</text>)?", it):
                kind = tag[1:].split()[0]
                # Guides and thin slate contour lines are backdrop: a label's
                # halo keeps it legible there, so they weigh little.
                guide = attr(tag, "stroke") == "#a0aec0" or (
                    attr(tag, "stroke") == "#4a5568" and float(attr(tag, "stroke-width", 2)) <= 1.3)
                wt = 0.15 if guide else 1.0
                if kind in ("polyline", "polygon"):
                    nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", attr(tag, "points", ""))]
                    p = list(zip(nums[0::2], nums[1::2]))
                    if kind == "polygon":
                        p.append(p[0])
                    for a, b in zip(p, p[1:]):
                        seg(a, b, wt)
                elif kind == "circle":
                    cx, cy, r = float(attr(tag, "cx")), float(attr(tag, "cy")), float(attr(tag, "r", 3))
                    # Scatter dots (small) weigh half: a halo label over a few
                    # of them is legible, over a marker it is not.
                    dot = 0.5 if r <= 2.5 else 1.0
                    for dx in (-r, 0, r):
                        for dy in (-r, 0, r):
                            pts.append((cx + dx, cy + dy, dot))
                elif kind == "line":
                    seg((float(attr(tag, "x1")), float(attr(tag, "y1"))), (float(attr(tag, "x2")), float(attr(tag, "y2"))), wt)
                elif kind == "rect":
                    if attr(tag, "fill", "none") != "none":
                        x, y = float(attr(tag, "x")), float(attr(tag, "y"))
                        boxes.append((x, y, x + float(attr(tag, "width")), y + float(attr(tag, "height"))))
                elif kind == "text":
                    size = float(attr(tag, "font-size", 12))
                    s = re.sub(r"<[^>]+>", "", tag.split(">", 1)[1])
                    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                    wd = text_width(s, size)
                    x, y = float(attr(tag, "x")), float(attr(tag, "y"))
                    anchor = attr(tag, "text-anchor", "start")
                    x0 = x - wd / 2 if anchor == "middle" else x - wd if anchor == "end" else x
                    texts.append((x0, y - size * 0.8, x0 + wd, y + size * 0.25, s))
        return pts, boxes, texts

    def _warn(self, msg):
        print(f"svgplot warning [{self.context}]: {msg}", file=sys.stderr)

    def _lint(self, scene, x0, x1, y1, y0):
        """Report defects the helper cannot repair: clipped, colliding or
        curve-crossing text, so the figure script can move the label."""
        pts, _, texts = scene
        for i, (a0, b0, a1, b1, s) in enumerate(texts):
            if a0 < x0 - 0.5 or a1 > x1 + 0.5 or b0 < y1 - 0.5 or b1 > y0 + 0.5:
                self._warn(f"label {s!r} leaves the plot area and is clipped")
            for (c0, d0, c1, d1, t) in texts[i + 1:]:
                if a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1:
                    self._warn(f"labels {s!r} and {t!r} overlap")
            hits = sum(1 for px, py, wt in pts if wt >= 1 and a0 <= px <= a1 and b0 + 2 <= py <= b1 - 2)
            if hits:
                self._warn(f"label {s!r} sits on drawn data ({hits} samples)")

    # ---- legend -----------------------------------------------------------

    def _legend_geometry(self, columns_px=None):
        """(entry widths, row height) for the legend entries."""
        widths = [26 + text_width(label, 11) + 10 for label, _, _ in self.legend]
        return widths, 17

    def _place_legend(self, scene, x0, x1, y1, y0):
        """Choose where the legend goes: ('in', bx, by, bw, bh) or ('below'|'right', rows)."""
        widths, rh = self._legend_geometry()
        pts, boxes, texts = scene
        bw = max(widths) + 12
        bh = len(widths) * rh + 10
        loc = self.legend_loc

        def inside(corner):
            m = 8
            bx = x1 - m - bw if corner[1] == "r" else x0 + m
            by = y1 + m if corner[0] == "u" else y0 - m - bh
            return bx, by

        def score(bx, by):
            s = sum(wt for px, py, wt in pts if bx - 4 <= px <= bx + bw + 4 and by - 4 <= py <= by + bh + 4)
            s += sum(5 for (a, b, c, d) in boxes if a < bx + bw and bx < c and b < by + bh and by < d)
            s += sum(50 for (a, b, c, d, _) in texts if a < bx + bw + 4 and bx - 4 < c and b < by + bh + 4 and by - 4 < d)
            return s

        corners = {"ur": "ur", "ul": "ul", "lr": "lr", "ll": "ll"}
        if loc in corners:
            bx, by = inside(loc)
            return ("in", bx, by, bw, bh)
        if loc == "auto":
            best = min((score(*inside(c)), i, c) for i, c in enumerate(("ur", "ul", "lr", "ll")))
            if best[0] < 0.5:
                bx, by = inside(best[2])
                return ("in", bx, by, bw, bh)
            loc = "below"
        return (loc,)

    def svg(self) -> str:
        x0, x1 = self.x(self.xlim[0]), self.x(self.xlim[1])
        y0, y1 = self.y(self.ylim[0]), self.y(self.ylim[1])
        xticks = _fit_ticks(*self.xlim, x1 - x0, True, lambda t: text_width(fmt(t)))
        yticks = _fit_ticks(*self.ylim, y0 - y1, False)
        scene = self._scene()
        self._lint(scene, x0, x1, y1, y0)
        place = self._place_legend(scene, x0, x1, y1, y0) if self.legend and self.legend_loc else None
        widths, rh = self._legend_geometry() if place else ([], 17)

        # Outside legends grow the canvas; the plot mapping stays untouched.
        W, H, rows = self.w, self.h, []
        if place and place[0] == "below":
            row, used = [], 0
            for i, wd in enumerate(widths):
                if row and used + wd > self.w - 20:
                    rows.append(row)
                    row, used = [], 0
                row.append(i)
                used += wd + 14
            rows.append(row)
            H = self.h + len(rows) * rh + 8
        elif place and place[0] == "right":
            W = self.w + int(max(widths)) + 18
        if self.title and text_width(self.title, 13, True) > W - 12:
            self._warn(f"title {self.title!r} is wider than the figure")

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
               '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#4a5568"/></marker></defs>',
               f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
        for t in xticks:
            out.append(f'<line x1="{fmt(self.x(t))}" y1="{fmt(y0)}" x2="{fmt(self.x(t))}" y2="{fmt(y1)}" stroke="#edf2f7"/>')
            out.append(f'<text x="{fmt(self.x(t))}" y="{fmt(y0 + 16)}" {FONT} font-size="11" fill="#4a5568" '
                       f'text-anchor="middle">{fmt(t)}</text>')
        for t in yticks:
            out.append(f'<line x1="{fmt(x0)}" y1="{fmt(self.y(t))}" x2="{fmt(x1)}" y2="{fmt(self.y(t))}" stroke="#edf2f7"/>')
            out.append(f'<text x="{fmt(x0 - 6)}" y="{fmt(self.y(t) + 4)}" {FONT} font-size="11" fill="#4a5568" '
                       f'text-anchor="end">{fmt(t)}</text>')
        out.append(f'<rect x="{fmt(x0)}" y="{fmt(y1)}" width="{fmt(x1 - x0)}" height="{fmt(y0 - y1)}" '
                   'fill="none" stroke="#718096"/>')
        out.append(f'<clipPath id="plot"><rect x="{fmt(x0)}" y="{fmt(y1)}" width="{fmt(x1 - x0)}" height="{fmt(y0 - y1)}"/></clipPath>')
        out.append('<g clip-path="url(#plot)">' + "".join(self.items) + "</g>")
        if self.xlabel:
            out.append(f'<text x="{fmt((x0 + x1) / 2)}" y="{fmt(self.h - 10)}" {FONT} font-size="12" fill="#1a202c" '
                       f'text-anchor="middle">{escape(self.xlabel)}</text>')
        if self.ylabel:
            cy = (y0 + y1) / 2
            out.append(f'<text x="14" y="{fmt(cy)}" {FONT} font-size="12" fill="#1a202c" text-anchor="middle" '
                       f'transform="rotate(-90 14 {fmt(cy)})">{escape(self.ylabel)}</text>')
        if self.title:
            out.append(f'<text x="{fmt((x0 + x1) / 2)}" y="18" {FONT} font-size="13" font-weight="bold" fill="#1a202c" '
                       f'text-anchor="middle">{escape(self.title)}</text>')

        def entry(i, ex, ey):
            label, c, kind = self.legend[i]
            if kind == "point":
                out.append(f'<circle cx="{fmt(ex + 10)}" cy="{fmt(ey - 4)}" r="3.5" fill="{c}"/>')
            else:
                d = ' stroke-dasharray="6 4"' if kind == "line-dash" else ""
                out.append(f'<line x1="{fmt(ex)}" y1="{fmt(ey - 4)}" x2="{fmt(ex + 20)}" y2="{fmt(ey - 4)}" '
                           f'stroke="{c}" stroke-width="2"{d}/>')
            out.append(f'<text x="{fmt(ex + 26)}" y="{fmt(ey)}" {FONT} font-size="11" fill="#1a202c">{escape(label)}</text>')

        if place and place[0] == "in":
            _, bx, by, bw, bh = place
            out.append(f'<rect x="{fmt(bx)}" y="{fmt(by)}" width="{fmt(bw)}" height="{fmt(bh)}" rx="3" fill="#ffffff" '
                       'fill-opacity="0.92" stroke="#cbd5e0"/>')
            for i in range(len(self.legend)):
                entry(i, bx + 8, by + 6 + rh * (i + 1) - 4)
        elif place and place[0] == "right":
            bx = self.w + 4
            for i in range(len(self.legend)):
                entry(i, bx, y1 + 14 + i * rh)
        elif place:
            for r, row in enumerate(rows):
                total = sum(widths[i] + 14 for i in row) - 14
                ex = (self.w - total) / 2
                for i in row:
                    entry(i, ex, self.h + 4 + (r + 1) * rh - 4)
                    ex += widths[i] + 14
        out.append("</svg>")
        return "\n".join(out) + "\n"
