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


class Plot:
    """A single x/y chart: axes, ticks, lines, points, text."""

    def __init__(self, xlim, ylim, width=560, height=340, xlabel="", ylabel="", title=""):
        self.xlim, self.ylim = xlim, ylim
        self.w, self.h = width, height
        self.ml, self.mr, self.mt, self.mb = 58, 18, 30 if title else 14, 46
        self.items: list[str] = []
        self.legend: list[tuple[str, str, str]] = []
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title

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

    def text(self, xv, yv, s, anchor="start", size=12, color="#1a202c"):
        self.items.append(f'<text x="{fmt(self.x(xv))}" y="{fmt(self.y(yv))}" {FONT} font-size="{size}" '
                          f'fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')

    def vline(self, xv, color="#a0aec0", dash=True):
        d = ' stroke-dasharray="4 4"' if dash else ""
        self.items.append(f'<line x1="{fmt(self.x(xv))}" y1="{fmt(self.y(self.ylim[0]))}" x2="{fmt(self.x(xv))}" '
                          f'y2="{fmt(self.y(self.ylim[1]))}" stroke="{color}"{d}/>')

    def hline(self, yv, color="#a0aec0", dash=True):
        d = ' stroke-dasharray="4 4"' if dash else ""
        self.items.append(f'<line x1="{fmt(self.x(self.xlim[0]))}" y1="{fmt(self.y(yv))}" x2="{fmt(self.x(self.xlim[1]))}" '
                          f'y2="{fmt(self.y(yv))}" stroke="{color}"{d}/>')

    def svg(self) -> str:
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">',
               '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#4a5568"/></marker></defs>',
               f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>']
        x0, x1 = self.x(self.xlim[0]), self.x(self.xlim[1])
        y0, y1 = self.y(self.ylim[0]), self.y(self.ylim[1])
        for t in nice_ticks(*self.xlim):
            out.append(f'<line x1="{fmt(self.x(t))}" y1="{fmt(y0)}" x2="{fmt(self.x(t))}" y2="{fmt(y1)}" stroke="#edf2f7"/>')
            out.append(f'<text x="{fmt(self.x(t))}" y="{fmt(y0 + 16)}" {FONT} font-size="11" fill="#4a5568" '
                       f'text-anchor="middle">{fmt(t)}</text>')
        for t in nice_ticks(*self.ylim):
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
            out.append(f'<text x="{fmt(self.w / 2)}" y="18" {FONT} font-size="13" font-weight="bold" fill="#1a202c" '
                       f'text-anchor="middle">{escape(self.title)}</text>')
        for i, (label, c, kind) in enumerate(self.legend):
            ly = y1 + 14 + i * 18
            lx = x1 - 170
            if kind == "point":
                out.append(f'<circle cx="{fmt(lx + 10)}" cy="{fmt(ly - 4)}" r="3.5" fill="{c}"/>')
            else:
                d = ' stroke-dasharray="6 4"' if kind == "line-dash" else ""
                out.append(f'<line x1="{fmt(lx)}" y1="{fmt(ly - 4)}" x2="{fmt(lx + 20)}" y2="{fmt(ly - 4)}" '
                           f'stroke="{c}" stroke-width="2"{d}/>')
            out.append(f'<text x="{fmt(lx + 26)}" y="{fmt(ly)}" {FONT} font-size="11" fill="#1a202c">{escape(label)}</text>')
        out.append("</svg>")
        return "\n".join(out) + "\n"
