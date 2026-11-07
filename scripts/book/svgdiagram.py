# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Hand-placed block diagrams for the book, on an exact grid.

Graphviz lays out each cluster on its own, so a figure of parallel panels
comes out with panels of different widths, columns that do not line up and
edge labels crowding the arrows. A block diagram whose geometry is the point
(the same three stages in three settings, a pipeline, a loop) is placed here
by coordinates instead: equal panels, aligned columns, arrows that start and
end on box edges, and labels set clear of the lines. The style (type, colours,
stroke weights) is the book's one visual vocabulary, shared with svgplot.

Swedish Embedded AB implements solutions for clear technical communication.
If your team needs expertise in machine learning education material, you can
procure our services by sending an email to info@swedishembedded.com.
"""
from html import escape

from svgplot import FONT, fmt

INK = "#1a202c"
MUTED = "#4a5568"
FRAME = "#a0aec0"
FILL = {"plain": "#ffffff", "model": "#e2e8f0", "result": "#fefcbf", "accent": "#ebf4ff"}
LINE_H = 16  # px between text lines at the body size


class Diagram:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.items: list[str] = []

    def panel(self, x, y, w, h, title):
        """A titled frame grouping related boxes."""
        self.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="6" '
                          f'fill="#fbfcfe" stroke="{FRAME}" stroke-width="1"/>')
        self.items.append(f'<text x="{fmt(x + 14)}" y="{fmt(y + 22)}" {FONT} font-size="13" font-weight="bold" '
                          f'fill="{INK}">{escape(title)}</text>')

    def text(self, cx, cy, lines, size=13, color=INK, weight="normal", anchor="middle"):
        """Lines of text centred vertically on cy."""
        top = cy - (len(lines) - 1) * LINE_H / 2
        for i, line in enumerate(lines):
            self.items.append(f'<text x="{fmt(cx)}" y="{fmt(top + i * LINE_H + size * 0.35)}" {FONT} '
                              f'font-size="{size}" font-weight="{weight}" fill="{color}" '
                              f'text-anchor="{anchor}">{escape(line)}</text>')

    def box(self, cx, cy, w, h, lines, kind="model"):
        """A rounded box centred on (cx, cy); returns its edge midpoints."""
        self.items.append(f'<rect x="{fmt(cx - w / 2)}" y="{fmt(cy - h / 2)}" width="{fmt(w)}" height="{fmt(h)}" '
                          f'rx="10" fill="{FILL[kind]}" stroke="{MUTED}" stroke-width="1.2"/>')
        self.text(cx, cy, lines)
        return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy), "t": (cx, cy - h / 2), "b": (cx, cy + h / 2)}

    def arrow(self, a, b, label=None, above=True, gap=3):
        """A straight arrow from point a to point b (stopping `gap` px short),
        with an optional label set above or below its middle."""
        (x0, y0), (x1, y1) = a, b
        dx, dy = x1 - x0, y1 - y0
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        x0, y0 = x0 + gap * dx / n, y0 + gap * dy / n
        x1, y1 = x1 - gap * dx / n, y1 - gap * dy / n
        self.items.append(f'<line x1="{fmt(x0)}" y1="{fmt(y0)}" x2="{fmt(x1)}" y2="{fmt(y1)}" stroke="{INK}" '
                          f'stroke-width="1.4" marker-end="url(#head)"/>')
        if label:
            lines = label if isinstance(label, list) else [label]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            off = 8 + (len(lines) - 1) * LINE_H / 2 + 5
            self.text(mx, my - off if above else my + off, lines, size=12, color=MUTED)

    def svg(self) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}">'
                f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>')
        return head + "\n" + "\n".join(self.items) + "\n</svg>\n"
