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
RED = "#c53030"  # backward (gradient) flow
FILL = {"plain": "#ffffff", "model": "#e2e8f0", "result": "#fefcbf", "accent": "#ebf4ff"}
LINE_H = 16  # px between text lines at the body size
CHAR_W = 0.56  # average glyph advance as a fraction of the font size


def text_width(lines, size=13):
    """Estimated width in px of the widest of `lines`, for sizing boxes."""
    lines = [lines] if isinstance(lines, str) else lines
    return max(len(line) for line in lines) * CHAR_W * size


def ellipse_x(ell, side, dy=0.0):
    """x of the ellipse (cx, cy, rx, ry) boundary at height cy + dy, on side
    "l" or "r"; lets parallel arrows meet an oval exactly."""
    cx, _, rx, ry = ell
    half = rx * (1 - (dy / ry) ** 2) ** 0.5
    return cx - half if side == "l" else cx + half


def ellipse_y(ell, side, dx=0.0):
    """y of the ellipse boundary at x = cx + dx, on side "t" or "b"."""
    _, cy, rx, ry = ell
    half = ry * (1 - (dx / rx) ** 2) ** 0.5
    return cy - half if side == "t" else cy + half


class Diagram:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.items: list[str] = []
        self.colors = {INK}  # arrowhead colours in use

    def panel(self, x, y, w, h, title):
        """A titled frame grouping related boxes."""
        self.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="6" '
                          f'fill="#fbfcfe" stroke="{FRAME}" stroke-width="1"/>')
        self.items.append(f'<text x="{fmt(x + 14)}" y="{fmt(y + 22)}" {FONT} font-size="13" font-weight="bold" '
                          f'fill="{INK}">{escape(title)}</text>')

    def group(self, x, y, w, h, color=FRAME, dash="5 3"):
        """A dashed outline around boxes that act together (a column that is
        summed, a set of copies); returns its edge midpoints like `box`."""
        self.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="12" '
                          f'fill="none" stroke="{color}" stroke-width="1.3" stroke-dasharray="{dash}"/>')
        cx, cy = x + w / 2, y + h / 2
        return {"l": (x, cy), "r": (x + w, cy), "t": (cx, y), "b": (cx, y + h)}

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

    def oval(self, cx, cy, w, h, lines, kind="plain"):
        """An ellipse (a value, as opposed to the box of an operation) centred
        on (cx, cy); returns its edge midpoints and the ellipse "ell" for
        ellipse_x / ellipse_y."""
        self.items.append(f'<ellipse cx="{fmt(cx)}" cy="{fmt(cy)}" rx="{fmt(w / 2)}" ry="{fmt(h / 2)}" '
                          f'fill="{FILL[kind]}" stroke="{MUTED}" stroke-width="1.2"/>')
        self.text(cx, cy, lines)
        return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy), "t": (cx, cy - h / 2), "b": (cx, cy + h / 2),
                "ell": (cx, cy, w / 2, h / 2)}

    def note(self, cx, cy, w, h, lines, kind="plain"):
        """A note: a box with a folded corner and left-aligned text; returns
        its edge midpoints."""
        x, y, fold = cx - w / 2, cy - h / 2, 12
        self.items.append(f'<path d="M{fmt(x)},{fmt(y)} H{fmt(x + w - fold)} L{fmt(x + w)},{fmt(y + fold)} '
                          f'V{fmt(y + h)} H{fmt(x)} Z" fill="{FILL[kind]}" stroke="{MUTED}" stroke-width="1.2"/>')
        self.items.append(f'<path d="M{fmt(x + w - fold)},{fmt(y)} V{fmt(y + fold)} H{fmt(x + w)}" fill="none" '
                          f'stroke="{MUTED}" stroke-width="1.2"/>')
        self.text(x + 14, cy, lines, anchor="start")
        return {"l": (x, cy), "r": (x + w, cy), "t": (cx, y), "b": (cx, y + h)}

    def swatch(self, x, y, w, h, kind):
        """A small filled rectangle with its top-left corner at (x, y), for legends."""
        self.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="3" '
                          f'fill="{FILL[kind]}" stroke="{MUTED}" stroke-width="1.2"/>')

    def _marker(self, color):
        self.colors.add(color)
        return "head" if color == INK else f"head-{color[1:]}"

    def _stroke(self, color, dash):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return f'stroke="{color}" stroke-width="1.4"{extra}'

    def _label(self, x, y, label, side, color):
        """Place a label beside the point (x, y) of a line: "above" or
        "below" centred on x, "left" or "right" centred on y."""
        lines = label if isinstance(label, list) else [label]
        half = (len(lines) - 1) * LINE_H / 2
        if side in ("above", "below"):
            off = 8 + half + 5
            self.text(x, y - off if side == "above" else y + off, lines, size=12, color=color)
        else:
            self.text(x + (9 if side == "right" else -9), y, lines, size=12, color=color,
                      anchor="start" if side == "right" else "end")

    def arrow(self, a, b, label=None, above=True, gap=3, color=INK, dash=None, side=None):
        """A straight arrow from point a to point b (stopping `gap` px short),
        with an optional label beside its middle: `side` is "above", "below",
        "left" or "right" of the line (default above, or below if above=False).
        `color` and `dash` (an SVG dash pattern) style the line; the label
        takes the line's colour unless it is the default ink."""
        (x0, y0), (x1, y1) = a, b
        dx, dy = x1 - x0, y1 - y0
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        x0, y0 = x0 + gap * dx / n, y0 + gap * dy / n
        x1, y1 = x1 - gap * dx / n, y1 - gap * dy / n
        self.items.append(f'<line x1="{fmt(x0)}" y1="{fmt(y0)}" x2="{fmt(x1)}" y2="{fmt(y1)}" '
                          f'{self._stroke(color, dash)} marker-end="url(#{self._marker(color)})"/>')
        if label:
            self._label((x0 + x1) / 2, (y0 + y1) / 2, label, side or ("above" if above else "below"),
                        MUTED if color == INK else color)

    def path(self, points, radius=10, gap=3, color=INK, dash=None):
        """An arrow along a polyline through `points` (axis-aligned elbows),
        corners rounded by `radius`, stopping `gap` px short of the last
        point."""
        pts = [tuple(p) for p in points]
        (px, py), (qx, qy) = pts[-2], pts[-1]
        n = ((qx - px) ** 2 + (qy - py) ** 2) ** 0.5 or 1.0
        pts[-1] = (qx - gap * (qx - px) / n, qy - gap * (qy - py) / n)
        d = f"M{fmt(pts[0][0])},{fmt(pts[0][1])}"
        for i in range(1, len(pts) - 1):
            (ax, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[i + 1]
            r = min(radius, abs(bx - ax) + abs(by - ay), abs(cx - bx) + abs(cy - by))
            u1 = ((bx - ax) / (abs(bx - ax) + abs(by - ay)), (by - ay) / (abs(bx - ax) + abs(by - ay)))
            u2 = ((cx - bx) / (abs(cx - bx) + abs(cy - by)), (cy - by) / (abs(cx - bx) + abs(cy - by)))
            d += (f" L{fmt(bx - r * u1[0])},{fmt(by - r * u1[1])} "
                  f"Q{fmt(bx)},{fmt(by)} {fmt(bx + r * u2[0])},{fmt(by + r * u2[1])}")
        d += f" L{fmt(pts[-1][0])},{fmt(pts[-1][1])}"
        self.items.append(f'<path d="{d}" fill="none" {self._stroke(color, dash)} '
                          f'marker-end="url(#{self._marker(color)})"/>')

    def line(self, a, b, dash="4 3"):
        """A plain dashed connector without an arrowhead."""
        self.items.append(f'<line x1="{fmt(a[0])}" y1="{fmt(a[1])}" x2="{fmt(b[0])}" y2="{fmt(b[1])}" '
                          f'stroke="{FRAME}" stroke-width="1.2" stroke-dasharray="{dash}"/>')

    def svg(self) -> str:
        markers = "".join(
            f'<marker id="{"head" if c == INK else "head-" + c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{c}"/></marker>' for c in sorted(self.colors))
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}"><defs>{markers}</defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>')
        return head + "\n" + "\n".join(self.items) + "\n</svg>\n"
