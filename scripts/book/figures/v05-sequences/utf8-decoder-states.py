# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The UTF-8 decoder as a state machine, drawn from the decoder's own transition table.

`text.DECODER_TABLE` is the table the chapter's decoder runs on; this figure groups its rows into edges and
places the eight states on a grid, so a change to the decoder changes the picture (a state the grid does not
know is an error here, not a silently missing edge). Edges that complete a character are blue.

Swedish Embedded AB implements solutions for getting text into models without silent corruption. If your
team needs expertise in text encodings and tokenizer pipelines, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import text  # noqa: E402
from svgdiagram import MUTED, Diagram, ellipse_y  # noqa: E402

BLUE = "#2b6cb0"
W, H = 900, 440
NODE_W, NODE_H = 104, 44
POS = {"start": (84, 210), "after F4": (340, 60), "after F0": (340, 130), "need3": (340, 200),
       "after E0": (340, 285), "after ED": (340, 340), "need2": (610, 130), "need1": (815, 255)}


def _range(lo, hi):
    return f"{lo:02X}" if lo == hi else f"{lo:02X}..{hi:02X}"


def render() -> str:
    edges = {}
    for src, lo, hi, dst, action in text.transition_rows():
        edges.setdefault((src, dst), [[], action in ("ascii", "cont") and dst == "start"])[0].append(_range(lo, hi))
    d = Diagram(W, H)
    box = {}
    for name, (x, y) in POS.items():
        box[name] = d.oval(x, y, NODE_W, NODE_H, [name], "result" if name == "start" else "plain")

    def label(ranges, emit):
        return ", ".join(ranges) + ("  (emit)" if emit else "")

    for (src, dst), (ranges, emit) in edges.items():
        color = BLUE if emit else "#1a202c"
        text_ = label(ranges, emit)
        if (src, dst) == ("start", "start"):  # ASCII: one byte, one character
            x, y = POS["start"]
            ex = box["start"]["ell"]
            d.path([(x - ellipse_x(ex, 14), y - 14), (x - 72, y - 14), (x - 72, y + 14), (x - ellipse_x(ex, 14), y + 14)],
                   radius=6, color=color)
            d.text(x - 72, y - 32, [text_], anchor="start", size=12, color=color)
        elif (src, dst) == ("start", "need2"):  # over the top, clear of the fan
            sx, sy = POS["start"]
            tx, ty = POS["need2"]
            d.path([(sx + 14, ellipse_y(box["start"]["ell"], "t", 14)), (sx + 14, 18), (tx, 18), (tx, box["need2"]["t"][1])], color=color)
            d.text((sx + tx) / 2 + 40, 10, [text_], size=12)
        elif (src, dst) == ("start", "need1"):  # along the bottom
            sx, sy = POS["start"]
            tx, ty = POS["need1"]
            d.path([(sx - 22, ellipse_y(box["start"]["ell"], "b", -22)), (sx - 22, H - 12), (tx + 30, H - 12), (tx + 30, ty + NODE_H / 2)], color=color)
            d.text((sx + tx) / 2, H - 22, [text_], size=12)
        elif (src, dst) == ("need1", "start"):  # the return of every completed multi-byte character
            sx, sy = POS["start"]
            tx, ty = POS["need1"]
            d.path([(tx - 22, ty + NODE_H / 2), (tx - 22, H - 52), (sx + 22, H - 52), (sx + 22, ellipse_y(box["start"]["ell"], "b", 22))],
                   color=color)
            d.text(sx + 70, H - 62, [text_], anchor="start", size=12, color=color)
        else:
            a, b = box[src], box[dst]
            (ax, ay), (bx, by) = a["r"], b["l"]
            d.arrow((ax, ay), (bx, by), color=color)
            if src == "start":  # a fan: the label sits beside the arrowhead, on the side away from the line
                d.text(bx - 6, by + (15 if by > ay else -12), [text_], anchor="end", size=12, color=color)
            else:  # beside the start of the line, on the side the line leaves
                d.text(ax + 8, ay + (14 if by < ay else -12), [text_], anchor="start", size=12, color=color)
    return d.svg()


def ellipse_x(ell, dy):
    """Half width of the ellipse at vertical offset dy from its centre (the left boundary is cx minus this)."""
    _, _, rx, ry = ell
    return rx * (1 - (dy / ry) ** 2) ** 0.5
