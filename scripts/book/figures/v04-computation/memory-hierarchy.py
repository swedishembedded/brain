# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The memory hierarchy of one core as a column of five levels, each with its typical capacity, typical
# latency and who shares it, read from the worked module's HIERARCHY table so the picture and the
# table of the chapter cannot disagree. The numbers are typical orders of magnitude, not measurements.
#
# Swedish Embedded AB implements solutions for making numerical code use the cache hierarchy of the
# machines it runs on. If your team needs expertise in CPU performance work for machine learning, you
# can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cpu  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, TOP, ROW_H, BOX_H = 780, 60, 72, 44
COLS = (150, 330, 500, 665)      # centres of: level, capacity, latency, shared by
COL_W = (200, 130, 130, 130)
KINDS = ("accent", "accent", "model", "model", "plain")


def render() -> str:
    h = TOP + len(cpu.HIERARCHY) * ROW_H + 24
    d = Diagram(W, h)
    d.panel(20, 12, W - 40, h - 24, "One core's memory, nearest the arithmetic at the top (typical values)")
    for cx, title in zip(COLS, ("level", "typical capacity", "typical latency", "shared by")):
        d.text(cx, TOP + 4, [title], size=12, color="#4a5568", weight="bold")
    boxes = []
    for r, (name, cap, who, lat, _) in enumerate(cpu.HIERARCHY):
        cy = TOP + 38 + r * ROW_H
        cells = (name, cpu._size(cap), f"{lat} cycle" + ("" if lat == 1 else "s"), who)
        row = []
        for c, (cx, w, text) in enumerate(zip(COLS, COL_W, cells)):
            row.append(d.box(cx, cy, w, BOX_H, [text], KINDS[r] if c == 0 else "plain"))
        boxes.append(row[0])
    for a, b in zip(boxes, boxes[1:]):
        d.arrow(a["b"], b["t"], "64-byte lines", side="right")
    return d.svg()
