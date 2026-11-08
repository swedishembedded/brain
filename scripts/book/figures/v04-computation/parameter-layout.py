# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The bytes of the example parameter block, drawn as rows of 16 bytes: the layout the kernel (WGSL rules) expects, with
# its padding, above the layout a host that aligns every field to 4 bytes produces for the same fields. Each field is a
# box over its bytes (a field that crosses a row is drawn in pieces); padding is unfilled. Offsets, gaps and sizes come
# from the worked module's `parameter_layout_rows`, so the picture and the tables of the chapter cannot disagree.
#
# Swedish Embedded AB implements solutions for keeping a GPU kernel and its host program in agreement about the bytes of
# every buffer. If your team needs expertise in GPU parameter and buffer layout, you can procure our services by sending
# an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import compute_kernels as ck  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W = 780
X0, BYTE_W, ROW_H, ROW_GAP = 96, 38, 34, 10     # 16 bytes x 38 px = 608 px
KINDS = {"n": "model", "scale": "model", "origin": "accent", "flag": "result", "dims": "model", "gain": "accent"}


def _panel(d, y, title, rows, size):
    total_rows = -(-size // 16)
    d.panel(20, y, W - 40, 56 + total_rows * (ROW_H + ROW_GAP), title)
    top = y + 44
    for t in range(0, 16, 4):
        d.text(X0 + (t + 2) * BYTE_W, top - 2, [f"+{t}"], size=11, color="#4a5568")
    for r in range(total_rows):
        d.text(X0 - 12, top + 10 + r * (ROW_H + ROW_GAP) + ROW_H / 2, [f"byte {16 * r}"], size=12, color="#4a5568", anchor="end")
        d.items.append(f'<rect x="{X0}" y="{top + 10 + r * (ROW_H + ROW_GAP)}" width="{16 * BYTE_W}" height="{ROW_H}" '
                       f'fill="none" stroke="#a0aec0" stroke-width="1" stroke-dasharray="2 3"/>')
    for fld in rows:
        start, end = fld["offset"], fld["offset"] + fld["size"]
        while start < end:
            r = start // 16
            stop = min(end, 16 * (r + 1))
            cx = X0 + (start % 16 + (stop - start) / 2) * BYTE_W
            cy = top + 10 + r * (ROW_H + ROW_GAP) + ROW_H / 2
            label = fld["name"] if start == fld["offset"] else fld["name"] + " (cont.)"
            d.box(cx, cy, (stop - start) * BYTE_W - 3, ROW_H - 4, [label], KINDS[fld["name"]])
            start = stop
    return y + 56 + total_rows * (ROW_H + ROW_GAP)


def render() -> str:
    krows, kalign, ksize, hrows, hsize = ck.parameter_layout_rows()
    h1 = 56 + (-(-ksize // 16)) * (ROW_H + ROW_GAP)
    h2 = 56 + (-(-hsize // 16)) * (ROW_H + ROW_GAP)
    d = Diagram(W, 14 + h1 + 20 + h2 + 14)
    y = _panel(d, 14, f"Kernel layout (WGSL rules, struct alignment {kalign}): {ksize} bytes; unfilled bytes are padding", krows, ksize)
    _panel(d, y + 20, f"Host layout (every field aligned to 4 bytes): {hsize} bytes, the same fields at other offsets", hrows, hsize)
    return d.svg()
