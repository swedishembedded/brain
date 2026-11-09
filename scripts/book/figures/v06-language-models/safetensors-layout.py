# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 242 bytes of the three-tensor safetensors file of the chapter: the length prefix, the header and the three tensors, with the
# byte offset at every boundary and the header's entries decoded below the strip. Every number comes from
# `importing_models.layout_data`, which builds the same bytes as the listing beside the figure.
#
# Swedish Embedded AB implements solutions for loading published model weights safely. If your team needs expertise in checkpoint
# formats and their validation, you can procure our services by sending an email to info@swedishembedded.com.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importing_models as im  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 800, 330
X0 = 48
WIDTHS = {"length N": 124, "header": 290, "w": 130, "b": 90, "h": 70}   # drawn widths: legible, not proportional to the bytes
KIND = {"length N": "accent", "header": "result", "w": "model", "b": "model", "h": "model"}


def render() -> str:
    segs, header, total = im.layout_data()
    meta = json.loads(header)
    d = Diagram(W, H)
    d.text(W / 2, 22, [f"A safetensors file of {total} bytes (drawn boxes are not to scale)"], size=14, weight="bold")
    y = 90
    x = X0
    boxes, edges = {}, []
    for label, a, b in segs:
        w = WIDTHS[label]
        if label == "length N":
            lines = ["length N", f"{b - a} bytes", "u64 little-endian"]
        elif label == "header":
            lines = ["JSON header", f"N = {b - a} bytes", "padded with spaces"]
        else:
            dtype, shape = meta[label]["dtype"], meta[label]["shape"]
            lines = [f"data of {label}", dtype, f"{shape}", f"{b - a} bytes"]
        boxes[label] = d.box(x + w / 2, y, w - 4, 76, lines, KIND[label])
        edges.append((x, a))
        x += w
    edges.append((x, total))
    for (ex, off) in edges:
        d.line((ex, y + 40), (ex, y + 54), dash="2 2")
        d.text(ex, y + 66, [str(off)], size=12, color=MUTED)
    d.text(X0 - 14, y + 66, ["byte"], size=12, color=MUTED, anchor="end")
    # the header decoded
    hx = X0 + WIDTHS["length N"]
    note_y = 230
    lines = [f'__metadata__: {json.dumps(meta["__metadata__"])}'] + [
        f'{k}: {v["dtype"]}, shape {v["shape"]}, data_offsets {v["data_offsets"]}' for k, v in meta.items() if k != "__metadata__"]
    n = d.note(hx + 235, note_y, 470, 20 + 18 * len(lines), lines, "plain")
    d.arrow(boxes["header"]["b"], (boxes["header"]["b"][0], n["t"][1]), label="parsed as", side="right")
    start = segs[1][2]
    d.text(W / 2, H - 14, [f"the data starts at byte {start} = 8 + {segs[1][2] - 8}; data_offsets count from there, so tensor w "
                           f"is bytes {segs[2][1]} to {segs[2][2]} of the file"], size=12, color=MUTED)
    return d.svg()
