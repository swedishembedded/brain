# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The toy convolutional network of the chapter with the shape of the tensor after each stage, its parameter count and the receptive field
# along one axis (`cnns.toy_layers`): conv 3x3 (4 filters), ReLU and 2x2 max pool, conv 3x3 (6 filters), ReLU and
# global average pooling, a linear layer to 3 classes and the softmax.
#
# Swedish Embedded AB implements solutions for small image classifiers that are verified layer by layer. If your team needs expertise in
# convolutional network design, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cnns  # noqa: E402
import convolution as cv  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 700, 400
BOX_W, BOX_H, GAP = 140, 124, 24
ROW_Y = (124, 290)


def render() -> str:
    c1, c2, k = cnns.C1, cnns.C2, cnns.NCLS
    rec = cv.receptive([(3, 1, 1), (2, 2, 1), (3, 1, 1)])
    p1 = c1 * 9 + c1
    p2 = c2 * c1 * 9 + c2
    pl = c2 * k + k
    stages = [
        ("accent", ["image", "1 x 8 x 8"]),
        ("model", ["conv 3x3, pad 1", f"{c1} filters", f"{p1} parameters", f"{c1} x 8 x 8", f"receptive field {rec[0][0]}"]),
        ("model", ["ReLU", "max pool 2x2, stride 2", f"{c1} x 4 x 4", f"receptive field {rec[1][0]}"]),
        ("model", ["conv 3x3, pad 1", f"{c2} filters", f"{p2} parameters", f"{c2} x 4 x 4", f"receptive field {rec[2][0]}"]),
        ("model", ["ReLU", "global average pool", f"{c2} values"]),
        ("model", ["linear", f"{c2} to {k}", f"{pl} parameters", f"{k} logits"]),
        ("result", ["softmax", "cross-entropy", f"{k} probabilities"]),
    ]
    d = Diagram(W, H)
    d.text(W / 2, 28, ["Toy network: 283 parameters, 5,778 multiply-adds per image"], size=14, weight="bold")
    per_row = 4
    x0 = (W - (per_row * BOX_W + (per_row - 1) * GAP)) / 2 + BOX_W / 2
    boxes = []
    for i, (kind, lines) in enumerate(stages):
        row, col = divmod(i, per_row)
        boxes.append(d.box(x0 + col * (BOX_W + GAP), ROW_Y[row], BOX_W, BOX_H, lines, kind))
    for i in range(len(stages) - 1):
        if i == per_row - 1:  # from the end of the first row to the start of the second
            (xa, ya), (xb, yb) = boxes[i]["b"], boxes[i + 1]["t"]
            mid = (ya + yb) / 2
            d.path([(xa, ya), (xa, mid), (xb, mid), (xb, yb)])
        else:
            d.arrow(boxes[i]["r"], boxes[i + 1]["l"])
    d.text(W / 2, H - 16, ["Each box shows the shape of the tensor it produces; the receptive field is along one axis, in input pixels."], size=12, color=MUTED)
    return d.svg()
