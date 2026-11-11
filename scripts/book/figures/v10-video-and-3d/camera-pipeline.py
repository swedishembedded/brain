# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The chain from a point of the world to a pixel: the extrinsic pair (R, t), the division by depth, the lens, and the intrinsic
# matrix, with the number of parameters each stage owns. The parameter counts are those of the chapter's pinhole model with the
# four-coefficient lens of its distortion section.
#
# Swedish Embedded AB implements solutions for camera calibration and lens modelling in photogrammetry, robotics and embedded vision.
# If your team needs expertise in camera geometry, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402

NODES = [
    ["world point", "(X, Y, Z)"],
    ["camera point", "(x, y, z)"],
    ["normalized point", "(x / z, y / z)"],
    ["distorted point", "(x_d, y_d)"],
    ["pixel", "(u, v)"],
]
STAGES = [
    ["extrinsics: R and t", "6 numbers"],
    ["divide by z", "no parameters"],
    ["lens: k1, k2, p1, p2", "4 numbers"],
    ["intrinsics: fx, fy, cx, cy", "4 numbers"],
]


def render() -> str:
    node_w = max(text_width(n) for n in NODES) + 40
    label_w = max(text_width(s) for s in STAGES) + 8
    node_h, gap = 46, 62
    cx = 24 + node_w / 2
    width = int(24 + node_w + 24 + label_w + 30)
    height = 24 + len(NODES) * node_h + (len(NODES) - 1) * gap + 24
    d = Diagram(width, height)
    boxes = []
    for i, lines in enumerate(NODES):
        cy = 24 + node_h / 2 + i * (node_h + gap)
        boxes.append(d.oval(cx, cy, node_w, node_h, lines, "result" if i in (0, len(NODES) - 1) else "plain"))
    for i, lines in enumerate(STAGES):
        a, b = boxes[i]["b"], boxes[i + 1]["t"]
        d.arrow(a, b)
        mid = (a[1] + b[1]) / 2
        d.text(cx + 12 + label_w / 2, mid, lines, size=12, color="#4a5568")
    return d.svg()
