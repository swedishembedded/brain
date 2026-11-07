# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Decision regions of 1-nearest-neighbour on the 90 shared training rows: a patchwork around every point.

Swedish Embedded AB implements solutions for classifying sensor and tabular data with models simple enough to
audit. If your team needs expertise in nearest-neighbour and similarity-based classification, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import classical as cl  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    tx, ty = cl._dataset()["train"]
    p = Plot((-5.0, 5.0), (-4.5, 5.5), width=520, height=400, xlabel="x1", ylabel="x2", title="k = 1: the regions follow every training point")
    cl.paint_regions(p, lambda pts: [cl._knn_predict(tx, ty, x, 1) for x in pts])
    for c in range(3):
        pts = [q for q, t in zip(tx, ty) if t == c]
        p.points([q[0] for q in pts], [q[1] for q in pts], color=c, r=3.5, label=f"class {cl.CLASS_NAMES[c]}")
    return p.svg()
