# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The stages of brain's photogrammetry path from photographs to a dense starting scene: structure from motion (`crates/sfm`), then
# multi-view stereo (`crates/mvs`), driven end to end by `crates/recon`.
#
# Swedish Embedded AB implements photogrammetry pipelines from photographs to dense geometry. If your team needs expertise in
# structure from motion and multi-view stereo, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402

SFM = [["features", "RootSIFT"], ["matching", "ratio, mutual,", "essential RANSAC"], ["tracks", "union-find"],
       ["initial cameras", "global or incremental"], ["bundle adjustment", "and lens choice"]]
MVS = [["source views", "and range bounds"], ["PatchMatch", "pyramid"], ["consistency", "filter"], ["fusion", "one point per sample"],
       ["splat start", "thin gaussians"]]


def render() -> str:
    box_w = max(text_width(b) for b in SFM + MVS) + 24
    box_h, gap, pad = 66, 30, 18
    row_w = 5 * box_w + 4 * gap
    W = int(row_w + 2 * pad + 40)
    panel_h = 130
    H = int(12 + 2 * panel_h + 52 + 12)
    d = Diagram(W, H)
    rows = []
    for r, (title, items) in enumerate([("crates/sfm", SFM),
                                         ("crates/mvs", MVS)]):
        y = 12 + r * (panel_h + 52)
        d.panel(20, y, W - 40, panel_h, title)
        cy = y + 82
        boxes = []
        for i, lines in enumerate(items):
            cx = 20 + pad + box_w / 2 + 12 + i * (box_w + gap)
            kind = "result" if i == len(items) - 1 else "model"
            boxes.append(d.box(cx, cy, box_w, box_h, lines, kind))
        for a, b in zip(boxes, boxes[1:]):
            d.arrow(a["r"], b["l"])
        rows.append(boxes)
    a, b = rows[0][-1], rows[1][0]
    ymid = 12 + panel_h + 26
    d.path([a["b"], (a["b"][0], ymid), (b["t"][0], ymid), b["t"]])
    d.text(W / 2, ymid - 10, ["cameras, calibration and sparse tracks"], size=12, color="#4a5568")
    return d.svg()
