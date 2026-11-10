# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The Volume VII project as a data-flow diagram: one frame read through two preprocessing paths (a letterbox for the detector, a
# stretch with normalization for the segmenter), the detector's tail that returns boxes in frame pixels, and the box prompts that
# carry those boxes to the segmenter and the tracker. The numbers are those of `project_vision_pipeline` (the frame size, the two input
# sizes, the grid of cells, the thresholds and the bank).
#
# Swedish Embedded AB implements solutions for video pipelines whose coordinate conventions are written down once. If your team needs
# expertise in detection, segmentation and tracking pipelines, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_vision_pipeline as pv  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 940, 440
BW, BH = 150, 84
XS = (92, 292, 470, 648, 836)
Y_TOP, Y_BOT = 120, 300


def render() -> str:
    s, nw, nh, px, py = pv.letterbox_plan()
    d = Diagram(W, H)
    frame = d.oval(XS[0], (Y_TOP + Y_BOT) / 2, 140, 70, ["frame t", f"{pv.W} x {pv.H}, bytes"], "accent")
    lb = d.box(XS[1], Y_TOP, BW, BH, ["1  letterbox", f"{nw} x {nh} in {pv.SIZE} x {pv.SIZE}", f"grey, pad ({px}, {py})"])
    head = d.box(XS[2], Y_TOP, BW, BH, ["2  head", f"{pv.GRID * pv.GRID:,} cells", "box and score"])
    nms = d.box(XS[3], Y_TOP, BW, BH, ["2  filter, NMS", f"score >= {pv.CONF}", f"IoU > {pv.NMS_IOU}"])
    inv = d.box(XS[4], Y_TOP, BW, BH, ["2  inverse", "letterbox", "boxes in frame px"], "result")
    st = d.box(XS[1], Y_BOT, BW, BH, ["1  stretch", f"{pv.SIZE} x {pv.SIZE}, ImageNet", "constants"])
    seg = d.box(XS[2], Y_BOT, BW, BH, ["3  segmenter", "box prompt, seed", "region growing"])
    trk = d.box(XS[3], Y_BOT, BW, BH, ["4  tracker", "memory attention", "bank of 7 slots"])
    out = d.box(XS[4], Y_BOT, BW, BH, ["masks, presence", "per frame, per", "track"], "result")
    d.path([frame["r"], (XS[0] + 98, frame["r"][1]), (XS[0] + 98, Y_TOP), lb["l"]], radius=8)
    d.path([frame["r"], (XS[0] + 98, frame["r"][1]), (XS[0] + 98, Y_BOT), st["l"]], radius=8)
    d.arrow(lb["r"], head["l"])
    d.arrow(head["r"], nms["l"])
    d.arrow(nms["r"], inv["l"])
    d.arrow(st["r"], seg["l"])
    d.arrow(seg["r"], trk["l"])
    d.arrow(trk["r"], out["l"])
    mid = (Y_TOP + Y_BOT) / 2
    d.path([inv["b"], (inv["b"][0], mid), (seg["t"][0], mid), seg["t"]], radius=8)
    d.text(XS[3] + 20, mid - 14, ["box prompts, frame 0 (frame pixels, scaled per axis)"], size=12, color=MUTED)
    d.text(W / 2, H - 30, ["The detector reads the letterboxed view and the segmenter reads the stretched one: the same frame in two coordinate systems,",
                           "joined only by the boxes, which are always stated in frame pixels. The numbers are the milestones; the fifth runs them together."],
           size=12, color=MUTED)
    return d.svg()
