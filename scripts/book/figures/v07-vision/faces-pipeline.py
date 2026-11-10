# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The three models of the face chapter and what passes between them: one detector whose five landmarks are cut into two crops, one for
# the identity embedding (112 template) and one for the parsing network (512 template). Counts are those of `faces.scrfd_counts` and
# `faces.iresnet_table`.
#
# Swedish Embedded AB implements solutions for face pipelines whose stages are verified one at a time. If your team needs expertise in
# face detection, recognition and parsing, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import faces  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 940, 420
BW, BH = 150, 76
XS = (96, 286, 476, 666, 846)
Y_TOP, Y_MID, Y_BOT = 104, 214, 324


def render() -> str:
    cand = sum((faces.SCRFD_SIZE // s) ** 2 * faces.SCRFD_ANCHORS for s in faces.SCRFD_STRIDES)
    _, params, _ = faces.iresnet_cost()
    d = Diagram(W, H)
    photo = d.oval(XS[0], Y_MID, 130, 62, ["photograph", "any size"], "accent")
    det = d.box(XS[1], Y_MID, BW, BH, ["detector", f"{cand:,} candidates", "box, score, 5 points"])
    a112 = d.box(XS[2], Y_TOP, BW, BH, ["align to the", "112 template", "fit + warp"])
    a512 = d.box(XS[2], Y_BOT, BW, BH, ["align to the", "512 template", "fit + warp"])
    net = d.box(XS[3], Y_TOP, BW, BH, ["embedding network", f"{params / 1e6:.1f} M parameters", "112 x 112 crop"])
    seg = d.box(XS[3], Y_BOT, BW, BH, ["parsing network", "19 class scores", "per pixel"])
    out1 = d.box(XS[4], Y_TOP, 130, BH, ["512 numbers", "unit length,", "compare by cosine"], "result")
    out2 = d.box(XS[4], Y_BOT, 130, BH, ["class map", "argmax, then", "mask by class"], "result")
    d.arrow(photo["r"], det["l"])
    d.path([det["r"], (XS[2] - BW / 2 - 24, Y_MID), (XS[2] - BW / 2 - 24, Y_TOP), a112["l"]], radius=8)
    d.path([det["r"], (XS[2] - BW / 2 - 24, Y_MID), (XS[2] - BW / 2 - 24, Y_BOT), a512["l"]], radius=8)
    d.arrow(a112["r"], net["l"])
    d.arrow(a512["r"], seg["l"])
    d.arrow(net["r"], out1["l"])
    d.arrow(seg["r"], out2["l"])
    d.text(XS[1] + 20, Y_TOP + 4, ["the five landmarks", "leave the detector"], size=12, color=MUTED)
    d.text(W / 2, H - 28, ["The detector knows nothing about identity and the other two networks know nothing about detection:",
                           "the landmarks are the only thing that passes between them, with a different template for each crop."],
           size=12, color=MUTED)
    return d.svg()
