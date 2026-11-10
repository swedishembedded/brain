# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The data flow of a feature pyramid on a 640 by 640 input: the backbone levels F3, F4, F5 of a ResNet-50 (strides 8, 16, 32), the
# top-down path (lateral 1x1 convolution, nearest upsampling and sum) and the bottom-up path (strided 3x3 convolution and sum), with the
# shape of each map (`cnns.fpn_rows`, `cnns.RESNET50_OUT`).
#
# Swedish Embedded AB implements solutions for multi-scale vision models whose shapes are checked before they are trained. If your team
# needs expertise in detection necks and feature pyramids, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cnns  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 760, 450
BW, BH = 132, 60
XS = {"F": 76, "M": 282, "P": 488, "N": 694}
YS = {3: 120, 4: 250, 5: 380}


def render() -> str:
    side = {l: cnns.PYRAMID_INPUT // 2 ** l for l in (3, 4, 5)}
    c = cnns.PYRAMID_C
    d = Diagram(W, H)
    for col, title in (("F", "backbone"), ("M", "merged maps"), ("P", "pyramid"), ("N", "bottom-up")):
        d.text(XS[col], 34, [title], size=14, weight="bold")
    d.text((XS["M"] + XS["P"]) / 2, 54, ["top-down path"], size=12, color=MUTED)
    d.text(XS["N"], 54, ["PAN path"], size=12, color=MUTED)
    d.text(XS["F"], 54, ["ResNet-50"], size=12, color=MUTED)
    box = {}
    for l in (3, 4, 5):
        shape = f"{c} x {side[l]} x {side[l]}"
        box["F", l] = d.box(XS["F"], YS[l], BW, BH, [f"F{l}, stride {2 ** l}", f"{cnns.RESNET50_OUT[l]} x {side[l]} x {side[l]}"], "model")
        box["M", l] = d.box(XS["M"], YS[l], BW, BH, [f"M{l}", shape], "plain")
        box["P", l] = d.box(XS["P"], YS[l], BW, BH, [f"P{l}", shape], "accent")
        box["N", l] = d.box(XS["N"], YS[l], BW, BH, [f"N{l}", shape], "result")
        d.arrow(box["F", l]["r"], box["M", l]["l"], label=["1x1 conv"])
        d.arrow(box["M", l]["r"], box["P", l]["l"], label=["3x3 conv"])
        d.arrow(box["P", l]["r"], box["N", l]["l"], label=["copy"] if l == 3 else ["add"])
    for l in (4, 5):
        d.arrow(box["M", l]["t"], box["M", l - 1]["b"], label=["upsample x2,", "add"], side="right")
    for l in (3, 4):
        d.arrow(box["N", l]["b"], box["N", l + 1]["t"], label=["3x3 conv,", "stride 2"], side="left")
    d.text(W / 2, H - 14, ["All pyramid maps have the same number of channels, so the sums are elementwise; maps are shown as C x H x W."], size=12, color=MUTED)
    return d.svg()
