# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The tiny encoder-decoder segmenter of the dense-prediction chapter with the shape of the tensor after each stage, its parameter count,
# and the skip connection that carries the full-resolution features of the first stage past the bottleneck to the concatenation
# (`dense_prediction.seg_shapes`, kind unet).
#
# Swedish Embedded AB implements solutions for per-pixel prediction models whose shapes are counted layer by layer. If your team needs
# expertise in segmentation network design, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dense_prediction as dp  # noqa: E402
from svgdiagram import MUTED, RED, Diagram  # noqa: E402

W, H = 760, 420
BOX_W, BOX_H = 126, 86
XS = [88, 238, 388, 538, 688]
TOP, BOTTOM = 130, 300


def render() -> str:
    rows = {name: (shape, params, macs) for name, shape, params, macs in dp.seg_shapes("unet")}
    conv1, conv2 = dp.seg_shapes("unet")[0], dp.seg_shapes("unet")[2]
    up, head = dp.seg_shapes("unet")[3], dp.seg_shapes("unet")[5]
    total_p = sum(r[2] for r in dp.seg_shapes("unet"))
    total_m = sum(r[3] for r in dp.seg_shapes("unet"))
    d = Diagram(W, H)
    d.text(W / 2, 28, [f"Encoder-decoder segmenter: {total_p} parameters, {total_m:,} multiply-adds per 8 by 8 image"], size=14, weight="bold")
    img = d.box(XS[0], TOP, BOX_W, BOX_H, ["image", "1 x 8 x 8"], "accent")
    e1 = d.box(XS[1], TOP, BOX_W, BOX_H, ["conv 3x3, ReLU", conv1[1].replace("[", "").replace("]", "").replace(",", " x"), f"{conv1[2]} parameters"])
    e2 = d.box(XS[2], BOTTOM, BOX_W, BOX_H, ["max pool 2x2,", "conv 3x3, ReLU", "8 x 4 x 4", f"{conv2[2]} parameters"])
    ups = d.box(XS[3], BOTTOM, BOX_W, BOX_H, ["transposed conv", "k 2, stride 2", "4 x 8 x 8", f"{up[2]} parameters"])
    cat = d.box(XS[3], TOP, BOX_W, BOX_H, ["concatenate", "4 + 4 channels", "8 x 8 x 8"])
    hd = d.box(XS[4], TOP, BOX_W, BOX_H, ["conv 3x3", "3 x 8 x 8", f"{head[2]} parameters", "softmax per pixel"], "result")
    d.arrow(img["r"], e1["l"])
    d.path([e1["b"], (e1["b"][0], BOTTOM), e2["l"]], radius=12)
    d.arrow(e2["r"], ups["l"])
    d.arrow(ups["t"], cat["b"])
    d.arrow(cat["r"], hd["l"])
    d.arrow(e1["r"], cat["l"], label=["skip: the 4 x 8 x 8 features,", "copied unchanged"], color=RED)
    d.text(W / 2, H - 38, ["Without the skip the head sees only the 4 x 8 x 8 map that came through the 4 x 4 bottleneck (579 parameters);"], size=12, color=MUTED)
    d.text(W / 2, H - 20, ["each bottleneck value reads 8 input pixels along one axis, the whole image, but only at half the resolution."], size=12, color=MUTED)
    return d.svg()
