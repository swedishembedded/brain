# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The skeleton of SDXL's U-Net on a 128 by 128 latent: three levels down, a middle block, three levels up, and the three skips that
# cross at each resolution. The counts are those of `denoiser_architectures.unet_walk`.
#
# Swedish Embedded AB implements solutions for denoising networks of image generators that are specified and counted from their
# configurations. If your team needs expertise in U-Nets, you can procure our services by sending an email to info@swedishembedded.com.
from svgdiagram import Diagram

W, H = 800, 560
DX, UX, MX = 150, 650, 400          # the down column, the up column, the middle block
LY = (130, 240, 350)                # the three levels
BW, BH = 220, 64


def render() -> str:
    d = Diagram(W, H)
    inp = d.oval(DX, 44, 150, 36, ["latent 4 x 128 x 128"], "plain")
    out = d.oval(UX, 44, 150, 36, ["prediction 4 x 128 x 128"], "plain")
    down = [
        d.box(DX, LY[0], BW, BH, ["down 0, 128 x 128, 320", "2 resnets"], "model"),
        d.box(DX, LY[1], BW, BH, ["down 1, 64 x 64, 640", "2 resnets, 4 attention blocks"], "accent"),
        d.box(DX, LY[2], BW, BH, ["down 2, 32 x 32, 1280", "2 resnets, 20 attention blocks"], "accent"),
    ]
    up = [
        d.box(UX, LY[0], BW, BH, ["up 2, 128 x 128, 320", "3 resnets"], "model"),
        d.box(UX, LY[1], BW, BH, ["up 1, 64 x 64, 640", "3 resnets, 6 attention blocks"], "accent"),
        d.box(UX, LY[2], BW, BH, ["up 0, 32 x 32, 1280", "3 resnets, 30 attention blocks"], "accent"),
    ]
    mid = d.box(MX, 470, 200, BH, ["middle, 32 x 32, 1280", "2 resnets, 10 attention blocks"], "accent")

    d.arrow(inp["b"], down[0]["t"])
    d.arrow(down[0]["b"], down[1]["t"], "stride-2 conv", side="right")
    d.arrow(down[1]["b"], down[2]["t"], "stride-2 conv", side="right")
    d.path([down[2]["b"], (DX, 470), mid["l"]])
    d.path([mid["r"], (UX, 470), up[2]["b"]])
    d.arrow(up[2]["t"], up[1]["b"], "x2, conv", side="left")
    d.arrow(up[1]["t"], up[0]["b"], "x2, conv", side="left")
    d.arrow(up[0]["t"], out["b"])

    skips = ("320, 320, 320", "320, 640, 640", "640, 1280, 1280")
    for i, wd in enumerate(skips):
        d.arrow(down[i]["r"], up[i]["l"], ["3 skips, concatenated", f"widths {wd}"], side="above")
    d.text(W / 2, 530, ["blue boxes have attention blocks; every resnet adds a projection of the time embedding to each channel,", "and every attention block reads 77 text tokens"],
           size=12, color="#4a5568")
    return d.svg()
