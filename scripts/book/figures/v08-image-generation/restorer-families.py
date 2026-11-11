# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The three ways brain restores a picture, side by side: a network that regresses the restored picture (Real-ESRGAN's RRDBNet), a
# network that reads the codes of a face from a learned codebook (CodeFormer), and a diffusion sampler steered by a control trunk
# (SUPIR). The sizes are those of `restoration.rrdbnet_counts`, `restoration.codeformer_counts` and `restoration.supir_counts`.
#
# Swedish Embedded AB implements solutions for restoring degraded photographs and faces with a model of what the restored picture
# should look like. If your team needs expertise in super-resolution and restoration, you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram  # noqa: E402

W, H = 1000, 560
ROWS = (110, 272, 436)


def render() -> str:
    d = Diagram(W, H)
    # lane 1: regression
    y = ROWS[0]
    d.text(24, y - 62, ["A network that returns the picture: Real-ESRGAN"], size=13, weight="bold", anchor="start")
    a = d.oval(80, y, 120, 46, ["low-resolution", "picture"], "plain")
    b = d.box(330, y, 290, 58, ["RRDBNet: 23 residual-in-residual", "dense blocks at the input size"], "model")
    c = d.box(640, y, 230, 58, ["two nearest doublings,", "each with a convolution"], "model")
    e = d.oval(890, y, 116, 46, ["picture", "4 x larger"], "result")
    d.arrow(a["r"], b["l"])
    d.arrow(b["r"], c["l"])
    d.arrow(c["r"], e["l"])

    # lane 2: codes
    y = ROWS[1]
    d.text(24, y - 62, ["A network that reads codes from a learned codebook: CodeFormer"], size=13, weight="bold", anchor="start")
    a = d.oval(70, y, 104, 46, ["degraded", "face"], "plain")
    enc = d.box(205, y, 112, 58, ["VQ encoder", "16 x 16 grid"], "model")
    pred = d.box(372, y, 142, 58, ["code predictor", "9 layers, 1,024-way"], "accent")
    cb = d.box(540, y, 112, 58, ["codebook", "1,024 x 256"], "model")
    gen = d.box(700, y, 128, 58, ["generator", "fused at 4 sizes"], "model")
    e = d.oval(890, y, 104, 46, ["restored", "face"], "result")
    d.arrow(a["r"], enc["l"])
    d.arrow(enc["r"], pred["l"])
    d.arrow(pred["r"], cb["l"])
    d.arrow(cb["r"], gen["l"])
    d.arrow(gen["r"], e["l"])
    d.path([enc["b"], (205, y + 52), (700, y + 52), gen["b"]], dash="5 3")
    d.text(452, y + 40, ["encoder features, scaled by the dial w"], size=12, color="#4a5568")

    # lane 3: diffusion
    y = ROWS[2]
    d.text(24, y - 62, ["A sampler that draws the picture: SUPIR"], size=13, weight="bold", anchor="start")
    a = d.oval(70, y, 104, 46, ["degraded", "photo"], "plain")
    enc = d.box(225, y, 140, 58, ["robust encoder", "hint z, 4 x 128 x 128"], "model")
    den = d.box(462, y, 244, 72, ["control trunk + 12 adaptors", "on the frozen SDXL U-Net", "50 steps, 2 passes each"], "accent")
    dec = d.box(698, y, 124, 58, ["decoder,", "colour fix"], "model")
    e = d.oval(892, y, 100, 46, ["restored", "photo"], "result")
    noise = d.oval(462, y - 80, 96, 34, ["noise"], "plain")
    d.arrow(a["r"], enc["l"])
    d.arrow(enc["r"], den["l"])
    d.arrow(noise["b"], den["t"])
    d.arrow(den["r"], dec["l"])
    d.arrow(dec["r"], e["l"])
    d.text(500, y + 56, ["each step: a denoiser pass, the pull toward a clean re-encode, a step to a lower noise level"], size=12, color="#4a5568")
    return d.svg()
