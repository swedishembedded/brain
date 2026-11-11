# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The latent system of the project (`project_image_generator.pipeline_data`): the path a training item takes (picture, encoder and
# standardization, the straight path to noise, the denoiser with the class or the null class, the velocity loss) and the path a sample
# takes (noise, guided Euler steps, undoing the standardization, decoder, pixel noise), with the FLOPs of one pass of each stage.
#
# Swedish Embedded AB implements solutions for conditional latent image generators whose stages are counted in operations and bytes. If
# your team needs expertise in training and sampling image generators, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_image_generator as pig  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 980, 440
XS = (100, 300, 500, 700, 880)
TOP, BOT = 140, 340
BW, BH = 160, 84


def _n(v):
    return f"{v:,}"


def render() -> str:
    q = pig.pipeline_data()
    d = Diagram(W, H)
    d.text(20, 24, ["training: one item"], size=13, color="#1a202c", weight="bold", anchor="start")
    pic = d.oval(XS[0], TOP, 110, 60, ["picture x", f"{pig.D} values"], "plain")
    enc = d.box(XS[1], TOP, BW, BH, ["encoder, then", "standardize", f"{_n(q['enc'])} FLOPs"], "model")
    path = d.box(XS[2], TOP, BW, BH, ["straight path", "x_s = (1 - s) x0 + s e", "target e - x0"], "accent")
    den = d.box(XS[3], TOP, BW, BH, ["denoiser", f"{_n(q['ln'])} values", f"{_n(q['fl'])} FLOPs"], "model")
    loss = d.oval(XS[4], TOP, 120, 60, ["squared error", "to the target"], "result")
    d.arrow(pic["r"], enc["l"], f"{pig.Z_DIM} values", side="above")
    d.arrow(enc["r"], path["l"], "x0")
    d.arrow(path["r"], den["l"], "x_s")
    d.arrow(den["r"], loss["l"])
    eps = d.oval(XS[2], TOP - 90, 170, 44, ["s uniform, e: 4 normal"], "plain")
    d.arrow(eps["b"], path["t"])
    cls = d.oval(XS[3], TOP - 90, 150, 44, ["class c, or null", "with probability 0.15"], "plain")
    d.arrow(cls["b"], den["t"])
    d.text(20, 224, ["sampling: one picture"], size=13, color="#1a202c", weight="bold", anchor="start")
    noise = d.oval(XS[0], BOT, 110, 60, ["noise z", f"{pig.Z_DIM} values"], "plain")
    loop = d.box(XS[1], BOT, BW, BH, [f"{pig.G_STEPS} Euler steps", "u_un + g (u_c - u_un)", f"{pig.G_STEPS * 2} denoiser passes"], "accent")
    inv = d.box(XS[2], BOT, BW, BH, ["undo the", "standardization", "z0 sd + mean"], "model")
    dec = d.box(XS[3], BOT, BW, BH, ["decoder", f"{_n(q['dec'])} FLOPs", f"{pig.D} values out"], "model")
    out = d.oval(XS[4], BOT, 120, 60, ["picture", "+ 0.25 normal()"], "result")
    d.arrow(noise["r"], loop["l"])
    d.arrow(loop["r"], inv["l"], "z0")
    d.arrow(inv["r"], dec["l"], "code")
    d.arrow(dec["r"], out["l"], "mean")
    cg = d.oval(XS[1], BOT - 96, 150, 44, ["class c, scale g"], "plain")
    d.arrow(cg["b"], loop["t"])
    d.text(490, H - 24, [f"the autoencoder ({_n(q['ae'])} values) is trained first and frozen; the denoiser sees only standardized codes"],
           size=12, color="#4a5568")
    return d.svg()
