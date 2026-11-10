# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The stages of one FLUX.2 klein-4b image at 1024 x 1024 with the work of each (`latent_diffusion_systems.stage_data`,
# `encode_flops`) and the bytes each holds on the device (`latent_diffusion_systems.parts`, int8 tier): the text encoder once, the
# optional reference encoder once, the denoising loop with its sampler, the mask blend and the guidance branch, and the decoder once.
#
# Swedish Embedded AB implements solutions for latent image generators whose stages are counted in operations and bytes. If your team
# needs expertise in image generation systems on a fixed memory budget, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latent_diffusion_systems as lds  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 960, 430
TOP, BOT = 120, 272


def render() -> str:
    name, fe, passes, fp, fdec = lds.stage_data()[1]
    enc = lds.encode_flops(32, True, 1024)
    p = {k: v[1] for k, v in lds.parts("klein-4b", "int8", 4096).items()}
    d = Diagram(W, H)
    prompt = d.oval(66, TOP, 96, 56, ["prompt"], "plain")
    te = d.box(236, TOP, 156, 92, ["text encoder", "Qwen3-4B, 27 blocks", f"{fe / 1e12:.1f} TFLOP once", f"{lds._gib(p['te'])} GiB weights"], "model")
    loop = d.box(520, TOP, 230, 132, ["denoising loop", f"{passes} passes of the DiT, Euler steps", f"{passes * fp / 1e12:.1f} TFLOP, {lds._gib(p['dit'])} GiB",
                                      "start: noise, or (1 - s) x0 + s eps", "after each step: blend under a mask"], "accent")
    dec = d.box(770, TOP, 124, 92, ["decoder", f"{fdec / 1e12:.1f} TFLOP once", f"{lds._gib(p['vae_dec'])} GiB graph"], "model")
    img = d.oval(896, TOP, 84, 56, ["image"], "result")
    ref = d.oval(66, BOT, 96, 56, ["reference", "image"], "plain")
    ve = d.box(236, BOT, 156, 92, ["VAE encoder", f"{enc / 1e12:.1f} TFLOP once", f"{lds._gib(p['vae_enc'])} GiB graph", "x0 and its tokens"], "model")
    d.arrow(prompt["r"], te["l"])
    d.arrow(te["r"], loop["l"], "512 rows")
    d.arrow(loop["r"], dec["l"], "tokens")
    d.arrow(dec["r"], img["l"])
    d.arrow(ref["r"], ve["l"])
    d.path([ve["r"], (396, BOT), (396, TOP + 90), (520 - 50, TOP + 90), (520 - 50, loop["b"][1])])
    d.text(560, BOT + 36, ["x0: the start, the preserved region,", "and the reference tokens of the joint sequence"], size=12, color="#4a5568")
    mk = d.oval(700, BOT - 10, 96, 56, ["mask", "(optional)"], "plain")
    d.path([mk["t"], (700, loop["b"][1])])
    d.text(480, 396, ["klein-4b, int8 tier, 1024 x 1024; the bytes are the layout the constructors upload"], size=12, color="#4a5568")
    return d.svg()
