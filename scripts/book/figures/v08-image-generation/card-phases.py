# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Bytes on the card through the three phases of a FLUX.2 klein-9b run at 1024 x 1024 with one reference, int8 tier
# (`latent_diffusion_systems.parts`, the uploaded layout): everything resident, the planner's phased charge on one card with the
# text encoder permanent, the encoder in a phase of its own, and the two-card layout.
#
# Swedish Embedded AB implements solutions for running large image generators inside a fixed memory budget. If your team needs
# expertise in device placement for generative pipelines, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latent_diffusion_systems as lds  # noqa: E402
from svgplot import Plot  # noqa: E402

CARD = 24.0


def data():
    p = {k: v[1] / lds.GIB for k, v in lds.parts("klein-9b", "int8", 4096).items()}
    te, dit, enc, dec = p["te"], p["dit"], p["vae_enc"], p["vae_dec"]
    return {
        "all parts resident on one card": [te + dit + enc + dec] * 3,
        "one card, encoder resident throughout": [te, te + dit + enc, te + dec],
        "one card, encoder in a phase of its own": [te, dit + enc, dec],
        "two cards: the denoiser's card": [0.0, dit + enc, dec],
    }


def render() -> str:
    pl = Plot((-0.5, 2.5), (0, 46), width=640, height=360, xlabel="phase: 0 the prompt is encoded, 1 the denoiser runs, 2 the image is decoded",
              ylabel="GiB on the card", legend_loc="below")
    xs = [0, 1, 2]
    for i, (name, ys) in enumerate(data().items()):
        pl.line(xs, ys, color=i, label=name)
        pl.points(xs, ys, color=i)
    pl.hline(CARD)
    pl.text(2.45, CARD + 1.2, "24 GiB card", anchor="end", size=12)
    return pl.svg()
