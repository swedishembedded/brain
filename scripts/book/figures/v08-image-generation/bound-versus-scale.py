# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The evidence lower bound of the trained variational autoencoder when its code is rescaled by c and the decoder's input divided by
# c: the reconstruction does not change, the KL term does, and the bound has a single maximum near c = 1. A plain autoencoder's
# reconstruction value is flat along the same line (the dashed level).
# The data are `vae.rescale_figure_data`, the arithmetic of the chapter's tables.
#
# Swedish Embedded AB implements solutions for generative models whose objective fixes the scale of the latent. If your team needs
# expertise in variational autoencoders and latent image models, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vae  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data, ae_level = vae.rescale_figure_data()
    xs = [a for a, _ in data]
    ys = [b for _, b in data]
    lo = min(ys)
    p = Plot((-3, 3), (round(lo - 1), round(max(max(ys), ae_level) + 3)), width=560, height=340, xlabel="log2 of the code scale c",
             ylabel="bound, nats per reading", title="The bound fixes the scale of the code", legend_loc="lr")
    p.line(xs, ys, color=0, label="variational autoencoder", width=2.0)
    p.points(xs, ys, color=0, r=2.5)
    best = max(range(len(ys)), key=lambda i: ys[i])
    p.vline(0.0)
    p.line([-3, 3], [ae_level, ae_level], color=5, label="plain autoencoder: reconstruction only", dash=True, width=1.4)
    p.points([xs[best]], [ys[best]], color=1, label=f"best on the grid, c = 2^{xs[best]:g}", r=5)
    return p.svg()
