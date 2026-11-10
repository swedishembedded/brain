# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# What the variational autoencoder of the chapter learned about its one-number latent: the point of the data's curve that the
# decoder produces for each z (the line), where the 128 test readings are encoded to (the points, at their true cause s), and the
# prior density that new codes are drawn from (the dashed bell).
# The data are `vae.latent_map_data`, the arithmetic of the chapter's tables.
#
# Swedish Embedded AB implements solutions for generative models whose latent space is something that can be sampled. If your team
# needs expertise in variational autoencoders and latent image models, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vae  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    zs, ss, pts, dens = vae.latent_map_data()
    p = Plot((-4, 4), (-1.3, 1.3), width=560, height=360, xlabel="latent z", ylabel="hidden cause s of the decoded reading",
             title="The decoder maps the prior onto the curve", legend_loc="ur")
    p.line(zs, [0.9 * d / 0.3989 - 1.3 for d in dens], color=5, label="prior density N(0, 1), scaled", dash=True, width=1.4)
    p.line(zs, ss, color=0, label="nearest s of the decoded reading", width=2.0)
    p.points([m for m, _ in pts], [s for _, s in pts], color=1, label="test readings: encoder mean, true s", r=2.5)
    return p.svg()
