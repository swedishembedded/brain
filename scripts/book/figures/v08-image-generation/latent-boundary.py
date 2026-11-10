# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Where a trained autoencoder sits in an image generator, with the shapes of a 1024 x 1024 image and a 32-channel autoencoder: the
# encoder gives moments, only the mean is kept, a normalization (or a pack and a standardization) puts it on the scale the denoiser
# works at, the denoiser acts on that, and the inverse maps and the decoder return to pixels.
#
# Swedish Embedded AB implements solutions for latent image pipelines whose stages are checked against a reference. If your team
# needs expertise in autoencoders for image generators, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

W, H = 800, 250
CY = 100


def render() -> str:
    d = Diagram(W, H)
    img = d.box(62, CY, 104, 56, ["image", "3 x 1024 x 1024"], "plain")
    enc = d.box(180, CY, 84, 56, ["encoder"], "model")
    mom = d.box(310, CY, 112, 56, ["moments", "64 x 128 x 128"], "result")
    mean = d.box(440, CY, 104, 56, ["mean (first half)", "32 x 128 x 128"], "result")
    pack = d.box(575, CY, 124, 56, ["pack, standardize", "128 x 64 x 64"], "accent")
    den = d.box(715, CY, 100, 56, ["denoiser", "4096 tokens"], "model")
    dec = d.box(180, 200, 84, 44, ["decoder"], "model")
    out = d.box(62, 200, 104, 44, ["image"], "plain")
    unp = d.box(575, 200, 124, 44, ["unstandardize,", "unpack"], "accent")
    d.arrow(img["r"], enc["l"])
    d.arrow(enc["r"], mom["l"])
    d.arrow(mom["r"], mean["l"])
    d.arrow(mean["r"], pack["l"])
    d.arrow(pack["r"], den["l"])
    d.path([den["b"], (den["b"][0], 200), unp["r"]])
    d.arrow(unp["l"], dec["r"], "32 x 128 x 128")
    d.arrow(dec["l"], out["r"])
    d.text(400, 36, ["frozen at generation time: the autoencoder only encodes and decodes, the denoiser does the generating"], size=12, color="#4a5568")
    return d.svg()
