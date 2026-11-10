# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The training graph of a vector-quantized autoencoder: the encoder's rows are replaced by the nearest rows of a codebook before they
# reach the decoder. The nearest-code step passes no gradient; the reconstruction gradient is copied past it from the quantized rows
# to the encoder's rows (top), and the two quantizer terms act in opposite directions, one reaching the codebook and the other the
# encoder (bottom left).
#
# Swedish Embedded AB implements solutions for image tokenizers whose gradients are routed on purpose. If your team needs expertise
# in vector quantization and discrete latent image models, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import RED, Diagram

W, H = 800, 290
CY = 96


def render() -> str:
    d = Diagram(W, H)
    x = d.oval(52, CY, 72, 50, ["reading x"], "plain")
    enc = d.box(150, CY, 84, 52, ["encoder"], "model")
    ze = d.box(262, CY, 84, 52, ["rows z_e"], "result")
    vq = d.box(392, CY, 116, 52, ["nearest code", "k = argmin"], "accent")
    zq = d.box(522, CY, 100, 52, ["z_q = C[k]"], "result")
    dec = d.box(640, CY, 84, 52, ["decoder"], "model")
    xh = d.oval(748, CY, 60, 50, ["x_hat"], "plain")
    cb = d.box(392, 206, 120, 44, ["codebook C", "K rows of d"], "model")

    d.arrow(x["r"], enc["l"])
    d.arrow(enc["r"], ze["l"])
    d.arrow(ze["r"], vq["l"])
    d.arrow(vq["r"], zq["l"])
    d.arrow(zq["r"], dec["l"])
    d.arrow(dec["r"], xh["l"])
    d.arrow(cb["t"], vq["b"])
    d.path([cb["r"], (522, 206), zq["b"]])

    d.path([zq["t"], (522, 22), (262, 22), ze["t"]], color=RED, dash="5 3")
    d.text(392, 48, ["straight-through:", "gradient copied"], size=12, color=RED)
    d.path([(250, CY + 26), (250, 221), (cb["l"][0], 221)], color=RED, dash="5 3")
    d.path([(cb["l"][0], 192), (274, 192), (274, CY + 26)], color=RED, dash="5 3")
    d.text(244, 160, ["codebook", "term"], size=12, color=RED, anchor="end")
    d.text(282, 160, ["commitment", "term"], size=12, color=RED, anchor="start")
    d.text(400, 266, ["solid: the forward pass; dashed red: where gradients go. The argmin passes none."], size=12, color="#4a5568")
    return d.svg()
