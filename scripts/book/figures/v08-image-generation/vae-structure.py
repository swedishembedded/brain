# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The training graph of a variational autoencoder: the encoder returns a mean and a log-variance, a draw of noise turns them into a
# latent, the decoder returns a reconstruction, and the loss has two parts, one pulling the reconstruction toward the input and one
# pulling the encoder's distribution toward the prior. The noise enters from the side, which is what lets gradients flow through
# the sample to the encoder.
#
# Swedish Embedded AB implements solutions for generative models that can be trained against a bound on likelihood. If your team
# needs expertise in variational autoencoders and latent image models, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

W, H = 780, 340
CY = 112          # the main row


def render() -> str:
    d = Diagram(W, H)
    x = d.oval(70, CY, 100, 54, ["reading x", "16 numbers"], "plain")
    enc = d.box(200, CY, 100, 60, ["encoder", "16-12-2"], "model")
    mean = d.box(340, CY - 38, 100, 40, ["mean m"], "result")
    logv = d.box(340, CY + 38, 120, 40, ["log-variance v"], "result")
    both = d.group(270, CY - 66, 140, 132)
    z = d.box(485, CY, 130, 60, ["z = m + e^(v/2) xi"], "accent")
    dec = d.box(640, CY, 100, 60, ["decoder", "1-12-16"], "model")
    xh = d.oval(745, CY, 56, 54, ["x_hat"], "plain")
    noise = d.oval(485, 22, 110, 34, ["xi ~ N(0, 1)"], "plain")
    kl = d.box(340, 232, 220, 44, ["KL(N(m, e^v) || N(0, 1))"], "plain")
    rec = d.box(600, 232, 210, 44, ["||x - x_hat||^2 / (2 sigma^2)"], "plain")

    d.arrow(x["r"], enc["l"])
    d.arrow(enc["r"], mean["l"])
    d.arrow(enc["r"], logv["l"])
    d.arrow(mean["r"], (z["l"][0], CY - 14))
    d.arrow(logv["r"], (z["l"][0], CY + 14))
    d.arrow((noise["b"][0], noise["b"][1]), z["t"], "fresh draw", side="right")
    d.arrow(z["r"], dec["l"])
    d.arrow(dec["r"], xh["l"])
    d.arrow(both["b"], kl["t"], dash="5 3")
    d.path([(745, CY + 27), (745, 232), (705, 232)], dash="5 3")
    d.path([(70, CY + 27), (70, 300), (600, 300), (600, 254)], dash="5 3")
    d.text(390, 326, ["loss = reconstruction + KL: the sum of the two lower boxes, averaged over the readings"], size=12, color="#4a5568")
    return d.svg()
