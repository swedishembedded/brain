# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The adversarial loop: noise goes through the generator to a fake sample; the discriminator scores real and fake samples; one loss
# trains the discriminator on both scores, and the generator is trained by the gradient that reaches it through the discriminator.
#
# Swedish Embedded AB implements solutions for generative models whose adversarial training is specified, checked and made
# reproducible. If your team needs expertise in generative adversarial networks and image synthesis, you can procure our services
# by sending an email to info@swedishembedded.com.
from svgdiagram import RED, Diagram

W, H = 800, 300


def render() -> str:
    d = Diagram(W, H)
    z = d.oval(52, 80, 72, 50, ["noise z"], "plain")
    g = d.box(170, 80, 96, 52, ["generator", "g(z)"], "model")
    fake = d.box(300, 80, 86, 52, ["fake x_f"], "result")
    dd = d.box(450, 150, 100, 60, ["discriminator", "D(x)"], "accent")
    real = d.box(300, 220, 86, 52, ["real x_r"], "result")
    data = d.oval(170, 220, 72, 50, ["data"], "plain")
    lg = d.box(600, 150, 100, 52, ["logits", "D(x_r), D(x_f)"], "result")
    loss = d.box(740, 150, 92, 52, ["two losses"], "accent")

    d.arrow(z["r"], g["l"])
    d.arrow(g["r"], fake["l"])
    d.arrow(data["r"], real["l"])
    d.path([fake["r"], (450, 80), dd["t"]])
    d.path([real["r"], (372, 220), (372, 166), (dd["l"][0], 166)])
    d.arrow(dd["r"], lg["l"])
    d.arrow(lg["r"], loss["l"])
    d.path([(740, 124), (740, 28), (170, 28), g["t"]], color=RED, dash="5 3")
    d.text(455, 46, ["generator: gradient of its loss reaches g through D and x_f"], size=12, color=RED)
    d.path([(740, 176), (740, 272), (450, 272), (450, 180)], color=RED, dash="5 3")
    d.text(640, 288, ["discriminator: gradient of its loss"], size=12, color=RED)
    d.text(215, 284, ["solid: the forward pass; dashed red: where gradients go."], size=12, color="#4a5568")
    return d.svg()
