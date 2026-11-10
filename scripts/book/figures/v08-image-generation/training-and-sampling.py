# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two loops of a diffusion model. Training (left): a clean value, a step t and a noise draw give a noisy value in one jump; the
# network sees the noisy value and t and is scored against the noise. Sampling (right): start from noise, and at each step from T down
# to 1 ask the network for its noise estimate, take the reverse step, and add fresh noise (except at the last step).
#
# Swedish Embedded AB implements solutions for generative models whose noising schedules, targets and samplers are specified,
# checked and made reproducible. If your team needs expertise in diffusion models and image synthesis, you can procure our services
# by sending an email to info@swedishembedded.com.
from svgdiagram import RED, Diagram

W, H = 800, 320


def render() -> str:
    d = Diagram(W, H)
    d.panel(10, 10, 380, 300, "training: one step")
    x0 = d.oval(70, 95, 76, 46, ["data x_0"], "plain")
    nz = d.oval(70, 175, 76, 46, ["noise e", "step t"], "plain")
    xt = d.box(210, 135, 100, 80, ["x_t = a_t x_0", "+ b_t e"], "model")
    net = d.box(210, 235, 100, 52, ["network", "(x_t, t)"], "accent")
    ls = d.box(330, 235, 90, 52, ["squared", "error to e"], "result")
    d.arrow(x0["r"], (160, 95))
    d.arrow(nz["r"], (160, 175))
    d.arrow(xt["b"], net["t"])
    d.arrow(net["r"], ls["l"])
    d.path([nz["b"], (70, 275), (330, 275), ls["b"]], color=RED, dash="5 3")
    d.text(200, 296, ["dashed: the target is the noise e that was added"], size=12, color=RED)
    d.panel(410, 10, 380, 300, "sampling: T steps")
    xT = d.oval(480, 95, 84, 46, ["x_T from", "N(0, 1)"], "plain")
    nt = d.box(620, 95, 100, 52, ["network", "(x_t, t)"], "accent")
    st = d.box(620, 235, 120, 52, ["reverse step", "+ fresh noise"], "model")
    out = d.oval(480, 235, 84, 46, ["x_0"], "result")
    d.arrow(xT["r"], nt["l"])
    d.arrow(nt["b"], st["t"])
    d.arrow(st["l"], out["r"])
    d.path([st["r"], (750, 235), (750, 95), nt["r"]])
    d.text(742, 165, ["x_(t-1)"], size=12, anchor="end")
    d.text(600, 296, ["repeat for t = T, T - 1, ..., 1"], size=12, color="#4a5568")
    return d.svg()
