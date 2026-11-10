# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One Z-Image block: attention and feed-forward sublayers each wrapped in two RMS norms, the first scaled by (1 + s) and the second
# by tanh(g), with s and g taken from one projection of the timestep vector (`denoiser_architectures.small_denoiser` builds it).
#
# Swedish Embedded AB implements solutions for denoising networks of image generators that are specified and checked against exact
# references. If your team needs expertise in diffusion transformers, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

W, H = 680, 720
X = 230                        # the main column
CX = 560                       # the conditioning column
YS = (46, 118, 190, 262, 334, 418, 490, 562, 634)


def render() -> str:
    d = Diagram(W, H)
    x0 = d.oval(X, YS[0], 200, 36, ["token rows x  [N, D]"], "plain")
    n1 = d.box(X, YS[1], 250, 44, ["RMS norm, gain x (1 + s_a)"], "model")
    at = d.box(X, YS[2], 250, 44, ["QK norm, rotary, attention", "output projection"], "model")
    n2 = d.box(X, YS[3], 250, 44, ["RMS norm, gain x tanh(g_a)"], "model")
    p1 = d.box(X, YS[4], 44, 36, ["+"], "result")
    n3 = d.box(X, YS[5], 250, 44, ["RMS norm, gain x (1 + s_m)"], "model")
    ff = d.box(X, YS[6], 250, 44, ["SwiGLU feed-forward"], "model")
    n4 = d.box(X, YS[7], 250, 44, ["RMS norm, gain x tanh(g_m)"], "model")
    p2 = d.box(X, YS[8], 44, 36, ["+"], "result")
    c = d.oval(CX, YS[0], 190, 36, ["c: timestep vector [256]"], "plain")
    mod = d.box(CX, 343, 150, 494, ["Linear 256 -> 4 D", "split into", "s_a, g_a, s_m, g_m", "each [D]", "", "one per block"], "accent")

    d.arrow(x0["b"], n1["t"])
    d.arrow(n1["b"], at["t"])
    d.arrow(at["b"], n2["t"])
    d.arrow(n2["b"], p1["t"])
    d.arrow(p1["b"], n3["t"])
    d.arrow(n3["b"], ff["t"])
    d.arrow(ff["b"], n4["t"])
    d.arrow(n4["b"], p2["t"])
    d.path([x0["l"], (60, YS[0]), (60, YS[4]), p1["l"]])
    d.path([(X, 376), (60, 376), (60, YS[8]), p2["l"]])
    d.arrow(c["b"], mod["t"])
    for ny, lab in ((YS[1], "s_a"), (YS[3], "g_a"), (YS[5], "s_m"), (YS[7], "g_m")):
        d.arrow((mod["l"][0], ny), (X + 125, ny), lab, side="above")
    return d.svg()
