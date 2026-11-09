# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The spectral decoder block and its mixer. Left: one block, whose residual stream is m Fourier coefficients per token; attention is the
# only step that reads other positions. Right: the mixer for one token and one channel c, the filters G_c, U_c and W_c acting per mode,
# the two syntheses S (m to D numbers), the gated product in the signal domain and the analysis S^T (D to m). Sizes are those of the chapter's
# toy model (`spectral_lm`: m = 8, D = 16, C = 4) and the arithmetic is the chapter's own count.
#
# Swedish Embedded AB implements solutions for structured and spectral neural-network operators. If your team needs expertise in
# frequency-domain model design, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spectral_lm as sl  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 470


def render() -> str:
    m, dd, c = sl.M_TOY, sl.D_TOY, sl.C_TOY
    d = Diagram(W, H)
    d.panel(10, 10, 300, 450, "One block, for every token")
    cx = 160
    x0 = d.oval(cx, 85, 200, 42, [f"x: {m} Fourier coefficients"], "accent")
    att = d.box(cx, 170, 230, 56, ["RMSNorm, then causal attention", "(reads the other positions)"])
    x1 = d.oval(cx, 255, 200, 42, ["x + attention"], "plain")
    mix = d.box(cx, 340, 230, 56, ["RMSNorm, then spectral mixer", "(reads this token only)"], "result")
    x2 = d.oval(cx, 425, 200, 42, ["x + mixer"], "accent")
    d.arrow(x0["b"], att["t"])
    d.arrow(att["b"], x1["t"])
    d.arrow(x1["b"], mix["t"])
    d.arrow(mix["b"], x2["t"])

    d.panel(325, 10, 665, 450, f"The mixer for one token: the path of one channel, summed over C = {c} channels")
    h = d.oval(385, 235, 80, 44, ["h", f"[{m}]"], "accent")
    g = d.box(500, 160, 110, 50, ["G_c * h", f"[{m}]"])
    u = d.box(500, 310, 110, 50, ["U_c * h", f"[{m}]"])
    sg = d.box(640, 160, 104, 50, ["S  (synthesis)", f"[{dd}]"])
    su = d.box(640, 310, 104, 50, ["S  (synthesis)", f"[{dd}]"])
    prod = d.box(775, 235, 112, 62, ["SiLU(a) * b", f"pointwise [{dd}]"], "result")
    an = d.box(910, 235, 110, 62, ["S^T  (analysis)", f"[{m}]"])
    d.arrow(h["r"], g["l"])
    d.arrow(h["r"], u["l"])
    d.arrow(g["r"], sg["l"])
    d.arrow(u["r"], su["l"])
    d.path([sg["r"], (prod["t"][0], sg["r"][1]), prod["t"]])
    d.path([su["r"], (prod["b"][0], su["r"][1]), prod["b"]])
    d.text(715, 140, ["a"], size=12, color=MUTED)
    d.text(715, 330, ["b"], size=12, color=MUTED)
    d.arrow(prod["r"], an["l"])
    wsum = d.box(910, 345, 110, 56, ["W_c * t", "summed over c"])
    y = d.oval(910, 425, 80, 36, [f"y [{m}]"], "accent")
    d.arrow(an["b"], wsum["t"])
    d.arrow(wsum["b"], y["t"])
    d.text(600, 395, ["The product is taken sample by sample in the signal domain, where it couples frequencies;"], size=12, color=MUTED)
    d.text(600, 414, ["the filters act mode by mode in the coefficient domain, where they cannot."], size=12, color=MUTED)
    d.text(600, 440, [f"{3 * c} transforms per token (two S and one S^T for each of the {c} channels), dense or FFT"], size=12, color=MUTED)
    return d.svg()
