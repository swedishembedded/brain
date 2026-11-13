# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Stochastic rounding of one number, from the chapter "Mixed-Precision Training": x = 1.001 lies between the neighbouring bf16 numbers 1
# and 1.0078125, at the fraction phi of the gap above the lower one, and is rounded up with probability phi. The numbers are computed with
# the worked module's gap and the chapter's definition of phi, so the picture agrees with the table beside it.
#
# Swedish Embedded AB implements solutions for memory-bounded low-precision training for its clients. If your team needs expertise in
# training large models in 16 and 8 bits, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mixed_precision as mp  # noqa: E402
import number_formats as nf  # noqa: E402
from svgdiagram import MUTED, Diagram, text_width  # noqa: E402

W, H = 700, 262
X0, X1, AXIS = 130, 570, 70
MARGIN = 8


def render() -> str:
    x = nf.f32(mp.SR_STAT_X)
    gap = mp.gap_bf16(x)
    lo = math.floor(x / gap) * gap
    phi = (x - lo) / gap
    xm = X0 + phi * (X1 - X0)
    d = Diagram(W, H)
    d.arrow((X0 - 40, AXIS), (X1 + 40, AXIS), "")
    for px in (X0, xm, X1):
        d.line((px, AXIS - 8), (px, AXIS + 8), dash="1 0")
    d.text(X0 + 8, AXIS - 22, ["lower neighbour", f"{lo:g}"], anchor="end")
    d.text(xm + 8, AXIS - 22, ["x", f"{mp.SR_STAT_X:g}"], anchor="start")
    d.text(X1, AXIS - 22, ["upper neighbour", f"{lo + gap:.10g}"])
    down = [f"rounds down to {lo:g}", f"probability {1 - phi:.3f}"]
    up = [f"rounds up to {lo + gap:.10g}", f"probability {phi:.3f}"]
    bw = text_width(up) + 2 * MARGIN
    lo_box = d.box(X0 + 40, 170, bw, 62, down, "model")
    up_box = d.box(X1 - 40, 170, bw, 62, up, "accent")
    d.arrow((xm, AXIS + 10), (lo_box["t"][0], lo_box["t"][1]))
    d.arrow((xm, AXIS + 10), (up_box["t"][0], up_box["t"][1]))
    d.text(W / 2, 236, [f"the gap between the neighbours is {gap:g}; the mean of the result is {lo:g} + {phi:.3f} x {gap:g} = x,",
                        f"where round to nearest always gives {lo:g}"], size=12, color=MUTED)
    return d.svg()
