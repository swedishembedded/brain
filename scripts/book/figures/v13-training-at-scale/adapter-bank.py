# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One batch of rows through a layer with an adapter bank, from the chapter "Parameter-Efficient Fine-Tuning": every row passes through
# the one resident base weight, a slot vector picks the adapter that corrects each row (or none), and the correction is added to the
# base output of that row.
#
# Swedish Embedded AB implements solutions for serving many fine-tuned variants of one model for its clients. If your team needs
# expertise in multi-tenant inference you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 380
XS = (110, 330, 610, 840)
Y1, Y2, Y3 = 62, 170, 280
BW, BH = 170, 66


def render() -> str:
    d = Diagram(W, H)
    rows = d.box(XS[0], Y1, 150, BH, ["batch rows x_i", "6 rows, 32 numbers"], "plain")
    slots = d.box(XS[0], Y3, 150, BH, ["slot vector", "[0, 2, none, 1, 0, 2]"], "plain")
    base = d.box(XS[1], Y1, BW, BH, ["one resident W0", "every row, one product"], "model")
    bank = d.box(XS[1], Y3, BW, BH, ["bank of adapters", "A_k, B_k, rank r_k"], "accent")
    gather = d.box(XS[2], Y3, BW, BH, ["per-row correction", "s_k B_k (A_k x_i)"], "accent")
    out = d.box(XS[3], Y2, BW, BH, ["y_i = W0 x_i + correction", "none: W0 x_i alone"], "result")
    d.arrow(rows["r"], base["l"], "", above=True)
    d.arrow(base["r"], (out["t"][0] - 40, out["t"][1]), "", above=True)
    d.arrow(slots["r"], bank["l"], "slot z_i", above=True)
    d.arrow(bank["r"], gather["l"], "rows of z_i", above=True)
    d.arrow(gather["r"], (out["b"][0] - 40, out["b"][1]), "", above=False)
    d.path([rows["b"], (rows["b"][0], Y2), (XS[2], Y2), gather["t"]])
    d.text(XS[3] - 40, Y3 + 62, ["a row with no adapter adds nothing;", "padding rows beyond r_k are never read"], size=12, color=MUTED)
    return d.svg()
