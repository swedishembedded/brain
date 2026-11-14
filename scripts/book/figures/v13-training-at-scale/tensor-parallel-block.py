# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One block of the chapter's stack on two tensor-parallel ranks: the first matrix is cut by rows and the second by columns, so each
# rank holds a partial sum of the output, and one all-reduce adds the partials before the input is added once.
#
# Swedish Embedded AB implements solutions for spreading training over several devices for its clients. If your team needs
# expertise in tensor-parallel training you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram  # noqa: E402

W, H = 1020, 340
XS = (95, 290, 470, 650, 830, 960)
Y0, Y1 = 90, 250
YM = (Y0 + Y1) // 2


def render() -> str:
    d = Diagram(W, H)
    x = d.box(XS[0], YM, 120, 66, ["input X", "[n, D]"], "plain")
    lanes = []
    for j, y in enumerate((Y0, Y1)):
        a = d.box(XS[1], y, 150, 66, [f"rows of W1, rank {j}", "[F/2, D]"], "model")
        t = d.box(XS[2], y, 120, 66, ["tanh", "[n, F/2]"], "accent")
        b = d.box(XS[3], y, 150, 66, [f"columns of W2, rank {j}", "[D, F/2]"], "model")
        lanes.append((a, t, b, y))
        d.arrow(a["r"], t["l"], "", above=True)
        d.arrow(t["r"], b["l"], "", above=True)
    ar = d.box(XS[4], YM, 120, 66, ["all-reduce", "adds the partials"], "result")
    y_out = d.box(XS[5], YM, 80, 66, ["+ X", "= Y"], "plain")
    for a, t, b, y in lanes:
        d.arrow(x["r"], a["l"], "", above=True)
        d.arrow(b["r"], (ar["l"][0], y + (YM - y) * 0.45), "", above=True)
    d.arrow(ar["r"], y_out["l"], "", above=True)
    d.text(XS[3], H - 24, ["no message between the two matrices; one all-reduce of [n, D] numbers per block in the forward pass"], size=12, color="#4a5568")
    return d.svg()
