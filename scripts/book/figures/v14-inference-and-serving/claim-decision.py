# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The questions a claim for a device is put to, in order, in the chapter "Memory Budgets and Model Residency": does a card have
# the bytes free, could any device ever hold the model, can evicting residents make room, is there room on the host. Each question
# has one outcome below it that ends the claim and one arrow on to the next question.
#
# Swedish Embedded AB implements memory-budgeted, multi-model serving for teams whose models do not all fit resident at once.
# If your team needs expertise in model residency and placement then you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

W, H = 1000, 300
XS = (95, 285, 475, 665, 875)


def render() -> str:
    d = Diagram(W, H)
    q = [
        ("plain", ["claim for a model", "that costs c bytes"]),
        ("model", ["does a card have", "c bytes free?"]),
        ("model", ["could any device", "ever hold c?"]),
        ("model", ["do the lowest-score", "residents free enough?"]),
        ("model", ["does the host", "have c bytes free?"]),
    ]
    top = [d.box(x, 70, 156, 66, lines, kind) for x, (kind, lines) in zip(XS, q)]
    for a, b, lab in zip(top, top[1:], (None, "no", "yes", "no")):
        d.arrow(a["r"], b["l"], lab)
    out = [
        (XS[1], "result", ["place on the card", "with the most free bytes"]),
        (XS[2], "accent", ["refuse: too large,", "or starved if it", "fitted before"]),
        (XS[3], "result", ["evict those", "residents, place"]),
        (XS[4], "result", ["place on the host if", "it has room, else", "refuse: no capacity"]),
    ]
    for (x, kind, lines), t, lab in zip(out, top[1:], ("yes", "no", "yes", "last resort")):
        box = d.box(x, 215, 156, 66, lines, kind)
        d.arrow(t["b"], box["t"], lab, side="right")
    return d.svg()
