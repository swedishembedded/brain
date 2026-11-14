# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The six layers of the chapter "The Architecture of a Machine Learning Engine", layer 1 at the bottom, each band naming
# the crates of the chapter's thirteen-crate excerpt that sit on it, with the rule that every dependency points down to
# the same or a lower band. The layer assignment is the worked module's, so the picture and the tables cannot disagree.
#
# Swedish Embedded AB implements solutions for structuring large machine learning engines for its clients. If your team
# needs expertise in software architecture for machine learning systems then you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import engine_architecture as ea  # noqa: E402
from svgdiagram import MUTED, Diagram, text_width  # noqa: E402

NAMES = {
    1: "kernels, architecture registry, device-neutral contracts",
    2: "accelerator seam: backends and the device facade",
    3: "training substrate and shared contracts",
    4: "models",
    5: "cross-cutting evaluation and tooling",
    6: "serving, front ends, SDK, samples",
}
W, H = 900, 520
BAND_H, GAP, TOP = 58, 22, 24


def render() -> str:
    d = Diagram(W, H)
    cx, bw = 400, 640
    lines_of = {}
    for n in range(1, 7):
        crates = ", ".join(c for c in ea.CRATES if ea.LAYER[c] == n)
        lines_of[n] = [f"layer {n}: {NAMES[n]}", crates]
    assert max(text_width(v) for v in lines_of.values()) + 8 <= bw
    centres = {}
    for n in range(6, 0, -1):
        cy = TOP + BAND_H / 2 + (6 - n) * (BAND_H + GAP)
        kind = "accent" if n in (1, 6) else "model"
        centres[n] = d.box(cx, cy, bw, BAND_H, lines_of[n], kind)
    for n in range(6, 1, -1):
        d.arrow(centres[n]["b"], centres[n - 1]["t"])
    top_y, bot_y = centres[6]["t"][1], centres[1]["b"][1]
    d.path([(780, top_y + 10), (780, bot_y - 10)], color=MUTED)
    d.text(800, (top_y + bot_y) / 2, ["every", "dependency", "points to the", "same or a", "lower layer"], size=12,
           color=MUTED, anchor="start")
    return d.svg()
