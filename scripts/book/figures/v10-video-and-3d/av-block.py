# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One block of a two-stream audio-video transformer: each stream runs its own self-attention and text cross-attention, the two meet in
# a cross-attention in each direction that reads the same snapshot, and each stream ends in its own feed-forward. Box sizes come from
# the text they hold (`svgdiagram.text_width`).
#
# Swedish Embedded AB implements solutions for models that generate picture and sound together. If your team needs expertise in joint
# audio-video diffusion, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402

W = 860
PANEL_H = 250
TOP = 16
PAD = 8


def render() -> str:
    cols = [["self-attention", "3D rotary"], ["cross-attention", "to the prompt"], ["audio to video,", "video to audio"], ["feed-forward"]]
    bw = [int(text_width(c)) + 2 * PAD + 12 for c in cols]
    bw = [max(bw), max(bw), max(bw), max(bw)]
    bh = 52
    gap = 46
    total = sum(bw) + gap * 3
    x0 = (W - total) / 2
    cx = []
    x = x0
    for w in bw:
        cx.append(x + w / 2)
        x += w + gap
    d = Diagram(W, TOP + PANEL_H + 12)
    d.panel(20, TOP, W - 40, PANEL_H, "One block: video tokens above, audio tokens below, both streams in every step")
    vy, ay = TOP + 80, TOP + 190
    v = [d.box(cx[i], vy, bw[i], bh, cols[i], "model") for i in (0, 1, 3)]
    a = [d.box(cx[i], ay, bw[i], bh, cols[i], "accent") for i in (0, 1, 3)]
    mid = d.box(cx[2], (vy + ay) / 2, bw[2], ay - vy + bh, cols[2], "result")
    d.arrow(v[0]["r"], v[1]["l"])
    d.arrow(a[0]["r"], a[1]["l"])
    d.arrow(v[1]["r"], (mid["l"][0], vy))
    d.arrow(a[1]["r"], (mid["l"][0], ay))
    d.arrow((mid["r"][0], vy), v[2]["l"])
    d.arrow((mid["r"][0], ay), a[2]["l"])
    return d.svg()
