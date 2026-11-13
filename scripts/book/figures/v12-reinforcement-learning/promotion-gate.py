# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The decision of the promotion gate: a candidate and the incumbent are scored on the same frozen probes, four bars are checked in a fixed
# order, and the first that fails names the cause; only a candidate that clears all of them replaces the incumbent.
#
# Swedish Embedded AB implements solutions for deciding, with a stated test, whether a retrained model may replace the one in use. If your
# team needs expertise in gating model promotions, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402


def render() -> str:
    pad, gap, h = 20, 34, 84
    left = [["candidate", "(trained from", "the incumbent)"], ["incumbent", "(what is served)"]]
    probes = ["same frozen probes,", "same order, both arms", "decoded from disk"]
    bars = [["1 significance", "sign test on the", "episode's probes:", "p <= 0.05"],
            ["2 effect size", "mean gain on those", "probes >= 0.15"],
            ["3 pooled anchor", "mean drop over all", "earlier blocks", "<= the budget"],
            ["4 block bar", "worst single", "block drop <= 0.20"]]
    result = ["promote:", "the candidate", "becomes the", "incumbent"]
    cols = [left[0] + left[1], probes] + bars + [result]
    widths = [int(max(text_width(c) for c in col) + 24) for col in (left[0], left[1])]
    lw = max(widths)
    pw = int(text_width(probes) + 24)
    bw = max(int(text_width(b) + 24) for b in bars)
    rw = int(text_width(result) + 24)
    width = pad * 2 + lw + pw + bw * 4 + rw + gap * 6
    ymid = pad + 90
    ytop, ybot = ymid - 54, ymid + 54
    rej_y = ymid + h / 2 + 90
    d = Diagram(int(width), int(rej_y + 40 + pad))
    x = pad
    cx_left = x + lw / 2
    x += lw + gap
    cx_probe = x + pw / 2
    x += pw + gap
    cx_bars = []
    for _ in bars:
        cx_bars.append(x + bw / 2)
        x += bw + gap
    cx_res = x + rw / 2
    cand = d.box(cx_left, ytop, lw, 66, left[0], "plain")
    inc = d.box(cx_left, ybot, lw, 66, left[1], "plain")
    pr = d.box(cx_probe, ymid, pw, h, probes, "model")
    d.path([cand["r"], (cand["r"][0] + gap / 2, ytop), (cand["r"][0] + gap / 2, ymid - 14), (pr["l"][0], ymid - 14)])
    d.path([inc["r"], (inc["r"][0] + gap / 2, ybot), (inc["r"][0] + gap / 2, ymid + 14), (pr["l"][0], ymid + 14)])
    prev = pr
    boxes = []
    for i, b in enumerate(bars):
        box = d.box(cx_bars[i], ymid, bw, h, b, "accent")
        d.arrow(prev["r"], box["l"], label="pass" if i else None)
        boxes.append(box)
        prev = box
    res = d.box(cx_res, ymid, rw, h, result, "result")
    d.arrow(prev["r"], res["l"], label="pass")
    rej = d.box((cx_bars[0] + cx_bars[-1]) / 2, rej_y, cx_bars[-1] - cx_bars[0] + bw, 44,
                ["reject, naming the first bar that failed; every number is still reported"], "plain")
    for box in boxes:
        d.arrow(box["b"], (box["b"][0], rej["t"][1]), label="fail", side="right")
    return d.svg()
