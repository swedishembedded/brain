# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The path of a series through each of the four pretrained forecasters of brain, in five columns that line up: how the series is scaled, how it
# becomes tokens, what the backbone is, what the head emits and how the output returns to the units of the series.
#
# Swedish Embedded AB implements solutions for forecasting with pretrained models whose behaviour is checked rather than assumed. If your team
# needs expertise in time-series foundation models and their evaluation, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402

LANES = (
    ("Chronos-2", (["standardise the", "context, arcsinh"], ["patches of 16 bars,", "one register token"], ["12 blocks: time and", "group attention"],
                   ["21 quantiles", "for each bar"], ["sinh, then", "loc + scale x"])),
    ("TimesFM-3", (["running mean and", "deviation per patch"], ["patches of 32 bars,", "a slot for the future"], ["20 blocks: causal, variate", "attention, feed-forward"],
                   ["9 quantiles, 64 bars", "for each patch"], ["undo the scaling,", "cross-fade patches"])),
    ("FinCast", (["mean and deviation", "of the first patch"], ["patches of 32 bars,", "padded to 512"], ["50 blocks: attention and", "top 2 of 4 experts"],
                 ["mean and 9 quantiles,", "128 bars at once"], ["undo the scaling,", "feed the mean back"])),
    ("Kronos", (["z-score of the bars,", "clipped to +-5"], ["sign code of each bar:", "tokens s1 and s2"], ["8 blocks and a", "dependency layer"],
                ["logits of s1, then", "of s2 given s1"], ["sample, decode the", "bars, undo the z-score"])),
)
HEADERS = ("scale", "tokens", "backbone", "head", "output")
GAP, ROW_H, BOX_H, TOP, LEFT = 26, 82, 50, 44, 16


def render() -> str:
    widths = [max(text_width(lane[1][c]) for lane in LANES) + 10 for c in range(5)]
    label_w = max(text_width([lane[0]]) for lane in LANES) + 10
    width = int(LEFT + label_w + 14 + sum(widths) + GAP * 4 + LEFT)
    d = Diagram(width, TOP + ROW_H * len(LANES) + 8)
    x = LEFT + label_w + 14
    centres = []
    for c in range(5):
        centres.append(x + widths[c] / 2)
        d.text(centres[c], 22, [HEADERS[c]], weight="bold")
        x += widths[c] + GAP
    for r, (name, cells) in enumerate(LANES):
        cy = TOP + r * ROW_H + ROW_H / 2
        d.text(LEFT + label_w / 2, cy, [name], weight="bold")
        prev = None
        for c in range(5):
            kind = "model" if c == 2 else ("result" if c == 4 else "plain")
            b = d.box(centres[c], cy, widths[c], BOX_H, cells[c], kind)
            if prev is not None:
                d.arrow(prev["r"], b["l"])
            prev = b
    return d.svg()
