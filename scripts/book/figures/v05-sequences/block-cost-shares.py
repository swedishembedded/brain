# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Where the parameters and the forward FLOPs of one block go for the four stated configurations of the chapter: two bars for each, the
# upper for the parameters (attention, feed-forward network, norm gains) and the lower for the FLOPs (projections, score and average
# products with the softmax, feed-forward products, norms, adds and activation). Numbers from `transformer_block.share_data`.
#
# Swedish Embedded AB implements solutions for sizing the compute and memory of transformer models. If your team needs expertise in
# the cost accounting of attention and feed-forward layers, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import transformer_block as tb  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, X0, BAR_W, BAR_H, ROW = 940, 190, 620, 26, 36
PARAM_KINDS = ("accent", "model", "plain")        # attention, feed-forward network, norm gains
FLOP_KINDS = ("accent", "result", "model", "plain")   # projections, score products, feed-forward products, other


def segments(d, y, shares, kinds, labels):
    x = X0
    for share, kind, label in zip(shares, kinds, labels):
        w = share * BAR_W
        d.swatch(x, y - BAR_H / 2, w, BAR_H, kind)
        if w >= 34:
            d.text(x + w / 2, y, [f"{share * 100:.0f}%"], size=12)
        x += w


def render() -> str:
    data = tb.share_data()
    height = 52 + len(data) * (2 * ROW + 14) + 70
    d = Diagram(W, height)
    y = 52
    for label, T, pshare, fshare in data:
        d.text(X0 - 14, y + ROW / 2 - 8, [label, f"T = {T:,}"], size=13, anchor="end")
        segments(d, y, pshare, PARAM_KINDS, ("attention", "feed-forward", "norm gains"))
        d.text(X0 + BAR_W + 12, y, ["parameters"], size=12, color=MUTED, anchor="start")
        segments(d, y + ROW, fshare, FLOP_KINDS, ("projections", "scores", "feed-forward", "other"))
        d.text(X0 + BAR_W + 12, y + ROW, ["FLOPs"], size=12, color=MUTED, anchor="start")
        y += 2 * ROW + 14
    # legend, two columns
    ly = y + 4
    entries = (("accent", "attention projections"), ("result", "score and average products, softmax"),
               ("model", "feed-forward network"), ("plain", "norm gains; norms, adds and activation"))
    for i, (kind, name) in enumerate(entries):
        lx, row = X0 + (i % 2) * 330, ly + (i // 2) * 24
        d.swatch(lx, row, 16, 14, kind)
        d.text(lx + 24, row + 7, [name], size=12, anchor="start")
    return d.svg()
