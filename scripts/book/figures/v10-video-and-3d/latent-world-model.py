# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The latent world model of the chapter as brain imports it: a tokenizer turns each 64 x 64 frame into 16 x 16 codes of a 1024-entry
# codebook, an action-conditioned masked-token transformer predicts the codes of the next frame, and the tokenizer's decoder draws them.
#
# Swedish Embedded AB implements solutions for tokenized world models whose parts are checked against a reference one by one. If your
# team needs expertise in discrete latent video models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402


def render() -> str:
    cols = [
        (["past frames", "64 x 64 x 3 each"], "plain"),
        (["tokenizer", "encoder, 8 blocks", "cosine VQ"], "model"),
        (["past tokens", "256 codes a frame", "of 1024"], "result"),
        (["dynamics", "12 blocks, width 519", "causal in time"], "accent"),
        (["next tokens", "256 codes"], "result"),
        (["tokenizer", "decoder, 8 blocks"], "model"),
        (["next frame", "64 x 64 x 3"], "plain"),
    ]
    gap, pad, h = 30, 24, 74
    widths = [int(text_width(c[0]) + 8 + 8) for c in cols]
    xs, x = [], pad
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    width = int(x - gap + pad)
    y = pad + h / 2 + 70
    d = Diagram(width, int(y + h / 2 + pad))
    bx = [d.box(xs[i], y, widths[i], h, cols[i][0], cols[i][1]) for i in range(len(cols))]
    for i in range(len(cols) - 1):
        d.arrow(bx[i]["r"], bx[i + 1]["l"])
    act = d.box(xs[3], pad + 22, int(text_width(["action one-hot", "(7 for CoinRun)"]) + 16), 44, ["action one-hot", "(7 for CoinRun)"], "plain")
    d.arrow(act["b"], bx[3]["t"], label="appended to every token", side="right")
    return d.svg()
