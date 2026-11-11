# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One layer of the two-way transformer of the mask decoder: the prompt and output tokens attend to themselves and then to the image
# cells, and the image cells attend back to the tokens. Shapes are those of the large model with one click (8 tokens, 64 x 64 cells).
#
# Swedish Embedded AB implements solutions for attention between a few prompts and a large image. If your team needs expertise in
# cross-attention for dense prediction, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 900, 500


def render() -> str:
    d = Diagram(W, H)
    xt, xi = 230, 670
    tok = d.oval(xt, 44, 190, 52, ["tokens", "8 x 256"], "accent")
    img = d.oval(xi, 44, 190, 52, ["image cells + position", "4,096 x 256"], "accent")
    sa = d.box(xt, 128, 220, 56, ["self-attention", "tokens read tokens"])
    t2i = d.box(xt, 222, 220, 56, ["token-to-image attention", "queries: tokens; keys, values: cells"])
    mlp = d.box(xt, 316, 220, 56, ["MLP", "256 -> 2,048 -> 256"])
    i2t = d.box(xi, 316, 220, 56, ["image-to-token attention", "queries: cells; keys, values: tokens"])
    tout = d.oval(xt, 450, 190, 52, ["tokens out", "8 x 256"], "result")
    iout = d.oval(xi, 450, 190, 52, ["image cells out", "4,096 x 256"], "result")
    d.arrow(tok["b"], sa["t"])
    d.arrow(sa["b"], t2i["t"])
    d.arrow(t2i["b"], mlp["t"])
    # the image stream feeds the keys and values of the token-to-image attention, then waits for the updated tokens
    d.path([(xi, img["b"][1]), (xi, 222), (t2i["r"][0], 222)], radius=8, dash="5 3")
    d.text(450, 208, ["keys and values"], size=12, color=MUTED)
    d.arrow(img["b"], i2t["t"], gap=3)
    d.arrow(mlp["r"], i2t["l"], label="keys and values", above=True)
    d.arrow(mlp["b"], tout["t"])
    d.arrow(i2t["b"], iout["t"])
    d.text(xt + 14, 388, ["LayerNorm after each step,", "a residual add around each"], size=12, color=MUTED, anchor="start")
    d.text(xi + 14, 388, ["the same, on the 4,096 cells"], size=12, color=MUTED, anchor="start")
    d.text(W / 2, H - 12, ["Two such layers, then one more token-to-image attention; the tokens that leave are the ones that make masks."],
           size=12, color=MUTED)
    return d.svg()
