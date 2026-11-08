# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The pipeline of the chapter from raw text to a trained generative model: the documents are cut into token ids by the byte-pair
# tokenizer, the ids of all documents (each opened by a separator) become one stream, the stream is packed into rows of T positions
# with inputs, targets, a loss mask and document ids; the GPT turns rows into logits, the masked cross-entropy scores them, AdamW with
# clipping and a learning-rate schedule updates the weights, and checkpoints feed validation and sampling.
#
# Swedish Embedded AB implements solutions for training language models end to end, from raw text to a served checkpoint. If your team
# needs expertise in language-model training pipelines, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, RED, Diagram  # noqa: E402

W, H = 1000, 380
Y1, Y2, Y3 = 62, 182, 314
XS = (110, 340, 570, 800)   # four equal columns, from the left


def render() -> str:
    d = Diagram(W, H)
    # first row, left to right: text becomes rows
    text = d.oval(XS[0], Y1, 150, 56, ["raw text", "documents"], "plain")
    tokz = d.box(XS[1], Y1, 150, 56, ["tokenizer", "byte-pair merges"], "model")
    ids = d.oval(XS[2], Y1, 150, 56, ["token stream", "with separators"], "plain")
    pack = d.box(XS[3], Y1, 150, 56, ["packing", "x, y, mask, doc"], "model")
    d.arrow(text["r"], tokz["l"])
    d.arrow(tokz["r"], ids["l"])
    d.arrow(ids["r"], pack["l"])
    # second row, right to left: rows become a better model
    gpt = d.box(XS[3], Y2, 150, 56, ["GPT", "L blocks, tied head"], "accent")
    loss = d.box(XS[2], Y2, 150, 56, ["masked loss", "mean over supervised"], "result")
    opt = d.box(XS[1], Y2, 150, 56, ["AdamW", "clip, warmup, cosine"], "model")
    ckpt = d.oval(XS[0], Y2, 150, 56, ["checkpoint", "weights, moments"], "plain")
    d.arrow(pack["b"], gpt["t"], "[B, T] ids", side="right")
    d.arrow(gpt["l"], loss["r"], "logits", side="above")
    d.arrow(loss["l"], opt["r"], "gradients", side="above", color=RED)
    d.arrow(opt["l"], ckpt["r"], "every k steps", side="above")
    # the weights go round again
    d.path([(XS[1], Y2 + 28), (XS[1], Y2 + 72), (XS[3], Y2 + 72), (XS[3], Y2 + 28)], color=MUTED, dash="5 3")
    d.text((XS[1] + XS[3]) / 2, Y2 + 90, ["updated weights, next step"], size=12, color=MUTED)
    # what a checkpoint is for
    out = d.oval(XS[0], Y3, 150, 56, ["validation loss", "and generated text"], "result")
    d.arrow(ckpt["b"], out["t"])
    return d.svg()
