# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A decoder-only language model as the chapter builds it: token ids, the embedding table, L identical blocks (each with its own weights)
# on one residual stream, a final normalization and the output head that turns the last stream into logits.
#
# Swedish Embedded AB implements solutions for generative language models, from the architecture to the training loop. If your team needs
# expertise in decoder-only transformers, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 250
Y = 105


def render() -> str:
    d = Diagram(W, H)
    ids = d.oval(60, Y, 92, 52, ["token ids", "T"], "plain")
    emb = d.box(170, Y, 100, 56, ["embedding", "table E"], "model")
    d.arrow(ids["r"], emb["l"])
    b1 = d.box(350, Y, 96, 56, ["block 1"], "accent")
    b2 = d.box(480, Y, 96, 56, ["block 2"], "accent")
    bl = d.box(640, Y, 96, 56, ["block L"], "accent")
    d.group(284, Y - 56, 424, 112)
    d.text(496, Y - 74, ["L blocks, each with its own weights, one stream"], size=12, color=MUTED)
    d.arrow(emb["r"], b1["l"], "[T, D]", side="below")
    d.arrow(b1["r"], b2["l"])
    d.arrow(b2["r"], bl["l"], "...", side="above")
    norm = d.box(785, Y, 96, 56, ["final", "RMS norm"], "model")
    head = d.box(905, Y, 96, 56, ["output head", "W_C"], "model")
    out = d.oval(905, Y + 82, 80, 44, ["logits", "T x V"], "result")
    d.arrow(bl["r"], norm["l"])
    d.arrow(norm["r"], head["l"])
    d.arrow(head["b"], out["t"])
    return d.svg()
