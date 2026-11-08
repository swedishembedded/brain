# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The chain rule applied to a sentence: one row for each token, with the tokens before it as its context, the
# conditional probability of the token given that context and the running product. The numbers are those of
# `language_modeling.chain_rule`, computed from the toy language's own sentence probabilities.
#
# Swedish Embedded AB implements solutions for scoring and generating text with language models. If your team needs
# expertise in autoregressive models and their probabilities, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import language_modeling as lm  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W = 780
ROW_H, TOP = 46, 96
CTX_END, TOK_W, TOK_H, GAP = 372, 62, 28, 6
TARGET_X, FACTOR_X, PRODUCT_X = 452, 576, 690


def render() -> str:
    sent = [lm.ID[t] for t in ["the", "cat", "sees", "fish", "."]]
    words = ["<s>"] + [lm.TOKENS[i] for i in sent]
    n = len(sent)
    h = TOP + n * ROW_H + 40
    d = Diagram(W, h)
    d.panel(10, 10, W - 20, h - 20, "P(sentence) = the product of one next-token probability for each token")
    for x, label in ((CTX_END - 120, "context (the tokens before)"), (TARGET_X, "token"),
                     (FACTOR_X, "p(token | context)"), (PRODUCT_X, "product so far")):
        d.text(x, 62, [label], size=12, color="#4a5568")
    running = Fraction(1)
    for t in range(n):
        cy = TOP + t * ROW_H
        cond = lm._prefix_prob(sent[:t + 1]) / lm._prefix_prob(sent[:t])
        running *= cond
        count = t + 1                              # `<s>` and the t tokens before this one
        right = None
        for j in range(count):
            cx = CTX_END - TOK_W / 2 - (count - 1 - j) * (TOK_W + GAP)
            right = d.box(cx, cy, TOK_W, TOK_H, [words[j]], "plain")
        tgt = d.box(TARGET_X, cy, TOK_W, TOK_H, [words[t + 1]], "result")
        d.arrow(right["r"], tgt["l"], gap=4)
        d.text(FACTOR_X, cy, [str(cond)], size=14)
        d.text(PRODUCT_X, cy, [str(running)], size=14, weight="bold" if t == n - 1 else "normal")
    d.text(W / 2, TOP + n * ROW_H + 8, ["every factor reads only what came before its token"], size=12, color="#2d3748")
    return d.svg()
