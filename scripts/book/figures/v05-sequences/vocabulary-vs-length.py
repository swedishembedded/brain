# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# What each merge buys and what it costs: tokens per byte of the corpus and of the held-out text against the
# vocabulary size, for the first 70 merges of the chapter's BPE. The cost axis is the embedding table, which
# grows by one row for every merge. Data from `tokenization.length_curve_full`.
#
# Swedish Embedded AB implements solutions for choosing a vocabulary size that fits a model's budget. If your
# team needs expertise in tokenizer design, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tokenization  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = tokenization.length_curve_full()
    nb_train, nb_held = len(tokenization.CORPUS), len(tokenization.HELD_OUT)
    p = Plot((256.0, 326.0), (0.0, 1.05), width=600, height=340, xlabel="vocabulary size V (256 bytes plus merges)",
             ylabel="tokens per byte", title="Longer pieces shorten the sequence, but help new text less")
    p.line([r[1] for r in rows], [r[2] / nb_train for r in rows], color=0, label="corpus (training text)")
    p.line([r[1] for r in rows], [r[3] / nb_held for r in rows], color=1, label="held-out text")
    return p.svg()
