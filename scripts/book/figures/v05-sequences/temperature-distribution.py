# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The next-token distribution after the word "the" of the toy bigram model, with the tokens in order of decreasing
# probability at temperature 1, drawn at three temperatures: a low one concentrates the probability on the first
# few tokens, a high one lifts the tail. Data from `decoding.temperature_curves`, the same arithmetic as the table of
# the chapter's temperature section.
#
# Swedish Embedded AB implements solutions for controlling the randomness of generated text. If your team needs
# expertise in sampling temperature and decoding, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import decoding  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ranks, curves = decoding.temperature_curves((0.5, 1.0, 2.0))
    p = Plot((0.5, 18.5), (0.0, 0.32), width=600, height=340, xlabel="token, from most to least probable at temperature 1",
             ylabel="probability", title="Dividing the logits by a temperature moves probability along the ranking")
    for color, (tau, ys) in enumerate(curves.items()):
        label = f"temperature {tau:g}"
        p.line(ranks, ys, color=color, label=label)
        p.points(ranks, ys, color=color, r=2.6)
    p.vline(6.5)
    p.text(6.7, 0.30, "six nouns | twelve other tokens", anchor="start", size=12, color="#4a5568")
    return p.svg()
