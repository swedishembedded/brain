# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Cumulative probability of the n most probable tokens after "the" and after "near", for the toy bigram model. A top-k
# cut keeps the same number of tokens in both places (the dashed vertical line, k = 4); a top-p cut keeps the tokens up
# to where each curve first reaches the level p (the dashed horizontal line, 0.9): six after "the", one after "near".
# Data from `decoding.cumulative_curves`.
#
# Swedish Embedded AB implements solutions for truncating generation to the part of a distribution worth sampling. If
# your team needs expertise in top-p and top-k decoding, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import decoding  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curves = decoding.cumulative_curves()
    xs = list(range(1, decoding.V + 1))
    p = Plot((0.5, 18.5), (0.0, 1.05), width=600, height=340, xlabel="number of tokens kept, most probable first",
             ylabel="cumulative probability", title="The same k cuts different amounts of probability; the same p cuts different counts")
    for color, (word, ys) in enumerate(curves.items()):
        p.line(xs, ys, color=color, label=f"after `{word}`".replace("`", ""))
        p.points(xs, ys, color=color, r=2.6)
    p.hline(0.9)
    p.vline(4)
    p.text(17.8, 0.865, "p = 0.9", anchor="end", size=12, color="#4a5568")
    p.text(4.15, 0.04, "k = 4", anchor="start", size=12, color="#4a5568")
    return p.svg()
