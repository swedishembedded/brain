# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The root-cause chain of the chapter "A Catalogue of Expensive Mistakes": from the symptom of a diverging fine-tune,
# through the measurement that located it and the mechanism, back to the decision that made it possible, with the three
# reasons the existing checks could not see it underneath and the regression gate that closes the chain.
#
# Swedish Embedded AB implements solutions for root-cause analysis of failing machine learning systems for its clients.
# If your team needs expertise in finding why a training run diverges then you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, RED, Diagram  # noqa: E402

W, H = 1000, 400
XS = (125, 375, 625, 875)
Y1, Y2, Y3 = 70, 215, 340
BW, BH = 210, 70


def render() -> str:
    d = Diagram(W, H)
    symptom = d.box(XS[0], Y1, BW, BH, ["symptom", "loss rises at every", "rate that moves"], "accent")
    measured = d.box(XS[1], Y1, BW, BH, ["measurement", "gradient norm 2.8e5", "where the true one is 0.98"], "model")
    mechanism = d.box(XS[2], Y1, BW, BH, ["mechanism", "row chunks added into", "an uncleared buffer"], "model")
    decision = d.box(XS[3], Y1, BW, BH, ["decision", "clear dropped: each", "chunk 'written whole'"], "model")
    d.arrow(symptom["r"], measured["l"], "why?")
    d.arrow(measured["r"], mechanism["l"], "why?")
    d.arrow(mechanism["r"], decision["l"], "why?")
    misses = [
        d.note(XS[0], Y2, BW, BH, ["CPU memory is", "zero-filled, so an add", "into it is a copy"]),
        d.note(XS[1], Y2, BW, BH, ["the gates ran 12 tokens,", "which never select", "the chunked path"]),
        d.note(XS[2], Y2, BW, BH, ["the evidence was the", "loss bits, which cannot", "see a gradient"]),
    ]
    for m in misses:
        d.line(m["t"], (m["t"][0], Y1 + BH / 2 + 8))
    gate = d.box((XS[1] + XS[2]) / 2, Y3, 460, 60, ["regression gate: device against CPU at the published geometry,",
                                                    "more than one chunk, memory that is not zero"], "result")
    d.path([decision["b"], (XS[3], Y3), gate["r"]], color=RED)
    d.path([misses[0]["b"], (XS[0], Y3), gate["l"]], dash="4 3", color=MUTED)
    for m in misses[1:]:
        d.arrow(m["b"], (m["b"][0], Y3 - 30), dash="4 3", color=MUTED)
    return d.svg()
