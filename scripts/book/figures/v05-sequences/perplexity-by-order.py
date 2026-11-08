# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Test perplexity against the order n of the counting model, for add-alpha smoothing and for interpolation with
# back-off, with the network and the process floor as references. Orders 2 to 5: the unigram model (perplexity
# about 15) would flatten the rest, and the unsmoothed estimates are infinite from order 2 on. Data from
# `language_modeling.perplexity_by_order`, the models trained on the 40 printed sentences.
#
# Swedish Embedded AB implements solutions for evaluating language models on held-out text. If your team needs
# expertise in perplexity and smoothing, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import language_modeling as lm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = lm.ppl_by_order_data()
    orders = data["orders"][1:]
    p = Plot((1.5, 5.5), (2.0, 6.5), width=600, height=340, xlabel="order n of the counting model (context of n - 1 tokens)",
             ylabel="test perplexity", title="More context stops helping when the counts run out")
    p.line(orders, data["addk"][1:], color=0, label="add-alpha smoothing")
    p.points(orders, data["addk"][1:], color=0)
    p.line(orders, data["interp"][1:], color=1, label="interpolation with back-off")
    p.points(orders, data["interp"][1:], color=1)
    p.hline(data["mlp"], color="#2f855a")
    p.hline(data["floor"], color="#718096")
    p.text(5.45, data["mlp"] + 0.08, "network, window of 4 tokens", anchor="end", size=12, color="#2f855a")
    p.text(5.45, data["floor"] + 0.08, "floor of the process", anchor="end", size=12, color="#4a5568")
    return p.svg()
