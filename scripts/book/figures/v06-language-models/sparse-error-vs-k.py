# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Mean relative error of the attention output when each query reads only k keys, against log2 of k, for the k largest true weights
# (the oracle), the k largest scores of the trained indexer and the k most recent keys, on the held-out toy sequence B. The data is
# `sparse_attention.curve_data`, the function behind the table of the chapter.
#
# Swedish Embedded AB implements solutions for serving long-context language models whose attention cost is set by what a token
# needs. If your team needs expertise in sparse attention, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sparse_attention as sa  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = sa.curve_data()
    xs = [math.log2(k) for k in data["k"]]
    p = Plot((-0.2, 7.3), (0, 3), width=620, height=360, xlabel="log2 of k, the number of keys read", ylabel="relative error of the output",
             title="Output error against the number of keys read")
    p.line(xs, data["window"][1], color=3, label="the k most recent keys")
    p.line(xs, data["indexer"][1], color=1, label="top k of the trained indexer")
    p.line(xs, data["oracle"][1], color=0, label="top k of the true weights")
    p.points(xs, data["indexer"][1], color=1)
    p.points(xs, data["oracle"][1], color=0)
    return p.svg()
