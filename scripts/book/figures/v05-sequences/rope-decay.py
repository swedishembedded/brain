# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The rotary score of a query and a key that agree on every pair, against their distance j - i, at width 64: the mean over the 32
# pairs of cos((j - i) w_m), for the base 10^4 and the base 10^6. Numbers from `positions.decay_data`, the same function as the
# table of the chapter.
#
# Swedish Embedded AB implements solutions for long-context transformer models, from the position encoding to the kernels. If your
# team needs expertise in rotary position embeddings, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import positions as ps  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs, (low, high) = ps.decay_data(64, (1e4, 1e6), 512)
    p = Plot((0, 512), (0.0, 1.05), width=640, height=360, xlabel="distance j - i between key and query",
             ylabel="score / score at distance 0", title="Rotary score against distance, width 64")
    p.line(xs, low, color=1, label="base 10^4", width=1.6)
    p.line(xs, high, color=0, label="base 10^6", width=1.6)
    return p.svg()
