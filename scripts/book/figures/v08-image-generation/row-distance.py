# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# For two prompts that exchange two colours, the distance between the final rows of the toy encoder at each of the nine positions,
# under the causal mask and under the full mask (`text_conditioning.order_data`).
#
# Swedish Embedded AB implements solutions for text-conditioned image generators whose prompt interface is specified and checked.
# If your team needs expertise in text encoders for diffusion models, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import text_conditioning  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = text_conditioning.order_data()
    xs = [float(i) for i, *_ in data]
    p = Plot((-0.4, 8.4), (-0.1, 3.2), width=560, height=340, xlabel="position (the colours sit at 2 and 6)",
             ylabel="distance between the two prompts' rows", title="Which rows know about the swap", legend_loc="ur")
    for k, (name, col) in enumerate((("causal mask", 3), ("full mask", 4))):
        ys = [row[col] for row in data]
        p.line(xs, ys, color=k, label=name, width=2.0)
        p.points(xs, ys, color=k, r=3.0)
    return p.svg()
