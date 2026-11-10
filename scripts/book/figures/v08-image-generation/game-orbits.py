# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The smallest game with no minimum: x lowers x y while y raises it, each by gradient steps. Played simultaneously the pair spirals
# outward; played alternately it stays on a closed orbit and never settles at the saddle (0, 0). The data are `gans.bilinear_data`.
#
# Swedish Embedded AB implements solutions for generative models whose adversarial training is specified, checked and made
# reproducible. If your team needs expertise in generative adversarial networks and image synthesis, you can procure our services
# by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gans  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    sim, alt = gans.bilinear_data()
    p = Plot((-4, 4), (-4, 4), width=420, height=420, xlabel="x", ylabel="y", title="Two ways to play x y", legend_loc="lr")
    p.line([a for a, _ in sim], [b for _, b in sim], color=1, label="simultaneous, 200 steps", width=1.6)
    p.line([a for a, _ in alt], [b for _, b in alt], color=0, label="alternating, 200 steps", width=2.2)
    p.points([0.0], [0.0], color=5, label="saddle (0, 0)", r=4)
    return p.svg()
