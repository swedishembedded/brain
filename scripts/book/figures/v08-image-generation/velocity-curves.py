# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The velocity the exact field gives at (x, s) against x at three values of s (solid), and the trained network's (dashed)
# (`flow_matching.velocity_curves_data`, the first run of the chapter's table).
#
# Swedish Embedded AB implements solutions for generative models whose paths, velocity targets and sampling schedules are specified,
# checked and made reproducible. If your team needs expertise in flow matching and image synthesis, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flow_matching  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = flow_matching.velocity_curves_data()
    p = Plot((-3.7, 3.7), (-8, 8), width=560, height=340, xlabel="position x", ylabel="velocity u(x, sigma)",
             title="Exact and learned velocity", legend_loc="below")
    for i, (s, (xs, exact, net)) in enumerate(data.items()):
        p.line(xs, exact, color=i, label=f"exact, sigma = {s}", width=2.2)
        p.line(xs, net, color=i, label=f"network, sigma = {s}", width=1.6, dash=True)
    return p.svg()
