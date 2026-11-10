# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The target logit of the angular-margin loss, divided by the scale, as the angle between a sample and its own class centre grows
# (`faces.margin_curve_data`): the plain cosine, the margin form cos(theta + m) and the margin form with the clamp.
#
# Swedish Embedded AB implements solutions for recognition losses whose margins are checked against a hand computation. If your team
# needs expertise in metric learning for faces, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import faces  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    th, plain, marg, clamp = faces.margin_curve_data()
    p = Plot((0, 180), (-1.3, 1.1), width=620, height=380, xlabel="angle between the sample and its own centre, degrees",
             ylabel="target logit / scale", legend_loc="ll")
    p.line(th, plain, color=5, label="plain cosine")
    p.line(th, marg, color=0, label="cos(theta + 0.5)")
    p.line(th, clamp, color=1, dash=True, label="with the clamp")
    sw = math.degrees(math.pi - 0.5)
    p.vline(sw)
    p.text(sw - 2, 0.55, "theta = pi - m", anchor="end", size=11)
    p.text(sw - 2, 0.38, "151.35 degrees", anchor="end", size=11)
    return p.svg()
