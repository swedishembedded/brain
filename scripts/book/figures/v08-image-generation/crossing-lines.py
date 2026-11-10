# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Twelve pairs (data value, noise value) of the independent coupling joined by straight lines, from s = 0 (data) to s = 1 (noise)
# (`flow_matching.crossing_lines_data`): the lines cross, so a point (x, s) lies on several of them with different slopes.
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
    p = Plot((0, 1), (-4.2, 4.2), width=560, height=340, xlabel="noise fraction sigma (0 = data, 1 = noise)", ylabel="position x_sigma",
             title="Twelve independent pairs, joined by straight lines", legend_loc=None)
    for i, (ss, xs) in enumerate(flow_matching.crossing_lines_data()):
        p.line(ss, xs, color=i % 6, width=1.8)
    return p.svg()
