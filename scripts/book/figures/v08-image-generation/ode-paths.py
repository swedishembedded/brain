# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Paths of the exact velocity field from eight starting values (solid), and the straight chord from each starting value to where
# its path ends (dashed): the field's paths bend toward the clusters and never cross, the chords are the second round's paths
# (`flow_matching.trajectory_data`).
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
    p = Plot((0, 1), (-4.2, 4.2), width=560, height=340, xlabel="noise fraction sigma (1 = noise, 0 = data)", ylabel="position x_sigma",
             title="Paths of the exact field and their chords", legend_loc="below")
    for i, (ss, xs, cs, cx) in enumerate(flow_matching.trajectory_data()):
        p.line(ss, xs, color=0, width=2.0, label="path of the exact field (round 1)" if i == 0 else "")
        p.line(cs, cx, color="#a0aec0", dash=True, width=1.6, label="straight chord (round 2)" if i == 0 else "")
    return p.svg()
