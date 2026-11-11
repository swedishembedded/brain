# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Image 1 of the pair with the epipolar lines of ten view-0 keypoints under the estimated essential matrix: a correct match lies on
# its line, a wrong one is somewhere else. The data are `structure_from_motion.epipolar_figure_data`, the pair of the chapter's
# RANSAC example.
#
# Swedish Embedded AB implements solutions for photogrammetry and structure from motion. If your team needs expertise in
# epipolar geometry and robust matching, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import structure_from_motion as sfm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = sfm.epipolar_figure_data()
    h = float(sfm.HEIGHT)
    p = Plot((0, sfm.WIDTH), (0, sfm.HEIGHT), width=600, height=450, xlabel="u (pixels)", ylabel="480 - v (pixels above the bottom edge)",
             title="Epipolar lines of ten view-0 keypoints, drawn in view 1", legend_loc="ul")
    for k, (u0, v0, line, (u1, v1), ok) in enumerate(data):
        color = 0 if ok else 2
        us = [0.0, float(sfm.WIDTH)]
        vs = [h + (line[0] * u + line[2]) / line[1] for u in us]
        p.line(us, vs, color=color, width=1.2)
    p.points([d[3][0] for d in data if d[4]], [h - d[3][1] for d in data if d[4]], color=0, label="matched keypoint, correct", r=4)
    p.points([d[3][0] for d in data if not d[4]], [h - d[3][1] for d in data if not d[4]], color=2, label="matched keypoint, wrong", r=4)
    return p.svg()
