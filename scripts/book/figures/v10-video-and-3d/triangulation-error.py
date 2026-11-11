# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Root mean square error of the depth of a point 5 m away, triangulated from two cameras, against the angle between the two rays:
# the linear method on noisy pixels against the first-order prediction sqrt(2) sigma Z^2 / (f B). The data are
# `structure_from_motion.triangulation_data`.
#
# Swedish Embedded AB implements solutions for photogrammetry and stereo geometry. If your team needs expertise in triangulation
# accuracy, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import structure_from_motion as sfm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    d = sfm.triangulation_data()
    p = Plot((0.3, 30), (0.01, 1.5), width=560, height=340, xlabel="angle between the two rays (degrees)",
             ylabel="RMS depth error (m)", title="Depth error of a triangulated point, 5 m away", legend_loc="ur", xlog=True,
             ylog=True)
    p.line([r[1] for r in d], [r[4] for r in d], color=3, label="prediction sqrt(2) sigma Z^2 / (f B)", dash=True, width=1.4)
    p.line([r[1] for r in d], [r[2] for r in d], color=0, label="linear triangulation, 200 trials", width=1.9)
    p.points([r[1] for r in d], [r[2] for r in d], color=0)
    return p.svg()
