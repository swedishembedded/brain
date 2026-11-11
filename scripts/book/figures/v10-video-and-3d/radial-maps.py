# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The radius a lens maps a normalized radius to: r (1 + k1 r^2 + k2 r^4) for a pincushion lens, a mild barrel lens and a barrel lens
# strong enough to fold. The data are `camera_geometry.radial_curves`, the arithmetic of the chapter's distortion section.
#
# Swedish Embedded AB implements solutions for lens modelling and distortion correction. If your team needs expertise in camera
# calibration, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import camera_geometry  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rs, curves, fold = camera_geometry.radial_curves()
    p = Plot((0, 1.6), (0, 1.8), width=560, height=340, xlabel="normalized radius r", ylabel="distorted radius r_d",
             title="Three radial lenses", legend_loc="ul")
    p.line(rs, rs, color=3, label="no lens (r_d = r)", dash=True, width=1.4)
    for i, (name, ys) in enumerate(curves):
        p.line(rs, ys, color=i, label=name, width=1.9)
    peak = fold * (1 - 0.35 * fold * fold)
    p.vline(fold)
    p.points([fold], [peak], color=2)
    p.text(fold + 0.03, 0.2, "fold at r = 0.976", anchor="start")
    return p.svg()
