# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The image radius in units of the focal length of a ray at angle theta from the axis: tan(theta) for the pinhole, theta for the
# equidistant fisheye, and the Kannala-Brandt polynomial, which stops being a lens where its slope reaches zero. The data are
# `camera_geometry.radius_curves`.
#
# Swedish Embedded AB implements solutions for wide-angle and fisheye camera models. If your team needs expertise in lens
# calibration, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import camera_geometry  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    degs, pin, eq, kbv, fold = camera_geometry.radius_curves()
    p = Plot((0, 150), (0, 3), width=560, height=340, xlabel="angle from the optical axis (degrees)",
             ylabel="image radius / f", title="Where a ray lands", legend_loc="ul")
    shown = [(d, t) for d, t in pin if t <= 3.0]
    p.line([d for d, t in shown], [t for d, t in shown], color=0, label="pinhole: tan(theta)", width=1.9)
    p.line(degs, eq, color=1, label="equidistant: theta", width=1.9)
    p.line(degs, kbv, color=2, label="Kannala-Brandt polynomial", width=1.9)
    p.vline(fold)
    p.text(fold - 2, 0.35, "polynomial folds at 126.4 degrees", anchor="end")
    return p.svg()
