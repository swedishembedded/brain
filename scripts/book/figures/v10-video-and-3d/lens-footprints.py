# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The width in pixels, at half the peak response, of one Gaussian (standard deviation 0.3 at distance 4, f = 200) as it moves out from
# the axis, seen by a pinhole and by an equidistant fisheye, each evaluated exactly along the pixel rays. Data: `gaussian_splatting._lens_curves`.
#
# Swedish Embedded AB implements renderers that image through any lens without resampling the photograph. If your team needs
# expertise in fisheye and wide-angle reconstruction, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gaussian_splatting as gs  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = gs._lens_curves()
    th = [r[0] for r in rows]
    p = Plot((0, 85), (10, 1000), ylog=True, width=560, height=340, xlabel="angle of the Gaussian from the optical axis (degrees)",
             ylabel="half-width at half response (pixels)", title="One Gaussian, two lenses", legend_loc="ul")
    p.line(th, [r[2] for r in rows], color=0, label="pinhole", width=2.0)
    p.line(th, [r[1] for r in rows], color=1, label="equidistant fisheye", width=2.0)
    return p.svg()
