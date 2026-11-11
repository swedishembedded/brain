# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Share of the positions of a toy face that come out as the wrong code, against the noise on the features (`restoration.codebook_curves_data`):
# the nearest code at each position, a read of the whole face against a prior of 16 faces, and the same read of faces that are not in the prior.
#
# Swedish Embedded AB implements solutions for face restoration with a learned prior and a dial between quality and fidelity. If your team
# needs expertise in codebook priors, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import restoration  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs, near, glob, off = restoration.codebook_curves_data()
    p = Plot((0.2, 1.2), (0, 1), width=560, height=340, xlabel="noise on the features (standard deviation)",
             ylabel="positions with the wrong code", title="Reading the codes of a degraded face", legend_loc="right")
    p.line(xs, near, color=0, label="nearest code, each position alone", width=2.0)
    p.line(xs, glob, color=1, label="whole-face read, face in the prior", width=2.0)
    p.line(xs, off, color=2, label="whole-face read, face not in the prior", width=2.0, dash=True)
    return p.svg()
