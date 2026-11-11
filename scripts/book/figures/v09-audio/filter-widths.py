# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The width in hertz of each of 40 mel filters over 0 to 8000 Hz, against its centre frequency, for the HTK scale and for the Slaney
# scale. A filter is as wide as the distance between the edges on either side of it. The data are `spectrograms.filter_widths_data`.
#
# Swedish Embedded AB implements solutions for turning raw audio into the features a speech model was trained on. If your team needs
# expertise in audio front ends for speech and music models, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spectrograms as sp  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = sp.filter_widths_data(40, 8000)
    p = Plot((0, 8000), (0, 1600), width=560, height=320, xlabel="filter centre (Hz)", ylabel="filter width (Hz)",
             title="Width of 40 mel filters, 0 to 8000 Hz", legend_loc="ul")
    for i, (scale, name) in enumerate((("htk", "HTK scale"), ("slaney", "Slaney scale"))):
        xs = [c for c, _ in data[scale]]
        ys = [w for _, w in data[scale]]
        p.line(xs, ys, color=i, label=name, width=2.0)
        p.points(xs, ys, color=i, r=2.5)
    return p.svg()
