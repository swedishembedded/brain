# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Ten Slaney mel filters from 0 to 4000 Hz, area-normalised, evaluated on the bins of a 1024-point transform of 8000 Hz audio: each
# triangle starts at the centre of the one before the previous and ends at the centre of the one after the next. The data are
# `spectrograms.figure_triangles`.
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
    freqs, rows = sp.figure_triangles(10, 4000.0, "slaney")
    top = 1000 * max(max(r) for r in rows)
    p = Plot((0, 4000), (0, round(top * 1.1, 1)), width=560, height=300, xlabel="frequency (Hz)", ylabel="weight (per kHz)",
             title="Ten Slaney mel filters, 0 to 4000 Hz", legend_loc=None)
    for i, row in enumerate(rows):
        p.line(freqs, [1000 * v for v in row], color=i % 6, width=1.8)
    return p.svg()
