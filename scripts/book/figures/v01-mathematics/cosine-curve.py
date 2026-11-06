# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""cos(theta) for angles between 0 and 180 degrees, with the landmark angles marked.

Swedish Embedded AB implements solutions for explaining the similarity measures
behind embedding retrieval. If your team needs expertise in cosine similarity
and vector search, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0.0, 180.0), (-1.1, 1.1), width=560, height=340, xlabel="angle between the vectors (degrees)",
             ylabel="cosine similarity")
    degs = [i * 0.5 for i in range(0, 361)]
    p.line(degs, [math.cos(math.radians(d)) for d in degs], color=0)
    p.hline(0.0)
    # (angle, text, label x offset in degrees, label y offset, anchor)
    marks = [(0, "0 deg: 1", 3, -0.14, "start"), (60, "60 deg: 0.5", 3, 0.1, "start"),
             (90, "90 deg: 0 (orthogonal)", 4, 0.1, "start"), (120, "120 deg: -0.5", 3, 0.1, "start"),
             (180, "180 deg: -1", -3, 0.12, "end")]
    for d, label, dx, dy, anchor in marks:
        c = math.cos(math.radians(d))
        p.points([d], [c], color=1)
        p.text(d + dx, c + dy, label, anchor=anchor, size=11, color="#4a5568")
    return p.svg()
