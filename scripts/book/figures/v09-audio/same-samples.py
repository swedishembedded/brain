# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A 10 kHz cosine and a 6 kHz cosine over half a millisecond, and the 16 kHz sample times where they take the same values. The data
# are `sound.alias_waves`, the arithmetic of the chapter's first table.
#
# Swedish Embedded AB implements solutions for audio that keeps its meaning when it is resampled. If your team needs expertise in
# sample-rate conversion for speech models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sound  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ts, ten, six, st, sv = sound.alias_waves()
    ms = [t * 1000 for t in ts]
    p = Plot((0, 0.5), (-1.25, 1.25), width=560, height=320, xlabel="time (ms)", ylabel="value",
             title="Two tones, one set of samples", legend_loc="below")
    p.line(ms, ten, color=0, label="10 kHz", width=1.8)
    p.line(ms, six, color=1, label="6 kHz", width=1.8, dash=True)
    p.points([t * 1000 for t in st], sv, color=3, label="samples at 16 kHz", r=4)
    return p.svg()
