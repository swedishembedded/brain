# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The gain of three 24 kHz to 16 kHz resamplers for a tone at each frequency, measured at the frequency the tone ends up at. The
# data are `sound.gain_curve`, the arithmetic of the chapter's tables.
#
# Swedish Embedded AB implements solutions for sample-rate conversion that keeps the sound and drops only what the new rate cannot
# carry. If your team needs expertise in resampling for speech and music models, you can procure our services by sending an email
# to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sound  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    freqs, series = sound.gain_curve()
    p = Plot((0, 12), (-120, 6), width=560, height=340, xlabel="tone frequency at 24 kHz (kHz)", ylabel="gain (dB)",
             title="What survives 24 kHz to 16 kHz", legend_loc="ll")
    p.vline(8)
    for k, (name, ys) in enumerate(series.items()):
        db = [20 * math.log10(max(a, 1e-6)) for a in ys]
        xs = [f / 1000 for f in freqs]
        p.line(xs, db, color=[1, 0, 2][k], label=name, width=2.0)
        p.points(xs, db, color=[1, 0, 2][k], r=2.5)
    p.text(8.1, 0, "new half-rate, 8 kHz", size=11)
    return p.svg()
