# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The four chunks of an eighteen-second song (450 frames) on the song's own time axis: which stretch of each chunk is carried in from
# the previous one, which is cropped away, and which reaches the song. The rows are `music.stitch_plan(450)`; a latent step is
# 512/44,100 of a second and a chunk starts 100 frames, 4 seconds, after the one before it.
#
# Swedish Embedded AB implements solutions for generating long audio in overlapping windows. If your team needs expertise in music
# and long-form audio generation, you can procure our services by sending an email to info@swedishembedded.com.
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import music as mu  # noqa: E402
from svgplot import Plot  # noqa: E402

GREY = "#cbd5e0"


def render() -> str:
    plan = mu.stitch_plan(450)
    n = len(plan)
    p = Plot((-3.0, 19), (n + 0.4, -0.7), width=620, height=300, xlabel="time in the song (seconds)", ylabel="",
             title="Four chunks of an 18-second song", legend_loc=None)
    sec = 1.0 / mu.LAT_RATE
    for c in plan:
        y = c["i"]
        t0 = c["start"] / mu.FRAME_RATE
        held = min(mu.OVERLAP, c["length"]) if c["i"] > 0 else 0
        p.line([t0, t0 + c["length"] * sec], [y, y], color=GREY, width=16)
        if held:
            p.line([t0, t0 + held * sec], [y, y], color=1, width=16)
        p.line([t0 + c["lo"] * sec, t0 + c["hi"] * sec], [y, y], color=0, width=16)
        if held:
            p.line([t0 + c["lo"] * sec, t0 + held * sec], [y, y], color=2, width=16)
        p.text(-0.2, y + 0.08, f"chunk {c['i']}", anchor="end", size=12)
    for c in plan[1:]:
        t0 = c["start"] / mu.FRAME_RATE
        p.line([t0 + c["lo"] * sec] * 2, [-0.5, n - 0.5], color="#2f855a", width=1.2, dash=True)
        p.line([t0 + min(mu.OVERLAP, c["length"]) * sec] * 2, [-0.5, n - 0.5], color="#c05621", width=1.2, dash=True)
    p.text(-2.8, n - 0.05, "grey: cropped", size=11)
    p.text(2.2, n - 0.05, "blue: kept", size=11, color="#2b6cb0")
    p.text(5.6, n - 0.05, "orange: carried in, then cropped", size=11, color="#c05621")
    p.text(12.6, n - 0.05, "green: carried in and kept", size=11, color="#2f855a")
    # the lanes are labelled in the figure itself; the numeric ticks of the vertical axis mean nothing here
    return re.sub(r'<text [^>]*font-size="11" fill="#4a5568" text-anchor="end">[^<]*</text>', '', p.svg())
