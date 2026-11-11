# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The ideal code length, in bytes, of each frame of the panning clip coded alone, as a difference from the previous frame, and with
# block motion vectors. The data are `video.toy_bytes_data`, the numbers of the chapter's table.
#
# Swedish Embedded AB implements solutions for storing and moving video at the size its content needs. If your team needs expertise
# in video coding and motion compensation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import video  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    t, alone, diff, motion = video.toy_bytes_data()
    p = Plot((0, 7), (0, 260), width=560, height=320, xlabel="frame", ylabel="bytes of the frame",
             title="A clip that pans one pixel per frame", legend_loc="below")
    p.line(t, diff, color=2, label="difference from the previous frame", width=1.8)
    p.line(t, alone, color=1, label="each frame alone", width=1.8, dash=True)
    p.line(t, motion, color=0, label="motion vectors and residual", width=1.8)
    return p.svg()
