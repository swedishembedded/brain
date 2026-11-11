# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Where the video of a minute sits on the T axis of the rotary position, second by second, against where its sound sits: under the
# time-aligned rule the picture of second s and the sound of second s share a position, and under brain's frame-ordinal rule the
# whole video lies after the whole sound. The series are `omni_models.clock_data()`.
#
# Swedish Embedded AB implements solutions for models that read sound and video on one clock. If your team needs expertise in
# omni-modal position encodings, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import omni_models as om  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    d = om.clock_data()
    s = d["seconds"]
    p = Plot((0, 60), (0, 860), width=620, height=320, xlabel="second of the clip", ylabel="position on the T axis",
             title="Where second s of a minute sits on the T axis", legend_loc="lr")
    p.line(s, d["aligned_sound"], color=0, label="sound row of second s (both rules)", width=3.0)
    p.line(s, d["aligned_video"], color=1, label="video group, time-aligned", dash=True, width=2.0)
    p.line(s, d["ordinal_video"], color=2, label="video group, frame-ordinal", width=2.5)
    return p.svg()
