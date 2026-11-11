# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The apparent speed of a six-spoke wheel filmed at 24 frames per second against its true speed. The data are `video.wheel_data`,
# the folding rule of the sound chapter applied to the rate at which spokes pass.
#
# Swedish Embedded AB implements solutions for video that is sampled and resampled without false motion. If your team needs
# expertise in video frame-rate conversion, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import video  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs, ys = video.wheel_data()
    p = Plot((0, 14), (-2.6, 2.6), width=560, height=320, xlabel="true speed (turns per second)",
             ylabel="apparent speed (turns per second)", title="A six-spoke wheel at 24 frames per second", legend_loc="below")
    p.hline(0.0)
    p.line([x for x in xs if x <= 2.5], [x for x in xs if x <= 2.5], color=1, label="true speed", dash=True, width=1.6)
    p.line(xs, ys, color=0, label="apparent speed", width=1.8)
    return p.svg()
