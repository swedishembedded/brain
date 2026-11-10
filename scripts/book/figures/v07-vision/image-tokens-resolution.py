# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Image tokens against the side of a square picture for the four designs of the chapter (`vlms.tokens_series`): a fixed square, a fixed count
# after pooling, slices with a source view, and dynamic resolution with the released limits.
#
# Swedish Embedded AB implements solutions for vision-language models whose image-token budget is counted before it is paid. If your team
# needs expertise in token budgets for vision-language models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vlms  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    sides, series = vlms.tokens_series()
    p = Plot((200, 3000), (30, 10000), width=640, height=380, xlabel="side of the square picture (pixels)", ylabel="image tokens",
             title="Image tokens against resolution", xlog=True, ylog=True)
    for i, (label, ys) in enumerate(series.items()):
        p.line(sides, ys, color=i, label=label)
        p.points(sides, ys, color=i, r=3)
    return p.svg()
