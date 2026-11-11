# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The pairs of positions one attention head scores for a 768 x 512 clip as its length grows: frame by frame, space then time, and all at
# once (`video_generation.attention_pairs_data`, the arithmetic of the chapter's table).
#
# Swedish Embedded AB implements solutions for attention that stays affordable as video gets longer. If your team needs expertise in
# the cost of spatiotemporal attention, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import video_generation  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = video_generation.attention_pairs_data(frames=(1, 9, 17, 25, 49, 73, 97, 121, 169, 241, 361, 481))
    tokens = [r[2] for r in data]
    p = Plot((300, 30000), (1e5, 1e9), width=560, height=340, xlabel="tokens in the clip (latent frames x 384)",
             ylabel="pairs scored per head", title="Pairs of positions, 768 x 512 clips", xlog=True, ylog=True, legend_loc="ul")
    p.line(tokens, [r[3] for r in data], color=1, label="frame by frame", width=2.0)
    p.line(tokens, [r[4] for r in data], color=2, label="space then time", width=2.0)
    p.line(tokens, [r[5] for r in data], color=0, label="all at once", width=2.0)
    return p.svg()
