# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# FLOPs of the 24 global blocks of the feed-forward model as the number of frames grows, at the native 518 pixel size: their matrix
# products and their attention, with the frame attention of the other 24 blocks for reference (`feed_forward_3d.trunk_flops`).
#
# Swedish Embedded AB implements solutions for multi-view attention whose cost is known before it is run. If your team needs expertise
# in the cost of transformers over many images, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import feed_forward_3d as ff  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    hp = wp = ff.IMG // ff.PATCH
    n = ff.SPECIALS + hp * wp
    frames = list(range(1, 25))
    data = [ff.trunk_flops(s, hp, wp) for s in frames]
    p = Plot((1, 24), (0.1, 200), width=560, height=340, xlabel="frames in the pass (518 by 518 pixels)",
             ylabel="TFLOP of 24 blocks", title="Global blocks: matrix products and attention", xlog=True, ylog=True, legend_loc="ul")
    p.line(frames, [r[0] / 2e12 for r in data], color=1, label="matrix products of the global blocks", width=2.0)
    p.line(frames, [r[1] / 1e12 for r in data], color=2, label="frame blocks, attention (for reference)", width=2.0)
    p.line(frames, [r[2] / 1e12 for r in data], color=0, label="global blocks, attention", width=2.0)
    cross = 6 * ff.D / n
    p.vline(cross)
    p.text(cross * 1.04, 0.35, f"{cross:.1f} frames: attention = products", size=12)
    return p.svg()
