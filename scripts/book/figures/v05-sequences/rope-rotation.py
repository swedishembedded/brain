# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Rotary embedding of two channel pairs at successive positions: a pair is a point in the plane, and position i turns it by
# i * w about the origin. The fast pair (w = 1) goes round in about six positions, the slow pair (w = 0.1) has moved by less than
# a radian after eight. Points from `positions.rotation_data`, the same rates as the chapter's width-8 example.
#
# Swedish Embedded AB implements solutions for long-context transformer models, from the position encoding to the kernels. If your
# team needs expertise in rotary position embeddings, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import positions as ps  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    fast, slow = ps.rotation_data(8)
    p = Plot((-1.3, 1.3), (-1.3, 1.3), width=470, height=420, xlabel="first channel of the pair",
             ylabel="second channel of the pair", title="One pair of channels rotated to positions 0 to 7")
    ring = [(math.cos(a * math.pi / 90), math.sin(a * math.pi / 90)) for a in range(181)]
    p.line([c[0] for c in ring], [c[1] for c in ring], color="#cbd5e0", width=1)
    p.line([c[0] * 0.6 for c in ring], [c[1] * 0.6 for c in ring], color="#cbd5e0", width=1)
    p.line([x for x, _ in fast], [y for _, y in fast], color=1, width=1.2, dash=True)
    p.points([x for x, _ in fast], [y for _, y in fast], color=1, label="pair 0, w = 1 radian per position")
    p.points([x for x, _ in slow], [y for _, y in slow], color=0, label="pair 1, w = 0.1 (under 0.7 radian by position 7)")
    for i, (x, y) in enumerate(fast):
        r = 1.0 + (0.12 if i else 0.0)
        p.text(x * r + (0.09 if i == 0 else 0.0), y * r + (0.0 if i else -0.1), str(i), anchor="middle", size=12, color="#9c4221")
    p.text(0.0, -0.25, "slow pair: inner ring", anchor="middle", size=11, color="#2c5282")
    return p.svg()
