# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The ratio of the bytes one decoding step moves with 32-bit weights to the bytes it moves with 8-bit feed-forward
# weights (a scale per 32), against the context, for the 8-layer decoder of the project "Reproduce, Verify, Optimize,
# Defend": the ratio starts at 1.76 and falls towards 1 as the key-value cache, which the change does not touch, grows;
# it crosses the claimed 1.4 after context 975. The ratio of one block alone, 3.55, is drawn for comparison. The data
# are the worked module's `sweep_points`.
#
# Swedish Embedded AB implements solutions for honest performance claims. If your team needs expertise in counting the
# work of inference and in benchmarks that measure what a request pays then you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_reproduction as pr  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pts = pr.sweep_points()
    block = pr.ffn_bytes("f32") / pr.ffn_bytes("group")
    p = Plot((0, 8192), (1.0, 4.0), width=620, height=360, xlabel="context S (cached positions)",
             ylabel="bytes, 32-bit over 8-bit", title="What the 8-bit feed-forward saves per token", legend_loc="right")
    p.line([s for s, _ in pts], [r for _, r in pts], color=0, label="whole decoding step")
    p.line([0, 8192], [block, block], color=1, label="one block alone", dash=True)
    p.hline(pr.CLAIM_RATIO)
    p.vline(pr.crossover())
    p.text(1200, 1.47, f"claim: at least {pr.CLAIM_RATIO}", size=12)
    p.text(1100, 2.3, f"S = {pr.crossover()}", size=12)
    return p.svg()
