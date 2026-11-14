# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The device work done so far against the tokens of the sequence that was already decoding, while eight prompts of 2,048
# tokens arrive, for the four prefill policies of the chapter "Batching and Scheduling Requests". The data are the worked
# module's `burst_timeline`, so the curves cannot disagree with the chapter's table.
#
# Swedish Embedded AB implements solutions for serving language models to many users at once on the hardware they already
# have. If your team needs expertise in request scheduling then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import batching as bt  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0, 96), (0, 6.6), width=640, height=400, xlabel="tokens of the sequence already decoding (one an iteration)",
             ylabel="work so far (10^12 equivalent bytes)", title="Eight prompts of 2,048 tokens arrive beside a running sequence",
             legend_loc="lr")
    for color, name in enumerate(bt.POLICIES):
        pts = bt.burst_timeline(name)
        p.line([k for k, _ in pts], [c for _, c in pts], color=color, label=name, dash=(color == 2))
    return p.svg()
