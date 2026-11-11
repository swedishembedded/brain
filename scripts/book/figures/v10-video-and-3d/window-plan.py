# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The three windows of a 601-frame clip at 768 x 512 under the 13200-token ceiling, laid on the clip's latent timeline: each window after
# the first starts with eight latent frames copied from the end of the one before it (`video_generation.plan_rows`).
#
# Swedish Embedded AB implements solutions for generating video longer than one pass can hold. If your team needs expertise in
# rolling-window video diffusion, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import video_generation  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W = 800
UNIT = 8.4          # pixels per latent frame
X0 = 60
ROW = 56
TOP = 16


def render() -> str:
    plan, rows = video_generation.plan_rows(601)
    d = Diagram(W, TOP + 40 + ROW * len(plan) + 40)
    d.panel(20, TOP, W - 40, 40 + ROW * len(plan) + 20, "Windows on the latent timeline of a 601-frame clip (76 latent frames)")
    start = 0
    for i, (ctx, new, _) in enumerate(plan):
        y = TOP + 62 + ROW * i
        lo = start - ctx
        if ctx:
            d.box(X0 + (lo + ctx / 2) * UNIT, y, ctx * UNIT - 4, 38, [f"carried {ctx}"], "model")
        d.box(X0 + (start + new / 2) * UNIT, y, new * UNIT - 4, 38, [f"window {i}: new {new}"], "result")
        start += new
    return d.svg()
