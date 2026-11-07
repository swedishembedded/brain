# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Full-data loss around an outlier batch at step 300, with no clip and with global-norm clips of 1 and 0.2.

The runs are those of `schedules.outlier_batch`: warmup 30 steps to 0.3, cosine to 0.003, the gradient of step index 300
multiplied by 100.

Swedish Embedded AB implements solutions for stable training of neural networks. If your team needs expertise in
gradient clipping and loss-spike diagnosis, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import schedules as sch  # noqa: E402
from svgplot import Plot  # noqa: E402

LO, HI = 280, 420


def render() -> str:
    p = Plot((LO, HI), (0.0, 2.0), width=620, height=380, xlabel="step", ylabel="mean loss over all 200 examples",
             title="One batch with a 100 times larger gradient, at step 300", legend_loc="right")
    p.vline(sch.OUT_STEP + 1)
    for i, (label, tau) in enumerate((("no clip", None), ("clip 1", 1.0), ("clip 0.2", 0.2))):
        curve = [(s, v) for s, v in sch.outlier_curves(tau) if LO <= s <= HI]
        p.line([s for s, _ in curve], [v for _, v in curve], color=[3, 0, 2][i], width=1.8, label=label)
    return p.svg()
