# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Training and validation loss of the tiny GPT of the chapter from step 80 on (the first 80 steps fall from the loss of a uniform guess
# to about 0.8 and would flatten the rest), against the entropy of the corpus generator, the loss no model can beat in expectation.
# Data from `training_gpt.training_curve_data` and `training_gpt.corpus_floor`, the functions behind the chapter's tables.
#
# Swedish Embedded AB implements solutions for monitoring language-model training with held-out data. If your team needs expertise
# in validation curves and checkpoint selection, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_gpt as tg  # noqa: E402
from svgplot import Plot  # noqa: E402

FIRST = 80


def render() -> str:
    steps, train, vsteps, val = tg.training_curve_data()
    ts = [(s, v) for s, v in zip(steps, train) if s >= FIRST]
    vs = [(s, v) for s, v in zip(vsteps, val) if s >= FIRST]
    p = Plot((FIRST, tg.TRAIN["steps"]), (0.62, 0.92), width=640, height=380, xlabel="optimizer step", ylabel="loss, nats per target token",
             title="Validation loss turns up as training loss keeps falling")
    p.line([s for s, _ in ts], [v for _, v in ts], color=1, label="training (mean of 10 steps)")
    p.line([s for s, _ in vs], [v for _, v in vs], color=0, label="validation")
    p.points([s for s, _ in vs], [v for _, v in vs], color=0, r=2.5)
    p.hline(tg.corpus_floor("val"), color="#718096")
    p.text(FIRST + 4, tg.corpus_floor("val") - 0.022, "entropy of the generator", anchor="start", size=12, color="#4a5568")
    return p.svg()
