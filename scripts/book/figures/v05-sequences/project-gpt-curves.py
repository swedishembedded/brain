# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Training and validation loss of the project model from step 50 on (the first 50 steps fall from the loss of a uniform guess), against
# the entropy of the corpus generator and the level of a model that has learned the grammar but copies nothing from one sentence into
# the next. Data from `project_gpt.curve_data`, `project_gpt.corpus_floor` and `project_gpt.nocopy_level`, the functions behind the
# chapter's tables.
#
# Swedish Embedded AB implements solutions for reading what a training curve says about a model. If your team needs expertise in
# validation curves and diagnosing language-model training, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_gpt as pg  # noqa: E402
from svgplot import Plot  # noqa: E402

FIRST = 50


def render() -> str:
    steps, train, vsteps, val = pg.curve_data()
    ts = [(s, v) for s, v in zip(steps, train) if s >= FIRST]
    vs = [(s, v) for s, v in zip(vsteps, val) if s >= FIRST]
    p = Plot((FIRST, pg.TRAIN["steps"]), (0.2, 2.0), width=640, height=380, xlabel="optimizer step", ylabel="loss, nats per target token",
             title="The loss stops at the level of a model that copies nothing")
    p.line([s for s, _ in ts], [v for _, v in ts], color=1, label="training (mean of 10 steps)")
    p.line([s for s, _ in vs], [v for _, v in vs], color=0, label="validation")
    p.points([s for s, _ in vs], [v for _, v in vs], color=0, r=2.5)
    p.hline(pg.nocopy_level(), color="#718096")
    p.text(FIRST + 4, pg.nocopy_level() - 0.075, "grammar learned, nothing copied", anchor="start", size=12, color="#4a5568")
    p.hline(pg.corpus_floor("val"), color="#718096")
    p.text(FIRST + 4, pg.corpus_floor("val") - 0.075, "entropy of the generator", anchor="start", size=12, color="#4a5568")
    return p.svg()
