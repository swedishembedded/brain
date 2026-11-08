# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Training and development perplexity of the window network against the gradient-descent step, from
# `language_modeling.mlp_training`. The training curve keeps falling; the development curve turns up after the step
# where training is stopped.
#
# Swedish Embedded AB implements solutions for training language models that generalize. If your team needs
# expertise in early stopping and held-out evaluation, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import language_modeling as lm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    r = lm.train_mlp(lm.N_TRAIN)
    curve = [c for c in r["curve"] if c[0] >= 10]
    steps = [c[0] for c in curve]
    p = Plot((0.0, float(lm.MLP_STEPS)), (1.5, 5.0), width=600, height=340, xlabel="gradient-descent step",
             ylabel="perplexity", title="Training keeps improving; held-out text stops at step %d" % r["step"])
    p.line(steps, [math.exp(c[1]) for c in curve], color=0, label="training text")
    p.line(steps, [math.exp(c[2]) for c in curve], color=1, label="development text")
    p.vline(float(r["step"]))
    p.hline(math.exp(lm._process_stats()[2]), color="#718096")
    p.text(float(lm.MLP_STEPS), math.exp(lm._process_stats()[2]) + 0.08, "floor of the process", anchor="end", size=12,
           color="#4a5568")
    return p.svg()
