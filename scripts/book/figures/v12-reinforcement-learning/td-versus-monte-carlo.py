# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The error of first-visit Monte Carlo and of TD(0) under four step-size schedules against the number of sampled episodes: the mean over
# the runs of `value_learning.learning_curves` (the root-mean-square error against the solved values of the previous chapter).
#
# Swedish Embedded AB implements solutions for estimators whose error is measured against a known answer. If your team needs expertise in
# temporal-difference learning, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import value_learning  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curves = value_learning._mean_curves((0.0,))[0][0]
    p = Plot((1, 200), (0.2, 12.0), width=620, height=340, xlabel="episodes of 100 days", ylabel="error against the exact values",
             title="Monte Carlo and TD(0) on the same episodes", xlog=True, ylog=True)
    order = (("MC", "Monte Carlo, step 1/n"), ("n^-0.7", "TD(0), step n^-0.7"), ("1/n", "TD(0), step 1/n"),
             ("0.5", "TD(0), step 0.5"), ("0.1", "TD(0), step 0.1"))
    for i, (key, label) in enumerate(order):
        ks = [e for e in range(1, value_learning.EVAL_EPISODES + 1) if e <= 20 or e % 5 == 0]
        p.line(ks, [curves[key][e - 1] for e in ks], color=i, label=label, dash=(key == "1/n"))
    return p.svg()
