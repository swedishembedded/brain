# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Population cross-entropy above the source's entropy against the training step, for the spectral mixer and its two SwiGLU controls, eight
# runs each of the chapter's main comparison: the thick line is the mean over the runs, the thin lines of the same colour the smallest and
# the largest run at each step. Numbers from `spectral_lm.curve_data` (the unnudged runs).
#
# Swedish Embedded AB implements solutions for testing new model architectures against matched baselines. If your team needs expertise in
# experiment design for neural architectures, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spectral_lm as sl  # noqa: E402
from svgplot import Plot  # noqa: E402

ORDER = (("S", 0, "spectral mixer, equal parameters to Q_par"), ("Q_par", 1, "SwiGLU, equal parameters"), ("Q_flop", 2, "SwiGLU, equal arithmetic"))


def render() -> str:
    h = sl.source_entropy()
    data = sl.curve_data("main")
    p = Plot((0, 240), (0.04, 2.5), width=620, height=380, xlabel="training step (full batch)",
             ylabel="population cross-entropy minus entropy rate (nats)", title="Eight runs per arm: mean and range", ylog=True, legend_loc="auto")
    for arm, color, label in ORDER:
        xs = [s for s, _ in data[arm]]
        mean = [sum(v) / len(v) - h for _, v in data[arm]]
        lo = [min(v) - h for _, v in data[arm]]
        hi = [max(v) - h for _, v in data[arm]]
        p.line(xs, lo, color=color, width=1.0)
        p.line(xs, hi, color=color, width=1.0)
        p.line(xs, mean, color=color, label=label, width=2.4)
    return p.svg()
