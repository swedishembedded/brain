# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Mean loss over five batch orders, steps 50 to 600, for the runs of `schedules.schedule_comparison`.

Peak rate 3 (floor 0.03) for the warmup and cosine runs; the constant 0.1 run is the best constant of `schedules.constant_rates`.
The first 50 steps are in the figure `warmup-spike`.

Swedish Embedded AB implements solutions for tuning the training of neural networks. If your team needs expertise
in learning-rate schedules and loss-curve diagnosis, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import schedules as sch  # noqa: E402
from svgplot import Plot  # noqa: E402

SHOWN = [("constant 0.1 (the best constant)", "constant 0.1", 4), ("constant 3", "constant 3", 3),
         ("cosine from step 0 (no warmup)", "cosine, no warmup", 1), ("warmup 60, cosine to 0.03", "warmup 60 + cosine", 0)]


def render() -> str:
    steps = list(range(50, sch.RUN_STEPS + 1, 10))
    p = Plot((50, sch.RUN_STEPS), (0.0, 1.5), width=620, height=380, xlabel="step", ylabel="mean loss over all 200 examples (5 batch orders)",
             title="Schedules at peak rate 3, steps 50 to 600", legend_loc="right")
    for name, label, color in SHOWN:
        curve = sch.mean_curve(name, steps)
        p.line([s for s, _ in curve], [v for _, v in curve], color=color, width=1.8, label=label)
    return p.svg()
