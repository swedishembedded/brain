# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training and validation loss of the unregularized network (seed 0), with the early-stopping point.

Computed by worked module `regularization_nn`. The vertical line at the best validation step is the checkpoint a
patience rule keeps; the line to its right is the step at which a patience of 10 evaluations (100 steps) fires.

Swedish Embedded AB implements solutions for stopping training at the point where a model generalizes best. If your
team needs expertise in early stopping and model selection, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import regularization_nn as rg  # noqa: E402
from svgplot import Plot  # noqa: E402

PATIENCE = 10


def render() -> str:
    run = rg._runs()[0]
    h = run["history"]
    i_stop, i_best = rg._stop_point(h, PATIENCE)
    steps = [row[0] for row in h]
    p = Plot((0, rg.STEPS), (0, 0.3), width=640, height=380, xlabel="gradient step", ylabel="mean squared error",
             title="No regularization (seed 0)")
    p.hline(rg.NOISE_SD ** 2)
    p.line(steps, [row[1] for row in h], color=0, label="training loss")
    p.line(steps, [row[2] for row in h], color=1, label="validation loss")
    best_step, stop_step = h[i_best][0], h[i_stop][0]
    p.vline(best_step, color="#2f855a")
    p.vline(stop_step, color="#9b2c2c")
    p.points([best_step], [h[i_best][2]], color=2, r=5)
    p.text(stop_step + 60, 0.285, f"best validation (green): step {best_step}", color="#2f855a")
    p.text(stop_step + 60, 0.262, f"patience {PATIENCE} fires (red): step {stop_step}", color="#9b2c2c")
    p.text(rg.STEPS - 20, rg.NOISE_SD ** 2 + 0.008, "noise floor 0.09", anchor="end", color="#4a5568")
    return p.svg()
