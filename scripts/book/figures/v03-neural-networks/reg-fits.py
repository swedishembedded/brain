# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The function the network has learned after 3000 steps, for seed 0, with no regularization, with weight decay and with dropout.

Computed by worked module `regularization_nn` (the same runs as the tables): the 16 noisy training points, the true curve
sin(2 pi x), and the output of each trained network on a grid of 201 points.

Swedish Embedded AB implements solutions for models that keep working on data they have not seen. If your team needs
expertise in regularizing neural networks, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import regularization_nn as rg  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = rg.comparison_data()
    wd, dp, _ = rg._chosen()
    xs, ys = rg._training_set(0, rg.N_TRAIN)
    grid = [i / 200 for i in range(201)]
    p = Plot((0, 1), (-1.8, 1.8), width=640, height=380, xlabel="x", ylabel="y", title="Fits after 3000 steps (seed 0)")
    p.line(grid, [math.sin(2 * math.pi * x) for x in grid], color=5, dash=True, width=1.6, label="true curve")
    p.points(xs, ys, color=5, r=4, label="16 training points")
    shown = {"none": 0, "weight decay": 2, "dropout": 1}
    labels = {"none": "no regularization", "weight decay": f"weight decay, lambda {wd:g}", "dropout": f"dropout, p {dp:g}"}
    for name, setting, runs in data:
        if name in shown:
            theta = runs[0]["theta"]
            out = rg._predict(theta, grid)
            p.line(grid, [max(-1.8, min(1.8, v)) for v in out], color=shown[name], width=2.2, label=labels[name])
    return p.svg()
