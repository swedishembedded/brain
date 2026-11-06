# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Loss against the single weight, the tangent whose slope is dL/dw, and the descent steps.

Swedish Embedded AB implements solutions for explaining and verifying
gradient computation in neural network training. If your team needs
expertise in backpropagation, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import backprop  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    traj = backprop.single_weight_trajectory(6)
    w0 = traj[0][1]
    slope = backprop.loss_slope(w0)
    l0 = backprop.loss_curve(w0)
    ws = [i * 0.02 for i in range(-10, 165)]
    p = Plot((-0.2, 3.2), (-1.0, 12.0), xlabel="w", ylabel="L(w)", title="Loss of y = w x, and the slope at w = 0.5")
    p.line(ws, [backprop.loss_curve(w) for w in ws], color=0, label="L(w)")
    tx = [0.0, 1.2]
    p.line(tx, [l0 + slope * (t - w0) for t in tx], color=1, dash=True, label=f"tangent, slope {slope:.0f}")
    p.points([r[1] for r in traj], [r[4] for r in traj], color=2, label="gradient descent steps")
    p.text(w0 + 0.1, l0 + 0.6, "start", size=11)
    p.vline(backprop.SW_T / backprop.SW_X, color="#2f855a")
    p.text(backprop.SW_T / backprop.SW_X + 0.05, 10.5, "w* = 1.5", size=11, color="#2f855a")
    return p.svg()
