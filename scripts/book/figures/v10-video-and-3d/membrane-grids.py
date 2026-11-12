# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The membrane of a leaky integrate-and-fire neuron under constant drive (`neural_simulation.lif_spike_times`): the exact curve, the
# same curve sampled by the exact step every 2 milliseconds, and forward Euler on the same grid, against the threshold.
#
# Swedish Embedded AB implements solutions for spiking simulation whose integrator error is measured, not assumed. If your team needs
# expertise in neuron models and numerical integration, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import neural_simulation as ns  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ts, exact, grid, ex, fe = ns._membrane_curves()
    keep = [i for i, g in enumerate(grid) if g <= 24.0]
    p = Plot((0, 24), (0, 1.6), width=560, height=340, xlabel="time (milliseconds)", ylabel="membrane potential",
             title="One neuron, three ways of stepping it")
    p.hline(1.0)
    p.text(0.5, 1.04, "threshold", size=11)
    n = [i for i, t in enumerate(ts) if t <= 24.0]
    p.line([ts[i] for i in n], [exact[i] for i in n], color=0, width=2.0, label="the ODE, exactly")
    p.points([grid[i] for i in keep], [ex[i] for i in keep], color=1, label="exact step, every 2 ms")
    p.points([grid[i] for i in keep], [fe[i] for i in keep], color=2, label="forward Euler, every 2 ms")
    return p.svg()
