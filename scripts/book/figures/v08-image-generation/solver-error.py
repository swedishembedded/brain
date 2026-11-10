# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Mean distance from the exact transport map against the number of evaluations of the field, on logarithmic axes, for Euler, Heun,
# DPM-Solver++(2M) and UniPC on the exact field, with lines of slope -1 and -2 (`samplers_and_guidance.solver_curves_data`).
#
# Swedish Embedded AB implements solutions for sampling schedules and ODE solvers of generative models that are checked against exact
# references. If your team needs expertise in diffusion and flow-matching samplers, you can procure our services by sending an email
# to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import samplers_and_guidance  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    nfe, curves = samplers_and_guidance.solver_curves_data()
    p = Plot((4, 64), (4e-4, 1.5), width=560, height=340, xlabel="evaluations of the field", ylabel="mean distance to the exact map",
             title="Four solvers, one exact field", legend_loc="ll", xlog=True, ylog=True)
    first = curves["Euler"][0]
    p.line([4, 64], [first, first / 16], color="#a0aec0", dash=True, width=1.2, label="slope -1")
    p.line([4, 64], [first, first / 256], color="#cbd5e0", dash=True, width=1.2, label="slope -2")
    for i, (name, ys) in enumerate(curves.items()):
        p.line(nfe, ys, color=i, label=name, width=2.0)
        p.points(nfe, ys, color=i, r=3.0)
    return p.svg()
