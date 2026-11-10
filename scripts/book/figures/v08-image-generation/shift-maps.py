# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The static shift s' = k s / (1 + (k - 1) s) for several k, and the four step values of the 512 x 512 FLUX.2 Klein schedule against
# the uniform grid they come from (`flow_matching.shift_curves_data`).
#
# Swedish Embedded AB implements solutions for generative models whose paths, velocity targets and sampling schedules are specified,
# checked and made reproducible. If your team needs expertise in flow matching and image synthesis, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flow_matching  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs, curves, grid, klein = flow_matching.shift_curves_data()
    p = Plot((0, 1), (0, 1), width=560, height=340, xlabel="unshifted noise fraction sigma", ylabel="shifted noise fraction sigma'",
             title="The timestep shift", legend_loc="lr")
    for i, (k, ys) in enumerate(curves.items()):
        p.line(xs, ys, color=i, label=f"k = {k}", width=2.0)
    p.points(grid, klein, color=3, label="Klein, 512 px, 4 steps", r=4.5)
    return p.svg()
