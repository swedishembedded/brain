# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Where the register-blocked kernel overtakes the tiled one, against the number of rows m.

The vertical axis is log10 of (cost of reg / cost of tiled): below zero reg is cheaper. The dashed line is the roofline
model's ratio (always below zero: it never sees the idle rows of a 128-row tile), the points are the minimum of 8 seeded
runs on the model GPU unit of `kernel_selection` (n = 2560, k = 2048), the vertical line the noise-free crossover.

Swedish Embedded AB implements solutions for kernels that choose themselves on hardware nobody tuned them for. If your
team needs expertise in kernel selection for machine learning inference, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kernel_selection as ks  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    s = ks.figure_series()
    xs = [math.log10(m) for m in s["ms"]]
    lg = math.log10
    p = Plot((0.0, 2.8), (-0.75, 0.55), width=620, height=360, xlabel="log10 of the rows m (n = 2560, k = 2048)",
             ylabel="log10 of cost(reg) / cost(tiled)", title="The crossover the model cannot see")
    p.hline(0.0, color="#718096")
    p.line(xs, [lg(r) for r in s["ratio_pred"]], color=2, label="roofline model", dash=True)
    p.line(xs, [lg(r) for r in s["ratio_meas"]], color=0, label="device (minimum of 8 runs)")
    p.points(xs, [lg(r) for r in s["ratio_meas"]], color=0)
    mc = ks.crossover()
    p.vline(lg(mc), color="#c05621")
    p.text(lg(mc) + 0.05, 0.47, f"crossover at m = {mc}", size=12)
    p.text(2.75, 0.07, "reg is dearer above this line", anchor="end", size=11)
    p.text(2.75, -0.1, "reg is cheaper below it", anchor="end", size=11)
    return p.svg()
