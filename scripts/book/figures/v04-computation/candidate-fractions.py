# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Fraction of the roofline floor against the number of rows m, for the three matrix-product candidates.

Solid lines are the model GPU unit of `kernel_selection` (the minimum of 8 seeded runs per point, n = 2560, k = 2048),
dashed lines are the roofline model's prediction for the two candidates that matter. The model ranks the register-blocked
kernel first at every m; the device ranks the tiled kernel first below the crossover.

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
    p = Plot((0.0, 2.8), (0.0, 1.05), width=620, height=380, xlabel="log10 of the rows m (n = 2560, k = 2048)",
             ylabel="fraction of the roofline floor", title="Predicted and device fractions of the three candidates")
    p.line(xs, s["meas"]["reg"], color=0, label="register-blocked, device")
    p.line(xs, s["pred"]["reg"], color=0, label="register-blocked, model", dash=True)
    p.line(xs, s["meas"]["tiled"], color=1, label="tiled, device")
    p.line(xs, s["pred"]["tiled"], color=1, label="tiled, model", dash=True)
    p.line(xs, s["meas"]["naive"], color=2, label="naive, device")
    return p.svg()
