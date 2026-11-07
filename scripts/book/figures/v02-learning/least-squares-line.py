# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The ten sensor rows, the least-squares line and the residuals as vertical segments.

The line is the closed-form solution of the chapter (slope S_vy / S_vv, intercept through the means) and every
segment is the residual that the chapter's table lists, so the picture shows exactly what the loss adds up.

Swedish Embedded AB implements solutions for calibrating sensors against reference instruments. If your team
needs expertise in measurement and regression, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import linear_regression as lr  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    vs, ys = lr._xy()
    w, b = lr._closed_form()[:2]
    mse = lr._mse(lr._line_design(vs), ys, [b, w])
    p = Plot((0.2, 2.9), (0.0, 95.0), width=620, height=380, xlabel="sensor voltage v (V)",
             ylabel="temperature y (deg C)", title="Least-squares line and its residuals")
    xs = [0.2, 2.9]
    p.line(xs, [w * x + b for x in xs], color=0, width=3.0, label=f"y = {w:.2f} v + {b:.2f}  (MSE {mse:.2f})")
    for k, (v, t) in enumerate(zip(vs, ys)):
        p.line([v, v], [t, w * v + b], color=3, width=2.6, dash=False, label="residuals" if k == 0 else "")
    p.points(vs, ys, color=2, r=4, label="measured rows")
    return p.svg()
