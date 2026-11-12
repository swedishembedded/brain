# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The training loss of the three-Gaussian fit of the chapter after each Adam update (`gaussian_splatting.fit_toy`).
#
# Swedish Embedded AB implements optimizers that fit splat scenes to photographs. If your team needs expertise in differentiable
# rendering, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gaussian_splatting as gs  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curve = gs._fit_curve()
    xs = list(range(1, len(curve)))
    p = Plot((1, len(curve) - 1), (1e-7, 1e-2), ylog=True, width=560, height=320, xlabel="update", ylabel="training loss (mean squared error)",
             title="Fitting three Gaussians to four views")
    p.line(xs, [curve[i] for i in xs], color=0, width=2.0)
    return p.svg()
