# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Mean distance from the exact transport map against the number of Euler steps, on logarithmic axes: the exact field (solid), the
# trained network (dashed), and a line of slope -1 (`flow_matching.error_curves_data`).
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
    n_ex, ex, n_net, net = flow_matching.error_curves_data()
    p = Plot((1, 64), (0.02, 3), width=560, height=340, xlabel="Euler steps n", ylabel="mean distance to the exact map",
             title="Sampling error against the number of steps", legend_loc="ur", xlog=True, ylog=True)
    p.line([1, 64], [ex[0], ex[0] / 64], color="#a0aec0", dash=True, width=1.4, label="slope -1")
    p.line(n_ex, ex, color=0, label="exact field", width=2.2)
    p.line(n_net, net, color=1, label="trained network", width=1.8, dash=True)
    return p.svg()
