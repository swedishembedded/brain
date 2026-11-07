# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A sine curve and the networks of k = 2, 4 and 8 ReLU units that interpolate it (mlp.relu_approx_error).

Swedish Embedded AB implements solutions for function approximation with small neural networks whose every
weight can be inspected. If your team needs expertise in compact neural models, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlp  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs = [i / 400 for i in range(401)]
    p = Plot((0.0, 1.0), (-1.2, 1.2), width=560, height=340, xlabel="x", ylabel="value",
             title="k ReLU units approximating sin(2 pi x)", legend_loc="ur")
    p.line(xs, [mlp._curve(x) for x in xs], color=0, width=3.0, label="sin(2 pi x)")
    for color, k in ((1, 2), (2, 4), (4, 8)):
        net = mlp._relu_net(k)
        err = mlp._curve_error(k)
        p.line(xs, [mlp._relu_eval(net, x) for x in xs], color=color, width=1.8, dash=(k == 2),
               label=f"k = {k}, error {err:.3f}")
    return p.svg()
