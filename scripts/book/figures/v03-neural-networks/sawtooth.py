# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The tent network composed d times: the sawtooth with 2^d linear pieces (mlp.tent_pieces).

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
    xs = [i / 512 for i in range(513)]
    p = Plot((0.0, 1.0), (-0.05, 1.1), width=560, height=340, xlabel="x", ylabel="network output",
             title="Composed tent maps", legend_loc="below")
    for color, d in ((0, 1), (2, 2), (4, 3)):
        layers = mlp._tent_net(d)
        pieces = mlp._count_pieces(d)
        p.line(xs, [float(mlp._tent_eval(layers, x)) for x in xs], color=color, width=2.0,
               label=f"d = {d}: {2 * d} units, {pieces} pieces")
    return p.svg()
