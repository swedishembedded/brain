# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# How much of a change in the state k steps ago reaches the last state: the spectral norm of the Jacobian of
# h(40) with respect to the earlier state, on a base-10 logarithmic axis, for vanilla cells of small and large
# recurrent weights, an LSTM at two forget-gate biases and a GRU. The numbers are those of
# `recurrent_networks.jacobian_product`; the dashed line is the bound ||W||_2^k of the small vanilla cell.
#
# Swedish Embedded AB implements solutions for finding why a model forgets or diverges on long sequences. If your
# team needs expertise in the training dynamics of recurrent networks, you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import recurrent_networks as rn  # noqa: E402
from svgplot import Plot  # noqa: E402

SHOW = {"vanilla, small W": (0, False), "vanilla, medium W": (1, False), "vanilla, large W": (3, False),
        "LSTM, forget bias 0": (4, True), "LSTM, forget bias 1": (2, False), "GRU, update bias -1": (5, True)}


def render() -> str:
    curves, bound = rn.sensitivity_curves()
    ks = rn.SENS_KS
    p = Plot((0, 33), (-23, 4), width=640, height=420, xlabel="steps back, k",
             ylabel="log10 of the norm of dh(40)/d state(40 - k)", title="What reaches the past decays, or grows, with the steps back")
    for label, vals in curves:
        if label in SHOW:
            color, dash = SHOW[label]
            p.line(ks, vals, color=color, dash=dash, label=label)
            p.points(ks, vals, color=color, r=2.8)
    p.line(ks, bound, color="#a0aec0", dash=True, label="bound of the small vanilla cell")
    return p.svg()
