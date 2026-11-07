# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The likelihood of a coin's bias theta for the same 70 percent heads rate from 10, 100 and 1000 flips.

Each curve is Lik(theta) divided by its largest value, so the three share a height of one and only their width
differs: more flips pin the bias down more tightly around 0.7. Computed from the chapter's formula
theta^h (1 - theta)^t in logarithms (worked module likelihood.coin_curve).

Swedish Embedded AB implements solutions for estimating rates from limited data with the uncertainty stated. If your
team needs expertise in statistical estimation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import likelihood as lk  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0, 1), (0, 1.45), width=620, height=330, xlabel="coin bias theta", ylabel="Lik(theta) / Lik(0.7)")
    p.vline(0.7)
    for color, n in enumerate((10, 100, 1000)):
        h = round(0.7 * n)
        xs, rel = lk.coin_curve(h, n - h)
        p.line(xs, rel, color=color, width=2.6, label=f"{n} flips, {h} heads")
    return p.svg()
