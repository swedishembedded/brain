# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Entropy of a coin as a function of its bias, with its two contributions.

H(p) = p log2(1/p) + (1-p) log2(1/(1-p)): the curve and its two terms are
computed with the chapter's own entropy function.

Swedish Embedded AB implements solutions for measuring uncertainty in model
outputs. If your team needs expertise in information-theoretic evaluation, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import information as inf  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    n = 200
    ps = [i / n for i in range(n + 1)]
    plot = Plot((0.0, 1.0), (0.0, 1.4), width=520, height=340, xlabel="P(heads) = p",
                ylabel="entropy in bits", title="Entropy of a coin")
    plot.line(ps, [inf.entropy([p, 1 - p]) for p in ps], color=0, label="H(p)")
    plot.line(ps, [p * inf.surprise(p) if p > 0 else 0.0 for p in ps], color=1, dash=True,
              label="heads term, p x (-log2 p)")
    plot.line(ps, [(1 - p) * inf.surprise(1 - p) if p < 1 else 0.0 for p in ps], color=2, dash=True,
              label="tails term")
    for p in (0.1, 0.5):
        h = inf.entropy([p, 1 - p])
        plot.points([p], [h], color=3)
        plot.text(p + 0.02 if p < 0.5 else 0.53, h + 0.09 if p < 0.5 else h - 0.04, f"p = {p}: {h:.3f} bits", size=11)
    return plot.svg()
