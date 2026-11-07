# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Twelve unlabelled readings and the two groups the assign-then-average loop finds.

Swedish Embedded AB implements solutions for discovering structure in
unlabelled machine and sensor data. If your team needs expertise in
unsupervised learning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import learning_problem as lp  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    points, history, centers = lp._grouping()
    assign = history[-1][1]
    p = Plot((-1.0, 5.0), (-0.5, 6.5), width=520, height=360, xlabel="feature 1", ylabel="feature 2",
             title="Groups found without any labels")
    for k, (colour, name) in enumerate(((0, "group A"), (1, "group B"))):
        pts = [pt for pt, a in zip(points, assign) if a == k]
        p.points([x for x, _ in pts], [y for _, y in pts], color=colour, r=5, label=name)
    p.points([c[0] for c in centers], [c[1] for c in centers], color="#1a202c", r=8, label="group centres")
    for r, (cs, _) in enumerate(history[:-1]):
        p.points([c[0] for c in cs], [c[1] for c in cs], color="#a0aec0", r=4,
                 label="earlier centres" if r == 0 else "")
    return p.svg()
