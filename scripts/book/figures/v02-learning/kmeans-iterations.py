# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Lloyd's algorithm from a poor start: the paths of the three centres over the rounds, on the 90 training rows without labels.

Swedish Embedded AB implements solutions for finding structure in unlabelled machine and sensor data. If your
team needs expertise in clustering and vector quantization, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import classical as cl  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pts = cl._train_points()
    (_, final, assign, rounds), _, poor, _ = cl._first_runs()
    p = Plot((-5.0, 5.0), (-4.5, 9.0), width=520, height=440, xlabel="x1", ylabel="x2",
             title=f"k-means from a poor start: {len(rounds)} rounds")
    for j in range(3):
        mine = [q for q, a in zip(pts, assign) if a == j]
        p.points([q[0] for q in mine], [q[1] for q in mine], color=j, r=3.5, label=f"points assigned to centre {j + 1}")
    paths = [[r[0][j] for r in rounds] + [final[j]] for j in range(3)]
    for j, path in enumerate(paths):
        p.line([q[0] for q in path], [q[1] for q in path], color="#4a5568", width=1.5, dash=True, label="path of the centres" if j == 0 else "")
        p.points([q[0] for q in path[:-1]], [q[1] for q in path[:-1]], color="#a0aec0", r=3.0)
        p.points([path[0][0]], [path[0][1]], color="#e53e3e", r=5.5, label="starting centres" if j == 0 else "")
    p.points([c[0] for c in final], [c[1] for c in final], color="#1a202c", r=8, label="final centres")
    return p.svg()
