# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The 90 training rows shared by every method of the chapter: three overlapping groups of machines.

Swedish Embedded AB implements solutions for telling machine states apart from a few sensor readings. If your
team needs expertise in classical classification of sensor data, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import classical as cl  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    tx, ty = cl._dataset()["train"]
    p = Plot((-5.0, 5.0), (-4.5, 5.5), width=520, height=400, xlabel="x1", ylabel="x2", title="The shared training set: 90 rows, three classes")
    for c in range(3):
        pts = [q for q, t in zip(tx, ty) if t == c]
        p.points([q[0] for q in pts], [q[1] for q in pts], color=c, r=4, label=f"class {cl.CLASS_NAMES[c]} (30 rows)")
    for c in range(3):
        p.points([cl.CENTRES[c][0]], [cl.CENTRES[c][1]], color="#1a202c", r=6, label="generating centres" if c == 0 else "")
    return p.svg()
