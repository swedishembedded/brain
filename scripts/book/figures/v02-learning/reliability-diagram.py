# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Reliability diagram of the screening model on the 3000 test parts, before and after recalibration.

Each point is one of ten equal-width bins of the predicted probability: its mean prediction against the share of parts
in it that were defective. A calibrated model lies on the diagonal. The raw model (a = 1) lies far below it where it is
confident; the single scale a fitted on the validation parts (temperature scaling) pulls the points toward the diagonal
but leaves them below it; a scale and a shift together bring them onto it.

Swedish Embedded AB implements solutions for probabilities that can be trusted when decisions depend on them. If your
team needs expertise in calibration, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import metrics as m  # noqa: E402
from svgplot import Plot  # noqa: E402


def _points(p, y):
    b = [t for t in m._bins(p, y, 10) if t[2] > 0]
    return [t[3] for t in b], [t[4] for t in b]


def render() -> str:
    zv, yv, zt, yt, *_ = m._scores()
    a = m._fit_scale(zv, yv)
    a2, b2 = m._fit_shift_scale(zv, yv)
    raw = [m.sigmoid(v) for v in zt]
    sc = [m.sigmoid(a * v) for v in zt]
    ss = [m.sigmoid(a2 * v + b2) for v in zt]
    p = Plot((0, 1), (0, 1), width=560, height=420, xlabel="mean predicted probability in the bin",
             ylabel="share of defective parts in the bin", title="Reliability diagram, 3000 test parts")
    p.line([0, 1], [0, 1], color="#a0aec0", dash=True, width=1.5)
    for pp, color in ((raw, 1), (sc, 0), (ss, 2)):
        xs, ys = _points(pp, yt)
        p.line(xs, ys, color=color, width=2.2)
        p.points(xs, ys, color=color, r=3.8)
    p.text(0.99, 0.20, f"raw (a = 1): ECE {m._ece(raw, yt):.3f}", anchor="end", color="#c05621")
    p.text(0.99, 0.13, f"scale only, a = {a:.2f}: ECE {m._ece(sc, yt):.3f}", anchor="end", color="#2b6cb0")
    p.text(0.99, 0.06, f"scale and shift: ECE {m._ece(ss, yt):.3f}", anchor="end", color="#2f855a")
    p.text(0.03, 0.95, "dashed: perfectly calibrated", color="#718096", size=11)
    return p.svg()
