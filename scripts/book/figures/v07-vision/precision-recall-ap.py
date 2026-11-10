# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The precision-recall curve of class 0 of the average-precision example at IoU 0.5, its monotone envelope, and the area under the
# envelope that is the average precision (`detection.pr_data`).
#
# Swedish Embedded AB implements solutions for detector evaluation that can be reproduced from a printed table. If your team needs
# expertise in mean average precision and benchmark protocols, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import detection as dt  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402


def render() -> str:
    pts, env, ap = dt.pr_data(0, 0.5)
    rs = [r for r, _ in pts]
    ps = [q for _, q in pts]
    p = Plot((0, 1), (0, 1.05), width=560, height=380, xlabel="recall", ylabel="precision", title=f"Precision-recall curve, AP = {ap:.2f}",
             legend_loc="ur")
    # the area under the envelope: for each step of recall, a rectangle of the envelope's height
    for k in range(1, len(rs)):
        if rs[k] > rs[k - 1]:
            x0, x1, y = p.x(rs[k - 1]), p.x(rs[k]), p.y(env[k])
            p.items.append(f'<rect x="{fmt(x0)}" y="{fmt(y)}" width="{fmt(x1 - x0)}" height="{fmt(p.y(0) - y)}" fill="#2b6cb0" fill-opacity="0.18"/>')
    p.line(rs[1:], ps[1:], color=1, label="precision after each detection")
    p.points(rs[1:], ps[1:], color=1, r=3.5)
    # the envelope, drawn only over the recall that the detections reach: a flat run per step of recall
    sx, sy = [], []
    for k in range(1, len(rs)):
        if rs[k] > rs[k - 1]:
            sx += [rs[k - 1], rs[k]]
            sy += [env[k], env[k]]
    p.line(sx, sy, color=0, label="envelope (area = AP)", width=2.4)
    return p.svg()
