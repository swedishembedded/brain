# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Precision, recall and F1 on the validation parts as the cut on the logit moves.

Computed with the chapter's confusion counts: for each cut c on the logit z a part is called defective when z >= c.
Raising the cut trades recall for precision; F1 peaks between them. Precision is undefined (and not drawn) once no
part reaches the cut.

Swedish Embedded AB implements solutions for choosing operating points of classifiers from held-out evidence. If your
team needs expertise in precision, recall and threshold selection, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import metrics as m  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    z, y, *_ = m._scores()
    cuts = [-8.0 + 0.1 * k for k in range(161)]
    xs, prec, rec, f1 = [], [], [], []
    for c in cuts:
        tp, fp, fn, _ = m._confusion([v >= c for v in z], y)
        if tp + fp == 0:
            continue
        xs.append(c)
        prec.append(tp / (tp + fp))
        rec.append(tp / (tp + fn))
        f1.append(2 * tp / (2 * tp + fp + fn))
    p = Plot((-8, 8), (0.0, 1.05), width=620, height=360, xlabel="cut on the logit z (a part is called defective when z >= cut)",
             ylabel="value", title="Validation parts: precision, recall and F1 against the cut")
    p.line(xs, rec, color=1, width=2.6)
    p.line(xs, prec, color=0, width=2.6)
    p.line(xs, f1, color=2, width=2.6, dash=True)
    p.text(-7.7, 0.975, "recall", color="#c05621")
    p.text(5.2, 0.86, "precision", color="#2b6cb0")
    p.text(-7.7, 0.56, "F1 (dashed)", color="#2f855a")
    p.vline(0.0)
    p.text(0.15, 0.04, "t = 0.5")
    return p.svg()
