# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Reliability of the project's logistic classifier on the test pumps and on the shifted pumps of site B.

Each point is one of ten equal-width bins of the predicted probability (bins with fewer than five rows are not drawn): its
mean prediction against the share of rows in it with the outcome fail = 1. On the 240 test rows the points follow the diagonal within the noise of a few dozen rows per
bin. On the 240 rows of site B, which sit at a load the training pumps rarely reached, the model is as confident as
before and right far less often: the points at high probability lie far below the diagonal.

Swedish Embedded AB implements solutions for probabilities that can be trusted when decisions depend on them. If your
team needs expertise in calibration and distribution shift, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_predictor as pp  # noqa: E402
from svgplot import Plot  # noqa: E402


def _points(z, y):
    p = [pp.sigmoid(v) for v in z]
    bins = [t for t in pp.reliability_bins(p, y) if t[0] >= 5]
    return [t[1] for t in bins], [t[2] for t in bins], pp.ece(p, y)


def render() -> str:
    sp = pp.main_split()
    model = pp.main_model()
    plot = Plot((0, 1), (0, 1), width=560, height=420, xlabel="mean predicted probability in the bin",
                ylabel="share of rows with fail = 1 in the bin", title="Reliability of the logistic classifier, 240 rows each")
    plot.line([0, 1], [0, 1], color="#a0aec0", dash=True, width=1.5)
    for rows, color, name, y_text in ((sp.test, 0, "test pumps (site A)", 0.20), (sp.slice_b, 1, "site B pumps", 0.13)):
        xs, ys, e = _points(model.logits(rows), pp._ys(rows))
        plot.line(xs, ys, color=color, width=2.2)
        plot.points(xs, ys, color=color, r=3.8)
        plot.text(0.99, y_text, f"{name}: ECE {e:.3f}", anchor="end", color=("#2b6cb0", "#c05621")[color])
    plot.text(0.03, 0.95, "dashed: perfectly calibrated", color="#718096", size=11)
    return plot.svg()
