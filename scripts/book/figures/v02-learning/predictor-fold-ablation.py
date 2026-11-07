# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The ablation of the project: how the cross-validation folds are drawn, seen on the nearest-neighbour model.

Each point is one of the 16 runs of the seed study. Horizontal: the cross-validation log loss that the run would report
for the neighbour count it chose. Vertical: the log loss of the refitted model on 240 test rows from pumps never seen.
Points on the dashed line are honest, points above it are runs whose test result is worse than their claim. With folds of
whole pumps the points scatter around the line; with random row folds most lie far above it, because a held-out row has
its sibling rows in the training folds and the estimate is too good to be true.

Swedish Embedded AB implements solutions for evaluation protocols whose numbers survive contact with new data. If your
team needs expertise in cross-validation and data leakage, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_predictor as pp  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    runs = pp._seed_runs()
    lo, hi = 0.46, 0.66
    plot = Plot((lo, hi), (lo, hi), width=560, height=420, xlabel="cross-validation log loss of the chosen k (the claim)",
                ylabel="test log loss of the refitted k-NN model", title="Does the estimate match the test, 16 runs?")
    plot.line([lo, hi], [lo, hi], color="#a0aec0", dash=True, width=1.5)
    for tag, color, label in (("group", 0, "folds of whole pumps"), ("row", 1, "random row folds")):
        xs = [r[tag]["cv_knn"] for r in runs]
        ys = [r[tag]["test_knn"] for r in runs]
        plot.points(xs, ys, color=color, label=label, r=4)
    plot.text(0.47, 0.655, "above the line: the test is worse than the claim", color="#718096", size=11)
    return plot.svg()
