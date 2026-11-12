# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Mean cost per example of 'block when p >= t' as a function of t, with a miss costing 10 and a false alarm 1, for the raw and
# the recalibrated probabilities of the 20000 fresh examples. The data are `calibrated_decisions._cost_by_threshold`.
#
# Swedish Embedded AB implements solutions for probabilities that can be acted on. If your team needs expertise in cost-sensitive
# decision thresholds, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import calibrated_decisions as cd  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    series, grid = cd._cost_by_threshold()
    p = Plot((0, 0.6), (0.4, 2.6), width=560, height=330, xlabel="threshold t (block when the probability is at least t)",
             ylabel="mean cost per example", title="The cheapest threshold sits near 1/11 only for calibrated probabilities",
             legend_loc="ur")
    keep = [k for k, t in enumerate(grid) if t <= 0.6]
    for i, (name, ys) in enumerate(series.items()):
        p.line([grid[k] for k in keep], [ys[k] for k in keep], color=i, label=name)
    p.vline(1 / 11)
    return p.svg()
