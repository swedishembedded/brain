# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The first repetition of the stitching experiment: the truth, the cross-fade of the TimesFM-3
# postprocess and a hard switch between patches. The data are `foundation_forecasters._stitch_first`.
#
# Swedish Embedded AB implements solutions for forecasting with pretrained models whose behaviour is checked rather than assumed. If your team
# needs expertise in time-series foundation models and their evaluation, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import foundation_forecasters as ff  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    truth, fade, hard, preds = ff._stitch_first()
    p = Plot((0, 19), (9, 20), width=560, height=340, xlabel="forecast bar", ylabel="value",
             title="Four output patches, stitched two ways", legend_loc="ul")
    xs = list(range(ff.STITCH_H))
    p.line(xs, truth, color=1, label="truth", width=2.6)
    p.line(xs, hard, color=3, label="hard switch")
    p.line(xs, fade, color=0, label="cross-fade")
    return p.svg()
