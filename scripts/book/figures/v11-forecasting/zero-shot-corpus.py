# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The scaled error of one shared linear map, pretrained on a corpus of series, on three families of test series as the corpus grows, beside the
# fitted autoregression of Chapter 1 on the same series. The data are `foundation_forecasters._zero_shot_results`.
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
    res = ff._zero_shot_results()
    sizes = ff.ZS_MARKS
    p = Plot((3, 600), (0.3, 1.4), width=560, height=340, xlabel="series in the pretraining corpus (log scale)", ylabel="scaled MAE",
             title="One pretrained map on series it was not fitted to", legend_loc="ur", xlog=True)
    for color, fam, m, mixed in ((0, "in", ff.ZS_M_IN, False), (2, "out", ff.ZS_M_OUT, False), (3, "new", ff.ZS_M_NEW, False)):
        ys = [ff._mean(res[fam][f"shared map, {n} series"]) for n in sizes]
        p.line(list(sizes), ys, color=color, label=f"test period {m}")
    ar = "AR(1) on seasonal differences, fitted to the 48 bars"
    p.hline(ff._mean(res["in"][ar]), color="#718096")
    p.text(4, ff._mean(res["in"][ar]) + 0.02, "fitted AR(1), period 12", size=11, color="#4a5568")
    mixed = "shared map, 200 series of period 12 and 200 of period 8"
    p.points([400], [ff._mean(res["out"][mixed])], color=2, r=5)
    p.text(360, ff._mean(res["out"][mixed]) + 0.04, "period 8 added to the corpus", anchor="end", size=11, color="#4a5568")
    return p.svg()
