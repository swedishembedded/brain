# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The ratio of the 12-bar error of the AR forecaster to that of seasonal naive with drift at each of the 169 origins of one series, with the 15
# origins 12 bars apart marked. The data are `evaluating_forecasts.origin_losses`, the first table of the chapter.
#
# Swedish Embedded AB implements solutions for forecast evaluation that separates skill from luck. If your team needs expertise in
# walk-forward evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import evaluating_forecasts as ef  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    origins, losses = ef.origin_losses()
    ratio = [a / b for a, b in zip(losses["AR"], losses["SND"])]
    p = Plot((ef.FIRST_ORIGIN, ef.LAST_ORIGIN), (0.4, 1.8), width=620, height=320, xlabel="origin o (index of the first forecast bar)",
             ylabel="error of AR / error of SND", title="One series, 169 origins: who wins depends on where you look", legend_loc="below")
    p.line(origins, ratio, color=0, label="every origin", width=1.4)
    sel = origins[::12]
    p.points(sel, [ratio[o - ef.FIRST_ORIGIN] for o in sel], color=1, r=3.2, label="the 15 origins 12 bars apart")
    p.hline(1.0)
    return p.svg()
