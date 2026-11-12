# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The skill that four ways of building and judging the same forecasting system report on the test block, the skill the same models
# have in the deployment block that follows it, and the skill the system shows on random walks, where it is zero. The data are
# `project_forecaster.figure_protocols` and the control rows of `project_forecaster.control`.
#
# Swedish Embedded AB implements solutions for forecasts that state their own uncertainty and are judged honestly. If your team needs
# expertise in honest forecast evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_forecaster as pfc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = pfc.figure_protocols()
    ctl = pfc.runs()[0]["control"]
    names = ("honest", "future scaler", "random split", "recalibration on the test block")
    xs = [1, 2, 3, 4]
    p = Plot((0.5, 4.5), (-0.8, 0.7), width=560, height=340, xlabel="1 as specified   2 future scaler   3 random split   4 test-block recalibration",
             ylabel="skill against the baseline", title="Skill reported, deployed and on random walks", legend_loc="lr")
    p.hline(0.0)
    p.points(xs, [r[1]["test"]["skill"]["recal"] for r in rows], color=0, label="test block", r=5)
    p.points(xs, [r[1]["deploy"]["skill"]["recal"] for r in rows], color=1, label="deployment block", r=5)
    p.points(xs, [ctl[n]["skill"] for n in names], color=2, label="random-walk control", r=5)
    return p.svg()
