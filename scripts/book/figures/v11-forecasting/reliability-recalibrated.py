# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Reliability diagram of the raw model and of two recalibrations fitted on the fit set, on the 20000 fresh examples. The data are
# `calibrated_decisions._reliability_series`.
#
# Swedish Embedded AB implements solutions for probabilities that can be acted on. If your team needs expertise in calibrating
# model outputs, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import calibrated_decisions as cd  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0, 1), (0, 1), width=520, height=420, xlabel="mean forecast in the bin", ylabel="share of labels 1 in the bin",
             title="Reliability on 20000 fresh examples", legend_loc="lr")
    p.line([0, 1], [0, 1], color="#a0aec0", dash=True)
    for i, (name, (xs, ys)) in enumerate(cd._reliability_series().items()):
        p.line(xs, ys, color=i, label=name)
        p.points(xs, ys, color=i)
    return p.svg()
