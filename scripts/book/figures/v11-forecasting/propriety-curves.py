# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# How much worse than the best report each report q is, in expected loss, for an event of probability 0.7 under four scoring
# rules. The data are `calibrated_decisions._excess_curve`.
#
# Swedish Embedded AB implements solutions for probabilities that can be acted on. If your team needs expertise in scoring
# rules for forecasts, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import calibrated_decisions as cd  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    qs = [k / 100 for k in range(10, 96)]
    p = Plot((0.1, 0.95), (0, 1.0), width=560, height=330, xlabel="reported probability q (the event has probability 0.7)",
             ylabel="expected loss above the best report", title="Only a proper rule is lowest at the truth", legend_loc="ur")
    for i, rule in enumerate(("absolute", "side of 1/2", "focal 2", "Brier", "log")):
        p.line(qs, cd._excess_curve(rule, qs), color=i, label=rule)
    p.vline(0.7)
    return p.svg()
