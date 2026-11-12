# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The risk-coverage curve of the classifier that answers by the side of 1/2 on three sets of probabilities and two references, on the 20000 fresh
# examples. The data are `calibrated_decisions._risk_curves`.
#
# Swedish Embedded AB implements solutions for probabilities that can be acted on. If your team needs expertise in selective
# prediction and abstention, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import calibrated_decisions as cd  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0, 1), (0, 0.3), width=560, height=330, xlabel="coverage (share of examples answered)",
             ylabel="error rate among the answered", title="Answering only what is confident", legend_loc="ul")
    for i, (name, curve) in enumerate(cd._risk_curves().items()):
        step = 100
        pts = curve[step - 1::step]
        p.line([c for c, _ in pts], [r for _, r in pts], color=i, label=name, dash=name in ("perfect ranking", "no ranking"))
    return p.svg()
