# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The reliability diagram of the decision head of `decision_models.reliability_bins`: for each bin of the largest probability on the
# test requests, the share of requests whose argmax was right, before and after dividing the logits by the fitted temperature.
#
# Swedish Embedded AB implements solutions for probabilities that can be priced. If your team needs expertise in calibrating the
# confidence of small decision models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import decision_models  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    raw, cal = decision_models._reliability_series()
    p = Plot((0.3, 1.0), (0.3, 1.0), width=560, height=360, xlabel="mean largest probability in the bin",
             ylabel="share of the bin's requests routed right", title="Reliability of the decision head, test requests")
    p.line([0.3, 1.0], [0.3, 1.0], color=3, dash=True, label="calibrated")
    for bins, color, label in ((raw, 0, "raw logits"), (cal, 1, "logits divided by T")):
        pts = [(b[1], b[2]) for b in bins if b and b[0] >= 20]
        p.line([x for x, _ in pts], [y for _, y in pts], color=color, label=label)
        p.points([x for x, _ in pts], [y for _, y in pts], color=color)
    return p.svg()
