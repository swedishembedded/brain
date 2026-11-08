# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Exact-match accuracy against the length of the input for the model without attention and the model with it (the two
# runs of each pooled). Numbers from `seq2seq.accuracy_data`; lengths to the right of the dotted line were not trained on.
#
# Swedish Embedded AB implements solutions for models that turn one sequence into another. If your team needs expertise in
# evaluating sequence models as inputs grow longer, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import seq2seq as s2s  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    exact, _ = s2s.accuracy_data()
    xs = s2s.TEST_LENGTHS
    p = Plot((0.5, 12.5), (-0.02, 1.04), width=640, height=380, xlabel="digits in the input, T", ylabel="exact-match accuracy",
             title="One vector against attention")
    for variant, color in (("plain", 1), ("attention", 3)):
        p.line(xs, exact[variant], color=color, label=s2s.LABEL[variant])
        p.points(xs, exact[variant], color=color, r=3)
    p.vline(s2s.LMAX_TRAIN + 0.5)
    p.text(s2s.LMAX_TRAIN + 0.7, 0.08, "not trained on", size=12, color="#4a5568")
    return p.svg()
