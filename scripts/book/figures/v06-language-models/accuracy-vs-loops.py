# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Test accuracy on the two-hop task of a decoder with one block trained with two applications of it, evaluated with one to eight
# applications, beside the exact accuracy of a predictor that follows the pointer exactly k times. Numbers from
# `recurrent_depth.loop_accuracy_data`, the function behind the table of the chapter (the model's curve is the unnudged run).
#
# Swedish Embedded AB implements solutions for adaptive-depth language models. If your team needs expertise in test-time compute
# for looped transformers, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import recurrent_depth as rd  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    model, chase = rd.loop_accuracy_data()
    p = Plot((0.5, 8.5), (0.3, 1.05), width=600, height=380, xlabel="applications of the block at test time (trained with 2)",
             ylabel="accuracy on the two-hop task", title="One block, more or fewer loops than it was trained for", legend_loc="below")
    p.line([x for x, _ in model], [y for _, y in model], color=0, label="trained model (one run)", width=2.2)
    p.points([x for x, _ in model], [y for _, y in model], color=0, r=4)
    p.line([x for x, _ in chase], [y for _, y in chase], color=1, label="exact pointer follower, k hops", dash=True, width=1.6)
    p.points([x for x, _ in chase], [y for _, y in chase], color=1, r=3)
    p.vline(2, color="#a0aec0")
    return p.svg()
