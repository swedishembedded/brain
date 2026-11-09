# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The largest attention logit of one head on six tokens as its query and key weights are multiplied by a growing factor, with and
# without an RMS norm of the queries and keys, against the bound sqrt(8) of the normalized logit. Numbers from
# `modern_decoder.logit_growth_data`, the function behind the table of the chapter.
#
# Swedish Embedded AB implements solutions for stable training of transformer models. If your team needs expertise in attention
# logit control, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import modern_decoder as md  # noqa: E402
from svgplot import Plot  # noqa: E402

SCALES = [0.5 * k for k in range(1, 33)]


def render() -> str:
    data = md.logit_growth_data(SCALES)
    p = Plot((0, 16.5), (-20, 400), width=620, height=380, xlabel="factor on W_Q and W_K",
             ylabel="largest absolute attention logit", title="Logits grow with the weights unless q and k are normalized", legend_loc="ul")
    p.line(SCALES, [d[1] for d in data], color=3, label="no normalization", width=1.8)
    p.line(SCALES, [d[2] for d in data], color=0, label="RMS norm of q and k", width=1.8)
    p.hline(math.sqrt(md.LOGIT_DH), color="#4a5568")
    p.text(16.3, math.sqrt(md.LOGIT_DH) + 14, "bound sqrt(8) = 2.83", anchor="end", size=11)
    marks = [d for d in md.logit_growth_data() if d[0] in (1.0, 4.0, 16.0)]
    p.points([d[0] for d in marks], [d[1] for d in marks], color=3, r=4)
    for d in marks:
        p.text(d[0] - 0.4, d[1] + 8, f"{d[1]:.1f}", anchor="end", size=11)
    return p.svg()
