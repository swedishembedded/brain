# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The attention term of the forward FLOPs per token over the parameter term, as the context T (or the number of packed rows R) grows,
# for the stated small decoder, Qwen3 8B, the text tower of Gemma-4 (40 sliding layers cap their keys at 1,024) and the diffusion
# transformer of MiniMax-H3 (every row attends to every row). Numbers from `model_cost.ratio_data`; the dashed line at 1 is where
# the two terms are equal.
#
# Swedish Embedded AB implements solutions for sizing language models before they are run. If your team needs expertise in
# FLOP budgeting for long-context models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import model_cost as mc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    series = mc.ratio_data()
    p = Plot((64, 131072), (0.001, 100), width=620, height=400, xlabel="keys seen by a token, or packed rows (log scale)",
             ylabel="attention term / parameter term (log scale)", title="When the attention term overtakes the parameters",
             legend_loc="below", xlog=True, ylog=True)
    p.hline(1.0)
    for color, (label, pts) in zip((0, 1, 2, 3), series):
        # MiniMax-H3 nearly coincides with Qwen3 8B, so it is drawn dashed on top of it.
        p.line([x for x, _ in pts], [y for _, y in pts], color=color, label=label, width=1.8, dash=(color == 3))
    return p.svg()
