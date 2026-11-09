# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Parameters of one feed-forward layer against the FLOPs a token spends in it (D = 2,048), for three ways of growing it from the same
# dense layer of hidden width 4,096: adding distinct dense layers (the diagonal), applying the same layer more times (a horizontal line:
# compute without parameters) and adding experts (a vertical line: parameters without compute). Numbers from
# `recurrent_depth.params_flops_data`, which uses the layer costs of the chapter on mixtures.
#
# Swedish Embedded AB implements solutions for sizing language models. If your team needs expertise in the trade between parameters
# and compute, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import recurrent_depth as rd  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    series = rd.params_flops_data()
    p = Plot((30, 2000), (20, 2000), width=560, height=420, xlabel="MFLOPs per token in the layer (log scale)",
             ylabel="parameters of the layer, millions (log scale)", title="Three ways to make a layer bigger",
             legend_loc="below", xlog=True, ylog=True)
    for color, (label, pts) in zip((3, 0, 1), series):
        p.line([x for x, _ in pts], [y for _, y in pts], color=color, label=label, width=1.6)
        p.points([x for x, _ in pts], [y for _, y in pts], color=color, r=3.5)
    (_, distinct), (_, looped), (_, mixture) = series
    p.text(distinct[-1][0] * 0.93, distinct[-1][1] * 1.12, f"16 layers: {distinct[-1][1]:.0f} M", anchor="end", size=10, color="#4a5568")
    p.text(looped[-1][0] * 0.93, looped[-1][1] * 0.82, f"16 passes: {looped[-1][1]:.0f} M", anchor="end", size=10, color="#4a5568")
    p.text(mixture[-1][0] * 1.12, mixture[-1][1], f"256 experts: {mixture[-1][1]:.0f} M", size=10, color="#4a5568")
    return p.svg()
