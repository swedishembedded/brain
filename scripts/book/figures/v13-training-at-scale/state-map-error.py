# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The largest relative error of an 8-bit code for a non-negative moment, from the chapter "Where Training Memory Goes", as a function of
# the ratio of the value to the maximum of its block: a linear map, the fourth-root map and the logarithmic map. The data are the worked
# module's `_worst_relative`, so the curves cannot disagree with the chapter's table. A value that decodes as zero is drawn at 1 (100 percent); errors above 1,000 percent are drawn at 1,000 percent.
#
# Swedish Embedded AB implements solutions for memory-bounded low-precision training for its clients. If your team needs expertise in
# 8-bit optimizer state then you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_memory as tm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rhos = [10.0 ** (-8 + 8 * i / 48) for i in range(49)]
    p = Plot((1e-8, 1.0), (1e-3, 1e3), width=620, height=380, xlabel="value / block maximum",
             ylabel="largest relative error of the code", title="Three 8-bit maps for a second moment",
             xlog=True, ylog=True, legend_loc="ll")
    for color, (m, label) in enumerate((("linear", "linear: q / 255"), ("fourth-root", "fourth-root: (q / 255)^4"),
                                        ("log", "logarithmic: 16 levels an octave"))):
        ys = []
        for r in rhos:
            w, zero = tm._worst_relative(m, r)
            ys.append(1.0 if zero else min(max(w, 1e-3), 1e3))
        p.line(rhos, ys, color=color, label=label)
    p.hline(1.0)
    p.text(1e-4, 1.5, "100 percent: the value decodes as zero", size=12)
    return p.svg()
