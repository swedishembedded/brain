# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The relative error of every hidden state of a 6-layer decoder, 32-bit against the stored 64-bit golden, for a correct import and
# for an import that exchanged gate and up in layer 3, against the bound from the rounding count. Numbers from
# `importing_models.bisection_data`, the function behind the table beside the figure.
#
# Swedish Embedded AB implements solutions for proving that a port computes the same function as its reference. If your team needs
# expertise in layer-by-layer numerical parity, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importing_models as im  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = im.bisection_data()
    xs = list(range(len(rows)))
    p = Plot((-0.3, len(rows) - 0.7), (-8.5, 0), width=620, height=380, xlabel="hidden state (0 = embeddings, k = output of layer k-1, 7 = final norm)",
             ylabel="log10 of rel_l2 against the golden", title="Where the error leaves the budget", legend_loc="ul")
    p.line(xs, [math.log10(r[3]) for r in rows], color="#4a5568", label="bound: rounding count x 2^-24", dash=True, width=1.6)
    p.line(xs[1:], [math.log10(r[1]) for r in rows[1:]], color=0, label="correct import", width=1.8)
    p.points(xs[1:], [math.log10(r[1]) for r in rows[1:]], color=0)
    p.line(xs[1:], [math.log10(r[2]) for r in rows[1:]], color=3, label="gate and up exchanged in layer 3", width=1.8)
    p.points(xs[1:], [math.log10(r[2]) for r in rows[1:]], color=3)
    p.vline(im.BIS_BUG_LAYER + 1)
    p.text(im.BIS_BUG_LAYER + 1.1, -2.8, "first tap above the bound", size=11)
    return p.svg()
