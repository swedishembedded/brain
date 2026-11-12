# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The largest change of value iteration against the sweep, for three discounts, with the geometric bound change_1 * gamma^(k-1) that
# the contraction argument gives. The data are `mdps._value_iteration`, the runs of `mdps.contraction_prediction`.
#
# Swedish Embedded AB implements solutions for dynamic programming that is stopped when the bound says it may be. If your team needs
# expertise in value iteration and its convergence, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mdps  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    runs = [(g, mdps._value_iteration(g)[2]) for g in (0.5, 0.9, 0.99)]
    p = Plot((1, 3000), (mdps.TOL / 2, 5.0), width=620, height=340, xlabel="sweep k of value iteration",
             ylabel="largest change of V in sweep k", title="Value iteration against its contraction bound", xlog=True, ylog=True)
    for i, (g, deltas) in enumerate(runs):
        n = len(deltas)
        ks = [k for k in range(1, n + 1) if k <= 40 or k % 10 == 0 or k == n]   # thinned: the curves are smooth
        p.line(ks, [deltas[k - 1] for k in ks], color=i, label=f"gamma = {g:g}")
        p.line(ks, [deltas[0] * g ** (k - 1) for k in ks], color=i, dash=True)
    p.hline(mdps.TOL)
    return p.svg()
