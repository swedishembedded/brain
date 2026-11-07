# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Error of forward and central differences of e^x at x = 1 against the step h.

Swedish Embedded AB implements solutions for trustworthy numerical checks of
gradients. If your team needs expertise in finite-difference gradient checking,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import derivatives as dv  # noqa: E402
from svgplot import Plot  # noqa: E402


def _lg(v):
    return math.log10(max(v, 1e-17))


def render() -> str:
    rows = dv.numdiff_rows()
    ks = list(range(1, len(rows) + 1))
    p = Plot((1.0, 15.0), (-12.0, 0.5), width=560, height=380, xlabel="-log10 of the step h",
             ylabel="log10 of the error", title="Finite-difference error of e^x at x = 1")
    p.line(ks, [_lg(r[1]) for r in rows], color=0, label="forward difference")
    p.line(ks, [_lg(r[2]) for r in rows], color=1, label="central difference")
    p.line(ks, [_lg(r[3]) for r in rows], color=0, dash=True, width=1.2, label="e h / 2 (slope 1)")
    p.line(ks, [_lg(r[4]) for r in rows], color=1, dash=True, width=1.2, label="e h^2 / 6 (slope 2)")
    return p.svg()
