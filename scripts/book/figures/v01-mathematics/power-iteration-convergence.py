# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Power iteration converges geometrically, at the rate |lambda_2| / |lambda_1|.

The matrix is the tridiagonal [[2,1,0],[1,2,1],[0,1,2]] shifted by s I for
s = -1, 0, 2 (a shift adds s to every eigenvalue and leaves the eigenvectors
alone), started from (1, 0, 0). The solid line is log10 of the distance from
each normalised iterate to the true dominant eigenvector; the dashed line is the
prediction error_5 * ratio^(k-5). The floor near 1e-16 is double-precision
round-off.

Swedish Embedded AB implements solutions for iterative numerical methods. If
your team needs expertise in eigenvalue computation and convergence analysis,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import decompositions as dc  # noqa: E402
from svgplot import Plot  # noqa: E402

STEPS = 40
FLOOR = 1e-17


def render() -> str:
    p = Plot((0, STEPS), (-17, 0.5), width=560, height=360, xlabel="iteration k",
             ylabel="log10 of |x_k - v_1|", title="Power iteration: straight lines on a log scale")
    ks = list(range(STEPS + 1))
    for color, s in enumerate(dc.SHIFTS):
        errs, vals = dc._power_errors(dc._shifted(dc.TRI3, s), [1.0, 0.0, 0.0], STEPS)
        ratio = max(abs(vals[1]), abs(vals[2])) / abs(vals[0])
        p.line(ks, [math.log10(max(e, FLOOR)) for e in errs], color=color,
               label=f"s = {s:g}: ratio {ratio:.3f}")
        anchor = errs[5]
        p.line(ks[5:], [max(math.log10(anchor * ratio ** (k - 5)), -17) for k in ks[5:]], color=color, dash=True, width=1.4)
    return p.svg()
