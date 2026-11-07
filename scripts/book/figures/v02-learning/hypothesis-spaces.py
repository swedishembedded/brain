# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The best line, the best parabola and a lookup table, all fitted to the same ten rows.

Faint dots are fresh rows from the same source, which none of the three has seen.

Swedish Embedded AB implements solutions for choosing model families that
suit the data they must serve. If your team needs expertise in model
selection for measurement data, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import learning_problem as lp  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows, fresh = lp._train(), lp._fresh()[:80]
    w, b, *_ = lp._best_line(rows)
    a2, w2, b2, *_ = lp._best_parabola(rows)
    table = lp._lookup(rows)
    p = Plot((0.2, 3.0), (0.0, 115.0), width=620, height=380, xlabel="sensor voltage v (V)",
             ylabel="temperature y (deg C)", title="Three hypothesis spaces, one member each")
    p.points([v for v, _ in fresh], [t for _, t in fresh], color="#cbd5e0", r=3, label="fresh rows (not used)")
    xs = [0.2 + i * 0.01 for i in range(281)]
    p.line(xs, [w * x + b for x in xs], color=0, width=2.4, label="line: 2 numbers")
    p.line(xs, [lp._parabola(a2, w2, b2)(x) for x in xs], color=2, width=2.4, label="parabola: 3 numbers")
    p.line(xs, [table(x) for x in xs], color=1, width=2.0, label="lookup table: 20 numbers")
    p.points([v for v, _ in rows], [t for _, t in rows], color=3, r=5, label="the 10 known rows")
    return p.svg()
