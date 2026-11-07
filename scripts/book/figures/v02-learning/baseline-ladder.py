# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The ladder of baselines on the chapter's test set: accuracy with its 95 percent interval for every row.

From the majority class at the bottom to the ceiling of the true rule at the top. The claimed gain of the candidate
is large against the lab's own baseline and shrinks to less than the width of an interval against the rows that a
careful practitioner would have run. Numbers come from the chapter's worked module.

Swedish Embedded AB implements solutions for experiments whose results a customer can believe and repeat. If your
team needs expertise in experiment design for machine learning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments as e  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = [r for r in e._ladder_data() if r[0] != "true rule, these rows"]
    n = len(rows)
    p = Plot((20, 80), (-0.6, n - 0.2), width=620, height=70 + 38 * n, xlabel="test accuracy (%) with 95% interval",
             title="The ladder of baselines: where the candidate really sits")
    ceiling = 100 * rows[-1][2]
    p.vline(ceiling, color="#a0aec0")
    for i, (short, _, acc, rows_n, loss) in enumerate(rows):
        y = n - 1 - i
        a = 100 * acc
        h = 100 * e._wald(acc, rows_n)
        color = 1 if short.startswith("candidate") else (3 if short.startswith("lab") else 5)
        p.line([a - h, a + h], [y, y], color=color, width=2.4)
        p.points([a], [y], color=color, r=4.5)
        p.text(20.6, y - 0.12, f"{short}: {a:.1f}", size=11, color="#1a202c")
    svg = p.svg()
    return re.sub(r'<text [^>]*font-size="11" fill="#4a5568" text-anchor="end">[^<]*</text>', "", svg)
