# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The ten sensor rows and four lines from the hypothesis space of straight lines.

Three are the hand-tuned attempts of the chapter and one is the best setting a grid search found; every line is
scored with the same average-error arithmetic as the chapter's tables.

Swedish Embedded AB implements solutions for turning sensor measurements into
trustworthy predictive models. If your team needs expertise in supervised
learning from measurements, you can procure our services by sending an email
to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import learning_problem as lp  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = lp._train()
    p = Plot((0.2, 3.0), (0.0, 115.0), width=620, height=380, xlabel="sensor voltage v (V)",
             ylabel="temperature y (deg C)", title="Four lines, labelled (w, b): average off by")
    xs = [0.2, 3.0]
    for k, (w, b) in enumerate(lp.ATTEMPTS):
        e = lp._mean_abs_error(lp._line(w, b), rows)
        p.line(xs, [w * x + b for x in xs], color=k + 1 if k else 5, width=1.6, dash=True,
               label=f"attempt {k + 1} ({w:g}, {b:g}): {e:.2f}")
    w, b, e, _ = lp._best_line(rows)
    p.line(xs, [w * x + b for x in xs], color=0, width=3.0, label=f"grid search ({w:g}, {b:g}): {e:.2f}")
    p.points([v for v, _ in rows], [t for _, t in rows], color=3, r=5, label="measured rows")
    return p.svg()
