# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bootstrap distribution of the accuracy gap B - A in the chapter's second scenario.

The histogram is the 2000 resampled gaps the chapter's table summarises. The
dashed lines mark the 2.5th and 97.5th percentiles (the interval), the
observed gap, and zero, the value the null hypothesis claims.

Swedish Embedded AB implements solutions for deciding, from finite test sets,
whether one model is really better than another. If your team needs expertise
in statistical evaluation of machine learning systems, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import statistics_ch as st  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402

LO, HI, BINS = -0.03, 0.10, 26


def render() -> str:
    reps = st.bootstrap_replicates(2)
    ordered = sorted(reps)
    lo, hi = st._percentile(ordered, 0.025), st._percentile(ordered, 0.975)
    width = (HI - LO) / BINS
    counts = [0] * BINS
    for v in reps:
        if LO <= v < HI:
            counts[int((v - LO) / width)] += 1
    top = 1.15 * max(counts) / (len(reps) * width)
    p = Plot((LO, HI), (0.0, top), width=560, height=320, xlabel="gap B - A on a resample of the 500 questions",
             ylabel="density", title="2000 bootstrap resamples")
    for i, c in enumerate(counts):
        h = c / (len(reps) * width)
        x0 = LO + i * width
        p.items.append(f'<rect x="{fmt(p.x(x0))}" y="{fmt(p.y(h))}" width="{fmt(p.x(x0 + width) - p.x(x0) - 0.6)}" '
                       f'height="{fmt(p.y(0) - p.y(h))}" fill="#90cdf4"/>')
    p.vline(lo, color="#2f855a")
    p.vline(hi, color="#2f855a")
    p.vline(0.0, color="#c53030")
    p.vline(sum(reps) / len(reps), color="#2b6cb0", dash=False)
    pad = 0.004 * (HI - LO)  # keep labels off their own guide lines
    p.text(lo - pad, top * 0.95, "2.5%", anchor="end", size=11, color="#2f855a")
    p.text(hi + pad, top * 0.95, "97.5%", anchor="start", size=11, color="#2f855a")
    p.text(0.0 - pad, top * 0.55, "no difference", anchor="end", size=11, color="#c53030")
    return p.svg()
