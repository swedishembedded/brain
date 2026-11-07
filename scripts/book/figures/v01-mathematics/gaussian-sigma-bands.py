# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The standard Gaussian density with the areas within 1, 2 and 3 standard deviations shaded.

The curve is e^(-x^2/2) / sqrt(2 pi) and every area printed on the figure is the
chapter's own Simpson integral, so the 68-95-99.7 percentages are computed, not
recalled.

Swedish Embedded AB implements solutions for explaining probabilistic models to
engineering teams. If your team needs expertise in noise modelling and
uncertainty estimation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import distributions as ds  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402

BANDS = ((3.0, "#bee3f8"), (2.0, "#90cdf4"), (1.0, "#63b3ed"))


def render() -> str:
    p = Plot((-4.0, 4.0), (0.0, 0.46), width=560, height=340, xlabel="z (standard deviations from the mean)",
             ylabel="density", title="Share of the area within k standard deviations")
    for k, color in BANDS:
        xs = [-k + 2 * k * i / 120 for i in range(121)]
        pts = [(-k, 0.0)] + [(x, ds._phi(x)) for x in xs] + [(k, 0.0)]
        poly = " ".join(f"{fmt(p.x(a))},{fmt(p.y(b))}" for a, b in pts)
        p.items.append(f'<polygon points="{poly}" fill="{color}" stroke="none"/>')
    grid = [-4 + 8 * i / 400 for i in range(401)]
    p.line(grid, [ds._phi(x) for x in grid], color=0, width=2.4)
    for k, _ in BANDS:
        p.vline(k, dash=True)
        p.vline(-k, dash=True)
    p.items.append(f'<rect x="{fmt(p.x(-3.98))}" y="{fmt(p.y(0.455))}" width="{fmt(p.x(-1.55) - p.x(-3.98))}" '
                   f'height="{fmt(p.y(0.345) - p.y(0.455))}" fill="#ffffff"/>')
    for k, y in ((1.0, 0.435), (2.0, 0.40), (3.0, 0.365)):
        area = ds._simpson(ds._phi, -k, k, 4000)
        p.text(-3.95, y, f"within {k:g} sigma: {100 * area:.2f}%", size=12, color="#2a4365")
    return p.svg()
