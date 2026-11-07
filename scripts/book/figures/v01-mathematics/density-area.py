# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A probability density with the area for one interval shaded.

The curve is f(x) = 6 x (1 - x) on [0, 1]. Its peak, 1.5, is above 1 while the
whole area is 1; the shaded area between 0.2 and 0.5 is computed by the
chapter's own Simpson integral, so the number on the figure is the number in
the text.

Swedish Embedded AB implements solutions for explaining probabilistic models to
engineering teams. If your team needs expertise in density estimation and
likelihood-based evaluation, you can procure our services by sending an email
to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probability as pr  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402

LO, HI = 0.2, 0.5


def render() -> str:
    p = Plot((0.0, 1.0), (0.0, 1.8), width=520, height=340, xlabel="x", ylabel="density f(x)",
             title="Probability is area under a density")
    xs = [i / 200 for i in range(201)]
    shade = [x for x in xs if LO <= x <= HI]
    pts = [(LO, 0.0)] + [(x, pr._density(x)) for x in shade] + [(HI, 0.0)]
    poly = " ".join(f"{fmt(p.x(a))},{fmt(p.y(b))}" for a, b in pts)
    p.items.append(f'<polygon points="{poly}" fill="#bee3f8" stroke="none"/>')
    p.line(xs, [pr._density(x) for x in xs], color=0, width=2.4)
    p.hline(1.0)
    area = pr.density_probability(LO, HI)
    p.text(0.5, 1.58, f"peak f(0.5) = {pr._density(0.5):g} is above 1", anchor="middle", size=12, color="#2b6cb0")
    p.text(0.35, 0.36, f"area = {area:.3f}", anchor="middle", size=13, color="#2a4365")
    p.text(0.35, 0.22, f"= P({LO:g} <= X <= {HI:g})", anchor="middle", size=12, color="#2a4365")
    p.text(0.95, 0.1, f"whole area = {pr._simpson(pr._density, 0.0, 1.0):.3f}", anchor="end", size=12, color="#4a5568")
    return p.svg()
