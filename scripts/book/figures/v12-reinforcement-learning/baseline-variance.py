# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The exact total variance of the single-sample policy-gradient estimate (y - b)(e_a - pi) of `policy_gradients.estimator_moments`
# as a function of the constant baseline b: a parabola whose lowest point is b*, with the baselines of the table marked.
#
# Swedish Embedded AB implements solutions for gradient estimators whose variance is measured before they are trusted. If your
# team needs expertise in policy-gradient training, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import policy_gradients  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    g, pi, _ = policy_gradients._bandit_gradient(policy_gradients.THETA_B)
    bs, j, b_star = policy_gradients._baselines()
    xs = [x * 0.1 for x in range(-10, 141)]
    ys = [policy_gradients._total_variance(b, pi, g) for b in xs]
    p = Plot((-1, 14), (0, 32), width=620, height=340, xlabel="constant baseline b", ylabel="total variance of one estimate",
             title="One sample of the policy gradient: variance against the baseline", legend_loc=None)
    p.line(xs, ys, color=0)
    marks = ((0.0, "b = 0", "start"), (j, "J", "start"), (b_star, "b*", "end"), (12.0, "b = 12", "end"))
    for b, label, anchor in marks:
        v = policy_gradients._total_variance(b, pi, g)
        p.points([b], [v], color=1)
        dx = 0.25 if anchor == "start" else -0.25
        p.text(b + dx, v + (1.8 if label in ("J", "b*") else 1.2), f"{label}: {v:.2f}", anchor=anchor, size=11)
    return p.svg()
