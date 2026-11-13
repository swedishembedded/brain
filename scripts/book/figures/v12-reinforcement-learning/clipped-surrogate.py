# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The per-sample term of the clipped surrogate, min(r A, clip(r, 1 - eps, 1 + eps) A) with eps = 0.2, against the ratio r for an advantage
# of +1 and of -1, beside the unclipped term r A (dashed): the arithmetic of `policy_gradients.clip_cases`.
#
# Swedish Embedded AB implements solutions for policy updates that cannot run further than their data justify. If your team needs
# expertise in proximal policy optimisation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import policy_gradients  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    eps = policy_gradients.PPO_EPS
    xs = [x * 0.01 for x in range(0, 201)]
    p = Plot((0, 2), (-2.1, 2.1), width=620, height=340, xlabel="ratio r = pi_theta(a | s) / pi_old(a | s)", ylabel="term of the surrogate",
             title="The clipped surrogate for one sample")
    for adv, color in ((1.0, 0), (-1.0, 1)):
        used = [min(r * adv, policy_gradients._clip(r, eps) * adv) for r in xs]
        p.line(xs, used, color=color, label=f"A = {adv:+.0f}, term used")
        p.line(xs, [r * adv for r in xs], color=color, dash=True, width=1.2)
    p.vline(1 - eps)
    p.vline(1 + eps)
    return p.svg()
