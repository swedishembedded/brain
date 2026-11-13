# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The exact return J(theta) of the shop after each episode, averaged over the ten runs of `policy_gradients.policy_gradient_runs`, for
# REINFORCE, REINFORCE with a learned baseline and one-step actor-critic.
#
# Swedish Embedded AB implements solutions for learners that are compared on the same problem with the same measure. If your team
# needs expertise in policy-gradient reinforcement learning, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import policy_gradients  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((1, policy_gradients.RUN_EPISODES), (2, 10.2), width=620, height=340, xlabel="episodes of 100 days (log scale)",
             ylabel="exact return J(theta), mean of 10 runs", title="Three policy-gradient learners on the shop", xlog=True)
    for i, (method, label) in enumerate(policy_gradients.RUN_METHODS):
        curve = policy_gradients._mean_curve(method)
        ks = list(range(1, policy_gradients.RUN_EPISODES + 1))
        p.line(ks, curve, color=i, label=label)
    p.hline(policy_gradients.J_STAR)
    return p.svg()
