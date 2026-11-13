# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The runs of `llm_alignment.grpo_checker_runs`: the mean over ten runs of the reward the weak checker pays and of the exact accuracy
# (the reward the exact checker would pay), for GRPO trained with the weak checker and for GRPO trained with the exact checker.
#
# Swedish Embedded AB implements solutions for reinforcement learning against rewards that cannot be argued with. If your team needs
# expertise in verifiable rewards and in finding the reward hacks of a checker, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import llm_alignment  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    steps = list(range(llm_alignment.GRPO_STEPS + 1))
    p = Plot((0, llm_alignment.GRPO_STEPS), (0, 1.05), width=620, height=340, xlabel="GRPO steps",
             ylabel="mean over 10 runs, exact from the policy", title="Training against a weak checker and against the exact one")
    p.line(steps, [llm_alignment._grpo_mean("weak", s, 0) for s in steps], color=0, label="weak checker's reward, trained on the weak checker")
    p.line(steps, [llm_alignment._grpo_mean("weak", s, 1) for s in steps], color=1, label="exact accuracy, trained on the weak checker")
    p.line(steps, [llm_alignment._grpo_mean("exact", s, 1) for s in steps], color=2, label="exact accuracy, trained on the exact checker")
    return p.svg()
