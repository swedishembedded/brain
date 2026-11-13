# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The closed-form optimum of the KL-regularised objective on the five answers of `llm_alignment.kl_optimum`, pi*(y) = pi_ref(y)
# exp(r(y) / beta) / Z, as a function of beta on a logarithmic axis: the policy is the reference for a large beta and puts its mass
# on the answer with the highest reward as beta falls.
#
# Swedish Embedded AB implements solutions for tuning language models toward a reward without letting them forget what they were. If
# your team needs expertise in KL-regularised reinforcement learning, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import llm_alignment  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    lo, hi, n = 0.05, 20.0, 80
    betas = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    curves = [llm_alignment._pi_star(llm_alignment.R_TRUE, b) for b in betas]
    p = Plot((lo, hi), (0, 1), width=620, height=340, xlabel="beta (log scale)", ylabel="probability pi*(answer)",
             title="The optimum of the KL-regularised objective against beta", xlog=True)
    for k in range(llm_alignment.K):
        p.line(betas, [c[k] for c in curves], color=k, label=f"a{k}  {llm_alignment.ANSWERS[k]}")
    p.vline(llm_alignment.PAIR_BETA)
    return p.svg()
