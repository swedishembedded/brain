# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The Markov chain that a policy leaves of the inventory problem: one node per stock, one edge per positive transition probability of
# the action the policy takes there, drawn from the transition tables of `mdps.inventory_mdp`.
#
# Swedish Embedded AB implements solutions for sequential decisions under uncertainty. If your team needs expertise in Markov decision
# processes and the policies that solve them, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mdps  # noqa: E402
from figures import dot_svg  # noqa: E402

POLICY = (1, 0, 0, 0)


def dot_source() -> str:
    lines = ["digraph M {", "rankdir=LR;", 'node [shape=circle, style=filled, fillcolor="#ebf4ff", color="#718096", fixedsize=true, width=0.9];',
             'edge [color="#718096"];']
    for s in range(mdps.N_STATES):
        a = POLICY[s]
        lines.append(f'n{s} [label="stock {s}\\norder {a}"];')
    for s in range(mdps.N_STATES):
        for j, p in enumerate(mdps.PROB[s][POLICY[s]]):
            if p > 0:
                lines.append(f'n{s} -> n{j} [label="{p:.1f}"];')
    lines.append("}")
    return "\n".join(lines)


def render() -> str:
    return dot_svg(dot_source(), "inventory-chain")
