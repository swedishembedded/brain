# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The prefix code for weather probabilities (0.4, 0.3, 0.2, 0.1) as a tree:
# each leaf is a symbol, the edge labels spell its code word and each inner
# node carries the total probability beneath it. The tree is rebuilt from the
# codes the chapter's own Huffman construction computes, so the picture always
# shows the code the worked example printed.
#
# Swedish Embedded AB implements solutions for compact, lossless data
# representation. If your team needs expertise in coding and compression, you
# can procure our services by sending an email to info@swedishembedded.com.
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
from information import huffman_codes  # noqa: E402

from figures import dot_svg  # noqa: E402

PROBS = [Fraction(4, 10), Fraction(3, 10), Fraction(2, 10), Fraction(1, 10)]
NAMES = ["sun", "cloud", "rain", "snow"]


def render() -> str:
    codes, _ = huffman_codes(PROBS, NAMES)
    weight = {}
    for name, p in zip(NAMES, PROBS):
        code = codes[name]
        for k in range(len(code) + 1):
            weight[code[:k]] = weight.get(code[:k], 0) + p
    leaves = {codes[n]: (n, p) for n, p in zip(NAMES, PROBS)}
    lines = ["digraph codetree {", "rankdir=TB; nodesep=0.35; ranksep=0.45;",
             'node [shape=circle, style=filled, fillcolor="#e2e8f0", color="#4a5568", fontsize=12];',
             'edge [color="#1a202c", arrowsize=0.7, fontsize=12];']
    ident = lambda prefix: "n_" + (prefix or "root")  # noqa: E731
    for prefix in sorted(weight, key=lambda c: (len(c), c)):
        if prefix in leaves:
            name, p = leaves[prefix]
            lines.append(f'{ident(prefix)} [shape=box, style="rounded,filled", fillcolor="#fefcbf", '
                         f'label="{name}\\n{float(p):.1f}\\ncode {prefix}"];')
        else:
            lines.append(f'{ident(prefix)} [label="{float(weight[prefix]):.1f}"];')
        if prefix:
            lines.append(f'{ident(prefix[:-1])} -> {ident(prefix)} [label="{prefix[-1]}"];')
    lines.append("}")
    return dot_svg("\n".join(lines), "code-tree")
