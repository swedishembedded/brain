# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The depth-3 entropy tree grown on the 90 shared training rows, drawn from the structure the algorithm returns.

Swedish Embedded AB implements solutions for models whose every decision can be read and audited. If your team
needs expertise in decision trees and interpretable classification, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import classical as cl  # noqa: E402
from figures import dot_svg  # noqa: E402

FILL = ("#bee3f8", "#fed7aa", "#c6f6d5")


def dot_source(tree) -> str:
    lines = ["digraph T {", "rankdir=TB;", 'node [shape=box, style="rounded,filled", fillcolor="#f7fafc", color="#718096"];', 'edge [color="#718096"];']
    counter = [0]

    def walk(node):
        i = counter[0]
        counter[0] += 1
        counts = ", ".join(f"{cl.CLASS_NAMES[c]}:{n}" for c, n in enumerate(node["counts"]))
        if "feature" in node:
            lines.append(f'n{i} [label="x{node["feature"] + 1} <= {node["threshold"]:.2f}\\nentropy {node["impurity"]:.2f} bits\\n[{counts}]"];')
            left, right = walk(node["left"]), walk(node["right"])
            lines.append(f'n{i} -> n{left} [label="yes"];')
            lines.append(f'n{i} -> n{right} [label="no"];')
        else:
            lines.append(f'n{i} [shape=box, style="filled", fillcolor="{FILL[node["class"]]}", label="class {cl.CLASS_NAMES[node["class"]]}\\n[{counts}]"];')
        return i

    walk(tree)
    lines.append("}")
    return "\n".join(lines)


def render() -> str:
    tx, ty = cl._dataset()["train"]
    tree = cl._build_tree(tx, ty, [0, 1, 2], cl._entropy_bits, 3)
    return dot_svg(dot_source(tree), "decision-tree")
