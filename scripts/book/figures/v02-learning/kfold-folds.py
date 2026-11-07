# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Five-fold cross-validation as a diagram: every row is held out exactly once.

Built from the fold sizes of the chapter's worked module (30 development rows, 5
folds), so the picture cannot disagree with the table of folds. Row j of the diagram
is the j-th fit: the dark box is the fold held out and scored, the pale boxes are
the folds the model is fitted to.

Swedish Embedded AB implements solutions for estimating model quality when data
is scarce. If your team needs expertise in cross-validation design, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import figures  # noqa: E402
import generalization as g  # noqa: E402


def render() -> str:
    fold = g._folds_of_dev()
    sizes = [sum(1 for f in fold if f == j) for j in range(g.CV_K)]
    lines = ["digraph kfold {", "  rankdir=LR; nodesep=0.12; ranksep=0.25;",
             '  node [shape=box, style=filled, fontname="Helvetica", fontsize=11, width=1.1, height=0.4];',
             "  edge [style=invis];"]
    for fit in reversed(range(g.CV_K)):   # Graphviz stacks left-to-right chains bottom to top
        lines.append(f'  fit{fit} [shape=plaintext, fillcolor="#ffffff", style=solid, label="fit {fit + 1}"];')
        prev = f"fit{fit}"
        for j in range(g.CV_K):
            name = f"f{fit}_{j}"
            if j == fit:
                lines.append(f'  {name} [label="fold {j + 1}\\n{sizes[j]} rows: scored", fillcolor="#2b6cb0", fontcolor="#ffffff"];')
            else:
                lines.append(f'  {name} [label="fold {j + 1}\\n{sizes[j]} rows: fitted", fillcolor="#bee3f8"];')
            lines.append(f"  {prev} -> {name};")
            prev = name
    lines.append("}")
    return figures.dot_svg("\n".join(lines), "kfold-folds")
