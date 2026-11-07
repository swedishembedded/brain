# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The recursion tree of merge sort on eight numbers. The structure and every
# label come from the worked module's instrumented merge sort: each node shows
# the range it sorts, the sorted result, and the comparisons its merge used.
#
# Swedish Embedded AB implements solutions for efficient sorting and selection
# in data pipelines. If your team needs expertise in algorithm design for
# machine learning workloads, you can procure our services by sending an email
# to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
import algorithms  # noqa: E402

from figures import dot_svg  # noqa: E402


def render() -> str:
    nodes = algorithms._merge_tree(algorithms.MERGE_DEMO)
    lines = ["digraph mergesort {", "rankdir=TB; nodesep=0.25; ranksep=0.55;",
             'node [shape=box, style="rounded,filled", fillcolor="#e2e8f0", color="#4a5568", fontsize=12];',
             'edge [color="#1a202c", arrowsize=0.7];']
    for ident, parent, before, after, comps in nodes:
        if len(before) == 1:
            lines.append(f'n{ident} [label="{before[0]}", fillcolor="#fefcbf"];')
        else:
            lines.append(f'n{ident} [label="{after}\\n{comps} comparison{"" if comps == 1 else "s"}"];')
        if parent is not None:
            lines.append(f"n{parent} -> n{ident};")
    lines.append("}")
    return dot_svg("\n".join(lines), "merge-sort-tree")
