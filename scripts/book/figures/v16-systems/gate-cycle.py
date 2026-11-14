# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Six crates of the chapter "The Architecture of a Machine Learning Engine" in two workspaces: on the left the gate inside
# rl and bench with the edge qwen3 -> rl that a model crate needs to reach it, on the right the gate moved down into its
# own crate. Rows are layers, highest at the top. The edges, the upward edge and the cycle group are computed by the
# worked module (`deps_gate_in_rl`, `DEPS_NOW`, `upward_edges`, `components`), restricted to these six crates.
#
# Swedish Embedded AB implements solutions for structuring large machine learning engines for its clients. If your team
# needs expertise in software architecture for machine learning systems then you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
import engine_architecture as ea  # noqa: E402

from figures import dot_svg  # noqa: E402

SHOWN = ["data", "model", "promote", "qwen3", "bench", "rl"]


def _cluster(name, title, deps):
    shown = [c for c in SHOWN if c in deps]
    sub = {c: [d for d in deps[c] if d in shown] for c in shown}
    up = set(ea.upward_edges(ea.LAYER, sub))
    cyc = {c for g in ea.components(sub) for c in g}
    out = [f"subgraph cluster_{name} {{", f'label="{title}"; fontsize=12; color="#a0aec0"; style=rounded;']
    for layer in (5, 4, 3):
        row = [f"{name}_{c}" for c in shown if ea.LAYER[c] == layer]
        out.append(f'{name}_L{layer} [label="layer {layer}", shape=plaintext, style="", fontcolor="#4a5568"];')
        out.append("{rank=same; " + f"{name}_L{layer}; " + "; ".join(row) + ";}")
    out.append(f"{name}_L5 -> {name}_L4 -> {name}_L3 [style=invis];")
    for c in shown:
        fill = "#fed7d7" if c in cyc else ("#fefcbf" if c == "promote" else "#e2e8f0")
        out.append(f'{name}_{c} [label="{c}", fillcolor="{fill}"];')
    for c in shown:
        for d in sub[c]:
            if (c, d) in up:
                out.append(f'{name}_{c}:w -> {name}_{d}:sw [color="#c53030", penwidth=1.8, constraint=false, '
                           'xlabel="upward   ", fontcolor="#c53030"];')
            else:
                out.append(f"{name}_{c} -> {name}_{d};")
    out.append("}")
    return out


def render() -> str:
    lines = ["digraph gate {", "rankdir=TB; nodesep=0.3; ranksep=0.55; newrank=true;",
             'node [shape=box, style="rounded,filled", color="#4a5568", fontsize=12];',
             'edge [color="#1a202c", arrowsize=0.7, fontsize=11];']
    lines += _cluster("a", "gate inside rl and bench, plus qwen3 -> rl", ea.deps_gate_in_rl())
    lines += _cluster("b", "gate in its own crate, promote", ea.DEPS_NOW)
    lines.append("}")
    return dot_svg("\n".join(lines), "gate-cycle")
