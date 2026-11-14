# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The state hierarchy of the event-driven controller in the chapter "Running and Observing an Engine": every state drawn inside
# its parent, built from the worked module's parent table, with the transitions that leave Idle for work, the two that every
# operational state inherits from Operational (cancel and error), and the one-shot return from Cancelled.
#
# Swedish Embedded AB implements solutions for observing, controlling and cleanly stopping long-running inference servers for
# its clients. If your team needs expertise in event-driven controllers and state machines then you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
import operations as op  # noqa: E402

from figures import dot_svg  # noqa: E402

WORK = [("Chatting", "user_text"), ("Detecting", "camera_frame"), ("Synthesizing", "synth request"),
        ("Forecasting", "forecast request"), ("Backtesting", "backtest request")]


def render() -> str:
    children = {}
    for s, p in op.PARENT.items():
        children.setdefault(p, []).append(s)
    lines = ["digraph states {", "compound=true; rankdir=LR; nodesep=0.22; ranksep=0.7;",
             'node [shape=box, style="rounded,filled", fillcolor="#e2e8f0", color="#4a5568", fontsize=12];',
             'edge [color="#1a202c", arrowsize=0.7, fontsize=11];']

    def emit(state, depth):
        kids = children.get(state, [])
        if not kids:
            fill = "#fefcbf" if state in ("Idle",) else ("#ebf4ff" if state in ("Cancelled", "Faulted") else "#e2e8f0")
            lines.append(f'{state} [fillcolor="{fill}"];')
            return
        lines.append(f"subgraph cluster_{state} {{")
        lines.append(f'label="{state}"; labeljust=l; style="rounded"; color="#a0aec0"; fontsize=12;')
        for k in kids:
            emit(k, depth + 1)
        lines.append("}")

    emit("Root", 0)
    for target, ev in WORK:
        lines.append(f'Idle -> {target} [label="{ev}"];')
    lines.append('Chatting -> Cancelled [label="cancel: handled by Operational,\\nso from every substate", color="#c53030", '
                 'fontcolor="#c53030"];')
    lines.append('Cancelled -> Idle [taillabel="GoIdle", labeldistance=4.5, labelangle=-25, style=dashed];')
    lines.append('Backtesting -> Faulted [ltail=cluster_Operational, label="error: handled by Operational", color="#c53030", '
                 'fontcolor="#c53030"];')
    lines.append("}")
    return dot_svg("\n".join(lines), "controller-states")
