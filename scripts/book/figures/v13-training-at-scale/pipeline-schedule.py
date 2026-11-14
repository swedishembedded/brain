# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The schedule of one step of a four-stage pipeline with four micro-batches, from the chapter "Parallel and Distributed Training":
# forwards (F), rebuilds of a stage's forward (r) and backwards (B) of each stage, in units of one stage's forward work. The cells
# are the worked module's `_simulate`, so the picture cannot disagree with the chapter's table.
#
# Swedish Embedded AB implements solutions for spreading training over several devices for its clients. If your team needs
# expertise in pipeline-parallel training you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import parallelism  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

P, M = 4, 4
UNIT, X0, Y0, ROW, CH = 28, 110, 70, 46, 36
W = X0 + 25 * UNIT + 30
H = Y0 + P * ROW + 110
KIND = {"F": "accent", "R": "plain", "B": "result"}


def render() -> str:
    tasks, span = parallelism._simulate(P, M)
    d = Diagram(W, H)
    d.text(W / 2, 24, [f"Pipeline of {P} stages, {M} micro-batches: the step lasts {span} units"], size=14, weight="bold")
    for s in range(P):
        cy = Y0 + s * ROW + ROW / 2
        d.text(X0 - 14, cy, [f"stage {s}"], anchor="end")
        for kind, k, a, b in tasks[s]:
            label = {"F": f"F{k}", "R": f"r{k}", "B": f"B{k}"}[kind]
            d.box(X0 + (a + b) / 2 * UNIT, cy, (b - a) * UNIT - 2, CH, [label], KIND[kind])
    yax = Y0 + P * ROW + 8
    for u in range(0, span + 1, 5):
        d.text(X0 + u * UNIT, yax + 10, [str(u)], size=12, color=MUTED)
    d.text(X0 + span * UNIT / 2, yax + 34, ["time in units of one stage's forward work"], size=12, color=MUTED)
    lx, ly = X0, yax + 62
    for kind, name in (("F", "forward of micro-batch k"), ("R", "rebuild of the forward"), ("B", "backward")):
        d.swatch(lx, ly - 8, 18, 16, KIND[kind])
        d.text(lx + 26, ly, [name], size=12, anchor="start")
        lx += 220
    return d.svg()
