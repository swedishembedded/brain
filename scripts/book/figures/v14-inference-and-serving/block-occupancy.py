# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Which request holds each of the six blocks of the pool after every step of the specified server, in the project "A
# Multi-Client Inference Server". The cells are the worked module's block tables after each step (`tables_by_step`), so the
# picture cannot disagree with the chapter's step-by-step table.
#
# Swedish Embedded AB implements solutions for serving language models to many clients from a fixed amount of device memory.
# If your team needs expertise in blockd caches and continuous batching then you can procure our services by sending an email
# to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_inference_server as pis  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

UNIT, X0, Y0, ROW, CELL = 34, 90, 64, 34, 28


def render() -> str:
    steps = pis.tables_by_step()
    n = len(steps)
    W = X0 + n * UNIT + 30
    H = Y0 + pis.NBLK * ROW + 64
    d = Diagram(W, H)
    d.text(W / 2, 26, ["The request that holds each block after every step (blank: the block is free)"], size=14, weight="bold")
    for p in range(pis.NBLK):
        cy = Y0 + p * ROW + ROW / 2
        d.text(X0 - 14, cy, [f"block {p}"], anchor="end")
        for k, (_, held) in enumerate(steps):
            cx = X0 + k * UNIT + UNIT / 2
            if p in held:
                d.box(cx, cy, CELL, CELL, [str(held[p])], "accent")
    yax = Y0 + pis.NBLK * ROW + 14
    for k, (s, _) in enumerate(steps):
        d.text(X0 + k * UNIT + UNIT / 2, yax, [str(s)], size=12, color=MUTED)
    d.text(X0 + n * UNIT / 2, yax + 26, ["step"], size=12, color=MUTED)
    return d.svg()
