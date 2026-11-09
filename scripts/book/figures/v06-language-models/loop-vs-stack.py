# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Three ways to spend three block applications: three distinct blocks, one block applied three times (the looped transformer), and the
# recurrent-depth layout (a prelude that embeds the input once, a core applied r times to a latent state that starts as noise and
# re-reads the prelude's output at every step, and a coda that reads the answer out). The shapes are those of the chapter's
# counting table and of the two crates it reads; nothing here is computed.
#
# Swedish Embedded AB implements solutions for weight-shared language models. If your team needs expertise in looped and
# recurrent-depth transformer architectures, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 600


def render() -> str:
    d = Diagram(W, H)

    # Row 1: three distinct blocks.
    d.panel(10, 10, 980, 150, "three distinct blocks: three sets of weights")
    y = 100
    x = d.oval(80, y, 90, 50, ["x"], "accent")
    prev = x
    for i, cx in enumerate((280, 480, 680), start=1):
        b = d.box(cx, y, 150, 56, [f"block {i}", f"weights W{i}"])
        d.arrow(prev["r"], b["l"])
        prev = b
    out = d.oval(880, y, 100, 50, ["head"], "result")
    d.arrow(prev["r"], out["l"])

    # Row 2: one block three times.
    d.panel(10, 180, 980, 170, "one block applied three times: one set of weights")
    y = 255
    x = d.oval(80, y, 90, 50, ["x"], "accent")
    prev = x
    boxes = []
    for i, cx in enumerate((280, 480, 680), start=1):
        b = d.box(cx, y, 150, 56, [f"pass {i}", "weights W"])
        d.arrow(prev["r"], b["l"])
        boxes.append(b)
        prev = b
    out = d.oval(880, y, 100, 50, ["head"], "result")
    d.arrow(prev["r"], out["l"])
    bottom = y + 28 + 28
    for b in boxes:
        d.line(b["b"], (b["b"][0], bottom))
    d.line((boxes[0]["b"][0], bottom), (boxes[2]["b"][0], bottom))
    d.text(480, bottom + 18, ["the same W is read three times; the cost grows with the passes, the parameters do not"], size=12, color=MUTED)

    # Row 3: prelude, core r times, coda.
    d.panel(10, 370, 980, 220, "recurrent depth: prelude, core applied r times, coda")
    y = 470
    tokens = d.oval(60, y, 84, 50, ["tokens"], "accent")
    prelude = d.box(190, y, 110, 56, ["prelude", "(runs once)"])
    d.arrow(tokens["r"], prelude["l"])
    cores = []
    for i, cx in enumerate((370, 530, 690), start=1):
        cores.append(d.box(cx, y, 130, 64, [f"step {i}", "adapter + core"]))
    s0 = d.oval(370, 552, 130, 44, ["state s0, random"], "accent")
    d.arrow(s0["t"], cores[0]["b"])
    d.arrow(cores[0]["r"], cores[1]["l"])
    d.arrow(cores[1]["r"], cores[2]["l"])
    coda = d.box(840, y, 100, 56, ["coda", "(runs once)"])
    d.arrow(cores[2]["r"], coda["l"])
    head = d.oval(950, y, 66, 50, ["head"], "result")
    d.arrow(coda["r"], head["l"])
    # The prelude's output is re-read by every step: a bus above the core boxes.
    bus = 428
    d.path([prelude["t"], (prelude["t"][0], bus), (cores[2]["t"][0], bus), cores[2]["t"]])
    for c in cores[:2]:
        d.path([(c["t"][0], bus), c["t"]])
    d.text(370, bus - 16, ["the prelude's output is concatenated with the state again at every step"], size=12, color=MUTED, anchor="start")
    d.text(610, y + 52, ["... r steps in all"], size=12, color=MUTED)
    return d.svg()
