# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The chain a numerical test runs through (an instance, the code under test
# beside its reference, a measure with a threshold, the verdict a runner
# reports), with the way each link can let a wrong program through set under
# it, and the conformance suite as the frame that runs the whole chain for
# every registered implementation.
#
# Swedish Embedded AB implements solutions for test suites whose green result
# can be trusted. If your team needs expertise in testing numerical and
# machine learning code, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram, text_width

W, H = 780, 268
COLS = (113, 298, 483, 668)      # centres of the four links
BOX_W, BOX_H = 150, 52
NOTE_W, NOTE_H = 168, 70
BOX_Y, NOTE_Y = 92, 196

LINKS = [
    (["test instance", "sizes and values"], "plain"),
    (["code under test", "and its reference"], "model"),
    (["measure and", "threshold"], "model"),
    (["verdict the", "runner reports"], "result"),
]
TRAPS = [
    ["degenerate instance:", "a wrong program", "computes the same"],
    ["shared reference:", "the oracle repeats", "the same mistake"],
    ["invariant measure:", "a family of errors", "scores perfectly"],
    ["gate never runs:", "no comparison made,", "reported as a pass"],
]


def render() -> str:
    assert max(text_width(t) for t in TRAPS) + 28 <= NOTE_W, "a trap note is too narrow for its text"
    assert max(text_width(b) for b, _ in LINKS) + 16 <= BOX_W, "a box is too narrow for its text"
    d = Diagram(W, H)
    d.panel(20, 16, W - 40, H - 28, "The chain of a numerical test and where a wrong program gets through")
    boxes = [d.box(cx, BOX_Y, BOX_W, BOX_H, lines, kind) for cx, (lines, kind) in zip(COLS, LINKS)]
    for a, b in zip(boxes, boxes[1:]):
        d.arrow(a["r"], b["l"])
    for cx, box, trap in zip(COLS, boxes, TRAPS):
        note = d.note(cx, NOTE_Y, NOTE_W, NOTE_H, trap)
        d.line(box["b"], note["t"])
    return d.svg()
