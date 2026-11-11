# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The forward lattice of connectionist temporal classification for the label `ab` in four frames: the states of the extended label
# `-a-b-` against time, only the states that lie on some path that writes `ab`, each with its forward probability alpha. The values are
# `speech_recognition.ctc_lattice_data`, the arithmetic of the chapter's table.
#
# Swedish Embedded AB implements solutions for aligning speech with text without frame-level labels. If your team needs expertise in
# alignment-free speech recognition training, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speech_recognition as sr  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

X0, DX, Y0, DY = 150, 120, 78, 62
NODE_W, NODE_H = 76, 36


def render() -> str:
    e, alpha, beta, total = sr.ctc_lattice_data()
    T, S = len(alpha), len(e)
    d = Diagram(X0 + DX * (T - 1) + 90, Y0 + DY * (S - 1) + 62)
    live = {(t, s) for t in range(T) for s in range(S) if alpha[t][s] * beta[t][s] > 0}
    for t in range(T):
        d.text(X0 + DX * t, 30, [f"frame {t}"], size=13, color="#4a5568")
    for s in range(S):
        d.text(24, Y0 + DY * s, [f"state {s}", "blank" if e[s] == 0 else "letter " + sr.SYM[e[s]]], size=13, color="#4a5568", anchor="start")
    nodes = {}
    for (t, s) in sorted(live):
        kind = "result" if t == T - 1 else "model"
        nodes[(t, s)] = d.oval(X0 + DX * t, Y0 + DY * s, NODE_W, NODE_H, [f"{alpha[t][s]:.4f}"], kind)
    for (t, s) in sorted(live):
        if t == 0:
            continue
        for back in (0, 1, 2):
            src = (t - 1, s - back)
            if src not in live or (back == 2 and not sr._skip_ok(e, s)):
                continue
            a, b = nodes[src]["ell"], nodes[(t, s)]["ell"]
            d.arrow(_edge(a, b), _edge(b, a), dash="4 3" if back == 2 else None)
    return d.svg()


def _edge(ell, toward):
    """The point where the line from the centre of `ell` to the centre of `toward` leaves the ellipse `ell` (cx, cy, rx, ry)."""
    cx, cy, rx, ry = ell
    dx, dy = toward[0] - cx, toward[1] - cy
    k = 1.0 / ((dx / rx) ** 2 + (dy / ry) ** 2) ** 0.5
    return (cx + k * dx, cy + k * dy)
