#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Independent reference for horizon's survival arithmetic, from PyTorch.

Writes crates/horizon/testdata/survival_reference.json: a small population of
subjects chosen to hit every edge of the exposure rules (an event on a knot,
left truncation, an event after an absorbing one, an event on the window's
end, follow-up past the last knot, no window at all), their per-piece
log-hazards, and what PyTorch computes:

- exposure and event piece per (subject, code), re-derived from the raw
  subject by the rules below, not from horizon's encoder;
- the weighted piecewise-exponential negative log-likelihood and its gradient
  with respect to the log-hazards, by autograd;
- survival and cumulative incidence at several times, by the matrix
  exponential of the competing-risks generator of each piece (a different
  method from the closed form horizon uses).

crates/horizon/tests/reference.rs holds the encoder, the loss kernels (on
the CPU backend) and `survival::Curves` to every number. No checkpoint is
involved, so the inputs are in the file.

Run: <venv>/bin/python tools/goldens/survival_likelihood_reference.py
(needs numpy and torch).
"""
import json
import math
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
KNOTS = [0.0, 1.0, 2.5, 4.0, 7.0, 12.0]
CODES = ["death:a", "death:b", "onset"]
ABSORBING = [True, True, False]
ENTRY = 50.0
P, K = len(KNOTS) - 1, len(CODES)

def subj(i, window, events, weight=1.0, onset_window=None):
    at_risk = [] if window is None else [{"code": "*", "from": window[0], "to": window[1]}]
    if onset_window is not None:
        at_risk.append({"code": "onset", "from": onset_window[0], "to": onset_window[1]})
    return {
        "subject_id": f"s{i}", "source": "ref", "weight": weight, "entry": ENTRY, "calendar_at_entry": 2000.0,
        "observations": [{"t": ENTRY, "var": "x", "value": float(i)}],
        "events": [{"t": t, "code": c} for t, c in events], "at_risk": at_risk,
    }

SUBJECTS = [
    subj(0, (50, 62), [(51.5, "death:a")]),
    subj(1, (50, 56), []),
    subj(2, (50, 62), [(54.0, "death:b")]),                      # event exactly on a knot: piece (2.5, 4]
    subj(3, (50, 62), [(52.0, "onset"), (58.0, "death:a")]),
    subj(4, (50, 62), [(50.5, "onset")], onset_window=(51, 62)), # event before its window opens: not counted
    subj(5, (50, 60), [(45.0, "onset"), (61.0, "death:b")]),     # history ignored; event past the window: censored
    subj(6, (50, 62), [(50.2, "death:a")], weight=2.5),
    subj(7, (50, 70), []),                                       # follow-up past the last knot
    subj(8, (50, 65), [(60.0, "onset")]),
    subj(9, (50, 62), [(62.0, "death:a")]),                      # event on the window's end
    subj(10, None, []),                                          # no window: no exposure
    subj(11, (50, 62), [(52.0, "death:a"), (53.0, "death:b")]),  # second absorbing event is not at risk
    subj(12, (50, 62), [(53.5, "onset"), (54.5, "death:b")], weight=0.5),
]

def first(events, code):
    t = [e["t"] for e in events if e["code"] == code and e["t"] > ENTRY]
    return min(t) if t else math.inf

def window(s, code):
    own = [w for w in s["at_risk"] if w["code"] == code]
    star = [w for w in s["at_risk"] if w["code"] == "*"]
    w = (own or star or [None])[0]
    return w

def outcome(s, k):
    """Exposure per piece and the event piece of code k, from the rules."""
    w = window(s, CODES[k])
    if w is None:
        return [0.0] * P, None
    absorbed = min(first(s["events"], c) for c, a in zip(CODES, ABSORBING) if a)
    ev = first(s["events"], CODES[k])
    to = w["to"] + 1e-9 * max(abs(w["to"]), 1.0)  # an event a rounding past the window's end is inside it
    end = min(to, absorbed, ev)
    a, b = w["from"] - ENTRY, end - ENTRY
    expo = [max(min(b, hi) - max(a, lo), 0.0) for lo, hi in zip(KNOTS[:-1], KNOTS[1:])]
    rel = ev - ENTRY
    piece = None
    if ev <= to and ev <= absorbed and math.isfinite(ev) and rel > a:
        for p, (lo, hi) in enumerate(zip(KNOTS[:-1], KNOTS[1:])):
            if lo < rel <= hi:
                piece = p
                break
    return expo, piece

expo = torch.zeros(len(SUBJECTS), P, K)
event = torch.zeros(len(SUBJECTS), P, K)
event_piece = []
for i, s in enumerate(SUBJECTS):
    row = []
    for k in range(K):
        e, piece = outcome(s, k)
        expo[i, :, k] = torch.tensor(e)
        if piece is not None:
            event[i, piece, k] = 1.0
        row.append(piece)
    event_piece.append(row)

gen = torch.Generator().manual_seed(20261006)
loglam = (math.log(0.05) + 0.7 * torch.randn(len(SUBJECTS), P, K, generator=gen)).requires_grad_(True)
w = torch.tensor([s["weight"] for s in SUBJECTS])
loss = ((w / w.sum())[:, None, None] * (loglam.exp() * expo - event * loglam)).sum()
loss.backward()

def cif_and_survival(lam, t):
    """P(code k happened by t) per code, and P(no absorbing event by t), by
    the matrix exponential of each piece's generator."""
    t = min(t, KNOTS[-1])  # held past the last knot
    cifs = []
    for k in range(K):
        p = torch.tensor([1.0, 0.0, 0.0])
        for q in range(P):
            dt = max(min(t, KNOTS[q + 1]) - KNOTS[q], 0.0)
            if dt == 0.0:
                break
            other = sum(lam[q, c] for c in range(K) if ABSORBING[c] and c != k)
            Q = torch.zeros(3, 3)
            Q[0, 1], Q[0, 2] = lam[q, k], other
            Q[0, 0], = -(lam[q, k] + other),
            p = p @ torch.linalg.matrix_exp(Q * dt)
        cifs.append(float(p[1]))
    s = torch.tensor([1.0, 0.0])
    for q in range(P):
        dt = max(min(t, KNOTS[q + 1]) - KNOTS[q], 0.0)
        if dt == 0.0:
            break
        rate = sum(lam[q, c] for c in range(K) if ABSORBING[c])
        Q = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        Q[0, 0], Q[0, 1] = -rate, rate
        s = s @ torch.linalg.matrix_exp(Q * dt)
    return cifs, float(s[0])

TIMES = [0.5, 1.0, 2.5, 3.3, 6.0, 12.0, 15.0]
with torch.no_grad():
    lam = loglam.exp()
    curves = []
    for i in range(len(SUBJECTS)):
        rows = [cif_and_survival(lam[i], t) for t in TIMES]
        curves.append({"cif": [r[0] for r in rows], "survival": [r[1] for r in rows]})

out = {
    "versions": {"torch": torch.__version__, "numpy": np.__version__},
    "knots": KNOTS, "codes": CODES, "absorbing": ABSORBING,
    "subjects": SUBJECTS,
    "log_hazards": loglam.detach().tolist(),
    "exposure": expo.tolist(), "event_piece": event_piece,
    "nll": float(loss.detach()), "nll_grad": loglam.grad.tolist(),
    "times": TIMES, "curves": curves,
}
path = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "horizon", "testdata", "survival_reference.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.normpath(path), "nll", float(loss))
