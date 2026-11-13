# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Recurrent Networks": a state that is rewritten at every step, and why it forgets.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own
description and independent of any engine code. Each public function without arguments returns a markdown string
that the chapter embeds between `<!-- worked: recurrent_networks.<name> -->` markers; the `*_data` functions supply
the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* a toy agreement task whose subject sits any number of phrases before the verb, and the best accuracy any model
  that reads only the last four tokens can reach, by exact enumeration;
* a recurrent cell of two units unrolled by hand over three tokens, backpropagation through time on it, and a
  check of the result against central differences; the same check for the LSTM and the GRU cell;
* the gradient that reaches a state k steps back (the spectral norm of a product of Jacobians) for a tanh cell of
  small, medium and large recurrent weights, for an LSTM at three forget-gate biases and for a GRU, and truncated
  backpropagation through time;
* an LSTM step by hand and the product of forget gates along the cell state;
* a vanilla recurrent cell, an LSTM and a GRU trained on the task by Adam in minibatches, three seeds each, and
  their accuracy against the distance of the subject;
* parameters and FLOPs of one step of each cell, and the work and the depth of a recurrent layer over T steps.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed.
The sums of dot products are plain left-to-right additions, the order a C loop would use.

Swedish Embedded AB implements solutions for sequence models that must remember across long inputs. If your team
needs expertise in recurrent networks, gated cells and the training of models on long sequences, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import hashlib
import json
import math
from fractions import Fraction
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import bookrng
import stable

# ------------------------------------------------------------------------------------------------ formatting


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _int(n):
    return f"{n:,}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _vec(v, d=4):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


# ------------------------------------------------------------------------------------------------ the task

TOKENS = ["the", "cat", "dog", "fox", "cats", "dogs", "foxes", "near", "sees", "chases", "see", "chase"]
V = len(TOKENS)            # 12 ids, the first twelve of the language of the previous chapter
THE, NEAR, FIRST_VERB = 0, 7, 8
SUBJECT_ID = 1             # position of the subject in a sequence: `the` is position 0, the subject position 1
TRAIN_KS = [0, 1, 2, 4, 8, 12, 16]
TEST_KS = [0, 1, 2, 4, 8, 12, 16, 24, 32]
N_TRAIN_PER_K, N_TEST_PER_K = 20, 40
WINDOW = 4
NAME_TRAIN, NAME_TEST = "recurrent_networks.train", "recurrent_networks.test"


def distance(k):
    """How many positions before the verb the subject sits, for k phrases between them."""
    return 1 + 3 * k


def make_set(name, ks, n_per_k):
    """(k, input tokens, class of the verb 0..3) for n_per_k sequences of each k in ks, in that order.

    Draws per sequence: integer(6) the subject, then k times integer(6) for the nouns of the phrases, then
    integer(2) for which of the two verbs of the subject's number."""
    rng = bookrng.generator(name)
    out = []
    for k in ks:
        for _ in range(n_per_k):
            s = rng.integer(6)
            nouns = [rng.integer(6) for _ in range(k)]
            v = rng.integer(2)
            toks = [THE, 1 + s]
            for x in nouns:
                toks += [NEAR, THE, 1 + x]
            out.append((k, toks, (2 if s >= 3 else 0) + v))
    return out


def _words(toks):
    return " ".join(TOKENS[t] for t in toks)


def _window_ceiling(k):
    """Best accuracy of any function of the last four tokens before the verb, by enumeration of all 6^(k+1)
    equally likely (subject, nouns) combinations: for each window content take the number (singular or plural)
    that is more frequent among the sequences showing that window."""
    seen = {}
    total = 0

    def rec(prefix, left, subj):
        nonlocal total
        if left == 0:
            win = tuple(prefix[-WINDOW:])
            seen.setdefault(win, [0, 0])[0 if subj < 3 else 1] += 1
            total += 1
            return
        for x in range(6):
            rec(prefix + [NEAR, THE, 1 + x], left - 1, subj)

    for s in range(6):
        rec([THE, 1 + s], k, s)
    return Fraction(sum(max(a) for a in seen.values()), total)


def task():
    rows = []
    for k, toks, y in make_set("recurrent_networks.example", [0, 1, 2, 3], 1):
        rows.append((k, distance(k), len(toks), f"`{_words(toks)}`", f"`{TOKENS[FIRST_VERB + y]}`"))
    t1 = _table(["phrases k", "distance d = 1 + 3k", "tokens read", "input", "verb drawn"], rows, "rrrll")
    ceil_rows = []
    for k in range(6):
        f = _window_ceiling(k)
        ceil_rows.append((k, distance(k), _int(6 ** (k + 1)), "yes" if distance(k) <= WINDOW else "no", f"{f.numerator}/{f.denominator}",
                          _fx(float(f), 4)))
    t2 = _table(["k", "d", "combinations 6^(k+1)", "subject inside the window of 4", "best accuracy of a window model", "as a decimal"],
                ceil_rows, "rrrlrr")
    return (
        f"The ids are those of the previous chapter's language, `{'`, `'.join(f'{i} {w}' for i, w in enumerate(TOKENS))}`. "
        "A sequence is `the`, a subject (ids 1 to 6), then k times the three tokens `near the` and a noun, and the model must "
        "name the verb that comes next: one of `sees`, `chases` (singular subject, ids 1 to 3) or `see`, `chase` "
        "(plural subject, ids 4 to 6), the two verbs of the right number equally likely. The nouns of the phrases are "
        "drawn from all six nouns and have nothing to do with the subject.\n\n"
        + _seed_line(NAME_TRAIN) + " Test: "
        + f"seed {bookrng.seed_of(NAME_TEST):#018x} (`{NAME_TEST}`). Draw order per sequence: `integer(6)` the subject, "
        "then k times `integer(6)` the nouns of the phrases, then `integer(2)` the verb among the two of the subject's number. "
        f"The training set has {N_TRAIN_PER_K} sequences for each k in {TRAIN_KS} in that order ({N_TRAIN_PER_K * len(TRAIN_KS)} in all), "
        f"the test set {N_TEST_PER_K} for each k in {TEST_KS} ({N_TEST_PER_K * len(TEST_KS)} in all). The four sequences below are "
        "drawn from the generator `recurrent_networks.example` (seed "
        f"{bookrng.seed_of('recurrent_networks.example'):#018x}), one for each k from 0 to 3.\n\n"
        + t1 + "\n\n"
        "The best accuracy a model can reach that sees only the last four tokens before the verb, found by listing all "
        "6^(k+1) equally likely combinations of subject and phrase nouns, grouping them by the window's contents and "
        "taking the more frequent number in each group:\n\n" + t2
    )


# ------------------------------------------------------------------------------------------------ the cells


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _matvec(m, v):
    return [_dot(r, v) for r in m]


def _transpose(m):
    return [list(c) for c in zip(*m)]


def _norm(v):
    return math.sqrt(_dot(v, v))


def power_iteration(m, iters, v0):
    """The book's `power_iteration(W, iters, v0)`: the largest singular value of m, by `iters` rounds of
    v <- normalize(m^T (m v)) from v0, answering norm(m v)."""
    mt = _transpose(m)
    v = list(v0)
    for _ in range(iters):
        w = _matvec(mt, _matvec(m, v))
        n = _norm(w)
        v = [x / n for x in w]
    return _norm(_matvec(m, v))


GATES = {"rnn": 1, "lstm": 4, "gru": 3}
CELL_NAMES = {"rnn": "vanilla", "lstm": "LSTM", "gru": "GRU"}


def init_params(kind, H, rng, wgain=1.0, forget_bias=1.0, u_scale=0.5):
    """Parameters of a cell with H units over V one-hot inputs, drawn in the order U, W, readout.

    U[v] is column v of the input matrix (the vector added when token v is read), W has G*H rows of H entries
    (G = 1, 4 or 3 blocks of H rows), b has G*H entries. LSTM blocks are ordered i, f, g, o; GRU blocks z, r, n.
    U entries are u_scale * normal(), W entries wgain * normal() / sqrt(H), the readout normal() / sqrt(H). The bias
    of the LSTM forget gate starts at forget_bias, the bias of the GRU update gate at -forget_bias, all others at 0."""
    G = GATES[kind]
    U = [[u_scale * rng.normal() for _ in range(G * H)] for _ in range(V)]
    W = [[wgain * rng.normal() / math.sqrt(H) for _ in range(H)] for _ in range(G * H)]
    Wo = [[rng.normal() / math.sqrt(H) for _ in range(H)] for _ in range(4)]
    b = [0.0] * (G * H)
    if kind == "lstm":
        for j in range(H):
            b[H + j] = forget_bias
    if kind == "gru":
        for j in range(H):
            b[j] = -forget_bias
    return {"U": U, "W": W, "b": b, "Wo": Wo, "bo": [0.0] * 4}


def _zeros_like(p):
    return {"U": [[0.0] * len(r) for r in p["U"]], "W": [[0.0] * len(r) for r in p["W"]], "b": [0.0] * len(p["b"]),
            "Wo": [[0.0] * len(r) for r in p["Wo"]], "bo": [0.0] * 4}


def _flat(p):
    return p["U"] + p["W"] + p["Wo"] + [p["b"], p["bo"]]


def step(kind, p, H, h, c, tok):
    """One step of a cell: (h, c) -> (h', c', cache). c is only used by the LSTM."""
    U, b, W = p["U"][tok], p["b"], p["W"]
    if kind == "rnn":
        h2 = [math.tanh(_dot(W[i], h) + U[i] + b[i]) for i in range(H)]
        return h2, c, (h, h2, tok)
    if kind == "lstm":
        a = [_dot(W[i], h) + U[i] + b[i] for i in range(4 * H)]
        ig = [_sigmoid(x) for x in a[0:H]]
        fg = [_sigmoid(x) for x in a[H:2 * H]]
        gg = [math.tanh(x) for x in a[2 * H:3 * H]]
        og = [_sigmoid(x) for x in a[3 * H:]]
        c2 = [fg[j] * c[j] + ig[j] * gg[j] for j in range(H)]
        tc = [math.tanh(x) for x in c2]
        return [og[j] * tc[j] for j in range(H)], c2, (h, c, ig, fg, gg, og, tc, tok)
    z = [_sigmoid(_dot(W[i], h) + U[i] + b[i]) for i in range(H)]
    r = [_sigmoid(_dot(W[H + i], h) + U[H + i] + b[H + i]) for i in range(H)]
    hr = [r[j] * h[j] for j in range(H)]
    n = [math.tanh(_dot(W[2 * H + i], hr) + U[2 * H + i] + b[2 * H + i]) for i in range(H)]
    return [(1.0 - z[j]) * h[j] + z[j] * n[j] for j in range(H)], c, (h, z, r, hr, n, tok)


def forward(kind, p, H, toks):
    h, c = [0.0] * H, [0.0] * H
    caches = []
    for t in toks:
        h, c, cache = step(kind, p, H, h, c, t)
        caches.append(cache)
    return h, c, caches


def _transposes(kind, p, H):
    W = p["W"]
    if kind == "gru":
        return _transpose(W[:2 * H]), _transpose(W[2 * H:])
    return (_transpose(W),)


def _accumulate(g, da, hin, tok, rows):
    """dW[rows] += da outer hin, dU[tok] += da, db += da (the same parameters at every step)."""
    for i in range(rows):
        gi, d = g["W"][i], da[i]
        for j in range(len(hin)):
            gi[j] += d * hin[j]


def back(kind, H, WT, cache, dh, dc, g):
    """One step of backpropagation through time. Given the gradient of the loss with respect to the state this
    step produced, (dh, dc), return the gradient with respect to the state it started from. When g is not None the
    parameter gradients of this step are added into it."""
    if kind == "rnn":
        hp, h2, tok = cache
        da = [(1.0 - h2[j] * h2[j]) * dh[j] for j in range(H)]
        if g is not None:
            _accumulate(g, da, hp, tok, H)
            for i in range(H):
                g["U"][tok][i] += da[i]
                g["b"][i] += da[i]
        return [_dot(WT[0][j], da) for j in range(H)], dc
    if kind == "lstm":
        hp, cp, ig, fg, gg, og, tc, tok = cache
        dcc = [dc[j] + dh[j] * og[j] * (1.0 - tc[j] * tc[j]) for j in range(H)]
        da = ([dcc[j] * gg[j] * ig[j] * (1.0 - ig[j]) for j in range(H)]
              + [dcc[j] * cp[j] * fg[j] * (1.0 - fg[j]) for j in range(H)]
              + [dcc[j] * ig[j] * (1.0 - gg[j] * gg[j]) for j in range(H)]
              + [dh[j] * tc[j] * og[j] * (1.0 - og[j]) for j in range(H)])
        if g is not None:
            _accumulate(g, da, hp, tok, 4 * H)
            for i in range(4 * H):
                g["U"][tok][i] += da[i]
                g["b"][i] += da[i]
        return [_dot(WT[0][j], da) for j in range(H)], [dcc[j] * fg[j] for j in range(H)]
    hp, z, r, hr, n, tok = cache
    da_n = [dh[j] * z[j] * (1.0 - n[j] * n[j]) for j in range(H)]
    da_z = [dh[j] * (n[j] - hp[j]) * z[j] * (1.0 - z[j]) for j in range(H)]
    dhr = [_dot(WT[1][j], da_n) for j in range(H)]
    da_r = [dhr[j] * hp[j] * r[j] * (1.0 - r[j]) for j in range(H)]
    da_zr = da_z + da_r
    dprev = [dh[j] * (1.0 - z[j]) + dhr[j] * r[j] + _dot(WT[0][j], da_zr) for j in range(H)]
    if g is not None:
        for i in range(2 * H):
            gi, d = g["W"][i], da_zr[i]
            for j in range(H):
                gi[j] += d * hp[j]
        for i in range(H):
            gi, d = g["W"][2 * H + i], da_n[i]
            for j in range(H):
                gi[j] += d * hr[j]
        da = da_zr + da_n
        for i in range(3 * H):
            g["U"][tok][i] += da[i]
            g["b"][i] += da[i]
    return dprev, dc


def bptt(kind, p, H, caches, dh, dc, g=None, reach=None):
    """Backpropagation through time from the final state: the gradient with respect to the state after t tokens,
    for t = T down to T - reach (all the way to 0 when reach is None), as {t: (dh_t, dc_t)}. Parameter gradients
    are added into g from every step that is visited."""
    WT = _transposes(kind, p, H)
    T = len(caches)
    out = {T: (dh, dc)}
    stop = 0 if reach is None else max(0, T - reach)
    for t in range(T - 1, stop - 1, -1):
        dh, dc = back(kind, H, WT, caches[t], dh, dc, g)
        out[t] = (dh, dc)
    return out


def _softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def _readout(p, h):
    return [_dot(p["Wo"][k], h) + p["bo"][k] for k in range(4)]


def loss_and_grad(kind, p, H, batch, g):
    """Mean cross-entropy of the verb over a batch of (k, tokens, class); adds the gradient of that mean into g."""
    n = len(batch)
    total = 0.0
    for _, toks, y in batch:
        h, _, caches = forward(kind, p, H, toks)
        pr = _softmax(_readout(p, h))
        total += -math.log(pr[y])
        d = [(pr[i] - (1.0 if i == y else 0.0)) / n for i in range(4)]
        for i in range(4):
            g["bo"][i] += d[i]
            for j in range(H):
                g["Wo"][i][j] += d[i] * h[j]
        dh = [sum(d[i] * p["Wo"][i][j] for i in range(4)) for j in range(H)]
        bptt(kind, p, H, caches, dh, [0.0] * H, g)
    return total / n


# ------------------------------------------------------------------------------------------------ unrolled by hand

TINY_TOKENS = [THE, 4, NEAR]            # `the cats near`
TINY_TARGET = [0.5, -0.5]
TINY_NAME = "recurrent_networks.tiny"


@lru_cache(maxsize=None)
def _tiny():
    rng = bookrng.generator(TINY_NAME)
    U = [[0.8 * rng.normal() for _ in range(2)] for _ in range(V)]
    W = [[0.8 * rng.normal() for _ in range(2)] for _ in range(2)]
    return {"U": U, "W": W, "b": [0.0, 0.0], "Wo": [[0.0, 0.0]] * 4, "bo": [0.0] * 4}


def _tiny_loss(p):
    h, _, _ = forward("rnn", p, 2, TINY_TOKENS)
    return 0.5 * sum((h[i] - TINY_TARGET[i]) ** 2 for i in range(2))


def unroll():
    p = _tiny()
    h = [0.0, 0.0]
    rows = []
    for t, tok in enumerate(TINY_TOKENS, 1):
        wh = _matvec(p["W"], h)
        a = [wh[i] + p["U"][tok][i] + p["b"][i] for i in range(2)]
        h = [math.tanh(x) for x in a]
        rows.append((t, f"`{TOKENS[tok]}`", _vec(wh), _vec(p["U"][tok]), _vec(a), _vec(h)))
    tu = _table(["token id", "token", "U column"], [(i, f"`{TOKENS[i]}`", _vec(p["U"][i])) for i in range(V)], "rll")
    tw = _table(["row", "W"], [(i, _vec(p["W"][i])) for i in range(2)], "rl")
    ts = _table(["t", "token", "W h(t-1)", "U x(t)", "a(t) = W h(t-1) + U x(t) + b", "h(t) = tanh(a(t))"], rows, "rlllll")
    return (
        _seed_line(TINY_NAME) + " Draws, in this order: the 12 columns of U one after the other, 2 values each (each "
        "`0.8 * normal()`), then W row by row, 4 values (each `0.8 * normal()`). The bias b is (0, 0) and the "
        "start state h(0) is (0, 0). The input at step t is the one-hot vector of the token, so `U x(t)` is the column "
        "of the token.\n\n" + tu + "\n\n" + tw + "\n\n"
        f"The sequence is `the cats near` (ids 0, 4, 7). Tolerance: values are rounded to 4 decimals.\n\n" + ts
        + f"\n\nEvery row used the same W, the same U and the same b."
    )


def bptt_by_hand():
    p = _tiny()
    h, _, caches = forward("rnn", p, 2, TINY_TOKENS)
    hs = [[0.0, 0.0]] + [c[1] for c in caches]
    loss = _tiny_loss(p)
    dh = [h[i] - TINY_TARGET[i] for i in range(2)]
    g = _zeros_like(p)
    states = bptt("rnn", p, 2, caches, dh, [0.0, 0.0], g)
    rows = []
    for t in (3, 2, 1):
        hv = hs[t]
        slope = [1.0 - x * x for x in hv]
        da = [slope[i] * states[t][0][i] for i in range(2)]
        rows.append((t, _vec(states[t][0]), _vec(slope), _vec(da), _vec(states[t - 1][0])))
    t1 = _table(["t", "dh(t)", "1 - h(t)^2", "da(t) = (1 - h(t)^2) * dh(t)", "dh(t-1) = W^T da(t)"], rows, "rllll")
    # per step contributions to dW and the totals
    contrib = []
    for t in (3, 2, 1):
        da = [(1.0 - hs[t][i] ** 2) * states[t][0][i] for i in range(2)]
        contrib.append((t, f"da({t}) h({t - 1})^T", "; ".join(_vec([da[i] * hs[t - 1][j] for j in range(2)]) for i in range(2))))
    contrib.append(("sum", "dW", "; ".join(_vec(g["W"][i]) for i in range(2))))
    t2 = _table(["t", "term", "rows of the 2 x 2 matrix"], contrib, "rll")
    # numerical check of every parameter that matters
    worst = 0.0
    n_checked = 0
    for key in ("W", "U", "b"):
        arr = p[key] if key != "b" else [p["b"]]
        for i, row in enumerate(arr):
            for j in range(len(row)):
                old = row[j]
                row[j] = old + 1e-6
                lp = _tiny_loss(p)
                row[j] = old - 1e-6
                lm = _tiny_loss(p)
                row[j] = old
                num = (lp - lm) / 2e-6
                ana = g[key][j] if key == "b" else g[key][i][j]
                worst = max(worst, abs(num - ana))
                n_checked += 1
    gu = "; ".join(f"`{TOKENS[tok]}` {_vec(g['U'][tok])}" for tok in TINY_TOKENS)
    return (
        f"Loss L = 1/2 * ||h(3) - y||^2 with y = {_vec(TINY_TARGET, 1)}; with the model above L = {_fx(loss, 6)}. "
        f"The gradient into the last state is dh(3) = h(3) - y = {_vec(dh)}. Each row below is one step of the backward pass, from the last step "
        "to the first.\n\n" + t1 + "\n\n"
        "The gradient of the shared matrix W is the sum of one outer product per step:\n\n" + t2 + "\n\n"
        f"The bias gradient is the sum of the three da vectors, db = {_vec(g['b'])}. The columns of U that were used get "
        f"the da of their step: {gu}; the other nine columns get 0.\n\n"
        f"Check: all {n_checked} entries of W, of the used and unused columns of U and of b, against central differences "
        f"of L with step 1e-6: the largest absolute difference is {'below' if worst < 1e-8 else 'NOT below'} 1e-8."
    )


def gradient_check_cells():
    rows = []
    for kind in ("rnn", "lstm", "gru"):
        H = 3
        p = init_params(kind, H, bookrng.generator(f"recurrent_networks.check.{kind}"), wgain=1.5, forget_bias=0.5, u_scale=0.7)
        batch = make_set("recurrent_networks.check.data", [2, 1], 1)
        # the readout is part of the model: give it a nonzero bias so that bo is exercised too
        p["bo"] = [0.1, -0.2, 0.05, 0.0]
        g = _zeros_like(p)
        loss_and_grad(kind, p, H, batch, g)

        def loss():
            gg = _zeros_like(p)
            return loss_and_grad(kind, p, H, batch, gg)

        worst, count = 0.0, 0
        for fp, fg in zip(_flat(p), _flat(g)):
            for j in range(len(fp)):
                old = fp[j]
                fp[j] = old + 1e-6
                lp = loss()
                fp[j] = old - 1e-6
                lm = loss()
                fp[j] = old
                worst = max(worst, abs((lp - lm) / 2e-6 - fg[j]))
                count += 1
        name = f"recurrent_networks.check.{kind}"
        rows.append((CELL_NAMES[kind], f"`{name}`", f"{bookrng.seed_of(name):#018x}", count, "yes" if worst < 1e-7 else "NO"))
    return (
        "Each cell has H = 3 units. Its parameters are drawn by the initialization recipe stated before the Jacobian table, from the generator "
        "`recurrent_networks.check.<cell>` with `<cell>` equal to `rnn` (the vanilla cell), `lstm` or `gru` (seeds in the table), with W gain 1.5, forget-gate bias 0.5 (the update-gate bias of the GRU -0.5) and U entries "
        "`0.7 * normal()`, and the readout bias set to (0.1, -0.2, 0.05, 0). The loss is the mean cross-entropy of two sequences, one of "
        "k = 2 phrases (8 tokens) and one of k = 1 (5 tokens), in that order, from the generator `recurrent_networks.check.data` "
        f"(seed {bookrng.seed_of('recurrent_networks.check.data'):#018x}). Every entry of every parameter is compared "
        "with a central difference of step 1e-6.\n\n"
        + _table(["cell", "generator", "seed", "parameters compared", "largest absolute difference below 1e-7"], rows, "llrrr")
    )


# ------------------------------------------------------------------------------------------------ the gradient k steps back

SENS_H, SENS_T = 16, 40
SENS_KS = [1, 2, 4, 8, 16, 24, 32]
SENS_MODELS = [
    # label, kind, wgain, forget bias
    ("vanilla, small W", "rnn", 0.25, 1.0),
    ("vanilla, medium W", "rnn", 1.0, 1.0),
    ("vanilla, large W", "rnn", 3.0, 1.0),
    ("LSTM, forget bias 0", "lstm", 1.0, 0.0),
    ("LSTM, forget bias 1", "lstm", 1.0, 1.0),
    ("LSTM, forget bias 3", "lstm", 1.0, 3.0),
    ("GRU, update bias -1", "gru", 1.0, 1.0),
]
SENS_TOKENS_NAME = "recurrent_networks.sens_tokens"
SPECTRAL_NAME = "recurrent_networks.spectral"
SPECTRAL_ITERS = 60


def _spectral(m):
    rng = bookrng.generator(SPECTRAL_NAME)
    return power_iteration(m, SPECTRAL_ITERS, [rng.normal() for _ in range(len(m[0]))])


def _sens_tokens():
    rng = bookrng.generator(SENS_TOKENS_NAME)
    return [rng.integer(V) for _ in range(SENS_T)]


@lru_cache(maxsize=None)
def sensitivity_data():
    """For each model: ||W||_2 of the recurrent block and, for each k, the spectral norm of the Jacobian of h(T)
    with respect to the state k steps earlier (h for the vanilla cell and the GRU, (h, c) for the LSTM)."""
    toks = _sens_tokens()
    out = {}
    for label, kind, gain, fb in SENS_MODELS:
        p = init_params(kind, SENS_H, bookrng.generator("recurrent_networks.sens." + label), wgain=gain, forget_bias=fb)
        _, _, caches = forward(kind, p, SENS_H, toks)
        rows = {k: [] for k in SENS_KS}
        for i in range(SENS_H):
            seed = [1.0 if j == i else 0.0 for j in range(SENS_H)]
            states = bptt(kind, p, SENS_H, caches, seed, [0.0] * SENS_H, None)
            for k in SENS_KS:
                dh, dc = states[SENS_T - k]
                rows[k].append(dh + (dc if kind == "lstm" else []))
        wblock = p["W"][:SENS_H] if kind != "rnn" else p["W"]
        out[label] = {"kind": kind, "wnorm": _spectral(wblock), "norms": {k: _spectral(rows[k]) for k in SENS_KS}}
    return out


def jacobian_product():
    data = sensitivity_data()
    head = ["model", "||W||_2"] + [f"k = {k}" for k in SENS_KS]
    rows = []
    for label, _, _, _ in SENS_MODELS:
        d = data[label]
        rows.append([label, _fx(d["wnorm"], 3) if d["kind"] == "rnn" else "-"] + [_sci(d["norms"][k], 2) for k in SENS_KS])
    small = data["vanilla, small W"]
    bound = [_sci(small["wnorm"] ** k, 2) for k in SENS_KS]
    large = data["vanilla, large W"]
    clip_k = max(SENS_KS, key=lambda k: large["norms"][k])
    return (
        f"All models have H = {SENS_H} units and read the same {SENS_T} tokens, ids drawn by `integer({V})` from the generator "
        f"`{SENS_TOKENS_NAME}` (seed {bookrng.seed_of(SENS_TOKENS_NAME):#018x}). Each model's parameters come from the generator "
        "`recurrent_networks.sens.<label>`, where the label is the full text of the first column (the generator of the first model is "
        "`recurrent_networks.sens.vanilla, small W`, with the comma and the space; the seeds are listed after the table) in the order "
        "U, W, readout (the readout is not used here), with U entries `0.5 * normal()`, W entries `gain * normal() / sqrt(16)` where the gain is "
        "0.25, 1 and 3 for the three vanilla models and 1 for the gated ones, and the biases of the initialization recipe "
        "of the text: LSTM forget bias 0, 1 or 3, GRU update-gate bias -1, all others 0. "
        "The entry for k is the spectral norm of the Jacobian of the last state h(40) with respect to the state k steps earlier "
        f"(for the LSTM the state is (h, c), a {SENS_H} by {2 * SENS_H} Jacobian), found by running the backward pass of "
        f"BPTT from each of the {SENS_H} unit vectors in h(40) and stacking the {SENS_H} gradients that arrive k steps back. "
        f"The spectral norm is `power_iteration` with {SPECTRAL_ITERS} iterations (the start vector, one `normal()` per column of the matrix, is not normalised first) from draws of the generator "
        f"`{SPECTRAL_NAME}` (seed {bookrng.seed_of(SPECTRAL_NAME):#018x}), restarted for each entry; the second column is the same estimate for the "
        "matrix W of a vanilla cell (a gated cell has no single recurrent matrix and shows a dash). Values are rounded to 3 significant digits "
        "and are stable to about that.\n\n"
        + _table(head, rows, "lr" + "r" * len(SENS_KS)) + "\n\n"
        + _table(["model", "generator of the parameters", "seed"],
                 [(label, f"`recurrent_networks.sens.{label}`", f"{bookrng.seed_of('recurrent_networks.sens.' + label):#018x}")
                  for label, _, _, _ in SENS_MODELS], "llr") + "\n\n"
        f"The bound ||W||_2^k for the small vanilla model is {', '.join(f'{b}' for b in bound)} at k = {', '.join(str(k) for k in SENS_KS)}. "
        f"The large vanilla model peaks at k = {clip_k} with {_sci(large['norms'][clip_k], 2)}; a global-norm clip with "
        f"threshold 5 would multiply a gradient of that norm by {_fx(5.0 / large['norms'][clip_k], 4)}."
    )


def sensitivity_curves():
    """(label, [log10 of the norm for each k in SENS_KS]) for the figure, plus the bound of the small vanilla model."""
    data = sensitivity_data()
    curves = [(label, [math.log10(data[label]["norms"][k]) for k in SENS_KS]) for label, _, _, _ in SENS_MODELS]
    w = data["vanilla, small W"]["wnorm"]
    return curves, [k * math.log10(w) for k in SENS_KS]


# ------------------------------------------------------------------------------------------------ truncated BPTT

TRUNC_KS = [0, 1, 2, 3, 4, 8]
TRUNC_REACH = 4


def truncation():
    H = 12
    p = init_params("lstm", H, bookrng.generator("recurrent_networks.trunc"))
    nouns = bookrng.generator("recurrent_networks.trunc_nouns")
    rows = []
    for k in TRUNC_KS:
        toks = [THE, 4]
        for _ in range(k):
            toks += [NEAR, THE, 1 + nouns.integer(6)]
        h, _, caches = forward("lstm", p, H, toks)
        pr = _softmax(_readout(p, h))
        y = 2
        d = [pr[i] - (1.0 if i == y else 0.0) for i in range(4)]
        dh = [sum(d[i] * p["Wo"][i][j] for i in range(4)) for j in range(H)]
        full = bptt("lstm", p, H, caches, dh, [0.0] * H, None)
        cut = bptt("lstm", p, H, caches, dh, [0.0] * H, None, reach=TRUNC_REACH)
        sub = 2   # the state after reading `the` and the subject

        def mag(states):
            if sub not in states:
                return "0 (never computed)"
            a, b = states[sub]
            return _sci(_norm(a + b), 2)
        T = len(toks)
        rows.append((k, distance(k), T, T - sub, mag(full), mag(cut), f"{T}", f"{min(T, TRUNC_REACH)}"))
    return (
        "A single LSTM with H = 12 units (parameters from the generator `recurrent_networks.trunc`, seed "
        f"{bookrng.seed_of('recurrent_networks.trunc'):#018x}, forget bias 1) reads the task's sequence with the subject `cats` and k phrases, "
        "the phrase nouns being `1 + integer(6)` draws, in order of the rows, from the generator `recurrent_networks.trunc_nouns` "
        f"(seed {bookrng.seed_of('recurrent_networks.trunc_nouns'):#018x}); the loss is the cross-entropy of the verb `see` (class 2). "
        "The entry is the norm of the gradient (h and c together) that arrives at the state right after the subject "
        f"has been read, from full backpropagation through time and from backpropagation truncated to the last {TRUNC_REACH} steps.\n\n"
        + _table(["k", "d", "tokens T", "steps from the subject to the loss", "norm of the gradient at the subject, full",
                  f"norm, truncated to {TRUNC_REACH} steps", "states stored, full", "states stored, truncated"], rows, "rrrrrrrr")
    )


# ------------------------------------------------------------------------------------------------ gates

def lstm_step():
    H = 2
    h_prev = [0.5, -0.5]
    c_prev = [1.0, -2.0]
    a = {"i": [0.0, 2.0], "f": [2.2, -2.2], "g": [1.0, -1.0], "o": [1.0, 0.0]}
    i = [_sigmoid(x) for x in a["i"]]
    f = [_sigmoid(x) for x in a["f"]]
    g = [math.tanh(x) for x in a["g"]]
    o = [_sigmoid(x) for x in a["o"]]
    keep = [f[j] * c_prev[j] for j in range(H)]
    write = [i[j] * g[j] for j in range(H)]
    c = [keep[j] + write[j] for j in range(H)]
    tc = [math.tanh(x) for x in c]
    h = [o[j] * tc[j] for j in range(H)]
    rows = [
        ("pre-activation a", _vec(a["i"], 1), _vec(a["f"], 1), _vec(a["g"], 1), _vec(a["o"], 1)),
        ("gate (sigmoid, sigmoid, tanh, sigmoid)", _vec(i), _vec(f), _vec(g), _vec(o)),
    ]
    t1 = _table(["", "input gate i", "forget gate f", "candidate g", "output gate o"], rows, "lllll")
    rows2 = [
        ("old cell c(t-1)", _vec(c_prev, 1)),
        ("kept, f * c(t-1)", _vec(keep)),
        ("written, i * g", _vec(write)),
        ("new cell c(t) = kept + written", _vec(c)),
        ("tanh(c(t))", _vec(tc)),
        ("new state h(t) = o * tanh(c(t))", _vec(h)),
    ]
    return (
        "One LSTM step with two units. The previous state is h(t-1) = (0.5, -0.5), c(t-1) = (1, -2), and the four "
        "pre-activations, which in a real cell are the four blocks of W h(t-1) + U x(t) + b, are given directly:\n\n"
        + t1 + "\n\n" + _table(["quantity", "unit 0, unit 1"], rows2, "ll")
    )


def forget_products():
    fs = [("0.50", 0.5), ("sigmoid(1) = 0.7311", _sigmoid(1.0)), ("sigmoid(3) = 0.9526", _sigmoid(3.0)), ("0.99", 0.99)]
    ks = [1, 4, 8, 16, 32, 64]
    rows = [(name, *[_sci(f ** k, 2) for k in ks]) for name, f in fs]
    return (
        "The factor by which the gradient of the cell state is multiplied after k steps if the forget gate stays at f "
        "(the product of k equal gates, f^k):\n\n" + _table(["forget gate f"] + [f"k = {k}" for k in ks], rows, "l" + "r" * len(ks))
    )


# ------------------------------------------------------------------------------------------------ training

TRAIN_H = 12
EPOCHS = 80
BATCH = 14
LR, BETA1, BETA2, EPS_ADAM = 0.01, 0.9, 0.999, 1e-8
CLIP = 5.0
SEEDS = [0, 1, 2]
CHECKPOINT_EPOCHS = [1, 10, 20, 40, 80]


def _run_name(kind, seed, part):
    return f"recurrent_networks.{kind}.{seed}.{part}"


def _train_job(job):
    kind, seed, nudge = job
    train = make_set(NAME_TRAIN, TRAIN_KS, N_TRAIN_PER_K)
    test = make_set(NAME_TEST, TEST_KS, N_TEST_PER_K)
    H = TRAIN_H
    p = init_params(kind, H, bookrng.generator(_run_name(kind, seed, "init")))
    p["U"][0][0] += nudge                       # 0.0 for the run itself; a tiny change for the copies of the stable-digits rule
    order = bookrng.generator(_run_name(kind, seed, "order"))
    fp = _flat(p)
    m = [[0.0] * len(r) for r in fp]
    v = [[0.0] * len(r) for r in fp]
    step_n = 0
    curve = {}
    for epoch in range(1, EPOCHS + 1):
        idx = list(range(len(train)))
        order.shuffle(idx)
        losses = []
        for start in range(0, len(train), BATCH):
            step_n += 1
            g = _zeros_like(p)
            losses.append(loss_and_grad(kind, p, H, [train[i] for i in idx[start:start + BATCH]], g))
            fg = _flat(g)
            gn = math.sqrt(sum(x * x for r in fg for x in r))
            scale = min(1.0, CLIP / gn) if gn > 0 else 1.0
            c1, c2 = 1.0 - BETA1 ** step_n, 1.0 - BETA2 ** step_n
            for pr, gr, mr, vr in zip(fp, fg, m, v):
                for j in range(len(pr)):
                    gj = gr[j] * scale
                    mr[j] = BETA1 * mr[j] + (1.0 - BETA1) * gj
                    vr[j] = BETA2 * vr[j] + (1.0 - BETA2) * gj * gj
                    pr[j] -= LR * (mr[j] / c1) / (math.sqrt(vr[j] / c2) + EPS_ADAM)
        if epoch in CHECKPOINT_EPOCHS:
            curve[epoch] = sum(losses) / len(losses)
    correct = {k: 0 for k in TEST_KS}
    for k, toks, y in test:
        h, _, _ = forward(kind, p, H, toks)
        pr = _softmax(_readout(p, h))
        if (pr[0] + pr[1] > 0.5) == (y < 2):
            correct[k] += 1
    return kind, seed, nudge, curve, correct


TRAIN_CACHE = Path(__file__).resolve().parents[3] / "build" / "book" / "recurrent_networks_training.json"


def _source_digest():
    """Digest of this module and of the generator it draws from: the key under which a finished training is kept."""
    h = hashlib.sha256()
    for f in (Path(__file__), Path(bookrng.__file__)):
        h.update(f.read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=None)
def _all_trainings():
    """{(kind, seed, nudge): (loss at the checkpoint epochs, {k: correct sequences of the test set})}: the nine runs and,
    for the stable-digits rule, four copies of each whose first weight is changed by one of `stable.NUDGES`. The 45
    trainings are independent and are spread over four worker processes. Training is deterministic, so a finished result
    is kept in build/book (keyed by the digest of this module and of bookrng.py) and the chapter's blocks and its figure,
    which are computed by different programs, share one training."""
    key = _source_digest()
    if TRAIN_CACHE.is_file():
        try:
            stored = json.loads(TRAIN_CACHE.read_text(encoding="utf-8"))
            if stored.get("key") == key:
                return {(r["kind"], r["seed"], r["nudge"]): ({int(e): v for e, v in r["curve"].items()},
                                                             {int(k): v for k, v in r["correct"].items()})
                        for r in stored["runs"]}
        except (ValueError, KeyError):
            pass
    jobs = [(kind, seed, nudge) for kind in ("rnn", "lstm", "gru") for seed in SEEDS for nudge in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_train_job, jobs, chunksize=1)
    TRAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_CACHE.write_text(json.dumps({"key": key, "runs": [{"kind": k, "seed": s, "nudge": n, "curve": c, "correct": r}
                                                            for k, s, n, c, r in res]}), encoding="utf-8")
    return {(kind, seed, nudge): (curve, correct) for kind, seed, nudge, curve, correct in res}


def training_results():
    """{(kind, seed): (loss at the checkpoint epochs, {k: correct sequences of the test set})} of the runs themselves."""
    return {(kind, seed): v for (kind, seed, nudge), v in _all_trainings().items() if nudge == 0.0}


def _stable_cell(kind, seed, f, full):
    """A number computed from a run, printed with the digits on which the run and its four nudged copies agree."""
    runs = _all_trainings()
    vals = [f(*runs[(kind, seed, n)]) for n in (0.0, *stable.NUDGES)]
    return stable.text(vals[0], vals[1:], full)


def _stable_count(kind, seed, f, total):
    """A count out of `total`: printed when the run and its copies agree on it, otherwise as 'unstable'."""
    runs = _all_trainings()
    vals = {f(*runs[(kind, seed, n)]) for n in (0.0, *stable.NUDGES)}
    return f"{vals.pop()}/{total}" if len(vals) == 1 else "unstable"


def _pooled(correct, ks):
    n = N_TEST_PER_K * len(ks)
    return sum(correct[k] for k in ks), n


def training():
    groups = [[k for k in TEST_KS if distance(k) <= WINDOW], [k for k in TEST_KS if distance(k) > WINDOW and k in TRAIN_KS],
              [k for k in TEST_KS if k not in TRAIN_KS]]
    rows = []
    for kind in ("rnn", "lstm", "gru"):
        for seed in SEEDS:
            rows.append((CELL_NAMES[kind], seed, f"{bookrng.seed_of(_run_name(kind, seed, 'init')):#018x}",
                         *[_stable_cell(kind, seed, lambda curve, correct, e=e: curve[e], lambda x: _fx(x, 3)) for e in CHECKPOINT_EPOCHS],
                         *[_stable_count(kind, seed, lambda curve, correct, ks=ks: sum(correct[k] for k in ks), N_TEST_PER_K * len(ks))
                           for ks in groups]))
    runs = _all_trainings()
    n_test = N_TEST_PER_K * len(TEST_KS)
    dist_rows = []
    for kind in ("rnn", "lstm", "gru"):
        shares = [sum(c.values()) / n_test for (kd, _, _), (_, c) in runs.items() if kd == kind]
        dist_rows.append((CELL_NAMES[kind], len(shares), sum(x == 1.0 for x in shares), sum(x >= 0.95 for x in shares),
                          sum(x < 0.7 for x in shares)))
    head = (["cell", "run", "seed of the initial parameters"] + [f"loss, epoch {e}" for e in CHECKPOINT_EPOCHS]
            + ["test, d <= 4", "test, d > 4 seen in training", "test, d beyond training"])
    return (
        f"Every cell has H = {TRAIN_H} units and reads the one-hot ids of the task; parameters are drawn by the initialization recipe "
        f"of the text (order U, W, readout; U entries `0.5 * normal()`, W gain 1, LSTM forget-gate bias 1, GRU update-gate bias -1) from the generator named "
        "`recurrent_networks.<cell>.<run>.init` with cell `rnn`, `lstm` or `gru` and run 0, 1 or 2; the seed of each is in the table. "
        f"Training is Adam (learning rate {LR}, beta1 {BETA1}, beta2 {BETA2}, epsilon {EPS_ADAM:g}, with the bias correction of "
        f"Volume III) on minibatches of {BATCH} of the {N_TRAIN_PER_K * len(TRAIN_KS)} training sequences "
        f"(10 batches per epoch, {EPOCHS} epochs, {EPOCHS * 10} updates), the batches of an epoch being a new shuffle of "
        "the sequence indices drawn by `shuffle` from the generator `recurrent_networks.<cell>.<run>.order`, and "
        f"the gradient of every update is clipped to a global norm of {CLIP:g}. The training sequences are the {N_TRAIN_PER_K * len(TRAIN_KS)} of "
        f"the generator `{NAME_TRAIN}` in the order in which it makes them (index 0 .. {N_TRAIN_PER_K * len(TRAIN_KS) - 1}). At the start of every epoch "
        "the index array is rebuilt as 0, 1, 2, ... and shuffled in place by the Fisher-Yates `shuffle` of Volume I, Chapter 17, with the one generator `order` of the "
        "run, which is never re-seeded within a run (so the permutation of epoch 2 is the next draws of that stream, not a shuffle of epoch 1's permutation); "
        f"batch b of the epoch holds the entries {BATCH} b to {BATCH} b + {BATCH - 1} of the shuffled array. The 1/{BATCH} of the mean is "
        "applied to the gradient of the loss with respect to the logits, (p - one-hot) / "
        f"{BATCH}, before anything is propagated back, so the parameter gradients are sums of already scaled terms and nothing is divided afterwards. "
        "The sequences of a batch are processed one after the other in batch order and their parameter gradients are added into one accumulator "
        "in that order; for one sequence the readout gradients come first, then backpropagation through time runs from the last token to the first, "
        "adding the gradient of each step into the accumulator. The global norm adds the squares of all entries left to right over the "
        "parameters in the order U (one row per token id), W (row by row), readout weights (row by row), b, readout bias, and takes the square root; "
        "the factor min(1, 5 / norm) multiplies every entry before Adam sees it. The loss is the mean cross-entropy over "
        "the four verbs; the table gives its mean over the batches of the epoch. A test sequence counts as correct when the "
        "probability of the two verbs of the subject's number exceeds one half. The columns on the right pool the test "
        f"sequences by distance: d <= 4 is k in {[k for k in TEST_KS if distance(k) <= WINDOW]} (the window reaches the subject), "
        f"d > 4 seen in training is k in {[k for k in TEST_KS if distance(k) > WINDOW and k in TRAIN_KS]}, and beyond training is "
        f"k in {[k for k in TEST_KS if k not in TRAIN_KS]}. Losses are rounded to 3 decimals. The runs are chaotic after "
        "the first few epochs (a change of 1e-12 in one weight moves the loss at epoch 40 in the second digit and can move a whole run from "
        "learned to not learned), so the stable-digits rule applies. " + stable.RULE + " A count is printed when all five runs give the same count and as 'unstable' otherwise.\n\n"
        + _table(head, rows, "lrl" + "r" * (len(head) - 3)) + "\n\n"
        "What survives the chaos is the distribution over trainings. Each cell was trained 15 times (the 3 runs, each with its 4 nudged copies and itself); "
        f"the table counts them by the share of the {sum(N_TEST_PER_K for _ in TEST_KS)} test sequences of all distances that they name correctly. "
        "A reimplementation should expect similar fractions, not these counts.\n\n"
        + _table(["cell", "trainings", "all test sequences right", "at least 0.95 right", "less than 0.7 right"], dist_rows, "lrrrr")
    )


def accuracy_data():
    """{cell: [mean accuracy over the three runs at each k of TEST_KS]} and the window ceiling at each k."""
    res = training_results()
    out = {}
    for kind in ("rnn", "lstm", "gru"):
        out[kind] = [sum(res[(kind, s)][1][k] for s in SEEDS) / (len(SEEDS) * N_TEST_PER_K) for k in TEST_KS]
    ceil = [float(_window_ceiling(k)) if k <= 5 else 0.5 for k in TEST_KS]
    return out, ceil


def _stable_mean(kind, k):
    """Mean accuracy over the three runs of a cell at k, by the stable-digits rule over the runs and their nudged copies."""
    runs = _all_trainings()
    vals = [sum(runs[(kind, s, n)][1][k] for s in SEEDS) / (len(SEEDS) * N_TEST_PER_K) for n in (0.0, *stable.NUDGES)]
    return stable.text(vals[0], vals[1:], lambda x: _fx(x, 3))


def accuracy_by_distance():
    out, ceil = accuracy_data()
    rows = []
    for idx, k in enumerate(TEST_KS):
        rows.append((k, distance(k), "trained" if k in TRAIN_KS else "beyond", _fx(ceil[idx], 3), *[_stable_mean(c, k) for c in ("rnn", "lstm", "gru")]))
    return (
        f"Accuracy on the {N_TEST_PER_K} test sequences of each k, averaged over the three runs of a cell "
        f"(so a multiple of 1/{N_TEST_PER_K * len(SEEDS)}); the window column is the ceiling of the previous table "
        "(0.5 from k = 2, where the last four tokens do not contain the subject). " + stable.RULE + "\n\n"
        + _table(["k", "d", "in training", "window of 4 (ceiling)", "vanilla", "LSTM", "GRU"], rows, "rrlrrrr")
    )


# ------------------------------------------------------------------------------------------------ cost


def cell_params(kind, H, D):
    G = GATES[kind]
    return G * H * (H + D) + G * H


def cell_flops(kind, H, D):
    """FLOPs of one step. A matrix-vector product of m by n counts 2 m n; every addition, multiplication and
    activation of a unit counts 1 (tanh and the sigmoid are counted as one operation, as the roofline chapters do for
    elementwise functions)."""
    if kind == "rnn":
        return 2 * H * (H + D) + 3 * H
    if kind == "lstm":
        return 8 * H * (H + D) + 17 * H
    return 6 * H * (H + D) + 14 * H


def cell_costs():
    rows = []
    for H, D, label in ((TRAIN_H, V, "the trained cells (H = 12, one-hot input of width 12 as a dense vector)"),
                        (1024, 1024, "a large layer (H = 1024, input width 1024)")):
        for kind in ("rnn", "lstm", "gru"):
            rows.append((label if kind == "rnn" else "", CELL_NAMES[kind], _int(cell_params(kind, H, D)), _int(cell_flops(kind, H, D)),
                         _fx(cell_flops(kind, H, D) / cell_flops("rnn", H, D), 2)))
    # the formulas against the parameter arrays that were actually built
    def built(kind):
        p = init_params(kind, TRAIN_H, bookrng.generator("recurrent_networks.count"))
        return sum(len(r) for r in p["U"]) + sum(len(r) for r in p["W"]) + len(p["b"])

    ok = all(built(k) == cell_params(k, TRAIN_H, V) for k in GATES)
    return (
        "Parameters of one cell are G H (H + D) + G H with G = 1, 4, 3 for the vanilla cell, the LSTM and the GRU, H the "
        "number of units and D the width of the input (the readout of 4 H + 4 parameters is not counted). FLOPs of one step are "
        "2 H (H + D) + 3 H (vanilla), 8 H (H + D) + 17 H (LSTM) and 6 H (H + D) + 14 H (GRU): a matrix-vector product of m by n "
        "counts 2 m n, and every addition, multiplication and activation of a unit counts 1.\n\n"
        + _table(["setting", "cell", "parameters", "FLOPs per step", "FLOPs relative to the vanilla cell"], rows, "llrrr")
        + f"\n\nThe parameter formula {'agrees' if ok else 'DISAGREES'} with the number of entries of U, W and b that the initialization recipe builds for the three "
        f"cells at H = {TRAIN_H}, D = {V}. With a one-hot input, the product U x(t) is a lookup of a column and "
        f"costs no multiplication; the trained cells then spend {_int(cell_flops('rnn', TRAIN_H, V) - 2 * TRAIN_H * V)}, "
        f"{_int(cell_flops('lstm', TRAIN_H, V) - 8 * TRAIN_H * V)} and {_int(cell_flops('gru', TRAIN_H, V) - 6 * TRAIN_H * V)} FLOPs per step."
    )


def sequential_cost():
    H = D = 1024
    per_step = cell_flops("lstm", H, D)
    seq_part = 2 * 4 * H * H           # the recurrent product W h(t-1): cannot start before step t-1 has finished
    par_part = 2 * 4 * H * D           # the input product U x(t): known for every t at the start
    w_bytes = 4 * H * (H + D) * 4      # weights of the LSTM in f32
    rows = []
    for T in (16, 256, 4096):
        rows.append((_int(T), _int(per_step * T), _int(T), _int(w_bytes * T), _fx(per_step / w_bytes, 3),
                     _int(par_part * T), _int(seq_part * T)))
    rows2 = []
    for T in (16, 256, 4096):
        rows2.append((_int(T), _int(per_step * T), "1", _int(w_bytes), _fx(per_step * T / w_bytes, 1)))
    return (
        f"One LSTM layer with H = {H} units and input width D = {D}, batch of one sequence, f32 weights. FLOPs per step "
        f"{_int(per_step)} (8 H (H + D) + 17 H); weights {_int(w_bytes)} bytes (4 H (H + D) numbers of 4 bytes). Of the "
        f"matrix FLOPs of a step, {_int(par_part)} are the input product U x(t), which does not depend on the state and can be done for "
        f"all steps at once, and {_int(seq_part)} the recurrent product W h(t-1), which cannot start before step t - 1 has finished.\n\n"
        "Run as a recurrence, reading the weights again at every step (they do not fit in a fast cache at this size):\n\n"
        + _table(["T", "FLOPs, forward", "sequential steps", "weight bytes read", "FLOPs per byte",
                  "FLOPs of U x(t), all t, in one pass", "FLOPs of W h(t-1), sequential"], rows)
        + "\n\nThe same T positions through a feed-forward layer of the same shape (one product of T rows with the weights, "
        "no dependence between rows):\n\n"
        + _table(["T", "FLOPs, forward", "sequential steps", "weight bytes read", "FLOPs per byte"], rows2)
    )
