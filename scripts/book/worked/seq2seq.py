# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Encoder-Decoder Models and the Birth of Attention": reading one sequence and writing another.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description
and independent of any engine code. Each public function without arguments returns a markdown string that the chapter
embeds between `<!-- worked: seq2seq.<name> -->` markers; the `*_data` functions supply the numbers of the chapter's
figures, so a picture and the table beside it cannot disagree.

* the task (write a string of digits backwards, then an end marker), its data and the floor of a decoder that sees
  nothing of the input;
* one step of additive attention by hand on three encoder states of two numbers;
* an encoder-decoder pair of GRUs, with and without attention, its backward pass written out, and the check of that
  backward pass against central differences;
* the parameters of both models and the FLOPs of reading and writing a sequence of T digits, counted by a closed form
  and again by walking the loops;
* the two models trained on the task by Adam in minibatches, two runs each, their accuracy against the length of the
  input, the digits each loses, three decoded sequences and the alignment of the attention model on one input;
* the parameters of the encoder-decoder transformer of `crates/toyseq2seq` for its convergence test, counted from the
  shapes of its weights.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of
products are plain left-to-right additions, the order a C loop would use.

Swedish Embedded AB implements solutions for models that turn one sequence into another, such as speech, translation
and structured extraction. If your team needs expertise in sequence-to-sequence models and attention, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import hashlib
import json
import math
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import bookrng

# ------------------------------------------------------------------------------------------------ formatting


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


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

NS = 4                      # the digits 0 .. 3
BOS, EOS = 4, 5             # start-of-output and end-of-output markers
V = 6                       # ids 0 .. 3 digits, 4 `<s>`, 5 `<end>`
TOKEN_NAMES = ["0", "1", "2", "3", "<s>", "<end>"]
LMAX_TRAIN = 10             # training inputs have 1 .. 10 digits
TEST_LENGTHS = list(range(1, 13))
N_TEST_PER_LENGTH = 60
NAME_TEST, NAME_EXAMPLE = "seq2seq.test", "seq2seq.example"


def _words(ids):
    return " ".join(TOKEN_NAMES[t] for t in ids)


def target_of(src):
    """The output the task asks for: the digits backwards, then the end marker."""
    return src[::-1] + [EOS]


def test_set():
    """{length: [input digits]} with N_TEST_PER_LENGTH inputs for each length, drawn by integer(4) in the order of the
    lengths 1 .. 12, input after input."""
    rng = bookrng.generator(NAME_TEST)
    return {L: [[rng.integer(NS) for _ in range(L)] for _ in range(N_TEST_PER_LENGTH)] for L in TEST_LENGTHS}


def task():
    rng = bookrng.generator(NAME_EXAMPLE)
    rows = []
    for L in (1, 4, 7, 10):
        src = [rng.integer(NS) for _ in range(L)]
        rows.append((L, f"`{_words(src)}`", f"`{_words(target_of(src))}`", L + 1, _int(NS ** L)))
    t1 = _table(["input digits T", "input", "output", "output tokens T + 1", "possible inputs 4^T"], rows, "rllrr")
    floor_rows = [(L, _fx(math.log(NS), 4), _fx(L * math.log(NS), 4), _fx(L * math.log(NS) / (L + 1), 4)) for L in (1, 4, 7, 10)]
    t2 = _table(["T", "ln 4 per digit", "ln 4 summed over the T digits", "mean over the T + 1 tokens"], floor_rows, "rrrr")
    return (
        "The ids are `0 0`, `1 1`, `2 2`, `3 3`, `4 <s>` (the start marker fed to the decoder first) and `5 <end>`, so V = 6. "
        "An input is T digits with T between 1 and 10 in training (and 1 to 12 in the test), each digit uniform on 0 .. 3; "
        "the output is the digits in the opposite order followed by `<end>`.\n\n"
        + _seed_line(NAME_EXAMPLE) + " Draw order: for each of the four examples, T times `integer(4)`.\n\n"
        + t1 + "\n\n"
        "A decoder that has no access to the input can only guess each digit uniformly: its cross-entropy is ln 4 nats on each "
        "digit. Suppose it were even told T, so that the `<end>` costs it nothing. Its loss per output token would then be, for "
        "a few lengths:\n\n" + t2
    )


# ------------------------------------------------------------------------------------------------ arithmetic helpers


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


def _sum(xs):
    s = 0.0
    for x in xs:
        s += x
    return s


def _softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = _sum(e)
    return [x / s for x in e]


def _mat(rng, rows, cols, scale):
    return [[scale * rng.normal() for _ in range(cols)] for _ in range(rows)]


# ------------------------------------------------------------------------------------------------ the model

H_TRAIN, DM_TRAIN, A_TRAIN = 16, 8, 16        # state width, embedding width, attention width


def init_params(attention, H, DM, A, rng):
    """Parameters drawn in this order, each matrix row by row, each entry `scale * normal()`:
    E [V, DM] (scale 0.5); encoder Wx [3H, DM], Wh [3H, H] (scales 1/sqrt(DM), 1/sqrt(H)); decoder Wx [3H, DX], Wh [3H, H]
    with DX = DM, or DM + H with attention (scales 1/sqrt(DX), 1/sqrt(H)); output Wo [V, H] (1/sqrt(H)); and with
    attention Wa [A, H], Ua [A, H] (1/sqrt(H)) and v [A] (1/sqrt(A)). Every bias is 0. The GRU blocks are ordered z, r, n."""
    DX = DM + (H if attention else 0)
    p = {"E": _mat(rng, V, DM, 0.5),
         "eWx": _mat(rng, 3 * H, DM, 1 / math.sqrt(DM)), "eWh": _mat(rng, 3 * H, H, 1 / math.sqrt(H)),
         "dWx": _mat(rng, 3 * H, DX, 1 / math.sqrt(DX)), "dWh": _mat(rng, 3 * H, H, 1 / math.sqrt(H)),
         "Wo": _mat(rng, V, H, 1 / math.sqrt(H))}
    if attention:
        p["Wa"] = _mat(rng, A, H, 1 / math.sqrt(H))
        p["Ua"] = _mat(rng, A, H, 1 / math.sqrt(H))
        p["v"] = _mat(rng, 1, A, 1 / math.sqrt(A))
    p["eb"] = [0.0] * (3 * H)
    p["db"] = [0.0] * (3 * H)
    p["bo"] = [0.0] * V
    return p


def n_params(p):
    return sum(len(r) for r in _rows(p))


def _zeros_like(p):
    return {k: ([[0.0] * len(r) for r in m] if isinstance(m[0], list) else [0.0] * len(m)) for k, m in p.items()}


def _rows(p):
    """Every row of every parameter (a bias is one row), in a fixed order."""
    out = []
    for k in p:
        m = p[k]
        out += m if isinstance(m[0], list) else [m]
    return out


def gru_step(Wx, Wh, b, H, h, x):
    """h' = (1 - z) h + z n with z = sigmoid(Wz x + Uz h + bz), r = sigmoid(...), n = tanh(Wn x + Un (r h) + bn)."""
    ax = [_dot(r, x) for r in Wx]
    ah = [_dot(Wh[i], h) for i in range(2 * H)]
    z = [_sigmoid(ax[i] + ah[i] + b[i]) for i in range(H)]
    r = [_sigmoid(ax[H + i] + ah[H + i] + b[H + i]) for i in range(H)]
    rh = [r[i] * h[i] for i in range(H)]
    n = [math.tanh(ax[2 * H + i] + _dot(Wh[2 * H + i], rh) + b[2 * H + i]) for i in range(H)]
    h2 = [(1.0 - z[i]) * h[i] + z[i] * n[i] for i in range(H)]
    return h2, (x, h, z, r, rh, n)


def gru_back(Wx, Wh, H, cache, dh2, g):
    """Gradient of the loss with respect to the input and the previous state of one GRU step, given the gradient dh2
    of the state it produced; the parameter gradients are added into g = (dWx, dWh, db)."""
    x, h, z, r, rh, n = cache
    gWx, gWh, gb = g
    dh = [dh2[i] * (1.0 - z[i]) for i in range(H)]
    dan = [dh2[i] * z[i] * (1.0 - n[i] * n[i]) for i in range(H)]
    daz = [dh2[i] * (n[i] - h[i]) * z[i] * (1.0 - z[i]) for i in range(H)]
    drh = [0.0] * H
    for i in range(H):
        row = Wh[2 * H + i]
        for j in range(H):
            drh[j] += dan[i] * row[j]
    dar = [drh[j] * h[j] * r[j] * (1.0 - r[j]) for j in range(H)]
    for j in range(H):
        dh[j] += drh[j] * r[j]
    da = daz + dar + dan
    dx = [0.0] * len(x)
    for i in range(3 * H):
        d = da[i]
        gb[i] += d
        row, grow = Wx[i], gWx[i]
        for j in range(len(x)):
            grow[j] += d * x[j]
            dx[j] += d * row[j]
        row, grow = Wh[i], gWh[i]
        src = h if i < 2 * H else rh
        for j in range(H):
            grow[j] += d * src[j]
            if i < 2 * H:
                dh[j] += d * row[j]
    return dx, dh


def encode(p, H, src):
    """The encoder states h_1 .. h_T and the caches of the steps; h_0 = 0 and the input of step t is row src[t] of E."""
    h = [0.0] * H
    states, caches = [], []
    for t in src:
        h, c = gru_step(p["eWx"], p["eWh"], p["eb"], H, h, p["E"][t])
        states.append(h)
        caches.append(c)
    return states, caches


def attend(p, H, keys, states, s_prev):
    """Additive attention: e_i = v . tanh(Wa s_prev + keys_i) with keys_i = Ua h_i, a = softmax(e), c = sum_i a_i h_i.
    Returns the context, the weights and the tanh activations (kept for the backward pass)."""
    q = [_dot(r, s_prev) for r in p["Wa"]]
    v = p["v"][0]
    acts, scores = [], []
    for k in keys:
        u = [math.tanh(q[a] + k[a]) for a in range(len(q))]
        acts.append(u)
        scores.append(_dot(v, u))
    w = _softmax(scores)
    c = [0.0] * H
    for wi, h in zip(w, states):
        for j in range(H):
            c[j] += wi * h[j]
    return c, w, acts


def _keys(p, states):
    return [[_dot(r, h) for r in p["Ua"]] for h in states]


def loss_and_grad(p, attention, H, DM, A, batch, g):
    """Mean cross-entropy per output token over a list of input sequences (the decoder is teacher-forced: its input is
    `<s>` then the target shifted right), with the gradient of every parameter added into g."""
    total, ntok = 0.0, sum(len(s) + 1 for s in batch)
    for src in batch:
        T = len(src)
        tgt = target_of(src)
        dec_in = [BOS] + tgt[:-1]
        states, ecaches = encode(p, H, src)
        keys = _keys(p, states) if attention else None
        s = states[-1]
        steps = []
        for j, tok in enumerate(dec_in):
            e = p["E"][tok]
            if attention:
                c, w, acts = attend(p, H, keys, states, s)
                x = e + c
            else:
                c = w = acts = None
                x = e
            s2, cache = gru_step(p["dWx"], p["dWh"], p["db"], H, s, x)
            probs = _softmax([_dot(r, s2) + b for r, b in zip(p["Wo"], p["bo"])])
            total -= math.log(probs[tgt[j]])
            steps.append((cache, s, s2, probs, w, acts, tok))
            s = s2
        ds = [0.0] * H
        dstates = [[0.0] * H for _ in range(T)]
        for j in range(len(dec_in) - 1, -1, -1):
            cache, sp, s2, probs, w, acts, tok = steps[j]
            dl = [q / ntok for q in probs]
            dl[tgt[j]] -= 1.0 / ntok
            for k in range(V):
                g["bo"][k] += dl[k]
                for i in range(H):
                    g["Wo"][k][i] += dl[k] * s2[i]
                    ds[i] += dl[k] * p["Wo"][k][i]
            dx, dsp = gru_back(p["dWx"], p["dWh"], H, cache, ds, (g["dWx"], g["dWh"], g["db"]))
            for i in range(DM):
                g["E"][tok][i] += dx[i]
            if attention:
                dc = dx[DM:]
                dw = [_dot(dc, h) for h in states]
                for wi, dh in zip(w, dstates):
                    for i in range(H):
                        dh[i] += wi * dc[i]
                mean_dw = _dot(w, dw)
                v = p["v"][0]
                dq = [0.0] * A
                for i in range(T):
                    de = w[i] * (dw[i] - mean_dw)           # softmax backward
                    for a in range(A):
                        g["v"][0][a] += de * acts[i][a]
                        dpre = (1.0 - acts[i][a] * acts[i][a]) * de * v[a]
                        dq[a] += dpre
                        for k in range(H):
                            g["Ua"][a][k] += dpre * states[i][k]
                            dstates[i][k] += dpre * p["Ua"][a][k]
                for a in range(A):
                    for k in range(H):
                        g["Wa"][a][k] += dq[a] * sp[k]
                        dsp[k] += dq[a] * p["Wa"][a][k]
            ds = dsp
        for i in range(H):
            dstates[-1][i] += ds[i]                          # the decoder started from the last encoder state
        dh = [0.0] * H
        for t in range(T - 1, -1, -1):
            tot = [a + b for a, b in zip(dh, dstates[t])]
            dx, dh = gru_back(p["eWx"], p["eWh"], H, ecaches[t], tot, (g["eWx"], g["eWh"], g["eb"]))
            for i in range(DM):
                g["E"][src[t]][i] += dx[i]
    return total / ntok


def generate(p, attention, H, src):
    """Greedy decoding for len(src) + 1 steps, each step fed the model's own previous output (first `<s>`).
    Returns the output ids and, with attention, the weights of every step."""
    states, _ = encode(p, H, src)
    keys = _keys(p, states) if attention else None
    s, tok, out, weights = states[-1], BOS, [], []
    for _ in range(len(src) + 1):
        x = p["E"][tok]
        if attention:
            c, w, _ = attend(p, H, keys, states, s)
            x = x + c
            weights.append(w)
        s, _ = gru_step(p["dWx"], p["dWh"], p["db"], H, s, x)
        logits = [_dot(r, s) + b for r, b in zip(p["Wo"], p["bo"])]
        tok = max(range(V), key=lambda k: logits[k])        # first index on a tie
        out.append(tok)
    return out, weights


# ------------------------------------------------------------------------------------------------ one attention step by hand

STEP_NAME = "seq2seq.step"


def attention_step():
    rng = bookrng.generator(STEP_NAME)
    Wa = _mat(rng, 2, 2, 0.8)
    Ua = _mat(rng, 2, 2, 0.8)
    v = [0.8 * rng.normal() for _ in range(2)]
    s = [0.8 * rng.normal() for _ in range(2)]
    hs = [[0.8 * rng.normal() for _ in range(2)] for _ in range(3)]
    q = [_dot(r, s) for r in Wa]
    rows, scores, acts = [], [], []
    for i, h in enumerate(hs):
        k = [_dot(r, h) for r in Ua]
        pre = [q[a] + k[a] for a in range(2)]
        u = [math.tanh(x) for x in pre]
        e = _dot(v, u)
        scores.append(e)
        acts.append(u)
        rows.append((i + 1, _vec(h), _vec(k), _vec(pre), _vec(u), _fx(e)))
    w = _softmax(scores)
    c = [_sum([w[i] * hs[i][j] for i in range(3)]) for j in range(2)]
    cc, ww, _ = attend({"Wa": Wa, "v": [v]}, 2, [[_dot(r, h) for r in Ua] for h in hs], hs, s)
    assert all(abs(a - b) < 1e-15 for a, b in zip(cc + ww, c + w))
    t = _table(["i", "encoder state h_i", "keys_i = Ua h_i", "Wa s + keys_i", "tanh of that", "score e_i = v . tanh"], rows, "rlllll")
    return (
        f"{_seed_line(STEP_NAME)} Draws, in this order, each `0.8 * normal()`: Wa (4 values, row by row), Ua (4), v (2), the decoder "
        "state s (2), then the three encoder states h_1, h_2, h_3 (2 values each). Two numbers per state (H = 2) and two in the "
        "attention layer (A = 2). Tolerance: values are rounded to 4 decimals.\n\n"
        f"Wa = {[_vec(r) for r in Wa]}, Ua = {[_vec(r) for r in Ua]}, v = {_vec(v)}, s = {_vec(s)}, so Wa s = {_vec(q)}.\n\n"
        + t + "\n\n"
        f"The softmax of the scores {_vec(scores)} gives the weights a = {_vec(w)}, which add to {_fx(_sum(w), 4)}, "
        f"and the context is c = sum of a_i h_i = {_vec(c)}."
    )


# ------------------------------------------------------------------------------------------------ gradient check

CHECK_H, CHECK_DM, CHECK_A, CHECK_SRC, EPS = 3, 2, 3, [2, 0, 3], 1e-5


def gradient_check():
    rows = []
    for attention in (False, True):
        name = f"seq2seq.check.{'attention' if attention else 'plain'}"
        rng = bookrng.generator(name)
        p = init_params(attention, CHECK_H, CHECK_DM, CHECK_A, rng)
        for k in ("eb", "db", "bo"):
            for i in range(len(p[k])):
                p[k][i] = 0.1 * rng.normal()                 # nonzero biases, so their gradients are tested too
        g = _zeros_like(p)
        loss_and_grad(p, attention, CHECK_H, CHECK_DM, CHECK_A, [CHECK_SRC], g)
        worst_abs, worst_rel, count, smallest = 0.0, 0.0, 0, 1e9
        for pr, gr in zip(_rows(p), _rows(g)):
            for j in range(len(pr)):
                old = pr[j]
                pr[j] = old + EPS
                up = loss_and_grad(p, attention, CHECK_H, CHECK_DM, CHECK_A, [CHECK_SRC], _zeros_like(p))
                pr[j] = old - EPS
                dn = loss_and_grad(p, attention, CHECK_H, CHECK_DM, CHECK_A, [CHECK_SRC], _zeros_like(p))
                pr[j] = old
                num = (up - dn) / (2 * EPS)
                worst_abs = max(worst_abs, abs(num - gr[j]))
                if max(abs(num), abs(gr[j])) > 1e-4:
                    worst_rel = max(worst_rel, abs(num - gr[j]) / max(abs(num), abs(gr[j])))
                    smallest = min(smallest, max(abs(num), abs(gr[j])))
                count += 1
        rows.append((f"{'attention' if attention else 'no attention'} (`{name}`)", f"{bookrng.seed_of(name):#018x}", count, f"{worst_abs:.1e}",
                     f"{worst_rel:.1e}"))
    return (
        f"Input {CHECK_SRC} (T = {len(CHECK_SRC)}, so the output has {len(CHECK_SRC) + 1} tokens), H = {CHECK_H}, embedding width {CHECK_DM}, "
        f"A = {CHECK_A}. Parameters drawn by the initialisation recipe of the text from the generator named `seq2seq.check.plain` for the model without attention and `seq2seq.check.attention` for the one with it (seeds below), "
        "after which the three biases get `0.1 * normal()` entries drawn from the same generator, in the order encoder bias, "
        f"decoder bias, output bias. Central differences with step {EPS:g} on every entry of every parameter, against the "
        "gradient of the backward pass. Tolerance: the table prints the largest absolute difference and the largest relative "
        "difference |a - n| / max(|a|, |n|) over the entries whose gradient is above 1e-4.\n\n"
        + _table(["model", "seed", "parameters checked", "largest absolute difference", "largest relative difference"], rows, "llrrr")
    )


# ------------------------------------------------------------------------------------------------ parameters and FLOPs


def model_param_count(attention, H, DM, A):
    DX = DM + (H if attention else 0)
    n = V * DM + 3 * H * (DM + H + 1) + 3 * H * (DX + H + 1) + V * H + V
    return n + (2 * A * H + A if attention else 0)


def parameters():
    rows = []
    for attention in (False, True):
        p = init_params(attention, H_TRAIN, DM_TRAIN, A_TRAIN, bookrng.generator("seq2seq.count"))
        counted = n_params(p)
        formula = model_param_count(attention, H_TRAIN, DM_TRAIN, A_TRAIN)
        assert counted == formula
        rows.append(("attention" if attention else "no attention", _int(counted),
                     _int(V * DM_TRAIN), _int(3 * H_TRAIN * (DM_TRAIN + H_TRAIN + 1)),
                     _int(3 * H_TRAIN * ((DM_TRAIN + (H_TRAIN if attention else 0)) + H_TRAIN + 1)),
                     _int(V * H_TRAIN + V), _int(2 * A_TRAIN * H_TRAIN + A_TRAIN) if attention else "0"))
    return (
        f"H = {H_TRAIN} state numbers, embedding width {DM_TRAIN}, attention width A = {A_TRAIN}, V = {V} ids. A GRU with input width D "
        "has 3 H (D + H + 1) parameters (the matrices on the input and on the state, and the bias); the encoder reads the embedding "
        "(D = embedding width), the decoder reads the embedding, plus the context of H numbers when it has attention. The table "
        "counts the entries of the parameter lists the training builds, and the formula agrees with that count.\n\n"
        + _table(["model", "all parameters", "embedding table", "encoder GRU", "decoder GRU", "output layer", "attention (Wa, Ua, v)"], rows)
    )


def gru_flops(H, D):
    """FLOPs of one GRU step with input width D: a matrix-vector product of m by n counts 2 m n, every addition,
    multiplication and activation of a unit counts 1 (the count of the previous chapter, 6 H (H + D) + 14 H)."""
    return 6 * H * (H + D) + 14 * H


def attention_flops_closed(T, H, A):
    """FLOPs of the attention of one decoder step over T encoder states, with the keys Ua h_i already formed:
    the query Wa s costs 2 A H; each position costs A additions, A tanh and a dot product with v (2 A), 4 A in all;
    the softmax costs 3 T (an exponential, an addition into the sum and a division for each position); the context costs
    2 T H (a multiplication and an addition for each entry)."""
    return 2 * A * H + T * 4 * A + 3 * T + 2 * T * H


def attention_flops_walked(T, H, A):
    """The same count, tallied by walking the loops of the computation one operation at a time."""
    n = 0
    for _ in range(A):                      # query: A dot products of length H
        n += 2 * H
    for _ in range(T):                      # one position
        for _ in range(A):
            n += 1                          # add the query to the key
            n += 1                          # tanh
            n += 2                          # multiply by v and add into the score
    for _ in range(T):                      # softmax
        n += 3
    for _ in range(H):                      # context entries
        for _ in range(T):
            n += 2
    return n


def pair_flops(attention, T, H, DM, A):
    """Forward FLOPs of reading T digits and writing T + 1 output tokens (the output layer included, the loss not)."""
    enc = T * gru_flops(H, DM)
    out = (T + 1) * (2 * V * H + V)
    if not attention:
        return enc + (T + 1) * gru_flops(H, DM) + out
    keys = T * 2 * A * H
    return enc + keys + (T + 1) * (gru_flops(H, DM + H) + attention_flops_closed(T, H, A)) + out


def attention_cost():
    for T in (1, 7, 64):
        assert attention_flops_closed(T, H_TRAIN, A_TRAIN) == attention_flops_walked(T, H_TRAIN, A_TRAIN)
    rows = []
    for T in (4, 8, 16, 32, 64, 128):
        att = (T + 1) * attention_flops_closed(T, H_TRAIN, A_TRAIN) + T * 2 * A_TRAIN * H_TRAIN
        base = pair_flops(False, T, H_TRAIN, DM_TRAIN, A_TRAIN)
        full = pair_flops(True, T, H_TRAIN, DM_TRAIN, A_TRAIN)
        rows.append((T, _int(base), _int(att), _int(full), _fx(full / base, 2), _fx(att / full, 3)))
    t1 = _table(["T", "FLOPs without attention", "FLOPs of the attention", "FLOPs with attention", "ratio", "attention share"], rows)
    big_h = 1024
    gru = gru_flops(big_h, 2 * big_h)
    per_pair = 4 * big_h + 3 + 2 * big_h
    query = 2 * big_h * big_h                  # the constant term 2 A H of the query Wa s
    cross = None
    for T in range(1, 100000):
        if query + T * per_pair >= gru:
            cross = T
            break
    ctx_bytes = lambda T, H, A: (4 * H, 4 * T * H, 4 * T * A, 4 * (T + 1) * T)
    brows = []
    for T in (8, 64, 512):
        c, st, ky, al = ctx_bytes(T, H_TRAIN, A_TRAIN)
        brows.append((T, _int(c), _int(st + ky), _int(al)))
    t2 = _table(["T", "context vector alone (bytes)", "encoder states and keys kept for attention (bytes)",
                 "weights of all output steps (bytes)"], brows)
    return (
        f"Counting rule as in the previous chapter: a matrix-vector product of m by n is 2 m n, every addition, multiplication and activation "
        f"is 1; a GRU step with input width D is 6 H (H + D) + 14 H FLOPs. The attention of one decoder step over T encoder states costs "
        f"2 A H + 4 A T + 3 T + 2 T H FLOPs, after the keys Ua h_i of the T states (2 A H each) have been formed once per input; "
        f"the closed form agrees with a tally made by walking the loops (checked for T = 1, 7 and 64). H = {H_TRAIN}, embedding width "
        f"{DM_TRAIN}, A = {A_TRAIN}, V = {V}; reading T digits and writing T + 1 tokens (output layer included, loss not):\n\n"
        + t1 + "\n\n"
        f"For a layer of the size that language models use, H = A = {big_h} and a decoder input of width 2 H (the embedding and the "
        f"context), a GRU step costs {_int(gru)} FLOPs and the attention step has a constant part, the query, of {_int(query)} FLOPs, and each encoder position adds {_int(per_pair)}, so the "
        f"attention step ({_int(query)} + {_int(per_pair)} T) first costs at least as much as the recurrent step itself at T = {cross}.\n\n"
        "Bytes of float32 numbers that a model must hold to answer one input of T digits:\n\n" + t2
    )


# ------------------------------------------------------------------------------------------------ training

VARIANTS = ("plain", "attention")
RUNS = (0, 1)
BATCH, STEPS = 16, 1500
LR, BETA1, BETA2, EPS_ADAM, CLIP = 0.01, 0.9, 0.999, 1e-8, 5.0
CHECKPOINTS = (100, 300, 600, 1000, 1500)
WINDOW = 100                    # a checkpoint reports the mean loss of the WINDOW batches that end at it
POS_LENGTH = 8                  # the length at which the digits lost are counted
EXAMPLE_LENGTH = 6              # the length of the printed alignment
SHOWN_LENGTHS = (4, 8, 10)      # lengths of the three decoded sequences shown


def _name(variant, run, part):
    return f"seq2seq.{variant}.{run}.{part}"


def _train_job(job):
    variant, run = job
    attention = variant == "attention"
    H, DM, A = H_TRAIN, DM_TRAIN, A_TRAIN
    p = init_params(attention, H, DM, A, bookrng.generator(_name(variant, run, "init")))
    data = bookrng.generator(f"seq2seq.train.{run}")        # the same batches for both variants of a run
    rows = _rows(p)
    m = [[0.0] * len(r) for r in rows]
    vv = [[0.0] * len(r) for r in rows]
    losses = []
    for step in range(1, STEPS + 1):
        batch = []
        for _ in range(BATCH):
            L = 1 + data.integer(LMAX_TRAIN)
            batch.append([data.integer(NS) for _ in range(L)])
        g = _zeros_like(p)
        losses.append(loss_and_grad(p, attention, H, DM, A, batch, g))
        grows = _rows(g)
        gn = math.sqrt(_sum([x * x for r in grows for x in r]))
        scale = min(1.0, CLIP / gn) if gn > 0 else 1.0
        c1, c2 = 1.0 - BETA1 ** step, 1.0 - BETA2 ** step
        for pr, gr, mr, vr in zip(rows, grows, m, vv):
            for j in range(len(pr)):
                gj = gr[j] * scale
                mr[j] = BETA1 * mr[j] + (1.0 - BETA1) * gj
                vr[j] = BETA2 * vr[j] + (1.0 - BETA2) * gj * gj
                pr[j] -= LR * (mr[j] / c1) / (math.sqrt(vr[j] / c2) + EPS_ADAM)
    curve = {c: _sum(losses[c - WINDOW:c]) / WINDOW for c in CHECKPOINTS}
    tests = test_set()
    correct, token_ok, lost = {}, {}, [0] * POS_LENGTH
    for L, seqs in tests.items():
        ok = tok = 0
        for src in seqs:
            out, _ = generate(p, attention, H, src)
            want = target_of(src)
            ok += out == want
            tok += sum(1 for a, b in zip(out, want) if a == b)
            if L == POS_LENGTH:
                for i in range(L):
                    lost[i] += out[L - 1 - i] == want[L - 1 - i]    # input digit i is written at output position L - 1 - i
        correct[L], token_ok[L] = ok, tok
    rng = bookrng.generator(NAME_EXAMPLE + ".align")
    align_src = [rng.integer(NS) for _ in range(EXAMPLE_LENGTH)]
    align_out, align_w = generate(p, attention, H, align_src)
    shown = {}
    srng = bookrng.generator(NAME_EXAMPLE + ".shown")
    for L in SHOWN_LENGTHS:
        src = [srng.integer(NS) for _ in range(L)]
        shown[L] = (src, generate(p, attention, H, src)[0])
    return {"variant": variant, "run": run, "curve": curve, "correct": correct, "token_ok": token_ok, "lost": lost,
            "align": {"src": align_src, "out": align_out, "w": align_w}, "shown": shown}


TRAIN_CACHE = Path(__file__).resolve().parents[3] / "build" / "book" / "seq2seq_training.json"


def _source_digest():
    """Digest of this module and of the generator it draws from: the key under which a finished training is kept."""
    h = hashlib.sha256()
    for f in (Path(__file__), Path(bookrng.__file__)):
        h.update(f.read_bytes())
    return h.hexdigest()


def _decode(r):
    r["curve"] = {int(k): v for k, v in r["curve"].items()}
    r["correct"] = {int(k): v for k, v in r["correct"].items()}
    r["token_ok"] = {int(k): v for k, v in r["token_ok"].items()}
    r["shown"] = {int(k): v for k, v in r["shown"].items()}
    return r


@lru_cache(maxsize=None)
def training_results():
    """{(variant, run): result}; the four runs are independent and are spread over four worker processes. Training is
    deterministic, so a finished result is kept in build/book (keyed by the digest of this module and of bookrng.py) and the
    chapter's blocks and its figures, which are computed by different programs, share one training."""
    key = _source_digest()
    if TRAIN_CACHE.is_file():
        try:
            stored = json.loads(TRAIN_CACHE.read_text(encoding="utf-8"))
            if stored.get("key") == key:
                return {(r["variant"], r["run"]): _decode(r) for r in stored["runs"]}
        except (ValueError, KeyError):
            pass
    jobs = [(variant, run) for variant in VARIANTS for run in RUNS]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_train_job, jobs, chunksize=1)
    TRAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_CACHE.write_text(json.dumps({"key": key, "runs": res}), encoding="utf-8")
    return {(r["variant"], r["run"]): _decode(json.loads(json.dumps(r))) for r in res}


LABEL = {"plain": "no attention", "attention": "attention"}


def _training_rows(variants):
    res = training_results()
    n = lambda lo, hi: N_TEST_PER_LENGTH * len([L for L in TEST_LENGTHS if lo <= L <= hi])
    rows = []
    for variant in variants:
        for run in RUNS:
            r = res[(variant, run)]
            short = sum(r["correct"][L] for L in TEST_LENGTHS if L <= 4)
            mid = sum(r["correct"][L] for L in TEST_LENGTHS if 5 <= L <= LMAX_TRAIN)
            beyond = sum(r["correct"][L] for L in TEST_LENGTHS if L > LMAX_TRAIN)
            rows.append((LABEL[variant], run, f"{bookrng.seed_of(_name(variant, run, 'init')):#018x}",
                         *[_fx(r["curve"][c], 3) for c in CHECKPOINTS],
                         f"{short}/{n(1, 4)}", f"{mid}/{n(5, LMAX_TRAIN)}", f"{beyond}/{n(LMAX_TRAIN + 1, 12)}"))
    head = (["model", "run", "seed of the initial parameters"] + [f"loss, step {c}" for c in CHECKPOINTS]
            + ["exact, T 1 to 4", f"exact, T 5 to {LMAX_TRAIN}", f"exact, T {LMAX_TRAIN + 1} to 12"])
    return _table(head, rows, "lrl" + "r" * (len(head) - 3))


def training_plain():
    return (
        f"The model is the one of the text, H = {H_TRAIN} state numbers, embedding width {DM_TRAIN}, V = {V}, the parameters drawn by the initialisation recipe "
        "of the text from the generator named `seq2seq.plain.<run>.init` with run 0 or 1 (`plain` is the model without attention; seeds in the table). "
        f"Training is Adam (learning rate {LR}, beta1 {BETA1}, beta2 {BETA2}, epsilon {EPS_ADAM:g}, with the bias correction of Volume III) "
        f"on minibatches of {BATCH} freshly drawn input sequences, {STEPS} updates ({STEPS * BATCH:,} sequences), every gradient clipped to a "
        f"global norm of {CLIP:g}. Batch element after batch element, the length is `1 + integer({LMAX_TRAIN})` and then that many "
        "`integer(4)` digits, from the generator `seq2seq.train.<run>`, so that the two models of a run (this one and the one of the "
        f"attention section) see the same sequences. The loss is the mean cross-entropy per output token over the batch; a checkpoint reports "
        f"the mean over the {WINDOW} batches that end at it. The test set has {N_TEST_PER_LENGTH} inputs for each length 1 to 12, drawn from "
        f"`{NAME_TEST}` (seed {bookrng.seed_of(NAME_TEST):#018x}), length by length; an input counts as exact when the greedy output, run "
        "for T + 1 steps on its own previous outputs, equals the reversed input followed by `<end>`. "
        "Losses are rounded to 3 decimals. They and the counts are reproduced digit for digit only by an implementation that adds in the "
        "same order; a trained network is a chaotic function of its arithmetic, so a reimplementation should expect the same picture "
        "and not the same counts.\n\n" + _training_rows(("plain",))
    )


def training_attention():
    return (
        "The model of the text with attention (A = 16), parameters drawn by the same initialisation recipe from the generator `seq2seq.attention.<run>.init` with run 0 or 1, trained "
        "by the protocol of the previous training block (same optimizer, batches, clipping, test set and counting; the sequences of "
        "run r are those of run r there). Same tolerance.\n\n" + _training_rows(("attention",))
    )


def accuracy_data():
    """{variant: [exact-match accuracy at each test length, pooled over the runs]} and the same for digits written correctly."""
    res = training_results()
    exact = {v: [sum(res[(v, r)]["correct"][L] for r in RUNS) / (len(RUNS) * N_TEST_PER_LENGTH) for L in TEST_LENGTHS] for v in VARIANTS}
    tokens = {v: [sum(res[(v, r)]["token_ok"][L] for r in RUNS) / (len(RUNS) * N_TEST_PER_LENGTH * (L + 1)) for L in TEST_LENGTHS]
              for v in VARIANTS}
    return exact, tokens


def accuracy_by_length():
    exact, tokens = accuracy_data()
    rows = []
    for idx, L in enumerate(TEST_LENGTHS):
        rows.append((L, "trained" if L <= LMAX_TRAIN else "beyond", _fx(exact["plain"][idx], 3), _fx(exact["attention"][idx], 3),
                     _fx(tokens["plain"][idx], 3), _fx(tokens["attention"][idx], 3)))
    return (
        f"Accuracy over the {len(RUNS) * N_TEST_PER_LENGTH} test inputs of each length (the {N_TEST_PER_LENGTH} of the two runs together, so a "
        f"multiple of 1/{len(RUNS) * N_TEST_PER_LENGTH}). Exact: the whole output is right. Token: the fraction of the T + 1 output tokens "
        "that are right, each compared with the same position of the target.\n\n"
        + _table(["T", "in training", "exact, no attention", "exact, attention", "token, no attention", "token, attention"], rows, "rlrrrr")
    )


def plain_accuracy():
    exact, tokens = accuracy_data()
    rows = []
    for idx, L in enumerate(TEST_LENGTHS):
        rows.append((L, "trained" if L <= LMAX_TRAIN else "beyond", _fx(exact["plain"][idx], 3), _fx(tokens["plain"][idx], 3)))
    return (
        f"Accuracy of the model without attention over the {len(RUNS) * N_TEST_PER_LENGTH} test inputs of each length (the {N_TEST_PER_LENGTH} of the "
        f"two runs together, so a multiple of 1/{len(RUNS) * N_TEST_PER_LENGTH}). Exact: the whole output is right. Token: the fraction of the T + 1 "
        "output tokens that are right, each compared with the same position of the target.\n\n"
        + _table(["T", "in training", "exact", "token"], rows, "rlrr")
    )


def lost_digits():
    res = training_results()
    n = len(RUNS) * N_TEST_PER_LENGTH
    rows = []
    for i in range(POS_LENGTH):
        rows.append((i + 1, POS_LENGTH - i, _fx(sum(res[("plain", r)]["lost"][i] for r in RUNS) / n, 3)))
    return (
        f"Inputs of T = {POS_LENGTH} digits (the {N_TEST_PER_LENGTH} test inputs of that length, for each of the two runs). For input digit "
        "number i (counting from 1 in the order of reading) the table gives the fraction of inputs for which the model writes that digit "
        "correctly at the place where it belongs, output position T + 1 - i, each output being generated on the model's own previous "
        "outputs. Digit 1 is read first and written last.\n\n"
        + _table(["input digit i", "written at output position", "fraction written correctly"], rows)
    )


def plain_outputs():
    res = training_results()
    rows = []
    for L in SHOWN_LENGTHS:
        src = res[("plain", 0)]["shown"][L][0]
        rows.append((L, f"`{_words(src)}`", f"`{_words(target_of(src))}`", f"`{_words(res[('plain', 0)]['shown'][L][1])}`"))
    return (
        f"Run 0 of the model without attention on three inputs of the generator `{NAME_EXAMPLE}.shown` (seed {bookrng.seed_of(NAME_EXAMPLE + '.shown'):#018x}; "
        f"draws, `integer(4)` for each digit, of the lengths {list(SHOWN_LENGTHS)} in that order), greedy output on its own previous outputs.\n\n"
        + _table(["T", "input", "target", "output"], rows, "rlll")
    )


def alignment_data():
    """(input digits, output ids, weights [T + 1][T]) of the attention model, run 0, on the printed example."""
    a = training_results()[("attention", 0)]["align"]
    return a["src"], a["out"], a["w"]


def alignment():
    src, out, w = alignment_data()
    names = [f"x{i + 1}={src[i]}" for i in range(len(src))]
    rows = []
    for j, row in enumerate(w):
        peak = max(range(len(row)), key=lambda i: row[i])
        rows.append((j + 1, f"`{TOKEN_NAMES[out[j]]}`", *[_fx(x, 2) for x in row], f"x{peak + 1}"))
    return (
        f"The attention model of run 0 on the input `{_words(src)}` (T = {EXAMPLE_LENGTH}, drawn by `integer(4)` from the generator "
        f"`{NAME_EXAMPLE}.align`, seed {bookrng.seed_of(NAME_EXAMPLE + '.align'):#018x}). Row j holds the weights a_ji of output step j over the "
        "encoder positions, the token written at that step is in the second column; each row adds to 1. Tolerance: 2 decimals "
        "of a trained network (see the training block on what a reimplementation reproduces).\n\n"
        + _table(["output step j", "token written"] + names + ["largest weight on"], rows, "rl" + "r" * len(src) + "l")
    )


# ------------------------------------------------------------------------------------------------ the case study

CS = dict(vocab=7, src=5, dec=5, n_enc=2, n_dec=2, d=32, heads=4, ff=128, batch=32)


def crate_scale():
    """Parameters of the encoder-decoder transformer of the convergence test, from the shapes of its weights."""
    c = CS
    d, ff, v = c["d"], c["ff"], c["vocab"]
    emb = v * d + c["src"] * d + c["dec"] * d
    enc_block = 2 * d + 3 * d * d + 3 * d + d * d + d + 2 * d + ff * d + ff + d * ff + d
    cross = d * d + d + 2 * d * d + 2 * d + d * d + d
    dec_block = enc_block + cross + 2 * d
    head = 2 * d + v * d
    total = emb + c["n_enc"] * enc_block + c["n_dec"] * dec_block + head
    scores_per_layer = c["batch"] * c["heads"] * c["dec"] * c["src"]
    rows = [
        ("token table, encoder positions, decoder positions", _int(emb)),
        (f"{c['n_enc']} encoder blocks of {_int(enc_block)}", _int(c["n_enc"] * enc_block)),
        (f"{c['n_dec']} decoder blocks of {_int(dec_block)}, of which cross-attention {_int(cross)} each", _int(c["n_dec"] * dec_block)),
        ("final norm and output head", _int(head)),
        ("all parameters", _int(total)),
    ]
    return (
        f"The configuration of the convergence test: vocabulary {v} (6 symbols and the start marker), source and target length {c['src']}, "
        f"{c['n_enc']} encoder blocks and {c['n_dec']} decoder blocks, width d = {d}, {c['heads']} heads, feed-forward width {ff}, batch {c['batch']}. "
        "Each weight matrix has the shape of its entry in the crate's parameter list (a matrix of [rows, columns] and its bias). "
        "Every normalisation layer is a gain and a bias of d numbers each (2 d), every linear layer has a bias except the output head, which has none, "
        "and the position tables are learned tables of one row per position.\n\n"
        + _table(["part", "parameters"], rows) + "\n\n"
        f"The cross-attention of one decoder block scores every target position against every source position, for every head and "
        f"every sequence of the batch: {c['batch']} x {c['heads']} x {c['dec']} x {c['src']} = {_int(scores_per_layer)} scores per block "
        f"({_int(scores_per_layer * c['n_dec'])} for the {c['n_dec']} blocks), and the number of weights per sequence is the "
        f"{c['dec']} by {c['src']} alignment matrix of the chapter, one for each of the {c['heads']} heads."
    )
