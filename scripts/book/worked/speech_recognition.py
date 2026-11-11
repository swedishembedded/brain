# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Speech Recognition": how a model turns speech into text when it does not know where each word
starts.

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: speech_recognition.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree. Earlier modules are imported, never copied: the edit-distance table of `ocr` and the receptive-field
recursion of `convolution`.

* how many alignments of a transcript with a run of frames exist, counted exactly;
* connectionist temporal classification on a vocabulary of two letters and a blank: every path, the forward lattice, the soft
  alignment and the gradient, the exact probability of every label sequence and what greedy and beam decoding make of it;
* the transducer on the same scale: the lattice of frames and labels, its forward and backward sums and the greedy walk;
* word error rate with its alignment, and the corpus rate that weighs a long utterance by its words;
* the geometry of the two recognisers: what three strided convolutions do to 100 frames a second, the windows of the Qwen3-ASR
  encoder, the chunked mask of the Nemotron encoder and how far each lets a row see ahead, the 12.5 Hz bookkeeping and the state a
  stream carries;
* the parameters and the work of both recognisers, counted from the shapes of their configurations;
* a toy of chunked attention, with the streamed and the whole-utterance computation compared.

Arithmetic is the language's 64-bit floating point, a sum of products running left to right in the order the text states. The only
random numbers are those of the attention toy, from the book's SplitMix64 (`bookrng.generator(name)`), whose name and draw order the
chapter prints. Nothing is trained, so no number is chaotic and every digit printed is reproducible. FLOP counts use 2 per
multiply-add and leave out normalisations, activations, softmaxes and biases; each table says so.

Swedish Embedded AB implements solutions for speech-to-text systems whose alignment, latency and memory are counted before they are
deployed. If your team needs expertise in speech recognition engineering, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import itertools
import math

import bookrng
import convolution as cv
import ocr

# ------------------------------------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _int(v):
    return f"{v:,}"


def _f(v, d=4):
    return f"{v:.{d}f}"


def _mega(v, d=1):
    return f"{v / 1e6:,.{d}f}"


def _giga(v, d=2):
    return f"{v / 1e9:,.{d}f}"


def _sci(n, digits=3):
    """An integer of any size as d.dd x 10^k, from its exact decimal digits."""
    s = str(n)
    if len(s) <= 12:
        return f"{n:,}"
    return f"{s[0]}.{s[1:digits]} x 10^{len(s) - 1}"


# ------------------------------------------------------------------------------------------------ configurations (from the crates)

HOP = 160
RATE = 16000

# Qwen3-ASR audio encoder (crates/qwen3asr/src/config.rs)
Q_MEL, Q_D, Q_HEADS, Q_FFN, Q_LAYERS = 128, 1024, 16, 4096, 24
Q_CONV, Q_OUT, Q_NWIN, Q_NINFER = 480, 2048, 50, 800
Q_CHUNK = 2 * Q_NWIN  # mel frames per conv chunk
# Qwen3-1.7B decoder behind it
D_LAYERS, D_H, D_HEADS, D_KV, D_HD, D_FFN, D_VOCAB = 28, 2048, 16, 8, 128, 6144, 151936
PROMPT_PREFIX, PROMPT_SUFFIX = 10, 6

# Nemotron 3.5 ASR Streaming 0.6B (crates/nemotronasr/src/config.rs)
N_MEL, N_D, N_LAYERS, N_HEADS, N_FFN, N_K = 128, 1024, 24, 8, 4096, 9
N_SUBC, N_SUBK, N_SUBS = 256, 3, 2
N_WINDOW, N_LOOK = 57, 3
N_DEC, N_LSTM, N_VOCAB, N_BLANK, N_MAXSYM = 640, 2, 13088, 13087, 10
N_PROMPTS, N_PROMPT_MID = 128, 2048


# ------------------------------------------------------------------------------------------------ alignments counted


def _collapse(path, blank=0):
    out, prev = [], None
    for k in path:
        if k != prev and k != blank:
            out.append(k)
        prev = k
    return tuple(out)


def _ext(labels, blank=0):
    e = [blank]
    for l in labels:
        e += [l, blank]
    return e


def _skip_ok(e, s):
    return s >= 2 and e[s] != 0 and e[s] != e[s - 2]


def ctc_path_count(labels, frames):
    """Number of length-`frames` sequences over blank and the labels that collapse to `labels`: paths of the lattice, each with weight 1."""
    e = _ext(labels)
    n = [0] * len(e)
    n[0] = 1
    if len(e) > 1:
        n[1] = 1
    for _ in range(1, frames):
        m = [0] * len(e)
        for s in range(len(e)):
            m[s] = n[s] + (n[s - 1] if s >= 1 else 0) + (n[s - 2] if _skip_ok(e, s) else 0)
        n = m
    return n[-1] + (n[-2] if len(e) > 1 else 0)


def _ctc_count_brute(labels, frames, vocab):
    return sum(1 for p in itertools.product(range(vocab), repeat=frames) if _collapse(p) == tuple(labels))


def alignment_counts():
    text = "the cat sat"
    labels = [ord(c) for c in text]  # 11 characters, no two neighbours equal
    u = len(labels)
    assert all(labels[i] != labels[i + 1] for i in range(u - 1))
    # check the lattice count against brute force on a case small enough to enumerate
    small = (1, 2, 1)
    assert ctc_path_count(small, 6) == _ctc_count_brute(small, 6, 3)
    rows = []
    for label, frames in (("1.2 seconds at 100 frames a second", 120), ("1.2 seconds at 12.5 rows a second", 15)):
        rows.append((label, frames, _sci(math.comb(frames - 1, u - 1)), _sci(ctc_path_count(labels, frames))))
    return (f"The transcript `{text}` has U = {u} characters (letters and spaces; no two neighbours are equal). Column 3: the ways to give each "
            "character one or more consecutive frames and leave no frame without a character, C(T - 1, U - 1). Column 4: the sequences of "
            "T symbols over a blank and the 27 characters that remove to the transcript after merging repeats and deleting blanks, counted "
            "on the lattice of the chapter with every path weighing 1 (checked against enumeration of all 3^6 = 729 sequences for the "
            "label `aba` over a blank and two letters in 6 frames).\n\n"
            + _table(["clip", "frames T", "no blanks, every frame labelled", "with blanks"], rows))


def ctc_minimum_frames():
    rows = []
    for name, labels in (("a", (1,)), ("ab", (1, 2)), ("aa", (1, 1)), ("aab", (1, 1, 2)), ("abba", (1, 2, 2, 1))):
        reps = sum(1 for i in range(len(labels) - 1) if labels[i] == labels[i + 1])
        need = len(labels) + reps
        counts = [ctc_path_count(labels, t) for t in (need - 1, need, need + 1, need + 2)]
        rows.append((f"`{name}`", len(labels), reps, need, counts[0], counts[1], counts[2], counts[3]))
    return ("Fewest frames in which a label sequence can be written, U plus the number of neighbouring equal labels (each pair needs a blank "
            "between), and the number of alignments in one frame fewer, in exactly that many, and in one and two more.\n\n"
            + _table(["label", "U", "equal neighbours", "fewest frames", "alignments, one fewer", "at the minimum", "one more", "two more"],
                     rows))


# ------------------------------------------------------------------------------------------------ CTC on a tiny example

CTC_P = [[0.25, 0.60, 0.15], [0.15, 0.45, 0.40], [0.30, 0.50, 0.20], [0.15, 0.50, 0.35]]  # frames x (blank, a, b)
SYM = "-ab"


def _path_str(p):
    return "".join(SYM[k] for k in p)


def _path_prob(p, P=CTC_P):
    pr = 1.0
    for t, k in enumerate(p):
        pr *= P[t][k]
    return pr


def _label_probs(P=CTC_P):
    d = {}
    for p in itertools.product(range(3), repeat=len(P)):
        d[_collapse(p)] = d.get(_collapse(p), 0.0) + _path_prob(p, P)
    return d


def ctc_paths():
    target = (1, 2)
    paths = [p for p in itertools.product(range(3), repeat=4) if _collapse(p) == target]
    rows = [(f"`{_path_str(p)}`", *[_f(CTC_P[t][k], 2) for t, k in enumerate(p)], _f(_path_prob(p), 8)) for p in paths]
    total = 0.0
    for p in paths:
        total += _path_prob(p)
    grid = [(f"{t}", *[_f(x, 2) for x in CTC_P[t]]) for t in range(4)]
    return ("Per-frame probabilities over blank `-`, `a` and `b` for T = 4 frames (each row sums to 1):\n\n"
            + _table(["frame t", "`-`", "`a`", "`b`"], grid)
            + f"\n\nAll 3^4 = 81 paths, of which {len(paths)} collapse to `ab`. A path's probability is the product of its four entries, "
            "multiplied in frame order.\n\n"
            + _table(["path", "t=0", "t=1", "t=2", "t=3", "product"], rows)
            + f"\n\nSum of the {len(paths)} products: {_f(total, 8)}, the probability of the label `ab`, so a loss of -ln {_f(total, 8)} = "
            f"{_f(-math.log(total), 4)}.")


def _ctc_lattice(P, labels):
    """(alpha, beta) over frames x extended states, in probabilities (no logs: the example is tiny)."""
    e = _ext(labels)
    T, S = len(P), len(e)
    alpha = [[0.0] * S for _ in range(T)]
    alpha[0][0] = P[0][e[0]]
    if S > 1:
        alpha[0][1] = P[0][e[1]]
    for t in range(1, T):
        for s in range(S):
            a = alpha[t - 1][s]
            if s >= 1:
                a += alpha[t - 1][s - 1]
            if _skip_ok(e, s):
                a += alpha[t - 1][s - 2]
            alpha[t][s] = a * P[t][e[s]]
    beta = [[0.0] * S for _ in range(T)]
    beta[T - 1][S - 1] = P[T - 1][e[S - 1]]
    if S > 1:
        beta[T - 1][S - 2] = P[T - 1][e[S - 2]]
    for t in range(T - 2, -1, -1):
        for s in range(S):
            b = beta[t + 1][s]
            if s + 1 < S:
                b += beta[t + 1][s + 1]
            if s + 2 < S and _skip_ok(e, s + 2):
                b += beta[t + 1][s + 2]
            beta[t][s] = b * P[t][e[s]]
    return alpha, beta


def ctc_lattice_data():
    alpha, beta = _ctc_lattice(CTC_P, (1, 2))
    e = _ext((1, 2))
    total = alpha[-1][-1] + alpha[-1][-2]
    return e, alpha, beta, total


def ctc_lattice():
    e, alpha, beta, total = ctc_lattice_data()
    _, _, _, _ = e, alpha, beta, total
    paths = sum(_path_prob(p) for p in itertools.product(range(3), repeat=4) if _collapse(p) == (1, 2))
    rows = []
    for s, sym in enumerate(e):
        rows.append((f"{s}", f"`{SYM[sym]}`", *[_f(alpha[t][s], 8) for t in range(4)]))
    # beta check: the same total from the other end
    beta_total = beta[0][0] + beta[0][1]
    assert abs(total - paths) < 1e-15 and abs(total - beta_total) < 1e-15
    return ("Forward lattice for the label `ab`, extended to `-a-b-` (states 0 to 4). alpha[t][s] is the summed probability of the paths that are in state s "
            "after frame t. Frame 0 can only be in state 0 or state 1, so alpha[0][0] = P[0][`-`], alpha[0][1] = P[0][`a`] and the other states are 0. For t >= 1, "
            "alpha[t][s] = (alpha[t-1][s] + alpha[t-1][s-1] + alpha[t-1][s-2] if state s is a letter that differs from the letter of state s-2, which here is state 3 only) "
            "times P[t][symbol of s], the terms that do not exist being 0.\n\n"
            + _table(["state s", "symbol", "t=0", "t=1", "t=2", "t=3"], rows)
            + f"\n\nP(`ab`) = alpha[3][4] + alpha[3][3] = {_f(alpha[3][4], 8)} + {_f(alpha[3][3], 8)} = {_f(total, 8)}. The sum over the paths "
            f"of the previous table is {_f(paths, 8)}; the backward lattice (beta) gives beta[0][0] + beta[0][1] = {_f(beta_total, 8)}. "
            "The three agree to within 1e-15.")


def _ctc_soft_alignment(P, labels):
    alpha, beta = _ctc_lattice(P, labels)
    e = _ext(labels)
    T = len(P)
    total = alpha[-1][-1] + alpha[-1][-2]
    occ = []
    for t in range(T):
        row = [0.0] * len(P[0])
        for s, sym in enumerate(e):
            # alpha counts the frame-t emission and so does beta: divide the double count once
            row[sym] += alpha[t][s] * beta[t][s] / P[t][sym]
        occ.append([x / total for x in row])
    return occ


def _ctc_loss_logits(z, labels):
    """-log P(labels) for logits z (frames x classes), softmax per frame."""
    P = []
    for row in z:
        m = max(row)
        ex = [math.exp(x - m) for x in row]
        s = sum(ex)
        P.append([x / s for x in ex])
    a, _ = _ctc_lattice(P, labels)
    return -math.log(a[-1][-1] + a[-1][-2])


def ctc_gradient():
    labels = (1, 2)
    soft = _ctc_soft_alignment(CTC_P, labels)
    grad = [[CTC_P[t][k] - soft[t][k] for k in range(3)] for t in range(4)]
    z = [[math.log(p) for p in row] for row in CTC_P]
    h = 1e-6
    worst = 0.0
    for t in range(4):
        for k in range(3):
            zp = [r[:] for r in z]
            zm = [r[:] for r in z]
            zp[t][k] += h
            zm[t][k] -= h
            fd = (_ctc_loss_logits(zp, labels) - _ctc_loss_logits(zm, labels)) / (2 * h)
            worst = max(worst, abs(fd - grad[t][k]))
    assert worst < 1e-8 and all(abs(sum(r) - 1) < 1e-15 for r in soft) and all(abs(sum(r)) < 1e-15 for r in grad)
    rows = [(f"{t}", *[_f(soft[t][k], 4) for k in range(3)], *[f"{grad[t][k]:+.4f}" for k in range(3)]) for t in range(4)]
    return ("The soft alignment of the label `ab` in the example: gamma[t][k] = (sum over the states s that emit k of alpha[t][s] beta[t][s] / "
            "P[t][k]) / P(`ab`), the probability, among the paths that write `ab`, that frame t holds symbol k. The gradient of the loss "
            "-ln P(`ab`) with respect to the logits z[t][k] of frame t, taking the logits to be the logarithms of the table, is "
            "P[t][k] - gamma[t][k].\n\n"
            + _table(["frame t", "gamma `-`", "gamma `a`", "gamma `b`", "grad `-`", "grad `a`", "grad `b`"], rows)
            + "\n\nEvery row of gamma sums to 1 and every gradient row to 0, each to within 1e-15. "
            "Central differences of the loss with step 1e-6 on each of the 12 logits agree with these gradients to below 1e-8.")


def _prefix_beam(P, width):
    beams = {(): (1.0, 0.0)}  # prefix -> (probability of ending in blank, probability of ending in a letter)
    for row in P:
        nxt = {}

        def add(pre, pb, pnb):
            ob, onb = nxt.get(pre, (0.0, 0.0))
            nxt[pre] = (ob + pb, onb + pnb)

        for pre, (pb, pnb) in beams.items():
            tot = pb + pnb
            add(pre, tot * row[0], 0.0)
            for k in (1, 2):
                if pre and pre[-1] == k:
                    add(pre + (k,), 0.0, pb * row[k])
                    add(pre, 0.0, pnb * row[k])
                else:
                    add(pre + (k,), 0.0, tot * row[k])
        beams = dict(sorted(nxt.items(), key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0]))[:width])
    return {k: sum(v) for k, v in beams.items()}


def ctc_decoding():
    d = _label_probs()
    ranked = sorted(d.items(), key=lambda kv: -kv[1])
    total = 0.0
    for _, v in ranked:
        total += v
    best = [max(range(3), key=lambda k: CTC_P[t][k]) for t in range(4)]
    greedy = _collapse(best)
    rows = [(f"`{''.join(SYM[k] for k in lab) or '(empty)'}`", _f(p, 8)) for lab, p in ranked[:6]]
    beam_rows = []
    for w in (1, 2, 3, 4):
        r = _prefix_beam(CTC_P, w)
        top = max(r.items(), key=lambda kv: kv[1])
        beam_rows.append((w, f"`{''.join(SYM[k] for k in top[0])}`", _f(top[1], 8)))
    exact = max(d.items(), key=lambda kv: kv[1])
    return (f"The same four frames. The {len(d)} different label sequences that some path writes have probabilities that sum to {_f(total, 8)} "
            "(summed in order of decreasing probability). The six most probable:\n\n"
            + _table(["label", "probability"], rows)
            + f"\n\nGreedy decoding takes the largest entry of each frame, `{_path_str(best)}`, and collapses it to `{''.join(SYM[k] for k in greedy)}`, "
            f"whose probability is {_f(d[greedy], 8)}. The most probable label is `{''.join(SYM[k] for k in exact[0])}` with {_f(exact[1], 8)}. "
            "Prefix beam search keeps the `width` prefixes of highest total probability after each frame (a prefix's probability split into "
            "the part that ends in a blank and the part that ends in a letter, ties broken by the order of the prefixes as tuples of symbol ids); the probability it "
            "reports is the mass of the paths it did not discard, so it is a lower bound.\n\n"
            + _table(["beam width", "best prefix at the end", "mass kept for it"], beam_rows))


# ------------------------------------------------------------------------------------------------ the transducer on a tiny example

R_ENC = [[1.5, 1.0], [1.0, 0.0], [1.0, -0.5]]  # one row per frame (the encoder's output, width 2)
R_PRED = [[-1.0, 1.0], [1.5, 0.5], [0.5, 1.5]]  # predictor output after: start, a, b
R_HEAD = [[0.0, 0.0], [-2.0, 2.0], [1.0, -2.0]]  # joint head rows: blank, a, b
R_BIAS = [1.0, 0.0, 0.0]
R_LABEL = (1, 2)


def _joint(t, tok):
    """Probabilities over blank, a, b at frame t when the last symbol written was `tok` (0 = nothing yet): head(relu(enc + pred))."""
    h = [max(e + p, 0.0) for e, p in zip(R_ENC[t], R_PRED[tok])]
    z = [R_HEAD[k][0] * h[0] + R_HEAD[k][1] * h[1] + R_BIAS[k] for k in range(3)]
    m = max(z)
    ex = [math.exp(x - m) for x in z]
    s = ex[0] + ex[1] + ex[2]
    return [x / s for x in ex]


def _lab_tok(u):
    return 0 if u == 0 else R_LABEL[u - 1]


def _rnnt_alpha_beta():
    T, U = len(R_ENC), len(R_LABEL)
    pb = [[_joint(t, _lab_tok(u))[0] for u in range(U + 1)] for t in range(T)]
    pl = [[_joint(t, _lab_tok(u))[R_LABEL[u]] if u < U else 0.0 for u in range(U + 1)] for t in range(T)]
    alpha = [[0.0] * (U + 1) for _ in range(T)]
    alpha[0][0] = 1.0
    for t in range(T):
        for u in range(U + 1):
            if t == 0 and u == 0:
                continue
            a = 0.0
            if t > 0:
                a += alpha[t - 1][u] * pb[t - 1][u]
            if u > 0:
                a += alpha[t][u - 1] * pl[t][u - 1]
            alpha[t][u] = a
    beta = [[0.0] * (U + 1) for _ in range(T)]
    beta[T - 1][U] = pb[T - 1][U]
    for t in range(T - 1, -1, -1):
        for u in range(U, -1, -1):
            if t == T - 1 and u == U:
                continue
            b = 0.0
            if t < T - 1:
                b += pb[t][u] * beta[t + 1][u]
            if u < U:
                b += pl[t][u] * beta[t][u + 1]
            beta[t][u] = b
    return pb, pl, alpha, beta


def transducer_lattice_data():
    pb, pl, alpha, beta = _rnnt_alpha_beta()
    T, U = len(R_ENC), len(R_LABEL)
    return pb, pl, alpha, alpha[T - 1][U] * pb[T - 1][U], beta[0][0]


def transducer_paths():
    T, U = len(R_ENC), len(R_LABEL)
    pb, pl, alpha, total, beta0 = transducer_lattice_data()
    grid = []
    for t in range(T):
        for u in range(U + 1):
            p = _joint(t, _lab_tok(u))
            grid.append((t, u, "start" if u == 0 else SYM[R_LABEL[u - 1]], *[_f(x, 4) for x in p]))
    # every path: an interleaving of U labels and T blanks that ends in the last blank
    paths = []
    for emit_after in itertools.combinations(range(T + U - 1), U):
        moves, t, u, pr = [], 0, 0, 1.0
        for i in range(T + U):
            if i in emit_after:
                moves.append(SYM[R_LABEL[u]])
                pr *= pl[t][u]
                u += 1
            else:
                moves.append("-")
                pr *= pb[t][u]
                t += 1
        paths.append(("".join(moves), pr))
    total_paths = 0.0
    for _, pr in paths:
        total_paths += pr
    assert abs(total - total_paths) < 1e-15 and abs(total - beta0) < 1e-15
    rows = [(f"`{m}`", _f(pr, 6)) for m, pr in paths]
    arows = [(t, *[_f(alpha[t][u], 6) for u in range(U + 1)]) for t in range(T)]
    return ("The joint network of the example has width 2 and reads the encoder row of frame t and the predictor output after the last symbol "
            "written (nothing yet, `a`, `b`): the three probabilities are softmax(head relu(enc[t] + pred[last]) + bias).\n\n"
            f"enc = {R_ENC}, pred (rows: start, a, b) = {R_PRED}, head (rows: blank, a, b) = {R_HEAD}, bias = {R_BIAS}.\n\n"
            + _table(["frame t", "labels written u", "last symbol", "P(`-`)", "P(`a`)", "P(`b`)"], grid)
            + f"\n\nFor the label `ab` in T = {T} frames a path is an order of T = {T} blanks and U = {U} letters whose last move is the blank that "
            f"closes the last frame: C(T + U - 1, U) = {math.comb(T + U - 1, U)} paths.\n\n"
            + _table(["path (moves)", "probability"], rows)
            + f"\n\nTheir sum is {_f(total_paths, 6)}. The forward lattice, alpha[t][u] = alpha[t-1][u] P(`-` | t-1, u) + alpha[t][u-1] P(label u | t, u-1), "
            "with alpha[0][0] = 1:\n\n"
            + _table(["frame t", "u=0", "u=1", "u=2"], arows)
            + f"\n\nP(`ab`) = alpha[{T - 1}][{U}] P(`-` | {T - 1}, {U}) = {_f(total, 6)}; the backward sum from the start is {_f(beta0, 6)}. "
            f"All three agree to within 1e-15; the loss is {_f(-math.log(total), 4)}.")


def transducer_greedy():
    t, tok, out, rows, sym = 0, 0, [], [], 0
    T = len(R_ENC)
    while t < T:
        p = _joint(t, tok)
        k = max(range(3), key=lambda i: p[i])  # first index wins a tie; none occurs here
        forced = sym >= 10
        action = "next frame" if (k == 0 or forced) else f"write `{SYM[k]}`"
        rows.append((len(rows) + 1, t, "start" if tok == 0 else SYM[tok], *[_f(x, 4) for x in p], SYM[k], action))
        if k == 0 or forced:
            t += 1
            sym = 0
        else:
            out.append(k)
            tok = k
            sym += 1
    pb, pl, alpha, total, _ = transducer_lattice_data()
    path_p = 1.0
    t = u = 0
    for r in rows:
        if r[-2] == "-":
            path_p *= pb[t][u]
            t += 1
        else:
            path_p *= pl[t][u]
            u += 1
    return ("Greedy transducer decoding on the same joint network: at frame t take the largest of the three probabilities given the last symbol "
            "written; a blank moves to the next frame, a letter is written and the predictor is run on it, and after at most 10 letters on one frame "
            "the frame is left whatever the joint says (the guard of the code, here never reached).\n\n"
            + _table(["step", "frame t", "last symbol", "P(`-`)", "P(`a`)", "P(`b`)", "largest", "action"], rows)
            + f"\n\nOutput `{''.join(SYM[k] for k in out)}` after {len(rows)} joint evaluations (T + U = {T + len(out)}); the path it followed has "
            f"probability {_f(path_p, 6)} of the label's {_f(total, 6)}.")


def transducer_memory():
    rows = []
    for name, T, U in (("10 seconds, 40 tokens", 125, 40), ("30 seconds, 120 tokens", 375, 120)):
        ctc = T * N_VOCAB
        rnnt = T * (U + 1) * N_VOCAB
        macs = T * (U + 1) * N_DEC * N_VOCAB
        rows.append((name, T, U, _int(ctc), _int(rnnt), _mega(4 * ctc), _mega(4 * rnnt), _mega(3 * 4 * rnnt), _giga(macs, 1)))
    head = N_DEC * N_VOCAB
    return (f"Logits a loss must hold: a connectionist loss needs T x V values, a transducer T x (U + 1) x V, for V = {_int(N_VOCAB)} outputs (the blank is one of them) and "
            "T rows at 12.5 a second (f32, 4 bytes each; the column 'three transducer arrays' counts the three arrays of that size that the reference loss of brain keeps, "
            "logits, log-softmax and gradient); the last column is the joint head applied once to every cell of the lattice, "
            "T x (U + 1) x 640 x 13,088 multiply-adds.\n\n"
            + _table(["utterance", "T", "U", "CTC values", "transducer values", "CTC MB", "transducer MB", "three transducer arrays, MB", "head multiply-adds, billions"], rows)
            + f"\n\nOne joint evaluation at decoding time reads the head, {N_DEC} x {_int(N_VOCAB)} = {_int(head)} weights, {_mega(4 * head)} MB "
            f"in f32, for {_mega(2 * head)} MFLOP: {2 * head / (4 * head):.1f} FLOP per byte.")


# ------------------------------------------------------------------------------------------------ word error rate


def _wer_ops(ref, hyp):
    d = ocr.edit_table(ref, hyp)
    i, j, ops = len(ref), len(hyp), []
    while i > 0 or j > 0:
        # preference when several steps are optimal: match or substitute, then delete, then insert
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], "-"))
            i -= 1
        else:
            ops.append(("ins", "-", hyp[j - 1]))
            j -= 1
    return d[len(ref)][len(hyp)], ops[::-1]


def wer_alignment():
    ref = "turn the lights off in the kitchen".split()
    hyp = "turn the light off the kitchen please".split()
    dist, ops = _wer_ops(ref, hyp)
    s = sum(o[0] == "sub" for o in ops)
    dl = sum(o[0] == "del" for o in ops)
    n = sum(o[0] == "ins" for o in ops)
    rows = [(i, o[1], o[2], o[0]) for i, o in enumerate(ops)]
    return (f"Reference: {' '.join(ref)} ({len(ref)} words). Hypothesis: {' '.join(hyp)}. The table of prefixes is filled as in Volume VII, Chapter 10; "
            "the alignment is read back from the last cell, preferring a match or substitution, then a deletion, then an insertion when several "
            "steps are optimal.\n\n"
            + _table(["position", "reference", "hypothesis", "operation"], rows, "rlll")
            + f"\n\nSubstitutions S = {s}, deletions D = {dl}, insertions I = {n}: the edit distance is {dist} = S + D + I, and the word error rate "
            f"(S + D + I) / N = {dist} / {len(ref)} = {_f(dist / len(ref), 4)}.")


def wer_corpus():
    pairs = [("a b", "a"), ("a b c d e f g h", "a b c d e f g x")]
    rows, edits, words, rates = [], 0, 0, []
    for r, h in pairs:
        d, _ = _wer_ops(r.split(), h.split())
        rows.append((f"`{r}`", f"`{h}`", len(r.split()), d, _f(d / len(r.split()), 4)))
        edits += d
        words += len(r.split())
        rates.append(d / len(r.split()))
    mean = (rates[0] + rates[1]) / 2
    return ("Two utterances, scored one by one and together.\n\n"
            + _table(["reference", "hypothesis", "reference words", "edits", "rate"], rows, "llrrr")
            + f"\n\nThe mean of the two rates is {_f(mean, 4)}. The corpus rate, total edits over total reference words, is {edits} / {words} = "
            f"{_f(edits / words, 4)}: a long utterance weighs by its words.")


# ------------------------------------------------------------------------------------------------ subsampling


def _qwen_chunk_rows(frames):
    """Rows the Qwen3-ASR encoder keeps for `frames` valid mel frames: whole chunks of 100 frames, the last partial, each through three
    convolutions of kernel 3, stride 2, padding 1 (length l -> (l - 1) // 2 + 1)."""
    def after(n):
        for _ in range(3):
            n = (n - 1) // 2 + 1 if n > 0 else 0
        return n
    full, rest = divmod(frames, Q_CHUNK)
    return full * after(Q_CHUNK) + after(rest)


def _nem_stage(n):
    """One causal stride-2 stage of kernel 3: padding (k - 1, s - 1) = (2, 1) gives (n + 2 + 1 - 3) // 2 + 1."""
    return (n + (N_SUBK - 1) + (N_SUBS - 1) - N_SUBK) // N_SUBS + 1


def _nem_rows(frames):
    for _ in range(3):
        frames = _nem_stage(frames)
    return frames


def subsampling_stages():
    qrows = [("input", Q_MEL, Q_CHUNK, 1, _int(Q_MEL * Q_CHUNK))]
    f, t = Q_MEL, Q_CHUNK
    for i in range(3):
        f, t = (f - 1) // 2 + 1, (t - 1) // 2 + 1
        qrows.append((f"conv {i + 1}", f, t, Q_CONV, _int(Q_CONV * f * t)))
    nrows = [("input", N_MEL, 100, 1, _int(N_MEL * 100))]
    f, t = N_MEL, 100
    for i in range(3):
        f, t = _nem_stage(f), _nem_stage(t)
        nrows.append((f"stage {i + 1}", f, t, N_SUBC, _int(N_SUBC * f * t)))
    qflat = Q_CONV * qrows[-1][1]
    nflat = N_SUBC * nrows[-1][1]
    return ("Shapes after each of the three strided convolutions, for 100 valid mel frames (one second at the hop of 160 samples). Qwen3-ASR: kernel 3, "
            "stride 2, padding 1 on both axes, length n -> (n - 1) // 2 + 1, 480 channels, on a chunk of 100 frames. Nemotron: kernel 3, stride 2, padding "
            "(k - 1, s - 1) = (2, 1) at the start and the end of each axis, length n -> (n + 2 + 1 - 3) // 2 + 1 = n // 2 + 1, 256 channels.\n\n"
            "Qwen3-ASR\n\n" + _table(["stage", "frequency", "time", "channels", "values"], qrows)
            + f"\n\nThe rows handed to the linear map are {qrows[-1][2]} time positions of width 480 x {qrows[-1][1]} = {_int(qflat)} (to 1024).\n\n"
            "Nemotron\n\n" + _table(["stage", "frequency", "time", "channels", "values"], nrows)
            + f"\n\nThe rows handed to the linear map are {nrows[-1][2]} time positions of width 256 x {nrows[-1][1]} = {_int(nflat)} (to 1024).")


def subsampling_rows():
    rows = []
    for secs in (0.5, 1, 1.5, 2, 3, 10, 30, 35):
        n = int(secs * RATE)
        frames = n // HOP
        rows.append((secs, _int(n), frames, _qwen_chunk_rows(frames), _nem_rows(frames), f"{_nem_rows(frames) / secs:.2f}"))
    return ("Rows the encoder produces for a clip of L samples at 16 kHz, from the valid mel frames floor(L / 160): Qwen3-ASR cuts the frames into chunks of "
            "100 and takes 13 rows from a full chunk (7 from 50 frames); Nemotron applies n -> n // 2 + 1 three times to all the frames.\n\n"
            + _table(["seconds", "samples L", "valid frames", "Qwen3-ASR rows", "Nemotron rows", "Nemotron rows per second"], rows))


def receptive_fields():
    stack = [(3, 2, 1)] * 3
    rf = cv.receptive(stack)
    rows = []
    for i, (r, j) in enumerate(rf):
        rows.append((i + 1, r, j, _f(10.0 * r + 15.0, 0), _f(10.0 * j, 0)))
    last_r = rf[-1][0]
    span = (last_r - 1) * HOP + 400
    return ("Three convolutions of kernel 3 and stride 2 on mel frames 10 milliseconds apart (Volume VII, Chapter 2): receptive field r in frames and jump j between "
            "neighbouring outputs, then in milliseconds (a frame's window of 400 samples adds 15 milliseconds beyond the first and last centres).\n\n"
            + _table(["convolution", "field r (frames)", "jump j (frames)", "field (ms)", "jump (ms)"], rows)
            + f"\n\nThe field after the third is {last_r} frames, {span} samples = {span / 16:.1f} milliseconds of audio; the jump is {rf[-1][1]} frames = {rf[-1][1] * 10} milliseconds.")


# ------------------------------------------------------------------------------------------------ windows and masks


def _qwen_spans(rows):
    window = _qwen_chunk_rows(Q_CHUNK) * (Q_NINFER // Q_CHUNK)
    spans = [window] * (rows // window)
    if rows % window:
        spans.append(rows % window)
    return window, spans


def qwen_windows():
    out = []
    for secs in (2, 10, 30):
        frames = secs * 100
        rows = _qwen_chunk_rows(frames)
        window, spans = _qwen_spans(rows)
        blockpairs = sum(s * s for s in spans)
        out.append((secs, rows, window, "+".join(str(s) for s in spans) if len(spans) < 6 else f"{spans.count(window)} x {window} + {spans[-1]}", _int(blockpairs), _int(rows * rows),
                    f"{rows * rows / blockpairs:.2f}"))
    return (f"Attention windows of the Qwen3-ASR encoder: n_window_infer = {Q_NINFER} mel frames is {Q_NINFER // Q_CHUNK} chunks of 13 rows, a window of "
            f"{_qwen_spans(1000)[0]} rows, and every layer attends inside windows only, row by row, with no position beyond a window's edge visible. "
            "Query-key pairs of one layer per head, with the windows against one dense attention over all rows.\n\n"
            + _table(["clip (s)", "rows", "window", "spans", "pairs, windows", "pairs, dense", "dense / windows"], out))


def _chunk_allow(i, j, left_chunks, chunk):
    qc, kc = i // chunk, j // chunk
    return qc >= kc and qc - kc <= left_chunks


def chunk_mask_toy():
    T, chunk, left_chunks = 12, 4, 2
    rows = []
    for i in range(T):
        rows.append((i, *["x" if _chunk_allow(i, j, left_chunks, chunk) else "." for j in range(T)], sum(_chunk_allow(i, j, left_chunks, chunk) for j in range(T))))
    # the real model
    chunk_n, left_n = N_LOOK + 1, (N_WINDOW - 1) // (N_LOOK + 1)
    keys_steady = (left_n + 1) * chunk_n
    max_off = left_n * chunk_n + chunk_n - 1
    min_off = -(chunk_n - 1)
    # verify the offsets by enumeration over a long utterance
    seen = set()
    for i in range(400):
        for j in range(max(0, i - 80), min(400, i + 10)):
            if _chunk_allow(i, j, left_n, chunk_n):
                seen.add(i - j)
    assert (max(seen), min(seen), len(seen)) == (max_off, min_off, max_off - min_off + 1)
    return ("The chunked mask: query row i may read key row j when floor(i / chunk) - floor(j / chunk) lies between 0 and the number of left chunks. "
            "Twelve rows, chunk 4 (three rows of look-ahead), two left chunks; x marks an allowed pair.\n\n"
            + _table(["query i"] + [str(j) for j in range(T)] + ["keys"], rows, "r" + "c" * T + "r")
            + f"\n\nNemotron: sliding_window {N_WINDOW} gives 56 rows of left context, so left chunks = 56 // 4 = {left_n} and every query in steady state reads "
            f"({left_n} + 1) x {chunk_n} = {keys_steady} keys. The offset i - j runs from {min_off} (the first row of a chunk reads the last row of its own chunk) to "
            f"{max_off} = {left_n} x {chunk_n} + {chunk_n - 1} (the last row of a chunk reads back to the start of the oldest chunk): {max_off - min_off + 1} offsets, "
            "the 63 rows of the relative-position table of the streaming path.")


def _reach(T, rule, layers):
    """lo[i], hi[i]: the rows of the input that output row i of a stack of `layers` attention layers depends on (the dependency set of each row is an interval)."""
    lo = list(range(T))
    hi = list(range(T))
    for _ in range(layers):
        nlo, nhi = [], []
        for i in range(T):
            a, b = rule(i, T)
            nlo.append(lo[a])  # lo and hi never decrease with the row, so the ends of the window decide
            nhi.append(hi[b])
        lo, hi = nlo, nhi
    return lo, hi


def lookahead_data(max_layers=24, T=3000, probe=1700):
    """(layers, rows ahead for a window sliding 56 back and 3 ahead, rows ahead for the chunked mask, left reach of the chunked mask)."""
    chunk = N_LOOK + 1
    left_n = (N_WINDOW - 1) // chunk

    def sliding(i, T):
        return max(0, i - (N_WINDOW - 1)), min(T - 1, i + N_LOOK)

    def chunked(i, T):
        qc = i // chunk
        return max(0, (qc - left_n) * chunk), min(T - 1, (qc + 1) * chunk - 1)
    out = []
    for layers in range(1, max_layers + 1):
        slo, shi = _reach(T, sliding, layers)
        clo, chi = _reach(T, chunked, layers)
        # worst case over the rows of one chunk, in the interior of the utterance
        ahead_c = max(chi[i] - i for i in range(probe, probe + chunk))
        last = probe + chunk - 1
        out.append((layers, shi[probe] - probe, ahead_c, last - clo[last]))
    return out


def lookahead():
    data = lookahead_data()
    rows = []
    for layers, s, c, left in data:
        if layers in (1, 2, 4, 6, 12, 24):
            rows.append((layers, s, f"{80 * s:,}", c, f"{80 * c:,}", _int(left), f"{left * 0.08:,.1f}"))
    return ("How far ahead of a row the output of a stack of attention layers can depend on, from the dependency sets of the masks composed layer by layer "
            "(rows are 80 milliseconds apart; 3,000 rows, measured in the interior at row 1,700): a window that slides, 56 rows back and 3 ahead for every row, against the chunked mask "
            "(chunks of 4, 14 chunks back). For the chunked mask the worst row of a chunk is reported.\n\n"
            + _table(["layers", "sliding: rows ahead", "sliding: ms ahead", "chunked: rows ahead", "chunked: ms ahead", "chunked: rows back", "chunked: seconds back"], rows))


# ------------------------------------------------------------------------------------------------ the chunked-attention toy


def _softmax(z):
    m = max(z)
    ex = [math.exp(x - m) for x in z]
    s = 0.0
    for x in ex:
        s += x
    return [x / s for x in ex]


def _attend(x, wq, wk, wv, allowed, d):
    """One head of attention with a residual: y[i] = x[i] + sum_j softmax_j(q_i . k_j / sqrt(d)) v_j over the allowed j, dot products summed left to right."""
    def mat(v, w):
        return [sum_ordered(v[a] * w[a][b] for a in range(d)) for b in range(d)]
    q = [mat(r, wq) for r in x]
    k = [mat(r, wk) for r in x]
    v = [mat(r, wv) for r in x]
    y = []
    for i in range(len(x)):
        js = [j for j in range(len(x)) if allowed(i, j)]
        sc = [sum_ordered(q[i][a] * k[j][a] for a in range(d)) / math.sqrt(d) for j in js]
        p = _softmax(sc)
        row = []
        for b in range(d):
            acc = 0.0
            for pj, j in zip(p, js):
                acc += pj * v[j][b]
            row.append(x[i][b] + acc)
        y.append(row)
    return y


def sum_ordered(it):
    acc = 0.0
    for v in it:
        acc += v
    return acc


TOY_T, TOY_D, TOY_CHUNK, TOY_LEFT = 16, 4, 4, 2
TOY_NAME = "speech_recognition.chunk_attention"


def _toy_inputs():
    g = bookrng.generator(TOY_NAME)
    x = [[g.normal() for _ in range(TOY_D)] for _ in range(TOY_T)]
    ws = []
    for _ in range(6):
        ws.append([[0.5 * g.normal() for _ in range(TOY_D)] for _ in range(TOY_D)])
    return x, ws


def _two_layers(x, ws, allowed):
    h = _attend(x, ws[0], ws[1], ws[2], allowed, TOY_D)
    return _attend(h, ws[3], ws[4], ws[5], allowed, TOY_D)


def _streamed(x, ws):
    """The two layers computed chunk by chunk with a cache of at most TOY_LEFT chunks of the layer's own input rows: no row of the future chunks is ever read."""
    d, c = TOY_D, TOY_CHUNK
    layers = [(ws[0], ws[1], ws[2]), (ws[3], ws[4], ws[5])]
    caches = [[], []]  # per layer: list of (global row index, key, value)
    out = []
    for q0 in range(0, TOY_T, c):
        cur = x[q0:q0 + c]
        for li, (wq, wk, wv) in enumerate(layers):
            def mat(v, w):
                return [sum_ordered(v[a] * w[a][b] for a in range(d)) for b in range(d)]
            ks = [mat(r, wk) for r in cur]
            vs = [mat(r, wv) for r in cur]
            qs = [mat(r, wq) for r in cur]
            caches[li] += [(q0 + n, ks[n], vs[n]) for n in range(c)]
            caches[li] = [e for e in caches[li] if e[0] >= q0 - TOY_LEFT * c]
            nxt = []
            for n in range(c):
                sc = [sum_ordered(qs[n][a] * e[1][a] for a in range(d)) / math.sqrt(d) for e in caches[li]]
                p = _softmax(sc)
                row = []
                for b in range(d):
                    acc = 0.0
                    for pj, e in zip(p, caches[li]):
                        acc += pj * e[2][b]
                    row.append(cur[n][b] + acc)
                nxt.append(row)
            cur = nxt
            caches[li] = [e for e in caches[li] if e[0] >= q0 + c - TOY_LEFT * c]  # what the next chunk can still read
        out += cur
    return out, [len(cc) for cc in caches]


def chunk_attention_toy():
    x, ws = _toy_inputs()

    def chunked(i, j):
        return _chunk_allow(i, j, TOY_LEFT, TOY_CHUNK)

    whole = _two_layers(x, ws, chunked)
    streamed, kept = _streamed(x, ws)
    worst = max(abs(a - b) for ra, rb in zip(whole, streamed) for a, b in zip(ra, rb))
    # perturb row 9 of the input and see which output rows move
    x2 = [r[:] for r in x]
    x2[9][0] += 1.0
    moved = []
    for name, rule in (("chunked mask, chunk 4", chunked), ("sliding window, 8 back and 3 ahead", lambda i, j: i - 8 <= j <= i + 3)):
        a = _two_layers(x, ws, rule)
        b = _two_layers(x2, ws, rule)
        rows = [i for i in range(TOY_T) if max(abs(p - q) for p, q in zip(a[i], b[i])) > 1e-12]
        moved.append((name, ", ".join(map(str, rows))))
    return (f"A two-layer toy of the streaming identity: {TOY_T} rows of width {TOY_D}, one head per layer, the residual form y = x + attention(x) (no normalisation, "
            "no relative positions, queries, keys and values from the layer's input), scores q . k / 2 over the allowed keys, softmax with the maximum subtracted, "
            "sums left to right. Draws from `bookrng.generator(\"speech_recognition.chunk_attention\")` "
            f"(seed {bookrng.seed_of(TOY_NAME)}): first the {TOY_T * TOY_D} values of x as normals, row by row, then six {TOY_D} x {TOY_D} matrices "
            "(Wq, Wk, Wv of layer 1, then of layer 2), each row by row and each value 0.5 times a normal. Chunk 4, two chunks of left context.\n\n"
            f"Streamed chunk by chunk with a cache of at most {TOY_LEFT} chunks of each layer's own keys and values, against the whole-sequence masked computation: "
            f"largest difference of any output value {'below 1e-12' if worst < 1e-12 else f'{worst:.3e}'}; cache rows held at the end: {kept[0]} for each layer "
            f"(two chunks) of the {TOY_T} rows.\n\n"
            "Add 1.0 to the first value of input row 9 (the second row of chunk 2). Output rows of the two-layer stack that change by more than 1e-12:\n\n"
            + _table(["mask", "output rows that change"], moved, "ll"))


# ------------------------------------------------------------------------------------------------ parameters


def qwen_counts():
    d = Q_D
    attn = 4 * (d * d + d)
    mlp = d * Q_FFN + Q_FFN + Q_FFN * d + d
    norms = 2 * 2 * d
    block = attn + mlp + norms
    conv = [Q_CONV * 1 * 9 + Q_CONV, Q_CONV * Q_CONV * 9 + Q_CONV, Q_CONV * Q_CONV * 9 + Q_CONV]
    f = Q_MEL
    for _ in range(3):
        f = (f - 1) // 2 + 1
    conv_out = d * Q_CONV * f
    tail = 2 * d + (d * d + d) + (Q_OUT * d + Q_OUT)
    audio = sum(conv) + conv_out + Q_LAYERS * block + tail
    # the decoder
    kvw = D_KV * D_HD
    dec_layer = D_H * D_H * 2 + D_H * kvw * 2 + 3 * D_H * D_FFN + 2 * D_H + 2 * D_HD
    dec = D_LAYERS * dec_layer + D_VOCAB * D_H + D_H
    return {"conv": sum(conv), "conv_out": conv_out, "block": block, "attn": attn, "mlp": mlp, "norms": norms, "tail": tail, "audio": audio,
            "dec_layer": dec_layer, "embed": D_VOCAB * D_H, "dec": dec, "total": audio + dec, "kvw": kvw, "flat": Q_CONV * f}


def nemotron_counts():
    c = N_D
    ff = 2 * (c * N_FFN + N_FFN * c)
    attn = 5 * c * c + 2 * c
    conv = c * 2 * c + c * N_K + 2 * c + c * c
    ln = 5 * 2 * c
    block = ff + attn + conv + ln
    f = N_MEL
    for _ in range(3):
        f = _nem_stage(f)
    sub = (N_SUBC * 9 + N_SUBC) + 2 * (N_SUBC * 9 + N_SUBC + N_SUBC * N_SUBC + N_SUBC) + (N_SUBC * f * c + c)
    prompt = (c + N_PROMPTS) * N_PROMPT_MID + N_PROMPT_MID + N_PROMPT_MID * c + c
    proj = c * N_DEC + N_DEC
    enc = sub + N_LAYERS * block + prompt + proj
    emb = N_VOCAB * N_DEC
    lstm = N_LSTM * (2 * 4 * N_DEC * N_DEC + 2 * 4 * N_DEC)
    dproj = N_DEC * N_DEC + N_DEC
    head = N_DEC * N_VOCAB + N_VOCAB
    dec = emb + lstm + dproj + head
    return {"ff": ff, "attn": attn, "conv": conv, "ln": ln, "block": block, "sub": sub, "prompt": prompt, "proj": proj, "enc": enc, "emb": emb,
            "lstm": lstm, "dproj": dproj, "head": head, "dec": dec, "total": enc + dec, "flat": N_SUBC * f}


def parameters():
    q, n = qwen_counts(), nemotron_counts()
    qrows = [("three strided convolutions (480 channels)", _int(q["conv"])), (f"conv_out, {q['flat']} -> 1024, no bias", _int(q["conv_out"])),
             (f"{Q_LAYERS} transformer blocks of {_int(q['block'])}", _int(Q_LAYERS * q["block"])), ("final norm, projector (1024 -> 1024 -> 2048)", _int(q["tail"])),
             ("audio tower", _int(q["audio"])),
             (f"decoder: {D_LAYERS} layers of {_int(q['dec_layer'])}", _int(D_LAYERS * q["dec_layer"])), ("decoder: embedding (also the output head)", _int(q["embed"])),
             ("decoder: final norm", _int(D_H)), ("Qwen3-ASR in all", _int(q["total"]))]
    nrows = [("subsampling (3 convolutions, linear 4352 -> 1024)", _int(n["sub"])), (f"{N_LAYERS} Conformer blocks of {_int(n['block'])}", _int(N_LAYERS * n["block"])),
             ("language-prompt projector", _int(n["prompt"])), ("encoder projector 1024 -> 640", _int(n["proj"])), ("encoder", _int(n["enc"])),
             ("predictor: embedding", _int(n["emb"])), (f"predictor: {N_LSTM} LSTM layers of width 640", _int(n["lstm"])),
             ("predictor: projector", _int(n["dproj"])), ("joint head 640 -> 13,088 with bias", _int(n["head"])), ("transducer", _int(n["dec"])),
             ("Nemotron in all", _int(n["total"]))]
    return ("Parameters counted from the configurations of the two recognisers (weights and biases that exist, from the shapes the loaders read).\n\n"
            "Qwen3-ASR\n\n" + _table(["part", "parameters"], qrows)
            + "\n\nNemotron 3.5 ASR Streaming\n\n" + _table(["part", "parameters"], nrows))


def block_counts():
    q, n = qwen_counts(), nemotron_counts()
    rows = [("Qwen3-ASR transformer block", _int(q["attn"]), "-", _int(q["mlp"]), _int(q["norms"]), _int(q["block"])),
            ("FastConformer block", _int(n["attn"]), _int(n["conv"]), _int(n["ff"]), _int(n["ln"]), _int(n["block"]))]
    return ("Parameters of one encoder block at width 1024, by part. The Conformer's feed-forward column is two half-step feed-forwards, each with two "
            "matrices of 1024 x 4096 and no bias; its attention has a fifth matrix for relative positions and two bias vectors of 1024 (u and v).\n\n"
            + _table(["block", "attention", "convolution module", "feed-forward", "normalisation", "block total"], rows))


# ------------------------------------------------------------------------------------------------ work


def qwen_encoder_macs(chunks, valid_rows):
    """Multiply-adds of the Qwen3-ASR audio tower for `chunks` chunks of 100 frames of which `valid_rows` rows are real."""
    f1, t1 = 64, 50
    f2, t2 = 32, 25
    f3, t3 = 16, 13
    c1 = f1 * t1 * Q_CONV * 1 * 9
    c2 = f2 * t2 * Q_CONV * Q_CONV * 9
    c3 = f3 * t3 * Q_CONV * Q_CONV * 9
    cout = t3 * (Q_CONV * f3) * Q_D
    stem = chunks * (c1 + c2 + c3 + cout)
    d = Q_D
    window, spans = _qwen_spans(valid_rows)
    dense = Q_LAYERS * valid_rows * (4 * d * d + 2 * d * Q_FFN)
    attn = Q_LAYERS * sum(s * s for s in spans) * 2 * d
    proj = valid_rows * (d * d + d * Q_OUT)
    return {"conv1": chunks * c1, "conv2": chunks * c2, "conv3": chunks * c3, "conv_out": chunks * cout, "stem": stem, "dense": dense, "attn": attn, "proj": proj,
            "total": stem + dense + attn + proj}


def qwen_decoder_macs(prompt, new):
    """Multiply-adds to prefill `prompt` tokens and then generate `new` tokens: the weights of every layer, the causal attention scores and the
    weighted sum of values (2 x position x 2048 per token and layer), and the head once per generated token."""
    per_token = D_LAYERS * (2 * D_H * D_H + 2 * D_H * D_KV * D_HD + 3 * D_H * D_FFN)
    attn = 0
    for p in range(1, prompt + new + 1):
        attn += D_LAYERS * p * 2 * D_HEADS * D_HD
    head = D_VOCAB * D_H
    return {"weights": (prompt + new) * per_token, "attn": attn, "head": new * head, "total": (prompt + new) * per_token + attn + new * head, "per_token": per_token}


def qwen_work():
    e = qwen_encoder_macs(1, 13)
    rows = [("conv 1 (1 -> 480)", _giga(2 * e["conv1"], 3)), ("conv 2 (480 -> 480)", _giga(2 * e["conv2"], 3)), ("conv 3 (480 -> 480)", _giga(2 * e["conv3"], 3)),
            ("conv_out (7680 -> 1024, on all 13 rows)", _giga(2 * e["conv_out"], 3)), ("24 blocks, dense parts of the 13 valid rows", _giga(2 * e["dense"], 3)),
            ("24 blocks, attention scores and weighted sums (13 keys)", _giga(2 * e["attn"], 3)), ("projector", _giga(2 * e["proj"], 3)),
            ("audio tower in all", _giga(2 * e["total"], 3))]
    d = qwen_decoder_macs(13, 0)
    rows.append(("decoder prefill of the 13 rows (weights only)", _giga(2 * 13 * d["per_token"], 3)))
    return ("GFLOP of Qwen3-ASR for one second of audio (one full chunk of 100 frames, 13 rows), counted as 2 per multiply-add from the shapes of the layers; "
            "normalisations, activations, softmaxes and biases are left out. The last line is what the decoder spends reading the same 13 rows as prompt tokens "
            "(the weight matrices of its 28 layers, without the attention over the cache and without the head).\n\n"
            + _table(["stage", "GFLOP"], rows))


def nemotron_row_macs():
    c = N_D
    f1, f2, f3 = 65, 33, 17
    sub = (4 * f1 * N_SUBC * 1 * 9) + (2 * f2 * N_SUBC * 9 + 2 * f2 * N_SUBC * N_SUBC) + (1 * f3 * N_SUBC * 9 + 1 * f3 * N_SUBC * N_SUBC) + f3 * N_SUBC * c
    keys = ((N_WINDOW - 1) // (N_LOOK + 1) + 1) * (N_LOOK + 1)
    ff = 2 * (2 * c * N_FFN)
    att_proj = 4 * c * c
    att_scores = 3 * keys * c
    conv = c * 2 * c + c * N_K + c * c
    layer = ff + att_proj + att_scores + conv
    proj = (c + N_PROMPTS) * N_PROMPT_MID + N_PROMPT_MID * c + c * N_DEC
    return {"sub": sub, "ff": ff, "att_proj": att_proj, "att_scores": att_scores, "conv": conv, "layer": layer, "layers": N_LAYERS * layer, "proj": proj,
            "row": sub + N_LAYERS * layer + proj, "keys": keys, "rel_table": 63 * c * c}


def nemotron_work():
    m = nemotron_row_macs()
    per_s = 12.5
    rows = [("subsampling (3 convolutions and the linear map)", _mega(2 * m["sub"]), _giga(2 * m["sub"] * per_s, 3)),
            (f"24 blocks: two feed-forwards", _mega(2 * N_LAYERS * m["ff"]), _giga(2 * N_LAYERS * m["ff"] * per_s, 3)),
            ("24 blocks: attention projections q, k, v, o", _mega(2 * N_LAYERS * m["att_proj"]), _giga(2 * N_LAYERS * m["att_proj"] * per_s, 3)),
            (f"24 blocks: scores, relative scores and weighted sum ({m['keys']} keys)", _mega(2 * N_LAYERS * m["att_scores"]), _giga(2 * N_LAYERS * m["att_scores"] * per_s, 3)),
            ("24 blocks: convolution module", _mega(2 * N_LAYERS * m["conv"]), _giga(2 * N_LAYERS * m["conv"] * per_s, 3)),
            ("prompt and encoder projectors", _mega(2 * m["proj"]), _giga(2 * m["proj"] * per_s, 3)),
            ("encoder in all", _mega(2 * m["row"]), _giga(2 * m["row"] * per_s, 3))]
    head = N_DEC * N_VOCAB
    lstm = N_LSTM * (2 * N_DEC * 4 * N_DEC) + N_DEC * N_DEC
    drows = [("joint head at every evaluation", _mega(2 * head, 2)), ("predictor step (2 LSTM layers and the projector), per letter written", _mega(2 * lstm, 2))]
    return ("MFLOP of Nemotron per encoder row (80 milliseconds of audio, steady state of the streaming path) and GFLOP per second at 12.5 rows a second, counted as 2 per "
            "multiply-add from the shapes (normalisations, activations, softmaxes and biases left out; the relative-position table, 63 rows through each layer's "
            f"matrix, is built once per layer: {_mega(2 * m['rel_table'])} MFLOP each).\n\n"
            + _table(["stage", "MFLOP per row", "GFLOP per second"], rows)
            + "\n\nThe transducer, per step:\n\n" + _table(["step", "MFLOP"], drows))


def cost_summary():
    q_enc = 2 * qwen_encoder_macs(1, 13)["total"]
    d = qwen_decoder_macs(0, 1)
    q_read = 2 * 13 * d["per_token"]
    q_tok = 2 * (d["per_token"] + D_VOCAB * D_H)
    m = nemotron_row_macs()
    n_enc = 2 * m["row"] * 12.5
    head = N_DEC * N_VOCAB
    lstm = N_LSTM * (2 * N_DEC * 4 * N_DEC) + N_DEC * N_DEC
    n_frames = 2 * head * 12.5
    n_tok = 2 * (head + lstm)
    rows = [("Qwen3-ASR", _giga(q_enc, 2), "reading 13 rows: " + _giga(q_read, 2), _giga(q_tok, 3)),
            ("Nemotron", _giga(n_enc, 2), "one joint step a row: " + _giga(n_frames, 2), _giga(n_tok, 3))]
    return ("Billions of floating-point operations (GFLOP) for one second of audio and for one written token, 2 per multiply-add, by the counting rule of the two "
            "tables above. Qwen3-ASR: the audio tower for 13 rows, the 28-layer decoder's weights applied to the same 13 rows as prompt tokens, and per generated token the "
            "28 layers' weights plus the 151,936-way head. Nemotron: the encoder at 12.5 rows a second, one joint evaluation for each row (the blank that moves on), and per "
            f"written token one more joint evaluation and one predictor step.\n\n"
            + _table(["recogniser", "encoder, per second of audio", "after the encoder, per second of audio", "per written token"], rows, "lrrr")
            + f"\n\nPer second of audio the two sum to {_giga(q_enc + q_read, 2)} and {_giga(n_enc + n_frames, 2)}, a ratio of {(q_enc + q_read) / (n_enc + n_frames):.1f}; "
            f"per written token the ratio is {q_tok / n_tok:.0f}.")


def fixed_window():
    rows = []
    for secs in (2, 10, 30):
        window_frames = 3000
        exact_chunks = math.ceil(secs * 100 / Q_CHUNK)
        exact_rows = _qwen_chunk_rows(secs * 100)
        win_rows = _qwen_chunk_rows(window_frames)
        en_exact = 2 * qwen_encoder_macs(exact_chunks, exact_rows)["total"]
        en_win = 2 * qwen_encoder_macs(30, win_rows)["total"]
        dec_exact = 2 * qwen_decoder_macs(PROMPT_PREFIX + exact_rows + PROMPT_SUFFIX, 0)["total"]
        dec_win = 2 * qwen_decoder_macs(PROMPT_PREFIX + win_rows + PROMPT_SUFFIX, 0)["total"]
        rows.append((secs, exact_rows, win_rows, _giga(en_exact, 1), _giga(en_win, 1), _giga(dec_exact, 1), _giga(dec_win, 1),
                     f"{(en_win + dec_win) / (en_exact + dec_exact):.2f}"))
    return ("A clip through the serving path of Qwen3-ASR, which assembles the decoder for a fixed window of 30 seconds (3,000 frames, 390 rows, a prompt of 10 + 390 + 6 "
            "tokens) and pads every clip to it, against a clip that is encoded at its own length (13 rows a second, a prompt of 10 + rows + 6). GFLOP of the "
            "encoder, and of the decoder's prefill (weights and attention scores of the prompt, no generated tokens), 2 per multiply-add.\n\n"
            + _table(["clip (s)", "rows at its own length", "rows in the window", "encoder, own length", "encoder, window", "prefill, own length", "prefill, window",
                      "window / own length, total"], rows))


def qwen_prompt():
    rows = []
    for rows_n in (26, 390):
        prompt = PROMPT_PREFIX + rows_n + PROMPT_SUFFIX
        kv_row = D_LAYERS * 2 * D_KV * D_HD * 4
        rows.append((rows_n, prompt, _int(kv_row), _int(prompt * kv_row), f"{prompt * kv_row / 1e6:,.1f}"))
    head_bytes = 4 * D_VOCAB * D_H
    return (f"The prompt of the Qwen3-ASR decoder is {PROMPT_PREFIX} fixed ids (the chat template up to the start-of-audio marker), one placeholder id per encoder row, then "
            f"{PROMPT_SUFFIX} fixed ids; the audio rows replace the placeholders' embeddings. Bytes of the key-value cache of the prompt in f32, 28 layers x 2 x 8 key-value "
            f"heads x 128 x 4 bytes per token.\n\n"
            + _table(["audio rows", "prompt tokens", "cache bytes per token", "cache bytes", "cache MB"], rows)
            + f"\n\nThe tied embedding and output head, {_int(D_VOCAB)} x {_int(D_H)} in f32, is {_int(head_bytes)} bytes ({head_bytes / 1e9:.2f} GB).")


def stream_state():
    chunk = N_LOOK + 1
    kv_rows = ((N_WINDOW - 1) // chunk) * chunk
    kv = N_LAYERS * 2 * kv_rows * N_D * 4
    glu = N_LAYERS * (N_K - 1) * N_D * 4
    rowsize = [1 * N_MEL, N_SUBC * 65, N_SUBC * 33]
    tails = sum((N_SUBK - 1) * r for r in rowsize) * 4
    pend = (chunk - 1) * N_D * 4
    total = kv + glu + tails + pend
    rows = [(f"attention band: {N_LAYERS} layers x (K and V) x {kv_rows} rows x 1024 x 4 bytes", _int(kv)),
            (f"causal convolution: {N_LAYERS} layers x {N_K - 1} rows x 1024 x 4 bytes", _int(glu)),
            ("subsampling: the last 2 input rows of each of 3 stages (128, 256 x 65 and 256 x 33 values a row)", _int(tails)),
            (f"rows waiting for a chunk to fill: at most {chunk - 1} x 1024 x 4 bytes", _int(pend)), ("total at a chunk boundary", _int(total))]
    return ("What one stream of the Nemotron recogniser carries between pushes, in f32 (the decoder's LSTM state, two layers of 2 x 640 values, is 10,240 bytes more, "
            "and the front end's last 512 samples a few kilobytes).\n\n" + _table(["part", "bytes"], rows)
            + f"\n\nThe serving path refuses a 257th open stream: 256 x {_int(total)} = {_int(256 * total)} bytes, {256 * total / 1e9:.2f} GB, bounds the state.")


def causal_conv_toy():
    k = 3
    x = [[1.0, 2.0], [3.0, 1.0], [2.0, 0.0], [0.0, 4.0], [1.0, 1.0], [2.0, 3.0]]
    w = [[0.5, 0.25, 0.25], [1.0, -1.0, 0.5]]  # per channel, taps for rows t-2, t-1, t

    def conv(rows, history):
        full = history + rows
        out = []
        for t in range(len(history), len(full)):
            vals = []
            for c in range(2):
                acc = 0.0
                for kk in range(k):
                    src = t - (k - 1) + kk
                    acc += (full[src][c] if src >= 0 else 0.0) * w[c][kk]
                vals.append(acc)
            out.append(vals)
        return out

    whole = conv(x, [])
    assert whole == conv(x[:3], []) + conv(x[3:], x[1:3])
    stitched = conv(x[:3], []) + conv(x[3:], x[1:3])  # the second piece starts with the last k - 1 = 2 rows of the first
    x2 = [r[:] for r in x]
    x2[4][0] += 10.0
    changed = conv(x2, [])
    moved = [t for t in range(6) if changed[t] != whole[t]]
    rows = [(t, f"({x[t][0]:g}, {x[t][1]:g})", f"({whole[t][0]:g}, {whole[t][1]:g})") for t in range(6)]
    return ("A causal depthwise convolution of kernel 3: output row t of channel c is the sum over taps kk = 0, 1, 2 of w[c][kk] x x[t - 2 + kk][c], with rows before "
            "the start read as zero. Taps for channel 0: (0.5, 0.25, 0.25); for channel 1: (1, -1, 0.5).\n\n"
            + _table(["row t", "input (c0, c1)", "output (c0, c1)"], rows)
            + f"\n\nComputed in two pieces, rows 0 to 2 and then rows 3 to 5 with the last 2 rows of the first piece carried over, the outputs are identical. "
            f"Adding 10 to channel 0 of row 4 changes the outputs at rows {', '.join(map(str, moved))} only: a row never reads the future.")


# ------------------------------------------------------------------------------------------------ figure data


def pipeline_rates():
    """The numbers of the pipeline figure: rows per second and widths at each stage of the two recognisers."""
    q = qwen_counts()
    return {"qwen_frames": RATE // HOP, "qwen_rows": 13, "qwen_width": Q_D, "qwen_out": Q_OUT, "qwen_layers": Q_LAYERS, "qwen_dec_layers": D_LAYERS,
            "nem_frames": RATE // HOP, "nem_rows": 12.5, "nem_width": N_D, "nem_layers": N_LAYERS, "nem_dec": N_DEC, "nem_vocab": N_VOCAB, "q_total": q["total"]}
