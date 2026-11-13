# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Positional Information": telling attention where each token is.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. Each public function without arguments returns a markdown string that the chapter embeds
between `<!-- worked: positions.<name> -->` markers; the `*_data` functions supply the numbers of the chapter's figures, so a
picture and the table beside it cannot disagree.

* the order test: attention on "dog bites man" and "man bites dog" gives the same vector to each word;
* a learned table of absolute position embeddings, the rows it has and the position it has no row for;
* the sinusoidal encoding: its table, the fixed linear map that shifts a position, and a dot product that depends on the offset;
* rotary embeddings at width 8: the frequencies, the rotated pairs, and the score of shifted pairs of positions;
* the two ways of pairing channels, and what mixing them does;
* the decay of the rotary score with distance, at two bases;
* relative position biases: T5-style buckets and the linear bias of ALiBi;
* positions beyond the trained length, position interpolation and the YaRN frequency table;
* multi-axis rotary embeddings: position triples of an image between two pieces of text, and the score they give;
* the work of the rotation against the work of the attention that follows it.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of products
are plain left-to-right additions, the order a C loop would use (never Python's `sum`, which may compensate). Library functions
are the correctly rounded-to-last-place `sin`, `cos`, `exp`, `log` and `pow` of the C library; rotation angles are products
`position * omega` in double precision.

Swedish Embedded AB implements solutions for long-context transformer models, from the position encoding to the kernels. If your
team needs expertise in rotary embeddings and context extension, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng

NAME_SENTENCE = "positions.sentence"
NAME_TABLE = "positions.table"
NAME_ROPE = "positions.rope"
NAME_MROPE = "positions.mrope"

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


def _vec(v, d=2):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


def _round1(x):
    """x rounded to one decimal, ties away from zero (the pseudocode's `round(x, 1)`)."""
    r = math.floor(abs(x) * 10 + 0.5) / 10
    return -r if x < 0 else r


def _below(x, limit):
    """'below 1e-9' when x is round-off noise under the limit, else the value: noise is not reproducible digit by digit."""
    return f"below {limit:.0e}".replace("e-0", "e-") if x < limit else f"{x:.3g}"


def _g(x, digits=4):
    return f"{x:.{digits}g}"


# ------------------------------------------------------------------------------------------------ small linear algebra


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


def matmul(A, B):
    cols = list(zip(*B))
    return [[_dot(row, col) for col in cols] for row in A]


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = _sum(e)
    return [v / s for v in e]


def attend(X, Wq, Wk, Wv):
    """Single-head attention without a mask on the rows of X: softmax(Q K^T / sqrt(d_k)) V, and the weights."""
    Q, K, V = matmul(X, Wq), matmul(X, Wk), matmul(X, Wv)
    dk = len(Q[0])
    P = [_softmax([_dot(q, k) / math.sqrt(dk) for k in K]) for q in Q]
    return matmul(P, V), P


# ------------------------------------------------------------------------------------------------ the order test

TOKENS = ("dog", "bites", "man")
ORDER_A = (0, 1, 2)       # dog bites man
ORDER_B = (2, 1, 0)       # man bites dog
D_MODEL, D_HEAD = 4, 2
POSITIONS_TRAINED = 4     # rows of the learned position table


def sentence():
    """The three embeddings and the three projections of the order test, in the draw order the chapter prints."""
    g = bookrng.generator(NAME_SENTENCE)
    X = [[_round1(g.normal()) for _ in range(D_MODEL)] for _ in TOKENS]
    W = {n: [[_round1(g.normal()) for _ in range(D_HEAD)] for _ in range(D_MODEL)] for n in ("Wq", "Wk", "Wv")}
    return X, W["Wq"], W["Wk"], W["Wv"]


def position_table():
    g = bookrng.generator(NAME_TABLE)
    return [[_round1(0.5 * g.normal()) for _ in range(D_MODEL)] for _ in range(POSITIONS_TRAINED)]


def _run(order, X, W, P=None):
    """Outputs of attention for the sentence in `order`; with a table P, the row of position a is added to the a-th input."""
    rows = []
    for a, tok in enumerate(order):
        x = list(X[tok])
        if P is not None:
            x = [u + w for u, w in zip(x, P[a])]
        rows.append(x)
    return attend(rows, *W)[0]


def sentence_setup():
    X, Wq, Wk, Wv = sentence()
    rows = [(i, TOKENS[i], _vec(X[i], 1)) for i in range(3)]
    wrows = [(f"row {r + 1}", _vec(Wq[r], 1), _vec(Wk[r], 1), _vec(Wv[r], 1)) for r in range(D_MODEL)]
    return (_seed_line(NAME_SENTENCE) + " Draws, in this order: the 12 entries of X (one row for each of the words dog, bites, man; "
            "four numbers each), then the 8 entries of W_Q, the 8 of W_K and the 8 of W_V (each 4 by 2, row by row), every "
            "one `round(normal(), 1)` (`round` to one decimal, ties away from zero). The embeddings are random stand-ins for "
            "learned ones; d_k = d_v = 2.\n\n"
            + _table(["row of X", "word", "embedding x (4 numbers)"], rows, "lll") + "\n\n"
            + _table(["", "W_Q (4 by 2)", "W_K (4 by 2)", "W_V (4 by 2)"], wrows, "llll"))


def order_test():
    X, Wq, Wk, Wv = sentence()
    W = (Wq, Wk, Wv)
    a, b = _run(ORDER_A, X, W), _run(ORDER_B, X, W)
    rows = []
    worst = 0.0
    for tok in range(3):
        pa, pb = ORDER_A.index(tok), ORDER_B.index(tok)
        worst = max(worst, max(abs(a[pa][c] - b[pb][c]) for c in range(D_HEAD)))
        rows.append((TOKENS[tok], pa + 1, _vec(a[pa], 3), pb + 1, _vec(b[pb], 3)))
    return ("Attention without a mask (the layer of Chapter 7) on the three embeddings of the table above, once in the order dog "
            "bites man and once in the order man bites dog.\n\n"
            + _table(["word", "place in dog bites man", "its output", "place in man bites dog", "its output"], rows, "lrlrl")
            + f"\n\nLargest absolute difference between the two outputs of any word: {_below(worst, 1e-12)}.")


def learned_positions():
    X, Wq, Wk, Wv = sentence()
    P = position_table()
    W = (Wq, Wk, Wv)
    a, b = _run(ORDER_A, X, W, P), _run(ORDER_B, X, W, P)
    rows = []
    worst = 0.0
    for tok in range(3):
        pa, pb = ORDER_A.index(tok), ORDER_B.index(tok)
        worst = max(worst, max(abs(a[pa][c] - b[pb][c]) for c in range(D_HEAD)))
        rows.append((TOKENS[tok], _vec(a[pa], 3), _vec(b[pb], 3)))
    prow = [(f"position {i}", _vec(P[i], 1)) for i in range(POSITIONS_TRAINED)]
    return (_seed_line(NAME_TABLE) + f" Draws: the {POSITIONS_TRAINED * D_MODEL} entries of the table W_pos (shape "
            f"[{POSITIONS_TRAINED}, {D_MODEL}]), row by row, each `round(0.5 * normal(), 1)`. The input at place a is the "
            "embedding of its word plus row a of the table; the projections are those above.\n\n"
            + _table(["row of W_pos", "values"], prow, "ll") + "\n\n"
            + _table(["word", "output in dog bites man", "output in man bites dog"], rows, "lll")
            + f"\n\nLargest absolute difference between the two outputs of any word: {_fx(worst, 3)}. The table has "
            f"{POSITIONS_TRAINED} rows, so a sequence of {POSITIONS_TRAINED + 1} tokens has a token at position "
            f"{POSITIONS_TRAINED} and no row to add to it. A table for 4,096 positions of width 1,024 holds "
            f"{_int(4096 * 1024)} numbers.")


# ------------------------------------------------------------------------------------------------ sinusoidal encoding

BASE = 10000.0


def omegas(d, base=BASE):
    """Rotation rate of pair m, base^(-2m/d), for m = 0 .. d/2 - 1 (library pow)."""
    return [base ** (-2 * m / d) for m in range(d // 2)]


def sinusoid(i, d, base=BASE):
    """Row of the sinusoidal encoding for position i: (sin(i w_0), cos(i w_0), sin(i w_1), cos(i w_1), ...)."""
    out = []
    for w in omegas(d, base):
        out += [math.sin(i * w), math.cos(i * w)]
    return out


def sinusoid_table():
    d = 8
    w = omegas(d)
    head = ["position i"] + [(["sin", "cos"][c % 2] + f" w{c // 2}") for c in range(d)]
    rows = [(i, *[_fx(v, 3) for v in sinusoid(i, d)]) for i in range(8)]
    return (f"Width D = 8, base 10000. The four rates are w_m = 10000^(-2m/8) (library pow): "
            + ", ".join(f"w{m} = {_g(w[m])}" for m in range(4))
            + ". Column 2m is sin(i w_m) and column 2m + 1 is cos(i w_m); three decimals.\n\n" + _table(head, rows))


def sinusoid_shift():
    d, k = 8, 3
    w = omegas(d)
    # the map of the chapter: for each pair, [[cos k w, sin k w], [-sin k w, cos k w]] applied to (sin i w, cos i w)
    blocks = [[[math.cos(k * x), math.sin(k * x)], [-math.sin(k * x), math.cos(k * x)]] for x in w]
    rows = []
    worst = 0.0
    for i in (0, 2, 10, 100):
        base_row = sinusoid(i, d)
        moved = []
        for m, B in enumerate(blocks):
            a, b = base_row[2 * m], base_row[2 * m + 1]
            moved += [B[0][0] * a + B[0][1] * b, B[1][0] * a + B[1][1] * b]
        target = sinusoid(i + k, d)
        diff = max(abs(u - v) for u, v in zip(moved, target))
        worst = max(worst, diff)
        dot = _dot(base_row, target)
        rows.append((i, i + k, _vec(target[:2], 4), _vec(moved[:2], 4), _fx(dot, 6)))
    dots = [_dot(sinusoid(i, d), sinusoid(i + k, d)) for i in (0, 2, 10, 100)]
    return (f"Width D = 8, base 10000, shift k = {k}. The map applied to each pair (sin(i w_m), cos(i w_m)) is the 2 by 2 matrix "
            f"with rows (cos(k w_m), sin(k w_m)) and (-sin(k w_m), cos(k w_m)); it does not mention i. First pair of the "
            "encoding at i + k, and the same pair obtained from position i by that map, and the dot product of the full "
            "rows at i and i + k.\n\n"
            + _table(["i", "i + k", "first pair at i + k (direct)", "first pair from i by the map", "dot(PE(i), PE(i + k))"], rows)
            + f"\n\nLargest absolute difference over all eight entries and the four positions between the mapped row and the "
            f"direct row: {_below(worst, 1e-12)}. The largest spread of the four dot products is {_below(max(dots) - min(dots), 1e-9)}.")


def sinusoid_grid_data(positions=32, d=32, base=BASE):
    """Rows of the encoding for positions 0 .. positions-1 at width d, for the heat map."""
    return [sinusoid(i, d, base) for i in range(positions)]


# ------------------------------------------------------------------------------------------------ rotary embeddings


def rotate(x, pos, ws, conv="adjacent"):
    """Rotate the pairs of x by pos * w_m. conv 'adjacent': pair m is (x[2m], x[2m+1]); 'half': pair m is (x[m], x[m + d/2])."""
    d = len(x)
    out = list(x)
    for m, w in enumerate(ws):
        c, s = math.cos(pos * w), math.sin(pos * w)
        a, b = (2 * m, 2 * m + 1) if conv == "adjacent" else (m, m + d // 2)
        out[a] = x[a] * c - x[b] * s
        out[b] = x[a] * s + x[b] * c
    return out


def rope_inputs(d=8):
    g = bookrng.generator(NAME_ROPE)
    q = [_round1(g.normal()) for _ in range(d)]
    k = [_round1(g.normal()) for _ in range(d)]
    return q, k


def rope_frequencies():
    d = 8
    q, k = rope_inputs(d)
    w = omegas(d)
    rows = [(m, _g(w[m]), _g(2 * math.pi / w[m])) for m in range(d // 2)]
    return (_seed_line(NAME_ROPE) + " Draws: the 8 entries of the query q, then the 8 of the key k, each `round(normal(), 1)`. "
            "Width d = 8, base 10000: pair m is the channels (2m, 2m + 1) and turns by the angle i * w_m at position i, with "
            "w_m = 10000^(-2m/8) (library pow).\n\n"
            f"q = {_vec(q, 1)}, k = {_vec(k, 1)}\n\n"
            + _table(["pair m", "w_m (radians per position)", "positions per full turn, 2 pi / w_m"], rows))


def rotary_score(q, k, i, j, ws, conv="adjacent"):
    return _dot(rotate(q, i, ws, conv), rotate(k, j, ws, conv))


def rope_shift():
    q, k = rope_inputs(8)
    ws = omegas(8)
    cases = [(0, 3), (5, 8), (40, 43), (100, 103), (1000, 1003), (3, 0), (8, 5)]
    rows = [(i, j, j - i, _fx(rotary_score(q, k, i, j, ws), 6)) for i, j in cases]
    same = [rotary_score(q, k, i, j, ws) for i, j in cases[:5]]
    qi, kj = rotate(q, 5, ws), rotate(k, 8, ws)
    return ("The q, k and rates of the last block, adjacent pairs. The score is the dot product of the query rotated to position "
            "i with the key rotated to position j.\n\n"
            + _table(["query position i", "key position j", "offset j - i", "score"], rows)
            + f"\n\nThe first five rows share the offset 3; their largest spread is {_below(max(same) - min(same), 1e-9)}. "
            f"For the second row, the rotated query is {_vec(qi, 4)} and the rotated key is {_vec(kj, 4)}.")


def rope_vs_position():
    """Rotated pair 0 of the query at positions 0 .. 3, to show that position alone moves the vector (length is kept)."""
    q, _ = rope_inputs(8)
    ws = omegas(8)
    rows = []
    for i in range(4):
        r = rotate(q, i, ws)
        rows.append((i, _vec(r[:2], 4), _fx(math.hypot(r[0], r[1]), 4), _fx(math.sqrt(_dot(r, r)), 4)))
    return ("The first pair of q, rotated at positions 0 to 3 (rate w_0 = 1), and the lengths: the pair's own and the whole "
            "vector's.\n\n" + _table(["position i", "rotated pair 0", "length of pair 0", "length of the rotated vector"], rows))


def rotation_data(steps=8):
    """Points for the figure: pair 0 (w = 1, radius 1) and pair 1 (w = 0.1, radius 0.6), positions 0 .. steps-1."""
    w0, w1 = omegas(8)[0], omegas(8)[1]
    a = [(math.cos(i * w0), math.sin(i * w0)) for i in range(steps)]
    b = [(0.6 * math.cos(i * w1), 0.6 * math.sin(i * w1)) for i in range(steps)]
    return a, b


# ------------------------------------------------------------------------------------------------ two ways to pair channels


def pairing():
    q, k = rope_inputs(8)
    ws = omegas(8)
    cases = [(2, 5), (10, 13), (100, 103)]
    rows = []
    cols = {"aa": [], "hh": [], "ah": [], "ha": []}
    for i, j in cases:
        aa = _dot(rotate(q, i, ws, "adjacent"), rotate(k, j, ws, "adjacent"))
        hh = _dot(rotate(q, i, ws, "half"), rotate(k, j, ws, "half"))
        ah = _dot(rotate(q, i, ws, "adjacent"), rotate(k, j, ws, "half"))
        ha = _dot(rotate(q, i, ws, "half"), rotate(k, j, ws, "adjacent"))
        for key, v in zip(cols, (aa, hh, ah, ha)):
            cols[key].append(v)
        rows.append((i, j, _fx(aa, 6), _fx(hh, 6), _fx(ah, 6), _fx(ha, 6)))

    def to_half(x):
        """The adjacent layout whose half-split reading is x: pair m of x is (x[m], x[m + d/2])."""
        d = len(x)
        out = [0.0] * d
        for m in range(d // 2):
            out[2 * m], out[2 * m + 1] = x[m], x[m + d // 2]
        return out

    qp, kp = to_half(q), to_half(k)
    perm = max(abs(_dot(rotate(qp, i, ws, "adjacent"), rotate(kp, j, ws, "adjacent")) - _dot(rotate(q, i, ws, "half"), rotate(k, j, ws, "half")))
               for i, j in cases)
    spread = lambda v: max(v) - min(v)  # noqa: E731
    return ("The q and k of the last blocks and the same rate w_m for pair m in both conventions. Adjacent: pair m is the channels "
            "(2m, 2m + 1). Half-split: pair m is the channels (m, m + 4). Scores at three position pairs with the same offset 3; "
            "'adj, half' means the query is rotated with the adjacent convention and the key with the half-split one.\n\n"
            + _table(["i", "j", "adjacent, adjacent", "half, half", "adj, half", "half, adj"], rows)
            + f"\n\nSpread over the three rows: adjacent, adjacent {_below(spread(cols['aa']), 1e-9)}; half, half "
            f"{_below(spread(cols['hh']), 1e-9)}; adj, half {_fx(spread(cols['ah']), 3)}; half, adj {_fx(spread(cols['ha']), 3)}. "
            "Reordering the channels of q and k so that the half-split pair (m, m + 4) lands at (2m, 2m + 1) and using the "
            f"adjacent convention gives the half-split scores: largest difference {_below(perm, 1e-9)}.")


# ------------------------------------------------------------------------------------------------ decay with distance

DECAY_DISTANCES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def decay_value(delta, d, base):
    """Score at offset delta of a query and a key that are equal on every pair (value 1 + 0i), divided by its value at 0."""
    ws = omegas(d, base)
    return _sum([math.cos(delta * w) for w in ws]) / len(ws)


def decay_data(d=64, bases=(1e4, 1e6), upto=512):
    xs = list(range(upto + 1))
    return xs, [[decay_value(x, d, b) for x in xs] for b in bases]


def rope_decay():
    d = 64
    rows = [(x, _fx(decay_value(x, d, 1e4), 3), _fx(decay_value(x, d, 1e6), 3)) for x in DECAY_DISTANCES]
    return ("Width d = 64 (32 pairs) and every pair of the query and of the key equal to 1 + 0i, so that the score at offset "
            "j - i is the sum over the 32 pairs of cos((j - i) w_m); divided by 32, so that offset 0 gives 1. Three decimals; "
            "cos is the library cosine.\n\n"
            + _table(["offset j - i", "base 10^4", "base 10^6"], rows))


# ------------------------------------------------------------------------------------------------ relative position biases

BUCKETS, MAX_DISTANCE = 32, 128


def t5_bucket(rel, buckets=BUCKETS, max_distance=MAX_DISTANCE):
    """Bucket of the offset rel = j - i (key position minus query position) of an encoder: half of the buckets for keys after the
    query (rel > 0), half for the others; inside a half, one bucket for each of the nearest max_exact distances, then
    logarithmic spacing up to max_distance, and the last bucket of the half for everything farther."""
    half = buckets // 2
    exact = half // 2
    near = abs(rel)
    if near < exact:
        off = near
    else:
        off = min(half - 1, exact + int(math.log(near / exact) / math.log(max_distance / exact) * (half - exact)))
    return (half if rel > 0 else 0) + off


def relative_buckets():
    ns = (0, 1, 3, 7, 8, 9, 12, 20, 40, 70, 100, 127, 128, 200)
    behind = [t5_bucket(-n) for n in ns]
    ahead = [t5_bucket(n) for n in ns]
    return (f"{BUCKETS} buckets and a largest distance of {MAX_DISTANCE}, so 16 buckets for the keys at or before the query "
            "(offset j - i <= 0) and 16 for the keys after it. With n = |j - i|: the bucket is n for n < 8; for n >= 8 it is "
            "8 + floor(log(n / 8) / log(128 / 8) * 8), at most 15 (natural logarithms), and 16 is added when the key is "
            "after the query.\n\n"
            + _table(["n = abs(j - i)", "key behind: j - i = -n", "key ahead: j - i = +n"],
                     [(n, b, a) for n, b, a in zip(ns, behind, ahead)]))


def alibi_slope(h, heads):
    return 2.0 ** (-8.0 * h / heads)


def alibi_bias():
    heads, T = 8, 5
    slopes = [alibi_slope(h, heads) for h in range(1, heads + 1)]
    m = slopes[0]
    rows = []
    for i in range(T):
        rows.append((i, *[(_fx(-m * (i - j), 3) if j <= i else "") for j in range(T)]))
    # weights of the last row of a position whose content scores are all equal, for three heads
    n = 6
    wrows = []
    for h in (1, 4, 8):
        mu = alibi_slope(h, heads)
        w = _softmax([-mu * (n - 1 - j) for j in range(n)])
        wrows.append((h, _g(mu, 3), *[_fx(v, 3) for v in w]))
    return (f"Eight heads: head h has the slope 2^(-8h/8), that is "
            + ", ".join(_g(s, 3) for s in slopes) + f". The bias added to the score of query i and key j (j <= i) in head 1 "
            f"(slope {_g(m, 3)}) is minus the slope times i - j; blank cells are hidden by the causal mask.\n\n"
            + _table(["i \\ j", *[str(j) for j in range(T)]], rows, "l" + "r" * T)
            + "\n\nWeights of query position 5 over the keys 0 to 5 when every content score is equal, so that only the bias "
            "decides:\n\n"
            + _table(["head h", "slope", *[f"j = {j}" for j in range(n)]], wrows, "lr" + "r" * n))


# ------------------------------------------------------------------------------------------------ beyond the trained length

TRAINED = 4096


def turns(d, base, length, m):
    return length * base ** (-2 * m / d) / (2 * math.pi)


def beyond_training():
    d, base = 64, 1e4
    pairs = d // 2
    short = [m for m in range(pairs) if turns(d, base, TRAINED, m) < 1]
    sel = (0, 8, 16, 20, 24, 28, 31)
    rows = [(m, _g(base ** (-2 * m / d), 4), _g(turns(d, base, TRAINED, m), 4), "yes" if turns(d, base, TRAINED, m) < 1 else "no")
            for m in sel]
    return (f"Width d = 64 (32 pairs), base 10000, a model trained on positions 0 to {TRAINED - 1}. Pair m turns "
            f"{TRAINED} w_m / (2 pi) times over the trained length.\n\n"
            + _table(["pair m", "w_m", f"turns over {TRAINED} positions", "less than one turn"], rows, "lrrl")
            + f"\n\nPairs that complete less than one turn over the trained length: {len(short)} of {pairs} "
            f"(m = {short[0]} to {short[-1]}).")


def _correction_dim(rot, d, base, length):
    return d * math.log(length / (rot * 2 * math.pi)) / (2 * math.log(base))


def yarn_table(d, base, factor, length, beta_fast=32.0, beta_slow=1.0):
    """(inv_freq, attention_factor, low, high): the YaRN frequencies as the chapter states them."""
    half = d // 2
    low = max(0, math.floor(_correction_dim(beta_fast, d, base, length)))
    high = min(half - 1, math.ceil(_correction_dim(beta_slow, d, base, length)))
    if low == high:
        high += 0.001
    out = []
    for m in range(half):
        w = base ** (-2 * m / d)
        ramp = min(1.0, max(0.0, (m - low) / (high - low)))
        out.append((w / factor) * ramp + w * (1.0 - ramp))
    return out, (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0), low, high


def context_extension():
    d, base, factor = 64, 1e4, 2.0
    w = omegas(d, base)
    yarn, att, low, high = yarn_table(d, base, factor, TRAINED)
    sel = (0, 4, 8, 10, 12, 16, 20, 23, 24, 27, 31)
    rows = [(m, _g(w[m], 4), _g(w[m] / factor, 4), _g(yarn[m], 4), _fx(min(1.0, max(0.0, (m - low) / (high - low))), 3)) for m in sel]
    return (f"Width d = 64, base 10000, trained length {TRAINED}, extension factor 2 (to 8192). Linear (position interpolation): "
            f"every rate divided by 2. YaRN with the rotation bounds 32 (fast) and 1 (slow): the correction range is "
            f"[{low}, {high}], the weight of the interpolated rate rises linearly from 0 at pair {low} to 1 at pair {high}, "
            f"and cos and sin are multiplied by 0.1 ln 2 + 1 = {att:.7f}.\n\n"
            + _table(["pair m", "unscaled w_m", "linear: w_m / 2", "YaRN", "weight of the interpolated rate"], rows)
            + f"\n\nPairs left unchanged by YaRN: {sum(1 for m in range(d // 2) if yarn[m] == w[m])} of {d // 2}; pairs divided by 2 "
            f"exactly: {sum(1 for m in range(d // 2) if abs(yarn[m] - w[m] / factor) <= 1e-15 * w[m])} of {d // 2}.")


def extension_reach():
    """The largest angle each scheme ever asks of pair m at position 8191, against the largest the trained length asked."""
    d, base, factor = 64, 1e4, 2.0
    w = omegas(d, base)
    yarn, _, _, _ = yarn_table(d, base, factor, TRAINED)
    new = 2 * TRAINED - 1
    rows = []
    for m in (0, 8, 16, 24, 31):
        rows.append((m, _g((TRAINED - 1) * w[m], 4), _g(new * w[m], 4), _g(new * w[m] / factor, 4), _g(new * yarn[m], 4)))
    return ("The angle in radians, position times rate, for a few pairs of the table above: at the last trained position "
            f"({TRAINED - 1}), at the last position of the extended window ({2 * TRAINED - 1}) without any scaling, and at that "
            "position with the linear and the YaRN rates.\n\n"
            + _table(["pair m", "angle at 4095, unscaled", "angle at 8191, unscaled", "angle at 8191, linear", "angle at 8191, YaRN"], rows))


# ------------------------------------------------------------------------------------------------ multi-axis rotary embeddings

MR_PIECES = (("what", 1), ("is", 1), ("image", 6), ("in", 1), ("the", 1), ("picture", 1))
MR_GRID = (1, 2, 3)           # (t, h, w) of the image


def mrope_index(pieces=MR_PIECES, grid=MR_GRID):
    """Position triples [t, h, w] of every token. A text token takes (c, c, c) and advances c by 1; the image takes
    (c + a, c + b, c + e) for a in 0..t-1, b in 0..h-1, e in 0..w-1 (t fastest-last: row-major), then c advances by max(t, h, w)."""
    c = 0
    out = []
    for name, count in pieces:
        if name == "image":
            t, h, w = grid
            assert t * h * w == count
            for a in range(t):
                for b in range(h):
                    for e in range(w):
                        out.append((name, (c + a, c + b, c + e)))
            c += max(t, h, w)
        else:
            out.append((name, (c, c, c)))
            c += 1
    return out


def mrope_positions():
    toks = mrope_index()
    rows = []
    for idx, (name, p) in enumerate(toks):
        rows.append((idx, name if name != "image" else "image patch", f"({p[0]}, {p[1]}, {p[2]})"))
    return ("A prompt of two words, a picture and three words. The picture is cut into 1 frame of 2 rows and 3 columns of "
            "patches (6 tokens, in row-major order). A word takes the position (c, c, c) and advances the counter c by 1; the "
            "picture takes the positions (c + t, c + h, c + w) for its frame t, row h and column w, and then advances c by "
            "max(1, 2, 3) = 3. The counter starts at 0. The index is also the position a one-axis scheme would "
            "give the token.\n\n" + _table(["index", "token", "(t, h, w)"], rows, "llr"))


def axis_owner(sections, half, layout):
    """Axis (0 = t, 1 = h, 2 = w) that drives each of the half pairs. 'blocks': t then h then w in runs of sections[a] pairs.
    'interleaved': pair r belongs to h if r mod 3 = 1 and r // 3 < sections[1], to w if r mod 3 = 2 and r // 3 < sections[2],
    and to t otherwise."""
    if layout == "blocks":
        owner = []
        for a, s in enumerate(sections):
            owner += [a] * s
        return owner
    owner = [0] * half
    for a in (1, 2):
        for u in range(sections[a]):
            if a + 3 * u < half:
                owner[a + 3 * u] = a
    return owner


MR_SECTIONS = (4, 2, 2)
MR_HEAD = 16
MR_BASE = 10000.0


def mrope_axes():
    half = MR_HEAD // 2
    ws = omegas(MR_HEAD, MR_BASE)
    names = "thw"
    rows = []
    for layout in ("blocks", "interleaved"):
        own = axis_owner(MR_SECTIONS, half, layout)
        spans = []
        for a in range(3):
            mine = [ws[m] for m in range(half) if own[m] == a]
            spans.append(f"{_g(max(mine), 3)} to {_g(min(mine), 3)}")
        rows.append((layout, " ".join(names[a] for a in own), *spans))
    return (f"Head width d = {MR_HEAD}, so {half} pairs with the rates w_m = 10000^(-2m/16); sections (t, h, w) = "
            f"{MR_SECTIONS} pairs. Two ways to assign the pairs to the axes, and the range of rates each axis ends up with "
            "(largest to smallest).\n\n"
            + _table(["layout", "axis of pairs 0 to 7", "rates of t", "rates of h", "rates of w"], rows, "lllll"))


def mrope_rotate(x, triple, own, ws, conv="half"):
    angles = [triple[own[m]] * ws[m] for m in range(len(ws))]
    d = len(x)
    out = list(x)
    for m, ang in enumerate(angles):
        c, s = math.cos(ang), math.sin(ang)
        a, b = (2 * m, 2 * m + 1) if conv == "adjacent" else (m, m + d // 2)
        out[a] = x[a] * c - x[b] * s
        out[b] = x[a] * s + x[b] * c
    return out


def mrope_offsets():
    g = bookrng.generator(NAME_MROPE)
    q = [_round1(g.normal()) for _ in range(MR_HEAD)]
    k = [_round1(g.normal()) for _ in range(MR_HEAD)]
    ws = omegas(MR_HEAD, MR_BASE)
    own = axis_owner(MR_SECTIONS, MR_HEAD // 2, "interleaved")

    def score(p, r):
        return _dot(mrope_rotate(q, p, own, ws), mrope_rotate(k, r, own, ws))

    pairs = [((2, 2, 2), (2, 2, 3)), ((5, 7, 9), (5, 7, 10)), ((2, 2, 2), (2, 3, 2)), ((8, 3, 6), (8, 4, 6)),
             ((0, 0, 0), (1, 2, 3)), ((4, 5, 6), (5, 7, 9)), ((4, 5, 6), (6, 5, 4))]
    rows = [(f"({p[0]}, {p[1]}, {p[2]})", f"({r[0]}, {r[1]}, {r[2]})", f"({r[0] - p[0]}, {r[1] - p[1]}, {r[2] - p[2]})",
             _fx(score(p, r), 6)) for p, r in pairs]
    same = [score(*pairs[0]), score(*pairs[1])], [score(*pairs[2]), score(*pairs[3])], [score(*pairs[4]), score(*pairs[5])]
    gaps = [max(v) - min(v) for v in same]
    # a text-only sequence: all three axes equal, against plain rotary embeddings at the same positions
    diag = 0.0
    for pos in (0, 3, 17, 100):
        a = mrope_rotate(q, (pos, pos, pos), own, ws)
        b = rotate(q, pos, ws, "half")
        diag = max(diag, max(abs(u - v) for u, v in zip(a, b)))
    return (_seed_line(NAME_MROPE) + " Draws: the 16 entries of q, then the 16 of k, each `round(normal(), 1)`. Head width 16, "
            "rates w_m = 10000^(-2m/16), half-split pairs (m, m + 8), interleaved sections (4, 2, 2): pair m turns by "
            "(position on its axis) * w_m. The score is the dot product of the rotated query at the first triple with the "
            "rotated key at the second.\n\n"
            + _table(["query triple", "key triple", "offset (t, h, w)", "score"], rows, "lllr")
            + f"\n\nRows 1 and 2 share an offset, as do rows 3 and 4, and rows 5 and 6: spreads {_below(gaps[0], 1e-9)}, "
            f"{_below(gaps[1], 1e-9)} and {_below(gaps[2], 1e-9)}. With the same position on all three axes the rotated query "
            f"equals the plain half-split rotation at that position: largest difference {_below(diag, 1e-12)} over the "
            "positions 0, 3, 17 and 100.")


# ------------------------------------------------------------------------------------------------ cost

COST_D, COST_H, COST_DH, COST_T = 1024, 16, 64, 4096


def cost_counts(D=COST_D, H=COST_H, dh=COST_DH, T=COST_T):
    pair = 6                              # 4 multiplications and 2 additions for one rotated pair
    rope = (H + H) * (dh // 2) * pair     # q and k, H heads each, dh/2 pairs per head
    attn = H * (2 * T * dh + 2 * T * dh)  # one query against T keys, then the weighted sum of T values
    proj = 8 * D * D                      # four projections of one token, 2 D^2 FLOPs each
    move = (H + H) * dh * 4 * 2           # read and write of q and k in float32
    tab_pairs = dh // 2
    return rope, attn, proj, move, tab_pairs


def rope_cost():
    D, H, dh, T = COST_D, COST_H, COST_DH, COST_T
    rope, attn, proj, move, half = cost_counts()
    tr_analytic = (H + H) * half * 3      # pow, sin, cos for each (head, pair), recomputed by every head
    tr_table = half * 2                   # sin and cos once per pair and position when a table is shared
    rows = [
        ("rotation of q and k", _int(rope), "6 per pair: 4 multiplications, 2 additions"),
        ("scores and weighted sum", _int(attn), f"4 T d_h per head, T = {T}"),
        ("four projections", _int(proj), "8 D^2"),
        ("scores and sum / rotation", f"{attn / rope:,.0f} times", "ratio of the rows 2 and 1"),
        ("projections / rotation", f"{proj / rope:,.0f} times", "ratio of the rows 3 and 1"),
    ]
    return (f"One token of one layer with D = {D}, H = {H} heads of width d_h = {dh} and a cache of T = {T} keys; a "
            "multiplication or an addition is one FLOP; sin, cos and the table lookups are counted apart.\n\n"
            + _table(["quantity", "FLOPs", "rule"], rows, "lrl")
            + f"\n\nThe rotation reads and writes q and k once each in float32: {_int(move)} bytes, so it does "
            f"{rope / move:.2f} FLOPs per byte. If every head computes its own angles (an analytic kernel), the token costs "
            f"{_int(tr_analytic)} evaluations of pow, sin and cos (each of the {H} heads of the query and each of the {H} heads of the key "
            f"rotates its {half} pairs, so q and k are counted separately: 2 x {H} x {half} x 3); if the angles come from a table shared by the heads, it "
            f"costs {_int(tr_table)} evaluations of sin and cos, or none when the table is read from memory.")

