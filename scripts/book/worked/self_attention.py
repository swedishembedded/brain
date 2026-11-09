# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Self-Attention": every position of a sequence looking at every other.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description
and independent of any engine code. Each public function without arguments returns a markdown string that the chapter
embeds between `<!-- worked: self_attention.<name> -->` markers; the `*_data` functions supply the numbers of the
chapter's figures, so a picture and the table beside it cannot disagree.

* four tokens of three numbers: queries, keys and values, the score matrix, the softmax of each row, the weighted sums;
* the variance of a dot product and the softmax that saturates without the scale;
* the shuffle test (attention treats its input as a set) and the causal mask, with the three versions of its boundary;
* two heads on a constructed example and the parameter count that does not depend on the number of heads;
* the backward pass of multi-head attention, checked against central differences, and three wrong backward passes;
* the FLOPs of the projections and of the score and average products, the bytes of the score matrix and the depth;
* a one-layer attention model trained on a lookup task, against the same model with uniform weights.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of
products are plain left-to-right additions, the order a C loop would use (never Python's `sum`, which may compensate).
Numbers of trained models are printed by the stable-digits rule of `stable.py`.

Swedish Embedded AB implements solutions for sequence models built on attention, from the kernels to the training loop.
If your team needs expertise in transformer attention and its gradients, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import stable

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
    """'below 1e-12' when x is round-off noise under the limit, else the value: noise is not reproducible digit by digit."""
    return f"below {limit:.0e}".replace("e-0", "e-") if x < limit else _fx(x, 3)


def _bound(x):
    """The power of ten above x, written 1e-7: a difference is reported by the bound it stays under, not by its digits."""
    if x <= 0:
        return "0"
    return f"1e{math.ceil(math.log10(x * 1.0000001)):d}".replace("e-0", "e-")


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


def transpose(A):
    return [list(r) for r in zip(*A)]


def _cols(M, a, b):
    return [row[a:b] for row in M]


def _hcat(blocks):
    return [sum(rows, []) for rows in zip(*blocks)]


def _softmax(z):
    """Stable softmax of a vector; entries that are -infinity get weight 0, and a row of only -infinity gives nan."""
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = _sum(e)
    return [v / s for v in e]


def _allowed(i, j, lookahead):
    return lookahead is None or j <= i + lookahead


# ------------------------------------------------------------------------------------------------ attention, forward and backward


def mha_forward(X, Wq, Wk, Wv, Wo, H, lookahead=None, uniform=False):
    """Multi-head attention over the rows of X [T, D]. Wq, Wk: [D, H dk]; Wv: [D, H dv]; Wo: [H dv, Dout] or None (no output
    projection). Head h uses columns h dk .. (h+1) dk of Q and K and h dv .. (h+1) dv of V; their outputs are concatenated.
    lookahead None: every pair is allowed; an integer c allows the pair (i, j) when j <= i + c, so 0 is the causal mask, -1 the
    mask that also hides the position itself and 1 the mask that lets a position see the next one. uniform=True replaces the
    weights by 1 over the allowed positions (the ablation of the lookup experiment). Returns (Y, cache)."""
    T = len(X)
    dk, dv = len(Wq[0]) // H, len(Wv[0]) // H
    Q, K, V = matmul(X, Wq), matmul(X, Wk), matmul(X, Wv)
    scale = 1.0 / math.sqrt(dk)
    heads = []
    for h in range(H):
        Qh, Kh, Vh = _cols(Q, h * dk, (h + 1) * dk), _cols(K, h * dk, (h + 1) * dk), _cols(V, h * dv, (h + 1) * dv)
        S = [[_dot(Qh[i], Kh[j]) * scale if _allowed(i, j, lookahead) else -math.inf for j in range(T)] for i in range(T)]
        if uniform:
            P = []
            for i in range(T):
                n = len([1 for j in range(T) if _allowed(i, j, lookahead)])
                P.append([1.0 / n if _allowed(i, j, lookahead) else 0.0 for j in range(T)])
        else:
            P = [_softmax(row) for row in S]
        heads.append({"Q": Qh, "K": Kh, "V": Vh, "S": S, "P": P, "O": matmul(P, Vh)})
    O = _hcat([hd["O"] for hd in heads])
    Y = matmul(O, Wo) if Wo is not None else O
    return Y, {"X": X, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "H": H, "dk": dk, "dv": dv, "heads": heads, "O": O,
               "uniform": uniform}


def mha_backward(cache, dY, bug=None):
    """The gradient of a loss with respect to Wq, Wk, Wv, Wo and X, given dY = dL/dY [T, Dout], by the chain of the chapter:
    dO = dY Wo^T; per head dP = dO V^T, dV = P^T dO, dS = P (dP - rowsum(P dP)), dQ = dS K / sqrt(dk), dK = dS^T Q / sqrt(dk);
    then dWq = X^T dQ (and likewise for K and V) and dX = dQ Wq^T + dK Wk^T + dV Wv^T. With uniform weights the scores do not
    reach the output, so dS = 0 and Wq and Wk receive no gradient. `bug` names a wrong variant for the
    check to catch: "no_rowsum" (softmax rule without the row term), "no_scale" (the 1/sqrt(dk) left out of dQ and dK) and
    "head_layout" (dO cut into heads as if the concatenation were [H, T, dv] instead of [T, H dv])."""
    X, H, dk, dv, heads = cache["X"], cache["H"], cache["dk"], cache["dv"], cache["heads"]
    T = len(X)
    scale = 1.0 / math.sqrt(dk)
    Wo = cache["Wo"]
    dO = matmul(dY, transpose(Wo)) if Wo is not None else dY
    g = {"Wo": matmul(transpose(cache["O"]), dY) if Wo is not None else None}
    flat = [x for row in dO for x in row]
    dQ, dK, dV = [], [], []
    for h in range(H):
        hd = heads[h]
        if bug == "head_layout":
            dOh = [flat[(h * T + i) * dv:(h * T + i + 1) * dv] for i in range(T)]
        else:
            dOh = _cols(dO, h * dv, (h + 1) * dv)
        P = hd["P"]
        dP = matmul(dOh, transpose(hd["V"]))
        dVh = matmul(transpose(P), dOh)
        dS = []
        for i in range(T):
            rowsum = 0.0 if bug == "no_rowsum" else _dot(P[i], dP[i])
            dS.append([0.0] * T if cache["uniform"] else [P[i][j] * (dP[i][j] - rowsum) for j in range(T)])
        f = 1.0 if bug == "no_scale" else scale
        dQ.append([[f * x for x in row] for row in matmul(dS, hd["K"])])
        dK.append([[f * x for x in row] for row in matmul(transpose(dS), hd["Q"])])
        dV.append(dVh)
    dQ, dK, dV = _hcat(dQ), _hcat(dK), _hcat(dV)
    Xt = transpose(X)
    g["Wq"], g["Wk"], g["Wv"] = matmul(Xt, dQ), matmul(Xt, dK), matmul(Xt, dV)
    parts = [matmul(dQ, transpose(cache["Wq"])), matmul(dK, transpose(cache["Wk"])), matmul(dV, transpose(cache["Wv"]))]
    g["X"] = [[_sum([parts[0][i][c], parts[1][i][c], parts[2][i][c]]) for c in range(len(X[0]))] for i in range(T)]
    return g


# ------------------------------------------------------------------------------------------------ the four-token example

NAME_EXAMPLE = "self_attention.example"
TOKENS = ["the", "cat", "sat", "down"]   # labels only: the numbers below are random stand-ins for learned embeddings


@lru_cache(maxsize=None)
def example():
    g = bookrng.generator(NAME_EXAMPLE)
    X = [[_round1(2 * g.normal()) for _ in range(3)] for _ in range(4)]
    W = {}
    for name in ("Wq", "Wk", "Wv"):
        W[name] = [[_round1(g.normal()) for _ in range(2)] for _ in range(3)]
    Q, K, V = matmul(X, W["Wq"]), matmul(X, W["Wk"]), matmul(X, W["Wv"])
    S = matmul(Q, transpose(K))
    Ss = [[v / math.sqrt(2) for v in row] for row in S]
    P = [_softmax(row) for row in Ss]
    Pc = [_softmax([v if j <= i else -math.inf for j, v in enumerate(row)]) for i, row in enumerate(Ss)]
    return {"X": X, **W, "Q": Q, "K": K, "V": V, "S": S, "Ss": Ss, "P": P, "O": matmul(P, V), "Pc": Pc, "Oc": matmul(Pc, V)}


def attention_data():
    e = example()
    return TOKENS, e["P"], e["Pc"]


def example_inputs():
    e = example()
    rows = [(i + 1, TOKENS[i], _vec(e["X"][i], 1)) for i in range(4)]
    wrows = [(f"row {r + 1}", _vec(e["Wq"][r], 1), _vec(e["Wk"][r], 1), _vec(e["Wv"][r], 1)) for r in range(3)]
    return (_seed_line(NAME_EXAMPLE) + " Draws, in this order: the 12 entries of X, row by row, each `round(2 * normal(), 1)`; then "
            "the 6 entries of W_Q, the 6 of W_K and the 6 of W_V, each row by row and each `round(normal(), 1)` (`round` to one "
            "decimal, ties away from zero). The words are only labels for the four positions; the numbers are random "
            "stand-ins for learned embeddings. The width of a query and a key is d_k = 2 and of a value d_v = 2.\n\n"
            + _table(["position i", "token", "x_i (3 numbers)"], rows, "lll") + "\n\n"
            + _table(["", "W_Q (3 by 2)", "W_K (3 by 2)", "W_V (3 by 2)"], wrows, "llll"))


def projections():
    e = example()
    x = e["X"][1]
    parts = [f"{_fx(x[r], 1)} * {_fx(e['Wq'][r][0], 1)}" for r in range(3)]
    terms = [x[r] * e["Wq"][r][0] for r in range(3)]
    rows = [(i + 1, TOKENS[i], _vec(e["Q"][i]), _vec(e["K"][i]), _vec(e["V"][i])) for i in range(4)]
    return ("Q = X W_Q, K = X W_K, V = X W_V with the X, W_Q, W_K, W_V above. Every entry is a sum of three products of numbers "
            "with one decimal, so two decimals are exact: for instance the first entry of q_2 is "
            + " + ".join(parts) + " = " + " + ".join(_fx(t, 2) for t in terms) + f" = {_fx(e['Q'][1][0], 2)}.\n\n"
            + _table(["i", "token", "query q_i", "key k_i", "value v_i"], rows, "lllll"))


def score_matrix():
    e = example()
    s = 1.0 / math.sqrt(2)
    rows = [(i + 1, TOKENS[i], *[_fx(v, 4) for v in e["S"][i]]) for i in range(4)]
    rows2 = [(i + 1, TOKENS[i], *[_fx(v, 3) for v in e["Ss"][i]]) for i in range(4)]
    head = ["i", "token"] + [f"j = {j + 1}" for j in range(4)]
    return ("The score matrix is S = Q K^T: entry (i, j) is the dot product of query q_i with key k_j, so row i holds what "
            "position i asks of every position. Four decimals are exact. Then every entry is divided by sqrt(d_k) = sqrt(2) = "
            f"{s:.6f} (to 3 decimals).\n\n" + _table(head, rows, "ll" + "r" * 4) + "\n\n"
            + _table(head, rows2, "ll" + "r" * 4))


def weights_and_outputs():
    e = example()
    P, O, V = e["P"], e["O"], e["V"]
    i = 1
    row = e["Ss"][i]
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    z = _sum(ex)
    detail = _table(["j", "scaled score", "minus the largest", "exp of that", "weight = exp / " + _fx(z, 3)],
                    [(j + 1, _fx(row[j], 3), _fx(row[j] - m, 3), _fx(ex[j], 3), _fx(ex[j] / z, 3)) for j in range(4)])
    wsum = " + ".join(f"{_fx(P[i][j], 3)} * {_vec(V[j])}" for j in range(4))
    rows = [(r + 1, TOKENS[r], *[_fx(v, 3) for v in P[r]], _fx(_sum(P[r]), 3), _vec(O[r], 3)) for r in range(4)]
    return ("Row 2 in detail (the softmax subtracts the largest score first, as the stable softmax of Volume III does):\n\n"
            + detail + f"\n\nThe output of position 2 is the weighted sum of the four value vectors: {wsum} = {_vec(O[i], 3)}.\n\n"
            "Every row, with the sum of its weights and its output o_i = sum_j p_ij v_j (3 decimals):\n\n"
            + _table(["i", "token", "p_i1", "p_i2", "p_i3", "p_i4", "row sum", "output o_i"], rows, "ll" + "r" * 5 + "l"))


# ------------------------------------------------------------------------------------------------ why divide by sqrt(d_k)

NAME_VARIANCE, NAME_SATURATION = "self_attention.variance", "self_attention.saturation"
WIDTHS = [1, 4, 16, 64, 256]
N_PAIRS = 2000
N_ROWS, N_KEYS = 500, 8


def scale_variance():
    g = bookrng.generator(NAME_VARIANCE)
    rows = []
    for d in WIDTHS:
        dots = []
        for _ in range(N_PAIRS):
            q = [g.normal() for _ in range(d)]
            k = [g.normal() for _ in range(d)]
            dots.append(_dot(q, k))
        mean = _sum(dots) / N_PAIRS
        var = _sum([(x - mean) ** 2 for x in dots]) / (N_PAIRS - 1)
        scaled = var / d
        rows.append((d, _fx(math.sqrt(d), 3), _fx(var, 2), _fx(math.sqrt(var), 3), _fx(scaled, 3)))
    return (_seed_line(NAME_VARIANCE) + f" For each width d_k in the table, {_int(N_PAIRS)} pairs in this order: d_k draws of "
            "`normal()` for the entries of q, then d_k for the entries of k; the dot product of each pair is formed. The "
            "variance is the sample variance with divisor n - 1. Tolerance: the sample variance of "
            f"{_int(N_PAIRS)} values has a relative standard error of about {_fx(math.sqrt(2 / (N_PAIRS - 1)), 3)} for wide "
            f"vectors and {_fx(math.sqrt(8 / (N_PAIRS - 1)), 3)} for d_k = 1, where the product of two normal numbers has "
            "heavier tails.\n\n"
            + _table(["d_k", "predicted standard deviation sqrt(d_k)", "sample variance of q . k", "sample standard deviation",
                      "sample variance of (q . k) / sqrt(d_k)"], rows))


def softmax_saturation():
    g = bookrng.generator(NAME_SATURATION)
    rows = []
    for d in WIDTHS:
        acc = {"raw": [0.0, 0.0, 0], "scaled": [0.0, 0.0, 0]}
        for _ in range(N_ROWS):
            q = [g.normal() for _ in range(d)]
            keys = [[g.normal() for _ in range(d)] for _ in range(N_KEYS)]
            raw = [_dot(q, k) for k in keys]
            for name, scores in (("raw", raw), ("scaled", [x / math.sqrt(d) for x in raw])):
                p = _softmax(scores)
                acc[name][0] += max(p)
                acc[name][1] += 1.0 - _sum([x * x for x in p])
                acc[name][2] += 1 if max(p) > 0.99 else 0
        r = [d]
        for name in ("raw", "scaled"):
            r += [_fx(acc[name][0] / N_ROWS, 3), _fx(acc[name][1] / N_ROWS, 3), f"{acc[name][2]}/{N_ROWS}"]
        rows.append(tuple(r))
    return (_seed_line(NAME_SATURATION) + f" For each width d_k, {N_ROWS} rows in this order: d_k draws of `normal()` for the "
            f"query, then {N_KEYS} keys of d_k draws each. A row is softmax of the {N_KEYS} dot products, once as they are and "
            "once divided by sqrt(d_k). Entries: the mean over the rows of the largest weight; the mean of 1 - sum of squared "
            "weights (the sum of the diagonal slopes p_i (1 - p_i) of the softmax, which is how much gradient the row can "
            f"pass on); and the number of rows whose largest weight exceeds 0.99. Eight equal weights would be 0.125 and "
            "0.875.\n\n"
            + _table(["d_k", "largest weight, unscaled", "gradient signal, unscaled", "rows above 0.99, unscaled",
                      "largest weight, scaled", "gradient signal, scaled", "rows above 0.99, scaled"], rows))


# ------------------------------------------------------------------------------------------------ attention as a set operation

NAME_PERMUTATION = "self_attention.shuffle"


def equivariance():
    e = example()
    g = bookrng.generator(NAME_PERMUTATION)
    perm = g.shuffle([0, 1, 2, 3])
    Xp = [e["X"][p] for p in perm]
    Y, _ = mha_forward(e["X"], e["Wq"], e["Wk"], e["Wv"], None, 1)
    Yp, _ = mha_forward(Xp, e["Wq"], e["Wk"], e["Wv"], None, 1)
    Yc, _ = mha_forward(e["X"], e["Wq"], e["Wk"], e["Wv"], None, 1, lookahead=0)
    Ycp, _ = mha_forward(Xp, e["Wq"], e["Wk"], e["Wv"], None, 1, lookahead=0)
    diff = max(abs(Yp[a][c] - Y[perm[a]][c]) for a in range(4) for c in range(2))
    diffc = max(abs(Ycp[a][c] - Yc[perm[a]][c]) for a in range(4) for c in range(2))
    rows = [(a + 1, TOKENS[perm[a]], perm[a] + 1, _vec(Yp[a], 3), _vec(Y[perm[a]], 3)) for a in range(4)]
    return (_seed_line(NAME_PERMUTATION) + " The permutation is `shuffle([0, 1, 2, 3], rng)` (Fisher-Yates as in Volume I): "
            f"{perm}, so position a of the shuffled input holds token perm[a] + 1 of the original. The example above, "
            "attention without a mask, run on the original and on the shuffled input.\n\n"
            + _table(["position a of the shuffled input", "its token", "perm[a] + 1", "output at a, shuffled input",
                      "output at perm[a] + 1, original input"], rows, "lllll")
            + f"\n\nLargest absolute difference between the two output columns: {_below(diff, 1e-12)}. With the "
            f"causal mask of the next section the same comparison gives a largest difference of {_fx(diffc, 3)}.")


# ------------------------------------------------------------------------------------------------ the causal mask


def causal_weights():
    e = example()
    rows = [(i + 1, TOKENS[i], *[_fx(e["Pc"][i][j], 3) for j in range(4)], _vec(e["Oc"][i], 3)) for i in range(4)]
    return ("The scaled scores of the example, with the entries above the diagonal replaced by -infinity before the softmax:\n\n"
            + _table(["i", "token", "p_i1", "p_i2", "p_i3", "p_i4", "output o_i"], rows, "ll" + "r" * 4 + "l"))


def mask_boundary():
    e = example()
    new = list(e["X"][2])
    new[0] += 1.0                      # the third token is replaced by one whose first entry is larger by 1
    X2 = [list(r) for r in e["X"]]
    X2[2] = new
    out = []
    for name, la in (("no mask", None), ("causal: j <= i", 0), ("one too tight: j < i", -1), ("one too loose: j <= i + 1", 1)):
        Y1, _ = mha_forward(e["X"], e["Wq"], e["Wk"], e["Wv"], None, 1, lookahead=la)
        Y2, _ = mha_forward(X2, e["Wq"], e["Wk"], e["Wv"], None, 1, lookahead=la)
        nan_rows = [i + 1 for i in range(4) if any(math.isnan(v) for v in Y1[i])]
        changed = [i + 1 for i in range(4) if i + 1 not in nan_rows and max(abs(Y1[i][c] - Y2[i][c]) for c in range(2)) > 1e-12]
        out.append((name, ", ".join(map(str, changed)) + (f" (position {', '.join(map(str, nan_rows))}: nan)" if nan_rows else ""),
                    "output undefined (nan) at position " + ", ".join(map(str, nan_rows)) if nan_rows else "-"))
    return ("The input of the example with the first entry of x_3 increased by 1 (token 3 is replaced). The table lists the "
            "positions whose output changes by more than 1e-12 when it is, for four versions of the boundary of the mask; a position whose output is nan "
            "(no score survives the mask) is named in brackets and left out of the comparison. "
            "A causal model must not change at positions 1 and 2, which come before the token that was replaced.\n\n"
            + _table(["mask (the pair i, j is kept when)", "positions whose output changes", "problem"], out, "lll"))


# ------------------------------------------------------------------------------------------------ several heads

NAME_HEADS = "self_attention.heads"
HEAD_TOKENS = [("A", "red"), ("B", "red"), ("A", "blue"), ("C", "blue")]
SHAPES, COLORS = ["A", "B", "C"], ["red", "blue", "green"]
GAIN = 2.0


def _onehot(n, i):
    return [1.0 if k == i else 0.0 for k in range(n)]


@lru_cache(maxsize=None)
def heads_example():
    """D = 6 (3 shape features, 3 colour features), two heads of width 3, four tokens. Head 1 compares the shape features
    (W_Q = W_K = 2 I on the first three inputs) and carries the colour features as values; head 2 compares the colour
    features and carries the shape features. W_O = I."""
    X = [_onehot(3, SHAPES.index(s)) + _onehot(3, COLORS.index(c)) for s, c in HEAD_TOKENS]
    def block(rows_on):
        return [[GAIN if (r == c + (0 if rows_on == 0 else 3)) else 0.0 for c in range(3)] for r in range(6)]
    def carry(rows_on):     # values: the features of the other kind
        return [[1.0 if (r == c + (3 if rows_on == 0 else 0)) else 0.0 for c in range(3)] for r in range(6)]
    Wq = _hcat([block(0), block(1)])
    Wv = _hcat([carry(0), carry(1)])
    Wo = [[1.0 if r == c else 0.0 for c in range(6)] for r in range(6)]
    Y, cache = mha_forward(X, Wq, Wq, Wv, Wo, 2)
    # one head of width 6 that compares all six features
    W1q = [[GAIN if r == c else 0.0 for c in range(6)] for r in range(6)]
    Y1, cache1 = mha_forward(X, W1q, W1q, [[1.0 if r == c else 0.0 for c in range(6)] for r in range(6)], None, 1)
    return {"X": X, "Y": Y, "cache": cache, "Y1": Y1, "cache1": cache1}


def heads_weights():
    h = heads_example()
    names = [f"{s}-{c}" for s, c in HEAD_TOKENS]
    out = [f"Four tokens with six features: three for the shape (A, B, C) and three for the colour (red, blue, green), each "
           f"a one-hot group. The tokens are {', '.join(f'{i + 1} = {n}' for i, n in enumerate(names))}. Head 1 has W_Q = W_K "
           f"= {GAIN:g} on the three shape features (zero on the colour features) and its values are the colour features; "
           f"head 2 does the same with the roles swapped; both have d_k = d_v = 3, so scores are divided by sqrt(3) = "
           f"{math.sqrt(3):.3f}; W_O is the identity. The single head for comparison has width 6, W_Q = W_K = {GAIN:g} on all "
           "six features and values equal to the inputs. No mask. Weights to 3 decimals, one row for each position i:"]
    for title, heads in (("Head 1 (matches on shape)", [h["cache"]["heads"][0]]), ("Head 2 (matches on colour)", [h["cache"]["heads"][1]]),
                         ("One head on all six features", [h["cache1"]["heads"][0]])):
        rows = [(i + 1, names[i], *[_fx(heads[0]["P"][i][j], 3) for j in range(4)]) for i in range(4)]
        out.append(f"{title}:\n\n" + _table(["i", "token", "j = 1", "j = 2", "j = 3", "j = 4"], rows, "ll" + "r" * 4))
    return "\n\n".join(out)


def heads_output():
    h = heads_example()
    names = [f"{s}-{c}" for s, c in HEAD_TOKENS]
    rows = []
    for i in range(4):
        rows.append((i + 1, names[i], _vec(h["Y"][i][:3], 3), _vec(h["Y"][i][3:], 3), _vec(h["Y1"][i], 3)))
    return ("Outputs of the two heads concatenated (the first three numbers are head 1's, the last three head 2's), and of "
            "the single head of width 6. The first group of the two-head output is a weighted mean of the colour features, "
            "the second a weighted mean of the shape features:\n\n"
            + _table(["i", "token", "head 1: colours of same-shape tokens", "head 2: shapes of same-colour tokens",
                      "one head: all six features at once"], rows, "lllll"))


def head_parameters():
    D, T = 6, 4
    rows = []
    for H in (1, 2, 3, 6):
        dh = D // H
        params = 3 * D * (H * dh) + (H * dh) * D
        qk_flops = H * 2 * T * T * dh
        rows.append((H, dh, _int(params), _int(qk_flops), _int(H * T * T)))
    return (f"The width D = {D} split into H heads of width D / H, T = {T} positions. Parameters: the three projections D by "
            "(H d_h) and the output projection (H d_h) by D, with no biases. FLOPs of the score products Q_h K_h^T of all "
            "heads, 2 T^2 d_h for each head. Scores held per sequence: H T^2.\n\n"
            + _table(["heads H", "head width d_h", "parameters", "FLOPs of all score products", "score entries"], rows))


# ------------------------------------------------------------------------------------------------ the backward pass

NAME_CHECK = "self_attention.check"
CHECK_T, CHECK_D, CHECK_H, CHECK_C = 5, 6, 2, 4
EPS_CHECK = 1e-5


def _check_setup():
    g = bookrng.generator(NAME_CHECK)
    p = {"X": [[g.normal() for _ in range(CHECK_D)] for _ in range(CHECK_T)]}
    for name, shape in (("Wq", (CHECK_D, CHECK_D)), ("Wk", (CHECK_D, CHECK_D)), ("Wv", (CHECK_D, CHECK_D)),
                        ("Wo", (CHECK_D, CHECK_D)), ("Wc", (CHECK_D, CHECK_C))):
        p[name] = [[0.5 * g.normal() for _ in range(shape[1])] for _ in range(shape[0])]
    targets = [g.integer(CHECK_C) for _ in range(CHECK_T)]
    return p, targets


def _check_loss(p, targets, lookahead):
    Y, cache = mha_forward(p["X"], p["Wq"], p["Wk"], p["Wv"], p["Wo"], CHECK_H, lookahead)
    logits = matmul(Y, p["Wc"])
    loss, dlog = 0.0, []
    for t, row in enumerate(logits):
        pr = _softmax(row)
        loss += -math.log(pr[targets[t]]) / CHECK_T
        dlog.append([(pr[c] - (1.0 if c == targets[t] else 0.0)) / CHECK_T for c in range(CHECK_C)])
    return loss, cache, Y, dlog


def _check_gradients(p, targets, lookahead, bug=None):
    loss, cache, Y, dlog = _check_loss(p, targets, lookahead)
    dY = matmul(dlog, transpose(p["Wc"]))
    g = mha_backward(cache, dY, bug)
    g["Wc"] = matmul(transpose(Y), dlog)
    return g


def gradient_check():
    p, targets = _check_setup()
    g = _check_gradients(p, targets, 0)
    rows = []
    for name in ("Wq", "Wk", "Wv", "Wo", "Wc", "X"):
        worst_abs, worst_rel, n = 0.0, 0.0, 0
        for r in range(len(p[name])):
            for c in range(len(p[name][0])):
                old = p[name][r][c]
                p[name][r][c] = old + EPS_CHECK
                lp = _check_loss(p, targets, 0)[0]
                p[name][r][c] = old - EPS_CHECK
                lm = _check_loss(p, targets, 0)[0]
                p[name][r][c] = old
                num = (lp - lm) / (2 * EPS_CHECK)
                ana = g[name][r][c]
                n += 1
                worst_abs = max(worst_abs, abs(ana - num))
                if max(abs(ana), abs(num)) > 1e-4:
                    worst_rel = max(worst_rel, abs(ana - num) / max(abs(ana), abs(num)))
        rows.append((name, n, _bound(worst_abs), _bound(worst_rel)))
    return (_seed_line(NAME_CHECK) + f" T = {CHECK_T} positions, D = {CHECK_D}, H = {CHECK_H} heads of width {CHECK_D // CHECK_H}, "
            f"{CHECK_C} classes, causal mask. Draws, in this order: X ({CHECK_T} by {CHECK_D}, each `normal()`), W_Q, W_K, W_V, "
            f"W_O ({CHECK_D} by {CHECK_D}) and W_C ({CHECK_D} by {CHECK_C}), each row by row with `0.5 * normal()`, then the "
            f"{CHECK_T} target classes `integer({CHECK_C})`. The model is multi-head attention, then the classifier "
            "logits = Y W_C at every position, and the loss is the mean over positions of the softmax cross-entropy of Volume III. "
            f"Central differences with step {EPS_CHECK:g} on every entry of every parameter and of the input X. Tolerance: "
            "each cell is the power of ten that the largest difference stays below (an implementation that adds in another order "
            "lands within a factor of ten of it; the criterion is 1e-6); the relative difference |a - n| / max(|a|, |n|) is "
            "taken over the entries whose gradient is above 1e-4.\n\n"
            + _table(["tensor", "entries checked", "largest absolute difference below", "largest relative difference below"], rows))


def wrong_backward_passes():
    p, targets = _check_setup()
    rows = []
    for label, bug in (("the chapter's backward pass", None), ("softmax rule without the row term", "no_rowsum"),
                       ("1 / sqrt(d_k) left out of dQ and dK", "no_scale"), ("dO cut into heads as [H, T, d_v]", "head_layout")):
        g = _check_gradients(p, targets, 0, bug)
        worst = {}
        for name in ("Wq", "Wk", "Wv", "Wo", "X"):
            w = 0.0
            for r in range(len(p[name])):
                for c in range(len(p[name][0])):
                    old = p[name][r][c]
                    p[name][r][c] = old + EPS_CHECK
                    lp = _check_loss(p, targets, 0)[0]
                    p[name][r][c] = old - EPS_CHECK
                    lm = _check_loss(p, targets, 0)[0]
                    p[name][r][c] = old
                    num, ana = (lp - lm) / (2 * EPS_CHECK), g[name][r][c]
                    if max(abs(ana), abs(num)) > 1e-4:
                        w = max(w, abs(ana - num) / max(abs(ana), abs(num)))
            worst[name] = w
        cells = [(_bound(worst[n]) if worst[n] < 1e-6 else ("about " + _fx(worst[n], 1))) for n in ("Wq", "Wk", "Wv", "Wo", "X")]
        rows.append((label, *cells))
    return ("The check of the previous block (same parameters, same step) run on a correct and on three wrong backward passes. "
            "Entries: the largest relative difference |a - n| / max(|a|, |n|) over the entries of the tensor whose gradient is "
            "above 1e-4; a difference of 1e-6 or more is written to one decimal, a smaller one as the power of ten it stays below "
            "(a tensor that the wrong step does not touch keeps its correct gradient).\n\n"
            + _table(["backward pass", "W_Q", "W_K", "W_V", "W_O", "X"], rows))


# ------------------------------------------------------------------------------------------------ cost

COST_D, COST_H = 1024, 16
SOFTMAX_FLOPS_PER_SCORE = 5     # scale, subtract the largest, exp, add to the normalizer, divide


def flops_closed(T, D, H, causal=False):
    """(projections, score and average products, softmax and scaling) for one layer: 8 T D^2, 4 T^2 D and 5 H T^2; a causal
    layer that skips the masked entries has T (T + 1) / 2 scores per row block instead of T^2."""
    pairs = T * (T + 1) // 2 if causal else T * T
    return 8 * T * D * D, 4 * pairs * D, SOFTMAX_FLOPS_PER_SCORE * H * pairs


def flops_counted(T, D, H, causal=False):
    """The same by walking the loops of the layer and counting a multiplication and an addition as one FLOP each."""
    dh = D // H
    proj = out = 0
    for _ in range(T):
        for _ in range(3):
            proj += D * D * 2
        out += D * D * 2
    score = apply = soft = 0
    for _h in range(H):
        for i in range(T):
            for j in range(T):
                if causal and j > i:
                    continue
                score += dh * 2
                soft += SOFTMAX_FLOPS_PER_SCORE
                apply += dh * 2
    return proj + out, score + apply, soft


def flops_data():
    """(T, projections, attention products) in GFLOPs for the figure: D = 1024, H = 16."""
    ts = list(range(256, 8193, 256))
    return ts, [8 * t * COST_D * COST_D / 1e9 for t in ts], [4 * t * t * COST_D / 1e9 for t in ts]


def cost_of_a_layer():
    for T, D, H in ((7, 12, 3), (16, 8, 2)):
        for causal in (False, True):
            assert flops_closed(T, D, H, causal) == flops_counted(T, D, H, causal), (T, D, H, causal)
    rows = []
    for T in (512, 1024, 2048, 4096, 8192, 32768):
        pr, at, sm = flops_closed(T, COST_D, COST_H)
        rows.append((_int(T), _int(pr), _int(at), _int(sm), _fx(at / pr, 3), _fx(at / (pr + at + sm), 3)))
    pr, at, _ = flops_closed(2 * COST_D, COST_D, COST_H)
    bwd = 8 * 4096 * 4096 * COST_D
    fwd = 4 * 4096 * 4096 * COST_D
    return (f"One attention layer on T positions of width D = {COST_D} with H = {COST_H} heads of width {COST_D // COST_H}, "
            "no biases. Counting rule: a matrix product of an m by k matrix with a k by n matrix is 2 m k n FLOPs; the softmax "
            f"and the scaling cost {SOFTMAX_FLOPS_PER_SCORE} FLOPs per score (scale, subtract the largest, exponential, add to the "
            "normalizer, divide). The closed forms 8 T D^2 (four projections), 4 T^2 D (the products Q K^T and P V of all heads "
            "together) and 5 H T^2 agree with a count made by walking the loops (checked for two small layers, with and without "
            "the causal skip of the masked scores).\n\n"
            + _table(["T", "projection FLOPs 8 T D^2", "score and average FLOPs 4 T^2 D", "softmax FLOPs 5 H T^2",
                      "attention products / projections", "attention products / whole layer"], rows)
            + f"\n\nThe two products are equal at T = 2 D = {2 * COST_D} ({_int(pr)} FLOPs against {_int(at)}). The backward pass "
            "of the attention products is four products of the same size (dP, dV, dQ and dK, each 2 T^2 D), so at T = 4096 it "
            f"costs {_int(bwd)} FLOPs against {_int(fwd)} for the forward products, twice as many. A causal layer that skips "
            "the masked scores does T (T + 1) / 2 of the T^2 score computations, "
            f"{_fx(4096 * 4097 / 2 / (4096 * 4096), 4)} of the work at T = 4096.")


def score_memory():
    rows = []
    for T in (2048, 8192, 32768):
        n = COST_H * T * T
        rows.append((_int(T), _int(n), _int(4 * n), _fx(4 * n / 2 ** 30, 2), _fx(24 * 4 * n / 2 ** 30, 1)))
    return (f"The score matrix of one layer for one sequence with H = {COST_H} heads, in float32 (4 bytes per entry), and "
            "the same for 24 layers if the weights of every layer are kept for the backward pass.\n\n"
            + _table(["T", "entries H T^2", "bytes", "GiB, one layer", "GiB, 24 layers"], rows)
            + "\n\nThe chain of three kernels that materialises the matrix writes the scores (4 bytes), reads them to form the softmax, "
            "writes the weights (4 bytes) and reads them for the weighted sum: 16 bytes of traffic per score. For a head width of "
            f"d_h = {COST_D // COST_H} one score costs 4 d_h + 5 = {4 * (COST_D // COST_H) + 5} FLOPs (two products and the softmax), so "
            f"the chain does {_fx((4 * (COST_D // COST_H) + 5) / 16, 2)} FLOPs for every byte moved through the matrix.")


def depth_and_work():
    H, D = 1024, 1024
    lstm_step = 8 * H * (H + D) + 17 * H
    rows = []
    for T in (16, 256, 4096):
        pr, at, sm = flops_closed(T, COST_D, COST_H)
        rows.append((_int(T), _int(lstm_step * T), _int(T), _int(pr + at + sm), 5, _int(T - 1), 1))
    return ("An LSTM layer of 1024 units on an input of width 1024 (8 H (H + D) + 17 H = "
            f"{_int(lstm_step)} FLOPs per step, as in Chapter 5) against one attention layer of width 1024 with 16 heads. "
            "Sequential stages: the longest chain of operations in which each one needs the result of the one before it, five for "
            "attention (the projections, the scores, the softmax, the average, the output projection) whatever T is. Path: "
            "the number of operations between the input at position 1 and the output at position T.\n\n"
            + _table(["T", "LSTM FLOPs", "LSTM sequential stages", "attention layer FLOPs", "attention sequential stages",
                      "LSTM path, position 1 to T", "attention path"], rows))


# ------------------------------------------------------------------------------------------------ a model that learns to look things up

NS_KEYS, NS_VALS = 4, 4
N_PAIRS_SHOWN = 3
VOCAB = NS_KEYS * NS_VALS + NS_KEYS         # 16 pair tokens and 4 query tokens
DM, DK = 8, 4
LR, BETA1, BETA2, EPS_ADAM = 0.01, 0.9, 0.999, 1e-8
BATCH, STEPS, WINDOW = 16, 600, 20
CHECKPOINTS = [50, 150, 300, 600]
N_TEST = 400
RUNS = [0, 1]
MODELS = ["attention", "uniform"]
NAME_LOOKUP_TEST = "self_attention.lookup.test"


def _lookup_example(rng):
    keys = rng.shuffle([0, 1, 2, 3])[:N_PAIRS_SHOWN]
    values = [rng.integer(NS_VALS) for _ in range(N_PAIRS_SHOWN)]
    q = rng.integer(N_PAIRS_SHOWN)
    ids = [NS_VALS * keys[i] + values[i] for i in range(N_PAIRS_SHOWN)] + [NS_KEYS * NS_VALS + keys[q]]
    return ids, values[q]


def _token_name(t):
    return f"Q{t - NS_KEYS * NS_VALS}" if t >= NS_KEYS * NS_VALS else f"K{t // NS_VALS}V{t % NS_VALS}"


def _init_lookup(model, run, nudge):
    g = bookrng.generator(f"self_attention.lookup.init.{model}.{run}")
    p = {"E": [[0.5 * g.normal() for _ in range(DM)] for _ in range(VOCAB)]}
    for name in ("Wq", "Wk", "Wv"):
        p[name] = [[0.5 * g.normal() for _ in range(DK)] for _ in range(DM)]
    p["Wo"] = [[0.5 * g.normal() for _ in range(NS_VALS)] for _ in range(DK)]
    p["E"][0][0] += nudge
    return p


def _lookup_forward(p, ids, uniform):
    X = [p["E"][t] for t in ids]
    Y, cache = mha_forward(X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], 1, 0, uniform)
    return Y[-1], Y, cache


def _train_lookup(job):
    model, run, nudge = job
    uniform = model == "uniform"
    p = _init_lookup(model, run, nudge)
    adam_m = {k: [[0.0] * len(r) for r in v] for k, v in p.items()}
    adam_v = {k: [[0.0] * len(r) for r in v] for k, v in p.items()}
    rng = bookrng.generator(f"self_attention.lookup.train.{run}")
    losses = []
    for step in range(1, STEPS + 1):
        grads = {k: [[0.0] * len(r) for r in v] for k, v in p.items()}
        total = 0.0
        for _ in range(BATCH):
            ids, y = _lookup_example(rng)
            logits, Y, cache = _lookup_forward(p, ids, uniform)
            pr = _softmax(logits)
            total += -math.log(pr[y]) / BATCH
            dY = [[0.0] * NS_VALS for _ in ids]
            dY[-1] = [(pr[c] - (1.0 if c == y else 0.0)) / BATCH for c in range(NS_VALS)]
            g = mha_backward(cache, dY)
            for name in ("Wq", "Wk", "Wv", "Wo"):
                for r, row in enumerate(g[name]):
                    for c, v in enumerate(row):
                        grads[name][r][c] += v
            for t, tok in enumerate(ids):
                for c, v in enumerate(g["X"][t]):
                    grads["E"][tok][c] += v
        losses.append(total)
        for name in p:
            for r in range(len(p[name])):
                for c in range(len(p[name][r])):
                    gv = grads[name][r][c]
                    adam_m[name][r][c] = BETA1 * adam_m[name][r][c] + (1 - BETA1) * gv
                    adam_v[name][r][c] = BETA2 * adam_v[name][r][c] + (1 - BETA2) * gv * gv
                    mh = adam_m[name][r][c] / (1 - BETA1 ** step)
                    vh = adam_v[name][r][c] / (1 - BETA2 ** step)
                    p[name][r][c] -= LR * mh / (math.sqrt(vh) + EPS_ADAM)
    curve = {c: _sum(losses[c - WINDOW:c]) / WINDOW for c in CHECKPOINTS}
    rng_t = bookrng.generator(NAME_LOOKUP_TEST)
    right = 0
    for _ in range(N_TEST):
        ids, y = _lookup_example(rng_t)
        logits, _, _ = _lookup_forward(p, ids, uniform)
        right += 1 if max(range(NS_VALS), key=lambda c: (logits[c], -c)) == y else 0
    # one shown example: the weights of the query position over the three pairs, and the reorder test
    ids, y = _lookup_example(bookrng.generator("self_attention.lookup.shown"))
    logits, Y, cache = _lookup_forward(p, ids, uniform)
    weights = cache["heads"][0]["P"][-1]
    order = [2, 0, 1, 3]
    logits2 = _lookup_forward(p, [ids[k] for k in order], uniform)[0]
    reorder = max(abs(a - b) for a, b in zip(logits, logits2))
    return {"model": model, "run": run, "nudge": nudge, "curve": curve, "right": right, "weights": weights, "reorder": reorder,
            "pred": max(range(NS_VALS), key=lambda c: (logits[c], -c))}


@lru_cache(maxsize=None)
def lookup_results():
    """{(model, run): (result, [nudged results])}; the 20 trainings are spread over four worker processes."""
    jobs = [(m, r, d) for m in MODELS for r in RUNS for d in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_train_lookup, jobs, chunksize=1)
    out = {}
    for r in res:
        entry = out.setdefault((r["model"], r["run"]), [None, []])
        if r["nudge"] == 0.0:
            entry[0] = r
        else:
            entry[1].append(r)
    return out


def lookup_task():
    shown_ids, shown_y = _lookup_example(bookrng.generator("self_attention.lookup.shown"))
    rng = bookrng.generator("self_attention.lookup.examples")
    ex = [_lookup_example(rng) for _ in range(3)]
    rows = [(" ".join(f"`{_token_name(t)}`" for t in ids), f"`V{y}`") for ids, y in ex]
    params = VOCAB * DM + 3 * DM * DK + DK * NS_VALS
    # a model that sees which values occur but not which key each belongs to: best guess is the commonest value
    best = 0
    for a in range(NS_VALS):
        for b in range(NS_VALS):
            for c in range(NS_VALS):
                best += max([a, b, c].count(v) for v in range(NS_VALS))
    ceiling = best / (3 * NS_VALS ** 3)
    return (f"Ids 0 .. 15 are the pair tokens `K<k>V<v>` (id 4 k + v: key k in 0 .. 3 paired with value v in 0 .. 3) and ids 16 .. "
            "19 are the query tokens `Q<k>` (id 16 + k), so the vocabulary has V = 20 ids. An example is generated by the "
            "draws, in this order: `shuffle([0, 1, 2, 3], rng)`, of which the first three entries are the keys of the three "
            f"pairs; `integer(4)` three times for their values; `integer(3)` for the index of the key that is asked. The input is "
            "the three pair tokens in that order followed by the query token (T = 4); the target is the value of the asked key. "
            f"Three examples from `self_attention.lookup.examples` ({bookrng.seed_of('self_attention.lookup.examples'):#018x}):\n\n"
            + _table(["input (T = 4 tokens)", "target"], rows, "ll")
            + f"\n\nA model that does not use the pairs can only guess one of 4 values: loss ln 4 = {_fx(math.log(4), 4)} nats, "
            f"accuracy 0.25. A model that sees which values occur among the three pairs but cannot tell which key each belongs to can "
            f"always answer with the commonest value. The asked pair is one of the three with equal chance, so for a triple of values "
            "the best chance is (the count of the commonest value) / 3; averaged over the 64 equally likely triples it is "
            f"{best} / {3 * NS_VALS ** 3} = {_fx(ceiling, 4)}, the most such a model can reach. The model is the one-layer attention of the text: embedding table E [{VOCAB}, {DM}] "
            f"(row by row, `0.5 * normal()`), then W_Q, W_K, W_V [{DM}, {DK}] and the readout W_O [{DK}, 4] "
            f"(each row by row, `0.5 * normal()`), {_int(params)} parameters in all, causal mask, one head; "
            "the logits are the output of the last position times W_O and the loss is the softmax cross-entropy of that position only.")


def _stable_cell(model, run, f, digits):
    base, twins = lookup_results()[(model, run)]
    return stable.text(f(base), [f(t) for t in twins], lambda x: _fx(x, digits))


def lookup_training():
    rows = []
    for model in MODELS:
        for run in RUNS:
            cells = [_stable_cell(model, run, lambda r, c=c: r["curve"][c], 4) for c in CHECKPOINTS]
            acc = _stable_cell(model, run, lambda r: r["right"] / N_TEST, 3)
            rows.append((model, run, f"{bookrng.seed_of(f'self_attention.lookup.init.{model}.{run}'):#018x}", *cells, acc))
    return (f"Training: Adam (learning rate {LR}, beta1 {BETA1}, beta2 {BETA2}, epsilon {EPS_ADAM:g}, with the bias correction of "
            f"Volume III), minibatches of {BATCH} fresh examples, {STEPS} updates, no clipping. Examples come from the generator "
            "`self_attention.lookup.train.<run>`, so the two models of a run see the same ones; the parameters are drawn "
            "from `self_attention.lookup.init.<model>.<run>` (seeds in the table; the uniform model draws the same parameters, "
            "including the W_Q and W_K it does not use). The 'uniform' model replaces every attention weight by 1 over the positions the "
            "mask allows, so the logits are W_O applied to the mean of the four value vectors x_t W_V of the whole input (weight 1/4 "
            "on each of the four positions, the query token included). The parameters that train are E, W_V and W_O; W_Q and W_K "
            "stay at their initial values, because the scores do not reach the output and the backward pass sets the gradient "
            "of the scores to zero; the gradient of E is the one through W_V alone. A loss column is the mean of the {WINDOW} batch losses that end at that step. Accuracy is the "
            f"fraction of the {N_TEST} test examples of `{NAME_LOOKUP_TEST}` ({bookrng.seed_of(NAME_LOOKUP_TEST):#018x}) "
            "whose largest logit (first on a tie) is the target. Rounding noise grows into any difference in a trained network. "
            + stable.RULE + "\n\n"
            + _table(["model", "run", "seed of the initial parameters"] + [f"loss, step {c}" for c in CHECKPOINTS] + ["test accuracy"],
                     rows, "llr" + "r" * 5)
            )


def lookup_attention():
    rows = []
    for model in MODELS:
        base, twins = lookup_results()[(model, 0)]
        w = [_stable_cell(model, 0, lambda r, j=j: r["weights"][j], 2) for j in range(4)]
        reorder = _below(max(r["reorder"] for r in [base, *twins]), 1e-9)
        rows.append((model, *w, reorder))
    ids, y = _lookup_example(bookrng.generator("self_attention.lookup.shown"))
    names = " ".join(f"`{_token_name(t)}`" for t in ids)
    return (f"Run 0 on the example from `self_attention.lookup.shown` ({bookrng.seed_of('self_attention.lookup.shown'):#018x}): "
            f"the input {names}, so the target is `V{y}`. The row of attention weights of the query position (the last) over "
            "the positions 1 to 4, and the largest difference of the logits when the three pairs are reordered "
            "(pairs 3, 1, 2 in that order, query last). Weights are printed by the "
            "stable-digits rule.\n\n"
            + _table(["model", "weight on pair 1", "pair 2", "pair 3", "the query itself", "change of the logits when the pairs are reordered"],
                     rows, "lrrrrl"))
