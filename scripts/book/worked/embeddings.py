# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Embeddings": from a token id to a vector, and from a vector to a meaning.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own
description and independent of any engine code. Each public function returns a markdown string that the chapter
embeds between `<!-- worked: embeddings.<name> -->` markers; a few `*_data` functions supply the numbers of the
chapter's figures.

* the lookup of rows of a six-row table and the same result as a product with a one-hot matrix;
* the gradient of the lookup (scatter-add), checked against central differences, and the "assign" mistake;
* lazy and dense Adam on one table entry, and what each costs in bytes;
* tied input and output tables: the two contributions to one gradient, checked the same way;
* a toy grammar, a next-token model with a separate output table trained by full-batch gradient descent, and the
  cosine similarities, neighbours and analogies of its learned rows;
* counts: parameters of a table against a whole model, the scale of the rows at initialisation.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed.

Swedish Embedded AB implements solutions for sizing and training the token tables of language models. If your
team needs expertise in embedding tables, tied output heads and their memory cost, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache

import bookrng

# --------------------------------------------------------------------------------------------- formatting


def _n(x, nd=4):
    s = f"{x:.{nd}f}"
    return f"{0.0:.{nd}f}" if float(s) == 0 else s


def _row(r, nd=4):
    return "(" + ", ".join(_n(x, nd) for x in r) + ")"


def _table(headers, rows, align=None):
    align = align or "l" * len(headers)
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _int(n):
    return f"{n:,}"


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cos(a, b):
    return _dot(a, b) / math.sqrt(_dot(a, a) * _dot(b, b))


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(a):
    return [list(r) for r in zip(*a)]


# --------------------------------------------------------------------------------------------- the small table

TOKENS = ["the", "cat", "dog", "sat", "ran", "."]
V, D = len(TOKENS), 3
TABLE_NAME = "embeddings.table"
TABLE_SCALE = 0.5


@lru_cache(maxsize=None)
def _table6():
    rng = bookrng.generator(TABLE_NAME)
    return tuple(tuple(round(TABLE_SCALE * rng.normal(), 2) + 0.0 for _ in range(D)) for _ in range(V))


def table6():
    return [list(r) for r in _table6()]


def _table_rows(E):
    return [(i, f"`{TOKENS[i]}`", *(_n(x, 2) for x in E[i])) for i in range(V)]


def _lookup(E, ids):
    return [list(E[i]) for i in ids]


def _onehot_matrix(ids, n):
    return [[1.0 if j == i else 0.0 for j in range(n)] for i in ids]


LOOKUP_IDS = (0, 1, 3)


def lookup():
    E = table6()
    ids = LOOKUP_IDS
    X = _lookup(E, ids)
    out = [
        _seed_line(TABLE_NAME),
        f"The table has V = {V} rows (one per token) and D = {D} columns. It is filled row by row, three draws a row, "
        f"each entry `{TABLE_SCALE} * rng.normal()` rounded to two decimals (18 draws of `normal`, so nine pairs of Box-Muller).",
        "",
        _table(["id", "token", "col 0", "col 1", "col 2"], _table_rows(E), "rlrrr"),
        "",
        f"The sentence `the cat sat` has the ids {tuple(ids)}. Lookup, one row of the table per id:",
        "",
        _table(["position t", "id", "token", "row of the table"],
               [(t, ids[t], f"`{TOKENS[ids[t]]}`", _row(X[t], 2)) for t in range(len(ids))], "rrlr"),
        "",
        f"The lookup reads T D = {len(ids)} * {D} = {len(ids) * D} numbers and multiplies nothing.",
    ]
    return "\n".join(out)


def onehot_product():
    E = table6()
    ids = LOOKUP_IDS
    X = _lookup(E, ids)
    N = _onehot_matrix(ids, V)
    P = _matmul(N, E)
    assert P == X
    out = [
        "The table is the one of the first example and the sentence is `the cat sat`, ids " + str(tuple(ids)) + ". "
        "N has one row per position, with a 1 in the column of that position's id:",
        "",
        _table(["position t", "row of N (V = 6 entries)", "row of N times the table"],
               [(t, "(" + ", ".join(f"{int(x)}" for x in N[t]) + ")", _row(P[t], 2)) for t in range(len(ids))], "rlr"),
        "",
        f"The three rows are the rows the lookup returned, entry for entry (largest difference 0): in every sum all terms but one are an entry times 0. "
        f"The product counts T V D = {len(ids)} * {V} * {D} = {len(ids) * V * D} multiply-adds, "
        f"of which {len(ids) * V * D - len(ids) * D} multiply by zero.",
    ]
    return "\n".join(out)


def lookup_figure_data():
    """The pieces the figure of the lookup draws: table, ids, the rows picked."""
    E = table6()
    return {"tokens": TOKENS, "table": E, "ids": list(LOOKUP_IDS), "rows": _lookup(E, LOOKUP_IDS)}


# Named configurations. The Qwen3 and GPT shapes are the ones brain's `QwenConfig` and `GptConfig` lay out.
CONFIGS = {
    "toy": dict(V=6, D=3),
    "gpt_tiny": dict(V=65, D=32, layers=2, ff=128, block=64),
    "gpt_small": dict(V=50257, D=768, layers=12, ff=3072, block=1024),
    "qwen3_0_6b": dict(V=151936, D=1024, layers=28, heads=16, kv_heads=8, head_dim=128, ff=3072),
    "qwen3_1_7b": dict(V=151936, D=2048, layers=28, heads=16, kv_heads=8, head_dim=128, ff=6144),
    "qwen3_4b": dict(V=151936, D=2560, layers=36, heads=32, kv_heads=8, head_dim=128, ff=9728),
}


def lookup_cost():
    rows = []
    cases = [("the table above", 6, 3, 3, 4), ("GPT, tiny", 65, 32, 64, 4), ("Qwen3, 0.6B", 151936, 1024, 1024, 2)]
    for name, v, d, t, b in cases:
        macs = t * v * d
        table_bytes = v * d * b
        n_bytes = t * v * b
        pick = t * d * b + 4 * t
        rows.append((name, _int(v), _int(d), _int(t), b, _int(macs), _int(table_bytes), _int(n_bytes), _int(pick)))
    out = [
        "T is the number of positions, B the bytes of one table entry (4 for `f32`, 2 for `bf16`). A multiply-add is "
        "two FLOPs. The product reads the whole table and the matrix N of T rows of V entries; the lookup reads T rows of "
        "D entries and T ids of 4 bytes.",
        "",
        _table(["case", "V", "D", "T", "B", "multiply-adds of N times table (T V D)", "bytes of the table (V D B)",
                "bytes of N (T V B)", "bytes the lookup reads (T D B + 4 T)"], rows, "lrrrrrrrr"),
    ]
    last = cases[-1]
    t, v, d, b = last[3], last[1], last[2], last[4]
    out += ["", f"For the last row the product needs {_int(2 * t * v * d)} FLOPs and reads at least "
                f"{_int(v * d * b)} bytes of table; the lookup needs no FLOPs and reads {_int(t * d * b + 4 * t)} bytes, "
                f"{(v * d * b) / (t * d * b + 4 * t):.1f} times fewer."]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------- the gradient

GRAD_IDS = (1, 3, 1, 5)    # cat sat cat .
GRAD_TARGET = [[0.50, -0.25, 0.00], [0.00, 0.25, -0.50], [-0.25, 0.50, 0.25], [0.25, 0.00, 0.50]]
FD_H = 1e-5


def _embed_loss(E, ids, target):
    X = _lookup(E, ids)
    return 0.5 * sum((X[t][j] - target[t][j]) ** 2 for t in range(len(ids)) for j in range(len(E[0])))


def scatter_add(dX, ids, n_rows):
    """The embedding gradient: row i of the result is the sum of the rows dX[t] with ids[t] == i."""
    dE = [[0.0] * len(dX[0]) for _ in range(n_rows)]
    for t, i in enumerate(ids):
        for j in range(len(dX[0])):
            dE[i][j] += dX[t][j]
    return dE


def _central(f, E, h=FD_H):
    out = [[0.0] * len(E[0]) for _ in E]
    for i in range(len(E)):
        for j in range(len(E[0])):
            old = E[i][j]
            E[i][j] = old + h
            fp = f(E)
            E[i][j] = old - h
            fm = f(E)
            E[i][j] = old
            out[i][j] = (fp - fm) / (2 * h)
    return out


def _maxdiff(A, B):
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def gradient_data():
    E = table6()
    ids = list(GRAD_IDS)
    X = _lookup(E, ids)
    dX = [[X[t][j] - GRAD_TARGET[t][j] for j in range(D)] for t in range(len(ids))]
    dE = scatter_add(dX, ids, V)
    return {"ids": ids, "X": X, "dX": dX, "dE": dE, "tokens": TOKENS}


def gradient():
    E = table6()
    d = gradient_data()
    ids, X, dX, dE = d["ids"], d["X"], d["dX"], d["dE"]
    T = len(ids)
    loss = _embed_loss(E, ids, GRAD_TARGET)
    num = _central(lambda M: _embed_loss(M, ids, GRAD_TARGET), [list(r) for r in E])
    err = _maxdiff(dE, num)
    assert err < 1e-8
    # the one-hot view: dE = N^T dX
    via_n = _matmul(_transpose(_onehot_matrix(ids, V)), dX)
    assert _maxdiff(via_n, dE) < 1e-15
    # the mistake: assign where the rule adds (the last use wins)
    dE_assign = [[0.0] * D for _ in range(V)]
    for t, i in enumerate(ids):
        dE_assign[i] = list(dX[t])
    lost = [dE[1][j] - dE_assign[1][j] for j in range(D)]
    count = [ids.count(i) for i in range(V)]
    out = [
        "The table is the one above. The input is `cat sat cat .`, ids " + str(tuple(ids)) + ", so `cat` (id 1) is used twice. "
        "The loss is L = 1/2 sum over t and j of (X[t, j] - R[t, j])^2, with the target R below, "
        "so the gradient arriving at the lookup is dX = X - R.",
        "",
        _table(["position t", "id", "X[t] (row of the table)", "R[t]", "dX[t] = X[t] - R[t]"],
               [(t, ids[t], _row(X[t], 2), _row(GRAD_TARGET[t], 2), _row(dX[t], 2)) for t in range(T)], "rrlll"),
        "",
        f"L = {_n(loss, 4)}. The gradient of the table, row by row:",
        "",
        _table(["id", "token", "uses", "dW[i] = sum of dX[t] with ids[t] = i"],
               [(i, f"`{TOKENS[i]}`", count[i], _row(dE[i], 2)) for i in range(V)], "rlrr"),
        "",
        f"Row 1 is dX[0] + dX[2] = {_row(dX[0], 2)} + {_row(dX[2], 2)} = {_row(dE[1], 2)}. Rows 0, 2 and 4 were never used and are zero. "
        f"The same table from the product N^T dX (N the matrix of the previous example): largest difference from the loop above, below 1e-15. "
        f"Largest difference from a central difference over all {V * D} entries (step h = 1e-5, applied to L): below 1e-8.",
        "",
        f"A loop that assigns, `dW[ids[t]] = dX[t]`, leaves row 1 equal to dX[2] = {_row(dE_assign[1], 2)}; the correct row minus "
        f"this wrong row is {_row(lost, 2)}, which is exactly dX[0], the use that was overwritten.",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------- Adam on a row

ADAM = dict(lr=0.1, b1=0.9, b2=0.999, eps=1e-8)
ADAM_GRADS = (0.5, 0.0, 0.0, 0.0, 0.0, 0.5)      # the token is in the batch at steps 1 and 6 only
ADAM_W0 = 0.30


def _adam_step(w, m, v, g, t):
    a = ADAM
    m = a["b1"] * m + (1 - a["b1"]) * g
    v = a["b2"] * v + (1 - a["b2"]) * g * g
    mh = m / (1 - a["b1"] ** t)
    vh = v / (1 - a["b2"] ** t)
    dw = a["lr"] * mh / (math.sqrt(vh) + a["eps"])
    return w - dw, m, v, dw


def adam_rows():
    wd, md, vd = ADAM_W0, 0.0, 0.0
    wl, ml, vl = ADAM_W0, 0.0, 0.0
    rows = []
    for t, g in enumerate(ADAM_GRADS, start=1):
        wd2, md, vd, dwd = _adam_step(wd, md, vd, g, t)
        if g != 0.0:
            wl2, ml, vl, dwl = _adam_step(wl, ml, vl, g, t)
        else:
            wl2, dwl = wl, 0.0
        wd, wl = wd2, wl2
        rows.append((t, _n(g, 2), _n(md, 6), _n(vd, 8), _n(-dwd, 5), _n(wd, 5), _n(-dwl, 5), _n(wl, 5)))
    a = ADAM
    return "\n".join([
        f"One entry of a table, starting at w = {ADAM_W0}. Adam with learning rate {a['lr']}, beta1 = {a['b1']}, beta2 = {a['b2']}, "
        f"epsilon = {a['eps']:g}, no weight decay, update w <- w - lr * mhat / (sqrt(vhat) + epsilon) with the bias corrections "
        f"mhat = m / (1 - beta1^t) and vhat = v / (1 - beta2^t). Its token is in the batch at steps 1 and 6 (gradient 0.5) and absent at "
        f"steps 2 to 5 (gradient 0). *Dense* Adam runs the rule on every entry at every step; *lazy* Adam runs it only at the steps where "
        f"the token is present, with the global step t in the corrections, and leaves m and v alone otherwise. m and v are those of dense Adam.",
        "",
        _table(["t", "g", "m (dense)", "v (dense)", "dense change of w", "w (dense)", "lazy change of w", "w (lazy)"], rows, "rrrrrrrr"),
        "",
        f"At the end, lazy Adam holds m = {_n(ml, 6)} and v = {_n(vl, 8)}, dense Adam m = {_n(md, 6)} and v = {_n(vd, 8)}.",
    ])


def update_cost():
    rows = []
    for name, cfg, tokens in (("Qwen3, 0.6B", CONFIGS["qwen3_0_6b"], 1024), ("Qwen3, 4B", CONFIGS["qwen3_4b"], 1024)):
        v, d = cfg["V"], cfg["D"]
        params = v * d
        dense = 7 * 4 * params
        lazy = 7 * 4 * tokens * d
        rows.append((name, _int(params), _int(16 * params), _int(dense), _int(lazy), f"{dense / lazy:.1f}"))
    return "\n".join([
        "The table of each model, with `f32` values, gradients and moments (4 bytes each). A step of Adam reads w, g, m and v and writes "
        "w, m and v: 7 words per entry. The lazy column assumes at most 1,024 distinct tokens in the batch (a batch of 1,024 positions "
        "has no more), each row D entries.",
        "",
        _table(["table of", "entries V D", "bytes held: w, g, m, v (16 per entry)", "bytes moved per step, dense", "bytes moved per step, lazy (at most)", "ratio"],
               rows, "lrrrrr"),
    ])


# --------------------------------------------------------------------------------------------- tied head

TIED_IDS = (1, 3, 1)         # cat sat cat
TIED_TARGETS = (3, 1, 5)     # sat cat .
TIED_WH = [[0.50, -0.25, 0.00], [0.25, 0.50, -0.25], [0.00, 0.25, 0.50]]


def _softmax(z):
    m = max(z)
    ex = [math.exp(x - m) for x in z]
    s = sum(ex)
    return [e / s for e in ex]


def _tied_forward(E):
    ids, tg = TIED_IDS, TIED_TARGETS
    X = _lookup(E, ids)
    H = [[_dot(TIED_WH[r], x) for r in range(D)] for x in X]        # h = Wh x
    logits = [[_dot(h, E[i]) for i in range(V)] for h in H]
    P = [_softmax(z) for z in logits]
    loss = -sum(math.log(P[t][tg[t]]) for t in range(len(ids))) / len(ids)
    return X, H, logits, P, loss


def tied_parts(E):
    ids, tg = TIED_IDS, TIED_TARGETS
    T = len(ids)
    X, H, logits, P, loss = _tied_forward(E)
    dz = [[(P[t][i] - (1.0 if i == tg[t] else 0.0)) / T for i in range(V)] for t in range(T)]
    dH = [[sum(dz[t][i] * E[i][j] for i in range(V)) for j in range(D)] for t in range(T)]
    dE_head = _matmul(_transpose(dz), H)
    dX = [[sum(TIED_WH[r][j] * dH[t][r] for r in range(D)) for j in range(D)] for t in range(T)]
    dE_embed = scatter_add(dX, ids, V)
    return loss, dz, dH, dX, dE_head, dE_embed


def tied():
    E = table6()
    loss, dz, dH, dX, dE_head, dE_embed = tied_parts(E)
    total = [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(dE_head, dE_embed)]
    num = _central(lambda M: _tied_forward(M)[4], [list(r) for r in E])
    err = _maxdiff(total, num)
    assert err < 1e-8
    err_head, err_embed = _maxdiff(dE_head, num), _maxdiff(dE_embed, num)
    assert err_head > 1e-3 and err_embed > 1e-3
    assert _maxdiff([[a + b for a, b in zip(ra, rb)] for ra, rb in zip(dE_head, dE_embed)], total) == 0
    X, H, logits, P, _ = _tied_forward(E)
    t_rows = [(t, f"`{TOKENS[TIED_IDS[t]]}`", f"`{TOKENS[TIED_TARGETS[t]]}`", _row(H[t], 4),
               _n(P[t][TIED_TARGETS[t]], 4)) for t in range(len(TIED_IDS))]
    out = [
        "The table is the one of the first example. The input is `cat sat cat` (ids 1, 3, 1) and the targets are the next tokens `sat cat .` "
        "(ids 3, 1, 5). A stand-in for the network turns each looked-up row x into h = Wh x with",
        "",
        "    Wh = " + str([list(r) for r in TIED_WH]),
        "",
        "and the head is tied: the logits are z = h E^T, row t of the table acting as the weight vector of token t. "
        "The loss is the mean over the T = 3 positions of -log softmax(z)[target].",
        "",
        _table(["t", "input", "target", "h[t] = Wh x[t]", "probability of the target"], t_rows, "rllrr"),
        "",
        f"L = {_n(loss, 4)}. Gradient with respect to the table, split into its two sources, and the total (rounded to 4 decimals):",
        "",
        _table(["id", "token", "head part: dz^T H", "embedding part: scatter-add of Wh^T dh", "total"],
               [(i, f"`{TOKENS[i]}`", _row(dE_head[i]), _row(dE_embed[i]), _row(total[i])) for i in range(V)], "rlrrr"),
        "",
        f"Largest difference of the total from a central difference of L over all {V * D} entries (h = 1e-5): below 1e-8. "
        f"With the head part alone it is {err_head:.2f}, with the embedding part alone {err_embed:.2f}. "
        f"Row 0 (`the`) is never an input but is a candidate output at every position, so its gradient is all head part.",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------- parameter counts


def params_qwen(c, tied_head=True):
    v, d, L = c["V"], c["D"], c["layers"]
    hq, hkv, hd, ff = c["heads"] * c["head_dim"], c["kv_heads"] * c["head_dim"], c["head_dim"], c["ff"]
    per = 2 * d + hq * d + 2 * hkv * d + d * hq + 2 * hd + 2 * ff * d + d * ff
    return v * d + L * per + d + (0 if tied_head else v * d)


def params_gpt(c, tied_head=False):
    v, d, L, ff, blk = c["V"], c["D"], c["layers"], c["ff"], c["block"]
    per = 4 * d + 3 * d * d + 3 * d + d * d + d + ff * d + ff + d * ff + d
    return v * d + blk * d + L * per + 2 * d + (0 if tied_head else v * d)


def _model_counts(name):
    c = CONFIGS[name]
    if name.startswith("qwen"):
        return params_qwen(c, True), params_qwen(c, False)
    return params_gpt(c, True), params_gpt(c, False)


MODEL_LABELS = [("gpt_tiny", "GPT, tiny (brain's `GptConfig::tiny`)"), ("gpt_small", "GPT, 12 layers of width 768"),
                ("qwen3_0_6b", "Qwen3, 0.6B"), ("qwen3_1_7b", "Qwen3, 1.7B"), ("qwen3_4b", "Qwen3, 4B")]


def params():
    a_rows, b_rows = [], []
    for key, label in MODEL_LABELS:
        c = CONFIGS[key]
        tied_total, sep_total = _model_counts(key)
        table = c["V"] * c["D"]
        a_rows.append((label, _int(c["V"]), _int(c["D"]), _int(table), _int(tied_total)))
        b_rows.append((label, _int(sep_total), _int(sep_total - tied_total), f"{100 * table / tied_total:.1f}%",
                       f"{100 * 2 * table / sep_total:.1f}%", _int(2 * table)))
    cfg_rows = []
    for key, label in MODEL_LABELS:
        c = CONFIGS[key]
        if key.startswith("qwen"):
            cfg_rows.append((label, c["layers"], c["heads"], c["kv_heads"], c["head_dim"], _int(c["ff"]), "-"))
        else:
            cfg_rows.append((label, c["layers"], "-", "-", "-", _int(c["ff"]), c["block"]))
    return "\n".join([
        "Counted from the layouts of brain's `QwenConfig` and `GptConfig`; a shared table is counted once and the separate head adds a "
        "second V D. The configurations, with D, V and the model's name in the tables below (query heads, key-value heads and head width "
        "are those of grouped-query attention; `ff` is the width of the feed-forward layer; the position table of the GPT has `block` rows):",
        "",
        _table(["model", "layers", "query heads", "key-value heads", "head width", "ff", "block"], cfg_rows, "lrrrrrr"),
        "",
        "Each layer of the Qwen3 models holds two RMS-norm scales (D each), a query projection (D by heads x head width), a key and a value "
        "projection (D by key-value heads x head width each), an output projection (heads x head width by D), two scales of head width "
        "(the query and key norms), and a gated feed-forward layer of three bias-free matrices (gate and up: D by ff each; down: ff by D); "
        "after the layers come one final RMS-norm scale (D). Every other weight of these models is bias-free. Each layer of the GPT holds "
        "two layer norms (scale and bias, 2 D each), a combined query-key-value projection (D by 3 D) with a bias of 3 D, an output "
        "projection (D by D) with a bias of D, and two feed-forward matrices (D by ff and ff by D) with biases of ff and D; after the layers "
        "come one final layer norm (2 D), and before them the token table (V D) and the position table (block D). The number of attention "
        "heads of the GPT does not change the count.",
        "",
        _table(["model", "V", "D", "table V D", "parameters with one shared table"], a_rows, "lrrrr"),
        "",
        _table(["model", "parameters with a separate head", "difference (= V D)", "share of the table, shared",
                "share of table plus head, separate", "bytes of the shared table in `bf16`"], b_rows, "lrrrrr"),
    ])


def params_share(key):
    tied_total, _ = _model_counts(key)
    c = CONFIGS[key]
    return c["V"] * c["D"], tied_total


# --------------------------------------------------------------------------------------------- scale

SCALE_NAME = "embeddings.scale"
SCALE_V, SCALE_D, SCALE_N = 256, 64, 100
SCALE_STDS = (0.02, 0.1, 0.5, 1.0)


def scale():
    rows = []
    for s in SCALE_STDS:
        rng = bookrng.generator(SCALE_NAME)
        W = [[s * rng.normal() for _ in range(SCALE_D)] for _ in range(SCALE_V)]
        norms = [math.sqrt(_dot(r, r)) for r in W]
        loss, zs = 0.0, []
        for _ in range(SCALE_N):
            h = [rng.normal() for _ in range(SCALE_D)]
            tgt = rng.integer(SCALE_V)
            z = [_dot(h, W[i]) for i in range(SCALE_V)]
            zs += z
            m = max(z)
            loss += math.log(sum(math.exp(x - m) for x in z)) + m - z[tgt]
        mean = sum(zs) / len(zs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in zs) / (len(zs) - 1))
        rows.append((s, _n(s * math.sqrt(SCALE_D), 3), _n(sum(norms) / len(norms), 3), _n(sd, 3), _n(loss / SCALE_N, 3)))
    return "\n".join([
        _seed_line(SCALE_NAME),
        f"A table of V = {SCALE_V} rows and D = {SCALE_D} columns with entries s * rng.normal(), drawn row by row ({SCALE_V * SCALE_D} draws), "
        f"then {SCALE_N} pairs of a hidden vector h (D draws of `normal`, so entries of standard deviation 1, as after a normalisation layer) "
        f"and a target token (`rng.integer({SCALE_V})`), drawn in that order. The logits are z = h W^T over the V rows; "
        f"each scale s reuses the same draws. The loss is the mean over the {SCALE_N} pairs of -log softmax(z)[target]; "
        f"a model that knows nothing has loss log V = {_n(math.log(SCALE_V), 3)}.",
        "",
        _table(["s", "s sqrt(D)", "mean row length", "standard deviation of the logits", "mean loss"], rows, "rrrrr"),
    ])


# --------------------------------------------------------------------------------------------- the toy grammar

VOCAB = ["<s>", "cat", "dog", "fox", "cats", "dogs", "foxes", "sees", "chases", "see", "chase", "ball", "bone", "fish", "."]
NV = len(VOCAB)
VIDX = {t: i for i, t in enumerate(VOCAB)}
SING, PLUR, VERB, OBJ = (1, 2, 3), (4, 5, 6), (7, 8, 9, 10), (11, 12, 13)
CLASSES = [("singular nouns", SING), ("plural nouns", PLUR), ("verbs", VERB), ("objects", OBJ)]
CORPUS_NAME, N_SENT = "embeddings.corpus", 60
TRAIN_NAME = "embeddings.train"
DIM, INIT_SD, LR, STEPS = 6, 0.5, 2.0, 300
SHOW_STEPS = (0, 10, 30, 100, 300)


@lru_cache(maxsize=None)
def _corpus():
    rng = bookrng.generator(CORPUS_NAME)
    sents, stream = [], []
    for _ in range(N_SENT):
        n = rng.integer(6)
        v = rng.integer(2)
        o = rng.integer(3)
        verb = ("sees", "chases")[v] if n < 3 else ("see", "chase")[v]
        words = [VOCAB[1 + n], verb, VOCAB[11 + o], "."]
        sents.append(" ".join(words))
        stream += ["<s>"] + words
    return tuple(sents), tuple(VIDX[w] for w in stream)


def _pairs():
    ids = _corpus()[1]
    return list(zip(ids[:-1], ids[1:]))


def _bigram_floor():
    pairs = _pairs()
    n = len(pairs)
    cnt, tot = {}, {}
    for a, b in pairs:
        cnt[(a, b)] = cnt.get((a, b), 0) + 1
        tot[a] = tot.get(a, 0) + 1
    return -sum(c * math.log(c / tot[a]) for (a, b), c in cnt.items()) / n


def corpus():
    sents, stream = _corpus()
    pairs = _pairs()
    classes = "; ".join(f"{name}: " + ", ".join(f"`{VOCAB[i]}`" for i in ids) for name, ids in CLASSES)
    return "\n".join([
        _seed_line(CORPUS_NAME),
        f"The vocabulary has V = {NV} tokens with ids 0 to {NV - 1} in this order: " + ", ".join(f"{i} `{t}`" for i, t in enumerate(VOCAB)) + ".",
        f"A sentence is a noun, a verb that agrees with it in number, an object and a full stop. Draw order for each of the {N_SENT} sentences: "
        f"`n = rng.integer(6)` picks the noun (ids 1 to 6 in the order above, so n = 0 is `cat` and n = 5 is `foxes`), "
        f"`v = rng.integer(2)` picks the verb of the noun's number (for a singular noun `sees` or `chases`, for a plural one `see` or `chase`), "
        f"`o = rng.integer(3)` picks the object (`ball`, `bone` or `fish`).",
        "",
        "The corpus, one sentence after another:",
        "",
        "    " + " ".join(sents[i] for i in range(0, 20)),
        "    " + " ".join(sents[i] for i in range(20, 40)),
        "    " + " ".join(sents[i] for i in range(40, 60)),
        "",
        f"The training text is the stream in which `<s>` (id 0) is written before every sentence: {len(stream)} tokens. "
        f"A training example is a pair of neighbours (current, next); the stream has N = {len(pairs)} of them. The classes used to read the results: {classes}.",
        f"If a model could store each current token's next-token frequencies exactly, its mean loss would be the conditional entropy of the pairs, "
        f"-(1/N) times the sum over the distinct pairs (a, b) of count(a, b) * log(count(a, b) / count(a)), where count(a) is the number of pairs that start with a: "
        f"{_n(_bigram_floor(), 4)} nats. For orientation, log 2 = {_n(math.log(2), 4)} is the entropy of a fair choice between the two verbs of a noun and "
        f"log 3 = {_n(math.log(3), 4)} that of a fair choice between the three objects.",
    ])


@lru_cache(maxsize=None)
def _train(tied_head):
    pairs = _pairs()
    N = len(pairs)
    rng = bookrng.generator(TRAIN_NAME)
    E = [[INIT_SD * rng.normal() for _ in range(DIM)] for _ in range(NV)]
    if tied_head:
        W = E
    else:
        W = [[INIT_SD * rng.normal() for _ in range(DIM)] for _ in range(NV)]
    E0 = [list(r) for r in E]
    hist = {}
    for step in range(STEPS + 1):
        dE = [[0.0] * DIM for _ in range(NV)]
        dW = [[0.0] * DIM for _ in range(NV)]
        loss = 0.0
        for a, b in pairs:
            e = E[a]
            z = [_dot(e, W[j]) for j in range(NV)]
            m = max(z)
            ex = [math.exp(x - m) for x in z]
            s = sum(ex)
            loss -= z[b] - m - math.log(s)
            for j in range(NV):
                dz = (ex[j] / s - (1.0 if j == b else 0.0)) / N
                for k in range(DIM):
                    dE[a][k] += dz * W[j][k]
                    dW[j][k] += dz * e[k]
        if step in SHOW_STEPS:
            hist[step] = loss / N
        if step == STEPS:
            break
        for i in range(NV):
            for k in range(DIM):
                if tied_head:
                    E[i][k] -= LR * (dE[i][k] + dW[i][k])
                else:
                    E[i][k] -= LR * dE[i][k]
                    W[i][k] -= LR * dW[i][k]
    return E0, [list(r) for r in E], hist


def _class_mean(E, c1, c2):
    vals = [_cos(E[a], E[b]) for a in c1 for b in c2 if a != b] if c1 == c2 else [_cos(E[a], E[b]) for a in c1 for b in c2]
    return sum(vals) / len(vals)


def _class_matrix(E):
    names = [n for n, _ in CLASSES]
    rows = []
    for i, (n1, c1) in enumerate(CLASSES):
        rows.append((n1, *(_n(_class_mean(E, c1, c2), 3) for _, c2 in CLASSES)))
    return names, rows


def training():
    E0, E1, hist = _train(False)
    names, before = _class_matrix(E0)
    _, after = _class_matrix(E1)
    pairs_list = [("cat", "dog"), ("cat", "fox"), ("dog", "fox"), ("cats", "dogs"), ("sees", "chases"), ("sees", "see"), ("ball", "fish"),
                  ("cat", "cats"), ("cat", "fish"), ("cat", "sees")]
    prow = [(f"`{a}`, `{b}`", _n(_cos(E0[VIDX[a]], E0[VIDX[b]]), 3), _n(_cos(E1[VIDX[a]], E1[VIDX[b]]), 3)) for a, b in pairs_list]
    return "\n".join([
        _seed_line(TRAIN_NAME),
        f"Model: logits z = W_out e, where e is row x of the embedding table (the current token x) and W_out is a second table of the same "
        f"shape; the next token is predicted with softmax(z). Both tables have {NV} rows and D = {DIM} columns. Start: the {NV * DIM} entries "
        f"of the embedding table are drawn first, row by row, as {INIT_SD} * `rng.normal()`, then the entries of W_out the same way. "
        f"Loss: the mean over the N = {len(_pairs())} pairs of -log softmax(z)[next]. Training: {STEPS} steps of full-batch gradient descent, "
        f"learning rate {LR}, every pair in every step, no regularisation.",
        "",
        _table(["step", "mean loss"], [(s, _n(hist[s], 4)) for s in SHOW_STEPS], "rr"),
        "",
        f"Mean cosine similarity of the rows of the embedding table, by class, before training (step 0). "
        f"The diagonal is the mean over pairs of different tokens of one class:",
        "",
        _table([""] + names, before, "l" + "r" * len(names)),
        "",
        "After 300 steps:",
        "",
        _table([""] + names, after, "l" + "r" * len(names)),
        "",
        "Some individual pairs:",
        "",
        _table(["pair", "cosine before", "cosine after"], prow, "lrr"),
    ])


def tied_comparison():
    rows = []
    for label, tied_head in (("separate output table", False), ("tied (the output table is the embedding table)", True)):
        E0, E1, hist = _train(tied_head)
        _, m = _class_matrix(E1)
        within = [float(m[i][1 + i]) for i in range(len(CLASSES))]
        rows.append((label, _int(NV * DIM * (1 if tied_head else 2)), _n(hist[STEPS], 4), *(_n(w, 3) for w in within)))
    return "\n".join([
        f"The same corpus, the same {STEPS} steps of full-batch gradient descent with learning rate {LR} (as in the previous training block) "
        f"and the same starting embedding table (the {NV * DIM} entries drawn first); the tied model has no second table "
        f"(its logits are z = E e, and the update of the table is E <- E - {LR} (dE + dW), the sum of the lookup's gradient and the head's, "
        f"as in the previous section). The final loss is the mean loss over the pairs at the last step. "
        f"Mean cosine of different tokens of one class after training:",
        "",
        _table(["model", "parameters", "final loss", "singular nouns", "plural nouns", "verbs", "objects"], rows, "lrrrrrr"),
    ])


def _neighbours(E, q, k=3):
    qi = VIDX[q]
    sc = sorted(((_cos(E[qi], E[j]), j) for j in range(NV) if j != qi), key=lambda p: (-p[0], p[1]))
    return [(VOCAB[j], c) for c, j in sc[:k]]


def neighbours():
    E0, E1, _ = _train(False)
    rows = []
    for q in ("cat", "cats", "sees", "ball", "<s>"):
        b = ", ".join(f"`{t}` {_n(c, 2)}" for t, c in _neighbours(E0, q))
        a = ", ".join(f"`{t}` {_n(c, 2)}" for t, c in _neighbours(E1, q))
        rows.append((f"`{q}`", b, a))
    return "\n".join([
        "The three rows of largest cosine similarity to each query (the query excluded; ties, which do not occur here, would go to the smaller id), "
        "from the model of the previous example before and after training.",
        "",
        _table(["query", "before training", "after training"], rows, "lll"),
    ])


ANALOGY_PAIRS = [("cat", "cats"), ("dog", "dogs"), ("fox", "foxes")]


def _analogy(E, a, b, c, k=3):
    q = [E[VIDX[b]][j] - E[VIDX[a]][j] + E[VIDX[c]][j] for j in range(DIM)]
    excl = {VIDX[a], VIDX[b], VIDX[c]}
    sc = sorted(((_cos(q, E[j]), j) for j in range(NV) if j not in excl), key=lambda p: (-p[0], p[1]))
    return [(VOCAB[j], cs) for cs, j in sc[:k]]


def analogies():
    E0, E1, _ = _train(False)
    rows, hit0, hit1, cls0, cls1, total = [], 0, 0, 0, 0, 0
    for a, b in ANALOGY_PAIRS:
        for c, d in ANALOGY_PAIRS:
            if c == a:
                continue
            r0, r1 = _analogy(E0, a, b, c), _analogy(E1, a, b, c)
            total += 1
            hit0 += r0[0][0] == d
            hit1 += r1[0][0] == d
            cls0 += VIDX[r0[0][0]] in PLUR
            cls1 += VIDX[r1[0][0]] in PLUR
            rows.append((f"`{a}` : `{b}` :: `{c}` : ?", f"`{d}`", f"`{r0[0][0]}` {_n(r0[0][1], 2)}", f"`{r1[0][0]}` {_n(r1[0][1], 2)}",
                         "yes" if r1[0][0] == d else "no"))
    vb = _analogy(E1, "sees", "see", "chases")
    vb0 = _analogy(E0, "sees", "see", "chases")
    c_see, c_chases = _cos(E1[VIDX["sees"]], E1[VIDX["see"]]), _cos(E1[VIDX["sees"]], E1[VIDX["chases"]])
    return "\n".join([
        "For a : b :: c : ? the query is the vector e(b) - e(a) + e(c), and the answer is the row of largest cosine similarity to it "
        "among the tokens other than a, b and c. Each of the six questions below takes a singular noun a with its plural b and another singular noun c, "
        "and asks for the plural of c; the best row and its cosine are shown, for the table before and after training.",
        "",
        _table(["question", "expected", "before training", "after training", "after: correct"], rows, "llllr"),
        "",
        f"Top-1 correct: {hit0} of {total} before training, {hit1} of {total} after. Answer is some plural noun: {cls0} of {total} before, {cls1} of {total} after.",
        f"For comparison, `sees` : `see` :: `chases` : ? answers `{vb0[0][0]}` before training and `{vb[0][0]}` after. After training "
        f"cos(`sees`, `see`) = {_n(c_see, 3)} is larger than cos(`sees`, `chases`) = {_n(c_chases, 3)}, so the offset e(`see`) - e(`sees`) is short "
        f"and the query is close to e(`chases`) itself.",
    ])


# --------------------------------------------------------------------------------------------- figure data


def _top2(rows):
    """First two principal directions of the rows (power iteration with deflation on the covariance, divisor N)."""
    n, d = len(rows), len(rows[0])
    mu = [sum(r[j] for r in rows) / n for j in range(d)]
    xc = [[r[j] - mu[j] for j in range(d)] for r in rows]
    C = [[sum(r[i] * r[j] for r in xc) / n for j in range(d)] for i in range(d)]
    dirs = []
    for _ in range(2):
        v = [1.0 / math.sqrt(d) * (1 + 0.1 * j) for j in range(d)]
        for _ in range(500):
            w = [sum(C[i][j] * v[j] for j in range(d)) for i in range(d)]
            nw = math.sqrt(_dot(w, w))
            v = [x / nw for x in w]
        lam = _dot(v, [sum(C[i][j] * v[j] for j in range(d)) for i in range(d)])
        k = max(range(d), key=lambda j: abs(v[j]))
        if v[k] < 0:
            v = [-x for x in v]
        dirs.append((lam, v))
        C = [[C[i][j] - lam * v[i] * v[j] for j in range(d)] for i in range(d)]
    return xc, dirs


def pca_points(stage):
    """2-D PCA coordinates of the embedding rows ("before" or "after" training), and the share of variance kept."""
    E0, E1, _ = _train(False)
    E = E0 if stage == "before" else E1
    xc, dirs = _top2(E)
    pts = [(_dot(r, dirs[0][1]), _dot(r, dirs[1][1])) for r in xc]
    total = sum(_dot(r, r) for r in xc) / len(xc)
    return pts, (dirs[0][0] + dirs[1][0]) / total


def pca_summary():
    rows = []
    for stage in ("before", "after"):
        pts, kept = pca_points(stage)
        rows.append((stage + " training", _n(100 * kept, 1) + "%"))
    return "\n".join([
        "The 15 rows of the embedding table (D = 6) are centred on their mean; C is the covariance of the centred rows with divisor N = 15. "
        "The first direction is found by power iteration on C: start from v = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5) divided by its length, "
        "repeat 500 times v <- C v / norm(C v); its eigenvalue is v . C v. The second direction is the same iteration on C - lambda_1 v v^T. "
        "Each direction is signed so that its entry of largest magnitude is positive, and a row's two coordinates are its dot products "
        "with the directions. The share of the variance kept is (lambda_1 + lambda_2) / trace(C).",
        "",
        _table(["table", "variance kept by the two directions"], rows, "lr"),
    ])


if __name__ == "__main__":
    for f in (lookup, onehot_product, lookup_cost, gradient, adam_rows, update_cost, tied, params, scale, corpus, training, tied_comparison, neighbours, analogies, pca_summary):
        print(f"--- {f.__name__}")
        print(f())
