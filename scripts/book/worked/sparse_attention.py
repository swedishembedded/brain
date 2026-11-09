# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Sparse Attention": how concentrated attention weights are, what fixed sparse patterns keep
and miss, the cost of a learned indexer against dense attention, an indexer trained by distillation toward the dense weights,
the output error of attending to the top few keys, a bucket index over the keys (latent sparse attention), and the traffic of
reading selected keys from a paged cache.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. Each public function without arguments returns a markdown string that the chapter embeds
between `<!-- worked: sparse_attention.<name> -->` markers; the `*_data` functions supply the numbers of the chapter's figures,
so a picture and the table beside it cannot disagree.

* a toy attention layer (64 topics, 256 positions, width 32) whose weights are printed, sorted and summed, and the share of the
  mass that the top k keys carry;
* fixed patterns (a window, a window with global tokens, a window with a stride, blocks, random keys) against the best choice of
  the same number of keys, and the cost of the strided pattern at long lengths;
* the per-query cost of a learned indexer against dense attention at the shapes of `GlmConfig::glm5_2()`, and the length above
  which the indexer pays for itself;
* an indexer (two projections to width 8) trained for 200 steps to match the dense attention weights, under the stable-digits
  rule, and the output error and captured mass of attending to the top k keys of the oracle, of that indexer and of a window;
* the Ward-merged bucket index of latent sparse attention on the toy, its read, and its sizes at the default configuration;
* the keys a selection touches in a paged cache, the bytes of the sparse kernels, and the index pages.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of products
are plain left-to-right additions, the order a C loop would use (never Python's `sum`, which may compensate).

Swedish Embedded AB implements solutions for serving long-context language models whose attention cost is set by what a token
needs, not by what the context holds. If your team needs expertise in sparse attention and paged key-value caches, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import stable

# ------------------------------------------------------------------------------------------------ formatting


def _int(n):
    return f"{n:,}"


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed_line(name):
    return f"`{name}` (seed {bookrng.seed_of(name):#018x})"


# ------------------------------------------------------------------------------------------------ arithmetic


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _norm(a):
    return math.sqrt(_dot(a, a))


def _softmax(s):
    m = max(s)
    e = [math.exp(x - m) for x in s]
    t = 0.0
    for x in e:
        t += x
    return [x / t for x in e]


def _order(scores, n):
    """The positions of the n largest scores, best first, a tie to the lower position."""
    return sorted(range(len(scores)), key=lambda j: (-scores[j], j))[:n]


# ------------------------------------------------------------------------------------------------ the toy layer

T, D, TOPICS = 256, 32, 64
GAIN, NOISE = 6.0, 0.5
EVAL_FROM = 128  # queries 128 .. 255 are evaluated: each has at least 129 keys
MODEL, SEQ_A, SEQ_B = "sparse_attention.model", "sparse_attention.seq_a", "sparse_attention.seq_b"


@lru_cache(maxsize=None)
def centers():
    g = bookrng.generator(MODEL)
    out = []
    for _ in range(TOPICS):
        v = [g.normal() for _ in range(D)]
        n = _norm(v)
        out.append([x / n for x in v])
    return out


@lru_cache(maxsize=None)
def sequence(name):
    """Queries, keys, values and topics of one sequence; the draw order is the one the chapter prints."""
    c = centers()
    g = bookrng.generator(name)
    z = [g.integer(TOPICS) for _ in range(T)]
    k = [[GAIN * c[z[j]][a] + NOISE * g.normal() for a in range(D)] for j in range(T)]
    v = [[g.normal() for _ in range(D)] for _ in range(T)]
    w = [g.integer(TOPICS) for _ in range(T)]
    q = [[GAIN * c[w[i]][a] + NOISE * g.normal() for a in range(D)] for i in range(T)]
    return {"q": q, "k": k, "v": v, "z": z, "w": w}


@lru_cache(maxsize=None)
def row(name, i):
    """(scores, weights) of query i over the keys 0 .. i of the named sequence."""
    s = sequence(name)
    sc = [_dot(s["q"][i], s["k"][j]) / math.sqrt(D) for j in range(i + 1)]
    return sc, _softmax(sc)


def _weighted(v, weights, idx):
    out = [0.0] * D
    for j in idx:
        for a in range(D):
            out[a] += weights[j] * v[j][a]
    return out


def attend(name, i, sel):
    """Attention of query i over the positions `sel` only: one softmax over their scores, values summed in ascending position."""
    sc, _ = row(name, i)
    sel = sorted(sel)
    p = _softmax([sc[j] for j in sel])
    out = [0.0] * D
    v = sequence(name)["v"]
    for pj, j in zip(p, sel):
        for a in range(D):
            out[a] += pj * v[j][a]
    return out


def measure(name, i, sel):
    """(mass, relative output error) of reading `sel` instead of every key, for query i."""
    _, p = row(name, i)
    sel = sorted(sel)
    mass = 0.0
    for j in sel:
        mass += p[j]
    dense = _weighted(sequence(name)["v"], p, range(i + 1))
    sparse = attend(name, i, sel)
    diff = [x - y for x, y in zip(dense, sparse)]
    return mass, _norm(diff) / _norm(dense)


def _setup_text():
    return (f"A toy layer of one head: {T} positions, head width d_h = {D}, {TOPICS} topics. Draw order: the {TOPICS} topic "
            f"centres from {_seed_line(MODEL)}, {D} normal draws per centre, row by row, each row divided by its length. A "
            f"sequence from its own generator, in this order: {T} key topics (integer draws below {TOPICS}); the key noise "
            f"({T} x {D} normals, key by key); the values ({T} x {D} normals, key by key); {T} query topics (integer draws "
            f"below {TOPICS}); the query noise ({T} x {D} normals, query by query). Key j is {GAIN:g} c[z_j] + {NOISE:g} n_j, "
            f"query i is {GAIN:g} c[w_i] + {NOISE:g} n_i, the value is the draw itself. Scores are s_ij = q_i . k_j / "
            f"sqrt({D}) for j <= i, softmax in the stable form. Sequence A is {_seed_line(SEQ_A)}, sequence B is "
            f"{_seed_line(SEQ_B)}.")


# ------------------------------------------------------------------------------------------------ 1. how concentrated


def weights_of_one_query():
    name, i = SEQ_A, T - 1
    _, p = row(name, i)
    s = sequence(name)
    order = _order(p, len(p))
    rows, cum = [], 0.0
    for r, j in enumerate(order[:8]):
        cum += p[j]
        rows.append([r + 1, j, "yes" if s["z"][j] == s["w"][i] else "no", _fx(p[j], 4), _fx(cum, 4)])
    need = {}
    cum = 0.0
    for r, j in enumerate(order):
        cum += p[j]
        for target in (0.5, 0.9, 0.99):
            if target not in need and cum >= target:
                need[target] = r + 1
    same = sum(1 for j in range(i + 1) if s["z"][j] == s["w"][i])
    head = (_setup_text() + f"\n\nSequence A, query i = {i} (topic {s['w'][i]}), which reads all {i + 1} keys; {same} of them share "
            "its topic. The eight largest weights:\n\n")
    tail = (f"\n\nThe smallest number of keys that carry 50%, 90% and 99% of the mass: {need[0.5]}, {need[0.9]} and {need[0.99]} "
            f"of {i + 1}. Tolerance: weights to 4 decimals.")
    return head + _table(["rank", "position j", "same topic", "weight p_ij", "cumulative mass"], rows) + tail


def mass_by_k():
    name = SEQ_A
    ks = (1, 2, 4, 8, 16, 32, 64)
    rows = []
    for k in ks:
        mass, share = 0.0, 0.0
        for i in range(EVAL_FROM, T):
            _, p = row(name, i)
            for j in _order(p, k):
                mass += p[j]
            share += k / (i + 1)
        n = T - EVAL_FROM
        rows.append([k, _fx(mass / n, 4), _fx(share / n, 4)])
    return ("Sequence A, the 128 queries i = 128 .. 255 (each reads i + 1 keys, between 129 and 256). For each k the mean over "
            "those queries of the mass held by the k largest weights of the query, and the mean share of the query's keys "
            "that k keys are. Tolerance: 4 decimals.\n\n"
            + _table(["k", "mean mass of the top k", "mean share of the keys read"], rows))


# ------------------------------------------------------------------------------------------------ 2. fixed patterns

BLOCK = 8


def _random_keys(g, i, n):
    """n distinct positions from 0 .. i: the first n steps of a Fisher-Yates shuffle of [0 .. i], step t swapping position t with
    t + integer(i + 1 - t)."""
    pos = list(range(i + 1))
    for t in range(min(n, i + 1)):
        r = t + g.integer(i + 1 - t)
        pos[t], pos[r] = pos[r], pos[t]
    return pos[: min(n, i + 1)]


def pattern_sets():
    """{pattern name: [set of positions for each query i in EVAL_FROM .. T-1]} on the toy."""
    g = bookrng.generator("sparse_attention.random_keys")
    out = {"window of 16": [], "window of 8 + 8 global tokens": [], "window of 4 + every 16th key": [],
           "blocks of 8: own, previous, first": [], "16 random keys": []}
    for i in range(EVAL_FROM, T):
        out["window of 16"].append(set(range(i - 15, i + 1)))
        out["window of 8 + 8 global tokens"].append(set(range(i - 7, i + 1)) | set(range(8)))
        out["window of 4 + every 16th key"].append(set(range(i - 3, i + 1)) | {i - 16 * m for m in range(1, i // 16 + 1)})
        b = i // BLOCK
        own = set(range(b * BLOCK, i + 1))
        out["blocks of 8: own, previous, first"].append(own | set(range((b - 1) * BLOCK, b * BLOCK)) | set(range(BLOCK)))
        out["16 random keys"].append(set(_random_keys(g, i, 16)))
    return out


def fixed_patterns():
    name = SEQ_A
    rows = []
    n = T - EVAL_FROM
    for label, sets in pattern_sets().items():
        used, mass, best = 0.0, 0.0, 0.0
        for idx, i in enumerate(range(EVAL_FROM, T)):
            _, p = row(name, i)
            sel = sets[idx]
            used += len(sel)
            for j in sel:
                mass += p[j]
            for j in _order(p, len(sel)):
                best += p[j]
        rows.append([label, _fx(used / n, 2), _fx(mass / n, 4), _fx(best / n, 4)])
    return (f"Sequence A, queries i = 128 .. 255. Window of w: positions i - w + 1 .. i. Global tokens: positions 0 .. 7. Every "
            "16th key: i - 16, i - 32, ... down to the first non-negative position. Blocks: position j is in block j // 8; "
            "a query reads its own block up to itself, the whole previous block and block 0. Random keys: 16 distinct "
            f"positions of 0 .. i by the first 16 steps of a Fisher-Yates shuffle of [0 .. i] (step t swaps entries t and t + "
            f"integer(i + 1 - t)), queries in ascending order, generator {_seed_line('sparse_attention.random_keys')}. The last "
            "column is the mass of the best choice of as many keys as the pattern reads, query by query (the top k of the true "
            "weights). Tolerance: 4 decimals.\n\n"
            + _table(["pattern", "mean keys read", "mean mass kept", "mass kept by the best keys of the same number"], rows))


def strided_cost():
    rows = []
    for t in (4096, 65536, 1048576):
        s = math.isqrt(t)
        w = 512
        strided = w + (t - 1) // s - (w // s)  # window plus the strided keys outside it
        rows.append([_int(t), _int(t), _int(w), _int(strided), _fx(t / strided, 1)])
    return ("The last query of a sequence of T keys. Dense reads T keys. A window reads 512. A window of 512 plus every s-th key "
            "back from the query, with stride s = floor(sqrt(T)), reads 512 + floor((T - 1) / s) minus the strided keys that "
            "fall inside the window, floor(512 / s).\n\n"
            + _table(["T", "dense keys", "window keys", "window + strided keys", "dense / (window + strided)"], rows))


def mask_data():
    """Four 16 x 16 masks for the figure: each row i is a query, "x" marks a key it reads. The learned panel keeps, per query, the
    4 keys of largest true weight on a separate toy of 16 positions and 4 topics (generator `sparse_attention.mask_toy`)."""
    n, d, topics = 16, 16, 4
    g = bookrng.generator("sparse_attention.mask_toy")
    cs = []
    for _ in range(topics):
        v = [g.normal() for _ in range(d)]
        nv = _norm(v)
        cs.append([x / nv for x in v])
    z = [g.integer(topics) for _ in range(n)]
    k = [[GAIN * cs[z[j]][a] + NOISE * g.normal() for a in range(d)] for j in range(n)]
    w = [g.integer(topics) for _ in range(n)]
    q = [[GAIN * cs[w[i]][a] + NOISE * g.normal() for a in range(d)] for i in range(n)]
    masks = {"window of 4": [], "window of 2 + every 4th key": [], "blocks of 2: own, previous, first": [], "top 4 by true weight": []}
    for i in range(n):
        masks["window of 4"].append({j for j in range(max(0, i - 3), i + 1)})
        masks["window of 2 + every 4th key"].append({j for j in range(max(0, i - 1), i + 1)} | {i - 4 * m for m in range(1, i // 4 + 1)})
        b = i // 2
        masks["blocks of 2: own, previous, first"].append(set(range(b * 2, i + 1)) | set(range(max(0, (b - 1) * 2), b * 2)) | set(range(2)))
        sc = [_dot(q[i], k[j]) / math.sqrt(d) for j in range(i + 1)]
        masks["top 4 by true weight"].append(set(_order(_softmax(sc), 4)))
    return n, masks


# ------------------------------------------------------------------------------------------------ 3. cost of an indexer

H_Q, D_QK, D_V = 64, 256, 256      # query heads, key width and value width per head (GlmConfig::glm5_2)
H_I, D_I, N_SEL = 32, 128, 2048    # indexer heads, indexer head width, keys read (index_n_heads, index_head_dim, index_topk)


def indexer_cost():
    dense_key = H_Q * (D_QK + D_V)
    idx_key = H_I * D_I
    rows = []
    for t in (2048, 4096, 16384, 65536, 131072):
        dense = dense_key * t
        sel = min(t, N_SEL)
        total = idx_key * t + dense_key * sel
        rows.append([_int(t), _int(dense), _int(idx_key * t), _int(dense_key * sel), _int(total), _fx(dense / total, 2)])
    t_star = N_SEL * dense_key / (dense_key - idx_key)
    return (f"Multiply-adds for one query of one layer. Dense attention, per cached key: {H_Q} heads x ({D_QK} + {D_V}) = "
            f"{_int(dense_key)} (a score of width {D_QK} and a weighted value of width {D_V} per head, each head reading its own "
            f"key, the form without absorption). The indexer, per cached key: {H_I} heads x {D_I} = {_int(idx_key)}. Sparse "
            f"attention reads min(T, {N_SEL}) keys. The selection itself (picking the {N_SEL} largest of T scores) is not "
            "counted.\n\n"
            + _table(["T", "dense", "indexer scores", "attention over the selected keys", "indexer + selected", "dense / total"], rows)
            + f"\n\nAbove T = {N_SEL} the indexer pays for itself when {idx_key} T + {_int(dense_key)} x {N_SEL} < "
            f"{_int(dense_key)} T, that is T > {N_SEL} x {_int(dense_key)} / ({_int(dense_key)} - {_int(idx_key)}) = "
            f"{t_star:.1f}. At or below {N_SEL} keys the selection keeps everything and the indexer is pure cost.")


# ------------------------------------------------------------------------------------------------ 4. distilling the indexer

DI, LR, STEPS = 8, 1.0, 200
TRAIN_QUERIES = list(range(64, T, 2))
KS = (1, 2, 4, 8, 16, 32, 64, 128)
CHECKPOINTS = (0, 50, 100, 200)


def _init_projections():
    g = bookrng.generator("sparse_attention.indexer_init")
    wq = [[g.normal() / math.sqrt(D) for _ in range(D)] for _ in range(DI)]
    wk = [[g.normal() / math.sqrt(D) for _ in range(D)] for _ in range(DI)]
    return wq, wk


def _project(w, xs):
    return [[_dot(w[r], x) for r in range(DI)] for x in xs]


def _index_scores(pq_i, pk, i):
    return [_dot(pq_i, pk[j]) / math.sqrt(DI) for j in range(i + 1)]


def _eval_indexer(wq, wk):
    """Mean mass and relative error of the indexer's top k on sequence B, queries 128 .. 255, for every k in KS, and the
    mean mass of the oracle."""
    seq = sequence(SEQ_B)
    pq, pk = _project(wq, seq["q"]), _project(wk, seq["k"])
    res = {k: [0.0, 0.0] for k in KS}
    for i in range(EVAL_FROM, T):
        sc = _index_scores(pq[i], pk, i)
        for k in KS:
            m, e = measure(SEQ_B, i, _order(sc, k))
            res[k][0] += m
            res[k][1] += e
    n = T - EVAL_FROM
    return {k: (res[k][0] / n, res[k][1] / n) for k in KS}


def _train_indexer(nudge):
    """200 steps of plain gradient descent (rate 1.0) on the mean cross-entropy between the dense weights of the 96 training
    queries of sequence A and the indexer's softmax; `nudge` is added to the first entry of the query projection."""
    wq, wk = _init_projections()
    wq[0][0] += nudge
    seq = sequence(SEQ_A)
    targets = {i: row(SEQ_A, i)[1] for i in TRAIN_QUERIES}
    entropy = 0.0
    for i in TRAIN_QUERIES:
        for pj in targets[i]:
            entropy -= pj * math.log(pj)
    entropy /= len(TRAIN_QUERIES)
    trace = {}
    for step in range(STEPS + 1):
        pq, pk = _project(wq, seq["q"]), _project(wk, seq["k"])
        dpq = [[0.0] * DI for _ in range(T)]
        dpk = [[0.0] * DI for _ in range(T)]
        ce = 0.0
        for i in TRAIN_QUERIES:
            r = _softmax(_index_scores(pq[i], pk, i))
            for j in range(i + 1):
                ce -= targets[i][j] * math.log(r[j])
                g = (r[j] - targets[i][j]) / len(TRAIN_QUERIES) / math.sqrt(DI)
                for a in range(DI):
                    dpk[j][a] += g * pq[i][a]
                    dpq[i][a] += g * pk[j][a]
        ce /= len(TRAIN_QUERIES)
        if step in CHECKPOINTS:
            trace[step] = {"kl": ce - entropy, "eval": _eval_indexer(wq, wk)}
        if step == STEPS:
            break
        for a in range(DI):
            for c in range(D):
                gq = 0.0
                for i in TRAIN_QUERIES:
                    gq += dpq[i][a] * seq["q"][i][c]
                gk = 0.0
                for j in range(T):
                    gk += dpk[j][a] * seq["k"][j][c]
                wq[a][c] -= LR * gq
                wk[a][c] -= LR * gk
    return {"nudge": nudge, "trace": trace}


@lru_cache(maxsize=None)
def training_runs():
    """The base run and the four nudged runs, spread over four processes."""
    jobs = [0.0, *stable.NUDGES]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_train_indexer, jobs, chunksize=1)
    return res[0], res[1:]


def indexer_training():
    base, twins = training_runs()
    rows = []
    for step in CHECKPOINTS:
        kl = stable.text(base["trace"][step]["kl"], [t["trace"][step]["kl"] for t in twins], lambda x: _fx(x, 3))
        m8 = stable.text(base["trace"][step]["eval"][8][0], [t["trace"][step]["eval"][8][0] for t in twins], lambda x: _fx(x, 3))
        m32 = stable.text(base["trace"][step]["eval"][32][0], [t["trace"][step]["eval"][32][0] for t in twins], lambda x: _fx(x, 3))
        rows.append([step, kl, m8, m32])
    return (f"Indexer: s'_ij = (Wq q_i) . (Wk k_j) / sqrt({DI}) with Wq, Wk of shape [{DI}, {D}], entries drawn as normals "
            f"divided by sqrt({D}) from {_seed_line('sparse_attention.indexer_init')}, Wq first and then Wk, row by row. "
            f"Trained on sequence A, the {len(TRAIN_QUERIES)} queries i = 64, 66, ..., 254, by {STEPS} steps of full-batch "
            f"gradient descent at rate {LR:g} on the mean over those queries of the cross-entropy between the dense weights "
            "p_i and softmax_j(s'_ij) over j <= i; the table's KL is that cross-entropy minus the mean entropy of the dense "
            "weights. Held-out columns: sequence B, queries 128 .. 255, the mean mass of the dense weights on the indexer's "
            f"top 8 and top 32 keys (a random choice of 8 keys keeps {_fx(sum(8 / (i + 1) for i in range(EVAL_FROM, T)) / (T - EVAL_FROM), 3)} of it on average, 32 keys {_fx(sum(32 / (i + 1) for i in range(EVAL_FROM, T)) / (T - EVAL_FROM), 3)}). "
            + stable.RULE + "\n\n"
            + _table(["step", "mean KL on the training queries", "held-out mass of the top 8", "held-out mass of the top 32"], rows))


def error_vs_k():
    base, twins = training_runs()
    rows = []
    worst, ident = 0.0, 0.0
    for k in KS:
        om, oe, wm, we = 0.0, 0.0, 0.0, 0.0
        for i in range(EVAL_FROM, T):
            _, p = row(SEQ_B, i)
            m, e = measure(SEQ_B, i, _order(p, k))
            om += m
            oe += e
            m, e = measure(SEQ_B, i, range(max(0, i + 1 - k), i + 1))
            wm += m
            we += e
        n = T - EVAL_FROM
        im = stable.text(base["trace"][STEPS]["eval"][k][0], [t["trace"][STEPS]["eval"][k][0] for t in twins], lambda x: _fx(x, 3))
        ie = stable.text(base["trace"][STEPS]["eval"][k][1], [t["trace"][STEPS]["eval"][k][1] for t in twins], lambda x: _fx(x, 3))
        rows.append([k, _fx(om / n, 3), _fx(oe / n, 3), im, ie, _fx(wm / n, 3), _fx(we / n, 3)])
    # the identity o - o_hat = (1 - m)(mu_R - mu_S) and its bound 2 (1 - m) max |v|, on every query and every k, for the oracle
    seq = sequence(SEQ_B)
    vmax = max(_norm(v) for v in seq["v"])
    for k in KS:
        for i in range(EVAL_FROM, T):
            _, p = row(SEQ_B, i)
            sel = set(_order(p, k))
            if len(sel) == i + 1:
                continue
            m = sum(p[j] for j in sorted(sel))
            dense = _weighted(seq["v"], p, range(i + 1))
            sparse = attend(SEQ_B, i, sel)
            err = _norm([x - y for x, y in zip(dense, sparse)])
            mu_s = _weighted(seq["v"], [p[j] / m if j in sel else 0.0 for j in range(i + 1)], sorted(sel))
            rest = [j for j in range(i + 1) if j not in sel]
            mu_r = _weighted(seq["v"], [p[j] / (1 - m) if j not in sel else 0.0 for j in range(i + 1)], rest)
            exact = (1 - m) * _norm([x - y for x, y in zip(mu_r, mu_s)])
            ident = max(ident, abs(err - exact))
            worst = max(worst, err / (2 * (1 - m) * vmax))
    full = max(measure(SEQ_B, i, range(i + 1))[1] for i in range(EVAL_FROM, T))
    assert full == 0.0 and ident < 1e-12
    return ("Sequence B, queries i = 128 .. 255, and every k below. 'mass' is the mean over queries of the dense weight the "
            "selected keys hold; 'error' is the mean over queries of the length of (dense output - sparse output) divided by "
            "the length of the dense output, the sparse output being one softmax over the selected keys' own scores. Oracle: "
            "the k largest true weights. Indexer: the k largest scores of the trained indexer of the previous table. "
            "Window: the k most recent keys. Oracle and window are exact; the indexer columns follow the stable-digits rule.\n\n"
            + _table(["k", "oracle mass", "oracle error", "indexer mass", "indexer error", "window mass", "window error"], rows)
            + "\n\nReading every causal key gives an error of exactly 0 (checked on all 128 queries). Over the "
            f"{len(KS) * (T - EVAL_FROM)} (query, k) pairs of the oracle, the largest difference between the error length "
            "and (1 - m) times the length of mu_R - mu_S is below 1e-12, and the largest ratio of the error length to "
            f"2 (1 - m) max_j |v_j| is {worst:.3f}, where max_j |v_j| = {_fx(vmax, 3)} on sequence B.")


def curve_data():
    """Figure data: ks and, per selector, the mean mass and mean relative error on sequence B (the un-nudged indexer)."""
    base, _ = training_runs()
    out = {"k": list(KS), "oracle": ([], []), "indexer": ([], []), "window": ([], [])}
    n = T - EVAL_FROM
    for k in KS:
        om = oe = wm = we = 0.0
        for i in range(EVAL_FROM, T):
            _, p = row(SEQ_B, i)
            m, e = measure(SEQ_B, i, _order(p, k))
            om, oe = om + m, oe + e
            m, e = measure(SEQ_B, i, range(max(0, i + 1 - k), i + 1))
            wm, we = wm + m, we + e
        out["oracle"][0].append(om / n)
        out["oracle"][1].append(oe / n)
        out["window"][0].append(wm / n)
        out["window"][1].append(we / n)
        out["indexer"][0].append(base["trace"][STEPS]["eval"][k][0])
        out["indexer"][1].append(base["trace"][STEPS]["eval"][k][1])
    return out


# ------------------------------------------------------------------------------------------------ 5. buckets of keys

BLOCK_KEYS, POOL, BUCKET, CAP = 32, 1, 4, 8
TOP_P, TOP_K, SINK, WINDOW = 4, 16, 2, 32


def _unit(x):
    n = _norm(x)
    return [v / n for v in x]


def ward_labels(pools, sizes, count, cap):
    """Merge pools into `count` buckets of at most `cap` keys: each round every live bucket finds its cheapest partner
    (cost a(b) + a(b') - a(b + b'), a = length of the sum; the cap must allow the pair; a tie to the lower index); pairs that
    choose each other merge into the lower member, cheapest first, at most as many as are still needed."""
    n = len(pools)
    sums = [list(p) for p in pools]
    size = list(sizes)
    alive = [True] * n
    parent = list(range(n))
    live = n
    while live > count:
        align = [_norm(s) if alive[i] else 0.0 for i, s in enumerate(sums)]
        best = [None] * n
        for i in range(n):
            if not alive[i]:
                continue
            for j in range(n):
                if j == i or not alive[j] or size[i] + size[j] > cap:
                    continue
                cost = align[i] + align[j] - _norm([x + y for x, y in zip(sums[i], sums[j])])
                if best[i] is None or cost < best[i][1]:
                    best[i] = (j, cost)
        mutual = [(best[i][1], i, best[i][0]) for i in range(n)
                  if best[i] is not None and i < best[i][0] and best[best[i][0]] is not None and best[best[i][0]][0] == i]
        if not mutual:
            raise ValueError("no pair can merge under the cap")
        mutual.sort(key=lambda t: (t[0], t[1]))
        for _, kept, absorbed in mutual[: live - count]:
            sums[kept] = [x + y for x, y in zip(sums[kept], sums[absorbed])]
            size[kept] += size[absorbed]
            alive[absorbed] = False
            parent[absorbed] = kept
            live -= 1
    number, nxt = {}, 0
    for i in range(n):
        if alive[i]:
            number[i] = nxt
            nxt += 1
    labels = []
    for i in range(n):
        while parent[i] != i:
            i = parent[i]
        labels.append(number[i])
    return labels


@lru_cache(maxsize=None)
def bucket_index(name):
    """Buckets of every closed block of the sequence's keys (unit rows): (centroids, members) in block order."""
    keys = [_unit(k) for k in sequence(name)["k"]]
    cents, members = [], []
    for b in range(T // BLOCK_KEYS):
        base = b * BLOCK_KEYS
        pools = []
        for p in range(BLOCK_KEYS // POOL):
            s = [0.0] * D
            for u in range(POOL):
                for a in range(D):
                    s[a] += keys[base + p * POOL + u][a]
            pools.append(s)
        labels = ward_labels(pools, [POOL] * len(pools), BLOCK_KEYS // BUCKET, CAP)
        for lab in range(BLOCK_KEYS // BUCKET):
            mem = [base + p * POOL + u for p in range(len(pools)) if labels[p] == lab for u in range(POOL)]
            s = [0.0] * D
            for j in mem:
                for a in range(D):
                    s[a] += keys[j][a]
            cents.append(_unit(s))
            members.append(mem)
    return cents, members


def lsa_read(name, i, trace=False):
    """Positions read by query i: the keys of its best TOP_P buckets (best first, outside the sinks and the window, at most
    TOP_K), then the window and the sinks."""
    cents, members = bucket_index(name)
    q = _unit(sequence(name)["q"][i])
    visible = (i // BLOCK_KEYS) * (BLOCK_KEYS // BUCKET)
    scores = [_dot(q, cents[b]) for b in range(visible)]
    picked = _order(scores, TOP_P)
    wstart = max(0, i + 1 - WINDOW)
    far = []
    for b in picked:
        for j in members[b]:
            if len(far) == TOP_K:
                break
            if SINK <= j < wstart:
                far.append(j)
    sel = far + list(range(wstart, i + 1)) + list(range(min(SINK, wstart)))
    return (sel, picked, visible) if trace else sel


def bucket_index_toy():
    name = SEQ_B
    cents, members = bucket_index(name)
    s = sequence(name)
    rows = [[b, ", ".join(str(j) for j in members[b]), ", ".join(str(s["z"][j]) for j in members[b])] for b in range(BLOCK_KEYS // BUCKET)]
    pairs = together = 0
    for b in range(T // BLOCK_KEYS):
        home = {}
        for lab in range(BLOCK_KEYS // BUCKET):
            for j in members[b * (BLOCK_KEYS // BUCKET) + lab]:
                home[j] = lab
        for x in range(b * BLOCK_KEYS, (b + 1) * BLOCK_KEYS):
            for y in range(x + 1, (b + 1) * BLOCK_KEYS):
                if s["z"][x] == s["z"][y]:
                    pairs += 1
                    together += 1 if home[x] == home[y] else 0
    return (f"Sequence B. Index vectors are the keys divided by their length. Blocks of {BLOCK_KEYS} keys; {BLOCK_KEYS // BUCKET} "
            f"buckets a block (average {BUCKET} keys, at most {CAP}); pools of {POOL} key, because the toy's neighbouring keys "
            "are unrelated by construction; the merge is the Ward rule of the chapter; a bucket's centroid is the sum of its "
            "keys divided by its length. Block 0 (positions 0 .. 31): the members of each bucket and their topics.\n\n"
            + _table(["bucket", "member positions", "member topics"], rows, "lll")
            + f"\n\nIn the {T // BLOCK_KEYS} blocks, {pairs} pairs of keys share a topic and lie in the same block; {together} of "
            f"those pairs were merged into one bucket. Two keys chosen at random from a block of {BLOCK_KEYS} share a bucket "
            f"with probability {BUCKET - 1}/{BLOCK_KEYS - 1} = {(BUCKET - 1) / (BLOCK_KEYS - 1):.3f}.")


def lsa_toy():
    name = SEQ_B
    sel, picked, visible = lsa_read(name, T - 1, trace=True)
    cents, members = bucket_index(name)
    z = sequence(name)["z"]
    w = sequence(name)["w"]
    first = (
        f"Query 255 (topic {w[255]}) sees {visible} buckets of the {T // BLOCK_KEYS - 1} closed blocks (it is in block 7, "
        f"still open). The {TOP_P} buckets it picks, best first: {', '.join(str(b) for b in picked)}, "
        f"holding {sum(len(members[b]) for b in picked)} keys, of which "
        f"{sum(1 for b in picked for j in members[b] if z[j] == w[255])} share its topic. It reads {len(sel)} of its 256 keys "
        f"({min(TOP_K, sum(len(members[b]) for b in picked))} far keys, the window 224 .. 255 and the sinks 0 and 1)."
    )
    n = T - EVAL_FROM
    cnt = lm = lo = lw = le = oe = we = 0.0
    for i in range(EVAL_FROM, T):
        sel = lsa_read(name, i)
        assert len(set(sel)) == len(sel)
        _, p = row(name, i)
        mass, err = measure(name, i, sel)
        cnt += len(sel)
        lm += mass
        le += err
        omass, oerr = measure(name, i, _order(p, len(sel)))
        lo += omass
        oe += oerr
        wmass, werr = measure(name, i, range(i + 1 - len(sel), i + 1))
        lw += wmass
        we += werr
    rows = [["LSA read", _fx(cnt / n, 2), _fx(lm / n, 4), _fx(le / n, 4)],
            ["best keys, same count", _fx(cnt / n, 2), _fx(lo / n, 4), _fx(oe / n, 4)],
            ["window, same count", _fx(cnt / n, 2), _fx(lw / n, 4), _fx(we / n, 4)]]
    return (f"Read of a query: score the centroids of the buckets of closed blocks, take the {TOP_P} best (a tie to the lower "
            f"bucket), walk their members best bucket first and keep those in [{SINK}, window start) until {TOP_K} are kept, "
            f"then add the window of the last {WINDOW} positions and the sinks 0 .. {SINK - 1}.\n\n" + first
            + f"\n\nSequence B, queries i = 128 .. 255, per query the same number of keys for the three rows. Tolerance: 4 decimals.\n\n"
            + _table(["selection", "mean keys read", "mean mass kept", "mean relative error"], rows))


def lsa_scale():
    d_idx, bsize, block, top_k, window, sink = 64, 8, 512, 4096, 512, 4
    kv_bytes = 8 * 128 * 2 * 2
    idx_bytes = (block // bsize) * (d_idx + 16) * 4 // block
    rows = []
    for t in (16384, 131072, 1048576):
        closed = t // block
        buckets = closed * (block // bsize)
        rows.append([_int(t), _int(min(t, top_k + window + sink)), _int(buckets), _int(buckets * d_idx), _int(t * d_idx)])
    return (f"The default index configuration (index width {d_idx}, buckets of {bsize} keys on average, blocks of {block} keys, "
            f"{top_k} far keys, a window of {window}, {sink} sinks), for a query at the end of a sequence of T keys with all "
            "blocks before it closed. 'Bucket scores' are the centroids a query scores (closed blocks x 64 buckets), each a "
            f"dot product of width {d_idx}.\n\n"
            + _table(["T", "keys read", "bucket scores", "bucket-score multiply-adds", "multiply-adds to score every key"], rows)
            + f"\n\nThe index of one block is {block // bsize} buckets x ({d_idx} centroid numbers + 16 member slots) x 4 bytes = "
            f"{(block // bsize) * (d_idx + 16) * 4:,} bytes for {block} tokens, {idx_bytes} bytes per token and layer. The "
            f"cached keys and values of a layer with 8 key-value heads of width 128 in bf16 are 8 x 128 x 2 x 2 = {kv_bytes:,} "
            f"bytes per token, so the index is {100 * idx_bytes / kv_bytes:.1f}% of them.")


# ------------------------------------------------------------------------------------------------ 6. paged storage

PAGE_SIZES = (1, 2, 4, 8, 16)
N_READ = 16


def page_traffic():
    name = SEQ_A
    rows = []
    n = T - EVAL_FROM
    for pg in PAGE_SIZES:
        touched = fetched = kmass = bmass = 0.0
        for i in range(EVAL_FROM, T):
            _, p = row(name, i)
            top = _order(p, N_READ)
            pages = {j // pg for j in top}
            touched += len(pages)
            fetched += sum(min((b + 1) * pg, i + 1) - b * pg for b in pages)
            kmass += sum(p[j] for j in top)
            n_pages = max(1, N_READ // pg)
            mass_of = [sum(p[j] for j in range(b * pg, min((b + 1) * pg, i + 1))) for b in range((i + pg) // pg)]
            bmass += sum(mass_of[b] for b in _order(mass_of, n_pages))
        rows.append([pg, _fx(touched / n, 2), _fx(fetched / n, 2), _fx(kmass / n, 4), _fx(bmass / n, 4)])
    return (f"Sequence A, queries i = 128 .. 255, a budget of {N_READ} keys. The page of size P holds the keys "
            "P b .. P b + P - 1 (the last page of a query stops at the query). 'Pages touched' counts the distinct pages that "
            f"hold the {N_READ} keys of largest true weight; 'rows if pages are fetched whole' adds the rows of those pages. "
            f"'Page-granular mass' is the mass kept when the selection chooses whole pages instead: the {N_READ} // P "
            "pages of largest total weight. Tolerance: 2 decimals for counts, 4 for mass.\n\n"
            + _table(["page size P", "pages touched by the top 16 keys", "rows if pages are fetched whole",
                      "key-granular mass (16 rows)", "page-granular mass"], rows))


def sparse_kernel_bytes():
    nh, nkv, hd, width, t = 32, 8, 128, 4096, 131072
    per_head = 8 * nh * width * hd
    per_kv = 8 * nkv * width * hd
    int8 = 2 * nkv * width * hd + 8 * nkv * width
    dense = 8 * nkv * t * hd
    rows = [["dense, per KV head, f32", _int(dense), "1.00"],
            ["sparse, one workgroup per query head, f32", _int(per_head), _fx(per_head / dense, 3)],
            ["sparse, one workgroup per KV head, f32", _int(per_kv), _fx(per_kv / dense, 3)],
            ["sparse, one workgroup per KV head, int8 pool", _int(int8), _fx(int8 / dense, 3)]]
    return (f"K and V bytes read for one query of one layer with {nh} query heads, {nkv} key-value heads of width {hd}, "
            f"{width} selected keys, out of T = {_int(t)}. An f32 key and value channel are 4 + 4 bytes. A kernel with a "
            f"workgroup per query head reads each selected key row once per query head ({nh} x), one per key-value head once "
            f"per key-value head ({nkv} x). An int8 pool stores a byte for K and one for V per channel and one f32 scale "
            "for each of K and V per key and key-value head. The dense row is the per-KV-head kernel over all T keys.\n\n"
            + _table(["kernel", "bytes", "fraction of dense"], rows))
