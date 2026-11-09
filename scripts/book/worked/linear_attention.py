# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Linear Attention, Recurrence and Hybrid Models": attention without the softmax as a recurrence
with a matrix state, the delta rule and the decay gate, the chunked form of the recurrence, the short convolution, state-space
models, and the cost of hybrid stacks.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and independent
of any engine code. Each public function without arguments returns a markdown string that the chapter embeds between
`<!-- worked: linear_attention.<name> -->` markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture
and the table beside it cannot disagree.

* linear attention in its parallel and recurrent forms, on a hand-sized case and on random data, with the multiply-adds each form
  performs counted by the loops that perform them, and the growth of the output without a normalisation;
* a key-value memory written additively and by the delta rule: overwriting by hand, and retrieval accuracy as more pairs are written;
* the decay gate of the gated delta layer, a hand-sized run of the full recurrence, and the chunked form of the recurrence checked
  against the token-by-token one, with its multiply-adds counted, its sequential depth and its state traffic;
* why the triangular inverse of a chunk is solved by substitution and not by repeated squaring, on keys that point the same way;
* the causal short convolution, whole-sequence and streaming, and the symmetric one of an encoder;
* the discretisation of a state-space model, the scan form of a linear recurrence, and the cost of a hybrid stack against a pure one.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of products are
plain left-to-right additions, the order a C loop would use (never Python's `sum`, which may compensate).

Swedish Embedded AB implements solutions for serving long-context language models within a memory budget. If your team needs
expertise in linear attention and hybrid decoder architectures, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng

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
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


def _vec(v, d=3):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


def _gib(b):
    return f"{b / 2**30:.3f}"


# ------------------------------------------------------------------------------------------------ arithmetic


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def _l2_normalize(x, eps=1e-6):
    """x / sqrt(sum of squares + eps): the normalisation of the layer, with eps inside the root."""
    r = 1.0 / math.sqrt(_dot(x, x) + eps)
    return [v * r for v in x]


def _read(S, q):
    """S^T q for S of shape [dk, dv]: entry c is the sum over a = 0 .. dk-1 of S[a][c] q[a], left to right."""
    return [_dot([S[a][c] for a in range(len(S))], q) for c in range(len(S[0]))]


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _flat(rows):
    return [x for r in rows for x in r]


def _softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


# ------------------------------------------------------------------------------------------------ the shapes of the chapter

# The decoder shape of the 27B model that `crates/qwen35` builds as `qwen38_27b`: 64 layers, model width 5120, full attention with
# 24 query heads and 4 key-value heads of width 256, every 4th layer full attention, the others Gated DeltaNet with 16 key heads
# and 48 value heads of width 128 and a causal convolution of width 4.
QW = dict(layers=64, interval=4, d=5120, heads=24, kv_heads=4, hd=256, nk=16, nv=48, dk=128, dv=128, conv_k=4, ctx=262144)


def _qw_derived():
    q = QW
    d = dict(q)
    d["q_dim"] = q["heads"] * q["hd"]
    d["kv_dim"] = q["kv_heads"] * q["hd"]
    d["key_dim"] = q["nk"] * q["dk"]
    d["value_dim"] = q["nv"] * q["dv"]
    d["conv_dim"] = 2 * d["key_dim"] + d["value_dim"]
    d["state_words"] = q["nv"] * q["dk"] * q["dv"]
    d["hist_words"] = d["conv_dim"] * (q["conv_k"] - 1)
    d["n_full"] = q["layers"] // q["interval"]
    d["n_lin"] = q["layers"] - d["n_full"]
    d["kv_bytes_token"] = 2 * d["kv_dim"] * 2          # K and V, 2 bytes each (bf16)
    d["state_bytes"] = (d["state_words"] + d["hist_words"]) * 4   # fp32
    # projection parameters of one layer's token mixer (the leaves of `layer_leaves` that belong to the mixer)
    d["full_proj"] = (2 * d["q_dim"] + 2 * d["kv_dim"]) * q["d"] + q["d"] * d["q_dim"]
    d["lin_proj"] = (d["conv_dim"] + d["value_dim"] + 2 * q["nv"]) * q["d"] + q["d"] * d["value_dim"]
    return d


# ------------------------------------------------------------------------------------------------ 1. the bill of a long cache


def state_vs_cache():
    z = _qw_derived()
    rows = []
    for t in (1024, 8192, 65536, 262144):
        full = z["kv_bytes_token"] * t
        rows.append([_int(t), _int(full), _int(z["state_bytes"]), f"{full / z['state_bytes']:.2f}"])
    cross = z["state_bytes"] / z["kv_bytes_token"]
    return (
        "Shape of `crates/qwen35` `qwen38_27b`: model width 5,120; full attention has 4 key-value heads of width 256, so a token adds "
        f"2 x 1,024 numbers of 2 bytes (bf16) = {_int(z['kv_bytes_token'])} bytes to a full-attention layer's cache. A Gated DeltaNet layer has "
        f"{z['nv']} value heads with a state of {z['dk']} x {z['dv']} numbers = {_int(z['state_words'])} numbers, plus a convolution history of "
        f"{_int(z['conv_dim'])} channels x {z['conv_k'] - 1} numbers = {_int(z['hist_words'])}, all kept in fp32 (4 bytes): {_int(z['state_bytes'])} bytes whatever "
        "the length.\n\n"
        + _table(["tokens in the sequence", "full-attention layer: cache bytes", "Gated DeltaNet layer: state bytes", "cache over state"], rows)
        + f"\n\nThe two are equal at {cross:.0f} tokens ({_int(z['state_bytes'])} / {_int(z['kv_bytes_token'])}); {_int(z['ctx'])} is the model's `max_position_embeddings`."
    )


# ------------------------------------------------------------------------------------------------ 2. linear attention, two forms

_PQ = [[1, 0], [0, 1], [1, 1], [1, -1]]
_PK = [[1, 0], [0, 1], [1, 1], [0, 1]]
_PV = [[1, 2], [3, 1], [0, 2], [2, 0]]


def _linear_parallel(Q, K, V):
    """Rows of (M o (Q K^T)) V with M the causal 0/1 mask, and the multiply-adds performed."""
    T, dv = len(Q), len(V[0])
    out, macs = [], 0
    for i in range(T):
        row = [0.0] * dv
        for j in range(i + 1):
            s = _dot(Q[i], K[j])
            macs += len(Q[i])
            for c in range(dv):
                row[c] += s * V[j][c]
                macs += 1
        out.append(row)
    return out, macs


def _linear_recurrent(Q, K, V):
    """o_t = S_t^T q_t with S_t = S_{t-1} + k_t v_t^T; returns the outputs, every state and the multiply-adds performed."""
    dk, dv = len(K[0]), len(V[0])
    S = _zeros(dk, dv)
    outs, states, macs = [], [], 0
    for t in range(len(Q)):
        for a in range(dk):
            for c in range(dv):
                S[a][c] += K[t][a] * V[t][c]
                macs += 1
        outs.append(_read(S, Q[t]))
        macs += dk * dv
        states.append([r[:] for r in S])
    return outs, states, macs


def two_forms_macs(T, dk, dv):
    """(parallel, recurrent) multiply-adds for T tokens, counted by running both forms on all-zero inputs of that shape."""
    Q = [[0.0] * dk for _ in range(T)]
    V = [[0.0] * dv for _ in range(T)]
    return _linear_parallel(Q, Q, V)[1], _linear_recurrent(Q, Q, V)[2]


def two_forms_by_hand():
    outs_r, states, _ = _linear_recurrent(_PQ, _PK, _PV)
    outs_p, _ = _linear_parallel(_PQ, _PK, _PV)
    rows = []
    for t in range(4):
        rows.append([t, _vec(_PQ[t], 0), _vec(_PK[t], 0), _vec(_PV[t], 0), "[" + ", ".join(_vec(r, 0) for r in states[t]) + "]",
                     _vec(outs_r[t], 0), _vec(outs_p[t], 0)])
    same = all(outs_r[t][c] == outs_p[t][c] for t in range(4) for c in range(2))
    return (
        "Four tokens, $d_k = d_v = 2$, whole-number inputs (so every number below is exact). The state rows are the rows of the "
        "$2 \\times 2$ matrix $\\mathbf{S}_t = \\sum_{s \\le t}\\mathbf{k}_s\\mathbf{v}_s^\\top$; the recurrent output is $\\mathbf{S}_t^\\top\\mathbf{q}_t$ and the "
        "parallel one is row $t$ of $(\\mathbf{M}\\odot\\mathbf{Q}\\mathbf{K}^\\top)\\mathbf{V}$.\n\n"
        + _table(["t", "q", "k", "v", "state S_t (rows)", "o_t recurrent", "o_t parallel"], rows, align="lllllll")
        + f"\n\nThe two forms agree exactly in all 8 numbers: {'yes' if same else 'no'}."
    )


def two_forms_random():
    name = "linear_attention.two_forms"
    T, dk, dv = 64, 8, 8
    g = bookrng.generator(name)
    Q, K, V = [], [], []
    for _ in range(T):
        Q.append([g.normal() for _ in range(dk)])
        K.append([g.normal() for _ in range(dk)])
        V.append([g.normal() for _ in range(dv)])
    op, mp = _linear_parallel(Q, K, V)
    orr, _, mr = _linear_recurrent(Q, K, V)
    diff = _max_abs_diff(_flat(op), _flat(orr))
    assert mp == (T * (T + 1) // 2) * (dk + dv)
    assert mr == T * 2 * dk * dv
    assert diff < 1e-12
    rows = [["parallel: scores, then weighted sum of values", f"T(T+1)/2 (d_k + d_v)", _int(mp)],
            ["recurrent: state update, then read", "2 T d_k d_v", _int(mr)]]
    return (
        _seed_line(name) + f" T = {T}, d_k = d_v = {dk}. Draws, in order, for t = 0 to {T - 1}: q_t ({dk} values of `normal()`), k_t ({dk}), v_t ({dv}); "
        "no normalisation of any vector. Multiply-adds are counted by the loops that perform them.\n\n"
        + _table(["form", "multiply-adds (formula)", "multiply-adds counted"], rows, align="llr")
        + f"\n\nLargest difference between the two outputs over all {T * dv} numbers: below 1e-12 (the bound the program asserts)."
    )


def norm_growth():
    name = "linear_attention.growth"
    T, d, runs = 256, 16, 10
    marks = (1, 4, 16, 64, 256)
    lin = {t: 0.0 for t in marks}
    sm = {t: 0.0 for t in marks}
    for r in range(runs):
        g = bookrng.generator(f"{name}.run{r}")
        Q, K, V = [], [], []
        for _ in range(T):
            Q.append(_l2_normalize([g.normal() for _ in range(d)], 0.0))
            K.append(_l2_normalize([g.normal() for _ in range(d)], 0.0))
            V.append([g.normal() for _ in range(d)])
        S = _zeros(d, d)
        for t in range(T):
            for a in range(d):
                for c in range(d):
                    S[a][c] += K[t][a] * V[t][c]
            if t + 1 in marks:
                o = _read(S, Q[t])
                lin[t + 1] += math.sqrt(_dot(o, o)) / runs
                sc = [_dot(Q[t], K[j]) * math.sqrt(d) for j in range(t + 1)]   # q.k / sqrt(d_k) with unit q and k scaled to d_k
                m = max(sc)
                e = [math.exp(x - m) for x in sc]
                z = 0.0
                for x in e:
                    z += x
                acc = [0.0] * d
                for j in range(t + 1):
                    for c in range(d):
                        acc[c] += e[j] / z * V[j][c]
                sm[t + 1] += math.sqrt(_dot(acc, acc)) / runs
    rows = [[t, _fx(lin[t], 3), _fx(lin[t] / math.sqrt(t), 3), _fx(sm[t], 3)] for t in marks]
    return (
        f"Seeds `{name}.run0` to `{name}.run{runs - 1}` (the seed of each is `seed_of` of its name). d_k = d_v = {d}, T = {T}. Draws per run, in order, for each "
        f"t: q_t, then k_t (each {d} values of `normal()`, then divided by its own length), then v_t ({d} values of `normal()`). Linear attention: o_t = "
        "S_t^T q_t with S_t the sum of k_s v_s^T. Softmax attention: scores sqrt(d_k) q.k (what the usual q.k / sqrt(d_k) gives for vectors "
        "of length sqrt(d_k)), then the softmax-weighted sum of the values. Mean over "
        f"{runs} runs of the length of o_t.\n\n"
        + _table(["t", "length of o_t, linear", "linear / sqrt(t)", "length of o_t, softmax"], rows)
    )


# ------------------------------------------------------------------------------------------------ 3. memory: additive and delta


def _write_additive(S, k, v, beta):
    for a in range(len(S)):
        for c in range(len(S[0])):
            S[a][c] += k[a] * v[c]


def _write_delta(S, k, v, beta):
    pred = _read(S, k)
    for a in range(len(S)):
        for c in range(len(S[0])):
            S[a][c] += beta * k[a] * (v[c] - pred[c])


def overwrite_by_hand():
    ka, kb = [1.0, 0.0], [0.6, 0.8]
    v1, v2, v3 = [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]
    steps = [("write k_a -> v1", ka, v1), ("write k_b -> v2", kb, v2), ("rewrite k_a -> v3", ka, v3)]
    rows = []
    for name, write in (("additive", _write_additive), ("delta, beta = 1", _write_delta)):
        S = _zeros(2, 2)
        for label, k, v in steps:
            write(S, k, v, 1.0)
            rows.append([name, label, "[" + ", ".join(_vec(r, 3) for r in S) + "]", _vec(_read(S, ka), 3), _vec(_read(S, kb), 3)])
    return (
        "Keys $\\mathbf{k}_a = (1, 0)$ and $\\mathbf{k}_b = (0.6, 0.8)$ (unit length, $\\mathbf{k}_a\\cdot\\mathbf{k}_b = 0.6$); values $\\mathbf{v}_1 = (1, 0)$, "
        "$\\mathbf{v}_2 = (0, 1)$, $\\mathbf{v}_3 = (0.5, 0.5)$. Three writes in the order shown, into a state that starts at zero; the last two columns "
        "are what reading the state with each key returns after that write.\n\n"
        + _table(["memory", "after", "state (rows)", "read with k_a", "read with k_b"], rows, align="lllll")
    )


MEM = dict(dk=16, dv=16, trials=20, rounds=3, ms=(2, 4, 6, 8, 12, 16, 20, 24))


def _unit(g, n):
    return _l2_normalize([g.normal() for _ in range(n)], 0.0)


def _memory_trial(workload, m, trial, writer, beta):
    """One trial. Returns how many of the m probes retrieve the right value.

    workload 'rewrite': m keys, then 3 rounds; each round shuffles the key indices and writes a fresh value to each key in that order;
    the target of a probe is the LAST value written to the key. workload 'once': m keys, then m values, each key written once."""
    dk, dv = MEM["dk"], MEM["dv"]
    g = bookrng.generator(f"linear_attention.memory.{workload}.m{m}.trial{trial}")
    keys = [_unit(g, dk) for _ in range(m)]
    S = _zeros(dk, dv)
    target = {}
    if workload == "rewrite":
        for _ in range(MEM["rounds"]):
            order = list(range(m))
            g.shuffle(order)
            for j in order:
                v = _unit(g, dv)
                target[j] = v
                writer(S, keys[j], v, beta)
    else:
        for j in range(m):
            v = _unit(g, dv)
            target[j] = v
            writer(S, keys[j], v, beta)
    hits = 0
    for j in range(m):
        r = _read(S, keys[j])
        nr = math.sqrt(_dot(r, r))
        sims = [_dot(r, target[x]) / nr for x in range(m)]
        best = max(range(m), key=lambda x: (sims[x], -x))
        hits += best == j
    return hits


def memory_accuracy_data():
    """{workload: {method label: [accuracy for each m]}}."""
    methods = (("additive", _write_additive, 1.0), ("delta, beta = 1", _write_delta, 1.0), ("delta, beta = 0.5", _write_delta, 0.5))
    out = {}
    for workload in ("rewrite", "once"):
        out[workload] = {}
        for label, writer, beta in methods:
            accs = []
            for m in MEM["ms"]:
                hits = sum(_memory_trial(workload, m, tr, writer, beta) for tr in range(MEM["trials"]))
                accs.append(hits / (m * MEM["trials"]))
            out[workload][label] = accs
    return out


def memory_accuracy():
    data = memory_accuracy_data()
    rows = []
    for i, m in enumerate(MEM["ms"]):
        row = [m]
        for workload in ("rewrite", "once"):
            for label in ("additive", "delta, beta = 1", "delta, beta = 0.5"):
                row.append(_fx(data[workload][label][i], 3))
        rows.append(row)
    names = [f"linear_attention.memory.{w}.m<m>.trial<r>" for w in ("rewrite", "once")]
    return (
        f"d_k = d_v = {MEM['dk']}; keys and values are random unit vectors (`normal()` draws, divided by their length). A trial draws the m keys, then, for "
        f"the 'rewritten' workload, {MEM['rounds']} rounds each of which shuffles the indices 0 to m-1 (`shuffle`) and writes a fresh value to every key in that order "
        "(the value is drawn just before its write); for the 'written once' workload, m values in key order, one write each. Generators: "
        f"`{names[0]}` and `{names[1]}` with m and the trial number r = 0 to {MEM['trials'] - 1} filled in. A probe reads the state with a key and counts as correct when "
        "its result has the highest cosine similarity, among the m target values, with the key's own target (the last value written to the key). "
        "Each entry is the fraction of correct probes over " + f"{MEM['trials']} trials.\n\n"
        + _table(["m keys", "rewritten: additive", "rewritten: delta 1", "rewritten: delta 0.5", "once: additive", "once: delta 1", "once: delta 0.5"], rows)
    )


# ------------------------------------------------------------------------------------------------ 4. the gate


def decay_gate_table():
    rows = []
    for label, a_log, x in (("fresh init: A = 8, a + bias = 1", math.log(8.0), 1.0),
                            ("A = 1, a + bias = 1", 0.0, 1.0),
                            ("A = 1, a + bias = -3", 0.0, -3.0),
                            ("A = 0.1, a + bias = -1", math.log(0.1), -1.0),
                            ("A = 0.1, a + bias = -3", math.log(0.1), -3.0),
                            ("A = 0.01, a + bias = -3", math.log(0.01), -3.0)):
        A = math.exp(a_log)
        delta = _softplus(x)
        g = -A * delta
        alpha = math.exp(g)
        half = math.log(2.0) / -g
        rows.append([label, _fx(delta, 4), f"{g:.4f}", f"{alpha:.4g}", f"{half:.3g}"])
    return (
        "The decay gate of one value head for one token: $\\Delta = \\mathrm{softplus}(a + \\mathrm{bias})$ with `softplus` of the pseudocode, "
        "$g = -A\\,\\Delta$ with $A = \\exp(A_{\\log})$, and $\\alpha = \\exp(g)$. The half-life is $\\ln 2 / (-g)$ tokens. The first row is the "
        "reference initialisation at its mean ($A$ uniform on $(0, 16)$, bias 1, $a = 0$).\n\n"
        + _table(["case", "Delta", "g", "alpha", "half-life in tokens"], rows)
    )


# ------------------------------------------------------------------------------------------------ 5. the full recurrence

GD_RAW = dict(
    q=[[2.0, 1.0], [0.0, 3.0], [1.0, -1.0]],
    k=[[3.0, 4.0], [0.0, 2.0], [5.0, 12.0]],
    v=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    g=[-0.2, -0.1, -0.3],
    beta=[0.9, 0.5, 0.8],
)


def _prepare(Qr, Kr, dk):
    scale = 1.0 / math.sqrt(dk)
    Q = [[x * scale for x in _l2_normalize(q)] for q in Qr]
    K = [_l2_normalize(k) for k in Kr]
    return Q, K


def gated_delta_tokenwise(Q, K, V, g, beta, S0=None):
    """The recurrence of the layer, one token at a time: decay, read, delta, write, read out with the (scaled) query. Returns outputs,
    final state and the states after every token."""
    dk, dv = len(K[0]), len(V[0])
    S = [r[:] for r in S0] if S0 else _zeros(dk, dv)
    outs, states = [], []
    for t in range(len(K)):
        a = math.exp(g[t])
        for i in range(dk):
            for c in range(dv):
                S[i][c] *= a
        pred = _read(S, K[t])
        for i in range(dk):
            for c in range(dv):
                S[i][c] += K[t][i] * (beta[t] * (V[t][c] - pred[c]))
        outs.append(_read(S, Q[t]))
        states.append([r[:] for r in S])
    return outs, S, states


def gated_delta_by_hand():
    R = GD_RAW
    Q, K = _prepare(R["q"], R["k"], 2)
    outs, S, states = gated_delta_tokenwise(Q, K, R["v"], R["g"], R["beta"])
    raw = [[t, _vec(R["q"][t], 0), _vec(R["k"][t], 0), _vec(R["v"][t], 0), f"{R['g'][t]}", f"{R['beta'][t]}"] for t in range(3)]
    made = [[t, _fx(math.exp(R["g"][t]), 4), _vec(K[t], 4), _vec(Q[t], 4)] for t in range(3)]
    steps = [[t, "[" + ", ".join(_vec(r, 4) for r in states[t]) + "]", _vec(outs[t], 4)] for t in range(3)]
    return (
        "Three tokens, one head, $d_k = d_v = 2$, zero initial state. Raw inputs:\n\n"
        + _table(["t", "q raw", "k raw", "v", "g", "beta"], raw, align="llllll")
        + "\n\nWhat the layer makes of them: $\\alpha = \\exp(g)$; $\\mathbf{k}$ and $\\mathbf{q}$ divided by $\\sqrt{\\lVert\\cdot\\rVert^2 + 10^{-6}}$ (the L2 "
        "normalisation), and $\\mathbf{q}$ then scaled by $1/\\sqrt{d_k}$.\n\n"
        + _table(["t", "alpha", "k normalised", "q normalised and scaled"], made, align="llll")
        + "\n\nThe recurrence per token: $\\mathbf{S} \\leftarrow \\alpha\\mathbf{S}$; $\\hat{\\mathbf{v}} = \\mathbf{S}^\\top\\mathbf{k}$; $\\mathbf{S} \\leftarrow \\mathbf{S} + "
        "\\mathbf{k}\\,\\beta(\\mathbf{v} - \\hat{\\mathbf{v}})^\\top$; $\\mathbf{o} = \\mathbf{S}^\\top\\mathbf{q}$. State (rows) and output after each token:\n\n"
        + _table(["t", "state S_t (rows)", "o_t"], steps, align="lll")
    )


# ------------------------------------------------------------------------------------------------ 6. the chunked form


def _gcs(g):
    out, s = [], 0.0
    for x in g:
        s += x
        out.append(s)
    return out


def gated_delta_chunked(Q, K, V, g, beta, C, S0=None, count=None):
    """The chunked form: inside a chunk of C tokens a triangular system and matrix products, across chunks the state S.

    For a chunk with cumulative log-decay c_i (the running sum of g inside the chunk), N[i][j] = beta_i exp(c_i - c_j) (k_i . k_j) for j < i.
    Solve (I + N) U = beta_i v_i (rows) and (I + N) W = beta_i exp(c_i) k_i (rows) by forward substitution; D = U - W S0 holds the corrections;
    o_i = exp(c_i) S0^T q_i + sum over j <= i of exp(c_i - c_j) (q_i . k_j) D_j; the chunk's final state is
    exp(c_last) S0 + sum over j of exp(c_last - c_j) k_j D_j^T. `count` (a dict) receives the multiply-adds, by kind."""
    T, dk, dv = len(K), len(K[0]), len(V[0])
    assert T % C == 0
    S = [r[:] for r in S0] if S0 else _zeros(dk, dv)
    outs = []

    def tick(kind, n):
        if count is not None:
            count[kind] = count.get(kind, 0) + n

    for s0 in range(0, T, C):
        q, k, v, b = Q[s0:s0 + C], K[s0:s0 + C], V[s0:s0 + C], beta[s0:s0 + C]
        c = _gcs(g[s0:s0 + C])
        N = _zeros(C, C)
        for i in range(C):
            for j in range(i):
                N[i][j] = b[i] * math.exp(c[i] - c[j]) * _dot(k[i], k[j])
                tick("gram of keys (the matrix N)", dk + 1)
        U, W = [], []
        for i in range(C):
            ui = [b[i] * x for x in v[i]]
            wi = [b[i] * math.exp(c[i]) * x for x in k[i]]
            for j in range(i):
                for col in range(dv):
                    ui[col] -= N[i][j] * U[j][col]
                for col in range(dk):
                    wi[col] -= N[i][j] * W[j][col]
                tick("triangular solves", dv + dk)
            U.append(ui)
            W.append(wi)
        D = []
        for i in range(C):
            di = U[i][:]
            for a in range(dk):
                for col in range(dv):
                    di[col] -= W[i][a] * S[a][col]
            tick("D = U - W S", dk * dv)
            D.append(di)
        for i in range(C):
            o = _read(S, q[i])
            e = math.exp(c[i])
            o = [x * e for x in o]
            tick("output from the old state", dk * dv)
            for j in range(i + 1):
                w = _dot(q[i], k[j]) * math.exp(c[i] - c[j])
                tick("output inside the chunk", dk + dv)
                for col in range(dv):
                    o[col] += w * D[j][col]
            outs.append(o)
        dec = math.exp(c[-1])
        newS = [[S[a][col] * dec for col in range(dv)] for a in range(dk)]
        tick("decay of the old state", dk * dv)
        for j in range(C):
            f = math.exp(c[-1] - c[j])
            for a in range(dk):
                for col in range(dv):
                    newS[a][col] += f * k[j][a] * D[j][col]
            tick("state update", dk * dv)
        S = newS
    return outs, S


def chunk_macs_formula(T, C, dk, dv):
    """Multiply-adds of the chunked form per head, in closed form: it must equal what gated_delta_chunked counts."""
    n = T // C
    tri = C * (C - 1) // 2
    per_chunk = tri * (dk + 1) + tri * (dk + dv) + C * dk * dv + C * dk * dv + (C * (C + 1) // 2) * (dk + dv) + dk * dv + C * dk * dv
    return n * per_chunk


def tokenwise_macs(T, dk, dv):
    """Multiply-adds of the token-by-token recurrence per head: decay, read, write, read out."""
    return T * 4 * dk * dv


def _random_inputs(name, T, dk, dv):
    g = bookrng.generator(name)
    Qr, Kr, V, gg, beta = [], [], [], [], []
    for _ in range(T):
        Qr.append([g.normal() for _ in range(dk)])
        Kr.append([g.normal() for _ in range(dk)])
        V.append([g.normal() for _ in range(dv)])
        gg.append(-0.5 * g.uniform())
        beta.append(0.2 + 0.7 * g.uniform())
    Q, K = _prepare(Qr, Kr, dk)
    return Q, K, V, gg, beta


def chunk_equivalence():
    name = "linear_attention.chunk"
    T, dk, dv = 8, 3, 4
    Q, K, V, g, beta = _random_inputs(name, T, dk, dv)
    ref_o, ref_S, _ = gated_delta_tokenwise(Q, K, V, g, beta)
    rows = []
    for C in (1, 2, 4, 8):
        cnt = {}
        o, S = gated_delta_chunked(Q, K, V, g, beta, C, count=cnt)
        d_o = _max_abs_diff(_flat(o), _flat(ref_o))
        d_s = _max_abs_diff(_flat(S), _flat(ref_S))
        assert d_o < 1e-12 and d_s < 1e-12
        total = sum(cnt.values())
        assert total == chunk_macs_formula(T, C, dk, dv), (C, total, chunk_macs_formula(T, C, dk, dv))
        rows.append([C, T // C, "below 1e-12", "below 1e-12", _int(total)])
    return (
        _seed_line(name) + f" T = {T}, d_k = {dk}, d_v = {dv}, one head, zero initial state. Draws, in order, for t = 0 to {T - 1}: raw q_t ({dk} values of "
        f"`normal()`), raw k_t ({dk}), v_t ({dv}), then u = `uniform()` for the decay ($g_t = -0.5\\,u$) and u = `uniform()` for the write strength "
        "($\\beta_t = 0.2 + 0.7\\,u$); q and k are L2-normalised (with the $10^{-6}$) and q scaled by $1/\\sqrt{d_k}$ as in the layer. The reference is the "
        f"token-by-token recurrence ({_int(tokenwise_macs(T, dk, dv))} multiply-adds).\n\n"
        + _table(["chunk size C", "chunks", "largest absolute difference of the outputs", "largest absolute difference of the final state", "multiply-adds counted"], rows)
        + "\n\nThe counts are those of the loops of the program, and each equals the closed form of the chapter."
    )


def chunk_costs():
    dk = dv = 128
    T = 4096
    rows = []
    rec = tokenwise_macs(T, dk, dv)
    rows.append(["1 (token by token)", _int(rec // T), _int(T), _int(2 * T * dk * dv * 4), _int(T)])
    for C in (16, 32, 64, 128, 256):
        m = chunk_macs_formula(T, C, dk, dv)
        rows.append([C, _int(m // T), _int(T // C), _int(2 * (T // C) * dk * dv * 4), _int(C + T // C)])
    return (
        f"One head, T = {_int(T)} tokens, d_k = d_v = {dk}. Multiply-adds per token from the closed form (equal to the counts of the program at small "
        "sizes). 'State round trips' is the number of times the state is read from memory and written back, counting one read and one write per "
        "step, with the state in fp32 (4 bytes), if it lives in memory between steps. 'Sequential depth' is the longest chain of steps that "
        "cannot overlap: for the chunked form, the C rows of the triangular solve (independent across chunks) plus the T/C steps of the state.\n\n"
        + _table(["chunk size C", "multiply-adds per token", "state round trips", "state bytes moved (read + write)", "sequential depth"], rows)
    )


# ------------------------------------------------------------------------------------------------ 7. the triangular solve


def neumann_failure():
    name = "linear_attention.neumann"
    C, dk = 64, 16
    g = bookrng.generator(name)
    base = [g.normal() for _ in range(dk)]
    K, gg, beta = [], [], []
    for _ in range(C):
        K.append(_l2_normalize([base[i] + 0.06 * g.normal() for i in range(dk)]))
        gg.append(-0.002 * g.uniform())
        beta.append(0.97 + 0.02 * g.uniform())
    c = _gcs(gg)
    N = _zeros(C, C)
    for i in range(C):
        for j in range(i):
            N[i][j] = beta[i] * math.exp(c[i] - c[j]) * _dot(K[i], K[j])

    def matmul(A, B):
        n = len(A)
        return [[_dot(A[i], [B[k][j] for k in range(n)]) for j in range(n)] for i in range(n)]

    ident = [[1.0 if i == j else 0.0 for j in range(C)] for i in range(C)]
    # forward substitution on the columns of the identity
    L = _zeros(C, C)
    for col in range(C):
        for i in range(C):
            r = 1.0 if i == col else 0.0
            for j in range(i):
                r -= N[i][j] * L[j][col]
            L[i][col] = r
    IpN = [[ident[i][j] + N[i][j] for j in range(C)] for i in range(C)]
    resid = _max_abs_diff(_flat(matmul(IpN, L)), _flat(ident))
    negN = [[-x for x in r] for r in N]
    prod, cur, rows = ident, negN, []
    for m in range(6):
        mx = max(abs(x) for r in cur for x in r)
        rows.append([f"N^{2 ** m}", f"{mx:.3e}"])
        prod = matmul(prod, [[ident[i][j] + cur[i][j] for j in range(C)] for i in range(C)])
        cur = matmul(cur, cur)
    err = _max_abs_diff(_flat(prod), _flat(L))
    assert err > 1.0 and max(abs(x) for r in prod for x in r) > 1.0
    return (
        _seed_line(name) + f" C = {C}, d_k = {dk}. Draws, in order: a base direction ({dk} values of `normal()`); then for each of the {C} tokens, {dk} values of "
        "`normal()` that are added after scaling by 0.06 to the base, the sum L2-normalised (with $10^{-6}$) into the key; $u$ = `uniform()` for the decay "
        "($g = -0.002\\,u$) and $u$ = `uniform()` for the write strength ($\\beta = 0.97 + 0.02\\,u$). $\\mathbf{N}_{ij} = \\beta_i\\exp(c_i - c_j)(\\mathbf{k}_i\\cdot\\mathbf{k}_j)$ "
        f"for $j < i$. Largest entry of $\\mathbf{{N}}$: {max(abs(x) for r in N for x in r):.3f}.\n\n"
        + _table(["power of N", "largest absolute entry (3 digits)"], rows, align="lr")
        + f"\n\nForward substitution gives $(\\mathbf{{I}}+\\mathbf{{N}})^{{-1}}$ with largest entry {max(abs(x) for r in L for x in r):.3f} and residual "
        f"$\\lVert(\\mathbf{{I}}+\\mathbf{{N}})\\mathbf{{L}} - \\mathbf{{I}}\\rVert_{{\\max}}$ below 1e-12. The product of the six factors $\\mathbf{{I}} + (-\\mathbf{{N}})^{{2^m}}$ "
        "has a largest entry above 1 and differs from the substitution result by more than 1: its error is larger than the answer itself "
        "(the program asserts both; the digits beyond that depend on rounding and are not printed)."
    )


# ------------------------------------------------------------------------------------------------ 8. the short convolution

CONV_W = [0.5, 0.25, -0.5, 1.0]
CONV_X = [1.0, 2.0, 0.0, -1.0, 3.0, 1.0]


def causal_conv(x, w):
    """y_t = sum over j of w[j] x[t - (K-1) + j], with x before the start equal to 0 (the K-th tap is the current token)."""
    K = len(w)
    out = []
    for t in range(len(x)):
        s = 0.0
        for j in range(K):
            i = t - (K - 1) + j
            s += w[j] * (x[i] if i >= 0 else 0.0)
        out.append(s)
    return out


def causal_conv_stream(x, w):
    """The same convolution one token at a time, from a history of the last K-1 inputs (oldest first), updated after each token."""
    K = len(w)
    hist = [0.0] * (K - 1)
    out = []
    for xt in x:
        window = hist + [xt]
        s = 0.0
        for j in range(K):
            s += w[j] * window[j]
        out.append(s)
        hist = window[1:]
    return out


def symmetric_conv(x, w):
    """Width K = 3 with one zero of padding on each side: y_t = w0 x[t-1] + w1 x[t] + w2 x[t+1]."""
    n = len(x)
    return [w[0] * (x[t - 1] if t >= 1 else 0.0) + w[1] * x[t] + w[2] * (x[t + 1] if t + 1 < n else 0.0) for t in range(n)]


def causal_conv_by_hand():
    y = causal_conv(CONV_X, CONV_W)
    ys = causal_conv_stream(CONV_X, CONV_W)
    rows = []
    hist = [0.0, 0.0, 0.0]
    for t, xt in enumerate(CONV_X):
        rows.append([t, _fx(xt, 2), "(" + ", ".join(_fx(h, 2) for h in hist) + ")", _fx(y[t], 4), _fx(ys[t], 4)])
        hist = (hist + [xt])[1:]
    same = y == ys
    return (
        f"One channel, taps $w_0, \\dots, w_3$ = {_vec(CONV_W, 2)} (the last tap multiplies the current token), input $x$ = {_vec(CONV_X, 2)}, zeros before the start. "
        "'History' is the three inputs a streaming step keeps (oldest first) before it reads $x_t$. $y_t = \\sum_{j=0}^{3} w_j\\,x_{t-3+j}$.\n\n"
        + _table(["t", "x_t", "history before", "y_t, whole sequence", "y_t, streaming"], rows)
        + f"\n\nThe two columns are equal bit for bit: {'yes' if same else 'no'}."
    )


def symmetric_vs_causal():
    w = [0.5, 1.0, -0.5]
    x = [1.0, 2.0, 0.0, -1.0, 3.0, 1.0]
    base_c = causal_conv(x, w)
    base_s = symmetric_conv(x, w)
    t0 = 2
    rows = []
    for name, f, base in (("causal, width 3 (zeros on the left)", causal_conv, base_c), ("symmetric, width 3 (one zero each side)", symmetric_conv, base_s)):
        for pos, label in ((t0 - 1, f"x_{t0 - 1} (the past)"), (t0 + 1, f"x_{t0 + 1} (the future)")):
            x2 = x[:]
            x2[pos] += 1.0
            y2 = f(x2, w)
            rows.append([name, label, _fx(base[t0], 3), _fx(y2[t0], 3), "yes" if y2[t0] != base[t0] else "no"])
    return (
        f"Taps $w$ = {_vec(w, 2)}, input $x$ = {_vec(x, 2)}. Each row adds 1 to one input and reads output $y_{t0}$ again.\n\n"
        + _table(["convolution", "input changed", f"y_{t0} before", f"y_{t0} after", f"y_{t0} changed"], rows, align="llrrl")
    )


# ------------------------------------------------------------------------------------------------ 9. state-space models


def ssm_discretization():
    A, B = -2.0, 1.0
    rows = []
    for dt in (0.1, 0.25, 0.5, 1.0):
        exact = math.exp(dt * A)
        euler = 1.0 + dt * A
        b_zoh = (exact - 1.0) / A * B
        rows.append([dt, _fx(exact, 4), _fx(euler, 4), _fx(b_zoh, 4), _fx(dt * B, 4)])
    return (
        "Scalar state-space model $h' = A h + B x$ with $A = -2$ and $B = 1$ (so a constant input $x$ drives $h$ toward $x/2$). One step of "
        "size $\\Delta$: the exact (zero-order-hold) transition $\\bar A = \\exp(\\Delta A)$ and input gain $\\bar B = (\\bar A - 1)B/A$, against the "
        "Euler step $1 + \\Delta A$ and $\\Delta B$.\n\n"
        + _table(["step Delta", "A-bar exact", "Euler 1 + Delta A", "B-bar exact", "Euler Delta B"], rows)
    )


SCAN_A = [0.9, 0.5, 0.8, 0.3, 1.0, 0.6, 0.7, 0.4]
SCAN_B = [1.0, 2.0, -1.0, 0.5, 3.0, 0.0, 1.0, -2.0]


def _compose(first, second):
    """(a1, b1) then (a2, b2): h -> a2 (a1 h + b1) + b2 = (a2 a1) h + (a2 b1 + b2)."""
    return (second[0] * first[0], second[0] * first[1] + second[1])


def scan_equivalence():
    n = len(SCAN_A)
    h, seq = 0.0, []
    for a, b in zip(SCAN_A, SCAN_B):
        h = a * h + b
        seq.append(h)
    x = list(zip(SCAN_A, SCAN_B))
    rounds = []
    d = 1
    while d < n:
        x = [x[i] if i < d else _compose(x[i - d], x[i]) for i in range(n)]
        rounds.append(list(x))
        d *= 2
    scan = [b for _, b in x]
    rows = []
    for t in range(n):
        rows.append([t, SCAN_A[t], SCAN_B[t], _fx(seq[t], 4), _fx(scan[t], 4)])
    same = _max_abs_diff(seq, scan) < 1e-12
    return (
        "The linear recurrence $h_t = a_t h_{t-1} + b_t$ with $h_{-1} = 0$ for the eight $(a_t, b_t)$ below, computed in order and by the "
        "doubling scan of Volume IV, Chapter 7 (Hillis and Steele) with the operator $(a_1, b_1) \\star (a_2, b_2) = (a_2 a_1,\\; a_2 b_1 + b_2)$: round "
        "$s$ combines position $i$ with position $i - 2^{s}$ for every $i \\ge 2^{s}$. Since $h_{-1} = 0$, the second component of the scanned pair at "
        f"position $t$ is $h_t$. {len(rounds)} rounds for {n} positions.\n\n"
        + _table(["t", "a_t", "b_t", "h_t in order", "h_t by the scan"], rows)
        + f"\n\nThe two columns agree to below 1e-12: {'yes' if same else 'no'}."
    )


# ------------------------------------------------------------------------------------------------ 10. hybrid stacks


def hybrid_costs_data(t):
    """(cache bytes per sequence, token-mixer FLOPs per decoded token) for pure attention, the hybrid and pure linear, at t cached tokens."""
    z = _qw_derived()
    L = z["layers"]
    full_flops = 2 * z["full_proj"] + 4 * z["heads"] * z["hd"] * t
    lin_flops = 2 * z["lin_proj"] + 8 * z["nv"] * z["dk"] * z["dv"] + 2 * z["conv_k"] * z["conv_dim"]
    cache_full = z["kv_bytes_token"] * t
    n_full, n_lin = z["n_full"], z["n_lin"]
    return {
        "attention": (L * cache_full, L * full_flops),
        "hybrid": (n_full * cache_full + n_lin * z["state_bytes"], n_full * full_flops + n_lin * lin_flops),
        "linear": (L * z["state_bytes"], L * lin_flops),
    }


def hybrid_costs():
    z = _qw_derived()
    mem_rows, flop_rows = [], []
    for t in (1024, 8192, 32768, 131072, 262144):
        d = hybrid_costs_data(t)
        mem_rows.append([_int(t)] + [_gib(d[k][0]) for k in ("attention", "hybrid", "linear")])
        flop_rows.append([_int(t)] + [f"{d[k][1] / 1e9:.2f}" for k in ("attention", "hybrid", "linear")])
    return (
        f"The {z['layers']}-layer shape of `qwen38_27b` (full attention in every 4th layer: {z['n_full']} full-attention layers and {z['n_lin']} Gated DeltaNet layers), against "
        f"a pure stack of the same width with full attention in all {z['layers']} layers (same 4 key-value heads of width 256), and a pure stack of Gated DeltaNet layers. "
        f"KV cache in bf16 ({z['kv_bytes_token']} bytes per token and full-attention layer), recurrent state and convolution history in fp32 ({_int(z['state_bytes'])} bytes per Gated DeltaNet layer). "
        f"Token-mixer projections: {_int(z['full_proj'])} parameters in a full-attention layer and {_int(z['lin_proj'])} in a Gated DeltaNet layer, 2 FLOPs each per token; "
        f"attention core $4 H d_h T$ = {_int(4 * z['heads'] * z['hd'])} T FLOPs per layer; Gated DeltaNet core 8 x {z['nv']} x {z['dk']} x {z['dv']} for the states plus 2 x {z['conv_k']} x {_int(z['conv_dim'])} "
        "for the convolution. Normalisations, gates, the feed-forward network and the output head are the same in all three stacks and left out.\n\n"
        "Bytes held for one sequence (GiB; a decode step reads all of the cache, and reads and writes the state):\n\n"
        + _table(["tokens", "pure attention", "hybrid", "pure Gated DeltaNet"], mem_rows)
        + "\n\nToken-mixer FLOPs per decoded token (GFLOP):\n\n"
        + _table(["tokens", "pure attention", "hybrid", "pure Gated DeltaNet"], flop_rows)
    )


def hybrid_curve_data(points=None):
    """Token-mixer GFLOP per decoded token against the context, in thousands of tokens, for the three stacks."""
    points = points or [i * 4096 for i in range(0, 33)]
    out = {"attention": [], "hybrid": [], "linear": []}
    for t in points:
        d = hybrid_costs_data(max(t, 1))
        for k in out:
            out[k].append(d[k][1] / 1e9)
    return [t / 1024 for t in points], out


# ------------------------------------------------------------------------------------------------ 11. LFM2.5 layers


LFM = dict(d=1024, heads=16, kv_heads=8, hd=64, ff=2560, conv_k=3,
           pattern="CCACACACACACAC")   # LFM2.5-Encoder-230M: C = gated short-convolution layer, A = bidirectional attention layer


def lfm_layers():
    L = LFM
    d = L["d"]
    attn_params = d * L["heads"] * L["hd"] + 2 * d * L["kv_heads"] * L["hd"] + L["heads"] * L["hd"] * d
    conv_params = 3 * d * d + d * L["conv_k"] + d * d
    ffn = 3 * d * L["ff"]
    n_c, n_a = L["pattern"].count("C"), L["pattern"].count("A")
    rows = []
    for t in (512, 2048, 8192):
        conv_core = 2 * L["conv_k"] * d
        attn_core = 4 * t * L["heads"] * L["hd"]
        rows.append([_int(t), _int(2 * conv_params + conv_core), _int(2 * attn_params + attn_core)])
    stack_rows = [["gated short-convolution layer", _int(conv_params), _int(ffn), _int(conv_params + ffn), n_c],
                  ["bidirectional attention layer", _int(attn_params), _int(ffn), _int(attn_params + ffn), n_a]]
    return (
        f"LFM2.5-Encoder-230M (`LfmConfig::lfm25_encoder_230m`): model width {d}, {L['heads']} query heads and {L['kv_heads']} key-value heads of width {L['hd']}, SwiGLU width {_int(L['ff'])}, "
        f"convolution width {L['conv_k']}; layer pattern (C convolution, A attention) `{L['pattern']}`. A convolution layer has the fused input projection $d \\to 3d$, "
        "the depthwise taps $d \\times 3$ and the output projection $d \\to d$; an attention layer has $\\mathbf{W}_Q$, $\\mathbf{W}_K$, $\\mathbf{W}_V$, $\\mathbf{W}_O$ (the two head-width "
        "gains of the query and key normalisation are left out).\n\n"
        + _table(["layer", "mixer parameters", "feed-forward parameters", "total", "layers of this kind"], stack_rows)
        + "\n\nToken-mixer FLOPs per token over a sequence of $T$ tokens: projections $2P$ plus, for the convolution, $2 \\cdot 3 d$, for attention $4 T H d_h$ (every token attends to all $T$):\n\n"
        + _table(["T", "convolution layer", "attention layer"], rows)
    )
