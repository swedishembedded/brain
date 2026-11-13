# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Attention Without the Quadratic Memory".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine
code. Each public function without arguments returns a markdown string that the chapter embeds between
`<!-- worked: flash_attention.<name> -->` markers; the `*_data` and other functions with arguments supply the
numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* the score matrix against the inputs it is computed from;
* the online softmax of a printed row of eight scores in chunks of two, with the weighted sum of values, and the merge
  of two partial states;
* the bytes a head moves when attention is materialised and when it is tiled, and the FLOPs of both;
* a tiled forward kernel run on the kernel simulator of Volume IV (Chapter 5) and checked against materialised attention;
* causal masking in tiles, and the first-tile hazard of the minus-infinity convention;
* the tiled backward pass from the saved output and log-sum-exp, checked against the materialised backward pass and
  against central differences, with three deliberate mistakes;
* one query against a long cache split across workgroups, with the combine of the partial states;
* float32 rounding: tiled and materialised attention agree to tolerance and not to the bit.

Arithmetic is IEEE double precision except where a block says float32. Sums of products are plain left-to-right
additions, the order a C loop would use (never Python's `sum`, which may compensate). Every random quantity comes from
the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed.

Swedish Embedded AB implements solutions for attention kernels that fit long sequences in the memory of the device. If
your team needs expertise in fused attention, its backward pass and its numerics, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import struct

import bookrng
import compute_kernels as ck
import fusion
import self_attention as sa

INF = math.inf
F32 = 4

# ------------------------------------------------------------------------------------------------ formatting


def _int(n):
    return f"{n:,}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _bound(x, floor=1e-12):
    """The power of ten above x (never below `floor`, the round-off level of the arithmetic), written 1e-12: a difference is
    reported by the bound it stays under, not by its digits, which another implementation would not reproduce."""
    x = max(x, floor / 10)
    return f"1e{math.ceil(math.log10(x * 1.0000001)):d}".replace("e-0", "e-")


def _gib(b):
    return b / 2 ** 30


def _vec(v, d=3):
    return "(" + ", ".join(f"{x:.{d}f}" for x in v) + ")"


# ------------------------------------------------------------------------------------------------ plain linear algebra


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


def _normals(g, rows, cols, factor=1.0):
    return [[factor * g.normal() for _ in range(cols)] for _ in range(rows)]


def _max_abs(a, b):
    """Largest absolute difference between two equally shaped lists (vectors or matrices)."""
    if a and isinstance(a[0], list):
        return max(_max_abs(x, y) for x, y in zip(a, b))
    return max(abs(x - y) for x, y in zip(a, b))


def _ceil_div(a, b):
    return -(-a // b)


# ------------------------------------------------------------------------------------------------ materialised attention


def attention_reference(Q, K, V, causal=False, scale=None):
    """Attention of one head as Chapter 7 defines it: O = softmax(scale Q K^T + mask) V.
    Returns (O, lse, P) with lse[i] the log-sum-exp of the scaled, masked scores of row i."""
    T, d = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    O, lse, P = [], [], []
    for i in range(T):
        s = [scale * _dot(Q[i], K[j]) if (not causal or j <= i) else -INF for j in range(T)]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        z = _sum(e)
        p = [v / z for v in e]
        P.append(p)
        lse.append(m + math.log(z))
        O.append([_sum([p[j] * V[j][c] for j in range(T)]) for c in range(len(V[0]))])
    return O, lse, P


def backward_reference(Q, K, V, dO, causal=False, scale=None):
    """The materialised backward of Chapter 7: dP = dO V^T, dV = P^T dO, dS = P (dP - rowsum(P dP)), dQ = scale dS K,
    dK = scale dS^T Q."""
    T, d = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    _, _, P = attention_reference(Q, K, V, causal, scale)
    dP = sa.matmul(dO, sa.transpose(V))
    dV = sa.matmul(sa.transpose(P), dO)
    dS = []
    for i in range(T):
        c = _dot(P[i], dP[i])
        dS.append([P[i][j] * (dP[i][j] - c) for j in range(T)])
    dQ = [[scale * x for x in row] for row in sa.matmul(dS, K)]
    dK = [[scale * x for x in row] for row in sa.matmul(sa.transpose(dS), Q)]
    return dQ, dK, dV


# ------------------------------------------------------------------------------------------------ online softmax and its merge


def empty_state(width):
    """The state of no keys at all: maximum minus infinity, sum zero, output zero. It is the identity of `merge`."""
    return (-INF, 0.0, [0.0] * width)


def update_state(state, scores, values):
    """Fold a chunk of scores and the values they weight into a running state (m, l, o): the online-softmax update.
    A chunk whose scores are all minus infinity changes nothing."""
    m, l, o = state
    top = max(scores)
    if top == -INF:
        return state
    m_new = max(m, top)
    corr = math.exp(m - m_new)                       # exp(-inf) = 0 when nothing has been seen yet
    p = [math.exp(s - m_new) for s in scores]
    l_new = l * corr + _sum(p)
    o_new = [o[c] * corr + _sum([p[j] * values[j][c] for j in range(len(p))]) for c in range(len(o))]
    return (m_new, l_new, o_new)


def merge_states(a, b, drop_correction=False):
    """Combine the states of two disjoint sets of keys. `drop_correction` leaves the factor of the second state out of
    the sum l, the mistake that the combine test of the chapter is built to catch."""
    ma, la, oa = a
    mb, lb, ob = b
    m = max(ma, mb)
    if m == -INF:
        return a
    ca, cb = math.exp(ma - m), math.exp(mb - m)
    l = la * ca + (lb if drop_correction else lb * cb)
    return (m, l, [oa[c] * ca + ob[c] * cb for c in range(len(oa))])


def state_of(scores, values):
    return update_state(empty_state(len(values[0])), scores, values)


# ------------------------------------------------------------------------------------------------ 1. the matrix against the inputs

H_BIG, DH_BIG = 16, 64


def matrix_versus_inputs():
    d_model = H_BIG * DH_BIG
    rows = []
    for t in (2048, 8192, 32768):
        mat = H_BIG * t * t * F32
        inputs = 4 * t * d_model * F32            # Q, K, V and the output O: four [T, D] matrices
        rows.append((_int(t), _int(mat), f"{_gib(mat):.2f}", _int(inputs), f"{inputs / 2 ** 20:,.0f}", f"{mat / inputs:.1f}"))
    return (f"One attention layer with H = {H_BIG} heads of width d_h = {DH_BIG} (D = {d_model}) in float32 (4 bytes per number). "
            "The score matrix of all heads has H T^2 entries; the inputs and result are four matrices of shape [T, D] (Q, K, V and "
            "the output O). Chapter 7 gave the same matrix for 24 layers; here one layer is set against the numbers it is computed from.\n\n"
            + _table(["T", "score matrix, bytes", "GiB", "Q, K, V and O, bytes", "MiB", "matrix / (Q, K, V, O)"], rows))


# ------------------------------------------------------------------------------------------------ 2. the online softmax

ROW_SCORES = [2.0, -1.5, 0.5, 3.0, 1.0, 4.5, -0.5, 4.0]
ROW_VALUES = [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [-1.0, 2.0], [0.5, -1.0], [1.0, 1.0], [-2.0, 0.0], [0.0, -2.0]]
CHUNK = 2


def softmax_stable(s):
    m = max(s)
    e = [math.exp(v - m) for v in s]
    z = _sum(e)
    return [v / z for v in e]


def online_row_data():
    """(chunk scores, m, correction, l) after each chunk of the row, and the final weights from the state."""
    m, l, out = -INF, 0.0, []
    for k in range(0, len(ROW_SCORES), CHUNK):
        chunk = ROW_SCORES[k:k + CHUNK]
        m_new = max(m, max(chunk))
        corr = math.exp(m - m_new)
        l = l * corr + _sum([math.exp(s - m_new) for s in chunk])
        m = m_new
        out.append((chunk, m, corr, l))
    weights = [math.exp(s - m) / l for s in ROW_SCORES]
    return out, weights


def online_row():
    steps, weights = online_row_data()
    want = softmax_stable(ROW_SCORES)
    rows = []
    for k, (chunk, m, corr, l) in enumerate(steps):
        rows.append((k, _vec(chunk, 1), f"{m:.1f}", f"{corr:.6f}", f"{l:.6f}"))
    ok = _max_abs(weights, want)
    return (f"The row of eight scores s = {_vec(ROW_SCORES, 1)}, read in chunks of {CHUNK}. The state starts at m = -infinity and l = 0; "
            "per chunk, m' = max(m, largest score of the chunk), the correction is exp(m - m'), and l' = l exp(m - m') + the sum of "
            "exp(s - m') over the chunk. exp is the library exp of double precision; the last column is printed to six decimals.\n\n"
            + _table(["chunk", "scores", "m", "exp(m_old - m)", "l"], rows)
            + f"\n\nWeights from the final state, exp(s_j - m) / l, are {_vec(weights, 6)}. The ordinary stable softmax of the whole row "
            f"gives the same weights; the largest difference over the eight is below {_bound(ok)}.")


def online_values():
    state = empty_state(2)
    rows = []
    for k in range(0, len(ROW_SCORES), CHUNK):
        state = update_state(state, ROW_SCORES[k:k + CHUNK], ROW_VALUES[k:k + CHUNK])
        m, l, o = state
        n = k + CHUNK
        prefix = softmax_stable(ROW_SCORES[:n])
        direct = [_sum([prefix[j] * ROW_VALUES[j][c] for j in range(n)]) for c in range(2)]
        got = [v / l for v in o]
        rows.append((k // CHUNK, f"{m:.1f}", f"{l:.6f}", _vec(o, 6), _vec(got, 6), _bound(_max_abs(got, direct))))
    w = softmax_stable(ROW_SCORES)
    full = [_sum([w[j] * ROW_VALUES[j][c] for j in range(8)]) for c in range(2)]
    final = [v / state[1] for v in state[2]]
    vals = ", ".join(_vec(v, 1) for v in ROW_VALUES)
    return (f"The same row with a value of width 2 for each score, v_0 .. v_7 = {vals}. The state is (m, l, o) with o the running "
            "weighted sum, rescaled by exp(m - m') whenever the maximum grows. The last two columns compare o / l with the attention "
            "output over the prefix read so far, softmax of the first scores applied to the first values; the bound is the power of "
            "ten the largest difference stays below.\n\n"
            + _table(["chunk", "m", "l", "o", "o / l", "differs from prefix attention by less than"], rows)
            + f"\n\nAfter the last chunk, o / l = {_vec(final, 6)}; the materialised softmax(s) V is {_vec(full, 6)}, and the two "
            f"agree to below {_bound(_max_abs(final, full))}.")


def merge_demo():
    parts = [(0, 3), (3, 5), (5, 8)]
    st = [state_of(ROW_SCORES[a:b], ROW_VALUES[a:b]) for a, b in parts]
    rows = []
    for (a, b), (m, l, o) in zip(parts, st):
        rows.append((f"keys {a} .. {b - 1}", f"{m:.1f}", f"{l:.6f}", _vec(o, 6)))
    left = merge_states(merge_states(st[0], st[1]), st[2])
    right = merge_states(st[0], merge_states(st[1], st[2]))
    swapped = merge_states(merge_states(st[2], st[0]), st[1])
    whole = state_of(ROW_SCORES, ROW_VALUES)
    for label, s in (("(A + B) + C", left), ("A + (B + C)", right), ("(C + A) + B", swapped), ("all eight in one pass", whole)):
        rows.append((label, f"{s[0]:.1f}", f"{s[1]:.6f}", _vec(s[2], 6)))
    spread = max(_max_abs(left[2], whole[2]), _max_abs(right[2], whole[2]), _max_abs(swapped[2], whole[2]))
    return ("The row of the previous blocks cut into three sets of keys, each summarised by its own state (m, l, o) from the empty state, "
            "then merged: m = max(m_A, m_B), l = l_A exp(m_A - m) + l_B exp(m_B - m), and o likewise.\n\n"
            + _table(["states", "m", "l", "o"], rows)
            + f"\n\nThe three orders of merging give the state of the single pass; the largest difference of the outputs o is below {_bound(spread)} "
            f"and that of l below {_bound(max(abs(left[1] - whole[1]), abs(right[1] - whole[1]), abs(swapped[1] - whole[1])))}.")


# ------------------------------------------------------------------------------------------------ 3. bytes moved and FLOPs

D_HEAD = 64
M_FLOATS = {"48 KiB": 12288, "96 KiB": 24576}


def words_materialised(t, d):
    """Words moved by the three-kernel chain of Chapter 8: 4 T^2 + 4 T d."""
    return 4 * t * t + 4 * t * d


def words_chain_tiled_products(t, d, tile):
    """The same chain when its two matrix products are tiled kernels of Chapter 6 with output tile `tile`: loads
    k (m ceil(n / T) + n ceil(m / T)) for each product (which includes reading S in the second), S written, S read and P
    written by the softmax, and the output written."""
    qk = d * (t * _ceil_div(t, tile) + t * _ceil_div(t, tile))               # [T, d] x [d, T]
    pv = t * (t * _ceil_div(d, tile) + d * _ceil_div(t, tile))               # [T, T] x [T, d]
    return qk + t * t + 2 * t * t + pv + t * d                                # + write S, softmax read S and write P, + write O


def words_tiled(t, d, bq):
    """Words moved by tiled attention with query blocks of bq rows: Q once, K and V once per query block, O and the
    log-sum-exp once."""
    return t * d + 2 * t * d * _ceil_div(t, bq) + t * d + t


def tile_for_memory(m_floats, d):
    """Largest B with a query block, an output block and a key tile and a value tile of B rows (4 B d floats) plus the B x B
    scores fitting in m_floats: 4 B d + B^2 <= M."""
    b = int(math.sqrt(4 * d * d + m_floats) - 2 * d)
    while 4 * b * d + b * b > m_floats:
        b -= 1
    return b


def bytes_cases(t=16384, d=D_HEAD, bqs=(16, 64, 256)):
    chain = F32 * words_materialised(t, d)
    return chain, [(bq, F32 * words_tiled(t, d, bq)) for bq in bqs]


def _mib(b):
    return f"{b / 2 ** 20:,.1f}"


def bytes_moved():
    rows = []
    for t in (1024, 4096, 16384, 65536):
        d = D_HEAD
        assert F32 * words_materialised(t, d) == fusion._attention(t, d)[1]       # the count of Chapter 8
        row = [_int(t), _mib(F32 * words_materialised(t, d)), _mib(F32 * words_chain_tiled_products(t, d, 64))]
        for bq in (16, 64, 256):
            row.append(_mib(F32 * words_tiled(t, d, bq)))
        rows.append(row)
    t, d = 16384, D_HEAD
    chain = words_materialised(t, d)
    notes = []
    for bq in (16, 32, 64, 256):
        notes.append(f"B_q = {bq}: {words_tiled(t, d, bq) / chain:.3f}")
    return (f"Bytes moved to and from global memory by one head of width d_h = {d} in float32, in MiB (1 MiB = 1,048,576 bytes), as a function "
            "of the sequence length T. Materialised: 16 T^2 + 16 T d_h bytes (the count of Volume IV, Chapter 8; asserted equal to the unfused count of "
            "`fusion.attention_bytes`). Chain with tiled products: the same three kernels when the two matrix products are tiled kernels of "
            "output tile 64 (loads k (m ceil(n / 64) + n ceil(m / 64)) for each, Volume IV, Chapter 6). Tiled attention: 4 (2 T d_h + 2 T d_h ceil(T / B_q) + T) "
            "bytes, the query block read once, K and V read again for every query block, the output and the log-sum-exp "
            "written once.\n\n"
            + _table(["T", "materialised", "chain, tiled products", "tiled, B_q = 16", "tiled, B_q = 64", "tiled, B_q = 256"], rows)
            + f"\n\nTiled words over materialised words at T = {t}: " + "; ".join(notes) + ".")


def tile_choice():
    rows = []
    t = 16384
    for label, m in M_FLOATS.items():
        for d in (64, 128):
            b = tile_for_memory(m, d)
            tiled = words_tiled(t, d, b)
            mat = words_materialised(t, d)
            rows.append((label, d, b, _int(4 * b * d + b * b), f"{tiled / mat:.2f}", f"{tiled / (8 * t * t * d * d / m):.2f}"))
    return (f"The tile that fits in M floats of fast memory: the largest B with 4 B d_h + B^2 <= M (a query block, an output block, a key "
            f"tile and a value tile of B rows each, and the B by B scores), for T = {t} and the two memory sizes of 48 and 96 KiB "
            "(M = 12,288 and 24,576 floats). 'Words tiled' is 2 T d_h + 2 T d_h ceil(T / B) + T and 'words materialised' is 4 T^2 + 4 T d_h; "
            "the last column is words tiled over the estimate 8 T^2 d_h^2 / M of the text.\n\n"
            + _table(["fast memory", "d_h", "B", "floats in use", "words tiled / words materialised", "words tiled / (8 T^2 d_h^2 / M)"], rows))


def flop_counts(t=4096, d=64):
    """FLOPs of one head, non-causal. Materialised forward: 4 T^2 d for the products and 5 T^2 for the softmax (scale, subtract,
    exp, add, divide). Tiled forward: the same products, 4 per score (the division moves to the T d outputs) and, per row and
    tile, d + 4 for the rescale (subtract, exp, multiply l, add, d multiplies of o)."""
    rows = []
    mat = 4 * t * t * d + 5 * t * t
    rows.append(("materialised", "-", _int(mat), "1.0000"))
    for bk in (8, 16, 64, 128):
        extra = t * _ceil_div(t, bk) * (d + 4)
        tiled = 4 * t * t * d + 4 * t * t + t * d + extra
        rows.append(("tiled", bk, _int(tiled), f"{tiled / mat:.4f}"))
    return (f"FLOPs of the forward pass of one head, T = {t}, d_h = {d}, no mask. Materialised: 4 T^2 d_h for the two products plus 5 T^2 "
            "for the softmax (Chapter 7). Tiled: the same products, 4 T^2 for scaling, subtracting the maximum, the exponential and "
            "the addition to l, T d_h for the final division, and T ceil(T / B_k) (d_h + 4) for the rescales (subtract, exponential, "
            "multiply l, add, and d_h multiplies of the output).\n\n"
            + _table(["algorithm", "B_k", "FLOPs", "relative to materialised"], rows))


# ------------------------------------------------------------------------------------------------ 4. a tiled kernel on the simulator

SIM_T, SIM_D, SIM_BQ, SIM_BK = 11, 4, 4, 3
NAME_SIM = "flash_attention.simulated"


def sim_inputs():
    g = bookrng.generator(NAME_SIM)
    Q = _normals(g, SIM_T, SIM_D)
    K = _normals(g, SIM_T, SIM_D)
    V = _normals(g, SIM_T, SIM_D)
    return Q, K, V


def _wg_tiles(w, t, bq, bk, causal):
    """Key tiles the workgroup w of query rows visits: all of them, or (causal) up to the tile of its last row."""
    total = _ceil_div(t, bk)
    if not causal:
        return total
    last_row = min(w * bq + bq - 1, t - 1)
    return min(total, last_row // bk + 1)


def kv_words_loaded(t, d, bq, bk, causal):
    """Key and value numbers loaded by all workgroups, counted from the tile geometry."""
    words = 0
    for w in range(_ceil_div(t, bq)):
        rows = min(_wg_tiles(w, t, bq, bk, causal) * bk, t)
        words += 2 * rows * d
    return words


def _flash_phases(counters, bq, bk, tiles_total, causal):
    """The tiled forward kernel as phases of the simulator of Chapter 5: one workgroup of bq invocations per block of bq query rows,
    each invocation owning one row (its q, m, l and o live in registers, the dict `regs`); the key tile and value tile of bk rows
    are shared memory; two phases per key tile (load, then compute) with a barrier between."""
    regs = {}

    def load(kt):
        def phase(inv, p):
            t, d = p["T"], p["d"]
            w = inv.wx
            row = w * bq + inv.l
            key = (w, inv.l)
            if kt == 0:
                if row < t:
                    q = [inv.load("Q", row * d + c) for c in range(d)]
                    counters["global_loads"] += d
                else:
                    q = [0.0] * d
                regs[key] = {"q": q, "m": -INF, "l": 0.0, "o": [0.0] * d}
            if kt >= _wg_tiles(w, t, bq, bk, causal):
                return
            for e in range(inv.l, bk * d, bq):                    # the invocations stride over the tile
                jr, c = divmod(e, d)
                j = kt * bk + jr
                if j < t:
                    inv.shared_store(e, inv.load("K", j * d + c))
                    inv.shared_store(bk * d + e, inv.load("V", j * d + c))
                    counters["global_loads"] += 2
                else:
                    inv.shared_store(e, 0.0)
                    inv.shared_store(bk * d + e, 0.0)
        return phase

    def compute(kt):
        def phase(inv, p):
            t, d = p["T"], p["d"]
            w = inv.wx
            row = w * bq + inv.l
            if row >= t or kt >= _wg_tiles(w, t, bq, bk, causal):
                return
            r = regs[(w, inv.l)]
            s = []
            for jr in range(bk):
                j = kt * bk + jr
                if j >= t or (causal and j > row):
                    s.append(-INF)
                else:
                    s.append(p["scale"] * _dot(r["q"], [inv.shared_load(jr * d + c) for c in range(d)]))
            counters["tiles"] += 1
            top = max(s)
            if top == -INF:
                return
            m_new = max(r["m"], top)
            corr = math.exp(r["m"] - m_new)
            pw = [math.exp(v - m_new) for v in s]
            r["l"] = r["l"] * corr + _sum(pw)
            r["o"] = [r["o"][c] * corr + _sum([pw[jr] * inv.shared_load(bk * d + jr * d + c) for jr in range(bk)]) for c in range(d)]
            r["m"] = m_new
        return phase

    def finish(inv, p):
        t, d = p["T"], p["d"]
        row = inv.wx * bq + inv.l
        if row >= t:
            return
        r = regs[(inv.wx, inv.l)]
        for c in range(d):
            inv.store("O", row * d + c, r["o"][c] / r["l"])
        inv.store("LSE", row, r["m"] + math.log(r["l"]))
        counters["global_stores"] += d + 1

    phases = []
    for kt in range(tiles_total):
        phases += [load(kt), compute(kt)]
    phases.append(finish)
    return phases


def simulate_forward(Q, K, V, bq, bk, causal):
    """Run the tiled forward on the simulator. Returns (O, lse, counters, memory buffer sizes)."""
    t, d = len(Q), len(Q[0])
    mem = ck.Memory()
    bufs = {"Q": mem.alloc("Q", [x for r in Q for x in r], "read"), "K": mem.alloc("K", [x for r in K for x in r], "read"),
            "V": mem.alloc("V", [x for r in V for x in r], "read"), "O": mem.alloc("O", [ck.UNWRITTEN] * (t * d)),
            "LSE": mem.alloc("LSE", [ck.UNWRITTEN] * t)}
    counters = {"global_loads": 0, "global_stores": 0, "tiles": 0}
    tiles_total = _ceil_div(t, bk)
    phases = _flash_phases(counters, bq, bk, tiles_total, causal)
    ck.launch(mem, phases, _ceil_div(t, bq), bq, {"T": t, "d": d, "scale": 1.0 / math.sqrt(d)}, bufs, shared_words=2 * bk * d)
    O = [mem.values("O")[i * d:(i + 1) * d] for i in range(t)]
    return O, mem.values("LSE"), counters, {name: b.n for name, b in bufs.items()}


def simulated_forward():
    Q, K, V = sim_inputs()
    t, d, bq, bk = SIM_T, SIM_D, SIM_BQ, SIM_BK
    rows = []
    for causal in (False, True):
        O, lse, c, sizes = simulate_forward(Q, K, V, bq, bk, causal)
        want_o, want_lse, _ = attention_reference(Q, K, V, causal)
        n_wg = _ceil_div(t, bq)
        visited = sum(_wg_tiles(w, t, bq, bk, causal) for w in range(n_wg))
        full = n_wg * _ceil_div(t, bk)
        formula = t * d + kv_words_loaded(t, d, bq, bk, causal)
        assert c["global_loads"] == formula
        rows.append(("causal" if causal else "no mask", n_wg, f"{visited} of {full}", _int(c["global_loads"]),
                     _int(c["global_stores"]), _bound(_max_abs(O, want_o)), _bound(_max_abs(lse, want_lse))))
    biggest = max(sizes.values())
    return (f"{('Generator: the book\'s SplitMix64, seed ' + _seed(NAME_SIM) + ' (`seed_of` of the name `' + NAME_SIM + '`).')} T = {t} rows, "
            f"d_h = {d}, query blocks of B_q = {bq} rows (one workgroup of {bq} invocations each) and key tiles of B_k = {bk} rows, so "
            f"{_ceil_div(t, bq)} workgroups and {_ceil_div(t, bk)} key tiles, neither a divisor of T. Draws, in this order: Q, K, V "
            f"(each {t} by {d}, row by row, `normal()`). The kernel runs on the simulator of Volume IV, Chapter 5, in double precision "
            f"(scale 1 / sqrt(d_h)); the key tile and the value tile are shared memory ({2 * bk * d} words), the query row, m, l and "
            f"the output row of each invocation are registers. Loads and stores count numbers (words) moved to or from the "
            f"buffers Q, K, V, O and the log-sum-exp; the loads equal T d_h plus the key and value rows each workgroup loads, counted from "
            f"the tile geometry. The largest buffer of the kernel has {biggest} words, against T^2 = {t * t}.\n\n"
            + _table(["mask", "workgroups", "key tiles visited", "global loads", "global stores", "O differs by less than",
                      "log-sum-exp differs by less than"], rows))


# ------------------------------------------------------------------------------------------------ 5. tiles under a causal mask


def tile_classes(t, bq, bk):
    """Grid of classes of the (query block, key block) pairs: 'skipped' (every key is after every query row of the block),
    'full' (every key is at or before every query row) and 'diagonal' (the mask cuts the tile)."""
    grid = []
    for qb in range(_ceil_div(t, bq)):
        r0, r1 = qb * bq, min(qb * bq + bq - 1, t - 1)
        row = []
        for kb in range(_ceil_div(t, bk)):
            k0, k1 = kb * bk, min(kb * bk + bk - 1, t - 1)
            row.append("skipped" if k0 > r1 else "full" if k1 <= r0 else "diagonal")
        grid.append(row)
    return grid


def causal_tile_rows(configs):
    rows = []
    for t, bq, bk in configs:
        grid = tile_classes(t, bq, bk)
        flat = [c for r in grid for c in r]
        full, diag, skip = flat.count("full"), flat.count("diagonal"), flat.count("skipped")
        computed = sum(min(bq, t - qb * bq) * min(bk, t - kb * bk) for qb, r in enumerate(grid) for kb, c in enumerate(r) if c != "skipped")
        rows.append((_int(t), f"{bq} x {bk}", _int(len(flat)), _int(full), _int(diag), _int(skip), f"{computed / (t * t):.4f}"))
    return rows


def causal_tiles():
    rows = causal_tile_rows([(24, 4, 4), (24, 8, 4), (4096, 64, 64), (4096, 64, 8), (32768, 64, 8)])
    return ("Tiles of the score matrix under the causal mask. A tile is the set of scores of B_q consecutive query rows against B_k "
            "consecutive keys: skipped when every key is after every row of the block, full when every key is at or before the first "
            "row, diagonal otherwise (the last block of a ragged T is smaller). The scores the kernel computes are those of the tiles that "
            "are not skipped, counted entry by entry as a fraction of T^2; the mask lets (T + 1) / (2 T) of them through "
            "(0.5208 at T = 24, 0.5001 at T = 4096).\n\n"
            + _table(["T", "B_q x B_k", "tiles", "full", "diagonal", "skipped", "scores computed / T^2"], rows))


FIRST_TILE_SCORES = [0.5, 1.0, 2.0, -1.0]      # scores of one row, in tiles of two
FIRST_TILE_VALUES = [[1.0, 2.0], [3.0, -1.0], [0.0, 1.5], [2.0, 2.0]]
FIRST_TILE_VISIBLE = [False, False, True, True]  # the first tile (keys 0 and 1) is masked for this row


def masked_first_tile():
    s = [v if ok else -INF for v, ok in zip(FIRST_TILE_SCORES, FIRST_TILE_VISIBLE)]
    visible = [k for k, ok in enumerate(FIRST_TILE_VISIBLE) if ok]
    p = softmax_stable([s[k] for k in visible])
    truth = [_sum([p[a] * FIRST_TILE_VALUES[k][c] for a, k in enumerate(visible)]) for c in range(2)]

    def run(minus_inf, guard):
        sentinel = -INF if minus_inf else -3.4e38
        m, l, o = sentinel, 0.0, [0.0, 0.0]
        for a in (0, 2):
            chunk = [v if ok else sentinel for v, ok in zip(FIRST_TILE_SCORES[a:a + 2], FIRST_TILE_VISIBLE[a:a + 2])]
            top = max(chunk)
            if guard and top == sentinel:
                continue
            m_new = max(m, top)
            corr = math.exp(m - m_new)
            pw = [math.exp(v - m_new) for v in chunk]
            l = l * corr + _sum(pw)
            o = [o[c] * corr + _sum([pw[j] * FIRST_TILE_VALUES[a + j][c] for j in range(2)]) for c in range(2)]
            m = m_new
        return [v / l for v in o]

    def show(v):
        return _vec(v, 6) if all(x == x for x in v) else "(nan, nan)"

    rows = [("the softmax over the visible keys only", show(truth), "-"),
            ("minus infinity, no test of the tile", show(run(True, False)), "nan from -inf minus -inf"),
            ("a finite sentinel -3.4e38, no test", show(run(False, False)), _bound(_max_abs(run(False, False), truth))),
            ("minus infinity, tile skipped when all of it is masked", show(run(True, True)), _bound(_max_abs(run(True, True), truth)))]
    # a row with no visible key at all
    none = [[-3.4e38, -3.4e38], [-3.4e38, -3.4e38]]
    m, l, o = -3.4e38, 0.0, [0.0, 0.0]
    for a in (0, 2):
        chunk = none[a // 2]
        top = max(chunk)
        m_new = max(m, top)
        corr = math.exp(m - m_new)
        pw = [math.exp(v - m_new) for v in chunk]
        l = l * corr + _sum(pw)
        o = [o[c] * corr + _sum([pw[j] * FIRST_TILE_VALUES[a + j][c] for j in range(2)]) for c in range(2)]
        m = m_new
    nothing = [v / l for v in o]
    mean_v = [_sum([FIRST_TILE_VALUES[j][c] for j in range(4)]) / 4 for c in range(2)]
    return ("One row of four keys in two tiles of two; the scores are " + _vec(FIRST_TILE_SCORES, 1) + " and the row sees only keys 2 and "
            "3, so its first tile is masked entirely. The values are " + ", ".join(_vec(v, 1) for v in FIRST_TILE_VALUES)
            + ". Output of the row from the exact softmax of its visible keys and from three ways of running the online update (double precision):\n\n"
            + _table(["method", "output", "differs from the exact softmax by less than"], rows, "lll")
            + f"\n\nA row that sees none of the four keys, with the sentinel and no test, ends with l = {l:.1f} and the output "
            f"{_vec(nothing, 6)}, the plain mean {_vec(mean_v, 6)} of the four values.")


# ------------------------------------------------------------------------------------------------ 6. the tiled backward pass

CHECK_T, CHECK_D, CHECK_BQ, CHECK_BK = 7, 3, 3, 2
NAME_CHECK = "flash_attention.backward"
EPS = 1e-5


def forward_tiled(Q, K, V, bq, bk, causal, scale=None):
    """The tiled forward in plain Python: returns (O, lse, m) where m is the final running maximum of each row and
    lse = m + log l."""
    t, d = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    O, lse, ms = [None] * t, [None] * t, [None] * t
    for r0 in range(0, t, bq):
        rows = range(r0, min(r0 + bq, t))
        state = {i: empty_state(len(V[0])) for i in rows}
        for k0 in range(0, t, bk):
            if causal and k0 > rows[-1]:
                continue
            keys = range(k0, min(k0 + bk, t))
            for i in rows:
                s = [scale * _dot(Q[i], K[j]) if (not causal or j <= i) else -INF for j in keys]
                state[i] = update_state(state[i], s, [V[j] for j in keys])
        for i in rows:
            m, l, o = state[i]
            O[i], lse[i], ms[i] = [v / l for v in o], m + math.log(l), m
    return O, lse, ms


def backward_tiled(Q, K, V, O, lse, dO, bq, bk, causal, scale=None, mistake=None, m=None, meter=None):
    """The tiled backward from the saved O and lse, in the two passes of brain's kernels: a query-side pass that forms
    D_i = dO_i . O_i and dQ, and a key-side pass that forms dK and dV. Every score is recomputed from Q and K and the weight
    is exp(s - lse_i); nothing T by T is stored. `mistake` is one of 'no_D' (D_i taken as 0), 'running_max' (the saved m_i
    used where lse_i belongs) or 'no_mask' (the causal mask left out of the recomputed scores). `meter` counts tile products."""
    t, d = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    masked = causal and mistake != "no_mask"
    D = [0.0 if mistake == "no_D" else _dot(dO[i], O[i]) for i in range(t)]
    L = list(m) if mistake == "running_max" else list(lse)

    def tile_weights(i, js):
        s = [scale * _dot(Q[i], K[j]) if (not masked or j <= i) else -INF for j in js]
        p = [math.exp(v - L[i]) for v in s]
        dp = [_dot(dO[i], V[j]) for j in js]
        return p, dp

    dQ = [[0.0] * d for _ in range(t)]
    for r0 in range(0, t, bq):
        rows = range(r0, min(r0 + bq, t))
        for k0 in range(0, t, bk):
            if masked and k0 > rows[-1]:
                continue
            js = range(k0, min(k0 + bk, t))
            if meter is not None:
                meter["products"] += 3 * len(rows) * len(js) * d            # S, dP and the product with K
            for i in rows:
                p, dp = tile_weights(i, js)
                for a, j in enumerate(js):
                    ds = p[a] * (dp[a] - D[i])
                    for c in range(d):
                        dQ[i][c] += scale * ds * K[j][c]
    dK = [[0.0] * d for _ in range(t)]
    dV = [[0.0] * d for _ in range(t)]
    for k0 in range(0, t, bk):
        js = range(k0, min(k0 + bk, t))
        for r0 in range(0, t, bq):
            rows = range(r0, min(r0 + bq, t))
            if masked and rows[-1] < k0:
                continue
            if meter is not None:
                meter["products"] += 4 * len(rows) * len(js) * d            # S, dP, and the products for dV and dK
            for i in rows:
                p, dp = tile_weights(i, js)
                for a, j in enumerate(js):
                    ds = p[a] * (dp[a] - D[i])
                    for c in range(d):
                        dV[j][c] += p[a] * dO[i][c]
                        dK[j][c] += scale * ds * Q[i][c]
    return dQ, dK, dV


def _check_inputs():
    g = bookrng.generator(NAME_CHECK)
    Q = _normals(g, CHECK_T, CHECK_D)
    K = _normals(g, CHECK_T, CHECK_D)
    V = _normals(g, CHECK_T, CHECK_D)
    dO = _normals(g, CHECK_T, CHECK_D)
    return Q, K, V, dO


def _relative(got, want, floor=1e-4):
    """Largest |a - n| / max(|a|, |n|) over entries whose larger magnitude is above `floor`."""
    worst = 0.0
    for r, w in zip(got, want):
        for a, b in zip(r, w):
            if max(abs(a), abs(b)) > floor:
                worst = max(worst, abs(a - b) / max(abs(a), abs(b)))
    return worst


def _central(Q, K, V, dO, causal, which):
    """Central differences of the loss sum(dO * O) with respect to every entry of Q, K or V, through the tiled forward."""
    mats = {"Q": Q, "K": K, "V": V}
    target = mats[which]
    out = [[0.0] * len(target[0]) for _ in target]
    for i in range(len(target)):
        for c in range(len(target[0])):
            old = target[i][c]
            vals = []
            for sign in (1, -1):
                target[i][c] = old + sign * EPS
                O = forward_tiled(Q, K, V, CHECK_BQ, CHECK_BK, causal)[0]
                vals.append(sum(_sum([dO[r][k] * O[r][k] for k in range(len(O[0]))]) for r in range(len(O))))
            target[i][c] = old
            out[i][c] = (vals[0] - vals[1]) / (2 * EPS)
    return out


def backward_check():
    Q, K, V, dO = _check_inputs()
    rows = []
    for causal in (False, True):
        O, lse, ms = forward_tiled(Q, K, V, CHECK_BQ, CHECK_BK, causal)
        ref = backward_reference(Q, K, V, dO, causal)
        got = backward_tiled(Q, K, V, O, lse, dO, CHECK_BQ, CHECK_BK, causal)
        P = attention_reference(Q, K, V, causal)[2]
        dP = sa.matmul(dO, sa.transpose(V))
        row_term = max(abs(_dot(dO[i], O[i]) - _dot(P[i], dP[i])) for i in range(CHECK_T))
        for name, a, b, which in (("dQ", got[0], ref[0], "Q"), ("dK", got[1], ref[1], "K"), ("dV", got[2], ref[2], "V")):
            fd = _central(Q, K, V, dO, causal, which)
            rows.append(("causal" if causal else "no mask", name, _bound(_max_abs(a, b)), _bound(_relative(a, fd)), _bound(row_term)))
    return (f"{('Generator: the book\'s SplitMix64, seed ' + _seed(NAME_CHECK) + ' (`seed_of` of the name `' + NAME_CHECK + '`).')} T = {CHECK_T}, "
            f"d_h = {CHECK_D}, query blocks of {CHECK_BQ} rows and key tiles of {CHECK_BK} rows. Draws, in this order: Q, K, V and the incoming "
            f"gradient dO (each {CHECK_T} by {CHECK_D}, row by row, `normal()`). The loss is the sum over all entries of dO times O, so the "
            f"gradient of the loss with respect to O is dO. 'Tiled backward' recomputes every weight tile by tile from the saved O and "
            f"log-sum-exp; 'materialised' is the backward of Chapter 7. Central differences have step {EPS:g}, are taken through the tiled "
            "forward on every entry of the tensor, and are compared as |a - n| / max(|a|, |n|) over the entries whose gradient is above "
            "1e-4. The last column is the largest |dO_i . O_i - sum_j P_ij dP_ij| over the rows. Every cell is the power of ten the "
            "difference stays below.\n\n"
            + _table(["mask", "tensor", "tiled vs materialised, absolute", "tiled vs central differences, relative",
                      "D_i identity, absolute"], rows, "llrrr"))


def wrong_backward_passes():
    Q, K, V, dO = _check_inputs()
    rows = []
    for causal in (False, True):
        O, lse, ms = forward_tiled(Q, K, V, CHECK_BQ, CHECK_BK, causal)
        ref = backward_reference(Q, K, V, dO, causal)
        variants = [("the chapter's backward pass", None)]
        variants += [("D_i left out (taken as 0)", "no_D"), ("the running maximum m_i saved in place of the log-sum-exp", "running_max")]
        if causal:
            variants.append(("the causal mask left out of the recomputed scores", "no_mask"))
        for label, mistake in variants:
            got = backward_tiled(Q, K, V, O, lse, dO, CHECK_BQ, CHECK_BK, causal, mistake=mistake, m=ms)
            cells = []
            for a, b in zip(got, ref):
                err = _relative(a, b)
                cells.append(_bound(err) if err < 1e-6 else f"about {err:.1f}")
            rows.append(("causal" if causal else "no mask", label, *cells))
    return (f"The check of the previous block (same inputs, same tiles) run on the correct tiled backward pass and on deliberately wrong "
            "ones. Cells are the largest relative difference |a - b| / max(|a|, |b|) of the tensor from the materialised backward pass over "
            "the entries whose gradient is above 1e-4; a difference of 1e-6 or more is written to one decimal, a smaller one as the "
            "power of ten it stays below. Dropping D_i leaves dV, which does not use it, correct. The mutation 'the causal mask left out' "
            "removes two things at once: the test of each recomputed score (a key after the query keeps its score) and the skip of the "
            "tiles that lie wholly after a block's last row (every tile is visited).\n\n"
            + _table(["mask", "backward pass", "dQ", "dK", "dV"], rows, "llrrr"))


def backward_flops(t=64, d=8, bq=16, bk=16):
    meter = {"products": 0}
    Q, K, V = (_normals(bookrng.generator("flash_attention.flops"), t, d) for _ in range(3))
    dO = _normals(bookrng.generator("flash_attention.flops_do"), t, d)
    O, lse, _ = forward_tiled(Q, K, V, bq, bk, False)
    backward_tiled(Q, K, V, O, lse, dO, bq, bk, False, meter=meter)
    return meter["products"], t, d


def backward_cost():
    t, d = 64, 8
    counted, _, _ = backward_flops(t, d)
    per = t * t * d                                       # multiply-adds of one product of the family, T^2 d
    unit = 2 * per                                        # FLOPs of one such product
    rows = [("materialised (Chapter 7): dP, dV, dQ, dK", 4, _int(4 * unit), "1.00"),
            ("S recomputed once, one pass: S, dP, dV, dK, dQ", 5, _int(5 * unit), "1.25"),
            ("two passes: (S, dP, dQ) and (S, dP, dV, dK)", 7, _int(7 * unit), "1.75")]
    assert 2 * counted == 7 * unit
    return (f"Products of the backward pass of one head, counted in units of one T x T x d_h product (2 T^2 d_h FLOPs, no mask). The last "
            f"row was counted by running the tiled backward of the previous blocks on a case with T = {t}, d_h = {d} and tiles of 16 and "
            f"counting the multiply-adds of every tile product: {counted:,} multiply-adds, which is 7 T^2 d_h = {7 * per:,}. The forward "
            "pass is 2 such products (S and P V), so a training step spends 2 + 4 = 6 on the materialised path and 2 + 5 = 7 or 2 + 7 = 9 "
            "when scores are recomputed.\n\n"
            + _table(["backward pass", "products", "FLOPs at T = 64, d_h = 8", "relative to materialised"], rows, "lrrr"))


# ------------------------------------------------------------------------------------------------ 7. one query, a long cache

NAME_DECODE = "flash_attention.decode"
DEC_D, DEC_BC, DEC_TILES_PER_SPLIT = 4, 8, 2
DEC_LENS = (1, 7, 8, 9, 41, 100)


def decode_inputs():
    g = bookrng.generator(NAME_DECODE)
    q = [g.normal() for _ in range(DEC_D)]
    K = _normals(g, max(DEC_LENS), DEC_D)
    V = _normals(g, max(DEC_LENS), DEC_D)
    return q, K, V


def decode_partials(q, K, V, length, n_splits, bc=DEC_BC, tiles_per_split=DEC_TILES_PER_SPLIT):
    """The partial states of the n_splits workgroups for the first `length` cache rows: split s walks the tiles
    s * tiles_per_split .. (s + 1) * tiles_per_split - 1, and one beyond the cache holds the empty state."""
    scale = 1.0 / math.sqrt(len(q))
    ntiles = _ceil_div(length, bc)
    out = []
    for s in range(n_splits):
        state = empty_state(len(V[0]))
        for kt in range(s * tiles_per_split, min((s + 1) * tiles_per_split, ntiles)):
            js = range(kt * bc, min(kt * bc + bc, length))
            state = update_state(state, [scale * _dot(q, K[j]) for j in js], [V[j] for j in js])
        out.append(state)
    return out


def combine(states, drop_correction=False):
    acc = states[0]
    for s in states[1:]:
        acc = merge_states(acc, s, drop_correction)
    m, l, o = acc
    return [v / l for v in o]


def combine_tree(states):
    level = list(states)
    while len(level) > 1:
        level = [merge_states(level[i], level[i + 1]) if i + 1 < len(level) else level[i] for i in range(0, len(level), 2)]
    m, l, o = level[0]
    return [v / l for v in o]


def decode_split():
    q, K, V = decode_inputs()
    n_splits = _ceil_div(_ceil_div(max(DEC_LENS), DEC_BC), DEC_TILES_PER_SPLIT)
    rows = []
    for length in DEC_LENS:
        scores = [_dot(q, K[j]) / math.sqrt(DEC_D) for j in range(length)]
        p = softmax_stable(scores)
        want = [_sum([p[j] * V[j][c] for j in range(length)]) for c in range(DEC_D)]
        states = decode_partials(q, K, V, length, n_splits)
        live = sum(1 for s in states if s[1] > 0.0)
        got = combine(states)
        rows.append((length, _ceil_div(length, DEC_BC), live, n_splits - live, _bound(_max_abs(got, want)),
                     _bound(_max_abs(combine_tree(states), want)), f"{_max_abs(combine(states, True), want):.3f}"))
    return (f"{('Generator: the book\'s SplitMix64, seed ' + _seed(NAME_DECODE) + ' (`seed_of` of the name `' + NAME_DECODE + '`).')} One query of "
            f"width d_h = {DEC_D} against a cache of up to {max(DEC_LENS)} rows (K and V each {max(DEC_LENS)} by {DEC_D}). Draws, in this order: the "
            f"query (`normal()` {DEC_D} times), then K and V row by row, `normal()`; a shorter cache is the first rows. Keys are read in tiles of "
            f"{DEC_BC}, each of the {n_splits} workgroups walks {DEC_TILES_PER_SPLIT} consecutive tiles and returns its state (m, l, o); "
            "the same number of workgroups is dispatched for every length, and one whose tiles lie beyond the cache returns the empty state "
            "(m = -infinity, l = 0, o = 0). The combine merges the states left to right (or as a balanced tree) and divides o by l. "
            "The last column is the combine with exp(m_i - m) left out of the sum l: its largest absolute error, to three decimals.\n\n"
            + _table(["cache length", "tiles of 8", "workgroups with keys", "empty workgroups", "combine differs from attention by less than",
                      "tree combine differs by less than", "combine that drops the factor, error"], rows))


def decode_traffic():
    rows = []
    d = 128
    for t in (4096, 32768):
        for s in (1, 8, 32):
            cache = F32 * 2 * t * d
            partial = F32 * 2 * s * (d + 2) if s > 1 else 0
            rows.append((_int(t), s, _int(cache), _int(partial), f"{partial / cache:.5f}", 16 * s))
    return (f"One decode step of one sequence: a cache of T rows of width d_h = {d} for K and for V in float32, split across S workgroups per "
            "head. Each split writes its state (m, l and the d_h numbers of o) and the combine reads it back, so the extra traffic is "
            "2 S (d_h + 2) numbers per head; the last column is the number of workgroups for a model with 16 heads and one sequence.\n\n"
            + _table(["T", "S", "cache bytes read per head", "partial states, bytes written and read", "partial / cache", "workgroups, 16 heads"], rows))


# ------------------------------------------------------------------------------------------------ 8. float32 rounding

NAME_F32 = "flash_attention.float32"
F32_T, F32_D = 48, 16


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _dot32(a, b):
    acc = 0.0
    for x, y in zip(a, b):
        acc = _f32(acc + _f32(x * y))
    return acc


def _exp32(x):
    return _f32(math.exp(x))


def attention_f32_materialised(Q, K, V, scale):
    t, d = len(Q), len(Q[0])
    O = []
    for i in range(t):
        s = [_f32(scale * _dot32(Q[i], K[j])) if j <= i else -INF for j in range(t)]
        m = max(s)
        e = [_exp32(_f32(v - m)) for v in s]
        z = 0.0
        for v in e:
            z = _f32(z + v)
        p = [_f32(v / z) for v in e]
        row = []
        for c in range(d):
            acc = 0.0
            for j in range(t):
                acc = _f32(acc + _f32(p[j] * V[j][c]))
            row.append(acc)
        O.append(row)
    return O


def attention_f32_tiled(Q, K, V, scale, bk):
    t, d = len(Q), len(Q[0])
    O = []
    for i in range(t):
        m, l, o = -INF, 0.0, [0.0] * d
        for k0 in range(0, i + 1, bk):
            js = range(k0, min(k0 + bk, i + 1))
            s = [_f32(scale * _dot32(Q[i], K[j])) for j in js]
            top = max(s)
            m_new = max(m, top)
            corr = _exp32(_f32(m - m_new)) if m != -INF else 0.0
            p = [_exp32(_f32(v - m_new)) for v in s]
            tile_l = 0.0
            for v in p:
                tile_l = _f32(tile_l + v)
            l = _f32(_f32(l * corr) + tile_l)
            for c in range(d):
                acc = 0.0
                for a, j in enumerate(js):
                    acc = _f32(acc + _f32(p[a] * V[j][c]))
                o[c] = _f32(_f32(o[c] * corr) + acc)
            m = m_new
        O.append([_f32(v / l) for v in o])
    return O


def float32_numerics():
    g = bookrng.generator(NAME_F32)
    t, d = F32_T, F32_D
    Q, K, V = [[[_f32(g.normal()) for _ in range(d)] for _ in range(t)] for _ in range(3)]
    scale = _f32(1.0 / math.sqrt(d))
    truth = attention_reference(Q, K, V, True, scale)[0]
    mat = attention_f32_materialised(Q, K, V, scale)
    n = t * d
    u = 2.0 ** -24                                        # the unit roundoff of float32 (Chapter 15 of Volume I)

    def units(x):
        return f"{math.ceil(x / u - 1e-9)}"

    rows = [("materialised", "-", units(_max_abs(mat, truth)), f"{n} of {n}")]
    for bk in (1, 2, 4, 8, 16, t):
        tiled = attention_f32_tiled(Q, K, V, scale, bk)
        same = sum(1 for a, b in zip(tiled, mat) for x, y in zip(a, b) if _bits(x) == _bits(y))
        rows.append((f"tiled, B_k = {bk}", units(_max_abs(tiled, mat)), units(_max_abs(tiled, truth)), f"{same} of {n}"))
    return (f"{('Generator: the book\'s SplitMix64, seed ' + _seed(NAME_F32) + ' (`seed_of` of the name `' + NAME_F32 + '`).')} Causal attention of one head, "
            f"T = {t}, d_h = {d}. Draws, in this order: Q, K, V (each {t} by {d}, row by row, `normal()` rounded to float32). Every product, "
            "sum, difference and quotient is rounded to float32 one operation at a time (no fused multiply-add: a product is rounded, then "
            "added), sums run left to right from 0.0 (a dot product over the head width, the normalizer, and the weighted sum over the keys), "
            "exp is the library exp of double precision rounded to float32, and the scale is the float32 nearest to 1 / sqrt(d). Row i reads "
            "the keys 0 to i. The materialised version computes s_j = scale * (q . k_j), the maximum m of the row, e_j = exp(s_j - m), the sum z "
            "of the e_j, the weights p_j = e_j / z, and each output channel as the sum of p_j v_j. The tiled version cuts the keys 0 to i into "
            "tiles of B_k consecutive keys (the last tile is shorter) and keeps (m, l, o). For each tile it computes the scores s_j, the new "
            "maximum m' = max(m, largest score of the tile), the correction corr = exp(m - m') (0 for the first tile), the weights "
            "p_j = exp(s_j - m'), the tile's own partial sum of the p_j (left to right from 0.0) and, per channel, the tile's own partial sum of "
            "the p_j v_j (left to right from 0.0), and updates l = l * corr + (the tile's sum of p) and o = o * corr + (the tile's sum of p v), "
            "each product and sum rounded to float32; the output is o / l at the end. This per-tile order of additions is not the key-by-key "
            "order of the kernel pseudocode above, which rounds differently; the table is for the order stated here. The reference "
            "in double is attention in double precision from the same float32 inputs. Differences are the largest absolute difference "
            "over the 768 outputs, in units of u = 2^-24 (the float32 unit roundoff) and rounded up to a whole number; the outputs are "
            "below 3 in magnitude. The last column counts outputs equal to the materialised float32 result in every bit; the count "
            "depends on the library exp, which is why the page names it.\n\n"
            + _table(["version", "difference from materialised float32, units of u", "difference from double precision, units of u",
                      "outputs bit-identical to materialised"], rows))


# ------------------------------------------------------------------------------------------------ 9. the shapes of brain's kernels

WG, BC, CH = 256, 8, 32
REGISTER_LIMIT = 255        # registers per invocation on the device the lane split was written for (the kernel's own header)


def kernel_geometry():
    rows = []
    for hd in (64, 128, 256):
        lanes = 4 if hd <= 128 else 8
        br = WG // lanes
        hdt = lanes * CH
        kv_bytes = 2 * BC * hdt * F32
        part_bytes = BC * WG * F32
        rows.append((hd, lanes, br, 2 * CH, f"{2 * hd}{' (over 255)' if 2 * hd > REGISTER_LIMIT else ''}", (kv_bytes + part_bytes) // 1024))
    return (f"The shape of the causal forward kernel of brain for the three head widths it serves: 256 invocations per workgroup, key tiles of "
            f"B_k = {BC} rows, {CH} channels per invocation. lanes = 4 up to head width 128 and 8 above; a workgroup covers 256 / lanes query "
            "rows. 'q and o per invocation' is 2 x 32 registers. 'If one invocation owned a row' is 2 d_h, to be compared with the 255 "
            "registers per invocation that the kernel's own header names for the device it was written for. Shared KiB is the key tile and the "
            f"value tile (2 x {BC} x lanes x 32 floats) plus the partial dot products ({BC} x 256 floats).\n\n"
            + _table(["head width", "lanes", "query rows per workgroup", "q and o per invocation", "if one invocation owned a row",
                      "shared KiB used"], rows))


def score_slabs():
    smallest = next(t for t in range(1, 100000) if 16 * t * t * 4 > 2 ** 31 - 1)
    cases = [(32, 7484), (16, smallest), (24, 16384), (24, 32768)]
    rows = [(h, _int(t), _int(h * t * t * F32), f"{_gib(h * t * t * F32):.2f}") for h, t in cases]
    return ("Score slabs in float32, H heads and T rows: 4 H T^2 bytes. The second row is the smallest T for which the slab of 16 heads "
            "exceeds 2^31 - 1 bytes, the largest storage-binding size the device of that history allowed.\n\n"
            + _table(["H", "T", "slab, bytes", "GiB"], rows))


def flash_buffers():
    t, hq, hkv, hd = 16384, 24, 4, 256
    q = t * hq * hd * F32
    kv = t * hkv * hd * F32
    lse = hq * t * F32
    fwd = q + 2 * kv + q + lse                               # q, k, v, ctx, lse
    bwd = q + q + 2 * kv + lse                               # d_ctx, d_q, d_k, d_v, dsum
    slab = hq * t * t * F32
    return (f"The attention buffers of one training layer of a model with {hq} query heads, {hkv} key-value heads and head width {hd} at "
            f"T = {t}, float32. Tiled path: q, k, v, ctx and the log-sum-exp for the forward, d_ctx, d_q, d_k, d_v and D_i for the backward. "
            "Materialised path: three slabs of 4 H T^2 bytes (scores, probabilities, score gradients).\n\n"
            + _table(["buffers", "bytes", "GiB"],
                     [("q and ctx (each)", _int(q), f"{_gib(q):.3f}"), ("k and v (each)", _int(kv), f"{_gib(kv):.3f}"),
                      ("log-sum-exp (and D_i)", _int(lse), f"{lse / 2 ** 20:.1f} MiB"), ("tiled forward", _int(fwd), f"{_gib(fwd):.2f}"),
                      ("tiled forward and backward", _int(fwd + bwd), f"{_gib(fwd + bwd):.2f}"), ("one slab", _int(slab), f"{_gib(slab):.1f}"),
                      ("three slabs", _int(3 * slab), f"{_gib(3 * slab):.1f}")], "lrr"))


# ------------------------------------------------------------------------------------------------ figures


def bytes_curve(d=D_HEAD):
    """(ts, chain GiB, {bq: tiled GiB}) for the figure of bytes against the sequence length."""
    ts = list(range(1024, 16385, 1024))
    chain = [_gib(F32 * words_materialised(t, d)) for t in ts]
    tiled = {bq: [_gib(F32 * words_tiled(t, d, bq)) for t in ts] for bq in (16, 64, 256)}
    return ts, chain, tiled
