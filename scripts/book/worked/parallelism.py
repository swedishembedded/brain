# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Parallel and Distributed Training".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: parallelism.<name> -->` markers. Helpers with a leading underscore, or that take arguments, are shared
with the chapter's figures so a picture and the table beside it cannot disagree.

Nothing here trains a chaotic run: every table is a few steps of deterministic arithmetic whose order is stated, so
the digits printed are the digits any implementation that follows the stated order reproduces. The block stack that
the chapter counts (width, hidden width, depth, tokens, weights) and the tiny executable stack come from
`training_memory` (imported, never copied).

Swedish Embedded AB implements solutions for spreading training over several devices for its clients. If your team
needs expertise in data, tensor and pipeline parallel training or in collective communication then you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math
import struct

import bookrng
import number_formats as nf
import training_memory as tm

_table = nf._table

D, F, L, N_TOK = tm.D, tm.F, tm.L, tm.N          # width, hidden width, blocks, tokens of one step
WEIGHTS = tm.P                                    # 2^32 trainable numbers of the stack
GIB = 2 ** 30
MIB = 2 ** 20


def _g(x, d=6):
    """d significant digits, written without an exponent when that is short."""
    return f"{float(f'{x:.{d}g}'):g}"


def _gib(b, d=2):
    return f"{b / GIB:.{d}f}"


# ------------------------------------------------------------------ data parallelism on a small regression

R_SHARD = 12                                       # rows of the batch
W_DIM = 3                                          # weights of the linear model


def _regression():
    """Rows X (12 x 3, row-major, each normal()), targets y (12) and the weight w (3, each normal() times 0.5), drawn
    in that order from one generator."""
    rng = bookrng.generator("parallelism.shards")
    X = [[rng.normal() for _ in range(W_DIM)] for _ in range(R_SHARD)]
    y = [rng.normal() for _ in range(R_SHARD)]
    w = [rng.normal() * 0.5 for _ in range(W_DIM)]
    return X, y, w


def _grad_sum(rows, X, y, w):
    """The sum over `rows` (in the order given) of (w . x_i - y_i) x_i, each component accumulated from 0.0."""
    g = [0.0] * W_DIM
    for i in rows:
        res = sum(wj * xj for wj, xj in zip(w, X[i])) - y[i]
        for j in range(W_DIM):
            g[j] += res * X[i][j]
    return g


def _mean_grad(rows, X, y, w):
    return [v / len(rows) for v in _grad_sum(rows, X, y, w)]


def _dev(a, b):
    return max(abs(p - q) for p, q in zip(a, b))


def shard_gradients():
    X, y, w = _regression()
    full = _mean_grad(list(range(R_SHARD)), X, y, w)
    rows = []
    for reps in (1, 2, 3, 4, 6, 12):
        per = R_SHARD // reps
        gs = [_mean_grad(list(range(r * per, (r + 1) * per)), X, y, w) for r in range(reps)]
        avg = [0.0] * W_DIM
        for g in gs:                                  # rank order, from 0.0
            avg = [a + b for a, b in zip(avg, g)]
        avg = [a / reps for a in avg]
        rows.append((f"{reps} replica{'s' if reps > 1 else ''}, {per} contiguous row{'s' if per > 1 else ''} each", "passes (below 1e-12)" if _dev(avg, full) < 1e-12 else "FAILS"))
    # three replicas, micro-batches of 2 rows dealt round-robin (replica r takes micro-batches r, r+3, ...), each
    # replica accumulating the sums of its micro-batches, the replicas' sums added in rank order, the total divided by 12
    reps, mb = 3, 2
    nmb = R_SHARD // mb
    sums = []
    for r in range(reps):
        acc = [0.0] * W_DIM
        for k in range(r, nmb, reps):
            part = _grad_sum(list(range(k * mb, (k + 1) * mb)), X, y, w)
            acc = [a + b for a, b in zip(acc, part)]
        sums.append(acc)
    tot = [0.0] * W_DIM
    for s in sums:
        tot = [a + b for a, b in zip(tot, s)]
    tot = [a / R_SHARD for a in tot]
    rows.append(("3 replicas, 6 micro-batches of 2 rows dealt round-robin", "passes (below 1e-12)" if _dev(tot, full) < 1e-12 else "FAILS"))
    head = (f"One generator seeded {bookrng.seed_of('parallelism.shards')} (seed_of 'parallelism.shards') draws, in this order and "
            f"each by normal(): the 12 rows x_i of 3 numbers (row-major), the 12 targets y_i and the 3 weights w (each times 0.5). "
            f"The loss of a batch is the mean over its rows of half the square of w . x_i - y_i, so its gradient is the mean of "
            f"(w . x_i - y_i) x_i; each component is accumulated in increasing row order from 0.0 and divided by the row count. "
            f"The full-batch gradient is ({', '.join(_g(v) for v in full)}). A replica's gradient is the same mean over its own "
            f"rows; the replicas' gradients are added in rank order from 0.0 and divided by their number. The deviation is the largest "
            f"absolute difference of a component from the full-batch gradient, and the criterion is that it is below 1e-12.\n\n")
    return head + _table(["layout", "average of the replicas' gradients against the full batch"], rows, "ll")


def weighted_shards():
    X, y, w = _regression()
    full = _mean_grad(list(range(R_SHARD)), X, y, w)
    cuts = [(0, 5), (5, 8), (8, 12)]
    gs = [_mean_grad(list(range(a, b)), X, y, w) for a, b in cuts]
    counts = [b - a for a, b in cuts]
    plain = [0.0] * W_DIM
    for g in gs:
        plain = [p + q for p, q in zip(plain, g)]
    plain = [p / len(gs) for p in plain]
    wtd = [0.0] * W_DIM
    for g, c in zip(gs, counts):
        wtd = [p + q * c for p, q in zip(wtd, g)]
    wtd = [p / sum(counts) for p in wtd]
    rows = [(f"replica {r}: rows {a} to {b - 1}", c, ", ".join(_g(v) for v in g)) for r, ((a, b), c, g) in enumerate(zip(cuts, counts, gs))]
    rows.append(("full batch, 12 rows", 12, ", ".join(_g(v) for v in full)))
    rows.append(("plain average of the three", "-", ", ".join(_g(v) for v in plain)))
    rows.append(("average weighted by row count", "-", ", ".join(_g(v) for v in wtd)))
    dist = math.sqrt(sum((p - q) ** 2 for p, q in zip(plain, full)))
    head = ("The data, weights and gradient rule of the previous table, with the 12 rows cut into three shards of 5, 3 and 4 rows "
            "(rows 0 to 4, 5 to 7, 8 to 11). The plain average adds the three shard gradients and divides by 3; the weighted one "
            "adds each shard gradient times its row count and divides by 12; both add in replica order from 0.0.\n\n")
    tail = (f"\n\nThe plain average is {_g(dist, 4)} away from the full-batch gradient (Euclidean distance); the weighted one "
            f"{'passes' if _dev(wtd, full) < 1e-12 else 'FAILS'} the criterion of the previous table (below 1e-12).")
    return head + _table(["gradient", "rows", "components (6 significant digits)"], rows, "lrl") + tail


# ------------------------------------------------------------------ the operations, on whole numbers

RANKS, ELEMS = 4, 4


def _vec(j):
    return [10 * j + i for i in range(ELEMS)]


def _fmt(v):
    return "[" + ", ".join(str(x) for x in v) + "]"


def collective_ops():
    vs = [_vec(j) for j in range(RANKS)]
    total = [sum(v[i] for v in vs) for i in range(ELEMS)]
    gathered = [x for v in vs for x in v]
    inputs = _table(["rank j", "input of rank j"], [(j, _fmt(v)) for j, v in enumerate(vs)], "rl")
    rows = [("broadcast, root 2", "every rank: " + _fmt(vs[2])),
            ("all-gather", "every rank: " + _fmt(gathered)),
            ("all-reduce (sum)", "every rank: " + _fmt(total)),
            ("reduce-scatter (sum)", "; ".join(f"rank {j}: {total[j]}" for j in range(RANKS)))]
    head = (f"{RANKS} ranks, each holding 4 whole numbers: element i of rank j is 10 j + i.\n\n")
    tail = ("\n\nAll-gather lists the inputs in rank order; the sums are over the four ranks; reduce-scatter hands rank j "
            "element j of the sum.\n\n")
    return head + inputs + tail + _table(["operation", "result"], rows, "ll")


def _ring(vs):
    """Ring all-reduce on lists of ints. Rank j sends chunk (j - s) mod R in reduce-scatter step s and chunk
    (j + 1 - s) mod R in all-gather step s, always to rank (j + 1) mod R. Returns the final buffers, the chunks sent
    per step (reduce-scatter steps then all-gather steps) and the elements each rank sent."""
    R = len(vs)
    n = len(vs[0])
    c = n // R
    buf = [list(v) for v in vs]
    plan, sent = [], 0
    for s in range(R - 1):                           # reduce-scatter: receiver adds to its own copy
        msgs = []
        for j in range(R):
            ch = (j - s) % R
            msgs.append((j, (j + 1) % R, ch, buf[j][ch * c:(ch + 1) * c]))
        for j, to, ch, data in msgs:
            for i, x in enumerate(data):
                buf[to][ch * c + i] += x
        plan.append(("reduce-scatter", s, [m[2] for m in msgs]))
        sent += c
    for s in range(R - 1):                           # all-gather: receiver overwrites
        msgs = []
        for j in range(R):
            ch = (j + 1 - s) % R
            msgs.append((j, (j + 1) % R, ch, buf[j][ch * c:(ch + 1) * c]))
        for j, to, ch, data in msgs:
            buf[to][ch * c:(ch + 1) * c] = data
        plan.append(("all-gather", s, [m[2] for m in msgs]))
        sent += c
    return buf, plan, sent


def ring_steps():
    vs = [_vec(j) for j in range(RANKS)]
    total = [sum(v[i] for v in vs) for i in range(ELEMS)]
    buf, plan, sent = _ring(vs)
    rows = [(ph, s, *[f"chunk {ch}" for ch in chs]) for ph, s, chs in plan]
    ok = all(b == total for b in buf)
    head = ("The inputs of the previous table (4 ranks, 4 numbers, so chunk c is element c). In "
            "every step, rank j sends the chunk named in its column to rank (j + 1) mod 4. In a reduce-scatter step the "
            "receiver adds it to its own copy of that chunk; in an all-gather step it overwrites its copy.\n\n")
    tail = (f"\n\nAfter the six steps every rank holds {_fmt(total) if ok else 'WRONG'}, the all-reduce of the previous table "
            f"({'passes' if ok else 'FAILS'}); each rank sent {sent} numbers, 2 (R - 1) / R of the 4, and in the sum of "
            f"chunk c the numbers are added in the order ((v_c + v_(c+1)) + v_(c+2)) + v_(c+3), indices mod 4.")
    return head + _table(["phase", "step s", "rank 0 sends", "rank 1 sends", "rank 2 sends", "rank 3 sends"], rows, "llllll") + tail


# ------------------------------------------------------------------ the order of the additions

def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _ulp32(x):
    """The spacing of the float32 numbers at x != 0."""
    _, e = math.frexp(abs(x))
    return 2.0 ** (e - 24)


def _order_row(R, n=840):
    rng = bookrng.generator("parallelism.order")
    v = [[_f32(rng.normal()) for _ in range(n)] for _ in range(R)]
    c = n // R
    diff, worst = 0, 0.0
    for i in range(n):
        a = 0.0
        for j in range(R):                           # the coordinator's order: rank 0, 1, ..., from 0.0
            a = _f32(a + v[j][i])
        ch = i // c
        b = v[ch][i]
        for t in range(1, R):                        # the ring's order for this chunk: ranks ch, ch + 1, ...
            b = _f32(b + v[(ch + t) % R][i])
        if a != b:
            diff += 1
            worst = max(worst, abs(a - b) / _ulp32(a))
    return R, n, diff, worst


def summation_order():
    rows = []
    for R in (2, 3, 4, 8):
        r, n, diff, worst = _order_row(R)
        rows.append((r, n, diff, f"{100 * diff / n:.1f}", f"{worst:.0f}"))
    head = (f"{bookrng.seed_of('parallelism.order')} is seed_of 'parallelism.order'; each row starts this generator afresh and draws "
            "R x 840 values by normal(), rank 0's 840 first, then rank 1's, and so on; every draw is rounded to the nearest float32, "
            "and every addition is done in float64 and rounded to the nearest float32 (ties to even). Element i is summed in "
            "the coordinator's order (0.0, then rank 0, rank 1, ...) and in the ring's order for its chunk c = floor(i / (840 / R)) "
            "(rank c, then c + 1, ..., indices mod R). A difference is counted when the two float32 results are not the same number; "
            "its size is measured in units of the spacing of the float32 numbers at the coordinator's result.\n\n")
    return head + _table(["ranks R", "elements", "elements that differ", "percent", "largest difference (spacings)"], rows)


# ------------------------------------------------------------------ bytes on the wire

def _star_ring(R, n_elems=WEIGHTS, width=4):
    full = n_elems * width
    star_busy = (R - 1) * full                       # the coordinator receives this and sends this
    ring = 2 * (R - 1) * full // R                   # every rank sends this and receives this
    return full, star_busy, ring


def wire_bytes():
    rows = []
    for R in (2, 4, 8, 16):
        full, busy, ring = _star_ring(R)
        rows.append((R, _gib(busy), _gib(full), _gib(ring), f"{busy / ring:.2f}", _gib(full // R)))
    head = ("The gradient of the 4,294,967,296-weight stack of Chapter 3 in fp32 (4 bytes a number, 16.00 GiB, 1 GiB = 2^30 bytes). "
            "Bytes are payload bytes only (framing and the one tag byte of a frame are not counted) and a rank's sent and received "
            "bytes are counted separately. In the coordinator star every worker sends its whole gradient to rank 0, which sends the "
            "whole sum back to each worker; in the ring every rank sends 2 (R - 1) chunks of 1/R of the gradient. The busiest rank of "
            "either topology handles 2 (R - 1) messages.\n\n")
    return head + _table(["ranks R", "star: rank 0 sends and receives (GiB)", "star: a worker sends and receives (GiB)",
                          "ring: every rank sends and receives (GiB)", "busiest rank, star over ring",
                          "ring: one message (GiB)"], rows)


# ------------------------------------------------------------------ the sharded optimizer

def sharded_state():
    rows = []
    for R in (1, 2, 4, 8, 16):
        rep = 16 * WEIGHTS
        shd = 8 * WEIGHTS + 8 * WEIGHTS // R
        grad = 4 * WEIGHTS
        comm_rep = 2 * (R - 1) * grad // R
        comm_shd = (R - 1) * grad // R + (R - 1) * grad // R
        rows.append((R, _gib(rep), _gib(shd), _gib(comm_rep), _gib(comm_shd)))
    head = ("The 4,294,967,296-weight stack of Chapter 3 trained with Adam in fp32, 1 GiB = 2^30 bytes. A rank always holds the "
            "weights (4 bytes a number) and the gradient (4 bytes); the replicated optimizer also holds the two moments (8 bytes) "
            "and the sharded one holds 1/R of them. Sent bytes are payload bytes per rank per step on a ring: the replicated "
            "optimizer all-reduces the gradient; the sharded one reduce-scatters the gradient and all-gathers the updated "
            "weights, both in fp32.\n\n")
    return head + _table(["ranks R", "replicated: state per rank (GiB)", "sharded: state per rank (GiB)",
                          "replicated: bytes sent per rank (GiB)", "sharded: bytes sent per rank (GiB)"], rows)


SH_R, SH_N, SH_STEPS = 4, 8, 5


def _adam(w, g, m, v, t, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    for i in range(len(w)):
        m[i] = b1 * m[i] + (1 - b1) * g[i]
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
        w[i] -= lr * (m[i] / (1 - b1 ** t)) / (math.sqrt(v[i] / (1 - b2 ** t)) + eps)


def _sharded_run():
    rng = bookrng.generator("parallelism.sharded")
    w0 = [rng.normal() * 0.5 for _ in range(SH_N)]
    grads = [[[rng.normal() for _ in range(SH_N)] for _ in range(SH_R)] for _ in range(SH_STEPS)]
    # replicated: every rank all-reduces (rank order, from 0.0), divides by R and updates all 8 weights with its own moments
    reps = [(list(w0), [0.0] * SH_N, [0.0] * SH_N) for _ in range(SH_R)]
    for t in range(1, SH_STEPS + 1):
        total = [0.0] * SH_N
        for r in range(SH_R):
            total = [a + b for a, b in zip(total, grads[t - 1][r])]
        mean = [a / SH_R for a in total]
        for w, m, v in reps:
            _adam(w, mean, m, v, t)
    # sharded: reduce-scatter hands rank r elements 2r, 2r+1 of the sum; it updates those with moments it alone holds;
    # the updated slices are all-gathered into every rank's weights
    c = SH_N // SH_R
    w = list(w0)
    ms = [[0.0] * c for _ in range(SH_R)]
    vs = [[0.0] * c for _ in range(SH_R)]
    for t in range(1, SH_STEPS + 1):
        total = [0.0] * SH_N
        for r in range(SH_R):
            total = [a + b for a, b in zip(total, grads[t - 1][r])]
        new = []
        for r in range(SH_R):
            sl = [w[r * c + i] for i in range(c)]
            g = [total[r * c + i] / SH_R for i in range(c)]
            _adam(sl, g, ms[r], vs[r], t)
            new += sl
        w = new
    return w0, reps, w


def sharded_equivalence():
    w0, reps, w = _sharded_run()
    same_ranks = all(reps[r][0] == reps[0][0] for r in range(SH_R))
    differ = sum(1 for a, b in zip(reps[0][0], w) if a != b)
    rows = [(i, _g(w0[i]), _g(reps[0][0][i]), _g(w[i])) for i in range(SH_N)]
    head = (f"{bookrng.seed_of('parallelism.sharded')} is seed_of 'parallelism.sharded'. One generator draws, by normal(), the 8 starting "
            "weights (each times 0.5) and then, for each of the 5 steps in turn, the gradients of the 4 ranks (rank 0's 8 numbers first), "
            "so the gradients are given and do not depend on the weights. Each step sums the four gradients in rank order from 0.0 and "
            "divides by 4. Adam is the adam_update of Chapter 4 (rate 0.01, beta1 0.9, beta2 0.999, epsilon 1e-8, bias-corrected, no weight decay, "
            "steps counted from 1), in float64, with 1 - beta1 and 1 - beta2 evaluated in float64, the second moment updated as ((1 - beta2) g) g, and the powers "
            "beta ^ t formed by exponentiation, not by repeated multiplication. The replicated run gives every rank all 8 weights and 16 moments; the sharded run gives rank "
            "r the elements 2 r and 2 r + 1 and their 4 moments, and all-gathers the updated pairs.\n\n")
    tail = (f"\n\nThe four replicated ranks end with identical weights ({'yes' if same_ranks else 'NO'}). The sharded run ends with "
            f"{differ} of the 8 weights different from them (bit patterns compared). A rank holds {8 + 16} numbers of state in the "
            f"replicated run and {8 + 4} in the sharded one.")
    return head + _table(["weight i", "start", "after 5 steps, replicated", "after 5 steps, sharded"], rows) + tail


# ------------------------------------------------------------------ tensor parallelism

TP_N, TP_D, TP_F = 4, 6, 8                        # rows, width and hidden width of the small block


def _tp_block():
    rng = bookrng.generator("parallelism.tensor")
    Ws, X = tm._draw_stack(rng, 1, TP_N, TP_D, TP_F)
    return Ws[0][0], Ws[0][1], X


def _add(P_, Q_):
    return [[a + b for a, b in zip(r, q)] for r, q in zip(P_, Q_)]


def _max_abs(P_, Q_):
    return max(abs(a - b) for r, q in zip(P_, Q_) for a, b in zip(r, q))


def _tp_run(K, wrong=False):
    """Forward and backward of the block Y = X + tanh(X W1^T) W2^T with the hidden width cut over K ranks. Returns Y, dX,
    dW1 and dW2 (the shards of the weight gradients put back in place) for the loss 0.5 * sum(Y^2)."""
    W1, W2, X = _tp_block()
    fk = TP_F // K
    A_s, partials = [], []
    for j in range(K):
        W1j = W1[j * fk:(j + 1) * fk]
        W2j = [row[j * fk:(j + 1) * fk] for row in W2]
        Aj = [[math.tanh(z) for z in r] for r in tm._mm(X, tm._t(W1j))]
        part = tm._mm(Aj, tm._t(W2j))
        if wrong:
            part = _add(part, X)                    # the input added inside every rank's partial
        A_s.append((W1j, W2j, Aj))
        partials.append(part)
    S = [[0.0] * TP_D for _ in range(TP_N)]
    for p in partials:                               # the all-reduce: rank order, from zeros
        S = _add(S, p)
    Y = S if wrong else _add(X, S)
    dY = Y
    dXs, dW1, dW2 = [], [], [[] for _ in range(TP_D)]
    for W1j, W2j, Aj in A_s:
        dA = tm._mm(dY, W2j)
        dZ = [[da * (1.0 - a * a) for da, a in zip(rda, ra)] for rda, ra in zip(dA, Aj)]
        dW1 += tm._mm(tm._t(dZ), X)
        for i, row in enumerate(tm._mm(tm._t(dY), Aj)):
            dW2[i] += row
        dXs.append(tm._mm(dZ, W1j))
    S = [[0.0] * TP_D for _ in range(TP_N)]
    for p in dXs:
        S = _add(S, p)
    dX = S if wrong else _add(dY, S)
    return Y, dX, dW1, dW2


def tensor_split():
    W1, W2, X = _tp_block()
    A = [[math.tanh(z) for z in r] for r in tm._mm(X, tm._t(W1))]
    _, Yd = tm._forward_block(X, W1, W2)
    dXd, dW1d, dW2d = tm._backward_block(Yd, W1, W2, X, A)
    rows = []
    for K in (1, 2, 4):
        Y, dX, dW1, dW2 = _tp_run(K)
        ok = max(_max_abs(Y, Yd), _max_abs(dX, dXd), _max_abs(dW1, dW1d), _max_abs(dW2, dW2d)) < 1e-12
        Yw, dXw, _, _ = _tp_run(K, wrong=True)
        rows.append((K, 2 * TP_N * TP_D if K > 1 else 0, "passes (below 1e-12)" if ok else "FAILS",
                     "-" if K == 1 else _g(_max_abs(Yw, Yd), 4)))
    head = (f"{bookrng.seed_of('parallelism.tensor')} is seed_of 'parallelism.tensor'. One generator draws, by the draw order of Chapter 3 "
            f"(W1 [{TP_F}, {TP_D}] row-major, W2 [{TP_D}, {TP_F}] row-major, each normal() times 0.5, then X [{TP_N}, {TP_D}] row-major, each normal()), "
            "one block Y = X + tanh(X W1^T) W2^T with the loss L = 0.5 sum(Y^2), so dY = Y. Rank j of K takes the rows "
            f"j {TP_F}/K to (j + 1) {TP_F}/K - 1 of W1 and the same columns of W2; the K partial products A_j W2_j^T are added in rank order "
            "from zeros (the all-reduce) and X is added once to the sum. In the backward, rank j forms dA_j = dY W2_j, dZ_j = dA_j times (1 - A_j^2), "
            "the weight gradients of its own shards and the partial dZ_j W1_j; the partials are added in rank order and dY is added once. "
            "The check compares Y, dX, dW1 and dW2 (shards put back in place) with the unsplit block of Chapter 3 and passes when every "
            "entry differs by less than 1e-12. The last column adds X inside every rank's partial instead, and gives the largest error of an entry of Y.\n\n")
    return head + _table(["ranks K", "numbers all-reduced per step (forward plus backward)", "against the unsplit block",
                          "X added inside each partial: largest error of an entry of Y"], rows, "rrlr")


def tensor_costs():
    rows = []
    for K in (1, 2, 4, 8):
        flops = 12 * N_TOK * D * F // K
        elems = 2 * N_TOK * D
        sent = 0 if K == 1 else 2 * 2 * (K - 1) * N_TOK * D * 4 // K
        rows.append((K, f"{flops / 1e9:,.2f}", f"{elems:,}" if K > 1 else "0", "-" if K == 1 else f"{sent / MIB:,.0f}",
                     "-" if K == 1 else f"{flops / sent:,.0f}"))
    head = (f"One block of the stack of Chapter 3 (D = {D}, F = {F}, {N_TOK} tokens in the step). A multiply-add counts as 2 FLOPs and only the "
            "matrix products are counted: the forward has 2, each 2 n D F FLOPs, and the backward has 4 of the same size, so 12 n D F in all; "
            "the tanh and the additions are not counted. Each of the forward and the backward all-reduces one "
            "[n, D] fp32 array (n D numbers), and on a ring a rank sends 2 (K - 1) / K of it. 1 MiB = 2^20 bytes.\n\n")
    return head + _table(["ranks K", "matrix FLOPs per rank per block (10^9)", "numbers all-reduced per block",
                          "bytes sent per rank per block (MiB)", "FLOPs per byte sent"], rows)


# ------------------------------------------------------------------ pipeline parallelism

F_UNIT, R_UNIT, B_UNIT = 1, 1, 2                  # stage forward, recompute of a stage's forward, stage backward


def _simulate(p, m, f=F_UNIT, r=R_UNIT, b=B_UNIT):
    """The schedule of one step with p stages and m micro-batches: every stage runs the forwards of micro-batches
    0 .. m - 1 (the forward of micro-batch k waits for the previous stage's), then, for micro-batches m - 1 down to 0,
    recomputes its forward and then waits for the next stage's gradient of that micro-batch (the last stage waits for
    nothing) before its backward. Returns tasks[s] = list of (kind, micro-batch, start, end) and the makespan."""
    t = [0] * p
    tasks = [[] for _ in range(p)]
    done = [[0] * m for _ in range(p)]
    for s in range(p):
        for k in range(m):
            start = t[s] if s == 0 else max(t[s], done[s - 1][k])
            t[s] = start + f
            done[s][k] = t[s]
            tasks[s].append(("F", k, start, t[s]))
    back = [[0] * m for _ in range(p)]
    for s in range(p - 1, -1, -1):
        for k in range(m - 1, -1, -1):
            start = t[s]
            tasks[s].append(("R", k, start, start + r))
            ready = start + r if s == p - 1 else max(start + r, back[s + 1][k])
            t[s] = ready + b
            back[s][k] = t[s]
            tasks[s].append(("B", k, ready, t[s]))
    return tasks, max(t)


def pipeline_schedule():
    rows = []
    p = 4
    busy_unit = F_UNIT + R_UNIT + B_UNIT
    for m in (1, 2, 4, 8, 16, 32):
        _, span = _simulate(p, m)
        assert span == busy_unit * m + (R_UNIT + B_UNIT) * (p - 1)
        rows.append((m, span, busy_unit * m, f"{100 * busy_unit * m / span:.1f}", f"{100 / p:.1f}"))
    rows2 = []
    for p_, m_ in ((2, 8), (4, 8), (8, 8), (8, 32)):
        _, span = _simulate(p_, m_)
        rows2.append((p_, m_, span, f"{100 * busy_unit * m_ / span:.1f}"))
    head = ("Stages of equal size, and one unit is the forward work of one stage on one micro-batch. A stage's backward is 2 units, and "
            "because the stage rebuilds its activations before the backward (as in Chapter 3) the rebuild is 1 more unit, so one stage does "
            "4 units of work on each micro-batch. The sends take no units. Each stage runs the forwards of the micro-batches 0 to m - 1 in "
            "order, waiting for the previous stage's output of the same micro-batch, and then, from micro-batch m - 1 down to 0, the rebuild "
            "(which waits for nothing) and the backward (which waits for the next stage's gradient of that micro-batch; the last stage waits "
            "for nothing). The step ends when the last stage has finished.\n\n")
    t1 = _table(["micro-batches m", "length of the step (units)", "work of one stage (units)", "share of the length a stage is busy (percent)",
                 "share if micro-batches go through one at a time (percent)"], rows)
    t2 = _table(["stages p", "micro-batches m", "length of the step (units)", "share busy (percent)"], rows2)
    return head + "Four stages:\n\n" + t1 + "\n\nOther depths:\n\n" + t2


BC_LAYERS, BC_STAGES, BC_EMBED, BC_HEAD = 32, 4, 3, 5     # layers, stages, extra cost of the first and the last stage


def _compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for i in range(1, n - k + 2):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


def _stage_costs(c):
    costs = [float(x) for x in c]
    costs[0] += BC_EMBED
    costs[-1] += BC_HEAD
    return costs


def balanced_cuts():
    best, winners = None, []
    for c in _compositions(BC_LAYERS, BC_STAGES):
        m = max(_stage_costs(c))
        if best is None or m < best:
            best, winners = m, [c]
        elif m == best:
            winners.append(c)
    assert len(winners) == 1
    even = tuple(BC_LAYERS // BC_STAGES for _ in range(BC_STAGES))
    rows = []
    for name, c in (("equal layers per stage", even), ("cuts that minimize the largest stage cost", winners[0])):
        costs = _stage_costs(c)
        rows.append((name, ", ".join(str(x) for x in c), ", ".join(_g(x) for x in costs), _g(max(costs))))
    head = (f"{BC_LAYERS} layers of cost 1 each are cut into {BC_STAGES} stages of consecutive layers, every stage with at least one layer; "
            f"the first stage also carries the embedding, of cost {BC_EMBED}, and the last the output head and loss, of cost {BC_HEAD} "
            "(costs in units of one layer). Every way of cutting is tried, and the one whose largest stage cost is smallest is kept "
            f"({len(winners)} way reaches that value).\n\n")
    return head + _table(["cut", "layers in stages 0 to 3", "cost of stages 0 to 3", "largest stage cost"], rows, "llll")


def schedule_cells(p, m):
    """For the figure: the forward phase only, as (stage, micro-batch, start) cells."""
    tasks, _ = _simulate(p, m)
    return [(s, k, a) for s in range(p) for (kind, k, a, _) in tasks[s] if kind == "F"]


PA_BLOCKS, PA_ROWS, PA_D, PA_F, PA_MB = 4, 8, 4, 6, 4


def _bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _flat(g):
    return [v for (dW1, dW2) in g for M in (dW1, dW2) for r in M for v in r]


def _fwd_bwd(Ws, X):
    """Forward through blocks Ws on rows X keeping each block's input; backward with dY = Y of the last block
    (the loss 0.5 * sum(Y^2)). Returns the gradients, block by block."""
    xs, Y = [], X
    for W1, W2 in Ws:
        xs.append(Y)
        _, Y = tm._forward_block(Y, W1, W2)
    dY, grads = Y, [None] * len(Ws)
    for l in range(len(Ws) - 1, -1, -1):
        A, _ = tm._forward_block(xs[l], Ws[l][0], Ws[l][1])
        dY, dW1, dW2 = tm._backward_block(dY, Ws[l][0], Ws[l][1], xs[l], A)
        grads[l] = (dW1, dW2)
    return grads


def _accumulate(acc, g):
    if acc is None:
        zero = lambda M: [[0.0] * len(r) for r in M]  # noqa: E731
        acc = [(zero(a), zero(b)) for (a, b) in g]
    return [(_add(a, c), _add(b, d)) for (a, b), (c, d) in zip(acc, g)]


def _stage_backward(Ws, x_in, dY):
    """One stage: rebuild each block's activations from the stashed stage input, then run the backward from dY."""
    xs, Y = [], x_in
    for W1, W2 in Ws:
        xs.append(Y)
        _, Y = tm._forward_block(Y, W1, W2)
    grads = [None] * len(Ws)
    for l in range(len(Ws) - 1, -1, -1):
        A, _ = tm._forward_block(xs[l], Ws[l][0], Ws[l][1])
        dY, dW1, dW2 = tm._backward_block(dY, Ws[l][0], Ws[l][1], xs[l], A)
        grads[l] = (dW1, dW2)
    return dY, grads


def pipeline_accumulation():
    rng = bookrng.generator("parallelism.pipeline")
    Ws, X = tm._draw_stack(rng, PA_BLOCKS, PA_ROWS, PA_D, PA_F)
    mb = PA_ROWS // PA_MB
    mbs = [X[k * mb:(k + 1) * mb] for k in range(PA_MB)]
    whole = _flat(_fwd_bwd(Ws, X))
    seq = None
    for xb in mbs:                                     # grad accumulation, micro-batches 0, 1, 2, 3
        seq = _accumulate(seq, _fwd_bwd(Ws, xb))
    cut = PA_BLOCKS // 2

    def pipeline(order):
        s0 = s1 = None
        stash = {}
        for k in range(PA_MB):                         # forward phase: stage 0, then stage 1 on its output
            h = mbs[k]
            Y = h
            for W1, W2 in Ws[:cut]:
                _, Y = tm._forward_block(Y, W1, W2)
            stash[k] = (h, Y)
        for k in order:                                # backward phase
            h0, h1 = stash[k]
            Y = h1
            for W1, W2 in Ws[cut:]:
                _, Y = tm._forward_block(Y, W1, W2)
            d1, g1 = _stage_backward(Ws[cut:], h1, Y)  # the loss's gradient is the last output
            d0, g0 = _stage_backward(Ws[:cut], h0, d1)  # the stage-1 input gradient crosses the cut
            s0 = _accumulate(s0, g0)
            s1 = _accumulate(s1, g1)
        return s0 + s1

    rev = _flat(pipeline([3, 2, 1, 0]))
    fwd = _flat(pipeline([0, 1, 2, 3]))
    sq = _flat(seq)
    n = len(sq)

    def row(name, a, ref):
        d = sum(1 for p, q in zip(a, ref) if _bits(p) != _bits(q))
        ok = max(abs(p - q) for p, q in zip(a, whole)) < 1e-12
        return (name, n, d, "passes (below 1e-12)" if ok else "FAILS")

    rows = [row("accumulation, micro-batches 0, 1, 2, 3 (no cut)", sq, sq),
            row("pipeline of 2 stages, backward over micro-batches 3, 2, 1, 0", rev, sq),
            row("pipeline of 2 stages, backward over micro-batches 0, 1, 2, 3", fwd, sq)]
    head = (f"{bookrng.seed_of('parallelism.pipeline')} is seed_of 'parallelism.pipeline'. One generator draws the 4 blocks and the input by the "
            f"draw order of Chapter 3 ({PA_BLOCKS} blocks with W1 [{PA_F}, {PA_D}] and W2 [{PA_D}, {PA_F}], each normal() times 0.5, then X [{PA_ROWS}, {PA_D}], each normal()). "
            "The loss is 0.5 sum(Y^2) over all 8 rows of the last block's output, so micro-batch gradients add with no scale. The 8 rows are cut "
            "into 4 micro-batches of 2 consecutive rows. The pipeline has 2 stages (blocks 0 and 1, then blocks 2 and 3); each stage keeps only the "
            "input of each micro-batch and rebuilds its activations before its backward; the gradient of the cut's activations crosses the cut. "
            "The gradient of one micro-batch is the matrix products of the backward (each entry a sum over the 2 rows, or over the 6 hidden units, "
            "taken with compensated (Neumaier) summation in float64, as Python's sum does from version 3.12) and tanh is the C library's double tanh; "
            "the buffers receive whole micro-batch gradients, entry by entry. "
            "Gradients are added into zeroed buffers in the order stated. The criterion compares every entry with the gradient of one backward over "
            "the whole batch (below 1e-12); the third column counts the entries (out of 4 blocks x 2 matrices x 24 = 192) whose 64-bit patterns differ "
            "from the first row (the zero of the third row and the pass of every row do not depend on the summation or the tanh; the count of the second row does, "
            "and is 68 with plain left-to-right summation).\n\n")
    return head + _table(["run", "gradient entries", "bit patterns differing from the first row", "against one backward over the whole batch"], rows, "lrrl")


# ------------------------------------------------------------------ the grid

GRID = (4, 2, 2)                                   # tensor, pipeline, data degrees; 16 ranks


def _rank(tp, pp, dp, g=GRID):
    return (dp * g[1] + pp) * g[0] + tp


def _coord(r, g=GRID):
    return (r % g[0], (r // g[0]) % g[1], r // (g[0] * g[1]))


def _group(r, axis, g=GRID):
    c = list(_coord(r, g))
    out = []
    for i in range(g[axis]):
        c2 = list(c)
        c2[axis] = i
        out.append(_rank(*c2, g))
    return out


def grid_layout():
    ok = all(_rank(*_coord(r)) == r for r in range(16))
    rows = []
    for r in (0, 1, 5, 9, 15):
        tp, pp, dp = _coord(r)
        rows.append((r, f"({tp}, {pp}, {dp})", _fmt(_group(r, 0)), _fmt(_group(r, 1)), _fmt(_group(r, 2))))
    head = ("A grid of 4 x 2 x 2 = 16 ranks, with rank = (d x 2 + p) x 4 + t for the coordinate (t, p, d) of tensor, pipeline and data position. "
            "A rank's tensor group is the ranks that differ only in t, its pipeline group those that differ only in p, its data group those that "
            "differ only in d, each listed in increasing order of the coordinate that varies.\n\n")
    tail = f"\n\nThe map from coordinates to ranks and back returns the rank it started from for all 16 ranks ({'passes' if ok else 'FAILS'})."
    return head + _table(["rank", "(t, p, d)", "tensor group", "pipeline group", "data group"], rows, "lllll") + tail


GRIDS = ((1, 1, 16), (4, 1, 4), (1, 4, 4), (2, 2, 4), (4, 4, 1), (2, 4, 2))


def _grid_cost(tp, pp, dp):
    state = 16 * WEIGHTS // (tp * pp)
    resid = N_TOK * D * 4
    tp_b = 0 if tp == 1 else (L // pp) * 2 * (2 * (tp - 1) * resid // tp)
    pp_b = 0 if pp == 1 else (resid if pp == 2 else 2 * resid)
    grad = 4 * WEIGHTS // (tp * pp)
    dp_b = 0 if dp == 1 else 2 * (dp - 1) * grad // dp
    return state, tp_b, pp_b, dp_b


def grid_costs():
    rows = []
    for tp, pp, dp in GRIDS:
        state, tp_b, pp_b, dp_b = _grid_cost(tp, pp, dp)
        rows.append((f"{tp} x {pp} x {dp}", _gib(state, 3), _gib(tp_b, 3), _gib(pp_b, 3), _gib(dp_b, 3), _gib(tp_b + pp_b + dp_b, 3)))
    head = (f"The stack of Chapter 3 (L = {L} blocks, D = {D}, {N_TOK} tokens per replica per step) on 16 ranks, cut tensor x pipeline x data. "
            "State is the weights, gradients and Adam moments in fp32 (16 bytes a weight) of the rank's share, 1 / (t p) of the weights; "
            "activations are not counted. The tensor column is L / p blocks, each with one forward and one backward all-reduce of n D fp32 "
            "numbers over t ranks on a ring, 2 (t - 1) / t of the array sent per rank; the pipeline column is the largest number of bytes sent by "
            "any stage, the [n, D] fp32 residual forward and its gradient back (each of the middle stages sends both, the end stages one); the "
            "data column is a ring all-reduce over d ranks of the rank's gradient, 4 bytes for each of its 1 / (t p) of the weights. "
            "All sizes in GiB (2^30 bytes) per rank per step.\n\n")
    return head + _table(["t x p x d", "state per rank", "tensor traffic sent", "pipeline traffic sent, busiest stage", "data traffic sent", "all traffic sent"], rows)


# ------------------------------------------------------------------ federated averaging

FED_COUNTS = (1, 3, 4)
FED_CENTRES = ((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0))
FED_CURV = ((1.0, 0.2), (0.2, 1.0), (1.0, 1.0))
FED_RATE, FED_STEPS = 0.5, 12


def _fed_opt(counts=FED_COUNTS):
    """The minimizer of the weighted objective sum_i n_i f_i / sum_i n_i, with f_i(w) = 1/2 sum_j h_ij (w_j - c_ij)^2."""
    return tuple(sum(n * h[j] * c[j] for n, h, c in zip(counts, FED_CURV, FED_CENTRES)) /
                 sum(n * h[j] for n, h in zip(counts, FED_CURV)) for j in range(2))


def _fed_run(local_steps, counts=FED_COUNTS):
    w = [0.0, 0.0]
    for _ in range(FED_STEPS // local_steps):
        locals_ = []
        for h, c in zip(FED_CURV, FED_CENTRES):
            wi = list(w)
            for _ in range(local_steps):
                wi = [wi[j] - FED_RATE * h[j] * (wi[j] - c[j]) for j in range(2)]
            locals_.append(wi)
        tot = sum(counts)
        w = [sum(n * wi[j] for n, wi in zip(counts, locals_)) / tot for j in range(2)]
    return w


def fedavg_counts():
    """The answer to the exercise that changes the sample counts of fedavg_rounds."""
    rows = []
    for counts in ((1, 1, 1), (1, 1, 100)):
        opt = _fed_opt(counts)
        for k in (1, 12):
            w = _fed_run(k, counts)
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(w, opt)))
            rows.append((", ".join(str(n) for n in counts), ", ".join(_g(v) for v in opt), k,
                         ", ".join(_g(v) for v in w), _g(d, 4)))
    head = ("The nodes, curvatures, centres, starting point, rate and 12 steps of fedavg_rounds, with only the sample counts changed; "
            "the counts weight both the objective and the average of the nodes' weights.\n\n")
    return head + _table(["sample counts", "minimizer", "local steps E per round", "final weights", "distance from the minimizer"],
                         rows, "llrll")


def fedavg_rounds():
    opt = _fed_opt()
    dist = lambda w: math.sqrt(sum((a - b) ** 2 for a, b in zip(w, opt)))  # noqa: E731
    ref = _fed_run(1)
    rows = []
    for k in (1, 3, 6, 12):
        w = _fed_run(k)
        rows.append((k, FED_STEPS // k, ", ".join(_g(v) for v in w), _g(dist(w), 4)))
    # the sample-weighted average of two small weight vectors, from the unit test of the distributed module
    num = [10.0 * 1 + 0.0 * 3, 20.0 * 1 + 0.0 * 3]
    avg = [v / 4 for v in num]
    head = (f"Three nodes with n = {FED_COUNTS[0]}, {FED_COUNTS[1]} and {FED_COUNTS[2]} samples. Node i has the loss f_i(w) = 1/2 sum over j of "
            f"h_ij (w_j - c_ij)^2 on two weights, with the curvatures h = {FED_CURV} and the centres c = {FED_CENTRES} (one pair per node, "
            "in node order). The common objective is the sample-weighted mean of the nodes' losses, whose minimizer is "
            f"({_g(opt[0])}, {_g(opt[1])}). Every run starts from w = (0, 0) and makes {FED_STEPS} gradient steps in all with the rate {FED_RATE}: "
            "each round, every node takes E steps of plain gradient descent on its own loss from the shared weights, and the new shared weights "
            "are the average of the nodes' weights weighted by their sample counts (summed in node order from 0.0 and divided by 1 + 3 + 4 = 8). "
            "The distance is the Euclidean distance of the final weights from the minimizer.\n\n")
    tail = (f"\n\nA run of gradient averaging, in which each of the 12 steps moves the weights along the sample-weighted mean of the nodes' "
            f"gradients at the current weights, reaches "
            f"{', '.join(_g(v) for v in _grad_avg_run())} (differences from the E = 1 row below 1e-12: "
            f"{'yes' if _dev(_grad_avg_run(), ref) < 1e-12 else 'NO'}). The sample-weighted average of the weight vectors (10, 20) with 1 sample and "
            f"(0, 0) with 3 samples is ({_g(avg[0])}, {_g(avg[1])}).")
    return head + _table(["local steps E per round", "rounds", "final weights", "distance from the minimizer"], rows, "rrll") + tail


def _grad_avg_run():
    w = [0.0, 0.0]
    tot = sum(FED_COUNTS)
    for _ in range(FED_STEPS):
        g = [sum(n * h[j] * (w[j] - c[j]) for n, h, c in zip(FED_COUNTS, FED_CURV, FED_CENTRES)) / tot for j in range(2)]
        w = [w[j] - FED_RATE * g[j] for j in range(2)]
    return w


# ------------------------------------------------------------------ expert shards of a checkpoint

def _expert_id(name):
    parts = name.split(".")
    if len(parts) > 4 and parts[0] == "blocks" and parts[1].isdigit() and parts[2] == "moe" and parts[3] == "experts" and parts[4].isdigit():
        return int(parts[4])
    return None


def expert_shards():
    names = ["token_emb.weight", "blocks.0.moe.router.weight"]
    for e in range(3):
        names += [f"blocks.0.moe.experts.{e}.{w}.weight" for w in ("w_gate", "w_up", "w_down")]
    rows = []
    for n in names:
        e = _expert_id(n)
        rows.append((f"`{n}`", "-" if e is None else e, "shared.safetensors" if e is None else f"experts/expert_{e:04d}.safetensors"))
    # the base holds 10 + e in every expert tensor; overlays replace expert 1 in turn by 42 and 77 (later overlay wins)
    base = {e: 10.0 + e for e in range(3)}
    one = dict(base)
    one[1] = 99.0
    three = dict(base)
    three[1] = 77.0                                 # the overlay of 42 is applied first and overwritten
    res = [("base alone", *[_g(base[e]) for e in range(3)]),
           ("base, then an overlay of expert 1 holding 99", *[_g(one[e]) for e in range(3)]),
           ("base, then overlays of expert 1 holding 42 and 77", *[_g(three[e]) for e in range(3)])]
    head = ("A checkpoint of one layer with three experts, as the unit tests of the shard module build it. A tensor belongs to expert E when its name "
            "has the form blocks.L.moe.experts.E.rest (L and E whole numbers); every other tensor is shared.\n\n")
    return (head + _table(["tensor", "expert", "file"], rows, "lrl") +
            "\n\nThe base holds the value 10 + E in every tensor of expert E. Assembling the base with overlays, each overlay applied in the order "
            "listed and the last writer of a tensor winning, gives these values for the first tensor of experts 0, 1 and 2:\n\n" +
            _table(["assembled from", "expert 0", "expert 1", "expert 2"], res))


# ------------------------------------------------------------------ data for the figures

def wire_curve():
    """For the figure: (R, star busiest rank GiB, ring GiB) for R = 2 .. 32."""
    out = []
    for R in range(2, 33):
        full, busy, ring = _star_ring(R)
        out.append((R, busy / GIB, ring / GIB))
    return out
