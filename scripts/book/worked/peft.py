# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Parameter-Efficient Fine-Tuning".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: peft.<name> -->` markers. Helpers with a leading underscore, or that take arguments, are shared with the
chapter's figures so a picture and the table beside it cannot disagree.

The task is a linear layer W0 of 16 rows and 32 columns that must be moved to a target W*: the loss is the mean squared
error of (W - W*) x over a fixed set of inputs x. The adapter is the chapter's W = W0 + s B A, trained with Adam; the
singular value decomposition comes from `decompositions` and the memory accounts of the block stack from
`training_memory` (imported, never copied).

Swedish Embedded AB implements solutions for adapting large models on modest hardware for its clients. If your team needs
expertise in parameter-efficient fine-tuning and adapter serving then you can procure our services by sending an email
to info@swedishembedded.com.
"""
import functools
import math

import bookrng
import decompositions as dc
import number_formats as nf
import stable
import training_memory as tm

_table = nf._table

OUT, INN, NS = 16, 32, 128           # rows and columns of the layer, number of inputs
STEPS, LR = 300, 0.01                # Adam steps and rate of every training run
B1, B2, EPS = 0.9, 0.999, 1e-8
GROUP = 32                           # elements that share one scale in a quantized row
NF4 = (-1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343, -0.091050036, 0.0,
       0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0)
NONE = 4294967295                    # the slot value that means "no adapter"


# ------------------------------------------------------------------ small matrix helpers

def _mm(P, Q):
    cols = list(zip(*Q))
    return [[sum(a * b for a, b in zip(row, col)) for col in cols] for row in P]


def _t(P):
    return [list(r) for r in zip(*P)]


def _fro(P):
    return math.sqrt(sum(x * x for r in P for x in r))


def _sub(P, Q):
    return [[a - b for a, b in zip(r, q)] for r, q in zip(P, Q)]


def _add(P, Q):
    return [[a + b for a, b in zip(r, q)] for r, q in zip(P, Q)]


def _scaled(P, s):
    return [[s * x for x in r] for r in P]


def _num(v, d=4):
    return nf._num(v, d)


# ------------------------------------------------------------------ the task

@functools.lru_cache(maxsize=None)
def _task():
    """W0, the target W*, the Gram matrix M = X^T X / NS of the inputs, X, and the exact change W* - W0.

    One generator, seeded seed_of('peft.task'), draws in this order: W0 (16 x 32, row-major, each normal() times 0.5), U (16 x 2,
    row-major), V (32 x 2, row-major), G (16 x 32, row-major) and the inputs X (128 x 32, row-major), all normal()."""
    g = bookrng.generator("peft.task")
    W0 = [[0.5 * g.normal() for _ in range(INN)] for _ in range(OUT)]
    U = [[g.normal() for _ in range(2)] for _ in range(OUT)]
    V = [[g.normal() for _ in range(2)] for _ in range(INN)]
    G = [[g.normal() for _ in range(INN)] for _ in range(OUT)]
    X = [[g.normal() for _ in range(INN)] for _ in range(NS)]
    delta = [[0.02 * U[o][0] * V[i][0] + 0.01 * U[o][1] * V[i][1] + 0.004 * G[o][i] for i in range(INN)] for o in range(OUT)]
    M = [[sum(X[n][i] * X[n][j] for n in range(NS)) / NS for j in range(INN)] for i in range(INN)]
    Wt = _add(W0, delta)
    return W0, Wt, M, X, delta


def _loss(W, M, Wt):
    """1/(2 NS) sum_n |(W - W*) x_n|^2 = 1/2 sum_o e_o M e_o^T with e_o the o-th row of W - W*."""
    E = _sub(W, Wt)
    return 0.5 * sum(sum(e[i] * sum(M[i][j] * e[j] for j in range(INN)) for i in range(INN)) for e in E)


def _dw(W, M, Wt):
    """The gradient of the loss with respect to the whole weight: (W - W*) M."""
    return _mm(_sub(W, Wt), M)


class _Adam:
    """Adam with beta1 0.9, beta2 0.999, epsilon 1e-8, bias correction, no weight decay, on a list of parameter matrices."""

    def __init__(self, shapes):
        self.m = [[[0.0] * c for _ in range(r)] for r, c in shapes]
        self.v = [[[0.0] * c for _ in range(r)] for r, c in shapes]
        self.t = 0

    def step(self, params, grads, lr=LR):
        self.t += 1
        c1, c2 = 1 - B1 ** self.t, 1 - B2 ** self.t
        for p, gr, m, v in zip(params, grads, self.m, self.v):
            for i, (prow, grow, mrow, vrow) in enumerate(zip(p, gr, m, v)):
                for j, gij in enumerate(grow):
                    mrow[j] = B1 * mrow[j] + (1 - B1) * gij
                    vrow[j] = B2 * vrow[j] + (1 - B2) * gij * gij
                    prow[j] -= lr * (mrow[j] / c1) / (math.sqrt(vrow[j] / c2) + EPS)


# ------------------------------------------------------------------ the adapter

def _init_A(r, name="peft.rank_sweep"):
    """A (r x 32): normal() / sqrt(32), row-major, from a generator seeded seed_of(name) that this run starts afresh."""
    g = bookrng.generator(name)
    return [[g.normal() / math.sqrt(INN) for _ in range(INN)] for _ in range(r)]


def _lora_weight(W0, A, B, s):
    """W = W0 + s B A."""
    return _add(W0, _scaled(_mm(B, A), s))


def _lora_gradients(dW, A, B, s):
    """(dA, dB) = (s B^T dW, s dW A^T) from the gradient dW with respect to the whole weight."""
    return _scaled(_mm(_t(B), dW), s), _scaled(_mm(dW, _t(A)), s)


def _train_lora(base, r, nudge=0.0, steps=STEPS, name="peft.rank_sweep", target=None, alpha=None):
    """Adam on A and B only, from B = 0, against the task's loss; the base stays as given. Returns (loss after `steps`, A, B).

    The scale is s = alpha / r with alpha = r unless given. `nudge` is added to A[0][0] before the first step."""
    W0, Wt, M, _, _ = _task()
    Wt = target or Wt
    s = (alpha if alpha is not None else r) / r
    A = _init_A(r, name)
    A[0][0] += nudge
    B = [[0.0] * r for _ in range(OUT)]
    opt = _Adam([(r, INN), (OUT, r)])
    for _ in range(steps):
        dW = _dw(_lora_weight(base, A, B, s), M, Wt)
        dA, dB = _lora_gradients(dW, A, B, s)
        opt.step([A, B], [dA, dB])
    return _loss(_lora_weight(base, A, B, s), M, Wt), A, B


def _train_full(nudge=0.0, steps=STEPS):
    """Adam on every entry of the weight, from W0, against the same loss."""
    W0, Wt, M, _, _ = _task()
    W = [list(r) for r in W0]
    W[0][0] += nudge
    opt = _Adam([(OUT, INN)])
    for _ in range(steps):
        opt.step([W], [_dw(W, M, Wt)])
    return _loss(W, M, Wt)


def _stable_loss(fn):
    """The loss of a run `fn(nudge)` written with the digits that survive the nudged copies of it."""
    base = fn(0.0)
    ws = [fn(d) for d in stable.NUDGES]
    if max([base] + ws) < 1e-9:
        return "below 1e-9"
    return stable.text(base, ws, lambda x: f"{x:.4g}")


# ------------------------------------------------------------------ 1. the change a fine-tune needs

def task_spectrum():
    W0, Wt, M, X, delta = _task()
    U, sig, V = dc._svd(delta)
    loss0 = _loss(W0, M, Wt)
    total = sum(x * x for x in sig)
    rows = []
    for r in (1, 2, 3, 4, 8):
        trunc = dc._truncate(U, sig, V, r)
        kept = sum(x * x for x in sig[:r]) / total
        rows.append([r, _num(sig[r - 1], 4), f"{100 * kept:.3f}", r * (OUT + INN), _num(_loss(_add(W0, trunc), M, Wt), 4)])
    seed = bookrng.seed_of("peft.task")
    head = (f"The layer has {OUT} rows and {INN} columns ({OUT * INN} weights). One generator seeded {seed} (seed_of 'peft.task') draws, "
            f"in this order and each by normal(): W0 ({OUT} x {INN}, row-major, every draw times 0.5), U ({OUT} x 2, row-major), "
            f"V ({INN} x 2, row-major), G ({OUT} x {INN}, row-major) and {NS} inputs x_n of {INN} numbers (row-major). "
            "The change the fine-tune needs is dW* = 0.02 u1 v1^T + 0.01 u2 v2^T + 0.004 G (u1, u2 the columns of U, v1, v2 the "
            "columns of V), the target is W* = W0 + dW*, and the loss of a weight W is the mean over the inputs of half the squared "
            f"length of (W - W*) x_n, written L(W) = (1 / (2 {NS})) sum_n |(W - W*) x_n|^2; its gradient with respect to the whole weight is "
            f"(W - W*) M with M = (1 / {NS}) sum_n x_n x_n^T. Singular values "
            "come from one-sided Jacobi rotations in float64. The loss of the base itself is "
            f"L(W0) = {_num(loss0, 4)}. Row r gives the r-th singular value of dW*, the share of its squared Frobenius norm held by "
            "the first r singular values, the count r (16 + 32) of numbers in two factors of rank r, and the loss of the weight "
            "W0 + (best rank-r approximation of dW* in the Frobenius norm), called the floor of rank r below. The loss weights the "
            "directions of the inputs by their spread, which the Frobenius norm does not, so a trained rank-r change can end "
            "slightly under its floor.\n\n")
    return head + _table(["r", "singular value r", "share of squared norm kept (percent)", "numbers in the two factors",
                          "loss with the best rank-r change"], rows, "rrrrr")


# ------------------------------------------------------------------ 2. the gradients of the two factors

def lora_gradient_check():
    W0, Wt, M, _, _ = _task()
    r, s = 3, 2.0 / 3
    g = bookrng.generator("peft.lora_gradient_check")
    A = [[g.normal() / math.sqrt(INN) for _ in range(INN)] for _ in range(r)]
    B = [[0.1 * g.normal() for _ in range(r)] for _ in range(OUT)]

    def f(A, B):
        return _loss(_lora_weight(W0, A, B, s), M, Wt)

    dA, dB = _lora_gradients(_dw(_lora_weight(W0, A, B, s), M, Wt), A, B, s)
    h = 1e-6
    worst = 0.0
    for P, dP in ((A, dA), (B, dB)):
        for i in range(len(P)):
            for j in range(len(P[0])):
                keep = P[i][j]
                P[i][j] = keep + h
                up = f(A, B)
                P[i][j] = keep - h
                down = f(A, B)
                P[i][j] = keep
                num = (up - down) / (2 * h)
                worst = max(worst, abs(num - dP[i][j]) / max(1.0, abs(dP[i][j])))
    # the same weight at the start of training: B = 0
    A0 = _init_A(r)
    B0 = [[0.0] * r for _ in range(OUT)]
    dA0, dB0 = _lora_gradients(_dw(_lora_weight(W0, A0, B0, s), M, Wt), A0, B0, s)
    seed = bookrng.seed_of("peft.lora_gradient_check")
    head = (f"Rank r = {r} and scale s = 2/3 (alpha = 2). One generator seeded {seed} (seed_of 'peft.lora_gradient_check') draws A "
            f"({r} x {INN}, row-major, each normal() divided by sqrt({INN})) and then B ({OUT} x {r}, row-major, each normal() times 0.1). "
            f"The analytic gradients dA = s B^T dW and dB = s dW A^T, with dW = (W - W*) M for W = W0 + s B A, are compared with "
            f"central differences of the loss with step {h:g} at every one of the {r * INN + OUT * r} entries; the deviation is "
            "|numeric - analytic| / max(1, |analytic|), and the criterion is that its largest value is below 1e-6.\n\n")
    ok = "passes" if worst < 1e-6 else "FAILS"
    rows = [[f"largest deviation over all {r * INN + OUT * r} entries", f"{ok} (below 1e-6)"],
            ["|dA| and |dB| (Frobenius) at B = 0 with s = 2/3 (alpha = 2, r = 3), A the rank-3 start of the sweep",
             f"{_num(_fro(dA0), 4)} and {_num(_fro(dB0), 4)}"]]
    return head + _table(["quantity", "value"], rows, "lr")


# ------------------------------------------------------------------ 3. how many numbers, how much loss

def rank_sweep():
    W0, Wt, M, _, delta = _task()
    U, sig, V = dc._svd(delta)
    rows = []
    for r in (1, 2, 4, 8):
        loss = _stable_loss(lambda d, r=r: _train_lora(W0, r, d)[0])
        floor = _loss(_add(W0, dc._truncate(U, sig, V, r)), M, Wt)
        rows.append([f"LoRA, rank {r}", r * (OUT + INN), f"{100 * r * (OUT + INN) / (OUT * INN):.2f}", loss, _num(floor, 4)])
    full = _stable_loss(lambda d: _train_full(d))
    rows.append(["every weight", OUT * INN, "100.00", full, "-"])
    seed = bookrng.seed_of("peft.rank_sweep")
    head = (f"The task of the first table, {STEPS} Adam steps (beta1 0.9, beta2 0.999, epsilon 1e-8, bias-corrected, no weight decay, "
            f"rate {LR}) on the trainable numbers only. LoRA starts from B = 0 and A = normal() / sqrt({INN}) in row-major order, "
            f"from a generator seeded {seed} (seed_of 'peft.rank_sweep') that every run starts afresh, so the rank-2 A begins with the rank-1 A; "
            "the scale is s = alpha / r with alpha = r, that is s = 1. The full run starts from W0 and updates all weights. "
            "The loss column is the loss after the last step, and a loss below 1e-9 is printed as such. " + stable.RULE +
            " In the nudged runs the first entry of A (of the weight, in the full run) starts at the stated tiny amount above its "
            "value. The last column repeats the floor of the first table.\n\n")
    return head + _table(["trained", "trainable numbers", "percent of the weights", f"loss after {STEPS} steps",
                          "floor of the best change of that rank"], rows, "lrrrr")


def rank_curve():
    """(rank, loss after training, floor of the best change of that rank) for the figure; the plain runs, without the nudged copies."""
    W0, Wt, M, _, delta = _task()
    U, sig, V = dc._svd(delta)
    return [(r, _train_lora(W0, r)[0], _loss(_add(W0, dc._truncate(U, sig, V, r)), M, Wt)) for r in (1, 2, 3, 4, 6, 8)]


# ------------------------------------------------------------------ 4. what the adapters cost in memory

def memory_bill():
    D, F, L, N = tm.D, tm.F, tm.L, tm.N
    r = 16
    weights = L * 2 * D * F
    adapter = L * 2 * r * (D + F)
    kept = L * (D + F) * N * 4                     # the two kept tensors of every block, fp32
    mids = L * 2 * r * N * 4                       # the r numbers per token that each adapter's first product leaves
    gib = lambda b: b / 2 ** 30
    rows = []
    for label, bpw in (("fp32", 4.0), ("bf16", 2.0), ("8-bit codes, one fp32 scale per 32", 1 + 4 / 32),
                       ("4-bit codes, one fp32 scale per 32", 0.5 + 4 / 32)):
        base = weights * bpw
        state = adapter * 16
        total = base + state + kept + mids
        rows.append([label, f"{bpw:.3f}", f"{gib(base):.2f}", f"{gib(state):.2f}", f"{gib(kept + mids):.2f}", f"{gib(total):.2f}"])
    full = weights * 16 + kept
    head = (f"The stack of Chapter 3: {L} blocks of two matrices each, [{F}, {D}] and [{D}, {F}], so {weights:,} weights, and a step of {N:,} "
            f"tokens. An adapter of rank {r} on both matrices of every block has {r} x ({D} + {F}) = {r * (D + F):,} numbers per matrix and "
            f"{adapter:,} in all ({100 * adapter / weights:.3f} percent of the weights). The adapter's state is 16 bytes a number (value, "
            "gradient and two Adam moments, all fp32); the base is frozen, so it has no gradient and no moments. The activation columns "
            f"add the {D + F:,} numbers per token and block that Chapter 3 keeps (fp32) and the {r} numbers per token that each of the two "
            "adapters' first products leave for the backward pass (two per block). Sizes are in GiB (2^30 bytes); the second column is "
            "bytes per frozen weight.\n\n")
    foot = (f"\n\nTraining every weight in fp32 costs {gib(full):.2f} GiB on the same stack (weights, gradients and moments at 16 bytes a "
            f"weight plus the same {gib(kept):.2f} GiB of kept tensors).")
    return head + _table(["frozen base stored as", "bytes per weight", "base", "adapter state", "activations", "total"], rows, "lrrrrr") + foot


# ------------------------------------------------------------------ 5. quantizing the frozen base

def _round_half_away(x):
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def _quantize_row(row, fmt):
    """Decode(encode(row)) for groups of 32: int8 (scale max|g| / 127, round half away from zero, clamp +-127), uniform 4-bit
    (scale max|g| / 7, same rounding, clamp +-7) or NF4 (scale max|g|, the nearest of the 16 codebook values, a tie to the lower
    code)."""
    out = []
    for a in range(0, len(row), GROUP):
        blk = row[a:a + GROUP]
        top = max(max(abs(x) for x in blk), 1e-8)
        if fmt == "int8":
            s = top / 127
            out += [max(-127, min(127, _round_half_away(x / s))) * s for x in blk]
        elif fmt == "int4":
            s = top / 7
            out += [max(-7, min(7, _round_half_away(x / s))) * s for x in blk]
        else:
            s = top / 1.0
            for x in blk:
                v = x / s
                best = min(range(16), key=lambda c: (abs(v - NF4[c]), c))
                out.append(NF4[best] * s)
    return out


def _quantize(W, fmt):
    return W if fmt == "fp32" else [_quantize_row(r, fmt) for r in W]


FORMATS = (("fp32", "fp32", 32.0), ("int8", "8-bit, steps of max / 127", 8 + 32 / GROUP),
           ("int4", "4-bit, steps of max / 7", 4 + 32 / GROUP), ("nf4", "4-bit, the 16 values of the codebook", 4 + 32 / GROUP))


def nf4_error():
    g = bookrng.generator("peft.nf4_error")
    rows_n, cols_n = 64, 256
    W = [[0.02 * g.normal() for _ in range(cols_n)] for _ in range(rows_n)]
    rows = []
    for fmt, label, bits in FORMATS[1:]:
        Q = _quantize(W, fmt)
        rel = _fro(_sub(Q, W)) / _fro(W)
        worst = max(abs(a - b) for rq, rw in zip(Q, W) for a, b in zip(rq, rw))
        rows.append([label, f"{bits:.2f}", f"{rel:.4f}", f"{worst:.5f}"])
    gaps = ", ".join(f"{NF4[c + 1] - NF4[c]:.4f}" for c in range(15))
    seed = bookrng.seed_of("peft.nf4_error")
    head = (f"A matrix of {rows_n} rows and {cols_n} columns, every entry 0.02 times normal(), drawn row-major from one generator seeded "
            f"{seed} (seed_of 'peft.nf4_error'). Each row is cut into groups of {GROUP} consecutive entries with one scale each; the scale is "
            "the group's largest magnitude divided by 127, 7 or 1 (1 is the largest magnitude of the codebook), the integer formats round "
            "half away from zero and clamp, and the codebook format takes the nearest of the 16 values below, a tie going to the lower "
            "code. The codebook is " + ", ".join(f"{v:g}" for v in NF4) + f", and the gaps between neighbouring values are {gaps}. "
            "Bits per weight include one fp32 scale (32 bits) per group. The relative error is the Frobenius norm of "
            "(decoded - original) over that of the original, and the worst error is the largest absolute difference of one entry.\n\n")
    return head + _table(["format", "bits per weight", "relative error", "worst error of one entry"], rows, "lrrr")


def lora_over_quantized_base():
    W0, Wt, M, _, delta = _task()
    rows = []
    for fmt, label, bits in FORMATS:
        Q = _quantize(W0, fmt)
        shifted = _add(Q, delta)                    # the target defined relative to the deployed base: Q + dW*
        before_q = _loss(Q, M, shifted)
        after_q = _stable_loss(lambda d, Q=Q, shifted=shifted: _train_lora(Q, 2, d, target=shifted)[0])
        before = _loss(Q, M, Wt)
        after = _stable_loss(lambda d, Q=Q: _train_lora(Q, 2, d)[0])
        rows.append([label, f"{bits:.2f}", _num(before_q, 4), after_q, _num(before, 4), after])
    head = ("The task of the first table with the frozen base replaced by its decoded copy Q (groups of 32, which is one group a row here). "
            f"A rank-2 adapter (alpha = 2, so s = 1) is trained for {STEPS} Adam steps exactly as in the rank sweep, twice. In the first "
            "run the target is Q + dW*, the same change asked of the deployed base; in the second the target is the original W* = W0 + dW*, "
            "so that the loss counts what the quantization took away as well as what the fine-tune must add. Each pair of columns gives "
            "the loss before training and after it. "
            "A loss below 1e-9 is printed as such. " + stable.RULE + "\n\n")
    return head + _table(["base", "bits per weight", "before, target Q + dW*", "after, target Q + dW*", "before, target W*", "after, target W*"],
                         rows, "lrrrrr")


# ------------------------------------------------------------------ 6. merging an adapter into a base

def _int8_encode(W):
    """The integer codes and group scales of an int8 matrix: one scale max|group| / 127 per group of 32, codes rounded half away
    from zero and clamped to +-127."""
    codes, scales = [], []
    for row in W:
        for a in range(0, len(row), GROUP):
            blk = row[a:a + GROUP]
            sc = max(max(abs(x) for x in blk), 1e-8) / 127
            scales.append(sc)
            codes.append([max(-127, min(127, _round_half_away(x / sc))) for x in blk])
    return codes, scales


def _int8_decode(codes, scales):
    rows = []
    per_row = INN // GROUP
    for o in range(len(codes) // per_row):
        row = []
        for g in range(per_row):
            row += [c * scales[o * per_row + g] for c in codes[o * per_row + g]]
        rows.append(row)
    return rows


def _fold_stats(Wq, s_delta):
    """(codes that stay the same, |survivor - intended| / |intended|) of re-encoding Wq + s_delta, where Wq is a decoded int8
    matrix (so it re-encodes to itself) and the survivor is the decoded result minus Wq."""
    base_codes, base_scales = _int8_encode(Wq)
    codes, scales = _int8_encode(_add(Wq, s_delta))
    unchanged = sum(1 for cb, cm in zip(base_codes, codes) for a, b in zip(cb, cm) if a == b)
    survived = _sub(_int8_decode(codes, scales), _int8_decode(base_codes, base_scales))
    return unchanged, _fro(_sub(survived, s_delta)) / _fro(s_delta)


def fold_into_quantized():
    W0, Wt, M, X, _ = _task()
    Wq = _quantize(W0, "int8")
    _, A, B = _train_lora(Wq, 2, 0.0)
    delta = _mm(B, A)                              # s = 1
    step = sum(max(abs(x) for x in r) for r in Wq) / OUT / 127
    rows = []
    for strength in (0.05, 0.1, 0.25, 0.5, 1.0):
        sd = _scaled(delta, strength)
        unchanged, rel = _fold_stats(Wq, sd)
        rms = _fro(sd) / math.sqrt(OUT * INN)
        # the runtime route: y = Wq x + (strength) B (A x) in float64, against the exact sum
        worst = 0.0
        for x in X[:16]:
            exact = [sum((Wq[o][i] + sd[o][i]) * x[i] for i in range(INN)) for o in range(OUT)]
            ax = [sum(A[k][i] * x[i] for i in range(INN)) for k in range(2)]
            run = [sum(Wq[o][i] * x[i] for i in range(INN)) + strength * sum(B[o][k] * ax[k] for k in range(2)) for o in range(OUT)]
            worst = max(worst, max(abs(a - b) for a, b in zip(exact, run)))
        rows.append([f"{strength:g}", f"{rms / step:.3f}", f"{100 * unchanged / (OUT * INN):.1f}", f"{rel:.3f}", "below 1e-12" if worst < 1e-12 else f"{worst:.1e}"])
    head = ("The int8 base Q of the task (W0 quantized in groups of 32, one group per row, scale = largest magnitude / 127, so Q is the "
            "decoded matrix) and the rank-2 adapter trained on it with the rank-sweep recipe (s = 1, the task's original target W*, "
            "not Q + the change), multiplied by a strength. 'Delta in steps' is the root mean square of one entry of "
            "strength x B A divided by the mean step of the base, the mean over the 16 rows of their scale. The fold re-encodes "
            "Q + strength x B A (the W of the next sentence is Q) with the same rule "
            "and subtracts the decoded base; 'same code' is the percent of the 512 weights whose integer code is the old one, and the next "
            "column the Frobenius norm of (what survived - what was intended) over the norm of what was intended. The runtime route "
            "computes y = W x + strength B (A x) with W the decoded base; its last column is the largest difference from "
            "(W + strength B A) x over the first 16 inputs, in float64. The strengths multiply the adapter, which makes the 'delta in steps' "
            "column the dial.\n\n")
    return head + _table(["strength", "delta in steps", "weights with the same code (percent)", "relative error of the fold",
                          "largest output difference of the runtime route"], rows, "lrrrr")


# ------------------------------------------------------------------ 7. several adapters over one base

def merge_adapters():
    W0, _, _, _, _ = _task()
    g = bookrng.generator("peft.merge_adapters")
    A1 = [[0.2 * g.normal() for _ in range(INN)] for _ in range(2)]
    B1 = [[0.2 * g.normal() for _ in range(2)] for _ in range(OUT)]
    A2 = [[0.2 * g.normal() for _ in range(INN)] for _ in range(3)]
    B2 = [[0.2 * g.normal() for _ in range(3)] for _ in range(OUT)]
    s1, s2 = 1.0, 0.5
    sums = _add(_scaled(_mm(B1, A1), s1), _scaled(_mm(B2, A2), s2))
    rev = _add(_scaled(_mm(B2, A2), s2), _scaled(_mm(B1, A1), s1))
    Bc = [b1 + b2 for b1, b2 in zip(_scaled(B1, s1), _scaled(B2, s2))]
    Ac = A1 + A2
    cat = _mm(Bc, Ac)
    U, sig, V = dc._svd(sums)
    d_cat = max(abs(a - b) for ra, rb in zip(sums, cat) for a, b in zip(ra, rb))
    d_rev = max(abs(a - b) for ra, rb in zip(sums, rev) for a, b in zip(ra, rb))
    rows = [["largest entry of |sum form - concatenated form|", "below 1e-15" if d_cat < 1e-15 else f"{d_cat:.1e}"],
            ["largest entry of |adapter 1 first - adapter 2 first|", "below 1e-15" if d_rev < 1e-15 else f"{d_rev:.1e}"],
            ["rank of the merged change (singular values above 1e-12)", sum(1 for x in sig if x > 1e-12)]]
    for r in (1, 2, 3, 4):
        rows.append([f"error of the best rank-{r} copy of the merged change, over its norm",
                     f"{_fro(_sub(sums, dc._truncate(U, sig, V, r))) / _fro(sums):.4f}"])
    seed = bookrng.seed_of("peft.merge_adapters")
    head = (f"Two adapters on the 16 x 32 layer: A1 (2 x 32), B1 (16 x 2), A2 (3 x 32) and B2 (16 x 3), every entry 0.2 times normal(), drawn "
            f"in the order A1, B1, A2, B2 (each row-major) from one generator seeded {seed} (seed_of 'peft.merge_adapters'). The strengths are "
            "s1 = 1 and s2 = 0.5 (the scale alpha / r of each adapter folded into its strength). The sum form is s1 B1 A1 + s2 B2 A2; the "
            "concatenated form is [s1 B1 | s2 B2] times A1 stacked over A2, a single product of rank 5. Singular values come from "
            "one-sided Jacobi rotations in float64.\n\n")
    return head + _table(["quantity", "value"], rows, "lr")


# ------------------------------------------------------------------ 8. many adapters in one batch

BANK_RANKS, R_MAX, ALPHA = (2, 4, 1), 4, 4.0
BANK_SLOTS = (0, 2, NONE, 1, 0, 2)


def _bank(poison=0.0):
    """The bank of the example: for slot k, A_k padded to R_MAX rows and (alpha / r_k) B_k transposed and padded to R_MAX rows;
    the padding rows hold `poison`. Returns (A_bank, Bt_bank, x rows)."""
    g = bookrng.generator("peft.bank")
    A_bank, Bt_bank, parts = [], [], []
    for r in BANK_RANKS:
        A = [[0.2 * g.normal() for _ in range(INN)] for _ in range(r)]
        B = [[0.2 * g.normal() for _ in range(r)] for _ in range(OUT)]
        parts.append((A, B, ALPHA / r))
        A_bank.append(A + [[poison] * INN for _ in range(R_MAX - r)])
        Bt = _t(_scaled(B, ALPHA / r))
        Bt_bank.append(Bt + [[poison] * OUT for _ in range(R_MAX - r)])
    xs = [[g.normal() for _ in range(INN)] for _ in BANK_SLOTS]
    return A_bank, Bt_bank, xs, parts


def _bank_rows(W0, A_bank, Bt_bank, xs):
    """y_i = W0 x_i + (t Bt_bank[z_i][0:r]) with t = A_bank[z_i][0:r] x_i, the sum over the r rows taken in order and then added to y_i
    (as bank_forward in the chapter); a slot outside the bank adds nothing."""
    ys = []
    for z, x in zip(BANK_SLOTS, xs):
        y = [sum(W0[o][i] * x[i] for i in range(INN)) for o in range(OUT)]
        if z < len(BANK_RANKS):
            r = BANK_RANKS[z]
            t = [sum(A_bank[z][c][i] * x[i] for i in range(INN)) for c in range(r)]
            y = [y[o] + sum(t[c] * Bt_bank[z][c][o] for c in range(r)) for o in range(OUT)]
        ys.append(y)
    return ys


def bank_semantics():
    W0, _, _, _, _ = _task()
    A_bank, Bt_bank, xs, parts = _bank()
    ys = _bank_rows(W0, A_bank, Bt_bank, xs)
    A_bad, Bt_bad, _, _ = _bank(poison=1e6)
    ys_bad = _bank_rows(W0, A_bad, Bt_bad, xs)
    rows = []
    for i, (z, x) in enumerate(zip(BANK_SLOTS, xs)):
        if z < len(parts):
            A, B, s = parts[z]
            Wm = _lora_weight(W0, A, B, s)
            ref = [sum(Wm[o][j] * x[j] for j in range(INN)) for o in range(OUT)]
        else:
            ref = [sum(W0[o][j] * x[j] for j in range(INN)) for o in range(OUT)]
        diff = max(abs(a - b) for a, b in zip(ys[i], ref))
        same = ys[i] == ys_bad[i]
        rows.append([i, "none (4294967295)" if z == NONE else z, "-" if z == NONE else BANK_RANKS[z],
                     "below 1e-12" if diff < 1e-12 else f"{diff:.1e}", "yes" if same else "no"])
    seed = bookrng.seed_of("peft.bank")
    head = (f"The 16 x 32 layer W0 of the task (the first 512 draws of the generator seeded seed_of 'peft.task', each normal() times 0.5, "
            f"row-major) and a bank of three adapters whose ranks are {BANK_RANKS[0]}, {BANK_RANKS[1]} and {BANK_RANKS[2]}, padded to "
            f"r_max = {R_MAX} rows. One generator seeded {seed} (seed_of 'peft.bank') draws, for slot 0, 1, 2 in turn, A_k (r_k x 32, row-major) "
            "and then B_k (16 x r_k, row-major), every entry 0.2 times normal(), and after the three slots draws the six input rows "
            "(32 numbers each, row-major, each normal() with no factor). The bank stores A_k padded to r_max rows and the transposed (alpha / r_k) B_k, alpha = 4, "
            "padded the same way; in the 'poisoned' copy every padding row holds 1e6 instead of zero. A batch of six rows carries the slot vector "
            f"{list(BANK_SLOTS[:2])} + [none] + {list(BANK_SLOTS[3:])}; row i computes t = A_k x_i over the first r_k rows of its slot and adds "
            "t B_k^T (scaled) to W0 x_i, and a row whose slot is outside the bank adds nothing. The 'difference' column is the largest "
            "entry of |bank row - (merged weight) x_i| in float64, where the merged weight is W0 + (alpha / r_k) B_k A_k (W0 for a row "
            "with no adapter); the last column asks whether the poisoned padding changes any bit of the row.\n\n")
    return head + _table(["row", "slot", "rank used", "difference from the merged weight", "same bits with poisoned padding"], rows, "rrrrr")


def kv_difference():
    W0, _, _, _, _ = _task()
    _, _, xs, parts = _bank()
    x = xs[0]
    base = [sum(W0[o][j] * x[j] for j in range(INN)) for o in range(OUT)]
    rows = []
    for k, (A, B, s) in enumerate(parts):
        Wm = _lora_weight(W0, A, B, s)
        key = [sum(Wm[o][j] * x[j] for j in range(INN)) for o in range(OUT)]
        rel = math.sqrt(sum((a - b) ** 2 for a, b in zip(key, base))) / math.sqrt(sum(b * b for b in base))
        rows.append([f"adapter {k}", f"{rel:.4f}"])
    head = ("The key a token gets from a layer, y = W x, with the first input row of the bank example (as drawn there, entries normal()) as x and "
            "the 16 x 32 layer W0 of the task (as in the bank example) as the key "
            "projection of one position, once without an adapter and once with each adapter of the bank merged in. The table gives "
            "|key with the adapter - key of the base| over |key of the base| (Euclidean norms).\n\n")
    return head + _table(["adapter", "relative change of the cached key"], rows, "lr")


# ------------------------------------------------------------------ 9. serving many adapters

def serving_cost():
    D, F, L = tm.D, tm.F, tm.L
    r = 16
    weights = L * 2 * D * F
    adapter = L * 2 * r * (D + F)
    gib = lambda b: b / 2 ** 30
    rows = []
    for n in (1, 10, 100, 1000):
        merged = n * weights * 4
        bank = weights * 4 + n * adapter * 4
        rows.append([f"{n:,}", f"{gib(merged):,.2f}", f"{gib(bank):,.2f}"])
    base_flops = 2 * weights
    extra = 2 * adapter
    head = (f"The stack of Chapter 3 ({weights:,} weights) with adapters of rank {r} on both matrices of every block ({adapter:,} numbers each, "
            "kept in fp32). A merged copy of the stack costs all the weights in fp32; the bank keeps one fp32 base and the adapters. "
            f"Per token, one multiply-add counts as 2 floating-point operations (FLOPs): the base matrices cost {base_flops:,} FLOPs, and the adapter of the row's slot "
            f"adds {extra:,} FLOPs ({100 * extra / base_flops:.3f} percent of the base), whatever the number of adapters in the bank. "
            "Sizes are in GiB.\n\n")
    return head + _table(["adapters served", "merged copies, one per adapter", "one base and a bank"], rows, "rrr")
