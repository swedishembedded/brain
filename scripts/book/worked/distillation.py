# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Distillation".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: distillation.<name> -->` markers. Helpers with a leading underscore, or that take arguments, are shared
with the chapter's figures so a picture and the table beside it cannot disagree.

Every trained number comes from a strongly convex problem (a softmax regression with an L2 penalty) run to a
gradient below 1e-12, so its digits are those of the unique minimiser; the stable-digits rule of `stable.py` is applied
on top, from five starting points. The grid searches and the expectations are exact sums, and the few sampled numbers
come from a named generator with a stated draw order. The step the chapter counts (tokens, weights) is the one of
`training_memory`, imported and never copied.

Swedish Embedded AB implements solutions for compressing large models into small ones for its clients. If your team
needs expertise in knowledge distillation of language models then you can procure our services by sending an email to
info@swedishembedded.com.
"""
import functools
import math

import bookrng
import number_formats as nf
import stable
import training_memory as tm

_table = nf._table


def _g(x, d=4):
    """d significant digits, written without an exponent when that is short."""
    return f"{float(f'{x:.{d}g}'):g}"


def _f(x, d=4):
    return f"{x:.{d}f}"


def softmax(z, tau=1.0):
    """The stable softmax of z / tau (max subtracted first)."""
    m = max(z)
    e = [math.exp((v - m) / tau) for v in z]
    s = sum(e)
    return [v / s for v in e]


def kl(q, p):
    """KL(q || p) in nats, terms with q = 0 contributing 0, summed in index order."""
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(q, p) if a > 0)


def entropy(q):
    return -sum(a * math.log(a) for a in q if a > 0)


def topk_teacher(z, k):
    """The ids and renormalised probabilities of the k largest entries of the logits z, ties to the lower id."""
    order = sorted(range(len(z)), key=lambda i: (-z[i], i))[:k]
    m = z[order[0]]
    e = [math.exp(z[i] - m) for i in order]
    s = sum(e)
    return order, [v / s for v in e]


def _dense(ids, q, size):
    out = [0.0] * size
    for i, v in zip(ids, q):
        out[i] = v
    return out


# ------------------------------------------------------------------ 1. the gradient of a soft target

ZT = (3.2, 2.1, 0.4, -0.5, -1.6)     # teacher logits, five classes
ZS = (1.0, 0.5, 0.3, 0.2, 0.0)       # student logits


def soft_target_gradient():
    q, p = softmax(ZT), softmax(ZS)
    k = kl(q, p)
    ce = -sum(a * math.log(b) for a, b in zip(q, p))
    h = entropy(q)
    grad = [b - a for a, b in zip(q, p)]
    hard = [b - (1.0 if i == 0 else 0.0) for i, b in enumerate(p)]

    def loss(z):
        return kl(q, softmax(z))

    dev = 0.0
    for i in range(5):
        zp, zm = list(ZS), list(ZS)
        zp[i] += 1e-6
        zm[i] -= 1e-6
        dev = max(dev, abs((loss(zp) - loss(zm)) / 2e-6 - grad[i]))
    rows = [[i, _f(q[i]), _f(p[i]), f"{grad[i]:+.4f}", f"{hard[i]:+.4f}"] for i in range(5)]
    head = ("One position with five classes. Teacher logits (3.2, 2.1, 0.4, -0.5, -1.6) and student logits (1.0, 0.5, 0.3, 0.2, "
            "0.0); q and p are their stable softmaxes (temperature 1). The soft column is the gradient of KL(q || p) with respect "
            "to the student's logits, p - q; the hard column is the gradient of the cross-entropy against the one-hot label of "
            "the teacher's largest class (class 0), p - e_0. Natural logarithms, four decimals.\n\n")
    tail = (f"\n\nKL(q || p) = {_f(k)}, the cross-entropy H(q, p) = {_f(ce)}, the teacher's entropy H(q) = {_f(h)}, and "
            f"H(q, p) - H(q) = {_f(ce - h)}. A central finite difference of KL(q || p) (step 1e-6 on each logit) differs from "
            f"p - q by less than 1e-8 (the criterion, checked). The components of each gradient column sum to zero to below 1e-15.")
    assert abs(sum(grad)) < 1e-15 and abs(sum(hard)) < 1e-15
    assert dev < 1e-8
    return head + _table(["class", "q (teacher)", "p (student)", "soft gradient p - q", "hard gradient p - e_0"], rows, "lrrrr") + tail


# ------------------------------------------------------------------ 2. what a soft target carries

V_C, D_T, D_S = 5, 6, 4         # classes, features the teacher reads, features the student reads
SCALE_T = 0.7                   # teacher weights = SCALE_T * normal()
LAMBDA, LR, STEPS = 0.05, 1.0, 800
N_MAX, N_TEST, N_FLOOR = 48, 400, 400
SIZES = (6, 12, 24, 48)
RUNS = 4


@functools.lru_cache(maxsize=None)
def _teacher():
    g = bookrng.generator("distillation.teacher")
    return [[SCALE_T * g.normal() for _ in range(D_T)] for _ in range(V_C)]


def _q_of(x):
    wt = _teacher()
    return softmax([sum(wt[v][j] * x[j] for j in range(D_T)) for v in range(V_C)])


def _contexts(name, n):
    g = bookrng.generator(name)
    return [[g.normal() for _ in range(D_T)] for _ in range(n)], g


@functools.lru_cache(maxsize=None)
def _test_set():
    xs, _ = _contexts("distillation.test", N_TEST)
    return xs, [_q_of(x) for x in xs]


def _student_p(w, x):
    return softmax([sum(w[v][j] * x[j] for j in range(D_S)) for v in range(V_C)])


def _train(xs, targets, nudge=0.0):
    """Gradient descent on the mean loss against the given target rows plus (LAMBDA / 2) times the squared weights, from
    zero (one weight starting at `nudge`). Returns the weights and the largest gradient component at the end."""
    n = len(xs)
    w = [[0.0] * D_S for _ in range(V_C)]
    w[0][0] = nudge
    gmax = 0.0
    for _ in range(STEPS):
        grad = [[LAMBDA * w[v][j] for j in range(D_S)] for v in range(V_C)]
        for x, t in zip(xs, targets):
            p = _student_p(w, x)
            for v in range(V_C):
                d = (p[v] - t[v]) / n
                for j in range(D_S):
                    grad[v][j] += d * x[j]
        gmax = max(abs(c) for row in grad for c in row)
        for v in range(V_C):
            for j in range(D_S):
                w[v][j] -= LR * grad[v][j]
    return w, gmax


def _test_kl(w):
    xs, qs = _test_set()
    return sum(kl(q, _student_p(w, x)) for x, q in zip(xs, qs)) / len(xs)


def _targets(kind, qs, us):
    out = []
    for q, u in zip(qs, us):
        if kind == "soft":
            out.append(q)
        elif kind == "top1":
            top = max(range(V_C), key=lambda v: (q[v], -v))
            out.append([1.0 if v == top else 0.0 for v in range(V_C)])
        elif kind == "top2":
            ids, qk = topk_teacher([math.log(a) for a in q], 2)
            out.append(_dense(ids, qk, V_C))
        else:
            c, y = 0.0, V_C - 1
            for v in range(V_C):
                c += q[v]
                if u < c:
                    y = v
                    break
            out.append([1.0 if v == y else 0.0 for v in range(V_C)])
    return out


@functools.lru_cache(maxsize=None)
def _run(kind, run, n, nudge=0.0):
    xs, g = _contexts(f"distillation.train.{run}", N_MAX)
    us = [g.uniform() for _ in range(N_MAX)]
    xs, us = xs[:n], us[:n]
    qs = [_q_of(x) for x in xs]
    w, gmax = _train([x[:D_S] for x in xs], _targets(kind, qs, us), nudge)
    assert gmax < 1e-12, gmax
    return _test_kl(w)


def _stable_mean(kind, n):
    base = sum(_run(kind, r, n) for r in range(RUNS)) / RUNS
    ws = [sum(_run(kind, r, n, d) for r in range(RUNS)) / RUNS for d in stable.NUDGES]
    return stable.text(base, ws, lambda x: f"{x:.4f}")


@functools.lru_cache(maxsize=None)
def _floor():
    xs, _ = _contexts("distillation.floor", N_FLOOR)
    qs = [_q_of(x) for x in xs]
    w, gmax = _train([x[:D_S] for x in xs], qs)
    assert gmax < 1e-12
    return _test_kl(w)


def curve():
    """(n, hard, soft, top-2) mean test KL over the runs, for the figure; plain runs."""
    return [(n, *(sum(_run(k, r, n) for r in range(RUNS)) / RUNS for k in ("hard", "soft", "top2"))) for n in SIZES]


def soft_versus_hard():
    seeds = ", ".join(f"{bookrng.seed_of(f'distillation.train.{r}')}" for r in range(RUNS))
    rows = []
    for n in SIZES:
        rows.append([n, _stable_mean("hard", n), _stable_mean("top2", n), _stable_mean("soft", n)])
    head = (f"A teacher and a student, both softmax classifiers over {V_C} classes. The teacher reads {D_T} features: its "
            f"{V_C} x {D_T} weight matrix is {SCALE_T} * normal() in row-major order from the generator seeded "
            f"{bookrng.seed_of('distillation.teacher')} (seed_of 'distillation.teacher'). The student reads only the first {D_S} "
            f"features, so it cannot copy the teacher. Each of the {RUNS} runs (index r = 0 to {RUNS - 1}) restarts its own "
            f"generator, named 'distillation.train.r' (seeds {seeds}), and draws the {N_MAX} training contexts first "
            f"({D_T} normal() each, in order) and then {N_MAX} uniform() values, one per context; a run with n contexts uses the "
            f"first n of both. The hard label of a context is the first class whose cumulative teacher probability exceeds its "
            f"uniform value (the last class if none does). The three targets are that one-hot label, the teacher's top 2 "
            f"classes renormalised, and the full teacher distribution. The student has no bias and starts at zero; it minimises "
            f"the mean cross-entropy against its targets plus {LAMBDA}/2 times the sum of its squared weights by {STEPS} steps "
            f"of gradient descent with rate {LR}, to a largest gradient component below 1e-12. The score is the mean over "
            f"{N_TEST} test contexts (generator 'distillation.test', seed {bookrng.seed_of('distillation.test')}, {D_T} normal() each) "
            f"of KL(q || p), q the teacher's and p the student's distribution, averaged over the {RUNS} runs. "
            f"The student fitted to the full teacher distribution of {N_FLOOR} other contexts (generator 'distillation.floor', "
            f"seed {bookrng.seed_of('distillation.floor')}) scores {_f(_floor())} on the same test set. " + stable.RULE + "\n\n")
    return head + _table(["training contexts", "one-hot label sampled from the teacher", "top 2 of the teacher", "full teacher distribution"],
                         rows, "rrrr")


# ------------------------------------------------------------------ 3. temperature

TAUS = (1, 2, 4, 8, 16, 32)


def temperature_gradient():
    c = len(ZT)
    zt_c = [v - sum(ZT) / c for v in ZT]
    zs_c = [v - sum(ZS) / c for v in ZS]
    approx = [(a - b) / c for a, b in zip(zs_c, zt_c)]
    rows = []
    for tau in TAUS:
        q, p = softmax(ZT, tau), softmax(ZS, tau)
        g = [tau * (b - a) for a, b in zip(q, p)]
        dev = max(abs(a - b) for a, b in zip(g, approx))
        # the gradient of KL(q_tau || p_tau) with respect to the student logits is (p_tau - q_tau) / tau; checked numerically
        h = 1e-6
        num = []
        for i in range(c):
            zp, zm = list(ZS), list(ZS)
            zp[i] += h
            zm[i] -= h
            num.append((kl(q, softmax(zp, tau)) - kl(q, softmax(zm, tau))) / (2 * h))
        assert max(abs(a - (b - qq) / tau) for a, b, qq in zip(num, p, q)) < 1e-8
        rows.append([tau, _f(max(q)), _f(entropy(q)), _f(tau * tau * kl(q, p), 5), _f(g[0], 5), f"{dev:.5f}"])
    head = ("The logits of the first table (teacher (3.2, 2.1, 0.4, -0.5, -1.6), student (1.0, 0.5, 0.3, 0.2, 0.0)), every "
            "probability the softmax of the logits divided by tau. 'largest q' and 'entropy' describe the teacher's distribution "
            "at that temperature (nats; ln 5 = 1.6094 is the entropy of the uniform distribution on five classes). The scaled "
            "loss is tau^2 KL(q_tau || p_tau). The gradient column is tau^2 times the first component of the gradient of that "
            "KL, that is tau (p_tau - q_tau) for class 0 (a central finite difference with step 1e-6 agrees with "
            "(p_tau - q_tau) / tau to below 1e-8). The last column is the largest difference, over the five classes, between "
            "tau (p_tau - q_tau) and (1/5) times the centred student logits minus the centred teacher logits, the limit of the "
            "derivation. The approximation for class 0 is "
            f"{_f(approx[0], 5)}.\n\n")
    return head + _table(["tau", "largest q", "entropy of q", "scaled loss", "scaled gradient, class 0", "largest gap to the limit"],
                         rows, "rrrrrr")


# ------------------------------------------------------------------ 4. top-K targets

ZT8 = (2.6, 2.1, 1.7, 0.9, 0.2, -0.4, -1.1, -2.0)     # teacher logits, eight tokens
ZS8 = (1.2, 1.9, 0.6, 1.1, 0.4, 0.0, -0.3, -0.5)      # student logits


def distill_gradient(zs, ids_q):
    """The chapter's routine: per-position gradient with respect to the logits as the sum over k of q_k (p - e_{id_k}),
    position by position, divided by the number of positions with a teacher (the K weighted passes)."""
    n_act = max(1, sum(1 for ids, _ in ids_q if ids))
    out = []
    for z, (ids, q) in zip(zs, ids_q):
        p = softmax(z)
        g = [0.0] * len(z)
        for i, qk in zip(ids, q):
            for v in range(len(z)):
                g[v] += (qk / n_act) * (p[v] - (1.0 if v == i else 0.0))
        out.append(g)
    return out


def distill_loss(zs, ids_q):
    """The chapter's routine: the mean over the positions that have a teacher of KL(q_K || p), written as the sum over
    the kept tokens of q_k (ln q_k - ln p_{id_k})."""
    n_act = max(1, sum(1 for ids, _ in ids_q if ids))
    total = 0.0
    for z, (ids, q) in zip(zs, ids_q):
        if ids:
            p = softmax(z)
            for i, qk in zip(ids, q):
                total += qk * (math.log(qk) - math.log(p[i]))
    return total / n_act


def topk_gap():
    q, p = softmax(ZT8), softmax(ZS8)
    full = kl(q, p)
    rows = []
    for k in range(1, 9):
        ids, qk = topk_teacher(list(ZT8), k)
        dense = _dense(ids, qk, 8)
        kept = sum(q[i] for i in ids)
        direct = [b - a for a, b in zip(dense, p)]
        via = distill_gradient([list(ZS8)], [(ids, qk)])[0]
        assert max(abs(a - b) for a, b in zip(direct, via)) < 1e-12
        l1 = sum(abs(a - b) for a, b in zip(dense, q))
        rows.append([k, _f(kept), _f(kl(dense, p)), _f(l1), 8 * k])
    head = ("One position, eight tokens. Teacher logits (2.6, 2.1, 1.7, 0.9, 0.2, -0.4, -1.1, -2.0), student logits (1.2, 1.9, "
            "0.6, 1.1, 0.4, 0.0, -0.3, -0.5), temperature 1. 'kept mass' is the teacher's probability on its K largest tokens "
            "before renormalising; KL is KL(q_K || p) for the renormalised top-K distribution q_K; 'distance' is the sum over "
            "the eight tokens of |q_K - q|, which is how far the gradient p - q_K is from the full gradient p - q in the sum of "
            "absolute components; 'bytes' is 4 for each stored id and 4 for each stored probability, 8 K. "
            f"For every K the gradient p - q_K equals the sum over the K kept tokens of q_k (p - e_k) to below 1e-12 (checked). "
            f"The full KL(q || p) is {_f(full)}.\n\n")
    return head + _table(["K", "kept mass", "KL(q_K \\|\\| p)", "distance of q_K from q", "bytes stored"], rows, "rrrrr")


VOCAB_EX = 100_000
TOKENS_STEP = tm.N                    # tokens per step in the volume's stack
STEP_FLOPS = 6 * tm.N * tm.P          # one forward and backward pass of the volume's stack


def topk_cost():
    rows = []
    for k in (1, 8, 32, 64, 128):
        dense = 4 * VOCAB_EX
        sparse = 8 * k
        rows.append([k, f"{sparse:,}", _g(100 * sparse / dense, 3), f"{sparse * TOKENS_STEP:,}", f"{k * STEP_FLOPS:,}"])
    head = (f"A vocabulary of {VOCAB_EX:,} tokens, one step of {TOKENS_STEP:,} positions, and the stack of Volume XIII, Chapter 3 "
            f"(2^32 weights, {STEP_FLOPS:,} FLOPs for one forward and backward pass over {TOKENS_STEP:,} tokens, 6 n P). A dense "
            f"teacher row is 4 x {VOCAB_EX:,} = {4 * VOCAB_EX:,} bytes of fp32 probabilities per position. A top-K row is 4 bytes "
            "for each id and 4 for each probability, 8 K bytes; it is smaller than the dense row while K is below half the "
            "vocabulary. The FLOP column counts K passes, one forward and one backward each, over the same tokens.\n\n")
    return head + _table(["K", "bytes per position", "percent of a dense row", f"bytes for {TOKENS_STEP:,} positions", "FLOPs of the K passes"],
                         rows, "rrrrr")


# ------------------------------------------------------------------ 5. one normalisation

V_N, D_N = 5, 3


def _norm_setup():
    g = bookrng.generator("distillation.normalisation")
    w = [[0.5 * g.normal() for _ in range(D_N)] for _ in range(V_N)]
    pos = []
    for _ in range(4):
        x = [g.normal() for _ in range(D_N)]
        zt = [1.5 * g.normal() for _ in range(V_N)]
        pos.append((x, topk_teacher(zt, 2)))
    return w, pos


def _norm_grad(w, pos, mean):
    n_act = len(pos)
    grad = [[0.0] * D_N for _ in range(V_N)]
    for x, (ids, qk) in pos:
        p = softmax([sum(w[v][j] * x[j] for j in range(D_N)) for v in range(V_N)])
        for v in range(V_N):
            gv = sum(q * (p[v] - (1.0 if v == i else 0.0)) for i, q in zip(ids, qk))
            for j in range(D_N):
                grad[v][j] += gv * x[j] / (n_act if mean else 1)
    return grad


def _norm_loss(w, pos):
    zs = [[sum(w[v][j] * x[j] for j in range(D_N)) for v in range(V_N)] for x, _ in pos]
    return distill_loss(zs, [t for _, t in pos])


def normalisation():
    w, pos = _norm_setup()
    rows = []
    for n in (1, 2, 4):
        sub = pos[:n]
        loss = _norm_loss(w, sub)
        fro = lambda m: math.sqrt(sum(c * c for r in m for c in r))
        summed, mean = fro(_norm_grad(w, sub, False)), fro(_norm_grad(w, sub, True))
        gm = _norm_grad(w, sub, True)
        dev = 0.0
        for v in range(V_N):
            for j in range(D_N):
                wp = [list(r) for r in w]
                wm = [list(r) for r in w]
                wp[v][j] += 1e-6
                wm[v][j] -= 1e-6
                dev = max(dev, abs((_norm_loss(wp, sub) - _norm_loss(wm, sub)) / 2e-6 - gm[v][j]))
        assert dev < 1e-7
        rows.append([n, _f(loss, 5), _f(summed, 5), _f(mean, 5), "passes (below 1e-7)"])
    seed = bookrng.seed_of("distillation.normalisation")
    head = (f"A student that is one linear softmax: logits W x with W of shape [{V_N}, {D_N}] and x of {D_N} numbers, the same W at "
            f"every position. One generator seeded {seed} (seed_of 'distillation.normalisation') draws, in this order: the {V_N * D_N} "
            f"entries of W (row-major, each 0.5 * normal()), then for each of 4 positions its x ({D_N} normal()) and its teacher "
            f"logits ({V_N} values, each 1.5 * normal()), whose top 2 renormalised is the position's target. An example is its first "
            "n positions. The loss is the mean over the n positions of KL(q_K || p). The two gradients are with respect to W, "
            "summed over positions: the older convention adds the positions' gradients of KL(q_K || p) without dividing, the "
            "current one divides the sum by n. The table gives the Frobenius norm of each (the square root of the sum of the "
            "squares of the 15 entries) and whether the current gradient agrees with a central finite difference "
            "of the mean loss (step 1e-6 on each entry) to below 1e-7.\n\n")
    return head + _table(["positions n", "mean KL(q_K \\|\\| p)", "norm, summed over positions", "norm, mean over positions", "finite-difference check"],
                         rows, "rrrrr")


# ------------------------------------------------------------------ 6. forward and reverse KL

POINTS = 10
MUS = [i * 0.25 for i in range(0, 37)]          # 0 to 9 in steps of 0.25
SIGMAS = [0.25 * i for i in range(1, 25)]       # 0.25 to 6 in steps of 0.25


def _bimodal(floor=0.005):
    raw = [floor + math.exp(-(v - 2) ** 2 / (2 * 0.6 ** 2)) + math.exp(-(v - 7) ** 2 / (2 * 0.6 ** 2)) for v in range(POINTS)]
    s = sum(raw)
    return [r / s for r in raw]


def _gauss(mu, sigma):
    raw = [math.exp(-(v - mu) ** 2 / (2 * sigma ** 2)) for v in range(POINTS)]
    s = sum(raw)
    return [r / s for r in raw]


def fits():
    """The grid search of the chapter's teacher (floor 0.005)."""
    return _fits(0.005)


@functools.lru_cache(maxsize=None)
def _fits(floor):
    """(teacher, forward-KL best (mu, sigma, KL, p), reverse-KL best (mu, sigma, KL, p)) by exhaustive grid search; the
    first of equal values in the order mu ascending, then sigma ascending wins."""
    q = _bimodal(floor)
    best_f = best_r = None
    for mu in MUS:
        for sigma in SIGMAS:
            p = _gauss(mu, sigma)
            f, r = kl(q, p), kl(p, q)
            if best_f is None or f < best_f[2]:
                best_f = (mu, sigma, f, p)
            if best_r is None or r < best_r[2]:
                best_r = (mu, sigma, r, p)
    return q, best_f, best_r


def forward_versus_reverse():
    q, bf, br = fits()
    rows = []
    for v in range(POINTS):
        rows.append([v, _f(q[v]), _f(bf[3][v]), _f(br[3][v])])
    low = lambda p: sum(b for a, b in zip(q, p) if a < 0.01)
    mass_left = lambda p: sum(p[:5])
    summary = [
        ["best mean mu, best sigma", f"{bf[0]}, {bf[1]}", f"{br[0]}, {br[1]}"],
        ["KL(q \\|\\| p), the forward divergence", _f(bf[2]), _f(kl(q, br[3]))],
        ["KL(p \\|\\| q), the reverse divergence", _f(kl(bf[3], q)), _f(br[2])],
        ["student mass on tokens where q < 0.01", _f(low(bf[3])), _f(low(br[3]))],
        ["student mass on tokens 0 to 4", _f(mass_left(bf[3])), _f(mass_left(br[3]))],
    ]
    head = ("A teacher over the ten tokens 0 to 9 with two peaks: q_v is proportional to 0.005 + exp(-(v - 2)^2 / (2 x 0.36)) + "
            "exp(-(v - 7)^2 / (2 x 0.36)), normalised to sum 1. The student is a discretised Gaussian, p_v proportional to "
            "exp(-(v - mu)^2 / (2 sigma^2)), with two numbers to choose: mu from 0 to 9 in steps of 0.25 and sigma from 0.25 to 6 "
            "in steps of 0.25 (37 x 24 = 888 students). Each objective is minimised by trying every student, mu ascending "
            "then sigma ascending, and keeping the first strictly smaller value. 'Forward' minimises KL(q || p), 'reverse' "
            "KL(p || q). Natural logarithms, four decimals.\n\n")
    return (head + _table(["token", "teacher q", "forward-KL student", "reverse-KL student"], rows, "rrrr") + "\n\n"
            + _table(["quantity", "forward-KL student", "reverse-KL student"], summary, "lrr"))


# ------------------------------------------------------------------ 7. the sampled gradient of the reverse KL

ZP4 = (0.5, 0.2, -0.3, 0.0)           # student logits of the four answers
HINT = 2.0                             # the privileged context adds this to the logit of answer 2


def _exact_reverse_gradient(z, logq):
    p = softmax(z)
    a = [math.log(pi) - lq for pi, lq in zip(p, logq)]
    m = sum(pi * ai for pi, ai in zip(p, a))
    return [pi * (ai - m) for pi, ai in zip(p, a)]


def sampled_gradient(z, logq, y):
    """The chapter's routine: the single-sample estimate -(ln q_y - ln p_y) (e_y - p) of the gradient of KL(p || q) with
    respect to the logits z, for the sampled answer y."""
    p = softmax(z)
    adv = logq[y] - math.log(p[y])
    return [(-adv) * ((1.0 if v == y else 0.0) - p[v]) for v in range(len(z))]


def _teacher4(z):
    t = list(z)
    t[2] += HINT
    return t


def reverse_kl_estimator():
    z = list(ZP4)
    q = softmax(_teacher4(z))
    logq = [math.log(v) for v in q]
    p = softmax(z)
    exact = _exact_reverse_gradient(z, logq)
    # expectation of the sampled estimate -A_y (e_y - p), A_y = log q_y - log p_y, over y ~ p, as an exact sum
    expect = [0.0] * 4
    for y in range(4):
        est = sampled_gradient(z, logq, y)
        for v in range(4):
            expect[v] += p[y] * est[v]
    num = []
    for i in range(4):
        zp, zm = list(z), list(z)
        zp[i] += 1e-6
        zm[i] -= 1e-6
        num.append((kl(softmax(zp), q) - kl(softmax(zm), q)) / 2e-6)
    g = bookrng.generator("distillation.estimator")
    n_draw = 2000
    mc = [0.0] * 4
    cum = []
    c = 0.0
    for v in p:
        c += v
        cum.append(c)
    for _ in range(n_draw):
        u = g.uniform()
        y = next((i for i, cv in enumerate(cum) if u < cv), 3)
        est = sampled_gradient(z, logq, y)
        for v in range(4):
            mc[v] += est[v] / n_draw
    dev = max(abs(a - b) for a, b in zip(exact, expect))
    devn = max(abs(a - b) for a, b in zip(exact, num))
    assert dev < 1e-12 and devn < 1e-8
    rows = [[i, _f(p[i]), _f(q[i]), f"{math.log(q[i]) - math.log(p[i]):+.4f}", f"{exact[i]:+.4f}", f"{expect[i]:+.4f}", f"{mc[i]:+.4f}"]
            for i in range(4)]
    seed = bookrng.seed_of("distillation.estimator")
    head = ("The student's logits are (0.5, 0.2, -0.3, 0.0) over four answers. The teacher is the same model shown a hint, which "
            f"we model as adding {HINT} to the logit of answer 2 (and nothing else), so its logits are (0.5, 0.2, 1.7, 0.0). "
            "The advantage of answer y is A_y = ln q_y - ln p_y. 'exact' is the gradient of KL(p || q) with respect to the "
            "student's logits, p_j (a_j - sum_i p_i a_i) with a = ln p - ln q. 'expected estimate' is the exact sum over the four "
            "answers of p_y x (-A_y) x (e_y - p), the expectation of the single-sample estimate. 'sample mean' averages that "
            f"estimate over {n_draw} answers drawn from p by one generator seeded {seed} (seed_of 'distillation.estimator'): "
            "each draw is one uniform(), and the answer is the first whose cumulative p exceeds it.\n\n")
    tail = (f"\n\nThe exact and the expected columns agree to below 1e-12 (the criterion, checked), and a central finite "
            f"difference of KL(p || q) (step 1e-6) agrees with the exact column to below 1e-8 (checked). "
            f"KL(p || q) is {_f(kl(p, q))} before any step.")
    return head + _table(["answer", "p", "q", "advantage A", "exact gradient", "expected estimate", f"sample mean of {n_draw}"],
                         rows, "rrrrrrr") + tail


def descent_on_the_advantage():
    z = list(ZP4)
    q = softmax(_teacher4(z))                       # frozen: the model with the hint, before any step
    rows = []
    for step in range(0, 51):
        p = softmax(z)
        if step in (0, 1, 2, 5, 10, 20, 50):
            rows.append([step, _f(p[2]), _f(p[0]), _f(kl(p, q), 5)])
        g = _exact_reverse_gradient(z, [math.log(v) for v in q])
        z = [a - 1.0 * b for a, b in zip(z, g)]
    head = ("Gradient descent on the student's four logits, rate 1.0, minimising KL(p || q) against the hinted teacher of the "
            "previous table, which stays frozen at its logits (0.5, 0.2, 1.7, 0.0), using the exact gradient. Four decimals, five "
            "for the divergence.\n\n")
    return head + _table(["step", "p(answer 2)", "p(answer 0)", "KL(p \\|\\| q)"], rows, "rrrr")


# ------------------------------------------------------------------ 8. guards on the advantage

LP_TEACHER = (-0.9, -2.2, -0.4, -3.1, -1.5, -0.7, -2.6, -1.1)
LP_OLD = (-1.0, -1.9, -2.4, -1.2, -1.6, -0.7, -0.3, -1.1)


def token_advantages(logp_teacher, logp_old, bound=None, cap=None):
    """The routine of the chapter: raw A_t = teacher - old, clamped to [-bound, bound], then scaled by cap / sum|A| when the
    sum of magnitudes exceeds cap. Returns the advantages and the number clamped."""
    a = [t - o for t, o in zip(logp_teacher, logp_old)]
    clipped = 0
    if bound is not None:
        clipped = sum(1 for v in a if abs(v) > bound)
        a = [max(-bound, min(bound, v)) for v in a]
    m = sum(abs(v) for v in a)
    if cap is not None and m > cap:
        a = [v * (cap / m) for v in a]
    return a, clipped


def advantage_guards():
    rows = []
    for label, bound, cap in (("no guard", None, None), ("bound 1.0", 1.0, None), ("cap 3.0", None, 3.0), ("bound 1.0 and cap 3.0", 1.0, 3.0)):
        a, clipped = token_advantages(LP_TEACHER, LP_OLD, bound, cap)
        rows.append([label, ", ".join(f"{v:+.3f}" for v in a), _f(sum(abs(v) for v in a), 3), _f(max(abs(v) for v in a), 3), clipped])
    head = ("Eight sampled tokens of one answer. The teacher's log-probabilities of them, in its privileged context, are (-0.9, "
            "-2.2, -0.4, -3.1, -1.5, -0.7, -2.6, -1.1); the log-probabilities the sampler recorded are (-1.0, -1.9, -2.4, -1.2, "
            "-1.6, -0.7, -0.3, -1.1). The advantage of a token is the first minus the second; a bound c clamps each advantage "
            "to [-c, c], and then, if the sum of the magnitudes exceeds a cap, every advantage is multiplied by cap divided by "
            "that sum. 'tokens clamped' counts the advantages whose magnitude exceeded the bound.\n\n")
    return head + _table(["guards", "advantages of the eight tokens", "sum of magnitudes", "largest magnitude", "tokens clamped"], rows, "lrrrr")


# ------------------------------------------------------------------ 9. staying on the teacher's path

def drift():
    rows = []
    for eps in (0.002, 0.01, 0.02):
        rows.append([eps] + [_g((1 - eps) ** t, 3) for t in (50, 100, 200, 500)])
    head = ("If at every token the student has a chance epsilon_tok of choosing a token its teacher would not, independently of the other "
            "tokens, the chance that a continuation of T tokens contains no such token is (1 - eps)^T, three significant "
            "digits with trailing zeros dropped (0.67 stands for 0.670).\n\n")
    return head + _table(["epsilon_tok", "T = 50", "T = 100", "T = 200", "T = 500"], rows, "rrrrr")


# ------------------------------------------------------------------ answers to the investigations of the chapter

def answer_top1():
    rows = [[n, _stable_mean("hard", n), _stable_mean("top1", n), _stable_mean("soft", n)] for n in SIZES]
    return ("The runs of the table 'soft versus hard' with one more target: the teacher's most likely class as a one-hot label "
            "('top 1'), beside the sampled label and the full distribution (same generators, penalty, steps and score).\n\n"
            + _table(["training contexts", "one-hot label sampled from the teacher", "top 1 of the teacher", "full teacher distribution"],
                     rows, "rrrr"))


def answer_floor():
    rows = []
    for floor in (0.005, 0.05, 0.5):
        q, bf, br = _fits(floor)
        rows.append([floor, f"{bf[0]}, {bf[1]}", f"{br[0]}, {br[1]}", _f(bf[2]), _f(br[2])])
    return ("The grid search of the table 'forward versus reverse' with the teacher's floor (the constant added to each raw "
            "weight before normalising) changed. mu, sigma are the best student's mean and spread; the divergences are those "
            "of the student chosen by that objective.\n\n"
            + _table(["floor", "forward best mu, sigma", "reverse best mu, sigma", "forward KL(q \\|\\| p)", "reverse KL(p \\|\\| q)"],
                     rows, "rrrrr"))
