# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Continual Learning and Promotion".

Plain Python 3, standard library only, written from the chapter's own description. One small network carries the chapter: a
two-input classifier with two hidden layers of six tanh units and one logistic output (67 numbers), trained by full-batch gradient
descent on the mean binary cross-entropy.

* Forgetting. Task A and task B ask the same question (which side of the line x1 = 0 is the point on) in two clusters of the plane
  with opposite answers. The network is trained on A, then on B alone, and the accuracy on A is counted as a function of the steps
  and of the step size.
* Remedies, counted against each other on the same toy: a buffer of old rows chosen by reservoir sampling and mixed into every step
  (replay), a quadratic anchor on the old weights (an L2 anchor, and elastic weight consolidation with the exact Fisher diagonal of
  the old task), and a rank-one adapter on one weight matrix with the base frozen.
* The probe bank, the paired comparison of a candidate with the incumbent on it, the arithmetic of why a pooled mean goes blind,
  a simulation of many blocks and of the margin of non-inferiority, and a continual reader on four episodes in five arms (the loop
  of brain's `audit` crate, with the gate of its `promote` crate, in miniature).

Every number that comes from training goes through `stable` (the rule of `value_learning`): the whole experiment is repeated with the
first initial weight changed by each nudge of `stable.py`, and a digit is printed only if the run and its four copies share it. The
experiments are independent of one another, so the five copies of each run in a pool of four worker processes. Each public function
without a leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: continual_learning.<name> -->` markers. Underscore functions and the constants are shared with the chapter's figures.

Swedish Embedded AB implements solutions for models that keep learning without quietly losing what they could do. If your team needs
expertise in continual learning and in deciding whether a retrained model may replace the one in use, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import hashlib
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import stable
from statistics_ch import _table, _tail_ge, _z_central

# ------------------------------------------------------------------ the network and its data

DATA = "continual_learning.data"
INIT = "continual_learning.init"
STREAM = "continual_learning.stream"
RESERVOIR = "continual_learning.reservoir"
REPLAY = "continual_learning.replay"
BLOCKS = "continual_learning.blocks"

H = 6                                   # units in each hidden layer
OFF = {"W1": 0, "b1": 12, "W2": 18, "b2": 54, "w3": 60, "b3": 66}
P = 67                                  # 2*6 + 6 + 6*6 + 6 + 6 + 1
INIT_SCALE = 0.5
CENTRE_A, CENTRE_B = -1.5, 1.5          # x0 of the clusters of task A and of task B
SPREAD = 0.5                            # standard deviation of x0 inside a cluster
N_TRAIN, N_TEST, N_BANK = 100, 400, 100
PRETRAIN_RATE, PRETRAIN_STEPS = 0.5, 300
RATES = (0.05, 0.2, 0.5)
CHECK_STEPS = (0, 10, 20, 50, 100, 200)
CURVE_EVERY = 5
PROTOCOL_RATE, PROTOCOL_STEPS = 0.2, 200
BUFFERS = (0, 5, 10, 20, 50, 100)
L2_LAMBDAS = (0.1, 0.3, 1.0, 3.0)
EWC_LAMBDAS = (10.0, 100.0, 1000.0)
JOINT_RATE, JOINT_STEPS = 0.5, 600
QUAD_STEPS = (1, 2, 5, 10, 20)
ADAPTER_SCALE = 0.5                     # the adapter's matrix A is drawn with this scale
ADAPTER_RATE, ADAPTER_STEPS = 0.5, 200
CENTRES = (-3.0, -1.0, 1.0, 3.0)        # the four episodes of the reader
SIGNS = (1, -1, -1, 1)                  # +1: y = 1 above x1 = 0, -1: y = 1 below it
EP_TRAIN, EP_PROBES, EP_TEST = 60, 48, 400
EP_RATE, EP_STEPS = 0.2, 200
KNOWN_BELOW = 0.20                      # triage: mean loss on the episode's own rows at or below this means already known
ALPHA, MIN_EFFECT, POOLED_BUDGET, BLOCK_DROP = 0.05, 0.15, 0.05, 0.20
GATE_BANK_BUDGET = 0.05                 # the toy gate of the two-capability bank: the pooled and the block bar share this scale
Z_ONE_SIDED = _z_central(0.90)          # 1.644854: the Gaussian value with 0.95 of the area to its left
M_BLOCKS, N_BLOCK, N_REPEATS = 20, 48, 400
MARGINS = (0.05, 0.10, 0.15, 0.20, 0.25)
NULL_OUTCOMES = (0.70, 0.90, 0.95)      # cumulative: both right, both wrong, only the candidate right, otherwise only the incumbent
REGRESSED_OUTCOMES = (0.55, 0.65, 0.70)


def _fmt(x, d=4):
    v = float(x) + 0.0
    s = f"{v:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def _seed_line(name, normal=False):
    seed = bookrng.seed_of(name)
    return f"`{name}` (seed {seed:#018x}, {seed})" + (". A normal() call that finds a held value returns it without drawing." if normal else "")


def _softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def _sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _forward(t, x):
    """The hidden activations and the logit of one input."""
    h1 = [math.tanh(t[OFF["W1"] + 2 * r] * x[0] + t[OFF["W1"] + 2 * r + 1] * x[1] + t[OFF["b1"] + r]) for r in range(H)]
    h2 = []
    for r in range(H):
        a = t[OFF["b2"] + r]
        for c in range(H):
            a += t[OFF["W2"] + H * r + c] * h1[c]
        h2.append(math.tanh(a))
    z = t[OFF["b3"]]
    for r in range(H):
        z += t[OFF["w3"] + r] * h2[r]
    return h1, h2, z


def _logit_gradient(t, x):
    """The gradient of the logit with respect to all 67 numbers, by backpropagation (the logit's own, with no loss in front)."""
    h1, h2, z = _forward(t, x)
    g = [0.0] * P
    d2 = [t[OFF["w3"] + r] * (1.0 - h2[r] * h2[r]) for r in range(H)]
    d1 = []
    for c in range(H):
        s = 0.0
        for r in range(H):
            s += d2[r] * t[OFF["W2"] + H * r + c]
        d1.append(s * (1.0 - h1[c] * h1[c]))
    for r in range(H):
        g[OFF["W1"] + 2 * r] = d1[r] * x[0]
        g[OFF["W1"] + 2 * r + 1] = d1[r] * x[1]
        g[OFF["b1"] + r] = d1[r]
        g[OFF["b2"] + r] = d2[r]
        g[OFF["w3"] + r] = h2[r]
        for c in range(H):
            g[OFF["W2"] + H * r + c] = d2[r] * h1[c]
    g[OFF["b3"]] = 1.0
    return z, g


def _loss_gradient(t, rows):
    """Mean binary cross-entropy softplus(z) - y z over the rows, and its gradient: the rows' gradients added in order, then divided by n."""
    G = [0.0] * P
    total = 0.0
    for x, y in rows:
        z, g = _logit_gradient(t, x)
        total += _softplus(z) - y * z
        c = _sigmoid(z) - y
        for k in range(P):
            G[k] += c * g[k]
    n = len(rows)
    return total / n, [v / n for v in G]


def _loss(t, rows):
    total = 0.0
    for x, y in rows:
        z = _forward(t, x)[2]
        total += _softplus(z) - y * z
    return total / len(rows)


def _correct(t, rows):
    """1.0 for a row the model gets right (the logit above 0 means class 1; a logit of exactly 0 means class 0), else 0.0."""
    return [1.0 if (_forward(t, x)[2] > 0.0) == (y == 1) else 0.0 for x, y in rows]


def _count(t, rows):
    return int(sum(_correct(t, rows)))


def _gd(theta, rows, lr, steps, penalty=None, snaps=()):
    """`steps` steps of t = t - lr * (data gradient + penalty gradient); returns the final numbers and the numbers after the steps in `snaps`."""
    t = list(theta)
    kept = {0: list(t)} if 0 in snaps else {}
    for s in range(1, steps + 1):
        g = _loss_gradient(t, rows)[1]
        if penalty is not None:
            g = [a + b for a, b in zip(g, penalty(t))]
        t = [a - lr * b for a, b in zip(t, g)]
        if s in snaps:
            kept[s] = list(t)
    return t, kept


def _init(nudge=0.0):
    """Both weight matrices from N(0, 0.5^2) in row-major order (W1 then W2, 48 normal() calls); every bias and the output layer are 0."""
    rng = bookrng.generator(INIT)
    t = [0.0] * P
    for k in range(12):
        t[OFF["W1"] + k] = INIT_SCALE * rng.normal()
    for k in range(36):
        t[OFF["W2"] + k] = INIT_SCALE * rng.normal()
    t[0] += nudge
    return t


def _rows(rng, centre, sign, n):
    """One row draws u = uniform(), then z1 = normal() and z0 = normal(); y = 1 if u < 0.5; x = (centre + 0.5 z0, sign * s * (0.3 + 0.7 |z1|)) with s = +1 for y = 1, -1 for y = 0."""
    out = []
    for _ in range(n):
        u = rng.uniform()
        z1 = rng.normal()
        z0 = rng.normal()
        y = 1 if u < 0.5 else 0
        s = 1 if y == 1 else -1
        out.append(([centre + SPREAD * z0, sign * s * (0.3 + 0.7 * abs(z1))], y))
    return out


@lru_cache(maxsize=None)
def _data():
    rng = bookrng.generator(DATA)
    d = {}
    d["A_train"] = _rows(rng, CENTRE_A, 1, N_TRAIN)
    d["B_train"] = _rows(rng, CENTRE_B, -1, N_TRAIN)
    d["A_test"] = _rows(rng, CENTRE_A, 1, N_TEST)
    d["B_test"] = _rows(rng, CENTRE_B, -1, N_TEST)
    d["A_bank"] = _rows(rng, CENTRE_A, 1, N_BANK)
    d["B_bank"] = _rows(rng, CENTRE_B, -1, N_BANK)
    return d


def _fisher_diagonal(t, rows):
    """The exact diagonal of the Fisher information of the model on these inputs: the mean over rows of p (1 - p) g_i^2, g the gradient of the logit."""
    F = [0.0] * P
    for x, _ in rows:
        z, g = _logit_gradient(t, x)
        p = _sigmoid(z)
        w = p * (1.0 - p)
        for k in range(P):
            F[k] += w * g[k] * g[k]
    return [v / len(rows) for v in F]


def _reservoir(n_items, cap):
    """Algorithm R: the first `cap` items fill the buffer, then item i (0-based) replaces slot j = rng.integer(i + 1) when j < cap."""
    rng = bookrng.generator(RESERVOIR)
    keep = list(range(min(cap, n_items)))
    for i in range(cap, n_items):
        j = rng.integer(i + 1)
        if j < cap:
            keep[j] = i
    return keep


def _lora_effective(base, ad):
    """The base numbers with the rank-one product u v^T added to W2 (u = ad[0:6], v = ad[6:12])."""
    t = list(base)
    for r in range(H):
        for c in range(H):
            t[OFF["W2"] + H * r + c] += ad[r] * ad[H + c]
    return t


def _lora_train(base, rows, lr, steps, ad0):
    """Gradient descent on (u, v) only; the gradient of the loss with respect to W2 is projected on u and v, and the base never changes."""
    ad = list(ad0)
    for _ in range(steps):
        g = _loss_gradient(_lora_effective(base, ad), rows)[1]
        gu = [sum(g[OFF["W2"] + H * r + c] * ad[H + c] for c in range(H)) for r in range(H)]
        gv = [sum(g[OFF["W2"] + H * r + c] * ad[r] for r in range(H)) for c in range(H)]
        ad = [a - lr * b for a, b in zip(ad, gu + gv)]
    return ad


def _paired(cand, inc):
    """Wins, losses and ties of two lists of 0/1 outcomes of the same probes in the same order."""
    w = sum(1 for c, i in zip(cand, inc) if c > i)
    l = sum(1 for c, i in zip(cand, inc) if c < i)
    return w, l, len(cand) - w - l


def _sign_p(w, l):
    """One-sided exact sign test over the discordant pairs: P(Binomial(w + l, 1/2) >= w); 1 when there is no discordant pair."""
    n = w + l
    return 1.0 if n == 0 else _tail_ge(n, w)


def _mean(v):
    return sum(v) / len(v)


def _gate(own_c, own_i, blocks, alpha=ALPHA, min_effect=MIN_EFFECT, pooled=POOLED_BUDGET, block=BLOCK_DROP):
    """The four bars of the chapter in the order of brain's gate; returns (decision, cause, numbers).

    own_c, own_i: 0/1 outcomes of the candidate and the incumbent on the episode's own probes.
    blocks: (candidate mean, incumbent mean) of each earlier block of probes."""
    w, l, ties = _paired(own_c, own_i)
    p = _sign_p(w, l)
    effect = _mean(own_c) - _mean(own_i)
    anchor = (_mean([i for _, i in blocks]) - _mean([c for c, _ in blocks])) if blocks else None
    worst = max([i - c for c, i in blocks], default=None)
    if p > alpha:
        cause = "not significant"
    elif effect < min_effect:
        cause = "effect too small"
    elif anchor is not None and anchor > pooled:
        cause = "anchor regressed"
    elif worst is not None and worst > block:
        cause = "block regressed"
    else:
        cause = None
    return ("promote" if cause is None else "reject"), cause, {"wins": w, "losses": l, "ties": ties, "p": p, "effect": effect,
                                                               "anchor": anchor, "worst": worst}


# ------------------------------------------------------------------ the forgetting experiment

def _forget(nudge):
    """Every number of the sections on forgetting and its remedies, from one run (nudge = change of the first initial weight)."""
    d = _data()
    A, B = d["A_train"], d["B_train"]
    At, Bt = d["A_test"], d["B_test"]
    run = {}
    t0 = _init(nudge)
    tA, _ = _gd(t0, A, PRETRAIN_RATE, PRETRAIN_STEPS)
    run["tA"] = tA
    run["t0"] = t0
    run["start"] = (_count(tA, At), _count(tA, Bt))
    run["loss_A"] = _loss(tA, A)
    # forgetting as a function of steps and of the rate
    run["grid"], run["curve"], run["snap"] = {}, {}, {}
    for lr in RATES:
        snaps = set(CHECK_STEPS) | set(QUAD_STEPS) | set(range(0, PROTOCOL_STEPS + 1, CURVE_EVERY))
        t, kept = _gd(tA, B, lr, PROTOCOL_STEPS, snaps=snaps)
        run["grid"][lr] = {s: (_count(kept[s], At), _count(kept[s], Bt)) for s in CHECK_STEPS}
        run["curve"][lr] = [(s, _count(kept[s], At), _count(kept[s], Bt)) for s in range(0, PROTOCOL_STEPS + 1, CURVE_EVERY)]
        if lr == PROTOCOL_RATE:
            run["snap"] = {s: kept[s] for s in QUAD_STEPS}
            run["naive"] = kept[PROTOCOL_STEPS]
    # the exact Fisher diagonal, and how well the quadratic form predicts the rise of the loss on A
    F = _fisher_diagonal(tA, A)
    run["F"] = F
    base_loss = _loss(tA, A)
    quad = {}
    for s in QUAD_STEPS:
        t = run["snap"][s]
        dt = [a - b for a, b in zip(t, tA)]
        full = 0.0
        for x, _ in A:
            z, g = _logit_gradient(tA, x)
            p = _sigmoid(z)
            u = sum(a * b for a, b in zip(g, dt))
            full += p * (1.0 - p) * u * u
        full = 0.5 * full / len(A)
        diag = 0.5 * sum(f * v * v for f, v in zip(F, dt))
        quad[s] = (_loss(t, A) - base_loss, full, diag, math.sqrt(sum(v * v for v in dt)))
    run["quad"] = quad
    # the gradient check at tA, against its criterion
    _, ga = _loss_gradient(tA, A)
    worst = 0.0
    h = 1e-6
    for k in range(P):
        up, dn = list(tA), list(tA)
        up[k] += h
        dn[k] -= h
        fd = (_loss(up, A) - _loss(dn, A)) / (2 * h)
        worst = max(worst, abs(fd - ga[k]))
    run["gradcheck"] = worst
    # replay
    run["replay"], run["replay_theta"] = {}, {}
    for m in BUFFERS:
        keep = _reservoir(N_TRAIN, m)
        rows = list(B) + [A[i] for i in keep]
        t, _ = _gd(tA, rows, PROTOCOL_RATE, PROTOCOL_STEPS)
        run["replay"][m] = (_count(t, At), _count(t, Bt))
        run["replay_theta"][m] = t
    # anchors
    run["l2"], run["l2_theta"] = {}, {}
    for lam in L2_LAMBDAS:
        t, _ = _gd(tA, B, PROTOCOL_RATE, PROTOCOL_STEPS, penalty=lambda t, lam=lam: [lam * (a - b) for a, b in zip(t, tA)])
        run["l2"][lam] = (_count(t, At), _count(t, Bt))
        run["l2_theta"][lam] = t
    run["ewc"], run["ewc_theta"] = {}, {}
    for lam in EWC_LAMBDAS:
        t, _ = _gd(tA, B, PROTOCOL_RATE, PROTOCOL_STEPS, penalty=lambda t, lam=lam: [lam * f * (a - b) for f, a, b in zip(F, t, tA)])
        run["ewc"][lam] = (_count(t, At), _count(t, Bt))
        run["ewc_theta"][lam] = t
    # the adapter: u = 0, v from N(0, 0.5^2) in the order of the generator `continual_learning.adapter`
    rng = bookrng.generator("continual_learning.adapter")
    ad0 = [0.0] * H + [ADAPTER_SCALE * rng.normal() for _ in range(H)]
    ad = _lora_train(tA, B, ADAPTER_RATE, ADAPTER_STEPS, ad0)
    on = _lora_effective(tA, ad)
    run["adapter_on"] = (_count(on, At), _count(on, Bt))
    run["adapter_off"] = (_count(tA, At), _count(tA, Bt))
    # the router is the stated rule applied to every test row of either task: base where x0 < 0, adapter where x0 >= 0
    def _routed(rows):
        return sum((_forward(tA if x[0] < 0.0 else on, x)[2] > 0.0) == (y == 1) for x, y in rows)
    routed, routed_b = _routed(At), _routed(Bt)
    run["adapter_routed"] = (routed, routed_b)
    run["adapter_theta"] = on
    # joint training from the same start, as the reference
    joint, _ = _gd(t0, A + B, JOINT_RATE, JOINT_STEPS)
    run["joint"] = (_count(joint, At), _count(joint, Bt))
    run["joint_theta"] = joint
    # the candidates on the probe bank, outcome by outcome
    cands = {"naive": run["naive"], "replay 20": run["replay_theta"][20], "replay 100": run["replay_theta"][100],
             "L2 0.3": run["l2_theta"][0.3], "L2 1": run["l2_theta"][1.0], "EWC 1000": run["ewc_theta"][1000.0],
             "adapter on": on, "joint": joint}
    run["cands"] = {}
    for name, t in cands.items():
        run["cands"][name] = {"A": _correct(t, d["A_bank"]), "B": _correct(t, d["B_bank"])}
    run["cands"]["incumbent"] = {"A": _correct(tA, d["A_bank"]), "B": _correct(tA, d["B_bank"])}
    return run


def _forget_one(nudge):
    return _forget(nudge)


@lru_cache(maxsize=None)
def _runs():
    args = (0.0,) + stable.NUDGES
    with get_context("fork").Pool(min(4, len(args))) as pool:
        return pool.map(_forget_one, args)


def _st(getter, full, runs=None):
    runs = runs if runs is not None else _runs()
    return stable.text(getter(runs[0]), [getter(r) for r in runs[1:]], full)


def _n0(x):
    return f"{x:.0f}"


def _n4(x):
    return f"{x:.4f}"


def _n3(x):
    return f"{x:.3f}"


def _n4(x):
    return f"{x:.4f}"


def _n5(x):
    return f"{x:.5f}"


def _sci(x):
    return f"{x:.2e}"


def _pn(i):
    """The name of number i of the flat vector."""
    for name in ("W1", "b1", "W2", "b2", "w3", "b3"):
        lo = OFF[name]
        hi = {"W1": 12, "b1": 18, "W2": 54, "b2": 60, "w3": 66, "b3": 67}[name]
        if lo <= i < hi:
            k = i - lo
            if name == "W1":
                return f"W1[{k // 2},{k % 2}]"
            if name == "W2":
                return f"W2[{k // H},{k % H}]"
            if name == "b3":
                return "b3"
            return f"{name}[{k}]"
    raise ValueError(i)


# ------------------------------------------------------------------ blocks of the chapter

def world_rule():
    d = _data()
    t0 = _init()
    out = []
    out.append(f"Network: x = (x0, x1), h1 = tanh(W1 x + b1) with 6 units, h2 = tanh(W2 h1 + b2) with 6 units, z = w3 . h2 + b3, "
               f"p = sigmoid(z); the loss of a row is softplus(z) - y z, the mean over the rows is the loss, and a row is classified as 1 when z > 0 "
               f"(a logit of exactly 0 is class 0). The 67 numbers are stored in the order W1 (row-major, 12), b1 (6), W2 (row-major, 36), "
               f"b2 (6), w3 (6), b3 (1); a gradient is the gradient of the mean loss, the rows' gradients added in row order and then divided by n.")
    out.append("")
    out.append(f"Initialization: generator {_seed_line(INIT, True)} The 12 numbers of W1 are 0.5 times 12 successive `normal()` calls, "
               f"then the 36 numbers of W2 are 0.5 times 36 more, both row-major (48 calls, 24 pairs, nothing is left held); every bias and "
               f"w3 and b3 are 0. Because w3 is 0 the starting network outputs the logit 0 on every input, so it answers class 0 everywhere.")
    out.append("")
    rows = []
    for name, shape, lo, hi in (("W1", "6 x 2", 0, 12), ("W2", "6 x 6", 18, 54)):
        vals = t0[lo:hi]
        width = 2 if name == "W1" else 6
        for r in range(len(vals) // width):
            rows.append([f"{name} row {r}"] + [_fmt(v, 4) for v in vals[r * width:(r + 1) * width]] + [""] * (6 - width))
    out.append(_table(["initial numbers", "c0", "c1", "c2", "c3", "c4", "c5"], rows))
    out.append("")
    out.append(f"Data: generator {_seed_line(DATA, True)} One generator for everything, in this order: the 100 training rows of A, the 100 of B, "
               f"the 400 test rows of A, the 400 of B, then the 100 bank rows of A and the 100 of B (the bank of the probe-bank section). "
               f"A row draws `u = uniform()`, then `z1 = normal()`, then `z0 = normal()` (the second call returns the held value, so a row "
               f"consumes three raw values); y = 1 if u < 0.5, s = +1 if y = 1 and -1 otherwise, x0 = centre + 0.5 z0 and "
               f"x1 = sign * s * (0.3 + 0.7 |z1|). Task A has centre -1.5 and sign +1 (y = 1 above the line x1 = 0), task B has centre +1.5 "
               f"and sign -1 (y = 1 below it): the same question with the opposite answer, in another cluster.")
    out.append("")
    rows = []
    for name in ("A_train", "B_train"):
        for k in range(3):
            (x, y) = d[name][k]
            rows.append([f"{name[0]} row {k}", _fmt(x[0], 4), _fmt(x[1], 4), str(y)])
    out.append(_table(["row", "x0", "x1", "y"], rows))
    out.append("")
    out.append(f"Counts of y = 1: A train {sum(y for _, y in d['A_train'])} of 100, B train {sum(y for _, y in d['B_train'])} of 100, "
               f"A test {sum(y for _, y in d['A_test'])} of 400, B test {sum(y for _, y in d['B_test'])} of 400.")
    return "\n".join(out)


def pretrained():
    runs = _runs()
    tA = runs[0]["tA"]
    out = [f"Training on A from the initial numbers above: full-batch gradient descent, step size {PRETRAIN_RATE}, {PRETRAIN_STEPS} steps, "
           f"the mean loss over the 100 rows of A. Numbers of the run are printed with the digits that the run and its four nudged copies "
           f"share (the rule of `stable.py`); a count is out of 400 test rows. The analytic gradient agrees with central differences "
           f"(step 1e-6) on all 67 numbers, at the trained numbers: the largest absolute difference is "
           + ("below 1e-6, the criterion" if all(r["gradcheck"] < 1e-6 for r in runs) else "NOT below the criterion 1e-6") + ".", ""]
    rows = []
    for name, lo, hi, width in (("W1", 0, 12, 2), ("b1", 12, 18, 6), ("W2", 18, 54, 6), ("b2", 54, 60, 6), ("w3", 60, 66, 6), ("b3", 66, 67, 1)):
        for r in range((hi - lo) // width):
            vals = [_st(lambda x, k=lo + r * width + c: x["tA"][k], lambda v: _fmt(v, 3)) for c in range(width)]
            label = name if hi - lo == width else f"{name} row {r}"
            rows.append([label] + vals + [""] * (6 - width))
    out.append(_table(["trained on A", "c0", "c1", "c2", "c3", "c4", "c5"], rows))
    out.append("")
    out.append(_table(["quantity", "value"], [
        ["mean loss on the 100 rows of A", _st(lambda r: r["loss_A"], _n4)],
        ["correct on A test (of 400)", _st(lambda r: r["start"][0], _n0)],
        ["correct on B test (of 400)", _st(lambda r: r["start"][1], _n0)],
    ]))
    return "\n".join(out)


def forgetting_grid():
    rows = []
    for lr in RATES:
        for s in CHECK_STEPS:
            a = _st(lambda r, lr=lr, s=s: r["grid"][lr][s][0], _n0)
            b = _st(lambda r, lr=lr, s=s: r["grid"][lr][s][1], _n0)
            rows.append([str(lr) if s == 0 else "", str(s), a, b])
    out = [f"Starting from the numbers trained on A, gradient descent on the 100 rows of B alone (mean loss over them), with the step size and for the number of steps shown. "
           f"Counts are correct classifications out of 400 test rows of each task; step 0 is the network trained on A.", "",
           _table(["step size", "steps on B", "A correct", "B correct"], rows)]
    return "\n".join(out)


def anchor_prediction():
    out = [f"The run with step size {PROTOCOL_RATE} on B alone. F is the exact Fisher information of the network trained on A, taken over the 100 rows of A: "
           f"F_ij = (1/100) sum over the rows of p (1 - p) g_i g_j with g the gradient of the logit; 'diagonal' keeps only i = j. With "
           f"d = (numbers after the steps) - (numbers trained on A), the quadratic prediction of the rise of the mean loss on the 100 rows of A is "
           f"(1/2) d^T F d (full) or (1/2) sum_i F_ii d_i^2 (diagonal), against the measured rise.", ""]
    rows = []
    for s in QUAD_STEPS:
        rows.append([str(s), _st(lambda r, s=s: r["quad"][s][3], _n3), _st(lambda r, s=s: r["quad"][s][0], _n3),
                     _st(lambda r, s=s: r["quad"][s][1], _n3), _st(lambda r, s=s: r["quad"][s][2], _n3)])
    out.append(_table(["steps on B", "length of d", "measured rise", "full prediction", "diagonal prediction"], rows))
    return "\n".join(out)


def fisher_diagonal():
    runs = _runs()
    F = runs[0]["F"]
    order = sorted(range(P), key=lambda i: (-F[i], i))
    out = [f"The exact Fisher diagonal of the network trained on A over the 100 rows of A (true Fisher: the label is averaged out under the model's own "
           f"probability, which for a logistic output gives p (1 - p) times the squared gradient of the logit; per row, divisor 100). "
           f"Its sum and its largest entries; entries are printed with the digits all five runs share.", ""]
    rows = [["sum of the 67 entries", _st(lambda r: sum(r["F"]), _n4)], ["largest entry", _st(lambda r: max(r["F"]), _n4)],
            ["number of entries below 1e-4", _st(lambda r: sum(1 for v in r["F"] if v < 1e-4), _n0)]]
    for i in order[:5]:
        rows.append([f"entry {_pn(i)}", _st(lambda r, i=i: r["F"][i], _n4)])
    out.append(_table(["quantity", "value"], rows))
    return "\n".join(out)


def replay_table():
    keep10 = _reservoir(N_TRAIN, 10)
    out = [f"Replay: the loss of every step is the mean over the 100 rows of B and the m buffered rows of A (all weighted alike), "
           f"step size {PROTOCOL_RATE}, {PROTOCOL_STEPS} steps, from the numbers trained on A. The buffer is Algorithm R over the 100 rows of A in their order "
           f"with generator {_seed_line(RESERVOIR)}. Each m starts the generator afresh; the first m rows fill the buffer and row i (0-based) replaces slot "
           f"`rng.integer(i + 1)` when that is below m, one draw for each i = m .. 99. For m = 10 the buffer holds the A rows "
           f"{', '.join(str(i) for i in keep10)} (slot order). A step lists the 100 rows of B in their order and then the buffer rows in slot order, and sums their gradients in that order.", ""]
    rows = []
    for m in BUFFERS:
        rows.append([str(m), f"{m / N_TRAIN:.2f}", str(N_TRAIN + m), _st(lambda r, m=m: r["replay"][m][0], _n0), _st(lambda r, m=m: r["replay"][m][1], _n0)])
    out.append(_table(["buffer m", "old rows per new row", "rows per step", "A correct", "B correct"], rows))
    return "\n".join(out)


def anchor_table():
    out = [f"An anchor on the old numbers, the same protocol: step size {PROTOCOL_RATE}, {PROTOCOL_STEPS} steps on the 100 rows of B. "
           f"The L2 anchor adds (lambda/2) sum_i (t_i - tA_i)^2 to the mean loss, so its gradient is lambda (t_i - tA_i), for all 67 numbers. "
           f"EWC adds (lambda/2) sum_i F_i (t_i - tA_i)^2 with the exact Fisher diagonal F of the previous block, gradient lambda F_i (t_i - tA_i). "
           f"For a step to be stable the stiffest direction needs step size times stiffness below 2: here {PROTOCOL_RATE} * lambda <= 0.6 for the L2 anchor, and "
           f"{PROTOCOL_RATE} * lambda * max F = " + _st(lambda r: PROTOCOL_RATE * 1000.0 * max(r["F"]), _n3) + " for EWC at lambda = 1000.", ""]
    rows = []
    for lam in L2_LAMBDAS:
        rows.append(["L2", f"{lam:g}", _st(lambda r, lam=lam: r["l2"][lam][0], _n0), _st(lambda r, lam=lam: r["l2"][lam][1], _n0)])
    for lam in EWC_LAMBDAS:
        rows.append(["EWC", f"{lam:g}", _st(lambda r, lam=lam: r["ewc"][lam][0], _n0), _st(lambda r, lam=lam: r["ewc"][lam][1], _n0)])
    out.append(_table(["anchor", "lambda", "A correct", "B correct"], rows))
    return "\n".join(out)


def adapter_table():
    rng = bookrng.generator("continual_learning.adapter")
    v0 = [ADAPTER_SCALE * rng.normal() for _ in range(H)]
    out = [f"A rank-one adapter on W2: the matrix used is W2 + u v^T with u, v in R^6 (12 trained numbers; the 67 of the base are frozen). u starts at 0 and v from "
           f"{ADAPTER_SCALE} times 6 `normal()` calls of generator {_seed_line('continual_learning.adapter', True)} (v starts at "
           + ", ".join(_fmt(v, 4) for v in v0) + f"). Gradient descent on the 100 rows of B only, step size {ADAPTER_RATE}, {ADAPTER_STEPS} steps; the gradient "
           f"of u is G_W2 v and that of v is G_W2^T u, with G_W2 the gradient of the loss with respect to the matrix in use. "
           f"'Routed' serves the base where x0 < 0 and the adapter where x0 >= 0, by the row's own x0 and for every test row of both tasks (the router does not see the task label, so a row of A with x0 >= 0 reaches the adapter).", ""]
    rows = [["adapter off (the base alone)", _st(lambda r: r["adapter_off"][0], _n0), _st(lambda r: r["adapter_off"][1], _n0)],
            ["adapter on, always", _st(lambda r: r["adapter_on"][0], _n0), _st(lambda r: r["adapter_on"][1], _n0)]]
    # routed accuracy: the rule is applied to every test row, so a row of A with x0 >= 0 goes to the adapter and is served wrongly
    rows.append(["routed by the sign of x0 (every test row of either task)", _st(lambda r: r["adapter_routed"][0], _n0),
                 _st(lambda r: r["adapter_routed"][1], _n0)])
    out.append(_table(["served", "A correct", "B correct"], rows))
    return "\n".join(out)


def remedy_summary():
    cost_rows = lambda m: N_TRAIN + m
    rows = []
    rows.append(["train on B alone", str(N_TRAIN), "0", _st(lambda r: r["grid"][PROTOCOL_RATE][200][0], _n0), _st(lambda r: r["grid"][PROTOCOL_RATE][200][1], _n0)])
    rows.append(["replay, m = 20", str(cost_rows(20)), "60 (20 rows of 3 numbers)", _st(lambda r: r["replay"][20][0], _n0), _st(lambda r: r["replay"][20][1], _n0)])
    rows.append(["replay, m = 100", str(cost_rows(100)), "300", _st(lambda r: r["replay"][100][0], _n0), _st(lambda r: r["replay"][100][1], _n0)])
    rows.append(["L2 anchor, lambda = 0.3", str(N_TRAIN), "67 (a copy of the old numbers)", _st(lambda r: r["l2"][0.3][0], _n0), _st(lambda r: r["l2"][0.3][1], _n0)])
    rows.append(["EWC, lambda = 1000", str(N_TRAIN), "134 (old numbers and F)", _st(lambda r: r["ewc"][1000.0][0], _n0), _st(lambda r: r["ewc"][1000.0][1], _n0)])
    rows.append(["adapter, always on", str(N_TRAIN), "12 (and the 67 kept)", _st(lambda r: r["adapter_on"][0], _n0), _st(lambda r: r["adapter_on"][1], _n0)])
    rows.append(["joint training from the start (600 steps on 200 rows)", "200", "100 rows of A", _st(lambda r: r["joint"][0], _n0), _st(lambda r: r["joint"][1], _n0)])
    out = [f"The remedies of the last blocks side by side (step size {PROTOCOL_RATE}, {PROTOCOL_STEPS} steps on B, from the numbers trained on A; the joint row is the "
           f"reference trained from the initial numbers with step size {JOINT_RATE}). 'Rows per step' is the number of rows whose gradient is computed in every step; "
           f"'kept besides the new numbers' counts the numbers that must be stored to apply the remedy.", "",
           _table(["remedy", "rows per step", "kept besides the new numbers", "A correct", "B correct"], rows)]
    return "\n".join(out)


def _bank_text():
    d = _data()
    lines = []
    for cap, key in (("A", "A_bank"), ("B", "B_bank")):
        for k, (x, y) in enumerate(d[key]):
            lines.append(f"{cap}|{k}|{x[0]:.6f}|{x[1]:.6f}|{y}")
    return "\n".join(lines)


def probe_bank():
    text = _bank_text()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    lines = text.split("\n")
    first = lines[0].rsplit("|", 1)
    lines[0] = first[0] + "|" + str(1 - int(first[1]))
    changed = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    d = _data()
    out = [f"The probe bank of the toy has two capabilities and 100 probes each, frozen before any candidate is trained: capability A is the 100 bank rows of A "
           f"and capability B the 100 bank rows of B (the last 200 draws of the data generator, after the 1,000 rows above), disjoint from the training and test rows by "
           f"construction (separate draws). A probe is `capability|index|x0|x1|y` with x0 and x1 written with 6 decimals; the identity of the bank is the SHA-256 of "
           f"the 200 lines joined by a line feed (no final one), in the order A then B.", "",
           _table(["item", "value"], [
               ["probes in capability A / B", f"{len(d['A_bank'])} / {len(d['B_bank'])}"],
               ["y = 1 in A / in B", f"{sum(y for _, y in d['A_bank'])} / {sum(y for _, y in d['B_bank'])}"],
               ["first probe", lines_first(text)],
               ["identity of the bank (first 16 hex digits of SHA-256)", digest[:16]],
               ["identity after flipping the label of probe A|0", changed[:16]],
           ])]
    return "\n".join(out)


def lines_first(text):
    return "`" + text.split("\n")[0] + "`"


def gate_on_candidates():
    names = ("naive", "replay 20", "replay 100", "L2 0.3", "L2 1", "EWC 1000", "adapter on", "joint")
    # the gate of the chapter: capability B is the target (the candidate's own probes), capability A the single earlier block
    def stats(run, name):
        c, i = run["cands"][name], run["cands"]["incumbent"]
        dec, cause, nums = _gate(c["B"], i["B"], [(_mean(c["A"]), _mean(i["A"]))], pooled=GATE_BANK_BUDGET)
        return dec, cause, nums
    rows = []
    for name in names:
        rows.append([name,
                     _st(lambda r, n=name: stats(r, n)[2]["wins"], _n0), _st(lambda r, n=name: stats(r, n)[2]["losses"], _n0),
                     _st(lambda r, n=name: stats(r, n)[2]["p"], lambda v: f"{v:.2e}" if v < 1e-3 else f"{v:.4f}"),
                     _st(lambda r, n=name: stats(r, n)[2]["effect"], _n3),
                     _st(lambda r, n=name: stats(r, n)[2]["anchor"], _n3),
                     _st_text(lambda r, n=name: stats(r, n)[0] + ("" if stats(r, n)[1] is None else ": " + stats(r, n)[1])),
                     ])
    out = [f"Each candidate of the previous blocks is compared with the incumbent (the network trained on A) on the probe bank. Per probe the outcome is 1 (right) or 0; "
           f"on the 100 probes of capability B (the target) a win is a probe the candidate gets right and the incumbent wrong, a loss the reverse, and ties (both right or both wrong) are "
           f"left out of the sign test. p is the one-sided exact P(Binomial(wins + losses, 1/2) >= wins), 1 when there is no discordant probe. "
           f"'Effect' is mean(candidate) - mean(incumbent) on B and 'anchor drop' is mean(incumbent) - mean(candidate) on the 100 probes of capability A. "
           f"The four bars in this order: p <= {ALPHA}, effect >= {MIN_EFFECT}, anchor drop <= {GATE_BANK_BUDGET} (the pooled bar; with one earlier block it is that block's drop) "
           f"and the same drop <= {BLOCK_DROP} (the block bar); the first bar that fails names the cause. The candidates are the ones of the remedy table: "
           f"'naive' is B alone (step size {PROTOCOL_RATE}, {PROTOCOL_STEPS} steps), 'replay m', 'L2 lambda', 'EWC lambda' as before, 'adapter on' the rank-one adapter "
           f"always on and 'joint' the network trained on both from the start.", "",
           _table(["candidate", "wins on B", "losses on B", "p", "effect on B", "anchor drop on A", "decision"], rows,
                  "lrrrrrl")]
    return "\n".join(out)


def _st_text(getter):
    runs = _runs()
    vals = [getter(r) for r in runs]
    return vals[0] if len(set(vals)) == 1 else "unstable"


def pooled_blindness():
    rows = []
    for d in (0.30, 1.00):
        for k in (1, 2, 5, 10, 15, 20, 50, 100):
            pooled = d / k
            rows.append([f"{d:.2f}", str(k), f"{pooled:.4f}", "yes" if pooled > 0.02 else "no", "yes" if d > 0.20 else "no"])
    out = ["One earlier block drops by d, the other k - 1 earlier blocks hold. The pooled mean of the blocks drops by d/k. 'Pooled bar fires' is d/k > 0.02 (the "
           "default budget of brain's gate); 'block bar fires' is d > 0.20 (the pre-registered per-block bar). Exact arithmetic; the pooled bar stops seeing a drop of "
           "d once k >= d / 0.02, which is 15 blocks for d = 0.30 and 50 for d = 1.00.", "",
           _table(["drop d", "earlier blocks k", "pooled drop d/k", "pooled bar fires", "block bar fires"], rows)]
    return "\n".join(out)


def _outcome(u, cum):
    """0 both right, 1 both wrong, +... returns (win, loss) of one probe: win if only the candidate is right, loss if only the incumbent is."""
    if u < cum[0] or u < cum[1]:
        return (0, 0)
    if u < cum[2]:
        return (1, 0)
    return (0, 1)


@lru_cache(maxsize=None)
def _block_simulation():
    rng = bookrng.generator(BLOCKS)
    pops = {}
    for pop in ("equal", "regressed"):
        reps = []
        for _ in range(N_REPEATS):
            blocks = []
            for b in range(M_BLOCKS):
                cum = REGRESSED_OUTCOMES if (pop == "regressed" and b == 0) else NULL_OUTCOMES
                w = l = 0
                for _ in range(N_BLOCK):
                    dw, dl = _outcome(rng.uniform(), cum)
                    w += dw
                    l += dl
                blocks.append((w, l))
            reps.append(blocks)
        pops[pop] = reps
    return pops


def _lcb(w, l, n=N_BLOCK):
    """Lower confidence bound of the mean paired difference: mean - z * sqrt(s^2 / n), s^2 the unbiased variance of the differences (divisor n - 1)."""
    m = (w - l) / n
    s2 = ((w + l) - n * m * m) / (n - 1)
    return m - Z_ONE_SIDED * math.sqrt(max(s2, 0.0) / n)


def many_blocks():
    pops = _block_simulation()
    z_bonf = _z_central(1.0 - 2 * ALPHA / M_BLOCKS)

    def blocked(rule, reps):
        return sum(1 for blocks in reps if rule(blocks)) / len(reps)

    def flag_p(alpha):
        return lambda blocks: any(_sign_p(l, w) <= alpha for (w, l) in blocks)

    def fixed(blocks):
        return any((l - w) / N_BLOCK > BLOCK_DROP for (w, l) in blocks)

    def noninferior(delta):
        return lambda blocks: any(_lcb(w, l) <= -delta for (w, l) in blocks)

    rules = [(f"any block significantly worse at {ALPHA} (sign test, one-sided)", flag_p(ALPHA)),
             (f"the same at {ALPHA}/{M_BLOCKS} = {ALPHA / M_BLOCKS:.4f} (Bonferroni)", flag_p(ALPHA / M_BLOCKS)),
             (f"fixed bar: some block drops by more than {BLOCK_DROP}", fixed)]
    for delta in MARGINS:
        rules.append((f"non-inferiority: some block's lower bound at or below -{delta:.2f}", noninferior(delta)))
    rows = [[name, f"{blocked(rule, pops['equal']):.4f}", f"{blocked(rule, pops['regressed']):.4f}"] for name, rule in rules]
    out = [f"{M_BLOCKS} earlier blocks of {N_BLOCK} probes each. Generator {_seed_line(BLOCKS)} draws everything: the 400 repeats of the first population "
           f"(candidate equal to the incumbent) are drawn first, then the 400 of the second (block 0 truly worse); a repeat draws its 20 blocks in order and a block its 48 "
           f"probes in order, one `uniform()` per probe. A probe is 'both right' if u < 0.70, 'both wrong' if u < 0.90, 'only the candidate right' (a win) if u < 0.95 "
           f"and otherwise 'only the incumbent right' (a loss); in the second population the thresholds of block 0 are 0.55, 0.65 and 0.70 "
           f"(wins 0.05, losses 0.30, a true mean drop of 0.25). A block's mean difference is (wins - losses)/48. The lower bound of the "
           f"non-inferiority rows is mean - {Z_ONE_SIDED:.6f} * sqrt(s^2/48) with s^2 the unbiased variance of the 48 differences (divisor 47) and the Gaussian value "
           f"{Z_ONE_SIDED:.6f}; the sign tests are one-sided exact on the discordant probes of a block, and the share is of the 400 repeats in which the rule blocks the candidate. "
           f"For comparison, independent continuous tests would give 1 - 0.95^20 = {1 - 0.95 ** 20:.4f} and 1 - (1 - {ALPHA / M_BLOCKS:.4f})^20 = {1 - (1 - ALPHA / M_BLOCKS) ** 20:.4f}.", "",
           _table(["rule that blocks the candidate", "share blocked, equal candidate", "share blocked, block 0 worse by 0.25"], rows)]
    return "\n".join(out)


def margin_example():
    rows = []
    for lost in (0, 1, 2, 4, 6, 8, 10):
        w, l = 0, lost
        m = (w - l) / N_BLOCK
        s2 = ((w + l) - N_BLOCK * m * m) / (N_BLOCK - 1)
        se = math.sqrt(s2 / N_BLOCK)
        lcb = _lcb(w, l)
        rows.append([str(lost), f"{m:.4f}", f"{se:.4f}", f"{lcb:.4f}"] + ["yes" if lcb > -d else "no" for d in (0.10, 0.20)] + ["yes" if -m <= 0.20 else "no"])
    out = [f"One block of {N_BLOCK} probes where the candidate wins none and loses 'lost' of them (so 'lost' discordant probes and the rest ties). Mean difference m = -lost/48, "
           f"s^2 = (lost - 48 m^2)/47, standard error sqrt(s^2/48), one-sided lower bound m - {Z_ONE_SIDED:.6f} * SE. A candidate is non-inferior within the margin delta when the "
           f"lower bound is above -delta; the last column is brain's fixed bar (a drop of at most 0.20, no inference).", "",
           _table(["probes lost", "mean difference", "standard error", "lower bound", "non-inferior, delta 0.10", "non-inferior, delta 0.20", "fixed bar passes"], rows)]
    return "\n".join(out)


# ------------------------------------------------------------------ the continual reader

@lru_cache(maxsize=None)
def _stream():
    rng = bookrng.generator(STREAM)
    eps = []
    for k in range(len(CENTRES)):
        eps.append({"train": _rows(rng, CENTRES[k], SIGNS[k], EP_TRAIN), "probes": _rows(rng, CENTRES[k], SIGNS[k], EP_PROBES),
                    "test": _rows(rng, CENTRES[k], SIGNS[k], EP_TEST)})
    return eps


def _replay_rows(bank_rows, rng, ratio):
    """Rows drawn without replacement from the promoted episodes' rows (in promotion order): a partial Fisher-Yates, one integer(len - i) per row drawn."""
    pool = [r for rows in bank_rows for r in rows]
    m = round(EP_TRAIN * ratio)
    idx = list(range(len(pool)))
    for i in range(min(m, len(pool))):
        j = i + rng.integer(len(pool) - i)
        idx[i], idx[j] = idx[j], idx[i]
    return [pool[k] for k in idx[:min(m, len(pool))]]


def _reader(arm, nudge):
    """One arm of the reader over the four episodes (see ARMS); the last arm raises the pooled anchor budget from 0.05 to 0.10."""
    eps = _stream()
    rng = bookrng.generator(REPLAY)
    inc = _init(nudge)
    bank, promoted_at, log = [], {}, []
    use_replay, use_gate = "replay" in arm, "gate" in arm
    pooled = 0.10 if arm.endswith("0.10") else POOLED_BUDGET
    for k, ep in enumerate(eps):
        known = _loss(inc, ep["train"])
        if known <= KNOWN_BELOW:
            log.append(("already known", None))
            continue
        rows = list(ep["train"])
        if use_replay and bank:
            rows += _replay_rows([eps[j]["train"] for j in bank], rng, 1.0)
        cand, _ = _gd(inc, rows, EP_RATE, EP_STEPS)
        own_c, own_i = _correct(cand, ep["probes"]), _correct(inc, ep["probes"])
        blocks = [(_mean(_correct(cand, eps[j]["probes"])), _mean(_correct(inc, eps[j]["probes"]))) for j in bank]
        dec, cause, nums = _gate(own_c, own_i, blocks, pooled=pooled)
        ok = True if not use_gate else dec == "promote"
        label = "promoted" if ok else "rejected: " + cause
        if not use_gate and dec != "promote":
            label = "promoted (the gate would say: " + cause + ")"
        log.append((label, nums))
        if ok:
            inc = cand
            bank.append(k)
            promoted_at[k] = _count(inc, ep["test"])
    final = [_count(inc, ep["test"]) for ep in eps]
    bwt_set = bank[:-1]
    bwt = (sum(final[j] - promoted_at[j] for j in bwt_set) / (EP_TEST * len(bwt_set))) if bwt_set else None
    return {"log": log, "bank": bank, "final": final, "bwt": bwt}


def _reader_one(args):
    return _reader(*args)


ARMS = ("naive", "replay", "gate", "replay+gate", "replay+gate, pooled budget 0.10")


@lru_cache(maxsize=None)
def _reader_runs():
    jobs = [(arm, nudge) for arm in ARMS for nudge in (0.0,) + stable.NUDGES]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_reader_one, jobs)
    out, i = {}, 0
    for arm in ARMS:
        out[arm] = res[i:i + 1 + len(stable.NUDGES)]
        i += 1 + len(stable.NUDGES)
    return out


def _joint_reader(nudge=0.0):
    eps = _stream()
    rows = [r for ep in eps for r in ep["train"]]
    t, _ = _gd(_init(nudge), rows, JOINT_RATE, 800)
    return [_count(t, ep["test"]) for ep in eps]


def reader_world():
    eps = _stream()
    out = [f"Four episodes in a fixed order, from generator {_seed_line(STREAM, True)} One generator, episode after episode; an episode draws its {EP_TRAIN} training rows, then its "
           f"{EP_PROBES} probes, then its {EP_TEST} test rows, each row as in the first block (a row draws `uniform()`, `normal()`, `normal()`). Episode k has the centre "
           f"{', '.join(f'{c:+.1f}' for c in CENTRES)} and the sign {', '.join(f'{s:+d}' for s in SIGNS)} for k = 0 .. 3 (so the answer above x1 = 0 is class 1 in episodes 0 and 3 and "
           f"class 0 in episodes 1 and 2). The starting incumbent is the network of the first block with the same initial numbers (generator `continual_learning.init`), answering class 0 everywhere.", ""]
    rows = []
    for k, ep in enumerate(eps):
        rows.append([str(k), f"{CENTRES[k]:+.1f}", f"{SIGNS[k]:+d}", str(sum(y for _, y in ep["train"])), str(sum(y for _, y in ep["probes"])), str(sum(y for _, y in ep["test"]))])
    out.append(_table(["episode", "centre", "sign", "y = 1 in train (of 60)", "y = 1 in probes (of 48)", "y = 1 in test (of 400)"], rows))
    return "\n".join(out)


def reader_arms():
    runs = _reader_runs()
    rows = []
    for arm in ARMS:
        rs = runs[arm]
        for k in range(len(CENTRES)):
            labels = {r["log"][k][0] for r in rs}
            lab = rs[0]["log"][k][0] if len(labels) == 1 else "unstable"
            rows.append([arm if k == 0 else "", str(k), lab])
    out = [f"The five arms on the stream of the previous block. Every arm starts from the same initial numbers and restarts the generator {_seed_line(REPLAY)}; a candidate is trained from the incumbent for {EP_STEPS} steps of size {EP_RATE} on the episode's own rows and, in the replay arms, "
           f"as many rows drawn without replacement from the rows of the promoted episodes (ratio 1.0, 60 rows: a partial Fisher-Yates, `rng.integer(len - i)` for i = 0 .. 59, "
           f"in the order drawn; a pool of exactly 60 still does the shuffle, and a pool is never smaller since an episode is held only after it is promoted). Before training, triage drops an episode whose mean loss under the incumbent on its own rows is at most {KNOWN_BELOW} ('already known'). "
           f"The gate is the one of the previous block: the sign test at {ALPHA} on the 48 own probes, effect at least {MIN_EFFECT}, the pooled anchor drop over all promoted episodes' "
           f"probes at most {POOLED_BUDGET} and the worst block drop at most {BLOCK_DROP}. An arm without the gate promotes whatever it trains. The outcome of each episode:", "",
           _table(["arm", "episode", "outcome"], rows, "lll"), "",
           "Correct classifications on the 400 test rows of each episode after the last episode, the number of promoted episodes, and backward transfer "
           "(the mean over the promoted episodes except the last of the test accuracy now minus the accuracy at its own promotion, in fractions of 400; "
           "none when fewer than two were promoted):", ""]
    rows = []
    for arm in ARMS:
        rs = runs[arm]
        finals = []
        for k in range(len(CENTRES)):
            finals.append(stable.text(rs[0]["final"][k], [r["final"][k] for r in rs[1:]], _n0))
        npro = {len(r["bank"]) for r in rs}
        bwts = [r["bwt"] for r in rs]
        if all(b is None for b in bwts):
            bwt = "none"
        elif any(b is None for b in bwts):
            bwt = "unstable"
        else:
            bwt = stable.text(bwts[0], bwts[1:], _n4)
        rows.append([arm] + finals + [str(npro.pop()) if len(npro) == 1 else "unstable", bwt])
    jruns = [_joint_reader(n) for n in (0.0,) + stable.NUDGES]
    joint = []
    for k in range(len(CENTRES)):
        lo, hi = min(r[k] for r in jruns), max(r[k] for r in jruns)
        joint.append(str(lo) if lo == hi else f"{lo} to {hi}")
    rows.append(["joint on all four (800 steps of size %s, 240 rows of episodes 0 to 3 in order), range over the run and its four nudged copies" % JOINT_RATE] + joint + ["4", "-"])
    out.append(_table(["arm", "episode 0", "episode 1", "episode 2", "episode 3", "promoted", "backward transfer"], rows))
    return "\n".join(out)


def audit_cost():
    rows = []
    per_tick = 12
    rotating = 9
    for n in (10, 100, 1000, 10000):
        full = 2 * N_BLOCK * n
        bounded = 2 * 192
        latency = -(-n // rotating)
        rows.append([str(n), str(full), str(bounded), str(latency)])
    out = ["Probe decodes spent on re-checking the bank in one episode, counted for both arms of the gate (the candidate and the incumbent). 'Full' re-checks every earlier "
           "episode's 48 probes: 2 x 48 x N decodes for N promoted episodes. 'Bounded' is the schedule of brain's `audit::schedule` with its default budget: 192 decodes per arm a tick, "
           "blocks of 16 probes, so 12 blocks a tick, of which 3 go to the canary and 9 to a strict round-robin; every episode is revisited within ceil(N / 9) ticks.", "",
           _table(["episodes in the bank N", "full re-check, decodes per episode read", "bounded re-check, decodes per episode read", "ticks to revisit every episode"], rows)]
    return "\n".join(out)


# ------------------------------------------------------------------ for the figures

def _curves():
    """For the forgetting figure: {step size: [(steps, A correct, B correct), ...]} from the unnudged run."""
    return _runs()[0]["curve"]
