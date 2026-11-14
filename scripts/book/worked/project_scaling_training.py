# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the project "Scaling a Training Run".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: project_scaling_training.<name> -->` markers. Helpers with a leading underscore, or that take arguments,
are shared with the chapter's figures so a picture and the table beside it cannot disagree.

The system is a six-layer ReLU network trained with Adam in 64-bit floating point. Everything it computes uses
addition, subtraction, multiplication, division and square root (each correctly rounded) and a comparison, on draws
that are exact (`uniform()` of the book's generator), so another implementation that follows the stated order of the
additions reproduces every digit of a run that does not depend on the number of ranks. The numbers that come out of
training go through the stable-digits rule of `stable`. The FNV-1a digest and the epoch order are imported from
`checkpoints_and_recovery` and `training_data`, and the ring all-reduce from `parallelism` (never copied).

Swedish Embedded AB implements solutions for moving training runs from one device to many without changing what they
learn. If your team needs expertise in data-parallel, memory-lean and resumable training then you can procure our
services by sending an email to info@swedishembedded.com.
"""
import json
import math
import struct

import bookrng
import checkpoints_and_recovery as cr
import number_formats as nf
import parallelism as par
import stable
import training_data as td

_table = nf._table

# ------------------------------------------------------------------ the task, the network and its initialisation

N_EX = 40                                  # examples; an epoch is 40 draws
BATCH = 10                                 # draws of one step
SIZES = [(4, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 1)]    # (inputs, outputs) of each layer
LAYERS = len(SIZES)
RANK, SCALE = 2, 1.0                       # adapter rank r and scale s = alpha / r with alpha = r
ADAPTED = (1, 2, 3, 4)                     # the layers that carry an adapter
LR, B1, B2, EPS = 0.01, 0.9, 0.999, 1e-8
PRE_STEPS, FT_STEPS = 120, 60              # steps of the base run and of the adapter run
EVERY = 10                                 # steps between two checkpoints of the adapter run
CRASH_AT = 37                              # the adapter run is killed after this many updates

OFF = []                                   # offset of each layer in the flat parameter vector: W row-major, then b
_o = 0
for _i, _ou in SIZES:
    OFF.append(_o)
    _o += _ou * _i + _ou
P_BASE = _o                                # 337 numbers

AD_OFF = {}                                # offset of each adapter in the flat adapter vector: A [r, in], then B [out, r]
_o = 0
for _l in ADAPTED:
    AD_OFF[_l] = _o
    _o += RANK * SIZES[_l][0] + SIZES[_l][1] * RANK
P_AD = _o                                  # 128 numbers

NAMES = {name: f"project_scaling_training.{name}" for name in
         ("data", "init", "adapters", "order", "order_ft")}


def _seed(name):
    return bookrng.seed_of(NAMES[name])


def _data():
    g = bookrng.generator(NAMES["data"])
    X = [[2.0 * g.uniform() - 1.0 for _ in range(4)] for _ in range(N_EX)]
    T0 = [(x[0] * x[1] + 0.5 * x[2]) - x[3] * x[3] for x in X]
    T1 = [(t + 0.5 * x[0]) - (0.25 * x[1]) * x[2] for t, x in zip(T0, X)]
    return X, T0, T1


X_ALL, T_BASE, T_FT = _data()


def _init_base(nudge=0.0):
    g = bookrng.generator(NAMES["init"])
    th = []
    for i, ou in SIZES:
        lim = math.sqrt(6.0 / (i + ou))
        for _ in range(ou * i):
            th.append((2.0 * g.uniform() - 1.0) * lim)
        th += [0.0] * ou
    th[0] += nudge
    return th


def _init_adapters(zero_b=True):
    """A entries (2u - 1) * 0.5 in layer order and row-major; B is zero (or, for the planted mistake, drawn the same way)."""
    g = bookrng.generator(NAMES["adapters"])
    ad = [0.0] * P_AD
    for l in ADAPTED:
        i, ou = SIZES[l]
        a0 = AD_OFF[l]
        for q in range(RANK * i):
            ad[a0 + q] = (2.0 * g.uniform() - 1.0) * 0.5
        if not zero_b:
            for q in range(ou * RANK):
                ad[a0 + RANK * i + q] = (2.0 * g.uniform() - 1.0) * 0.5
    return ad


# ------------------------------------------------------------------ one example: forward, backward, with or without recomputation

class _Meter:
    """The activations alive at once, in numbers (one per element), and the FLOPs of the base layers' products."""

    def __init__(self):
        self.cur = self.peak = self.flops = 0

    def alloc(self, n):
        self.cur += n
        self.peak = max(self.peak, self.cur)

    def free(self, n):
        self.cur -= n


def _layer_forward(l, x, th, ad):
    i, ou = SIZES[l]
    w0 = OFF[l]
    b0 = w0 + ou * i
    t = None
    if ad is not None and l in AD_OFF:
        a0 = AD_OFF[l]
        t = []
        for r in range(RANK):
            c = 0.0
            for q in range(i):
                c += ad[a0 + r * i + q] * x[q]
            t.append(c)
    out = []
    for o in range(ou):
        z = th[b0 + o]
        for q in range(i):
            z += th[w0 + o * i + q] * x[q]
        if t is not None:
            c = 0.0
            for r in range(RANK):
                c += ad[AD_OFF[l] + RANK * i + o * RANK + r] * t[r]
            z += SCALE * c
        out.append(z if (l == LAYERS - 1 or z > 0.0) else 0.0)
    return out


def _layer_backward(l, x, out, dy, th, ad, acc, base_off, want_base):
    """Adds this layer's gradient entries into acc (adapter entries at their own offset, base entries at base_off) and returns dx."""
    i, ou = SIZES[l]
    w0 = OFF[l]
    b0 = w0 + ou * i
    dz = [dy[o] if (l == LAYERS - 1 or out[o] > 0.0) else 0.0 for o in range(ou)]
    if want_base:
        for o in range(ou):
            for q in range(i):
                acc[base_off + w0 + o * i + q] += dz[o] * x[q]
            acc[base_off + b0 + o] += dz[o]
    dx = []
    for q in range(i):
        s = 0.0
        for o in range(ou):
            s += dz[o] * th[w0 + o * i + q]
        dx.append(s)
    if ad is not None and l in AD_OFF:
        a0 = AD_OFF[l]
        t = []
        for r in range(RANK):
            c = 0.0
            for q in range(i):
                c += ad[a0 + r * i + q] * x[q]
            t.append(c)
        for o in range(ou):
            for r in range(RANK):
                acc[a0 + RANK * i + o * RANK + r] += (SCALE * dz[o]) * t[r]
        dt = []
        for r in range(RANK):
            c = 0.0
            for o in range(ou):
                c += dz[o] * ad[a0 + RANK * i + o * RANK + r]
            dt.append(SCALE * c)
        for r in range(RANK):
            for q in range(i):
                acc[a0 + r * i + q] += dt[r] * x[q]
        for q in range(i):
            c = 0.0
            for r in range(RANK):
                c += dt[r] * ad[a0 + r * i + q]
            dx[q] = dx[q] + c
    return dx


def _example_gradient(th, ad, x, y, k, acc, base_off, want_base, meter, perturb=None):
    """Adds the gradient of half the squared error of one example into acc. k is None for keeping every activation, or the
    segment length of recomputation. perturb = (j, delta) adds delta to th[j] after the forward pass and before the backward
    (th must then be a copy)."""
    width = [SIZES[0][0]] + [ou for _, ou in SIZES]
    if k is None:
        v = [x]
        meter.alloc(width[0])
        for l in range(LAYERS):
            v.append(_layer_forward(l, v[l], th, ad))
            meter.alloc(width[l + 1])
            meter.flops += 2 * SIZES[l][0] * SIZES[l][1]
        dy = [v[LAYERS][0] - y]
        meter.free(width[LAYERS])
        if perturb:
            th[perturb[0]] += perturb[1]
        for l in range(LAYERS - 1, -1, -1):
            dy = _layer_backward(l, v[l], v[l + 1], dy, th, ad, acc, base_off, want_base)
            meter.flops += 4 * SIZES[l][0] * SIZES[l][1]
            if l + 1 < LAYERS:
                meter.free(width[l + 1])
        meter.free(width[0])
        return
    kept = {0: x}
    meter.alloc(width[0])
    cur = x
    for l in range(LAYERS):
        nxt = _layer_forward(l, cur, th, ad)
        meter.alloc(width[l + 1])
        meter.flops += 2 * SIZES[l][0] * SIZES[l][1]
        if l not in kept:
            meter.free(width[l])
        if (l + 1) % k == 0 and l + 1 < LAYERS:
            kept[l + 1] = nxt
        cur = nxt
    dy = [cur[0] - y]
    meter.free(width[LAYERS])
    if perturb:
        th[perturb[0]] += perturb[1]
    for g in range((LAYERS - 1) // k, -1, -1):
        s, e = g * k, min(g * k + k, LAYERS)
        v = {s: kept[s]}
        for l in range(s, e):
            v[l + 1] = _layer_forward(l, v[l], th, ad)
            meter.alloc(width[l + 1])
            meter.flops += 2 * SIZES[l][0] * SIZES[l][1]
        for l in range(e - 1, s - 1, -1):
            dy = _layer_backward(l, v[l], v[l + 1], dy, th, ad, acc, base_off, want_base)
            meter.flops += 4 * SIZES[l][0] * SIZES[l][1]
            meter.free(width[l + 1])
        meter.free(width[s])


# ------------------------------------------------------------------ one step of a run on R ranks

def _views(cfg, p):
    """(base, adapters) the forward pass reads, from the trainable vector p."""
    if cfg["stage"] == "base":
        return p, None
    if cfg["train_base"]:
        return p[P_AD:], p[:P_AD]
    return cfg["base"], p[:P_AD]


def _rows_of_step(cfg, d0):
    seed = cfg["seed"]
    return [td.epoch_order(seed, N_EX, d // N_EX)[d % N_EX] for d in range(d0, d0 + BATCH)]


def _shard_vectors(cfg, p, rows, meter):
    """Rank j's sum of per-example gradients over the draws b of the step with b mod R = j, in draw order from zeros."""
    th, ad = _views(cfg, p)
    n = len(p)
    base_off = 0 if cfg["stage"] == "base" else P_AD
    want_base = cfg["stage"] == "base" or cfg["train_base"]
    R = cfg["R"]
    vecs, counts = [], []
    for j in range(R):
        mine = [rows[b] for b in range(BATCH) if b % R == (0 if cfg["share"] == "first" else j)]
        acc = [0.0] * n
        for i in mine:
            _example_gradient(th, ad, X_ALL[i], cfg["T"][i], cfg["k"], acc, base_off, want_base, meter)
        vecs.append(acc)
        counts.append(len(mine))
    return vecs, counts


def _reduce(vecs, mode):
    """The sum over ranks: the coordinator adds rank 0, 1, ... from zeros; the ring pads to a multiple of R and reduces chunks."""
    R, n = len(vecs), len(vecs[0])
    if mode == "coordinator" or R == 1:
        total = [0.0] * n
        for v in vecs:
            for q in range(n):
                total[q] += v[q]
        return total, True
    pad = (-n) % R
    buf, _, _ = par._ring([v + [0.0] * pad for v in vecs])
    agree = all(b == buf[0] for b in buf)
    return buf[0][:n], agree


def _batch_gradient(cfg, p, rows, meter):
    vecs, counts = _shard_vectors(cfg, p, rows, meter)
    n = len(p)
    if cfg["combine"] == "unweighted":              # the mean of the ranks' means
        means = [[x / c for x in v] for v, c in zip(vecs, counts)]
        total, _ = _reduce(means, cfg["reduce"])
        return [x / len(vecs) for x in total]
    total, _ = _reduce(vecs, cfg["reduce"])
    if cfg["combine"] == "sum":                      # forgets to divide by the batch
        return total
    return [x / BATCH for x in total]


def _adam(p, m, v, g, t, lr=LR, eps=EPS):
    p1 = p2 = 1.0
    for _ in range(t):
        p1 *= B1
        p2 *= B2
    for j in range(len(p)):
        m[j] = B1 * m[j] + 0.1 * g[j]
        v[j] = B2 * v[j] + 0.001 * g[j] * g[j]
        p[j] = p[j] - lr * (m[j] / (1.0 - p1)) / (math.sqrt(v[j] / (1.0 - p2)) + eps)


def _fresh(p):
    return {"p": list(p), "m": [0.0] * len(p), "v": [0.0] * len(p), "t": 0, "d": 0}


def _advance(cfg, st, until, after=None, meter=None):
    meter = meter or _Meter()
    while st["t"] < until:
        rows = _rows_of_step(cfg, st["d"])
        g = _batch_gradient(cfg, st["p"], rows, meter)
        st["t"] += 1
        st["d"] += BATCH
        _adam(st["p"], st["m"], st["v"], g, st["t"], eps=cfg["eps"])
        if after:
            after(st)
    return st


def _config(stage, **kw):
    cfg = {"stage": stage, "R": 1, "reduce": "coordinator", "combine": "weighted", "share": "own", "k": None,
           "train_base": False, "base": None, "eps": EPS}
    cfg["seed"] = _seed("order" if stage == "base" else "order_ft")
    cfg["T"] = T_BASE if stage == "base" else T_FT
    cfg.update(kw)
    return cfg


def _loss(th, ad, T):
    s = 0.0
    for x, y in zip(X_ALL, T):
        v = x
        for l in range(LAYERS):
            v = _layer_forward(l, v, th, ad)
        s += 0.5 * (v[0] - y) * (v[0] - y)
    return s / N_EX


def _stage_loss(cfg, p, T=None):
    th, ad = _views(cfg, p)
    return _loss(th, ad, T if T is not None else cfg["T"])


def _maxdev(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _bytes(xs):
    return b"".join(struct.pack("<d", x) for x in xs)


def _digest(xs):
    return f"{cr.fnv1a64(_bytes(xs)):016x}"


_MEMO = {}


def _base_run(nudge=0.0, **kw):
    """The run of the base: every number trained, PRE_STEPS updates, from the initialisation (first weight nudged)."""
    key = ("base", nudge, tuple(sorted(kw.items())))
    if key not in _MEMO:
        cfg = _config("base", **kw)
        st = _advance(cfg, _fresh(_init_base(nudge)), PRE_STEPS)
        _MEMO[key] = (cfg, st)
    return _MEMO[key]


REF_BASE = ("base", 0.0, ())


def _ft_run(base, nudge=0.0, tag="", ad0=None, steps=FT_STEPS, **kw):
    """The run of the adapters on the frozen `base` (a list), FT_STEPS updates, from B = 0."""
    cfg = _config("ft", base=base, **kw)
    st = _advance(cfg, _fresh(ad0 if ad0 is not None else _init_adapters()), steps)
    return cfg, st


def _stable(fn, full=lambda x: f"{x:.6g}"):
    """fn(nudge) -> float; the digits on which the run and the four nudged copies agree."""
    return stable.text(fn(0.0), [fn(n) for n in stable.NUDGES], full)


def _crit(dev, bound):
    return f"passes (below {bound:g})".replace("e-0", "e-") if dev < bound else "FAILS"


# ------------------------------------------------------------------ milestone 1: the data, the network and its gradient

def data_and_network():
    rows = [(i, f"{X_ALL[i][0]:.6f}", f"{X_ALL[i][1]:.6f}", f"{X_ALL[i][2]:.6f}", f"{X_ALL[i][3]:.6f}",
             f"{T_BASE[i]:.6f}", f"{T_FT[i]:.6f}") for i in (0, 1, 2, 39)]
    flat_x = [c for x in X_ALL for c in x]
    th = _init_base()
    inv = []
    for l, (i, ou) in enumerate(SIZES):
        inv.append((l, f"{ou} x {i}", ou * i, ou, ou * i + ou, OFF[l], "none" if l == LAYERS - 1 else "ReLU"))
    inv.append(("all", "", sum(o * i for i, o in SIZES), sum(o for _, o in SIZES), P_BASE, "", ""))
    head = (f"The generator `{NAMES['data']}` (seed {_seed('data')}) draws, in this order and each by uniform(), the 4 coordinates of "
            f"example 0, then of example 1, and so on to example 39: x = 2u - 1. The targets are evaluated in this order of "
            f"operations: T0 = (x0 x1 + 0.5 x2) - x3 x3 and T1 = (T0 + 0.5 x0) - (0.25 x1) x2 (products first, then the sums left "
            f"to right). Each row prints an example, its four coordinates and its two targets to 6 decimals. The 160 "
            f"coordinates, written as 64-bit floats (IEEE 754, little-endian) in this order, have the FNV-1a 64 digest "
            f"{_digest(flat_x)}; the 40 targets T0 and the 40 targets T1 have the digests {_digest(T_BASE)} and {_digest(T_FT)}.\n\n")
    t1 = _table(["example", "x0", "x1", "x2", "x3", "T0", "T1"], rows, "rrrrrrr")
    mid = (f"\n\nThe network has {LAYERS} layers, each y = W x + b followed by max(., 0) except the last, which is linear. A neuron's "
           f"sum starts at its bias and adds the products W[o][i] x[i] in increasing i. The flat vector of the {P_BASE} numbers "
           f"holds the layers in order, each as W (row-major, [outputs, inputs]) and then b. The generator `{NAMES['init']}` "
           f"(seed {_seed('init')}) draws, layer by layer and row-major, one uniform() for every entry of W, W = (2u - 1) limit "
           f"with limit = sqrt(6 / (inputs + outputs)) (the square root correctly rounded); every b starts at 0.0 and takes no "
           f"draw, so {sum(o * i for i, o in SIZES)} numbers are drawn. The initial vector has the FNV-1a 64 digest {_digest(th)}, and its "
           f"first three numbers are {th[0]:.8f}, {th[1]:.8f}, {th[2]:.8f}.\n\n")
    t2 = _table(["layer", "W shape", "weights", "biases", "numbers", "offset", "activation"], inv, "rlrrrrl")
    return head + t1 + mid + t2


def _forward(th, x, ad=None):
    v = x
    for l in range(LAYERS):
        v = _layer_forward(l, v, th, ad)
    return v[0]


CHECK_ROWS = [0, 5, 9, 17, 23, 31, 38, 12, 25, 3]


def _numeric_check(loss_of, vec, g):
    h = 1e-6
    worst = 0.0
    for j in range(len(vec)):
        up, dn = list(vec), list(vec)
        up[j] += h
        dn[j] -= h
        worst = max(worst, abs((loss_of(up) - loss_of(dn)) / (2 * h) - g[j]) / max(1.0, abs(g[j])))
    return worst


def gradient_check():
    th = _init_base()
    acc = [0.0] * P_BASE
    meter = _Meter()
    for i in CHECK_ROWS:
        _example_gradient(th, None, X_ALL[i], T_BASE[i], None, acc, 0, True, meter)
    g = [a / BATCH for a in acc]
    worst = _numeric_check(lambda vec: sum(0.5 * (_forward(vec, X_ALL[i]) - T_BASE[i]) ** 2 for i in CHECK_ROWS) / BATCH, th, g)
    return (f"The initial network and the 10 examples 0, 5, 9, 17, 23, 31, 38, 12, 25, 3 (in this order). The analytic gradient is "
            f"the sum of the per-example gradients in this order divided by 10; the numeric one is the central difference of the "
            f"mean loss with step 1e-6 at every one of the {P_BASE} numbers. The deviation is |numeric - analytic| / max(1, "
            f"|analytic|), and the criterion is that its largest value is below 1e-6.\n\n"
            + _table(["quantity", "value"], [(f"largest deviation over all {P_BASE} numbers", _crit(worst, 1e-6))], "lr"))


# ------------------------------------------------------------------ milestone 2: the reference run on one device

def _trace(n, marks, stage_cfg=None):
    """Losses against T0 and T1 of the base run (first weight nudged by n) after each number of updates in marks."""
    cfg = stage_cfg or _config("base")
    st = _fresh(_init_base(n))
    out = {}
    for t in marks:
        _advance(cfg, st, t)
        out[t] = (_loss(st["p"], None, T_BASE), _loss(st["p"], None, T_FT))
    return out


def reference_run():
    marks = (0, 20, 40, 60, 80, 100, 120)
    runs = [_trace(n, marks) for n in (0.0,) + stable.NUDGES]
    rows = []
    for t in marks:
        rows.append((t, stable.text(runs[0][t][0], [r[t][0] for r in runs[1:]], lambda x: f"{x:.6g}"),
                     stable.text(runs[0][t][1], [r[t][1] for r in runs[1:]], lambda x: f"{x:.6g}")))
    head = ("One device, one rank, every activation kept. Step t draws the 10 examples of draws 10 (t - 1) to 10 t - 1 from the "
            f"order of `training_data.epoch_order` with the seed {_seed('order')} (seed_of '{NAMES['order']}'), computed afresh "
            "from the epoch of each draw; the gradient is the sum over the 10 rows in draw order, from 0.0, divided by 10, and the "
            "update is `adam_update` of Chapter 4 with rate 0.01 and step count t (the bias corrections by repeated "
            "multiplication from 1.0). The losses are the mean over all 40 examples, summed in example order from 0.0, of half "
            "the squared error of the network after t updates, against T0 (what this run trains on) and T1 (what the adapters of "
            "milestone 5 will be asked for). " + stable.RULE + "\n\n")
    return head + _table(["updates t", "loss against T0", "loss against T1"], rows, "rrr")


# ------------------------------------------------------------------ milestone 3: recomputation

def _batch_acc(th, ad, rows, T, k, want_base=True, perturb=None, meter=None):
    acc = [0.0] * (P_BASE if ad is None else P_AD)
    meter = meter or _Meter()
    for i in rows:
        _example_gradient(list(th), ad, X_ALL[i], T[i], k, acc, 0, want_base, meter, perturb)
    return acc


def _rel_dist(a, b):
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(y * y for y in b))
    return num / den


def recomputation():
    th = _init_base()
    rows = _rows_of_step(_config("base"), 0)
    ref = _batch_acc(th, None, rows, T_BASE, None)
    out = []
    for name, k in (("keep every activation", None), ("segments of 1 layer", 1), ("segments of 2 layers", 2),
                    ("segments of 3 layers", 3), ("one segment of 6 layers", 6)):
        meter = _Meter()
        one = _Meter()
        acc = _batch_acc(th, None, rows, T_BASE, k, meter=meter)
        _batch_acc(th, None, rows[:1], T_BASE, k, meter=one)
        differ = sum(1 for a, b in zip(acc, ref) if struct.pack("<d", a) != struct.pack("<d", b))
        out.append((name, one.peak, one.flops, f"{differ} of {P_BASE}"))
    head = ("The initial network, the 10 rows of step 1 (draws 0 to 9 of the order of `project_scaling_training.order`) and the "
            "gradient of their summed loss, accumulated in draw order. For each policy one example is run alone to count what is "
            "alive at once: the input and every layer's output is a vector of 4, 8, 8, 8, 8, 8 and 1 numbers; a vector is alive "
            "from the operation that makes it until the last operation that reads it (the output of the last layer is read "
            "once, to form the first gradient, and the gradient vectors that flow backward are not counted, being the same in "
            "every policy). A policy keeps a vector as a boundary when its index is a multiple of the segment length (the input "
            "has index 0) and the backward pass replays a segment from its boundary before differentiating it. FLOPs count 2 for "
            "every multiply-add of a layer's product and nothing else: 2 x inputs x outputs per layer for a forward, twice that "
            "for the backward (the products for the weights and for the input), whatever the layer. The last column counts the "
            "entries of the 337-number gradient of the batch whose 64-bit pattern differs from the keep run.\n\n")
    rows2 = []
    for name, k in (("keep every activation", None), ("segments of 2 layers", 2)):
        cells = []
        for j in (1, OFF[1] + 1):
            acc = _batch_acc(th, None, rows, T_BASE, k, perturb=(j, 0.3))
            cells.append(f"{_rel_dist(acc, ref):.4g}")
        rows2.append((name, *cells))
    mid = ("\n\nThe same batch when a pending update adds 0.3 to one weight after the forward pass of every example and before its "
           "backward pass: W[0][1] of layer 0 (flat index 1) or W[0][1] of layer 1 (flat index 41). The distance is the Euclidean "
           "norm of the difference from the keep gradient of the unchanged weights, divided by the norm of that gradient.\n\n")
    return (head + _table(["policy", "numbers alive at the peak (one example)", "FLOPs, one example", "gradient entries that differ"],
                          out, "lrrr")
            + mid + _table(["backward after the late update", "weight of layer 0", "weight of layer 1"], rows2, "lrr"))


# ------------------------------------------------------------------ milestone 4: dividing the batch

def _shard_sizes(R):
    return [len([b for b in range(BATCH) if b % R == j]) for j in range(R)]


def shard_layouts():
    th = _init_base()
    rows = _rows_of_step(_config("base"), 0)
    ref = _batch_gradient(_config("base"), th, rows, _Meter())
    out = []
    for R in (1, 2, 3, 4):
        sizes = _shard_sizes(R)
        d1 = _maxdev(_batch_gradient(_config("base", R=R), th, rows, _Meter()), ref)
        d2 = _maxdev(_batch_gradient(_config("base", R=R, reduce="ring"), th, rows, _Meter()), ref)
        out.append((R, " + ".join(map(str, sizes)), _crit(d1, 1e-12), _crit(d2, 1e-12)))
    bad = []
    for name, kw in (("mean of the four shard means", {"combine": "unweighted"}),
                     ("sum of the shards, not divided by the batch", {"combine": "sum"}),
                     ("every rank reads the shard of rank 0", {"share": "first"})):
        g = _batch_gradient(_config("base", R=4, **kw), th, rows, _Meter())
        bad.append((name, f"{_rel_dist(g, ref):.4g}"))
    head = ("The initial network and the gradient of step 1 (the 10 draws 0 to 9). Draw b of the step belongs to rank b mod R; a "
            "rank adds the gradients of its rows in draw order from zeros; the coordinator adds the ranks in rank order from "
            "zeros and the ring adds each chunk in the order of the ring (the vector is padded with zeros to a multiple of R); "
            "the sum is divided by 10. The deviation is the largest absolute difference of an entry from the gradient of one "
            "rank, and the criterion is that it is below 1e-12.\n\n")
    mid = ("\n\nThe same step on 4 ranks (shards of 3, 3, 2 and 2 rows) with three mistakes; the distance is the Euclidean norm of "
           "the difference from the one-rank gradient divided by the norm of that gradient.\n\n")
    return (head + _table(["ranks R", "rows of each rank", "coordinator", "ring"], out, "rlll")
            + mid + _table(["build", "relative distance"], bad, "lr"))


def _dev_final(kw_a, kw_b, nudge):
    """(deviation of the base numbers, final loss against T0) of two base runs differing in kw."""
    a = _base_run(nudge, **kw_a)[1]["p"]
    b = _base_run(nudge, **kw_b)[1]["p"]
    return _maxdev(a, b)


def parallel_runs():
    cases = [("1 rank", {}), ("2 ranks, coordinator", {"R": 2}), ("3 ranks, coordinator", {"R": 3}),
             ("4 ranks, coordinator", {"R": 4}), ("4 ranks, ring", {"R": 4, "reduce": "ring"}),
             ("4 ranks, ring, segments of 2 layers", {"R": 4, "reduce": "ring", "k": 2})]
    rows = []
    for name, kw in cases:
        devs = [_dev_final(kw, {}, n) for n in (0.0,) + stable.NUDGES]
        loss = _stable(lambda n, kw=kw: _loss(_base_run(n, **kw)[1]["p"], None, T_BASE))
        agree = _reduce_agree(kw)
        rows.append((name, _crit(max(devs), 1e-9), loss, agree))
    bad = []
    for name, kw in (("mean of the shard means", {"combine": "unweighted"}),
                     ("sum, not divided by the batch", {"combine": "sum"}),
                     ("every rank reads the shard of rank 0", {"share": "first"})):
        kw = dict(R=4, **kw)
        dev = _stable(lambda n, kw=kw: _dev_final(kw, {}, n), lambda x: f"{x:.3g}")
        loss = _stable(lambda n, kw=kw: _loss(_base_run(n, **kw)[1]["p"], None, T_BASE))
        bad.append((name, dev, loss))
    head = ("The base run of milestone 2, all 120 updates, on R ranks. The deviation is the largest absolute difference, over the "
            "337 numbers after 120 updates, from the one-rank run started from the same (nudged) initial vector, and the criterion "
            "is that it is below 1e-9 for the run and its four nudged copies. 'Ranks hold the same bits' is whether, after the "
            "last reduction, every rank's copy of the summed gradient is the same list of 64-bit patterns (checked at the final weights, "
            "on the rows of the last step). " + stable.RULE + "\n\n")
    mid = "\n\nThe same run on 4 ranks with three planted mistakes. The deviation is the largest absolute difference from the one-rank run.\n\n"
    return (head + _table(["build", "deviation from the 1-rank run", "loss against T0 after 120 updates", "ranks hold the same bits"],
                          rows, "lllr")
            + mid + _table(["build", "deviation from the 1-rank run", "loss against T0 after 120 updates"], bad, "lrr"))


def _reduce_agree(kw):
    """Whether every rank ends the all-reduce with the same bits, at the final weights and on the rows of the last step."""
    cfg = _config("base", **kw)
    st = _base_run(0.0, **kw)[1]
    vecs, _ = _shard_vectors(cfg, st["p"], _rows_of_step(cfg, (PRE_STEPS - 1) * BATCH), _Meter())
    return "yes" if _reduce(vecs, cfg["reduce"])[1] else "no"


def wire_per_step():
    th = _init_base()
    rows_step = _rows_of_step(_config("base"), 0)
    rows = []
    for R in (1, 2, 3, 4):
        r = []
        for n in (P_BASE, P_AD):
            padded = n + (-n) % R
            ring = 2 * (R - 1) * (padded // R) * 8
            star0 = (R - 1) * n * 8
            r += [ring, star0]
        flops = []
        for k in (None, 2):                              # rank 0 holds the most rows: draws b with b mod R = 0
            meter = _Meter()
            acc = [0.0] * P_BASE
            for b in range(0, BATCH, R):
                _example_gradient(th, None, X_ALL[rows_step[b]], T_BASE[rows_step[b]], k, acc, 0, True, meter)
            flops.append(meter.flops)
        busiest = max(_shard_sizes(R))
        rows.append((R, busiest, *flops, *r))
    head = ("Work of one step of the base run (10 rows) on R ranks, counted by the rules of milestones 3 and 4. A rank's FLOPs are "
            "those of its rows (counted while the rank runs them); the busiest rank, rank 0, has the most rows. Bytes are the payload of one gradient all-reduce, 8 bytes to a number (64-bit floats), without framing: "
            "the ring pads the vector with zeros to a multiple of R and a rank sends 2 (R - 1) chunks of one R-th of it; in the "
            "coordinator star rank 0 receives (R - 1) whole vectors (and sends as many back). 'Full' is the 337 numbers of the base "
            "run, 'adapters' the 128 numbers of the adapter run.\n\n")
    return head + _table(["R", "rows of the busiest rank", "its FLOPs, keep", "its FLOPs, segments of 2", "full: ring bytes sent per rank",
                          "full: star bytes received by rank 0", "adapters: ring bytes per rank", "adapters: star bytes received by rank 0"],
                         rows, "rrrrrrrr")


# ------------------------------------------------------------------ milestone 5: adapters on a frozen base

def _ft(n=0.0, **kw):
    key = ("ft", n, repr(sorted(kw.items(), key=lambda kv: kv[0])))
    if key not in _MEMO:
        base = kw.pop("base", None)
        base = base if base is not None else _base_run(n)[1]["p"]
        ad0 = kw.pop("ad0", None)
        steps = kw.pop("steps", FT_STEPS)
        cfg = _config("ft", base=list(base), **kw)
        p0 = ad0 if ad0 is not None else _init_adapters()
        if cfg["train_base"]:
            p0 = p0 + list(base)
        _MEMO[key] = (cfg, _advance(cfg, _fresh(p0), steps))
    return _MEMO[key]


def adapter_layout():
    rows = []
    for l in ADAPTED:
        i, ou = SIZES[l]
        rows.append((l, f"{RANK} x {i}", f"{ou} x {RANK}", RANK * i + ou * RANK, AD_OFF[l]))
    rows.append(("all", "", "", P_AD, ""))
    ad = _init_adapters(zero_b=False)
    th = _base_run(0.0)[1]["p"]
    base0 = _loss(th, None, T_FT)
    zero0 = _loss(th, _init_adapters(), T_FT)
    rnd0 = _loss(th, ad, T_FT)
    # a gradient check with a nonzero B
    rows_c = CHECK_ROWS
    acc = [0.0] * P_AD
    meter = _Meter()
    for i in rows_c:
        _example_gradient(list(th), ad, X_ALL[i], T_FT[i], None, acc, 0, False, meter)
    g = [a / BATCH for a in acc]
    worst = _numeric_check(lambda vec: sum(0.5 * (_forward(th, X_ALL[i], vec) - T_FT[i]) ** 2 for i in rows_c) / BATCH, ad, g)
    acc0 = [0.0] * P_AD
    ad0 = _init_adapters()
    for i in rows_c:
        _example_gradient(list(th), ad0, X_ALL[i], T_FT[i], None, acc0, 0, False, meter)
    a_nonzero = sum(1 for l in ADAPTED for q in range(RANK * SIZES[l][0]) if acc0[AD_OFF[l] + q] != 0.0)
    b_nonzero = sum(1 for l in ADAPTED for q in range(SIZES[l][1] * RANK) if acc0[AD_OFF[l] + RANK * SIZES[l][0] + q] != 0.0)
    head = (f"Rank r = {RANK}, scale s = alpha / r = 1.0 (alpha = r), on layers 1 to 4 (the 8 by 8 layers). A layer's adapter is A "
            f"of [{RANK}, in] and then B of [out, {RANK}] in the flat vector of the {P_AD} adapter numbers. In the forward pass "
            f"t = A x (sums from 0.0 in increasing index), and the neuron's sum is the base sum (as in milestone 1) plus s times "
            f"the sum over r in increasing order of B[o][r] t[r] (from 0.0), and then max(., 0). The generator "
            f"`{NAMES['adapters']}` (seed {_seed('adapters')}) draws, for layers 1 to 4 in order, one uniform() for each entry of A "
            f"row-major, A = (2u - 1) x 0.5, so 64 numbers are drawn; B starts at zero (the planted mistake below draws B in the "
            f"same way right after the A of the same layer).\n\n")
    t1 = _table(["layer", "A shape", "B shape", "numbers", "offset"], rows, "rllrr")
    mid = (f"\n\nWith the trained base of milestone 2 the loss against T1 is {base0:.10g} for the base alone and "
           f"{zero0:.10g} with the adapters at their start (a difference of {abs(base0 - zero0):g}); with a nonzero B (every entry "
           f"drawn as (2u - 1) x 0.5) it is {rnd0:.6g}. A gradient check on the 10 rows of milestone 1 with B drawn that way "
           f"compares all {P_AD} analytic entries with central differences of step 1e-6: it {_crit(worst, 1e-6)}. At B = 0 "
           f"the gradient of the same rows has {a_nonzero} nonzero entries among the {RANK * 8 * 4} of A and {b_nonzero} among the "
           f"{8 * RANK * 4} of B.\n")
    return head + t1 + mid


def adapter_runs():
    refbase = lambda n: _base_run(n)[1]["p"]
    rows = []
    cases = [("1 rank, every activation kept", {}), ("4 ranks, ring, segments of 2 layers", {"R": 4, "reduce": "ring", "k": 2})]
    for name, kw in cases:
        devs = [_maxdev(_ft(n, **kw)[1]["p"], _ft(n)[1]["p"]) for n in (0.0,) + stable.NUDGES]
        loss = _stable(lambda n, kw=kw: _loss(refbase(n), _ft(n, **kw)[1]["p"], T_FT))
        loss0 = _stable(lambda n, kw=kw: _loss(refbase(n), _ft(n, **kw)[1]["p"], T_BASE))
        rows.append((name, _crit(max(devs), 1e-9), loss, loss0))
    bloss = _stable(lambda n: _loss(refbase(n), None, T_FT))
    bloss0 = _stable(lambda n: _loss(refbase(n), None, T_BASE))
    rows.insert(0, ("the base alone (no adapter)", "-", bloss, bloss0))
    base = refbase(0.0)
    frozen = _digest(base) == _digest(_ft(0.0)[0]["base"])
    bad = []
    zero = _ft(0.0, ad0=[0.0] * P_AD)[1]["p"]
    bad.append(("A and B both start at zero", f"{max(abs(x) for x in zero):g}", _stable(lambda n: _loss(refbase(n), _ft(n, ad0=[0.0] * P_AD)[1]["p"], T_FT))))
    rb = _init_adapters(zero_b=False)
    bad.append(("B starts nonzero", f"{_loss(base, rb, T_FT):.6g}", _stable(lambda n: _loss(refbase(n), _ft(n, ad0=rb)[1]["p"], T_FT))))
    uf = [_ft(n, train_base=True) for n in (0.0,) + stable.NUDGES]
    moved = _stable(lambda n: _maxdev(_ft(n, train_base=True)[1]["p"][P_AD:], refbase(n)), lambda x: f"{x:.3g}")
    bad.append(("the base is trained too (465 numbers)", moved,
                _stable(lambda n: _loss(_ft(n, train_base=True)[1]["p"][P_AD:], _ft(n, train_base=True)[1]["p"][:P_AD], T_FT))))
    head = ("The adapter run: the base is the trained vector of milestone 2 (after 120 updates, from the same nudged start), "
            "frozen; the 128 adapter numbers are trained for 60 updates against T1 with the Adam of milestone 2 (rate 0.01), the "
            f"draws from the order of `{NAMES['order_ft']}` (seed {_seed('order_ft')}), batches of 10 as before. The deviation is "
            "the largest absolute difference of the 128 adapter numbers from those of the one-rank run, and the criterion is "
            "that it is below 1e-9 for the run and its four nudged copies. After the 60 updates the digest of the base, written as "
            f"64-bit floats, is {'unchanged' if frozen else 'CHANGED'}. " + stable.RULE + "\n\n")
    t1 = _table(["run", "deviation of the adapters", "loss against T1", "loss against T0"], rows, "llrr")
    mid = ("\n\nThree planted mistakes on one rank. 'After 60 updates' is the largest absolute adapter number for the first row, "
           "the loss against T1 before the first update for the second and the largest absolute change of a base number for the "
           "third.\n\n")
    t2 = _table(["build", "check", "loss against T1 after 60 updates"], bad, "lrr")
    return head + t1 + mid + t2


# ------------------------------------------------------------------ milestone 6: generations on a disk that can be killed

def _meta(st, base_digest, values, state):
    body = {"base": base_digest, "step": st["t"], "values": {"digest": f"{cr.fnv1a64(values):016x}", "length": len(values)},
            "state": {"digest": f"{cr.fnv1a64(state):016x}", "length": len(state)}}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _commit_ops(st, base_digest):
    """The writes of one commit in order: values, state, meta last, then the rename of the temporary directory."""
    values = _bytes(st["p"])
    state = _bytes(st["m"]) + _bytes(st["v"]) + struct.pack("<QQ", st["t"], st["d"])
    tmp = f"tmp-{st['t']:04d}"
    return [("write", f"{tmp}/values.bin", values), ("write", f"{tmp}/state.bin", state),
            ("write", f"{tmp}/meta.json", _meta(st, base_digest, values, state)), ("rename", tmp, f"gen-{st['t']:04d}")]


def _apply(disk, op):
    if op[0] == "write":
        disk[op[1]] = op[2]
    else:
        for name in [n for n in disk if n.startswith(op[1] + "/")]:
            disk[op[2] + name[len(op[1]):]] = disk.pop(name)


def _generations(disk):
    return sorted({n.split("/")[0] for n in disk if n.startswith("gen-")}, reverse=True)


def _prune(disk, base_digest):
    """Keeps the two newest generations that verify and removes everything else (temporaries included)."""
    keep = [g for g in _generations(disk) if _read_generation(disk, g, base_digest, True) is not None][:2]
    for name in [n for n in disk if n.split("/")[0] not in keep]:
        del disk[name]


def _read_generation(disk, gen, base_digest, check):
    try:
        meta = json.loads(disk[f"{gen}/meta.json"])
        values, state = disk[f"{gen}/values.bin"], disk[f"{gen}/state.bin"]
    except (KeyError, ValueError):
        return None
    if meta.get("base") != base_digest:
        return None
    if check:
        for blob, key in ((values, "values"), (state, "state")):
            if len(blob) != meta[key]["length"] or f"{cr.fnv1a64(blob):016x}" != meta[key]["digest"]:
                return None
    n = len(values) // 8
    if len(values) != 8 * n or len(state) != 16 * n + 16:
        return None
    dec = lambda blob, a, b: list(struct.unpack(f"<{b - a}d", blob[8 * a:8 * b]))
    t, d = struct.unpack("<QQ", state[16 * n:])
    return {"p": dec(values, 0, n), "m": dec(state, 0, n), "v": dec(state, n, 2 * n), "t": t, "d": d}


def _restore(disk, base_digest, check=True):
    for gen in _generations(disk):
        st = _read_generation(disk, gen, base_digest, check)
        if st is not None:
            return st, gen
    return None, None


def _ft_killed(n, crash_after, damage=None, check=True, omit=(), resume_kw=None, **kw):
    """The adapter run, killed after `crash_after` updates and resumed from the newest generation that is accepted. Returns
    (final state, the generation resumed from, the disk after the crash)."""
    key = ("kill", n, crash_after, damage, check, omit, tuple(sorted((resume_kw or {}).items())), tuple(sorted(kw.items())))
    if key in _MEMO:
        return _MEMO[key]
    base = _base_run(n)[1]["p"]
    bd = _digest(base)
    cfg = _config("ft", base=list(base), **kw)
    disk = {}

    def commit(st):
        if st["t"] % EVERY == 0:
            for op in _commit_ops(st, bd):
                _apply(disk, op)
            _prune(disk, bd)

    st = _advance(cfg, _fresh(_init_adapters()), crash_after, after=commit)
    st = None                                                     # the process is gone; only the disk is left
    if damage:
        _apply_damage(disk, damage)
    got, gen = _restore(disk, bd, check)
    if "moments" in omit:
        got["m"], got["v"] = [0.0] * len(got["p"]), [0.0] * len(got["p"])
    if "draws" in omit:
        got["d"] = 0
    cfg2 = dict(cfg)
    cfg2.update(resume_kw or {})
    final = _advance(cfg2, got, FT_STEPS)
    _MEMO[key] = (final, gen, disk)
    return _MEMO[key]


def _apply_damage(disk, damage):
    gen = _generations(disk)[0]
    if damage == "tail zeroed":                                   # a power loss left the last 16 bytes of values.bin unwritten
        v = disk[f"{gen}/values.bin"]
        disk[f"{gen}/values.bin"] = v[:-16] + b"\x00" * 16


def commit_protocol():
    """The crash points of one commit, at the update that follows 30 and the disk those leave."""
    base = _base_run(0.0)[1]["p"]
    bd = _digest(base)
    cfg = _config("ft", base=list(base))
    disk = {}
    st = _fresh(_init_adapters())
    for stop in (EVERY, 2 * EVERY, 3 * EVERY):
        _advance(cfg, st, stop)
        for op in _commit_ops(st, bd):
            _apply(disk, op)
        _prune(disk, bd)
    before = _generations(disk)
    _advance(cfg, st, 4 * EVERY)
    ops = _commit_ops(st, bd)
    files = {name.split("/")[1]: len(blob) for name, blob in [(o[1], o[2]) for o in ops if o[0] == "write"]}
    sizes = _table(["file", "bytes", "content"],
                   [("values.bin", files["values.bin"], f"the {P_AD} adapter numbers as 64-bit floats ({P_AD} x 8)"),
                    ("state.bin", files["state.bin"], f"m ({P_AD} x 8), v ({P_AD} x 8), then t and d as 64-bit unsigned integers (2 x 8)"),
                    ("meta.json", files["meta.json"], "keys sorted, no spaces: base digest, step, length and FNV-1a digest of the two files")], "lrl")
    rows = []
    for done in range(len(ops) + 1):
        d2 = dict(disk)
        for op in ops[:done]:
            _apply(d2, op)
        got, gen = _restore(d2, bd, True)
        label = ["nothing written", "values.bin written", "state.bin written", "meta.json written", "directory renamed to gen-0040"][done]
        rows.append((label, ", ".join(_generations(d2)), f"{gen} (step {got['t']})"))
    head = (f"The adapter run on one rank with a checkpoint every {EVERY} updates. A commit writes three files into a temporary "
            f"directory `tmp-<step>`, in this order, and renames it to `gen-<step>` (4-digit step): `values.bin`, `state.bin` and, last, "
            f"`meta.json`. A disk is a map from path to bytes. After a commit the two newest generations that verify are kept. The "
            f"base is the trained vector of milestone 2; its FNV-1a digest, {bd}, is a field of every meta file and a generation "
            f"whose base differs is not accepted. A reader opens generations from the newest, accepts the first whose meta parses, "
            f"whose base digest matches and whose two files have the recorded lengths and FNV-1a digests, and reads nothing else. "
            f"After the updates 10, 20 and 30 are committed the disk holds {', '.join(before)} (step 10 was pruned). The files of the "
            f"commit of step 40:\n\n")
    mid = ("\n\nThe process is killed after each prefix of that commit's operations (everything written before the kill survives, "
           "nothing after; power loss is the subject of Chapter 7):\n\n")
    return head + sizes + mid + _table(["killed after", "generations on the disk", "generation the reader accepts"], rows, "lll")


def _bits_equal(a, b):
    return _bytes(a) == _bytes(b)


def resume_runs():
    rows = []
    ref = _ft(0.0, R=4, reduce="ring", k=2)[1]
    ref1 = _ft(0.0)[1]
    kw = dict(R=4, reduce="ring", k=2)
    cases = [("every part restored", dict(crash_after=CRASH_AT)),
             ("first and second moments set to zero", dict(crash_after=CRASH_AT, omit=("moments",))),
             ("examples drawn set to zero", dict(crash_after=CRASH_AT, omit=("draws",))),
             ("resumed on 2 ranks (ring) instead of 4", dict(crash_after=CRASH_AT, resume_kw={"R": 2})),
             ("newest generation damaged, digests checked", dict(crash_after=47, damage="tail zeroed")),
             ("newest generation damaged, accepted by name", dict(crash_after=47, damage="tail zeroed", check=False))]
    for name, c in cases:
        fin, gen, _ = _ft_killed(0.0, **c, **kw)
        same = _bits_equal(fin["p"], ref["p"])
        d0 = max(_maxdev(_ft_killed(n, **c, **kw)[0]["p"], _ft(n, **kw)[1]["p"]) for n in (0.0,) + stable.NUDGES)
        if same:
            dev = "0"
        elif d0 < 1e-9:
            dev = "below 1e-9"
        else:
            dev = _stable(lambda n, c=c: _maxdev(_ft_killed(n, **c, **kw)[0]["p"], _ft(n, **kw)[1]["p"]), lambda x: f"{x:.3g}")
        d1 = max(_maxdev(_ft_killed(n, **c, **kw)[0]["p"], _ft(n)[1]["p"]) for n in (0.0,) + stable.NUDGES)
        rows.append((name, gen, "yes" if same else "no", dev, _crit(d1, 1e-9)))
    head = ("The adapter run of milestone 5 on 4 ranks (ring, segments of 2 layers), 60 updates, a checkpoint every 10 updates "
            "as in the previous table. The run is killed after the stated number of updates (37, or 47 for the last two rows), "
            "everything in memory is lost, and a second process restores from the disk, applies the change named in the first "
            "column, and carries on to update 60 with the same step function. 'Same bits' compares the 128 adapter numbers, as 64-bit "
            "patterns, with the uninterrupted 4-rank run; the deviation is the largest absolute difference from it. The last "
            "column compares with the one-rank run of milestone 5 (criterion below 1e-9, for the run and its four nudged copies). "
            "The damage is that a power loss left the last 16 bytes of the newest `values.bin` as zeros. " + stable.RULE + "\n\n")
    return head + _table(["what the second process does", "generation restored", "same bits as the uninterrupted run",
                          "deviation from it", "against the 1-rank run"], rows, "llrrl")


# ------------------------------------------------------------------ milestone 7: the whole run against the reference

def _whole(n, **kw):
    """Both stages on 4 ranks (ring, segments of 2 layers) in the complete build; returns (base, adapters)."""
    sys_kw = dict(R=4, reduce="ring", k=2)
    base = _base_run(n, **sys_kw)[1]["p"]
    p = _ft_whole(n, base, **kw)
    if kw.get("train_base"):                       # the trained vector is the 128 adapter numbers, then the base
        return p[P_AD:], p[:P_AD]
    return base, p


def _ft_whole(n, base, crash_after=CRASH_AT, **kw):
    key = ("whole", n, crash_after, tuple(sorted(kw.items())))
    if key not in _MEMO:
        bd = _digest(base)
        sys_kw = dict(R=4, reduce="ring", k=2)
        sys_kw.update({k: v for k, v in kw.items() if k in ("combine", "train_base")})
        cfg = _config("ft", base=list(base), **sys_kw)
        disk = {}

        def commit(st):
            if st["t"] % EVERY == 0:
                for op in _commit_ops(st, bd):
                    _apply(disk, op)
                _prune(disk, bd)

        p0 = _init_adapters() + (list(base) if cfg["train_base"] else [])
        _advance(cfg, _fresh(p0), crash_after, after=commit)
        if kw.get("damage"):
            _apply_damage(disk, kw["damage"])
        got, _ = _restore(disk, bd, kw.get("check", True))
        if kw.get("omit") == "moments":
            got["m"], got["v"] = [0.0] * len(got["p"]), [0.0] * len(got["p"])
        _MEMO[key] = _advance(cfg, got, FT_STEPS)["p"]
    return _MEMO[key]


def whole_run():
    kinds = [("the specified system, killed after 37 updates and resumed", {}),
             ("a gradient averaged without the row counts (mistake of milestone 4)", {"combine": "unweighted"}),
             ("the base trained beside the adapters (mistake of milestone 5)", {"train_base": True}),
             ("resumed without the moments (mistake of milestone 6)", {"omit": "moments"}),
             ("resumed from a damaged generation accepted by name (mistake of milestone 6)",
              {"damage": "tail zeroed", "check": False, "crash_after": 47})]
    rows = []
    for name, kw in kinds:
        kw = dict(kw)

        def devs(n, kw=kw):
            base, ad = _whole(n, **kw)
            rb, rad = _base_run(n)[1]["p"], _ft(n)[1]["p"]
            return _maxdev(base, rb), _maxdev(ad[:P_AD], rad), base, ad

        allb = [devs(n) for n in (0.0,) + stable.NUDGES]
        if max(max(d[0], d[1]) for d in allb) < 1e-9:
            dev = _crit(max(max(d[0], d[1]) for d in allb), 1e-9)
        else:
            dev = _stable(lambda n: max(devs(n)[0], devs(n)[1]), lambda x: f"{x:.3g}")

        def final(n, kw=kw):
            base, ad = _whole(n, **kw)
            return _loss(base, ad[:P_AD], T_FT)

        rows.append((name, dev, _stable(final)))
    ref = _stable(lambda n: _loss(_base_run(n)[1]["p"], _ft(n)[1]["p"], T_FT))
    rows.insert(0, ("the reference: one rank, every activation kept, never killed", "0", ref))
    head = ("Both runs of the project end to end. The reference trains the base for 120 updates and then the 128 adapter numbers for "
            "60 on one rank with every activation kept. The system trains the base on 4 ranks (ring, segments of 2 layers), then "
            "the adapters on the same 4 ranks with a checkpoint every 10 updates, killed after 37 (or 47) updates and resumed as "
            "in milestone 6; each planted mistake is made in the adapter stage, after a correct 4-rank base. The deviation is the largest absolute difference, over the 337 base numbers and the 128 adapter numbers, "
            "from the reference run started from the same (nudged) initial vector, and the criterion is that it is below 1e-9 for "
            "the run and its four nudged copies. The loss is the mean over the 40 examples against T1 of the final network. " +
            stable.RULE + "\n\n")
    return head + _table(["build", "largest deviation from the reference", "loss against T1"], rows, "lrr")


# ------------------------------------------------------------------ the opening numbers and the traces of the figure

def opening():
    cases = [("one rank", {}), ("four ranks, the shard gradients weighted by their row counts", {"R": 4}),
             ("four ranks, the mean of the four shard means", {"R": 4, "combine": "unweighted"})]
    rows = []
    for name, kw in cases:
        loss = _stable(lambda n, kw=kw: _loss(_base_run(n, **kw)[1]["p"], None, T_BASE))
        dev = _stable(lambda n, kw=kw: _dev_final(kw, {}, n), lambda x: f"{x:.3g}") if kw.get("combine") else \
            ("0" if not kw else "below 1e-9")
        rows.append((name, loss, dev))
    head = ("The base run of milestone 2 (the network of milestone 1, 120 updates of 10 draws, Adam with rate 0.01) trained three "
            "ways from the same initial vector and the same draws. 'Loss' is the mean over the 40 examples of half the squared "
            "error against T0 of the network after the 120 updates; 'deviation' is the largest absolute difference, over the "
            "337 numbers, from the one-rank run. " + stable.RULE + "\n\n")
    return head + _table(["build", "loss against T0 after 120 updates", "deviation from the one-rank run"], rows, "lrr")


def deviation_traces(n=0.0):
    """Largest absolute deviation of the 128 adapter numbers from the one-rank reference after each of the 60 updates, for four
    builds that start from the reference base: the data is (name, list of 60 floats)."""
    base = _base_run(n)[1]["p"]
    ref_p = []
    cfg = _config("ft", base=list(base))
    _advance(cfg, _fresh(_init_adapters()), FT_STEPS, after=lambda st: ref_p.append(list(st["p"])))

    def run(after_step=None, change=None, **kw):
        c = _config("ft", base=list(base), R=4, reduce="ring", k=2, **kw)
        out = []
        st = _fresh(_init_adapters())
        while st["t"] < FT_STEPS:
            _advance(c, st, st["t"] + 1)
            if after_step is not None and st["t"] == after_step:
                change(st)
            out.append(_maxdev(st["p"], ref_p[st["t"] - 1]))
        return out

    def zero_moments(st):
        st["m"], st["v"] = [0.0] * len(st["p"]), [0.0] * len(st["p"])

    def zero_tail(st):
        st["p"][-2:] = [0.0, 0.0]

    return [("the specified system", run()),
            ("mean of the shard means", run(combine="unweighted")),
            ("moments lost at update 30", run(30, zero_moments)),
            ("damaged generation of update 40 accepted", run(40, zero_tail))]


def deviation_table():
    marks = (1, 10, 30, 31, 40, 41, 60)
    runs = [deviation_traces(n) for n in (0.0,) + stable.NUDGES]
    rows = []
    for b, (name, _) in enumerate(runs[0]):
        cells = []
        for t in marks:
            vals = [r[b][1][t - 1] for r in runs]
            if max(vals) < 1e-15:
                cells.append("below 1e-15")
            else:
                cells.append(stable.text(vals[0], vals[1:], lambda x: f"{x:.3g}"))
        rows.append((name, *cells))
    head = ("The adapter run of milestone 5 on 4 ranks (ring, segments of 2 layers) from the reference base, and three planted mistakes "
            "(the mean of the shard means from the first update; the moments set to zero after update 30; the last two numbers of the "
            "adapter vector set to zero after update 40, which is what accepting a generation whose `values.bin` ends in zeros does). "
            "Each cell is the largest absolute difference of the 128 adapter numbers after the update named in the column head from "
            "the one-rank run after the same update. " + stable.RULE + "\n\n")
    return head + _table(["build"] + [f"update {t}" for t in marks], rows, "l" + "r" * len(marks))


def blind_mistake():
    rows = []
    for eps in (EPS, 1e-300):
        f = lambda n, eps=eps: _maxdev(_base_run(n, R=4, combine="sum", eps=eps)[1]["p"], _base_run(n, eps=eps)[1]["p"])
        dev = _crit(max(f(n) for n in (0.0,) + stable.NUDGES), 1e-9)
        if dev == "FAILS":
            dev = _stable(f, lambda x: f"{x:.3g}")
        loss = _stable(lambda n, eps=eps: _loss(_base_run(n, R=4, combine="sum", eps=eps)[1]["p"], None, T_BASE))
        ref = _stable(lambda n, eps=eps: _loss(_base_run(n, eps=eps)[1]["p"], None, T_BASE))
        rows.append((f"{eps:g}", dev, loss, ref))
    head = ("The base run of milestone 2 on 4 ranks with the gradient left as the sum of the ten rows (not divided by 10), against "
            "the one-rank run with the same epsilon in the denominator of Adam. The deviation is the largest absolute difference "
            "over the 337 numbers after 120 updates. " + stable.RULE + "\n\n")
    return head + _table(["epsilon of Adam", "deviation of the undivided sum", "its loss against T0", "loss against T0 of the one-rank run"],
                         rows, "lrrr")
