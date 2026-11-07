# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Normalization Layers".

Plain Python 3 on floats (IEEE double precision), written from the chapter's own
description and independent of any engine code. Each public function returns a
markdown string that the chapter embeds between `<!-- worked: normalization.<name> -->`
markers. The same module supplies the data of the chapter's figures.

One routine does all the normalizations of the chapter. A tensor is a flat list in
row-major order, a *group* is the list of flat indices that are averaged together, and
the four layers differ only in the groups they are given and in whether the mean is
subtracted:

  batch norm   one group per column of a [rows, d] array (the rows are the batch)
  layer norm   one group per row
  rms norm     one group per row, mean not subtracted
  group norm   one group per (example, channel group)

The running example of the stack experiments is the deep stack of the initialization
chapter (ten layers of 64 units, ReLU, He start): the data and starting weights are the
streams of that chapter, read through its worked module.

Swedish Embedded AB implements solutions for making deep networks train stably. If your
team needs expertise in normalization layers and activation statistics, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math
import struct
from functools import lru_cache

import bookrng
import initialization as ini

EPS = 1e-5            # the epsilon inside the square root, in every example unless stated
KINDS = ("batch", "layer", "rms")
KIND_LABEL = {"none": "no normalization", "batch": "batch norm", "layer": "layer norm", "rms": "RMS norm"}
STACK_RATES = (0.01, 0.03, 0.1)
STACK_STEPS = 100
STACK_RECORD = (0, 1, 10, 30, 100)
LAYERS_SHOWN = (1, 2, 3, 5, 7, 10)


# ------------------------------------------------------------------ formatting

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _g(v, d=4):
    """Fixed decimals for ordinary sizes, scientific for very small or large values."""
    if v == 0.0:
        return "0"
    if not math.isfinite(v):
        return "not finite"
    return f"{v:.{d}f}" if 1e-3 <= abs(v) < 1e4 else _sci(v)


def _vec(v, d=4):
    return "(" + ", ".join(_g(x, d) for x in v) + ")"


def _gen_note(name):
    s = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed 0x{s:016x} ({s})"


def _loss(v):
    return _g(v) if math.isfinite(v) and v < 1e30 else "not finite"


# ------------------------------------------------------- the one normalization routine

def groups_for(kind, rows, d):
    """Flat indices of each group of a row-major [rows, d] array, and whether the mean is removed."""
    if kind == "batch":
        return [[i * d + j for i in range(rows)] for j in range(d)], True
    if kind == "layer":
        return [[i * d + j for j in range(d)] for i in range(rows)], True
    if kind == "rms":
        return [[i * d + j for j in range(d)] for i in range(rows)], False
    raise ValueError(kind)


def norm_forward(x, groups, centre, gamma, beta, eps=EPS):
    """y[i] = gamma[ch(i)] * xhat[i] + beta[ch(i)], with ch(i) = i mod d (d = len(gamma)).

    Each group has mean mu (0 when `centre` is false) and v = mean((x - mu)^2), divisor n;
    xhat = (x - mu) * r with r = 1 / sqrt(v + eps). Returns y and the cache (xhat, r per group)."""
    d = len(gamma)
    xhat = [0.0] * len(x)
    rs = []
    for idx in groups:
        n = len(idx)
        mu = sum(x[i] for i in idx) / n if centre else 0.0
        v = sum((x[i] - mu) ** 2 for i in idx) / n
        r = 1.0 / math.sqrt(v + eps)
        rs.append(r)
        for i in idx:
            xhat[i] = (x[i] - mu) * r
    y = [gamma[i % d] * xhat[i] + beta[i % d] for i in range(len(x))]
    return y, (xhat, rs)


def norm_backward(dy, cache, groups, centre, gamma):
    """dx, dgamma, dbeta from the gradient dy arriving at y.

    With g = dy * gamma (the gradient arriving at xhat), inside each group
    dx_i = r * (g_i - mean(g) - xhat_i * mean(g * xhat)); the mean(g) term is absent when the mean
    was not removed. dgamma and dbeta sum dy * xhat and dy over every element of a channel."""
    xhat, rs = cache
    d = len(gamma)
    dx = [0.0] * len(dy)
    for idx, r in zip(groups, rs):
        n = len(idx)
        g = [dy[i] * gamma[i % d] for i in idx]
        m1 = sum(g) / n if centre else 0.0
        m2 = sum(gi * xhat[i] for gi, i in zip(g, idx)) / n
        for gi, i in zip(g, idx):
            dx[i] = r * (gi - m1 - xhat[i] * m2)
    dgamma = [0.0] * d
    dbeta = [0.0] * d
    for i, dyi in enumerate(dy):
        dgamma[i % d] += dyi * xhat[i]
        dbeta[i % d] += dyi
    return dx, dgamma, dbeta


def _central(f, x, h):
    """Central-difference gradient of a scalar function f of a flat list."""
    out = []
    for i in range(len(x)):
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        out.append((f(xp) - f(xm)) / (2.0 * h))
    return out


# --------------------------------------------------------------- the deep stack

def _relu(v):
    return v if v > 0.0 else 0.0


@lru_cache(maxsize=None)
def stack_run(kind, lr, steps=STACK_STEPS, nudge=0.0):
    """Full-batch gradient descent on the deep stack of the initialization chapter, with a
    normalization of the given kind (or none) between every linear layer and its ReLU.

    Returns (losses before every step, {step: rms of the ReLU input of each of the 10 layers},
    {step: the values of the ReLU input of layer 10}, norm of db per layer at step 0,
    {step: rms of the pre-activation z of each layer, before any normalization})."""
    n = ini.N_TRAIN
    X, t = ini.make_data(n)
    Ws, bs = ini.init_stack("he")
    Ws[0][0][0] += nudge
    depth, width = ini.DEPTH, ini.WIDTH
    gam = [[1.0] * width for _ in range(depth)]
    bet = [[0.0] * width for _ in range(depth)]
    if kind != "none":
        groups, centre = groups_for(kind, n, width)
    losses, rms_at, last_at, db0, z_at = [], {}, {}, None, {}
    for step in range(steps + 1):
        A, As, Ns, caches = X, [X], [], []
        zs_out, zs_all = None, []
        for l in range(depth + 1):
            Z = [[ini.dot(w, a) + bj for w, bj in zip(Ws[l], bs[l])] for a in A]
            zs_all.append(Z)
            if l == depth:
                zs_out = Z
                A = Z
            else:
                if kind == "none":
                    Nn = Z
                else:
                    y, cache = norm_forward([v for r in Z for v in r], groups, centre, gam[l], bet[l])
                    Nn = [y[i * width:(i + 1) * width] for i in range(n)]
                    caches.append(cache)
                Ns.append(Nn)
                A = [[_relu(v) for v in row] for row in Nn]
            As.append(A)
        loss = sum((r[0] - tt) ** 2 for r, tt in zip(zs_out, t)) / n
        losses.append(loss)
        if step in STACK_RECORD:
            rms_at[step] = [ini.rms(v for r in Ns[l] for v in r) for l in range(depth)]
            last_at[step] = [v for r in Ns[depth - 1] for v in r]
            z_at[step] = [ini.rms(v for r in zs_all[l] for v in r) for l in range(depth)]
        if not math.isfinite(loss) or loss > 1e30 or step == steps:
            break
        dZ = [[2.0 * (r[0] - tt) / n] for r, tt in zip(zs_out, t)]
        gW, gb, gg, gbt = [None] * (depth + 1), [None] * (depth + 1), [None] * depth, [None] * depth
        for l in range(depth, -1, -1):
            Dt, At = list(zip(*dZ)), list(zip(*As[l]))
            gW[l] = [[ini.dot(Dt[j], At[i]) for i in range(len(Ws[l][0]))] for j in range(len(Ws[l]))]
            gb[l] = [sum(Dt[j]) for j in range(len(Ws[l]))]
            if l == 0:
                break
            Wt = list(zip(*Ws[l]))
            dA = [[ini.dot(col, dn) for col in Wt] for dn in dZ]
            dN = [[da * (1.0 if v > 0.0 else 0.0) for da, v in zip(ra, rn)] for ra, rn in zip(dA, Ns[l - 1])]
            if kind == "none":
                dZ = dN
            else:
                dx, gg[l - 1], gbt[l - 1] = norm_backward([v for r in dN for v in r], caches[l - 1], groups, centre,
                                                          gam[l - 1])
                dZ = [dx[i * width:(i + 1) * width] for i in range(n)]
        if step == 0:
            db0 = [math.sqrt(sum(v * v for v in g)) for g in gb[:depth]]
        for l in range(depth + 1):
            for j in range(len(Ws[l])):
                Ws[l][j] = [w - lr * g for w, g in zip(Ws[l][j], gW[l][j])]
                bs[l][j] -= lr * gb[l][j]
        if kind != "none":
            for l in range(depth):
                gam[l] = [g - lr * d for g, d in zip(gam[l], gg[l])]
                if kind != "rms":                       # RMS normalization has no shift: one learned vector, gamma
                    bet[l] = [g - lr * d for g, d in zip(bet[l], gbt[l])]
    return losses, rms_at, last_at, db0, z_at


def _stack_setup(extra=""):
    return (f"Setup: the deep stack of the initialization chapter (10 layers of 64 units, a read-out; ReLU), He start "
            f"(the stream `initialization.weights` of that chapter, seed 0x{bookrng.seed_of(ini.WEIGHTS_NAME):016x}, scale sqrt(2 / 64) times each normal() in the order of that chapter's table: layer by layer, row by row, read-out last), "
            f"biases 0, the first {ini.N_TRAIN} inputs and targets of the stream `initialization.data` "
            f"(seed 0x{bookrng.seed_of(ini.DATA_NAME):016x}), mean squared error, full batch, plain gradient "
            f"descent{extra}.")


def stack_drift():
    """The problem: the rms of the pre-activation z of each layer of the plain stack, as it trains."""
    losses, _, _, _, rms_at_z = stack_run("none", 0.01)
    rows = []
    for s in STACK_RECORD:
        rows.append((s, _loss(losses[s])) + tuple(f"{rms_at_z[s][l - 1]:.2f}" for l in LAYERS_SHOWN))
    return (_stack_setup(" with rate 0.01 for 100 steps") +
            " Entries are the rms of the pre-activations z_l of the layer, over the 32 examples and 64 units; "
            "\"loss\" is the loss before that step.\n\n" +
            _table(["step", "loss"] + [f"rms z_{l}" for l in LAYERS_SHOWN], rows))


# -------------------------------------------------------------------- the idea

ZX = (2.0, 4.0, 4.0, 6.0, 9.0)


def zscore_row():
    n = len(ZX)
    mu = sum(ZX) / n
    v = sum((x - mu) ** 2 for x in ZX) / n
    rows = []
    for x in ZX:
        rows.append((len(rows), _g(x, 1), _g(x - mu, 1), _g((x - mu) / math.sqrt(v), 4),
                     _g((x - mu) / math.sqrt(v + EPS), 6)))
    xh = [(x - mu) / math.sqrt(v + EPS) for x in ZX]
    mean_h = sum(xh) / n
    var_h = sum((a - mean_h) ** 2 for a in xh) / n
    return (f"The five numbers x = {_vec(ZX, 0)}: mean mu = {_g(mu, 4)}, variance sigma2 = {_g(v, 4)} (divisor n = 5), "
            f"standard deviation {_g(math.sqrt(v), 4)}.\n\n" +
            _table(["i", "x_i", "x_i - mu", "(x_i - mu) / sqrt(sigma2)", f"(x_i - mu) / sqrt(sigma2 + {EPS:g})"], rows) +
            f"\n\nWith epsilon = {EPS:g} the standardized values have mean {_g(mean_h, 6)} (a rounding error) and "
            f"variance {_g(var_h, 8)}, which is sigma2 / (sigma2 + epsilon) = {_g(v / (v + EPS), 8)}.")


# ----------------------------------------------------------------- batch norm

BN_X = [[1.0, 10.0, -2.0], [3.0, 20.0, 0.0], [5.0, 10.0, 2.0], [7.0, 40.0, 4.0]]
BN_GAMMA = [1.0, 0.5, 2.0]
BN_BETA = [0.0, 1.0, -1.0]


def _rows(flat, d):
    return [flat[i:i + d] for i in range(0, len(flat), d)]


def batchnorm_forward():
    rows, d = len(BN_X), len(BN_X[0])
    flat = [v for r in BN_X for v in r]
    groups, centre = groups_for("batch", rows, d)
    y, (xhat, rs) = norm_forward(flat, groups, centre, BN_GAMMA, BN_BETA)
    mus = [sum(BN_X[i][j] for i in range(rows)) / rows for j in range(d)]
    vs = [sum((BN_X[i][j] - mus[j]) ** 2 for i in range(rows)) / rows for j in range(d)]
    xh = _rows(xhat, d)
    yy = _rows(y, d)
    out = [_table(["column j", "mean mu_j", "variance sigma2_j", "r_j = 1/sqrt(sigma2_j + eps)"],
                  [(j, _g(mus[j])) + (_g(vs[j]), _g(rs[j], 6)) for j in range(d)])]
    out.append(_table(["example i", "x_i", "xhat_i", "y_i = gamma * xhat_i + beta"],
                      [(i, _vec(BN_X[i], 1), _vec(xh[i]), _vec(yy[i])) for i in range(rows)], "lrrr"))
    cm = [sum(yy[i][j] for i in range(rows)) / rows for j in range(d)]
    cv = [sum((yy[i][j] - cm[j]) ** 2 for i in range(rows)) / rows for j in range(d)]
    return (f"Input x of shape [B, D] = [4, 3]; gamma = {_vec(BN_GAMMA, 1)}, beta = {_vec(BN_BETA, 1)}, "
            f"epsilon = {EPS:g}. Statistics are taken down each column (over the 4 examples), divisor 4.\n\n" +
            "\n\n".join(out) +
            f"\n\nColumn means of y: {_vec(cm, 4)} (= beta); column variances of y: {_vec(cv, 4)} "
            f"(= gamma^2 sigma2 / (sigma2 + epsilon) = {_vec([g * g * v / (v + EPS) for g, v in zip(BN_GAMMA, vs)], 4)}).")


# ------------------------------------------------------ running statistics, modes

POP_MEAN, POP_SD = 3.0, 2.0
RUN_B, BETA_RUN = 8, 0.9
RUN_STEPS = 40
RUN_NAME = "normalization.running"
RUN_RECORD = (1, 2, 5, 10, 20, 40)
PROBE = 5.0


@lru_cache(maxsize=None)
def _running_series():
    """Batch means and variances of 40 batches of 8 draws of 3 + 2 normal(), the running averages
    (start mean 0, variance 1, momentum 0.1), and the mean of the batch variance over 2000 more batches."""
    g = bookrng.generator(RUN_NAME)
    rm, rv = 0.0, 1.0
    series = {}
    for step in range(1, RUN_STEPS + 1):
        b = [POP_MEAN + POP_SD * g.normal() for _ in range(RUN_B)]
        bm = sum(b) / RUN_B
        bv = sum((v - bm) ** 2 for v in b) / RUN_B
        rm = BETA_RUN * rm + (1.0 - BETA_RUN) * bm
        rv = BETA_RUN * rv + (1.0 - BETA_RUN) * bv
        series[step] = (bm, bv, rm, rv)
    total = 0.0
    for _ in range(2000):
        b = [POP_MEAN + POP_SD * g.normal() for _ in range(RUN_B)]
        bm = sum(b) / RUN_B
        total += sum((v - bm) ** 2 for v in b) / RUN_B
    return series, total / 2000.0


def running_statistics():
    series, mean_bv = _running_series()
    rows = []
    for s in RUN_RECORD:
        bm, bv, rm, rv = series[s]
        rows.append((s, _g(bm), _g(bv), _g(rm), _g(rv), _g(BETA_RUN ** s, 4)))
    s0 = bookrng.seed_of(RUN_NAME)
    return (f"Setup: one feature whose values are 3 + 2 * normal() (mean 3, variance 4). {_gen_note(RUN_NAME)}; "
            f"batch t takes the next 8 draws in order. Running averages start at mean 0 and variance 1 and update "
            f"as run = beta_run * run + (1 - beta_run) * batch with beta_run = 0.9 (weight 0.1 on the new batch), batch variance with divisor 8.\n\n" +
            _table(["step t", "batch mean", "batch variance", "running mean", "running variance",
                    "weight of the start, beta_run^t"], rows) +
            f"\n\nThe mean of the batch variance over 2000 further batches of 8 (the next draws of the same stream) is "
            f"{_g(mean_bv)}; the variance of the source is 4 and 4 * 7/8 = 3.5.")


def _train_out(probe, batch):
    m = sum(batch) / len(batch)
    v = sum((x - m) ** 2 for x in batch) / len(batch)
    return (probe - m) / math.sqrt(v + EPS)


def _eval_out(probe, rm, rv):
    return (probe - rm) / math.sqrt(rv + EPS)


BATCH_SIZES = (1, 2, 4, 8, 32, 128)
BATCH_TRIALS = 1000
BATCH_NAME = "normalization.batchsize"


@lru_cache(maxsize=None)
def batch_size_data():
    """For each batch size B: the train-mode output of the probe over 1000 batches (probe first, then B - 1 draws)."""
    g = bookrng.generator(BATCH_NAME)
    out = {}
    for b in BATCH_SIZES:
        vals = []
        for _ in range(BATCH_TRIALS):
            batch = [PROBE] + [POP_MEAN + POP_SD * g.normal() for _ in range(b - 1)]
            vals.append(_train_out(PROBE, batch))
        out[b] = vals
    return out


def _quantile(sorted_vals, q):
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def batch_size_summary():
    """(B, mean, sd, p5, p95) of the train-mode output of the probe, per batch size."""
    rows = []
    for b, vals in batch_size_data().items():
        s = sorted(vals)
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        rows.append((b, m, sd, _quantile(s, 0.05), _quantile(s, 0.95)))
    return rows


def batch_size_dependence():
    rows = []
    for b, m, sd, p5, p95 in batch_size_summary():
        rows.append((b, _g(m), _g(sd), _g(p5), _g(p95)))
    ev = _eval_out(PROBE, POP_MEAN, POP_SD ** 2)
    return (f"Setup: the feature of the previous example (population mean 3, variance 4). The probe example has the "
            f"value {PROBE:g}, whose standardized value is ({PROBE:g} - 3) / 2 = 1. It is placed first in a batch of B values "
            f"(the other B - 1 are 3 + 2 * normal() draws, in order). Train mode normalizes with the batch's mean and "
            f"variance (divisor B, epsilon = {EPS:g}, gamma = 1, beta = 0). 1000 batches per B; 'sd of output' is the standard deviation of the 1000 outputs with divisor 1000, "
            "and the percentiles use the book's percentile rule. "
            f"{_gen_note(BATCH_NAME)}; sizes are run in the order 1, 2, 4, 8, 32, 128, continuing one stream.\n\n" +
            _table(["B", "mean of output", "sd of output", "5th percentile", "95th percentile"], rows) +
            f"\n\nEval mode with the exact population statistics (mean 3, variance 4) gives {_g(ev, 6)} for every batch.")


def eval_mode_bug():
    series, _ = _running_series()
    probe = PROBE
    rows = [
        ("train mode, a batch of one", "the batch's own", "0", _g(_train_out(probe, [probe]), 4)),
        ("eval mode, running statistics never updated", "0", "1", _g(_eval_out(probe, 0.0, 1.0), 4)),
        ("eval mode, after 5 updates", _g(series[5][2]), _g(series[5][3]), _g(_eval_out(probe, series[5][2], series[5][3]), 4)),
        ("eval mode, after 40 updates", _g(series[40][2]), _g(series[40][3]), _g(_eval_out(probe, series[40][2], series[40][3]), 4)),
        ("eval mode, the exact statistics", "3", "4", _g(_eval_out(probe, POP_MEAN, POP_SD ** 2), 4)),
    ]
    return (f"The probe value {PROBE:g} through a batch-norm layer with gamma = 1, beta = 0, epsilon = {EPS:g}; the running "
            f"statistics are those of the table above (beta_run = 0.9). The correct standardized value is 1.\n\n" +
            _table(["situation", "mean used", "variance used", "output for the probe"], rows, "lrrr"))


# --------------------------------------------------------- gradient accumulation

GA_NAME = "normalization.accumulation"
GA_B, GA_D, GA_MICRO = 32, 4, 8
GA_GAMMA = [1.0, 1.5, 0.5, 2.0]
GA_BETA = [0.0, 0.5, -0.5, 1.0]


def accumulation():
    g = bookrng.generator(GA_NAME)
    x = [2.0 + 1.5 * g.normal() for _ in range(GA_B * GA_D)]
    t = [g.normal() for _ in range(GA_B * GA_D)]

    def dgamma(rows_idx, kind):
        """Gradient of the mean over the chosen rows' elements of (y - t)^2 with respect to gamma."""
        xs = [x[i * GA_D + j] for i in rows_idx for j in range(GA_D)]
        ts = [t[i * GA_D + j] for i in rows_idx for j in range(GA_D)]
        groups, centre = groups_for(kind, len(rows_idx), GA_D)
        y, cache = norm_forward(xs, groups, centre, GA_GAMMA, GA_BETA)
        dy = [2.0 * (a - b) / len(y) for a, b in zip(y, ts)]
        return norm_backward(dy, cache, groups, centre, GA_GAMMA)[1]

    rows = []
    for kind in ("batch", "layer"):
        full = dgamma(list(range(GA_B)), kind)
        micro = [dgamma(list(range(k * GA_MICRO, (k + 1) * GA_MICRO)), kind) for k in range(GA_B // GA_MICRO)]
        acc = [sum(m[j] for m in micro) / len(micro) for j in range(GA_D)]
        rows.append((KIND_LABEL[kind], _vec(full), _vec(acc), _sci(max(abs(a - b) for a, b in zip(full, acc)))))
    return (f"Setup: B = {GA_B} examples of D = {GA_D} features, x = 2 + 1.5 * normal() and targets t = normal(), both drawn "
            f"in row-major order, x first. {_gen_note(GA_NAME)}. The layer applies the normalization then "
            f"y = gamma * xhat + beta with gamma = {_vec(GA_GAMMA, 1)}, beta = {_vec(GA_BETA, 1)}; the loss is the mean "
            f"of (y - t)^2 over all elements. \"Accumulated\" is the mean of the gradients of the four micro-batches of "
            f"{GA_MICRO} rows, each with the mean loss of its own rows, as in the gradient-accumulation section of chapter 5.\n\n" +
            _table(["layer", "d loss / d gamma, one batch of 32", "mean over 4 micro-batches of 8", "largest difference"], rows,
                   "lrrr"))


# ------------------------------------------------------------------- the axes

def norm_axes():
    def row(name, shape_txt, groups, per, params):
        return (name, shape_txt, groups, per, params)
    b, c, h, w, gcount = 8, 64, 16, 16, 32
    t, dm = 128, 512
    rows = [
        row("batch norm", f"[{b}, {c}, {h}, {w}]", f"C = {c}", f"B*H*W = {b * h * w}", f"2C = {2 * c}"),
        row("layer norm (over C, H, W)", f"[{b}, {c}, {h}, {w}]", f"B = {b}", f"C*H*W = {c * h * w}", f"2C = {2 * c}"),
        row(f"group norm, G = {gcount}", f"[{b}, {c}, {h}, {w}]", f"B*G = {b * gcount}", f"(C/G)*H*W = {c // gcount * h * w}",
            f"2C = {2 * c}"),
        row("instance norm (G = C)", f"[{b}, {c}, {h}, {w}]", f"B*C = {b * c}", f"H*W = {h * w}", f"2C = {2 * c}"),
        row("channels-first layer norm (over C at each position)", f"[{b}, {c}, {h}, {w}]", f"B*H*W = {b * h * w}", f"C = {c}", f"2C = {2 * c}"),
        row("batch norm", f"[{b}, {t}, {dm}]", f"D = {dm}", f"B*T = {b * t}", f"2D = {2 * dm}"),
        row("layer norm", f"[{b}, {t}, {dm}]", f"B*T = {b * t}", f"D = {dm}", f"2D = {2 * dm}"),
        row("RMS norm", f"[{b}, {t}, {dm}]", f"B*T = {b * t}", f"D = {dm}", f"D = {dm}"),
    ]
    return _table(["layer", "input shape", "number of groups", "values per group", "learned numbers (gamma, beta)"], rows,
                  "lllll")


GN_NAME = "normalization.groupnorm"
GN_SHAPE = (2, 4, 2, 2)    # N, C, H, W


@lru_cache(maxsize=None)
def _gn_input():
    g = bookrng.generator(GN_NAME)
    n, c, h, w = GN_SHAPE
    return [float(g.integer(10)) for _ in range(n * c * h * w)]


def _gn_groups(groups_per_sample):
    """Flat-index groups of the [N, C, H, W] tensor for G groups of C/G channels per example."""
    n, c, h, w = GN_SHAPE
    cpg = c // groups_per_sample
    out = []
    for i in range(n):
        for k in range(groups_per_sample):
            out.append([((i * c + ch) * h + hh) * w + ww for ch in range(k * cpg, (k + 1) * cpg)
                        for hh in range(h) for ww in range(w)])
    return out


def _bn_groups_4d():
    n, c, h, w = GN_SHAPE
    return [[((i * c + ch) * h + hh) * w + ww for i in range(n) for hh in range(h) for ww in range(w)] for ch in range(c)]


def _gn_apply(x, groups):
    n, c, h, w = GN_SHAPE
    # parameters are per channel: the channel of flat index i is (i // (h * w)) mod c
    xhat = [0.0] * len(x)
    for idx in groups:
        mu = sum(x[i] for i in idx) / len(idx)
        v = sum((x[i] - mu) ** 2 for i in idx) / len(idx)
        r = 1.0 / math.sqrt(v + EPS)
        for i in idx:
            xhat[i] = (x[i] - mu) * r
    return xhat


def groupnorm_example():
    n, c, h, w = GN_SHAPE
    x = _gn_input()
    parts = []
    for i in range(n):
        rows = []
        for ch in range(c):
            base = (i * c + ch) * h * w
            rows.append((ch, *(f"{x[base + k]:g}" for k in range(h * w))))
        parts.append(f"Example {i} (rows are channels c, columns are the positions (h, w) = (0,0), (0,1), (1,0), (1,1)):\n\n" +
                     _table(["c", "(0,0)", "(0,1)", "(1,0)", "(1,1)"], rows))
    stat_rows = []
    for gcount in (1, 2, 4):
        groups = _gn_groups(gcount)
        cpg = c // gcount
        for k in range(gcount):
            idx = groups[k]    # example 0
            vals = [x[i] for i in idx]
            mu = sum(vals) / len(vals)
            v = sum((a - mu) ** 2 for a in vals) / len(vals)
            stat_rows.append((gcount, k, (f"{k * cpg} .. {(k + 1) * cpg - 1}" if cpg > 1 else f"{k}"), len(vals), _g(mu), _g(v)))
    inv_rows = []
    for name, groups in (("batch norm (per channel over N, H, W)", _bn_groups_4d()),
                         ("group norm, G = 1 (= layer norm over C, H, W)", _gn_groups(1)),
                         ("group norm, G = 2", _gn_groups(2)),
                         ("group norm, G = 4 (= instance norm)", _gn_groups(4))):
        base = _gn_apply(x, groups)
        x2 = list(x)
        for k in range(c * h * w, 2 * c * h * w):    # replace example 1 by zeros
            x2[k] = 0.0
        alt = _gn_apply(x2, groups)
        inv_rows.append((name, _sci(max(abs(a - b) for a, b in zip(base[:c * h * w], alt[:c * h * w])))))
    return ("Input x of shape [N, C, H, W] = [2, 4, 2, 2]: 32 integers, each `integer(10)` of the stream below in "
            f"row-major order. {_gen_note(GN_NAME)}.\n\n" + "\n\n".join(parts) +
            "\n\nStatistics of example 0 (divisor = values per group):\n\n" +
            _table(["groups per example G", "group k", "channels", "values", "mean", "variance"], stat_rows) +
            "\n\nLargest change of the normalized example 0 when example 1 is replaced by zeros (epsilon = "
            f"{EPS:g}):\n\n" + _table(["layer", "largest change"], inv_rows, "lr"))


# --------------------------------------------------------- the backward derivation

BG_X = (1.0, 3.0, 2.0, 6.0)
BG_G = (1.0, -2.0, 0.5, 3.0)


def backward_graph():
    n = len(BG_X)
    x, g = list(BG_X), list(BG_G)
    mu = sum(x) / n
    c = [a - mu for a in x]
    v = sum(a * a for a in c) / n
    r = 1.0 / math.sqrt(v + EPS)
    xh = [a * r for a in c]
    # backward node by node, with the rules of chapter 5
    dr = sum(gi * ci for gi, ci in zip(g, c))                 # xhat_i = c_i * r : r is broadcast, so its gradient is a sum
    dc_direct = [gi * r for gi in g]                          # and c_i gets g_i * r
    dv = dr * (-0.5) * r ** 3                                 # r = (v + eps)^(-1/2)
    dc_var = [dv * 2.0 * ci / n for ci in c]                  # v = mean(c^2): the backward of a mean spreads 1/n
    dc = [a + b for a, b in zip(dc_direct, dc_var)]
    dmu = -sum(dc)                                            # c_i = x_i - mu : mu is broadcast, minus sign, sum
    dx_path = [dci + dmu / n for dci in dc]                   # x_i gets dc_i directly and dmu / n through the mean
    # closed form
    m1 = sum(g) / n
    m2 = sum(gi * a for gi, a in zip(g, xh)) / n
    closed = [r * (gi - m1 - a * m2) for gi, a in zip(g, xh)]
    # central difference of L = sum g_i xhat_i(x)
    def loss(xx):
        m = sum(xx) / n
        vv = sum((a - m) ** 2 for a in xx) / n
        rr = 1.0 / math.sqrt(vv + EPS)
        return sum(gi * (a - m) * rr for gi, a in zip(g, xx))
    cd = _central(loss, x, 1e-6)
    fwd_rows = [(i, _g(x[i], 1), _g(c[i], 1), _g(xh[i], 5)) for i in range(n)]
    bwd_rows = [(i, _g(g[i], 1), _g(dc_direct[i], 5), _g(dc_var[i], 5), _g(dc[i], 5), _g(dmu / n, 5),
                 _g(dx_path[i], 6), _g(closed[i], 6), _g(cd[i], 6)) for i in range(n)]
    sdx = sum(dx_path)
    sdxh = sum(a * b for a, b in zip(dx_path, xh))
    return (f"One group of n = {n} values x = {_vec(x, 1)}, epsilon = {EPS:g}, and an arriving gradient dxhat = {_vec(g, 1)} "
            f"at xhat (so the loss is L = sum_i dxhat_i * xhat_i). Forward: mu = {_g(mu, 4)}, sigma2 = {_g(v, 4)}, "
            f"r = {_g(r, 6)}. Backward: dr = sum_i dxhat_i c_i = {_g(dr, 5)}, dsigma2 = -r^3 dr / 2 = {_g(dv, 6)}, "
            f"dmu = -sum_i dc_i = {_g(dmu, 6)}.\n\n" +
            _table(["i", "x_i", "c_i = x_i - mu", "xhat_i"], fwd_rows) + "\n\n" +
            _table(["i", "dxhat_i", "dxhat_i r", "dsigma2 * 2 c_i / n", "dc_i", "dmu / n", "dx_i (graph)",
                    "dx_i (closed form)", "central difference"], bwd_rows) +
            f"\n\nSum of dx_i: {_sci(sdx)}. Sum of dx_i * xhat_i: {_sci(sdxh)}; the closed form says it equals "
            f"r * n * mean(dxhat xhat) * epsilon * r^2 = {_sci(r * n * m2 * EPS * r * r)}. Largest difference between the graph, "
            f"the closed form and the central difference (step 1e-6): "
            f"{_sci(max(max(abs(a - b), abs(a - d)) for a, b, d in zip(dx_path, closed, cd)))}.")


GC_NAME = "normalization.gradcheck"
GC_ROWS, GC_D = 4, 4


def _gc_forward_cases():
    d = GC_D
    cases = {}
    for kind in KINDS:
        cases[kind] = groups_for(kind, GC_ROWS, d)
    half = d // 2
    cases["group (2 per row)"] = ([[i * d + k * half + j for j in range(half)] for i in range(GC_ROWS) for k in range(2)], True)
    return cases


def _noise(v):
    """A central-difference error is rounding noise of size about 1e-10: only its size is reproducible."""
    return "below 1e-8" if v < 1e-8 else _sci(v)


def _below(v, name):
    """The sum of dx over a group is zero up to rounding for the centred layers; shown for contrast for RMS norm."""
    return _sci(v) if name == "rms" else "below 1e-14"


def norm_gradcheck():
    g = bookrng.generator(GC_NAME)
    n = GC_ROWS * GC_D
    x = [g.normal() for _ in range(n)]
    gamma = [1.0 + 0.3 * g.normal() for _ in range(GC_D)]
    beta = [0.3 * g.normal() for _ in range(GC_D)]
    dy = [g.normal() for _ in range(n)]
    h = 1e-6
    rows = []
    for name, (groups, centre) in _gc_forward_cases().items():
        bta = [0.0] * GC_D if name == "rms" else beta       # RMS normalization has no shift
        def loss(xx, gg=gamma, bb=bta):
            y, _ = norm_forward(xx, groups, centre, gg, bb)
            return sum(a * b for a, b in zip(dy, y))
        _, cache = norm_forward(x, groups, centre, gamma, bta)
        dx, dgam, dbet = norm_backward(dy, cache, groups, centre, gamma)
        ndx = _central(loss, x, h)
        ndg = _central(lambda gg: loss(x, gg, bta), gamma, h)
        ndb = _central(lambda bb: loss(x, gamma, bb), bta, h)
        dbeta_err = "none (no shift)" if name == "rms" else _noise(max(abs(a - b) for a, b in zip(dbet, ndb)))
        rows.append((name, _noise(max(abs(a - b) for a, b in zip(dx, ndx))), _noise(max(abs(a - b) for a, b in zip(dgam, ndg))),
                     dbeta_err, _below(abs(sum(dx[i] for i in groups[0])), name)))
    return (f"Setup: a [4, 4] array x, gamma (1 + 0.3 * normal() per column), beta (0.3 * normal() per column) and an arriving "
            f"gradient dy, all drawn in that order, x first, row-major. {_gen_note(GC_NAME)}. The loss is L = sum(dy * y), "
            f"epsilon = {EPS:g}, central differences with step {h:g}; the differences of a central difference are rounding noise, so values under 1e-8 are printed as a bound. "
            f"\"Sum of dx over a group\" is for the first group (column 0 for batch norm, row 0 otherwise); for RMS norm it is not zero and is shown for contrast. "
            f"RMS norm has no shift, so its beta is zero and it has no dbeta.\n\n" +
            _table(["layer", "largest |dx - numeric|", "largest |dgamma - numeric|", "largest |dbeta - numeric|",
                    "sum of dx over a group"], rows))


# ------------------------------------------------------ scale invariance of the weights

SI_NAME = "normalization.scaleinvariance"


def scale_invariance():
    g = bookrng.generator(SI_NAME)
    b, din, dout = 8, 5, 4
    x = [g.normal() for _ in range(b * din)]
    w = [g.normal() / math.sqrt(din) for _ in range(dout * din)]
    dy = [g.normal() for _ in range(b * dout)]
    gamma, beta = [1.0] * dout, [0.0] * dout
    groups, centre = groups_for("layer", b, dout)

    def layer(wt):
        z = [sum(x[i * din + k] * wt[j * din + k] for k in range(din)) for i in range(b) for j in range(dout)]
        y, cache = norm_forward(z, groups, centre, gamma, beta)
        return y, cache

    def dw_of(wt):
        _, cache = layer(wt)
        dz, _, _ = norm_backward(dy, cache, groups, centre, gamma)
        return [sum(dz[i * dout + j] * x[i * din + k] for i in range(b)) for j in range(dout) for k in range(din)]

    y1, _ = layer(w)
    dw1 = dw_of(w)
    n1 = math.sqrt(sum(a * a for a in dw1))
    rows = []
    for a in (0.1, 0.5, 1.0, 2.0, 10.0):
        wa = [a * v for v in w]
        ya, _ = layer(wa)
        dwa = dw_of(wa)
        na = math.sqrt(sum(v * v for v in dwa))
        inner = sum(p * q for p, q in zip(wa, dwa))
        rows.append((_g(a, 1), _sci(max(abs(p - q) for p, q in zip(y1, ya))), _g(na / n1, 4), _g(1.0 / a, 4), _sci(abs(inner))))
    return (f"Setup: a layer z = x W^T (no bias) with x of shape [8, 5] and W of shape [4, 5], followed by layer norm "
            f"(gamma = 1, beta = 0, epsilon = {EPS:g}), and the loss L = sum(dy * y). Draws in order: x (40 values, "
            f"normal()), W (20 values, normal() / sqrt(5)), dy (32 values, normal()). {_gen_note(SI_NAME)}. "
            f"W is multiplied by a.\n\n" +
            _table(["a", "largest change of y", "|dW(aW)| / |dW(W)|", "1 / a", "|sum of W * dW|"], rows))


# --------------------------------------------------------------- the epsilon

EPS_BASE = (0.5, -1.5, 1.0, 2.0, -0.5, 1.5, -1.0, 0.0)
EPS_SCALES = (1.0, 0.1, 0.01, 0.003, 0.001, 0.0001)


def _ln_row(x, eps):
    n = len(x)
    mu = sum(x) / n
    v = sum((a - mu) ** 2 for a in x) / n
    r = 1.0 / math.sqrt(v + eps)
    return [(a - mu) * r for a in x], v, r


def epsilon_effect():
    n = len(EPS_BASE)
    rows = []
    for s in EPS_SCALES:
        x = [s * a for a in EPS_BASE]
        y5, v, r5 = _ln_row(x, 1e-5)
        y6, _, r6 = _ln_row(x, 1e-6)
        rel = math.sqrt(sum((a - b) ** 2 for a, b in zip(y5, y6)) / sum(b * b for b in y6))
        rows.append((_g(s, 4), _g(math.sqrt(v), 6), _g(math.sqrt(sum(a * a for a in y5) / n), 4),
                     _g(math.sqrt(sum(a * a for a in y6) / n), 4), _sci(rel)))
    return (f"Setup: layer norm (gamma = 1, beta = 0) of the row s * u with u = {_vec(EPS_BASE, 1)} (mean "
            f"{_g(sum(EPS_BASE) / n, 4)}, sd with divisor n = 8 {_g(math.sqrt(sum((a - sum(EPS_BASE) / n) ** 2 for a in EPS_BASE) / n), 4)}), for a scale s. \"Relative difference\" is "
            f"the Euclidean norm of the difference between the two outputs divided by the norm of the output at epsilon = 1e-6.\n\n" +
            _table(["scale s", "sd of the row", "rms of output, eps = 1e-5", "rms of output, eps = 1e-6", "relative difference"],
                   rows))


# ------------------------------------------------------------------ training table

def _stable_cell(v, w):
    """A loss in the table's precision when a change of 1e-12 in one starting weight leaves its third significant digit unchanged
    (a loss that has overflowed counts as 'not finite'), otherwise only the leading digits that survive, or 'unstable'."""
    v = v if v is not None and math.isfinite(v) and v < 1e30 else None
    w = w if w is not None and math.isfinite(w) and w < 1e30 else None
    return ini._stable_loss(v, w).replace("not finite", "not finite")


def stack_training():
    rows = []
    for lr in STACK_RATES:
        for kind in ("none", "batch", "layer", "rms"):
            losses = stack_run(kind, lr)[0]
            twin = stack_run(kind, lr, nudge=ini.NUDGE)[0]
            cells = []
            for s in (0, 10, 30, 100):
                cells.append(_stable_cell(losses[s] if s < len(losses) else None, twin[s] if s < len(twin) else None))
            rows.append((lr, KIND_LABEL[kind]) + tuple(cells))
    return (_stack_setup(", 100 steps") + " A normalization (gamma = 1 and, for batch and layer norm, beta = 0 at the start, trained with the same rate, "
            f"epsilon = {EPS:g}) sits between each of the ten linear layers and its ReLU; batch norm takes its statistics "
            "over the 32 examples, layer norm and RMS norm over the 64 units of an example. "
            "\"not finite\" means the loss overflowed or exceeded 1e30 before that step. A loss that changes in its third significant digit when the first weight of the first layer is changed by 1e-12 "
            "is printed with the leading digits that survive ('about 0.45') or as 'unstable'. RMS norm has only the vector gamma (no shift beta); batch norm and layer norm have gamma and beta, trained with the same rate; "
            "the weights, biases and gamma and beta are all updated by plain gradient descent.\n\n" +
            _table(["rate", "layers", "step 0", "step 10", "step 30", "step 100"], rows, "rlrrrr"))


def scale_by_layer():
    rows = []
    cases = [("none", 0), ("none", 10), ("none", 100), ("batch", 10), ("layer", 10), ("rms", 10)]
    for kind, step in cases:
        z = stack_run(kind, 0.01)[4][step]
        label = KIND_LABEL[kind] + ("" if kind == "none" else ": z before the norm")
        rows.append((label, step) + tuple(f"{z[l - 1]:.2f}" for l in LAYERS_SHOWN))
    for kind in KINDS:
        out = stack_run(kind, 0.01)[1][10]
        rows.append((KIND_LABEL[kind] + ": output of the norm", 10) + tuple(f"{out[l - 1]:.2f}" for l in LAYERS_SHOWN))
    return (_stack_setup(" with rate 0.01") + " Entries are root-mean-square values over the 32 examples and 64 units of "
            "the layer: of the pre-activation z_l = W_l a_{l-1} + b_l, and, for the normalized stacks, of the output "
            "of the normalization layer, which is the input of the ReLU.\n\n" +
            _table(["layers", "step"] + [f"layer {l}" for l in LAYERS_SHOWN], rows, "lr" + "r" * len(LAYERS_SHOWN)))


def bias_before_norm():
    rows = []
    for kind in ("none", "batch", "layer", "rms"):
        db = stack_run(kind, 0.01)[3]
        rows.append((KIND_LABEL[kind],) + tuple(("below 1e-14" if db[l - 1] < 1e-14 else _sci(db[l - 1])) for l in (1, 5, 10)))
    return (_stack_setup() + " Entries are the Euclidean norm of the gradient of the bias vector b_l (64 numbers) at step 0; "
            "a value at the level of rounding error (under 1e-14) is printed as a bound.\n\n" +
            _table(["layers", "layer 1", "layer 5", "layer 10"], rows))


def figure_scales():
    """(label, rms per layer) series of the scale-by-layer figure: the ReLU input of the plain stack at three
    steps, and of the normalized stacks at step 10 (the three kinds coincide to the digits shown)."""
    out = []
    for step in (0, 10, 100):
        out.append((f"no normalization, step {step}", stack_run("none", 0.01)[1][step]))
    for kind in KINDS:
        out.append((f"{KIND_LABEL[kind]}, step 10", stack_run(kind, 0.01)[1][10]))
    return out


def figure_histogram():
    """(label, values) of the ReLU input of layer 10 over the 32 examples and 64 units: plain stack at steps 0 and 10,
    batch-normalized stack at step 10."""
    return [("no normalization, step 0", stack_run("none", 0.01)[2][0]),
            ("no normalization, step 10", stack_run("none", 0.01)[2][10]),
            ("batch norm, step 10", stack_run("batch", 0.01)[2][10])]


def figure_losses(lr):
    return {kind: stack_run(kind, lr)[0] for kind in ("none", "batch", "layer", "rms")}


# --------------------------------------------------------------- work and memory

def _flops_ln(d):
    """Forward flops of one row of layer norm as (a, b) in a * d + b: one add, subtract, multiply, divide or
    square root counts 1; a reduction of d values is d - 1 adds; r = 1 / sqrt(.) is a square root and a division."""
    items = [("mean: sum of d values", (1, -1)), ("mean: divide by d", (0, 1)), ("c = x - mu", (1, 0)),
             ("v: squares", (1, 0)), ("v: sum", (1, -1)), ("v: divide by d", (0, 1)), ("v + eps", (0, 1)),
             ("r: sqrt and reciprocal", (0, 2)), ("xhat = c * r", (1, 0)), ("gamma * xhat", (1, 0)), ("+ beta", (1, 0))]
    return items


def _flops_rms(d):
    items = [("squares", (1, 0)), ("sum", (1, -1)), ("divide by d", (0, 1)), ("+ eps", (0, 1)),
             ("r: sqrt and reciprocal", (0, 2)), ("xhat = x * r", (1, 0)), ("gamma * xhat", (1, 0))]
    return items


def _bwd_ln():
    return [("g = dy * gamma", (1, 0)), ("mean(g): sum and divide", (1, 0)), ("g * xhat", (1, 0)),
            ("mean(g xhat): sum and divide", (1, 0)), ("xhat * mean(g xhat)", (1, 0)), ("g - mean(g)", (1, 0)),
            ("subtract the product", (1, 0)), ("times r", (1, 0)), ("dgamma: dy * xhat and accumulate", (2, 0)),
            ("dbeta: accumulate", (1, 0))]


def _bwd_rms():
    return [("g = dy * gamma", (1, 0)), ("g * xhat", (1, 0)), ("mean(g xhat): sum and divide", (1, 0)),
            ("xhat * mean(g xhat)", (1, 0)), ("g - the product", (1, 0)), ("times r", (1, 0)),
            ("dgamma: dy * xhat and accumulate", (2, 0))]


def _total(items):
    return (sum(a for _, (a, b) in items), sum(b for _, (a, b) in items))


def _fmt_ab(ab):
    a, b = ab
    if b == 0:
        return f"{a}d"
    return f"{a}d {'+' if b > 0 else '-'} {abs(b)}"


def norm_costs():
    d1, d2 = 64, 4096
    fl_ln, fl_rms = _total(_flops_ln(0)), _total(_flops_rms(0))
    bw_ln, bw_rms = _total(_bwd_ln()), _total(_bwd_rms())

    def val(ab, d):
        return ab[0] * d + ab[1]
    rows = [
        ("layer norm", "forward", _fmt_ab(fl_ln), val(fl_ln, d1), val(fl_ln, d2)),
        ("RMS norm", "forward", _fmt_ab(fl_rms), val(fl_rms, d1), val(fl_rms, d2)),
        ("layer norm", "backward", _fmt_ab(bw_ln), val(bw_ln, d1), val(bw_ln, d2)),
        ("RMS norm", "backward", _fmt_ab(bw_rms), val(bw_rms, d1), val(bw_rms, d2)),
    ]
    mm1, mm2 = 2 * d1 * d1, 2 * d2 * d2
    # traffic of one forward call over R rows: x in, y out, parameters read once, 4 bytes per number
    r_rows = 1024
    t_ln = 4 * (2 * r_rows * d2 + 2 * d2)
    t_rms = 4 * (2 * r_rows * d2 + d2)
    return ("Counting rule: one addition, subtraction, multiplication, division or square root is one flop; a sum of d "
            "numbers is d - 1 additions; r = 1 / sqrt(v + eps) is a square root and a division. Per row of d numbers:\n\n"
            + _table(["layer", "pass", "flops per row", "d = 64", "d = 4096"], rows, "llrrr") +
            f"\n\nA linear layer of width d to d costs 2d^2 flops per row (d multiplications and d additions per output): "
            f"{mm1} at d = 64 and {mm2} at d = 4096. The ratio of forward layer norm to it is {val(fl_ln, d1) / mm1:.4f} at d = 64 and "
            f"{val(fl_ln, d2) / mm2:.5f} at d = 4096; the ratio of RMS norm forward to layer norm forward is "
            f"{val(fl_rms, d2) / val(fl_ln, d2):.3f} at d = 4096.\n\n"
            f"Memory traffic of one forward call on R = {r_rows} rows of d = 4096 in 32-bit numbers: x read, y written, parameters "
            f"read once: layer norm {t_ln} bytes ({t_ln / 2**20:.2f} MiB), RMS norm {t_rms} bytes ({t_rms / 2**20:.2f} MiB). "
            f"Flops per byte: layer norm {val(fl_ln, d2) * r_rows / t_ln:.2f}, RMS norm {val(fl_rms, d2) * r_rows / t_rms:.2f}, "
            f"against {mm2 * r_rows / (4 * (2 * r_rows * d2 + d2 * d2)):.1f} for the linear layer on the same rows "
            f"(x in, y out, the d x d weight read once).")


def fused_front_end():
    """Words moved by the chain add, RMS norm, max |.| per group, int8 pack against one fused pass."""
    d = 5120
    chain = [("add: read a, b; write sum", 2 * d, d), ("RMS norm: read sum, gain w; write xn", 2 * d, d),
             ("group maxima: read xn; write d/32 scales", d, d // 32),
             ("pack: read xn and the scales; write d/4 words", d + d // 32, d // 4)]
    fused = (3 * d, d + d + d // 4 + d // 32)
    rows = [(n, r, w, r + w) for n, r, w in chain]
    tot = (sum(r for _, r, _ in chain), sum(w for _, _, w in chain))
    rows.append(("the four passes", tot[0], tot[1], sum(tot)))
    rows.append(("one fused pass: read a, b, w; write sum, xn, d/4 words, d/32 scales", fused[0], fused[1], sum(fused)))
    return (f"One row of d = {d} numbers, counted in 4-byte words, each pass reading and writing exactly the arrays named, "
            "once, with no reuse between passes:\n\n" +
            _table(["pass", "words read", "words written", "words moved"], rows) +
            f"\n\nThe ratio of the words moved by the fused pass to those of the chain is {sum(fused) / sum(tot):.3f}; the arithmetic is the same.")


# ------------------------------------------------------------------ folding batch norm

FOLD_NAME = "normalization.fold"


def fold_batchnorm():
    g = bookrng.generator(FOLD_NAME)
    din, dout = 3, 2
    w = [[g.normal() for _ in range(din)] for _ in range(dout)]
    b = [g.normal() for _ in range(dout)]
    gamma = [1.0 + 0.3 * g.normal() for _ in range(dout)]
    beta = [0.3 * g.normal() for _ in range(dout)]
    mu = [g.normal() for _ in range(dout)]
    var = [1.0 + 0.5 * abs(g.normal()) for _ in range(dout)]
    xs = [[g.normal() for _ in range(din)] for _ in range(5)]
    s = [gamma[j] / math.sqrt(var[j] + EPS) for j in range(dout)]
    wf = [[s[j] * w[j][k] for k in range(din)] for j in range(dout)]
    bf = [s[j] * (b[j] - mu[j]) + beta[j] for j in range(dout)]
    rows, worst = [], 0.0
    for x in xs:
        z = [sum(w[j][k] * x[k] for k in range(din)) + b[j] for j in range(dout)]
        y = [gamma[j] * (z[j] - mu[j]) / math.sqrt(var[j] + EPS) + beta[j] for j in range(dout)]
        yf = [sum(wf[j][k] * x[k] for k in range(din)) + bf[j] for j in range(dout)]
        worst = max(worst, max(abs(p - q) for p, q in zip(y, yf)))
        rows.append((_vec(x, 3), _vec(y, 6), _vec(yf, 6)))
    # fold the gain of a norm into the next linear layer: W (gamma * xhat) = (W diag(gamma)) xhat
    w2 = [[g.normal() for _ in range(din)] for _ in range(dout)]
    xhat = xs[0]
    gg = [1.0 + 0.3 * g.normal() for _ in range(din)]
    lhs = [sum(w2[j][k] * (gg[k] * xhat[k]) for k in range(din)) for j in range(dout)]
    rhs = [sum((w2[j][k] * gg[k]) * xhat[k] for k in range(din)) for j in range(dout)]
    d_in, k2 = 64, 9
    flops_extra, bytes_extra = 3, 8
    conv = 2 * d_in * k2
    return (f"Setup: a linear layer with W of shape [2, 3] and bias b, followed by batch norm in eval mode with running "
            f"mean mu, running variance var, gamma, beta and epsilon = {EPS:g}. Draws in order: W row by row (6), b (2), gamma "
            f"= 1 + 0.3 * normal() (2), beta = 0.3 * normal() (2), mu (2, normal()), var = 1 + 0.5 * |normal()| (2), then five "
            f"inputs of 3 values. {_gen_note(FOLD_NAME)}.\n\n"
            f"W = {[ [round(v, 4) for v in r] for r in w]}, b = {_vec(b)}, gamma = {_vec(gamma)}, beta = {_vec(beta)}, "
            f"mu = {_vec(mu)}, var = {_vec(var)}.\n\n"
            f"s = gamma / sqrt(var + eps) = {_vec(s, 6)}; W' = diag(s) W = {[ [round(v, 6) for v in r] for r in wf]}; "
            f"b' = s * (b - mu) + beta = {_vec(bf, 6)}.\n\n" +
            _table(["input x", "batch norm after the layer", "folded layer W' x + b'"], rows, "lrr") +
            f"\n\nLargest difference over the five inputs: {_sci(worst)}. The gain of a norm folds into the next layer in "
            f"the same way: for the second layer W2 of shape [2, 3] and gain gamma_g, W2 (gamma_g * xhat) and "
            f"(W2 diag(gamma_g)) xhat differ by {_sci(max(abs(p - q) for p, q in zip(lhs, rhs)))}.\n\n"
            f"Work saved per output element of a 3x3 convolution with 64 input channels: the convolution costs "
            f"2 * 64 * 9 = {conv} flops; the separate eval-mode batch norm then costs {flops_extra} flops "
            f"(subtract, multiply, add, with s precomputed) and one extra pass that reads and writes the element "
            f"({bytes_extra} bytes in 32-bit numbers); folding removes both.")


# ----------------------------------------------------- variance in one pass (32-bit)

def _f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


VAR_OFFSETS = (0.0, 10.0, 100.0, 1000.0, 10000.0)
VAR_U = (0.1, 0.4, 0.2, 0.7, 0.3, 0.6, 0.5, 0.0)


def variance_one_pass():
    """Three ways to get the variance of c + u in 32-bit arithmetic, rounding after every operation."""
    n = len(VAR_U)
    rows = []
    for c in VAR_OFFSETS:
        x = [_f32(c + a) for a in VAR_U]
        s1 = s2 = 0.0
        for a in x:
            s1 = _f32(s1 + a)
            s2 = _f32(s2 + _f32(a * a))
        mean = _f32(s1 / n)
        naive = _f32(_f32(s2 / n) - _f32(mean * mean))
        acc = 0.0
        for a in x:
            dv = _f32(a - mean)
            acc = _f32(acc + _f32(dv * dv))
        two_pass = _f32(acc / n)
        k = x[0]
        t1 = t2 = 0.0
        for a in x:
            dv = _f32(a - k)
            t1 = _f32(t1 + dv)
            t2 = _f32(t2 + _f32(dv * dv))
        m1 = _f32(t1 / n)
        shifted = _f32(_f32(t2 / n) - _f32(m1 * m1))
        rows.append((_g(c, 0) if c else "0", _g(naive, 4), _g(two_pass, 4), _g(shifted, 4)))
    true = sum((a - sum(VAR_U) / n) ** 2 for a in VAR_U) / n
    return (f"The row x = c + u with u = {_vec(VAR_U, 1)} has variance {_g(true, 4)} for every c (in exact arithmetic; the inputs themselves are rounded to 32 bits). Each entry is computed in "
            "32-bit floating point, rounding after every addition, subtraction, multiplication and division, with divisor 8. "
            "Naive: mean of squares minus square of the mean. Two-pass: the mean first, then the mean of the squared deviations. "
            "Shifted: the same formula as the naive one applied to x - x_0, where x_0 is the first element of the row.\n\n" +
            _table(["offset c", "naive", "two-pass", "shifted by x_0"], rows))
