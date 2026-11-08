# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Initialization: Where Training Starts".

Plain Python 3 on floats (IEEE double precision), written from the chapter's own
description and independent of any engine code. Each public function returns a
markdown string that the chapter embeds between `<!-- worked: initialization.<name> -->`
markers. The same module supplies the data of the chapter's figures.

The running example is one network, called the deep stack in the chapter:

  input x in R^64  ->  10 layers  z_l = W_l a_{l-1} + b_l,  a_l = g(z_l)  (64 units each)
                   ->  read-out   y = w_out . a_10 + b_out              (one number)

trained on the mean squared error of a smooth target. Everything random comes from
the book's SplitMix64 (`bookrng.generator(name)`); a weight is a scale times a
standard normal draw, and every scheme reads the same stream of draws, so two
schemes differ only in their scale.

Swedish Embedded AB implements solutions for making neural network training start
well and reproducibly. If your team needs expertise in weight initialization and
training diagnostics, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from functools import lru_cache
from operator import mul

import bookrng
import stable

INPUT_DIM = 64      # width of the input x
WIDTH = 64          # units per hidden layer
DEPTH = 10          # hidden layers (the read-out is an eleventh weight matrix)
N_MEASURE = 128     # inputs used to measure activations and gradients
N_TRAIN = 32        # examples of the training runs (full batch)
TRAIN_STEPS = 100
TRAIN_LR = 0.01
TRAIN_RECORD = (0, 10, 30, 60, 100)
SCHEMES = ("zero", "small", "large", "xavier", "he")
SMALL_STD = 0.01
LARGE_STD = 1.0
LAYERS_SHOWN = (1, 2, 3, 5, 7, 10)

WEIGHTS_NAME = "initialization.weights"
DATA_NAME = "initialization.data"


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


def _gen_note(name):
    s = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed 0x{s:016x} ({s})"


def stack_note():
    """The complete recipe of the deep stack's weights and data, printed with the first training table."""
    ws, ds = bookrng.seed_of(WEIGHTS_NAME), bookrng.seed_of(DATA_NAME)
    return (f"Weights: one generator `{WEIGHTS_NAME}` (seed 0x{ws:016x}, {ws}). The 10 hidden weight matrices (shape [64, 64], row j holds the weights of unit j) and then the read-out vector (64 entries) "
            "are drawn in this order, layer by layer, row by row, entry i = 0, ..., 63 within a row; each weight is scale * normal(), and every scheme draws every normal() (a zero scale still consumes it), "
            f"so all schemes see the same standard normals. Scales: zeros 0; small {SMALL_STD:g}; large {LARGE_STD:g}; Xavier sqrt(2 / (n_in + n_out)) with n_in = n_out = 64 in the hidden layers and "
            "n_in = 64, n_out = 1 in the read-out, so sqrt(2 / 65); He sqrt(2 / n_in) = sqrt(2 / 64) in every layer. "
            f"Data: one generator `{DATA_NAME}` (seed 0x{ds:016x}, {ds}). First the direction a: 64 draws of normal() each multiplied by 3/16 = 0.1875; then the examples, example after example, "
            "each 64 consecutive normal() draws (so the 128 inputs of the later tables extend the 32 training inputs, which are their first 32); the target of an example x is t = 1 + sin(a . x) (radians). "
            "The loss is the mean over the examples of (y - t)^2, with y = w_out . a_10 + b_out.")


def _loss(v):
    return f"{v:.4f}" if 1e-3 <= v < 1e3 else _sci(v)


def _stable_loss(run, twins, s):
    """The loss after s steps of `run` with only the digits that survive in the nudged runs `twins` (a run that left
    the finite range earlier has no entry at s: 'not finite' if it does so in every run, else 'unstable')."""
    at = lambda r: r[s] if s < len(r) else None   # noqa: E731
    return stable.text(at(run), [at(t) for t in twins], _loss)


def _is_bad(v):
    return not math.isfinite(v) or v > 1e30


# --------------------------------------------------------------- the deep stack

def dot(a, b):
    return sum(map(mul, a, b))


def activation(name, z):
    if name == "tanh":
        return math.tanh(z)
    if name == "relu":
        return z if z > 0.0 else 0.0
    if name == "identity":
        return z
    raise ValueError(name)


def slope(name, z, a):
    """g'(z) given the pre-activation z and the activation a = g(z)."""
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return 1.0 if z > 0.0 else 0.0
    return 1.0


def scale_of(scheme, fan_in, fan_out):
    """Standard deviation of a weight under a named scheme."""
    if scheme == "zero":
        return 0.0
    if scheme == "small":
        return SMALL_STD
    if scheme == "large":
        return LARGE_STD
    if scheme == "xavier":
        return math.sqrt(2.0 / (fan_in + fan_out))
    if scheme == "he":
        return math.sqrt(2.0 / fan_in)
    raise ValueError(scheme)


def layer_shapes():
    """(fan_out, fan_in) of the 11 weight matrices: 10 hidden layers and the read-out."""
    return [(WIDTH, INPUT_DIM)] + [(WIDTH, WIDTH)] * (DEPTH - 1) + [(1, WIDTH)]


def init_stack(scheme, name=WEIGHTS_NAME, readout_scale=1.0):
    """W[l][j][i] = scale * normal(), drawn layer by layer, row j by row, i = 0 .. fan_in - 1.

    The draws are taken for every scheme (a zero scale still consumes them) so that
    all schemes see the same standard normals."""
    g = bookrng.generator(name)
    Ws = []
    for l, (fo, fi) in enumerate(layer_shapes()):
        s = scale_of(scheme, fi, fo) * (readout_scale if l == DEPTH else 1.0)
        Ws.append([[s * g.normal() for _ in range(fi)] for _ in range(fo)])
    bs = [[0.0] * len(W) for W in Ws]
    return Ws, bs


@lru_cache(maxsize=None)
def make_data(n):
    """a: 64 draws of normal() times 1.5/8; then X[n, j]: n rows of 64 normal() draws. t = 1 + sin(a . x)."""
    g = bookrng.generator(DATA_NAME)
    a = [1.5 / 8.0 * g.normal() for _ in range(INPUT_DIM)]
    X = [[g.normal() for _ in range(INPUT_DIM)] for _ in range(n)]
    t = [1.0 + math.sin(dot(a, x)) for x in X]
    return X, t


def forward(Ws, bs, X, act):
    """Pre-activations Z[l][n][j] and the inputs of each layer A[l][n][i] (A[0] = X)."""
    A, Zs, As = X, [], [X]
    for l, (W, b) in enumerate(zip(Ws, bs)):
        Z = [[dot(w, a) + bj for w, bj in zip(W, b)] for a in A]
        Zs.append(Z)
        A = [[activation(act, z) for z in row] for row in Z] if l < len(Ws) - 1 else Z
        As.append(A)
    return Zs, As


def backward(Ws, Zs, As, t, act):
    """deltas[l][n][j] = d(loss of example n)/d z_l[n][j], the loss being (y - t)^2 per example."""
    deltas = [None] * len(Ws)
    deltas[-1] = [[2.0 * (r[0] - tt)] for r, tt in zip(Zs[-1], t)]
    for l in range(len(Ws) - 1, 0, -1):
        Wt = list(zip(*Ws[l]))
        new = []
        for dn, zr, ar in zip(deltas[l], Zs[l - 1], As[l]):
            new.append([dot(col, dn) * slope(act, z, a) for col, z, a in zip(Wt, zr, ar)])
        deltas[l - 1] = new
    return deltas


def gradients(Ws, deltas, As):
    """Mean over the batch of the per-example gradients: dW[l][j][i], db[l][j]."""
    n = len(As[0])
    gW, gb = [], []
    for l, W in enumerate(Ws):
        Dt = list(zip(*deltas[l]))
        At = list(zip(*As[l]))
        gW.append([[dot(Dt[j], At[i]) / n for i in range(len(W[0]))] for j in range(len(W))])
        gb.append([sum(Dt[j]) / n for j in range(len(W))])
    return gW, gb


def rms(values):
    v = list(values)
    return math.sqrt(sum(x * x for x in v) / len(v))


def mean_loss(Zs, t):
    return sum((r[0] - tt) ** 2 for r, tt in zip(Zs[-1], t)) / len(t)


@lru_cache(maxsize=None)
def layer_statistics(act, scheme):
    """(rms of z_l, rms of delta_l relative to delta_10, z_10 values) over the N_MEASURE inputs."""
    X, t = make_data(N_MEASURE)
    Ws, bs = init_stack(scheme)
    Zs, As = forward(Ws, bs, X, act)
    deltas = backward(Ws, Zs, As, t, act)
    fz = [rms(v for r in Zs[l] for v in r) for l in range(DEPTH)]
    bd = [rms(v for r in deltas[l] for v in r) for l in range(DEPTH)]
    top = bd[-1]
    rel = [b / top for b in bd] if top > 0.0 else [0.0] * DEPTH
    z10 = [v for r in Zs[DEPTH - 1] for v in r]
    return fz, rel, z10


@lru_cache(maxsize=None)
def train_run(act, scheme, name=WEIGHTS_NAME, steps=TRAIN_STEPS, lr=TRAIN_LR, readout_scale=1.0, nudge=0.0):
    """Full-batch gradient descent from a named start. Returns the loss before every step 0 .. steps
    (shorter if the loss leaves the finite range, in which case the last entry is the bad value)."""
    X, t = make_data(N_TRAIN)
    Ws, bs = init_stack(scheme, name, readout_scale)
    Ws[0][0][0] += nudge
    out = []
    for s in range(steps + 1):
        Zs, As = forward(Ws, bs, X, act)
        loss = mean_loss(Zs, t)
        out.append(loss)
        if _is_bad(loss) or s == steps:
            break
        deltas = backward(Ws, Zs, As, t, act)
        gW, gb = gradients(Ws, deltas, As)
        for l in range(len(Ws)):
            for j in range(len(Ws[l])):
                Ws[l][j] = [w - lr * g for w, g in zip(Ws[l][j], gW[l][j])]
                bs[l][j] -= lr * gb[l][j]
    return out


def _target_stats():
    _, t = make_data(N_TRAIN)
    mean = sum(t) / len(t)
    return sum(v * v for v in t) / len(t), sum((v - mean) ** 2 for v in t) / len(t)


# ------------------------------------------------------------------ example 1

def _tanh_unit_step(w1, w2, x, t):
    """One example, H = 3 identical tanh units (all weights equal): returns the pieces of the gradient."""
    z = [sum(wi * xi for wi, xi in zip(w, x)) for w in w1]
    h = [math.tanh(v) for v in z]
    y = sum(wj * hj for wj, hj in zip(w2, h))
    dy = 2.0 * (y - t)
    dw2 = [dy * hj for hj in h]
    dh = [dy * wj for wj in w2]
    dz = [d * (1.0 - hj * hj) for d, hj in zip(dh, h)]
    dw1 = [[d * xi for xi in x] for d in dz]
    return z, h, y, dy, dw2, dz, dw1


def symmetry_one_step():
    x, t = (1.0, 2.0), 1.0
    w1 = [[0.5, 0.5]] * 3
    w2 = [0.5] * 3
    z, h, y, dy, dw2, dz, dw1 = _tanh_unit_step(w1, w2, x, t)
    rows = [(j + 1, f"{z[j]:.4f}", f"{h[j]:.4f}", f"{dz[j]:.4f}", f"({dw1[j][0]:.4f}, {dw1[j][1]:.4f})", f"{dw2[j]:.4f}")
            for j in range(3)]
    return (f"Setup: one example x = (1, 2) with target t = 1; three tanh hidden units, every incoming weight 0.5 "
            f"(W1 is [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]), every outgoing weight 0.5, all biases 0. "
            f"Output y = {y:.4f}, error e = y - t = {y - t:.4f}, loss e^2 = {(y - t) ** 2:.4f}, dL/dy = {dy:.4f}.\n\n"
            + _table(["unit j", "z_j", "h_j = tanh(z_j)", "dL/dz_j", "row j of dL/dW1", "dL/dW2_j"], rows))


SYM_X = [(1.0, 2.0), (-1.0, 0.5), (0.5, -1.0), (-2.0, -1.0)]
SYM_T = [1.0, 0.0, -1.0, 0.5]
SYM_H = 4
SYM_LR = 0.1
SYM_RECORD = (0, 1, 10, 100, 1000)


def _sym_train(w1, w2):
    """Gradient descent on the mean squared error of the 4 examples; returns {step: (loss, spread of W1 rows, spread of W2)}."""
    n = len(SYM_X)
    out = {}
    for s in range(max(SYM_RECORD) + 1):
        zs = [[dot(w, x) for w in w1] for x in SYM_X]
        hs = [[math.tanh(v) for v in z] for z in zs]
        ys = [dot(w2, h) for h in hs]
        if s in SYM_RECORD:
            loss = sum((y - t) ** 2 for y, t in zip(ys, SYM_T)) / n
            spread1 = max(abs(w1[j][i] - w1[k][i]) for j in range(SYM_H) for k in range(SYM_H) for i in range(2))
            spread2 = max(w2) - min(w2)
            out[s] = (loss, spread1, spread2)
        gw1 = [[0.0, 0.0] for _ in range(SYM_H)]
        gw2 = [0.0] * SYM_H
        for x, h, y, t in zip(SYM_X, hs, ys, SYM_T):
            dy = 2.0 * (y - t) / n
            for j in range(SYM_H):
                gw2[j] += dy * h[j]
                dz = dy * w2[j] * (1.0 - h[j] * h[j])
                gw1[j][0] += dz * x[0]
                gw1[j][1] += dz * x[1]
        w1 = [[w - SYM_LR * g for w, g in zip(w1[j], gw1[j])] for j in range(SYM_H)]
        w2 = [w - SYM_LR * g for w, g in zip(w2, gw2)]
    return out


def symmetry_run():
    name = "initialization.symmetry"
    g = bookrng.generator(name)
    r1 = [[0.5 * g.normal() for _ in range(2)] for _ in range(SYM_H)]
    r2 = [0.5 * g.normal() for _ in range(SYM_H)]
    same = _sym_train([[0.5, 0.5] for _ in range(SYM_H)], [0.5] * SYM_H)
    rand = _sym_train(r1, r2)
    rows = []
    for s in SYM_RECORD:
        rows.append((s, f"{same[s][0]:.6f}", f"{same[s][1]:.6f}", f"{same[s][2]:.6f}",
                     f"{rand[s][0]:.6f}", f"{rand[s][1]:.6f}", f"{rand[s][2]:.6f}"))
    inputs = "; ".join(f"x = ({x[0]:g}, {x[1]:g}), t = {t:g}" for x, t in zip(SYM_X, SYM_T))
    return (f"Setup: four examples ({inputs}); a network with {SYM_H} tanh hidden units and no biases (y = sum_j W2_j tanh(W1_j . x)); "
            f"mean squared error over the four examples; gradient descent with rate {SYM_LR}; "
            f"the identical start has every weight 0.5. {_gen_note(name)}. The random start draws "
            f"W1[j][i] = 0.5 * normal() for j = 0 .. {SYM_H - 1}, i = 0, 1 (row by row), then W2[j] = 0.5 * normal(). "
            f"\"Spread\" is the largest difference between two units' entries (of W1 rows, or of W2).\n\n"
            + _table(["step", "loss (identical)", "spread W1 (identical)", "spread W2 (identical)",
                      "loss (random)", "spread W1 (random)", "spread W2 (random)"], rows))


def start_gradients():
    """Gradient norms by layer at the first step: all-zero start against He-ReLU with zero biases."""
    X, t = make_data(N_TRAIN)
    cols = []
    for act, scheme in (("tanh", "zero"), ("relu", "zero"), ("relu", "he")):
        Ws, bs = init_stack(scheme)
        Zs, As = forward(Ws, bs, X, act)
        deltas = backward(Ws, Zs, As, t, act)
        gW, gb = gradients(Ws, deltas, As)
        cols.append(([math.sqrt(sum(v * v for r in gw for v in r)) for gw in gW],
                     [math.sqrt(sum(v * v for v in g)) for g in gb]))
    rows = []
    for l in range(DEPTH + 1):
        label = f"{l + 1}" if l < DEPTH else "11 (read-out)"
        rows.append((label, *[_sci(c[0][l]) if k % 2 == 0 else _sci(c[1][l]) for c in cols for k in range(2)]))
    return (f"Setup: the deep stack (10 layers of 64 units, a read-out), the first {N_TRAIN} inputs of the data, all biases 0, "
            f"loss = mean of (y - t)^2. Entries are the Euclidean norm of the gradient of each layer's weight matrix and bias vector "
            f"at step 0. Weights: all zero (tanh and ReLU), or He normal (ReLU); weights and data are those of the first training table "
            f"(generators `{WEIGHTS_NAME}` and `{DATA_NAME}`, seeds 0x{bookrng.seed_of(WEIGHTS_NAME):016x} and 0x{bookrng.seed_of(DATA_NAME):016x}).\n\n"
            + _table(["layer", "tanh, zeros: norm of dW", "tanh, zeros: norm of db", "ReLU, zeros: norm of dW", "ReLU, zeros: norm of db",
                      "ReLU, He: norm of dW", "ReLU, He: norm of db"], rows))


# ------------------------------------------------------------------ variance

VAR_FAN_IN = 64
VAR_TRIALS = 4000


def variance_of_a_sum():
    name = "initialization.sum"
    g = bookrng.generator(name)
    n = VAR_FAN_IN
    sigma = 1.0 / math.sqrt(n)
    cases = [("independent", "independent"), ("independent", "equal"), ("equal", "independent"), ("equal", "equal")]
    rows = []
    for wmode, xmode in cases:
        total = 0.0
        for _ in range(VAR_TRIALS):
            if xmode == "independent":
                x = [g.normal() for _ in range(n)]
            else:
                s = g.normal()
                x = [s] * n
            if wmode == "independent":
                w = [sigma * g.normal() for _ in range(n)]
            else:
                v = sigma * g.normal()
                w = [v] * n
            z = dot(w, x)
            total += z * z
        measured = total / VAR_TRIALS
        predicted = n * sigma ** 2 * 1.0 if (wmode == "independent" or xmode == "independent") else n * n * sigma ** 2
        rows.append((wmode, xmode, f"{predicted:.1f}", f"{measured:.3f}"))
    return (f"{_gen_note(name)}. z = sum_i w_i x_i with n = {n} terms, weight standard deviation sigma_w = 1/sqrt(n) = {sigma:.4f}, "
            f"E[x_i^2] = 1. Each of {VAR_TRIALS} trials in turn draws x first (n calls of normal() for \"independent\" inputs, "
            f"one call whose value is copied n times for \"equal\"), then w (n calls of normal() times sigma_w, or one call "
            f"copied n times). The measured value is the mean of z^2 over the trials.\n\n"
            + _table(["weights", "inputs", "prediction", "measured E[z^2]"], rows, align="llrr"))


def _relu_moment_rows(g, draws):
    rows = []
    specs = [("N(0, 0.1^2)", lambda: 0.1 * g.normal()), ("N(0, 1)", lambda: g.normal()),
             ("N(0, 3^2)", lambda: 3.0 * g.normal()),
             ("uniform on [-sqrt 3, sqrt 3]", lambda: math.sqrt(3.0) * (2.0 * g.uniform() - 1.0))]
    for label, draw in specs:
        zs = [draw() for _ in range(draws)]
        m2 = sum(z * z for z in zs) / draws
        r2 = sum((z if z > 0.0 else 0.0) ** 2 for z in zs) / draws
        t2 = sum(math.tanh(z) ** 2 for z in zs) / draws
        rows.append((label, f"{m2:.4f}", f"{r2:.4f}", f"{r2 / m2:.4f}", f"{t2:.4f}", f"{t2 / m2:.4f}"))
    return rows


def relu_second_moment():
    name = "initialization.relu-moment"
    draws = 50000
    g = bookrng.generator(name)
    rows = _relu_moment_rows(g, draws)
    return (f"{_gen_note(name)}. For each distribution in turn, {draws} draws, one call each (normal() times the standard deviation, "
            f"or sqrt(3) * (2 * uniform() - 1)), all from one generator in the order of the rows.\n\n"
            + _table(["distribution of z", "E[z^2]", "E[relu(z)^2]", "ratio", "E[tanh(z)^2]", "ratio"], rows, align="lrrrrr"))


def _gauss_second_moment(f, steps=4000, lim=10.0):
    """E[f(zeta)^2] for zeta ~ N(0, 1) by the trapezoid rule on [-lim, lim] with `steps` intervals."""
    h = 2.0 * lim / steps
    total = 0.0
    for k in range(steps + 1):
        z = -lim + k * h
        w = 0.5 if k in (0, steps) else 1.0
        total += w * f(z) ** 2 * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return total * h


def _gelu(z):
    return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))


def _silu(z):
    return z / (1.0 + math.exp(-z))


GAIN_ACTS = [("identity", lambda z: z), ("relu", lambda z: max(0.0, z)), ("tanh", math.tanh), ("gelu", _gelu), ("silu", _silu)]


def tanh_kappa():
    return _gauss_second_moment(math.tanh)


def gain_by_activation():
    rows = []
    for name, f in GAIN_ACTS:
        k = _gauss_second_moment(f)
        rows.append((name, f"{k:.4f}", f"{1.0 / k:.4f}", f"{math.sqrt(1.0 / k):.4f}"))
    return ("kappa = E[g(zeta)^2] for zeta standard normal, computed by the trapezoid rule on [-10, 10] with 4000 intervals. "
            "A layer with weight variance 1/(kappa n) and inputs of second moment 1 gives pre-activations of second moment 1.\n\n"
            + _table(["activation g", "kappa = E[g(zeta)^2]", "n * Var(w) = 1/kappa", "gain sqrt(1/kappa)"], rows))


# ------------------------------------------------------------ depth: measured

def _scheme_label(scheme):
    return {"zero": "zeros", "small": f"small ({SMALL_STD:g})", "large": f"large ({LARGE_STD:g})",
            "xavier": "Xavier", "he": "He"}[scheme]


def _forward_rms(act):
    rows = []
    X, _ = make_data(N_MEASURE)
    x_rms = rms(v for r in X for v in r)
    for scheme in SCHEMES:
        if scheme == "zero":
            rows.append((_scheme_label(scheme), *["0"] * len(LAYERS_SHOWN)))
            continue
        fz, _, _ = layer_statistics(act, scheme)
        rows.append((_scheme_label(scheme), *[_sci(fz[l - 1]) for l in LAYERS_SHOWN]))
    return (f"Setup: the deep stack with {act} units, {N_MEASURE} inputs (rms of the inputs {x_rms:.3f}), biases 0. "
            f"Entries: the root mean square of z_l over all {N_MEASURE} x 64 pre-activations of layer l. "
            f"Weights and data: as in the first training table of the chapter (weights generator `{WEIGHTS_NAME}`, seed 0x{bookrng.seed_of(WEIGHTS_NAME):016x}; data generator `{DATA_NAME}`, "
            f"seed 0x{bookrng.seed_of(DATA_NAME):016x}, the {N_MEASURE} inputs being the first {N_MEASURE} examples of that stream); scale * normal(), the same draws for every row.\n\n"
            + _table(["start"] + [f"z_{l}" for l in LAYERS_SHOWN], rows))


def _backward_rms(act):
    rows = []
    for scheme in SCHEMES:
        if scheme == "zero":
            rows.append((_scheme_label(scheme), *["0"] * len(LAYERS_SHOWN)))
            continue
        _, rel, _ = layer_statistics(act, scheme)
        rows.append((_scheme_label(scheme), *[_sci(rel[l - 1]) for l in LAYERS_SHOWN]))
    return (f"Setup: as above, {act} units. Entries: the root mean square of d(loss of one example)/d z_l over the "
            f"{N_MEASURE} examples and 64 units, divided by the same quantity at layer 10 (so layer 10 reads 1). "
            f"The loss of one example is (y - t)^2 with the target t = 1 + sin(a . x) of the first training table; weights and data as in that table "
            f"(generators `{WEIGHTS_NAME}` and `{DATA_NAME}`).\n\n"
            + _table(["start"] + [f"layer {l}" for l in LAYERS_SHOWN], rows))


def forward_rms_tanh():
    return _forward_rms("tanh")


def forward_rms_relu():
    return _forward_rms("relu")


def backward_rms_tanh():
    return _backward_rms("tanh")


def backward_rms_relu():
    return _backward_rms("relu")


def forward_prediction():
    """Per-layer factor c = fan_in Var(w) kappa for ReLU and the predicted against the measured z_10."""
    X, _ = make_data(N_MEASURE)
    q0 = sum(v * v for r in X for v in r) / (len(X) * INPUT_DIM)
    rows = []
    for scheme in ("small", "large", "xavier", "he"):
        s = scale_of(scheme, WIDTH, WIDTH)
        c = WIDTH * s * s * 0.5
        q1 = INPUT_DIM * s * s * q0
        pred_z1 = math.sqrt(q1)
        pred_z10 = math.sqrt(q1 * c ** (DEPTH - 1))
        fz, _, _ = layer_statistics("relu", scheme)
        rows.append((_scheme_label(scheme), f"{s * s:.6f}", f"{c:.4f}", _sci(pred_z1), _sci(fz[0]), _sci(pred_z10), _sci(fz[-1])))
    return (f"Setup: ReLU deep stack (weights and data as in the first training table of the chapter), mean of x^2 over the {N_MEASURE} inputs q_0 = {q0:.4f}. Prediction: "
            f"E[z_1^2] = 64 Var(w) q_0 and E[z_l^2] = c E[z_(l-1)^2] with c = 64 Var(w) / 2 for the 64-unit layers.\n\n"
            + _table(["start", "Var(w)", "c = 64 Var(w)/2", "predicted rms z_1", "measured rms z_1",
                      "predicted rms z_10", "measured rms z_10"], rows))


def layer10_summary():
    rows = []
    for act, scheme in (("tanh", "xavier"), ("tanh", "he"), ("tanh", "large"), ("relu", "he"), ("relu", "small"), ("relu", "large")):
        _, _, z10 = layer_statistics(act, scheme)
        n = len(z10)
        if act == "tanh":
            frac = sum(1 for z in z10 if 1.0 - math.tanh(z) ** 2 < 0.01) / n
            what = "slope tanh'(z) below 0.01"
        else:
            frac = sum(1 for z in z10 if z > 0.0) / n
            what = "z > 0 (unit active)"
        mean = sum(z10) / n
        rows.append((act, _scheme_label(scheme), f"{mean:+.3e}", _sci(rms(z10)), what, f"{frac:.3f}"))
    return (f"Setup: z_10 of the deep stack (weights and data as in the first training table of the chapter) for the {N_MEASURE} inputs and 64 units ({N_MEASURE * WIDTH} values per row).\n\n"
            + _table(["activation", "start", "mean of z_10", "rms of z_10", "measure", "share"], rows, align="llrrlr"))


# ------------------------------------------------------- Xavier: the funnel

FUNNEL = (256, 128, 64, 32, 16, 8)
FUNNEL_N = 128


def _funnel_run(rule, g):
    widths = FUNNEL
    shapes = [(widths[i + 1], widths[i]) for i in range(len(widths) - 1)]
    X = [[g.normal() for _ in range(widths[0])] for _ in range(FUNNEL_N)]
    Ws = []
    for fo, fi in shapes:
        var = {"fan_in": 1.0 / fi, "fan_out": 1.0 / fo, "xavier": 2.0 / (fi + fo)}[rule]
        s = math.sqrt(var)
        Ws.append([[s * g.normal() for _ in range(fi)] for _ in range(fo)])
    a = X
    for W in Ws:
        a = [[dot(w, r) for w in W] for r in a]
    fwd = sum(v * v for r in a for v in r) / (len(a) * widths[-1]) / (sum(v * v for r in X for v in r) / (FUNNEL_N * widths[0]))
    d = [[g.normal() for _ in range(widths[-1])] for _ in range(FUNNEL_N)]
    d0 = sum(v * v for r in d for v in r) / (FUNNEL_N * widths[-1])
    for W in reversed(Ws):
        Wt = list(zip(*W))
        d = [[dot(col, r) for col in Wt] for r in d]
    bwd = sum(v * v for r in d for v in r) / (len(d) * widths[0]) / d0
    pf = 1.0
    pb = 1.0
    for fo, fi in shapes:
        var = {"fan_in": 1.0 / fi, "fan_out": 1.0 / fo, "xavier": 2.0 / (fi + fo)}[rule]
        pf *= fi * var
        pb *= fo * var
    return pf, fwd, pb, bwd


def funnel():
    name = "initialization.funnel"
    g = bookrng.generator(name)
    rows = []
    for rule, label in (("fan_in", "1/fan_in (forward-exact)"), ("fan_out", "1/fan_out (backward-exact)"),
                        ("xavier", "2/(fan_in + fan_out) (Xavier)")):
        pf, fwd, pb, bwd = _funnel_run(rule, g)
        rows.append((label, f"{pf:.4f}", f"{fwd:.4f}", f"{pb:.4f}", f"{bwd:.4f}"))
    ws = " -> ".join(str(w) for w in FUNNEL)
    return (f"{_gen_note(name)}. A linear network (identity activations, no biases) with widths {ws}: five weight matrices "
            f"W_l of shape [fan_out, fan_in]. For each rule in the order of the rows: first {FUNNEL_N} inputs of 256 normal() draws "
            f"(row by row), then the weights layer by layer (row by row, scale times normal()), then, for the backward pass, "
            f"{FUNNEL_N} upstream gradient vectors of 8 normal() draws. Forward: output second moment divided by input second "
            f"moment. Backward: the upstream gradient is multiplied by W_5^T, ..., W_1^T and the second moment of the result "
            f"divided by that of the upstream gradient. The prediction is the product over layers of fan_in Var(w) (forward) and of "
            f"fan_out Var(w) (backward).\n\n"
            + _table(["Var(w) of each layer", "forward: predicted", "forward: measured", "backward: predicted",
                      "backward: measured"], rows))


# ------------------------------------------------------- uniform, normal, truncated

SHAPE_DRAWS = 100000
TRUNC = 2.0


def truncated_variance(c=TRUNC):
    """Variance of a standard normal conditioned on |z| <= c: 1 - 2 c phi(c) / (2 Phi(c) - 1)."""
    phi = math.exp(-0.5 * c * c) / math.sqrt(2.0 * math.pi)
    big = 0.5 * (1.0 + math.erf(c / math.sqrt(2.0)))
    return 1.0 - 2.0 * c * phi / (2.0 * big - 1.0)


def _draw_shape(shape, sigma, g):
    if shape == "normal":
        return sigma * g.normal()
    if shape == "uniform":
        return sigma * math.sqrt(3.0) * (2.0 * g.uniform() - 1.0)
    while True:  # truncated normal: redraw until |z| <= TRUNC, then rescale to the target variance
        z = g.normal()
        if abs(z) <= TRUNC:
            return sigma * z / math.sqrt(truncated_variance())


def draw_shapes():
    name = "initialization.shapes"
    g = bookrng.generator(name)
    sigma = 1.0
    rows = []
    for shape, formula in (("normal", "sigma * normal()"), ("uniform", "sigma * sqrt(3) * (2 uniform() - 1)"),
                           ("truncated normal", f"sigma * z / sqrt({truncated_variance():.6f}), z = normal() redrawn while abs(z) > 2")):
        key = "truncated" if shape.startswith("trunc") else shape
        ws = [_draw_shape(key, sigma, g) for _ in range(SHAPE_DRAWS)]
        var = sum(w * w for w in ws) / SHAPE_DRAWS
        kurt = sum(w ** 4 for w in ws) / SHAPE_DRAWS / var ** 2
        rows.append((shape, formula, f"{var:.4f}", f"{kurt:.3f}", f"{max(abs(w) for w in ws):.3f}"))
    return (f"{_gen_note(name)}. sigma = {sigma:g}. {SHAPE_DRAWS} draws per shape, one after the other from one generator in "
            f"the order of the rows. The variance of a standard normal conditioned on abs(z) <= 2 is {truncated_variance():.6f} (use this value or the formula, not a shorter one). "
            f"Fourth moment = mean of w^4 divided by the squared variance (3 for a normal, 1.8 for a uniform). "
            f"Variance is the mean of w^2 (the mean is 0 by symmetry).\n\n"
            + _table(["shape", "draw", "variance", "fourth moment", "largest absolute w"], rows, align="llrrr"))


def shape_in_network():
    rows = []
    seeds = []
    for shape in ("normal", "uniform", "truncated"):
        seeds.append(f"`initialization.shape-net.{shape}` = 0x{bookrng.seed_of(f'initialization.shape-net.{shape}'):016x}")
        g = bookrng.generator(f"initialization.shape-net.{shape}")
        X, _ = make_data(N_MEASURE)
        a = X
        zr = []
        for l, (fo, fi) in enumerate(layer_shapes()[:DEPTH]):
            s = scale_of("he", fi, fo)
            W = [[_draw_shape(shape, s, g) for _ in range(fi)] for _ in range(fo)]
            Z = [[dot(w, r) for w in W] for r in a]
            zr.append(rms(v for r in Z for v in r))
            a = [[max(0.0, v) for v in r] for r in Z]
        rows.append((shape, *[_sci(zr[l - 1], 3) for l in (1, 5, 10)]))
    return ("Setup: the ReLU deep stack with He scale sigma = sqrt(2/fan_in), but each weight drawn with the named shape "
            f"(one generator per row: {', '.join(seeds)}; weights layer by layer, row by row, one draw per weight in the shape's rule, the truncated shape redrawing a normal() until it passes). "
            f"{N_MEASURE} inputs, the first {N_MEASURE} examples of the data stream of the first training table (generator `{DATA_NAME}`, seed 0x{bookrng.seed_of(DATA_NAME):016x}), biases 0. The predicted rms of z_1 is sqrt(2 q_0) with q_0 the mean of x^2.\n\n"
            + _table(["weight shape", "rms z_1", "rms z_5", "rms z_10"], rows))


# ------------------------------------------------------------ training results

def training_table(act):
    e_t2, var_t = _target_stats()
    rows = []
    for scheme in SCHEMES:
        run, twins = train_run(act, scheme), [train_run(act, scheme, nudge=d) for d in stable.NUDGES]
        cells = [_stable_loss(run, twins, s) for s in TRAIN_RECORD]
        rows.append((_scheme_label(scheme), *cells))
    return (f"Setup: the deep stack with {act} units, {N_TRAIN} training examples (full batch), mean squared error, "
            f"plain gradient descent with rate {TRAIN_LR:g}, {TRAIN_STEPS} steps, biases start at 0. The target is "
            f"t = 1 + sin(a . x) with mean of t^2 = {e_t2:.4f}; predicting the mean of the targets gives the loss "
            f"{var_t:.4f}. {stack_note() if act == "tanh" else "Weights and data exactly as in the tanh table above (same generators, seeds, draw order and scales)."} " + stable.RULE + "\n\n"
            + _table(["start"] + [f"step {s}" for s in TRAIN_RECORD], rows))


def training_tanh():
    return training_table("tanh")


def training_relu():
    return training_table("relu")


REPEAT_RUNS = 5


def seed_table():
    rows = []
    names = [f"initialization.seed-run.{k}" for k in range(REPEAT_RUNS)] + ["initialization.seed-run.0"]
    for k, name in enumerate(names):
        run = train_run("relu", "he", name)
        label = f"{k}" if k < REPEAT_RUNS else "0 again"
        rows.append((label, f"0x{bookrng.seed_of(name):016x}", _loss(run[0]), _loss(run[30]), _loss(run[TRAIN_STEPS])))
    return (f"Setup: the ReLU deep stack, He normal start, the training run of the previous tables, but the weights come from the "
            f"generator named `initialization.seed-run.<k>` (k = 0 .. {REPEAT_RUNS - 1}); the last row repeats k = 0. "
            f"The data and everything else are unchanged.\n\n"
            + _table(["run k", "seed", "loss at step 0", "loss at step 30", f"loss at step {TRAIN_STEPS}"], rows, align="lrrrr"))


def output_scale():
    X, t = make_data(N_TRAIN)
    rows = []
    for m in (1.0, 0.1, 0.0):
        Ws, bs = init_stack("he", readout_scale=m)
        Zs, _ = forward(Ws, bs, X, "relu")
        ys = [r[0] for r in Zs[-1]]
        rows.append((f"{m:g}", _sci(rms(ys), 2), f"{mean_loss(Zs, t):.4f}"))
    e_t2, _ = _target_stats()
    return (f"Setup: ReLU deep stack (weights and data as in the first training table), He normal in the ten hidden layers; the read-out weights are multiplied by the factor "
            f"in the first column. {N_TRAIN} examples; the mean of t^2 is {e_t2:.4f}.\n\n"
            + _table(["read-out factor", "rms of the output y", "loss at step 0"], rows))


# ------------------------------------------------------------ special cases

RES_N = 128


def residual_growth():
    name = "initialization.residual"
    g = bookrng.generator(name)
    rows = []
    for blocks in (10, 20, 40):
        for label, gamma in (("1", 1.0), ("1/(2L)", 1.0 / (2 * blocks))):
            x = [[g.normal() for _ in range(WIDTH)] for _ in range(RES_N)]
            q0 = sum(v * v for r in x for v in r) / (RES_N * WIDTH)
            s1 = math.sqrt(2.0 / WIDTH)
            s2 = math.sqrt(gamma / WIDTH)
            for _ in range(blocks):
                W1 = [[s1 * g.normal() for _ in range(WIDTH)] for _ in range(WIDTH)]
                W2 = [[s2 * g.normal() for _ in range(WIDTH)] for _ in range(WIDTH)]
                hidden = [[max(0.0, dot(w1, r)) for w1 in W1] for r in x]
                x = [[xi + dot(w2, h) for xi, w2 in zip(r, W2)] for r, h in zip(x, hidden)]
            ql = sum(v * v for r in x for v in r) / (RES_N * WIDTH)
            rows.append((blocks, label, f"{gamma:.4f}", _sci((1.0 + gamma) ** blocks, 3), _sci(ql / q0, 3)))
    return (f"{_gen_note(name)}. A stack of L blocks x <- x + W2 relu(W1 x) on 64 units; W1 is He normal (sigma^2 = 2/64), "
            f"W2 has sigma^2 = gamma/64, no biases. Cells in the order of the rows; for each cell, {RES_N} inputs of 64 normal() draws "
            f"(row by row), then for each block W1 and W2 (row by row, scale times normal()), all from one generator. "
            f"Growth = mean of x^2 after the last block divided by the mean of x^2 at the input.\n\n"
            + _table(["blocks L", "gamma", "gamma value", "predicted (1 + gamma)^L", "measured growth"], rows, align="rlrrr"))


def _entropy(p):
    return -sum(v * math.log(v) for v in p if v > 0.0)


def class_prior_bias():
    rows = []
    for pi in (0.5, 0.1, 0.01, 0.001):
        b = math.log(pi / (1.0 - pi))
        h0 = math.log(2.0)
        h1 = _entropy([pi, 1.0 - pi])
        # plain gradient descent on the bias alone: d/db of the mean cross-entropy is sigmoid(b) - pi
        bb, steps = 0.0, 0
        while steps < 100000:
            loss = -(pi * math.log(1.0 / (1.0 + math.exp(-bb))) + (1.0 - pi) * math.log(1.0 / (1.0 + math.exp(bb))))
            if loss - h1 < 0.01:
                break
            bb -= 0.5 * (1.0 / (1.0 + math.exp(-bb)) - pi)
            steps += 1
        rows.append((f"{pi:g}", f"{b:+.4f}", f"{h0:.4f}", f"{h1:.4f}", steps))
    prior = [0.7, 0.2, 0.07, 0.03]
    multi = (f"Four classes with prior {prior}: all logits zero give the loss log 4 = {math.log(4.0):.4f}; biases log(prior) = "
             f"{[round(math.log(p), 4) for p in prior]} give the entropy of the prior, {_entropy(prior):.4f}.")
    return ("Binary case: the logit is the bias b alone (the rest of the network contributes nothing yet), the label is 1 with "
            "frequency pi, loss = mean cross-entropy. \"Steps\" counts gradient steps of rate 0.5 on b alone, starting at b = 0, "
            "until the loss is within 0.01 of its minimum H(pi).\n\n"
            + _table(["frequency pi of label 1", "b = log(pi/(1-pi))", "loss at b = 0", "loss at b = log odds = H(pi)",
                      "steps from b = 0 to within 0.01"], rows)
            + "\n\n" + multi)


def lora_zero_start():
    name = "initialization.lora"
    g = bookrng.generator(name)
    W = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    x = [1.0, 2.0, 3.0, 4.0]
    target = [2.0, 1.0, 3.5]
    r, lr = 2, 0.05
    A = [[0.5 * g.normal() for _ in range(4)] for _ in range(r)]
    Wx = [dot(w, x) for w in W]

    def run(B, A, steps):
        out = []
        for s in range(steps + 1):
            Ax = [dot(a, x) for a in A]
            y = [wx + dot(b, Ax) for wx, b in zip(Wx, B)]
            dy = [2.0 * (yy - tt) for yy, tt in zip(y, target)]
            dB = [[d * ax for ax in Ax] for d in dy]
            dA = [[sum(B[o][k] * dy[o] for o in range(3)) * xi for xi in x] for k in range(r)]
            nB = math.sqrt(sum(v * v for row in B for v in row))
            nA = math.sqrt(sum(v * v for row in A for v in row))
            ndB = math.sqrt(sum(v * v for row in dB for v in row))
            ndA = math.sqrt(sum(v * v for row in dA for v in row))
            out.append((s, sum(d * d / 4.0 for d in dy), nB, nA, ndB, ndA))
            B = [[b - lr * gb for b, gb in zip(brow, gbrow)] for brow, gbrow in zip(B, dB)]
            A = [[a - lr * ga for a, ga in zip(arow, garow)] for arow, garow in zip(A, dA)]
        return out

    rows = []
    for tag, B0, A0 in (("B = 0, A random", [[0.0] * r for _ in range(3)], A), ("A = 0, B = 0", [[0.0] * r for _ in range(3)], [[0.0] * 4 for _ in range(r)])):
        traj = run(B0, A0, 40)
        for s, loss, nB, nA, ndB, ndA in traj:
            if tag == "A = 0, B = 0" and s not in (0, 40):
                continue
            if s in (0, 1, 2, 5, 10, 20, 40):
                rows.append((tag, s, f"{loss:.4f}", _sci(nB), _sci(ndB), _sci(ndA)))
    return (f"{_gen_note(name)}. A frozen layer W (3 x 4) = [identity of size 3 | zero column] applied to x = (1, 2, 3, 4), so W x = (1, 2, 3); "
            f"the target is (2, 1, 3.5). The adapter adds B (A x) with A of shape [{r}, 4] and B of shape [3, {r}]; "
            f"A = 0.5 * normal() (draws row by row) in the first start. Loss = squared distance of the output from the target, "
            f"|y - target|^2 (its gradient with respect to y is 2 (y - target)); gradient descent on A and B with rate {lr}. "
            f"Columns: loss, norm of B, norm of dL/dB, norm of dL/dA.\n\n"
            + _table(["start", "step", "loss", "norm of B", "norm of dB", "norm of dA"], rows, align="lrrrrr"))


# ------------------------------------------------------------------ figure data

def figure_forward(act):
    return {s: layer_statistics(act, s)[0] for s in ("small", "large", "xavier", "he")}


def figure_backward(act):
    return {s: layer_statistics(act, s)[1] for s in ("small", "large", "xavier", "he")}


def figure_histogram(act, scheme):
    return layer_statistics(act, scheme)[2]


def figure_training_curves(act):
    return {scheme: train_run(act, scheme) for scheme in SCHEMES}
