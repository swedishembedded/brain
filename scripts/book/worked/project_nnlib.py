# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Reference implementation and acceptance tables for the Volume III project.

Plain Python 3, standard library only, written from the project chapter's own
description. The first half of the module is the reference library: layers that
save what their backward pass needs (linear, tanh, ReLU, GELU, SiLU, layer
norm, RMS norm, dropout, a residual wrapper, a sequential container), two
losses (softmax cross-entropy fused, mean squared error), a gradient checker
(per entry and directional), the optimizers (SGD, momentum, Adam, AdamW, Lion),
the learning-rate schedule, global-norm clipping, the initialisation schemes
and a training loop with logging, early stopping, checkpointing and an exact
resume. The second half builds the acceptance tables: each public function
without arguments returns a markdown string that the chapter embeds between
`<!-- worked: project_nnlib.<name> -->` markers, every expected value computed
here by the reference itself.

Layout of numbers: every tensor is a flat list in row-major order, so a
parameter vector is one list and an optimizer needs no knowledge of layers.
All arithmetic is IEEE double precision.

Randomness: every generator is the book's SplitMix64 and every seed comes from
`bookrng.seed_of("project_nnlib.<name>")` (printed with its draw order), except
the sensor data and the autoencoder's start, which are the streams of the
chapter "First Complete Models" so that the project can be checked against that
chapter's table.

Swedish Embedded AB implements solutions for neural network libraries whose
gradients are proven rather than hoped for. If your team needs expertise in
building and verifying training code, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import json
import math
from functools import lru_cache
from operator import mul

import bookrng

NAME = "project_nnlib"

# gradient-check gate of the project (double precision, chapter 6)
H_CHECK, ATOL, RTOL = 1e-5, 1e-7, 1e-5
SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _name(n):
    return n if "." in n else f"{NAME}.{n}"


def _gen(n):
    return bookrng.generator(_name(n))


def _seed(n):
    s = bookrng.seed_of(_name(n))
    return f"seed {s:#018x} ({s})"


def _f(v, d=6):
    return f"{v:+.{d}f}"


def _g(v, d=3):
    return f"{v:.{d}e}"


def fix(n, freq, phase, scale):
    """The fixture rule of the project: scale * sin(freq * k + phase) for k = 0 .. n-1."""
    return [scale * math.sin(freq * k + phase) for k in range(n)]


def rows_of(flat, n_cols):
    return [flat[i:i + n_cols] for i in range(0, len(flat), n_cols)]


def sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def gelu(z):
    return 0.5 * z * (1.0 + math.erf(z / SQRT2))


def gelu_slope(z):
    return 0.5 * (1.0 + math.erf(z / SQRT2)) + z * INV_SQRT_2PI * math.exp(-0.5 * z * z)


# ------------------------------------------------------------------ the layers

class Linear:
    """y = x W^T + b, W of shape [n_out, n_in]; backward accumulates into g."""

    def __init__(self, n_in, n_out, W=None, b=None):
        self.n_in, self.n_out = n_in, n_out
        self.p = {"W": list(W) if W is not None else [0.0] * (n_out * n_in),
                  "b": list(b) if b is not None else [0.0] * n_out}
        self.g = {k: [0.0] * len(v) for k, v in self.p.items()}

    def _w_rows(self):
        W, n = self.p["W"], self.n_in
        return [W[j * n:(j + 1) * n] for j in range(self.n_out)]

    def forward(self, x, mode="train", rng=None):
        self.x = x
        rows, b = self._w_rows(), self.p["b"]
        return [[sum(map(mul, w, xr)) + bj for w, bj in zip(rows, b)] for xr in x]

    def backward(self, dy):
        dyt, xt = list(zip(*dy)), list(zip(*self.x))
        gW, gb, k = self.g["W"], self.g["b"], 0
        for j, dj in enumerate(dyt):
            for xi in xt:
                gW[k] += sum(map(mul, dj, xi))
                k += 1
            gb[j] += sum(dj)
        cols = list(zip(*self._w_rows()))
        return [[sum(map(mul, dr, wc)) for wc in cols] for dr in dy]


ACTIVATIONS = {
    "tanh": (math.tanh, lambda z: 1.0 - math.tanh(z) ** 2),
    "relu": (lambda z: z if z > 0.0 else 0.0, lambda z: 1.0 if z > 0.0 else 0.0),
    "gelu": (gelu, gelu_slope),
    "silu": (lambda z: z * sigmoid(z), lambda z: sigmoid(z) * (1.0 + z * (1.0 - sigmoid(z)))),
}


class Activation:
    def __init__(self, kind):
        self.kind = kind
        self.f, self.df = ACTIVATIONS[kind]
        self.p, self.g = {}, {}

    def forward(self, x, mode="train", rng=None):
        self.z = x
        return [[self.f(v) for v in r] for r in x]

    def backward(self, dy):
        return [[d * self.df(v) for d, v in zip(dr, zr)] for dr, zr in zip(dy, self.z)]


class LayerNorm:
    """Per-row normalisation with a learned gain and shift (RMS norm: no mean, no shift)."""

    def __init__(self, d, eps=1e-5, gamma=None, beta=None, rms=False):
        self.d, self.eps, self.rms = d, eps, rms
        self.p = {"gamma": list(gamma) if gamma is not None else [1.0] * d}
        if not rms:
            self.p["beta"] = list(beta) if beta is not None else [0.0] * d
        self.g = {k: [0.0] * d for k in self.p}

    def forward(self, x, mode="train", rng=None):
        d, gamma = self.d, self.p["gamma"]
        self.xhat, self.inv = [], []
        out = []
        for r in x:
            mu = 0.0 if self.rms else sum(r) / d
            var = sum((v - mu) ** 2 for v in r) / d
            inv = 1.0 / math.sqrt(var + self.eps)
            xh = [(v - mu) * inv for v in r]
            self.xhat.append(xh)
            self.inv.append(inv)
            if self.rms:
                out.append([gamma[k] * xh[k] for k in range(d)])
            else:
                beta = self.p["beta"]
                out.append([gamma[k] * xh[k] + beta[k] for k in range(d)])
        return out

    def backward(self, dy, centre_term=True):
        """centre_term=False is the planted fault of milestone 2: it drops the mean(dxhat) term."""
        d, gamma = self.d, self.p["gamma"]
        gg = self.g["gamma"]
        dx = []
        for dr, xh, inv in zip(dy, self.xhat, self.inv):
            for k in range(d):
                gg[k] += dr[k] * xh[k]
            if not self.rms:
                gb = self.g["beta"]
                for k in range(d):
                    gb[k] += dr[k]
            dxh = [dr[k] * gamma[k] for k in range(d)]
            m1 = sum(dxh) / d if (not self.rms and centre_term) else 0.0
            m2 = sum(a * b for a, b in zip(dxh, xh)) / d
            dx.append([inv * (dxh[k] - m1 - xh[k] * m2) for k in range(d)])
        return dx


class Dropout:
    """Inverted dropout: one uniform() per element in row-major order, kept when the draw is >= p."""

    def __init__(self, p):
        assert 0.0 <= p < 1.0
        self.p_drop = p
        self.p, self.g = {}, {}

    def forward(self, x, mode="train", rng=None):
        if mode != "train" or self.p_drop == 0.0:
            self.mask = None
            return x
        scale = 1.0 / (1.0 - self.p_drop)
        self.mask = [[scale if rng.uniform() >= self.p_drop else 0.0 for _ in r] for r in x]
        return [[v * m for v, m in zip(r, mr)] for r, mr in zip(x, self.mask)]

    def backward(self, dy):
        if self.mask is None:
            return dy
        return [[d * m for d, m in zip(dr, mr)] for dr, mr in zip(dy, self.mask)]


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, mode="train", rng=None):
        for layer in self.layers:
            x = layer.forward(x, mode, rng)
        return x

    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy


class Residual:
    """y = x + body(x); the gradient has an identity path."""

    def __init__(self, body):
        self.body = body

    def forward(self, x, mode="train", rng=None):
        y = self.body.forward(x, mode, rng)
        return [[a + b for a, b in zip(xr, yr)] for xr, yr in zip(x, y)]

    def backward(self, dy):
        dx = self.body.backward(dy)
        return [[a + b for a, b in zip(dr, xr)] for dr, xr in zip(dy, dx)]


def leaves(layer):
    """The layers that own parameters or state, in forward order."""
    if isinstance(layer, Sequential):
        for sub in layer.layers:
            yield from leaves(sub)
    elif isinstance(layer, Residual):
        yield from leaves(layer.body)
    else:
        yield layer


def tensors(model):
    """(label, parameter list, gradient list) for every parameter tensor, in order."""
    out = []
    for k, leaf in enumerate(leaves(model)):
        for key in leaf.p:
            out.append((f"{k}.{key}", leaf.p[key], leaf.g[key]))
    return out


def get_theta(model):
    return [v for _, p, _ in tensors(model) for v in p]


def set_theta(model, theta):
    k = 0
    for _, p, _ in tensors(model):
        p[:] = theta[k:k + len(p)]
        k += len(p)


def get_grad(model):
    return [v for _, _, g in tensors(model) for v in g]


def zero_grad(model):
    for _, _, g in tensors(model):
        g[:] = [0.0] * len(g)


# ------------------------------------------------------------------ the losses

class SoftmaxCrossEntropy:
    """Mean over the batch of -log softmax(z)[label]; fused, with the row maximum subtracted."""

    def forward(self, z, labels):
        self.p, self.labels = [], labels
        total = 0.0
        for r, c in zip(z, labels):
            m = max(r)
            e = [math.exp(v - m) for v in r]
            s = sum(e)
            self.p.append([v / s for v in e])
            total += (m + math.log(s)) - r[c]
        return total / len(z)

    def backward(self):
        B = len(self.p)
        return [[(pj - (1.0 if j == c else 0.0)) / B for j, pj in enumerate(pr)]
                for pr, c in zip(self.p, self.labels)]


class MeanSquaredError:
    """Mean over batch and outputs of (y - t)^2; `divisor` overrides B*O (the planted fault of milestone 2)."""

    def __init__(self, divisor=None):
        self.divisor = divisor

    def forward(self, y, t):
        self.y, self.t = y, t
        self.n = self.divisor or len(y) * len(y[0])
        return sum((a - b) ** 2 for yr, tr in zip(y, t) for a, b in zip(yr, tr)) / (len(y) * len(y[0]))

    def backward(self):
        return [[2.0 * (a - b) / self.n for a, b in zip(yr, tr)] for yr, tr in zip(self.y, self.t)]


# ------------------------------------------------------------------ the gradient checker

def gate_ratio(a, n, atol=ATOL, rtol=RTOL):
    return abs(a - n) / (atol + rtol * max(abs(a), abs(n)))


def check_entries(loss, theta, grad, indices, h=H_CHECK, atol=ATOL, rtol=RTOL):
    """Per-entry central differences: (failing indices, worst gate ratio)."""
    failing, worst = [], 0.0
    for i in indices:
        up, down = list(theta), list(theta)
        up[i] += h
        down[i] -= h
        num = (loss(up) - loss(down)) / (2.0 * h)
        r = gate_ratio(grad[i], num, atol, rtol)
        worst = max(worst, r)
        if r > 1.0:
            failing.append(i)
    return failing, worst


def check_directional(loss, grad, w, rng, h=H_CHECK, n_dirs=4, atol=ATOL, rtol=RTOL):
    """Best of n_dirs random +-1 directions by relative error; True when that direction passes the gate."""
    best_err, best_ok = math.inf, False
    for _ in range(n_dirs):
        v = [1.0 if rng.integer(2) == 1 else -1.0 for _ in w]
        ana = sum(g * s for g, s in zip(grad, v))
        up = [a + h * s for a, s in zip(w, v)]
        down = [a - h * s for a, s in zip(w, v)]
        num = (loss(up) - loss(down)) / (2.0 * h)
        err = abs(ana - num) / max(abs(ana), abs(num), 0.001)
        if err < best_err:
            best_err, best_ok = err, gate_ratio(ana, num, atol, rtol) <= 1.0
    return best_ok


def distinct_dims(dims):
    """The test-dimension policy: every size above 1 and no two sizes equal."""
    return all(d > 1 for d in dims) and len(set(dims)) == len(dims)


def layer_probe(layer, x, R, mode="train", seed_name=None, backward=None):
    """Probe loss L = sum(y * R). Returns (L, dx, loss closure over theta = [x; params], analytic gradient)."""
    B, D = len(x), len(x[0])
    seed = bookrng.seed_of(_name(seed_name)) if seed_name else None

    def run(xs):
        rng = bookrng.SplitMix64(seed) if seed is not None else None
        y = layer.forward(xs, mode, rng)
        return y, sum(a * b for yr, rr in zip(y, R) for a, b in zip(yr, rr))

    zero_grad(layer)
    y, L = run(x)
    dx = (backward or layer.backward)(R)
    own = [(lab, p, g) for lab, p, g in tensors(layer)]
    grad = [v for r in dx for v in r] + [v for _, _, g in own for v in g]
    theta0 = [v for r in x for v in r] + [v for _, p, _ in own for v in p]

    def loss(theta):
        n = B * D
        xs = rows_of(theta[:n], D)
        k = n
        for _, p, _ in own:
            p[:] = theta[k:k + len(p)]
            k += len(p)
        return run(xs)[1]

    return L, y, dx, loss, theta0, grad, own


def check_layer(layer, x, R, mode="train", seed_name=None, backward=None):
    """Per-entry and directional check of the input and of every parameter tensor of one layer."""
    L, y, dx, loss, theta0, grad, own = layer_probe(layer, x, R, mode, seed_name, backward)
    groups = [("input", 0, len(x) * len(x[0]))]
    k = len(x) * len(x[0])
    for lab, p, _ in own:
        groups.append((lab.split(".", 1)[1], k, k + len(p)))
        k += len(p)
    rng = _gen("directions")
    rows = []
    for name, lo, hi in groups:
        failing, worst = check_entries(loss, theta0, grad, range(lo, hi))
        sub_loss = (lambda lo, hi: lambda w: loss(theta0[:lo] + w + theta0[hi:]))(lo, hi)
        ok = check_directional(sub_loss, grad[lo:hi], theta0[lo:hi], rng)
        rows.append((name, hi - lo, len(failing), worst, ok))
    loss(theta0)  # restore the parameters
    return rows


# ------------------------------------------------------------------ optimizers, schedule, clip

def sgd_step(theta, g, eta):
    for i in range(len(theta)):
        theta[i] -= eta * g[i]


def momentum_step(theta, g, m, eta, beta, nesterov=False):
    for i in range(len(theta)):
        m[i] = beta * m[i] + g[i]
        d = g[i] + beta * m[i] if nesterov else m[i]
        theta[i] -= eta * d


def adam_step(theta, g, m, v, t, eta, beta1=0.9, beta2=0.999, eps=1e-8, lam=0.0, decoupled=True):
    bc1, bc2 = 1.0 - beta1 ** t, 1.0 - beta2 ** t
    for i in range(len(theta)):
        gi = g[i] if decoupled else g[i] + lam * theta[i]
        m[i] = beta1 * m[i] + (1.0 - beta1) * gi
        v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi
        if decoupled:
            theta[i] -= eta * lam * theta[i]
        theta[i] -= eta * (m[i] / bc1) / (math.sqrt(v[i] / bc2) + eps)


def lion_step(theta, g, m, eta, beta1, beta2, lam):
    for i in range(len(theta)):
        c = beta1 * m[i] + (1.0 - beta1) * g[i]
        u = 1.0 if c > 0.0 else (-1.0 if c < 0.0 else 0.0)
        theta[i] -= eta * (u + lam * theta[i])
        m[i] = beta2 * m[i] + (1.0 - beta2) * g[i]


def schedule_rate(t, peak, floor, warmup, hold, total):
    if t < warmup:
        return peak * (t + 1) / warmup
    if t < warmup + hold:
        return peak
    u = min(1.0, (t - warmup - hold) / (total - warmup - hold))
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * u))


def global_norm_clip(grads, tau):
    """Scale every tensor in place by min(1, tau / (norm + 1e-6)); return the norm before scaling."""
    norm = math.sqrt(sum(a * a for g in grads for a in g))
    c = min(1.0, tau / (norm + 1e-6))
    for g in grads:
        g[:] = [c * a for a in g]
    return norm


# ------------------------------------------------------------------ initialisation

def init_linear(n_out, n_in, scheme, shape, rng, factor=1.0):
    var = {"he": 2.0 / n_in, "xavier": 2.0 / (n_in + n_out), "fan_in": 1.0 / n_in}[scheme]
    sigma = factor * math.sqrt(var)
    W = []
    for _ in range(n_out * n_in):
        if shape == "normal":
            W.append(sigma * rng.normal())
        elif shape == "uniform":
            W.append(sigma * math.sqrt(3.0) * (2.0 * rng.uniform() - 1.0))
        else:  # truncated at two standard deviations, rescaled to the same variance
            while True:
                z = rng.normal()
                if abs(z) <= 2.0:
                    break
            W.append(sigma * z / math.sqrt(0.7737))
    return W


# ------------------------------------------------------------------ the training loop

def evaluate(model, loss_fn, X, Y, mode="eval", rng=None):
    out = model.forward(X, mode, rng)
    return loss_fn.forward(out, Y), out


def accuracy(out, labels):
    return sum(1 for r, c in zip(out, labels) if r.index(max(r)) == c) / len(labels)


class Trainer:
    """The training loop of chapter 13 with logging, early stopping, checkpointing and exact resume.

    cfg keys: epochs, batch, peak, floor_ratio, tau, lam, patience (None for no early stopping),
    total_override (planted fault: a schedule horizon in the wrong unit), eval_mode (planted fault)."""

    def __init__(self, model, loss_fn, X, Y, Xv, Yv, cfg, batch_stream, dropout_stream=None, classify=False):
        self.model, self.loss_fn, self.cfg, self.classify = model, loss_fn, cfg, classify
        self.X, self.Y, self.Xv, self.Yv = X, Y, Xv, Yv
        self.rng_b = _gen(batch_stream)
        self.rng_d = _gen(dropout_stream) if dropout_stream else None
        P = len(get_theta(model))
        self.m, self.v = [0.0] * P, [0.0] * P
        self.order = list(range(len(X)))
        self.S = len(X) // cfg["batch"]
        assert len(X) % cfg["batch"] == 0
        self.total = cfg.get("total_override") or cfg["epochs"] * self.S
        self.warmup = self.total // 10
        self.floor = cfg["peak"] * cfg.get("floor_ratio", 0.01)
        self.t, self.epoch, self.wait, self.stopped = 0, 0, 0, False
        self.log = []
        self.start = self._scores()
        self.best_val, self.best, self.best_epoch = self.start["val"], get_theta(model), 0

    def _scores(self):
        mode = self.cfg.get("eval_mode", "eval")
        tr, _ = evaluate(self.model, self.loss_fn, self.X, self.Y, "eval")
        va, out = evaluate(self.model, self.loss_fn, self.Xv, self.Yv, mode, self.rng_d if mode == "train" else None)
        row = {"train": tr, "val": va}
        if self.classify:
            row["acc"] = accuracy(out, self.Yv)
        return row

    def run(self, until):
        cfg, model = self.cfg, self.model
        while self.epoch < until and not self.stopped:
            self.rng_b.shuffle(self.order)
            norms, clipped = [], 0
            for k in range(self.S):
                idx = self.order[k * cfg["batch"]:(k + 1) * cfg["batch"]]
                xb, yb = [self.X[i] for i in idx], [self.Y[i] for i in idx]
                zero_grad(model)
                out = model.forward(xb, "train", self.rng_d)
                loss = self.loss_fn.forward(out, yb)
                if not math.isfinite(loss):
                    raise ValueError("the loss is not finite")
                model.backward(self.loss_fn.backward())
                gs = [g for _, _, g in tensors(model)]
                norm = global_norm_clip(gs, cfg["tau"])
                clipped += cfg["tau"] / (norm + 1e-6) < 1.0
                theta, grad = get_theta(model), get_grad(model)
                eta = schedule_rate(self.t, cfg["peak"], self.floor, self.warmup, 0, self.total)
                adam_step(theta, grad, self.m, self.v, self.t + 1, eta, 0.9, 0.999, 1e-8, cfg["lam"], True)
                set_theta(model, theta)
                norms.append(norm)
                self.t += 1
            self.epoch += 1
            row = {"epoch": self.epoch, **self._scores(), "norm": sum(norms) / len(norms),
                   "clipped": clipped, "lr": eta}
            self.log.append(row)
            if row["val"] < self.best_val:
                self.best_val, self.best, self.best_epoch, self.wait = row["val"], get_theta(model), self.epoch, 0
            else:
                self.wait += 1
                if cfg.get("patience") and self.wait >= cfg["patience"]:
                    self.stopped = True
        return self

    # --- checkpoint: everything the next step reads
    def save(self):
        return json.dumps({"theta": get_theta(self.model), "m": self.m, "v": self.v, "t": self.t,
                           "epoch": self.epoch, "order": self.order, "rng_b": self.rng_b.state,
                           "rng_d": self.rng_d.state if self.rng_d else None, "best": self.best,
                           "best_val": self.best_val, "best_epoch": self.best_epoch, "wait": self.wait,
                           "stopped": self.stopped, "log": self.log})

    def load(self, text):
        s = json.loads(text)
        set_theta(self.model, s["theta"])
        self.m, self.v, self.t, self.epoch, self.order = s["m"], s["v"], s["t"], s["epoch"], s["order"]
        self.rng_b = bookrng.SplitMix64(s["rng_b"])
        self.rng_d = bookrng.SplitMix64(s["rng_d"]) if s["rng_d"] is not None else None
        self.best, self.best_val, self.best_epoch = s["best"], s["best_val"], s["best_epoch"]
        self.wait, self.stopped, self.log = s["wait"], s["stopped"], s["log"]


def early_stop_epoch(val_losses, patience):
    """The rule of the chapter. val_losses[0] is the loss before training (epoch 0).
    Returns (epoch of the best loss, epoch at which training stops or None)."""
    best, best_epoch, wait = val_losses[0], 0, 0
    for e, v in enumerate(val_losses[1:], start=1):
        if v < best:
            best, best_epoch, wait = v, e, 0
        else:
            wait += 1
            if wait >= patience:
                return best_epoch, e
    return best_epoch, None


# ------------------------------------------------------------------ the two models

N_IN, N_CLASSES = 8, 3
READOUT_FACTOR = 0.1
N_TRAIN, N_VAL, N_TEST = 512, 256, 256


@lru_cache(maxsize=None)
def task():
    """Teacher vectors u_0..u_2, v_0..v_2 (8 normals each), then 1024 inputs of 8 normals; label = argmax_k
    2 (u_k . x) + (v_k . x)^2 with the first index winning a tie."""
    g = _gen("task")
    U = [[g.normal() for _ in range(N_IN)] for _ in range(N_CLASSES)]
    V = [[g.normal() for _ in range(N_IN)] for _ in range(N_CLASSES)]
    X = [[g.normal() for _ in range(N_IN)] for _ in range(N_TRAIN + N_VAL + N_TEST)]
    y = []
    for x in X:
        sc = [2.0 * sum(map(mul, U[k], x)) + sum(map(mul, V[k], x)) ** 2 for k in range(N_CLASSES)]
        y.append(sc.index(max(sc)))
    a, b = N_TRAIN, N_TRAIN + N_VAL
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def make_classifier(p_drop=0.3, init="classifier_init"):
    g = _gen(init)

    def lin(n_in, n_out, scheme, factor=1.0):
        return Linear(n_in, n_out, W=init_linear(n_out, n_in, scheme, "normal", g, factor))

    first = lin(N_IN, 32, "he")
    inner = lin(32, 32, "he")
    last = lin(32, N_CLASSES, "fan_in", READOUT_FACTOR)
    return Sequential([first, Activation("gelu"),
                       Residual(Sequential([LayerNorm(32), inner, Activation("gelu"), Dropout(p_drop)])),
                       last])


CLS_RUN = dict(epochs=40, batch=32, peak=3e-3, tau=1.0, lam=0.01, patience=8)


def train_classifier(cfg=None, labels=None, p_drop=0.3, epochs=None):
    (X, y), (Xv, yv), _ = task()
    model = make_classifier(p_drop)
    c = dict(CLS_RUN, **(cfg or {}))
    if epochs:
        c["epochs"] = epochs
    tr = Trainer(model, SoftmaxCrossEntropy(), X, labels if labels is not None else y, Xv, yv, c,
                 "classifier_batches", "classifier_dropout", classify=True)
    return tr.run(c["epochs"])


# the autoencoder of chapter 13: 16 -> 12 -> 3 -> 12 -> 16, GELU after the two hidden layers
SENSORS = 16
POSITIONS = [j / (SENSORS - 1) for j in range(SENSORS)]
PATTERNS = [[math.cos(math.pi * p) for p in POSITIONS], [math.sin(2.0 * math.pi * p) for p in POSITIONS],
            [math.cos(2.0 * math.pi * p) for p in POSITIONS]]
AE_SCALES, AE_NOISE = (1.0, 0.5, 0.25), 0.05
AE_RUN = dict(epochs=300, batch=32, peak=1e-2, tau=1.0, lam=0.01, patience=None)


@lru_cache(maxsize=None)
def sensor_data():
    """512 readings; per example s1, s2, s3 then n_0 .. n_15, all normal(); split 256 / 128 / 128."""
    g = _gen("first_models.ae_data")
    X = []
    for _ in range(512):
        s = [g.normal() for _ in range(3)]
        n = [g.normal() for _ in range(SENSORS)]
        X.append([sum(AE_SCALES[k] * s[k] * PATTERNS[k][j] for k in range(3)) + AE_NOISE * n[j]
                  for j in range(SENSORS)])
    return X[:256], X[256:384], X[384:]


def make_autoencoder(init="first_models.ae_init"):
    g = _gen(init)
    dims = [(16, 12), (12, 3), (3, 12), (12, 16)]
    L = [Linear(i, o, W=init_linear(o, i, "fan_in", "normal", g)) for i, o in dims]
    return Sequential([L[0], Activation("gelu"), L[1], L[2], Activation("gelu"), L[3]])


@lru_cache(maxsize=None)
def train_autoencoder():
    tr, va, te = sensor_data()
    model = make_autoencoder()
    t = Trainer(model, MeanSquaredError(), tr, tr, va, va, AE_RUN, "first_models.batches")
    t.run(AE_RUN["epochs"])
    return t


# ================================================================== acceptance tables

B_T, D_T, H_T = 3, 5, 7          # distinct test sizes of the layer fixtures (batch, features, hidden)
LABEL_PRIOR_CLS = None


def _fixture_cases():
    """(label, layer, input, probe, mode, dropout stream name) for every layer of milestone 1."""
    x = rows_of(fix(B_T * D_T, 0.7, 0.3, 1.0), D_T)
    gamma = [1.0 + v for v in fix(D_T, 1.7, 0.5, 0.3)]
    beta = fix(D_T, 2.3, 0.9, 0.2)
    probe_d = rows_of(fix(B_T * D_T, 1.1, 0.4, 1.0), D_T)
    probe_h = rows_of(fix(B_T * H_T, 1.1, 0.4, 1.0), H_T)
    lin = Linear(D_T, H_T, W=fix(H_T * D_T, 0.9, 0.2, 0.5), b=fix(H_T, 1.3, 0.1, 0.3))
    res = Residual(Sequential([Linear(D_T, H_T, W=fix(H_T * D_T, 0.9, 0.2, 0.5), b=fix(H_T, 1.3, 0.1, 0.3)),
                               Activation("gelu"),
                               Linear(H_T, D_T, W=fix(D_T * H_T, 0.6, 0.7, 0.5), b=fix(D_T, 1.9, 0.3, 0.2))]))
    cases = [("linear 5 -> 7", lin, x, probe_h, "train", None)]
    for kind in ("tanh", "relu", "gelu", "silu"):
        cases.append((kind, Activation(kind), x, probe_d, "train", None))
    cases += [("layer norm", LayerNorm(D_T, gamma=gamma, beta=beta), x, probe_d, "train", None),
              ("RMS norm", LayerNorm(D_T, gamma=gamma, rms=True), x, probe_d, "train", None),
              ("dropout p = 0.4", Dropout(0.4), x, probe_d, "train", "dropout_fixture"),
              ("residual (5 -> 7 -> 5)", res, x, probe_d, "train", None)]
    return cases


def layer_values():
    cases = _fixture_cases()
    x = cases[0][2]
    margin = min(abs(v) for r in x for v in r)
    rows, prow = [], []
    for label, layer, xin, R, mode, dname in cases:
        L, y, dx, loss, theta0, grad, own = layer_probe(layer, xin, R, mode, dname)
        rows.append([label, f"{len(y)} x {len(y[0])}", _f(L, 8), _f(y[0][0], 8), _f(y[-1][-1], 8),
                     _f(dx[0][0], 8), _f(dx[-1][-1], 8)])
        for lab, p, g in own:
            prow.append([label, lab.split(".", 1)[1] if label.startswith(("linear", "layer", "RMS")) else lab,
                         _f(g[0], 8), _f(g[-1], 8), f"{sum(a * a for a in g):.8f}"])
    draws = _gen("dropout_fixture")
    mask = [[(1.0 / 0.6) if draws.uniform() >= 0.4 else 0.0 for _ in range(D_T)] for _ in range(B_T)]
    head = (f"Input x: [3, 5], x[b, i] = sin(0.7 k + 0.3) with k = 5 b + i (smallest |x| = {margin:.4f}, so ReLU is "
            f"{margin / H_CHECK:.0f} steps of the checker from its kink). Probe R: the same rule with frequency 1.1 and "
            f"phase 0.4, shape [3, out]; the layer's scalar loss is L = sum(y * R), so dy = R.\n"
            "linear: W[j, i] = 0.5 sin(0.9 k + 0.2) with k = 5 j + i, b[j] = 0.3 sin(1.3 j + 0.1). "
            "norms: gamma[i] = 1 + 0.3 sin(1.7 i + 0.5), beta[i] = 0.2 sin(2.3 i + 0.9), eps = 1e-5. "
            f"dropout: p = 0.4 with Generator {_seed('dropout_fixture')}, 15 uniform() draws in row-major order, kept when the draw is at least p "
            f"(mask times 1/0.6: {' '.join(str(int(round(v * 0.6))) for r in mask for v in r)}). "
            "residual: body = linear 5 -> 7 (the linear above), gelu, linear 7 -> 5 with W[j, i] = 0.5 sin(0.6 k + 0.7), "
            "b[j] = 0.2 sin(1.9 j + 0.3).\n\n")
    t1 = _table(["layer", "y shape", "L", "y[0,0]", "y[2,last]", "dx[0,0]", "dx[2,4]"], rows)
    t2 = _table(["layer", "tensor", "g[0]", "g[last]", "sum of squares"], prow)
    return head + t1 + "\n\nParameter gradients after one backward pass from zero (tensor labels are k.name, k the index of the " \
        "parameter-owning layer inside the layer under test):\n\n" + t2


def loss_values():
    z = rows_of(fix(12, 0.8, 0.5, 2.0), 4)
    labels = [2, 0, 3]
    ce = SoftmaxCrossEntropy()
    L = ce.forward(z, labels)
    dz = ce.backward()
    big = SoftmaxCrossEntropy()
    l0 = big.forward([[710.0, 0.0, -5.0]], [0])
    l1 = SoftmaxCrossEntropy().forward([[710.0, 0.0, -5.0]], [1])
    y = rows_of(fix(12, 0.8, 0.5, 2.0), 4)
    t = rows_of(fix(12, 1.4, 0.2, 1.5), 4)
    mse = MeanSquaredError()
    Lm = mse.forward(y, t)
    dy = mse.backward()
    r1 = [["softmax cross-entropy", _f(L, 8), _f(dz[0][0], 8), _f(dz[0][2], 8), _f(dz[2][3], 8), str(abs(sum(sum(r) for r in dz)) < 1e-15)],
          ["mean squared error", _f(Lm, 8), _f(dy[0][0], 8), _f(dy[0][2], 8), _f(dy[2][3], 8), "-"]]
    return ("Logits z (softmax cross-entropy): z[b, j] = 2 sin(0.8 k + 0.5), k = 4 b + j, shape [3, 4]; labels (2, 0, 3). "
            "Mean squared error: the outputs y are the same 12 numbers as z (y = z, shape [3, 4]) and the targets are t[b, j] = 1.5 sin(1.4 k + 0.2), the mean taken over all 12 entries.\n\n"
            + _table(["loss", "value", "d[0,0]", "d[0,2]", "d[2,3]", "sum of d below 1e-15"], r1)
            + f"\n\nOverflow case, one row z = (710, 0, -5): label 0 gives loss {l0:.8f}, label 1 gives loss {l1:.8f}; "
              f"every value is finite and the gradient row sums to zero (below 1e-15: {abs(sum(big.backward()[0])) < 1e-15}).")


def check_loss_fn(loss_fn, out, target):
    """Directional/per-entry check of a loss with respect to its input; returns (entries, failing, worst)."""
    flat = [v for r in out for v in r]
    D = len(out[0])

    def f(theta):
        return loss_fn.forward(rows_of(theta, D), target)

    f(flat)
    grad = [v for r in loss_fn.backward() for v in r]
    failing, worst = check_entries(f, flat, grad, range(len(flat)))
    return len(flat), len(failing), worst


def gradient_check_layers():
    rows = []
    for label, layer, x, R, mode, dname in _fixture_cases():
        for name, n, nfail, worst, ok in check_layer(layer, x, R, mode, dname):
            rows.append([label, name, n, nfail, f"{worst:.2e}", "pass" if ok else "FAIL"])
    z = rows_of(fix(12, 0.8, 0.5, 2.0), 4)
    n, nf, w = check_loss_fn(SoftmaxCrossEntropy(), z, [2, 0, 3])
    rows.append(["softmax cross-entropy", "logits", n, nf, f"{w:.2e}", "-"])
    t = rows_of(fix(12, 1.4, 0.2, 1.5), 4)
    n, nf, w = check_loss_fn(MeanSquaredError(), z, t)
    rows.append(["mean squared error", "outputs", n, nf, f"{w:.2e}", "-"])
    return ("Fixtures of the previous table. Per-entry check: central difference with h = 1e-5 on every entry of the input and of "
            "every parameter tensor, gate |a - n| <= 1e-7 + 1e-5 max(|a|, |n|); 'worst' is the largest of |a - n| / (1e-7 + 1e-5 max(|a|, |n|)), "
            "which must stay below 1. Directional check: best of 4 random +-1 directions, h = 1e-5, directions from "
            f"Generator {_seed('directions')} (one integer(2) per entry, tensors in the order of the table, 4 directions each; "
            "+1 when the draw is 1). The two losses are checked per entry only. The dropout rows hold the mask fixed by "
            "restarting the same generator before every evaluation.\n\n"
            + _table(["layer", "tensor", "entries", "failing", "worst ratio", "directional"], rows))


def broken_layer():
    cases = _fixture_cases()
    label, layer, x, R, mode, _ = next(c for c in cases if c[0] == "layer norm")
    rows = []
    for tag, bw in (("correct", None), ("without the centring term", lambda dy: layer.backward(dy, centre_term=False))):
        L, y, dx, loss, theta0, grad, own = layer_probe(layer, x, R, mode, None, bw)
        rng = _gen("directions_broken")
        off = 0
        for name, lo, hi in [("input", 0, B_T * D_T), ("gamma", B_T * D_T, B_T * D_T + D_T), ("beta", B_T * D_T + D_T, B_T * D_T + 2 * D_T)]:
            fl, worst = check_entries(loss, theta0, grad, range(lo, hi))
            sub = (lambda lo, hi: lambda w: loss(theta0[:lo] + w + theta0[hi:]))(lo, hi)
            ok = check_directional(sub, grad[lo:hi], theta0[lo:hi], rng)
            rows.append([tag, name, hi - lo, len(fl), f"{worst:.2e}", "pass" if ok else "FAIL"])
        loss(theta0)
    return ("The layer-norm fixture of the previous tables. The planted fault is the backward pass of RMS norm used for layer norm: "
            "dx = inv_std (dxhat - xhat mean(dxhat xhat)), the term mean(dxhat) is missing. Directions: "
            f"Generator {_seed('directions_broken')}, 4 per tensor, in the order of the table, correct layer first.\n\n"
            + _table(["backward pass", "tensor", "entries", "failing", "worst ratio", "directional"], rows))


def dims_policy():
    rows = []
    for (B, D, O) in [(1, 1, 1), (3, 5, 1), (3, 5, 3), (3, 5, 2)]:
        lin = Linear(D, O, W=fix(O * D, 0.9, 0.2, 0.5), b=fix(O, 1.3, 0.1, 0.3))
        x = rows_of(fix(B * D, 0.7, 0.3, 1.0), D)
        t = rows_of(fix(B * O, 1.4, 0.2, 1.5), O)
        res = []
        for tag, div in (("correct", None), ("divides by B", B)):
            mse = MeanSquaredError(divisor=div)
            lossgood = MeanSquaredError()

            def loss(theta, lin=lin, x=x, t=t):
                set_theta(lin, theta)
                return lossgood.forward(lin.forward(x), t)

            theta0 = get_theta(lin)
            zero_grad(lin)
            mse.forward(lin.forward(x), t)
            lin.backward(mse.backward())
            fl, _ = check_entries(loss, theta0, get_grad(lin), range(len(theta0)))
            res.append(f"{len(fl)} of {len(theta0)}")
        rows.append([f"({B}, {D}, {O})", "yes" if distinct_dims([B, D, O]) else "no", res[0], res[1]])
    return ("Fault: the gradient of the mean squared error divides by B where B * O belongs. Model: linear D -> O on a batch of B, "
            "fixtures as in the layer table (weights 0.5 sin(0.9 k + 0.2), bias 0.3 sin(1.3 j + 0.1), inputs sin(0.7 k + 0.3), "
            "targets 1.5 sin(1.4 k + 0.2)); per-entry check on every parameter, same gate. 'Distinct' is the policy test: every size above 1, no two equal.\n\n"
            + _table(["(B, D, O)", "distinct", "failing entries, correct backward", "failing entries, with the fault"], rows))


@lru_cache(maxsize=None)
def _model_check_rows(which):
    if which.startswith("classifier"):
        (X, y), _, _ = task()
        y = y[1:] + y[:1] if which == "classifier_shifted" else y
        model, loss_fn, xb, yb = make_classifier(), SoftmaxCrossEntropy(), X[:5], y[:5]
        seed = bookrng.seed_of(_name("model_check_dropout"))
    else:
        tr, _, _ = sensor_data()
        model, loss_fn, xb, yb = make_autoencoder(), MeanSquaredError(), tr[:8], tr[:8]
        seed = None

    def forward_loss():
        rng = bookrng.SplitMix64(seed) if seed is not None else None
        return loss_fn.forward(model.forward(xb, "train", rng), yb)

    zero_grad(model)
    L0 = forward_loss()
    model.backward(loss_fn.backward())
    grad = get_grad(model)
    theta0 = get_theta(model)

    def loss(theta):
        set_theta(model, theta)
        return forward_loss()

    rng = _gen("directions_model_" + which.split("_")[0])
    rows, k = [], 0
    for lab, p, _ in tensors(model):
        lo, hi = k, k + len(p)
        k = hi
        if which == "autoencoder" or len(p) <= 64:
            fl, worst = check_entries(loss, theta0, grad, range(lo, hi))
            rows.append([lab, len(p), "per entry", len(fl), f"{worst:.2e}"])
        else:
            sub = (lambda lo, hi: lambda w: loss(theta0[:lo] + w + theta0[hi:]))(lo, hi)
            ok = check_directional(sub, grad[lo:hi], theta0[lo:hi], rng)
            rows.append([lab, len(p), "directional", "-", "pass" if ok else "FAIL"])
    loss(theta0)
    return L0, len(theta0), rows


def model_gradient_checks():
    L0, P, rows = _model_check_rows("classifier")
    L1, P1, rows1 = _model_check_rows("autoencoder")
    return ("Classifier of milestone 6 (hidden width 32, dropout p = 0.3 active, mask held fixed by restarting "
            f"Generator {_seed('model_check_dropout')} before every evaluation) on training rows 0 to 4: loss {L0:.6f}, {P} parameters. "
            "Tensors of at most 64 entries are checked per entry (gate ratio below 1 means pass), the larger ones by the directional check with 4 directions from "
            f"Generator {_seed('directions_model_classifier')}.\n\n"
            + _table(["tensor", "entries", "check", "failing", "worst ratio / verdict"], rows)
            + f"\n\nAutoencoder of milestone 6 on training readings 0 to 7: loss {L1:.6f}, {P1} parameters, every entry checked per entry; "
              f"failing entries in total: {sum(r[3] for r in rows1)}; the largest ratio of any tensor: {max(float(r[4]) for r in rows1):.2e}.")


def optimizer_trajectories():
    theta0 = [1.0, -2.0, 0.5]
    eta = 0.1

    def grad(t):
        return [0.5 * math.sin(0.8 * (3 * t + i) + 0.3) for i in range(3)]

    rules = [
        ("SGD", lambda th, g, st, t: sgd_step(th, g, eta)),
        ("momentum, beta 0.9", lambda th, g, st, t: momentum_step(th, g, st["m"], eta, 0.9)),
        ("Nesterov, beta 0.9", lambda th, g, st, t: momentum_step(th, g, st["m"], eta, 0.9, True)),
        ("Adam", lambda th, g, st, t: adam_step(th, g, st["m"], st["v"], t + 1, eta, 0.9, 0.999, 1e-8, 0.0, True)),
        ("Adam, coupled L2 0.1", lambda th, g, st, t: adam_step(th, g, st["m"], st["v"], t + 1, eta, 0.9, 0.999, 1e-8, 0.1, False)),
        ("AdamW, decay 0.1", lambda th, g, st, t: adam_step(th, g, st["m"], st["v"], t + 1, eta, 0.9, 0.999, 1e-8, 0.1, True)),
        ("Lion, 0.9, 0.99, decay 0.1", lambda th, g, st, t: lion_step(th, g, st["m"], eta, 0.9, 0.99, 0.1)),
    ]
    shown = (1, 2, 3, 10)
    rows = []
    for name, step in rules:
        th = list(theta0)
        st = {"m": [0.0] * 3, "v": [0.0] * 3}
        for t in range(10):
            step(th, grad(t), st, t)
            if t + 1 in shown:
                rows.append([name, t + 1] + [f"{v:+.8f}" for v in th])
    g0 = grad(0)
    gl = [", ".join(f"{v:+.6f}" for v in grad(t)) for t in range(3)]
    return ("Start theta = (1.0, -2.0, 0.5). The gradient at step t (t = 0, 1, 2, ...) is g_t[i] = 0.5 sin(0.8 (3 t + i) + 0.3), "
            f"independent of theta: g_0 = ({gl[0]}), g_1 = ({gl[1]}), g_2 = ({gl[2]}). Rate 0.1 for every rule and every step; Adam and AdamW "
            "use (0.9, 0.999, 1e-8), Lion (0.9, 0.99); t is the 1-based step in the bias correction. Rows give theta after the "
            "stated number of steps.\n\n"
            + _table(["rule", "steps", "theta[0]", "theta[1]", "theta[2]"], rows))


def schedule_and_clip():
    ts = (0, 4, 9, 10, 11, 20, 55, 99)
    rows = [[t, f"{schedule_rate(t, 0.01, 1e-4, 10, 0, 100):.8f}", f"{schedule_rate(t, 0.01, 1e-4, 10, 10, 100):.8f}"] for t in ts]
    last = schedule_rate(99, 0.01, 1e-4, 10, 0, 100)
    at_total = schedule_rate(100, 0.01, 1e-4, 10, 0, 100)
    out = ("Schedule: peak 0.01, floor 1e-4, warmup 10, total 100; the second column holds the peak for 10 steps after the warmup. "
           "t is the 0-based step. The last step of the run is t = 99 = total - 1; its rate is not yet the floor (the floor is reached at t = total = 100, "
           f"where the rate is {at_total:.8f}, and stays there).\n\n" + _table(["t", "hold 0", "hold 10"], rows))
    crows = []
    for tau in (5.0, 20.0):
        g = [[3.0, 4.0], [0.0, 12.0]]
        n = global_norm_clip(g, tau)
        crows.append([tau, f"{n:.8f}", f"{min(1.0, tau / (n + 1e-6)):.8f}", f"({g[0][0]:.6f}, {g[0][1]:.6f})", f"({g[1][0]:.6f}, {g[1][1]:.6f})"])
    out += ("\n\nClip: two gradient tensors (3, 4) and (0, 12), global norm 13; scale = min(1, tau / (norm + 1e-6)).\n\n"
            + _table(["tau", "norm before", "scale", "tensor 1 after", "tensor 2 after"], crows))
    return out


def initialization_statistics():
    n_out, n_in = 128, 96
    rows = []
    for scheme, shape in (("he", "normal"), ("he", "uniform"), ("he", "truncated"), ("xavier", "normal"), ("fan_in", "normal")):
        g = _gen(f"init.{scheme}.{shape}")
        W = init_linear(n_out, n_in, scheme, shape, g)
        mean = sum(W) / len(W)
        sd = math.sqrt(sum((v - mean) ** 2 for v in W) / len(W))
        var = {"he": 2.0 / n_in, "xavier": 2.0 / (n_in + n_out), "fan_in": 1.0 / n_in}[scheme]
        rows.append([scheme, shape, ", ".join(f"{v:+.6f}" for v in W[:3]), f"{math.sqrt(var):.6f}", f"{sd:.6f}", f"{max(abs(v) for v in W):.4f}"])
    return ("A weight matrix with n_out = 128 and n_in = 96 (12288 entries, row-major), drawn from a fresh generator per row named "
            "init.<scheme>.<shape> exactly, with nothing before it, e.g. the string init.he.normal (the seed of a stream is the first 8 bytes, read as a big-endian unsigned integer, of the SHA-256 digest of its name; the seeds are printed below, so no derivation is needed; the draw is the one call of the chapter-9 rule per entry, "
            "the truncated shape redrawing normal() until |z| <= 2). 'target sd' is sqrt(2/n_in), sqrt(2/(n_in+n_out)) or sqrt(1/n_in). 'sd of the 12288' is the population standard deviation with divisor n = 12288 after subtracting the mean of the entries (not the divisor n - 1 of the book's sd).\n\n"
            + _table(["scheme", "shape", "first three weights", "target sd", "sd of the 12288 (divisor n)", "largest |w|"], rows, "llrrrr")
            + "\n\nSeeds: " + "; ".join(f"init.{s}.{h} {_seed('init.' + s + '.' + h)}" for s, h in
                                       (("he", "normal"), ("he", "uniform"), ("he", "truncated"), ("xavier", "normal"), ("fan_in", "normal"))))


def early_stopping_rule():
    vals = [1.20, 0.90, 0.70, 0.55, 0.50, 0.51, 0.49, 0.50, 0.52, 0.53, 0.54, 0.40]
    tie = [1.0, 0.8, 0.6, 0.6, 0.6, 0.6]
    rows = []
    for p in (1, 2, 3, 5):
        b, st = early_stop_epoch(vals, p)
        rows.append([p, b, "never" if st is None else st])
    tb, ts = early_stop_epoch(tie, 3)
    return ("Validation losses by epoch, epoch 0 (before training) first: " + ", ".join(f"{v:.2f}" for v in vals)
            + ". A loss is an improvement only when strictly below the best so far; training stops at the epoch where the number of "
              "consecutive non-improvements reaches the patience.\n\n"
            + _table(["patience", "best epoch (checkpoint kept)", "stops after epoch"], rows)
            + f"\n\nA run of equal losses, {', '.join(f'{v:.1f}' for v in tie)}, with patience 3 keeps epoch {tb}, the first of the equal values, "
              f"and stops after epoch {ts}.")


def accumulation():
    x = rows_of(fix(B_T * D_T, 0.7, 0.3, 1.0), D_T)
    dy = rows_of(fix(B_T * H_T, 1.1, 0.4, 1.0), H_T)
    lin = Linear(D_T, H_T, W=fix(H_T * D_T, 0.9, 0.2, 0.5), b=fix(H_T, 1.3, 0.1, 0.3))
    lin.forward(x)
    lin.backward(dy)
    once = get_grad(lin)
    lin.backward(dy)
    twice = get_grad(lin)
    zero_grad(lin)
    lin.backward(dy)
    again = get_grad(lin)
    zero_grad(lin)
    for lo, hi in ((0, 1), (1, 3)):
        lin.forward(x[lo:hi])
        lin.backward(dy[lo:hi])
    chunks = get_grad(lin)
    bound = lambda v: "below 1e-15" if v < 1e-15 else f"{v:.1e}"   # noqa: E731  (rounding-level differences depend on the order of the sums)
    rows = [["backward twice without zero_grad", bound(max(abs(a - 2 * b) for a, b in zip(twice, once))), "twice the gradient"],
            ["zero_grad, then backward again", bound(max(abs(a - b) for a, b in zip(again, once))), "the gradient"],
            ["rows 0 and rows 1 to 2 as two backward calls", bound(max(abs(a - b) for a, b in zip(chunks, once))), "the gradient"]]
    return ("Linear 5 -> 7 of the layer fixtures, x of shape [3, 5], dy of shape [3, 7] (the probe). 'once' is one backward call after "
            "zero_grad. The last row cuts the batch into rows {0} and {1, 2} and calls forward and backward on each with the matching rows of dy. "
            "A difference at the level of one rounding error (under 1e-15) is printed as a bound: its value depends on the order in which the sums are formed.\n\n"
            + _table(["experiment", "largest |difference| from the stated value", "expected"], rows, "lrl")
            + f"\n\nThe 'once' gradient has first entry {once[0]:+.8f} (the weight W[0,0]) and last entry {once[-1]:+.8f} (the bias b[6]).")


def resume_identity():
    (X, y), (Xv, yv), _ = task()
    X, y = X[:128], y[:128]
    cfg = dict(epochs=12, batch=32, peak=3e-3, tau=1.0, lam=0.01, patience=None)

    def fresh():
        return Trainer(make_classifier(), SoftmaxCrossEntropy(), X, y, Xv, yv, cfg, "resume_batches", "resume_dropout", classify=True)

    a = fresh().run(12)
    first = fresh().run(6)
    ckpt = first.save()
    rows = []

    def resumed(mutate=None):
        s = json.loads(ckpt)
        if mutate:
            mutate(s)
        t = fresh()
        t.load(json.dumps(s))
        return t.run(12)

    def row(tag, t):
        ta, tb = get_theta(a.model), get_theta(t.model)
        rows.append([tag, "yes" if ta == tb and a.log == t.log else "no", f"{max(abs(p - q) for p, q in zip(ta, tb)):.2e}",
                     f"{t.log[-1]['train']:.6f}", f"{t.log[-1]['val']:.6f}"])

    rows.append(["12 epochs without a stop", "-", "-", f"{a.log[-1]['train']:.6f}", f"{a.log[-1]['val']:.6f}"])
    row("6 epochs, save, load, 6 epochs", resumed())
    row("same, run twice in one process", fresh().run(12))
    row("checkpoint without the shuffle order (sorted instead)", resumed(lambda s: s.__setitem__("order", sorted(s["order"]))))
    row("checkpoint without the dropout generator state", resumed(lambda s: s.__setitem__("rng_d", bookrng.seed_of(_name("resume_dropout")))))
    row("checkpoint without the optimizer moments", resumed(lambda s: (s.__setitem__("m", [0.0] * len(s["m"])), s.__setitem__("v", [0.0] * len(s["v"])))))
    return ("Classifier of milestone 6 (dropout p = 0.3, initial weights from Generator " + _seed('classifier_init') + ") on the first 128 training rows (4 steps of 32 per epoch, the 256 validation rows for the log), "
            "12 epochs, no early stopping; AdamW (0.9, 0.999, 1e-8) with decay 0.01, clip 1, and the schedule of milestone 3 with peak 3e-3, warmup 4 steps (a tenth of the 48 steps), floor 3e-5 and total 48 "
            f"(hold 0). Streams: shuffles {_seed('resume_batches')}, dropout {_seed('resume_dropout')}. The three damaged checkpoints are made from the checkpoint saved after epoch 6: "
            "the order replaced by the sorted order 0 .. 127, the dropout generator replaced by a generator at its seed (the start of its stream), the moments m and v replaced by zeros (the step count kept). "
            "'Identical' means every parameter and every logged number equal with ==, not within a tolerance.\n\n"
            + _table(["run", "identical to the first row", "largest parameter difference", "training loss, epoch 12", "validation loss, epoch 12"], rows, "lrrrr"))


def overfit_one_batch(model, loss_fn, xb, yb, eta, steps):
    """AdamW without clip or decay on one fixed batch in eval mode; returns (first loss, last loss, accuracy or None)."""
    m, v = [0.0] * len(get_theta(model)), [0.0] * len(get_theta(model))
    first = None
    for t in range(steps + 1):
        zero_grad(model)
        out = model.forward(xb, "eval")
        loss = loss_fn.forward(out, yb)
        first = loss if first is None else first
        if t == steps:
            break
        model.backward(loss_fn.backward())
        theta = get_theta(model)
        adam_step(theta, get_grad(model), m, v, t + 1, eta, 0.9, 0.999, 1e-8, 0.0, True)
        set_theta(model, theta)
    acc = accuracy(out, yb) if isinstance(loss_fn, SoftmaxCrossEntropy) else None
    return first, loss, acc


def prior_scores(y, yv):
    counts = [y.count(k) for k in range(N_CLASSES)]
    p = [c / len(y) for c in counts]
    vloss = -sum(math.log(p[c]) for c in yv) / len(yv)
    major = counts.index(max(counts))
    return counts, vloss, sum(1 for c in yv if c == major) / len(yv)


@lru_cache(maxsize=None)
def linear_baseline(labels_key="true"):
    (X, y), (Xv, yv), _ = task()
    yy = y if labels_key == "true" else y[1:] + y[:1]
    g = _gen("linear_init")
    model = Sequential([Linear(N_IN, N_CLASSES, W=init_linear(N_CLASSES, N_IN, "fan_in", "normal", g))])
    cfg = dict(epochs=20, batch=32, peak=1e-2, tau=1.0, lam=0.01, patience=None)
    t = Trainer(model, SoftmaxCrossEntropy(), X, yy, Xv, yv, cfg, "classifier_batches", None, classify=True)
    t.run(20)
    return t


@lru_cache(maxsize=None)
def classifier_run():
    return train_classifier()


def classifier_task():
    (X, y), (Xv, yv), (Xt, yt) = task()
    g = _gen("task")
    u0 = [g.normal() for _ in range(3)]
    rows = [[name, len(lab), *[lab.count(k) for k in range(3)]] for name, lab in (("training", y), ("validation", yv), ("test", yt))]
    return (f"Generator {_seed('task')}; draws in order: u_0, u_1, u_2, v_0, v_1, v_2 (8 normal() each), then 1024 inputs (8 normal() each, row by row). "
            "Label = the first index of the largest of 2 (u_k . x) + (v_k . x)^2, k = 0, 1, 2. Rows 0 to 511 are the training set, 512 to 767 the validation set, "
            f"768 to 1023 the test set. u_0 starts ({u0[0]:+.6f}, {u0[1]:+.6f}, {u0[2]:+.6f}); training row 0 starts ({X[0][0]:+.6f}, {X[0][1]:+.6f}, {X[0][2]:+.6f}) with label {y[0]}; "
            f"training row 511 ends ({X[511][5]:+.6f}, {X[511][6]:+.6f}, {X[511][7]:+.6f}) with label {y[511]}; sum of the 4096 training inputs {sum(sum(r) for r in X):+.6f}.\n\n"
            + _table(["split", "rows", "class 0", "class 1", "class 2"], rows))


def classifier_training():
    (X, y), (Xv, yv), (Xt, yt) = task()
    t = classifier_run()
    model = make_classifier()
    P = len(get_theta(model))
    counts, pv, pacc = prior_scores(y, yv)
    lb = linear_baseline("true")
    of = overfit_one_batch(make_classifier(), SoftmaxCrossEntropy(), X[:8], y[:8], 0.01, 150)
    rows = [[0, f"{t.start['train']:.5f}", f"{t.start['val']:.5f}", f"{t.start['acc']:.4f}", "", "", ""]]
    shown = {1, 2, 5, 10, 15, t.best_epoch, t.epoch}
    for r in t.log:
        if r["epoch"] in shown:
            rows.append([r["epoch"], f"{r['train']:.5f}", f"{r['val']:.5f}", f"{r['acc']:.4f}", f"{r['norm']:.4f}", r["clipped"], f"{r['lr']:.3e}"])
    best_model = make_classifier()
    set_theta(best_model, t.best)
    tl, tout = evaluate(best_model, SoftmaxCrossEntropy(), Xt, yt)
    vl, vout = evaluate(best_model, SoftmaxCrossEntropy(), Xv, yv)
    nb = len(X) // CLS_RUN["batch"]
    total = CLS_RUN["epochs"] * nb
    head = (f"Model: Linear 8 -> 32 (He), GELU, residual block [layer norm 32, Linear 32 -> 32 (He), GELU, dropout 0.3], Linear 32 -> 3 "
            f"(1/n_in scale times {READOUT_FACTOR}); {P} parameters, biases and layer-norm shifts 0, gains 1. Start draws: Generator {_seed('classifier_init')}, "
            "weights of the three linear layers in order, row-major, one normal() each. "
            f"Loop: batch 32 ({nb} steps per epoch, at most {CLS_RUN['epochs']} epochs = {total} steps), shuffles of 0 .. 511 from Generator {_seed('classifier_batches')} "
            f"(one in-place shuffle per epoch), dropout draws from Generator {_seed('classifier_dropout')} (one uniform() per element of the 32 x 32 block, row-major, per step). "
            f"AdamW (0.9, 0.999, 1e-8), decay {CLS_RUN['lam']}, clip {CLS_RUN['tau']}, peak rate {CLS_RUN['peak']}, warmup {total // 10} steps, cosine to {CLS_RUN['peak'] / 100:.1e}, "
            f"early stopping with patience {CLS_RUN['patience']} on the validation loss. Losses and accuracy are computed in eval mode over the whole set after each epoch; "
            "'grad norm' is the mean norm before the clip, 'clipped' the number of clipped steps in the epoch, 'rate' the rate of the epoch's last step. "
            f"Linear softmax baseline: one linear layer 8 -> 3 (weights N(0, 1/8) row-major from Generator {_seed('linear_init')}, biases 0), trained on the same data by the same loop (batch 32, shuffles from the classifier's shuffle stream, "
            "no dropout, no early stopping) for 20 epochs of 16 steps with AdamW decay 0.01, clip 1 and peak rate 0.01 (warmup 32 steps, cosine to 1e-4); validation loss and accuracy in eval mode after epoch 20. "
            "Overfit one batch: the classifier above with dropout in eval mode (off), training rows 0 to 7, a constant rate 0.01 (no schedule), AdamW without clip and without decay, 150 steps, "
            "the loss before the first step and after the 150th and the accuracy on those 8 rows.\n\n")
    pre = _table(["before training", "value"], [
        ["ln 3, the loss of a model that ignores the input", f"{math.log(3):.5f}"],
        ["initial loss, training set (eval mode)", f"{t.start['train']:.5f}"],
        ["training class counts; majority-class validation accuracy", f"{counts}; {pacc:.4f}"],
        ["validation loss of the training prior", f"{pv:.5f}"],
        ["linear softmax baseline (20 epochs), validation loss and accuracy", f"{lb.log[-1]['val']:.5f}, {lb.log[-1]['acc']:.4f}"],
        ["overfit one batch (8 rows, 150 steps of AdamW 0.01, eval mode): loss and accuracy", f"{of[0]:.5f} -> {of[1]:.2e}, {of[2]:.3f}"],
    ], "lr")
    tail = (f"\n\nTraining stopped after epoch {t.epoch}; the checkpoint kept is epoch {t.best_epoch}, with validation loss {vl:.5f} and accuracy "
            f"{accuracy(vout, yv):.4f}. On the test set the checkpoint scores loss {tl:.5f} and accuracy {accuracy(tout, yt):.4f}.")
    return head + pre + "\n\n" + _table(["epoch", "training loss", "validation loss", "validation accuracy", "grad norm", "clipped", "rate"], rows) + tail


def autoencoder_training():
    tr, va, te = sensor_data()
    t = train_autoencoder()
    model = make_autoencoder()
    P = len(get_theta(model))
    L0, _, rows1 = _model_check_rows("autoencoder")
    fb = overfit_one_batch(make_autoencoder(), MeanSquaredError(), tr[:8], tr[:8], 0.01, 400)
    mean = [sum(r[j] for r in tr) / len(tr) for j in range(SENSORS)]
    base = lambda S: sum((v - mean[j]) ** 2 for r in S for j, v in enumerate(r)) / (len(S) * SENSORS)
    shown = [1, 2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300]
    rows = [[0, f"{t.start['train']:.5f}", f"{t.start['val']:.5f}", "", "", ""]]
    for r in t.log:
        if r["epoch"] in shown:
            rows.append([r["epoch"], f"{r['train']:.5f}", f"{r['val']:.5f}", f"{r['norm']:.4f}", r["clipped"], f"{r['lr']:.2e}"])
    best_model = make_autoencoder()
    set_theta(best_model, t.best)
    te_loss = evaluate(best_model, MeanSquaredError(), te, te)[0]
    head = (f"Data, start and shuffles are the streams of chapter 13: `first_models.ae_data` (draws per reading: s1, s2, s3, then n_0 .. n_15; 512 readings split 256 / 128 / 128), "
            f"`first_models.ae_init` (weights N(0, 1/n_in) of the four linear layers in order, row-major, biases 0) and `first_models.batches` (one shuffle of 0 .. 255 per epoch). "
            f"{P} parameters. AdamW (0.9, 0.999, 1e-8), decay 0.01, clip 1, batch 32 (8 steps per epoch), 300 epochs = 2400 steps, peak rate 0.01, warmup 240 steps, cosine to 1e-4, no early stopping. "
            "The target of every reading is itself.\n\n")
    pre = _table(["check before training", "value"], [
        ["mean predictor, training and validation mean squared error", f"{base(tr):.4f}, {base(va):.4f}"],
        ["initial loss, training and validation", f"{t.start['train']:.4f}, {t.start['val']:.4f}"],
        ["gradient check, all 499 entries on readings 0 to 7: entries failing", str(sum(r[3] for r in rows1))],
        ["overfit one batch (8 readings, 400 steps of AdamW 0.01): loss", f"{fb[0]:.4f} -> {fb[1]:.2e}"],
    ], "lr")
    tail = (f"\n\nBest validation loss {t.best_val:.5f} at epoch {t.best_epoch}; the checkpoint kept is that epoch's, with test loss {te_loss:.5f}. "
            f"Mean predictor on the test readings: {base(te):.5f}.")
    return head + pre + "\n\n" + _table(["epoch", "training loss", "validation loss", "grad norm", "clipped", "rate"], rows) + tail


# ------------------------------------------------------------------ milestone 7: the planted defects

DEBUG_EPOCHS = 40
STAGES = [("0: as shipped", True, True, True), ("1: labels aligned", False, True, True),
          ("2: and horizon in steps", False, False, True), ("3: and eval mode (healthy)", False, False, False)]


@lru_cache(maxsize=None)
def debug_run(shift, short_horizon, train_mode_eval):
    (X, y), _, _ = task()
    labels = y[1:] + y[:1] if shift else y
    cfg = dict(epochs=DEBUG_EPOCHS, patience=None)
    if short_horizon:
        cfg["total_override"] = DEBUG_EPOCHS
    if train_mode_eval:
        cfg["eval_mode"] = "train"
    return train_classifier(cfg, labels)


def debug_stages():
    rows = []
    for tag, a, b, c in STAGES:
        t = debug_run(a, b, c)
        for e in (1, 2, 5, 10, 20, 40):
            r = t.log[e - 1]
            rows.append([tag if e == 1 else "", e, f"{r['train']:.4f}", f"{r['val']:.4f}", f"{r['acc']:.4f}", f"{r['lr']:.3e}"])
    return ("The classifier of milestone 6 with every other setting unchanged, early stopping off, 40 epochs of 16 steps. Same streams as milestone 6; "
            "in the shipped configuration the validation pass runs in train mode and so consumes the dropout generator (once before training, 8192 draws, "
            "and once after each epoch, in that order relative to the training steps). Loss columns are eval-mode losses for the training set and the "
            "configuration's own validation pass for the validation set; 'rate' is the rate of the epoch's last step.\n\n"
            + _table(["stage", "epoch", "training loss", "validation loss", "validation accuracy", "rate"], rows, "lrrrrr"))


def debug_diagnostics():
    (X, y), (Xv, yv), _ = task()
    bad, good = debug_run(True, True, True), debug_run(False, False, False)
    shifted = y[1:] + y[:1]
    agree = sum(1 for a, b in zip(y, shifted) if a == b) / len(y)
    lb_bad, lb_good = linear_baseline("shifted"), linear_baseline("true")
    of_bad = overfit_one_batch(make_classifier(), SoftmaxCrossEntropy(), X[:8], shifted[:8], 0.01, 150)
    of_good = overfit_one_batch(make_classifier(), SoftmaxCrossEntropy(), X[:8], y[:8], 0.01, 150)

    def two_calls(t):
        m = make_classifier()
        set_theta(m, get_theta(t.model))
        rng = _gen("determinism_probe")
        a = evaluate(m, SoftmaxCrossEntropy(), Xv, yv, "train", rng)[0]
        b = evaluate(m, SoftmaxCrossEntropy(), Xv, yv, "train", rng)[0]
        e1 = evaluate(m, SoftmaxCrossEntropy(), Xv, yv, "eval")[0]
        e2 = evaluate(m, SoftmaxCrossEntropy(), Xv, yv, "eval")[0]
        return abs(a - b), abs(e1 - e2)

    d_tr_bad, d_ev_bad = two_calls(bad)
    d_tr, d_ev = two_calls(good)
    counts, pv, pacc = prior_scores(y, yv)
    rows = [
        ["initial loss against ln 3 = 1.09861", f"{bad.start['train']:.5f}", f"{good.start['train']:.5f}", "same: not the fault"],
        ["gradient check of the model on rows 0 to 4 (milestone 2): failing entries, directional verdicts",
         *[f"{sum(r[3] for r in _model_check_rows(k)[2] if r[3] != '-')} failing, " + ("all pass" if all(r[4] == "pass" for r in _model_check_rows(k)[2] if r[3] == "-") else "FAIL")
           for k in ("classifier_shifted", "classifier")], "same: the gradient is of the loss it is given"],
        ["linear baseline, validation accuracy (majority class " + f"{pacc:.4f})", f"{lb_bad.log[-1]['acc']:.4f}", f"{lb_good.log[-1]['acc']:.4f}", "shipped: barely above the majority class"],
        ["overfit one batch: final loss", f"{of_bad[1]:.2e}", f"{of_good[1]:.2e}", "both pass: memorising needs no signal"],
        ["epoch-1 rate and rate at epoch 5 (peak 0.003, floor 3e-05)", f"{bad.log[0]['lr']:.3e}, {bad.log[4]['lr']:.3e}", f"{good.log[0]['lr']:.3e}, {good.log[4]['lr']:.3e}", "shipped: the floor by epoch 5"],
        ["rate of the last step of epoch 40", f"{bad.log[-1]['lr']:.3e}", f"{good.log[-1]['lr']:.3e}", "both at the floor 3e-05 to three digits; the shipped one got there 600 steps early"],
        ["two evaluations of one model, train mode: loss difference", f"{d_tr_bad:.4f}", f"{d_tr:.4f}", "different answers for one input"],
        ["two evaluations of one model, eval mode: loss difference", f"{d_ev_bad:.1e}", f"{d_ev:.1e}", "eval mode: identical, as it must be"],
    ]
    return ("Checklist measurements of the shipped configuration against the healthy one (stage 3 of the previous table). Validation labels are never "
            "shifted. The linear baseline and the overfit-one-batch test have the settings of the milestone-6 table, trained on the shifted labels (training label i replaced by the label of row i + 1, the last by the first) in the shipped column. The determinism probe evaluates the final weights of the run of its own column (shipped: stage 0, healthy: stage 3) twice on the 256 validation rows, in train mode with a fresh generator "
            f"{_seed('determinism_probe')} per column (the draws continue between the two calls) and then twice in eval mode.\n\n"
            + _table(["measurement", "shipped", "healthy", "reading"], rows, "lrrl"))


def acceptance_fingerprint():
    rows = []
    vals = {}
    for label, layer, x, R, mode, dname in _fixture_cases():
        vals[label] = layer_probe(layer, x, R, mode, dname)[0]
    rows += [["1", "probe loss L, linear 5 -> 7", f"{vals['linear 5 -> 7']:+.8f}", "5e-9"],
             ["1", "probe loss L, layer norm", f"{vals['layer norm']:+.8f}", "5e-9"],
             ["1", "probe loss L, residual (5 -> 7 -> 5)", f"{vals['residual (5 -> 7 -> 5)']:+.8f}", "5e-9"]]
    z = rows_of(fix(12, 0.8, 0.5, 2.0), 4)
    rows.append(["1", "softmax cross-entropy on the fixture", f"{SoftmaxCrossEntropy().forward(z, [2, 0, 3]):+.8f}", "5e-9"])
    cases = _fixture_cases()
    label, layer, x, R, mode, _ = next(c for c in cases if c[0] == "layer norm")
    L, y, dx, loss, theta0, grad, own = layer_probe(layer, x, R, mode, None, lambda dy: layer.backward(dy, centre_term=False))
    rows.append(["2", "layer-norm input entries failing without the centring term", f"{len(check_entries(loss, theta0, grad, range(15))[0])} of 15", "exact"])
    th = [1.0, -2.0, 0.5]
    m, v = [0.0] * 3, [0.0] * 3
    for t in range(10):
        adam_step(th, [0.5 * math.sin(0.8 * (3 * t + i) + 0.3) for i in range(3)], m, v, t + 1, 0.1, 0.9, 0.999, 1e-8, 0.1, True)
    rows.append(["3", "AdamW theta[0] after 10 steps", f"{th[0]:+.8f}", "5e-9"])
    rows.append(["3", "schedule rate at t = 55 (hold 0)", f"{schedule_rate(55, 0.01, 1e-4, 10, 0, 100):.8f}", "5e-9"])
    rows.append(["4", "sd (divisor n = 12288) of He normal, 128 x 96 (stream init.he.normal)", f"{_init_sd():.6f}", "5e-7 bit-exact; 3% of the target otherwise"])
    rows.append(["5", "patience 3: best epoch, stop epoch", "6, 9", "exact"])
    t = classifier_run()
    rows += [["6", "classifier: initial training loss", f"{t.start['train']:.5f}", "1e-4"],
             ["6", "classifier: best epoch, stop epoch", f"{t.best_epoch}, {t.epoch}", "exact"],
             ["6", "classifier: validation loss at the checkpoint", f"{t.best_val:.5f}", "1e-4"]]
    a = train_autoencoder()
    rows += [["6", "autoencoder: initial training loss", f"{a.start['train']:.5f}", "1e-4"],
             ["6", "autoencoder: best epoch", str(a.best_epoch), "exact"],
             ["6", "autoencoder: best validation loss", f"{a.best_val:.5f}", "1e-5"]]
    g = debug_run(False, False, False)
    h = debug_run(False, True, False)
    rows += [["7", "stage 3 equals milestone 6 without early stopping: validation loss, epoch 10", f"{g.log[9]['val']:.4f}", "1e-4"],
             ["7", "stage 2 (eval-mode fault only): validation accuracy, epoch 40", f"{debug_run(False, False, True).log[-1]['acc']:.4f}", "1e-4"],
             ["7", "horizon fault only: rate of the last step, epoch 40", f"{h.log[-1]['lr']:.3e}", "5e-9"]]
    return _table(["milestone", "quantity", "value", "tolerance"], rows, "llrl")


def _init_sd():
    W = init_linear(128, 96, "he", "normal", _gen("init.he.normal"))
    mean = sum(W) / len(W)
    return math.sqrt(sum((v - mean) ** 2 for v in W) / len(W))
