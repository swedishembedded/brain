# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Optimizers: From SGD to AdamW".

Plain Python 3 on floats (IEEE double precision), written from the chapter's
own description of each update rule and independent of any engine code. Every
public function returns a markdown string that the chapter embeds between
`<!-- worked: optimizers.<name> -->` markers; the same module supplies the
curves and trajectories of the chapter's figures.

Each rule is a small class with a `step(w, g)` method over plain lists (a
single weight is a list of length one). The rules are the ones the chapter
writes in pseudocode: SGD, heavy-ball momentum (the "sum" form and the
Nesterov variant), RMSProp, Adam with bias correction, AdamW (decoupled decay)
or Adam with the penalty inside the gradient (coupled L2), and Lion.

Counting rule used for work: one multiplication, addition, subtraction,
division or square root is one FLOP; a comparison or a choice of sign is not
counted. Bytes are the bytes read from and written to memory by one update of
one parameter, every array stored in 32-bit floats unless a row says otherwise.

Swedish Embedded AB implements solutions for training neural networks with
memory-bounded optimisers. If your team needs expertise in optimiser design
and optimiser-state memory for large-model training, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng

# ------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _int(v):
    return f"{v:,}"


def _seed_line(name):
    s = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed 0x{s:016x} ({s})."


# ------------------------------------------------------------------ the update rules


class SGD:
    """w <- w - eta g."""

    def __init__(self, lr):
        self.lr = lr

    def step(self, w, g):
        for i in range(len(w)):
            w[i] -= self.lr * g[i]


class Momentum:
    """Heavy ball in the sum form: m <- beta m + g, w <- w - eta m (Nesterov: w <- w - eta (g + beta m))."""

    def __init__(self, lr, beta, nesterov=False):
        self.lr, self.beta, self.nesterov = lr, beta, nesterov
        self.m = None

    def step(self, w, g):
        if self.m is None:
            self.m = [0.0] * len(w)
        for i in range(len(w)):
            self.m[i] = self.beta * self.m[i] + g[i]
            d = g[i] + self.beta * self.m[i] if self.nesterov else self.m[i]
            w[i] -= self.lr * d


class RMSProp:
    """v <- beta2 v + (1 - beta2) g^2, w <- w - eta g / (sqrt(v) + eps); no bias correction."""

    def __init__(self, lr, beta2=0.9, eps=1e-8):
        self.lr, self.beta2, self.eps = lr, beta2, eps
        self.v = None

    def step(self, w, g):
        if self.v is None:
            self.v = [0.0] * len(w)
        for i in range(len(w)):
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g[i] * g[i]
            w[i] -= self.lr * g[i] / (math.sqrt(self.v[i]) + self.eps)


class Adam:
    """Adam with bias correction. `decoupled` decay (AdamW) shrinks w directly; otherwise lam is an L2 penalty in g."""

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, lam=0.0, decoupled=True):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.lam, self.decoupled = lam, decoupled
        self.m = self.v = None
        self.t = 0

    def step(self, w, g):
        if self.m is None:
            self.m, self.v = [0.0] * len(w), [0.0] * len(w)
        self.t += 1
        bc1, bc2 = 1 - self.beta1 ** self.t, 1 - self.beta2 ** self.t
        for i in range(len(w)):
            gi = g[i] if self.decoupled else g[i] + self.lam * w[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gi
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gi * gi
            mhat, vhat = self.m[i] / bc1, self.v[i] / bc2
            if self.decoupled:
                w[i] -= self.lr * self.lam * w[i]
            w[i] -= self.lr * mhat / (math.sqrt(vhat) + self.eps)


class Lion:
    """w <- w - eta (u + lam w) with u the three-way sign of beta1 m + (1 - beta1) g; then m <- beta2 m + (1 - beta2) g."""

    def __init__(self, lr, beta1=0.9, beta2=0.99, lam=0.0):
        self.lr, self.beta1, self.beta2, self.lam = lr, beta1, beta2, lam
        self.m = None

    def step(self, w, g):
        if self.m is None:
            self.m = [0.0] * len(w)
        for i in range(len(w)):
            c = self.beta1 * self.m[i] + (1 - self.beta1) * g[i]
            u = 1.0 if c > 0 else (-1.0 if c < 0 else 0.0)
            w[i] -= self.lr * (u + self.lam * w[i])
            self.m[i] = self.beta2 * self.m[i] + (1 - self.beta2) * g[i]


# ------------------------------------------------------------------ 1. the cost of a step

COST_N, COST_D = 1_000_000, 1_000


def epoch_cost():
    rows = []
    for b in (1, 32, 1024, COST_N):
        step_flops = 4 * COST_D * b
        updates = COST_N // b
        rows.append([_int(b), _int(step_flops), _int(updates), _int(step_flops * updates),
                     _int(updates * 12 * COST_D)])
    head = (f"Squared error of a linear model with d = {_int(COST_D)} weights on N = {_int(COST_N)} examples. One example's "
            "loss and gradient cost 4d FLOPs (2d for the prediction, 2d for the gradient entries); one update of the weights "
            "reads w and g and writes w, 12 d bytes in 32-bit floats. The number of updates in an epoch is floor(N / B): a last batch smaller than B (for B = 1,024, the last 576 of the 1,000,000 examples) is dropped, and the later columns use that count.\n\n")
    return head + _table(["batch size B", "FLOPs of one step (4 d B)", "updates in one epoch (floor(N / B))",
                          "FLOPs in one epoch", "weight-update bytes in one epoch (12 d floor(N / B))"], rows)


# ------------------------------------------------------------------ 2. minibatch gradients

MB_N, MB_R = 2000, 2000
MB_W0 = (1.0, -2.0, 0.5)


def _mb_data(rng):
    xs, ys = [], []
    for _ in range(MB_N):
        x = [rng.normal() for _ in range(3)]
        noise = rng.normal()
        xs.append(x)
        ys.append(sum(a * b for a, b in zip(x, MB_W0)) + 0.5 * noise)
    return xs, ys


def _example_gradient(x, y, w):
    r = sum(a * b for a, b in zip(x, w)) - y
    return [2 * r * a for a in x]


MB_SIZES = (1, 4, 16, 64, 256)
_MB = {}


def _minibatch_stats():
    """The measurements of the minibatch experiment: the generator name, the full gradient, sigma_g^2 and one record per batch size."""
    if _MB:
        return _MB
    name = "optimizers.minibatch_gradient"
    rng = bookrng.generator(name)
    xs, ys = _mb_data(rng)
    w = [0.0, 0.0, 0.0]
    per = [_example_gradient(x, y, w) for x, y in zip(xs, ys)]
    big = [sum(p[j] for p in per) / MB_N for j in range(3)]
    sigma2 = sum(sum((p[j] - big[j]) ** 2 for j in range(3)) for p in per) / MB_N
    recs = []
    for b in MB_SIZES:
        mean_g, sq = [0.0, 0.0, 0.0], 0.0
        for _ in range(MB_R):
            g = [0.0, 0.0, 0.0]
            for _ in range(b):
                p = per[rng.integer(MB_N)]
                for j in range(3):
                    g[j] += p[j]
            g = [a / b for a in g]
            for j in range(3):
                mean_g[j] += g[j] / MB_R
            sq += sum((g[j] - big[j]) ** 2 for j in range(3))
        bias = math.sqrt(sum((mean_g[j] - big[j]) ** 2 for j in range(3)))
        recs.append({"B": b, "bias": bias, "var": sq / MB_R})
    _MB.update(name=name, big=big, sigma2=sigma2, recs=recs)
    return _MB


def minibatch_noise_points():
    """[(B, measured rms noise, predicted rms noise sigma_g / sqrt(B))] for the figure."""
    st = _minibatch_stats()
    return [(r["B"], math.sqrt(r["var"]), math.sqrt(st["sigma2"] / r["B"])) for r in st["recs"]]


def minibatch_gradient():
    st = _minibatch_stats()
    name, big, sigma2 = st["name"], st["big"], st["sigma2"]
    big_norm = math.sqrt(sum(a * a for a in big))
    rows = []
    for r in st["recs"]:
        b, var = r["B"], r["var"]
        rows.append([str(b), _fx(r["bias"], 4), _fx(math.sqrt(sigma2 / (b * MB_R)), 4), _fx(var, 3), _fx(sigma2 / b, 3),
                     _fx(var * b / sigma2, 3), _fx(math.sqrt(var) / big_norm, 3), str(12 * b)])
    head = (f"{_seed_line(name)} Data (draw order, i = 0 .. {MB_N - 1}): three calls of normal() for x_i, then one for the label noise; "
            f"y_i = x_i . (1, -2, 0.5) + 0.5 * noise. Loss of an example: (x . w - y)^2, evaluated at w = (0, 0, 0). "
            f"Then, for each batch size in the order of the table, R = {MB_R} minibatches, each B calls of integer({MB_N}) "
            "(sampling with replacement) and the average of the B example gradients.\n\n"
            f"Full gradient: ({_fx(big[0], 4)}, {_fx(big[1], 4)}, {_fx(big[2], 4)}), norm {_fx(big_norm, 4)}. "
            f"One example's total variance sigma_g^2 = mean of |grad_i - full|^2 = {_fx(sigma2, 3)}. "
            "'Bias' is the distance from the full gradient to the average of the R minibatch gradients, to be compared with the next column, "
            "the size that sampling noise alone gives it; "
            "'variance' is the mean over the R minibatches of |g - full|^2; 'noise / signal' is its square root over the norm of the full gradient; "
            "'work' is 4 d B FLOPs with d = 3.\n\n")
    return head + _table(["B", "bias of the average", "its sampling scale sqrt(sigma_g^2 / (B R))", "measured variance",
                          "predicted sigma_g^2 / B", "measured x B / sigma_g^2", "noise / signal", "work (FLOPs)"], rows)


NOISE_STEPS, NOISE_BURN = 200_000, 1_000
NOISE_LAM, NOISE_S = 1.0, 2.0


def noise_floor():
    name = "optimizers.noise_floor"
    rng = bookrng.generator(name)
    rows = []
    for b in (1, 4):
        for eta in (0.4, 0.2, 0.1, 0.05):
            e, total, count = 1.0, 0.0, 0
            sd = NOISE_S / math.sqrt(b)
            for t in range(NOISE_STEPS):
                g = NOISE_LAM * e + sd * rng.normal()
                e -= eta * g
                if t >= NOISE_BURN:
                    total += e * e
                    count += 1
            measured = total / count
            predicted = eta * sd * sd / (NOISE_LAM * (2 - eta * NOISE_LAM))
            rows.append([str(b), f"{eta:g}", _fx(measured, 4), _fx(predicted, 4), _fx(measured / predicted, 3)])
    head = (f"{_seed_line(name)} One weight error e with curvature lam = {NOISE_LAM:g}, so the exact gradient is lam e, and a minibatch gradient "
            f"g = lam e + (s / sqrt(B)) * normal() with s = {NOISE_S:g}. Start e = 1; each step draws one normal() and sets e <- e - eta g; "
            f"{_int(NOISE_STEPS)} steps per row, the rows in the order of the table, one generator. 'Measured' is the mean of e^2 over steps "
            f"{_int(NOISE_BURN + 1)} .. {_int(NOISE_STEPS)}; 'predicted' is eta (s^2 / B) / (lam (2 - eta lam)).\n\n")
    return head + _table(["B", "eta", "measured mean of e^2", "predicted", "ratio"], rows)


# ------------------------------------------------------------------ 3. epochs and shuffling


def epoch_batches():
    name = "optimizers.epoch_batches"
    rng = bookrng.generator(name)
    n, b = 10, 4
    order = list(range(n))
    rows = []
    for epoch in range(2):
        rng.shuffle(order)
        for j, start in enumerate(range(0, n, b)):
            idx = order[start:start + b]
            rows.append([str(epoch), str(j), "(" + ", ".join(str(i) for i in idx) + ")", str(len(idx)),
                         f"{len(idx)}/{n}"])
    head = (f"{_seed_line(name)} N = {n} examples, batch size B = {b}. The index list "
            "`order` starts as (0, 1, ..., 9); at the start of every epoch `shuffle(order, rng)` permutes the list as the previous epoch left it; "
            "batch j takes the positions 4j .. min(4j + 4, 10) - 1. The last batch is short.\n\n")
    return head + _table(["epoch", "batch j", "example indices", "size B_j", "weight B_j / N in the epoch mean"], rows)


# ------------------------------------------------------------------ 4. momentum


def momentum_buffers():
    ks = (1, 2, 5, 10, 20, 50, 100)
    rows_const, rows_alt = [], []
    for beta in (0.5, 0.9, 0.99):
        m, const = 0.0, {}
        for k in range(1, 101):
            m = beta * m + 1.0
            const[k] = m
        rows_const.append([str(beta)] + [_fx(const[k], 3) for k in ks] + [_fx(1 / (1 - beta), 3)])
        m, alt = 0.0, {}
        for k in range(1, 101):
            m = beta * m + (1.0 if k % 2 == 1 else -1.0)
            alt[k] = m
        rows_alt.append([str(beta)] + [_fx(alt[k], 3) for k in ks] + ["+-" + _fx(1 / (1 + beta), 3)])
    heads = ["beta"] + [f"k = {k}" for k in ks] + ["limit"]
    return ("Buffer m_k = beta m_(k-1) + g_k, m_0 = 0. First a constant gradient g_k = 1 (limit 1 / (1 - beta)):\n\n"
            + _table(heads, rows_const)
            + "\n\nThen a gradient that alternates, g_k = +1 for odd k and -1 for even k. The buffer then alternates in sign too (positive at odd k) "
            "and its size tends to 1 / (1 + beta):\n\n"
            + _table(heads, rows_alt))


def momentum_modes():
    lam_hi, lam_lo = 6 + 4 * math.sqrt(2), 6 - 4 * math.sqrt(2)
    eta_gd = 2 / (lam_hi + lam_lo)
    eta_m, beta = 4 / (math.sqrt(lam_hi) + math.sqrt(lam_lo)) ** 2, 0.5
    rows = []
    for lam in (lam_lo, 0.5, 1.0, 3.0, 6.0, lam_hi):
        gd = abs(1 - eta_gd * lam)
        b = 1 + beta - eta_m * lam
        disc = b * b - 4 * beta
        if abs(disc) < 1e-9:
            rate, kind = abs(b) / 2, "double root"
        elif disc < 0:
            rate, kind = math.sqrt(beta), "complex pair"
        else:
            rate, kind = (abs(b) + math.sqrt(disc)) / 2, "real roots"
        rows.append([_fx(lam, 4), _fx(gd, 4), _fx(eta_m * lam, 4), kind, _fx(rate, 4)])
    kappa = lam_hi / lam_lo
    head = (f"The bowl of Volume I, Chapter 8 has curvatures {_fx(lam_hi, 4)} and {_fx(lam_lo, 4)} (ratio {_fx(kappa, 2)}). "
            f"Gradient descent at its best step {_fx(eta_gd, 4)} = 1/6 shrinks every mode by (kappa - 1)/(kappa + 1) = {_fx((kappa - 1) / (kappa + 1), 4)}. "
            f"Momentum with eta = 4 / (sqrt(lam_max) + sqrt(lam_min))^2 = {_fx(eta_m, 4)} and beta = ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^2 = {_fx(beta, 4)}: "
            "the per-step factor of a mode of curvature lam is the larger modulus of the two roots of z^2 - (1 + beta - eta lam) z + beta.\n\n")
    return head + _table(["curvature lam", "gradient descent |1 - eta lam|", "eta lam (momentum)", "roots", "momentum factor"], rows)


def momentum_vs_condition():
    rows = []
    for kappa in (9.0, 33.97, 100.0, 1000.0):
        gd = (kappa - 1) / (kappa + 1)
        sk = math.sqrt(kappa)
        mom = (sk - 1) / (sk + 1)
        steps_gd = math.log(1e-3) / math.log(gd)
        steps_mom = math.log(1e-3) / math.log(mom)
        rows.append([_fx(kappa, 2), _fx(gd, 4), _fx(mom, 4), _fx(steps_gd, 1), _fx(steps_mom, 1)])
    head = ("The per-step shrink factor of the error at the best settings: gradient descent (kappa - 1)/(kappa + 1), heavy-ball momentum "
            "(sqrt(kappa) - 1)/(sqrt(kappa) + 1). 'Steps' is ln(0.001) / ln(factor), the (fractional) number of steps for a factor-1000 reduction.\n\n")
    return head + _table(["condition number kappa", "gradient descent factor", "momentum factor", "steps, gradient descent", "steps, momentum"], rows)


# ------------------------------------------------------------------ 5. Adam: one step, bias correction


def adam_two_steps():
    lr, b1, b2, eps, wd = 0.1, 0.9, 0.999, 1e-8, 0.01
    w, m, v = 1.0, 0.0, 0.0
    rows = []
    for t, g in ((1, 0.5), (2, 0.3)):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        mhat, vhat = m / bc1, v / bc2
        decay = lr * wd * w
        step = lr * mhat / (math.sqrt(vhat) + eps)
        w_new = w - decay - step
        rows.append([str(t), _fx(g, 1), _fx(m, 6), _fx(v, 8), _fx(bc1, 4), _fx(bc2, 6), _fx(mhat, 6), _fx(vhat, 6),
                     _fx(decay, 6), _fx(step, 6), _fx(w_new, 6)])
        w = w_new
    head = ("One weight, start w = 1.0, m = v = 0; eta = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, decay lam = 0.01; "
            "gradients g_1 = 0.5 and g_2 = 0.3. Per step: m, v; bc1 = 1 - beta1^t, bc2 = 1 - beta2^t; mhat = m / bc1, vhat = v / bc2; "
            "decay = eta lam w (computed from the w before the step); step = eta mhat / (sqrt(vhat) + eps); w after = w - decay - step.\n\n")
    return head + _table(["t", "g", "m", "v", "bc1", "bc2", "mhat", "vhat", "decay", "step", "w after"], rows)


def bias_correction_factors():
    ts = (1, 2, 5, 10, 20, 50, 100, 1000, 5000)
    rows = []
    for beta in (0.9, 0.99, 0.999):
        rows.append([str(beta)] + [_fx(1 / (1 - beta ** t), 3) for t in ts])
    t1 = _table(["beta"] + [f"t = {t}" for t in ts], rows)
    rows = []
    for b1, b2 in ((0.9, 0.999), (0.9, 0.99), (0.9, 0.95), (0.9, 0.9)):
        raw = (1 - b1) / math.sqrt(1 - b2)
        rows.append([f"{b1}, {b2}", _fx(raw, 4), _fx(1.0, 4)])
    t2 = _table(["beta1, beta2", "first step / eta without correction", "first step / eta with correction"], rows)
    return ("The factor 1 / (1 - beta^t) that divides a moment at step t:\n\n" + t1
            + "\n\nSize of the first step (t = 1, gradient g, any size) of the Adam update measured in units of eta, "
            "|m / (sqrt(v) + eps)| with eps negligible against |g|:\n\n" + t2)


BIAS_SIGMA, BIAS_BETA2, BIAS_RUNS, BIAS_T = 2.0, 0.99, 4000, 300


def second_moment_bias():
    name = "optimizers.second_moment_bias"
    rng = bookrng.generator(name)
    checkpoints = (1, 2, 5, 10, 50, 100, 300)
    sums = {t: 0.0 for t in checkpoints}
    for _ in range(BIAS_RUNS):
        v = 0.0
        for t in range(1, BIAS_T + 1):
            g = BIAS_SIGMA * rng.normal()
            v = BIAS_BETA2 * v + (1 - BIAS_BETA2) * g * g
            if t in sums:
                sums[t] += v
    rows = []
    for t in checkpoints:
        mean_v = sums[t] / BIAS_RUNS
        bc = 1 - BIAS_BETA2 ** t
        rows.append([str(t), _fx(mean_v, 4), _fx(bc * BIAS_SIGMA ** 2, 4), _fx(mean_v / BIAS_SIGMA ** 2, 4),
                     _fx(bc, 4), _fx(mean_v / bc, 4)])
    head = (f"{_seed_line(name)} The gradient at every step is g_t = {BIAS_SIGMA} * normal(), so E[g^2] = {BIAS_SIGMA ** 2}; "
            f"v_0 = 0, v_t = beta2 v_(t-1) + (1 - beta2) g_t^2 with beta2 = {BIAS_BETA2}. {BIAS_RUNS} runs, one after another from one generator, "
            f"each drawing t = 1 .. {BIAS_T} in order; the table averages v_t over the runs.\n\n")
    return head + _table(["t", "mean of v_t", "(1 - beta2^t) E[g^2]", "mean of v_t / E[g^2]", "1 - beta2^t", "mean of v_t / (1 - beta2^t)"], rows)


# ------------------------------------------------------------------ 6. scale, epsilon


SCALE_STEPS = 20


def gradient_scale():
    name = "optimizers.gradient_scale"
    rng = bookrng.generator(name)
    base = [rng.normal() for _ in range(SCALE_STEPS)]
    makers = [("SGD", lambda: SGD(0.01)),
              ("Adam, eps = 1e-8", lambda: Adam(0.01, eps=1e-8)),
              ("Adam, eps = 0.1", lambda: Adam(0.01, eps=0.1))]
    scales = (1.0, 0.25, 4.0, 1e-3, 1e-6)
    rows = []
    for label, make in makers:
        finals = {}
        for c in scales:
            opt, w = make(), [0.0]
            for g in base:
                opt.step(w, [c * g])
            finals[c] = w[0]
        rows.append([label, _fx(finals[1.0], 6)] + [f"{finals[c] / finals[1.0]:.4g}" for c in scales[1:]])
    head = (f"{_seed_line(name)} Twenty gradients g_1 .. g_20 = normal() in order, the same list for every row; the gradient fed to the rule is a * g_t. "
            f"One weight from w = 0, eta = 0.01, beta1 = 0.9, beta2 = 0.999 for Adam. The table gives w_20 for a = 1 and the ratio w_20(a) / w_20(1) for the other scales; "
            f"base gradients: g_1 = {_fx(base[0], 4)}, g_2 = {_fx(base[1], 4)}, g_3 = {_fx(base[2], 4)}.\n\n")
    return head + _table(["rule", "w_20 at a = 1"] + [f"a = {c:g}" for c in scales[1:]], rows)


def epsilon_floor():
    lr, b1, b2 = 1.0, 0.9, 0.999
    t, g_late = 100, 1e-4
    m = (1 - b1) * g_late
    v = (1 - b2) * g_late ** 2
    mhat, vhat = m / (1 - b1 ** t), v / (1 - b2 ** t)
    rows = []
    for eps in (1e-8, 1e-6, 1e-5, 1e-4, 1e-3):
        honest = lr * mhat / (math.sqrt(vhat) + eps)
        zero_v = lr * mhat / eps
        rows.append([f"{eps:g}", _fx(honest, 4), _fx(zero_v, 1)])
    head = (f"A weight whose gradient was exactly 0 at steps 1 .. {t - 1} and is {g_late:g} at step t = {t}; beta1 = {b1}, beta2 = {b2}. "
            f"Then m = {_sci(m)}, v = {_sci(v)}, mhat = {_sci(mhat)}, vhat = {_sci(vhat)}, sqrt(vhat) = {_sci(math.sqrt(vhat))}. "
            "The table gives the step in units of eta: with the true vhat, mhat / (sqrt(vhat) + eps), and in the last column "
            "the same step if a lossy store had decoded vhat as exactly 0, mhat / eps.\n\n")
    return head + _table(["eps", "step / eta with true vhat", "step / eta with vhat decoded as 0"], rows)


# ------------------------------------------------------------------ 7. coupled and decoupled decay

DECAY_LR, DECAY_LAM = 0.01, 0.1


def decay_trajectories():
    # (label, rule, lam, penalty in the gradient fed to the rule: only SGD is fed it by hand)
    rules = [("SGD, penalty in the gradient", lambda lam: SGD(DECAY_LR), DECAY_LAM, True),
             ("AdamW, decoupled decay", lambda lam: Adam(DECAY_LR, lam=lam, decoupled=True), DECAY_LAM, False),
             ("AdamW, decoupled decay", lambda lam: Adam(DECAY_LR, lam=lam, decoupled=True), 1.0, False),
             ("Adam, penalty in the gradient", lambda lam: Adam(DECAY_LR, lam=lam, decoupled=False), 0.01, False),
             ("Adam, penalty in the gradient", lambda lam: Adam(DECAY_LR, lam=lam, decoupled=False), DECAY_LAM, False),
             ("Adam, penalty in the gradient", lambda lam: Adam(DECAY_LR, lam=lam, decoupled=False), 1.0, False)]
    ks = (0, 10, 50, 100, 200, 500, 1000, 2000)
    rows = []
    for label, make, lam, add_penalty in rules:
        opt = make(lam)
        w, vals = [1.0], {0: 1.0}
        for k in range(1, ks[-1] + 1):
            g = [lam * w[0]] if add_penalty else [0.0]
            opt.step(w, g)
            vals[k] = w[0]
        rows.append([label, f"{lam:g}"] + [_fx(vals[k], 4) for k in ks])
    head = (f"A weight that the data never touches: the loss gradient is 0 at every step, so only the decay acts. w = 1 at step 0, "
            f"eta = {DECAY_LR}, Adam with beta1 = 0.9, beta2 = 0.999, eps = 1e-8. SGD is fed the gradient lam w (the penalty's gradient); "
            "the Adam rows with the penalty in the gradient are the `decoupled = false` variant of the chapter's pseudocode (the rule adds lam w to the "
            "gradient 0 it is given); AdamW is given a gradient of 0 and shrinks the weight itself. For SGD and AdamW the value after k steps is (1 - eta lam)^k.\n\n")
    return head + _table(["rule", "lam"] + [f"k = {k}" for k in ks], rows, align="lr" + "r" * len(ks))


DECAY_STEPS, DECAY_TAIL = 4000, 2000


def decay_strength():
    name = "optimizers.decay_strength"
    rng = bookrng.generator(name)
    lam, lr = 0.01, 0.01
    rows = []
    for s in (0.01, 1.0, 100.0):
        noise = [s * rng.normal() for _ in range(DECAY_STEPS)]
        with_pen, without = Adam(lr, lam=lam, decoupled=False), Adam(lr)
        applied, rms_acc = 0.0, 0.0
        for t in range(DECAY_STEPS):
            w1, w0 = [1.0], [1.0]             # the weight is held at 1: the probe measures the pull, not the path
            with_pen.step(w1, [noise[t]])     # the rule adds lam * w to this gradient itself
            without.step(w0, [noise[t]])
            if t >= DECAY_STEPS - DECAY_TAIL:
                applied += w0[0] - w1[0]      # extra amount subtracted because of the penalty
                rms_acc += (noise[t] + lam) ** 2
        unit = lr * lam * 1.0
        rms = math.sqrt(rms_acc / DECAY_TAIL)
        rows.append([_fx(s, 2), _fx(applied / DECAY_TAIL / unit, 3), _fx(1 / rms, 3), "1.000"])
    head = (f"{_seed_line(name)} The data gradient of a weight is noise with standard deviation s and mean 0: g_t = s * normal(), "
            f"{DECAY_STEPS} draws per row, rows in the order of the table, one generator. The weight is held at w = 1 so that the probe measures the pull "
            f"of the penalty lam w = {lam} on the step, not the path: two Adam instances (eta = {lr}, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) see the same noise, "
            "one with lam w added to its gradient and one without; the 'applied decay' is the mean over the last "
            f"{DECAY_TAIL} steps of the difference of their steps, in units of eta lam w. 'Predicted' is 1 divided by the root mean square, over the same last steps, of the gradient the penalised Adam sees, g_t + lam w (data gradient plus penalty gradient). Decoupled decay applies exactly 1 in these units.\n\n")
    return head + _table(["noise s", "applied decay, penalty inside Adam (measured)", "1 / rms of g + lam w (predicted)", "decoupled decay"], rows)


# ------------------------------------------------------------------ 8. the rules side by side on one weight


def update_sizes():
    gs = [0.5, 0.3, -0.2, 0.4, 0.4, 0.4]
    lr = 0.1
    makers = [("SGD", lambda: SGD(lr)),
              ("momentum, beta = 0.9", lambda: Momentum(lr, 0.9)),
              ("RMSProp, beta2 = 0.9", lambda: RMSProp(lr, 0.9)),
              ("Adam", lambda: Adam(lr)),
              ("Lion", lambda: Lion(lr))]
    rows = []
    for label, make in makers:
        for c in (1.0, 100.0):
            opt, w = make(), [0.0]
            steps = []
            for g in gs:
                before = w[0]
                opt.step(w, [c * g])
                steps.append(before - w[0])
            rows.append([label, f"{c:g}"] + [_fx(s, 4) for s in steps])
    head = (f"One weight from w = 0 with eta = 0.1 for every rule; the gradients are a * (0.5, 0.3, -0.2, 0.4, 0.4, 0.4) with a = 1 and a = 100; "
            "Adam uses beta1 = 0.9, beta2 = 0.999, eps = 1e-8; Lion beta1 = 0.9, beta2 = 0.99. The entries are the amounts subtracted from w "
            "(positive: w decreases).\n\n")
    return head + _table(["rule", "scale a"] + [f"step {k}" for k in range(1, len(gs) + 1)], rows)


# ------------------------------------------------------------------ 9. the ill-conditioned bowl

BOWL_START, BOWL_MIN = (0.0, 3.5), (1.0, 2.0)
_L_HI, _L_LO = 6 + 4 * math.sqrt(2), 6 - 4 * math.sqrt(2)
BOWLS = {"rotated": [[2.0, 4.0], [4.0, 10.0]], "aligned": [[_L_HI, 0.0], [0.0, _L_LO]]}


def bowl_loss(H, w):
    e = [w[0] - BOWL_MIN[0], w[1] - BOWL_MIN[1]]
    return 0.5 * (e[0] * (H[0][0] * e[0] + H[0][1] * e[1]) + e[1] * (H[1][0] * e[0] + H[1][1] * e[1]))


def bowl_grad(H, w):
    e = [w[0] - BOWL_MIN[0], w[1] - BOWL_MIN[1]]
    return [H[0][0] * e[0] + H[0][1] * e[1], H[1][0] * e[0] + H[1][1] * e[1]]


def bowl_contour(kind, level, n=121):
    """Points w with loss(w) = level: the ellipse e = a u1 / sqrt(lam1) + b u2 / sqrt(lam2), a^2 + b^2 = 2 level."""
    H = BOWLS[kind]
    a, b, c = H[0][0], H[0][1], H[1][1]
    mid, rad = (a + c) / 2, math.hypot((a - c) / 2, b)
    lams = (mid + rad, mid - rad)
    if abs(b) < 1e-12:
        vecs = ((1.0, 0.0), (0.0, 1.0)) if a >= c else ((0.0, 1.0), (1.0, 0.0))
    else:
        v = (b, lams[0] - a)
        nrm = math.hypot(*v)
        v = (v[0] / nrm, v[1] / nrm)
        vecs = (v, (-v[1], v[0]))
    pts = []
    for k in range(n):
        th = 2 * math.pi * k / (n - 1)
        ca = math.sqrt(2 * level / lams[0]) * math.cos(th)
        cb = math.sqrt(2 * level / lams[1]) * math.sin(th)
        pts.append((BOWL_MIN[0] + ca * vecs[0][0] + cb * vecs[1][0], BOWL_MIN[1] + ca * vecs[0][1] + cb * vecs[1][1]))
    return pts


def _dist(x):
    return _fx(x, 4) if x < 1e6 else f"{x:.1e}"


def bowl_distance(w):
    return math.hypot(w[0] - BOWL_MIN[0], w[1] - BOWL_MIN[1])


BOWL_RULES = [
    ("SGD", "eta = 1/6", lambda: SGD(1 / 6)),
    ("momentum", "eta = 0.25, beta = 0.5", lambda: Momentum(0.25, 0.5)),
    ("momentum", "eta = 0.05, beta = 0.9", lambda: Momentum(0.05, 0.9)),
    ("Nesterov", "eta = 0.1, beta = 0.7", lambda: Momentum(0.1, 0.7, nesterov=True)),
    ("Nesterov", "eta = 0.25, beta = 0.5", lambda: Momentum(0.25, 0.5, nesterov=True)),
    ("RMSProp", "eta = 0.05, beta2 = 0.9", lambda: RMSProp(0.05, 0.9)),
    ("Adam", "eta = 0.1, 0.9 / 0.999", lambda: Adam(0.1)),
]


def bowl_path(kind, make, steps):
    H = BOWLS[kind]
    opt, w = make(), list(BOWL_START)
    path = [tuple(w)]
    for _ in range(steps):
        opt.step(w, bowl_grad(H, w))
        path.append(tuple(w))
    return path


BOWL_STEPS, BOWL_TOL = 200, 0.01


def bowl_comparison():
    rows = []
    for kind in ("rotated", "aligned"):
        for name, hp, make in BOWL_RULES:
            path = bowl_path(kind, make, BOWL_STEPS)
            d = [bowl_distance(p) for p in path]
            first = next((k for k in range(len(d)) if d[k] < BOWL_TOL), None)
            stay = next((k for k in range(len(d)) if all(x < BOWL_TOL for x in d[k:])), None)
            rows.append([kind, name, hp, _dist(d[20]), _dist(d[50]), _dist(d[BOWL_STEPS]),
                         str(first) if first is not None else "never",
                         str(stay) if stay is not None else "never"])
    head = (f"Both bowls have minimum (1, 2), start (0, 3.5) and curvatures {_fx(_L_HI, 4)} and {_fx(_L_LO, 4)} (condition number {_fx(_L_HI / _L_LO, 2)}). "
            "'Rotated' is the bowl of Volume I, Chapter 8, with Hessian [[2, 4], [4, 10]] whose steep direction is not a coordinate axis; "
            f"'aligned' has the Hessian diag({_fx(_L_HI, 4)}, {_fx(_L_LO, 4)}), the same bowl turned so that its axes are the coordinates, the first coordinate the steep one. In both, the loss is 0.5 e^T H e for the error e = w - (1, 2), the gradient is H e, and the start error is e = (-1, 1.5). "
            "Each rule starts with empty state and takes the exact gradient; RMSProp and Adam use eps = 1e-8 (added to the square root of the second-moment average, as in the pseudocode), the average is updated as beta2 v + (1.0 - beta2) * g * g, and momentum is the sum form of the pseudocode. "
            f"Columns: the distance to the minimum after 20, 50 and {BOWL_STEPS} steps; the first step at which the distance is below {BOWL_TOL}; "
            f"the first step after which it stays below {BOWL_TOL} up to step {BOWL_STEPS}.\n\n")
    return head + _table(["bowl", "rule", "hyperparameters", "dist. at 20", "dist. at 50", f"dist. at {BOWL_STEPS}", "first below", "stays below"],
                         rows, align="lllrrrrr")


# ------------------------------------------------------------------ 10. a small network

MLP_N, MLP_B, MLP_STEPS, MLP_H = 200, 20, 300, 8
LR_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MLP_RULES = [
    ("SGD", lambda lr: SGD(lr)),
    ("momentum 0.9", lambda lr: Momentum(lr, 0.9)),
    ("RMSProp 0.9", lambda lr: RMSProp(lr, 0.9)),
    ("Adam", lambda lr: Adam(lr, lam=0.0)),
    ("AdamW 0.05", lambda lr: Adam(lr, lam=0.05)),
    ("Lion", lambda lr: Lion(lr)),
]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


class MLP:
    """2 -> H tanh -> 1 sigmoid network with the mean binary cross-entropy; parameters in one flat list
    [W1 (H x 2, row-major), b1 (H), W2 (H), b2]."""

    def __init__(self, hidden=MLP_H):
        self.h = hidden
        self.size = hidden * 2 + hidden + hidden + 1

    def loss(self, theta, xs, ys):
        h, total = self.h, 0.0
        for (x1, x2), y in zip(xs, ys):
            out = theta[3 * h + h]
            for j in range(h):
                out += theta[3 * h + j] * math.tanh(theta[2 * j] * x1 + theta[2 * j + 1] * x2 + theta[2 * h + j])
            total += _softplus(out) - y * out
        return total / len(xs)

    def gradient(self, theta, xs, ys):
        h = self.h
        g = [0.0] * self.size
        for (x1, x2), y in zip(xs, ys):
            a = [math.tanh(theta[2 * j] * x1 + theta[2 * j + 1] * x2 + theta[2 * h + j]) for j in range(h)]
            out = theta[4 * h] + sum(theta[3 * h + j] * a[j] for j in range(h))
            dout = _sigmoid(out) - y
            g[4 * h] += dout
            for j in range(h):
                g[3 * h + j] += dout * a[j]
                dz = dout * theta[3 * h + j] * (1.0 - a[j] * a[j])
                g[2 * j] += dz * x1
                g[2 * j + 1] += dz * x2
                g[2 * h + j] += dz
        return [v / len(xs) for v in g]


def mlp_problem():
    """The data and the starting weights, drawn from one generator in a fixed order."""
    rng = bookrng.generator("optimizers.mlp")
    xs, ys = [], []
    for _ in range(MLP_N):
        x1, x2 = 2 * rng.uniform() - 1, 2 * rng.uniform() - 1
        xs.append((x1, x2))
        ys.append(1.0 if x1 * x2 > 0 else 0.0)
    net = MLP()
    theta = [0.0] * net.size
    h = MLP_H
    for k in range(2 * h):
        theta[k] = 0.5 * rng.normal()
    for j in range(h):
        theta[3 * h + j] = 0.5 * rng.normal()
    return net, xs, ys, theta


BATCH_STREAM = "optimizers.mlp_batches"


def mlp_batches(stream=BATCH_STREAM):
    """The batch index lists of the whole run: the same for every rule."""
    rng = bookrng.generator(stream)
    order = list(range(MLP_N))
    batches = []
    while len(batches) < MLP_STEPS:
        rng.shuffle(order)
        for s in range(0, MLP_N, MLP_B):
            batches.append(order[s:s + MLP_B])
    return batches[:MLP_STEPS]


MLP_CURVE_EVERY = 10
MLP_TAIL = (250, 260, 270, 280, 290, 300)


def mlp_train(make, lr, stream=BATCH_STREAM, nudge=0.0):
    """Train from the common start; returns the full-data loss every MLP_CURVE_EVERY steps as [(step, loss)]
    (step 0 included), or None for a run whose gradient stopped being finite."""
    net, xs, ys, theta = mlp_problem()
    theta[0] += nudge
    batches = mlp_batches(stream)
    opt = make(lr)
    curve = [(0, net.loss(theta, xs, ys))]
    for t, idx in enumerate(batches, start=1):
        g = net.gradient(theta, [xs[i] for i in idx], [ys[i] for i in idx])
        if not all(math.isfinite(v) for v in g):
            return None
        opt.step(theta, g)
        if t % MLP_CURVE_EVERY == 0:
            curve.append((t, net.loss(theta, xs, ys)))
    return curve


def tail_loss(curve):
    """The score of a run: the mean full-data loss at steps 250, 260, ..., 300 (one step's loss is too noisy to rank by)."""
    if curve is None:
        return None
    d = dict(curve)
    v = sum(d[t] for t in MLP_TAIL) / len(MLP_TAIL)
    return v if math.isfinite(v) else None


_SWEEP = {}


def mlp_sweep_results():
    if not _SWEEP:
        for name, make in MLP_RULES:
            _SWEEP[name] = [mlp_train(make, lr) for lr in LR_GRID]
    return _SWEEP


NUDGE = 1e-12   # a relative change of 1e-12 of one starting weight: the test of whether a run is reproducible in another implementation


def mlp_unstable(make, lr, v):
    """True when the score of a run changes in its third significant digit under a nudge of 1e-12 of one starting
    weight, i.e. the run is chaotic and no other implementation can reproduce its digits."""
    w = tail_loss(mlp_train(make, lr, nudge=NUDGE))
    return v is not None and (w is None or abs(w - v) > 1e-3 * max(abs(v), 1e-12))


def mlp_best_lr(name):
    scores = [tail_loss(c) for c in mlp_sweep_results()[name]]
    ok = [(v, i) for i, v in enumerate(scores) if v is not None]
    return LR_GRID[min(ok)[1]]


def mlp_sweep():
    net, xs, ys, theta = mlp_problem()
    start = net.loss(theta, xs, ys)
    res = mlp_sweep_results()
    rows = []
    for name, _ in MLP_RULES:
        best = mlp_best_lr(name)
        cells = []
        make = dict(MLP_RULES)[name]
        for lr, c in zip(LR_GRID, res[name]):
            v = tail_loss(c)
            cell = "diverged" if v is None else ("unstable" if mlp_unstable(make, lr, v) else _fx(v, 4))
            cells.append(f"**{cell}**" if lr == best else cell)
        rows.append([name] + cells)
    head = (f"Data and starting weights from one generator (seed 0x{bookrng.seed_of('optimizers.mlp'):016x}): for i = 0 .. {MLP_N - 1}, "
            "two calls of uniform() give x1 = 2 u - 1 and x2 = 2 u - 1, and the label is 1 when x1 x2 > 0 and 0 otherwise; then 16 calls of "
            "normal() times 0.5 for W1 (row-major, 8 x 2), then 8 calls times 0.5 for W2; biases start at 0. "
            f"Network: 2 -> {MLP_H} tanh -> 1 sigmoid, mean binary cross-entropy. The loss at the starting weights is {_fx(start, 4)}. "
            f"Every rule runs {MLP_STEPS} steps of batch size {MLP_B}, with batches from a second generator "
            f"(seed 0x{bookrng.seed_of('optimizers.mlp_batches'):016x}): shuffle the list 0 .. {MLP_N - 1}, cut it into consecutive batches of {MLP_B}, "
            "reshuffle when the epoch ends; the same batches for every rule and learning rate. "
            "Entries: the mean loss over all 200 examples, averaged over the six checkpoints at steps 250, 260, ..., 300 (the loss after k updates is the loss at step k); the best entry of each row is in bold. "
            "Rule details: momentum is the sum form m = 0.9 m + g, theta = theta - eta m; RMSProp uses beta2 = 0.9 and eps = 1e-8; Adam uses 0.9, 0.999, 1e-8; AdamW adds decoupled decay 0.05; Lion uses beta1 = 0.9, beta2 = 0.99, the three-way sign (0 at 0) and decay 0. "
            "Rounding noise grows into any difference in a run whose learning rate is too large, so a run that changes in its third significant digit when one starting weight is changed by 1e-12 is printed as 'unstable' (its digits cannot be reproduced in another implementation; its loss is of the order of the neighbouring entries or larger); "
            "'diverged' means the gradient stopped being finite.\n\n")
    return head + _table(["rule"] + [f"eta = {lr:g}" for lr in LR_GRID], rows)


def mlp_curves():
    """{rule name: (best learning rate, [(step, loss)])} for the figure and the table."""
    res = mlp_sweep_results()
    return {name: (mlp_best_lr(name), res[name][LR_GRID.index(mlp_best_lr(name))]) for name, _ in MLP_RULES}


def mlp_curve_table():
    curves = mlp_curves()
    steps = [0, 10, 20, 50, 100, 200, 300]
    rows = []
    for name, _ in MLP_RULES:
        lr, curve = curves[name]
        d = dict(curve)
        rows.append([name, f"{lr:g}"] + [_fx(d[s], 4) for s in steps])
    head = ("Each rule at its best learning rate from the sweep above; mean loss over all 200 examples after the stated number of steps.\n\n")
    return head + _table(["rule", "eta"] + [f"step {s}" for s in steps], rows)


# ------------------------------------------------------------------ 11. state and bytes

# (name, state slots, FLOPs per element)
RULE_COST = [
    ("SGD", 0, 2),
    ("SGD with momentum", 1, 4),
    ("RMSProp", 1, 9),
    ("Adam / AdamW", 2, 16),
    ("Lion", 1, 10),
]


def update_cost():
    rows = []
    for name, slots, flops in RULE_COST:
        read = 4 + 4 + 4 * slots          # w, g, state
        wrote = 4 + 4 * slots             # w, state
        rows.append([name, str(slots), str(4 + 4 + 4 * slots), str(flops), str(read + wrote), _fx(flops / (read + wrote), 2)])
    head = ("Per parameter, all arrays in 32-bit floats. Training memory = value (4) + gradient (4) + 4 per state slot. "
            "FLOPs of one update (rule written with 1 - beta and lr * lam precomputed once per step): SGD w - eta g, 2; momentum m = beta m + g, w - eta m, 4; "
            "RMSProp v: 4 (g g, times 1 - beta2, beta2 v, add), sqrt 1, + eps 1, divide 1, eta times that 1, subtract 1 = 9; "
            "Adam / AdamW: m 3, v 4, divisions by bc1 and bc2 2, sqrt 1, + eps 1, divide 1, eta mhat 1, subtract 1, decay 2 = 16; "
            "Lion: interpolation 3, step eta (u + lam w) 4, momentum update 3 = 10. Bytes moved by one update: read w, g and the state, write w and the state. "
            "The last column is FLOPs per byte moved.\n\n")
    return head + _table(["rule", "state slots", "bytes of training memory", "FLOPs per update", "bytes moved per update", "FLOPs per byte"], rows)


PARAMS_7B = 7_000_000_000
INT8_BLOCK = 64


def _q8(n_slots):
    """Bytes per element of n_slots blockwise-int8 state slots: one byte a code plus a 4-byte scale per block."""
    return n_slots * (1 + 4 / INT8_BLOCK)


# (name, state slots, bytes of one slot element; None = blockwise int8)
STATE_CASES = [("SGD", 0, 4.0), ("SGD with momentum", 1, 4.0), ("Lion", 1, 4.0), ("AdamW", 2, 4.0),
               ("AdamW, bf16 moments", 2, 2.0), ("Lion, int8 state", 1, None), ("AdamW, int8 state", 2, None)]


def _state_bytes(slots, width):
    return _q8(slots) if width is None else slots * width


def state_bar_rows():
    """[(name, [(part, bytes)], total)] for the chapter's memory figure."""
    out = []
    for name, slots, width in STATE_CASES:
        state = _state_bytes(slots, width)
        parts = [("value", 4.0), ("gradient", 4.0)] + ([("state", state)] if state else [])
        out.append((name, parts, 8.0 + state))
    return out


def state_memory():
    rows = []
    for name, slots, width in STATE_CASES:
        state = _state_bytes(slots, width)
        total = 4 + 4 + state
        rows.append([name, _fx(state, 4), _fx(total, 4), _fx(total * PARAMS_7B / 1e9, 1), _fx(state * PARAMS_7B / 1e9, 1)])
    head = (f"Per parameter: value 4 bytes and gradient 4 bytes in 32-bit floats, plus the state. Blockwise int8 state stores one byte a code "
            f"and one 4-byte scale per block of {INT8_BLOCK} elements, so {_fx(1 + 4 / INT8_BLOCK, 4)} bytes per slot. "
            f"The last two columns are for a model of {_int(PARAMS_7B)} parameters, in decimal gigabytes.\n\n")
    return head + _table(["rule", "state bytes per parameter", "total bytes per parameter", "total for 7e9 parameters (GB)", "of which state (GB)"], rows)


def state_traffic():
    rows = []
    cases = [("AdamW, 32-bit state", 2, 4.0), ("AdamW, bf16 state", 2, 2.0), ("AdamW, int8 state", 2, None)]
    for name, slots, width in cases:
        state = _q8(slots) if width is None else slots * width
        read, wrote = 4 + 4 + state, 4 + state
        rows.append([name, _fx(read, 4), _fx(wrote, 4), _fx(read + wrote, 4), _fx((read + wrote) / 28, 3)])
    head = ("Bytes moved by one AdamW update per parameter when the weight and gradient stay in 32 bits and only the two moments are stored narrower: "
            "read w, g and both moments, write w and both moments (the 16 FLOPs per update do not change).\n\n")
    return head + _table(["state format", "bytes read", "bytes written", "bytes moved", "fraction of the 32-bit state's 28 bytes"], rows)


REPEAT_STREAMS = [f"optimizers.mlp_batches.{k}" for k in range(1, 6)]


def mlp_repeats():
    rows = []
    for name, make in MLP_RULES:
        lr = mlp_best_lr(name)
        scores = [tail_loss(mlp_train(make, lr, stream)) for stream in REPEAT_STREAMS]
        mean = sum(scores) / len(scores)
        rows.append([name, f"{lr:g}", _fx(mean, 4), _fx(min(scores), 4), _fx(max(scores), 4)])
    seeds = ", ".join(f"0x{bookrng.seed_of(n):016x}" for n in REPEAT_STREAMS)
    head = ("The same start, data, rule details and learning rates as the sweep, five more batch orders from the generators named "
            f"optimizers.mlp_batches.1 .. optimizers.mlp_batches.5 (seeds {seeds}). Entries are the score of the sweep (mean full-data loss at steps 250, 260, ..., 300).\n\n")
    return head + _table(["rule", "eta", "mean of 5", "smallest", "largest"], rows)


def ema_of_constant(beta, steps):
    """[m_0 .. m_steps] of m_t = beta m_(t-1) + (1 - beta) * 1 from m_0 = 0: an average of a constant gradient 1."""
    m, out = 0.0, [0.0]
    for _ in range(steps):
        m = beta * m + (1 - beta)
        out.append(m)
    return out
