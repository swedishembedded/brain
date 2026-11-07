# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Maximum Likelihood, Priors and Regularization".

Plain Python 3, standard library only, written from the chapter's own
description: the likelihood of a coin's bias and its logarithm, the maximum
likelihood estimates of a Gaussian, the identities that turn a Gaussian
likelihood into the squared error and a Bernoulli likelihood into the binary
cross-entropy, a Beta prior on the coin, and the penalties that a Gaussian and
a Laplace prior put on weights (ridge regression, soft-thresholding, a lasso by
coordinate descent, a regularized logistic fit of the separable data of the
previous chapter). Each public function without a leading underscore that takes
no arguments returns a markdown string that the chapter embeds between
`<!-- worked: likelihood.<name> -->` markers. Underscore functions and the
constants are shared with the chapter's figures, so a picture and the table
beside it cannot disagree.

Swedish Embedded AB implements solutions for estimating models from noisy
measurements with the uncertainty stated honestly. If your team needs
expertise in maximum likelihood and regularized estimation, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng

BIAS_NAME = "likelihood.variance_bias"
RIDGE_NAME = "likelihood.ridge_data"
LASSO_NAME = "likelihood.lasso_data"

# The coin: ten flips, in order. 7 heads (H), 3 tails (T).
FLIPS = "HHTHHTHTHH"
HEADS = FLIPS.count("H")
TAILS = FLIPS.count("T")

# Six thermometer readings (degrees Celsius).
READINGS = (20.3, 19.8, 20.9, 20.1, 19.5, 20.4)

# Six measurements (x, y) and the candidate lines (intercept, slope) compared by NLL and MSE.
LINE_DATA = ((0.0, 1.1), (1.0, 2.9), (2.0, 5.2), (3.0, 6.8), (4.0, 9.1), (5.0, 10.9))
LINES = ((1.0, 2.0), (0.5, 2.0), (1.0, 1.8), (1.0, 2.2), (0.0, 2.5), (1.5, 1.8))

# Four labelled examples and the probabilities of label 1 that a model gave them.
BERNOULLI_LABELS = (1, 0, 1, 1)
BERNOULLI_PROBS = (0.9, 0.2, 0.6, 0.7)

# Ridge data: 8 equally spaced x in [-1, 1], y = 1 + 2x - 3x^2 + noise (sd RIDGE_NOISE), a degree-5 polynomial fit.
RIDGE_N, RIDGE_DEGREE, RIDGE_NOISE = 8, 5, 0.3
RIDGE_TRUTH = (1.0, 2.0, -3.0)
RIDGE_LAMBDAS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)

# Lasso data: N examples, D features, the true weights and the noise sd.
LASSO_N, LASSO_D, LASSO_NOISE = 20, 8, 0.5
LASSO_TRUE = (3.0, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
LASSO_LAMBDAS = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
CD_TOL, CD_MAX_SWEEPS = 1e-12, 100_000

# The separable data of the previous chapter, with a constant-one column first.
SEP_DATA = ((0.5, 0), (1.0, 0), (2.0, 1), (3.0, 1))
SEP_ETA = 0.5
SEP_RECORD = (1000, 10_000, 100_000)
SEP_LAMBDAS = (0.0, 0.001, 0.01, 0.1)


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _generator_line(name):
    seed = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed})."


def _sci(x, digits=3):
    if x == 0:
        return "0"
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m} x 10^{int(e)}"


def _rel(x):
    return f"{x:.4f}" if x >= 1e-3 else f"{x:.2e}"


def coin_loglik(theta, heads, tails):
    """log Lik(theta) of a specific sequence with the given counts: heads log(theta) + tails log(1 - theta)."""
    return heads * math.log(theta) + tails * math.log(1.0 - theta)


def grid(lo, hi, steps):
    """steps + 1 equally spaced values from lo to hi inclusive."""
    return [lo + (hi - lo) * i / steps for i in range(steps + 1)]


def trapezoid(xs, ys):
    return sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))


def coin_curve(heads, tails, steps=1000):
    """Likelihood of the coin on theta = i / steps, i = 1 .. steps - 1, relative to its largest value."""
    xs = grid(0.0, 1.0, steps)[1:-1]
    logs = [coin_loglik(t, heads, tails) for t in xs]
    top = max(logs)
    return xs, [math.exp(v - top) for v in logs]


def beta_posterior_curve(a, b, heads, tails, steps=1000):
    """Density on (0, 1) proportional to theta^(heads + a - 1) (1 - theta)^(tails + b - 1), normalised to unit area."""
    xs = grid(0.0, 1.0, steps)[1:-1]
    logs = [coin_loglik(t, heads + a - 1, tails + b - 1) for t in xs]
    top = max(logs)
    ys = [math.exp(v - top) for v in logs]
    area = trapezoid(xs, ys)
    return xs, [y / area for y in ys]


def _solve(a, b):
    """A w = b by Gaussian elimination with partial pivoting."""
    n = len(a)
    m = [list(r) + [bb] for r, bb in zip(a, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    w = [0.0] * n
    for r in range(n - 1, -1, -1):
        w[r] = (m[r][n] - sum(m[r][k] * w[k] for k in range(r + 1, n))) / m[r][r]
    return w


def ridge_solve(x, y, lam):
    """Minimiser of 1/2 sum (y - X w)^2 + lam/2 ||w||^2: (X^T X + lam I) w = X^T y."""
    d = len(x[0])
    a = [[sum(row[i] * row[j] for row in x) + (lam if i == j else 0.0) for j in range(d)] for i in range(d)]
    b = [sum(row[i] * t for row, t in zip(x, y)) for i in range(d)]
    return _solve(a, b)


def soft_threshold(a, lam):
    """The minimiser of 1/2 (w - a)^2 + lam |w|: move a toward zero by lam, stop at zero."""
    if a > lam:
        return a - lam
    if a < -lam:
        return a + lam
    return 0.0


def lasso_cd(x, y, lam, w0=None):
    """Minimise 1/2 sum (y - X w)^2 + lam ||w||_1 by cyclic coordinate descent; returns (w, sweeps)."""
    n, d = len(x), len(x[0])
    w = list(w0) if w0 else [0.0] * d
    col_sq = [sum(x[i][j] ** 2 for i in range(n)) for j in range(d)]
    r = [y[i] - sum(x[i][j] * w[j] for j in range(d)) for i in range(n)]
    for sweep in range(1, CD_MAX_SWEEPS + 1):
        biggest = 0.0
        for j in range(d):
            rho = sum(x[i][j] * (r[i] + x[i][j] * w[j]) for i in range(n))
            new = soft_threshold(rho, lam) / col_sq[j]
            delta = new - w[j]
            if delta != 0.0:
                for i in range(n):
                    r[i] -= x[i][j] * delta
                w[j] = new
            biggest = max(biggest, abs(delta))
        if biggest < CD_TOL:
            return w, sweep
    return w, CD_MAX_SWEEPS


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logistic_gd(lam, steps):
    """Gradient descent on mean BCE + lam/2 ||w||^2 from w = 0; yields (k, w) for each recorded step."""
    x = [[1.0, v] for v, _ in SEP_DATA]
    y = [float(t) for _, t in SEP_DATA]
    n = len(x)
    w = [0.0, 0.0]
    out = {}
    for k in range(1, max(steps) + 1):
        r = [_sigmoid(w[0] * row[0] + w[1] * row[1]) - t for row, t in zip(x, y)]
        g = [sum(row[j] * ri for row, ri in zip(x, r)) / n + lam * w[j] for j in range(2)]
        w = [w[j] - SEP_ETA * g[j] for j in range(2)]
        if k in steps:
            out[k] = (list(w), math.hypot(*g))
    return out


# ------------------------------------------------------------------ the coin

def coin_likelihood():
    """The likelihood of the coin's bias theta for the sequence HHTHHTHTHH, on a grid."""
    rows = []
    for i in range(1, 10):
        t = i / 10
        lik = t ** HEADS * (1 - t) ** TAILS
        rows.append((f"{t:.1f}", f"{lik:.7f}", f"{math.log(lik):.4f}", f"{math.comb(10, HEADS) * lik:.4f}"))
    t = HEADS / (HEADS + TAILS)
    best = t ** HEADS * (1 - t) ** TAILS
    head = (f"Flips, in order: {FLIPS} ({HEADS} heads, {TAILS} tails). Lik(theta) = theta^{HEADS} (1 - theta)^{TAILS}; "
            f"the last column is 120 Lik(theta), the probability of getting exactly {HEADS} heads in any order.\n\n")
    return head + _table(["theta", "Lik(theta)", "log Lik(theta)", "P(7 heads in 10 flips)"], rows) + (
        f"\n\nAt theta = 0.7: Lik = {best:.7f}, log Lik = {math.log(best):.4f}.")


def likelihood_vs_probability():
    """The same formula with theta fixed (a probability distribution over k) and with k fixed (a likelihood)."""
    theta = 0.7
    pk = [math.comb(10, k) * theta ** k * (1 - theta) ** (10 - k) for k in range(11)]
    rows = [(k, f"{p:.6f}") for k, p in enumerate(pk)]
    t1 = _table(["k heads", "P(k heads | theta = 0.7)"], rows)
    xs = grid(0.0, 1.0, 1000)
    ys = [math.comb(10, HEADS) * t ** HEADS * (1 - t) ** (10 - HEADS) for t in xs]
    area = trapezoid(xs, ys)
    tail = (f"\n\nSum over k = {sum(pk):.6f}.\n\n"
            f"Fixing k = 7 and varying theta instead: the area under 120 theta^7 (1 - theta)^3 on [0, 1] "
            f"(trapezoid rule, theta = i/1000, i = 0 .. 1000) is {area:.6f}; 1/11 = {1 / 11:.6f}.")
    return t1 + tail


def log_trick():
    """Products of many probabilities underflow; sums of their logarithms do not."""
    theta = 0.7
    rows = []
    for n in (10, 100, 1000, 10_000):
        h = round(0.7 * n)
        t = n - h
        prod = 1.0
        for i in range(h + t):
            prod *= theta if i < h else 1 - theta
        ll = coin_loglik(theta, h, t)
        rows.append((n, h, _sci(prod) if prod else "0.0 (underflow)", f"{ll:.4f}", f"{-ll / n:.6f}"))
    head = ("theta = 0.7; the product is taken one flip at a time in 64-bit floating point, the heads first "
            "and then the tails. The last column is the negative log-likelihood divided by n.\n\n")
    out = head + _table(["n flips", "heads", "product of probabilities", "log Lik", "-log Lik / n"], rows)
    ll = coin_loglik(theta, 7000, 3000)
    return out + (f"\n\nFor n = 10000 the true product is 10^({ll / math.log(10):.1f}); the smallest positive "
                  "64-bit floating-point number is about 4.9 x 10^-324.")


def coin_data_sizes():
    """How sharply the likelihood peaks as the same 70 percent heads rate comes from more flips."""
    rows = []
    for n in (10, 100, 1000):
        h = round(0.7 * n)
        xs, rel = coin_curve(h, n - h)
        top = max(range(len(xs)), key=lambda i: rel[i])
        inside = [x for x, r in zip(xs, rel) if r >= 0.5]
        at = {t: math.exp(coin_loglik(t, h, n - h) - coin_loglik(0.7, h, n - h)) for t in (0.5, 0.6, 0.8)}
        se = math.sqrt(0.7 * 0.3 / n)
        rows.append((n, h, f"{xs[top]:.3f}", _rel(at[0.5]), _rel(at[0.6]), _rel(at[0.8]),
                     f"{inside[0]:.3f} to {inside[-1]:.3f}", f"{se:.4f}"))
    head = "Relative likelihood Lik(theta) / Lik(0.7), theta on a grid of step 0.001.\n\n"
    return head + _table(["n", "heads", "argmax", "at 0.5", "at 0.6", "at 0.8", "theta with relative >= 1/2", "sqrt(0.7*0.3/n)"], rows)


# ------------------------------------------------------------------ the Gaussian

def _nll_gauss(xs, mu, var):
    return len(xs) / 2 * math.log(2 * math.pi * var) + sum((x - mu) ** 2 for x in xs) / (2 * var)


def gaussian_mle():
    """The sample mean and the 1/n variance minimise the Gaussian negative log-likelihood."""
    xs = READINGS
    n = len(xs)
    mean = sum(xs) / n
    ss = sum((x - mean) ** 2 for x in xs)
    v_mle, v_unb = ss / n, ss / (n - 1)
    rows = [("(mean, 1/n variance)", f"{mean:.4f}", f"{v_mle:.4f}", f"{_nll_gauss(xs, mean, v_mle):.4f}"),
            ("(mean, 1/(n-1) variance)", f"{mean:.4f}", f"{v_unb:.4f}", f"{_nll_gauss(xs, mean, v_unb):.4f}"),
            ("mean + 0.2", f"{mean + 0.2:.4f}", f"{v_mle:.4f}", f"{_nll_gauss(xs, mean + 0.2, v_mle):.4f}"),
            ("mean - 0.2", f"{mean - 0.2:.4f}", f"{v_mle:.4f}", f"{_nll_gauss(xs, mean - 0.2, v_mle):.4f}"),
            ("variance x 0.7", f"{mean:.4f}", f"{0.7 * v_mle:.4f}", f"{_nll_gauss(xs, mean, 0.7 * v_mle):.4f}"),
            ("variance x 1.5", f"{mean:.4f}", f"{1.5 * v_mle:.4f}", f"{_nll_gauss(xs, mean, 1.5 * v_mle):.4f}")]
    best = None
    for i in range(0, 201):          # mu = 19 + 0.01 i
        for j in range(5, 151):      # var = 0.01 j
            v = _nll_gauss(xs, 19 + 0.01 * i, 0.01 * j)
            if best is None or v < best[0]:
                best = (v, 19 + 0.01 * i, 0.01 * j)
    head = (f"Readings: {', '.join(str(x) for x in xs)} (n = {n}). Sum of squared deviations from the mean: {ss:.4f}.\n\n")
    tail = (f"\n\nBrute-force search of the NLL over mu = 19.00, 19.01, ..., 21.00 and variance = 0.05, 0.06, ..., 1.50: "
            f"smallest at mu = {best[1]:.2f}, variance = {best[2]:.2f} (NLL {best[0]:.4f}).")
    return head + _table(["parameters", "mu", "variance", "NLL"], rows, "lrrr") + tail


def variance_bias():
    """Average of the 1/n and 1/(n-1) variance estimates over many samples of size 5 from N(0, 1)."""
    g = bookrng.generator(BIAS_NAME)
    n, reps = 5, 20_000
    s_mle = s_unb = 0.0
    for _ in range(reps):
        xs = [g.normal() for _ in range(n)]
        m = sum(xs) / n
        ss = sum((x - m) ** 2 for x in xs)
        s_mle += ss / n
        s_unb += ss / (n - 1)
    rows = [("1/n (maximum likelihood)", f"{s_mle / reps:.4f}", f"{(n - 1) / n:.4f}"),
            ("1/(n-1) (unbiased)", f"{s_unb / reps:.4f}", "1.0000")]
    head = (f"{_generator_line(BIAS_NAME)}\nDraw order: for each of {reps} repetitions, {n} calls of normal() "
            f"(a sample from N(0, 1)); the true variance is 1.\n\n")
    return head + _table(["divisor", "average estimate", "expected value"], rows, "lrr")


def _line_stats():
    n = len(LINE_DATA)
    out = []
    for w0, w1 in LINES:
        res = [y - (w0 + w1 * x) for x, y in LINE_DATA]
        mse = sum(r * r for r in res) / n
        out.append((w0, w1, mse, n / 2 * math.log(2 * math.pi) + sum(r * r for r in res) / 2))
    return out


def gaussian_nll_is_mse():
    """With sigma = 1 the NLL of a line is N/2 log(2 pi) plus N/2 times its MSE; the order of lines is the same."""
    n = len(LINE_DATA)
    stats = _line_stats()
    rows = [(f"({w0}, {w1})", f"{mse:.4f}", f"{nll:.4f}", f"{n / 2 * math.log(2 * math.pi) + n / 2 * mse:.4f}")
            for w0, w1, mse, nll in stats]
    order_mse = sorted(range(len(stats)), key=lambda i: stats[i][2])
    order_nll = sorted(range(len(stats)), key=lambda i: stats[i][3])
    head = ("Data (x, y): " + ", ".join(f"({x:g}, {y:g})" for x, y in LINE_DATA) +
            f". Line y = w0 + w1 x, sigma = 1, N = {n}, N/2 log(2 pi) = {n / 2 * math.log(2 * math.pi):.4f}.\n\n")
    t1 = _table(["(w0, w1)", "MSE", "NLL", "N/2 log(2 pi) + N/2 MSE"], rows, "lrrr")
    # least-squares line and the maximum likelihood sigma
    xm = sum(x for x, _ in LINE_DATA) / n
    ym = sum(y for _, y in LINE_DATA) / n
    w1 = sum((x - xm) * (y - ym) for x, y in LINE_DATA) / sum((x - xm) ** 2 for x, _ in LINE_DATA)
    w0 = ym - w1 * xm
    mse = sum((y - w0 - w1 * x) ** 2 for x, y in LINE_DATA) / n
    rows2 = []
    for s in (0.1, 0.2, math.sqrt(mse), 0.4, 0.8, 1.0):
        nll = n * math.log(s) + n / 2 * math.log(2 * math.pi) + n * mse / (2 * s * s)
        rows2.append((f"{s:.4f}", f"{nll:.4f}"))
    t2 = _table(["sigma", "NLL of the least-squares line"], rows2, "lr")
    return (head + t1 + "\n\nOrder from best to worst by MSE: " + ", ".join(str(i + 1) for i in order_mse) +
            "; by NLL: " + ", ".join(str(i + 1) for i in order_nll) + ".\n\n"
            f"Least-squares line: w0 = {w0:.4f}, w1 = {w1:.4f}, MSE = {mse:.4f}, sqrt(MSE) = {math.sqrt(mse):.4f}.\n\n" + t2)


def bernoulli_nll_is_bce():
    """The probability a model gives to labelled data, and its negative logarithm, against the binary cross-entropy."""
    rows, prod = [], 1.0
    for i, (y, p) in enumerate(zip(BERNOULLI_LABELS, BERNOULLI_PROBS), 1):
        pr = p if y == 1 else 1 - p
        prod *= pr
        bce = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        rows.append((i, y, f"{p:.1f}", f"{pr:.1f}", f"{-math.log(pr):.4f}", f"{bce:.4f}"))
    n = len(rows)
    total = -math.log(prod)
    tail = (f"\n\nProbability of the four labels together: {prod:.4f}. NLL = -log {prod:.4f} = {total:.4f}. "
            f"NLL / N = {total / n:.4f}, the mean binary cross-entropy.")
    return _table(["example", "y", "p-hat", "probability of the label", "-log of it", "BCE"], rows) + tail


# ------------------------------------------------------------------ the prior

def coin_map():
    """MAP estimates of the coin's bias for Beta priors, with the posterior counts and a grid check."""
    priors = ((1, 1), (2, 2), (5, 5), (20, 20), (8, 2))
    out = []
    for h, t in ((7, 3), (70, 30)):
        rows = []
        for a, b in priors:
            map_ = (h + a - 1) / (h + t + a + b - 2)
            mean = (h + a) / (h + t + a + b)
            xs = grid(0.0, 1.0, 1000)[1:-1]
            best = max(xs, key=lambda th: coin_loglik(th, h + a - 1, t + b - 1))
            rows.append((f"Beta({a}, {b})", f"{a - 1} heads, {b - 1} tails", f"{h + a - 1}/{h + t + a + b - 2}",
                         f"{map_:.4f}", f"{best:.3f}", f"{mean:.4f}"))
        out.append(f"{h} heads, {t} tails (maximum likelihood {h / (h + t):.4f}):\n\n" +
                   _table(["prior", "pseudo-counts", "posterior counts", "MAP", "grid argmax", "posterior mean"], rows, "llrrrr"))
    return "\n\n".join(out)


# ------------------------------------------------------------------ priors on weights

def _ridge_data():
    g = bookrng.generator(RIDGE_NAME)
    xs = [-1 + 2 * i / (RIDGE_N - 1) for i in range(RIDGE_N)]
    f = lambda v: RIDGE_TRUTH[0] + RIDGE_TRUTH[1] * v + RIDGE_TRUTH[2] * v * v
    ys = [f(v) + RIDGE_NOISE * g.normal() for v in xs]
    return xs, ys, f


def _poly(v):
    return [v ** j for j in range(RIDGE_DEGREE + 1)]


def ridge_data_table():
    """The eight training points of the ridge example."""
    xs, ys, f = _ridge_data()
    rows = [(i, f"{x:.4f}", f"{f(x):.4f}", f"{y:.4f}") for i, (x, y) in enumerate(zip(xs, ys))]
    head = (f"{_generator_line(RIDGE_NAME)}\nDraw order: for i = 0 .. {RIDGE_N - 1} in turn, one call of normal(); "
            f"y_i = 1 + 2 x_i - 3 x_i^2 + {RIDGE_NOISE} * normal(); x_i = -1 + 2 i / {RIDGE_N - 1}.\n\n")
    return head + _table(["i", "x", "true value", "y"], rows)


def ridge_path():
    """Degree-5 polynomial fits to eight points under an increasing penalty lambda."""
    xs, ys, f = _ridge_data()
    x = [_poly(v) for v in xs]
    test = grid(-1.0, 1.0, 100)
    rows = []
    for lam in RIDGE_LAMBDAS:
        w = ridge_solve(x, ys, lam)
        train = math.sqrt(sum((yi - sum(wj * xj for wj, xj in zip(w, row))) ** 2 for row, yi in zip(x, ys)) / len(xs))
        err = math.sqrt(sum((f(v) - sum(wj * pj for wj, pj in zip(w, _poly(v)))) ** 2 for v in test) / len(test))
        rows.append((f"{lam:g}", *[f"{wj:.3f}" for wj in w], f"{math.sqrt(sum(wj * wj for wj in w)):.3f}",
                     f"{train:.4f}", f"{err:.4f}"))
    head = ("Features 1, x, ..., x^5 (D = 6), the eight points above. Minimise 1/2 sum (y - w.x)^2 + lambda/2 ||w||^2 "
            "by (X^T X + lambda I) w = X^T y. 'Error to the true curve' is the RMS difference between the fitted "
            "polynomial and 1 + 2x - 3x^2 at the 101 points x = -1, -0.98, ..., 1.\n\n")
    return head + _table(["lambda", "w0", "w1", "w2", "w3", "w4", "w5", "norm of w", "training RMSE", "error to the true curve"], rows)


def soft_threshold_1d():
    """The one-weight problem under an L2 and an L1 penalty, with a brute-force check of both."""
    lam = 1.0
    rows = []
    for a in (-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        w2 = a / (1 + lam)
        w1 = soft_threshold(a, lam)
        pts = [i / 1000 for i in range(-4000, 4001)]
        b2 = min(pts, key=lambda w: 0.5 * (w - a) ** 2 + lam / 2 * w * w)
        b1 = min(pts, key=lambda w: 0.5 * (w - a) ** 2 + lam * abs(w))
        rows.append((f"{a:g}", f"{w2:.4f}", f"{b2:.3f}", f"{w1:.4f}", f"{b1:.3f}"))
    head = ("lambda = 1. L2 objective 1/2 (w - a)^2 + lambda/2 w^2, L1 objective 1/2 (w - a)^2 + lambda |w|; "
            "the check column minimises the objective over w = -4.000, -3.999, ..., 4.000.\n\n")
    return head + _table(["a", "L2 minimiser a/(1+lambda)", "L2 by search", "L1 minimiser", "L1 by search"], rows)


def _lasso_data():
    g = bookrng.generator(LASSO_NAME)
    x = [[g.normal() for _ in range(LASSO_D)] for _ in range(LASSO_N)]
    y = [sum(wj * xj for wj, xj in zip(LASSO_TRUE, row)) + LASSO_NOISE * g.normal() for row in x]
    return x, y


def lasso_path():
    """Ridge and lasso weights of the same data as the penalty grows: ridge shrinks, lasso removes."""
    x, y = _lasso_data()
    ridge_rows, lasso_rows = [], []
    for lam in LASSO_LAMBDAS:
        wr = ridge_solve(x, y, lam)
        wl, sweeps = lasso_cd(x, y, lam)
        ridge_rows.append((f"{lam:g}", *[f"{v:.3f}" for v in wr], sum(1 for v in wr if v == 0.0)))
        lasso_rows.append((f"{lam:g}", *[("0" if v == 0.0 else f"{v:.3f}") for v in wl], sum(1 for v in wl if v == 0.0), sweeps))
    cols = [f"w{j + 1}" for j in range(LASSO_D)]
    head = (f"{_generator_line(LASSO_NAME)}\nDraw order: first the feature matrix X of shape [{LASSO_N}, {LASSO_D}] row by row "
            f"(one normal() per entry), then the {LASSO_N} noise values, one normal() each; "
            f"y_n = x_n . w_true + {LASSO_NOISE} * noise_n with w_true = {LASSO_TRUE}.\n\n")
    t1 = "Ridge: minimise 1/2 sum (y - X w)^2 + lambda/2 ||w||^2.\n\n" + _table(["lambda", *cols, "exact zeros"], ridge_rows)
    t2 = ("Lasso: minimise 1/2 sum (y - X w)^2 + lambda ||w||_1 by cyclic coordinate descent from w = 0, "
          f"stopping when no weight changes by {CD_TOL:g}; 0 is an exact zero.\n\n" +
          _table(["lambda", *cols, "exact zeros", "sweeps"], lasso_rows))
    return head + t1 + "\n\n" + t2


def separable_regularized():
    """The separable data of the previous chapter, trained with an L2 penalty on both weights."""
    rows = []
    for lam in SEP_LAMBDAS:
        runs = _logistic_gd(lam, SEP_RECORD)
        w, g = runs[max(SEP_RECORD)]
        norms = [math.hypot(*runs[k][0]) for k in SEP_RECORD]
        p1 = _sigmoid(w[0] + w[1] * 1.0)
        p2 = _sigmoid(w[0] + w[1] * 2.0)
        rows.append((f"{lam:g}", *[f"{v:.4f}" for v in norms], f"{w[0]:.4f}", f"{w[1]:.4f}", f"{-w[0] / w[1]:.4f}",
                     f"{p1:.4f}", f"{p2:.4f}", _sci(g, 2)))
    head = ("Data (x, y): (0.5, 0), (1.0, 0), (2.0, 1), (3.0, 1), with a constant 1 column; start w = (0, 0), eta = 0.5. "
            "Objective: mean binary cross-entropy + lambda/2 (w0^2 + w1^2).\n\n")
    return head + _table(["lambda", "norm at k=1000", "norm at k=10000", "norm at k=100000", "w0", "w1",
                          "boundary -w0/w1", "p at x=1", "p at x=2", "gradient norm"], rows)


# ------------------------------------------------------------------ pictures of the penalties

CONTOUR_CENTER = (1.2, 1.9)           # the unpenalised optimum of the 2D loss
CONTOUR_A = ((1.0, 0.6), (0.6, 1.0))  # the loss is 1/2 (w - c)^T A (w - c)


def contour_problem():
    """Solve the two-weight penalised problem with an L1 and an L2 penalty; returns the numbers the figure draws."""
    c, a = CONTOUR_CENTER, CONTOUR_A
    # loss = 1/2 w^T A w - b^T w + const, b = A c; coordinate descent for the lasso, direct solve for ridge.
    b = [a[0][0] * c[0] + a[0][1] * c[1], a[1][0] * c[0] + a[1][1] * c[1]]

    def loss(w):
        d = [w[0] - c[0], w[1] - c[1]]
        return 0.5 * (d[0] * (a[0][0] * d[0] + a[0][1] * d[1]) + d[1] * (a[1][0] * d[0] + a[1][1] * d[1]))

    lam1 = 2.0
    w = [0.0, 0.0]
    for _ in range(100_000):
        old = list(w)
        for j in range(2):
            k = 1 - j
            rho = b[j] - a[j][k] * w[k]
            w[j] = soft_threshold(rho, lam1) / a[j][j]
        if max(abs(w[0] - old[0]), abs(w[1] - old[1])) < CD_TOL:
            break
    w_l1 = list(w)
    r1 = abs(w_l1[0]) + abs(w_l1[1])
    # ridge whose solution has the same L2 length as the lasso solution: find lambda2 by bisection
    target = math.hypot(*w_l1)

    def ridge2(l2):
        m = [[a[0][0] + l2, a[0][1]], [a[1][0], a[1][1] + l2]]
        return _solve(m, b)

    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.hypot(*ridge2(mid)) > target:
            lo = mid
        else:
            hi = mid
    lam2 = (lo + hi) / 2
    w_l2 = ridge2(lam2)
    return {"c": c, "a": a, "lam1": lam1, "lam2": lam2, "w_l1": w_l1, "w_l2": w_l2, "r1": r1, "loss": loss}


def penalty_corners():
    """Where the penalised optimum of a two-weight problem lies, for an L1 and an L2 penalty."""
    p = contour_problem()
    rows = [("none", "-", f"{p['c'][0]:.4f}", f"{p['c'][1]:.4f}", f"{p['loss'](p['c']):.4f}"),
            ("L1", f"{p['lam1']:g}", f"{p['w_l1'][0]:.4f}", f"{p['w_l1'][1]:.4f}", f"{p['loss'](p['w_l1']):.4f}"),
            ("L2", f"{p['lam2']:.4f}", f"{p['w_l2'][0]:.4f}", f"{p['w_l2'][1]:.4f}", f"{p['loss'](p['w_l2']):.4f}")]
    head = ("Loss 1/2 (w - c)^T A (w - c) with c = (1.2, 1.9) and A = [[1.0, 0.6], [0.6, 1.0]]. The L1 row minimises "
            "loss + lambda ||w||_1; the L2 row minimises loss + lambda/2 ||w||^2 with lambda found by bisection "
            "so that ||w||_2 equals the L1 solution's.\n\n")
    return head + _table(["penalty", "lambda", "w1", "w2", "loss"], rows)
