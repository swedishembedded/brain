# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The Distributions Machine Learning Uses".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: distributions.<name> -->` markers. Helpers with a leading
underscore (or that take arguments) are shared with the chapter's figures, so a
picture and the table beside it cannot disagree. Small tables are computed
exactly (fractions or enumeration), areas under the bell curve with Simpson's
rule, and every simulation draws from a named `bookrng` stream. Gaussian draws
use Python's `gauss` on such a stream; how a Gaussian is built from uniform
numbers is the subject of the randomness chapter, not of this one.

Swedish Embedded AB implements solutions for noise, uncertainty and sampling in
machine learning systems. If your team needs expertise in probabilistic
modelling, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from fractions import Fraction

import bookrng


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fx(v, d=4):
    v = float(v) + 0.0  # turns -0.0 into 0.0
    s = f"{v:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def _simpson(f, a, b, n=2000):
    """Area under f on [a, b] by Simpson's rule (Chapter 9); n must be even."""
    h = (b - a) / n
    s = f(a) + f(b)
    s += 4 * sum(f(a + (2 * i - 1) * h) for i in range(1, n // 2 + 1))
    s += 2 * sum(f(a + 2 * i * h) for i in range(1, n // 2))
    return s * h / 3


def _bell(x):
    return math.exp(-x * x / 2)


def _phi(x):
    """The standard Gaussian density."""
    return _bell(x) / math.sqrt(2 * math.pi)


def _mean(v):
    return sum(v) / len(v)


def _var(v):
    m = _mean(v)
    return sum((a - m) ** 2 for a in v) / len(v)


def _cov(x, y):
    mx, my = _mean(x), _mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / len(x)


def _draw(pmf, rng):
    """One index from a table of probabilities: lay them end to end, see where u lands."""
    u, acc = rng.random(), 0.0
    for k, p in enumerate(pmf):
        acc += p
        if u < acc:
            return k
    return len(pmf) - 1


# ------------------------------------------------- the opening: coins to a bell

COIN_RUNS, COIN_FLIPS = 5000, 30


def _coin_head_counts():
    rng = bookrng.stream("distributions.coin_sum_counts")
    return [sum(rng.random() < 0.5 for _ in range(COIN_FLIPS)) for _ in range(COIN_RUNS)]


def coin_sum_counts():
    counts = _coin_head_counts()
    rows, shown = [], 0
    for k in range(8, 23):
        c = counts.count(k)
        shown += c
        rows.append([k, c, "#" * round(c / 20)])
    rows.append(["other (7 or fewer, 23 or more)", COIN_RUNS - shown, ""])
    return _table(["heads in 30 flips", "runs of 5000", "bar (one # per 20 runs)"], rows, "lrl")


def coin_sum_sigma():
    counts = _coin_head_counts()
    n, p = COIN_FLIPS, 0.5
    sigma = math.sqrt(n * p * (1 - p))
    rows = []
    for k, rule in ((1, 0.682689), (2, 0.954500), (3, 0.997300)):
        reach = math.floor(k * sigma)
        lo, hi = 15 - reach, 15 + reach
        observed = sum(1 for c in counts if lo <= c <= hi) / COIN_RUNS
        exact = sum(math.comb(n, j) for j in range(lo, hi + 1)) / 2 ** n
        rows.append([f"within {k} sigma: {lo} to {hi} heads", _fx(observed), _fx(exact), _fx(rule, 4)])
    head = f"Mean {n * p:g} heads, variance {n * p * (1 - p):g}, standard deviation {sigma:.4f}.\n\n"
    return head + _table(["window", "observed share of the 5000 runs", "exact binomial share",
                          "bell-curve rule"], rows)


# ------------------------------------------------------------------- Bernoulli

def bernoulli_table():
    rows = []
    for p in (0.1, 0.15, 0.3, 0.5, 0.9):
        mean = 0 * (1 - p) + 1 * p
        second = 0 * (1 - p) + 1 * p            # X^2 = X for a 0/1 variable
        var = second - mean ** 2
        rows.append([f"{p:g}", _fx(mean, 2), _fx(second, 2), _fx(var, 4), _fx(p * (1 - p), 4), _fx(math.sqrt(var), 4)])
    return _table(["p", "E[X]", "E[X^2]", "Var[X] = E[X^2] - E[X]^2", "p(1-p)", "std"], rows)


def _binomial_by_convolution(n, p):
    pmf = [1.0]                                  # zero Bernoulli variables: the count is 0
    for _ in range(n):
        nxt = [0.0] * (len(pmf) + 1)
        for k, mass in enumerate(pmf):
            nxt[k] += mass * (1 - p)             # this example is wrong: count stays k
            nxt[k + 1] += mass * p               # this example is right: count becomes k + 1
        pmf = nxt
    return pmf


def binomial_from_bernoullis():
    n, p = 10, 0.9
    pmf = _binomial_by_convolution(n, p)
    rows = []
    for k in range(n + 1):
        closed = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
        rows.append([k, _fx(pmf[k], 6), _fx(closed, 6)])
    mean = sum(k * m for k, m in enumerate(pmf))
    var = sum(k * k * m for k, m in enumerate(pmf)) - mean ** 2
    tail = (f"\n\nFrom the table: mean {mean:.4f} (n p = {n * p:g}), variance {var:.4f} "
            f"(n p (1-p) = {n * p * (1 - p):.4f}), total probability {sum(pmf):.6f}.")
    return _table(["correct out of 10", "adding one Bernoulli at a time", "C(10,k) p^k (1-p)^(10-k)"], rows) + tail


# ----------------------------------------------------------------- categorical

TOKENS = ("mat", "floor", "sofa", "moon", "the")
LOGITS = (3.0, 2.0, 1.0, -1.0, 0.0)


def _softmax(z):
    m = max(z)
    e = [math.exp(a - m) for a in z]
    s = sum(e)
    return [a / s for a in e]


def categorical_softmax():
    p = _softmax(LOGITS)
    raw = [math.exp(z) for z in LOGITS]
    rows = [[t, f"{z:g}", _fx(r), _fx(q)] for t, z, r, q in zip(TOKENS, LOGITS, raw, p)]
    rows.append(["total", "", _fx(sum(raw)), _fx(sum(p))])
    return _table(["next token", "score z", "e^z", "probability e^z / sum"], rows)


def categorical_sampling():
    p = _softmax(LOGITS)
    rng = bookrng.stream("distributions.categorical_sampling")
    n = 20000
    counts = [0] * len(p)
    for _ in range(n):
        counts[_draw(p, rng)] += 1
    rows = []
    for k, t in enumerate(TOKENS):
        onehot_mean = counts[k] / n               # mean of the indicator "token k was drawn"
        rows.append([t, _fx(p[k]), counts[k], _fx(onehot_mean), _fx(p[k] * (1 - p[k])),
                     _fx(onehot_mean * (1 - onehot_mean))])
    return _table(["token", "p", "draws of 20,000", "frequency = mean of indicator", "p(1-p)",
                   "variance of the indicators"], rows)


def categorical_covariance():
    p = _softmax(LOGITS)
    k = len(p)
    sigma = [[(p[i] if i == j else 0.0) - p[i] * p[j] for j in range(k)] for i in range(k)]
    rows = [[TOKENS[i]] + [_fx(sigma[i][j]) for j in range(k)] + [_fx(sum(sigma[i]), 4)] for i in range(k)]
    return _table(["Cov of indicators"] + list(TOKENS) + ["row sum"], rows)


# ------------------------------------------------------------------- Gaussian

def gaussian_shape():
    rows = []
    for i in range(9):
        x = i / 2
        rows.append([f"{x:g}", _fx(_bell(x), 5), _fx(_phi(x), 5), _fx((x * x - 1) * _phi(x), 5)])
    return _table(["x", "e^(-x^2/2)", "density = e^(-x^2/2) / sqrt(2 pi)", "second derivative (x^2 - 1) * density"], rows)


def _normal_pdf(x, mu, sigma):
    return _phi((x - mu) / sigma) / sigma


def gaussian_moments():
    def moments(mu, sigma):
        f = lambda x: _normal_pdf(x, mu, sigma)  # noqa: E731
        lo, hi = mu - 8 * sigma, mu + 8 * sigma
        area = _simpson(f, lo, hi)
        mean = _simpson(lambda x: x * f(x), lo, hi)
        var = _simpson(lambda x: (x - mean) ** 2 * f(x), lo, hi)
        return area, mean, var

    a1, m1, v1 = moments(0.0, 1.0)
    a2, m2, v2 = moments(50.0, 0.2)
    rows = [["area (should be 1)", _fx(a1, 6), _fx(a2, 6)],
            ["mean (integral of x f(x))", _fx(m1, 6), _fx(m2, 6)],
            ["variance (integral of (x - mean)^2 f(x))", _fx(v1, 6), _fx(v2, 6)]]
    return _table(["quantity from Simpson's rule, 8 sigma each side", "mu = 0, sigma = 1", "mu = 50, sigma = 0.2"], rows)


def gaussian_standardise():
    mu, sigma = 50.0, 0.2
    rows = []
    for x in (49.7, 49.9, 50.0, 50.3, 50.5):
        z = (x - mu) / sigma
        rows.append([f"{x:.1f}", _fx(z, 2), _fx(_phi(z), 5), _fx(_normal_pdf(x, mu, sigma), 5)])
    t1 = _table(["x (mm)", "z = (x - mu) / sigma", "standard density at z", "density of x = that / sigma"], rows)
    in_x = _simpson(lambda x: _normal_pdf(x, mu, sigma), 49.8, 50.3)
    in_z = _simpson(_phi, -1.0, 1.5)
    t2 = _table(["probability that 49.8 <= X <= 50.3", "value"],
                [["integrating the density of x from 49.8 to 50.3", _fx(in_x, 6)],
                 ["integrating the standard density from z = -1 to z = 1.5", _fx(in_z, 6)]])
    return t1 + "\n\n" + t2


def sigma_rule():
    rows = []
    for k in (1.0, 1.645, 1.96, 2.0, 2.5, 2.576, 3.0, 4.0):
        inside = _simpson(_phi, -k, k, 4000)
        rows.append([f"{k:g}", _fx(inside, 6), _fx(1 - inside, 6), f"{1 / (1 - inside):,.0f}"])
    return _table(["k", "P(-k <= Z <= k)", "P(outside)", "outside once in about this many draws"], rows)


def gaussian_sum():
    rng = bookrng.stream("distributions.gaussian_sum")
    n = 40000
    xs = [rng.gauss(1.0, 2.0) for _ in range(n)]
    ys = [rng.gauss(-0.5, 1.5) for _ in range(n)]
    ss = [a + b for a, b in zip(xs, ys)]
    m, v = _mean(ss), _var(ss)
    inside = sum(abs(s - 0.5) <= 2.5 for s in ss) / n
    rows = [["X ~ N(1, 2^2)", _fx(_mean(xs), 3), _fx(_var(xs), 3), "1", "4"],
            ["Y ~ N(-0.5, 1.5^2)", _fx(_mean(ys), 3), _fx(_var(ys), 3), "-0.5", "2.25"],
            ["X + Y", _fx(m, 3), _fx(v, 3), "0.5", "6.25"]]
    tail = f"\n\nShare of X + Y within one predicted standard deviation (2.5) of 0.5: {inside:.4f}; the bell-curve rule says 0.6827."
    return _table(["variable", "sample mean", "sample variance", "mean predicted", "variance predicted"], rows) + tail


def noised_coordinate():
    rng = bookrng.stream("distributions.noised_coordinate")
    x0, n = 2.0, 20000
    eps = [rng.gauss(0.0, 1.0) for _ in range(n)]
    rows = []
    for s in (0.1, 0.3, 0.5, 0.9):
        xs = [(1 - s) * x0 + s * e for e in eps]
        rows.append([f"{s:g}", _fx((1 - s) * x0, 3), _fx(_mean(xs), 3), f"{s:g}", _fx(math.sqrt(_var(xs)), 3)])
    return _table(["noise level s", "predicted mean (1-s) x0", "sample mean", "predicted std s", "sample std"], rows)


# ----------------------------------------------------------------- covariance

RATER_A = (1, 2, 3, 4, 5)
RATER_B = (2, 1, 4, 3, 5)


def covariance_table():
    x, y = RATER_A, RATER_B
    n = len(x)
    mx, my = _mean(x), _mean(y)
    rows = [[i + 1, a, b, f"{a - mx:g}", f"{b - my:g}", f"{(a - mx) * (b - my):g}"] for i, (a, b) in enumerate(zip(x, y))]
    rows.append(["mean / sum", f"{mx:g}", f"{my:g}", "", "", f"{sum((a - mx) * (b - my) for a, b in zip(x, y)):g}"])
    cov = _cov(x, y)
    exy = _mean([a * b for a, b in zip(x, y)])
    tail = (f"\n\nCov[A, B] = {sum((a - mx) * (b - my) for a, b in zip(x, y)):g} / {n} = {cov:g}. "
            f"Second form: E[AB] = {exy:g}, E[A] E[B] = {mx * my:g}, difference {exy - mx * my:g}.\n"
            f"Var[A] = {_var(x):g}, Var[B] = {_var(y):g}, sum of the two plus 2 Cov = {_var(x) + _var(y) + 2 * cov:g}; "
            f"Var[A + B] computed directly = {_var([a + b for a, b in zip(x, y)]):g}.")
    return _table(["answer", "rater A", "rater B", "A - mean", "B - mean", "(A - mean)(B - mean)"], rows) + tail


def correlation_cosine():
    x, y = list(RATER_A), list(RATER_B)

    def stats(x, y):
        mx, my = _mean(x), _mean(y)
        dx, dy = [a - mx for a in x], [b - my for b in y]
        dot = sum(a * b for a, b in zip(dx, dy))
        cos = dot / (math.sqrt(sum(a * a for a in dx)) * math.sqrt(sum(b * b for b in dy)))
        c = _cov(x, y)
        rho = c / (math.sqrt(_var(x)) * math.sqrt(_var(y)))
        return c, rho, cos

    c1, r1, k1 = stats(x, y)
    x2 = [100 * a for a in x]
    y2 = [b + 7 for b in y]
    c2, r2, k2 = stats(x2, y2)
    rows = [["original scores", _fx(c1, 4), _fx(r1, 4), _fx(k1, 4)],
            ["A times 100, B plus 7", _fx(c2, 4), _fx(r2, 4), _fx(k2, 4)]]
    return _table(["data", "covariance", "correlation = Cov / (std A * std B)", "cosine of the two centred vectors"], rows)


def correlation_shapes():
    rng = bookrng.stream("distributions.correlation_shapes")
    n = 2000
    xs = [i / (n - 1) * 4 - 2 for i in range(n)]
    cases = [("y = 2x + 1", [2 * a + 1 for a in xs]),
             ("y = -x", [-a for a in xs]),
             ("y = x + noise (std 1)", [a + rng.gauss(0.0, 1.0) for a in xs]),
             ("y independent noise", [rng.gauss(0.0, 1.0) for _ in xs]),
             ("y = x^2", [a * a for a in xs])]
    rows = []
    for name, ys in cases:
        c = _cov(xs, ys)
        rows.append([name, _fx(c, 4), _fx(c / math.sqrt(_var(xs) * _var(ys)), 4)])
    return _table(["relation between x and y", "covariance", "correlation"], rows)


def uncorrelated_not_independent():
    third = Fraction(1, 3)
    joint = {(-1, 1): third, (0, 0): third, (1, 1): third}     # Y = X^2, X uniform on {-1, 0, 1}
    ex = sum(x * p for (x, _), p in joint.items())
    ey = sum(y * p for (_, y), p in joint.items())
    exy = sum(x * y * p for (x, y), p in joint.items())
    rows = [[x, y, str(p)] for (x, y), p in sorted(joint.items())]
    t1 = _table(["X", "Y = X^2", "probability"], rows)
    p_y1 = sum(p for (_, y), p in joint.items() if y == 1)
    p_x1 = sum(p for (x, _), p in joint.items() if x == 1)
    p_both = joint[(1, 1)]
    t2 = _table(["quantity", "value"],
                [["E[X], E[Y], E[XY]", f"{ex}, {ey}, {exy}"],
                 ["Cov[X, Y] = E[XY] - E[X] E[Y]", str(exy - ex * ey)],
                 ["P(Y = 1)", str(p_y1)],
                 ["P(Y = 1 given X = 1) = P(X = 1, Y = 1) / P(X = 1)", str(p_both / p_x1)],
                 ["P(X = 1) P(Y = 1) versus P(X = 1, Y = 1)", f"{p_x1 * p_y1} versus {p_both}"]], "lr")
    return t1 + "\n\n" + t2


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(a):
    return [list(r) for r in zip(*a)]


def _empirical_covariance(rows):
    """Covariance matrix of data given as a list of observations (each a list of D numbers)."""
    n, d = len(rows), len(rows[0])
    mu = [_mean([r[j] for r in rows]) for j in range(d)]
    centred = [[r[j] - mu[j] for j in range(d)] for r in rows]
    s = _matmul(_transpose(centred), centred)
    return [[v / n for v in row] for row in s]


def covariance_matrix():
    data = [[a, b] for a, b in zip(RATER_A, RATER_B)]
    s = _empirical_covariance(data)
    rows = [["A", _fx(s[0][0], 2), _fx(s[0][1], 2)], ["B", _fx(s[1][0], 2), _fx(s[1][1], 2)]]
    tail = ("\n\nCentred data C (5 x 2), then C^T C / 5:\n\n"
            + _table(["row", "A - 3", "B - 3"], [[i + 1, a - 3, b - 3] for i, (a, b) in enumerate(data)]))
    return _table(["Sigma", "A", "B"], rows, "lrr") + tail


def _sym2_eigen(s):
    """Eigenvalues (largest first) and unit eigenvectors of a symmetric 2x2 matrix, in closed form."""
    a, b, c = s[0][0], s[0][1], s[1][1]
    t, d = a + c, a * c - b * b
    disc = math.sqrt(max(t * t / 4 - d, 0.0))
    l1, l2 = t / 2 + disc, t / 2 - disc
    v = (b, l1 - a) if abs(b) > 1e-12 else ((1.0, 0.0) if a >= c else (0.0, 1.0))
    norm = math.hypot(*v)
    v1 = (v[0] / norm, v[1] / norm)
    if v1[0] < 0 or (v1[0] == 0 and v1[1] < 0):
        v1 = (-v1[0], -v1[1])
    v2 = (-v1[1], v1[0])
    return (l1, l2), (v1, v2)


def direction_variance():
    data = [[a, b] for a, b in zip(RATER_A, RATER_B)]
    s = _empirical_covariance(data)
    rows = []
    for deg in (0, 30, 45, 90, 135):
        t = math.radians(deg)
        u = (math.cos(t), math.sin(t))
        proj = [u[0] * r[0] + u[1] * r[1] for r in data]
        quad = sum(u[i] * s[i][j] * u[j] for i in range(2) for j in range(2))
        rows.append([deg, f"({u[0]:.4f}, {u[1]:.4f})", _fx(_var(proj), 4), _fx(quad, 4)])
    (l1, l2), (v1, v2) = _sym2_eigen(s)
    tail = (f"\n\nEigenvalues of Sigma: {l1:.4f} and {l2:.4f}, with unit eigenvectors ({v1[0]:.4f}, {v1[1]:.4f}) "
            f"and ({v2[0]:.4f}, {v2[1]:.4f}).")
    return _table(["angle (degrees)", "unit direction u", "variance of u . x measured on the data", "u^T Sigma u"], rows) + tail


# ------------------------------------------------- multivariate Gaussian

SIGMA = [[2.0, 1.2], [1.2, 1.0]]
MU = (1.0, -0.5)


def _cholesky2(s):
    a, b, c = s[0][0], s[0][1], s[1][1]
    if a <= 0 or a * c - b * b <= 0:
        raise ValueError("matrix is not positive definite")
    l11 = math.sqrt(a)
    l21 = b / l11
    l22 = math.sqrt(c - l21 * l21)
    return [[l11, 0.0], [l21, l22]]


def _eigen_factor(s):
    (l1, l2), (v1, v2) = _sym2_eigen(s)
    # columns of A are sqrt(lambda_i) * v_i, so that A A^T = Q Lambda Q^T
    return [[math.sqrt(l1) * v1[0], math.sqrt(l2) * v2[0]], [math.sqrt(l1) * v1[1], math.sqrt(l2) * v2[1]]]


def mvn_factors():
    (l1, l2), (v1, v2) = _sym2_eigen(SIGMA)
    L = _cholesky2(SIGMA)
    A = _eigen_factor(SIGMA)
    LLt = _matmul(L, _transpose(L))
    AAt = _matmul(A, _transpose(A))
    m = lambda M: "[[" + ", ".join(f"{v:.4f}" for v in M[0]) + "], [" + ", ".join(f"{v:.4f}" for v in M[1]) + "]]"  # noqa: E731
    rows = [["Sigma", m(SIGMA)],
            ["det Sigma", f"{SIGMA[0][0] * SIGMA[1][1] - SIGMA[0][1] ** 2:.4f}"],
            ["eigenvalues", f"{l1:.4f}, {l2:.4f}"],
            ["unit eigenvectors", f"({v1[0]:.4f}, {v1[1]:.4f}), ({v2[0]:.4f}, {v2[1]:.4f})"],
            ["lower-triangular L", m(L)],
            ["L L^T", m(LLt)],
            ["eigen factor A = Q Lambda^(1/2)", m(A)],
            ["A A^T", m(AAt)]]
    return _table(["quantity", "value"], rows, "ll")


def _sample_cholesky(rng, mu, s, n):
    L = _cholesky2(s)
    out = []
    for _ in range(n):
        z = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        out.append([mu[0] + L[0][0] * z[0], mu[1] + L[1][0] * z[0] + L[1][1] * z[1]])
    return out


def _sample_eigen(rng, mu, s, n):
    A = _eigen_factor(s)
    out = []
    for _ in range(n):
        z = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        out.append([mu[0] + A[0][0] * z[0] + A[0][1] * z[1], mu[1] + A[1][0] * z[0] + A[1][1] * z[1]])
    return out


def mvn_sampling():
    n = 20000
    rows = []
    for name, fn in (("Cholesky", _sample_cholesky), ("eigen-decomposition", _sample_eigen)):
        pts = fn(bookrng.stream(f"distributions.mvn_sampling.{name}"), MU, SIGMA, n)
        s = _empirical_covariance(pts)
        mu = [_mean([p[j] for p in pts]) for j in range(2)]
        rows.append([name, f"({mu[0]:.3f}, {mu[1]:.3f})", _fx(s[0][0], 3), _fx(s[0][1], 3), _fx(s[1][1], 3)])
    rows.append(["target", f"({MU[0]:g}, {MU[1]:g})", _fx(SIGMA[0][0], 3), _fx(SIGMA[0][1], 3), _fx(SIGMA[1][1], 3)])
    return _table(["method (20,000 samples)", "sample mean", "Sigma_11", "Sigma_12", "Sigma_22"], rows)


def mvn_ellipse_mass():
    pts = _sample_cholesky(bookrng.stream("distributions.mvn_ellipse_mass"), MU, SIGMA, 20000)
    det = SIGMA[0][0] * SIGMA[1][1] - SIGMA[0][1] ** 2
    inv = [[SIGMA[1][1] / det, -SIGMA[0][1] / det], [-SIGMA[1][0] / det, SIGMA[0][0] / det]]
    (l1, l2), _ = _sym2_eigen(SIGMA)

    def d2(p):
        d = (p[0] - MU[0], p[1] - MU[1])
        return d[0] * (inv[0][0] * d[0] + inv[0][1] * d[1]) + d[1] * (inv[1][0] * d[0] + inv[1][1] * d[1])

    dist = [d2(p) for p in pts]
    rows = []
    for c in (1.0, 4.0, 9.0):
        rows.append([f"{c:g}", f"{math.sqrt(c * l1):.4f}", f"{math.sqrt(c * l2):.4f}",
                     _fx(sum(d <= c for d in dist) / len(dist)), _fx(1 - math.exp(-c / 2))])
    return _table(["c in (x-mu)^T Sigma^-1 (x-mu) <= c", "half-length along v1", "half-length along v2",
                   "share of 20,000 samples inside", "1 - e^(-c/2)"], rows)


def high_dim_norm():
    rows = []
    for d in (1, 10, 100, 1000, 10000):
        rng = bookrng.stream(f"distributions.high_dim_norm.{d}")
        reps = 400 if d <= 1000 else 100
        sq = []
        for _ in range(reps):
            sq.append(sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(d)) / d)
        norms = [math.sqrt(s * d) for s in sq]
        rows.append([d, _fx(_mean(sq), 4), _fx(math.sqrt(_var(sq)), 4), _fx(math.sqrt(2 / d), 4),
                     _fx(_mean(norms), 3), _fx(math.sqrt(d), 3)])
    return _table(["dimension d", "mean of squared length / d", "std of squared length / d", "sqrt(2/d)", "mean length", "sqrt(d)"], rows)


def init_variance():
    rows = []
    for n in (16, 64, 256, 1024):
        rng = bookrng.stream(f"distributions.init_variance.{n}")
        x = [rng.gauss(0.0, 1.0) for _ in range(n)]
        norm2 = sum(a * a for a in x)
        for label, sw in (("1", 1.0), ("1/sqrt(n)", 1 / math.sqrt(n))):
            ys = [sum(rng.gauss(0.0, sw) * a for a in x) for _ in range(1500)]
            rows.append([n, label, _fx(norm2, 1), _fx(sw ** 2 * norm2, 2), _fx(_var(ys), 2)])
    return _table(["inputs n", "weight std", "sum of x_i^2", "predicted Var[y] = std^2 * sum of x_i^2", "measured Var[y], 1500 weight draws"], rows)


# --------------------------------------------------- large numbers and the CLT

def _chebyshev_row(n, p, eps, reps, rng):
    sigma2 = p * (1 - p)
    means = [sum(rng.random() < p for _ in range(n)) / n for _ in range(reps)]
    rmse = math.sqrt(sum((m - p) ** 2 for m in means) / reps)
    far = sum(abs(m - p) > eps for m in means) / reps
    return [n, _fx(rmse, 4), _fx(math.sqrt(sigma2 / n), 4), _fx(far, 4), _fx(min(1.0, sigma2 / (n * eps * eps)), 4)]


def lln_bernoulli():
    p, eps, reps = 0.3, 0.05, 300
    rng = bookrng.stream("distributions.lln_bernoulli")
    rows = [_chebyshev_row(n, p, eps, reps, rng) for n in (10, 100, 1000, 10000)]
    return _table(["n draws per mean", "RMS error of the mean over 300 repeats", "sigma / sqrt(n)",
                   "share of repeats with error > 0.05", "Chebyshev bound sigma^2 / (n * 0.05^2)"], rows)


def lln_heavy_tail():
    checkpoints = (100, 1000, 10000, 100000)
    out = []
    for rep in range(3):
        rng = bookrng.stream(f"distributions.lln_heavy_tail.{rep}")
        g_sum = r_sum = 0.0
        g_means, r_means = [], []
        n = 0
        for cp in checkpoints:
            while n < cp:
                g_sum += rng.gauss(0.0, 1.0)
                den = rng.gauss(0.0, 1.0)
                r_sum += rng.gauss(0.0, 1.0) / den if den != 0.0 else 0.0
                n += 1
            g_means.append(g_sum / n)
            r_means.append(r_sum / n)
        out.append((g_means, r_means))
    rows = []
    for i, cp in enumerate(checkpoints):
        rows.append([f"{cp:,}"] + [_fx(out[r][0][i], 4) for r in range(3)] + [_fx(out[r][1][i], 3) for r in range(3)])
    return _table(["n", "Gaussian, run 1", "run 2", "run 3", "ratio of two Gaussians, run 1", "run 2", "run 3"], rows)


CLT_VALUES = (0, 1, 2, 10)
CLT_PROBS = (0.6, 0.25, 0.1, 0.05)


def _clt_moments():
    mu = sum(v * p for v, p in zip(CLT_VALUES, CLT_PROBS))
    var = sum((v - mu) ** 2 * p for v, p in zip(CLT_VALUES, CLT_PROBS))
    skew = sum((v - mu) ** 3 * p for v, p in zip(CLT_VALUES, CLT_PROBS)) / var ** 1.5
    return mu, var, skew


def _clt_standardised(n, reps=10000):
    """Sample means of n draws of the skewed variable, shifted and scaled to mean 0 and std 1."""
    mu, var, _ = _clt_moments()
    rng = bookrng.stream(f"distributions.clt_table.n{n}")
    out = []
    for _ in range(reps):
        s = sum(CLT_VALUES[_draw(CLT_PROBS, rng)] for _ in range(n))
        out.append((s / n - mu) / math.sqrt(var / n))
    return out


def clt_setup():
    mu, var, skew = _clt_moments()
    rows = [[v, f"{p:g}"] for v, p in zip(CLT_VALUES, CLT_PROBS)]
    tail = (f"\n\nE[X] = {mu:g}, Var[X] = {var:.4f}, standard deviation {math.sqrt(var):.4f}, "
            f"skewness {skew:.4f}.")
    return _table(["value of X", "probability"], rows) + tail


def clt_table():
    mu, var, skew = _clt_moments()
    rows = []
    for n in (1, 2, 5, 30, 100):
        z = _clt_standardised(n)
        reps = len(z)
        rows.append([n, _fx(_mean(z), 3), _fx(math.sqrt(_var(z)), 3), _fx(_mean([a ** 3 for a in z]), 3),
                     _fx(skew / math.sqrt(n), 3), _fx(sum(abs(a) <= 1 for a in z) / reps, 3),
                     _fx(sum(abs(a) <= 2 for a in z) / reps, 3)])
    rows.append(["bell curve", "0", "1", "0", "0", "0.683", "0.954"])
    return _table(["n", "mean of z", "std of z", "skewness of z", "skewness of X / sqrt(n)",
                   "share with z in [-1, 1]", "share with z in [-2, 2]"], rows)
