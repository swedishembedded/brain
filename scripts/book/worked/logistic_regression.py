# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Classification and Logistic Regression".

Plain Python 3, standard library only, written from the chapter's own
description (odds and the logit, the sigmoid and its slope, binary
cross-entropy, its gradient and Hessian, gradient descent, the stable form of
the loss, and the behaviour on separable data). Each public function without
a leading underscore that takes no arguments returns a markdown string that
the chapter embeds between `<!-- worked: logistic_regression.<name> -->`
markers. Underscore functions are shared with the chapter's figures, so a
picture and the table beside it cannot disagree.

The data are two small sets of inspected machine parts. A hidden law (written
here, and only here, because the book plays the part of nature) decides how
likely a part is to be rejected from its measurements; the learner sees only
the measurements and the yes-or-no outcomes.

Swedish Embedded AB implements solutions for classifying measurements into
decisions that can be trusted, such as pass or fail inspection of parts. If
your team needs expertise in classification models and their numerical
behaviour, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import struct
from decimal import Decimal, getcontext
from functools import lru_cache

import bookrng

ONE_NAME = "logistic_regression.one_feature"
PARTS_NAME = "logistic_regression.parts"

# One measurement: nature's law is P(reject) = sigmoid(ONE_LAW[0] + ONE_LAW[1] * x).
ONE_LAW = (-5.5, 2.2)
N_ONE = 12
X_ONE_MAX = 5.0
# A grossly oversize part appended to the one-feature data (it is a clear reject).
FAR_PART = (15.0, 1)

# Two measurements: P(reject) = sigmoid(PARTS_LAW[0] + PARTS_LAW[1] * x1 + PARTS_LAW[2] * x2).
PARTS_LAW = (-6.0, 1.2, 1.0)
N_PARTS = 60
X_MAX = 5.0

ETA_SCALE = 1.0        # learning rate = ETA_SCALE / lam_cap, lam_cap = lambda_max(X^T X / N) / 4
GRAD_TOL = 1e-5        # "converged": the gradient norm is below this
STEP_CAP = 200_000


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _generator_line(name):
    seed = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed})."


def _sci(x, digits=2):
    """Plain decimal for moderate numbers, m x 10^e otherwise (zero and inf as such)."""
    if x == 0:
        return "0"
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    if 1e-3 <= abs(x) < 1e6:
        return f"{x:.{digits + 2}f}"
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m} x 10^{int(e)}"


# ------------------------------------------------------- scalar functions

def sigmoid(z):
    """The two-branch form: the exponent is never positive, so exp cannot overflow."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def softplus(z):
    """log(1 + e^z) as max(z, 0) + log1p(e^-|z|)."""
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def logit(p):
    return math.log(p / (1.0 - p))


def bce_from_logit(z, y):
    """Binary cross-entropy of label y in {0, 1} against logit z: softplus(z) - y z."""
    return softplus(z) - y * z


def bce_from_prob(p, y):
    """The textbook form -[y log p + (1 - y) log(1 - p)]; fails at p = 0 or 1."""
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


# ------------------------------------------------------------------- data

def _draw_one(rng):
    rows = []
    for _ in range(N_ONE):
        x = round(X_ONE_MAX * rng.uniform(), 1)
        u = rng.uniform()
        y = 1 if u < sigmoid(ONE_LAW[0] + ONE_LAW[1] * x) else 0
        rows.append((x, y))
    return tuple(sorted(rows))


@lru_cache(maxsize=None)
def _one_feature():
    """Twelve (x, y) rows, sorted by x."""
    return _draw_one(bookrng.generator(ONE_NAME))


def _draw_parts(rng):
    rows = []
    for _ in range(N_PARTS):
        x1 = round(X_MAX * rng.uniform(), 1)
        x2 = round(X_MAX * rng.uniform(), 1)
        u = rng.uniform()
        y = 1 if u < sigmoid(PARTS_LAW[0] + PARTS_LAW[1] * x1 + PARTS_LAW[2] * x2) else 0
        rows.append((x1, x2, y))
    return tuple(rows)


@lru_cache(maxsize=None)
def _parts():
    """Sixty (x1, x2, y) rows in draw order."""
    return _draw_parts(bookrng.generator(PARTS_NAME))


def _design(rows):
    """[N, 3] design matrix (1, x1, x2) and the label list."""
    return [[1.0, r[0], r[1]] for r in rows], [float(r[2]) for r in rows]


# ----------------------------------------------------------- linear algebra

def _dot(a, b):
    return sum(u * v for u, v in zip(a, b))


def _matvec(a, v):
    return [_dot(r, v) for r in a]


def _solve(a, b):
    """Gaussian elimination with partial pivoting (square, nonsingular)."""
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def _gram(x, weights=None):
    """X^T diag(weights) X (weights default to ones)."""
    d = len(x[0])
    g = [[0.0] * d for _ in range(d)]
    for n, row in enumerate(x):
        s = 1.0 if weights is None else weights[n]
        for i in range(d):
            for j in range(d):
                g[i][j] += s * row[i] * row[j]
    return g


def _eigenvalues(a, sweeps=20):
    """Eigenvalues of a small symmetric matrix by cyclic Jacobi rotations, ascending."""
    n = len(a)
    a = [list(r) for r in a]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-24:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
    return sorted(a[i][i] for i in range(n))


# ------------------------------------------------------- the logistic model

def _probs(x, w):
    return [sigmoid(_dot(r, w)) for r in x]


def _loss(x, y, w):
    """Mean binary cross-entropy, computed from the logits (stable)."""
    return sum(bce_from_logit(_dot(r, w), t) for r, t in zip(x, y)) / len(y)


def _gradient(x, y, w):
    """(1/N) X^T (sigmoid(Xw) - y)."""
    n, d = len(x), len(w)
    p = _probs(x, w)
    return [sum((p[i] - y[i]) * x[i][j] for i in range(n)) / n for j in range(d)]


def _mse_loss(x, y, w):
    return sum((sigmoid(_dot(r, w)) - t) ** 2 for r, t in zip(x, y)) / len(y)


def _descend(x, y, w0, eta, steps, record=()):
    """`steps` plain gradient steps; returns (final w, {step: (w, loss)} for steps in `record`)."""
    w = list(w0)
    seen = {}
    if 0 in record:
        seen[0] = (list(w), _loss(x, y, w))
    for k in range(1, steps + 1):
        g = _gradient(x, y, w)
        w = [wi - eta * gi for wi, gi in zip(w, g)]
        if k in record:
            seen[k] = (list(w), _loss(x, y, w))
    return w, seen


def _smoothness(x):
    """lambda_cap = lambda_max(X^T X / N) / 4, an upper bound on the largest curvature of the loss."""
    n = len(x)
    g = [[v / n for v in r] for r in _gram(x)]
    return _eigenvalues(g)[-1] / 4.0


@lru_cache(maxsize=None)
def _fit_parts():
    """Train on the sixty parts from w = 0 with eta = ETA_SCALE / lambda_cap until the gradient is tiny.

    Returns (w, steps, eta, lambda_cap, x, y).
    """
    x, y = _design(_parts())
    lam_cap = _smoothness(x)
    eta = ETA_SCALE / lam_cap
    w = [0.0, 0.0, 0.0]
    for k in range(1, STEP_CAP + 1):
        g = _gradient(x, y, w)
        if math.sqrt(_dot(g, g)) < GRAD_TOL:
            return tuple(w), k - 1, eta, lam_cap, x, y
        w = [wi - eta * gi for wi, gi in zip(w, g)]
    return tuple(w), STEP_CAP, eta, lam_cap, x, y


NEWTON_ITERATIONS = 30


def _hessian(x, w):
    """(1/N) X^T S X with S = diag(p (1 - p))."""
    s = [p * (1.0 - p) for p in _probs(x, w)]
    return [[v / len(x) for v in r] for r in _gram(x, s)]


def _fit_one_feature(rows):
    """The minimiser of the mean cross-entropy for (x, y) rows with one feature.

    Newton's method from w = 0: w <- w - solve(H, g), NEWTON_ITERATIONS times. The loss is convex and the data
    are not separable, so the iteration converges to the unique minimiser (gradient norm far below 1e-12).
    """
    x = [[1.0, r[0]] for r in rows]
    y = [float(r[1]) for r in rows]
    w = [0.0, 0.0]
    for _ in range(NEWTON_ITERATIONS):
        step = _solve(_hessian(x, w), _gradient(x, y, w))
        w = [wi - si for wi, si in zip(w, step)]
    return w


def _ols(rows):
    """Least-squares line through (x, y) rows: returns (intercept, slope)."""
    n = len(rows)
    mx = sum(r[0] for r in rows) / n
    my = sum(r[1] for r in rows) / n
    sxx = sum((r[0] - mx) ** 2 for r in rows)
    sxy = sum((r[0] - mx) * (r[1] - my) for r in rows)
    slope = sxy / sxx
    return my - slope * mx, slope


# ------------------------------------------------------------ worked blocks

def one_feature_table():
    rows = _one_feature()
    out = [_generator_line(ONE_NAME),
           f"Draw order, part by part (N = {N_ONE}): x = round(5 u1, 1) with u1 = uniform(), then u2 = uniform(); "
           f"y = 1 (rejected) when u2 < sigmoid({ONE_LAW[0]} + {ONE_LAW[1]} x). Rows are shown sorted by x and numbered in that order.",
           ""]
    out.append(_table(["part", "x (0.01 mm)", "y"], [(i + 1, f"{x:.1f}", y) for i, (x, y) in enumerate(rows)]))
    return "\n".join(out)


def linear_fit_on_labels():
    rows = _one_feature()
    far = _one_feature() + (FAR_PART,)
    b0, b1 = _ols(rows)
    c0, c1 = _ols(far)

    def errors(i0, i1, data):
        return sum(1 for x, y in data if (1 if i0 + i1 * x >= 0.5 else 0) != y)

    def cross(i0, i1):
        return (0.5 - i0) / i1

    out = [_table(["fit", "intercept", "slope", "prediction at x = 0", "prediction at x = 5",
                   "x where line = 0.5", "rows misclassified"],
                  [("twelve parts", f"{b0:.4f}", f"{b1:.4f}", f"{b0:.4f}", f"{b0 + 5 * b1:.4f}",
                    f"{cross(b0, b1):.3f}", f"{errors(b0, b1, rows)} of {len(rows)}"),
                   (f"plus one part at x = {FAR_PART[0]:.1f} (y = 1)", f"{c0:.4f}", f"{c1:.4f}",
                    f"{c0:.4f}", f"{c0 + 5 * c1:.4f}", f"{cross(c0, c1):.3f}",
                    f"{errors(c0, c1, far)} of {len(far)}")])]
    return "\n".join(out)


def logistic_fit_one_feature():
    rows = _one_feature()
    far = rows + (FAR_PART,)
    out, fits = [], []
    for name, data in (("twelve parts", rows), (f"plus one part at x = {FAR_PART[0]:.1f}", far)):
        w = _fit_one_feature(data)
        fits.append(w)
        errs = sum(1 for x, y in data if (1 if w[0] + w[1] * x >= 0 else 0) != y)
        out.append((name, f"{w[0]:.4f}", f"{w[1]:.4f}", f"{-w[0] / w[1]:.3f}", f"{errs} of {len(data)}"))
    shift = [abs(a - b) for a, b in zip(*fits)]
    out.append(("absolute change of the weights", _sci(shift[0], 1), _sci(shift[1], 1), "", ""))
    head = (f"Weights that minimise the mean binary cross-entropy of the next sections, found by Newton's method from "
            f"w = (0, 0): {NEWTON_ITERATIONS} updates w <- w - solve(H, g, true), with the gradient g and Hessian H (both of the mean loss, with the factor 1/N) of the sections "
            "\"The gradient\" and \"Convexity: one valley\" (the gradient norm is then below 1e-12). The last row "
            "needs a solution that accurate; the others are printed to four decimals.\n\n")
    return head + _table(["fit", "w0", "w1", "x where p = 0.5", "rows misclassified"], out)


def odds_table():
    rows = []
    for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        odds = p / (1 - p)
        z = logit(p)
        rows.append((f"{p:.2f}", f"{odds:.4f}", f"{z:.4f}", f"{sigmoid(z):.4f}"))
    return _table(["p", "odds p/(1-p)", "logit z = log(odds)", "sigmoid(z)"], rows)


def sigmoid_properties():
    rows = []
    for z in (-8, -4, -2, -1, 0, 1, 2, 4, 8):
        s = sigmoid(z)
        sm = sigmoid(-z)
        h = 1e-5
        fd = (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)
        rows.append((z, f"{s:.6f}", f"{sm:.6f}", f"{s + sm:.6f}", f"{s * (1 - s):.6f}", f"{fd:.6f}"))
    return _table(["z", "sigmoid(z)", "sigmoid(-z)", "sum", "s (1 - s)", "central difference"], rows)


def single_example_losses():
    """One positive example (y = 1): both losses and their slopes with respect to the logit z."""
    rows = []
    for z in (-8, -4, -2, 0, 2, 4):
        p = sigmoid(z)
        bce = bce_from_logit(z, 1)
        se = (p - 1) ** 2
        g_bce = p - 1
        g_se = 2 * (p - 1) * p * (1 - p)
        rows.append((z, f"{p:.5f}", f"{bce:.5f}", f"{se:.5f}", f"{g_bce:.5f}", f"{g_se:.5f}"))
    return _table(["z", "p", "BCE", "squared error", "dBCE/dz", "dSE/dz"], rows)


def nonconvex_squared_error():
    """One example x = 1, y = 1, no intercept: loss as a function of the single weight w."""
    h = 0.01

    def se(w):
        return (sigmoid(w) - 1) ** 2

    def bce(w):
        return bce_from_logit(w, 1)

    def second(f, w):
        return (f(w + h) - 2 * f(w) + f(w - h)) / (h * h)

    rows = []
    for w in (-6, -4, -2, -1, 0, 1, 2, 4):
        rows.append((w, f"{se(w):.5f}", f"{second(se, w):+.5f}", f"{bce(w):.5f}", f"{second(bce, w):+.5f}"))
    return _table(["w", "squared error", "its second difference", "BCE", "its second difference"], rows)


def descent_squared_vs_bce():
    """Gradient descent on the one-example problem from w = -6 with eta = 1."""
    eta = 1.0
    record = {0, 5, 10, 50, 100, 500, 2000}

    def run(grad):
        w, seen = -6.0, {}
        for k in range(0, max(record) + 1):
            if k in record:
                seen[k] = w
            w = w - eta * grad(w)
        return seen

    se = run(lambda w: 2 * (sigmoid(w) - 1) * sigmoid(w) * (1 - sigmoid(w)))
    bce = run(lambda w: sigmoid(w) - 1)
    rows = [(k, f"{se[k]:.4f}", f"{(sigmoid(se[k]) - 1) ** 2:.5f}", f"{bce[k]:.4f}", f"{bce_from_logit(bce[k], 1):.5f}")
            for k in sorted(record)]
    head = ("One part with x = 1 and y = 1, no intercept, so z = w. Start w = -6, eta = 1. Each step is "
            "w <- w - eta g with g = 2 (p - 1) p (1 - p) for the squared error (p - y)^2 and g = p - 1 for the "
            "BCE; row k shows w after k steps.\n\n")
    return head + _table(["step k", "w (squared error)", "its loss", "w (BCE)", "its loss"], rows)


def gradient_check():
    x, y = _design(_parts())
    w = [-1.0, 0.5, 0.25]
    g = _gradient(x, y, w)
    h = 1e-6
    rows = []
    for j in range(3):
        wp, wm = list(w), list(w)
        wp[j] += h
        wm[j] -= h
        fd = (_loss(x, y, wp) - _loss(x, y, wm)) / (2 * h)
        rows.append((f"w{j}", f"{g[j]:.8f}", f"{fd:.8f}"))
    head = f"At w = {tuple(w)} on the sixty parts; finite differences with h = {h:g}.\n\n"
    return head + _table(["component", "(1/N) X^T (p - y)", "central difference of the loss"], rows)


def parts_table():
    rows = _parts()
    out = [_generator_line(PARTS_NAME),
           f"Draw order, part by part (N = {N_PARTS}): x1 = round(5 u1, 1), x2 = round(5 u2, 1), u3, each u = uniform(); "
           f"y = 1 (rejected) when u3 < sigmoid({PARTS_LAW[0]} + {PARTS_LAW[1]} x1 + {PARTS_LAW[2]} x2). "
           "First ten rows, then the totals.",
           ""]
    out.append(_table(["part", "x1 (0.01 mm)", "x2 (micrometres)", "y"],
                      [(i + 1, f"{r[0]:.1f}", f"{r[1]:.1f}", r[2]) for i, r in enumerate(rows[:10])]))
    n1 = sum(r[2] for r in rows)
    out.append("")
    out.append(f"Rejected parts (y = 1): {n1} of {len(rows)}; passed (y = 0): {len(rows) - n1}.")
    return "\n".join(out)


def training_run():
    w, steps, eta, lam_cap, x, y = _fit_parts()
    record = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
    record = tuple(k for k in record if k <= steps)
    _, seen = _descend(x, y, [0.0, 0.0, 0.0], eta, max(record), record)
    rows = []
    for k in record:
        wk, lk = seen[k]
        g = _gradient(x, y, wk)
        rows.append((k, f"{wk[0]:.4f}", f"{wk[1]:.4f}", f"{wk[2]:.4f}", f"{lk:.6f}", f"{math.sqrt(_dot(g, g)):.6f}"))
    g = _gradient(x, y, list(w))
    rows.append((f"{steps} (stop)", f"{w[0]:.4f}", f"{w[1]:.4f}", f"{w[2]:.4f}", f"{_loss(x, y, list(w)):.6f}",
                 f"{math.sqrt(_dot(g, g)):.2e}"))
    head = (f"Start w = (0, 0, 0). lambda_cap = lambda_max(X^T X / N) / 4 = {lam_cap:.6f}, learning rate eta = 1/lambda_cap = {eta:.6f}, "
            f"stop when the gradient norm is below {GRAD_TOL:g}.\n\n")
    return head + _table(["step k", "w0", "w1", "w2", "loss", "gradient norm"], rows)


def decision_boundary():
    w, _, _, _, x, y = _fit_parts()
    rows = []
    w = list(w)
    for (x1, x2) in ((1.0, 1.0), (2.0, 2.0), (2.5, 2.5), (3.0, 3.0), (4.0, 2.0), (0.5, 4.5)):
        z = w[0] + w[1] * x1 + w[2] * x2
        rows.append((f"({x1:.1f}, {x2:.1f})", f"{z:+.4f}", f"{sigmoid(z):.4f}", "reject" if z >= 0 else "pass"))
    x2_at = lambda x1: -(w[0] + w[1] * x1) / w[2]
    head = (f"Fitted w = ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}). The boundary w0 + w1 x1 + w2 x2 = 0 is the line "
            f"x2 = {-w[0] / w[2]:.4f} - {w[1] / w[2]:.4f} x1 (crossing x2 = 0 at x1 = {-w[0] / w[1]:.4f} "
            f"and x1 = 0 at x2 = {x2_at(0):.4f}).\n\n")
    return head + _table(["part (x1, x2)", "z = w.x", "p", "decision at 0.5"], rows)


def threshold_table():
    w, _, _, _, x, y = _fit_parts()
    p = _probs(x, list(w))
    n_pos = int(sum(y))
    rows = []
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        fp = sum(1 for pi, yi in zip(p, y) if pi >= t and yi == 0)
        fn = sum(1 for pi, yi in zip(p, y) if pi < t and yi == 1)
        rows.append((f"{t:.1f}", f"{logit(t):+.4f}", fp, fn, f"{(len(y) - fp - fn) / len(y):.4f}"))
    head = (f"The sixty training parts ({n_pos} rejected, {len(y) - n_pos} passed). A part is rejected when p >= t.\n\n")
    return head + _table(["threshold t", "logit(t)", "good parts rejected", "bad parts passed", "fraction correct"], rows)


def hessian_psd():
    w, _, _, lam_cap, x, y = _fit_parts()
    rows = []
    for name, ww in (("w = (0, 0, 0)", [0.0, 0.0, 0.0]), ("w = (-3, 0.5, 0.5)", [-3.0, 0.5, 0.5]),
                     ("fitted w", list(w)), ("w = (-30, 6, 6)", [-30.0, 6.0, 6.0])):
        s = [sigmoid(_dot(r, ww)) * (1 - sigmoid(_dot(r, ww))) for r in x]
        g = [[v / len(x) for v in r] for r in _gram(x, s)]
        ev = _eigenvalues(g)
        rows.append((name, _sci(ev[0], 2), _sci(ev[1], 2), _sci(ev[2], 2)))
    head = ("Hessian of the mean loss, H = (1/N) X^T S X with S = diag(p (1 - p)), on the sixty parts; eigenvalues by "
            "jacobi_eigen(H, 20). "
            f"Bound lambda_cap = {lam_cap:.6f}; fitted w = ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}), "
            "the weights at the stop of the run in \"Training by gradient descent\".\n\n")
    return head + _table(["where", "smallest eigenvalue", "middle", "largest"], rows)


def naive_vs_stable():
    """log(sigmoid(z)) and the loss of a y = 0 part, in 32-bit arithmetic, naive and stable."""
    getcontext().prec = 60

    def f32(v):
        if math.isinf(v) or math.isnan(v):
            return v
        try:
            return struct.unpack("f", struct.pack("f", v))[0]
        except OverflowError:
            return math.copysign(math.inf, v)

    def exp32(v):
        try:
            return f32(math.exp(v))
        except OverflowError:
            return math.inf

    def log32(v):
        if v == 0.0:
            return -math.inf
        return f32(math.log(v))

    def sig32(z):
        e = exp32(-z)
        den = f32(1.0 + e)
        return f32(1.0 / den) if not math.isinf(den) else 0.0

    def exact_logsig(z):
        # -log(1 + e^-z) in 60-digit decimal arithmetic
        return -((Decimal(1) + (Decimal(-z)).exp()).ln())

    rows = []
    for z in (-100, -90, -30, 0, 20, 30, 60):
        naive = log32(sig32(z))
        stable = f32(min(z, 0.0) - f32(math.log1p(exp32(-abs(z)))))
        ex = exact_logsig(z)
        rows.append((z, _sci(naive, 3), _sci(stable, 3), f"{float(ex):.6e}"))
    t1 = _table(["z", "naive log(sigmoid(z))", "stable -softplus(-z)", "exact"], rows)

    # the loss of a rejected-label-0 part with a confident wrong logit
    rows2 = []
    for z in (5, 15, 17, 20, 30, 100):
        p = sig32(z)
        one_minus = f32(1.0 - p)
        naive = -log32(one_minus)
        stable = f32(max(z, 0.0) + f32(math.log1p(exp32(-abs(z)))))
        rows2.append((z, f"{p:.9f}", _sci(one_minus, 3), _sci(naive, 3), _sci(stable, 3)))
    t2 = _table(["z", "p = sigmoid(z)", "1 - p", "naive -log(1 - p)", "stable softplus(z)"], rows2)
    return (t1 + "\n\nLoss of a part with label y = 0 whose logit is z (the true value is softplus(z)):\n\n" + t2)


def separable_growth():
    """Four parts that a threshold separates exactly: the weights never stop growing."""
    data = [(0.5, 0), (1.0, 0), (2.0, 1), (3.0, 1)]
    x = [[1.0, v] for v, _ in data]
    y = [float(t) for _, t in data]
    eta = 0.5
    record = (10, 100, 1000, 10_000, 100_000)
    w = [0.0, 0.0]
    rows = []
    for k in range(1, max(record) + 1):
        g = _gradient(x, y, w)
        w = [wi - eta * gi for wi, gi in zip(w, g)]
        if k in record:
            p = _probs(x, w)
            rows.append((k, f"{w[0]:.4f}", f"{w[1]:.4f}", f"{math.hypot(*w):.4f}", f"{-w[0] / w[1]:.4f}",
                         f"{_loss(x, y, w):.6f}", f"{p[1]:.4f}", f"{p[2]:.4f}"))
    head = ("Data (x, y): (0.5, 0), (1.0, 0), (2.0, 1), (3.0, 1). Start w = (0, 0), eta = 0.5, plain gradient steps.\n\n")
    return head + _table(["step k", "w0", "w1", "norm of w", "boundary -w0/w1", "loss", "p at x = 1", "p at x = 2"], rows)


def reading_the_fit():
    """What the fitted weights say: each weight multiplies the odds by exp(weight) per unit of its feature."""
    w, _, _, _, x, y = _fit_parts()
    rows = [("w1 (x1, diameter error)", f"{w[1]:.4f}", f"{math.exp(w[1]):.4f}"),
            ("w2 (x2, roughness)", f"{w[2]:.4f}", f"{math.exp(w[2]):.4f}")]
    z = w[0] + w[1] * 3.0 + w[2] * 3.0
    z1 = z + w[1]
    tail = (f"\nPart (3.0, 3.0): z = {z:.4f}, p = {sigmoid(z):.4f}, odds = {math.exp(z):.4f}. "
            f"One more unit of x1: z = {z1:.4f}, p = {sigmoid(z1):.4f}, odds = {math.exp(z1):.4f}, "
            f"the ratio of the odds is {math.exp(z1) / math.exp(z):.4f}.")
    return _table(["weight", "value", "odds multiplier exp(w)"], rows) + "\n" + tail


def margin_loss():
    """A logistic loss on a margin m: loss = softplus(-m) = -log sigmoid(m), gradient weight sigmoid(-m)."""
    beta_pref, count = 0.5, 8.0
    lp_chosen, lp_chosen_ref, lp_rejected, lp_rejected_ref = -2.0, -3.0, -4.0, -3.5
    m = beta_pref * ((lp_chosen - lp_chosen_ref) - (lp_rejected - lp_rejected_ref))
    rows = [("margin m = beta_pref ((lp_chosen - lp_chosen_ref) - (lp_rejected - lp_rejected_ref))", f"{m:.6f}"),
            ("loss softplus(-m)", f"{softplus(-m):.6f}"),
            ("same as -log(sigmoid(m))", f"{-math.log(sigmoid(m)):.6f}"),
            ("slope magnitude sigmoid(-m)", f"{sigmoid(-m):.6f}"),
            ("weight beta_pref sigmoid(-m) count", f"{beta_pref * sigmoid(-m) * count:.6f}"),
            ("loss at m = 0 (ln 2)", f"{softplus(0.0):.6f}")]
    head = (f"beta_pref = {beta_pref}, count = {count:g}, lp_chosen = {lp_chosen}, lp_chosen_ref = {lp_chosen_ref}, "
            f"lp_rejected = {lp_rejected}, lp_rejected_ref = {lp_rejected_ref}.\n\n")
    return head + _table(["quantity", "value"], rows, "lr")
