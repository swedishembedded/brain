# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Linear Regression and the First Descent".

Plain Python 3, standard library only, written from the chapter's own
description (loss, closed-form line, design matrix and normal equations,
elimination, gradient descent and its stability bound, feature scaling). Each
public function without a leading underscore that takes no arguments returns a
markdown string that the chapter embeds between
`<!-- worked: linear_regression.<name> -->` markers. Underscore functions are
shared with the chapter's figures, so a picture and the table beside it cannot
disagree.

The data are the ten sensor rows of the previous chapter (read from that
chapter's worked module, which plays the part of nature), so this chapter's
lines can be compared with the grid-search lines found there.

Swedish Embedded AB implements solutions for fitting models to sensor
measurements quickly and verifiably. If your team needs expertise in
regression, calibration and numerical optimisation, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache

import bookrng
import learning_problem as lp

SECOND_METER_NAME = "linear_regression.second_meter"
POWER_LAW_NAME = "linear_regression.power_law"

TOL = 1e-3          # "converged": the loss is within this of its minimum (squared degrees)
START = [30.0, 0.0]  # weights (w0, w1): the constant guess "30 degrees", slope zero
STEP_CAP = 200_000  # a run that has not converged by then is reported as such


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _sci(x, digits=2):
    """A number as 1.23e+05 style text, plain when it is of moderate size."""
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1:
        return f"{x:.4f}"
    if 1 <= abs(x) < 1e6:
        return f"{x:,.{digits}f}"
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m} x 10^{int(e)}"


def _rows():
    return lp._train()


def _xy():
    rows = _rows()
    return [v for v, _ in rows], [t for _, t in rows]


def _design(features, n_cols_fn):
    """Design matrix: one row per example, columns given by n_cols_fn(feature)."""
    return [n_cols_fn(f) for f in features]


def _line_design(vs):
    return [[1.0, v] for v in vs]


def _poly_design(vs, degree):
    return [[v ** k for k in range(degree + 1)] for v in vs]


def _transpose(a):
    return [list(col) for col in zip(*a)]


def _matmul(a, b):
    bt = _transpose(b)
    return [[sum(x * y for x, y in zip(r, c)) for c in bt] for r in a]


def _matvec(a, v):
    return [sum(x * y for x, y in zip(r, v)) for r in a]


def _gram(x):
    """X^T X for a design matrix x of shape [N, d]."""
    return _matmul(_transpose(x), x)


def _vanish_threshold(a):
    """A pivot counts as zero below this size: 1e-12 times the larger of 1 and the largest absolute entry of A."""
    return 1e-12 * max(1.0, max(abs(e) for r in a for e in r))


def _pivots(a, b):
    """Elimination with partial pivoting: the pivots (after each row swap) and the upper-triangular system [U | c]."""
    n = len(a)
    m = [list(r) + [bb] for r, bb in zip(a, b)]
    pivots = []
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        pivots.append(m[c][c])
        if abs(m[c][c]) < _vanish_threshold(a):
            break
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return pivots, m


def _solve(a, b):
    """Solve A w = b by Gaussian elimination with partial pivoting; None if a pivot vanishes."""
    n = len(a)
    pivots, m = _pivots(a, b)
    if len(pivots) < n or abs(pivots[-1]) < _vanish_threshold(a):
        return None
    w = [0.0] * n
    for r in range(n - 1, -1, -1):
        w[r] = (m[r][n] - sum(m[r][k] * w[k] for k in range(r + 1, n))) / m[r][r]
    return w


def _least_squares(x, y):
    """The weights solving the normal equations X^T X w = X^T y."""
    return _solve(_gram(x), _matvec(_transpose(x), y))


def _mse(x, y, w):
    return sum((p - t) ** 2 for p, t in zip(_matvec(x, w), y)) / len(y)


def _mae(x, y, w):
    return sum(abs(p - t) for p, t in zip(_matvec(x, w), y)) / len(y)


def _eigs(a):
    """Eigenvalues (descending) and eigenvectors (columns) of a symmetric matrix, by cyclic Jacobi rotations."""
    n = len(a)
    a = [list(r) for r in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(60):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-30 * max(1.0, sum(a[i][i] ** 2 for i in range(n))):
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                th = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = math.copysign(1.0, th) / (abs(th) + math.sqrt(th * th + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: -a[i][i])
    return [a[i][i] for i in order], [[v[k][i] for i in order] for k in range(n)]


def _hessian(x):
    """The Hessian of the MSE: (2/N) X^T X."""
    n = len(x)
    return [[2.0 * e / n for e in r] for r in _gram(x)]


def _zscore(vs):
    """(mean, standard deviation with divisor n - 1, z-scores) as in Volume I, chapter 12."""
    m = sum(vs) / len(vs)
    s = math.sqrt(sum((a - m) ** 2 for a in vs) / (len(vs) - 1))
    return m, s, [(a - m) / s for a in vs]


def _gradient(x, y, w):
    """The MSE gradient (2/N) X^T (X w - y)."""
    r = [p - t for p, t in zip(_matvec(x, w), y)]
    xt = _transpose(x)
    return [2.0 * sum(c * ri for c, ri in zip(col, r)) / len(y) for col in xt]


def _descend(x, y, w0, eta, steps):
    """Gradient descent: the list of the weight vectors w_0 .. w_steps."""
    path = [list(w0)]
    w = list(w0)
    for _ in range(steps):
        g = _gradient(x, y, w)
        w = [wi - eta * gi for wi, gi in zip(w, g)]
        path.append(w)
    return path


class _Quadratic:
    """The MSE of a linear model as a quadratic bowl, held by its moments so long runs cost O(d^2) a step."""

    def __init__(self, x, y):
        self.n = len(y)
        self.gram = _gram(x)
        self.xty = _matvec(_transpose(x), y)
        self.w_star = _solve(self.gram, self.xty)
        self.hess = [[2.0 * e / self.n for e in r] for r in self.gram]
        self.lams, self.vecs = _eigs(self.hess)
        self.d = len(self.xty)

    def gap(self, w):
        """Loss minus the minimum: (1/N) e^T X^T X e with e = w - w*."""
        e = [a - b for a, b in zip(w, self.w_star)]
        ge = _matvec(self.gram, e)
        return sum(a * b for a, b in zip(e, ge)) / self.n

    def grad(self, w):
        e = [a - b for a, b in zip(w, self.w_star)]
        return _matvec(self.hess, e)

    def dist(self, w):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(w, self.w_star)))

    def run(self, w0, eta, stop, cap=STEP_CAP):
        """Descend until `stop(k, w, gap, previous_gap, gradient_norm)` is true; returns (steps, w, diverged)."""
        w = list(w0)
        prev = self.gap(w)
        for k in range(cap + 1):
            g = self.grad(w)
            gn = math.sqrt(sum(a * a for a in g))
            cur = self.gap(w)
            if cur > 1e12:
                return k, w, True
            if stop(k, w, cur, prev, gn):
                return k, w, False
            prev = cur
            w = [wi - eta * gi for wi, gi in zip(w, g)]
        return cap, w, False

    @property
    def best_eta(self):
        return 2.0 / (self.lams[0] + self.lams[-1])

    @property
    def kappa(self):
        return self.lams[0] / self.lams[-1]


@lru_cache(maxsize=None)
def _raw():
    vs, ys = _xy()
    x = _line_design(vs)
    return x, ys, _Quadratic(x, ys)


@lru_cache(maxsize=None)
def _scaled():
    vs, ys = _xy()
    m, s, zs = _zscore(vs)
    x = _line_design(zs)
    return x, ys, _Quadratic(x, ys), (m, s)


def _steps_to_tol(q, w0, eta, tol=TOL):
    k, _, div = q.run(w0, eta, lambda k, w, g, p, gn: g < tol)
    return "diverges" if div else (f"> {STEP_CAP:,}" if k >= STEP_CAP else f"{k:,}")


# ------------------------------------------------------- the loss, defined

def rescored_attempts():
    vs, ys = _xy()
    x = _line_design(vs)
    w_ls, b_ls = _closed_form()[:2]
    w_grid, b_grid = lp._best_line(lp._train())[:2]
    cases = [(f"attempt {k + 1} (hand)", w, b) for k, (w, b) in enumerate(lp.ATTEMPTS)]
    cases += [("best of the grid", w_grid, b_grid), ("least squares", w_ls, b_ls)]
    body = []
    for name, w, b in cases:
        wv = [b, w]
        mse = _mse(x, ys, wv)
        body.append((name, f"{w:.3f}", f"{b:.3f}", f"{_mae(x, ys, wv):.2f}", f"{mse:.2f}", f"{math.sqrt(mse):.2f}"))
    return ("The same lines as the previous chapter, scored by the average absolute error E, by the mean squared "
            "error and by its square root (in degrees). Weights are shown to 3 decimals, scores to 2.\n\n"
            + _table(["line", "w", "b", "E (deg C)", "MSE (deg C^2)", "sqrt(MSE) (deg C)"], body))


def constant_model():
    _, ys = _xy()
    srt = sorted(ys)
    med = (srt[4] + srt[5]) / 2
    mean = sum(ys) / len(ys)
    cands = [("50.0", 50.0), (f"median {med:.2f}", med), (f"mean {mean:.2f}", mean), ("60.0", 60.0)]
    body = []
    for name, c in cands:
        body.append((name, f"{sum(abs(c - t) for t in ys) / len(ys):.3f}", f"{sum((c - t) ** 2 for t in ys) / len(ys):.3f}"))
    return ("A model with no feature at all: it answers one constant c for every voltage. Both losses over the ten "
            "labels of the sensor table. The median of ten labels is the mean of the 5th and 6th smallest; the "
            "candidates c are the median and the mean at full precision (shown to 2 decimals), and 50 and 60.\n\n"
            + _table(["c", "mean absolute error", "MSE"], body))


def mse_first_attempt():
    vs, ys = _xy()
    w, b = 24.0, 9.0
    body, tot = [], 0.0
    for v, t in zip(vs, ys):
        p = w * v + b
        r = p - t
        tot += r * r
        body.append((f"{v:.2f}", f"{t:.1f}", f"{p:.2f}", f"{r:+.2f}", f"{r * r:.4f}"))
    body.append(("", "", "", "total", f"{tot:.4f}"))
    body.append(("", "", "", f"MSE = total / {len(ys)}", f"{tot / len(ys):.4f}"))
    return "Attempt 1 (w = 24, b = 9) with squared errors, computed from the unrounded predictions.\n\n" + _table(
        ["v", "label y", "prediction", "prediction - y", "squared"], body)


# ---------------------------------------------- the line, exactly (1D case)

@lru_cache(maxsize=None)
def _closed_form():
    vs, ys = _xy()
    n = len(vs)
    mv, my = sum(vs) / n, sum(ys) / n
    sxx = sum((a - mv) ** 2 for a in vs)
    sxy = sum((a - mv) * (c - my) for a, c in zip(vs, ys))
    w = sxy / sxx
    b = my - w * mv
    return w, b, mv, my, sxx, sxy


def closed_form_line():
    vs, ys = _xy()
    w, b, mv, my, sxx, sxy = _closed_form()
    x = _line_design(vs)
    g = _gradient(x, ys, [b, w])
    rows = [("mean of v", f"{mv:.4f}"), ("mean of y", f"{my:.4f}"),
            ("S_vv = sum (v - mean v)^2", f"{sxx:.4f}"), ("S_vy = sum (v - mean v)(y - mean y)", f"{sxy:.4f}"),
            ("slope w = S_vy / S_vv", f"{w:.4f}"), ("intercept b = mean y - w mean v", f"{b:.4f}"),
            ("MSE at (w, b)", f"{_mse(x, ys, [b, w]):.4f}"),
            ("dMSE/dw at (w, b)", f"{g[1]:.1e}"), ("dMSE/db at (w, b)", f"{g[0]:.1e}")]
    return ("Gradient entries and the sums below are zero up to rounding noise: a reproduction agrees when their size is "
            "below 1e-9, whatever digits it prints.\n\n" + _table(["quantity", "value"], rows, "lr"))


def line_residuals():
    vs, ys = _xy()
    w, b = _closed_form()[:2]
    body = [(f"{v:.2f}", f"{t:.1f}", f"{w * v + b:.2f}", f"{(w * v + b) - t:+.2f}") for v, t in zip(vs, ys)]
    sr = sum((w * v + b) - t for v, t in zip(vs, ys))
    body.append(("", "", "sum of residuals", f"{sr:+.1e}"))
    return ("The least-squares line and its residuals r = prediction - label, the sign convention of the whole chapter. "
            "The sum of residuals is zero up to rounding noise (agreement: size below 1e-9, whatever digits it prints).\n\n"
            + _table(["v", "label y", "prediction", "residual (prediction - y)"], body))


# ------------------------------------------------ many features: the matrix

def design_matrix_system():
    vs, ys = _xy()
    x = _line_design(vs)
    xtx = _gram(x)
    xty = _matvec(_transpose(x), ys)
    out = [f"X has shape [{len(x)}, 2]: a column of ones and the column of voltages. First three rows: "
           + ", ".join(f"[{r[0]:.0f}, {r[1]:.2f}]" for r in x[:3]) + ".", ""]
    out.append("```text")
    out.append("X^T X  (shape [2, 2])                 X^T y  (shape [2])")
    out.append(f"[ {xtx[0][0]:9.4f}  {xtx[0][1]:9.4f} ]            [ {xty[0]:10.4f} ]")
    out.append(f"[ {xtx[1][0]:9.4f}  {xtx[1][1]:9.4f} ]            [ {xty[1]:10.4f} ]")
    out.append("```")
    f = xtx[1][0] / xtx[0][0]
    a11 = xtx[1][1] - f * xtx[0][1]
    r2 = xty[1] - f * xty[0]
    w1 = r2 / a11
    w0 = (xty[0] - xtx[0][1] * w1) / xtx[0][0]
    out += ["", "Elimination (Volume I, chapter 4):", "",
            "```text",
            f"R2 <- R2 - {f:.4f} R1     [ 0  {a11:.4f} | {r2:.4f} ]",
            f"back substitution: w1 = {r2:.4f} / {a11:.4f} = {w1:.4f}",
            f"                   w0 = ({xty[0]:.4f} - {xtx[0][1]:.4f} * {w1:.4f}) / {xtx[0][0]:.0f} = {w0:.4f}",
            "```", "",
            f"Closed form of the previous section: w = {_closed_form()[0]:.4f}, b = {_closed_form()[1]:.4f}."]
    return "\n".join(out)


def projection_check():
    vs, ys = _xy()
    x = _line_design(vs)
    w = _least_squares(x, ys)
    yh = _matvec(x, w)
    r = [p - t for t, p in zip(ys, yh)]
    xt = _transpose(x)
    rows = [("sum of residuals  (ones column . r)", f"{sum(r):+.1e}"),
            ("sum of v * residual  (voltage column . r)", f"{sum(a * b for a, b in zip(xt[1], r)):+.1e}"),
            ("y . y", f"{sum(t * t for t in ys):.4f}"),
            ("y_hat . y_hat", f"{sum(p * p for p in yh):.4f}"),
            ("r . r", f"{sum(e * e for e in r):.4f}"),
            ("y_hat . y_hat + r . r", f"{sum(p * p for p in yh) + sum(e * e for e in r):.4f}"),
            ("y_hat . r", f"{sum(p * e for p, e in zip(yh, r)):+.1e}")]
    return ("The fitted vector y_hat = X w and the residual r = X w - y (prediction minus label), as vectors in 10 dimensions. The two dot products "
            "with the columns of X and the last one are zero up to rounding noise (agreement: size below 1e-9, whatever digits it prints); "
            "the other rows are shown to 4 decimals.\n\n"
            + _table(["quantity", "value"], rows, "lr"))


# ----------------------------------------- when X^T X breaks: collinearity

METER_SD = 0.003


def _second_meter(vs):
    """The second meter's readings: for each voltage in table order, v + METER_SD * rng.normal(), rounded to 4 decimals."""
    rng = bookrng.generator(SECOND_METER_NAME)
    return [round(v + METER_SD * rng.normal(), 4) for v in vs]


def collinear_features():
    vs, ys = _xy()
    v2 = _second_meter(vs)
    out = []
    # exact copy: the same voltage in millivolts
    xe = [[1.0, v, 1000.0 * v] for v in vs]
    ge = _gram(xe)
    out.append("Case A: the third column is the voltage in millivolts, exactly 1000 v.")
    out.append("")
    wl = _closed_form()[0]
    pivots, _ = _pivots(ge, _matvec(_transpose(xe), ys))
    vanished = _solve(ge, _matvec(_transpose(xe), ys)) is None
    out.append(f"The millivolt column is computed as 1000.0 * v in floating point. Elimination with partial pivoting on "
               f"X^T X (the pivot of each column is the entry of largest size at or below the diagonal of the partly "
               f"eliminated matrix; threshold {_vanish_threshold(ge):.1e} = 1e-12 times the largest absolute entry of X^T X "
               f"as formed): the first two pivots are {pivots[0]:.4g} and {pivots[1]:.4g}, and the third is "
               f"{'rounding noise, below the threshold, so X^T X has no inverse' if vanished else 'above the threshold'}. "
               f"The rank is the number of eigenvalues of X^T X larger than 1e-9 times the largest eigenvalue: "
               f"rank {_rank(ge)} of 3. Every (w1, w2) with w1 + 1000 w2 = {wl:.4f} gives the same predictions, for example "
               f"({wl:.4f}, 0) and (0, {wl / 1000:.7f}).")
    out.append("")
    # near copy: a second meter with a small disagreement
    xn = [[1.0, a, c] for a, c in zip(vs, v2)]
    gn = _gram(xn)
    lam, _ = _eigs(gn)
    ys2 = list(ys)
    ys2[4] += 0.5
    w_a = _least_squares(xn, ys)
    w_b = _least_squares(xn, ys2)
    p = lambda w: w[0] + (w[1] + w[2]) * lp.ASK_V
    out.append(f"Case B: the third column is a second meter reading the same voltage, differing by a few millivolts. "
               f"{lp._generator_line(SECOND_METER_NAME)} For the ten rows in table order, one `rng.normal()` each, the "
               f"second meter reads v + {METER_SD} * rng.normal() rounded to 4 decimals. Labels of the second run: "
               f"row 5 raised by 0.5.")
    out.append("")
    out.append(_table(["row n"] + [str(i + 1) for i in range(len(vs))],
                      [("meter 1 (V)",) + tuple(f"{v:.2f}" for v in vs), ("meter 2 (V)",) + tuple(f"{v:.4f}" for v in v2)]))
    out.append("")
    out.append(_table(
        ["labels", "b", "w (meter 1)", "w' (meter 2)", f"prediction at {lp.ASK_V:.2f} V on both"],
        [("original", f"{w_a[0]:.2f}", f"{w_a[1]:.1f}", f"{w_a[2]:.1f}", f"{p(w_a):.2f}"),
         ("row 5 + 0.5", f"{w_b[0]:.2f}", f"{w_b[1]:.1f}", f"{w_b[2]:.1f}", f"{p(w_b):.2f}")]))
    out.append("")
    out.append(f"Eigenvalues of X^T X in case B: {', '.join(_sci(l, 3) for l in lam)}; condition number "
               f"{_sci(lam[0] / lam[-1], 3)}. Weights are shown to 1 decimal (2 for b); they are the solution of the "
               f"normal equations by elimination with partial pivoting and the same threshold, accurate to far more digits than "
               f"shown; the eigenvalues come from a symmetric eigensolver such as `jacobi_eigen`.")
    return "\n".join(out)


def _rank(a, tol=1e-9):
    lam, _ = _eigs(a)
    return sum(1 for l in lam if abs(l) > tol * abs(lam[0]))


def cost_comparison():
    cases = [("sensor line", 10, 2), ("house prices", 10**5, 20), ("tabular model", 10**6, 1_000),
             ("wide model", 10**6, 10_000), ("a 7B-parameter model", 10**9, 7 * 10**9)]
    body = []
    for name, n, d in cases:
        form = n * d * d
        solve = d ** 3 / 3
        step = 2 * n * d
        mem = 4.0 * d * d / 2 ** 30
        body.append((f"{name}", f"{n:,}", f"{d:,}", _sci(form), _sci(solve), _sci(step), _sci(mem)))
    return ("Multiply-adds for the normal equations (forming X^T X, then elimination) against one gradient-descent "
            "step, and the memory of X^T X in f32.\n\n"
            + _table(["case", "N", "d", "form X^T X: N d^2", "solve: d^3 / 3", "one descent step: 2 N d", "X^T X in GiB"], body))


# --------------------------------------------------------- gradient descent

def gradient_check():
    vs, ys = _xy()
    x = _line_design(vs)
    w0 = [9.0, 24.0]
    g = _gradient(x, ys, w0)
    h = 1e-5
    fd = []
    for k in range(2):
        up, dn = list(w0), list(w0)
        up[k] += h
        dn[k] -= h
        fd.append((_mse(x, ys, up) - _mse(x, ys, dn)) / (2 * h))
    names = ["dMSE/db (intercept)", "dMSE/dw (slope)"]
    body = [(names[k], f"{g[k]:.5f}", f"{fd[k]:.5f}") for k in range(2)]
    return ("At attempt 1 (b = 9, w = 24): the formula (2/N) X^T (X w - y) against central differences with step 1e-5.\n\n"
            + _table(["component", "formula", "finite difference"], body))


def curvature():
    x, ys, q = _raw()
    h = q.hess
    eta_max = 2.0 / q.lams[0]
    rows = [("Hessian (2/N) X^T X", f"[[{h[0][0]:.4f}, {h[0][1]:.4f}], [{h[1][0]:.4f}, {h[1][1]:.4f}]]"),
            ("eigenvalues", f"{q.lams[0]:.4f} and {q.lams[1]:.4f}"),
            ("condition number", f"{q.kappa:.2f}"),
            ("largest stable learning rate 2 / lambda_max", f"{eta_max:.4f}"),
            ("best learning rate eta* = 2 / (lambda_max + lambda_min)", f"{q.best_eta:.6f}"),
            ("loss at the start (w, b) = (0, 30) minus the minimum", f"{q.gap(START):.1f}")]
    return _table(["quantity", "value"], rows, "lr")


RATES = (("too small", 0.01), ("good", 0.20), ("too large", 0.24))


def _sig(x, digits):
    """x with `digits` significant digits: plain for moderate sizes, mantissa x 10^e otherwise."""
    if 1e-3 <= abs(x) < 1e6:
        return f"{x:.{digits}g}"
    m, e = f"{x:.{digits - 1}e}".split("e")
    return f"{m} x 10^{int(e)}"


def _wnum(v):
    """A weight to 2 decimals (thousands separated), or in 3-digit scientific form from 10^6 in size."""
    return _sci(v, 2) if abs(v) >= 1e6 else f"{v:,.2f}"


def gd_three_rates():
    x, ys, q = _raw()
    shown = (0, 1, 2, 5, 10, 20, 50, 100, 200)
    paths = {name: _descend(x, ys, START, eta, max(shown)) for name, eta in RATES}
    body = []
    for k in shown:
        row = [str(k)]
        for name, _ in RATES:
            wv = paths[name][k]
            row += [f"({_wnum(wv[1])}, {_wnum(wv[0])})", _sci(_mse(x, ys, wv), 2)]
        body.append(tuple(row))
    heads = ["step k"]
    for name, eta in RATES:
        heads += [f"(w, b), eta={eta}", "MSE"]
    return ("Gradient descent from (w, b) = (0, 30), the constant guess of 30 degrees, on the ten sensor rows, unscaled voltage. "
            "Row k shows the weights after k updates w <- w - eta * gradient (k = 0 is the start), with learning rates "
            "0.01, 0.2 and 0.24, and the MSE there. Weights and MSE values are shown to 2 decimals (thousands separated by commas), and numbers of "
            "10^6 or more in size as a mantissa with 3 significant digits times a power of ten. The eta = 0.24 "
            "column diverges (it grows by a factor of about 1.08 per step), so its late rows agree only in their leading digits. "
            f"The minimum is at ({q.w_star[1]:.2f}, {q.w_star[0]:.2f}) with MSE {_mse(x, ys, q.w_star):.4f}.\n\n"
            + _table(heads, body))


def stability_table():
    x, ys, q = _raw()
    lam1, lam2 = q.lams
    etas = [0.01, 0.05, 0.10, 0.20, q.best_eta, 0.23, 0.24]
    g0 = q.gap(START)
    body = []
    for eta in etas:
        f1, f2 = abs(1 - eta * lam1), abs(1 - eta * lam2)
        worst = max(f1, f2)
        if worst < 1:
            bound = math.ceil(math.log(TOL / g0) / (2 * math.log(worst)))
            bound_s = f"{bound:,}"
        else:
            bound_s = "none"
        label = f"{eta:.6f} (eta*)" if eta == q.best_eta else f"{eta:.4f}"
        body.append((label, f"{f1:.4f}", f"{f2:.4f}", f"{worst:.4f}", bound_s, _steps_to_tol(q, START, eta)))
    return (f"Curvatures lambda_max = {lam1:.4f}, lambda_min = {lam2:.4f}; 2 / lambda_max = {2 / lam1:.4f}. "
            f"Steps k: the smallest k with loss(w_k) - minimum < {TOL}, from (w, b) = (0, 30); a run whose loss "
            f"exceeds 1e12 above the minimum is reported as diverging, and one that needs more than {STEP_CAP:,} steps "
            f"as such. Bound: ceil(ln({TOL} / gap_0) / (2 ln(worst factor))) with gap_0 = {g0:.1f} the starting "
            f"loss minus the minimum. The row marked eta* uses the unrounded best rate eta* = 2 / (lambda_max + "
            f"lambda_min) = {q.best_eta:.6f}; every other row uses the rate as printed.\n\n"
            + _table(["learning rate", "|1 - eta lambda_max|", "|1 - eta lambda_min|", "worst factor",
                      "bound from the worst factor", "steps measured"], body))


def convergence_rate():
    x, ys, q = _raw()
    eta = 0.20
    worst = max(abs(1 - eta * l) for l in q.lams)
    path = _descend(x, ys, START, eta, 400)
    gaps = [q.gap(w) for w in path]
    body = [(str(k), f"{gaps[k]:.3e}", f"{gaps[k + 1] / gaps[k]:.4f}") for k in (0, 1, 2, 5, 10, 50, 100, 300)]
    return (f"Gradient descent at learning rate {eta} from (w, b) = (0, 30). With gap_k the loss at step k minus the "
            f"minimum, the table lists gap_k and gap_(k+1) / gap_k. The factors are |1 - eta lambda| = "
            f"{abs(1 - eta * q.lams[0]):.4f} and {abs(1 - eta * q.lams[1]):.4f}; the predicted late ratio is the square of the "
            f"larger, {worst ** 2:.4f}.\n\n"
            + _table(["step k", "loss - minimum", "ratio of the next gap to this one"], body))


def stopping_criteria():
    x, ys, q = _raw()
    lam_min = q.lams[-1]
    rows = []
    for eta in (0.20, 0.02, 0.002):
        ka, wa, _ = q.run(START, eta, lambda k, w, g, p, gn: k > 0 and abs(p - g) < 1e-6)
        kb, wb, _ = q.run(START, eta, lambda k, w, g, p, gn: gn < 1e-3)
        rows.append((f"{eta}", f"{ka:,}", _sci(q.gap(wa), 2), f"{q.dist(wa):.4f}",
                     f"{kb:,}", _sci(q.gap(wb), 2), f"{q.dist(wb):.4f}"))
    return (f"Two stopping rules from (w, b) = (0, 30). Loss-change rule: stop at the first k >= 1 for which "
            f"|loss(w_(k-1)) - loss(w_k)| < 1e-6. Gradient rule: stop at the first k >= 0 for which the Euclidean norm of "
            f"the gradient at w_k is below 1e-3. The step count is k, the number of updates made. For the second, the distance to the minimum is at most 1e-3 / lambda_min = "
            f"{1e-3 / lam_min:.4f}.\n\n"
            + _table(["learning rate", "loss-change rule: steps", "loss - minimum", "distance to minimum",
                      "gradient rule: steps", "loss - minimum", "distance to minimum"], rows))


# ------------------------------------------------------------ feature scaling

def scaling_conditioning():
    vs, ys = _xy()
    mv, sv, zs = _zscore(vs)
    variants = [("volts, raw", _line_design(vs)),
                ("millivolts, raw", _line_design([1000.0 * v for v in vs])),
                ("z-scored", _line_design(zs))]
    body = []
    for name, x in variants:
        q = _Quadratic(x, ys)
        eta = q.best_eta
        worst = (q.kappa - 1) / (q.kappa + 1)
        predicted = math.ceil(math.log(TOL / q.gap(START)) / (2 * math.log(worst)))
        measured = _steps_to_tol(q, START, eta)
        body.append((name, _sci(q.lams[0], 3), _sci(q.lams[-1], 3), _sci(q.kappa, 3), _sig(eta, 6), measured,
                     f"{predicted:,}" if predicted < 10**6 else _sci(predicted, 2)))
    return (f"Same ten rows, same model, three ways of writing the voltage (z-score z = (v - mean) / s with mean "
            f"{mv:.3f} and standard deviation s = {sv:.3f}, divisor N - 1 = 9; millivolts are 1000 v). Best learning rate "
            f"eta* = 2 / (lambda_max + lambda_min), used unrounded (shown to 6 significant digits), and steps k, the smallest k with loss(w_k) - minimum < {TOL}, from (w, b) = "
            f"(0, 30) in the units of the column (for z-scores, the weights of the column of ones and of z are "
            f"started at 30 and 0); a run is cut off at {STEP_CAP:,} steps. The predicted bound is "
            f"ceil(ln({TOL} / gap_0) / (2 ln((kappa - 1) / (kappa + 1)))), shown with 3 significant digits when it is 10^6 or more.\n\n"
            + _table(["feature", "lambda_max", "lambda_min", "condition number", "best eta", "steps measured",
                      "steps predicted (bound)"], body))


def scaled_fit_matches():
    vs, ys = _xy()
    x, y, q, (m, s) = _scaled()
    w0, w1 = q.w_star
    w_raw = w1 / s
    b_raw = w0 - w1 * m / s
    wr, br = _closed_form()[:2]
    return _table(["fit", "slope", "intercept", f"prediction at {lp.ASK_V:.2f} V"],
                  [("z-scored: (w1, w0) = " + f"({w1:.4f}, {w0:.4f})", "", "", f"{w0 + w1 * (lp.ASK_V - m) / s:.4f}"),
                   ("mapped back: w1 / s, w0 - w1 m / s", f"{w_raw:.4f}", f"{b_raw:.4f}", f"{w_raw * lp.ASK_V + b_raw:.4f}"),
                   ("raw volts, closed form", f"{wr:.4f}", f"{br:.4f}", f"{wr * lp.ASK_V + br:.4f}")], "lrrr")


# ---------------------------------------------------- linear in disguise

def polynomial_fit():
    vs, ys = _xy()
    fresh = lp._fresh()
    fv, fy = [r[0] for r in fresh], [r[1] for r in fresh]
    body = []
    for name, deg in (("line", 1), ("parabola", 2), ("cubic", 3)):
        x = _poly_design(vs, deg)
        w = _least_squares(x, ys)
        xf = _poly_design(fv, deg)
        cond_raw = _cond(_gram(x))
        _, _, zs = _zscore(vs)
        m, s, _ = _zscore(vs)
        cond_z = _cond(_gram(_poly_design(zs, deg)))
        body.append((name, "(" + ", ".join(f"{c:.2f}" for c in w) + ")", f"{_mse(x, ys, w):.3f}",
                     f"{_mse(xf, fy, w):.3f}", _sci(cond_raw, 2), _sci(cond_z, 2)))
    return ("Polynomial features of the voltage (columns 1, v, v^2, ...), fitted by the normal equations on the ten rows. "
            "Weights are (w0, w1, ...) from the constant term up. The fresh rows are the 2000 rows of the source of the previous chapter, "
            f"drawn from the book's SplitMix64 with seed {bookrng.seed_of(lp.FRESH_NAME):#018x} ({bookrng.seed_of(lp.FRESH_NAME)}). Each row is made by two draws in this order: v = {lp.V_LO} + {lp.V_HI - lp.V_LO:.1f} * "
            f"rng.uniform(), then z = rng.normal(); y = {lp.LAW[0]:g} + {lp.LAW[1]:g} v - {-lp.LAW[2]:g} v^2 + {lp.NOISE_SD} z "
            "computed from the unrounded v; then v is rounded to 2 decimals and y to 1. The fits are not repeated on them: the "
            "weights found on the ten rows are scored on the fresh rows. MSE is shown to 3 decimals. cond(X^T X) is the largest "
            "eigenvalue over the smallest; the last column builds the columns 1, z, z^2, ... from the z-score z = (v - mean) / s "
            "of v (mean and standard deviation s of the ten voltages, divisor 9) instead of from v.\n\n"
            + _table(["model", "weights", "MSE, 10 known rows", "MSE, 2000 fresh rows",
                      "cond(X^T X), raw v", "cond(X^T X), z-scored v"], body))


def _cond(a):
    lam, _ = _eigs(a)
    return lam[0] / lam[-1]


def power_law_log_fit():
    rng = bookrng.generator(POWER_LAW_NAME)
    floor_true, scale_true, alpha_true = 0.5, 10.0, 0.3
    sizes = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    losses = [(floor_true + scale_true * n ** (-alpha_true)) * (1 + 0.01 * rng.normal()) for n in sizes]
    out = []
    cands = []
    for e in (0.0, 0.25, 0.5, 0.6):
        xs = [math.log(n) for n in sizes]
        ys = [math.log(l - e) for l in losses]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((a - mx) ** 2 for a in xs)
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        slope = sxy / sxx
        icpt = my - slope * mx
        ss_res = sum((b - (icpt + slope * a)) ** 2 for a, b in zip(xs, ys))
        ss_tot = sum((b - my) ** 2 for b in ys)
        cands.append((e, -slope, math.exp(icpt), 1 - ss_res / ss_tot))
    body1 = [(f"{n:,.0f}", f"{l:.4f}") for n, l in zip(sizes, losses)]
    out.append(f"Seven (size n_p, loss) points generated from loss = 0.5 + 10 n_p^-0.3 with multiplicative noise of one percent. "
               f"{lp._generator_line(POWER_LAW_NAME)} For the sizes in the order listed, one `rng.normal()` each, the loss "
               f"is (0.5 + 10 n_p^-0.3) * (1 + 0.01 * rng.normal()), shown to 4 decimals.")
    out.append("")
    out.append(_table(["n_p", "loss"], body1))
    out.append("")
    out.append("For each candidate floor, an ordinary least-squares line (closed form of the chapter, natural logarithms, "
               "unrounded losses) of log(loss - floor) against log n_p (slope = -alpha, intercept = log zeta, so the scale is "
               "zeta = exp(intercept)); R^2 = 1 - SS_res / SS_tot of that line, with SS_res the sum of the squared "
               "differences between the seven values of log(loss - floor) and the line, and SS_tot the sum of squares of "
               "log(loss - floor) about its mean:")
    out.append("")
    out.append(_table(["candidate floor", "alpha", "scale zeta", "R^2 of the log-space fit"],
                      [(f"{e:.2f}", f"{al:.4f}", f"{a:.3f}", f"{r2:.5f}") for e, al, a, r2 in cands]))
    return "\n".join(out)
