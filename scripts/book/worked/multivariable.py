# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Gradients, Jacobians and Many Variables".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: multivariable.<name> -->` markers. Helpers with a leading
underscore (or that take arguments) are shared with the chapter's figures, so
a picture and the table beside it cannot disagree.

The running example is a model with two knobs, w = (w1, w2), and two examples:
inputs (1, 1) and (1, 3) with targets 3 and 7. The loss is half the sum of the
squared errors, a quadratic bowl whose minimum, at w = (1, 2), has loss 0.

Swedish Embedded AB implements solutions for numerically trustworthy machine
learning code. If your team needs expertise in gradients, Jacobians and the
finite-difference checks that prove a backward pass correct, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math

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
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _vec(v, d=4):
    return "(" + ", ".join(_fx(a, d) for a in v) + ")"


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _matvec(m, v):
    return [_dot(row, v) for row in m]


def _matmul(a, b):
    cols = list(zip(*b))
    return [[_dot(row, col) for col in cols] for row in a]


def _transpose(m):
    return [list(c) for c in zip(*m)]


def _mat(m, d=4):
    return "[" + "; ".join(", ".join(_fx(a, d) for a in row) for row in m) + "]"


# ------------------------------------------- the two-knob loss (the bowl)

DATA_X = [[1.0, 1.0], [1.0, 3.0]]
DATA_T = [3.0, 7.0]
W_STAR = [1.0, 2.0]
HESSIAN = [[2.0, 4.0], [4.0, 10.0]]  # X^T X: the second derivatives of the loss
P_MAIN = [0.5, 1.5]  # the point the chapter examines most closely


def loss(w):
    """Half the sum of squared errors of the two examples."""
    return 0.5 * sum((_dot(x, w) - t) ** 2 for x, t in zip(DATA_X, DATA_T))


def grad(w):
    """The two partial derivatives of loss, by the formula sum_k r_k x_k."""
    r = [_dot(x, w) - t for x, t in zip(DATA_X, DATA_T)]
    return [sum(rk * x[j] for rk, x in zip(r, DATA_X)) for j in range(2)]


def central_partial(f, w, i, h=1e-5):
    """Central difference of f in coordinate i with the other coordinates frozen."""
    up, dn = list(w), list(w)
    up[i] += h
    dn[i] -= h
    return (f(up) - f(dn)) / (2.0 * h)


def central_gradient(f, w, h=1e-5):
    return [central_partial(f, w, i, h) for i in range(len(w))]


def central_directional(f, w, u, h=1e-5):
    up = [a + h * b for a, b in zip(w, u)]
    dn = [a - h * b for a, b in zip(w, u)]
    return (f(up) - f(dn)) / (2.0 * h)


def sym2_eigen(m):
    """Eigenvalues (largest first) and unit eigenvectors of a symmetric 2x2 matrix."""
    a, b, c = m[0][0], m[0][1], m[1][1]
    mid, rad = 0.5 * (a + c), math.hypot(0.5 * (a - c), b)
    pairs = []
    for lam in (mid + rad, mid - rad):
        v = [b, lam - a] if abs(b) > 1e-15 else ([1.0, 0.0] if abs(lam - a) < 1e-15 else [0.0, 1.0])
        n = _norm(v)
        pairs.append((lam, [v[0] / n, v[1] / n]))
    return pairs


def contour_points(level, n=181, center=None, hess=None):
    """Points of the contour {w : 1/2 (w - c)^T H (w - c) = level} of a quadratic bowl."""
    center = center or W_STAR
    (l1, e1), (l2, e2) = sym2_eigen(hess or HESSIAN)
    r = math.sqrt(2.0 * level)
    pts = []
    for k in range(n):
        th = 2.0 * math.pi * k / (n - 1)
        a, b = r * math.cos(th) / math.sqrt(l1), r * math.sin(th) / math.sqrt(l2)
        pts.append([center[0] + a * e1[0] + b * e2[0], center[1] + a * e1[1] + b * e2[1]])
    return pts


def descend(w0, eta, steps, gradient=grad):
    """The iterates w <- w - eta * gradient(w), starting with w0 itself."""
    path = [list(w0)]
    for _ in range(steps):
        g = gradient(path[-1])
        path.append([a - eta * b for a, b in zip(path[-1], g)])
    return path


# ---------------------------------------------------------- the problem

def loss_grid():
    w1s = [-1.0, 0.0, 1.0, 2.0, 3.0]
    w2s = [4.0, 3.0, 2.0, 1.0, 0.0]
    rows = [[_fx(w2, 1)] + [_fx(loss([w1, w2]), 1) for w1 in w1s] for w2 in w2s]
    return ("L(w1, w2) = 1/2 [(w1 + w2 - 3)^2 + (w1 + 3 w2 - 7)^2], w2 down the left, w1 across the top.\n\n"
            + _table(["w2 \\ w1"] + [_fx(a, 1) for a in w1s], rows))


# ------------------------------------------------------ partial derivatives

def freeze_and_nudge():
    p = list(P_MAIN)
    base = loss(p)
    rows = []
    for h in (0.1, 0.01, 0.001):
        for i, name in ((0, "w1"), (1, "w2")):
            q = list(p)
            q[i] += h
            rows.append([name, f"{h:g}", _fx(loss(q), 6), f"{loss(q) - base:+.6f}", f"{(loss(q) - base) / h:+.4f}"])
    return (f"At p = (0.5, 1.5): L = {_fx(base, 4)}. One knob nudged by h, the other frozen.\n\n"
            + _table(["knob nudged", "h", "L(p + h e)", "change in L", "change per unit of h"], rows))


def partial_formulas():
    p = list(P_MAIN)
    rA, rB = _dot(DATA_X[0], p) - 3.0, _dot(DATA_X[1], p) - 7.0
    w1_slice = 2 * p[0] - 4  # frozen w2 = 1.5: L(w1) = 1/2[(w1 - 1.5)^2 + (w1 - 2.5)^2]
    w2_slice = 10 * p[1] - 22  # frozen w1 = 0.5: L(w2) = 1/2[(w2 - 2.5)^2 + (3 w2 - 6.5)^2]
    return (f"Residuals at p = (0.5, 1.5): rA = {_fx(rA, 2)}, rB = {_fx(rB, 2)}.\n\n"
            + _table(["quantity", "value"], [
                ["slice with w2 frozen at 1.5: derivative 2 w1 - 4 at w1 = 0.5", _fx(w1_slice, 2)],
                ["slice with w1 frozen at 0.5: derivative 10 w2 - 22 at w2 = 1.5", _fx(w2_slice, 2)],
                ["rA + rB (first partial from the residuals)", _fx(rA + rB, 2)],
                ["rA + 3 rB (second partial from the residuals)", _fx(rA + 3 * rB, 2)],
            ]))


def gradient_check_points():
    """Central differences with h = 1e-5, one knob at a time; the last column is round-off."""
    rows = []
    for p in ([0.0, 0.0], [0.5, 1.5], [1.0, 2.0], [2.0, 0.0], [3.0, 4.0], [-1.0, 3.0]):
        g, n = grad(p), central_gradient(loss, p)
        rows.append([_vec(p, 1), _fx(loss(p), 3), _vec(g, 3), _vec(n, 3), _sci(max(abs(a - b) for a, b in zip(g, n)))])
    return ("Central differences with h = 1e-5, one knob at a time. The last column is the largest absolute "
            "difference of the two gradient entries; it is round-off, so its digits depend on h and are compared by order of magnitude.\n\n"
            + _table(["point w", "L(w)", "gradient (formula)", "central differences", "largest difference"], rows))


# ----------------------------------------------------------- direction

def directional_scan():
    p = list(P_MAIN)
    g = grad(p)
    gn = _norm(g)
    rows = []
    for deg in range(0, 360, 45):
        u = [math.cos(math.radians(deg)), math.sin(math.radians(deg))]
        rows.append([f"{deg} deg", _vec(u, 3), _fx(_dot(g, u), 4), _fx(central_directional(loss, p, u), 4)])
    special = [("along the gradient", [a / gn for a in g]),
               ("against the gradient", [-a / gn for a in g]),
               ("perpendicular to the gradient", [-g[1] / gn, g[0] / gn])]
    for name, u in special:
        rows.append([name, _vec(u, 3), _fx(_dot(g, u), 4), _fx(central_directional(loss, p, u), 4)])
    return (f"At p = (0.5, 1.5) the gradient is {_vec(g, 2)} with length {_fx(gn, 4)}. Angles are measured anticlockwise "
            "from the w1 axis; the perpendicular direction is (-g2, g1)/|g|, the gradient direction turned a quarter turn anticlockwise. "
            "Central difference along u: (L(p + h u) - L(p - h u)) / 2h with h = 1e-5.\n\n"
            + _table(["unit direction u", "u", "grad . u", "central difference along u"], rows))


def cauchy_schwarz_gradient():
    g = grad(P_MAIN)
    gn = _norm(g)
    name = "multivariable.cauchy_schwarz_gradient"
    rng = bookrng.generator(name)
    worst, best = 0.0, 0.0
    for _ in range(100000):
        th = 2.0 * math.pi * rng.uniform()
        d = _dot(g, [math.cos(th), math.sin(th)])
        worst, best = min(worst, d), max(best, d)
    return (f"Each direction is u = (cos t, sin t) with t = 2 pi times one uniform draw of the book's "
            f"generator (Volume I, chapter 17), seed {bookrng.seed_of(name)}; 100000 draws in order.\n\n"
            + _table(["quantity", "value"], [
        ["length of the gradient", _fx(gn, 4)],
        ["largest grad . u over 100000 random unit u", _fx(best, 4)],
        ["smallest grad . u over the same u", _fx(worst, 4)],
    ]))


def contour_tangent():
    p = list(P_MAIN)
    g = grad(p)
    gn = _norm(g)
    t = [-g[1] / gn, g[0] / gn]
    n = [g[0] / gn, g[1] / gn]
    rows = []
    for eps in (0.1, 0.01, 0.001):
        along_t = loss([p[0] + eps * t[0], p[1] + eps * t[1]]) - loss(p)
        along_n = loss([p[0] + eps * n[0], p[1] + eps * n[1]]) - loss(p)
        rows.append([f"{eps:g}", f"{along_t:+.3e}", f"{along_t / eps:+.4f}", f"{along_n:+.3e}", f"{along_n / eps:+.4f}"])
    return (f"At p = (0.5, 1.5), tangent direction {_vec(t, 4)} (perpendicular to the gradient), gradient direction {_vec(n, 4)}.\n\n"
            + _table(["step length", "change in L along tangent", "per unit step", "change in L along gradient", "per unit step"], rows))


# ---------------------------------------------------------- a first look

def first_steps():
    p0 = [0.0, 0.0]
    small = descend(p0, 0.05, 8)
    large = descend(p0, 0.18, 8)
    rows = []
    for k in range(9):
        rows.append([str(k), _vec(small[k], 4), _fx(loss(small[k]), 4), _vec(grad(small[k]), 3), _vec(large[k], 3), _fx(loss(large[k]), 2)])
    return ("Start at w = (0, 0). Each step moves w by -eta times the gradient.\n\n"
            + _table(["step", "w (eta = 0.05)", "L", "gradient there", "w (eta = 0.18)", "L"], rows))


def step_length_scan():
    p = list(P_MAIN)
    g = grad(p)
    curv = _dot(g, _matvec(HESSIAN, g))
    best = _dot(g, g) / curv
    rows = []
    for t in (0.0, 0.02, 0.05, best, 0.12, 0.17, 0.25):
        q = [a - t * b for a, b in zip(p, g)]
        rows.append([_fx(t, 4), _vec(q, 3), _fx(loss(q), 4)])
    return (f"From p = (0.5, 1.5) along the line w = p - t g, g = {_vec(g, 2)}.\n\n"
            + _table(["t", "w", "L(w)"], rows)
            + f"\n\nThe parabola L(p - t g) is lowest at t = (g . g) / (g^T H g) = {_fx(_dot(g, g), 4)} / {_fx(curv, 4)} = {_fx(best, 4)}.")


# ------------------------------------------------------------ Jacobians

def _f_2x2(x):
    return [x[0] ** 2 * x[1], 5.0 * x[0] + math.sin(x[1])]


def _jac_2x2(x):
    return [[2.0 * x[0] * x[1], x[0] ** 2], [5.0, math.cos(x[1])]]


def central_jacobian(f, x, h=1e-6):
    """Column j of the Jacobian is the central difference of f along coordinate j."""
    cols = []
    for j in range(len(x)):
        up, dn = list(x), list(x)
        up[j] += h
        dn[j] -= h
        cols.append([(a - b) / (2.0 * h) for a, b in zip(f(up), f(dn))])
    return _transpose(cols)


def jacobian_2x2():
    p = [1.0, 2.0]
    j = _jac_2x2(p)
    n = central_jacobian(_f_2x2, p)
    rows = [[f"row {i} (output {i})", _vec(j[i], 4), _vec(n[i], 4)] for i in range(2)]
    return (f"f(x, y) = (x^2 y, 5x + sin y) at (1, 2): f = {_vec(_f_2x2(p), 4)}. Central differences with h = 1e-6, one input at a time.\n\n"
            + _table(["Jacobian", "formula", "central differences"], rows, "lrr"))


def linear_approximation_error():
    p = [1.0, 2.0]
    j = _jac_2x2(p)
    d0 = [0.6, -0.8]
    rows = []
    for s in (1.0, 0.1, 0.01, 0.001):
        d = [s * a for a in d0]
        exact = [a - b for a, b in zip(_f_2x2([p[0] + d[0], p[1] + d[1]]), _f_2x2(p))]
        lin = _matvec(j, d)
        err = _norm([a - b for a, b in zip(exact, lin)])
        rows.append([f"{s:g}", _vec(exact, 5), _vec(lin, 5), _sci(err), _fx(err / s ** 2, 4)])
    return ("Step d = s (0.6, -0.8) from (1, 2), length s. Actual change of f against J d.\n\n"
            + _table(["s", "f(p + d) - f(p)", "J d", "length of the difference", "difference / s^2"], rows))


_W_LINEAR = [[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]]


def linear_layer_jacobian():
    x = [1.0, 2.0, -1.0]
    f = lambda v: _matvec(_W_LINEAR, v)
    n = central_jacobian(f, x, 1e-3)
    rows = [[f"row {i}", _vec(_W_LINEAR[i], 4), _vec(n[i], 4)] for i in range(2)]
    other = central_jacobian(f, [10.0, -4.0, 7.0], 1e-3)
    same = max(abs(a - b) for r1, r2 in zip(n, other) for a, b in zip(r1, r2))
    return (f"y = W x with W of shape [2, 3] at x = {_vec(x, 1)}: y = {_vec(f(x), 2)}. Central differences with h = 1e-3.\n\n"
            + _table(["Jacobian row", "row of W", "central differences"], rows, "lrr")
            + f"\n\nThe same central-difference Jacobian at x = (10, -4, 7) differs from the one above by at most {_sci(same)}.")


# -------------------------------------------------------- the chain rule

def chain_path_sum():
    # z = y1 * y2 with y1 = x^2 and y2 = sin x, at x = 1.2
    x = 1.2
    y1, y2 = x ** 2, math.sin(x)
    dz_dy1, dz_dy2 = y2, y1
    dy1, dy2 = 2 * x, math.cos(x)
    path1, path2 = dz_dy1 * dy1, dz_dy2 * dy2
    z = lambda v: (v ** 2) * math.sin(v)
    h = 1e-6
    fd = (z(x + h) - z(x - h)) / (2 * h)
    return _table(["path", "factors", "contribution"], [
        ["through y1 = x^2", f"{_fx(dz_dy1, 4)} * {_fx(dy1, 4)}", _fx(path1, 6)],
        ["through y2 = sin x", f"{_fx(dz_dy2, 4)} * {_fx(dy2, 4)}", _fx(path2, 6)],
        ["sum of the paths", "", _fx(path1 + path2, 6)],
        ["central difference of z(x) = x^2 sin x at 1.2", "", _fx(fd, 6)],
    ], "llr")


_W_CHAIN = [[0.5, -1.0], [1.0, 0.5], [-0.5, 2.0]]
_T_CHAIN = [0.2, -0.3, 0.4]


def _chain_forward(x):
    u = _matvec(_W_CHAIN, x)
    h = [math.tanh(a) for a in u]
    return u, h, 0.5 * sum((a - b) ** 2 for a, b in zip(h, _T_CHAIN))


def _chain_loss(x):
    return _chain_forward(x)[2]


def chain_jacobian_product():
    x = [0.6, -0.4]
    u, h, loss_v = _chain_forward(x)
    jl = [[a - b for a, b in zip(h, _T_CHAIN)]]  # [1, 3] row: dL/dh
    jt = [[(1 - h[i] ** 2) if i == j else 0.0 for j in range(3)] for i in range(3)]  # [3, 3]
    jw = _W_CHAIN  # [3, 2]
    step1 = _matmul(jl, jt)  # [1, 3]
    total = _matmul(step1, jw)  # [1, 2]
    n = central_gradient(_chain_loss, x)
    return (f"W = {_mat(_W_CHAIN, 1)}, target t = {_vec(_T_CHAIN, 1)}, L = (1/2)|h - t|^2. "
            f"x = {_vec(x, 1)}, u = W x = {_vec(u, 4)}, h = tanh(u) = {_vec(h, 4)}, L = {_fx(loss_v, 6)}. "
            "Central differences of L(x) with h = 1e-5.\n\n"
            + _table(["factor", "shape", "value"], [
                ["dL/dh = (h - t)^T", "[1, 3]", _mat(jl)],
                ["dh/du = diag(1 - h^2)", "[3, 3]", _mat(jt)],
                ["du/dx = W", "[3, 2]", _mat(jw)],
                ["(dL/dh)(dh/du)", "[1, 3]", _mat(step1)],
                ["(dL/dh)(dh/du)(du/dx)", "[1, 2]", _mat(total)],
                ["central differences of L(x)", "[2]", _vec(n, 4)],
            ], "llr"))


def chain_transposed():
    x = [0.6, -0.4]
    u, h, _ = _chain_forward(x)
    gh = [a - b for a, b in zip(h, _T_CHAIN)]
    gu = [(1 - hi ** 2) * g for hi, g in zip(h, gh)]
    gx = _matvec(_transpose(_W_CHAIN), gu)
    return _table(["step (backwards)", "formula", "result"], [
        ["gradient at h", "h - t", _vec(gh, 4)],
        ["gradient at u", "(1 - h^2) * (gradient at h), elementwise", _vec(gu, 4)],
        ["gradient at x", "W^T (gradient at u)", _vec(gx, 4)],
    ], "llr")


def chain_remainder():
    # g(f(x)) with f = _f_2x2 : R^2 -> R^2 and g(a, b) = a * b.  Error of J_g J_f d.
    p = [1.0, 2.0]
    g = lambda v: v[0] * v[1]
    comp = lambda x: g(_f_2x2(x))
    fp = _f_2x2(p)
    jg = [[fp[1], fp[0]]]
    jf = _jac_2x2(p)
    jc = _matmul(jg, jf)
    d0 = [0.6, -0.8]
    rows = []
    for s in (0.1, 0.01, 0.001):
        d = [s * a for a in d0]
        exact = comp([p[0] + d[0], p[1] + d[1]]) - comp(p)
        lin = _dot(jc[0], d)
        rows.append([f"{s:g}", _fx(exact, 6), _fx(lin, 6), _sci(abs(exact - lin)), _fx(abs(exact - lin) / s ** 2, 3)])
    return (f"g(a, b) = a b composed after f(x, y) = (x^2 y, 5x + sin y) at (1, 2). J_g = {_mat(jg)}, J_f = {_mat(jf)}, product {_mat(jc)}.\n\n"
            + _table(["s", "actual change", "(J_g J_f) d", "error", "error / s^2"], rows))


# ------------------------------------------------------ Taylor, one variable

def taylor_exp_partial_sums():
    rows = []
    s = 0.0
    term = 1.0
    for n in range(0, 9):
        if n > 0:
            term *= 1.0 / n
        s += term
        rows.append([str(n), f"1/{math.factorial(n)}", _fx(s, 8), _sci(math.e - s)])
    return ("Order-n Taylor polynomial of e^x at 0, evaluated at x = 1 (e = 2.71828183).\n\n"
            + _table(["order n", "coefficient of x^n", "value at x = 1", "e minus value"], rows))


def _taylor_exp(x, n):
    return sum(x ** k / math.factorial(k) for k in range(n + 1))


def taylor_error_order():
    rows = []
    prev = {}
    for h in (0.4, 0.2, 0.1, 0.05):
        row = [f"{h:g}"]
        for n in (1, 2, 3):
            err = math.exp(h) - _taylor_exp(h, n)
            ratio = "" if n not in prev else f" ({prev[n] / err:.1f}x)"
            row.append(_sci(err) + ratio)
            prev[n] = err
        rows.append(row)
    return ("Error of the order-n Taylor polynomial of e^x at 0, evaluated at x = h; in brackets, how much the error fell from the previous (twice as large) h.\n\n"
            + _table(["h", "order 1", "order 2", "order 3"], rows)
            + "\n\nPredicted fall factors: 4x, 8x, 16x for orders 1, 2, 3.")


def taylor_remainder_constant():
    rows = []
    for n in (1, 2, 3, 4):
        h = 1e-2
        err = math.exp(h) - _taylor_exp(h, n)
        rows.append([str(n), _sci(err), _sci(h ** (n + 1) / math.factorial(n + 1)), _fx(err / (h ** (n + 1) / math.factorial(n + 1)), 4)])
    return ("Error at h = 0.01 against the first dropped term h^(n+1) / (n+1)!.\n\n"
            + _table(["order n", "error", "h^(n+1) / (n+1)!", "ratio"], rows))


def taylor_explains_numdiff():
    h = 0.1
    f_exact = math.e
    fwd = (math.exp(1 + h) - math.exp(1)) / h
    err = fwd - f_exact
    t1 = 0.5 * math.e * h
    t2 = t1 + math.e * h * h / 6.0
    t3 = t2 + math.e * h ** 3 / 24.0
    return _table(["estimate of the forward-difference error at h = 0.1", "value"], [
        ["measured: (e^(1.1) - e) / 0.1 - e", _fx(err, 6)],
        ["first term f'' h / 2", _fx(t1, 6)],
        ["plus f''' h^2 / 6", _fx(t2, 6)],
        ["plus f'''' h^3 / 24", _fx(t3, 6)],
    ])


# ------------------------------------------------- Hessians, second order

def _f_saddle(x):
    return math.exp(x[0]) * math.cos(x[1])


def _hess_saddle(x):
    e, c, s = math.exp(x[0]), math.cos(x[1]), math.sin(x[1])
    return [[e * c, -e * s], [-e * s, -e * c]]


def central_hessian(f, x, h=1e-4):
    n = len(x)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            def at(si, sj):
                y = list(x)
                y[i] += si * h
                y[j] += sj * h
                return f(y)
            out[i][j] = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / (4.0 * h * h)
    return out


def hessian_example():
    rows = []
    for p in ([0.0, 0.0], [0.0, 1.0], [0.5, 2.0]):
        h = _hess_saddle(p)
        n = central_hessian(_f_saddle, p)
        rows.append([_vec(p, 1), _mat(h, 4), _mat(n, 4), _fx(h[0][1] - h[1][0], 1)])
    return ("f(x, y) = e^x cos y. Central second differences "
            "(f(+h,+h) - f(+h,-h) - f(-h,+h) + f(-h,-h)) / 4h^2 with h = 1e-4.\n\n"
            + _table(["point", "Hessian (formulas)", "central second differences", "H[0][1] - H[1][0]"], rows, "lrrr"))


def hessian_of_loss():
    (l1, e1), (l2, e2) = sym2_eigen(HESSIAN)
    rows = []
    for name, u in (("along w1", [1.0, 0.0]), ("along w2", [0.0, 1.0]),
                    ("along (1,1)/sqrt2", [math.sqrt(0.5), math.sqrt(0.5)]),
                    ("along (1,-1)/sqrt2", [math.sqrt(0.5), -math.sqrt(0.5)]),
                    ("steepest curvature direction", e1), ("flattest curvature direction", e2)):
        rows.append([name, _vec(u, 4), _fx(_dot(u, _matvec(HESSIAN, u)), 4)])
    tr, det = HESSIAN[0][0] + HESSIAN[1][1], HESSIAN[0][0] * HESSIAN[1][1] - HESSIAN[0][1] ** 2
    return (f"H = X^T X = {_mat(HESSIAN, 0)}, trace {_fx(tr, 0)}, determinant {_fx(det, 0)}. "
            f"The two numbers lambda with det(H - lambda I) = 0 are {_fx(l1, 4)} and {_fx(l2, 4)}; their ratio is {_fx(l1 / l2, 2)}.\n\n"
            + _table(["unit direction u", "u", "curvature u^T H u"], rows))


def quadratic_taylor_exact():
    p = list(P_MAIN)
    g = grad(p)
    rows = []
    for d in ([1.0, 0.0], [-0.4, 0.3], [2.0, -1.5]):
        exact = loss([p[0] + d[0], p[1] + d[1]])
        second = loss(p) + _dot(g, d) + 0.5 * _dot(d, _matvec(HESSIAN, d))
        first = loss(p) + _dot(g, d)
        rows.append([_vec(d, 1), _fx(exact, 6), _fx(first, 6), _fx(second, 6)])
    return (f"Expansion about p = (0.5, 1.5): L(p) = {_fx(loss(p), 4)}, g = {_vec(g, 2)}.\n\n"
            + _table(["step d", "L(p + d) exactly", "first order", "second order"], rows))


def taylor2_errors():
    p = [0.0, 0.0]
    g = [1.0, 0.0]
    hs = _hess_saddle(p)
    d0 = [0.5, 1.0]
    rows = []
    prev = {}
    for t in (0.4, 0.2, 0.1, 0.05):
        d = [t * a for a in d0]
        exact = _f_saddle([p[0] + d[0], p[1] + d[1]])
        f0 = _f_saddle(p)
        o1 = f0 + _dot(g, d)
        o2 = o1 + 0.5 * _dot(d, _matvec(hs, d))
        row = [f"{t:g}", _fx(exact, 6)]
        for k, approx in ((1, o1), (2, o2)):
            err = abs(exact - approx)
            ratio = "" if k not in prev else f" ({prev[k] / err:.1f}x)"
            row.append(_sci(err) + ratio)
            prev[k] = err
        rows.append(row)
    return ("f(x, y) = e^x cos y about the origin, step t (0.5, 1). Gradient (1, 0), Hessian diag(1, -1); "
            "second-order polynomial 1 + x + (x^2 - y^2)/2.\n\n"
            + _table(["t", "f exactly", "error of first order", "error of second order"], rows))


# ----------------------------------------------------- conditioning

def _equal_error_start():
    """w* plus one unit of error along each of the two curvature directions."""
    (_, e1), (_, e2) = sym2_eigen(HESSIAN)
    return [W_STAR[0] + e1[0] + e2[0], W_STAR[1] + e1[1] + e2[1]]


def _steps_to(path_fn, tol=1e-3, limit=100000):
    w = _equal_error_start()
    k = 0
    while k < limit:
        if _norm([a - b for a, b in zip(w, W_STAR)]) < tol:
            return k
        w = path_fn(w)
        k += 1
        if not all(math.isfinite(a) and abs(a) < 1e12 for a in w):
            return None
    return None


def step_size_limit():
    (l1, _), (l2, _) = sym2_eigen(HESSIAN)
    rows = []
    for eta in (0.02, 0.05, 0.1, 2.0 / (l1 + l2), 0.17, 0.18):
        factor = max(abs(1 - eta * l1), abs(1 - eta * l2))
        n = _steps_to(lambda w: [a - eta * b for a, b in zip(w, grad(w))])
        best = abs(eta - 2.0 / (l1 + l2)) < 1e-12
        rows.append(["1/6 (best, 0.1667)" if best else _fx(eta, 4), _fx(abs(1 - eta * l1), 4),
                     _fx(abs(1 - eta * l2), 4), _fx(factor, 4),
                     str(n) if n is not None else "never (diverges)"])
    return (f"Curvatures lambda1 = {_fx(l1, 4)}, lambda2 = {_fx(l2, 4)}; 2 / lambda1 = {_fx(2 / l1, 4)}; "
            "the step size 2/(lambda1 + lambda2) = 2/12 = 1/6 exactly (printed rounded). "
            "Start w = w* + v1 + v2 with v1, v2 the unit eigenvectors of H (distance sqrt 2 = 1.414); "
            "each step w <- w - eta grad L(w); steps = the first count n with distance |w_n - w*| < 0.001. "
            "'never' means the distance exceeds 10^12 or 100000 steps pass.\n\n"
            + _table(["step size eta", "|1 - eta lambda1|", "|1 - eta lambda2|", "worst factor per step", "steps needed"], rows))


ZIGZAG_START = [0.0, 3.5]
ZIGZAG_ETA = 0.15


def zigzag_path():
    path = descend(ZIGZAG_START, ZIGZAG_ETA, 40)
    rows = []
    for k in (0, 1, 2, 3, 4, 5, 10, 20, 30, 40):
        w = path[k]
        rows.append([str(k), _vec(w, 4), _fx(loss(w), 5), _fx(_norm([a - b for a, b in zip(w, W_STAR)]), 4)])
    return (f"Steps of size {ZIGZAG_ETA} from {_vec(ZIGZAG_START, 1)}.\n\n"
            + _table(["step", "w", "L(w)", "distance to the minimum (1, 2)"], rows))


def conditioning_compare():
    (l1, _), (l2, _) = sym2_eigen(HESSIAN)
    rows = []
    for name, lams in (("round bowl", (3.0, 3.0)), ("mild", (4.0, 1.0)), ("this chapter's bowl", (l1, l2)), ("severe", (10.0, 0.01))):
        a, b = lams
        eta = 2.0 / (a + b)
        factor = (a - b) / (a + b)
        # steps until the error shrinks by 1000x from a start with equal parts in both directions
        k, e1, e2 = 0, 1.0, 1.0
        while math.hypot(e1, e2) > 1e-3 and k < 10 ** 7:
            e1 *= 1 - eta * a
            e2 *= 1 - eta * b
            k += 1
        rows.append([name, f"{_fx(a, 3)} and {_fx(b, 3)}", _fx(a / b, 2), _fx(eta, 4), _fx(factor, 4), str(k)])
    return _table(["bowl", "curvatures", "ratio (condition number)", "best step 2/(l1 + l2)", "shrink factor (k-1)/(k+1)", "steps until distance < 0.001"], rows)


# ------------------------------------------------- checking gradients

def gradcheck_toy():
    """A missing example in the backward pass: ±1 directional check versus per-entry check."""
    atol, rtol, eps = 4e-3, 8e-2, 1e-4
    n, m = 24, 6
    trials = 1000
    dir_pass, entry_caught, frac_bad, rel_missing = 0, 0, 0.0, 0.0
    name = "multivariable.gradcheck_toy"
    rng = bookrng.generator(name)
    for trial in range(trials):
        X = [[rng.normal() for _ in range(n)] for _ in range(m)]
        t = [rng.normal() for _ in range(m)]
        w = [rng.normal() for _ in range(n)]

        def L(v):
            return 0.5 * sum((_dot(x, v) - tk) ** 2 for x, tk in zip(X, t))

        r = [_dot(x, w) - tk for x, tk in zip(X, t)]
        g_true = [sum(r[k] * X[k][j] for k in range(m)) for j in range(n)]
        g_bug = [sum(r[k] * X[k][j] for k in range(1, m)) for j in range(n)]  # example 0 never reaches the gradient
        rel_missing += _norm([a - b for a, b in zip(g_true, g_bug)]) / _norm(g_true)
        best = None
        for _ in range(4):
            v = [1.0 if rng.integer(2) else -1.0 for _ in range(n)]
            analytic = _dot(g_bug, v)
            numeric = (L([a + eps * b for a, b in zip(w, v)]) - L([a - eps * b for a, b in zip(w, v)])) / (2 * eps)
            err = abs(analytic - numeric)
            rel = err / max(abs(analytic), abs(numeric), 1e-3)
            ok = err <= atol + rtol * max(abs(analytic), abs(numeric))
            if best is None or rel < best[0]:
                best = (rel, ok)
        dir_pass += best[1]
        bad = 0
        for j in range(n):
            up, dn = list(w), list(w)
            up[j] += eps
            dn[j] -= eps
            numeric = (L(up) - L(dn)) / (2 * eps)
            if abs(g_bug[j] - numeric) > atol + rtol * max(abs(g_bug[j]), abs(numeric)):
                bad += 1
        entry_caught += bad > 0
        frac_bad += bad / n
    return ("One generator (Volume I, chapter 17), seed " + str(bookrng.seed_of(name)) + ", draws the "
            "1000 problems one after another: for each, X row by row, then t, then w, all with normal(), "
            "then the four directions, each entry +1 when integer(2) is 1 and -1 otherwise.\n\n"
            + _table(["check", "result over 1000 random problems"], [
        ["relative size of the missing part, mean of ||missing|| / ||gradient||", _fx(rel_missing / trials, 3)],
        ["best-of-4 random +-1 directions: bug accepted (check passes)", f"{dir_pass} of {trials}"],
        ["per-entry check: at least one entry fails", f"{entry_caught} of {trials}"],
        ["per-entry check: mean share of the 24 entries that fail", _fx(frac_bad / trials, 3)],
    ], "lr"))
