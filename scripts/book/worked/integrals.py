# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Integrals and Differential Equations".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: integrals.<name> -->` markers. Helpers with a leading underscore
(or that take arguments) are shared with the chapter's figures, so a picture and
the table beside it cannot disagree.

Swedish Embedded AB implements solutions for numerically trustworthy machine
learning code. If your team needs expertise in numerical integration, ODE
solvers and step-size stability, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math


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
    v = v + 0.0  # turns -0.0 into 0.0
    return f"{v:.{d}f}"


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _ratio(prev, cur):
    return f" ({prev / cur:.1f}x)" if prev is not None and cur != 0 else ""


# ------------------------------------------------ the functions of the text

def speed(t):
    """The speed, in metres per second, of the opening problem: t (4 - t)."""
    return t * (4.0 - t)


SPEED_AREA_0_3 = 9.0  # the exact distance in the first 3 seconds, 2 t^2 - t^3 / 3 at t = 3


def rule_sum(f, a, b, n, rule):
    """Approximate the area under f on [a, b] with n panels of equal width.

    rule: 'left', 'right', 'mid', 'trap' or 'simpson' (n must be even).
    """
    h = (b - a) / n
    if rule == "left":
        return h * sum(f(a + i * h) for i in range(n))
    if rule == "right":
        return h * sum(f(a + (i + 1) * h) for i in range(n))
    if rule == "mid":
        return h * sum(f(a + (i + 0.5) * h) for i in range(n))
    if rule == "trap":
        return h * (0.5 * f(a) + sum(f(a + i * h) for i in range(1, n)) + 0.5 * f(b))
    if rule == "simpson":
        if n % 2:
            raise ValueError("Simpson's rule needs an even number of panels")
        total = f(a) + f(b)
        total += 4.0 * sum(f(a + i * h) for i in range(1, n, 2))
        total += 2.0 * sum(f(a + i * h) for i in range(2, n, 2))
        return total * h / 3.0
    raise ValueError(rule)


def integrate(f, a, b, n=2000):
    """The chapter's reference integral: Simpson's rule with many panels."""
    return rule_sum(f, a, b, n, "simpson")


# ------------------------------------------------------- riemann sums

def speed_readings():
    rows = [(t, _fx(speed(t), 1)) for t in range(0, 4)]
    left = sum(speed(t) for t in range(0, 3))
    right = sum(speed(t) for t in range(1, 4))
    return (_table(["time t (s)", "speed v(t) (m/s)"], rows)
            + f"\n\nLeft readings (t = 0, 1, 2): {left:.1f} m. Right readings (t = 1, 2, 3): {right:.1f} m.")


def riemann_shrinking():
    rows = []
    for n in (3, 6, 12, 30, 300, 3000):
        h = 3.0 / n
        left = rule_sum(speed, 0.0, 3.0, n, "left")
        right = rule_sum(speed, 0.0, 3.0, n, "right")
        rows.append((n, f"{h:.4f}", f"{left:.4f}", f"{right:.4f}", f"{right - left:.4f}",
                     f"{(left + right) / 2:.4f}"))
    return ("Distance travelled in 3 seconds from n readings of v(t) = t (4 - t).\n\n"
            + _table(["readings n", "width h", "left sum", "right sum", "right minus left", "average"], rows))


def signed_area():
    f = lambda t: t - 2.0  # a velocity: backwards for t < 2, forwards after
    parts = [("0 to 2", 0.0, 2.0), ("2 to 4", 2.0, 4.0), ("0 to 4", 0.0, 4.0)]
    rows = [(name, _fx(integrate(f, a, b), 4)) for name, a, b in parts]
    travelled = integrate(lambda t: abs(f(t)), 0.0, 4.0, 2000)
    return ("Integral of v(t) = t - 2 (a velocity) over each interval.\n\n"
            + _table(["interval (s)", "integral (m, signed)"], rows)
            + f"\n\nIntegral of |v(t)| over 0 to 4: {travelled:.4f} m (the distance actually driven).")


# ----------------------------------------------- fundamental theorem

def strip_over_width():
    f = speed
    x = 1.0
    rows = []
    for h in (1.0, 0.5, 0.1, 0.01, 0.001):
        strip = integrate(f, x, x + h, 200)
        rows.append((h, _fx(strip, 6), _fx(strip / h, 6)))
    return (f"A(x) = area under v from 0 to x. Strip beside x = {x:g}, v({x:g}) = {f(x):g}.\n\n"
            + _table(["strip width h", "A(x + h) - A(x)", "(A(x + h) - A(x)) / h"], rows))


def antiderivative_table():
    cases = [
        ("x^2", lambda x: x * x, lambda x: x ** 3 / 3.0, 0.0, 2.0, "x^3 / 3"),
        ("x^3", lambda x: x ** 3, lambda x: x ** 4 / 4.0, 0.0, 2.0, "x^4 / 4"),
        ("3 x^2 + 2 x", lambda x: 3 * x * x + 2 * x, lambda x: x ** 3 + x * x, 0.0, 2.0, "x^3 + x^2"),
        ("e^x", math.exp, math.exp, 0.0, 2.0, "e^x"),
        ("e^(-x)", lambda x: math.exp(-x), lambda x: -math.exp(-x), 0.0, 2.0, "-e^(-x)"),
        ("1 / x", lambda x: 1.0 / x, math.log, 1.0, 2.0, "log x"),
    ]
    rows = []
    for name, f, F, a, b, fname in cases:
        exact = F(b) - F(a)
        num = integrate(f, a, b)
        rows.append((name, f"{a:g} to {b:g}", fname, _fx(exact, 6), _fx(num, 6), _sci(abs(exact - num))))
    return _table(["f(x)", "interval", "antiderivative F", "F(b) - F(a)", "area by Simpson, 2000 panels",
                   "abs difference"], rows)


def taylor_remainder_exact():
    n = 2
    rows = []
    for h in (1.0, 0.5, 0.25, 0.1):
        rem = math.exp(h) - sum(h ** k / math.factorial(k) for k in range(n + 1))
        kern = integrate(lambda s: math.exp(s) * (h - s) ** n / math.factorial(n), 0.0, h, 400)
        bound = math.exp(h) * h ** (n + 1) / math.factorial(n + 1)
        rows.append((h, _sci(rem, 6), _sci(kern, 6), _sci(bound, 6), f"{rem / bound:.3f}"))
    return ("Remainder of the order-2 Taylor polynomial of e^x at 0, evaluated at x = h.\n\n"
            + _table(["h", "remainder e^h - (1 + h + h^2/2)", "integral of e^s (h - s)^2 / 2 over 0 to h",
                      "bound M h^3 / 3!, M = e^h", "remainder / bound"], rows))


def normalising_constant():
    f = lambda x: math.exp(-0.5 * x * x)
    target = math.sqrt(2.0 * math.pi)
    rows = []
    for half in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        area = integrate(f, -half, half, 4000)
        rows.append((f"-{half:g} to {half:g}", _fx(area, 6), _sci(target - area)))
    return (f"Area under exp(-x^2 / 2). The limit is sqrt(2 pi) = {target:.6f}.\n\n"
            + _table(["interval", "area (Simpson, 4000 panels)", "sqrt(2 pi) minus area"], rows))


# ----------------------------------------------- numerical integration

def rules_error_order():
    exact = math.e - 1.0
    f = math.exp
    rows = []
    prev = {}
    for n in (2, 4, 8, 16, 32, 64):
        cells = [n]
        for rule in ("left", "trap", "simpson"):
            err = abs(rule_sum(f, 0.0, 1.0, n, rule) - exact)
            cells.append(_sci(err) + _ratio(prev.get(rule), err))
            prev[rule] = err
        rows.append(tuple(cells))
    return (f"Error of each rule for the area under e^x on [0, 1] (exact value e - 1 = {exact:.6f}); in brackets, "
            "how much the error fell when the number of panels n doubled.\n\n"
            + _table(["panels n", "left rectangles", "trapezoids", "Simpson"], rows)
            + "\n\nPredicted fall factors: 2x, 4x, 16x.")


def rules_polynomials():
    rows = []
    for k in range(5):
        exact = 1.0 / (k + 1)
        f = lambda x, k=k: x ** k
        cells = [f"x^{k}"]
        for rule in ("left", "trap", "simpson"):
            err = rule_sum(f, 0.0, 1.0, 2, rule) - exact
            cells.append(_sci(err) if abs(err) > 1e-15 else "0 (exact)")
        rows.append(tuple(cells))
    return ("Error (rule minus exact) of the area under x^k on [0, 1] with two panels.\n\n"
            + _table(["f(x)", "left rectangles", "trapezoids", "Simpson"], rows))


def evaluations_needed():
    exact = math.e - 1.0
    rows = []
    for rule, label in (("left", "left rectangles"), ("trap", "trapezoids"), ("simpson", "Simpson")):
        n = 2
        while abs(rule_sum(math.exp, 0.0, 1.0, n, rule) - exact) >= 1e-6:
            n *= 2
        err = abs(rule_sum(math.exp, 0.0, 1.0, n, rule) - exact)
        evals = n if rule == "left" else n + 1
        rows.append((label, n, evals, _sci(err)))
    return ("Smallest power-of-two number of panels that brings the error on e^x over [0, 1] below 1e-6.\n\n"
            + _table(["rule", "panels n", "function evaluations", "error"], rows))


# ------------------------------------------------------------- the ODE

def compounding_is_euler():
    r, t, balance = 0.05, 1.0, 100.0
    rows = []
    for n in (1, 2, 12, 365, 1_000_000):
        h = t / n
        y = balance
        for _ in range(n):
            y += h * (r * y)  # one Euler step of dy/dt = r y
        rows.append((n, f"{h:.6f}", _fx(y, 4), _fx(balance * (1 + r / n) ** n, 4)))
    exact = balance * math.exp(r * t)
    return (f"Balance 100, rate 5 % a year, one year. The ODE dy/dt = 0.05 y has the exact answer "
            f"100 e^0.05 = {exact:.4f}.\n\n"
            + _table(["Euler steps n", "step h (years)", "Euler result", "100 (1 + 0.05 / n)^n"], rows))


def euler(f, t0, x0, h, n):
    """Euler's method for dx/dt = f(t, x): returns the list of (t, x) for steps 0..n."""
    t, x = t0, x0
    out = [(t, x)]
    for i in range(n):
        x = x + h * f(t, x)
        t = t0 + (i + 1) * h
        out.append((t, x))
    return out


def euler_by_hand():
    h = 0.5
    path = euler(lambda t, x: -x, 0.0, 1.0, h, 4)
    rows = []
    for i, (t, x) in enumerate(path):
        exact = math.exp(-t)
        rows.append((i, _fx(t, 2), _fx(x, 4), _fx(-x, 4) if i < 4 else "", _fx(exact, 4), _sci(exact - x)))
    return ("Euler's method for dx/dt = -x, x(0) = 1, step h = 0.5.\n\n"
            + _table(["step i", "t", "x_i", "slope -x_i", "exact e^(-t)", "exact minus Euler"], rows))


def euler_error_order():
    exact = math.exp(-2.0)
    rows = []
    prev = None
    for h in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        n = round(2.0 / h)
        x = euler(lambda t, x: -x, 0.0, 1.0, h, n)[-1][1]
        err = exact - x
        local = math.exp(-h) - (1.0 - h)  # one step from the exact value at t = 0
        rows.append((h, n, _sci(local), _sci(h * h / 2), _fx(x, 6), _sci(err) + _ratio(prev, err),
                     f"{err / h:.4f}"))
        prev = err
    return (f"Euler on dx/dt = -x from x(0) = 1 up to t = 2 (exact e^-2 = {exact:.6f}). Local error: one step from "
            "the exact value at t = 0.\n\n"
            + _table(["step h", "steps n", "local error", "h^2 / 2", "Euler x(2)", "global error at t = 2",
                      "global error / h"], rows)
            + f"\n\nPredicted limit of the last column: t e^(-t) / 2 = {exact:.4f}.")


def euler_stability():
    rows = []
    for h in (0.5, 1.0, 1.5, 2.0, 2.5):
        factor = 1.0 - h
        x = 1.0
        for _ in range(10):
            x *= factor
        if factor == 0:
            kind = "lands on 0 in one step"
        elif 0 < factor < 1:
            kind = "decays, same sign as the true solution"
        elif -1 < factor < 0:
            kind = "decays, sign flips each step"
        elif factor == -1:
            kind = "never decays, sign flips each step"
        else:
            kind = "grows, sign flips each step"
        rows.append((h, f"{factor:+.2f}", f"{x:+.4f}", kind))
    return ("Euler on dx/dt = -x: each step multiplies x by 1 - h. Value after 10 steps from x(0) = 1.\n\n"
            + _table(["step h", "factor 1 - h", "x after 10 steps", "behaviour"], rows, align="lrrl"))


def gd_is_euler():
    lam = 3.0
    rows = []
    cols = {}
    for eta in (0.2, 0.7):
        w = 1.0
        gd = [w]
        for _ in range(4):
            w = w - eta * (lam * w)  # gradient descent on L(w) = lam w^2 / 2
            gd.append(w)
        cols[eta] = gd
    for n in range(5):
        row = [n]
        for eta in (0.2, 0.7):
            row.append(_fx(cols[eta][n], 4))
            row.append(_fx(math.exp(-lam * eta * n), 4))
        rows.append(tuple(row))
    return ("L(w) = 3 w^2 / 2, gradient 3 w, start w = 1. Gradient descent iterates beside the exact flow "
            "w(t) = e^(-3 t) at t = n eta.\n\n"
            + _table(["step n", "descent, eta = 0.2", "flow at t = 0.2 n", "descent, eta = 0.7", "flow at t = 0.7 n"],
                     rows)
            + f"\n\nThe stability limit 2 / lambda = {2.0 / lam:.4f}.")


# ---------------------------------------------------- several variables

def rotation_field():
    rows = []
    for n in (20, 40, 100, 1000):
        h = 2.0 * math.pi / n
        x = [1.0, 0.0]
        y = [1.0, 0.0]
        for _ in range(n):
            x = [x[0] - h * x[1], x[1] + h * x[0]]  # Euler
            k1 = [-y[1], y[0]]
            mid = [y[0] + h * k1[0], y[1] + h * k1[1]]
            k2 = [-mid[1], mid[0]]
            y = [y[0] + h * (k1[0] + k2[0]) / 2, y[1] + h * (k1[1] + k2[1]) / 2]  # Heun
        rows.append((n, f"{h:.4f}", f"{math.hypot(*x):.6f}", f"{(1 + h * h) ** (n / 2):.6f}",
                     f"{math.hypot(*y):.6f}"))
    return ("The rotation field (x, y)' = (-y, x) from (1, 0) for one full turn (t = 2 pi), n steps. "
            "The true radius stays 1.\n\n"
            + _table(["steps n", "step h", "Euler radius", "(1 + h^2)^(n/2)", "Heun radius"], rows))


_H = [[2.0, 4.0], [4.0, 10.0]]  # the Hessian of the chapter-8 loss
_WSTAR = [1.0, 2.0]


def _loss(w):
    return 0.5 * ((w[0] + w[1] - 3.0) ** 2 + (w[0] + 3.0 * w[1] - 7.0) ** 2)


def _grad(w):
    e = [w[0] - _WSTAR[0], w[1] - _WSTAR[1]]
    return [_H[0][0] * e[0] + _H[0][1] * e[1], _H[1][0] * e[0] + _H[1][1] * e[1]]


def _eigen_sym2(m):
    a, b, c = m[0][0], m[0][1], m[1][1]
    mean = (a + c) / 2.0
    rad = math.sqrt(((a - c) / 2.0) ** 2 + b * b)
    out = []
    for lam in (mean + rad, mean - rad):
        v = [b, lam - a]
        n = math.hypot(*v)
        out.append((lam, [v[0] / n, v[1] / n]))
    return out


def flow_at(w0, t):
    """Exact solution of dw/dt = -grad L(w) for the chapter-8 loss, by its two curvature directions."""
    e0 = [w0[0] - _WSTAR[0], w0[1] - _WSTAR[1]]
    w = list(_WSTAR)
    for lam, v in _eigen_sym2(_H):
        c = (v[0] * e0[0] + v[1] * e0[1]) * math.exp(-lam * t)
        w[0] += c * v[0]
        w[1] += c * v[1]
    return w


def descend(w0, eta, steps):
    w = list(w0)
    path = [list(w)]
    for _ in range(steps):
        g = _grad(w)
        w = [w[0] - eta * g[0], w[1] - eta * g[1]]
        path.append(list(w))
    return path


def loss_rate_along_flow():
    rows = []
    for t in (0.0, 0.1, 0.3, 0.8):
        w = flow_at([0.0, 0.0], t)
        eps = 1e-5
        rate = (_loss(flow_at([0.0, 0.0], t + eps)) - _loss(flow_at([0.0, 0.0], t - eps))) / (2 * eps)
        g = _grad(w)
        rows.append((t, f"({w[0]:.4f}, {w[1]:.4f})", _fx(_loss(w), 4), _fx(rate, 4), _fx(-(g[0] ** 2 + g[1] ** 2), 4)))
    return ("Along the exact flow from (0, 0): the rate of change of L against minus the squared gradient length.\n\n"
            + _table(["t", "w(t)", "L(w(t))", "dL/dt by central difference", "-|grad L|^2"], rows))


def flow_vs_descent():
    w0 = [0.0, 0.0]
    rows = []
    for eta, steps in ((0.05, (0, 2, 4, 8, 20)), (0.18, (0, 2, 4, 8))):
        path = descend(w0, eta, max(steps))
        for k in steps:
            t = eta * k
            wf = flow_at(w0, t)
            rows.append((eta, k, _fx(t, 2), f"({path[k][0]:.4f}, {path[k][1]:.4f})", _fx(_loss(path[k]), 4),
                         f"({wf[0]:.4f}, {wf[1]:.4f})", _fx(_loss(wf), 4)))
    return ("Gradient descent with step eta after k steps, against the exact flow at t = eta k, both from (0, 0).\n\n"
            + _table(["eta", "k", "t", "descent w", "descent L", "flow w(t)", "flow L"], rows))


# --------------------------------------------------------- better solvers

def _heun_step(f, t, x, h):
    k1 = f(t, x)
    k2 = f(t + h, x + h * k1)
    return x + h * (k1 + k2) / 2.0


def _midpoint_step(f, t, x, h):
    return x + h * f(t + h / 2.0, x + h / 2.0 * f(t, x))


def _euler_step(f, t, x, h):
    return x + h * f(t, x)


def _solve(step, f, x0, t_end, n):
    h = t_end / n
    x = x0
    for i in range(n):
        x = step(f, i * h, x, h)
    return x


def second_order_methods():
    f = lambda t, x: -x
    exact = math.exp(-2.0)
    rows = []
    prev = {}
    for budget in (8, 16, 32, 64, 128):
        cells = [budget]
        for name, step, per_step in (("euler", _euler_step, 1), ("heun", _heun_step, 2)):
            n = budget // per_step
            err = abs(_solve(step, f, 1.0, 2.0, n) - exact)
            cells.append(_sci(err) + _ratio(prev.get(name), err))
            prev[name] = err
        mid = abs(_solve(_midpoint_step, f, 1.0, 2.0, budget // 2) - exact)
        cells.append(_sci(mid))
        rows.append(tuple(cells))
    return ("dx/dt = -x, x(0) = 1, up to t = 2, with a fixed budget of evaluations of f (Euler uses one per step, "
            "Heun and midpoint two). Error at t = 2; in brackets, the fall when the budget doubled.\n\n"
            + _table(["evaluations of f", "Euler", "Heun", "midpoint"], rows))


def euler_is_riemann():
    f = speed
    n = 6
    h = 3.0 / n
    ode = lambda t, x: speed(t)  # the slope does not depend on x
    eu = _solve(_euler_step, ode, 0.0, 3.0, n)
    heun = _solve(_heun_step, ode, 0.0, 3.0, n)
    mid = _solve(_midpoint_step, ode, 0.0, 3.0, n)
    rows = [
        ("Euler", _fx(eu, 6), "left rectangles", _fx(rule_sum(f, 0.0, 3.0, n, "left"), 6)),
        ("Heun", _fx(heun, 6), "trapezoids", _fx(rule_sum(f, 0.0, 3.0, n, "trap"), 6)),
        ("midpoint", _fx(mid, 6), "midpoint rectangles", _fx(rule_sum(f, 0.0, 3.0, n, "mid"), 6)),
    ]
    return (f"dx/dt = v(t) = t (4 - t), x(0) = 0, up to t = 3 with {n} steps of h = {h:g}; exact answer "
            f"{SPEED_AREA_0_3:g}.\n\n"
            + _table(["ODE solver", "x(3)", "quadrature rule", "area"], rows, align="lrlr"))


def backward_in_time():
    rows = []
    for n in (2, 4, 8, 16):
        h = -1.0 / n  # sigma runs from 1 down to 0: negative steps
        cells = [n]
        for step in (_euler_step, _heun_step):
            x = 1.0
            for i in range(n):
                x = step(lambda s, _x: 3.0 * s * s, 1.0 + i * h, x, h)
            cells.append(_fx(x, 6))
        x = 1.0
        for i in range(n):
            x = _euler_step(lambda s, _x: 0.7, 1.0 + i * h, x, h)
        cells.append(_fx(x, 6))
        rows.append(tuple(cells))
    return ("Integrating from sigma = 1 down to 0 in n equal steps, x(1) = 1. Exact end values: 0 for "
            "dx/dsigma = 3 sigma^2, and 1 - 0.7 = 0.3 for dx/dsigma = 0.7.\n\n"
            + _table(["steps n", "Euler, v = 3 sigma^2", "Heun, v = 3 sigma^2", "Euler, v = 0.7"], rows))


def exponential_step():
    rows = []
    for h in (0.5, 1.0, 2.0, 2.5, 5.0, 10.0):
        rows.append((h, f"{1.0 - h:+.4f}", f"{math.exp(-h):.4f}"))
    return ("One step of size h on dx/dt = -x multiplies x by this factor: Euler uses 1 - h, "
            "the exact solution over the step is e^(-h).\n\n"
            + _table(["step h", "Euler factor 1 - h", "exact factor e^(-h)"], rows))
