# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Derivatives: The Mathematics of Change".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: derivatives.<name> -->` markers. Helpers with a leading
underscore (or that take arguments) are shared with the chapter's figures, so
a picture and the table beside it cannot disagree.

Swedish Embedded AB implements solutions for numerically trustworthy machine
learning code. If your team needs expertise in gradients, finite-difference
checks and second-order optimisation, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import math
from decimal import Decimal, getcontext
from fractions import Fraction


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


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def secant_slope(f, x, h):
    """Slope of the line through (x, f(x)) and (x + h, f(x + h))."""
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2.0 * h)


# ----------------------------------------------- the pieces named in text

def loss_one_weight(w):
    """The one-parameter loss of the opening problem: input 2, target 3."""
    return (2.0 * w - 3.0) ** 2


def cubic_wiggle(x):
    """x^4 - 3x^2 + x, the non-convex curve with two valleys."""
    return x ** 4 - 3.0 * x ** 2 + x


def cubic_wiggle_slope(x):
    return 4.0 * x ** 3 - 6.0 * x + 1.0


def tilted_exp(x):
    """e^x - 2x, the convex curve with its minimum at log 2."""
    return math.exp(x) - 2.0 * x


def newton_root_step(f, df, x):
    return x - f(x) / df(x)


# ------------------------------------------------------------ the problem

def nudge_loss():
    w = 0.5
    base = loss_one_weight(w)
    rows = []
    for h in (0.5, 0.1, 0.01, 0.001, -0.1, -0.01):
        new = loss_one_weight(w + h)
        rows.append([f"{h:+g}", _fx(w + h, 3), _fx(new, 6), f"{new - base:+.6f}", f"{(new - base) / h:+.4f}"])
    return (f"Loss L(w) = (2w - 3)^2 at w = {w}: L = {_fx(base, 4)}.\n\n"
            + _table(["nudge h", "new w", "L(w + h)", "change in L", "change per unit of h"], rows))


# ------------------------------------------------------ secants, limits

def secant_table():
    f = lambda x: x * x
    rows = []
    for h in (1.0, 0.5, 0.1, 0.01, 0.001, 1e-6, -0.1, -0.01, -0.001):
        s = secant_slope(f, 1.0, h)
        rows.append([f"{h:g}", _fx(f(1.0 + h), 9), _fx(s, 6), f"{s - 2.0:+.2e}"])
    return "f(x) = x^2 at x = 1, f(1) = 1.\n\n" + _table(["h", "f(1 + h)", "secant slope", "slope - 2"], rows)


def limit_examples():
    g = lambda x: (x * x - 1.0) / (x - 1.0)
    rows = []
    for h in (0.1, 0.01, 0.001, 1e-6):
        rows.append([f"{h:g}", _fx(g(1.0 - h), 6), _fx(g(1.0 + h), 6)])
    first = _table(["h", "g(1 - h)", "g(1 + h)"], rows)
    a = lambda x: abs(x)
    rows = []
    for h in (0.1, 0.001, -0.1, -0.001):
        rows.append([f"{h:+g}", _fx(secant_slope(a, 0.0, h), 3)])
    second = _table(["h", "secant slope of abs(x) at 0"], rows)
    return ("g(x) = (x^2 - 1)/(x - 1) approaching x = 1 from both sides (g(1) itself is 0/0):\n\n"
            + first + "\n\nThe absolute value at x = 0:\n\n" + second)


def tangent_error():
    f = lambda x: x ** 3
    rows = []
    for h in (1.0, 0.5, 0.1, 0.01, 0.001):
        exact = f(1.0 + h)
        line = f(1.0) + 3.0 * h
        err = exact - line
        rows.append([f"{h:g}", _fx(exact, 6), _fx(line, 6), _fx(err, 6), _fx(err / h ** 2, 4)])
    return ("f(x) = x^3 at x = 1, where f(1) = 1 and f'(1) = 3.\n\n"
            + _table(["h", "f(1 + h)", "tangent line 1 + 3h", "error", "error / h^2"], rows))


# --------------------------------------------------------- the rules

def power_rule():
    x = 1.5
    rows = []
    for n in range(1, 7):
        formula = n * x ** (n - 1)
        numeric = central_difference(lambda t: t ** n, x)
        rows.append([n, _fx(x ** n, 6), _fx(formula, 6), _fx(numeric, 6), _sci(abs(formula - numeric))])
    return ("Derivative of x^n at x = 1.5, rule n x^(n-1) against a central difference with h = 1e-5.\n\n"
            + _table(["n", "x^n", "rule n x^(n-1)", "central difference", "abs difference"], rows))


def derivative_rules():
    x = 2.0
    cases = [
        # name, f, g, f', g', combined value and its rule
        ("x^2 + x^3 (sum)", lambda t: t ** 2, lambda t: t ** 3, lambda t: 2 * t, lambda t: 3 * t ** 2,
         lambda f, g, df, dg: df + dg, lambda t: t ** 2 + t ** 3),
        ("5 x^3 (constant multiple)", lambda t: 5.0, lambda t: t ** 3, lambda t: 0.0, lambda t: 3 * t ** 2,
         lambda f, g, df, dg: 5.0 * dg, lambda t: 5.0 * t ** 3),
        ("x^2 (x + 1) (product)", lambda t: t ** 2, lambda t: t + 1.0, lambda t: 2 * t, lambda t: 1.0,
         lambda f, g, df, dg: df * g + f * dg, lambda t: t ** 2 * (t + 1.0)),
        ("1 / (x + 1) (reciprocal)", lambda t: 1.0, lambda t: t + 1.0, lambda t: 0.0, lambda t: 1.0,
         lambda f, g, df, dg: -dg / g ** 2, lambda t: 1.0 / (t + 1.0)),
        ("x^2 / (x + 1) (quotient)", lambda t: t ** 2, lambda t: t + 1.0, lambda t: 2 * t, lambda t: 1.0,
         lambda f, g, df, dg: (df * g - f * dg) / g ** 2, lambda t: t ** 2 / (t + 1.0)),
    ]
    rows = []
    for name, f, g, df, dg, rule, whole in cases:
        r = rule(f(x), g(x), df(x), dg(x))
        n = central_difference(whole, x)
        rows.append([name, _fx(f(x), 2), _fx(g(x), 2), _fx(df(x), 2), _fx(dg(x), 2), _fx(r, 6), _fx(n, 6)])
    return ("At x = 2, with f and g the two pieces of each expression.\n\n"
            + _table(["expression", "f(2)", "g(2)", "f'(2)", "g'(2)", "rule", "central difference"], rows))


# ------------------------------------------------ the exponential and e

def exp_base_slopes():
    rows = []
    for a, label in ((2.0, "2"), (3.0, "3"), (math.e, "e"), (10.0, "10")):
        cells = [(a ** h - 1.0) / h for h in (0.1, 0.01, 0.001, 1e-6)]
        rows.append([label] + [_fx(c, 5) for c in cells] + [_fx(math.log(a), 5)])
    return ("The slope at 0 of a^x is the limit of (a^h - 1)/h.\n\n"
            + _table(["base a", "h = 0.1", "h = 0.01", "h = 0.001", "h = 1e-6", "log a"], rows))


def exp_limit():
    rows = []
    for n in (1, 10, 100, 1000, 10 ** 6):
        rows.append([f"{n}", _fx((1.0 + 1.0 / n) ** n, 6), _fx(n * (math.exp(1.0 / n) - 1.0), 6)])
    return ("With h = 1/n, e^h - 1 over h should approach 1 if (1 + 1/n)^n approaches e.\n\n"
            + _table(["n", "(1 + 1/n)^n", "(e^h - 1)/h with h = 1/n"], rows))


def exp_poly():
    """Building a function equal to its own derivative, one power at a time."""
    x = 1.0
    rows = []
    total, term = 0.0, 1.0
    for k in range(0, 11):
        if k > 0:
            term = term * x / k
        total += term
        # power rule on x^j / j! gives j x^(j-1) / j! = x^(j-1) / (j-1)!
        d = sum(x ** (j - 1) / math.factorial(j - 1) for j in range(1, k + 1))
        rows.append([k, _fx(total, 9), _fx(d, 9), _fx(math.e - total, 9)])
    return ("The polynomial p_k(x) = 1 + x + x^2/2! + ... + x^k/k! at x = 1, with its derivative by the power rule.\n\n"
            + _table(["degree k", "p_k(1)", "p_k'(1) (equals p_(k-1)(1))", "e - p_k(1)"], rows))


def exp_self_derivative():
    rows = []
    for x in (-1.0, 0.0, 1.0, 2.0):
        rows.append([f"{x:g}", _fx(math.exp(x), 6), _fx(central_difference(math.exp, x), 6)])
    return _table(["x", "e^x", "central difference of e^x"], rows)


def log_derivative():
    rows = []
    for x in (0.5, 1.0, 2.0, 10.0):
        rows.append([f"{x:g}", _fx(math.log(x), 6), _fx(1.0 / x, 6), _fx(central_difference(math.log, x), 6)])
    return _table(["x", "log x", "1/x", "central difference of log x"], rows)


# --------------------------------------------------------- chain rule

def chain_gears():
    rows = []
    for p in (0.0, 1.0, 2.0, 2.5, 4.0):
        wheel = 3.0 * p
        dist = 2.0 * wheel
        rows.append([f"{p:g}", f"{wheel:g}", f"{dist:g}"])
    return ("Three wheel turns per pedal turn, two metres per wheel turn.\n\n"
            + _table(["pedal turns p", "wheel turns 3p", "distance in metres 2 * 3p"], rows))


def chain_numbers():
    f_outer = lambda u: u * u
    g_inner = lambda x: 3.0 * x + 1.0
    whole = lambda x: f_outer(g_inner(x))
    x = 1.0
    u = g_inner(x)
    first = _table(
        ["quantity", "value at x = 1"],
        [["inner u = 3x + 1", _fx(u, 4)], ["du/dx", "3.0000"], ["outer u^2", _fx(f_outer(u), 4)],
         ["d(u^2)/du = 2u", _fx(2 * u, 4)], ["product (du/dx)(2u)", _fx(3.0 * 2 * u, 4)],
         ["expanded 9x^2 + 6x + 1, derivative 18x + 6", _fx(18 * x + 6, 4)],
         ["central difference of the whole", _fx(central_difference(whole, x), 4)]])

    # three links: u = -x, v = e^u, w = 1 + v, s = 1/w
    x = 0.5
    links = []
    u = -x
    v = math.exp(u)
    w = 1.0 + v
    s = 1.0 / w
    local = [("u = -x", u, -1.0), ("v = e^u", v, v), ("w = 1 + v", w, 1.0), ("s = 1 / w", s, -1.0 / (w * w))]
    running = 1.0
    rows = []
    for name, val, d in local:
        running *= d
        rows.append([name, _fx(val, 6), _fx(d, 6), _fx(running, 6)])
    chain = _table(["link", "value", "local derivative", "derivative of this quantity w.r.t. x"], rows)
    check = _table(
        ["quantity at x = 0.5", "value"],
        [["product of local derivatives", _fx(running, 6)], ["s (1 - s)", _fx(s * (1 - s), 6)],
         ["central difference of s(x)", _fx(central_difference(_sigmoid, x), 6)]])
    return ("Chain of two links, (3x + 1)^2:\n\n" + first + "\n\nChain of four links, s(x) = 1/(1 + e^(-x)) at x = 0.5:\n\n"
            + chain + "\n\n" + check)


def sigmoid_slope():
    rows = []
    for x in (-4.0, -1.0, 0.0, 1.0, 4.0, 10.0):
        s = _sigmoid(x)
        rows.append([f"{x:g}", _fx(s, 6), _fx(s * (1 - s), 8), _fx(central_difference(_sigmoid, x), 8)])
    return _table(["x", "s(x)", "s (1 - s)", "central difference"], rows)


# ------------------------------------------- critical points, convexity

def critical_points():
    cases = [
        ("x^3 - 3x", lambda t: t ** 3 - 3 * t, lambda t: 3 * t ** 2 - 3, lambda t: 6 * t, (-1.0, 1.0)),
        ("x^3", lambda t: t ** 3, lambda t: 3 * t ** 2, lambda t: 6 * t, (0.0,)),
        ("x^4", lambda t: t ** 4, lambda t: 4 * t ** 3, lambda t: 12 * t ** 2, (0.0,)),
        ("-x^4", lambda t: -t ** 4, lambda t: -4 * t ** 3, lambda t: -12 * t ** 2, (0.0,)),
    ]
    rows = []
    for name, f, df, d2f, pts in cases:
        for p in pts:
            if d2f(p) > 0:
                verdict = "minimum"
            elif d2f(p) < 0:
                verdict = "maximum"
            else:
                lo, hi = f(p - 0.1), f(p + 0.1)
                if lo > f(p) and hi > f(p):
                    verdict = "minimum (test silent)"
                elif lo < f(p) and hi < f(p):
                    verdict = "maximum (test silent)"
                else:
                    verdict = "neither (test silent)"
            rows.append([name, f"{p:g}", _fx(f(p), 3), _fx(df(p), 3), _fx(d2f(p), 3), verdict])
    return _table(["function", "x", "f(x)", "f'(x)", "f''(x)", "verdict"], rows, "lrrrrl")


def _descend(df, x, eta, steps):
    for _ in range(steps):
        x -= eta * df(x)
    return x


def descent_endpoints():
    starts = (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0)
    rows = []
    for s in starts:
        a = _descend(cubic_wiggle_slope, s, 0.01, 2000)
        b = _descend(lambda t: math.exp(t) - 2.0, s, 0.05, 2000)
        rows.append([f"{s:g}", _fx(a, 4), _fx(cubic_wiggle(a), 4), _fx(b, 4), _fx(tilted_exp(b), 4)])
    return ("Step against the slope, x <- x - eta f'(x), 2000 times. Left: f = x^4 - 3x^2 + x with eta = 0.01. "
            "Right: g = e^x - 2x with eta = 0.05 (log 2 = " + _fx(math.log(2), 4) + ").\n\n"
            + _table(["start", "f ends at", "f there", "g ends at", "g there"], rows))


def chord_example():
    rows = []
    for name, f, a, b in (("x^2", lambda t: t * t, -1.0, 2.0), ("x^4 - 3x^2 + x", cubic_wiggle, -1.0, 1.0)):
        for lam in (0.5,):
            m = lam * a + (1 - lam) * b
            chord = lam * f(a) + (1 - lam) * f(b)
            rows.append([name, f"{a:g}", f"{b:g}", f"{m:g}", _fx(f(m), 4), _fx(chord, 4),
                         "chord above" if f(m) <= chord else "curve above chord"])
    return _table(["function", "a", "b", "midpoint m", "f(m)", "chord value at m", "result"], rows, "lrrrrrl")


def convexity_scan():
    s = _sigmoid
    funcs = [
        ("x^2", lambda t: t * t, lambda t: 2.0, (-3.0, 3.0)),
        ("e^x", math.exp, math.exp, (-3.0, 3.0)),
        ("-log x", lambda t: -math.log(t), lambda t: 1.0 / t ** 2, (0.1, 4.0)),
        ("(2w - 3)^2", loss_one_weight, lambda t: 8.0, (-3.0, 3.0)),
        ("x^4 - 3x^2 + x", cubic_wiggle, lambda t: 12 * t ** 2 - 6.0, (-2.0, 2.0)),
        ("(s(w) - 1)^2", lambda t: (s(t) - 1.0) ** 2,
         lambda t: -2.0 * s(t) * (1 - s(t)) ** 2 * (1.0 - 3.0 * s(t)), (-6.0, 6.0)),
    ]
    rows = []
    for name, f, d2, (lo, hi) in funcs:
        n = 40
        grid = [lo + (hi - lo) * i / n for i in range(n + 1)]
        bad = total = 0
        for i, a in enumerate(grid):
            for b in grid[i + 1:]:
                for lam in (0.25, 0.5, 0.75):
                    m = lam * a + (1 - lam) * b
                    total += 1
                    if f(m) > lam * f(a) + (1 - lam) * f(b) + 1e-12:
                        bad += 1
        lows = min(d2(t) for t in grid)
        rows.append([name, f"[{lo:g}, {hi:g}]", _fx(lows, 4), f"{bad} of {total}"])
    return _table(["function", "interval", "smallest f'' on a 41-point grid", "chords with the curve above them"],
                  rows, "llrr")


# ------------------------------------------------------------ Newton

def _newton_sqrt2_iterates(steps):
    xs = [Fraction(1)]
    for _ in range(steps):
        x = xs[-1]
        xs.append(x - (x * x - 2) / (2 * x))
    return xs


def newton_sqrt2():
    """Errors from the exact identity e_n = (x_n^2 - 2) / (x_n + sqrt 2).

    x_n is a fraction p/q, so x_n^2 - 2 = (p^2 - 2 q^2) / q^2 is an exact rational and is
    tiny; only the harmless second factor is rounded. Subtracting sqrt(2) from a
    64-bit x_n cannot show an error below about 1e-16.
    """
    getcontext().prec = 80
    root = Decimal(2).sqrt()
    xs = _newton_sqrt2_iterates(5)

    def error(x):
        numerator = Decimal((x * x - 2).numerator) / Decimal((x * x - 2).denominator)
        return numerator / (Decimal(x.numerator) / Decimal(x.denominator) + root)

    err = [error(x) for x in xs]
    rows = []
    for n in range(0, 5):
        xd = Decimal(xs[n].numerator) / Decimal(xs[n].denominator)
        pred = err[n] ** 2 / (2 * xd)
        rows.append([n, f"{xs[n]}", f"{float(err[n]):.3e}", f"{float(pred):.3e}",
                     f"{float(err[n + 1]):.3e}", f"{float(err[n + 1] / err[n] ** 2):.6f}"])
    limit = 1 / (2 * root)
    return (f"Newton's method for x^2 - 2 from x_0 = 1, fractions kept exact; 1/(2 sqrt 2) = {float(limit):.6f}. "
            "Each error is computed as e_n = (x_n^2 - 2) / (x_n + sqrt 2) with x_n^2 - 2 an exact rational.\n\n"
            + _table(["n", "x_n", "error e_n", "e_n^2 / (2 x_n)", "error e_(n+1)", "e_(n+1) / e_n^2"], rows))


def newton_failures():
    f = lambda t: t * t - 2.0
    df = lambda t: 2.0 * t
    x = 0.01
    rows = []
    for n in range(0, 14):
        rows.append([n, _fx(x, 6), _sci(abs(x - math.sqrt(2.0)))])
        x = newton_root_step(f, df, x)
    far = _table(["n", "x_n", "distance to sqrt 2"], rows)

    g = lambda t: t ** 3 - 2.0 * t + 2.0
    dg = lambda t: 3.0 * t ** 2 - 2.0
    x = 0.0
    rows = []
    for n in range(0, 7):
        rows.append([n, _fx(x, 4), _fx(g(x), 4), _fx(dg(x), 4)])
        x = newton_root_step(g, dg, x)
    cyc = _table(["n", "x_n", "g(x_n)", "g'(x_n)"], rows)
    return ("A far start for x^2 - 2, x_0 = 0.01, 64-bit floats, update x - f(x)/f'(x) = x - (x*x - 2)/(2*x), "
            "distance |x_n - 1.4142135623730951| (the double nearest sqrt 2):\n\n" + far + "\n\nA cycle for g(x) = x^3 - 2x + 2, x_0 = 0:\n\n" + cyc)


def newton_minimize():
    d1 = lambda t: math.exp(t) - 2.0   # g'
    d2 = lambda t: math.exp(t)         # g''
    target = math.log(2.0)
    xn = xa = xb = 0.0
    rows = []
    for k in range(0, 8):
        rows.append([k, _sci(abs(xn - target)), _sci(abs(xa - target)), _sci(abs(xb - target))])
        xn = xn - d1(xn) / d2(xn)
        xa = xa - 0.1 * d1(xa)
        xb = xb - 0.5 * d1(xb)
    first = _table(["step", "Newton |x - log 2|", "fixed step 0.1", "fixed step 0.5"], rows)

    def count(step):
        x, n = 0.0, 0
        while abs(x - target) > 1e-10 and n < 100000:
            x = step(x)
            n += 1
        return n
    counts = _table(["method", "steps until |x - log 2| <= 1e-10"],
                    [["Newton", count(lambda t: t - d1(t) / d2(t))],
                     ["fixed step 0.5", count(lambda t: t - 0.5 * d1(t))],
                     ["fixed step 0.1", count(lambda t: t - 0.1 * d1(t))]], "lr")

    q = lambda t: 3.0 * (t - 2.0) ** 2 + 1.0
    dq, d2q = (lambda t: 6.0 * (t - 2.0)), 6.0
    quad = _table(["start x", "f'(x)", "f''", "Newton step x - f'/f''"],
                  [["10", _fx(dq(10.0), 2), _fx(d2q, 2), _fx(10.0 - dq(10.0) / d2q, 4)]], "lrrr")

    x = 0.0
    rows = []
    for k in range(0, 6):
        rows.append([k, _fx(x, 6), _fx(cubic_wiggle_slope(x), 6), _fx(12 * x * x - 6.0, 6)])
        x = x - cubic_wiggle_slope(x) / (12 * x * x - 6.0)
    wrong = _table(["step", "x", "f'(x)", "f''(x)"], rows)
    return ("Minimising g(x) = e^x - 2x from x = 0 (minimum at log 2 = " + _fx(target, 6) + "):\n\n" + first
            + "\n\n" + counts + "\n\nA quadratic, f(x) = 3(x - 2)^2 + 1:\n\n" + quad
            + "\n\nNewton's method on the non-convex x^4 - 3x^2 + x from x = 0, where f'' < 0:\n\n" + wrong)


# ------------------------------------------- numerical differentiation

def numdiff_rows():
    """(h, forward error, central error, f''h/2, f'''h^2/6) for f = e^x at x = 1."""
    x = 1.0
    rows = []
    for k in range(1, 16):
        h = 10.0 ** (-k)
        fwd = (math.exp(x + h) - math.exp(x)) / h
        cen = (math.exp(x + h) - math.exp(x - h)) / (2.0 * h)
        rows.append((h, abs(fwd - math.e), abs(cen - math.e), math.e * h / 2.0, math.e * h * h / 6.0))
    return rows


def numdiff_errors():
    rows = [[f"1e-{k}", _sci(fe), _sci(pf), _sci(ce), _sci(pc)]
            for k, (h, fe, ce, pf, pc) in zip(range(1, 16), numdiff_rows())]
    return ("Derivative of e^x at x = 1 (exactly e), in 64-bit floating point.\n\n"
            + _table(["h", "forward error", "predicted e h / 2", "central error", "predicted e h^2 / 6"], rows))


def numdiff_best_step():
    """Step that balances truncation error against round-off, by order of magnitude."""
    rows = []
    for name, u in (("64-bit (u = 2^-53)", 2.0 ** -53), ("32-bit (u = 2^-24)", 2.0 ** -24)):
        rows.append([name, _sci(u), _sci(u ** 0.5), _sci(u ** (1.0 / 3.0))])
    return _table(["format", "unit round-off u", "forward: sqrt(u)", "central: u^(1/3)"], rows)


# ------------------------------------------ activations, for the brain section

def _gelu_tanh(x):
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _gelu_tanh_slope(x):
    """Product rule on 0.5 x (1 + t), chain rule through t = tanh(u), u = k (x + 0.044715 x^3)."""
    k = math.sqrt(2.0 / math.pi)
    t = math.tanh(k * (x + 0.044715 * x ** 3))
    du = k * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du


def _gelu_erf_slope(x):
    """Slope of 0.5 x (1 + erf(x / sqrt 2)): product rule, with d erf(u)/du = 2 exp(-u^2) / sqrt(pi)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) + x * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def activation_derivatives():
    cases = [
        ("tanh", math.tanh, lambda x: 1.0 - math.tanh(x) ** 2),
        ("x s(x)", lambda x: x * _sigmoid(x), lambda x: _sigmoid(x) + x * _sigmoid(x) * (1.0 - _sigmoid(x))),
        ("tanh-form GELU", _gelu_tanh, _gelu_tanh_slope),
    ]
    rows = []
    for name, f, df in cases:
        for x in (-2.0, -0.5, 0.0, 1.0, 3.0):
            a, n = df(x), central_difference(f, x)
            rows.append([name, f"{x:g}", _fx(f(x), 6), _fx(a, 6), _fx(n, 6), _sci(abs(a - n))])
    return _table(["function", "x", "value", "formula", "central difference", "abs difference"], rows, "lrrrrr")


def gelu_pair_gap():
    """The two GELU derivatives over [-4, 4] and what a gate with atol 4e-3, rtol 8e-2 can see."""
    atol, rtol = 4e-3, 8e-2
    grid = [i / 100.0 for i in range(-400, 401)]
    max_abs = (0.0, 0.0)
    max_rel = (0.0, 0.0)
    flagged = 0
    for x in grid:
        a, b = _gelu_tanh_slope(x), _gelu_erf_slope(x)
        d = abs(a - b)
        if d > max_abs[0]:
            max_abs = (d, x)
        r = d / max(abs(a), abs(b))
        if r > max_rel[0]:
            max_rel = (r, x)
        if d > atol + rtol * max(abs(a), abs(b)):
            flagged += 1
    rows = [["largest absolute difference", _sci(max_abs[0]), f"{max_abs[1]:g}"],
            ["largest relative difference", f"{max_rel[0]:.3f}", f"{max_rel[1]:g}"],
            ["grid points the gate rejects", f"{flagged} of {len(grid)}", ""]]
    return ("Slope of the tanh-form GELU against the slope of the exact erf-form GELU, x from -4 to 4 in steps of 0.01.\n\n"
            + _table(["quantity", "value", "at x"], rows, "lrr"))
