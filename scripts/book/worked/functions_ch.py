# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Functions and Their Shapes".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: functions_ch.<name> -->` markers. Functions whose names start
with an underscore (or that take arguments) are shared with the chapter's
figures, so a picture and the table beside it cannot disagree.

The module is called `functions_ch` to leave room for any future module that
wants the plain name.

Swedish Embedded AB implements solutions for numerically trustworthy machine
learning code. If your team needs expertise in model families, learning-rate
schedules and the numerical behaviour of elementary functions, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math
import struct
from fractions import Fraction


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    cell = lambda c: str(c).replace("|", "\\|")  # a literal pipe inside a cell must not split the column
    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def _g(v, digits=4):
    """Compact decimal: up to `digits` significant digits, no trailing zeros, sci for huge/tiny."""
    if isinstance(v, str):
        return v
    if isinstance(v, Fraction):
        v = float(v)
    if v == 0:
        return "0"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if abs(v) >= 1e7 or abs(v) < 1e-4:
        return f"{v:.{digits - 1}e}"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(v))))
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{v:.0f}"


def _f32(v):
    """Round a Python float to the nearest 32-bit float; inf on overflow."""
    try:
        return struct.unpack("f", struct.pack("f", v))[0]
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def _frac(q):
    q = Fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


UNDEF = "undefined"


# ------------------------------------------------- the pieces named in text

def relu(x):
    return max(0.0, x)


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


def leaky(x, slope=0.1):
    return x if x >= 0 else slope * x


def horner(coeffs, x):
    """Value of c0 + c1 x + ... + cn x^n from the highest coefficient down."""
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = acc * x + c
    return acc


def _solve_exact(a, b):
    """Gaussian elimination with partial row swaps on exact fractions; returns the solution."""
    n = len(a)
    m = [[Fraction(v) for v in row] + [Fraction(rhs)] for row, rhs in zip(a, b)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        m[col] = [v / m[col][col] for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[r][n] for r in range(n)]


def _interpolate(points):
    """Coefficients c0..cn of the unique degree-n polynomial through n+1 points (exact)."""
    n = len(points)
    a = [[Fraction(x) ** j for j in range(n)] for x, _ in points]
    return _solve_exact(a, [y for _, y in points])


# QUADRATIC_POINTS is shared with the figure.
QUADRATIC_POINTS = [(0, 1), (1, 0), (2, 3)]
# SCHEDULE is the piecewise learning-rate schedule of the chapter, shared with the figure.
SCHEDULE = dict(peak=0.1, warmup=10, gamma=0.97, floor=0.01)


def schedule_rate(step, peak=0.1, warmup=10, gamma=0.97, floor=0.01):
    """Linear ramp to `peak` over `warmup` steps, then peak * gamma^(step - warmup), never below `floor`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    return max(floor, peak * gamma ** (step - warmup))


# ---------------------------------------------------------------- functions

def domain_checks():
    xs = [-2, -1, 0, 1, 4]
    rows = []
    for x in xs:
        rows.append((
            x,
            _g(1 / x) if x != 0 else UNDEF,
            _g(math.sqrt(x)) if x >= 0 else UNDEF,
            _g(math.log(x)) if x > 0 else UNDEF,
            _g(x * x),
            _g(2.0 ** x),
        ))
    return _table(["x", "1/x", "sqrt(x)", "log(x)", "x^2", "2^x"], rows)


# --------------------------------------------------------------------- lines

def line_family():
    params = [(1, 0), (2, 0), (0.5, 0), (1, 2), (-1, 1)]
    xs = [-2, -1, 0, 1, 2]
    rows = []
    for x in xs:
        rows.append((x,) + tuple(_g(w * x + b) for w, b in params))
    heads = ["x"] + [f"w={_g(w)}, b={_g(b)}" for w, b in params]
    return _table(heads, rows)


def line_through_two_points():
    (x1, y1), (x2, y2) = (1, 3), (4, 9)
    w = Fraction(y2 - y1, x2 - x1)
    b = y1 - w * x1
    rows = [
        ("slope w = (y2 - y1) / (x2 - x1)", f"({y2} - {y1}) / ({x2} - {x1}) = {_frac(w)}"),
        ("intercept b = y1 - w x1", f"{y1} - {_frac(w)} * {x1} = {_frac(b)}"),
        (f"check x = {x1}", f"{_frac(w)} * {x1} + {_frac(b)} = {_frac(w * x1 + b)}"),
        (f"check x = {x2}", f"{_frac(w)} * {x2} + {_frac(b)} = {_frac(w * x2 + b)}"),
    ]
    return _table(["step", "value"], rows, "ll")


# --------------------------------------------------------------- polynomials

def horner_steps():
    coeffs = [-4, 3, -5, 2]  # c0 .. c3 of 2x^3 - 5x^2 + 3x - 4
    x = 3
    acc = coeffs[-1]
    rows = [(f"start with c3", f"{acc}")]
    for k in range(len(coeffs) - 2, -1, -1):
        new = acc * x + coeffs[k]
        rows.append((f"multiply by x = {x}, add c{k} = {coeffs[k]}", f"{acc} * {x} + ({coeffs[k]}) = {new}"))
        acc = new
    direct = sum(c * x ** j for j, c in enumerate(coeffs))
    assert acc == direct == horner(coeffs, x)
    rows.append(("direct power-by-power evaluation", f"{direct}"))
    return _table(["step", "value"], rows, "ll")


def quadratic_fit():
    pts = QUADRATIC_POINTS
    c = _interpolate(pts)  # c0, c1, c2
    rows = []
    for x, y in pts:
        p = c[0] + c[1] * x + c[2] * x * x
        rows.append((x, y, _frac(p), "yes" if p == y else "NO"))
    head = (f"Solution of the three equations: c = {_frac(c[0])}, b = {_frac(c[1])}, a = {_frac(c[2])}, "
            f"so p(x) = {_frac(c[2])} x^2 + ({_frac(c[1])}) x + {_frac(c[0])}.\n\n")
    return head + _table(["x", "y wanted", "p(x)", "matches"], rows)


def quadratic_roots_and_vertex():
    c, b, a = [float(v) for v in _interpolate(QUADRATIC_POINTS)]
    disc = b * b - 4 * a * c
    r1, r2 = (-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)
    xv = -b / (2 * a)
    yv = a * xv * xv + b * xv + c
    rows = [("a, b, c", f"{_g(a)}, {_g(b)}, {_g(c)}"),
            ("b^2 - 4ac", _g(disc)),
            ("zeros (x where p(x) = 0)", f"{_g(r1)} and {_g(r2)}"),
            ("lowest point (x, p(x))", f"({_g(xv)}, {_g(yv)})")]
    return _table(["quantity", "value"], rows, "ll")


_NOISY = [(0, 1.1), (1, 2.9), (2, 5.2), (3, 6.8), (4, 9.1)]


def interpolation_overfit():
    pts = [(Fraction(x), Fraction(str(y))) for x, y in _NOISY]
    coef = _interpolate(pts)
    w = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
    b = pts[0][1] - w * pts[0][0]
    rows = []
    for x in [0, 1, 2, 3, 4, 5, 6, 8]:
        poly = sum(c * Fraction(x) ** j for j, c in enumerate(coef))
        rows.append((x, _g(2 * x + 1), _g(w * x + b), _g(poly)))
    head = ("Degree-4 coefficients c0..c4: " + ", ".join(_g(c, 5) for c in coef) + ".\n\n")
    return head + _table(["x", "true rule 2x + 1", "line through first and last point", "degree-4 polynomial through all five"], rows)


def runge():
    def target(x):
        return 1.0 / (1.0 + 25.0 * x * x)

    def lagrange(xs, ys, x):
        total = 0.0
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            term = yi
            for j, xj in enumerate(xs):
                if j != i:
                    term *= (x - xj) / (xi - xj)
            total += term
        return total

    grid = [-1.0 + 2.0 * i / 1000 for i in range(1001)]
    rows = []
    for n in (3, 5, 9, 13):
        xs = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]
        ys = [target(x) for x in xs]
        worst = max(abs(lagrange(xs, ys, x) - target(x)) for x in grid)
        rows.append((n, n - 1, _g(worst)))
    return _table(["points n", "degree n-1", "largest error on [-1, 1]"], rows)


# --------------------------------------------------------------- composition

def composition_steps():
    rows = []
    for x in (-1, 0, 1, 2):
        a = 3 * x - 2
        b = a * a
        c = math.log(b + 1)
        rows.append((x, a, b, b + 1, _g(c)))
    return _table(["x", "a = 3x - 2", "b = a^2", "b + 1", "c = log(b + 1)"], rows)


def composition_order():
    def f(x):
        return x + 1

    def g(x):
        return x * x

    rows = []
    for x in (-2, 0, 1, 2, 3):
        rows.append((x, g(x), f(g(x)), f(x), g(f(x))))
    return _table(["x", "g(x) = x^2", "(f o g)(x) = x^2 + 1", "f(x) = x + 1", "(g o f)(x) = (x + 1)^2"], rows)


def domain_of_composition():
    rows = []
    for x in (-3, -1, -0.5, 0, 1, 1.5, 3):
        inner = x * x - 1
        outer = _g(math.log(inner)) if inner > 0 else UNDEF
        rows.append((_g(x), _g(inner), outer))
    return _table(["x", "inner x^2 - 1", "log(x^2 - 1)"], rows)


def lines_collapse():
    w1, b1, w2, b2 = 2, 1, -3, 4
    w, b = w2 * w1, w2 * b1 + b2
    rows = []
    for x in (-2, -1, 0, 1, 2):
        u = w1 * x + b1
        y = w2 * u + b2
        assert y == w * x + b
        rows.append((x, u, y, w * x + b))
    head = (f"Layer 1: u = {w1}x + {b1}. Layer 2: y = ({w2})u + {b2}. "
            f"Collapsed: w = {w2} * {w1} = {w}, b = {w2} * {b1} + {b2} = {b}.\n\n")
    return head + _table(["x", "u = 2x + 1", "y = -3u + 4", f"{w}x + {b}"], rows)


def relu_breaks_collapse():
    rows = []
    for x in (-2, -1, 0, 1, 2):
        u = 2 * x + 1
        r = max(0, u)
        y = -3 * r + 4
        rows.append((x, u, r, y, -6 * x + 1))
    return _table(["x", "u = 2x + 1", "max(0, u)", "y = -3 max(0, u) + 4", "the collapsed line -6x + 1"], rows)


# ------------------------------------------------------------------ inverses

def inverse_roundtrip():
    rows = []
    for x in (-2, 0, 1, 3):
        y = 2 * x + 1
        rows.append((x, y, _frac(Fraction(y - 1, 2)), _g(math.log(math.exp(x))), _g(math.exp(math.log(abs(y)))) if y > 0 else UNDEF))
    return _table(["x", "y = 2x + 1", "(y - 1) / 2", "log(exp(x))", "exp(log(y))"], rows)


def inverse_of_composite():
    rows = []
    for x in (-1.0, 0.0, 1.0, 2.0):
        y = 3 * math.exp(x) - 2        # (f o g)(x) with g = exp, f(u) = 3u - 2
        u = (y + 2) / 3                # undo f first
        back = math.log(u)             # then undo g
        rows.append((_g(x), _g(y), _g(u), _g(back)))
    return _table(["x", "y = 3 exp(x) - 2", "undo f: (y + 2) / 3", "undo g: log(.)"], rows)


def square_not_invertible():
    rows = []
    for x in range(-3, 4):
        y = x * x
        rows.append((x, y, _g(math.sqrt(y)), "yes" if math.sqrt(y) == x else "NO"))
    return _table(["x", "y = x^2", "sqrt(y)", "sqrt(y) equals x"], rows)


def bisection_inverse():
    def f(x):
        return x ** 3 + x

    target, lo, hi, tol = 10.0, 0.0, 3.0, 1e-3
    rows = []
    k = 0
    while hi - lo > tol:
        k += 1
        mid = (lo + hi) / 2
        fm = f(mid)
        rows.append((k, _g(lo, 6), _g(hi, 6), _g(mid, 6), _g(fm, 6), "too small, raise lo" if fm < target else "too big, lower hi"))
        if fm < target:
            lo = mid
        else:
            hi = mid
    return _table(["step", "lo", "hi", "mid", "f(mid)", "decision"], rows, "lrrrrl")


# -------------------------------------------------------------- exponentials

def compounding():
    rows = []
    for t in (0, 1, 5, 10, 20, 50):
        rows.append((t, f"{100 * 1.05 ** t:.2f}"))
    doubling = math.log(2) / math.log(1.05)
    head = f"Doubling time: log 2 / log 1.05 = {doubling:.3f} years.\n\n"
    return head + _table(["year t", "balance 100 * 1.05^t"], rows)


def decay_table():
    lr0 = 0.1
    gammas = (0.99, 0.97, 0.94)
    rows = []
    for t in (0, 10, 50, 100, 200):
        rows.append((t,) + tuple(_g(lr0 * g ** t, 3) for g in gammas))
    return _table(["step t"] + [f"gamma = {g}" for g in gammas], rows)


def decay_times():
    rows = []
    for g in (0.99, 0.97, 0.94):
        k = -math.log(g)
        half = math.log(2) / k
        tenfold = math.log(10) / k
        rows.append((g, _g(k, 4), _g(half, 4), _g(tenfold, 4), math.ceil(tenfold)))
    return _table(["gamma", "rate k = -log(gamma)", "half-life log 2 / k", "10x lower: log 10 / k", "first whole step"], rows)


def growth_comparison():
    rows = []
    for x in (1, 5, 10, 20, 50, 100):
        rows.append((x, _g(float(x) ** 2), _g(float(x) ** 10), _g(2.0 ** x), _g(math.exp(x))))
    return _table(["x", "x^2", "x^10", "2^x", "e^x"], rows)


# ----------------------------------------------------------------- monotone

def monotone_argmax():
    scores = [2.0, -1.0, 0.5, 3.1]

    def argmax(v):
        return max(range(len(v)), key=lambda i: v[i])

    transforms = [
        ("identity", lambda s: s),
        ("7 + 3 * s", lambda s: 3 * s + 7),
        ("exp(s)", math.exp),
        ("s^3", lambda s: s ** 3),
        ("-s", lambda s: -s),
        ("s^2", lambda s: s * s),
    ]
    rows = []
    for name, t in transforms:
        v = [t(s) for s in scores]
        rows.append((name, ", ".join(_g(a, 4) for a in v), argmax(v)))
    return _table(["transform g", "g(s) for s = (2.0, -1.0, 0.5, 3.1)", "arg max (0-indexed)"], rows, "llr")


def monotone_ranking_of_models():
    models = {"A": [0.5, 0.4, 0.3, 0.2], "B": [0.9, 0.1, 0.5, 0.4], "C": [0.6, 0.6, 0.1, 0.3]}
    rows = []
    for name, ps in models.items():
        prod = math.prod(ps)
        rows.append((name, _g(prod, 4), _g(sum(math.log(p) for p in ps), 4)))
    best_p = max(models, key=lambda n: math.prod(models[n]))
    best_l = max(models, key=lambda n: sum(math.log(p) for p in models[n]))
    assert best_p == best_l
    return _table(["model", "product of chances", "sum of logs"], rows) + f"\n\nBest by either: model {best_p}."


def monotone_mean_warning():
    runs = {"steady": [1.0, 1.0, 1.0], "spiky": [0.0, 0.0, 2.4]}
    rows = []
    for name, v in runs.items():
        m = sum(v) / len(v)
        me = sum(math.exp(a) for a in v) / len(v)
        rows.append((name, ", ".join(_g(a) for a in v), _g(m), _g(me, 4), _g(math.exp(m), 4)))
    return _table(["run", "scores", "mean of s", "mean of exp(s)", "exp(mean of s)"], rows, "llrrr")


# ----------------------------------------------------------------- piecewise

def piecewise_shapes():
    rows = []
    for x in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3):
        rows.append((_g(x), _g(abs(x)), _g(relu(x)), _g(clamp(x, -1, 2)), _g(leaky(x)), 1 if x > 0 else 0))
    return _table(["x", "abs(x)", "relu(x) = max(0, x)", "clamp(x, -1, 2)", "leaky(x), slope 0.1", "step(x)"], rows)


def relu_tent():
    rows = []
    for k in range(-2, 9):
        x = k / 2
        r0, r1, r2 = relu(x), relu(x - 1), relu(x - 2)
        rows.append((_g(x), _g(r0), _g(r1), _g(r2), _g(r0 - 2 * r1 + r2)))
    return _table(["x", "relu(x)", "relu(x - 1)", "relu(x - 2)", "relu(x) - 2 relu(x - 1) + relu(x - 2)"], rows)


def schedule_table():
    s = SCHEDULE
    rows = []
    for t in (0, 4, 9, 10, 11, 20, 50, 85, 86, 87, 150):
        if t < s["warmup"]:
            piece = "warmup (line)"
        elif s["peak"] * s["gamma"] ** (t - s["warmup"]) > s["floor"]:
            piece = "decay (exponential)"
        else:
            piece = "floor (constant)"
        rows.append((t, piece, _g(schedule_rate(t, **s), 4)))
    k = s["warmup"] + math.log(s["floor"] / s["peak"]) / math.log(s["gamma"])
    head = (f"Warmup {s['warmup']} steps to peak {s['peak']}; then peak * {s['gamma']}^(t - {s['warmup']}); "
            f"floor {s['floor']}. The decay meets the floor at t = {s['warmup']} + log({s['floor']}/{s['peak']}) / "
            f"log({s['gamma']}) = {k:.2f}, so the floor takes over from step {math.ceil(k)}.\n\n")
    return head + _table(["step t", "piece", "rate"], rows, "lll")


def schedule_monotone_check():
    s = SCHEDULE
    rates = [schedule_rate(t, **s) for t in range(200)]
    tol = 1e-12
    peak_step = next(i for i in range(len(rates)) if rates[i] >= max(rates) - tol)
    up = all(rates[i] < rates[i + 1] for i in range(s["warmup"] - 1))
    down = all(rates[i] >= rates[i + 1] for i in range(s["warmup"] - 1, 199))
    jumps = max(abs(rates[i + 1] - rates[i]) for i in range(199))
    rows = [("strictly increasing on steps 0 .. warmup - 1", "yes" if up else "NO"),
            ("never increasing from step warmup - 1 on", "yes" if down else "NO"),
            ("first step at the peak", peak_step),
            ("largest change between neighbouring steps", _g(jumps, 3))]
    return _table(["property (steps 0 .. 199)", "result"], rows, "lr")


# ------------------------------------------------------------ making it work

def exp_overflow():
    rows = []
    for x in (88, 89, 709, 710):
        e64 = math.exp(x) if x < 710 else math.inf
        e32 = _f32(math.exp(x)) if x < 710 else math.inf
        rows.append((x, _g(e32, 4), _g(e64, 4)))
    return _table(["x", "exp(x) in 32-bit", "exp(x) in 64-bit"], rows)


def log_exp_roundtrip():
    rows = []
    for x in (1.0, 20.0, 700.0, 710.0, -700.0, -740.0, -750.0):
        try:
            e = math.exp(x)
        except OverflowError:
            e = math.inf
        if e == 0:
            back = "log(0) = -inf"
        elif math.isinf(e):
            back = "log(inf) = inf"
        else:
            back = _g(math.log(e), 8)
        rows.append((_g(x), _g(e, 4), back))
    return _table(["x", "exp(x)", "log(exp(x))"], rows)


def silu_tail():
    rows = []
    for x in (-5.0, -20.0, -100.0):
        direct = x / (1.0 + math.exp(-x))
        rows.append((_g(x), _g(math.exp(-x), 4), _g(direct, 4)))
    return _table(["x", "exp(-x)", "x / (1 + exp(-x))"], rows)
