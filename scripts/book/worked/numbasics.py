# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Numbers, Quantities and Notation".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: numbasics.<name> -->` markers. The same module supplies the
numbers for the chapter's figures, so a picture and the table beside it cannot
disagree.

The module is called `numbasics`, not `numbers`, because a module named
`numbers` would shadow the standard library's `numbers` (imported by
`fractions` and `decimal`).

Swedish Embedded AB implements solutions for numerically trustworthy machine
learning code. If your team needs expertise in log-space probability
arithmetic and floating-point pitfalls, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import struct
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


def _f32(v):
    """Round a Python float to the nearest 32-bit float; inf on overflow."""
    try:
        return struct.unpack("f", struct.pack("f", v))[0]
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def _sci(v, digits=3):
    return f"{v:.{digits}e}"


# ---------------------------------------------------------------- integers

def integer_ranges():
    rows = []
    for bits in (8, 16, 32, 64):
        rows.append((bits, f"{-2 ** (bits - 1):,}", f"{2 ** (bits - 1) - 1:,}", f"{2 ** bits - 1:,}"))
    return _table(["bits n", "smallest signed -2^(n-1)", "largest signed 2^(n-1) - 1", "largest unsigned 2^n - 1"],
                  rows, "rrrr")


def integer_division():
    rows = []
    for a, b in ((17, 5), (-17, 5), (17, -5), (-17, -5)):
        qf, rf = a // b, a % b  # floor convention: remainder has the sign of b
        qt = math.trunc(a / b)
        rt = a - b * qt  # truncating convention: remainder has the sign of a
        assert a == b * qf + rf and a == b * qt + rt
        rows.append((f"{a}, {b}", qf, rf, qt, rt))
    return _table(["a, b", "floor q", "floor r", "truncating q", "truncating r"], rows, "lrrrr")


# ------------------------------------------------------------ real numbers

def decimal_expansions():
    """Long division of p/q, stopping when the remainder repeats or becomes 0."""
    def expand(p, q, limit=40):
        whole, rem = divmod(p, q)
        digits, seen = [], {}
        while rem and rem not in seen and len(digits) < limit:
            seen[rem] = len(digits)
            rem *= 10
            digits.append(str(rem // q))
            rem %= q
        if rem == 0:
            return f"{whole}." + ("".join(digits) or "0"), "terminates"
        start = seen[rem]
        text = f"{whole}." + "".join(digits[:start]) + "(" + "".join(digits[start:]) + ")"
        return text, f"repeats {len(digits) - start} digit(s)"
    rows = []
    for p, q in ((1, 3), (1, 8), (1, 7), (22, 7), (5, 12)):
        text, note = expand(p, q)
        rows.append((f"{p}/{q}", f"`{text}`", note))
    root2 = str(math.isqrt(2 * 10 ** 40))
    rows.append(("sqrt(2)", f"`{root2[0]}.{root2[1:]}...`", "never repeats (proved in the text)"))
    return _table(["number", "decimal expansion (parentheses mark the repeating block)", "behaviour"], rows, "lll")


def sqrt2_approximations():
    """Fractions closing in on sqrt(2) by repeatedly averaging x and 2/x."""
    x, rows = Fraction(1), []
    for k in range(5):
        rows.append((k, f"{x.numerator}/{x.denominator}" if x.denominator > 1 else "1",
                     f"{float(x):.12f}", f"{float(x * x):.12f}", _sci(float(x * x - 2), 2)))
        x = (x + 2 / x) / 2
    return _table(["step", "fraction x", "x as decimal", "x squared", "x squared minus 2"], rows, "rrrrr")


# --------------------------------------------------------------------- sets

def set_operations():
    a, b = {1, 2, 3, 4}, {3, 4, 5}
    ordered = lambda s: "{" + ", ".join(str(v) for v in sorted(s)) + "}"
    squares_small = {n for n in range(-10, 11) if n * n < 20}
    rows = [
        ("A", ordered(a), f"|A| = {len(a)}"),
        ("B", ordered(b), f"|B| = {len(b)}"),
        ("union A U B", ordered(a | b), f"{len(a | b)} elements"),
        ("intersection A n B", ordered(a & b), f"{len(a & b)} elements"),
        ("difference A \\ B", ordered(a - b), f"{len(a - b)} elements"),
        ("{3, 4} subset of A?", str({3, 4} <= a).lower(), ""),
        ("{3, 4} subset of B?", str({3, 4} <= b).lower(), ""),
        ("{n in Z : n^2 < 20}", ordered(squares_small), f"{len(squares_small)} elements"),
        ("{1, 2, 2, 3, 2}", ordered({1, 2, 2, 3, 2}), "repeats collapse: 3 elements"),
        ("A x B (ordered pairs)", f"{len(a) * len(b)} pairs", "|A| * |B|"),
    ]
    return _table(["expression", "value", "note"], rows, "lll")


# ---------------------------------------------------------------- exponents

def exponent_rules():
    rows = []

    def row(name, left_text, left, right_text, right):
        rows.append((name, left_text, f"{left:.10g}", right_text, f"{right:.10g}", "yes" if math.isclose(left, right, rel_tol=1e-12) else "NO"))

    row("product", "2^3 * 2^5", 2 ** 3 * 2 ** 5, "2^(3+5)", 2 ** (3 + 5))
    row("quotient", "3^7 / 3^4", 3 ** 7 / 3 ** 4, "3^(7-4)", 3 ** (7 - 4))
    row("power of power", "(2^3)^2", (2 ** 3) ** 2, "2^(3*2)", 2 ** (3 * 2))
    row("power of product", "(2*5)^3", (2 * 5) ** 3, "2^3 * 5^3", 2 ** 3 * 5 ** 3)
    row("zero exponent", "7^0", 7.0 ** 0, "1", 1.0)
    row("negative exponent", "2^(-3)", 2.0 ** -3, "1 / 2^3", 1 / 2 ** 3)
    row("fractional exponent", "9^(1/2)", 9.0 ** 0.5, "sqrt(9)", math.sqrt(9))
    row("fractional, then power", "(2^(1/2))^2", (2.0 ** 0.5) ** 2, "2^1", 2.0)
    return _table(["rule", "left side", "value", "right side", "value", "equal"], rows, "llrlrl")


def scientific_notation():
    items = [
        ("parameters in a large model", 7e9, 0),
        ("training tokens seen", 2e12, 0),
        ("a typical learning rate", 3e-4, 4),
        ("a small constant added to avoid dividing by zero", 1e-8, 8),
    ]
    rows = []
    for name, v, places in items:
        mant, exp = f"{v:e}".split("e")
        mant = f"{float(mant):g}"
        plain = f"{v:,.0f}" if places == 0 else f"{v:.{places}f}"
        rows.append((name, f"`{plain}`", f"{mant} x 10^{int(exp)}", f"`{v:g}`"))
    return _table(["quantity", "written out", "scientific notation", "as a program writes it"], rows, "llll")


def parameter_counts():
    """A square grid of d x d numbers plus d extra numbers, repeated 32 times."""
    rows = []
    for d in (1024, 4096, 8192):
        one = d * d + d
        total = 32 * one
        rows.append((f"{d:,}", f"{one:,}", _sci(one, 2), f"{total:,}", _sci(total, 2), f"{total * 4 / 2 ** 30:.2f}"))
    return _table(["d", "numbers per grid d*d + d", "scientific", "32 grids", "scientific", "GiB at 4 bytes each"], rows, "rrrrrr")


def memory_for_7b():
    n = 7e9
    rows = []
    for bytes_each, name in ((4, "32-bit float"), (2, "16-bit float"), (1, "8-bit integer")):
        total = n * bytes_each
        rows.append((name, bytes_each, _sci(total, 2), f"{total / 1e9:.2f}", f"{total / 2 ** 30:.2f}"))
    gap = (2 ** 30 / 1e9 - 1) * 100
    return (_table(["format", "bytes each", "total bytes", "GB (10^9)", "GiB (2^30)"], rows, "lrrrr")
            + f"\n\nOne GiB is {2 ** 30:,} bytes, which is {gap:.1f} percent more than one GB of {10 ** 9:,}.")


# --------------------------------------------------------------- logarithms

def e_limit():
    rows = []
    for k in (0, 1, 2, 3, 4, 6, 8, 12, 15, 16):
        n = 10 ** k
        v = (1 + 1 / n) ** n
        rows.append((f"10^{k}", f"{v:.12f}", _sci(v - math.e, 2)))
    head = f"e = {math.e:.12f}\n\n"
    return head + _table(["n", "(1 + 1/n)^n", "difference from e"], rows, "lrr")


def log_rules():
    a, b, k = 8.0, 32.0, 3.5
    rows = []

    def row(name, left_text, left, right_text, right):
        rows.append((name, left_text, f"{left:.10f}", right_text, f"{right:.10f}", _sci(abs(left - right), 1)))

    row("log of a product", "log(8 * 32)", math.log(a * b), "log 8 + log 32", math.log(a) + math.log(b))
    row("log of a quotient", "log(8 / 32)", math.log(a / b), "log 8 - log 32", math.log(a) - math.log(b))
    row("log of a power", "log(8^3.5)", math.log(a ** k), "3.5 * log 8", k * math.log(a))
    row("log of 1", "log 1", math.log(1.0), "0", 0.0)
    row("change of base", "log_2 32", math.log2(b), "log 32 / log 2", math.log(b) / math.log(2))
    row("change of base", "log_10 1000", math.log10(1000.0), "log 1000 / log 10", math.log(1000.0) / math.log(10))
    row("reciprocal", "log(1/8)", math.log(1 / a), "-log 8", -math.log(a))
    return _table(["rule", "left side", "value", "right side", "value", "difference"], rows, "llrlrr")


def exp_log_inverse():
    rows = []
    for x in (-2.0, -0.5, 0.0, 1.0, 2.5):
        y = math.exp(x)
        rows.append((f"{x}", f"{y:.6f}", f"{math.log(y):.6f}"))
    out = ["Going exp then log returns the starting number:", "",
           _table(["x", "exp(x)", "log(exp(x))"], rows), "",
           "Going log then exp does too, for positive inputs:", ""]
    rows = []
    for x in (0.1, 0.5, 1.0, math.e, 10.0, 1000.0):
        y = math.log(x)
        rows.append((f"{x:.6g}", f"{y:.6f}", f"{math.exp(y):.6g}"))
    out.append(_table(["x", "log(x)", "exp(log(x))"], rows))
    return "\n".join(out)


def digits_and_bits():
    rows = []
    for name, v in (("a count of 1,000", 1e3), ("a model with 7 x 10^9 parameters", 7e9),
                    ("a corpus of 2 x 10^12 tokens", 2e12), ("an epsilon of 10^-8", 1e-8),
                    ("a vocabulary of 50,000 tokens", 50000.0), ("a vocabulary of 128,000 tokens", 128000.0),
                    ("one million", 1e6)):
        rows.append((name, f"{math.log10(v):.4f}", f"{math.log2(v):.4f}", math.ceil(math.log2(v)) if v >= 1 else "-"))
    return _table(["quantity x", "log_10 x", "log_2 x", "bits to number x things (ceil of log_2)"], rows, "lrrr")


# ------------------------------------------------------ function notation

def function_evaluation():
    f = lambda x: 3 * x - 2
    g = lambda x: x * x
    h = lambda x, w, b: w * x + b
    rows = [(x, f(x), g(x), h(x, 3, -2), h(x, 0.5, 1)) for x in (-1, 0, 1, 2, 2.5)]
    return _table(["x", "f(x) = 3x - 2", "g(x) = x^2", "h(x; w=3, b=-2)", "h(x; w=0.5, b=1)"], rows, "rrrrr")


DATASET = [(0.0, 1.2), (1.0, 2.8), (2.0, 5.3), (3.0, 7.1), (4.0, 8.8)]


def _mse(w, b=1.0):
    return sum((w * x + b - y) ** 2 for x, y in DATASET) / len(DATASET)


def loss_dataset():
    w, b = 2.0, 1.0
    rows, total = [], 0.0
    for n, (x, y) in enumerate(DATASET, start=1):
        pred = w * x + b
        sq = (pred - y) ** 2
        total += sq
        rows.append((n, f"{x:.1f}", f"{y:.1f}", f"{pred:.1f}", f"{pred - y:+.1f}", f"{sq:.2f}", f"{total:.2f}"))
    table = _table(["n", "x_n", "y_n", "f(x_n) = 2 x_n + 1", "error", "error squared", "running sum"], rows, "rrrrrrr")
    mean = total / len(DATASET)
    return table + f"\n\nSum of squared errors {total:.2f}, divided by N = {len(DATASET)}: L = {mean:.3f}."


def loss_as_function_of_w():
    rows = [(f"{w:.1f}", f"{_mse(w):.4f}") for w in (1.0, 1.5, 2.0, 2.5, 3.0)]
    return _table(["w", "L(w) with b = 1"], rows, "rr")


# ----------------------------------------------------------------- summation

def sum_expanded():
    rows, run = [], 0
    for i in range(1, 6):
        run += i * i
        rows.append((i, f"{i}^2", i * i, run))
    closed = 5 * 6 * 11 // 6
    return (_table(["i", "term i^2", "value", "running sum"], rows, "rlrr")
            + f"\n\nThe sum is {run}; the closed form n(n+1)(2n+1)/6 with n = 5 gives {closed}.")


def sum_closed_form():
    rows = []
    for n in (4, 10, 100, 1000, 10 ** 6):
        direct = sum(range(1, n + 1))
        rows.append((f"{n:,}", f"{direct:,}", f"{n * (n + 1) // 2:,}"))
    return _table(["n", "1 + 2 + ... + n by adding", "n(n+1)/2"], rows, "rrr")


def sum_linearity():
    x, y = [2, 4, 6, 8], [1, -1, 3, 5]
    c = 3
    left = sum(c * xi + yi for xi, yi in zip(x, y))
    right = c * sum(x) + sum(y)
    rows = [(i + 1, x[i], y[i], c * x[i] + y[i]) for i in range(4)]
    return (_table(["i", "x_i", "y_i", "3 x_i + y_i"], rows, "rrrr")
            + f"\n\nSum of the last column: {left}. Three times the sum of x ({sum(x)}) plus the sum of y ({sum(y)}): {right}.")


def double_sum():
    a = [[1, 2, 3], [4, 5, 6]]
    row_sums = [sum(r) for r in a]
    col_sums = [sum(a[i][j] for i in range(2)) for j in range(3)]
    rows = [(f"row {i + 1}", *a[i], row_sums[i]) for i in range(2)]
    rows.append(("column sums", *col_sums, sum(col_sums)))
    return (_table(["", "j = 1", "j = 2", "j = 3", "sum"], rows, "lrrrr")
            + f"\n\nSumming rows first gives {sum(row_sums)}; summing columns first gives {sum(col_sums)}.")


def term_counts():
    rows = [(f"{a} to {b}", len(range(a, b + 1)), b - a + 1 if b >= a else 0) for a, b in ((1, 10), (0, 9), (3, 3), (4, 3), (-2, 2))]
    return _table(["i from .. to", "terms by counting", "max(b - a + 1, 0)"], rows, "lrr")


# ------------------------------------------------------------------- products

def product_expanded():
    rows, run = [], Fraction(1)
    for i in range(1, 7):
        factor = Fraction(i + 1, i)
        run *= factor
        rows.append((i, f"{factor.numerator}/{factor.denominator}", str(run)))
    return (_table(["i", "factor (i+1)/i", "running product"], rows, "rrr")
            + "\n\nEvery numerator cancels the next denominator, so the product of the first n factors is n + 1.")


def factorials():
    rows, limit = [], 2 ** 63 - 1
    first_overflow = None
    for n in (0, 1, 2, 3, 5, 10, 15, 20, 21, 25):
        v = math.factorial(n)
        fits = v <= limit
        if not fits and first_overflow is None:
            first_overflow = n
        rows.append((n, f"{v:,}", "yes" if fits else "NO"))
    smallest = next(n for n in range(1, 100) if math.factorial(n) > limit)
    return (_table(["n", "n!", "fits in a signed 64-bit integer"], rows, "rrr")
            + f"\n\nThe first n whose n! exceeds 2^63 - 1 = {limit:,} is n = {smallest}.")


# -------------------------------------------------- products of probabilities

TOKEN_PROBS = [0.5, 0.2, 0.1, 0.4]


def log_of_product():
    rows, prod, total = [], 1.0, 0.0
    for i, p in enumerate(TOKEN_PROBS, start=1):
        prod *= p
        total += math.log(p)
        rows.append((i, p, f"{math.log(p):.4f}", f"{prod:.4f}", f"{total:.4f}", f"{math.exp(total):.4f}"))
    return _table(["i", "p_i", "log p_i", "running product", "running sum of logs", "exp of that sum"], rows, "rrrrrr")


def underflow_rows(ns=(10, 38, 45, 46, 100, 300, 323, 324, 400)):
    """(n, product in 64 bits, product in 32 bits, sum of natural logs) for n factors of 0.1."""
    out = []
    for n in ns:
        p64, p32 = 1.0, 1.0
        for _ in range(n):
            p64 *= 0.1
            p32 = _f32(p32 * _f32(0.1))
        out.append((n, p64, p32, n * math.log(0.1)))
    return out


def underflow():
    rows = []
    for n, p64, p32, s in underflow_rows():
        rows.append((n, f"1e-{n}", _sci(p64, 3), _sci(p32, 3), f"{s:.2f}"))
    return _table(["n factors of 0.1", "true product", "multiplied in 64 bits", "multiplied in 32 bits", "sum of n natural logs"], rows, "rlrrr")


def exp_limits():
    def exp64(x):
        try:
            return math.exp(x)
        except OverflowError:
            return math.inf
    rows = []
    for x in (10, 50, 88, 89, 100, 709, 710, -87, -104, -745, -746):
        v64 = exp64(x)
        v32 = _f32(v64) if math.isfinite(v64) else math.inf
        rows.append((x, _sci(v64, 3), _sci(v32, 3)))
    return _table(["x", "exp(x) in 64 bits", "exp(x) in 32 bits"], rows, "lrr")


def stable_log_sum_demo():
    """A document of 400 tokens, each with probability 0.1: the log-likelihood and the per-token average."""
    n = 400
    total = sum(math.log(0.1) for _ in range(n))
    prod = 1.0
    for _ in range(n):
        prod *= 0.1
    return (f"Direct product of {n} factors of 0.1 in 64 bits: {prod!r}.\n\n"
            f"Sum of {n} logs: {total:.4f}; average per token: {total / n:.4f}; "
            f"and exp of that average: {math.exp(total / n):.4f}.")


def common_mistakes():
    """Four identities that look true and are not, each with the values that expose it."""
    x = [1, 2, 3]
    rows = [
        ("log(a + b) = log a + log b", "a = 2, b = 3", f"{math.log(2 + 3):.4f}", f"{math.log(2) + math.log(3):.4f}"),
        ("(a + b)^2 = a^2 + b^2", "a = 2, b = 3", f"{(2 + 3) ** 2}", f"{2 ** 2 + 3 ** 2}"),
        ("1 / (a + b) = 1/a + 1/b", "a = 2, b = 3", f"{1 / (2 + 3):.4f}", f"{1 / 2 + 1 / 3:.4f}"),
        ("sum of x_i^2 = (sum of x_i)^2", "x = (1, 2, 3)", f"{sum(v * v for v in x)}", f"{sum(x) ** 2}"),
    ]
    return _table(["claimed identity", "values", "left side", "right side"], rows, "llrr")


# ------------------------------------------------------------- figure data

def exp_curve(x):
    return math.exp(x)


def log_curve(x):
    return math.log(x)


def number_line_points():
    """(label, value, kind) for the number line figure."""
    pts = [(str(n), float(n), "integer") for n in range(-2, 5)]
    pts += [("-3/2", -3 / 2, "rational"), ("1/2", 1 / 2, "rational"), ("7/4", 7 / 4, "rational"), ("10/3", 10 / 3, "rational")]
    pts += [("sqrt 2", math.sqrt(2), "irrational"), ("e", math.e, "irrational"), ("pi", math.pi, "irrational")]
    return pts
