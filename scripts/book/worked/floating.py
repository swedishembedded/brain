# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Floating Point and Numerical Error".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: floating.<name> -->` markers. Helpers with a leading underscore
(or that take arguments) are shared with the chapter's figures, so a picture
and the table beside it cannot disagree.

32-bit arithmetic is emulated by rounding a 64-bit result to the nearest
32-bit number (`struct` pack and unpack). For one addition, subtraction,
multiplication, division or square root of 32-bit operands this is exactly the
correctly rounded 32-bit result, because a 64-bit significand has more than
twice the bits plus two of a 32-bit one, so rounding twice cannot differ from
rounding once.

Swedish Embedded AB implements solutions for numerically trustworthy machine
learning code. If your team needs expertise in floating-point error analysis,
mixed-precision accumulation and numerically stable loss functions, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math
import struct
from decimal import Decimal, getcontext
from fractions import Fraction
from functools import lru_cache

import bookrng

getcontext().prec = 60

UNIFORM_NAME = "floating.sum_uniform"   # names the generator of the uniform summands
U32 = 2.0 ** -24   # unit round-off of the 32-bit format
U64 = 2.0 ** -53   # unit round-off of the 64-bit format
INF = math.inf
NAN = math.nan


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _g(v, d=9):
    """A float in plain or scientific notation, whichever is shorter, with d significant digits."""
    if v != v:
        return "nan"
    if v in (INF, -INF):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{d}g}"


def f32(x):
    """The nearest 32-bit float to x (ties to even); infinity when x is beyond the largest."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(INF, x)


def bits32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits64(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _fmt_for(name):
    return f32 if name == "32-bit" else (lambda v: v)


def _exact(x):
    """A float as an exact fraction."""
    return Fraction(x)


def _rel(approx, exact):
    """Relative error |approx - exact| / |exact| for exact nonzero; approx may be a float or a Fraction."""
    if isinstance(approx, float) and (approx != approx or approx in (INF, -INF)):
        return INF
    return abs(Fraction(approx) - Fraction(exact)) / abs(Fraction(exact))


def _fr(x):
    """A Fraction as a float for printing (correctly rounded)."""
    return float(x)


# ------------------------------------------------ rounding to any format

def round_to_format(x, p, emin, emax):
    """Round the exact rational x to the nearest number of a binary format, ties to even.

    The format has p significand bits, normal exponents emin..emax and
    subnormals below 2^emin. Returns a Fraction, or the string "inf" / "-inf"
    when the rounded result would exceed the largest number.
    """
    x = Fraction(x)
    if x == 0:
        return Fraction(0)
    sign = 1 if x > 0 else -1
    a = abs(x)
    e = 0
    while a >= 2 ** (e + 1):
        e += 1
    while a < 2 ** e:
        e -= 1
    e = max(e, emin)                       # below 2^emin the spacing stops shrinking: subnormals
    spacing = Fraction(2) ** (e - (p - 1))
    n = round(a / spacing)                 # Fraction rounding: nearest, ties to even
    r = n * spacing
    largest = (2 - Fraction(2) ** (1 - p)) * Fraction(2) ** emax
    if r > largest:
        return "inf" if sign > 0 else "-inf"
    return sign * r


def _round32(x):
    return round_to_format(x, 24, -126, 127)


# ------------------------------------------------------- the opening problem

def surprises():
    d = Decimal(0.1)
    rows = [
        ["0.1 + 0.2", repr(0.1 + 0.2)],
        ["0.1 + 0.2 == 0.3", str(0.1 + 0.2 == 0.3)],
        ["0.1 + 0.2 - 0.3", _sci(0.1 + 0.2 - 0.3)],
        ["0.1 * 3", repr(0.1 * 3)],
        ["sum of ten copies of 0.1", repr(sum([0.1] * 10))],
        ["(0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3)", str((0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3))],
        ["the number stored for 0.1, exactly", f"{d:.40f}"],
    ]
    return _table(["expression (64-bit floating point)", "result"], rows, "ll")


def _naive32(xs):
    s = 0.0
    for x in xs:
        s = f32(s + x)
    return s


def _pairwise32(xs):
    layer = list(xs)
    while len(layer) > 1:
        nxt = [f32(layer[i] + layer[i + 1]) for i in range(0, len(layer) - 1, 2)]
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    return layer[0] if layer else 0.0


def _kahan32(xs):
    s, c = 0.0, 0.0
    for x in xs:
        y = f32(x - c)
        t = f32(s + y)
        c = f32(f32(t - s) - y)
        s = t
    return s


def _sum64(xs):
    s = 0.0
    for x in xs:
        s += x
    return s


def _exact_decimal(q, places):
    """The Fraction q to `places` decimals, rounded half up, from exact integer arithmetic."""
    scaled = (q * 10 ** places * 2 + 1) // 2
    whole, frac = divmod(scaled, 10 ** places)
    return f"{whole}.{frac:0{places}d}"


def million_tenths():
    n = 10 ** 6
    tenth32 = f32(0.1)
    xs = [tenth32] * n
    exact = n * Fraction(tenth32)
    s64 = 0.0
    for _ in range(n):
        s64 += 0.1
    rows = [["64-bit, left to right", repr(s64), _sci(_rel(s64, n * Fraction(0.1)))],
            ["32-bit, left to right", _g(_naive32(xs), 9), _sci(_rel(_naive32(xs), exact))],
            ["32-bit, pairwise", _g(_pairwise32(xs), 9), _sci(_rel(_pairwise32(xs), exact))],
            ["32-bit, compensated (Kahan)", _g(_kahan32(xs), 9), _sci(_rel(_kahan32(xs), exact))],
            ["32-bit terms added in a 64-bit accumulator, total rounded to 32 bits", _g(f32(_sum64(xs)), 9),
             _sci(_rel(f32(_sum64(xs)), exact))]]
    return (f"The sum of 10^6 copies of the stored value of 0.1. Relative errors are measured against the exact "
            f"sum of the stored copies, computed in exact rational arithmetic (a Fraction): "
            f"{float(exact):.6f} for the 32-bit copies of 0.1 ({_g(float(tenth32), 12)} each) and, for the first row, "
            f"{_exact_decimal(n * Fraction(0.1), 21)} (exact, 21 decimals) for the 64-bit copies. The 64-bit accumulator of the last row holds "
            f"{_sum64(xs):.9f} before it is rounded to the nearest 32-bit number.\n\n"
            + _table(["method", "computed sum", "relative error"], rows, "lrr"))


# --------------------------------------------------------- the toy format

TOY = dict(p=3, emin=-2, emax=1)


def toy_numbers(p=3, emin=-2, emax=1):
    """(normal numbers, subnormals) of the toy format, as Fractions, positive part only."""
    normals, subs = [], []
    for e in range(emin, emax + 1):
        for m in range(2 ** (p - 1)):
            normals.append((e, Fraction(2 ** (p - 1) + m, 2 ** (p - 1)) * Fraction(2) ** e))
    step = Fraction(2) ** (emin - (p - 1))
    for k in range(1, 2 ** (p - 1)):
        subs.append(k * step)
    return normals, subs


def toy_grid():
    p, emin, emax = TOY["p"], TOY["emin"], TOY["emax"]
    normals, subs = toy_numbers(**TOY)
    rows = []
    for e in range(emin, emax + 1):
        vals = [v for ee, v in normals if ee == e]
        rows.append([f"[{Fraction(2) ** e}, {Fraction(2) ** (e + 1)})".replace("Fraction", ""),
                     str(float(Fraction(2) ** (e - (p - 1)))),
                     ", ".join(f"{float(v):g}" for v in vals)])
    rows.append(["below 0.25 (subnormal)", str(float(Fraction(2) ** (emin - (p - 1)))),
                 "0, " + ", ".join(f"{float(v):g}" for v in subs)])
    return (f"A toy format: significand 1.xx (p = {p} bits), exponents {emin} to {emax}. "
            f"{len(normals)} positive normal numbers, largest {float(normals[-1][1]):g}.\n\n"
            + _table(["binade", "spacing in the binade", "the numbers"], rows, "lrl"))


def toy_rounding():
    p, emin, emax = TOY["p"], TOY["emin"], TOY["emax"]
    u = Fraction(1, 2 ** p)
    rows = []
    for x in (Fraction(3, 10), Fraction(11, 10), Fraction(9, 8), Fraction(11, 8), Fraction(29, 10), Fraction(10, 3),
              Fraction(1, 10)):
        r = round_to_format(x, p, emin, emax)
        err = abs(r - x) / x
        rows.append([f"{float(x):g}", f"{float(r):g}", _sci(float(err), 3), f"{float(err / u):.3f}"])
    return (f"Rounding to the toy format. u = 2^-{p} = {float(u):g}; 1.125 and 1.375 are exact ties.\n\n"
            + _table(["x", "nearest toy number", "relative error", "error / u"], rows))


# --------------------------------------------------------- the real formats

def fp32_bit_patterns():
    vals = [1.0, 0.5, -2.0, 0.1, 3.14159265358979, 2.0 ** -126, 2.0 ** -149, 3.4028234663852886e38, INF, NAN]
    names = ["1", "0.5", "-2", "0.1", "3.14159265358979", "2^-126", "2^-149", "largest", "inf", "nan"]
    rows = []
    for name, v in zip(names, vals):
        b = bits32(f32(v))
        sign, exp, frac = b >> 31, (b >> 23) & 0xFF, b & 0x7FFFFF
        stored = f32(v)
        if stored != stored or stored in (INF, -INF):
            exact = _g(stored)
        else:
            exact = format(Decimal(stored), ".12g") if abs(stored) < 1e-30 or abs(stored) > 1e30 \
                else format(Decimal(stored), ".20g")
        rows.append([name, f"{b:08x}", str(sign), f"{exp:08b} ({exp})", f"{frac:023b}", exact])
    return _table(["real number", "hex", "s", "exponent field", "fraction field", "value stored"], rows,
                  "lllll" + "l")


def _eps_by_halving(rnd):
    e = 1.0
    while rnd(1.0 + e / 2.0) != 1.0:
        e /= 2.0
    return e


def format_parameters():
    fmts = [("32-bit", 24, 8, -126, 127, f32), ("64-bit", 53, 11, -1022, 1023, lambda v: v)]
    cols = []
    for name, p, eb, emin, emax, rnd in fmts:
        largest = (2 - 2.0 ** (1 - p)) * 2.0 ** emax
        cols.append([str(p), str(eb), f"{emin} to {emax}", _sci(2.0 ** -p, 3), _sci(2.0 ** (1 - p), 3),
                     _sci(_eps_by_halving(rnd), 3), f"{p * math.log10(2):.2f}", _sci(2.0 ** emin, 3),
                     _sci(2.0 ** (emin - p + 1), 3), _sci(largest, 3)])
    labels = ["significand bits p", "exponent bits", "exponents of normal numbers", "u = 2^-p", "eps = 2^(1-p)",
              "eps found by halving until 1 + eps/2 == 1", "decimal digits (p log10 2)", "smallest normal number",
              "smallest subnormal number", "largest number"]
    return _table(["quantity", "32-bit", "64-bit"], [[l, a, b] for l, a, b in zip(labels, cols[0], cols[1])])


def rounding_error_table():
    pi = Fraction("3.14159265358979323846264338327950288")
    rows = []
    for name, x in (("0.1", Fraction(1, 10)), ("1/3", Fraction(1, 3)), ("pi", pi), ("100.1", Fraction(1001, 10)),
                    ("0.001", Fraction(1, 1000)), ("16777217", Fraction(16777217)), ("5e-39", Fraction(5, 10 ** 39))):
        r = _round32(x)
        err = abs(r - x) / x
        rows.append([name, _g(float(r), 10), _sci(float(abs(r - x)), 2), _sci(float(err), 2), f"{float(err / Fraction(1, 2 ** 24)):.3f}"])
    return ("Rounding exact real numbers to the 32-bit format, ties to even. u = 2^-24 = "
            + _sci(U32, 3) + ".\n\n" + _table(["x", "nearest 32-bit number", "absolute error", "relative error", "error / u"], rows)
            + "\n\nThe last row lies in the subnormal range, where the bound u no longer holds.")


def rounding_agreement():
    """Check the exact rounder against the hardware conversion on random doubles."""
    name = "floating.rounding_agreement"
    rng = bookrng.generator(name)
    n, bad = 20000, 0
    for _ in range(n):
        a = rng.uniform()
        b_ = rng.uniform()
        x = (2.0 * a - 1.0) * 10.0 ** (60.0 * b_ - 30.0)
        a = _round32(Fraction(x))
        b = f32(x)
        if (a == "inf" or a == "-inf") or Fraction(b) != a:
            bad += 1
    return (f"Exact rounding of {n} random doubles to 32 bits against the hardware conversion: {bad} disagreements. "
            f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x}. Each double is x = (2a - 1) * 10^(60b - 30) "
            f"from two draws, a then b, both `uniform()`.")


def ties_to_even():
    rows = []
    for name, x in (("1 + 2^-24", Fraction(1) + Fraction(1, 2 ** 24)),
                    ("1 + 3 * 2^-24", Fraction(1) + Fraction(3, 2 ** 24)),
                    ("2^24 + 1", Fraction(2 ** 24 + 1)),
                    ("2^24 + 3", Fraction(2 ** 24 + 3)),
                    ("1 + 2^-24 + 2^-40", Fraction(1) + Fraction(1, 2 ** 24) + Fraction(1, 2 ** 40))):
        r = _round32(x)
        rows.append([name, f"{float(r):.10f}".rstrip("0").rstrip("."), f"{bits32(f32(float(x))) & 1}"])
    return _table(["exact value", "rounds to", "last bit of the result"], rows, "lrr")


# ------------------------------------------------ the edges of the range

def special_values():
    nan, inf = NAN, INF
    big = 1e308
    rows = [
        ["1e308 * 10", _g(big * 10)],
        ["-1e308 * 10", _g(-big * 10)],
        ["1 / inf", _g(1 / inf)],
        ["inf - inf", _g(inf - inf)],
        ["0 * inf", _g(0 * inf)],
        ["nan + 1", _g(nan + 1)],
        ["nan == nan", str(nan == nan)],
        ["nan != nan", str(nan != nan)],
        ["nan < 1  and  nan > 1", f"{nan < 1}  and  {nan > 1}"],
        ["max(nan, 1)", _g(max(nan, 1.0))],
        ["max(1, nan)", _g(max(1.0, nan))],
        ["sum of [1, nan, 2]", _g(sum([1.0, nan, 2.0]))],
        ["sorted([3, nan, 1, 2])", str(sorted([3.0, nan, 1.0, 2.0])).replace("nan", "nan")],
        ["inf > 1e308", str(inf > 1e308)],
    ]
    return _table(["expression (64-bit)", "result"], rows, "ll")


def gradual_underflow():
    rows = []
    for name, rnd, p, emin, xs in (("32-bit", f32, 24, -126, (-126, -127, -140, -149, -150)),
                                   ("64-bit", lambda v: v, 53, -1022, (-1022, -1023, -1050, -1074, -1075))):
        for k in xs:
            x = 2.0 ** k
            r = rnd(x)
            if r == 0:
                rows.append([name, f"2^{k}", "0", "-", "-"])
                continue
            sp = 2.0 ** (max(math.floor(math.log2(r)), emin) - (p - 1))
            rows.append([name, f"2^{k}", _g(r, 6), _sci(sp, 2), f"{sp / r:.3g}"])
    return _table(["format", "exact value", "stored", "spacing at that size", "spacing / value"], rows)


def overflow_edge():
    top = 3.4028234663852886e38
    rows = []
    for text in ("3.4028235e38", "3.40282356e38", "3.40282357e38", "1e39"):
        rows.append([text, _g(f32(float(text)), 10)])
    return (f"Largest 32-bit number: {top!r}. Decimal inputs near the top:\n\n"
            + _table(["decimal input", "rounded to 32 bits"], rows, "lr"))


# ------------------------------------------------ summation

@lru_cache(maxsize=None)
def _sum_errors(kind, ns):
    """Relative errors of naive, pairwise, and Kahan 32-bit sums at the prefix lengths in ns.

    kind "tenths": every term is the stored 32-bit value of 0.1.
    kind "uniform": terms drawn uniformly from [0, 1) by the book generator and rounded to 32 bits;
    the row for n sums the first n terms of one list of max(ns) draws.
    """
    n_max = max(ns)
    if kind == "tenths":
        xs = [f32(0.1)] * n_max
    else:
        rng = bookrng.generator(UNIFORM_NAME)
        xs = [f32(rng.uniform()) for _ in range(n_max)]
    out = {}
    s, k_s, c = 0.0, 0.0, 0.0
    marks = set(ns)
    exact = Fraction(0)
    naive_at, kahan_at, exact_at = {}, {}, {}
    for i, x in enumerate(xs, 1):
        s = f32(s + x)
        y = f32(x - c)
        t = f32(k_s + y)
        c = f32(f32(t - k_s) - y)
        k_s = t
        exact += Fraction(x)
        if i in marks:
            naive_at[i], kahan_at[i], exact_at[i] = s, k_s, exact
    for n in ns:
        pw = _pairwise32(xs[:n])
        out[n] = (exact_at[n], _rel(naive_at[n], exact_at[n]), _rel(pw, exact_at[n]), _rel(kahan_at[n], exact_at[n]))
    return out


def sum_error_curves(ns):
    """Relative error (naive, pairwise, Kahan) of the 32-bit sum of n copies of 0.1, for each n in ns."""
    errs = _sum_errors("tenths", tuple(ns))
    return [(n,) + tuple(errs[n][1:]) for n in ns]


def sum_methods():
    rows = []
    for kind, label, ns in (("tenths", "copies of 0.1", (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6)),
                            ("uniform", "uniform in [0, 1)", (10 ** 3, 10 ** 4, 10 ** 5))):
        errs = _sum_errors(kind, ns)
        for n in ns:
            ex, a, b, c = errs[n]
            rows.append([label, f"10^{round(math.log10(n))}", _sci(float(ex), 4), _sci((n - 1) * U32, 1),
                         _sci(float(a)), _sci(float(b)), _sci(float(c))])
    return ("Relative error of a 32-bit sum of n terms, against the exact sum of the stored terms (exact rational "
            "arithmetic). Copies of 0.1: the 32-bit number nearest 0.1, n times. Uniform: the first n of 10^5 values "
            f"`uniform()` drawn in order from the book's SplitMix64 with seed {bookrng.seed_of(UNIFORM_NAME):#018x}, "
            "each rounded to 32 bits; the three row sizes use prefixes of the same list. Pairwise is the layer-by-layer "
            "algorithm below.\n\n"
            + _table(["terms", "n", "exact sum", "worst-case bound (n-1)u", "left to right", "pairwise", "Kahan"], rows,
                     "llrrrrr"))


def kahan_trace():
    x = f32(4e-8)
    s, c = 1.0, 0.0
    naive = 1.0
    exact = Fraction(1)
    rows = []
    for step in range(1, 9):
        y = f32(x - c)
        t = f32(s + y)
        c_new = f32(f32(t - s) - y)
        naive = f32(naive + x)
        exact += Fraction(x)
        rows.append([step, _sci(y, 3), f"{t - 1:.4e}", _sci(c_new, 3), f"{naive - 1:.4e}", f"{float(exact - 1):.4e}"])
        s, c = t, c_new
    return (f"The loop of `kahan_sum` started from s = 1, c = 0 (instead of s = 0, c = 0) in 32 bits, adding "
            f"x = f32(4e-8) = {float(x):.10e} eight times (half the gap above 1 is "
            f"{U32:.3e}, so one such x alone is lost). Columns show the part above 1: 'Kahan s - 1' is the stored s minus 1 "
            f"(exact, by Sterbenz), 'left to right' is the plain 32-bit sum started at 1, and 'exact sum - 1' is "
            f"k * f32(4e-8) for step k in exact rational arithmetic, the sum of the stored copies, not k * 4e-8.\n\n"
            + _table(["step", "y = x - c", "Kahan s - 1", "new c", "left to right s - 1", "exact sum - 1"], rows))


def ch1_loose_ends():
    rows = []
    for k in (8, 12, 15, 16):
        n = 10.0 ** k
        direct = (1.0 + 1.0 / n) ** n
        stable = math.exp(n * math.log1p(1.0 / n))
        rows.append([f"10^{k}", f"{direct:.12f}", f"{stable:.12f}", _sci(stable - math.e)])
    first = _table(["n", "(1 + 1/n)^n directly", "exp(n * log1p(1/n))", "stable minus e"], rows)
    n = 10 ** 6
    lg = f32(math.log(0.1))
    xs = [lg] * n
    exact = n * Fraction(lg)
    rows = [["left to right", _g(_naive32(xs), 9), _sci(_rel(_naive32(xs), exact))],
            ["pairwise", _g(_pairwise32(xs), 9), _sci(_rel(_pairwise32(xs), exact))],
            ["Kahan", _g(_kahan32(xs), 9), _sci(_rel(_kahan32(xs), exact))]]
    second = _table(["32-bit sum of 10^6 copies of log 0.1", "computed", "relative error"], rows, "lrr")
    return ("The limit (1 + 1/n)^n of chapter 1, in 64 bits:\n\n" + first + "\n\nThe sum of logs of chapter 1, "
            "exact sum " + f"{float(exact):.4f}" + ":\n\n" + second)


# ------------------------------------------------ cancellation

def one_plus_x():
    rows = []
    for k in (1, 4, 8, 12, 15, 16, 17):
        x = 10.0 ** -k
        d = (1.0 + x) - 1.0
        ex = Decimal(x).exp() - 1
        naive = math.exp(x) - 1.0
        stable = math.expm1(x)
        rows.append([f"1e-{k}", repr(d) if d != 0 else "0.0", _sci(float(abs(Decimal(d) - Decimal(x)) / Decimal(x))),
                     _sci(float(abs(Decimal(naive) - ex) / ex)), _sci(float(abs(Decimal(stable) - ex) / ex))])
    return ("In 64 bits. x is the 64-bit number nearest the decimal shown. The first error column compares (1 + x) - 1 "
            "with that stored x; the other two compare e^x - 1 computed naively and with `expm1` against the true "
            "value of e^x - 1 at the same stored x, computed in 60-digit decimal arithmetic. Errors are relative.\n\n"
            + _table(["x", "(1 + x) - 1", "relative error", "exp(x) - 1: error", "expm1(x): error"], rows))


def sterbenz():
    name = "floating.sterbenz"
    rng = bookrng.generator(name)
    n = 100000
    near_bad = far_bad = 0
    for _ in range(n):
        y = f32((1.0 + rng.uniform()) * 2.0 ** (rng.integer(41) - 20))
        x = f32(y * (0.5 + 1.5 * rng.uniform()))
        if not (y / 2 <= x <= 2 * y):
            x = y
        d = x - y                      # exact in 64 bits for 32-bit operands of similar size
        if f32(d) != d:
            near_bad += 1
        x2 = f32(y * (4.0 + rng.uniform()))
        d2 = x2 - y
        if f32(d2) != d2:
            far_bad += 1
    return (f"Subtracting two 32-bit numbers x, y: when y/2 <= x <= 2y the stored difference was inexact in "
            f"{near_bad} of {n} random cases; when x is 4 to 5 times y it was inexact in {far_bad} of {n}. "
            f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x}. Each case draws, in this order, "
            f"`uniform()` a, `integer(41)` k, `uniform()` b, `uniform()` c: y = f32((1 + a) * 2^(k - 20)), "
            f"x = f32(y * (0.5 + 1.5 b)) (replaced by y if that falls outside [y/2, 2y]), "
            f"and for the second count x = f32(y * (4 + c)). The difference x - y is exact in 64 bits for these "
            f"operands, and 'inexact' means it is not a 32-bit number.")


def quadratic():
    rows = []
    for k in (2, 4, 6, 8):
        b, c = 10.0 ** k, 1.0
        disc = b * b - 4 * c
        naive = (-b + math.sqrt(disc)) / 2
        q = -(b + math.sqrt(disc)) / 2
        stable = c / q
        D = Decimal(b)
        exact = (-D + (D * D - 4 * Decimal(c)).sqrt()) / 2
        rows.append([f"1e{k}", _g(naive, 10), _g(stable, 10), _g(float(exact), 10),
                     _sci(float(abs(Decimal(naive) - exact) / abs(exact))), _sci(float(abs(Decimal(stable) - exact) / abs(exact)))])
    return ("The smaller-magnitude root of x^2 + b x + 1 = 0, in 64 bits (b exactly 10^k, c = 1). The true root is "
            "(-b + sqrt(b^2 - 4)) / 2 evaluated in 60-digit decimal arithmetic from the same b; the errors are "
            "relative.\n\n"
            + _table(["b", "(-b + sqrt(b^2 - 4)) / 2", "c / q with q = -(b + sqrt(b^2 - 4)) / 2", "true root",
                      "error of the first", "error of the second"], rows))


def _var_methods(data, rnd):
    n = len(data)
    mean = 0.0
    sq = 0.0
    for x in data:
        mean = rnd(mean + x)
        sq = rnd(sq + rnd(x * x))
    mean = rnd(mean / n)
    naive = rnd(rnd(sq / n) - rnd(mean * mean))
    ss = 0.0
    for x in data:
        d = rnd(x - mean)
        ss = rnd(ss + rnd(d * d))
    two_pass = rnd(ss / n)
    m, m2 = 0.0, 0.0
    for k, x in enumerate(data, 1):
        d = rnd(x - m)
        m = rnd(m + rnd(d / k))
        m2 = rnd(m2 + rnd(d * rnd(x - m)))
    return naive, two_pass, rnd(m2 / n)


def variance():
    rows = []
    n = 1000
    for name, rnd, offsets in (("32-bit", f32, (1e1, 1e2, 1e3, 1e4)), ("64-bit", lambda v: v, (1e4, 1e8, 1e9))):
        for off in offsets:
            gen_name = f"floating.variance.{name}.{off:g}"
            rng = bookrng.generator(gen_name)
            data = [rnd(off + rng.normal()) for _ in range(n)]
            fr = [Fraction(x) for x in data]
            mu = sum(fr) / n
            true = sum((x - mu) ** 2 for x in fr) / n
            a, b, c = _var_methods(data, rnd)
            rows.append([name, f"10^{round(math.log10(off))}", f"{float(true):.6f}", f"{a:.6g}", _sci(float(_rel(a, true))),
                         _sci(float(_rel(b, true))), _sci(float(_rel(c, true))),
                         f"{bookrng.seed_of(gen_name):#018x}"])
    return (f"Variance of {n} values offset + z, z standard normal, population form (divide by n). The values are "
            f"rounded to the format after the addition. Each row has its own generator, the book's SplitMix64 with the "
            f"seed in the last column; the {n} values are its first {n} `normal()` draws in order. The true variance and "
            f"all errors are against the exact rational variance of the stored values (mean of the squared deviations "
            f"from the exact mean). Each format's arithmetic rounds every operation to that format; the one-pass "
            f"formula accumulates sum x and sum x^2 left to right, divides both by n, and subtracts.\n\n"
            + _table(["format", "offset", "true variance", "E[X^2] - E[X]^2", "its error", "two-pass error",
                      "Welford error", "seed"], rows))


# ------------------------------------------------ conditioning

def _dec_fn(name):
    return {"sqrt": lambda x: x.sqrt(), "exp": lambda x: x.exp(), "log": lambda x: x.ln(),
            "x^10": lambda x: x ** 10, "x - 1": lambda x: x - 1, "1/x": lambda x: 1 / x,
            "exp(x) - 1": lambda x: x.exp() - 1}[name]


def condition_numbers():
    delta = Decimal("1e-12")
    cases = [("sqrt(x)", "sqrt", Decimal(2), lambda x: 0.5), ("x^10", "x^10", Decimal(2), lambda x: 10.0),
             ("1/x", "1/x", Decimal(7), lambda x: 1.0), ("exp(x)", "exp", Decimal(30), lambda x: x),
             ("log(x)", "log", Decimal("1.001"), lambda x: 1 / math.log(x)),
             ("log(x)", "log", Decimal(2), lambda x: 1 / math.log(x)),
             ("x - 1", "x - 1", Decimal("1.001"), lambda x: x / (x - 1)),
             ("x - 1", "x - 1", Decimal(10 ** 8 + 1) / Decimal(10 ** 8), lambda x: x / (x - 1)),
             ("exp(x) - 1", "exp(x) - 1", Decimal("1e-8"), lambda x: x * math.exp(x) / math.expm1(x))]
    rows = []
    for label, fn, x, kappa in cases:
        f = _dec_fn(fn)
        y0, y1 = f(x), f(x * (1 + delta))
        measured = abs((y1 - y0) / y0) / delta
        k = kappa(float(x)) if not isinstance(kappa(float(x)), Decimal) else float(kappa(float(x)))
        rows.append([label, _g(float(x), 10), _g(float(k), 5), _g(float(measured), 5)])
    return ("Relative condition number |x f'(x) / f(x)| against the measured ratio (relative change of the "
            "output) / (relative change of the input) for an input change of 1e-12, computed with 60-digit "
            "arithmetic.\n\n" + _table(["f", "x", "|x f'(x) / f(x)|", "measured ratio"], rows))


# ------------------------------------------------ numerical differentiation

def _fwd(rnd, h, x=1.0):
    hh = rnd(h)
    xp = rnd(x + hh)
    step = xp - x
    if step == 0:
        return 0.0
    return (rnd(math.exp(xp)) - rnd(math.exp(x))) / step if rnd is not f32 else \
        (f32(math.exp(xp)) - f32(math.exp(x))) / step


def _cen(rnd, h, x=1.0):
    hh = rnd(h)
    xp, xm = rnd(x + hh), rnd(x - hh)
    step = xp - xm
    if step == 0:
        return 0.0
    return (rnd(math.exp(xp)) - rnd(math.exp(xm))) / step


def numdiff_curve(fmt, kind, hs):
    """|estimate - e| of the derivative of e^x at 1 for each step h, evaluating in `fmt` ("32-bit"/"64-bit")."""
    rnd = _fmt_for(fmt)
    fn = _fwd if kind == "forward" else _cen
    return [abs(fn(rnd, h) - math.e) for h in hs]


def numdiff_precision():
    rows = []
    ks = list(range(1, 16))
    hs = [10.0 ** -k for k in ks]
    cols = [numdiff_curve(f, kd, hs) for f in ("32-bit", "64-bit") for kd in ("forward", "central")]
    # columns: 32-bit forward, 32-bit central, 64-bit forward, 64-bit central
    for i, k in enumerate(ks):
        rows.append([f"1e-{k}"] + [_sci(c[i]) for c in cols])
    return ("Error of the difference quotient for the derivative of e^x at x = 1 (exactly e), with the function "
            "evaluated and the arithmetic done in each format.\n\n"
            + _table(["h", "32-bit forward", "32-bit central", "64-bit forward", "64-bit central"],
                     rows))


def numdiff_optimum():
    hs = [2.0 ** -j for j in range(1, 45)]
    rows = []
    for fmt, u in (("32-bit", U32), ("64-bit", U64)):
        for kind in ("forward", "central"):
            errs = numdiff_curve(fmt, kind, hs)
            best = min(range(len(hs)), key=lambda i: errs[i])
            if kind == "forward":
                hstar = 2 * math.sqrt(u)
                emin = 2 * math.e * math.sqrt(u)
            else:
                hstar = (3 * u) ** (1 / 3)
                emin = (math.e * hstar ** 2 / 6 + u * math.e / hstar)
            rows.append([fmt, kind, _sci(hstar, 2), _sci(hs[best], 2), _sci(emin, 2), _sci(errs[best], 2)])
    return ("Best step for the derivative of e^x at x = 1, scanning h = 2^-1 ... 2^-44. Predicted: "
            "forward h* = 2 sqrt(u), error 2e sqrt(u); central h* = (3u)^(1/3), error e h*^2/6 + u e / h*.\n\n"
            + _table(["format", "difference", "predicted h*", "best h found", "predicted worst-case error",
                      "best error found"], rows))


# ------------------------------------------------ log-sum-exp

def _exp_fmt(z, rnd):
    try:
        return rnd(math.exp(z))
    except OverflowError:
        return INF


def lse_naive(zs, rnd=lambda v: v):
    s = 0.0
    for z in zs:
        s = rnd(s + _exp_fmt(z, rnd))
    if s == 0.0:
        return -INF
    if s == INF:
        return INF
    return rnd(math.log(s))


def lse_stable(zs, rnd=lambda v: v):
    m = max(zs)
    s = 0.0
    for z in zs:
        s = rnd(s + _exp_fmt(rnd(z - m), rnd))
    return rnd(m + rnd(math.log(s)))


def lse_exact(zs):
    return float(sum(Decimal(z).exp() for z in zs).ln())


def lse_table():
    cases = [[1.0, 2.0, 3.0], [88.0, 88.0, 88.0], [89.0, 89.0, 89.0], [1000.0, 1000.0, 1000.0],
             [1000.0, 0.0, -1000.0], [-1000.0, -1000.0, -1000.0], [-800.0, -801.0]]
    rows = []
    for zs in cases:
        row = ["[" + ", ".join(f"{z:g}" for z in zs) + "]"]
        for rnd in (f32, lambda v: v):
            row += [_g(lse_naive(zs, rnd), 8), _g(lse_stable(zs, rnd), 8)]
        row.append(_g(lse_exact(zs), 8))
        rows.append(row)
    return ("32-bit exp and log are the exact function (evaluated in 64 bits) rounded to the nearest 32-bit number, "
            "and every sum, difference and addition is rounded to 32 bits; the 64-bit columns use 64-bit exp and log. "
            "Entries are printed with 8 significant digits; a library `expf` or `logf` may differ in the last "
            "32-bit digit by one unit in the last place. The exact column is computed with 60-digit decimal arithmetic. "
            "The last row has two scores, the others three.\n\n"
            + _table(["z", "32-bit naive", "32-bit shifted", "64-bit naive", "64-bit shifted", "exact"], rows))


def logaddexp(a, b):
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


def mixture_loglik():
    logw = [math.log(0.5), math.log(0.3), math.log(0.2)]
    ll = [-921.0, -950.0, -1100.0]
    terms = [w + l for w, l in zip(logw, ll)]
    direct = 0.0
    for t in terms:
        direct += math.exp(t)
    direct_log = -INF if direct == 0 else math.log(direct)
    acc = terms[0]
    for t in terms[1:]:
        acc = logaddexp(acc, t)
    rows = [["log w_k + log p_k", ", ".join(f"{t:.4f}" for t in terms)],
            ["sum of the probabilities, then log", f"sum = {direct}, log = {_g(direct_log)}"],
            ["log-sum-exp (shift by the largest)", f"{lse_stable(terms):.6f}"],
            ["one pair at a time with log1p", f"{acc:.6f}"],
            ["exact (60-digit arithmetic)", f"{lse_exact(terms):.6f}"]]
    return ("A text has log-likelihood -921, -950, -1100 under three models with mixture weights "
            "0.5, 0.3, 0.2.\n\n" + _table(["step", "value"], rows, "ll"))


def cross_entropy_row():
    logits = [1000.0, 998.0, 990.0, 0.0, -5.0]
    target = 1
    rows = []
    for name, rnd in (("32-bit", f32), ("64-bit", lambda v: v)):
        naive = lse_naive(logits, rnd) - logits[target]
        stable = lse_stable(logits, rnd) - logits[target]
        rows.append([name, _g(naive, 8), _g(stable, 8)])
    exact = lse_exact(logits) - logits[target]
    return (f"Cross-entropy -log softmax(z)[t] = lse(z) - z[t] for z = {logits}, t = {target}; exact value "
            f"{exact:.8f}.\n\n" + _table(["format", "unshifted lse - z[t]", "shifted lse - z[t]"], rows))


# ------------------------------------------------ accumulation precision

def cosine_accumulation():
    name = "floating.cosine"
    rng = bookrng.generator(name)
    d, trials = 512, 200
    err32, err64 = [], []
    for _ in range(trials):
        a = [f32(rng.normal()) for _ in range(d)]
        b = [f32(x + 0.1 * rng.normal()) for x in a]
        fa, fb = [Fraction(x) for x in a], [Fraction(x) for x in b]
        dot = sum(x * y for x, y in zip(fa, fb))
        na, nb = sum(x * x for x in fa), sum(x * x for x in fb)
        exact = float(dot) / math.sqrt(float(na) * float(nb))   # exact sums, one final double evaluation
        d32 = n1 = n2 = 0.0
        for x, y in zip(a, b):
            d32 = f32(d32 + f32(x * y))
            n1 = f32(n1 + f32(x * x))
            n2 = f32(n2 + f32(y * y))
        c32 = f32(d32 / f32(f32(math.sqrt(n1)) * f32(math.sqrt(n2))))
        d64 = n1_ = n2_ = 0.0
        for x, y in zip(a, b):
            d64 += x * y
            n1_ += x * x
            n2_ += y * y
        c64 = f32(d64 / (math.sqrt(n1_) * math.sqrt(n2_)))
        err32.append(abs(c32 - exact))
        err64.append(abs(c64 - exact))
    rows = [["32-bit products and sums", _sci(sum(err32) / trials), _sci(max(err32))],
            ["64-bit sums of the same 32-bit inputs, result rounded to 32 bits", _sci(sum(err64) / trials), _sci(max(err64))]]
    return (f"Cosine of {trials} pairs of 512-dimensional vectors (the second is the first plus noise of one tenth of its standard deviation, "
            f"cosine near 0.995), against the exact value (exact rational sums, one final evaluation in 64 bits). "
            f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x}; each trial draws {d} `normal()` values "
            f"for the first vector (rounded to 32 bits), then {d} for the noise, b = f32(a + 0.1 z). "
            f"The 32-bit row computes dot, na = a . a and nb = b . b by running f32 sums of f32 products, then "
            f"cos = dot / (sqrt(na) * sqrt(nb)) with both square roots, their product and the quotient rounded to f32 "
            f"(not dot / sqrt(na * nb)); the 64-bit row accumulates dot, na, nb in 64 bits and rounds only the final cosine to f32.\n\n" + _table(["accumulation", "mean abs. error", "max abs. error"], rows, "lrr"))


# ------------------------------------------------ the sum 0.1 + 0.2

def why_point_three():
    a, b = Fraction(0.1), Fraction(0.2)
    exact_sum = a + b
    stored_three = Fraction(0.3)
    up = Fraction(0.1 + 0.2)
    rows = [["stored 0.1", f"{Decimal(a.numerator) / Decimal(a.denominator):.30f}"],
            ["stored 0.2 (exactly twice the stored 0.1)", f"{Decimal(b.numerator) / Decimal(b.denominator):.30f}"],
            ["their exact sum", f"{Decimal(exact_sum.numerator) / Decimal(exact_sum.denominator):.30f}"],
            ["nearest number below it", f"{Decimal(stored_three.numerator) / Decimal(stored_three.denominator):.30f}"],
            ["nearest number above it", f"{Decimal(up.numerator) / Decimal(up.denominator):.30f}"],
            ["distance to the lower / to the upper", f"{float(exact_sum - stored_three):.4e} / {float(up - exact_sum):.4e}"]]
    return _table(["quantity", "value"], rows, "ll")


# ------------------------------------------------ a stable and an unstable evaluation of one inverse

def _lower_ones_matrix(c, a):
    """The strictly lower triangular matrix with every entry below the diagonal equal to -a."""
    return [[-a if j < i else 0.0 for j in range(c)] for i in range(c)]


def _matmul(x, y, rnd):
    n = len(x)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = x[i]
        for k in range(i + 1):               # all matrices here are lower triangular
            xik = xi[k]
            if xik == 0.0:
                continue
            yk = y[k]
            row = out[i]
            for j in range(k + 1):
                row[j] = rnd(row[j] + rnd(xik * yk[j]))
    return out


def _identity_plus(m):
    n = len(m)
    return [[m[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]


def _inverse_by_powers(a_mat, rnd):
    """(I - A)^-1 as the product of (I + A^(2^m)), valid because A^c = 0 (A strictly lower triangular).

    Returns the inverse and the largest entry of any power of A that the product had to form.
    """
    power, result, top = [r[:] for r in a_mat], None, 0.0
    while any(v != 0.0 for r in power for v in r):
        factor = _identity_plus(power)
        result = factor if result is None else _matmul(result, factor, rnd)
        top = max(top, max(abs(v) for r in power for v in r))
        power = _matmul(power, power, rnd)
    return result, top


def _inverse_by_substitution(a_mat, rnd):
    n = len(a_mat)
    t = [[0.0] * n for _ in range(n)]
    for i in range(n):
        t[i][i] = 1.0
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s = rnd(s + rnd(a_mat[i][k] * t[k][j]))
            t[i][j] = s
    return t


def _inverse_exact(a_mat):
    n = len(a_mat)
    am = [[Fraction(v) for v in r] for r in a_mat]
    t = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        t[i][i] = Fraction(1)
        for j in range(i):
            t[i][j] = sum(am[i][k] * t[k][j] for k in range(j, i))
    return t


def nilpotent_inverse():
    c = 64
    rows = []
    for a in (0.1, 0.5, 0.98):
        a_mat = _lower_ones_matrix(c, a)
        exact = _inverse_exact(a_mat)
        biggest = float(max(abs(v) for r in exact for v in r))
        cells = []
        for rnd in (f32, lambda v: v):
            for fn in (_inverse_by_powers, _inverse_by_substitution):
                got = fn(a_mat, rnd)
                got = got[0] if isinstance(got, tuple) else got
                err = max(abs(Fraction(got[i][j]) - exact[i][j]) for i in range(c) for j in range(i + 1))
                cells.append(_sci(float(err), 2))
        _, top = _inverse_by_powers(a_mat, lambda v: v)
        rows.append([f"{a:g}", _sci(top, 2), f"{biggest:.3g}"] + cells)
    return (f"The inverse of I - A for the {c} x {c} strictly lower triangular matrix A whose entries below the "
            "diagonal all equal -a, computed two ways in each format. Largest entry of any power of A used by the "
            "product form, largest entry of the true inverse, and the largest error of each computation.\n\n"
            + _table(["a", "largest entry among the powers of A", "largest entry of the inverse", "32-bit, product of powers",
                      "32-bit, substitution", "64-bit, product of powers", "64-bit, substitution"], rows))


# ------------------------------------------------ a floor inside a loss

def clamped_gradient():
    """Cross-entropy against the target (0.8, 0.2) when each probability is floored at 1e-12 before use."""
    floor = 1e-12
    target = (0.8, 0.2)
    rows = []
    for gap in (10.0, 27.0, 30.0, 100.0, 400.0):
        z = (-gap, 0.0)
        m = max(z)
        e = [math.exp(v - m) for v in z]
        p = [v / sum(e) for v in e]
        slope = [-t / max(pi, floor) for t, pi in zip(target, p)]      # d loss / d p_i with the floor
        mean_ps = sum(pi * si for pi, si in zip(p, slope))
        grad0 = p[0] * (slope[0] - mean_ps)                            # through the softmax Jacobian
        rows.append([f"{gap:g}", _sci(p[0], 2), f"{grad0:.3e}", f"{p[0] - target[0]:+.3f}"])
    return ("Gradient of the loss with respect to the first score, for scores (-gap, 0) and target (0.8, 0.2). "
            "The last column is the closed form p - target.\n\n"
            + _table(["gap", "p of the first option", "gradient with the floor", "p - target"], rows))
