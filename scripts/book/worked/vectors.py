# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Vectors: Lists of Numbers with Geometry".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: vectors.<name> -->` markers. Functions whose names start with an
underscore (or that take arguments) are shared with the chapter's figures, so
a picture and the table beside it cannot disagree.

Every vector here is a plain Python list of floats; every operation is the
loop the chapter writes out, with no library.

Swedish Embedded AB implements solutions for trustworthy similarity search and
embedding retrieval. If your team needs expertise in vector arithmetic,
similarity measures and numerical parity testing, you can procure our services
by sending an email to info@swedishembedded.com.
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


def _g(v, digits=4):
    """Compact decimal: up to `digits` significant digits, no trailing zeros, sci for huge/tiny."""
    if isinstance(v, str):
        return v
    if isinstance(v, Fraction):
        v = float(v)
    if abs(v) < 1e-12:  # rounding residue of an exact zero (sin 180 degrees, 2 - 2 cos 0)
        return "0"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if abs(v) >= 1e7 or abs(v) < 1e-4:
        return f"{v:.{digits - 1}e}"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(v))))
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{v:.0f}"


def _v(x):
    """A vector as text: (1, -2, 0.5)."""
    return "(" + ", ".join(_g(c) for c in x) + ")"


def _f32(v):
    """Round a Python float to the nearest 32-bit float; inf on overflow."""
    try:
        return struct.unpack("f", struct.pack("f", v))[0]
    except OverflowError:
        return math.inf if v > 0 else -math.inf


# ------------------------------------------------- the vector operations

def add(a, b):
    assert len(a) == len(b)
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    assert len(a) == len(b)
    return [x - y for x, y in zip(a, b)]


def scale(c, a):
    return [c * x for x in a]


def dot(a, b):
    assert len(a) == len(b)
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def norm(a):
    return math.sqrt(dot(a, a))


def norm1(a):
    return sum(abs(x) for x in a)


def norm_max(a):
    return max(abs(x) for x in a)


def unit(a):
    n = norm(a)
    assert n > 0
    return [x / n for x in a]


def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def project(v, u):
    """The projection of v onto the line through u, and the residual."""
    t = dot(v, u) / dot(u, u)
    p = scale(t, u)
    return t, p, sub(v, p)


def angle_deg(a, b):
    """The angle of the 2-D vector b measured from a, using atan2 (no dot product)."""
    d = math.degrees(math.atan2(b[1], b[0]) - math.atan2(a[1], a[0]))
    return abs((d + 180.0) % 360.0 - 180.0)


# Toy 4-D "embeddings" used by several blocks: the numbers are invented for
# the chapter, chosen so that related words point roughly the same way.
EMB = {
    "cat": [0.9, 0.8, 0.1, 0.0],
    "dog": [0.8, 0.9, 0.2, 0.1],
    "kitten": [0.45, 0.4, 0.05, 0.0],  # same direction as cat, half the length
    "car": [0.1, 0.0, 0.9, 0.8],
    "truck": [0.2, 0.1, 0.8, 0.9],
}


# --------------------------------------------------------- worked blocks

def features_and_geometry():
    houses = {
        "A": [120.0, 4.0, 30.0],
        "B": [80.0, 3.0, 5.0],
        "C": [200.0, 6.0, 12.0],
    }
    rows = []
    for k, x in houses.items():
        rows.append((k, _v(x), len(x), _g(x[0]), _g(x[2])))
    t = _table(["house", "vector (area m2, rooms, age years)", "n", "x_1", "x_3"], rows, align="llrrr")
    n = len(houses)
    cols = list(zip(*houses.values()))
    mean = [sum(c) / n for c in cols]
    return t + f"\n\nThe mean of the three vectors is {_v(mean)}."


def add_and_scale():
    a, b = [3.0, 1.0], [1.0, 2.0]
    rows = [
        ("a + b", add(a, b)),
        ("a - b", sub(a, b)),
        ("b - a", sub(b, a)),
        ("2a", scale(2, a)),
        ("-b", scale(-1, b)),
        ("0.5 b", scale(0.5, b)),
        ("2a - 3b", sub(scale(2, a), scale(3, b))),
        ("0 a", scale(0, a)),
    ]
    t = _table(["expression, a = (3, 1), b = (1, 2)", "result"], [(k, _v(v)) for k, v in rows])
    u = [0.9, 0.8, 0.1, 0.0]
    w = [0.8, 0.9, 0.2, 0.1]
    mid = scale(0.5, add(u, w))
    return t + f"\n\nIn four dimensions the same rules hold entry by entry: the midpoint (cat + dog)/2 is {_v(mid)}."


def dot_products():
    cases = [
        ("(2, 3) . (4, -1)", [2, 3], [4, -1]),
        ("(1, 2, 3) . (4, 5, 6)", [1, 2, 3], [4, 5, 6]),
        ("(1, 0, 0) . (7, 8, 9)", [1, 0, 0], [7, 8, 9]),
        ("(1, 1, 1, 1) . (5, 2, -3, 4)", [1, 1, 1, 1], [5, 2, -3, 4]),
        ("(3, 4) . (3, 4)", [3, 4], [3, 4]),
    ]
    rows = []
    for name, a, b in cases:
        terms = " + ".join(f"({_g(x * y)})" for x, y in zip(a, b))
        rows.append((name, terms, _g(dot(a, b))))
    t = _table(["dot product", "terms x_i y_i", "sum"], rows, align="llr")
    a, b, c = [2.0, 3.0], [4.0, -1.0], [1.0, 5.0]
    lhs = dot(add(a, b), c)
    rhs = dot(a, c) + dot(b, c)
    return (t + f"\n\nSymmetry: a.b = {_g(dot(a, b))}, b.a = {_g(dot(b, a))}. "
            f"Linearity with a = (2, 3), b = (4, -1), c = (1, 5): (a + b).c = {_g(lhs)} and a.c + b.c = {_g(rhs)}.")


def weighted_sum():
    w, x, bias = [0.5, -1.0, 2.0], [4.0, 3.0, 1.0], 0.25
    terms = [wi * xi for wi, xi in zip(w, x)]
    rows = [(i + 1, _g(wi), _g(xi), _g(t)) for i, (wi, xi, t) in enumerate(zip(w, x, terms))]
    t = _table(["i", "w_i", "x_i", "w_i x_i"], rows)
    return (t + f"\n\nw.x = {_g(sum(terms))}, so w.x + b = {_g(dot(w, x) + bias)} with b = {_g(bias)}. "
            f"With w = (1, 1, 1) the dot product is the sum of the entries ({_g(dot([1, 1, 1], x))}); "
            f"with w = (0, 1, 0) it picks the middle entry ({_g(dot([0, 1, 0], x))}).")


def unit_circle():
    rows = []
    for d in (0, 30, 45, 60, 90, 120, 135, 180, 270):
        r = math.radians(d)
        c, s = math.cos(r), math.sin(r)
        c = 0.0 if abs(c) < 1e-12 else c
        rows.append((d, _g(r), _g(c), _g(s), _g(c * c + s * s)))
    return _table(["angle (degrees)", "angle (radians)", "cos", "sin", "cos^2 + sin^2"], rows)


def law_of_cosines():
    cases = [([3.0, 1.0], [1.0, 2.0]), ([4.0, 0.0], [2.0, 2.0]), ([-1.0, 2.0], [3.0, 1.0]), ([2.0, 1.0], [-4.0, -2.0])]
    rows = []
    for a, b in cases:
        th = angle_deg(a, b)
        lhs = dot(sub(a, b), sub(a, b))
        rhs = dot(a, a) + dot(b, b) - 2 * norm(a) * norm(b) * math.cos(math.radians(th))
        rows.append((_v(a), _v(b), f"{th:.2f}", _g(lhs), _g(rhs), _g(dot(a, b)),
                     _g(norm(a) * norm(b) * math.cos(math.radians(th)))))
    return _table(["a", "b", "angle by atan2 (deg)", "norm(a-b)^2", "norm(a)^2 + norm(b)^2 - 2 norm(a) norm(b) cos", "a.b",
                   "norm(a) norm(b) cos"], rows, align="llrrrrr")


def norms():
    xs = [[3.0, 4.0], [3.0, -4.0, 0.0, 12.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 5.0], [-2.0, 0.5]]
    rows = []
    for x in xs:
        rows.append((_v(x), _g(norm1(x)), _g(norm(x)), _g(norm_max(x)),
                     "yes" if norm_max(x) <= norm(x) <= norm1(x) else "NO"))
    t = _table(["x", "L1 = sum of abs(x_i)", "L2 = sqrt(sum of x_i^2)", "max of abs(x_i)", "max <= L2 <= L1"], rows)
    a, b = [3.0, 4.0], [-1.0, 2.0]
    lhs, rhs = norm(add(a, b)), norm(a) + norm(b)
    c = -2.5
    return (t + f"\n\nTriangle inequality with a = (3, 4), b = (-1, 2): norm(a + b) = {_g(lhs)} <= norm(a) + norm(b) = {_g(rhs)}. "
            f"Scaling: norm({_g(c)} a) = {_g(norm(scale(c, a)))} = abs({_g(c)}) norm(a) = {_g(abs(c) * norm(a))}.")


def normalisation():
    xs = [[3.0, 4.0], [1.0, 2.0, 2.0], [0.0, -6.0, 8.0, 0.0], [0.9, 0.8, 0.1, 0.0]]
    rows = []
    for x in xs:
        u = unit(x)
        rows.append((_v(x), _g(norm(x)), _v(u), _g(norm(u))))
    t = _table(["x", "L2 norm of x", "x / norm(x)", "norm of result"], rows, align="lrlr")
    return t + "\n\nThe zero vector has norm 0 and cannot be normalised."


def embedding_cosines():
    names = list(EMB)
    rows = []
    for a in names:
        rows.append((a, *[_g(cosine(EMB[a], EMB[b]), 3) for b in names]))
    t = _table(["cosine"] + names, rows)
    lens = _table(["word", "vector", "L2 norm"], [(k, _v(EMB[k]), _g(norm(EMB[k]))) for k in names], align="llr")
    return lens + "\n\n" + t


def dot_vs_cosine_ranking():
    q = [1.0, 1.0, 0.0, 0.0]
    docs = {
        "short, same direction": [0.5, 0.5, 0.0, 0.0],
        "long, off direction": [3.0, 2.0, 2.0, 2.0],
        "medium, same direction": [2.0, 2.0, 0.0, 0.0],
        "long, opposite": [-4.0, -4.0, 0.0, 0.0],
    }
    rows = [(k, _v(d), _g(dot(q, d)), _g(cosine(q, d), 3)) for k, d in docs.items()]
    by_dot = sorted(docs, key=lambda k: -dot(q, docs[k]))
    by_cos = sorted(docs, key=lambda k: -cosine(q, docs[k]))
    t = _table(["document", "vector", "q . d", "cos(q, d)"], rows, align="llrr")
    return (t + f"\n\nQuery q = {_v(q)}. Ranked by dot product: {' > '.join(by_dot)}. "
            f"Ranked by cosine: {' > '.join(by_cos)}.")


def distance_and_cosine():
    pairs = [("cat", "dog"), ("cat", "car"), ("car", "truck"), ("cat", "kitten")]
    rows = []
    for a, b in pairs:
        ua, ub = unit(EMB[a]), unit(EMB[b])
        d2 = dot(sub(ua, ub), sub(ua, ub))
        c = dot(ua, ub)
        rows.append((f"{a}, {b}", _g(c, 5), _g(d2, 5), _g(2 - 2 * c, 5)))
    return _table(["pair (normalised first)", "cosine", "norm(u - v)^2", "2 - 2 cos"], rows)


def scale_blindness():
    # A "reference" output and three candidates that differ from it.
    want = [0.8, -1.2, 0.5, 2.0, -0.3, 1.1]
    good = [w + d for w, d in zip(want, [0.001, -0.002, 0.001, 0.0, 0.002, -0.001])]
    candidates = [
        ("exact copy", want),
        ("tiny noise (about 1e-3)", good),
        ("scaled by 1.05", scale(1.05, want)),
        ("scaled by 1.001", scale(1.001, want)),
        ("scaled by 0.5", scale(0.5, want)),
        ("one entry wrong (2.0 -> 1.0)", [1.0 if i == 3 else w for i, w in enumerate(want)]),
    ]
    rows = []
    for k, got in candidates:
        rel = norm(sub(got, want)) / norm(want)
        mx = max(abs(g - w) for g, w in zip(got, want))
        rows.append((k, f"{cosine(got, want):.9f}", _g(rel, 3), _g(mx, 3)))
    return _table(["candidate vs reference", "cosine", "relative L2", "max abs difference"], rows)


def orthogonality():
    pairs = [([1.0, 1.0, 0.0], [1.0, -1.0, 5.0]), ([2.0, 1.0], [-1.0, 2.0]), ([1.0, 2.0, 3.0], [3.0, 0.0, -1.0]),
             ([1.0, 2.0], [2.0, 1.0])]
    rows = []
    for a, b in pairs:
        d = dot(a, b)
        s = norm(add(a, b)) ** 2
        rows.append((_v(a), _v(b), _g(d), _g(norm(a) ** 2 + norm(b) ** 2), _g(s)))
    return _table(["a", "b", "a.b", "norm(a)^2 + norm(b)^2", "norm(a + b)^2"], rows, align="llrrr")


def projection():
    v, u = [3.0, 4.0], [2.0, 1.0]
    t, p, r = project(v, u)
    rows = [
        ("v . u", _g(dot(v, u))),
        ("u . u", _g(dot(u, u))),
        ("t = v.u / u.u", _g(t)),
        ("projection t u", _v(p)),
        ("residual r = v - t u", _v(r)),
        ("r . u", _g(dot(r, u))),
        ("norm(v)^2", _g(dot(v, v))),
        ("norm(t u)^2 + norm(r)^2", _g(dot(p, p) + dot(r, r))),
        ("scalar projection v.u / norm(u)", _g(dot(v, u) / norm(u))),
        ("norm(v) cos(angle)", _g(norm(v) * cosine(v, u))),
    ]
    return _table(["quantity, v = (3, 4), u = (2, 1)", "value"], rows, align="lr")


def projection_four_dimensions():
    v = [2.0, -1.0, 3.0, 1.0]
    u = [1.0, 1.0, 0.0, 2.0]
    t, p, r = project(v, u)
    rows = [
        ("v . u", _g(dot(v, u))),
        ("u . u", _g(dot(u, u))),
        ("t", _g(t)),
        ("projection", _v(p)),
        ("residual", _v(r)),
        ("r . u", _g(dot(r, u), 3) if abs(dot(r, u)) > 1e-12 else "0"),
        ("norm(v)^2", _g(dot(v, v))),
        ("norm(projection)^2 + norm(residual)^2", _g(dot(p, p) + dot(r, r))),
        ("projecting again: t", _g(project(p, u)[0])),
    ]
    return _table(["quantity, v = (2, -1, 3, 1), u = (1, 1, 0, 2)", "value"], rows, align="lr")


def projection_minimises():
    v, u = [3.0, 4.0], [2.0, 1.0]
    t_star = dot(v, u) / dot(u, u)
    rows = []
    for t in (0.0, 1.0, 1.5, t_star, 2.5, 3.0, 4.0):
        e = sub(v, scale(t, u))
        rows.append((_g(t), _v(e), _g(dot(e, e)), "yes" if abs(t - t_star) < 1e-12 else ""))
    return _table(["t", "v - t u", "norm(v - t u)^2", "t = v.u/u.u"], rows, align="lrrl")


def cauchy_schwarz_check():
    cases = [([3.0, 4.0], [2.0, 1.0]), ([1.0, 2.0, 3.0], [-3.0, -6.0, -9.0]), ([1.0, 0.0], [0.0, 5.0]),
             ([0.9, 0.8, 0.1, 0.0], [0.8, 0.9, 0.2, 0.1])]
    rows = []
    for a, b in cases:
        rows.append((_v(a), _v(b), _g(abs(dot(a, b))), _g(norm(a) * norm(b)), _g(cosine(a, b) if norm(a) and norm(b) else 0, 4)))
    return _table(["a", "b", "abs(a.b)", "norm(a) norm(b)", "cosine"], rows, align="llrrr")


def _lcg_signs(n, state):
    """n entries of +1/-1 from a 64-bit linear congruential generator (state in, state out)."""
    out = []
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(1.0 if (state >> 40) & 1 else -1.0)
    return out, state


def random_directions():
    rows = []
    state = 12345
    pairs = 300
    for n in (2, 8, 32, 128, 512, 2048):
        total, big = 0.0, 0.0
        for _ in range(pairs):
            a, state = _lcg_signs(n, state)
            b, state = _lcg_signs(n, state)
            c = abs(dot(a, b)) / n  # both have norm sqrt(n)
            total += c
            big = max(big, c)
        rows.append((n, _g(total / pairs, 3), _g(big, 3), _g(math.sqrt(2 / (math.pi * n)), 3)))
    return _table(["n", "mean abs(cos)", "largest abs(cos)", "sqrt(2 / (pi n))"], rows)


def summation_order():
    x = [1e8, 1.0, -1e8, 1.0]
    y = [1.0, 1.0, 1.0, 1.0]
    prods = [_f32(a * b) for a, b in zip(x, y)]
    fwd = 0.0
    for p in prods:
        fwd = _f32(fwd + p)
    pair = _f32(_f32(prods[0] + prods[2]) + _f32(prods[1] + prods[3]))
    exact = sum(Fraction(a) * Fraction(b) for a, b in zip(x, y))
    rows = [
        ("left to right in 32-bit", _g(fwd)),
        ("(x1 y1 + x3 y3) + (x2 y2 + x4 y4) in 32-bit", _g(pair)),
        ("exact rational arithmetic", _g(exact)),
    ]
    return _table(["x = (1e8, 1, -1e8, 1), y = (1, 1, 1, 1)", "x . y"], rows, align="lr")
