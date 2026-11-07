# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Complex Numbers, Frequencies and the Fourier Transform".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: fourier.<name> -->` markers. Functions whose names start with an
underscore (or that take arguments) are shared with the chapter's figures, so
a picture and the table beside it cannot disagree.

Complex numbers are Python's built-in `complex`, but the chapter's definition
of the product (a rotation-and-scaling of the plane) is also implemented
directly on pairs of reals and checked against it. The transforms are written
from the definitions: the DFT as a double loop, the FFT as the even/odd
recursion and as the iterative butterfly schedule, with operation counters.
Numbers smaller than 1e-12 in magnitude print as 0 (they are rounding error).

Swedish Embedded AB implements solutions for frequency-domain signal
processing in machine learning pipelines. If your team needs expertise in
spectral methods, fast transforms and audio front ends, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _g(v, digits=4):
    """Compact decimal: up to `digits` significant digits, no trailing zeros."""
    if isinstance(v, str):
        return v
    if abs(v) < 1e-12:
        return "0"
    if abs(v) >= 1e7 or abs(v) < 1e-4:
        return f"{v:.{digits - 1}e}"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(v))))
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{v:.0f}"


def _c(z, digits=4):
    """A complex number as 'a + bi' (parts below 1e-12 print as 0)."""
    z = complex(z)
    re, im = _g(z.real, digits), _g(z.imag, digits)
    if im == "0":
        return re
    sign = "-" if z.imag < 0 else "+"
    mag = _g(abs(z.imag), digits)
    mag = "" if mag == "1" else mag
    if re == "0":
        return ("-" if z.imag < 0 else "") + mag + "i"
    return f"{re} {sign} {mag}i"


def _vec(x, digits=4):
    return "(" + ", ".join(_g(v, digits) for v in x) + ")"


def _cvec(x, digits=4):
    return "(" + ", ".join(_c(v, digits) for v in x) + ")"


def _deg(rad):
    return math.degrees(rad)


# ----------------------------------------------- complex numbers on pairs

def _cmul_pairs(z, w):
    """(a + bi)(c + di) = (ac - bd) + (ad + bc)i, on pairs of reals."""
    (a, b), (c, d) = z, w
    return (a * c - b * d, a * d + b * c)


def _cmul_matrix(z, w):
    """The same product as the matrix [[c, -d], [d, c]] applied to the vector (a, b)."""
    (a, b), (c, d) = z, w
    return (c * a - d * b, d * a + c * b)


def _expi(t):
    """e^{it} built from the unit circle: cos t + i sin t."""
    return complex(math.cos(t), math.sin(t))


# ---------------------------------------------------------- the transforms

def _omega(n_points, m):
    """The N-th root of unity w^m with w = e^{-2 pi i / N}; m is reduced mod N."""
    return _expi(-2 * math.pi * (m % n_points) / n_points)


def _dft(x):
    """X[k] = sum_n x[n] w^{kn}, the definition, N^2 products."""
    n = len(x)
    return [sum(x[j] * _omega(n, k * j) for j in range(n)) for k in range(n)]


def _idft(spectrum):
    """x[n] = (1/N) sum_k X[k] w^{-kn}."""
    n = len(spectrum)
    return [sum(spectrum[k] * _omega(n, -k * j) for k in range(n)) / n for j in range(n)]


def _fft(x, counter=None):
    """Recursive radix-2 FFT: split into even and odd samples, combine with twiddles.

    `counter`, if given, is a dict whose "mul" and "add" entries count the
    complex multiplications and additions performed.
    """
    n = len(x)
    if n == 1:
        return [complex(x[0])]
    assert n % 2 == 0, "radix-2 FFT needs a power-of-two length"
    even = _fft(x[0::2], counter)
    odd = _fft(x[1::2], counter)
    out = [0j] * n
    for k in range(n // 2):
        t = _omega(n, k) * odd[k]
        out[k] = even[k] + t
        out[k + n // 2] = even[k] - t
        if counter is not None:
            counter["mul"] += 1
            counter["add"] += 2
    return out


def _bit_reverse(i, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def _butterflies(n):
    """The iterative decimation-in-time schedule for a power-of-two length n.

    Returns (perm, stages): `perm[i]` is the index of the input sample placed
    in slot i, and each stage is a list of butterflies (top, bottom, e), meaning
    u, v = slot[top], slot[bottom]; slot[top] = u + w^e v; slot[bottom] = u - w^e v
    with w = e^{-2 pi i / n}.
    """
    bits = n.bit_length() - 1
    assert 1 << bits == n and n >= 2
    perm = [_bit_reverse(i, bits) for i in range(n)]
    stages = []
    size = 2
    while size <= n:
        stage = []
        for start in range(0, n, size):
            for k in range(size // 2):
                stage.append((start + k, start + k + size // 2, k * (n // size)))
        stages.append(stage)
        size *= 2
    return perm, stages


def _fft_iterative(x):
    n = len(x)
    perm, stages = _butterflies(n)
    slot = [complex(x[perm[i]]) for i in range(n)]
    for stage in stages:
        for top, bottom, e in stage:
            u, t = slot[top], _omega(n, e) * slot[bottom]
            slot[top], slot[bottom] = u + t, u - t
    return slot


def _max_abs_diff(a, b):
    return max(abs(p - q) for p, q in zip(a, b))


def _below(err):
    """A bound on a rounding error as text (never a noisy digit): 1e-12 unless err is larger."""
    if err < 1e-12:
        return "below 1e-12"
    return f"below 1e{math.floor(math.log10(err)) + 1}"


def _circular_convolve(x, h):
    n = len(x)
    assert len(h) == n
    return [sum(h[m] * x[(j - m) % n] for m in range(n)) for j in range(n)]


def _linear_convolve(x, h):
    out = [0.0] * (len(x) + len(h) - 1)
    for j in range(len(out)):
        for m in range(len(h)):
            if 0 <= j - m < len(x):
                out[j] += h[m] * x[j - m]
    return out


# ------------------------------------------------- the running signal

PROBLEM_N = 16


def _problem_signal():
    """16 samples in one second: a 2 Hz hum (cosine, amplitude 3), a 5 Hz tone
    (sine, amplitude 1.5) and a constant offset 0.5."""
    n = PROBLEM_N
    return [0.5 + 3 * math.cos(2 * math.pi * 2 * j / n) + 1.5 * math.sin(2 * math.pi * 5 * j / n)
            for j in range(n)]


NOISY_NAME = "fourier.noisy_signal"
COUNTS_NAME = "fourier.operation_counts"
ACCURACY_NAME = "fourier.fft_accuracy"


def _noisy_signal(n=64):
    """The figure's signal: 64 samples in one second, a 3 Hz hum of amplitude 2
    and an 11 Hz tone of amplitude 1 with phase 1 rad, plus Gaussian noise of
    standard deviation 0.3: one rng.normal() per sample, in sample order."""
    rng = bookrng.generator(NOISY_NAME)
    return [2 * math.cos(2 * math.pi * 3 * j / n) + 1.0 * math.cos(2 * math.pi * 11 * j / n + 1.0)
            + 0.3 * rng.normal() for j in range(n)]


def noisy_spectrum():
    """Amplitudes 2 abs(X[k]) / N of the noisy signal at the planted frequencies and elsewhere."""
    n = 64
    X = _fft(_noisy_signal(n))
    amp = [2 * abs(X[k]) / n for k in range(n // 2 + 1)]  # every k, k = 32 included, is doubled
    others = [a for k, a in enumerate(amp) if k not in (0, 3, 11)]
    rows = [("k = 3 (planted 2)", _g(amp[3], 3)), ("k = 11 (planted 1)", _g(amp[11], 3)),
            ("largest of k = 1 .. 32 except 3 and 11", _g(max(others), 4)),
            ("mean of k = 1 .. 32 except 3 and 11", _g(sum(others) / len(others), 4))]
    return ("Amplitude 2 abs X[k] / N for every k listed, k = 32 included (it is doubled like the others); "
            "the mean is over the 30 values k = 1 .. 32 except 3 and 11.\n\n"
            + _table(["amplitude 2 abs X[k] / N", "value"], rows, "lr"))


DEMO_X = [3, 1, 4, 1, 5, 9, 2, 6]


# =============================================================== the problem

def signal_samples():
    x = _problem_signal()
    rows = [(j, _g(x[j]), j + 8, _g(x[j + 8])) for j in range(8)]
    return ("16 samples taken over one second, x[n] = 0.5 + 3 cos(2 pi 2 n/16) + 1.5 sin(2 pi 5 n/16).\n\n"
            + _table(["n", "x[n]", "n", "x[n]"], rows))


def cosine_correlations():
    x = _problem_signal()
    n = len(x)
    rows = []
    for k in range(n // 2 + 1):
        c = sum(x[j] * math.cos(2 * math.pi * k * j / n) for j in range(n))
        s = sum(x[j] * math.sin(2 * math.pi * k * j / n) for j in range(n))
        rows.append((k, _g(c), _g(s), _g(math.hypot(c, s))))
    return ("Dot products of the signal with the sampled cosine and sine that complete k cycles in 16 samples.\n\n"
            + _table(["k (cycles)", "x . cos", "x . sin", "sqrt(cos^2 + sin^2)"], rows))


# ============================================================ complex numbers

def complex_products():
    cases = [((2, 1), (0.6, 0.8)), ((3, 0), (0, 1)), ((1, 1), (1, 1)), ((0, 1), (0, 1)), ((-1, 2), (2, -0.5))]
    rows = []
    for z, w in cases:
        p = _cmul_pairs(z, w)
        q = _cmul_matrix(z, w)
        assert abs(p[0] - q[0]) < 1e-12 and abs(p[1] - q[1]) < 1e-12
        zc, wc, pc = complex(*z), complex(*w), complex(*p)
        assert abs(zc * wc - pc) < 1e-12
        ang = lambda v: _g(_deg(math.atan2(v.imag, v.real)), 4)  # noqa: E731
        rows.append((_c(zc), _c(wc), _c(pc), _vec(q), f"{_g(abs(zc))}, {_g(abs(wc))}, {_g(abs(pc))}",
                     f"{ang(zc)}, {ang(wc)}, {ang(pc)}"))
    return ("Products from the rule (ac - bd) + (ad + bc)i, the same product as the matrix [[c, -d], [d, c]] applied to (a, b), "
            "and the lengths and angles (degrees) of the three numbers.\n\n"
            + _table(["z", "w", "z w by the rule", "by the matrix", "lengths of z, w, z w", "angles of z, w, z w"], rows,
                     align="lllllr"))


def powers_of_i():
    i = complex(0, 1)
    rows = []
    p = complex(1, 0)
    for k in range(9):
        rows.append((k, _c(p), _g(_deg(math.atan2(p.imag, p.real)) % 360)))
        p = p * i
    return ("Powers of i = (0, 1), each a further quarter turn.\n\n"
            + _table(["k", "i^k", "angle (degrees)"], rows))


def polar_form():
    z = complex(3, 4)
    r, th = abs(z), math.atan2(z.imag, z.real)
    back = r * _expi(th)
    z1, z2 = (2, math.radians(30)), (3, math.radians(45))
    prod_polar = (z1[0] * z2[0], z1[1] + z2[1])
    zz1, zz2 = z1[0] * _expi(z1[1]), z2[0] * _expi(z2[1])
    rect = zz1 * zz2
    pol = prod_polar[0] * _expi(prod_polar[1])
    return _table(["quantity", "value"], [
        ("z", _c(z)),
        ("radius r = sqrt(3^2 + 4^2)", _g(r)),
        ("angle theta = atan2(4, 3), degrees", _g(_deg(th))),
        ("r (cos theta + i sin theta)", _c(back)),
        ("z1 = 2 at 30 degrees, rectangular", _c(zz1)),
        ("z2 = 3 at 45 degrees, rectangular", _c(zz2)),
        ("z1 z2 multiplying the rectangular forms", _c(rect)),
        ("z1 z2 as radius 2 x 3 = 6 at angle 30 + 45 = 75 degrees", _c(pol)),
    ], align="ll")


def conjugate_and_inverse():
    z = complex(3, 4)
    zc = z.conjugate()
    inv = zc / (abs(z) ** 2)
    return _table(["quantity", "value"], [
        ("z", _c(z)),
        ("conjugate of z", _c(zc)),
        ("z times its conjugate", _c(z * zc)),
        ("abs(z)^2", _g(abs(z) ** 2)),
        ("1/z = conjugate / abs(z)^2", _c(inv)),
        ("z * (1/z)", _c(z * inv)),
    ], align="ll")


# ========================================================== Euler's formula

def sine_cosine_limits():
    rows = []
    for h in (0.5, 0.1, 0.01, 0.001, 0.0001):
        rows.append((_g(h), f"{math.sin(h) / h:.8f}", f"{(math.cos(h) - 1) / h:.8f}"))
    return _table(["h", "sin(h) / h", "(cos(h) - 1) / h"], rows)


def euler_derivative():
    t, h = 0.9, 1e-6
    e = lambda s: complex(math.cos(s), math.sin(s))  # noqa: E731
    fwd = (e(t + h) - e(t - h)) / (2 * h)
    return _table(["quantity", "value"], [
        ("E(t) = cos t + i sin t at t = 0.9", _c(e(t), 6)),
        ("i E(t)", _c(1j * e(t), 6)),
        ("central difference (E(t+h) - E(t-h)) / 2h, h = 1e-6", _c(fwd, 6)),
        ("-sin t + i cos t", _c(complex(-math.sin(t), math.cos(t)), 6)),
    ], align="ll")


def euler_series_one():
    t = 1.0
    rows = []
    s = 0j
    term = complex(1, 0)
    for k in range(0, 11):
        if k > 0:
            term = term * (1j * t) / k
        s += term
        rows.append((k, _c(s, 8), f"{abs(s - _expi(t)):.2e}"))
    return (f"Partial sums p_k(it) = 1 + (it) + (it)^2/2! + ... + (it)^k/k! at t = 1; "
            f"the target is cos 1 + i sin 1 = {_c(_expi(1.0), 8)}.\n\n"
            + _table(["degree k", "p_k(i)", "distance to cos 1 + i sin 1"], rows))


def euler_series_pi():
    t = math.pi
    rows = []
    s = 0j
    term = complex(1, 0)
    for k in range(0, 25):
        if k > 0:
            term = term * (1j * t) / k
        s += term
        if k in (1, 2, 4, 8, 12, 16, 20, 24):
            rows.append((k, _g(s.real, 8), _g(s.imag, 8)))
    return ("Partial sums p_k(i pi); the target is e^{i pi} = -1 + 0i.\n\n"
            + _table(["degree k", "real part", "imaginary part"], rows))


def euler_identities():
    rows = []
    for name, val in [("e^{i pi/2}", _expi(math.pi / 2)), ("e^{i pi}", _expi(math.pi)),
                      ("e^{2 pi i}", _expi(2 * math.pi)), ("e^{i pi/4}", _expi(math.pi / 4)),
                      ("e^{-i pi/2}", _expi(-math.pi / 2))]:
        rows.append((name, _c(val), _g(abs(val))))
    a, b = 0.7, 1.1
    rows.append(("e^{0.7 i} e^{1.1 i}", _c(_expi(a) * _expi(b)), _g(abs(_expi(a) * _expi(b)))))
    rows.append(("e^{1.8 i}", _c(_expi(a + b)), _g(abs(_expi(a + b)))))
    z = math.e * _expi(math.pi / 2)
    rows.append(("e^{1 + i pi/2} = e^1 e^{i pi/2}", _c(z), _g(abs(z))))
    return _table(["expression", "value", "modulus"], rows)


def roots_of_unity():
    n = 8
    rows = []
    total = 0j
    for m in range(n):
        w = _omega(n, m)
        total += w
        rows.append((m, _g(-360 * m / n), _g(w.real), _g(w.imag)))
    prod = _omega(n, 3) * _omega(n, 6)
    return (_table(["m", "angle (degrees)", "real part of w^m", "imaginary part of w^m"], rows)
            + f"\n\nSum of all eight: {_c(total)}. w^3 w^6 = {_c(prod)} = w^9 = w^1 = {_c(_omega(n, 1))}. "
              f"(w^4)^2 = {_c(_omega(n, 4) ** 2)}.")


# ================================================================ sinusoids

def sinusoid_parameters():
    rows = []
    for a, f, phi in [(3, 2, 0.0), (1.5, 5, -math.pi / 2), (2, 1, math.pi / 3), (1, 0.5, math.pi)]:
        period = 1 / f
        peak = (-phi / (2 * math.pi * f)) % period
        rows.append((_g(a), _g(f), _g(phi), _g(period), _g(a * math.cos(phi)), _g(peak)))
    return _table(["amplitude A", "frequency f (cycles/s)", "phase phi (rad)", "period 1/f (s)", "value at t = 0",
                   "first peak (s)"], rows)


def amplitude_phase():
    rows = []
    theta = 1.0
    for a, b in [(3, 0), (0, 1.5), (2, -2), (-1, 1), (1, 1)]:
        amp = math.hypot(a, b)
        phi = math.atan2(-b, a)
        c = complex(a, -b)
        lhs = a * math.cos(theta) + b * math.sin(theta)
        rhs = amp * math.cos(theta + phi)
        assert abs(lhs - rhs) < 1e-12
        rows.append((_g(a), _g(b), _g(amp), _g(phi), _c(c), _g(lhs), _g(rhs)))
    return _table(["a", "b", "A = sqrt(a^2 + b^2)", "phi = atan2(-b, a)", "c = a - bi", "a cos 1 + b sin 1",
                   "A cos(1 + phi)"], rows)


def cosine_decay_sinusoid():
    rows = []
    for u in (0, 0.25, 0.5, 0.75, 1.0):
        rows.append((_g(u), _g(0.5 * (1 + math.cos(math.pi * u))), _g(0.5 + 0.5 * math.cos(2 * math.pi * 0.5 * u))))
    return _table(["u (fraction of the decay)", "0.5 (1 + cos(pi u))", "offset 0.5 + amplitude 0.5 at 1/2 cycle per unit"], rows)


def same_samples_different_frequencies():
    n = 8
    f4 = lambda v: f"{v:.4f}".replace("-0.0000", "0.0000")  # noqa: E731
    rows = []
    for j in range(n):
        rows.append((j, *(f4(math.cos(2 * math.pi * k * j / n)) for k in (1, 7, 9)),
                     *(f4(math.sin(2 * math.pi * k * j / n)) for k in (1, 7))))
    return ("Eight samples of cosines and sines that complete k cycles in the window.\n\n"
            + _table(["n", "cos k=1", "cos k=7", "cos k=9", "sin k=1", "sin k=7"], rows))


# ===================================================================== DFT

def dft_exponent_table():
    n = 8
    rows = [(k, *[(k * j) % n for j in range(n)]) for k in range(n)]
    return _table(["k \\ n"] + [str(j) for j in range(n)], rows)


def dft_terms_k1():
    x = DEMO_X
    n = len(x)
    rows = []
    total = 0j
    for j in range(n):
        w = _omega(n, j)
        term = x[j] * w
        total += term
        rows.append((j, x[j], j % n, _c(w), _c(term)))
    rows.append(("sum", "", "", "", "X[1] = " + _c(total)))
    return (f"x = {_vec(x)}, k = 1, so the exponent is k n mod 8 = n.\n\n"
            + _table(["n", "x[n]", "exponent", "w^exponent", "x[n] w^exponent"], rows))


def dft_eight_points():
    x = DEMO_X
    X = _dft(x)
    n = len(x)
    rows = []
    for k in range(n):
        re = X[k].real if abs(X[k].real) > 1e-12 else 0.0   # flush BEFORE the phase
        im = X[k].imag if abs(X[k].imag) > 1e-12 else 0.0
        rows.append((k, _g(X[k].real), _g(X[k].imag), _g(abs(X[k])), _g(_deg(math.atan2(im, re))),
                     _c(X[n - k].conjugate() if k else X[0].conjugate())))
    return (f"The DFT of x = {_vec(x)}. Parts below 1e-12 in magnitude are flushed to +0 before the "
            "phase is taken, and atan2(0, -3) = 180 degrees.\n\n"
            + _table(["k", "real part", "imaginary part", "abs X[k]", "phase (degrees)", "conjugate of X[8-k]"], rows))


def dft_orthogonality():
    n = 8
    rows = []
    for k, l in [(0, 0), (1, 1), (4, 4), (1, 2), (1, 7), (2, 6), (3, 5), (0, 4)]:
        ip = sum(_omega(n, k * j).conjugate() * _omega(n, l * j) for j in range(n))
        rows.append((f"({k}, {l})", _c(ip)))
    g = 0.0
    diag = []
    for k in range(n):
        for l in range(n):
            ip = sum(_omega(n, k * j).conjugate() * _omega(n, l * j) for j in range(n))
            if k == l:
                diag.append(ip.real)
            else:
                g = max(g, abs(ip))
    return (_table(["pair of rows (k, l)", "sum over n of conj(w^{kn}) w^{ln}"], rows, align="ll")
            + f"\n\nAll eight diagonal entries of F F^H: {_vec(diag)}. Largest off-diagonal magnitude: {_below(g)}.")


def dft_inverse_roundtrip():
    x = DEMO_X
    X = _dft(x)
    n = len(x)
    back = _idft(X)
    trick = [v.conjugate() / n for v in _dft([v.conjugate() for v in X])]
    energy_t = sum(v * v for v in x)
    energy_f = sum(abs(v) ** 2 for v in X) / n
    return _table(["quantity", "value"], [
        ("x", _vec(x)),
        ("(1/N) sum_k X[k] w^{-kn}", _cvec(back)),
        ("largest difference from x", _below(_max_abs_diff(back, x))),
        ("conj(DFT(conj(X))) / N", _cvec(trick)),
        ("largest difference from x (conjugate trick)", _below(_max_abs_diff(trick, x))),
        ("sum of x[n]^2", _g(energy_t)),
        ("(1/N) sum of abs(X[k])^2", _g(energy_f)),
    ], align="ll")


def truncated_reconstruction():
    x = DEMO_X
    X = _dft(x)
    n = len(x)
    norm_x = math.sqrt(sum(v * v for v in x))
    rows = []
    for kmax in range(0, n // 2 + 1):
        keep = [k for k in range(n) if min(k, n - k) <= kmax]
        Z = [X[k] if k in keep else 0j for k in range(n)]
        xr = _idft(Z)
        err = math.sqrt(sum(abs(a - b) ** 2 for a, b in zip(x, xr)))
        dropped = math.sqrt(sum(abs(X[k]) ** 2 for k in range(n) if k not in keep) / n)
        rows.append((kmax, len(keep), _vec([v.real for v in xr], 3), _g(err / norm_x), _g(dropped / norm_x)))
    return ("Keep only the coefficients with frequency at most K (and their mirror images), set the rest to zero, "
            "and invert.\n\n"
            + _table(["K", "coefficients kept", "reconstruction", "relative error", "predicted from the dropped coefficients"], rows,
                     align="rrlrr"))


def dft_leakage():
    n = 16
    rows = []
    for k in range(n // 2 + 1):
        a = _dft([math.cos(2 * math.pi * 3 * j / n) for j in range(n)])[k]
        b = _dft([math.cos(2 * math.pi * 3.5 * j / n) for j in range(n)])[k]
        scale = 1 if k in (0, n // 2) else 2
        rows.append((k, _g(scale * abs(a) / n), _g(scale * abs(b) / n)))
    return ("Amplitude estimate 2 abs(X[k])/N for a cosine with 3 whole cycles and one with 3.5 cycles in 16 samples.\n\n"
            + _table(["k", "3 cycles", "3.5 cycles"], rows))


def real_fourier_basis():
    d = 8
    cols = []
    cols.append(("constant", [1 / math.sqrt(d)] * d))
    for k in range(1, d // 2):
        cols.append((f"cos, k={k}", [math.sqrt(2 / d) * math.cos(2 * math.pi * k * j / d) for j in range(d)]))
        cols.append((f"sin, k={k}", [math.sqrt(2 / d) * math.sin(2 * math.pi * k * j / d) for j in range(d)]))
    cols.append(("alternating", [(-1) ** j / math.sqrt(d) for j in range(d)]))
    assert len(cols) == d
    gram_dev = max(abs(sum(a * b for a, b in zip(cols[i][1], cols[j][1])) - (1 if i == j else 0))
                   for i in range(d) for j in range(d))
    x = DEMO_X
    z = [sum(a * b for a, b in zip(c[1], x)) for c in cols]
    X = _dft(x)
    via = [X[0].real / math.sqrt(d)]
    for k in range(1, d // 2):
        via += [math.sqrt(2 / d) * X[k].real, -math.sqrt(2 / d) * X[k].imag]
    via.append(X[d // 2].real / math.sqrt(d))
    rows = [(i, cols[i][0], _g(z[i]), _g(via[i])) for i in range(d)]
    energy = math.sqrt(sum(v * v for v in z)), math.sqrt(sum(v * v for v in x))
    return (_table(["i", "basis function", "z[i] = column i . x", "from the complex DFT"], rows, align="rlrr")
            + f"\n\nLargest deviation of the Gram matrix from the identity: {_below(gram_dev)}. "
              f"Largest difference between the two columns of z: {_below(max(abs(a - b) for a, b in zip(z, via)))}. "
              f"Length of z: {_g(energy[0])}; length of x: {_g(energy[1])}.")


# ====================================================================== FFT

def fft_split():
    x = DEMO_X
    n = len(x)
    ev, od = _dft(x[0::2]), _dft(x[1::2])
    X = _dft(x)
    rows = []
    for k in range(n // 2):
        w = _omega(n, k)
        t = w * od[k]
        rows.append((k, _c(ev[k]), _c(od[k]), _c(w), _c(t), _c(ev[k] + t), _c(ev[k] - t)))
    diff = max(max(abs(ev[k] + _omega(n, k) * od[k] - X[k]), abs(ev[k] - _omega(n, k) * od[k] - X[k + n // 2]))
               for k in range(n // 2))
    return (f"Even samples {_vec(x[0::2])}, odd samples {_vec(x[1::2])}.\n\n"
            + _table(["k", "E[k]", "O[k]", "w^k", "w^k O[k]", "X[k] = E + w^k O", "X[k+4] = E - w^k O"], rows, align="lllllll")
            + f"\n\nLargest difference from the direct 8-point DFT: {_below(diff)}.")


def butterfly_schedule():
    n = 8
    perm, stages = _butterflies(n)
    rows = []
    for s, stage in enumerate(stages, start=1):
        rows.append((s, "; ".join(f"({a},{b}) w^{e}" for a, b, e in stage)))
    x = DEMO_X
    err = _max_abs_diff(_fft_iterative(x), _dft(x))
    return (f"Slot i starts with sample x[{perm[0]}], x[{perm[1]}], ... in bit-reversed order: {_vec(perm)}.\n\n"
            + _table(["stage", "butterflies (top, bottom) with twiddle"], rows, align="rl")
            + f"\n\nRunning this schedule on x = {_vec(x)} differs from the direct DFT by an amount {_below(err)}.")


def operation_counts():
    rows = []
    rng = bookrng.generator(COUNTS_NAME)
    for p in (3, 4, 6, 8, 10, 12):
        n = 1 << p
        x = [rng.normal() for _ in range(n)]
        cnt = {"mul": 0, "add": 0}
        _fft(x, cnt)
        assert cnt["mul"] == n // 2 * p and cnt["add"] == n * p
        rows.append((f"{n:,}", f"{n * n:,}", f"{cnt['mul']:,}", f"{cnt['add']:,}", _g(n * n / cnt["mul"], 3)))
    return ("Complex multiplications by the definition (N rows of N products), and multiplications and additions "
            "counted while the recursive FFT ran, trivial twiddle multiplications by w^0 = 1 included; they equal "
            "(N/2) log2 N and N log2 N exactly.\n\n"
            + _table(["N", "DFT multiplications N^2", "FFT multiplications", "FFT additions", "N^2 / FFT multiplications"], rows))


def fft_accuracy():
    rng = bookrng.generator(ACCURACY_NAME)
    rows = []
    for n in (8, 64, 256, 1024):
        x = [rng.normal() for _ in range(n)]
        direct, fast, itr = _dft(x), _fft(x), _fft_iterative(x)
        rows.append((n, _below(_max_abs_diff(fast, direct)), _below(_max_abs_diff(itr, direct)),
                     _below(_max_abs_diff(_idft(fast), x))))
    return ("Random Gaussian signals; largest difference in double precision.\n\n"
            + _table(["N", "recursive FFT vs DFT", "iterative FFT vs DFT", "inverse DFT of the FFT vs signal"], rows))


# ============================================================== convolution

CONV_X = [1, 3, 2, 0, -1, 2, 0, -2]
CONV_H = [1, 2, 1]


def linear_convolution():
    x, h = CONV_X, CONV_H
    y = _linear_convolve(x, h)
    rows = []
    for j, v in enumerate(y):
        terms = [f"{h[m]}*{x[j - m]}" for m in range(len(h)) if 0 <= j - m < len(x)]
        rows.append((j, " + ".join(terms), _g(v)))
    return (f"x = {_vec(x)}, h = {_vec(h)}; y[n] = sum over m of h[m] x[n - m] (terms outside x are zero).\n\n"
            + _table(["n", "terms h[m] x[n-m]", "y[n]"], rows, align="rlr"))


def circular_convolution():
    x, h = CONV_X, CONV_H
    n = len(x)
    hp = h + [0] * (n - len(h))
    lin = _linear_convolve(x, h)
    circ = _circular_convolve(x, hp)
    rows = []
    for j in range(n):
        wrap = lin[j + n] if j + n < len(lin) else 0
        rows.append((j, _g(lin[j]), _g(wrap), _g(lin[j] + wrap), _g(circ[j])))
    return ("Circular convolution with indices taken modulo 8, beside the linear convolution of the previous table.\n\n"
            + _table(["n", "linear y[n]", "linear y[n+8]", "sum of the two", "circular y[n]"], rows))


def convolution_theorem():
    x, h = CONV_X, CONV_H
    n = len(x)
    hp = h + [0] * (n - len(h))
    circ = _circular_convolve(x, hp)
    X, H, Y = _dft(x), _dft(hp), _dft(circ)
    rows = []
    for k in range(n):
        rows.append((k, _c(X[k]), _c(H[k]), _c(X[k] * H[k]), _c(Y[k])))
    diff = _max_abs_diff([a * b for a, b in zip(X, H)], Y)
    return (_table(["k", "X[k]", "H[k]", "X[k] H[k]", "DFT of the circular convolution"], rows, align="lllll")
            + f"\n\nLargest difference between X[k] H[k] and the DFT of the convolution: {_below(diff)}.")


def convolution_via_fft():
    x, h = CONV_X, CONV_H
    size = 16
    xp = list(x) + [0] * (size - len(x))
    hp = list(h) + [0] * (size - len(h))
    prod = [a * b for a, b in zip(_fft(xp), _fft(hp))]
    y = [v.real for v in _idft(prod)]
    direct = _linear_convolve(x, h)
    err = max(abs(a - b) for a, b in zip(y, direct + [0] * (size - len(direct))))
    return _table(["quantity", "value"], [
        ("length of x, length of h, length of the full linear convolution", f"{len(x)}, {len(h)}, {len(direct)}"),
        ("transform length used (a power of two, at least 10)", size),
        ("inverse transform of the product, first 10 entries", _vec(y[:10], 3)),
        ("direct linear convolution", _vec(direct, 3)),
        ("largest difference over all 16 entries", _below(err)),
    ], align="ll")


def filter_gain():
    x = _problem_signal()
    n = len(x)
    h = [1 / 3, 1 / 3, 1 / 3] + [0] * (n - 3)
    y = _circular_convolve(x, h)
    X, Y = _dft(x), _dft(y)
    H = _dft(h)
    rows = []
    for k in (0, 2, 5):
        scale = 1 if k == 0 else 2
        a_in = scale * abs(X[k]) / n
        gain = abs(H[k])
        formula = abs(1 + 2 * math.cos(2 * math.pi * k / n)) / 3
        rows.append((k, _g(a_in), _g(gain), _g(formula), _g(scale * abs(Y[k]) / n), _g(a_in * gain)))
    return ("A 3-point moving average h = (1/3, 1/3, 1/3), applied circularly to the 16-sample signal of the opening.\n\n"
            + _table(["k", "input amplitude", "abs H[k] from the DFT", "abs(1 + 2 cos(2 pi k/16)) / 3", "output amplitude",
                      "input x gain"], rows))


def _convolution_costs(n, m):
    """(direct, transform length L, FFT route) in real multiplications for a signal of n and a kernel of m."""
    length = 1
    while length < n + m - 1:
        length *= 2
    p = length.bit_length() - 1
    return n * m, length, 4 * (3 * (length // 2) * p + length)


def convolution_crossover():
    n = 4096
    equal = [m for m in range(1, n + 1) if _convolution_costs(n, m)[0] == _convolution_costs(n, m)[2]]
    first = next(m for m in range(1, n + 1) if _convolution_costs(n, m)[0] > _convolution_costs(n, m)[2])
    assert all(_convolution_costs(n, m)[0] > _convolution_costs(n, m)[2] for m in range(first, n + 1))
    d_prev, _, f_prev = _convolution_costs(n, first - 1)
    d_first, length, f_first = _convolution_costs(n, first)
    rows = [(first - 1, f"{d_prev:,}", f"{length:,}", f"{f_prev:,}", "equal cost" if d_prev == f_prev else "sliding cheaper"),
            (first, f"{d_first:,}", f"{length:,}", f"{f_first:,}", "transform route strictly cheaper")]
    return (f"Cost model of the previous table, N = {n}. Kernel lengths with equal cost: {', '.join(map(str, equal))}. "
            f"The first M at which the transform route is strictly cheaper (direct cost N M above FFT route cost) is "
            f"M = {first}, and it stays cheaper for every M up to {n}.\n\n"
            + _table(["kernel length M", "direct", "transform length L", "FFT route", "verdict"], rows, "rrrrl"))


def convolution_cost():
    n = 4096
    rows = []
    for m in (3, 9, 33, 129, 513, 4096):
        direct, length, fast = _convolution_costs(n, m)
        rows.append((f"{m:,}", f"{direct:,}", f"{length:,}", f"{fast:,}", _g(direct / fast, 3)))
    return ("Signal length N = 4096. Direct: N M real multiplications. Through the FFT: three transforms of length L and "
            "L pointwise complex products, a complex product counted as 4 real multiplications.\n\n"
            + _table(["kernel length M", "direct", "transform length L", "FFT route", "direct / FFT route"], rows))


# ======================================================== where this is used

def rope_rates():
    rows = []
    d = 8
    for m in range(d // 2):
        row = [m]
        for base in (10000, 1000000):
            omega = base ** (-2 * m / d)
            row += [_g(omega), _g(2 * math.pi / omega)]
        rows.append(tuple(row))
    return ("Rotation rate omega_m = base^(-2m/d) radians per position for head size d = 8, and the number of "
            "positions per full turn, 2 pi / omega_m.\n\n"
            + _table(["pair m", "omega, base 10^4", "positions per turn", "omega, base 10^6", "positions per turn"], rows))


def rope_as_complex():
    q, k, omega = complex(0.6, 0.8), complex(1.0, 0.5), 0.5
    rows = []
    for p in (0, 3, 10):
        th = p * omega
        formula = complex(q.real * math.cos(th) - q.imag * math.sin(th), q.imag * math.cos(th) + q.real * math.sin(th))
        prod = q * _expi(th)
        assert abs(formula - prod) < 1e-12
        rows.append((p, _c(formula), _c(prod)))
    rot = _table(["position p", "pair rotated by the RoPE formulas", "(x + iy) e^{i p omega}"], rows)
    rows2 = []
    for p, s in [(2, 5), (10, 13), (100, 103), (7, 3), (50, 50)]:
        a, b = q * _expi(p * omega), k * _expi(s * omega)
        dot = a.real * b.real + a.imag * b.imag
        rel = (q.conjugate() * k * _expi((s - p) * omega)).real
        rows2.append((p, s, s - p, _g(dot), _g(rel)))
    rel_t = _table(["p", "s", "offset s - p", "dot product of the rotated pairs", "Re(conj(q) k e^{i (s-p) omega})"], rows2)
    return (f"q = {_c(q)}, k = {_c(k)}, omega = 0.5.\n\n" + rot + "\n\n" + rel_t)
