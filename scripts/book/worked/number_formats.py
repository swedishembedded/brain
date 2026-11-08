# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Bits and Number Formats".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: number_formats.<name> -->` markers. Helpers with a leading
underscore, or that take arguments, are shared with the chapter's figures so a
picture and the table beside it cannot disagree.

Every format is defined by its bit split alone: `FORMATS` lists the number of
exponent bits n_e, fraction bits n_m and whether the top exponent field is
reserved for infinity and NaN (`finite_only` False) or only the single
all-ones pattern is NaN (`finite_only` True, the E4M3 convention). `decode`
and `encode` are written from the rule the chapter derives, with exact
rational arithmetic; `_self_check` compares them with Python's `struct`
(binary32, binary16) and with the integer shortcut for bf16.

Swedish Embedded AB implements solutions for memory-bounded low-precision
machine learning for its clients. If your team needs expertise in choosing
number formats, rounding rules and scales for training and serving, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math
import struct
from fractions import Fraction

import bookrng

LOG10_2 = math.log10(2.0)

# name -> (exponent bits n_e, fraction bits n_m, finite_only)
FORMATS = {
    "fp32": (8, 23, False),
    "fp16": (5, 10, False),
    "bf16": (8, 7, False),
    "E4M3": (4, 3, True),
    "E5M2": (5, 2, False),
}
SMALL = ("fp16", "bf16", "E4M3", "E5M2")  # formats with at most 65536 codes
PARAMS = 7_000_000_000  # the model size of the byte tables
LOSS_SCALE_DRAWS = 10_000
ROUNDING_DRAWS = 100_000


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _num(v, d=4):
    """A float with d significant digits, plain for moderate sizes, scientific otherwise."""
    if v != v:
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0:
        return "0"
    if float(v).is_integer() and abs(v) < 10 ** 7:
        return str(int(v))
    if 1e-3 <= abs(v) < 1e5:
        return f"{v:.{d}g}"
    return f"{v:.{d - 1}e}"


def _bits(code, n):
    return format(code, f"0{n}b")


def _split(code, f):
    """The bit string of a code with the sign, exponent and fraction separated by spaces."""
    ne, nm, _ = FORMATS[f]
    b = _bits(code, 1 + ne + nm)
    return f"{b[0]} {b[1:1 + ne]} {b[1 + ne:]}"


def _hex(code, n):
    return "0x" + format(code, f"0{(n + 3) // 4}X")


def width(f):
    ne, nm, _ = FORMATS[f]
    return 1 + ne + nm


def bias(f):
    return 2 ** (FORMATS[f][0] - 1) - 1


def round_even(x):
    """The nearest integer to the rational x, ties to the even integer."""
    x = Fraction(x)
    fl = x.numerator // x.denominator
    r = x - fl
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and fl % 2 == 1):
        return fl + 1
    return fl


def exponent_of(a):
    """The integer e with 2^e <= a < 2^(e+1), for a positive rational a."""
    a = Fraction(a)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    return e


# ------------------------------------------------------------------ decode / encode

def decode(code, f):
    """The value of the bit pattern `code` in format `f` (a float; nan and +-inf where the format has them)."""
    ne, nm, fin = FORMATS[f]
    s = code >> (ne + nm)
    E = (code >> nm) & (2 ** ne - 1)
    m = code & (2 ** nm - 1)
    top = 2 ** ne - 1
    if E == top and (not fin or m == 2 ** nm - 1):
        if fin or m != 0:
            return math.nan
        return -math.inf if s else math.inf
    if E == 0:
        v = math.ldexp(m, 1 - bias(f) - nm)
    else:
        v = math.ldexp(2 ** nm + m, E - bias(f) - nm)
    return -v if s else v


def max_code(f):
    """The code of the largest finite positive value."""
    ne, nm, fin = FORMATS[f]
    return (((2 ** ne - 1) << nm) | (2 ** nm - 2)) if fin else ((2 ** ne - 2) << nm) + 2 ** nm - 1


def nan_code(f):
    ne, nm, fin = FORMATS[f]
    return (2 ** (ne + nm) - 1) if fin else ((2 ** ne - 1) << nm) | (1 << (nm - 1))


def encode(x, f, saturate=False):
    """The code of the number of format `f` nearest to x, ties to the even code.

    A result beyond the largest finite number becomes infinity (the largest
    finite number when `saturate`); in a finite-only format the overflow
    value is NaN unless `saturate`."""
    ne, nm, fin = FORMATS[f]
    sbit = 1 << (ne + nm)
    if x != x:
        return nan_code(f)
    sign = sbit if (x < 0 or (x == 0 and math.copysign(1.0, x) < 0)) else 0
    if math.isinf(x):
        over = True
    elif x == 0:
        return sign
    else:
        a = Fraction(abs(x))
        e = max(exponent_of(a), 1 - bias(f))
        n = round_even(a / Fraction(2) ** (e - nm))
        if n == 2 ** (nm + 1):
            e, n = e + 1, 2 ** nm
        if n < 2 ** nm:           # subnormal or zero: exponent field 0
            return sign | n
        E, m = e + bias(f), n - 2 ** nm
        code = (E << nm) | m
        over = code > max_code(f)
        if not over:
            return sign | code
    if saturate:
        return sign | max_code(f)
    return sign | (nan_code(f) if fin else ((2 ** ne - 1) << nm))


def quantize(x, f, saturate=False):
    """x rounded to format f and decoded back."""
    return decode(encode(x, f, saturate), f)


def f32(x):
    """x rounded to the nearest 32-bit float (a float)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bf16_rne_bits(b):
    """The bf16 pattern of the fp32 pattern b: add 0x7FFF plus the lowest kept bit, keep the top 16 bits."""
    return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFFFFFF) >> 16


def bf16_trunc_bits(b):
    return b >> 16


def positive_values(f):
    """The finite positive values of format f in increasing order, with their codes."""
    out = []
    for c in range(1, 1 << (width(f) - 1)):
        v = decode(c, f)
        if v == v and not math.isinf(v):
            out.append((v, c))
    return out


def _self_check():
    """Compare decode/encode with struct (fp32, fp16), the integer shortcut (bf16) and each other."""
    for c in range(1 << 16):
        h = struct.unpack("<e", struct.pack("<H", c))[0]
        d = decode(c, "fp16")
        assert (h != h and d != d) or h == d, c
        b = bits_f32(c << 16)
        d = decode(c, "bf16")
        assert (b != b and d != d) or b == d, c
    for f in SMALL:
        for c in range(1 << width(f)):
            v = decode(c, f)
            if v == v and not math.isinf(v) and (v != 0 or c == 0 or c == 1 << (width(f) - 1)):
                assert encode(v, f) == c, (f, c)
    rng = bookrng.generator("number_formats.self_check")
    for _ in range(2000):
        x = f32(math.ldexp(rng.uniform() + 0.5, rng.integer(60) - 30) * (1 if rng.integer(2) else -1))
        assert decode(encode(x, "fp32"), "fp32") == x
        if abs(x) < 65000:
            assert encode(x, "fp16") == struct.unpack("<H", struct.pack("<e", x))[0], x
        assert encode(x, "bf16") == bf16_rne_bits(f32_bits(x)), x
    assert encode(65520.0, "fp16") == 0x7C00 and encode(65519.0, "fp16") == 0x7BFF
    return True


# ------------------------------------------------------------------ the formats from their bit splits

def _props(f):
    ne, nm, fin = FORMATS[f]
    mx = decode(max_code(f), f)
    mn = math.ldexp(1.0, 1 - bias(f))
    sub = math.ldexp(1.0, 1 - bias(f) - nm)
    return dict(bits=1 + ne + nm, ne=ne, nm=nm, bias=bias(f), max=mx, min_normal=mn, min_sub=sub,
                eps=2.0 ** -nm, digits=(nm + 1) * LOG10_2, decades=math.log10(mx / sub))


def float_formats():
    _self_check()
    rows = []
    for f in FORMATS:
        p = _props(f)
        rows.append([f, p["bits"], f"{p['ne']} / {p['nm']}", p["bias"], _num(p["max"]), _num(p["min_normal"], 3),
                     _num(p["min_sub"], 3), _num(p["eps"], 3), f"{p['digits']:.2f}", f"{p['decades']:.1f}"])
    return ("Everything below is computed from the bit split n_e / n_m by the decoding rule of the chapter (bias = 2^(n_e-1) - 1, "
            "eps = 2^-n_m, digits = (n_m + 1) log10 2, decades = log10(max / smallest subnormal)). "
            "fp32, fp16, bf16 and E5M2 use the IEEE-style rule (the all-ones exponent field is infinity and NaN); E4M3 sets the "
            "decoder's finite_only flag: its all-ones exponent field 1111 is an ordinary exponent, only the pattern with "
            "fraction 111 as well is NaN, and there is no infinity, so its largest finite number is (1 + 6/8) * 2^8 = 448 "
            "(the IEEE-style rule would stop at (1 + 7/8) * 2^7 = 240). "
            "Decoding was checked against the machine's own binary16 for all 65536 codes, and encoding against its binary32 and binary16 on 2000 random values.\n\n"
            + _table(["format", "bits", "exponent / fraction bits", "bias", "largest finite", "smallest normal",
                      "smallest subnormal", "eps = 2^-n_m", "decimal digits", "decades of range"], rows))


def integer_formats():
    rows = []
    for name, n, signed in [("uint4", 4, False), ("int4", 4, True), ("uint8", 8, False), ("int8", 8, True),
                            ("int16", 16, True), ("int32", 32, True)]:
        lo, hi = (-(2 ** (n - 1)), 2 ** (n - 1) - 1) if signed else (0, 2 ** n - 1)
        rows.append([name, n, 2 ** n, f"{lo:,}", f"{hi:,}"])
    return _table(["format", "bits", "codes 2^n", "smallest", "largest"], rows)


# ------------------------------------------------------------------ one weight in every format

WEIGHT = 0.0731
GROUP = [0.0731, -0.412, 0.0098, 0.9, -0.236, 0.0515, 0.1604, -0.0302]


def one_weight():
    rows = []
    for f in ["fp32", "fp16", "bf16", "E4M3", "E5M2"]:
        c = encode(WEIGHT, f)
        v = decode(c, f)
        rows.append([f, width(f), _hex(c, width(f)), _split(c, f), _num(v, 8), _num(v - WEIGHT, 3),
                     _num(abs(v - WEIGHT) / WEIGHT, 3)])
    delta = max(abs(w) for w in GROUP) / 127
    q = max(-127, min(127, round_even(WEIGHT / delta)))
    v = delta * q
    rows.append(["int8 with scale", 8, _hex(q & 0xFF, 8), _bits(q & 0xFF, 8), _num(v, 8), _num(v - WEIGHT, 3),
                 _num(abs(v - WEIGHT) / WEIGHT, 3)])
    head = (f"The weight is w = {WEIGHT} (first entry of the group {GROUP}); fp32 is the binary32 rounding of w, "
            f"the others round w once, ties to even. The int8 row uses the scale Delta = max|group| / 127 = "
            f"{delta:.8f} and the code q = round(w / Delta) = {q}, decoded as Delta q. "
            "Bit strings are sign, exponent, fraction (int8: the two's complement pattern).\n\n")
    return head + _table(["format", "bits", "hex", "bit pattern", "decoded value", "error", "relative error"], rows)


# ------------------------------------------------------------------ integers

def unsigned_binary():
    n = 37
    rows, k, parts = [], n, []
    while k > 0:
        rows.append([k, k // 2, k % 2])
        parts.append(k % 2)
        k //= 2
    s = "".join(str(b) for b in reversed(parts))
    places = " + ".join(f"{2 ** i}" for i, b in enumerate(parts) if b)
    head = f"Repeated division of {n} by 2: the remainders, read from last to first, are the bits.\n\n"
    tail = f"\n\nReading the remainders from the bottom up gives {s}; check: {places} = {n}."
    return head + _table(["k", "k // 2", "k mod 2 (next bit, lowest first)"], rows) + tail


def unsigned_wrap():
    rows = []
    for label, exact in [("200 + 55", 255), ("200 + 56", 256), ("250 + 10", 260), ("255 + 1", 256), ("5 - 6", -1), ("16 * 16", 256)]:
        st = exact % 256
        rows.append([label, f"{exact:,}", st, _bits(st, 8)])
    return _table(["operation on 8-bit unsigned", "exact result", "stored (result mod 256)", "stored pattern"], rows)


def _pattern(x, n):
    return x % (2 ** n)


def _signed(a, n):
    return a if a < 2 ** (n - 1) else a - 2 ** n


def twos_complement():
    rows = []
    for m in [0, 1, 5, 64, 127, 128]:
        mag = _pattern(m, 8)
        inv = 255 - mag
        neg = (inv + 1) % 256
        rows.append([m, _bits(mag, 8), _bits(inv, 8), _bits(neg, 8), _bits(_pattern(-m, 8), 8), _signed(neg, 8),
                     (mag + neg) % 256])
    head = ("For each magnitude m: its 8-bit pattern, that pattern inverted (every bit flipped), the inverted pattern plus one "
            "(mod 256), the pattern of -m computed directly as (-m) mod 256, the signed reading of the inverted-plus-one pattern, "
            "and the sum of the patterns of m and of inverted-plus-one, mod 256.\n\n")
    return head + _table(["m", "pattern of m", "inverted", "inverted + 1", "pattern of -m = (-m) mod 256", "read as signed",
                          "sum mod 256"], rows)


def twos_complement_arithmetic():
    rows = []
    for a, b in [(100, 27), (100, 28), (-100, -28), (-100, -29), (-5, 3), (64, 64)]:
        s = a + b
        st = _signed(_pattern(s, 8), 8)
        rows.append([f"{a} + {b}", s, _bits(_pattern(a, 8), 8), _bits(_pattern(b, 8), 8), _bits(_pattern(s, 8), 8), st,
                     "wrapped" if st != s else "exact"])
    ext = []
    for x in [5, -5, -128]:
        ext.append([x, _bits(_pattern(x, 8), 8), _bits(_pattern(x, 16), 16)])
    return (_table(["int8 addition", "exact", "pattern a", "pattern b", "pattern of the sum (low 8 bits)", "stored value", "result"],
                   rows) + "\n\nWidening to 16 bits copies the sign bit into the new high bits:\n\n"
            + _table(["x", "8-bit pattern", "16-bit pattern"], ext))


# ------------------------------------------------------------------ special values and subnormals

def special_codes():
    rows = []
    sel = {
        "E4M3": [0x00, 0x01, 0x07, 0x08, 0x38, 0x7E, 0x7F, 0xFF, 0x80],
        "E5M2": [0x00, 0x01, 0x03, 0x04, 0x3C, 0x7B, 0x7C, 0x7D, 0x80],
        "fp16": [0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x7C00, 0x7E00, 0xFC00],
    }
    for f, codes in sel.items():
        for c in codes:
            v = decode(c, f)
            kind = ("NaN" if v != v else "infinity" if math.isinf(v) else "zero" if v == 0 else
                    "subnormal" if ((c >> FORMATS[f][1]) & (2 ** FORMATS[f][0] - 1)) == 0 else "normal")
            rows.append([f, _hex(c, width(f)), _split(c, f), _num(v, 6), kind])
    # the value the ordinary rule would give the E4M3 NaN pattern
    ordinary = math.ldexp(2 ** 3 + 7, 15 - 7 - 3)
    return (_table(["format", "hex", "sign exponent fraction", "value", "kind"], rows, "lllrl")
            + f"\n\nThe ordinary normal-number rule applied to E4M3's reserved pattern (exponent 1111, fraction 111) would give "
              f"(1 + 7/8) * 2^8 = {_num(ordinary)}; the format spends that one pattern on NaN, so its largest finite value is "
              f"{_num(decode(max_code('E4M3'), 'E4M3'))}.")


def subnormals():
    rows = []
    for f in FORMATS:
        p = _props(f)
        nm = p["nm"]
        count_pos = 2 ** nm - 1
        total = max_code(f)
        rows.append([f, _num(p["min_normal"], 3), _num(p["min_sub"], 3), f"{count_pos:,}", f"{total:,}",
                     f"{100 * count_pos / total:.4f}%", _num(p["min_sub"] / 2, 3)])
    return _table(["format", "smallest normal", "subnormal spacing = smallest subnormal", "positive subnormals 2^n_m - 1",
                   "positive finite values", "share that are subnormal", "values below this round to 0"], rows)


# ------------------------------------------------------------------ range versus precision

def range_vs_precision():
    xs = [(1 / 3, "1/3"), (1000.1, "1000.1"), (70000.0, "70000"), (3e-5, "3e-5"), (1e-8, "1e-8"), (1e-40, "1e-40")]
    rows = []
    for x, label in xs:
        x32 = f32(x)
        r = [label, _num(x32, 8)]
        for f in ["fp16", "bf16"]:
            v = quantize(x32, f)
            rel = "-" if math.isinf(v) or v == 0 else _num(abs(v - x32) / abs(x32), 2)
            r += [_num(v, 8), rel]
        rows.append(r)
    head = "Each x is first rounded to fp32, then once to the narrower format, ties to even. 'rel' is |stored - fp32 value| / |fp32 value|.\n\n"
    return head + _table(["x", "fp32", "fp16", "rel", "bf16", "rel"], rows)


def loss_scaling():
    rng = bookrng.generator("number_formats.loss_scaling")
    gs = [10.0 ** (-12 + 10 * rng.uniform()) for _ in range(LOSS_SCALE_DRAWS)]
    rows = []

    def stats(vals, f):
        zero = sub = over = 0
        err = []
        mn = math.ldexp(1.0, 1 - bias(f))
        for v in vals:
            c = encode(v, f)
            q = decode(c, f)
            if math.isinf(q) or q != q:
                over += 1
            elif q == 0:
                zero += 1
            else:
                if q < mn:
                    sub += 1
                err.append(abs(q - v) / v)
        n = len(vals)
        return zero / n, sub / n, over / n, (sum(err) / len(err) if err else float("nan"))

    for f, scales in [("fp16", [1, 2 ** 8, 2 ** 12, 2 ** 16, 2 ** 20, 2 ** 24]), ("bf16", [1])]:
        for S in scales:
            z, s, o, e = stats([S * g for g in gs], f)
            rows.append([f, f"2^{int(math.log2(S))}" if S > 1 else "1", f"{100 * z:.2f}%", f"{100 * s:.2f}%", f"{100 * o:.2f}%",
                         f"{100 * (1 - z - o):.2f}%", _num(e, 3)])
    seed = bookrng.seed_of("number_formats.loss_scaling")
    head = (f"{LOSS_SCALE_DRAWS} stand-in gradient magnitudes g_i = 10^(-12 + 10 u_i), u_i = uniform() drawn in order i = 0, 1, ... "
            f"from the generator seeded {seed} (seed_of 'number_formats.loss_scaling'); the scaled values S g_i are rounded to the "
            "format once, ties to even. 'zero' is the share stored as 0, 'subnormal' the share stored with fewer than the full "
            "significand bits, 'inf' the share that overflows, 'kept' the share stored as a nonzero finite number; "
            "'mean rel. error' is the mean of |stored - S g| / (S g) over the kept values. Shares are printed with two decimals "
            "(each is a count out of 10000, so two decimals are exact).\n\n")
    return head + _table(["format", "scale S", "zero", "subnormal", "inf", "kept", "mean rel. error of kept"], rows)


# ------------------------------------------------------------------ fp8 with a block scale

FP8_BLOCK = [3000.0, -1200.0, 87.5, 0.9, 12.0, -0.04, 450.0, -2300.0]


def fp8_block_scale():
    amax = max(abs(x) for x in FP8_BLOCK)
    mx = decode(max_code("E4M3"), "E4M3")
    s = amax / mx
    rows = []
    for x in FP8_BLOCK:
        c_raw = encode(x, "E4M3", saturate=True)
        v_raw = decode(c_raw, "E4M3")
        c_sc = encode(x / s, "E4M3", saturate=True)
        v_sc = decode(c_sc, "E4M3") * s
        v52 = quantize(x, "E5M2", saturate=True)
        rows.append([_num(x, 5), _num(v_raw, 5), _num(abs(v_raw - x) / abs(x), 2), _hex(c_sc, 8), _num(v_sc, 5),
                     _num(abs(v_sc - x) / abs(x), 2), _num(v52, 5), _num(abs(v52 - x) / abs(x), 2)])
    head = (f"One block of {len(FP8_BLOCK)} values, amax = {_num(amax)}. The scale is s = amax / 448 = {s:.6f}; a value is stored as "
            "the E4M3 code of x / s and read back as s times the decoded code. 'E4M3 raw' stores x directly, saturating at 448. "
            "'E5M2 raw' stores x directly in E5M2 (largest finite 57344). Rounding: nearest, ties to even.\n\n")
    return head + _table(["x", "E4M3 raw", "rel. error", "code of x / s", "E4M3 with scale", "rel. error", "E5M2 raw", "rel. error"], rows)


# ------------------------------------------------------------------ integer quantization

OUTLIER = [0.12, -0.05, 0.31, -0.22, 2.40, 0.18, -0.27, 0.09]
ASYM = [-1.0, -0.3, 0.0, 1.7, 4.2, 6.0]


def _sym_codes(vals, qmax):
    delta = max(abs(v) for v in vals) / qmax
    return delta, [max(-qmax, min(qmax, round_even(v / delta))) for v in vals]


def int8_group_scale():
    d1, q1 = _sym_codes(OUTLIER, 127)
    dA, qA = _sym_codes(OUTLIER[:4], 127)
    dB, qB = _sym_codes(OUTLIER[4:], 127)
    rows, e1, e2 = [], [], []
    for i, w in enumerate(OUTLIER):
        d, q = (dA, qA[i]) if i < 4 else (dB, qB[i - 4])
        v1, v2 = d1 * q1[i], d * q
        e1.append(abs(v1 - w))
        e2.append(abs(v2 - w))
        rows.append([w, q1[i], _num(v1, 5), _num(abs(v1 - w), 2), q, _num(v2, 5), _num(abs(v2 - w), 2)])
    head = (f"Eight weights {OUTLIER}. One scale for all eight: Delta = 2.40 / 127 = {d1:.6f}. Two scales, one per group of four: "
            f"Delta_A = 0.31 / 127 = {dA:.6f} and Delta_B = 2.40 / 127 = {dB:.6f}. Code q = round(w / Delta) (ties to even), "
            "decoded Delta q.\n\n")
    tail = (f"\n\nMean absolute error over the four weights of the first group: {sum(e1[:4]) / 4:.5f} with one scale, "
            f"{sum(e2[:4]) / 4:.5f} with two. Over all eight: {sum(e1) / 8:.5f} against {sum(e2) / 8:.5f}.")
    return head + _table(["w", "q (one scale)", "decoded", "abs. error", "q (group scale)", "decoded", "abs. error"], rows) + tail


def int8_zero_point():
    lo, hi = min(ASYM), max(ASYM)
    ds, qs = _sym_codes(ASYM, 127)
    da = (hi - lo) / 255
    q0 = round_even(-lo / da)
    rows = []
    for x, q in zip(ASYM, qs):
        qa = max(0, min(255, round_even(x / da) + q0))
        va = da * (qa - q0)
        rows.append([x, q, _num(ds * q, 5), _num(abs(ds * q - x), 2), qa, _num(va, 5), _num(abs(va - x), 2)])
    head = (f"Six values {ASYM} with range [{lo}, {hi}]. Symmetric: Delta = 6 / 127 = {ds:.6f}, q = round(x / Delta), decoded Delta q. "
            f"With a zero point: Delta = (6 - (-1)) / 255 = {da:.6f}, q_0 = round(1 / Delta) = {q0}, q = clamp(round(x / Delta) + q_0, 0, 255), "
            "decoded Delta (q - q_0).\n\n")
    return head + _table(["x", "q symmetric", "decoded", "abs. error", "q with zero point", "decoded", "abs. error"], rows)


INT4_WEIGHTS = [0.5, -0.2, 0.7, 0.1, -0.7, 0.3, 0.0, -0.4]


def int4_pack():
    delta, q = _sym_codes(INT4_WEIGHTS, 7)
    rows, packed = [], []
    for j in range(0, 8, 2):
        lo, hi = _pattern(q[j], 4), _pattern(q[j + 1], 4)
        byte = lo | (hi << 4)
        packed.append(byte)
        rows.append([j // 2, f"w[{j}] = {INT4_WEIGHTS[j]}", q[j], _bits(lo, 4), f"w[{j + 1}] = {INT4_WEIGHTS[j + 1]}", q[j + 1],
                     _bits(hi, 4), _bits(byte, 8), _hex(byte, 8)])
    back = []
    for byte in packed:
        back += [_signed(byte & 0xF, 4), _signed(byte >> 4, 4)]
    head = (f"Eight weights {INT4_WEIGHTS}; Delta = max|w| / 7 = {delta:.1f}, q = round(w / Delta) in -7 .. 7. Byte b holds the even-index code "
            "in its low four bits and the odd-index code in its high four bits: byte = (lo and 0xF) or ((hi and 0xF) << 4).\n\n")
    tail = (f"\n\nUnpacking (low = byte and 0xF, high = byte >> 4, each read as a 4-bit two's complement number) returns the codes {back}; "
            f"they equal the codes {q}: {back == q}. The eight weights now take 4 bytes plus one scale.")
    return head + _table(["byte", "even element", "q", "low nibble", "odd element", "q", "high nibble", "byte (binary)", "hex"], rows) + tail


# ------------------------------------------------------------------ conversion and rounding

def bf16_rounding_bias():
    rng = bookrng.generator("number_formats.bf16_rounding_bias")
    xs = [f32(rng.normal()) for _ in range(ROUNDING_DRAWS)]
    sums = {"exact": (sum(abs(x) for x in xs), sum(x * x for x in xs))}
    rows = []
    for name, fn in [("round to nearest even", bf16_rne_bits), ("truncate (drop the low 16 bits)", bf16_trunc_bits)]:
        ys = [bits_f32(fn(f32_bits(x)) << 16) for x in xs]
        mean_rel = sum((abs(y) - abs(x)) / abs(x) for x, y in zip(xs, ys) if x != 0) / len(xs)
        worst = max(abs(y - x) / abs(x) for x, y in zip(xs, ys) if x != 0)
        s1 = sum(abs(y) for y in ys) / sums["exact"][0] - 1
        s2 = sum(y * y for y in ys) / sums["exact"][1] - 1
        toward = sum(1 for x, y in zip(xs, ys) if abs(y) < abs(x)) / len(xs)
        rows.append([name, f"{100 * mean_rel:+.4f}%", f"{100 * worst:.4f}%", f"{100 * toward:.1f}%", f"{100 * s1:+.4f}%", f"{100 * s2:+.4f}%"])
    # agreement of the integer shortcut with the exact rule on the first 2000 draws
    agree = sum(1 for x in xs[:2000] if encode(x, "bf16") == bf16_rne_bits(f32_bits(x)))
    seed = bookrng.seed_of("number_formats.bf16_rounding_bias")
    head = (f"{ROUNDING_DRAWS} values x_i = f32(normal()) drawn in order from the generator seeded {seed} (seed_of "
            "'number_formats.bf16_rounding_bias'), each rounded to bf16. 'mean change of |x|' is the mean of (|y| - |x|) / |x|; "
            "'worst relative error' is the largest |y - x| / |x|; 'moved toward zero' is the share with |y| < |x|; "
            "the last two columns are the relative error of the sums of |x_i| and of x_i^2. "
            f"The integer rule agreed with the exact rounding rule on {agree} of the first 2000 values.\n\n")
    return head + _table(["rounding", "mean change of |x|", "worst relative error", "moved toward zero", "error of sum |x|", "error of sum x^2"], rows)


def accumulator_stall():
    rows = []
    for f in SMALL:
        nm = FORMATS[f][1]
        s, n = 0.0, 0
        while n < 100000:
            t = quantize(s + 1.0, f)
            if t == s:
                break
            s, n = t, n + 1
        rows.append([f, nm + 1, f"2^{nm + 1} = {2 ** (nm + 1)}", int(s), n])
    x = 2.0 ** 24
    r32 = f32(x + 1.0)
    rows.append(["fp32 (one step)", 24, "2^24 = 16777216", int(r32), "16777216 + 1 stores " + str(int(r32))])
    head = "Adding 1.0 repeatedly to a running total that starts at 0, rounding to the format after each addition (ties to even); the loop stops when an addition changes nothing.\n\n"
    return head + _table(["format", "significand bits p = n_m + 1", "predicted stall 2^p", "total where it stalls", "additions made"], rows)


def update_absorbed():
    w = 1.0
    ups = [0.0005, 0.001, 0.002, 0.004, 0.005, 0.01]
    rows = []
    for u in ups:
        r = [u]
        for f in ["fp32", "fp16", "bf16"]:
            v = quantize(w + u, f) if f != "fp32" else f32(w + u)
            r.append(_num(v - w, 5))
        rows.append(r)
    gaps = ["gap above 1 = 2^-n_m"] + [_num(2.0 ** -FORMATS[f][1], 4) for f in ["fp32", "fp16", "bf16"]]
    rows.append(gaps)
    head = ("The weight is w = 1. The update u is added in exact arithmetic and the sum is rounded once to the format (ties to even); "
            "the cell is the change that was stored, (stored sum) - 1.\n\n")
    return head + _table(["update u", "fp32", "fp16", "bf16"], rows)


def format_relations():
    rows = []
    # widening is exact
    bad = 0
    n = 0
    for f in SMALL:
        for c in range(1 << width(f)):
            v = decode(c, f)
            if v == v and not math.isinf(v):
                n += 1
                if f32(v) != v:
                    bad += 1
    rows.append(["every finite value of fp16, bf16, E4M3, E5M2 is an fp32 value", n, bad])
    # bf16 is the top half of fp32
    bad = sum(1 for c in range(1 << 16) if (lambda v, w: not (v != v and w != w) and v != w)(decode(c, "bf16"), bits_f32(c << 16)))
    rows.append(["bf16 code c decodes like the fp32 pattern c << 16", 1 << 16, bad])
    # E5M2 is the top byte of fp16
    bad = sum(1 for c in range(256) if (lambda v, w: not (v != v and w != w) and v != w)(decode(c, "E5M2"), decode(c << 8, "fp16")))
    rows.append(["E5M2 code c decodes like the fp16 pattern c << 8", 256, bad])
    # order: positive codes sort like their values
    ok = 0
    tot = 0
    for f in SMALL:
        vals = positive_values(f)
        tot += 1
        ok += all(vals[i][0] < vals[i + 1][0] and vals[i][1] < vals[i + 1][1] for i in range(len(vals) - 1))
    rows.append(["the finite positive codes of each format increase with their values", tot, tot - ok])
    # encode(decode(c)) == c for every finite code; the one exception is the negative-zero code of each format,
    # because a real number has no sign of zero: it decodes to 0 and encodes as the positive-zero code
    n_codes = 0
    bad = 0
    neg_zero = 0
    neg_zero_bad = 0
    for f in SMALL:
        for c in range(1 << width(f)):
            v = decode(c, f)
            if v != v or math.isinf(v):
                continue
            if v == 0 and c != 0:
                neg_zero += 1
                neg_zero_bad += encode(0.0, f) != 0
            else:
                n_codes += 1
                bad += encode(v, f) != c
    rows.append(["encode(decode(c)) = c for every finite code, negative zero excepted", n_codes, bad])
    rows.append(["the negative-zero code decodes to 0 and 0 encodes as the positive-zero code (sign of zero is dropped)",
                 neg_zero, neg_zero_bad])
    return _table(["relation", "cases checked", "violations"], rows)


# ------------------------------------------------------------------ memory

def memory_per_format():
    fp8_scale = 4 / (128 * 128)
    entries = [
        ("fp32", 4.0, "none"),
        ("fp16 or bf16", 2.0, "none"),
        ("E4M3 with an f32 scale per 128 x 128 block", 1 + fp8_scale, "4 bytes per 16384 values"),
        ("int8 with an f32 scale per 32 values", 1 + 4 / 32, "4 bytes per 32 values"),
        ("int4 with an f32 scale per 32 values", 0.5 + 4 / 32, "4 bytes per 32 values"),
    ]
    rows = []
    for name, b, note in entries:
        gb = PARAMS * b / 1e9
        rows.append([name, f"{b:.4f}", f"{gb:.3f}", f"{b / 4:.3f}", f"{2 / b:.2f}"])
    head = (f"A dense model of P = {PARAMS:,} parameters. Producing one token at batch size 1 reads every weight once and does "
            "2 FLOPs per weight (a multiply and an add), so the bytes moved per token are P times the bytes per parameter and the "
            "FLOPs per byte are 2 divided by the bytes per parameter. GB = 10^9 bytes.\n\n")
    return head + _table(["storage of the weights", "bytes per parameter", "GB for the weights = bytes read per token", "fraction of fp32",
                          "FLOPs per byte"], rows)


def training_bytes():
    block = 1 + 4 / 64
    policies = [
        ("all fp32", 4, 4, 4, 0),
        ("bf16 value and gradient, fp32 moments, fp32 master", 2, 2, 4, 4),
        ("bf16 value and gradient, bf16 moments, fp32 master", 2, 2, 2, 4),
        ("bf16 value and gradient, int8-block moments, fp32 master", 2, 2, block, 4),
        ("bf16 everything, no master", 2, 2, 2, 0),
    ]
    rows = []
    for name, v, g, m, ms in policies:
        tot = v + g + 2 * m + ms
        rows.append([name, v, g, f"{2 * m:g}", ms, f"{tot:g}", f"{PARAMS * tot / 1e9:.1f}"])
    head = ("Bytes of training memory per parameter for AdamW (two moment slots, each in the moment format), as the sum of the value, "
            "the gradient, the moments and the optional master copy; int8-block moments cost 1 + 4/64 = 1.0625 bytes each. "
            f"The last column is the total for {PARAMS:,} parameters in decimal GB.\n\n")
    return head + _table(["policy", "value", "gradient", "moments (2 slots)", "master", "total bytes", "GB for 7e9"], rows)


# ------------------------------------------------------------------ data for the figures

def density_by_decade(f):
    """{decade k: number of finite positive values v of format f with 10^k <= v < 10^(k+1)}."""
    out = {}
    for v, _ in positive_values(f):
        k = math.floor(math.log10(v))
        out[k] = out.get(k, 0) + 1
    return out


def error_envelope(f):
    """[(log10 x, largest relative rounding error)] for each binade [2^e, 2^(e+1)) of the positive values of f.

    The largest error of a binade is at a midpoint between two neighbours, where
    rounding errs by half the gap: (gap / 2) / midpoint."""
    vals = [0.0] + [v for v, _ in positive_values(f)]
    best = {}
    for a, b in zip(vals, vals[1:]):
        mid = (a + b) / 2
        e = exponent_of(Fraction(mid))
        err = (b - a) / 2 / mid
        best[e] = max(best.get(e, 0.0), err)
    return [((e + 0.5) * math.log10(2.0), best[e]) for e in sorted(best)]
