# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Quantization: Fewer Bits per Weight".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: quantization.<name> -->` markers. Helpers with a leading
underscore, or that take arguments, are shared with the chapter's figures so a
picture and the table beside it cannot disagree.

Conventions fixed once for the whole module:

* rounding of a quotient to a code is round to nearest, ties to even (Python's
  `round` on a float is exactly that rule);
* a scale that is stored is rounded to the stored format first (binary16 for
  `f16`, binary32 for `f32`, through `struct`), and the codes are computed
  against the stored scale, so that the decoder sees the scale the encoder used;
* the error of a weight is `w' - w` (decoded minus original), its mean square
  is the MSE, and the signal-to-quantization-noise ratio is
  `10 log10(mean(w^2) / MSE)` decibels;
* a random draw comes from the book's generator, `bookrng.generator(name)`,
  and each function states the order of its draws.

The Q4_K block is built and read here from the layout the chapter derives
(field offsets, 6-bit scale packing, nibble order), not from any library.

Swedish Embedded AB implements solutions for storing machine learning models in
few bits without losing the answers that matter. If your team needs expertise in
weight quantization and mixed-precision deployment, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import math
import struct

import bookrng
import number_formats as nf

# ---------------------------------------------------------------- formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _g(v, d=4):
    """d significant digits, plain for moderate sizes, scientific otherwise."""
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e5:
        return f"{v:.{d}g}"
    return f"{v:.{d - 1}e}"


def _seed_line(name):
    return f"seed {bookrng.seed_of(name)} (seed_of '{name}', defined on the pseudocode page)"


# ---------------------------------------------------------------- number helpers


def _round_scale(delta, bits):
    """The scale as it is stored: binary16, binary32 or (for 64) exactly."""
    if bits == 16:
        return struct.unpack("<e", struct.pack("<e", delta))[0]
    if bits == 32:
        return struct.unpack("<f", struct.pack("<f", delta))[0]
    return delta


def _f16_bits(v):
    return struct.unpack("<H", struct.pack("<e", v))[0]


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def _sqnr_db(w, wq):
    """10 log10 (mean w^2 / MSE) in decibels; infinity when the error is zero."""
    e = _mse(w, wq)
    p = sum(x * x for x in w) / len(w)
    return math.inf if e == 0 else 10 * math.log10(p / e)


# ---------------------------------------------------------------- the quantizers


def affine_params(w, nbits):
    """(delta, q0): the scale and zero point that spread the range of `w`,
    extended to contain 0, over the unsigned codes 0 .. 2^nbits - 1."""
    lo, hi = min(min(w), 0.0), max(max(w), 0.0)
    delta = (hi - lo) / (2 ** nbits - 1)
    q0 = _clamp(round(-lo / delta), 0, 2 ** nbits - 1)
    return delta, q0


def quantize_affine(w, nbits, delta, q0):
    return [_clamp(round(x / delta) + q0, 0, 2 ** nbits - 1) for x in w]


def dequantize_affine(q, delta, q0):
    return [delta * (c - q0) for c in q]


def symmetric_group(w, nbits, scale_bits=16):
    """One group, codes -qmax .. qmax with qmax = 2^(nbits-1) - 1 and
    delta = max|w| / qmax stored in `scale_bits`; returns (delta, codes)."""
    qmax = 2 ** (nbits - 1) - 1
    amax = max(abs(x) for x in w)
    if amax == 0:
        return 0.0, [0] * len(w)
    delta = _round_scale(amax / qmax, scale_bits)
    return delta, [_clamp(round(x / delta), -qmax, qmax) for x in w]


def symmetric_roundtrip(w, nbits, group, scale_bits=16):
    """Decoded weights when each consecutive `group` weights share one symmetric scale."""
    out = []
    for i in range(0, len(w), group):
        delta, q = symmetric_group(w[i:i + group], nbits, scale_bits)
        out.extend(delta * c for c in q)
    return out


def q4_0_roundtrip(w):
    """GGUF Q4_0 layout: 32 weights, f16 d = (the signed element of largest
    magnitude) / -8, code = clamp(round(w / d) + 8, 0, 15), w' = d (code - 8)."""
    out = []
    for i in range(0, len(w), 32):
        blk = w[i:i + 32]
        ext = 0.0
        for x in blk:
            if abs(x) > abs(ext):
                ext = x
        d = _round_scale(ext / -8, 16) if ext != 0 else 0.0
        out.extend(d * (_clamp(round(x / d) + 8, 0, 15) - 8) if d != 0 else 0.0 for x in blk)
    return out


# ---------------------------------------------------------------- K-quant superblock


def kq_encode(w256, nbits):
    """Two-level affine quantization of one super-block of 256 weights.

    Eight sub-blocks of 32. Sub-block j: lo, hi its extremes; floor m_j =
    max(0, -lo); step D_j = (hi + m_j) / (2^nbits - 1). The super-block keeps
    d = max D_j / 63 and dmin = max m_j / 63 (both stored in f16), and each
    sub-block the 6-bit integers sc_j = round(D_j / d), mm_j = round(m_j / dmin).
    Codes are found against the stored scales: q = clamp(round((w + dmin mm_j)
    / (d sc_j)), 0, 2^nbits - 1)."""
    assert len(w256) == 256
    qmax = 2 ** nbits - 1
    ideal_step, ideal_floor = [], []
    for j in range(8):
        sub = w256[32 * j:32 * j + 32]
        lo, hi = min(sub), max(sub)
        m = max(0.0, -lo)
        ideal_step.append((hi + m) / qmax if hi + m > 0 else 0.0)
        ideal_floor.append(m)
    d = _round_scale(max(ideal_step) / 63, 16)
    dmin = _round_scale(max(ideal_floor) / 63, 16)
    sc = [_clamp(round(s / d), 0, 63) if d > 0 else 0 for s in ideal_step]
    mm = [_clamp(round(m / dmin), 0, 63) if dmin > 0 else 0 for m in ideal_floor]
    q = []
    for j in range(8):
        step, floor = d * sc[j], dmin * mm[j]
        for x in w256[32 * j:32 * j + 32]:
            q.append(_clamp(round((x + floor) / step), 0, qmax) if step > 0 else 0)
    return {"nbits": nbits, "d": d, "dmin": dmin, "sc": sc, "mm": mm, "q": q,
            "ideal_step": ideal_step, "ideal_floor": ideal_floor}


def kq_direct(f):
    """The direct formula w' = d sc_j q - dmin mm_j for every weight, from the fields."""
    return [f["d"] * f["sc"][i // 32] * c - f["dmin"] * f["mm"][i // 32] for i, c in enumerate(f["q"])]


def kq_roundtrip(w, nbits):
    out = []
    for i in range(0, len(w), 256):
        out.extend(kq_direct(kq_encode(w[i:i + 256], nbits)))
    return out


# the 12-byte packing of eight 6-bit scales and eight 6-bit mins (Q4_K and Q5_K)


def pack_scale_min(sc, mm):
    s = [0] * 12
    for j in range(4):
        s[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6)
        s[j + 4] = (mm[j] & 63) | ((mm[j + 4] >> 4) << 6)
        s[j + 8] = (sc[j + 4] & 15) | ((mm[j + 4] & 15) << 4)
    return s


def scale_min(j, s):
    """(sc_j, m_j) read back from the 12 packed bytes."""
    if j < 4:
        return s[j] & 63, s[j + 4] & 63
    return (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)


# the fields of the 144-byte Q4_K super-block, in file order, with their sizes in bytes
Q4K_FIELDS = (("d", 2), ("dmin", 2), ("scales", 12), ("qs", 128))


def q4k_offsets():
    """name -> (first byte, size) of every field of the Q4_K super-block."""
    out, at = {}, 0
    for name, size in Q4K_FIELDS:
        out[name] = (at, size)
        at += size
    return out


def q4k_pack(f):
    """The 144 bytes of a Q4_K super-block from the fields of `kq_encode(.., 4)`."""
    out = list(struct.pack("<e", f["d"])) + list(struct.pack("<e", f["dmin"]))
    out += pack_scale_min(f["sc"], f["mm"])
    q = f["q"]
    for chunk in range(4):
        for l in range(32):
            out.append(q[64 * chunk + l] | (q[64 * chunk + 32 + l] << 4))
    assert len(out) == sum(size for _, size in Q4K_FIELDS)
    return out


def q4k_decode(b):
    """Decode 144 bytes the way the file format defines it: 64 weights per
    chunk of 32 bytes, low nibbles are sub-block 2c and high nibbles 2c+1."""
    d = struct.unpack("<e", bytes(b[0:2]))[0]
    dmin = struct.unpack("<e", bytes(b[2:4]))[0]
    s = b[4:16]
    qs = b[16:144]
    out = []
    for chunk in range(4):
        sc1, m1 = scale_min(2 * chunk, s)
        sc2, m2 = scale_min(2 * chunk + 1, s)
        for l in range(32):
            out.append(d * sc1 * (qs[32 * chunk + l] & 15) - dmin * m1)
        for l in range(32):
            out.append(d * sc2 * (qs[32 * chunk + l] >> 4) - dmin * m2)
    return out


# ---------------------------------------------------------------- shared datasets

HEAVY_N = 16384
HEAVY_SIGMA = 0.02
HEAVY_P = 0.004
HEAVY_FACTOR = 10.0


def heavy_weights(name="quantization.heavy_tailed", n=HEAVY_N, outliers=True):
    """n weights 0.02 z_i, with z_i drawn first (n normals) and then, for the outlier flags, n uniforms u_i;
    with outliers, the weight is multiplied by 10 where u_i < 0.004. Returns (w, flags)."""
    g = bookrng.generator(name)
    z = [g.normal() for _ in range(n)]
    u = [g.uniform() for _ in range(n)]
    flags = [ui < HEAVY_P for ui in u]
    w = [HEAVY_SIGMA * zi * (HEAVY_FACTOR if (outliers and f) else 1.0) for zi, f in zip(z, flags)]
    return w, flags


def gaussian_weights(name, n, sigma=1.0):
    g = bookrng.generator(name)
    return [sigma * g.normal() for _ in range(n)]


# ================================================================ chapter blocks


def eight_weights():
    name = "quantization.eight_weights"
    g = bookrng.generator(name)
    w = [round(0.03 + 0.05 * g.normal(), 4) for _ in range(8)]
    # symmetric, codes -7 .. 7
    amax = max(abs(x) for x in w)
    ds = amax / 7
    qs_ = [_clamp(round(x / ds), -7, 7) for x in w]
    ws = [ds * c for c in qs_]
    # affine, codes 0 .. 15
    da, q0 = affine_params(w, 4)
    qa = quantize_affine(w, 4, da, q0)
    wa = dequantize_affine(qa, da, q0)
    rows = []
    for i in range(8):
        rows.append((i, f"{w[i]:+.4f}", qs_[i], f"{ws[i]:+.5f}", f"{ws[i] - w[i]:+.5f}",
                     qa[i], f"{wa[i]:+.5f}", f"{wa[i] - w[i]:+.5f}"))
    t = _table(["i", "w", "q sym", "w' sym", "error sym", "q affine", "w' affine", "error affine"], rows)
    e_s = [ws[i] - w[i] for i in range(8)]
    e_a = [wa[i] - w[i] for i in range(8)]
    head = (f"Eight weights w_i = round(0.03 + 0.05 z_i, 4), z_i = normal() drawn in order i = 0 .. 7 from the generator, {_seed_line(name)}. "
            f"Symmetric: Delta = max|w| / 7 = {amax:.4f} / 7 = {ds:.6f}, q = clamp(round(w / Delta), -7, 7), w' = Delta q. "
            f"Affine: range extended to contain 0, [{min(min(w), 0):+.4f}, {max(max(w), 0):+.4f}], Delta = (max - min) / 15 = {da:.6f}, "
            f"q0 = round(-min / Delta) = {q0}, q = clamp(round(w / Delta) + q0, 0, 15), w' = Delta (q - q0).\n\n")
    tail = (f"\n\nSymmetric: largest |error| {max(abs(e) for e in e_s):.5f} (bound Delta/2 = {ds / 2:.5f}), MSE {_mse(w, ws):.3e}, "
            f"codes used {len(set(qs_))} of 15. Affine: largest |error| {max(abs(e) for e in e_a):.5f} (bound Delta/2 = {da / 2:.5f}), "
            f"MSE {_mse(w, wa):.3e}, codes used {len(set(qa))} of 16.")
    return head + t + tail


def uniform_error_model():
    name = "quantization.uniform_error"
    n = 100_000
    w = gaussian_weights(name, n)
    power = sum(x * x for x in w) / n
    rows, prev = [], None
    for nb in range(2, 9):
        qmax = 2 ** (nb - 1) - 1
        delta = max(abs(x) for x in w) / qmax
        wq = [delta * _clamp(round(x / delta), -qmax, qmax) for x in w]
        mse = _mse(w, wq)
        pred = delta ** 2 / 12
        sq = 10 * math.log10(power / mse)
        worst = max(abs(a - b) for a, b in zip(w, wq))
        rows.append((nb, 2 * qmax + 1, f"{delta:.5f}", f"{pred:.4e}", f"{mse:.4e}", f"{mse / pred:.3f}",
                     f"{worst / (delta / 2):.3f}", f"{sq:.2f}", "" if prev is None else f"{sq - prev:+.2f}"))
        prev = sq
    head = (f"{n} weights w_i = normal() drawn in order from the generator, {_seed_line(name)} (mean w^2 = {power:.4f}). One scale for all of them, "
            "Delta = max|w| / q_max with q_max = 2^(n-1) - 1, q = clamp(round(w / Delta), -q_max, q_max), w' = Delta q; "
            "'predicted' is Delta^2 / 12, 'ratio' is measured over predicted, 'worst / (Delta/2)' the largest |error| in units of the bound, "
            "SQNR = 10 log10(mean w^2 / MSE) in dB, 'gain' its change from the row above.\n\n")
    t = _table(["n bits", "levels", "Delta", "predicted MSE", "measured MSE", "ratio", "worst / (Delta/2)", "SQNR dB", "gain"], rows)
    return head + t


def outliers():
    name = "quantization.heavy_tailed"
    n = HEAVY_N
    rows = []
    for label, with_out in (("no outliers", False), ("with outliers", True)):
        w, flags = heavy_weights(name, outliers=with_out)
        keep = [i for i in range(n) if not flags[i]]
        wk = [w[i] for i in keep]
        for scope, grp in (("one scale per tensor", n), ("one scale per 32", 32)):
            wq = symmetric_roundtrip(w, 4, grp, 32)
            zero = sum(1 for i in keep if wq[i] == 0) / len(keep)
            mse_keep = _mse(wk, [wq[i] for i in keep])
            top = max(abs(x) for x in w) / 7 if grp == n else None
            rows.append((label, scope, "" if top is None else f"{top:.5f}", f"{100 * zero:.1f}%", f"{mse_keep:.3e}",
                         f"{10 * math.log10((sum(x * x for x in wk) / len(wk)) / mse_keep):.2f}"))
    n_out = sum(1 for f in heavy_weights(name)[1] if f)
    head = (f"{n} weights, {_seed_line(name)}: first z_i = normal() for i = 0 .. {n - 1}, then u_i = uniform() for i = 0 .. {n - 1}; "
            f"w_i = 0.02 z_i, and in the 'with outliers' rows w_i = 0.2 z_i where u_i < 0.004 ({n_out} weights; the 'no outliers' rows ignore u). "
            "4-bit symmetric codes -7 .. 7, scale = max|group| / 7 stored as f32. The columns after the scale are computed over the weights that are not outliers "
            "(the same indices in both datasets): the share that decode to 0, their MSE and their SQNR in dB. 'Delta' is the single tensor scale.\n\n")
    t = _table(["data", "scales", "Delta", "non-outliers decoded as 0", "MSE of non-outliers", "SQNR of non-outliers dB"], rows,
               align="llrrrr")
    return head + t


def group_size_tradeoff():
    name = "quantization.heavy_tailed"
    w, _ = heavy_weights(name)
    wg = gaussian_weights("quantization.group_gauss", HEAVY_N, HEAVY_SIGMA)
    rows = []
    for G in (HEAVY_N, 4096, 1024, 256, 128, 64, 32, 16, 8, 4):
        bpw = 4 + 16 / G
        s_h = _sqnr_db(w, symmetric_roundtrip(w, 4, G, 16))
        s_g = _sqnr_db(wg, symmetric_roundtrip(wg, 4, G, 16))
        rows.append((G, f"{bpw:.4f}", f"{bpw / 8:.5f}", f"{100 * (16 / G) / 4:.2f}%", f"{s_h:.2f}", f"{s_g:.2f}"))
    head = ("4-bit symmetric codes -7 .. 7, scale max|group| / 7 stored as f16 (16 bits), so bits per weight = 4 + 16 / G. "
            f"'heavy-tailed': the {HEAVY_N} weights of the previous example with outliers ({_seed_line(name)}). "
            f"'gaussian': {HEAVY_N} weights 0.02 normal() in draw order from the generator, {_seed_line('quantization.group_gauss')}. "
            "'scale overhead' is the scale bits as a share of the 4 code bits. SQNR in dB over all weights.\n\n")
    t = _table(["G", "bits per weight", "bytes per weight", "scale overhead", "SQNR heavy-tailed dB", "SQNR gaussian dB"], rows)
    return head + t


def factored_dot():
    """Dequantize-then-multiply against the factored dot product of weight-only quantization, one group of 32."""
    name = "quantization.factored_dot"
    g = bookrng.generator(name)
    w = [round(0.05 * g.normal(), 4) for _ in range(32)]
    x = [round(g.normal(), 4) for _ in range(32)]
    delta, q0 = affine_params(w, 4)
    q = quantize_affine(w, 4, delta, q0)
    wd = dequantize_affine(q, delta, q0)
    direct = sum(a * b for a, b in zip(x, wd))
    sxq = sum(a * c for a, c in zip(x, q))
    sx = sum(x)
    fact = delta * (sxq - q0 * sx)
    exact = sum(a * b for a, b in zip(x, w))
    rows = [("exact: sum x_j w_j", f"{exact:.8f}"),
            ("dequantize first: sum x_j Delta (q_j - q0)", f"{direct:.8f}"),
            ("factored: Delta (sum x_j q_j - q0 sum x_j)", f"{fact:.8f}"),
            ("sum x_j q_j", f"{sxq:.6f}"), ("sum x_j (shared by every row of W)", f"{sx:.6f}"),
            ("difference of the two quantized forms", "below 1e-15" if abs(direct - fact) < 1e-15 else f"{abs(direct - fact):.2e}")]
    head = (f"One group of 32 weights w_j = round(0.05 z, 4) then 32 activations x_j = round(normal(), 4), all normal() in draw order from the generator, {_seed_line(name)}. "
            f"4-bit affine codes: Delta = {delta:.6f}, q0 = {q0}. Every sum is in binary64 (double precision), added from j = 0 to 31 in order; each form is then rounded to the digits shown. "
            "The difference of the two quantized forms is a few units of the last place of a double (an implementation that adds in another order, or fuses a multiply-add, gets 0 or about 2.2e-16): the table prints only that it is below 1e-15.\n\n")
    return head + _table(["form", "value"], rows, align="lr")


def block_formats():
    """Byte accounting of the formats of the chapter, from their field lists."""
    fmt = [
        ("fp16", 1, [("value", 2)]),
        ("Q8_0", 32, [("d (f16)", 2), ("32 int8 codes", 32)]),
        ("Q4_0", 32, [("d (f16)", 2), ("32 nibbles", 16)]),
        ("Q5_0", 32, [("d (f16)", 2), ("32 fifth bits", 4), ("32 low nibbles", 16)]),
        ("Q4_K", 256, [("d (f16)", 2), ("dmin (f16)", 2), ("8 sc + 8 m, 6 bits each", 12), ("256 nibbles", 128)]),
        ("Q5_K", 256, [("d (f16)", 2), ("dmin (f16)", 2), ("8 sc + 8 m, 6 bits each", 12), ("256 fifth bits", 32), ("256 low nibbles", 128)]),
        ("Q6_K", 256, [("256 low nibbles", 128), ("256 high 2-bit pairs", 64), ("16 sc (int8)", 16), ("d (f16)", 2)]),
    ]
    rows = []
    for nm, n, fields in fmt:
        b = sum(sz for _, sz in fields)
        rows.append((nm, n, " + ".join(f"{sz}" for _, sz in fields), b, f"{8 * b / n:.4f}", f"{b / n:.5f}"))
    head = ("Each format is a block of weights and the bytes of its fields (the order is the order in the file, except that the fp16 'block' is one value). "
            "Bits per weight = 8 x bytes / weights.\n\n")
    return head + _table(["format", "weights per block", "field bytes", "bytes per block", "bits per weight", "bytes per weight"], rows)


def decode_traffic():
    P = 7_000_000_000
    fmts = [("fp16", 2.0, None), ("Q8_0", 34 / 32, 32), ("Q6_K", 210 / 256, 16), ("Q5_K", 176 / 256, 32), ("Q4_K", 144 / 256, 32),
            ("Q4_0", 18 / 32, 32), ("brain int4, f32 scale per 32", (16 + 4) / 32, 32)]
    rows = []
    for nm, b, grp in fmts:
        gb = P * b / 1e9
        rows.append((nm, f"{b:.5f}", f"{gb:.3f}", f"{gb / (P * 2 / 1e9):.4f}", f"{2 / b:.3f}",
                     "2" if grp is None else f"{2 + 3 / grp:.3f}", "2" if grp is None else "3"))
    head = (f"A dense model of P = {P:,} weights read once per token (batch size 1), as in Chapter 3: bytes per token = P b, FLOPs per token = 2 P, "
            "intensity 2 / b. GB = 10^9 bytes. 'fused' is the FLOPs per weight when the scale is applied once per group of G weights "
            "(2 for the fused multiply-add of x_j q_j, plus about 3 / G per weight for the group's multiply, subtract and add, taking G = 32 except Q6_K's 16); "
            "'dequantize first' is 3 per weight (convert and scale, then the fused multiply-add).\n\n")
    return head + _table(["stored as", "bytes per weight", "GB per token", "bytes vs fp16", "FLOP per byte", "FLOPs per weight, fused", "FLOPs per weight, dequantize first"], rows)


# -------------------------------------------------------------- the Q4_K block, byte by byte


def _q4k_example():
    name = "quantization.q4k_block"
    g = bookrng.generator(name)
    sig = [0.02 * (1 + j % 3) for j in range(8)]
    mu = [0.01 * (j - 3) for j in range(8)]
    w = []
    for j in range(8):
        for _ in range(32):
            w.append(round(mu[j] + sig[j] * g.normal(), 4))
    f = kq_encode(w, 4)
    return name, w, f, q4k_pack(f), sig, mu


def q4k_inputs():
    name, w, f, b, sig, mu = _q4k_example()
    rows = []
    for j in range(8):
        sub = w[32 * j:32 * j + 32]
        rows.append((j, f"{sig[j]:.2f}", f"{mu[j]:+.2f}", f"{min(sub):+.4f}", f"{max(sub):+.4f}", f"{f['ideal_floor'][j]:.4f}",
                     f"{f['ideal_step'][j]:.6f}"))
    head = (f"One super-block of 256 weights, {_seed_line(name)}: for sub-block j = 0 .. 7 in order, 32 weights w = round(mu_j + sigma_j z, 4) with z = normal() "
            "drawn in order (256 draws in all), sigma_j = 0.02 (1 + j mod 3), mu_j = 0.01 (j - 3). "
            "Per sub-block: the smallest and largest weight, the floor m_j = max(0, -min) and the step D_j = (max + m_j) / 15 that would use all 16 codes.\n\n")
    return head + _table(["j", "sigma_j", "mu_j", "min", "max", "floor m_j", "step D_j"], rows)


def q4k_scales():
    name, w, f, b, sig, mu = _q4k_example()
    d, dmin = f["d"], f["dmin"]
    rows = []
    for j in range(8):
        step, floor = d * f["sc"][j], dmin * f["mm"][j]
        sub = w[32 * j:32 * j + 32]
        dec = [step * f["q"][32 * j + i] - floor for i in range(32)]
        worst = max(abs(a - c) for a, c in zip(sub, dec))
        rows.append((j, f["sc"][j], f["mm"][j], f"{step:.6f}", f"{floor:.5f}", f"{worst:.5f}", f"{worst / (step / 2):.3f}"))
    head = (f"The two-level scales of the super-block. d = max D_j / 63 = {max(f['ideal_step']):.6f} / 63 stored in f16 as {d:.8f} (bits 0x{_f16_bits(d):04X}); "
            f"dmin = max m_j / 63 = {max(f['ideal_floor']):.6f} / 63 stored as {dmin:.8f} (bits 0x{_f16_bits(dmin):04X}). "
            "sc_j = round(D_j / d) and mm_j = round(m_j / dmin) are the 6-bit integers (0 .. 63); effective step = d sc_j, effective floor = dmin mm_j; "
            "'largest error' is max |w' - w| in the sub-block and 'in units of step/2' divides it by half the effective step.\n\n")
    return head + _table(["j", "sc_j", "mm_j", "effective step", "effective floor", "largest error", "in units of step/2"], rows)


def q4k_bytes():
    name, w, f, b, sig, mu = _q4k_example()
    hexs = lambda xs: " ".join(f"{x:02X}" for x in xs)
    rows = []
    for r in range(9):
        lo = 16 * r
        rows.append((f"{lo:3d}..{lo + 15:3d}", hexs(b[lo:lo + 16])))
    t1 = _table(["bytes", "contents (hexadecimal)"], rows, align="ll")
    s = b[4:16]
    sc_rows = []
    for k in range(12):
        field = ("sc_%d (bits 0-5), sc_%d bits 4-5 (bits 6-7)" % (k, k + 4)) if k < 4 else \
                (("m_%d (bits 0-5), m_%d bits 4-5 (bits 6-7)" % (k - 4, k)) if k < 8 else
                 ("sc_%d bits 0-3 (low nibble), m_%d bits 0-3 (high nibble)" % (k - 4, k - 4)))
        sc_rows.append((4 + k, f"{s[k]:02X}", format(s[k], "08b"), field))
    t2 = _table(["byte", "hex", "binary", "holds"], sc_rows, align="rrll")
    back = [scale_min(j, s) for j in range(8)]
    ok = back == list(zip(f["sc"], f["mm"]))
    chunks = []
    for c in range(4):
        i0 = 16 + 32 * c
        chunks.append((f"{i0}..{i0 + 31}", f"{64 * c}..{64 * c + 31}", 2 * c, f"{64 * c + 32}..{64 * c + 63}", 2 * c + 1))
    t3 = _table(["bytes", "low nibbles: weights", "sub-block", "high nibbles: weights", "sub-block"], chunks, align="lrrrr")
    first = b[16:24]
    ex = [(16 + i, f"{first[i]:02X}", first[i] & 15, 64 * 0 + i, first[i] >> 4, 32 + i) for i in range(8)]
    t4 = _table(["byte", "hex", "low nibble q", "weight", "high nibble q", "weight"], ex, align="rrrrrr")
    head = "The 144 bytes of the super-block of the two previous tables (index of the first byte of each row at the left).\n\n"
    mid1 = "\n\nBytes 4 to 15, the twelve bytes that hold eight 6-bit scales and eight 6-bit mins:\n\n"
    mid2 = (f"\n\nUnpacking the twelve bytes with the rule of the chapter returns the pairs {back}; they equal (sc_j, mm_j): {ok}.\n\n"
            "The 128 code bytes, in four chunks of 32 bytes:\n\n")
    mid3 = "\n\nThe first eight bytes of the first chunk:\n\n"
    return head + t1 + mid1 + t2 + mid2 + t3 + mid3 + t4


def q4k_walk():
    """Decode two weights of the super-block by hand: one from a low nibble, one from a high nibble."""
    name, w, f, b, sig, mu = _q4k_example()
    dec = q4k_decode(b)
    rows = []
    for i in (5, 37):
        chunk, l = i // 64, i % 32
        byte = 16 + 32 * chunk + l
        high = (i % 64) >= 32
        nib = b[byte] >> 4 if high else b[byte] & 15
        j = i // 32
        sc, m = scale_min(j, b[4:16])
        rows.append((i, byte, f"{b[byte]:02X}", "high" if high else "low", nib, j, sc, m,
                     f"{f['d']:.8f} x {sc} x {nib} - {f['dmin']:.8f} x {m}", f"{dec[i]:+.5f}", f"{w[i]:+.4f}"))
    return _table(["weight i", "byte", "hex", "nibble", "q", "sub-block j", "sc_j", "m_j", "d sc_j q - dmin m_j", "decoded", "original"], rows,
                  align="rrrlrrrrlrr")


def q4k_check():
    name, w, f, b, sig, mu = _q4k_example()
    dec_bytes = q4k_decode(b)
    dec_direct = kq_direct(f)
    diff = max(abs(a - c) for a, c in zip(dec_bytes, dec_direct))
    err = [a - c for a, c in zip(dec_bytes, w)]
    mse = sum(e * e for e in err) / 256
    rows = [("bytes in the block", len(b)), ("bits per weight", f"{8 * len(b) / 256:.4f}"),
            ("largest |decode from bytes - direct formula|", f"{diff:.1e}"),
            ("MSE of the decoded weights", f"{mse:.4e}"),
            ("SQNR dB", f"{_sqnr_db(w, dec_bytes):.2f}"),
            ("largest |error|", f"{max(abs(e) for e in err):.5f}"),
            ("sum of all 144 bytes", sum(b))]
    return _table(["quantity", "value"], rows, align="lr")


def q4k_reassign():
    """Codes found against the stored scales (the rule of the chapter) against codes found against the ideal, unquantized scales."""
    name, w, f, b, sig, mu = _q4k_example()
    qmax = 15
    ideal = []
    for j in range(8):
        step, floor = f["ideal_step"][j], f["ideal_floor"][j]
        for x in w[32 * j:32 * j + 32]:
            ideal.append(_clamp(round((x + floor) / step), 0, qmax) if step > 0 else 0)
    stale = dict(f)
    stale["q"] = ideal
    rows = []
    for label, fields in (("codes against the stored scales (d sc_j, dmin mm_j)", f), ("codes against the ideal step D_j and floor m_j", stale)):
        dec = kq_direct(fields)
        rows.append((label, f"{_mse(w, dec):.4e}", f"{_sqnr_db(w, dec):.2f}", f"{max(abs(a - c) for a, c in zip(w, dec)):.5f}"))
    return ("Both rows decode with the stored scales d sc_j and dmin mm_j of the super-block above; they differ only in the scales the codes were found against.\n\n" +
            _table(["how the codes were chosen", "MSE", "SQNR dB", "largest |error|"], rows, align="lrrr"))


# ---------------------------------------------------------------- formats and error against bits


def _fmt_points(w):
    """(label, bits per weight, SQNR dB, MSE, max |error|) for every format of the comparison."""
    out = []

    def add(label, bpw, wq):
        out.append((label, bpw, _sqnr_db(w, wq), _mse(w, wq), max(abs(a - b) for a, b in zip(w, wq))))

    n = len(w)
    add("int4, one f32 scale for the tensor", 4 + 32 / n, symmetric_roundtrip(w, 4, n, 32))
    add("int4, f32 scale per 32 (brain int4)", 4 + 32 / 32, symmetric_roundtrip(w, 4, 32, 32))
    add("Q4_0 (f16 scale per 32)", 4 + 16 / 32, q4_0_roundtrip(w))
    add("Q4_K recipe (4-bit, two-level)", 4 + 128 / 256, kq_roundtrip(w, 4))
    add("Q5_K recipe (5-bit, two-level)", 5 + 128 / 256, kq_roundtrip(w, 5))
    add("Q8_0 (f16 scale per 32)", 8 + 16 / 32, symmetric_roundtrip(w, 8, 32, 16))
    add("int8, one f32 scale for the tensor", 8 + 32 / n, symmetric_roundtrip(w, 8, n, 32))
    return out


def format_comparison():
    name = "quantization.heavy_tailed"
    w, _ = heavy_weights(name)
    rows = [(l, f"{b:.4f}", f"{s:.2f}", f"{m:.3e}", f"{x:.5f}") for l, b, s, m, x in _fmt_points(w)]
    head = (f"The {HEAVY_N} heavy-tailed weights of the outlier example ({_seed_line(name)}, outliers included; max |w| = {max(abs(x) for x in w):.4f}) "
            "stored in each format. 'Q4_0': per group of 32, d = (signed element of largest magnitude) / -8 rounded to f16 (nearest, ties to even), code = clamp(round_even(w / d) + 8, 0, 15) (round_even: nearest, ties to even), w' = d (code - 8); a group of zeros decodes to zeros. 'recipe': eight sub-blocks of 32 with a 6-bit step and 6-bit "
            "floor each, an f16 d and dmin per 256 weights, as derived in the chapter (n-bit codes, 128 bits of scales per 256 weights). "
            "Bits per weight count every stored scale.\n\n")
    return head + _table(["format", "bits per weight", "SQNR dB", "MSE", "largest |error|"], rows)


def curve_points():
    """The three families of the error-against-bits figure on the heavy-tailed data: SQNR dB against bits per weight."""
    w, _ = heavy_weights("quantization.heavy_tailed")
    n = len(w)
    pt = [(nb + 32 / n, _sqnr_db(w, symmetric_roundtrip(w, nb, n, 32))) for nb in range(3, 9)]
    gw = [(nb + 0.5, _sqnr_db(w, symmetric_roundtrip(w, nb, 32, 16))) for nb in range(3, 9)]
    kq = [(nb + 0.5, _sqnr_db(w, kq_roundtrip(w, nb))) for nb in range(3, 8)]
    return {"one scale per tensor": pt, "scale per 32 (symmetric)": gw, "two-level recipe (affine)": kq}


def bits_curve():
    c = curve_points()
    rows = []
    for fam, pts in c.items():
        for i, (b, s) in enumerate(pts):
            rows.append((fam, f"{b:.3f}", f"{s:.2f}"))
    return "SQNR in dB against bits per weight for the three families of the figure, on the heavy-tailed weights; the codes have 3 to 8 bits.\n\n" + \
        _table(["family", "bits per weight", "SQNR dB"], rows, align="lrr")


def histogram_data():
    """(bin edges, counts, symmetric levels, affine levels) for 4096 gaussian weights, 4-bit codes."""
    name = "quantization.histogram"
    w = gaussian_weights(name, 4096, 0.02)
    delta = max(abs(x) for x in w) / 7
    sym = [delta * c for c in range(-7, 8)]
    da, q0 = affine_params(w, 4)
    aff = [da * (c - q0) for c in range(16)]
    return w, delta, sym, da, q0, aff


def histogram_summary():
    w, delta, sym, da, q0, aff = histogram_data()
    name = "quantization.histogram"
    return (f"4096 weights 0.02 normal() in draw order from the generator, {_seed_line(name)}; max |w| = {max(abs(x) for x in w):.5f}. "
            f"Symmetric levels: Delta = max|w| / 7 = {delta:.6f}, 15 levels from {sym[0]:+.4f} to {sym[-1]:+.4f}. "
            f"Affine levels: Delta = {da:.6f}, q0 = {q0}, 16 levels from {aff[0]:+.4f} to {aff[-1]:+.4f}. "
            f"Weights within half a step of zero: {100 * sum(1 for x in w if abs(x) < delta / 2) / len(w):.1f}% decode to 0 under the symmetric levels.")


# ---------------------------------------------------------------- the effect on a layer's output


def layer_output():
    name = "quantization.layer_output"
    g = bookrng.generator(name)
    n_out, k, n_in = 64, 256, 32
    W = [[0.05 * g.normal() for _ in range(k)] for _ in range(n_out)]
    X = [[g.normal() for _ in range(k)] for _ in range(n_in)]
    flat = [v for row in W for v in row]
    y = [[sum(a * b for a, b in zip(row, x)) for row in W] for x in X]
    ys = [v for r in y for v in r]
    rows = []
    for nb in range(2, 9):
        wq_flat = symmetric_roundtrip(flat, nb, 32, 16)
        Wq = [wq_flat[r * k:(r + 1) * k] for r in range(n_out)]
        yq = [[sum(a * b for a, b in zip(row, x)) for row in Wq] for x in X]
        yqs = [v for r in yq for v in r]
        sw = _sqnr_db(flat, wq_flat)
        so = _sqnr_db(ys, yqs)
        dot = sum(a * b for a, b in zip(ys, yqs))
        cos = dot / math.sqrt(sum(a * a for a in ys) * sum(b * b for b in yqs))
        rows.append((nb, f"{nb + 0.5:.1f}", f"{sw:.2f}", f"{so:.2f}", f"{so - sw:+.2f}", f"{cos:.6f}",
                     f"{100 * math.sqrt(_mse(ys, yqs) / (sum(a * a for a in ys) / len(ys))):.2f}%"))
    head = (f"A linear layer y = W x with W of shape [{n_out}, {k}], W_ij = 0.05 normal() drawn in row-major order, then {n_in} inputs x with entries normal() "
            f"drawn in order, one input after another, from the generator, {_seed_line(name)}. W is stored in n-bit symmetric codes with one f16 scale per 32 weights along each row "
            "(bits per weight = n + 0.5). 'weight SQNR' is over the weights, 'output SQNR' over all "
            f"{n_out * n_in} output values; 'relative output error' is the RMS error over the RMS output; cosine between the exact and the quantized outputs, both taken over the flattened vector of all {n_out * n_in} outputs (input after input, output unit within it); weight and output SQNR use the mean of the squared exact values as the signal power.\n\n")
    return head + _table(["n bits", "bits per weight", "weight SQNR dB", "output SQNR dB", "difference", "cosine", "relative output error"], rows)


# ---------------------------------------------------------------- floating-point scale against integer


def fp8_vs_int():
    name = "quantization.heavy_tailed"
    w, _ = heavy_weights(name)
    n = len(w)

    def block_rt(kind):
        out = []
        for i in range(0, n, 32):
            blk = w[i:i + 32]
            amax = max(abs(x) for x in blk)
            if kind == "int8":
                s = amax / 127
                out.extend(s * _clamp(round(x / s), -127, 127) for x in blk)
            elif kind == "int4":
                s = amax / 7
                out.extend(s * _clamp(round(x / s), -7, 7) for x in blk)
            else:
                top = 448.0 if kind == "E4M3" else 57344.0
                s = amax / top
                out.extend(s * nf.quantize(x / s, kind, saturate=True) for x in blk)
        return out

    kinds = ["int8", "E4M3", "E5M2", "int4"]
    rt = {k: block_rt(k) for k in kinds}
    # relative error by magnitude: t = |w| / (largest |w| of its group of 32)
    amaxes = [max(abs(x) for x in w[i - i % 32:i - i % 32 + 32]) for i in range(n)]
    edges = [1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32, 1 / 64, 0.0]
    rows = []
    for a, b in zip(edges, edges[1:]):
        idx = [i for i in range(n) if b <= abs(w[i]) / amaxes[i] < a or (a == 1.0 and abs(w[i]) / amaxes[i] == 1.0)]
        cells = []
        for k in kinds:
            r = [abs(rt[k][i] - w[i]) / abs(w[i]) for i in idx if w[i] != 0]
            cells.append(f"{100 * sum(r) / len(r):.2f}%")
        lab = f"[{b:.4g}, {a:.4g})" if a < 1.0 else f"[{b:.4g}, 1]"
        rows.append((lab, len(idx), *cells))
    summ = [("all weights, SQNR dB", "", *[f"{_sqnr_db(w, rt[k]):.2f}" for k in kinds])]
    head = (f"The {n} heavy-tailed weights ({_seed_line(name)}), stored in groups of 32 with one scale each: int8 (scale max|group| / 127), E4M3 (scale max|group| / 448, "
            "saturating, ties to even, with the decode of Chapter 1), E5M2 (scale max|group| / 57344) and int4 (scale max|group| / 7). "
            "Rows are bins of t = |w| / max|group| (the position of the weight under its group's largest); each cell is the mean of |w' - w| / |w| over the weights of the bin. "
            "Scales are kept exactly here, to isolate the codes.\n\n")
    return head + _table(["t of the weight", "weights", "int8", "E4M3", "E5M2", "int4"], rows + summ)


# ---------------------------------------------------------------- mixed-precision search

TIER_BITS = (2, 3, 4, 6, 8)
SCALE_BITS = 16
GROUP = 32


def _rels():
    """Relative MSE (MSE / variance) of n-bit symmetric codes with an f16 scale per 32, on 65536 standard normals."""
    w = gaussian_weights("quantization.rel", 65536)
    var = sum(x * x for x in w) / len(w)
    return {nb: _mse(w, symmetric_roundtrip(w, nb, GROUP, SCALE_BITS)) / var for nb in TIER_BITS}


def _tensors(n=12):
    """(size, sigma, sensitivity) of n tensors, rounded to 3 significant digits as printed."""
    g = bookrng.generator("quantization.tensors")
    out = []
    for _ in range(n):
        size = 4096 * (1 + g.integer(4))
        sigma = float(f"{0.02 * 2 ** (0.5 * g.normal()):.3g}")
        sens = float(f"{10 ** (2 * g.uniform()):.3g}")
        out.append((size, sigma, sens))
    return out


def _bytes(size, nb):
    return size * (nb + SCALE_BITS / GROUP) / 8


def _error(t, nb, kap):
    size, sigma, sens = t
    return 0.5 * sens * size * sigma ** 2 * kap[nb]


def rel_table():
    kap = _rels()
    rows = [(nb, f"{nb + SCALE_BITS / GROUP:.1f}", f"{kap[nb]:.4e}", f"{10 * math.log10(1 / kap[nb]):.2f}") for nb in TIER_BITS]
    head = ("The error of one weight at each tier: n-bit symmetric codes, f16 scale per 32 (bits per weight n + 0.5), measured as MSE over variance, the variance being the mean of w^2 (the weights have mean about 0, and it is not centred), on 65536 standard normals "
            f"normal() drawn in order from the generator, {_seed_line('quantization.rel')}. This relative MSE is the rel_n of the search below.\n\n")
    return head + _table(["n bits", "bits per weight", "rel_n", "SQNR dB"], rows)


def search_inputs():
    ts = _tensors()
    rows = [(i, f"{size:,}", f"{sigma:.3g}", f"{sens:.3g}") for i, (size, sigma, sens) in enumerate(ts)]
    head = ("Twelve tensors, i = 0 .. 11, each drawn in order from the generator, "
            f"{_seed_line('quantization.tensors')}: size N_i = 4096 (1 + integer(4)), sigma_i = 0.02 * 2^(0.5 normal()), sensitivity sens_i = 10^(2 uniform()); "
            "the draws of one tensor are made in the order integer, normal, uniform, and sigma_i and sens_i are rounded to 3 significant digits before use (the table prints them rounded). "
            "normal() is the book's Box-Muller with the held partner: tensor 0's normal() draws two uniforms and holds the sine partner, tensor 1's normal() returns the held value without drawing, tensor 2's draws two uniforms again, and so on (the held value carries across tensors). "
            "The estimated loss increase of tensor i stored at n bits is E_in = (1/2) sens_i N_i sigma_i^2 rel_n, its bytes N_i (n + 0.5) / 8.\n\n")
    return head + _table(["i", "N_i", "sigma_i", "sensitivity sens_i"], rows)


def _frontier(options):
    """options: list of (bytes, error, tag). Cheapest first; keep only options strictly more accurate than every cheaper one."""
    out = []
    for o in sorted(options, key=lambda t: (t[0], t[1])):
        if not out or o[1] < out[-1][1]:
            out.append(o)
    return out


def greedy(ts, tiers, budget, kap):
    """The multiple-choice knapsack by greedy upgrades. Every tensor starts at its cheapest tier; while some upgrade fits
    in the remaining bytes, take the one that removes the most error per extra byte (ties: lowest tensor index, then lowest tier)."""
    fronts = []
    for t in ts:
        fronts.append(_frontier([(_bytes(t[0], nb), _error(t, nb, kap), nb) for nb in tiers]))
    at = [0] * len(ts)
    used = sum(f[0][0] for f in fronts)
    assert used <= budget, "budget below the cheapest plan"
    while True:
        best = None
        for i, f in enumerate(fronts):
            here = f[at[i]]
            for to in range(at[i] + 1, len(f)):
                extra = f[to][0] - here[0]
                if used + extra > budget:
                    continue
                gain = (here[1] - f[to][1]) / extra
                if best is None or gain > best[0]:
                    best = (gain, i, to, extra)
        if best is None:
            break
        _, i, to, extra = best
        at[i] = to
        used += extra
    plan = [fronts[i][at[i]][2] for i in range(len(ts))]
    return plan, used, sum(fronts[i][at[i]][1] for i in range(len(ts)))


def exhaustive(ts, tiers, budget, kap):
    """The best assignment of a tier to every tensor within the budget, by trying all len(tiers)^len(ts)."""
    best = None

    def rec(i, used, err, plan):
        nonlocal best
        if used > budget:
            return
        if i == len(ts):
            if best is None or err < best[0]:
                best = (err, used, list(plan))
            return
        for nb in tiers:
            plan.append(nb)
            rec(i + 1, used + _bytes(ts[i][0], nb), err + _error(ts[i], nb, kap), plan)
            plan.pop()

    rec(0, 0.0, 0.0, [])
    return best


def greedy_vs_exhaustive():
    kap = _rels()
    ts = _tensors()[:6]
    tiers = (2, 3, 4, 8)
    total_w = sum(t[0] for t in ts)
    rows = []
    for bpw in (2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.5, 7.5):
        budget = bpw * total_w / 8
        plan, used, err = greedy(ts, tiers, budget, kap)
        err_x, used_x, plan_x = exhaustive(ts, tiers, budget, kap)
        rows.append((f"{bpw:.2f}", f"{budget:,.0f}", "".join(map(str, plan)), f"{used:,.0f}", f"{err:.4f}",
                     "".join(map(str, plan_x)), f"{used_x:,.0f}", f"{err_x:.4f}", f"{err / err_x:.4f}"))
    sizes = ", ".join(f"{t[0]:,}" for t in ts)
    head = (f"The first six tensors of the previous table (sizes {sizes}), tiers n in {{2, 3, 4, 8}} bits ({4 ** 6} assignments). "
            "Budget = bits per weight (mean over the six, scales included) x total weights / 8 bytes. 'plan' lists the bits of tensors 0 .. 5. "
            "Greedy: start every tensor at 2 bits, then repeatedly take the upgrade that removes the most estimated error per extra byte among those that still fit. "
            "Exhaustive: the plan of least total estimated error among all that fit. 'ratio' = greedy error / exhaustive error.\n\n")
    # a finer sweep of the same instance: 0.05 bit steps from 2.50 to 8.50 bits per weight (121 budgets)
    fine = []
    for i in range(50, 171):
        bpw = i / 20
        budget = bpw * total_w / 8
        err = greedy(ts, tiers, budget, kap)[2]
        err_x = exhaustive(ts, tiers, budget, kap)[0]
        fine.append((bpw, err / err_x))
    worse = [(b, r) for b, r in fine if r > 1 + 1e-9]
    worst_b, worst_r = max(worse, key=lambda t: (t[1], -t[0]))
    top = [b for b, r in worse if abs(r - worst_r) < 1e-9]
    tail = (f"\n\nThe nine budgets are a sample. The same instance on a finer sweep, every budget from 2.50 to 8.50 bits per weight in steps of 0.05 ({len(fine)} budgets): "
            f"greedy equals the exhaustive optimum at {len(fine) - len(worse)} of them and is worse at {len(worse)}, by at most a factor {worst_r:.4f} "
            f"(budgets {top[0]:.2f} to {top[-1]:.2f}); at 6.00 bits per weight the factor is "
            f"{next(r for b, r in fine if abs(b - 6.0) < 1e-9):.4f}.")
    return head + _table(["budget bits/weight", "budget bytes", "greedy plan", "greedy bytes", "greedy error", "exhaustive plan", "exhaustive bytes", "exhaustive error", "ratio"], rows,
                         align="rrlrrlrrr") + tail


def greedy_plan():
    kap = _rels()
    ts = _tensors()
    total_w = sum(t[0] for t in ts)
    bpw = 4.5
    budget = bpw * total_w / 8
    plan, used, err = greedy(ts, TIER_BITS, budget, kap)
    rows = []
    for i, (t, nb) in enumerate(zip(ts, plan)):
        rows.append((i, f"{t[0]:,}", f"{t[2]:.3g}", f"{t[1] ** 2 * t[2]:.3e}", nb, f"{_bytes(t[0], nb):,.0f}", f"{_error(t, nb, kap):.4f}"))
    # baselines
    uni = []
    for nb in TIER_BITS:
        b = sum(_bytes(t[0], nb) for t in ts)
        if b <= budget:
            uni.append((nb, b, sum(_error(t, nb, kap) for t in ts)))
    ub = min(uni, key=lambda u: u[2])
    g = bookrng.generator("quantization.random_plans")
    errs, tries = [], 0
    while len(errs) < 200:
        tries += 1
        p = [TIER_BITS[g.integer(len(TIER_BITS))] for _ in ts]
        if sum(_bytes(t[0], nb) for t, nb in zip(ts, p)) <= budget:
            errs.append(sum(_error(t, nb, kap) for t, nb in zip(ts, p)))
    errs.sort()
    med = 0.5 * (errs[99] + errs[100])
    t = _table(["i", "N_i", "sens_i", "sigma_i^2 sens_i", "chosen bits", "bytes", "estimated error"], rows)
    summary = [("greedy plan", f"{used:,.0f}", f"{8 * used / total_w:.3f}", f"{err:.4f}", "1.00"),
               (f"best uniform tier that fits ({ub[0]} bits)", f"{ub[1]:,.0f}", f"{8 * ub[1] / total_w:.3f}", f"{ub[2]:.4f}", f"{ub[2] / err:.2f}"),
               ("median of 200 random plans that fit", "", "", f"{med:.4f}", f"{med / err:.2f}"),
               ("all tensors at 8 bits", f"{sum(_bytes(t[0], 8) for t in ts):,.0f}", f"{8 * sum(_bytes(t[0], 8) for t in ts) / total_w:.3f}",
                f"{sum(_error(t, 8, kap) for t in ts):.4f}", "")]
    t2 = _table(["plan", "bytes", "bits per weight", "estimated error", "error / greedy"], summary, align="lrrrr")
    head = (f"The twelve tensors of the input table, tiers n in {{2, 3, 4, 6, 8}}, budget {bpw} bits per weight x {total_w:,} weights / 8 = {budget:,.0f} bytes. "
            "A random plan draws n for tensors 0 .. 11 in order as TIERS[integer(5)] with TIERS = (2, 3, 4, 6, 8) from the generator, "
            f"{_seed_line('quantization.random_plans')}, and is kept when it fits (drawing until 200 are kept; {tries} were drawn).\n\n")
    return head + t + "\n\n" + t2


def device_layouts():
    """Bits per weight of the device layout of six formats against the file's own: codes in a slot of `slot` bits, a (sc, m)
    byte pair per group of `group` weights, an f16 pair (d, dmin) per `spb` weights."""
    rows_in = [("Q4_K", 4, 32, 256, 144, 256), ("Q5_K", 8, 32, 256, 176, 256), ("Q6_K", 8, 16, 256, 210, 256),
               ("Q4_0", 4, 32, 32, 18, 32), ("Q5_0", 8, 32, 32, 22, 32), ("Q8_0", 8, 32, 32, 34, 32)]
    rows = []
    for nm, slot, group, spb, fbytes, fw in rows_in:
        dev = 8 * (slot / 8 + 2 / group + 4 / spb)
        file_bits = 8 * fbytes / fw
        rows.append((nm, slot, group, spb, f"{dev:.4f}", f"{file_bits:.4f}", f"{dev / file_bits:.3f}"))
    old = 8 * (4 / 8 + 8 / 32)
    return ("Device bits per weight = 8 (slot / 8 + 2 / group + 4 / spb): the codes in a slot of `slot` bits, two bytes (sc, m) per group, "
            "four bytes (the f16 pair d, dmin) per `spb` weights. The file's bits per weight are 8 x block bytes / block weights.\n\n" +
            _table(["format", "slot bits", "group", "spb", "device bits per weight", "file bits per weight", "device / file"], rows) +
            f"\n\nThe layout this replaced kept one f32 pair (scale, min) per group, 8 bytes per group of 32: for Q4_K that is {old:.1f} bits per weight "
            f"(4 for the codes, 2 for the pair), against {8 * (4 / 8 + 2 / 32 + 4 / 256) :.3f} now.")


# ---------------------------------------------------------------- the repository's own outlier fixture


def repo_outlier_fixture():
    """The fixture that gated the move to group-wise scales: two rows of k = 512 weights, one outlier of 100.0 at column k - 3,
    8-bit codes with scale max|.|/127, once per group of 32 and once per row. Rounding is to nearest, ties away from zero (as that code does)."""
    n, k, grp = 2, 512, 32
    w = [(i % 37) * 0.017 - 0.3 for i in range(n * k)]
    out = k - 3
    for r in range(n):
        w[r * k + out] = 100.0

    def away(x):
        return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)

    def rt(row, g):
        res = []
        for i in range(0, len(row), g):
            blk = row[i:i + g]
            s = max(max(abs(x) for x in blk), 1e-8) / 127
            res.extend(s * _clamp(away(x / s), -127, 127) for x in blk)
        return res

    rows = []
    for label, g in (("one scale per 32", grp), ("one scale per row", k)):
        errs = []
        for r in range(n):
            row = w[r * k:(r + 1) * k]
            dq = rt(row, g)
            errs.append([abs(a - b) for a, b in zip(dq, row)])
        outside = max(e for r in range(n) for c, e in enumerate(errs[r]) if c // grp != out // grp)
        bad = sum(1 for r in range(n) for c, e in enumerate(errs[r]) if c != out and e > 0.1)
        rows.append((label, f"{outside:.6f}", bad, n * (k - 1)))
    ratio = float(rows[1][1]) / float(rows[0][1])
    head = ("Two rows of k = 512 weights, w_i = (i mod 37) 0.017 - 0.3 for i = 0 .. 1023 (row-major), with the weight at column 509 of each row replaced by 100.0. "
            "8-bit symmetric codes, scale = max|group| / 127 (floored at 1e-8), q = clamp(round(w / scale), -127, 127) with ties away from zero, w' = scale q, error |w' - w|. "
            "'largest error outside' is over every weight not in the outlier's own group of 32; 'destroyed' counts weights other than the outlier with error above 0.1.\n\n")
    return head + _table(["scales", "largest error outside the outlier's group", "weights destroyed", "of"], rows) + \
        f"\n\nRatio of the two largest errors: {ratio:.0f}."
