# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Images as Data".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: images.<name> -->` markers; the
`*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* a 4 by 4 RGB image, its two memory layouts and the 8-bit to unit-interval conversion with its two rounding rules;
* the sRGB transfer function, its properties, and the average of two pixels taken in code space and in linear light;
* BT.601 / BT.709 conversion coefficients derived from two constants, a packed 4:2:2 and a planar 4:2:0 buffer, and the effect of
  a wrong matrix;
* the source coordinate of an output pixel under the half-pixel, corner-aligned and asymmetric conventions, bilinear and nearest
  resampling on stated inputs, the letterbox and its half-pixel disagreement;
* aliasing of a stripe pattern under naive and averaging downsamplers, the cubic kernels, and a fixed-point resampler checked
  byte for byte against the published values of the library it reproduces;
* per-channel normalization, the effect of wrong constants on a linear probe, and the tolerance of a parity test.

Arithmetic is the language's 64-bit floating point unless a block says that 32-bit arithmetic is emulated, which is done by
rounding a 64-bit result to the nearest 32-bit number after every operation (`struct`); that is exactly the correctly rounded
32-bit result for one addition, subtraction, multiplication or division of 32-bit operands. No example uses random numbers.

Swedish Embedded AB implements solutions for getting camera and file pixels into models exactly as the models were trained to see
them. If your team needs expertise in image preprocessing and its parity with reference pipelines, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import struct

# ------------------------------------------------------------------------------------------------ helpers


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def r32(x):
    """The nearest 32-bit floating-point number to x."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def half_up(x):
    """Round to the nearest integer, ties upward (x >= 0 in every use)."""
    return math.floor(x + 0.5)


def _num(v, d=4):
    s = f"{v:.{d}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


# ------------------------------------------------------------------------------------------------ the 4 by 4 image

W = H = 4
C = 3


def image_codes():
    """The test image: planes[c][y][x] as 8-bit codes. Red is 20 + 13 k, green 10 + 9 k and blue 5 + 15 k, where k = 4 y + x is
    the row-major index of the pixel."""
    base = (20, 10, 5)
    step = (13, 9, 15)
    return [[[base[c] + step[c] * (4 * y + x) for x in range(W)] for y in range(H)] for c in range(C)]


def to_hwc(planes):
    return [planes[c][y][x] for y in range(H) for x in range(W) for c in range(C)]


def to_chw(planes):
    return [planes[c][y][x] for c in range(C) for y in range(H) for x in range(W)]


def image_4x4():
    p = image_codes()
    rows = []
    for y in range(H):
        rows.append([f"({p[0][y][x]}, {p[1][y][x]}, {p[2][y][x]})" for x in range(W)])
    out = ["The test image has 4 rows and 4 columns; each cell is the pixel (red, green, blue) as three 8-bit codes. "
           "Red of the pixel with row-major index k = 4y + x is 20 + 13k, green 10 + 9k and blue 5 + 15k.", ""]
    out.append(_table(["row \\ column", "x = 0", "x = 1", "x = 2", "x = 3"],
                      [[f"y = {y}"] + rows[y] for y in range(H)], "lllll"))
    return "\n".join(out)


def layouts():
    p = image_codes()
    hwc, chw = to_hwc(p), to_chw(p)
    x, y, c = 2, 1, 1
    off_hwc = (y * W + x) * C + c
    off_chw = (c * H + y) * W + x
    assert hwc[off_hwc] == chw[off_chw] == p[c][y][x]
    out = [f"The same image stored two ways. Interleaved HWC, shape [4, 4, 3], strides (12, 3, 1) in elements; planar CHW, shape "
           f"[3, 4, 4], strides (16, 4, 1). The sample at x = {x}, y = {y}, channel {c} (green) has offset (({y} * 4 + {x}) * 3 + {c}) = "
           f"{off_hwc} in HWC and ({c} * 4 + {y}) * 4 + {x} = {off_chw} in CHW, and both hold {p[c][y][x]}.", ""]
    out.append("HWC, 48 values, one pixel (R, G, B) per group of three:")
    out.append("")
    out.append("```text")
    for y0 in range(H):
        out.append("  ".join(f"{hwc[(y0 * W + x0) * 3]:3d} {hwc[(y0 * W + x0) * 3 + 1]:3d} {hwc[(y0 * W + x0) * 3 + 2]:3d}" for x0 in range(W)))
    out.append("```")
    out.append("")
    out.append("CHW, 48 values, the red plane then the green plane then the blue plane (each 16 values, four rows of four):")
    out.append("")
    out.append("```text")
    for c0 in range(C):
        for y0 in range(H):
            out.append(" ".join(f"{chw[(c0 * H + y0) * W + x0]:3d}" for x0 in range(W)) + ("   plane " + "RGB"[c0] if y0 == 0 else ""))
    out.append("```")
    # the test of the crate: R = [1, 2], G = [3, 4], B = [5, 6] for a 1 x 2 image
    tiny_chw = [1, 2, 3, 4, 5, 6]
    tiny_hwc = [tiny_chw[(c * 1 + 0) * 2 + x] for x in range(2) for c in range(3)]
    out.append("")
    out.append(f"A 1 by 2 image with R = [1, 2], G = [3, 4], B = [5, 6] is the CHW buffer {tiny_chw}; interleaved it is {tiny_hwc}.")
    return "\n".join(out)


def byte_to_float():
    def back_half(v):
        return int(math.floor(r32(r32(v * 255.0) + 0.5)))

    def back_trunc(v):
        return int(r32(v * 255.0))

    plain_t = plain_h = 0
    after_t, after_h, ex = [], [], None
    for b in range(256):
        u = r32(b / 255.0)
        plain_h += back_half(u) != b
        plain_t += back_trunc(u) != b
        out = resize_aa_float([[u] * 7 for _ in range(5)], 4, 3, "f32")
        vals = [v for row in out for v in row]
        if any(back_trunc(v) != b for v in vals):
            after_t.append(b)
            if ex is None:
                ex = (b, min(vals, key=lambda v: v))
        if any(back_half(v) != b for v in vals):
            after_h.append(b)
    out = ["Every 8-bit code b becomes the 32-bit float u = b / 255 (one division) and returns by floor(u * 255 + 0.5), clamped to "
           "[0, 255], or by floor(u * 255); each operation is rounded to 32 bits. Second test: the constant 7 by 5 image of u is "
           "resized to 3 by 4 with the antialiased bilinear filter of the stripes example (32-bit arithmetic, window order, horizontal "
           "pass first, defined in the text; the weights are computed in 64-bit and rounded to 32 bits, each pass starts its accumulator at 0 "
           "and rounds every product and then every sum) and every one of the 12 outputs converted back.", ""]
    out.append(_table(["back-conversion", "codes (of 256) that fail to return directly", "codes that fail after the resize"],
                      [["floor(u * 255 + 0.5)", plain_h, len(after_h)], ["floor(u * 255)", plain_t, len(after_t)]], "lrr"))
    out.append("")
    out.append("The codes that truncation loses after the resize begin " + ", ".join(map(str, after_t[:8])) + ". For example the constant "
               f"{ex[0]}/255 comes out of the resize as {ex[1]!r} in at least one pixel, which truncation turns into "
               f"{back_trunc(ex[1])}.")
    out.append("")
    out.append(f"A value of exactly 0.5 gives 0.5 * 255 = 127.5, which floor(x + 0.5) rounds up to {half_up(127.5)} and truncation to 127.")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ sRGB


def dec(e):
    """sRGB encoded value in [0, 1] -> linear light in [0, 1] (IEC 61966-2-1)."""
    return e / 12.92 if e <= 0.04045 else ((e + 0.055) / 1.055) ** 2.4


def enc(l):
    """Linear light -> sRGB encoded value; the inverse of dec."""
    return l * 12.92 if l <= 0.0031308 else 1.055 * l ** (1 / 2.4) - 0.055


def srgb_steps():
    codes = [1, 2, 3, 10, 11, 50, 51, 128, 129, 200, 201, 254, 255]
    rows = []
    for c in codes:
        l = dec(c / 255)
        prev = dec((c - 1) / 255)
        step = l - prev
        rows.append([c, f"{l:.6f}", f"{step:.6f}", f"{(step / prev * 100) if prev > 0 else float('inf'):.2f}" if prev > 0 else "from black"])
    out = ["For each code c: the linear light dec(c / 255), the increase over the code c - 1 below it, and that increase as a "
           "percentage of the light at c - 1.", ""]
    out.append(_table(["code c", "linear light", "step from c - 1", "step relative to c - 1 (%)"], rows))
    lin_rel = [100.0 / (c - 1) for c in (2, 3, 11, 51, 129, 201, 255)]
    out.append("")
    out.append("For comparison, 255 codes spaced evenly in linear light would make the same relative step from c - 1 to c equal to "
               "100 / (c - 1) percent: " + ", ".join(f"{v:.2f}" for v in lin_rel) + " at c = 2, 3, 11, 51, 129, 201, 255.")
    return "\n".join(out)


def srgb_properties():
    e0 = 0.04045
    lin_val = e0 / 12.92
    pow_val = ((e0 + 0.055) / 1.055) ** 2.4
    lin_slope = 1 / 12.92
    pow_slope = 2.4 / 1.055 * ((e0 + 0.055) / 1.055) ** 1.4
    worst = 0.0
    ok = True
    for c in range(256):
        back = half_up(enc(dec(c / 255)) * 255)
        ok = ok and back == c
        worst = max(worst, abs(enc(dec(c / 255)) - c / 255))
    eff = math.log(dec(0.5)) / math.log(0.5)
    eff2 = math.log(dec(0.8)) / math.log(0.8)
    rows = [
        ["join at E = 0.04045: linear piece value E / 12.92", f"{lin_val:.7f}"],
        ["join: power piece value ((E + 0.055) / 1.055)^2.4", f"{pow_val:.7f}"],
        ["join: linear piece slope 1 / 12.92", f"{lin_slope:.5f}"],
        ["join: power piece slope (2.4 / 1.055) ((E + 0.055) / 1.055)^1.4", f"{pow_slope:.5f}"],
        ["slope of enc at L = 0 (the linear toe, 12.92)", "12.92"],
        ["dec(0.5)", f"{dec(0.5):.6f}"],
        ["effective exponent log dec(0.5) / log 0.5", f"{eff:.4f}"],
        ["effective exponent log dec(0.8) / log 0.8", f"{eff2:.4f}"],
        ["all 256 codes: half_up(255 enc(dec(c / 255))) == c", "yes" if ok else "NO"],
        ["largest |enc(dec(c / 255)) - c / 255| over the codes", f"{worst:.2e}"],
    ]
    return ("Properties of the two sRGB functions, computed in 64-bit arithmetic with the library power function.\n\n"
            + _table(["property", "value"], rows, "lr"))


def curve_data(n=101):
    xs = [i / (n - 1) for i in range(n)]
    return xs, [dec(x) for x in xs]


def averaging_pairs():
    pairs = [(0, 255), (64, 192), (0, 64), (100, 200), (30, 40), (200, 255)]
    rows = []
    for a, b in pairs:
        code_avg = (a + b) / 2
        lin = (dec(a / 255) + dec(b / 255)) / 2
        e = enc(lin) * 255
        # exclude ties: the printed roundings must not sit on a .5 boundary
        assert abs((e % 1) - 0.5) > 1e-3 and abs((code_avg % 1) - 0.5) >= 0
        rows.append([f"{a} and {b}", _num(code_avg, 1), f"{dec(code_avg / 255):.4f}", f"{lin:.4f}", f"{e:.2f}", half_up(e)])
    out = ["Two pixels of one channel with the given codes are averaged in two ways. In code space the average is (a + b) / 2. "
           "In linear light each code is decoded with dec, the two lights are averaged, and the mean is encoded again with enc. "
           "'light of the code average' is dec of the code-space result, to compare with the true mean light.", ""]
    out.append(_table(["codes", "code-space average", "light of that average", "mean light", "its code (exact)", "its code (rounded)"], rows))
    return "\n".join(out)


def edge_downsample():
    row = [0, 0, 0, 255, 255, 255, 255, 255]
    code_pairs = [(row[2 * i] + row[2 * i + 1]) / 2 for i in range(4)]
    lin_pairs = [enc((dec(row[2 * i] / 255) + dec(row[2 * i + 1] / 255)) / 2) * 255 for i in range(4)]
    src_light = sum(dec(v / 255) for v in row) / 8
    code_light = sum(dec(v / 255) for v in code_pairs) / 4
    lin_light = sum(dec(v / 255) for v in lin_pairs) / 4
    rows = [
        ["source (8 pixels)", " ".join(str(v) for v in row), f"{src_light:.4f}"],
        ["pairs averaged as codes", " ".join(_num(v, 1) for v in code_pairs), f"{code_light:.4f}"],
        ["pairs averaged in linear light", " ".join(_num(v, 1) for v in lin_pairs), f"{lin_light:.4f}"],
    ]
    return ("A row of 8 pixels with one edge, halved by averaging each pair (the exact 2x area average). "
            "'mean light' is the average of dec(code / 255) over the row.\n\n"
            + _table(["row", "codes", "mean light"], rows, "llr"))


# ------------------------------------------------------------------------------------------------ YUV


def yuv_coefficients(kr, kb):
    kg = 1 - kr - kb
    return {
        "kg": kg,
        "r_cr": 2 * (1 - kr),
        "g_cb": 2 * (1 - kb) * kb / kg,
        "g_cr": 2 * (1 - kr) * kr / kg,
        "b_cb": 2 * (1 - kb),
    }


def yuv_derivation():
    rows = []
    for name, kr, kb in (("BT.601", 0.299, 0.114), ("BT.709", 0.2126, 0.0722)):
        c = yuv_coefficients(kr, kb)
        rows.append([name, f"{kr}", f"{kb}", f"{c['kg']:.4f}", f"{c['r_cr']:.4f}", f"{c['g_cb']:.6f}", f"{c['g_cr']:.6f}", f"{c['b_cb']:.4f}"])
    return ("Two constants, K_r and K_b, fix a standard: K_g = 1 - K_r - K_b, then R = Y + 2 (1 - K_r) Cr, B = Y + 2 (1 - K_b) Cb and "
            "G = Y - [2 (1 - K_b) K_b / K_g] Cb - [2 (1 - K_r) K_r / K_g] Cr, with Cb and Cr already centred on zero.\n\n"
            + _table(["standard", "K_r", "K_b", "K_g", "R: Cr", "G: Cb (minus)", "G: Cr (minus)", "B: Cb"], rows))


def rgb_to_ycc(rgb, kr, kb):
    r, g, b = rgb
    y = kr * r + (1 - kr - kb) * g + kb * b
    return y, (b - y) / (2 * (1 - kb)), (r - y) / (2 * (1 - kr))


def ycc_to_rgb(y, cb, cr, kr, kb):
    c = yuv_coefficients(kr, kb)
    return y + c["r_cr"] * cr, y - c["g_cb"] * cb - c["g_cr"] * cr, y + c["b_cb"] * cb


def clamp8(v):
    return int(min(max(v, 0.0), 255.0) + 0.5)


def wrong_matrix():
    px = [(200, 80, 40), (40, 160, 60), (30, 60, 200), (128, 128, 128)]
    rows = []
    for rgb in px:
        y, cb, cr = rgb_to_ycc(rgb, 0.2126, 0.0722)
        right = tuple(clamp8(v) for v in ycc_to_rgb(y, cb, cr, 0.2126, 0.0722))
        wrong = tuple(clamp8(v) for v in ycc_to_rgb(y, cb, cr, 0.299, 0.114))
        rows.append([rgb, f"{y:.2f}", f"{cb:+.2f}", f"{cr:+.2f}", right, wrong])
    out = ["Each RGB pixel is encoded with the BT.709 constants (K_r = 0.2126, K_b = 0.0722) into (Y, Cb, Cr), "
           "then decoded with the BT.709 coefficients and with the BT.601 ones. Values are not rounded before decoding.", ""]
    out.append(_table(["pixel (R, G, B)", "Y", "Cb", "Cr", "decoded with BT.709", "decoded with BT.601"], rows, "lrrrll"))
    # range
    out.append("")
    out.append("Limited range puts black at Y = 16 and white at Y = 235. A decoder that assumes full range reads Y = 16 as the grey "
               "code 16 instead of 0 and Y = 235 as 235 instead of 255; the correct map is 255 (Y - 16) / 219, which gives "
               f"{_num(255 * (16 - 16) / 219, 1)} and {_num(255 * (235 - 16) / 219, 1)}.")
    return "\n".join(out)


# the 4 x 2 pixel row pair of the subsampling example (R, G, B), row-major
YUV_IMG = [(200, 30, 30), (200, 30, 30), (30, 30, 200), (30, 30, 200),
           (200, 30, 30), (30, 30, 200), (30, 30, 200), (200, 30, 30)]
KR601, KB601 = 0.299, 0.114


def _encode_pixel(rgb):
    return rgb_to_ycc(rgb, KR601, KB601)


def yuyv_encode(img, w=4, h=2):
    out = []
    for y in range(h):
        for x in range(0, w, 2):
            a, b = img[y * w + x], img[y * w + x + 1]
            ya, cba, cra = _encode_pixel(a)
            yb, cbb, crb = _encode_pixel(b)
            cb = 128 + (cba + cbb) / 2
            cr = 128 + (cra + crb) / 2
            out += [half_up(ya), half_up(cb), half_up(yb), half_up(cr)]
    return out


def yuyv_decode(buf, w=4, h=2):
    """The decoder: for each 4 bytes Y0 U Y1 V, R = Y + 1.402 (V - 128), G = Y - 0.344136 (U - 128) - 0.714136 (V - 128),
    B = Y + 1.772 (U - 128), each clamped to [0, 255] and rounded to the nearest integer (ties away from zero)."""
    rgb = []
    for i in range(0, len(buf), 4):
        y0, u, y1, v = buf[i], buf[i + 1] - 128.0, buf[i + 2], buf[i + 3] - 128.0
        for yy in (y0, y1):
            vals = (yy + 1.402 * v, yy - 0.344136 * u - 0.714136 * v, yy + 1.772 * u)
            for t in vals:
                assert abs((min(max(t, 0), 255) % 1) - 0.5) > 1e-3
            rgb.append(tuple(clamp8(t) for t in vals))
    return rgb


def nv12_encode(img, w=4, h=2):
    ys = [half_up(_encode_pixel(p)[0]) for p in img]
    uv = []
    for by in range(0, h, 2):
        for bx in range(0, w, 2):
            blk = [img[(by + dy) * w + bx + dx] for dy in (0, 1) for dx in (0, 1)]
            cb = 128 + sum(_encode_pixel(p)[1] for p in blk) / 4
            cr = 128 + sum(_encode_pixel(p)[2] for p in blk) / 4
            uv += [half_up(cb), half_up(cr)]
    return ys, uv


def nv12_decode(ys, uv, w=4, h=2):
    rgb = []
    for y in range(h):
        for x in range(w):
            u = uv[2 * ((y // 2) * (w // 2) + x // 2)] - 128.0
            v = uv[2 * ((y // 2) * (w // 2) + x // 2) + 1] - 128.0
            yy = ys[y * w + x]
            vals = (yy + 1.402 * v, yy - 0.344136 * u - 0.714136 * v, yy + 1.772 * u)
            rgb.append(tuple(clamp8(t) for t in vals))
    return rgb


def subsampling():
    yuyv = yuyv_encode(YUV_IMG)
    dec422 = yuyv_decode(yuyv)
    ys, uv = nv12_encode(YUV_IMG)
    dec420 = nv12_decode(ys, uv)
    out = ["The 4 by 2 image (pixels in row-major order) is "
           + ", ".join(str(p) for p in YUV_IMG) + ". It is encoded with the BT.601 constants (K_r = 0.299, K_b = 0.114), full range, "
           "Y rounded half up, and the chroma code 128 + Cb (128 + Cr) of a shared pair or block taken as the mean of the "
           "unrounded chroma of its pixels, then rounded half up.", ""]
    out.append(f"Packed 4:2:2 (YUYV), {len(yuyv)} bytes, groups Y0 U Y1 V: {yuyv}")
    out.append("")
    out.append(f"Planar 4:2:0 (NV12), {len(ys) + len(uv)} bytes: Y plane (8 bytes) {ys}, then the U V plane (one interleaved pair per 2 by 2 "
               f"block, 4 bytes) {uv}.")
    out.append("")
    rows = []
    for i, p in enumerate(YUV_IMG):
        e1 = max(abs(a - b) for a, b in zip(p, dec422[i]))
        e2 = max(abs(a - b) for a, b in zip(p, dec420[i]))
        rows.append([i, p, dec422[i], e1, dec420[i], e2])
    out.append(_table(["pixel", "original", "from 4:2:2", "largest error", "from 4:2:0", "largest error"], rows, "rlrrrr"))
    out.append("")
    out.append("Bytes per pixel: RGB 3, packed 4:2:2 2, planar 4:2:0 1.5; a 640 by 480 frame is "
               f"{640 * 480 * 3:,}, {640 * 480 * 2:,} and {640 * 480 * 3 // 2:,} bytes.")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ source coordinates


def src_half(o, n_in, n_out):
    return (o + 0.5) * n_in / n_out - 0.5


def src_corners(o, n_in, n_out):
    return 0.0 if n_out == 1 else o * (n_in - 1) / (n_out - 1)


def src_asym(o, n_in, n_out):
    return o * n_in / n_out


def source_positions():
    out = []
    for n_in, n_out in ((4, 8), (4, 3), (4, 2)):
        rows = []
        for o in range(n_out):
            rows.append([o, _num(src_half(o, n_in, n_out), 4), _num(src_corners(o, n_in, n_out), 4), _num(src_asym(o, n_in, n_out), 4)])
        out.append(f"Resizing {n_in} pixels to {n_out}. For each output index o, the source coordinate (in units of source pixel indices) "
                   f"under the three conventions.")
        out.append("")
        out.append(_table(["o", "half-pixel (o + 0.5) n_in / n_out - 0.5", "corners o (n_in - 1) / (n_out - 1)", "asymmetric o n_in / n_out"], rows))
        out.append("")
    return "\n".join(out).rstrip()


def positions_data():
    return {(a, b): ([src_half(o, a, b) for o in range(b)], [src_corners(o, a, b) for o in range(b)], [src_asym(o, a, b) for o in range(b)])
            for a, b in ((4, 8), (4, 2))}


# ------------------------------------------------------------------------------------------------ bilinear and nearest


def lerp_axis(values, o, n_out, mode):
    n_in = len(values)
    s = src_half(o, n_in, n_out) if mode == "half" else src_corners(o, n_in, n_out)
    s = min(max(s, 0.0), n_in - 1.0)
    i0 = int(math.floor(s))
    t = s - i0
    i1 = min(i0 + 1, n_in - 1)
    return values[i0] * (1 - t) + values[i1] * t


def resize2d(plane, th, tw, mode):
    """Bilinear resize of plane[y][x]: for each output pixel, interpolate the two horizontal neighbours on the two rows the
    vertical coordinate falls between, then interpolate those two results."""
    h, w = len(plane), len(plane[0])
    out = []
    for oy in range(th):
        sy = src_half(oy, h, th) if mode == "half" else src_corners(oy, h, th)
        sy = min(max(sy, 0.0), h - 1.0)
        y0 = int(math.floor(sy))
        ty = sy - y0
        y1 = min(y0 + 1, h - 1)
        row = []
        for ox in range(tw):
            sx = src_half(ox, w, tw) if mode == "half" else src_corners(ox, w, tw)
            sx = min(max(sx, 0.0), w - 1.0)
            x0 = int(math.floor(sx))
            tx = sx - x0
            x1 = min(x0 + 1, w - 1)
            top = plane[y0][x0] * (1 - tx) + plane[y0][x1] * tx
            bot = plane[y1][x0] * (1 - tx) + plane[y1][x1] * tx
            row.append(top * (1 - ty) + bot * ty)
        out.append(row)
    return out


def resize_separable(plane, th, tw, mode):
    h, w = len(plane), len(plane[0])
    mid = [[lerp_axis(plane[y], ox, tw, mode) for ox in range(tw)] for y in range(h)]
    return [[lerp_axis([mid[y][ox] for y in range(h)], oy, th, mode) for ox in range(tw)] for oy in range(th)]


def bilinear_examples():
    red = image_codes()[0]
    out = ["The red plane of the test image (20 + 13 k) is resized from 4 by 4 to 3 by 3 with bilinear interpolation under the two "
           "conventions. Because the plane is a linear ramp in x and y, the value at source coordinate (sx, sy) is exactly "
           "20 + 13 (4 sy + sx), which the entries can be checked against by hand.", ""]
    for mode, name in (("half", "half-pixel"), ("corners", "corners")):
        r = resize2d(red, 3, 3, mode)
        s = resize_separable(red, 3, 3, mode)
        diff = max(abs(a - b) for ra, rb in zip(r, s) for a, b in zip(ra, rb))
        out.append(f"{name}: source coordinates {[_num(src_half(o, 4, 3) if mode == 'half' else src_corners(o, 4, 3), 4) for o in range(3)]} on each axis.")
        out.append("")
        out.append("```text")
        for row in r:
            out.append("  ".join(f"{v:8.4f}" for v in row))
        out.append("```")
        out.append("")
        out.append(f"Computed instead as two passes (every row along x, then every column along y), the 9 entries differ from these by "
                   f"{'less than 1e-12' if diff < 1e-12 else f'{diff:.1e}'} at most.")
        out.append("")
    ident = resize2d(red, 4, 4, "half")
    out.append(f"Resizing to the same size is the identity: the largest difference from the input is {max(abs(a - b) for ra, rb in zip(ident, red) for a, b in zip(ra, rb)):.1f}.")
    two = [lerp_axis([0.0, 1.0], o, 4, "half") for o in range(4)]
    out.append("")
    out.append(f"Doubling [0, 1] to four samples with the half-pixel rule gives {[_num(v, 4) for v in two]}; the corner rule gives "
               f"{[_num(lerp_axis([0.0, 1.0], o, 4, 'corners'), 4) for o in range(4)]}.")
    return "\n".join(out).rstrip()


def nearest_floor(o, n_in, n_out):
    return o * n_in // n_out


def nearest_rules():
    out = ["Nearest-neighbour source index of output o for two rules: half-pixel, round_half_up((o + 0.5) n_in / n_out - 0.5) "
           "= floor((2o + 1) n_in / (2 n_out)), and asymmetric, floor(o n_in / n_out). The table counts the outputs for which they "
           "pick different source pixels. The counts use exact integer arithmetic, the rule of real numbers.", ""]
    rows = []
    for n_in, n_out in ((3, 2), (4, 2), (5, 3), (7, 4), (479, 640), (640, 224), (1080, 640), (2048, 576)):
        diff = sum(1 for o in range(n_out) if nearest_tap(o, n_in, n_out) != nearest_floor(o, n_in, n_out))
        rows.append([n_in, n_out, diff, n_out, f"{100 * diff / n_out:.1f}"])
    out.append(_table(["n_in", "n_out", "outputs that differ", "of", "%"], rows))
    out.append("")
    out.append(f"For 3 to 2: half-pixel picks source indices {[nearest_tap(o, 3, 2) for o in range(2)]}, asymmetric picks "
               f"{[nearest_floor(o, 3, 2) for o in range(2)]}.")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ letterbox


def nearest_tap(o, n_in, n_out):
    """The source index half-pixel nearest neighbour takes for output o: the integer form of round_half_up((o + 0.5) n_in / n_out
    - 0.5), clamped to the last source index. n_in is the source extent and n_out the extent of the resized image."""
    return min((2 * o + 1) * n_in // (2 * n_out), n_in - 1)


def letterbox_plan(w0, h0, size):
    """Scale, content size, and the whole-pixel pad above/left (floor of half the slack). The extra pixel of an odd slack goes below
    and right."""
    s = min(size / w0, size / h0)
    new_w, new_h = half_up(w0 * s), half_up(h0 * s)
    return s, new_w, new_h, (size - new_w) // 2, (size - new_h) // 2


def letterbox_grid():
    w0, h0, size = 7, 4, 8
    s, nw, nh, px, py = letterbox_plan(w0, h0, size)
    sy = [nearest_tap(yi, h0, nh) for yi in range(nh)]
    sx = [nearest_tap(xi, w0, nw) for xi in range(nw)]
    grid = [["  ." for _ in range(size)] for _ in range(size)]
    for yi in range(nh):
        for xi in range(nw):
            grid[yi + py][xi + px] = f"{sy[yi] * w0 + sx[xi]:3d}"
    out = [f"A {w0} by {h0} image (pixels numbered 0 to {w0 * h0 - 1} in row-major order) is letterboxed onto an {size} by {size} "
           f"square. Scale s = min({size}/{w0}, {size}/{h0}) = {s:.6f}; content {nw} by {nh}; slack ({size - nw}, {size - nh}); "
           f"padding above and left (pad_x, pad_y) = ({px}, {py}) = floor(slack / 2), below and right ({size - nw - px}, {size - nh - py}). "
           f"Source column of each content column: {sx}; source row of each content row: {sy}. "
           "Each axis takes content cell o from source index min(floor((2 o + 1) n_in / (2 n_out)), n_in - 1), "
           "where n_in is the source extent of the axis (7 and 4) and n_out the content extent (8 and 5): "
           "the half-pixel rule above written in integers, with the sizes of the two images and not the scale. "
           "Each cell shows the number of the source pixel that fills it, and . is padding.", ""]
    out.append("```text")
    for row in grid:
        out.append(" ".join(row))
    out.append("```")
    return "\n".join(out)


def letterbox_data():
    w0, h0, size = 7, 4, 8
    s, nw, nh, px, py = letterbox_plan(w0, h0, size)
    sy = [nearest_tap(yi, h0, nh) for yi in range(nh)]
    sx = [nearest_tap(xi, w0, nw) for xi in range(nw)]
    return w0, h0, size, nw, nh, px, py, sx, sy


def box_iou_shifted(k, shift):
    """IoU of the box (0, 0, k, k) with the same box moved down by `shift` rows (areas of rectangles, x extents equal)."""
    inter = k * max(k - shift, 0.0)
    return inter / (2 * k * k - inter)


def letterbox_pad():
    out = ["The pad of a letterbox is whole pixels: floor(slack / 2) above and left, the rest below and right. Pixels and boxes "
           "share it, so the box mapping sends the original's row 0 to the row where the content starts (pad above). "
           "slack = size - content.", ""]
    rows = []
    for w0, h0, size in ((4, 3, 4), (640, 479, 640), (640, 480, 640)):
        s, nw, nh, px, py = letterbox_plan(w0, h0, size)
        slack = size - nh
        rows.append([f"{w0} x {h0} to {size}", _num(s, 4), f"{nw} x {nh}", slack, py, slack - py, py])
    out.append(_table(["image to square", "scale", "content", "slack in y", "pad above", "pad below", "row 0 maps to y"], rows))
    out.append("")
    out.append("What a frame that disagrees with its pixels by half a row would cost. A box of k by k pixels, written (x1, y1, x2, y2) = "
               "(0, 0, k, k), is compared with the same box moved down by 0.5; the IoU is the overlap area divided by the union "
               "area, k (k - 0.5) / (2 k^2 - k (k - 0.5)) = (k - 0.5) / (k + 0.5).")
    out.append("")
    out.append(_table(["k", "IoU after a shift of 0.5"], [[k, f"{box_iou_shifted(k, 0.5):.4f}"] for k in (1, 2, 4, 16, 64)]))
    return "\n".join(out)


def depth_sizes():
    w0, h0, inp = 2048, 1375, 384

    def make_div(v, m):
        return max(half_up(v / m) * m, m)

    scale = inp / min(w0, h0)
    th, tw = make_div(h0 * scale, 32), make_div(w0 * scale, 32)
    s, nw, nh, px, py = letterbox_plan(w0, h0, inp)
    rows = [
        ["aspect-preserving, shorter side 384, sides rounded to a multiple of 32", f"{tw} x {th}", f"{tw * th:,}", f"{tw * th:,}", "0"],
        ["letterbox onto a 384 x 384 square", f"{nw} x {nh} inside 384 x 384", f"{inp * inp:,}", f"{nw * nh:,}", f"{inp * inp - nw * nh:,}"],
    ]
    out = [f"A {w0} by {h0} photograph (width by height) prepared for a model whose input side is {inp}. The first row is the rule "
           "scale = 384 / min(w0, h0), each side times the scale divided by 32, rounded half up, times 32 (at least 32); the second "
           "is the letterbox with scale min(384 / w0, 384 / h0).", ""]
    out.append(_table(["preparation", "input size (w x h)", "pixels the model reads", "pixels that come from the photograph", "padding pixels"], rows, "lrrrr"))
    out.append("")
    out.append(f"The letterbox scale is {s:.4f} against {scale:.4f}, so the photograph is shrunk by a factor {scale / s:.2f} more and "
               f"{nw * nh / (tw * th) * 100:.1f}% as many of its pixels reach the model as in the first row.")
    return "\n".join(out)


def pad_after_norm():
    clip_mean = (0.481, 0.458, 0.408)
    clip_std = (0.269, 0.261, 0.276)
    pad = tuple(int(m * 255) for m in clip_mean)
    normed = tuple((p / 255 - m) / s for p, m, s in zip(pad, clip_mean, clip_std))
    grey = 114 / 255
    out = ["Two padding colours. A detector with no normalization pads with 114 / 255 = %.4f in each channel. A tower that normalizes "
           "with mean (0.481, 0.458, 0.408) and standard deviation (0.269, 0.261, 0.276) can pad with int(255 mean) = %s, whose "
           "normalized value (c / 255 - mean) / std is %s." % (grey, pad, tuple(round(v, 4) for v in normed)),
           "", "Padding with the mean makes the border close to the average pixel: after normalization the network sees values near 0 there."]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ stripes and area filters


def stripes(period_pattern, n):
    return [period_pattern[i % len(period_pattern)] for i in range(n)]


def naive_down(row, factor):
    n_out = len(row) // factor
    return [lerp_axis([float(v) for v in row], o, n_out, "half") for o in range(n_out)]


def area_down(row, factor):
    n_out = len(row) // factor
    return [sum(row[o * factor:(o + 1) * factor]) / factor for o in range(n_out)]


def tri_weights(n_in, n_out):
    """Antialiased bilinear (triangle) weights: for output i, the window of source pixels and their weights, from the centre
    (i + 0.5) scale, the filter stretched by the scale, normalized to sum 1."""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = 1.0 * fs
    res = []
    for i in range(n_out):
        center = (i + 0.5) * scale
        xmin = max(int(math.floor(center - support + 0.5)), 0)
        xmax = min(int(math.floor(center + support + 0.5)), n_in)
        ws = [max(1 - abs((j - center + 0.5) / fs), 0.0) for j in range(xmin, xmax)]
        t = sum(ws)
        res.append((xmin, [w / t for w in ws]))
    return res


def aa_down(row, factor):
    n_out = len(row) // factor
    return [sum(w * row[x0 + k] for k, w in enumerate(ws)) for x0, ws in tri_weights(len(row), n_out)]


def stripe_cases():
    return [
        ("period 4, phase 0 (0 255 255 0)", stripes([0, 255, 255, 0], 24), 4),
        ("period 4, phase 2 (255 0 0 255)", stripes([255, 0, 0, 255], 24), 4),
        ("period 6 (0 0 0 255 255 255)", stripes([0, 0, 0, 255, 255, 255], 24), 4),
    ]


def stripe_data():
    return [(name, row, naive_down(row, f), area_down(row, f), aa_down(row, f)) for name, row, f in stripe_cases()]


def stripes_table():
    out = ["Rows of 24 pixels with stripes, shrunk by a factor 4 to 6 pixels three ways: naive (bilinear with the half-pixel rule, "
           "which reads two neighbouring pixels), area (the mean of the 4 pixels that the output pixel covers) and the "
           "antialiased triangle filter (window and weights as in the text, normalized). Values are shown to one decimal. The last column is the largest minus the smallest of the four middle outputs, because the two outer outputs of the triangle filter have a window clipped by the image border.", ""]
    rows = []
    for name, row, nv, ar, aa in stripe_data():
        rows.append([name, "naive", " ".join(f"{v:.1f}" for v in nv), f"{max(nv[1:5]) - min(nv[1:5]):.1f}"])
        rows.append(["", "area", " ".join(f"{v:.1f}" for v in ar), f"{max(ar[1:5]) - min(ar[1:5]):.1f}"])
        rows.append(["", "antialiased triangle", " ".join(f"{v:.1f}" for v in aa), f"{max(aa[1:5]) - min(aa[1:5]):.1f}"])
    out.append(_table(["stripes", "method", "6 output pixels", "range of outputs 1 to 4"], rows, "llrr"))
    tw = tri_weights(24, 6)
    out.append("")
    out.append(f"The antialiased weights of output pixel 0 are on source pixels {tw[0][0]} to {tw[0][0] + len(tw[0][1]) - 1}: "
               + ", ".join(f"{w:.4f}" for w in tw[0][1]) + " (they sum to 1; the window is clipped at the image border). "
               f"Those of output pixel 2 are on pixels {tw[2][0]} to {tw[2][0] + len(tw[2][1]) - 1}: " + ", ".join(f"{w:.4f}" for w in tw[2][1]) + ".")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ cubic and fixed-point


def cubic(x, a):
    x = abs(x)
    if x < 1:
        return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
    if x < 2:
        return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
    return 0.0


def cubic_taps():
    rows = []
    for a in (-0.5, -0.75):
        taps = [cubic(1.5, a), cubic(0.5, a), cubic(0.5, a), cubic(1.5, a)]
        edge = [0.0, 255.0, 255.0, 0.0]
        val = sum(w * v for w, v in zip(taps, edge))
        rows.append([a, " ".join(f"{t:.5f}" for t in taps), f"{sum(taps):.1f}", f"{val:.2f}"])
    return ("The four weights of cubic convolution for a sample exactly halfway between two source pixels (offsets -1.5, -0.5, +0.5, +1.5 "
            "from the sample), with k(x) = (a + 2)|x|^3 - (a + 3)|x|^2 + 1 for |x| < 1 and a|x|^3 - 5a|x|^2 + 8a|x| - 4a for 1 <= |x| < 2. "
            "The last column applies them to four pixels [0, 255, 255, 0] (before any clamp to 255).\n\n"
            + _table(["a", "weights at -1.5, -0.5, +0.5, +1.5", "sum", "value on [0, 255, 255, 0]"], rows, "rlrr"))


def _cubic05(x):
    return cubic(x, -0.5)


def fixed_coeffs(n_in, n_out, bits_rule):
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = 2.0 * fs
    ksize = math.ceil(support) * 2 + 1
    inv = 1.0 / fs
    bounds, weights = [], []
    for xx in range(n_out):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), n_in) - xmin
        w = [_cubic05((x + xmin - center + 0.5) * inv) for x in range(xmax)]
        t = sum(w)
        weights.append([v / t if t != 0 else v for v in w] + [0.0] * (ksize - xmax))
        bounds.append((xmin, xmax))
    if bits_rule == "pillow":
        bits = 22
    else:
        mx = max(max(w) for w in weights)
        bits = next((b for b in range(22) if int(0.5 + mx * (1 << (b + 1))) >= 1 << 15), 22)
    one = float(1 << bits)
    kk = [[int(-0.5 + k * one) if k < 0 else int(0.5 + k * one) for k in w] for w in weights]
    return bounds, kk, bits


def _pass_h(img, w, h, n_w, rule):
    bounds, kk, bits = fixed_coeffs(w, n_w, rule)
    out = []
    for y in range(h):
        row = img[y]
        orow = []
        for o, (mn, n) in enumerate(bounds):
            ss = [1 << (bits - 1)] * 3
            for i in range(n):
                px = row[mn + i]
                for ch in range(3):
                    ss[ch] += px[ch] * kk[o][i]
            orow.append(tuple(min(max(v >> bits, 0), 255) for v in ss))
        out.append(orow)
    return out


def _pass_v(img, w, h, n_h, rule):
    bounds, kk, bits = fixed_coeffs(h, n_h, rule)
    out = []
    for o, (mn, n) in enumerate(bounds):
        orow = []
        for x in range(w):
            ss = [1 << (bits - 1)] * 3
            for i in range(n):
                px = img[mn + i][x]
                for ch in range(3):
                    ss[ch] += px[ch] * kk[o][i]
            orow.append(tuple(min(max(v >> bits, 0), 255) for v in ss))
        out.append(orow)
    return out


def fixture(w=7, h=5):
    return [[tuple((37 * x + 11 * y + 53 * c) % 256 for c in range(3)) for x in range(w)] for y in range(h)]


def resize_fixed(img, n_w, n_h, rule):
    h, w = len(img), len(img[0])
    height_cut, width_cut = h - n_h, w - n_w
    vertical_first = rule == "pillow" and height_cut > 0 and height_cut > width_cut * 2
    if vertical_first:
        mid = _pass_v(img, w, h, n_h, rule) if n_h != h else img
        return _pass_h(mid, w, n_h, n_w, rule) if n_w != w else mid
    mid = _pass_h(img, w, h, n_w, rule) if n_w != w else img
    return _pass_v(mid, n_w, h, n_h, rule) if n_h != h else mid


def flat(img):
    return [v for row in img for px in row for v in px]


# the 3 x 4 bytes published for Pillow 11.3 and torch 2.14.1 by the reference script the library's tests are pinned to
# (crates/imaging/tests/it/resize_reference.rs); they are the independent reference of this block, not an input of it.
PIL_BICUBIC_3X4 = [29, 81, 137, 113, 174, 201, 197, 157, 82, 42, 95, 157, 126, 182, 156, 210, 122, 49, 56, 110, 171, 140, 188, 175,
                   227, 61, 65, 69, 123, 184, 160, 202, 98, 158, 80, 78]
TORCH_U8_BICUBIC_3X4 = [29, 81, 137, 112, 174, 201, 196, 157, 82, 42, 95, 157, 126, 182, 156, 210, 122, 49, 56, 110, 171, 140, 188,
                        175, 227, 61, 65, 69, 123, 184, 160, 202, 98, 158, 80, 78]


def fixed_point_resize():
    img = fixture()
    pil = flat(resize_fixed(img, 3, 4, "pillow"))
    tor = flat(resize_fixed(img, 3, 4, "torch"))
    assert pil == PIL_BICUBIC_3X4, "fixed-point resampler disagrees with the published Pillow bytes"
    assert tor == TORCH_U8_BICUBIC_3X4, "int16 variant disagrees with the published torch bytes"
    diff = [i for i, (a, b) in enumerate(zip(pil, tor)) if a != b]
    _, kk22, b22 = fixed_coeffs(7, 3, "pillow")
    _, kkt, bt = fixed_coeffs(7, 3, "torch")
    out = ["The 7 by 5 fixture image has the code (37 x + 11 y + 53 c) mod 256 at column x, row y, channel c. It is resized to 3 "
           "columns by 4 rows by the two fixed-point bicubic resamplers. Both use the cubic with a = -0.5, a window stretched by the "
           "scale when shrinking, normalized weights, and round to integer weights with 2^bits as one: Pillow uses bits = "
           f"{b22}, the uint8 path of torch the smallest bits for which the largest scaled weight reaches 2^14, so that it stays below 2^15 and "
           f"fits an int16 (here bits = {bt}). The horizontal pass runs first (Pillow runs the vertical pass first only when the height loses "
           "more than twice as many pixels as the width, which this shrink does not) and each pass is rounded and clamped to 8 bits.", ""]
    out.append("Pillow-style output, pixels in row-major order as (R, G, B):")
    out.append("")
    out.append("```text")
    r = resize_fixed(img, 3, 4, "pillow")
    for row in r:
        out.append("  ".join(f"({p[0]:3d},{p[1]:3d},{p[2]:3d})" for p in row))
    out.append("```")
    out.append("")
    out.append(f"The int16-weight variant differs from it in {len(diff)} of 36 bytes, at positions {diff}, by at most "
               f"{max(abs(pil[i] - tor[i]) for i in diff)}. Both outputs equal, byte for byte, the values published for Pillow 11.3 and "
               "torch 2.14.1 (the reference of the library's own test).")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ float antialiased vs 32-bit, tolerance


def bilinear_aa_weights(n_in, n_out):
    return tri_weights(n_in, n_out)


def resize_aa_float(plane, th, tw, arith):
    """Separable antialiased bilinear resize of plane[y][x]: horizontal pass then vertical pass, accumulated left to right
    in the order of the window; arith is "f64" or "f32" (every product and sum rounded to 32 bits)."""
    h, w = len(plane), len(plane[0])
    wx, wy = tri_weights(w, tw), tri_weights(h, th)
    rnd = (lambda v: v) if arith == "f64" else r32
    mid = []
    for y in range(h):
        row = []
        for x0, ws in wx:
            acc = 0.0
            for k, wt in enumerate(ws):
                acc = rnd(acc + rnd(rnd(wt) * rnd(plane[y][x0 + k])))
            row.append(acc)
        mid.append(row)
    out = []
    for y0, ws in wy:
        row = []
        for x in range(tw):
            acc = 0.0
            for k, wt in enumerate(ws):
                acc = rnd(acc + rnd(rnd(wt) * mid[y0 + k][x]))
            row.append(acc)
        out.append(row)
    return out


def resize_aa_direct(plane, th, tw):
    """The same resize without separating the axes: each output is the double sum of weight_y * weight_x * pixel."""
    h, w = len(plane), len(plane[0])
    wx, wy = tri_weights(w, tw), tri_weights(h, th)
    return [[sum(wyk * wxk * plane[y0 + j][x0 + i] for j, wyk in enumerate(ys) for i, wxk in enumerate(xs))
             for (x0, xs) in wx] for (y0, ys) in wy]


def parity_tolerance():
    img = fixture()
    plane = [[img[y][x][0] / 255 for x in range(7)] for y in range(5)]
    a64 = resize_aa_float(plane, 4, 3, "f64")
    a32 = resize_aa_float(plane, 4, 3, "f32")
    direct = resize_aa_direct(plane, 4, 3)
    sep_vs_direct = max(abs(a - b) for ra, rb in zip(a64, direct) for a, b in zip(ra, rb))
    f32_vs_f64 = max(abs(a - b) for ra, rb in zip(a32, a64) for a, b in zip(ra, rb))
    u = 2.0 ** -24
    kh = max(len(ws) for _, ws in tri_weights(7, 3))
    kv = max(len(ws) for _, ws in tri_weights(5, 4))
    out = ["The red channel of the 7 by 5 fixture (code / 255) is resized to 3 by 4 with the antialiased bilinear filter. Separable "
           "64-bit result against the double sum without separating the axes, and 32-bit result (every product and sum rounded to 32 bits, "
           "window order, horizontal pass first) against the 64-bit one. The weights are computed and normalized in 64-bit and then "
           "rounded to 32 bits, the input is code / 255 rounded to 32 bits, and each pass starts its accumulator at 0 and rounds every "
           "product and then every sum.", ""]
    rows = [
        ["separable 64-bit vs double sum (64-bit)", f"{sep_vs_direct:.1e}"],
        ["32-bit separable vs 64-bit separable", f"{f32_vs_f64:.1e}"],
        ["unit round-off u of 32 bits = 2^-24", f"{u:.1e}"],
        [f"bound (4 + K_h + K_v) u, with K_h = {kh} and K_v = {kv} taps in the longest windows", f"{(4 + kh + kv) * u:.1e}"],
        ["tolerance used by the parity tests for float resizes", "1e-6"],
    ]
    out.append(_table(["comparison", "largest absolute difference"], rows, "lr"))
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ normalization

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF = ((0.5,) * 3, (0.5,) * 3)
PROBE_W = (1.2, -0.8, 0.5)
PROBE_B = 0.1


def normalize_pixel(code, mean, std, order=(0, 1, 2)):
    x = [code[order[c]] / 255 for c in range(3)]
    return [(x[c] - mean[c]) / std[c] for c in range(3)]


def probe(z):
    return sum(w * v for w, v in zip(PROBE_W, z)) + PROBE_B


def scale_shift_table():
    rows = []
    for name, (m, s) in (("ImageNet", (IMAGENET_MEAN, IMAGENET_STD)), ("(x - 0.5) / 0.5", HALF), ("identity", ((0.0,) * 3, (1.0,) * 3))):
        sc = [1 / s[c] for c in range(3)]
        sh = [-m[c] / s[c] for c in range(3)]
        rows.append([name, " ".join(f"{v:.5f}" for v in sc), " ".join(f"{v:+.5f}" for v in sh)])
    out = ["The per-channel map (x - mean) / std written as y = x * scale + shift with scale = 1 / std and shift = -mean / std "
           "(x in [0, 1]), and its inverse x = y * std + mean.", ""]
    out.append(_table(["constants", "scale (R G B)", "shift (R G B)"], rows, "lll"))
    return "\n".join(out)


def wrong_constants():
    pixels = [(20, 10, 5), (110, 70, 80), (215, 145, 230), (200, 30, 30)]
    variants = [
        ("trained: ImageNet, RGB", IMAGENET_MEAN, IMAGENET_STD, (0, 1, 2)),
        ("(x - 0.5) / 0.5", HALF[0], HALF[1], (0, 1, 2)),
        ("forgotten: x in [0, 1] as is", (0.0,) * 3, (1.0,) * 3, (0, 1, 2)),
        ("ImageNet, channels swapped to BGR", IMAGENET_MEAN, IMAGENET_STD, (2, 1, 0)),
        ("ImageNet scaled by 255, codes as they are", tuple(m * 255 for m in IMAGENET_MEAN), tuple(s * 255 for s in IMAGENET_STD), (0, 1, 2)),
    ]
    out = [f"A first-layer probe z = w . y + b with w = {list(PROBE_W)} and b = {PROBE_B} was fitted to inputs y that were normalized with the "
           "ImageNet constants from RGB pixels in [0, 1]. Four pixels (8-bit RGB codes) are prepared in five ways: as trained; with "
           "(x - 0.5) / 0.5; with no normalization; with the channels in BGR order before the ImageNet constants; and, as a check "
           "that is the same function as the first row, from the codes 0 to 255 with mean and std multiplied by 255. The last column "
           "is the largest change of z from the first row over the four pixels.", ""]
    heads = ["preparation"] + [str(p) for p in pixels] + ["largest |z - z_trained|"]
    base = [probe(normalize_pixel(p, IMAGENET_MEAN, IMAGENET_STD)) for p in pixels]
    rows = []
    for name, m, s, order in variants:
        if name.startswith("ImageNet scaled by 255"):
            zs = [probe([(p[c] - m[c]) / s[c] for c in range(3)]) for p in pixels]
        else:
            zs = [probe(normalize_pixel(p, m, s, order)) for p in pixels]
        rows.append([name] + [f"{z:+.3f}" for z in zs] + [f"{max(abs(a - b) for a, b in zip(zs, base)):.3f}"])
    out.append(_table(heads, rows, "l" + "r" * (len(pixels) + 1)))
    return "\n".join(out)


def input_statistics():
    p = image_codes()
    px = [[p[c][y][x] for c in range(3)] for y in range(H) for x in range(W)]
    rows = []
    for name, m, s in (("ImageNet", IMAGENET_MEAN, IMAGENET_STD), ("(x - 0.5) / 0.5", HALF[0], HALF[1]), ("identity", (0.0,) * 3, (1.0,) * 3)):
        stats = []
        for c in range(3):
            v = [(q[c] / 255 - m[c]) / s[c] for q in px]
            mu = sum(v) / len(v)
            sd = math.sqrt(sum((a - mu) ** 2 for a in v) / len(v))
            stats.append(f"{mu:+.3f} / {sd:.3f}")
        rows.append([name] + stats)
    return ("Mean / standard deviation (divisor 16) of each channel of the 16 pixels of the test image after normalization.\n\n"
            + _table(["constants", "R", "G", "B"], rows, "lrrr"))


def fold_into_first_layer():
    w = [[0.7, -0.2, 0.4], [-0.5, 0.3, 0.9]]
    b = [0.05, -0.1]
    m, s = IMAGENET_MEAN, IMAGENET_STD
    wf = [[w[o][c] / s[c] for c in range(3)] for o in range(2)]
    bf = [b[o] - sum(w[o][c] * m[c] / s[c] for c in range(3)) for o in range(2)]
    rows = []
    worst = 0.0
    for code in [(20, 10, 5), (110, 70, 80), (215, 145, 230), (255, 255, 255)]:
        x = [v / 255 for v in code]
        y = [(x[c] - m[c]) / s[c] for c in range(3)]
        a = [sum(w[o][c] * y[c] for c in range(3)) + b[o] for o in range(2)]
        f = [sum(wf[o][c] * x[c] for c in range(3)) + bf[o] for o in range(2)]
        # 32-bit versions
        a32 = [r32(sum_f32([r32(r32(w[o][c]) * r32(r32(r32(x[c]) - r32(m[c])) / r32(s[c]))) for c in range(3)]) + r32(b[o])) for o in range(2)]
        f32 = [r32(sum_f32([r32(r32(wf[o][c]) * r32(x[c])) for c in range(3)]) + r32(bf[o])) for o in range(2)]
        d64 = max(abs(a[o] - f[o]) for o in range(2))
        d32 = max(abs(a32[o] - f32[o]) for o in range(2))
        worst = max(worst, d32)
        rows.append([code, f"{a[0]:+.6f} {a[1]:+.6f}", f"{d64:.1e}", f"{d32:.1e}"])
    out = ["A two-output first layer with weights [[0.7, -0.2, 0.4], [-0.5, 0.3, 0.9]] and bias [0.05, -0.1] sees ImageNet-normalized "
           "input. Folding the normalization into it gives weights W[o][c] / std[c] and bias b[o] - sum_c W[o][c] mean[c] / std[c], "
           "which take the unnormalized x in [0, 1] directly.", ""]
    out.append(_table(["pixel codes", "outputs (normalize, then layer)", "|difference| in 64-bit", "|difference| in 32-bit"], rows, "lrrr"))
    out.append("")
    out.append("The 32-bit columns take x = code / 255 computed in 64-bit and rounded to 32 bits, round the weights, mean, std, the folded "
               "W / std and the folded bias (computed in 64-bit) to 32 bits, and round every product, difference, quotient and sum to 32 bits, "
               "adding the three products left to right from 0 and then the bias.")
    return "\n".join(out)


def sum_f32(vals):
    acc = 0.0
    for v in vals:
        acc = r32(acc + v)
    return acc


# ------------------------------------------------------------------------------------------------ camera frames


def frame_slot():
    """Simulate the single-slot latest-frame buffer: push overwrites an untaken frame (counted as dropped); take empties the slot."""
    slot, st = None, {"pushed": 0, "dropped": 0, "taken": 0}
    log = []
    # the consumer takes at ticks 1 and 2, stalls until tick 62, then takes every tick again until tick 65
    takes = {1, 2, 62, 63, 64, 65}
    got = []
    for tick in range(1, 66):
        if slot is not None:
            st["dropped"] += 1
        st["pushed"] += 1
        slot = tick
        if tick in takes:
            if slot is not None:
                got.append(slot)
                st["taken"] += 1
                slot = None
    queue_frames = 60
    out = ["A camera delivers frame number 1 to 65 (one per tick, 30 ticks make a second). The consumer takes the slot at ticks 1, 2, 62, 63, 64 and 65 "
           "(it stalls for 59 ticks in between). The slot has capacity one and the producer overwrites an untaken frame, counting it "
           "as dropped.", ""]
    out.append(_table(["quantity", "single slot", "unbounded queue"],
                      [["frames pushed", st["pushed"], 65],
                       ["frames dropped (and counted)", st["dropped"], 0],
                       ["frames taken by the consumer", st["taken"], st["taken"]],
                       ["frames never taken, still queued after tick 65", 0, 65 - st["taken"]],
                       ["frame numbers the consumer sees", ", ".join(map(str, got)), "1, 2, 3, 4, 5, 6"],
                       ["frames waiting at tick 62", 1, queue_frames],
                       ["bytes waiting at tick 62 (640 x 480 RGB, 921,600 per frame)", f"{921_600:,}", f"{queue_frames * 921_600:,}"],
                       ["age of the frame the consumer gets at tick 62", "0 ticks", "59 ticks (frame 3)"]],
                      "lrr"))
    return "\n".join(out)


def frame_formats():
    rows = []
    for name, bpp in (("RGB, 3 bytes", 3.0), ("YUYV (4:2:2), 2 bytes", 2.0), ("NV12 (4:2:0), 1.5 bytes", 1.5)):
        rows.append([name] + [f"{int(w * h * bpp):,}" for w, h in ((640, 480), (1280, 720), (1920, 1080))])
    return "Bytes in one frame by pixel format and size.\n\n" + _table(["format", "640 x 480", "1280 x 720", "1920 x 1080"], rows)
