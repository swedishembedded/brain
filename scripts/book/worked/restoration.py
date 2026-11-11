# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Restoration and Super-Resolution".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: restoration.<name> -->` markers;
the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree. Earlier
modules are imported, never copied: `denoiser_architectures` (the SDXL U-Net counted from its configuration), `control_and_identity`
(the skip stack of that U-Net) and `stable` (the digits of trained numbers).

* four fully specified degradations (blur, downsampling, noise, compression-like quantisation) on one 8 x 8 picture, and the exact
  ways in which they lose information: the number of pictures that share one low-resolution pixel, the frequency that a blur kills;
* a world of 8 x 8 pictures whose texture an average over 2 x 2 blocks removes, in which the best squared-error restorer can be
  computed in closed form, and a small convolutional network trained on it to show that it finds that restorer and nothing sharper;
  the same network trained for one degradation and for a mixture of them (blind restoration);
* the toy of a texture amplitude: mean prediction, sampling and a feature weight on one pair of pixels, in closed form;
* a codebook prior in miniature (the nearest code against a predictor that reads the whole face) and the fidelity dial;
* the SUPIR sampler's arithmetic on the SDXL noise grid and a sampler built from it on a two-pixel world whose exact denoiser is known;
* brain's three restorers counted from their configurations: parameters, tensors and operations.

Everything random comes from the book's SplitMix64 (`bookrng.generator(name)`); every stream name is printed by the chapter. Trained
numbers are printed with the digits that agree between the run and four copies whose first weight is changed by the nudges of
`stable.py`. The five runs of each network are trained in at most four worker processes. The forward numbers depend on the library
`tanh`, `exp`, `cos` and `sqrt`, and another implementation agrees with them to the rounding of those functions.

Swedish Embedded AB implements solutions for image restoration whose loss of detail is measured rather than assumed. If your team
needs expertise in super-resolution and face and photo restoration, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import multiprocessing
from functools import lru_cache

import bookrng
import control_and_identity as ci
import denoiser_architectures as da
import stable


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _seed(name):
    return f"`{name}` (seed 0x{bookrng.seed_of(name):016x})"


def _int(n):
    return f"{n:,}"


def _f(x, d=4):
    out = f"{x:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _g(x, d=4):
    return f"{x:.{d}g}"


def _gf(x, d=1):
    """A number of GFLOP in math mode, so that the number is not written next to its unit (the units gate of the book)."""
    return "$" + f"{x:,.{d}f}".replace(",", "{,}") + "$ GFLOP"


def _tf(x, d=1):
    """A number of TFLOP in math mode, for the same reason as _gf."""
    return "$" + f"{x:,.{d}f}".replace(",", "{,}") + "$ TFLOP"


def _shared(vals, d):
    """The digits of vals[0] that every nudged run shares (stable.text), written with d decimals."""
    return stable.text(vals[0], vals[1:], lambda x: f"{x:.{d}f}")


# ------------------------------------------------------------------ four degradations on one picture

DEGRADE = "restoration.degradations"
NOISE_SD = 6.0           # standard deviation of the added noise, in grey levels
STEP_DC, STEP_AC = 2.0, 16.0   # quantisation steps of a block's mean and of its three differences, in grey levels


def hr_picture():
    """The 8 x 8 picture of the chapter, in grey levels: a dark field (40) with a bright half (200) below the anti-diagonal
    x + y = 8, and a checkerboard of plus and minus 24 on the dark 4 x 4 corner at the top left."""
    pic = [[0.0] * 8 for _ in range(8)]
    for y in range(8):
        for x in range(8):
            v = 200.0 if x + y >= 8 else 40.0
            if x < 4 and y < 4:
                v += 24.0 if (x + y) % 2 == 0 else -24.0
            pic[y][x] = v
    return pic


def blur_binomial(pic):
    """Horizontal then vertical pass of the kernel (1, 2, 1) / 4; a position outside the picture reads the nearest edge value."""
    n = len(pic)

    def at(row, i):
        return row[min(max(i, 0), n - 1)]

    h = [[(at(r, x - 1) + 2 * r[x] + at(r, x + 1)) / 4 for x in range(n)] for r in pic]
    return [[(h[max(y - 1, 0)][x] + 2 * h[y][x] + h[min(y + 1, n - 1)][x]) / 4 for x in range(n)] for y in range(n)]


def box_down(pic):
    """The mean of each non-overlapping 2 x 2 block."""
    n = len(pic) // 2
    return [[(pic[2 * r][2 * q] + pic[2 * r][2 * q + 1] + pic[2 * r + 1][2 * q] + pic[2 * r + 1][2 * q + 1]) / 4 for q in range(n)]
            for r in range(n)]


def _rnd(v, step):
    """v rounded to the nearest multiple of step, halves upward: step * floor(v / step + 0.5)."""
    return step * math.floor(v / step + 0.5)


def haar_quantise(pic):
    """Compression in miniature. Each 2 x 2 block (p00 p01 / p10 p11) becomes a mean m = (p00 + p01 + p10 + p11) / 4 and three
    differences h = (p00 + p01 - p10 - p11) / 4, v = (p00 - p01 + p10 - p11) / 4, d = (p00 - p01 - p10 + p11) / 4; the mean is rounded
    to a multiple of STEP_DC and each difference to a multiple of STEP_AC; the block is rebuilt as m + h + v + d, m + h - v - d,
    m - h + v - d, m - h - v + d."""
    n = len(pic) // 2
    out = [[0.0] * (2 * n) for _ in range(2 * n)]
    for r in range(n):
        for q in range(n):
            a, b = pic[2 * r][2 * q], pic[2 * r][2 * q + 1]
            c, d = pic[2 * r + 1][2 * q], pic[2 * r + 1][2 * q + 1]
            m = _rnd((a + b + c + d) / 4, STEP_DC)
            h = _rnd((a + b - c - d) / 4, STEP_AC)
            v = _rnd((a - b + c - d) / 4, STEP_AC)
            e = _rnd((a - b - c + d) / 4, STEP_AC)
            out[2 * r][2 * q], out[2 * r][2 * q + 1] = m + h + v + e, m + h - v - e
            out[2 * r + 1][2 * q], out[2 * r + 1][2 * q + 1] = m - h + v - e, m - h - v + e
    return out


def nearest_up(low):
    return [[low[y // 2][x // 2] for x in range(2 * len(low))] for y in range(2 * len(low))]


def _mean_sq(a, b):
    n = len(a)
    return sum((a[y][x] - b[y][x]) ** 2 for y in range(n) for x in range(n)) / (n * n)


def _row(vals, d=1):
    return " ".join(f"{v:.{d}f}" for v in vals)


def degraded_picture():
    """(clean low-resolution picture, blurred then downsampled, plus noise, quantised) of hr_picture, and the noise draws."""
    hr = hr_picture()
    plain = box_down(hr)
    low = box_down(blur_binomial(hr))
    g = bookrng.generator(DEGRADE)
    noisy = [[low[r][q] + NOISE_SD * g.normal() for q in range(4)] for r in range(4)]
    quant = haar_quantise(noisy)
    return hr, plain, low, noisy, quant


def degradations():
    hr, plain, low, noisy, quant = degraded_picture()
    blurred = blur_binomial(hr)
    t1 = _table(["row", "picture x", "blurred"], [[y, _row(hr[y], 0), _row(blurred[y])] for y in range(8)], "rll")
    rows = [[r, _row(plain[r]), _row(low[r]), _row(noisy[r]), _row(quant[r])] for r in range(4)]
    t2 = _table(["row", "box mean only", "blur, then box mean", "+ noise", "+ quantised"], rows, "rllll")
    errs = [("box mean only", plain), ("blur, then box mean", low), ("+ noise", noisy), ("+ quantised", quant)]
    t3 = _table(["stage", "mean squared change of the 4 by 4 picture"],
                [[n, f"{_mean_sq(p, plain):.2f}"] for n, p in errs], "lr")
    return (
        "The picture x (grey levels 0 to 255: 40 on the dark side, 200 on the bright side, a checkerboard of plus and minus 24 on the "
        "dark top-left 4 by 4 corner; rows are y, columns x, the bright side is x + y >= 8) and its blur by (1, 2, 1) / 4 along x and then "
        f"along y with the edge value repeated beyond the border.\n\n{t1}\n\n"
        "The degraded 4 by 4 pictures. 'Box mean only' averages each non-overlapping 2 by 2 block of x; 'blur, then box mean' does so "
        f"after the blur; '+ noise' adds {NOISE_SD:g} times one draw of `normal()` of the stream {_seed(DEGRADE)} to each of the 16 "
        f"pixels in row-major order; '+ quantised' rounds a block's mean to a multiple of {STEP_DC:g} and its three differences to a "
        f"multiple of {STEP_AC:g} (rounding halves upward) as described in the text.\n\n{t2}\n\n"
        f"How far each stage moves the 4 by 4 picture from the box mean of x alone (the mean over the 16 pixels of the squared "
        f"change).\n\n{t3}"
    )


# ------------------------------------------------------------------ how many pictures share one low-resolution pixel

def ways_to_sum(n_values, levels=256):
    """ways[S] = the number of n_values-tuples of integers in 0 .. levels - 1 whose sum is S (exact, by repeated convolution)."""
    ways = [1]
    for _ in range(n_values):
        new = [0] * (len(ways) + levels - 1)
        for i, w in enumerate(ways):
            for v in range(levels):
                new[i + v] += w
        ways = new
    return ways


def ill_posed():
    ways = ways_to_sum(4)
    rows = [[s, _int(ways[s])] for s in (0, 1, 2, 3, 255, 510, 765, 1020)]
    n510 = ways[510]
    digits = len(str(n510 ** 16))
    a = [[60 + 20 * (i + j) for j in range(4)] for i in range(4)]
    b = [[a[i][j] + (30 if (i + j) % 2 == 0 else -30) for j in range(4)] for i in range(4)]

    def box_sums(p):
        return [[p[2 * r][2 * q] + p[2 * r][2 * q + 1] + p[2 * r + 1][2 * q] + p[2 * r + 1][2 * q + 1] for q in range(2)] for r in range(2)]

    sa, sb = box_sums(a), box_sums(b)
    assert sa == sb
    rows2 = [[i, " ".join(f"{v:3d}" for v in a[i]), " ".join(f"{v:3d}" for v in b[i]), " ".join(f"{v:3d}" for v in
             [x - y for x, y in zip(b[i], a[i])])] for i in range(4)]
    return (
        "The number of ways to choose the four pixels of a 2 by 2 block (integers 0 to 255) so that they sum to a given S, which is "
        "all that the 2 by 2 box mean of the block (S / 4) records. Counts are exact integers.\n\n"
        + _table(["sum S", "blocks with that sum"], rows)
        + f"\n\nA 4 by 4 picture all of whose blocks sum to 510 (every low-resolution pixel 127.5) is the box mean of {n510:,}^16 "
        f"different 8 by 8 pictures, a number of {digits} digits. Two 4 by 4 pictures with the same four block sums: A, B = A plus a "
        "checkerboard of plus and minus 30, and the difference B - A (the block sums of both are "
        f"{sa[0][0]}, {sa[0][1]} / {sa[1][0]}, {sa[1][1]}).\n\n"
        + _table(["row", "A", "B", "B - A"], rows2, "rlll")
    )


def blur_response():
    n = 16
    h = [(1 + math.cos(2 * math.pi * k / n)) / 2 for k in range(n // 2 + 1)]
    alt = [(-1.0) ** i for i in range(n)]
    blurred = [(alt[(i - 1) % n] + 2 * alt[i] + alt[(i + 1) % n]) / 4 for i in range(n)]
    gain = math.sqrt(sum(1.0 / h[min(k, n - k)] ** 2 for k in range(n) if k != n // 2) / n)
    rows = [[k, _f(h[k], 6), "no inverse" if h[k] < 1e-12 else _f(1 / h[k], 3)] for k in range(n // 2 + 1)]
    return (
        "The response of the kernel (1, 2, 1) / 4 on a periodic row of 16 pixels at frequency k (k periods per row): "
        "H_k = (1 + cos(2 pi k / 16)) / 2, and the factor 1 / H_k by which an inverse filter must multiply that frequency.\n\n"
        + _table(["k", "H_k", "1 / H_k"], rows)
        + f"\n\nThe alternating row (+1, -1, +1, ...) blurred by this kernel is {"exactly 0 at every pixel" if all(v == 0.0 for v in blurred) else "not 0"} after the blur. An inverse filter that leaves out k = 8 and divides every other frequency by its H_k "
        f"multiplies the standard deviation of white noise by {gain:.2f}."
    )


# ------------------------------------------------------------------ what a pair of pixels can tell: mean, sample, feature weight

def _judge_accuracy(real, fake):
    """The best accuracy of a judge that sees one number (the signed gap) when real and fake examples are equally likely: 1/2 plus half
    the total variation distance between the two distributions, each given as {value: probability}."""
    keys = set(real) | set(fake)
    return 0.5 + 0.25 * sum(abs(real.get(k, 0.0) - fake.get(k, 0.0)) for k in keys)


def pair_world():
    # a pair (m + t, m - t): the low-resolution pixel is m, the lost half-difference t = +1 or -1 (units of delta) with probability 1/2
    real = {2.0: 0.5, -2.0: 0.5}                      # the signed gap, first pixel minus second, of a real pair
    rows = []
    for name, outputs in (
        ("the mean: (m, m)", {0.0: 1.0}),
        ("a sample: (m + t', m - t')", {1.0: 0.5, -1.0: 0.5}),
        ("a fixed guess: (m + 1, m - 1)", {1.0: 1.0}),
    ):
        mse = sum(0.5 * po * (t - o) ** 2 for t in (1.0, -1.0) for o, po in outputs.items())
        fake = {2 * o: po for o, po in outputs.items()}
        acc = _judge_accuracy(real, fake)
        rows.append([name, f"{mse:.2f}", " or ".join(f"{k:+g}" if k else "0" for k in sorted(fake)), f"{acc * 100:.0f}%"])
    return (
        "A pair of pixels (m + t, m - t) in units of delta, with the pair average m known to the restorer and t = +1 or -1 with probability 1/2 "
        "each, lost by the averaging. For three restorers: the mean squared error per pixel in units of delta squared, the signed gap (first "
        "pixel minus second, real pairs have +2 or -2) that their outputs show, and the best accuracy of a judge that sees only that gap and "
        "is shown real and restored pairs equally often (50% is a coin toss).\n\n"
        + _table(["restorer", "mean squared error", "gap shown", "judge accuracy"], rows, "lrrr")
    )


def feature_weight_rows(lams=(0.0, 0.25, 1.0, 4.0, 16.0, 1000.0)):
    """Texture amplitude a (in units of delta) minimising a^2 + lam (a - 1)^2: a* = lam / (1 + lam); the pair's error is 1 + a^2."""
    rows = []
    for lam in lams:
        a = lam / (1 + lam)
        # a ternary search over [0, 2] confirms the closed form
        lo, hi = 0.0, 2.0
        for _ in range(100):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if m1 ** 2 + lam * (m1 - 1) ** 2 < m2 ** 2 + lam * (m2 - 1) ** 2:
                hi = m2
            else:
                lo = m1
        assert abs((lo + hi) / 2 - a) < 1e-6
        rows.append((lam, a, 1 + a * a))
    return rows


def feature_weight():
    rows = [[_g(lam), f"{a:.4f}", f"{mse:.4f}", "seen (a < 1)" if a < 1 - 1e-12 else "not seen"] for lam, a, mse in feature_weight_rows()]
    return (
        "Restorers that return (m + a s, m - a s) with s = +1 or -1 drawn like t, and a the amplitude of the texture they produce, "
        "in units of delta. Minimising a^2 + lambda (a - 1)^2 (the first term is what a restorer's squared error would add to the "
        "unavoidable error, the second a penalty on a gap that differs from the real gap 2, with weight lambda) gives a* = lambda / (1 + "
        "lambda), and the mean squared error per pixel is 1 + a*^2 in units of delta squared. Whether the gap judge of the previous "
        "table can see the difference from real pairs (a* = 1 is the only amplitude it cannot).\n\n"
        + _table(["lambda", "a*", "mean squared error", "judge"], rows)
    )


# ------------------------------------------------------------------ a world of pictures whose texture the average removes

DELTA = 0.08                                   # amplitude of one texture coefficient
N_TRAIN, N_TEST, BATCH, STEPS = 256, 256, 32, 300
LR_RATE, ADAM_B1, ADAM_B2, ADAM_EPS = 0.01, 0.9, 0.999, 1e-8
CHECKPOINTS = (0, 50, 100, 200, 300)
SR_TRAIN, SR_TEST, SR_INIT, SR_SAMPLE = "restoration.sr.train", "restoration.sr.test", "restoration.sr.init", "restoration.sr.sample"
SR_LEVELS, SR_NOISE_TRAIN = "restoration.sr.levels", "restoration.sr.noise.train"
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.1)          # standard deviations of the noise added to a low-resolution pixel
LEAK = 0.2
N_W1, N_B1, N_W2, N_B2 = 27, 3, 108, 4         # the weights of the network: 142 numbers


def noise_test_stream(level):
    return f"restoration.sr.noise.test.{level:g}"


def draw_texture(g):
    """The 8 x 8 texture of one picture: for each of the 8 blocks of the right half (low-resolution rows r = 0..3, columns q = 2, 3, in
    that order) three signs, each +1 when `uniform()` < 0.5 and -1 otherwise, in the order h, v, d. Pixel (iy, ix) of the block gets
    DELTA * (h*sh + v*sv + d*sh*sv) with sh = +1 for iy = 0 and -1 for iy = 1, and sv = +1 for ix = 0 and -1 for ix = 1."""
    t = [[0.0] * 8 for _ in range(8)]
    for r in range(4):
        for q in (2, 3):
            h, v, d = (1 if g.uniform() < 0.5 else -1 for _ in range(3))
            for iy in range(2):
                for ix in range(2):
                    sh, sv = (1 if iy == 0 else -1), (1 if ix == 0 else -1)
                    t[2 * r + iy][2 * q + ix] = DELTA * (h * sh + v * sv + d * sh * sv)
    return t


def draw_picture(g):
    """(picture, texture): a plane a + b (x - 3.5) + c (y - 3.5) with a = 0.4 + 0.2 u1, b = -0.06 + 0.12 u2, c = -0.06 + 0.12 u3 (the
    three `uniform()` draws in that order), plus the texture of draw_texture, drawn after them."""
    a = 0.4 + 0.2 * g.uniform()
    b = -0.06 + 0.12 * g.uniform()
    c = -0.06 + 0.12 * g.uniform()
    t = draw_texture(g)
    return [[a + b * (x - 3.5) + c * (y - 3.5) + t[y][x] for x in range(8)] for y in range(8)], t


@lru_cache(maxsize=None)
def make_set(name, n):
    """n pictures from the stream `name`: a list of (picture, low-resolution picture, texture)."""
    g = bookrng.generator(name)
    out = []
    for _ in range(n):
        x, t = draw_picture(g)
        out.append((x, box_down(x), t))
    return out


def noisy_lows(lows, levels, stream):
    """Each low-resolution picture plus `level` times one `normal()` per pixel, row-major, the pictures in order, from one stream."""
    g = bookrng.generator(stream)
    return [[[lo[r][q] + s * g.normal() for q in range(4)] for r in range(4)] for lo, s in zip(lows, levels)]


def init_theta():
    """The 142 weights, in the order W1 (3 channels x 3 x 3, row-major), b1, W2 (4 x 3 channels x 3 x 3), b2: the weights of W1 are
    `normal()` times sqrt(2/9) and those of W2 `normal()` times 0.1, drawn in that order from SR_INIT; the biases are 0."""
    g = bookrng.generator(SR_INIT)
    w1 = [g.normal() * math.sqrt(2.0 / 9.0) for _ in range(N_W1)]
    w2 = [g.normal() * 0.1 for _ in range(N_W2)]
    return w1 + [0.0] * N_B1 + w2 + [0.0] * N_B2


def _unpack(th):
    w1 = [[[th[9 * c + 3 * ky + kx] for kx in range(3)] for ky in range(3)] for c in range(3)]
    b1 = th[27:30]
    o = 30
    w2 = [[[[th[o + 27 * co + 9 * ci + 3 * ky + kx] for kx in range(3)] for ky in range(3)] for ci in range(3)] for co in range(4)]
    return w1, b1, w2, th[o + N_W2:o + N_W2 + N_B2]


def sr_forward(th, low):
    """(pre-activations of layer 1, activations, 8 x 8 output): conv 3 x 3 (1 -> 3 channels, zero padding) + bias, leaky ReLU 0.2,
    conv 3 x 3 (3 -> 4 channels, zero padding) + bias, then channel 2i + j of position (r, q) is added to the nearest-enlargement
    pixel (2r + i, 2q + j)."""
    w1, b1, w2, b2 = _unpack(th)
    z1 = [[[0.0] * 4 for _ in range(4)] for _ in range(3)]
    for c in range(3):
        w = w1[c]
        for r in range(4):
            for q in range(4):
                s = b1[c]
                for ky in range(3):
                    rr = r + ky - 1
                    if 0 <= rr < 4:
                        for kx in range(3):
                            qq = q + kx - 1
                            if 0 <= qq < 4:
                                s += w[ky][kx] * low[rr][qq]
                z1[c][r][q] = s
    a1 = [[[v if v > 0 else LEAK * v for v in row] for row in ch] for ch in z1]
    y = [[[0.0] * 4 for _ in range(4)] for _ in range(4)]
    for co in range(4):
        for r in range(4):
            for q in range(4):
                s = b2[co]
                for ci in range(3):
                    w, a = w2[co][ci], a1[ci]
                    for ky in range(3):
                        rr = r + ky - 1
                        if 0 <= rr < 4:
                            for kx in range(3):
                                qq = q + kx - 1
                                if 0 <= qq < 4:
                                    s += w[ky][kx] * a[rr][qq]
                y[co][r][q] = s
    out = [[low[yy // 2][xx // 2] + y[2 * (yy % 2) + xx % 2][yy // 2][xx // 2] for xx in range(8)] for yy in range(8)]
    return z1, a1, out


def sr_gradient(th, low, target, scale):
    """The gradient of scale * sum over the 64 pixels of (output - target)^2 with respect to the 142 weights, same order as th."""
    z1, a1, out = sr_forward(th, low)
    _, _, w2, _ = _unpack(th)
    dy = [[[2 * scale * (out[2 * r + i][2 * q + j] - target[2 * r + i][2 * q + j]) for q in range(4)] for r in range(4)]
          for i in range(2) for j in range(2)]
    g_w2 = [0.0] * N_W2
    g_b2 = [0.0] * 4
    da1 = [[[0.0] * 4 for _ in range(4)] for _ in range(3)]
    for co in range(4):
        for r in range(4):
            for q in range(4):
                g = dy[co][r][q]
                g_b2[co] += g
                for ci in range(3):
                    a, w, da = a1[ci], w2[co][ci], da1[ci]
                    base = 27 * co + 9 * ci
                    for ky in range(3):
                        rr = r + ky - 1
                        if 0 <= rr < 4:
                            for kx in range(3):
                                qq = q + kx - 1
                                if 0 <= qq < 4:
                                    g_w2[base + 3 * ky + kx] += g * a[rr][qq]
                                    da[rr][qq] += g * w[ky][kx]
    g_w1 = [0.0] * N_W1
    g_b1 = [0.0] * 3
    for c in range(3):
        for r in range(4):
            for q in range(4):
                g = da1[c][r][q] * (1.0 if z1[c][r][q] > 0 else LEAK)
                g_b1[c] += g
                for ky in range(3):
                    rr = r + ky - 1
                    if 0 <= rr < 4:
                        for kx in range(3):
                            qq = q + kx - 1
                            if 0 <= qq < 4:
                                g_w1[9 * c + 3 * ky + kx] += g * low[rr][qq]
    return g_w1 + g_b1 + g_w2 + g_b2


def sr_loss(th, lows, targets):
    """The mean over the pictures and the 64 pixels of (output - target)^2, each picture's 64 terms added row-major, the pictures in order."""
    tot = 0.0
    for lo, x in zip(lows, targets):
        out = sr_forward(th, lo)[2]
        s = 0.0
        for y in range(8):
            for xx in range(8):
                s += (out[y][xx] - x[y][xx]) ** 2
        tot += s
    return tot / (64 * len(lows))


def sr_batch_gradient(th, lows, targets):
    scale = 1.0 / (64 * len(lows))
    total = [0.0] * len(th)
    for lo, x in zip(lows, targets):
        g = sr_gradient(th, lo, x, scale)
        total = [a + b for a, b in zip(total, g)]
    return total


def sr_train(theta0, lows, targets, nudge=0.0):
    """STEPS Adam steps (rate LR_RATE, betas ADAM_B1 and ADAM_B2, epsilon ADAM_EPS, bias-corrected moments) on batches of BATCH
    pictures taken in order and cyclically from the training set (step t uses pictures (t-1)*BATCH mod N_TRAIN onwards). The first
    weight is changed by `nudge` before the first step. Returns the final weights and the mean squared error on the whole training set
    after 0, 50, 100, 200 and 300 steps."""
    th = list(theta0)
    th[0] += nudge
    m, v = [0.0] * len(th), [0.0] * len(th)
    log = {}
    n = len(lows)
    for t in range(1, STEPS + 1):
        if t - 1 in CHECKPOINTS:
            log[t - 1] = sr_loss(th, lows, targets)
        k = ((t - 1) * BATCH) % n
        g = sr_batch_gradient(th, lows[k:k + BATCH], targets[k:k + BATCH])
        for i in range(len(th)):
            m[i] = ADAM_B1 * m[i] + (1 - ADAM_B1) * g[i]
            v[i] = ADAM_B2 * v[i] + (1 - ADAM_B2) * g[i] * g[i]
            th[i] -= LR_RATE * (m[i] / (1 - ADAM_B1 ** t)) / (math.sqrt(v[i] / (1 - ADAM_B2 ** t)) + ADAM_EPS)
    log[STEPS] = sr_loss(th, lows, targets)
    return th, log


def bilinear_up(low):
    """Each output pixel (Y, X) is 9/16 of its own block's pixel, 3/16 of the neighbour above or below (the side Y is nearer to), 3/16
    of the neighbour left or right, and 1/16 of the diagonal neighbour; a neighbour outside the picture is replaced by the nearest edge pixel."""
    def at(r, q):
        return low[min(max(r, 0), 3)][min(max(q, 0), 3)]

    out = [[0.0] * 8 for _ in range(8)]
    for yy in range(8):
        r, dr = yy // 2, (-1 if yy % 2 == 0 else 1)
        for xx in range(8):
            q, dq = xx // 2, (-1 if xx % 2 == 0 else 1)
            out[yy][xx] = (9 * at(r, q) + 3 * at(r + dr, q) + 3 * at(r, q + dq) + at(r + dr, q + dq)) / 16
    return out


def detail_energy(p):
    """(left half, right half): the mean over the pixels of the half of (pixel - mean of its 2 x 2 block)^2."""
    tot = [0.0, 0.0]
    for r in range(4):
        for q in range(4):
            blk = [p[2 * r][2 * q], p[2 * r][2 * q + 1], p[2 * r + 1][2 * q], p[2 * r + 1][2 * q + 1]]
            mu = sum(blk) / 4
            tot[q // 2] += sum((v - mu) ** 2 for v in blk)
    return tot[0] / 32, tot[1] / 32


def _mse_images(outs, targets):
    return sum(sum((o[y][x] - t[y][x]) ** 2 for y in range(8) for x in range(8)) / 64 for o, t in zip(outs, targets)) / len(targets)


def _mean_detail(outs):
    d = [detail_energy(o) for o in outs]
    return sum(a for a, _ in d) / len(d), sum(b for _, b in d) / len(d)


def sr_gradcheck():
    """Relative error of the whole gradient of the loss on 4 training pictures at the initial weights against central differences of
    step 1e-5: the norm of (analytic - numeric) over the norm of analytic."""
    tr = make_set(SR_TRAIN, N_TRAIN)[:4]
    lows, tgts = [t[1] for t in tr], [t[0] for t in tr]
    th = init_theta()
    ana = sr_batch_gradient(th, lows, tgts)
    num = []
    for i in range(len(th)):
        p, m = list(th), list(th)
        p[i] += 1e-5
        m[i] -= 1e-5
        num.append((sr_loss(p, lows, tgts) - sr_loss(m, lows, tgts)) / 2e-5)
    err = math.sqrt(sum((a - b) ** 2 for a, b in zip(ana, num))) / math.sqrt(sum(a * a for a in ana))
    return err


def _test_lows(level):
    lows = [t[1] for t in make_set(SR_TEST, N_TEST)]
    if level == 0.0:
        return lows
    return noisy_lows(lows, [level] * N_TEST, noise_test_stream(level))


def _train_lows(kind):
    """The inputs a network of this kind is trained on: the clean low-resolution pictures ('known'), or the same pictures each with
    noise of a level drawn from NOISE_LEVELS by `integer(4)` (one draw per picture, stream SR_LEVELS) ('blind')."""
    lows = [t[1] for t in make_set(SR_TRAIN, N_TRAIN)]
    if kind == "known":
        return lows
    g = bookrng.generator(SR_LEVELS)
    levels = [NOISE_LEVELS[g.integer(len(NOISE_LEVELS))] for _ in lows]
    return noisy_lows(lows, levels, SR_NOISE_TRAIN)


def sample_texture_set():
    """The textures a sampling restorer adds to the test pictures: draw_texture from the stream SR_SAMPLE, one per picture."""
    g = bookrng.generator(SR_SAMPLE)
    return [draw_texture(g) for _ in range(N_TEST)]


def _sr_job(args):
    kind, nudge = args
    train = make_set(SR_TRAIN, N_TRAIN)
    th, log = sr_train(init_theta(), _train_lows(kind), [t[0] for t in train], nudge)
    test = make_set(SR_TEST, N_TEST)
    targets = [t[0] for t in test]
    res = {"log": [log[c] for c in CHECKPOINTS]}
    res["test"] = [sr_loss(th, _test_lows(level), targets) for level in NOISE_LEVELS]
    outs = [sr_forward(th, t[1])[2] for t in test]
    res["detail"] = list(_mean_detail(outs))
    textures = sample_texture_set()
    res["sampled"] = _mse_images([[[o[y][x] + tex[y][x] for x in range(8)] for y in range(8)] for o, tex in zip(outs, textures)], targets)
    res["theta"] = th
    return res


@lru_cache(maxsize=None)
def sr_runs(kind):
    """The run of this kind ('known' or 'blind') and its four nudged copies, trained in four worker processes: a list of 5 dicts."""
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(4) as pool:
        return pool.map(_sr_job, [(kind, d) for d in (0.0, *stable.NUDGES)])


def sr_setup():
    g = bookrng.generator(SR_TRAIN)
    a, b, c = 0.4 + 0.2 * g.uniform(), -0.06 + 0.12 * g.uniform(), -0.06 + 0.12 * g.uniform()
    first = make_set(SR_TRAIN, N_TRAIN)[0]
    th = init_theta()
    start = [sr_loss(th, [t[1] for t in make_set(SR_TRAIN, N_TRAIN)], [t[0] for t in make_set(SR_TRAIN, N_TRAIN)])]
    rows = [
        ["`W1`", "3 x 1 x 3 x 3 (output channel, input channel, row, column)", N_W1, "`normal()` x sqrt(2/9)"],
        ["`b1`", "3", N_B1, "0"],
        ["`W2`", "4 x 3 x 3 x 3", N_W2, "`normal()` x 0.1"],
        ["`b2`", "4", N_B2, "0"],
    ]
    t1 = _table(["tensor", "shape", "numbers", "initial value, in draw order"], rows, "llrl")
    t2 = _table(["quantity", "value"], [
        ["streams", f"{_seed(SR_TRAIN)} (256 pictures), {_seed(SR_TEST)} (256), {_seed(SR_INIT)}"],
        ["first training picture: a, b, c", f"{a:.6f}, {b:.6f}, {c:.6f}"],
        ["its low-resolution row 0", _row(first[1][0], 6)],
        ["mean squared error of the initial network on the training set", f"{start[0]:.6f}"],
    ], "ll")
    return (
        "The network and its starting point. The picture is 8 by 8, the input 4 by 4; the input and every hidden map are zero-padded by "
        "one pixel for the 3 by 3 convolutions. The 142 numbers are drawn from the stream `restoration.sr.init` with `normal()`, W1 first and W2 "
        "second, each in row-major order of its shape.\n\n" + t1 + "\n\n" + t2
    )


def texture_energy(test):
    """The mean over the test pictures and the 64 pixels of the squared texture: the part of the error that no restorer sees."""
    return sum(sum(v * v for row in t[2] for v in row) / 64 for t in test) / len(test)


def sr_training():
    runs = sr_runs("known")
    test = make_set(SR_TEST, N_TEST)
    targets = [t[0] for t in test]
    near = _mse_images([nearest_up(t[1]) for t in test], targets)
    bil = _mse_images([bilinear_up(t[1]) for t in test], targets)
    tex = texture_energy(test)
    d6 = lambda key, i=None: _shared([(r[key] if i is None else r[key][i]) for r in runs], 6)  # noqa: E731
    t1 = _table(["steps taken", "mean squared error on the 256 training pictures"],
                [[c, d6("log", i)] for i, c in enumerate(CHECKPOINTS)])
    drows = [
        ["the pictures themselves", _f(sum(detail_energy(t[0])[0] for t in test) / len(test), 6), _f(sum(detail_energy(t[0])[1] for t in test) / len(test), 6)],
        ["nearest enlargement", "0.000000", "0.000000"],
        ["bilinear enlargement", *[_f(v, 6) for v in _mean_detail([bilinear_up(t[1]) for t in test])]],
        ["the network", d6("detail", 0), d6("detail", 1)],
    ]
    t3 = _table(["output", "detail energy, left half", "detail energy, right half"], drows)
    rrows = [
        ["nearest enlargement", _f(near, 6), _f(near - tex, 6)],
        ["bilinear enlargement", _f(bil, 6), _f(bil - tex, 6)],
        ["the network, 300 steps", d6("test", 0), _shared([r["test"][0] - tex for r in runs], 6)],
        ["the network plus a texture drawn like the real one", d6("sampled"), _shared([r["sampled"] - tex for r in runs], 6)],
    ]
    t2 = _table(["restorer", "mean squared error", "minus the texture energy"], rrows)
    return (
        f"The network trained for {STEPS} Adam steps (learning rate {LR_RATE:g}, first moment {ADAM_B1:g}, second moment {ADAM_B2:g}, "
        f"epsilon {ADAM_EPS:g}, moments corrected for their start) on batches of {BATCH} training pictures taken in order and cyclically; "
        "step t uses pictures (t - 1) x 32 mod 256 onwards. The loss is the mean over the batch and the 64 pixels of the squared error. "
        f"{stable.RULE}\n\n"
        f"The check of the hand-written gradient: on four training pictures at the initial weights the norm of (analytic - numeric) "
        f"over the norm of the analytic gradient of the loss, with central differences of step 1e-5 on every one of the 142 weights, is "
        f"{'below 1e-8' if sr_gradcheck() < 1e-8 else 'NOT below 1e-8'} (the criterion).\n\n{t1}\n\n"
        f"On the 256 test pictures. The texture energy is the mean squared value of the texture the average removed, {tex:.6f}: the same "
        f"for every picture, since the three patterns of a block are orthogonal and each carries a sign of size 0.08, so a block's four "
        f"pixels have squared values adding to 12 x 0.08^2 and half the pixels of a picture are in such blocks (1.5 x 0.08^2 = {1.5 * DELTA ** 2:.6f}). "
        "The last row adds to the network's picture a fresh texture drawn by `draw_texture` from the stream "
        f"{_seed(SR_SAMPLE)}, one per test picture.\n\n{t2}\n\n"
        "The detail energy of a picture is the mean over its pixels of the squared difference between a pixel and the mean of its 2 by 2 block, "
        "measured separately on the left half (no texture) and the right half (texture) and averaged over the 256 test pictures.\n\n" + t3
    )


def blind_restoration():
    known, blind = sr_runs("known"), sr_runs("blind")
    test = make_set(SR_TEST, N_TEST)
    targets = [t[0] for t in test]
    rows = []
    for i, s in enumerate(NOISE_LEVELS):
        lows = _test_lows(s)
        near = _mse_images([nearest_up(lo) for lo in lows], targets)
        bil = _mse_images([bilinear_up(lo) for lo in lows], targets)
        rows.append([f"{s:g}", _f(near, 6), _f(bil, 6), _shared([r["test"][i] for r in known], 6), _shared([r["test"][i] for r in blind], 6)])
    t = _table(["noise level", "nearest", "bilinear", "trained on clean pictures", "trained on a mixture"], rows)
    return (
        "The same network (the same 142 initial numbers, the same 300 steps) trained twice: on the clean low-resolution pictures, and on the "
        "same pictures with noise added to each: its level one of 0, 0.02, 0.05, 0.1 chosen by one `integer(4)` draw per picture from the stream "
        f"{_seed(SR_LEVELS)}, then the noise of each picture drawn row-major from {_seed(SR_NOISE_TRAIN)} (16 draws per picture, every "
        "picture included, each multiplied by the level of its picture, so that a picture of level 0 is unchanged). Mean squared error on "
        "the 256 test pictures when 4 by 4 pixels of noise of the given standard deviation are added to every low-resolution picture "
        "(from the stream `restoration.sr.noise.test.<level>`, 16 draws per picture, row-major; level 0 adds nothing).\n\n" + t
    )


def sr_pictures_data():
    """(truth, low-resolution, the network's output, the output plus a sampled texture) of the first test picture."""
    test = make_set(SR_TEST, N_TEST)
    th = sr_runs("known")[0]["theta"]
    x, lo, _ = test[0]
    out = sr_forward(th, lo)[2]
    tex = sample_texture_set()[0]
    return x, lo, out, [[out[y][xx] + tex[y][xx] for xx in range(8)] for y in range(8)]


# ------------------------------------------------------------------ a codebook prior in miniature

CB_K, CB_T, CB_M, CB_RADIUS, CB_N = 8, 8, 16, 1.5, 2000
CB_TEMPLATES, CB_FACES = "restoration.codebook.templates", "restoration.codebook.faces"
CB_SIGMAS = (0.25, 0.5, 0.75, 1.0)


def codebook():
    """Code k is the point (1.5 cos(2 pi k / 8), 1.5 sin(2 pi k / 8)) of the plane, k = 0 .. 7."""
    return [(CB_RADIUS * math.cos(2 * math.pi * k / CB_K), CB_RADIUS * math.sin(2 * math.pi * k / CB_K)) for k in range(CB_K)]


@lru_cache(maxsize=None)
def templates():
    """16 faces of 8 positions each: template m, position t is `integer(8)`, drawn for m = 0 .. 15 and t = 0 .. 7 in that order."""
    g = bookrng.generator(CB_TEMPLATES)
    return tuple(tuple(g.integer(CB_K) for _ in range(CB_T)) for _ in range(CB_M))


@lru_cache(maxsize=None)
def cb_faces():
    """(the template number of each in-prior face, the codes of each off-prior face, the unit noise): `integer(16)` for each of the
    CB_N in-prior faces; then `integer(8)` for each of the 8 positions of each of the CB_N off-prior faces; then two `normal()` draws
    (x then y) for each position of each face, faces in order: the noise a face carries at standard deviation s is s times these."""
    g = bookrng.generator(CB_FACES)
    ids = [g.integer(CB_M) for _ in range(CB_N)]
    off = [[g.integer(CB_K) for _ in range(CB_T)] for _ in range(CB_N)]
    noise = [[(g.normal(), g.normal()) for _ in range(CB_T)] for _ in range(CB_N)]
    return ids, off, noise


def _d2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def nearest_codes(feats, cb):
    """The code nearest to each feature (the lowest index on a tie): what a vector quantizer does."""
    return [min(range(CB_K), key=lambda k: (_d2(f, cb[k]), k)) for f in feats]


def read_whole_face(feats, cb, tpl):
    """The best template: the m whose codes have the smallest sum over the positions of the squared distance to the features (the
    lowest m on a tie), and its codes. For features that are a template's codes plus equal Gaussian noise it is the most probable
    template; a transformer that has learned the faces approximates it."""
    best = min(range(CB_M), key=lambda m: (sum(_d2(feats[t], cb[tpl[m][t]]) for t in range(CB_T)), m))
    return list(tpl[best])


def codebook_rates(sigma):
    cb, tpl = codebook(), templates()
    ids, off, noise = cb_faces()
    res = {"near_in": 0, "glob_in": 0, "whole_in": 0, "near_off": 0, "glob_off": 0}
    for i in range(CB_N):
        feats_in = [(cb[tpl[ids[i]][t]][0] + sigma * noise[i][t][0], cb[tpl[ids[i]][t]][1] + sigma * noise[i][t][1]) for t in range(CB_T)]
        feats_off = [(cb[off[i][t]][0] + sigma * noise[i][t][0], cb[off[i][t]][1] + sigma * noise[i][t][1]) for t in range(CB_T)]
        n_in, g_in = nearest_codes(feats_in, cb), read_whole_face(feats_in, cb, tpl)
        n_off, g_off = nearest_codes(feats_off, cb), read_whole_face(feats_off, cb, tpl)
        res["near_in"] += sum(a != b for a, b in zip(n_in, tpl[ids[i]]))
        res["glob_in"] += sum(a != b for a, b in zip(g_in, tpl[ids[i]]))
        res["whole_in"] += g_in == list(tpl[ids[i]])
        res["near_off"] += sum(a != b for a, b in zip(n_off, off[i]))
        res["glob_off"] += sum(a != b for a, b in zip(g_off, off[i]))
    pos = CB_N * CB_T
    return {k: (v / CB_N if k == "whole_in" else v / pos) for k, v in res.items()}


def codebook_prior():
    rows = []
    for s in CB_SIGMAS:
        r = codebook_rates(s)
        rows.append([f"{s:g}", f"{r['near_in'] * 100:.1f}%", f"{r['glob_in'] * 100:.1f}%", f"{r['whole_in'] * 100:.1f}%",
                     f"{r['near_off'] * 100:.1f}%", f"{r['glob_off'] * 100:.1f}%"])
    cb = codebook()
    tpl = templates()
    z = lambda v: 0.0 if abs(v) < 5e-5 else v  # noqa: E731  (a sine of pi that is -1e-16 is written 0.0000, not -0.0000)
    cbrows = [[k, f"({z(cb[k][0]):.4f}, {z(cb[k][1]):.4f})"] for k in range(CB_K)]
    t1 = _table(["code k", "vector"], cbrows, "rl")
    t2 = _table(["noise level", "nearest code, wrong", "whole-face read, wrong", "whole face right", "nearest, wrong (not a template)",
                 "whole-face read, wrong (not a template)"], rows)
    return (
        "A codebook of 8 vectors of the plane and a prior of 16 faces of 8 positions each. The codes are the points "
        f"(1.5 cos(2 pi k / 8), 1.5 sin(2 pi k / 8)); face m is the list of 8 codes drawn from the stream {_seed(CB_TEMPLATES)} "
        "(`integer(8)`, m = 0 .. 15, positions 0 .. 7). A degraded face has, at each position, the vector of its code plus the noise level times two "
        f"`normal()` draws (x then y) from the stream {_seed(CB_FACES)}, which draws first 2,000 face numbers (`integer(16)`), then the "
        "codes of 2,000 faces that are not templates (`integer(8)` for each of 8 positions), then the noise (two draws per position, "
        "faces in order). 'Nearest code' replaces each position's vector by the index of the nearest code; the whole-face read picks "
        "the template with the smallest summed squared distance and returns its codes. Shares of the 16,000 positions that come out "
        "wrong, and of the 2,000 faces that come out right.\n\n" + t1 + "\n\n" + t2
    )


def codebook_curves_data():
    xs = [0.2 + 0.1 * i for i in range(11)]
    return xs, [codebook_rates(s)["near_in"] for s in xs], [codebook_rates(s)["glob_in"] for s in xs], [codebook_rates(s)["glob_off"] for s in xs]


DIAL_STREAM, DIAL_DETAIL, DIAL_SIGMA = "restoration.dial", 0.5, 0.5


def dial_rates(ws=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Errors of out(w) = d + w (e - d) against the true feature f, per coordinate, for faces on the prior (f = the template's codes)
    and for faces with detail the codebook cannot carry (f = codes + u, u of standard deviation DIAL_DETAIL), when the encoder feature e
    is f plus noise of standard deviation DIAL_SIGMA and d is the codes of the whole-face read of e."""
    cb, tpl = codebook(), templates()
    ids, _, noise = cb_faces()
    g = bookrng.generator(DIAL_STREAM)
    detail = [[(g.normal(), g.normal()) for _ in range(CB_T)] for _ in range(CB_N)]
    out = {w: [0.0, 0.0, 0.0] for w in ws}
    for i in range(CB_N):
        base = [cb[tpl[ids[i]][t]] for t in range(CB_T)]
        for k, u_scale in ((0, 0.0), (1, DIAL_DETAIL)):
            f = [(base[t][0] + u_scale * detail[i][t][0], base[t][1] + u_scale * detail[i][t][1]) for t in range(CB_T)]
            e = [(f[t][0] + DIAL_SIGMA * noise[i][t][0], f[t][1] + DIAL_SIGMA * noise[i][t][1]) for t in range(CB_T)]
            d = [cb[c] for c in read_whole_face(e, cb, tpl)]
            for w in ws:
                o = [(d[t][0] + w * (e[t][0] - d[t][0]), d[t][1] + w * (e[t][1] - d[t][1])) for t in range(CB_T)]
                out[w][k] += sum(_d2(o[t], f[t]) for t in range(CB_T)) / (2 * CB_T)
                if k == 1:
                    out[w][2] += sum(_d2(o[t], e[t]) for t in range(CB_T)) / (2 * CB_T)
    return {w: [v / CB_N for v in vals] for w, vals in out.items()}


def fidelity_dial():
    r = dial_rates()
    rows = [[f"{w:g}", f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}"] for w, v in r.items()]
    w_star = DIAL_DETAIL ** 2 / (DIAL_DETAIL ** 2 + DIAL_SIGMA ** 2)
    return (
        "The fidelity dial on the toy faces. The encoder's feature e is the true feature f plus noise of standard deviation "
        f"{DIAL_SIGMA:g} (the noise of the previous table, stream {_seed(CB_FACES)}); d is the codes of the whole-face read of e; "
        "the output is out(w) = d + w (e - d). 'On the prior' faces have f equal to their template's codes; 'with detail' faces have "
        f"f equal to those codes plus a personal offset, two `normal()` draws of standard deviation {DIAL_DETAIL:g} per position, from the "
        f"stream {_seed(DIAL_STREAM)} (drawn per position, x then y, faces in order). Mean squared error per coordinate against f, and for "
        "the faces with detail the mean squared distance to the degraded input e.\n\n"
        + _table(["w", "error, on the prior", "error, with detail", "distance to e, with detail"], rows)
        + f"\n\nWith d equal to the codes, the error per coordinate is {DIAL_SIGMA ** 2:g} w^2 on the prior and {DIAL_DETAIL ** 2:g} (1 - w)^2 "
        f"+ {DIAL_SIGMA ** 2:g} w^2 with detail (the squared offset and the squared noise), which is smallest at w = "
        f"{DIAL_DETAIL ** 2:g} / ({DIAL_DETAIL ** 2:g} + {DIAL_SIGMA ** 2:g}) = {w_star:g}."
    )


# ------------------------------------------------------------------ the sampler of a diffusion restorer

SIGMA_MAX = 14.6146            # the constant of the sampler: the largest level of the schedule, and the scale of the first noise
S_CHURN, S_NOISE, CFG_END, CFG_START, RESTORE_FLOOR = 5.0, 1.01, 4.0, 1.0, 0.05
TOY_STREAM, TOY_N, TOY_DELTA, TOY_Y, TOY_CENTER = "restoration.diffusion", 2000, 1.0, 0.5, 0.4
TOY_STEPS = 50


@lru_cache(maxsize=None)
def sdxl_sigmas():
    """sigma_t = sqrt((1 - abar_t) / abar_t) for t = 0 .. 999, abar_t the running product of 1 - beta_s, beta the square of a line from
    sqrt(0.00085) to sqrt(0.012) in 1000 equal steps (the scaled-linear schedule of SDXL)."""
    lo, hi = math.sqrt(0.00085), math.sqrt(0.012)
    out, ab = [], 1.0
    for i in range(1000):
        ab *= 1.0 - (lo + (hi - lo) * i / 999) ** 2
        out.append(math.sqrt((1.0 - ab) / ab))
    return out


def reference_grid(n):
    """n levels from the training grid at the indices floor(999 - 999 i / n), i = 0 .. n-1 (largest first), then 0."""
    sig = sdxl_sigmas()
    return [sig[math.floor(999 - 999 * i / n)] for i in range(n)] + [0.0]


def leading_grid(n):
    """n levels at the indices i (1000 // n) + 1, i = n-1 down to 0, then 0 (the spacing diffusers calls 'leading', offset 1)."""
    sig = sdxl_sigmas()
    return [sig[i * (1000 // n) + 1] for i in range(n - 1, -1, -1)] + [0.0]


def churn_gamma(n):
    return min(S_CHURN / (n - 1), math.sqrt(2) - 1)


def sampler_grids():
    sig = sdxl_sigmas()
    ref, lead = reference_grid(TOY_STEPS), leading_grid(TOY_STEPS)
    rows = [[f"sigma[{i}]", f"{sig[i]:.6f}"] for i in (0, 1, 20, 500, 981, 999)]
    t1 = _table(["training level", "value"], rows, "lr")
    rows2 = [[i, f"{ref[i]:.4f}", f"{lead[i]:.4f}"] for i in (0, 1, 2, 3, 47, 48, 49)] + [[50, f"{ref[50]:.4f}", f"{lead[50]:.4f}"]]
    t2 = _table(["step i", "equal-spaced grid", "diffusers 'leading' grid"], rows2)
    return (
        "The 1000 noise levels of the SDXL training schedule, sigma_t = sqrt((1 - abar_t) / abar_t) with beta_t the square of a line from "
        "sqrt(0.00085) to sqrt(0.012) in 1000 equal steps, t = 0 .. 999 (abar_t the running product of 1 - beta_s), at a few indices.\n\n"
        + t1 + f"\n\nTwo ways to take {TOY_STEPS} of them, largest first with a final 0: the grid at the indices floor(999 - 999 i / 50), and the "
        "grid at the indices 20 i + 1 for i = 49 down to 0.\n\n" + t2
    )


def sampler_schedule():
    n = TOY_STEPS
    ref = reference_grid(n)
    gamma = churn_gamma(n)
    rows = []
    for i in (0, 1, 5, 15, 25, 35, 45, 48, 49):
        s, s_next = ref[i], ref[i + 1]
        sh = s * (1 + gamma)
        cfg = (CFG_START - CFG_END) * sh / SIGMA_MAX + CFG_END
        rows.append([i, f"{s:.4f}", f"{s_next:.4f}", f"{sh:.4f}", f"{cfg:.3f}", f"{(s / SIGMA_MAX) ** 1:.5f}", f"{(s / SIGMA_MAX) ** 4:.2e}",
                     "yes" if s_next > RESTORE_FLOOR else "no"])
    t = _table(["step i", "sigma", "next sigma", "sigma-hat", "guidance g", "(sigma / max)^1", "(sigma / max)^4", "restoration pull on"], rows)
    return (
        f"The numbers of the sampler for {n} steps on the equal-spaced grid, with churn {S_CHURN:g} (so gamma = min({S_CHURN:g} / {n - 1}, "
        f"sqrt(2) - 1) = {gamma:.6f}, sigma-hat = sigma (1 + gamma)), the guidance scale ramp g = ({CFG_START:g} - {CFG_END:g}) sigma-hat / "
        f"{SIGMA_MAX:g} + {CFG_END:g} (1 at the largest level, {CFG_END:g} at 0), the factor (sigma / {SIGMA_MAX:g})^r that scales the "
        f"restoration pull for r = 1 and r = 4, and whether the pull is applied (only while the next level exceeds {RESTORE_FLOOR}).\n\n" + t
    )


def toy_denoiser(x, sigma, y=TOY_Y, delta=TOY_DELTA):
    """The exact denoiser of the two-pixel world: the pair average is known to be y, and the half-difference t = (x0 - x1) / 2 is
    +delta or -delta with probability 1/2, so a noisy half-difference g = t + sigma n (n of variance 1/2) has posterior mean
    delta tanh(2 delta g / sigma^2). Returns (y + that, y - that)."""
    g = (x[0] - x[1]) / 2
    t = delta * math.tanh(2 * delta * g / (sigma * sigma))
    return y + t, y - t


def toy_sample(g, grid, r, s_churn=S_CHURN):
    """One run of the sampler: x = SIGMA_MAX times two `normal()` draws; then for each step: if gamma > 0, add s_noise sqrt(sigma-hat^2 -
    sigma^2) times two more `normal()` draws; denoise at sigma-hat; if r > 0 and the next level exceeds 0.05, move the estimate by
    -(estimate - (y + c, y - c)) (sigma / SIGMA_MAX)^r (sigma, not sigma-hat); then x <- x + (x - estimate)/sigma-hat * (next - sigma-hat)."""
    n = len(grid) - 1
    gamma = min(s_churn / (n - 1), math.sqrt(2) - 1)
    x = [SIGMA_MAX * g.normal(), SIGMA_MAX * g.normal()]
    for i in range(n):
        s, s_next = grid[i], grid[i + 1]
        sh = s * (gamma + 1)
        if gamma > 0:
            extra = S_NOISE * math.sqrt(max(sh * sh - s * s, 0.0))
            n0, n1 = g.normal(), g.normal()
            x = [x[0] + n0 * extra, x[1] + n1 * extra]
        d = toy_denoiser(x, sh)
        if r > 0 and s_next > RESTORE_FLOOR:
            ratio = (s / SIGMA_MAX) ** r
            d = tuple(di - (di - ref) * ratio for di, ref in zip(d, (TOY_Y + TOY_CENTER, TOY_Y - TOY_CENTER)))
        x = [xi + (xi - di) / sh * (s_next - sh) for xi, di in zip(x, d)]
    return x


@lru_cache(maxsize=None)
def toy_stats(r, s_churn=S_CHURN):
    grid = reference_grid(TOY_STEPS)
    g = bookrng.generator(TOY_STREAM)
    ts, dev = [], 0.0
    for _ in range(TOY_N):
        x = toy_sample(g, grid, r, s_churn)
        ts.append((x[0] - x[1]) / 2)
        dev = max(dev, abs((x[0] + x[1]) / 2 - TOY_Y))
    mean_abs = sum(abs(t) for t in ts) / TOY_N
    mean_sq = sum(t * t for t in ts) / TOY_N
    return {"pos": sum(t > 0 for t in ts) / TOY_N, "mean": sum(ts) / TOY_N, "abs": mean_abs, "sq": mean_sq, "mse": TOY_DELTA ** 2 + mean_sq,
            "dev": dev}


def diffusion_toy():
    rows = [["posterior mean (what squared error picks)", "-", "0.0000", f"{TOY_DELTA ** 2:.4f}"]]
    configs = (("sampler, no churn, no pull", -1, 0.0), ("sampler, churn 5, no pull", -1, S_CHURN), ("... pull r = 0.25", 0.25, S_CHURN),
               ("... pull r = 0.5", 0.5, S_CHURN), ("... pull r = 1", 1, S_CHURN), ("... pull r = 2", 2, S_CHURN), ("... pull r = 4", 4, S_CHURN))
    for label, r, sc in configs:
        st = toy_stats(r, sc)
        rows.append([label, f"{st['pos'] * 100:.1f}%", f"{st['abs']:.4f}", f"{st['mse']:.4f}"])
    worst = max(toy_stats(r, sc)["dev"] for _, r, sc in configs)
    return (
        "The sampler of the previous table run on a world of two pixels: the pair average is known to be 0.5, the half-difference t of "
        "the true pair is +1 or -1 with probability 1/2, and the denoiser is the exact posterior mean. The reference that the pull aims at "
        f"is the pair (0.9, 0.1), whose half-difference is +{TOY_CENTER:g}. {TOY_N:,} runs of {TOY_STEPS} steps on the equal-spaced grid, every "
        f"configuration from the stream {_seed(TOY_STREAM)} read from its start (two draws for the first latent, then two per step when "
        "the churn is on), so the first latents are the same in every row. Share of the runs whose half-difference ends positive, the mean "
        "of its absolute value, and the mean squared error per pixel against a true pair that is +1 or -1 with equal probability "
        "independently of the run (1 plus the mean square of the half-difference). The pair average of every run ends within "
        f"{'1e-12' if worst < 1e-12 else format(worst, '.0e')} of 0.5.\n\n"
        + _table(["restorer", "t positive", "mean |t|", "error per pixel"], rows)
    )


# ------------------------------------------------------------------ brain's three restorers, counted from their configurations

def conv_params(cin, cout, k=3):
    return cin * cout * k * k + cout


def conv_flops(cin, cout, k, hw):
    """Two operations per multiply-add; bias additions are not counted, nor are normalisations, activations and softmax."""
    return 2 * cin * cout * k * k * hw


class Tally:
    """Parameters, tensors (a weight and a bias are two) and operations of the parts of a network, by part name."""

    def __init__(self):
        self.parts = {}

    def add(self, part, params=0, tensors=0, flops=0):
        p = self.parts.setdefault(part, [0, 0, 0])
        p[0] += params
        p[1] += tensors
        p[2] += flops

    def get(self, part):
        return tuple(self.parts[part])

    def total(self):
        return tuple(sum(p[i] for p in self.parts.values()) for i in range(3))


def rrdbnet_tally(num_feat=64, grow=32, blocks=23, scale=4, h=128, w=128, c_in=3, c_out=3):
    """RRDBNet: conv_first, `blocks` residual-in-residual dense blocks (3 dense blocks of 5 convolutions whose inputs have
    num_feat + (c - 1) grow channels for c = 1 .. 5, the fifth returning num_feat), conv_body, one 'nearest doubling then convolution'
    per factor of 2 of the scale, conv_hr and conv_last; all 3 x 3 convolutions with bias."""
    t = Tally()
    hw = h * w
    t.add("conv_first", conv_params(c_in, num_feat), 2, conv_flops(c_in, num_feat, 3, hw))
    for _ in range(blocks):
        for _ in range(3):
            for c in range(1, 6):
                cin = num_feat + (c - 1) * grow
                cout = num_feat if c == 5 else grow
                t.add("trunk (RRDB blocks)", conv_params(cin, cout), 2, conv_flops(cin, cout, 3, hw))
    t.add("conv_body", conv_params(num_feat, num_feat), 2, conv_flops(num_feat, num_feat, 3, hw))
    ups = scale.bit_length() - 1
    for i in range(1, ups + 1):
        t.add(f"conv_up{i}", conv_params(num_feat, num_feat), 2, conv_flops(num_feat, num_feat, 3, hw * 4 ** i))
    t.add("conv_hr", conv_params(num_feat, num_feat), 2, conv_flops(num_feat, num_feat, 3, hw * scale * scale))
    t.add("conv_last", conv_params(num_feat, c_out), 2, conv_flops(num_feat, c_out, 3, hw * scale * scale))
    return t


def rrdbnet_counts():
    t = rrdbnet_tally()
    tot_p, tot_t, tot_f = t.total()
    rows = [[name, _int(p), f, f"{fl / 1e9:,.1f}", f"{fl / tot_f * 100:.1f}%"] for name, (p, f, fl) in t.parts.items()]
    rows.append(["total", _int(tot_p), tot_t, f"{tot_f / 1e9:,.1f}", "100.0%"])
    t1 = _table(["part", "parameters", "tensors", "GFLOP", "share of the operations"], rows)
    rows2 = []
    for name, nb in (("23 blocks", 23), ("6 blocks", 6)):
        tt = rrdbnet_tally(blocks=nb)
        p, f, fl = tt.total()
        rows2.append([name, _int(p), f, f"{fl / 1e9:,.1f}", f"{fl / (512 * 512) / 1e3:,.1f}"])
    t2 = _table(["configuration", "parameters", "tensors", "GFLOP for 128 x 128", "kilo-FLOP per output pixel"], rows2)
    dense = 3 * sum(conv_params(64 + (c - 1) * 32, 64 if c == 5 else 32) for c in range(1, 6))
    return (
        "Real-ESRGAN's RRDBNet (64 channels, growth 32, 23 blocks, scale 4) counted from its configuration for a 128 by 128 input "
        "and a 512 by 512 output. A multiply-add is two operations; bias additions, activations and the concatenations are not counted.\n\n"
        + t1 + f"\n\nOne RRDB block has {_int(dense)} parameters (three dense blocks of {_int(dense // 3)}). The 6-block configuration "
        "keeps every other number of the architecture:\n\n" + t2
    )


def vqgan_tally(nf=64, mult=(1, 2, 2, 4, 4, 8), nres=2, dec_res=2, img=512, attn_res=(16,), emb=256, codes=1024):
    """The VQ autoencoder of CodeFormer: encoder, codebook, generator. A residual block has GroupNorm, 3 x 3 convolution,
    GroupNorm, 3 x 3 convolution and, when the width changes, a 1 x 1 convolution on the shortcut; an attention block has GroupNorm
    and four 1 x 1 convolutions (query, key, value, output) and one head of width c; a downsampling is a stride-2 3 x 3 convolution
    and an upsampling a nearest doubling followed by a 3 x 3 convolution. Attention operations: 4 hw^2 c (scores and weighted sum)."""
    t = Tally()

    def res(part, cin, cout, hw):
        t.add(part, 2 * cin, 2)
        t.add(part, conv_params(cin, cout), 2, conv_flops(cin, cout, 3, hw))
        t.add(part, 2 * cout, 2)
        t.add(part, conv_params(cout, cout), 2, conv_flops(cout, cout, 3, hw))
        if cin != cout:
            t.add(part, conv_params(cin, cout, 1), 2, conv_flops(cin, cout, 1, hw))

    def attn(part, c, hw):
        t.add(part, 2 * c, 2)
        t.add(part, 4 * conv_params(c, c, 1), 8, 4 * conv_flops(c, c, 1, hw) + 4 * hw * hw * c)

    n = len(mult)
    r = img
    t.add("encoder", conv_params(3, nf), 2, conv_flops(3, nf, 3, r * r))
    cin = nf
    for i in range(n):
        cout = nf * mult[i]
        for _ in range(nres):
            res("encoder", cin, cout, r * r)
            cin = cout
            if r in attn_res:
                attn("encoder", cin, r * r)
        if i != n - 1:
            r //= 2
            t.add("encoder", conv_params(cin, cin), 2, conv_flops(cin, cin, 3, r * r))
    res("encoder", cin, cin, r * r)
    attn("encoder", cin, r * r)
    res("encoder", cin, cin, r * r)
    t.add("encoder", 2 * cin, 2)
    t.add("encoder", conv_params(cin, emb), 2, conv_flops(cin, emb, 3, r * r))
    t.add("codebook", codes * emb, 1)
    t.add("generator", conv_params(emb, cin), 2, conv_flops(emb, cin, 3, r * r))
    res("generator", cin, cin, r * r)
    attn("generator", cin, r * r)
    res("generator", cin, cin, r * r)
    for i in reversed(range(n)):
        cout = nf * mult[i]
        for _ in range(dec_res):
            res("generator", cin, cout, r * r)
            cin = cout
            if r in attn_res:
                attn("generator", cin, r * r)
        if i != 0:
            r *= 2
            t.add("generator", conv_params(cin, cin), 2, conv_flops(cin, cin, 3, r * r))
    t.add("generator", 2 * cin, 2)
    t.add("generator", conv_params(cin, 3), 2, conv_flops(cin, 3, 3, r * r))
    return t


def codeformer_tally():
    t = vqgan_tally()
    e, mlp, tok, heads, layers, codes, emb = 512, 1024, 256, 8, 9, 1024, 256
    t.add("code predictor", tok * e, 1)                                                     # position_emb
    t.add("code predictor", emb * e + e, 2, 2 * tok * emb * e)                              # feat_emb
    for _ in range(layers):
        t.add("code predictor", 2 * e * e + 2 * e + e * e + e, 4, 2 * tok * e * 2 * e + 2 * tok * e * e)   # q|k and v projections
        t.add("code predictor", 0, 0, 4 * tok * tok * e)                    # scores and weighted sum, all 8 heads
        t.add("code predictor", e * e + e, 2, 2 * tok * e * e)                              # output projection
        t.add("code predictor", e * mlp + mlp + mlp * e + e, 4, 4 * tok * e * mlp)          # the two MLP layers
        t.add("code predictor", 4 * e, 4)                                                   # two layer norms
    t.add("code predictor", 2 * e + e * codes, 3, 2 * tok * e * codes)                      # final norm and the head without bias
    for size, c in ((32, 256), (64, 256), (128, 128), (256, 128)):
        hw = size * size
        p = 2 * (2 * c) + conv_params(2 * c, c) + 2 * c + conv_params(c, c) + conv_params(2 * c, c, 1)    # ResBlock(2c -> c)
        f = conv_flops(2 * c, c, 3, hw) + conv_flops(c, c, 3, hw) + conv_flops(2 * c, c, 1, hw)
        p += 4 * conv_params(c, c)                                                          # two towers of two 3 x 3 convolutions
        f += 4 * conv_flops(c, c, 3, hw)
        t.add("feature transformation (4 taps)", p, 10 + 8, f)
    return t


def codeformer_counts():
    t = codeformer_tally()
    order = ["encoder", "codebook", "code predictor", "generator", "feature transformation (4 taps)"]
    tot_p, tot_t, tot_f = t.total()
    rows = [[n, _int(t.get(n)[0]), t.get(n)[1], f"{t.get(n)[2] / 1e9:,.1f}", f"{t.get(n)[2] / tot_f * 100:.1f}%"] for n in order]
    rows.append(["total", _int(tot_p), tot_t, f"{tot_f / 1e9:,.1f}", "100.0%"])
    vq = vqgan_tally().total()
    return (
        "CodeFormer counted from its configuration for a 512 by 512 face: the VQ autoencoder (width 64, multipliers 1, 2, 2, 4, 4, 8, two "
        "residual blocks per level, attention at 16 by 16, a 1,024 by 256 codebook), the code predictor (width 512, 8 heads, 9 layers, feed-forward "
        "width 1,024, 256 positions, position embedding, a 256 to 512 input projection, a layer norm and a 512 to 1,024 head without bias), "
        "and the controllable feature transformation at four sizes (32, 64, 128, 256; 256, 256, 128, 128 channels). A multiply-add is two "
        "operations; normalisations, activations, softmax and the elementwise feature mix are not counted.\n\n"
        + _table(["part", "parameters", "tensors", "GFLOP", "share of the operations"], rows)
        + f"\n\nThe VQ autoencoder alone has {_int(vq[0])} parameters in {vq[1]} tensors; the rest is what CodeFormer adds. Stored in 32-bit "
        f"floats the parameters are {tot_p * 4 / 2 ** 20:,.1f} MiB."
    )


def adaptor_sites(h=128):
    """The 12 adaptors of SUPIR on the SDXL U-Net at a latent of h by h, derived from its skip stack: ('join', control channels, joined
    channels, tokens), the mid site, and ('cross', query width, context width, tokens), in the order in which the up path meets them
    (the control tensor of the join k from the top of the stack is entry 8 - k of the stack of 9)."""
    shapes = ci.sdxl_skip_shapes(h)             # 9 skip entries (finest first) and the middle block's output
    chans = da.SDXL_UNET["ch"]
    out = []
    prev = chans[-1]
    k = 0
    for i in range(3):
        level = 2 - i
        for j in range(3):
            c, hh, ww = shapes[8 - k]
            h_ori = prev if j == 0 else chans[level]
            out.append(("join", c, h_ori + c, hh * ww))
            prev = chans[level]
            k += 1
            if j == 2 and i < 2:
                out.append(("cross", chans[level], c, hh * ww))
    c, hh, ww = shapes[9]
    out.insert(0, ("mid", c, c, hh * ww))
    return out


def adaptor_tally(h=128):
    t = Tally()
    for kind, c, c2, tok in adaptor_sites(h):
        if kind in ("join", "mid"):
            p = conv_params(c, 128) + 2 * c2 + 2 * conv_params(128, c2) + conv_params(c, c, 1)
            f = conv_flops(c, 128, 3, tok) + 2 * conv_flops(128, c2, 3, tok) + conv_flops(c, c, 1, tok)
            t.add("ZeroSFT (9 joins and the middle)", p, 10, f)
        else:
            x, ctx = c, c2
            p = 2 * x + 2 * ctx + x * x + 2 * x * ctx + x * x + x
            f = 2 * tok * x * x + 2 * tok * ctx * 2 * x + 4 * tok * tok * x + 2 * tok * x * x
            t.add("ZeroCrossAttn (2 sites)", p, 8, f)
    return t


def vae_encoder_tally(h=1024, widths=(128, 256, 512, 512), layers=2, latent=4):
    t = Tally()
    r = h
    t.add("denoise encoder", conv_params(3, widths[0]), 2, conv_flops(3, widths[0], 3, r * r))
    cin = widths[0]

    def res(cin, cout, hw):
        t.add("denoise encoder", 2 * cin + conv_params(cin, cout) + 2 * cout + conv_params(cout, cout), 8,
              conv_flops(cin, cout, 3, hw) + conv_flops(cout, cout, 3, hw))
        if cin != cout:
            t.add("denoise encoder", conv_params(cin, cout, 1), 2, conv_flops(cin, cout, 1, hw))

    for i, cout in enumerate(widths):
        for _ in range(layers):
            res(cin, cout, r * r)
            cin = cout
        if i + 1 < len(widths):
            r //= 2
            t.add("denoise encoder", conv_params(cin, cin), 2, conv_flops(cin, cin, 3, r * r))
    res(cin, cin, r * r)
    t.add("denoise encoder", 2 * cin + 4 * conv_params(cin, cin, 1), 10, 4 * conv_flops(cin, cin, 1, r * r) + 4 * (r * r) ** 2 * cin)
    res(cin, cin, r * r)
    t.add("denoise encoder", 2 * cin + conv_params(cin, 2 * latent), 4, conv_flops(cin, 2 * latent, 3, r * r))
    return t


def supir_counts():
    n_full, _ = da.unet_count(da.SDXL_UNET, 128, 128)
    half, _ = da.unet_count(da.SDXL_UNET, 128, 128, through_mid=True)
    hint_p, hint_f = conv_params(4, 320), conv_flops(4, 320, 3, 128 * 128)
    trunk = (half.total_params() + hint_p, half.tensors + 2, half.total_flops() + hint_f)
    ad = adaptor_tally()
    adp, adt, adf = ad.total()
    enc = vae_encoder_tally().total()
    backbone = (n_full.total_params(), n_full.tensors, n_full.total_flops())
    per_eval = backbone[2] + trunk[2] + adf
    rows = [
        ["frozen SDXL U-Net (the backbone)", _int(backbone[0]), backbone[1], f"{backbone[2] / 1e9:,.1f}"],
        ["control trunk (GLVControl)", _int(trunk[0]), trunk[1], f"{trunk[2] / 1e9:,.1f}"],
        ["12 adaptors", _int(adp), adt, f"{adf / 1e9:,.1f}"],
        ["one denoiser evaluation", _int(backbone[0] + trunk[0] + adp), backbone[1] + trunk[1] + adt, f"{per_eval / 1e9:,.1f}"],
        ["degradation-robust encoder (once, at 1024 x 1024)", _int(enc[0]), enc[1], f"{enc[2] / 1e9:,.1f}"],
    ]
    sites = adaptor_sites()
    srows = []
    for kind, c, c2, tok in sites:
        if kind == "cross":
            heads = c // 64
            srows.append([kind, f"{c} (query), {c2} (context)", _int(tok), f"{heads}", _int(heads * tok * tok * 4)])
    delta_p = trunk[0] + adp + enc[0]
    all_p = delta_p + backbone[0]
    return (
        "SUPIR counted from its configuration, at a 128 by 128 latent (a 1024 by 1024 image). The backbone and the trunk are the SDXL U-Net "
        "of `denoiser_architectures` (the trunk cut after its middle block, plus the 4 to 320 convolution that embeds the hint). Multiply-adds "
        "count two operations; normalisations, activations and softmax do not count.\n\n"
        + _table(["part", "parameters", "tensors", "GFLOP"], rows)
        + f"\n\nThe parts that SUPIR adds to SDXL (trunk, adaptors and encoder) have {_int(delta_p)} parameters in {trunk[1] + adt + enc[1]} tensors; "
        f"with the backbone the total is {_int(all_p)} parameters, {all_p * 4 / 1e9:.2f} GB in 32-bit floats. A restoration of 50 steps with "
        f"a guidance branch runs the denoiser 100 times: {_tf(100 * per_eval / 1e12)}, against {_tf(2 * enc[2] / 1e12)} for the two "
        "encoder passes of the pipeline (the hint encoder and the re-encoding of the first-stage picture).\n\n"
        "The two ZeroCrossAttn sites attend between a level of the up path and the control tensor of the same size, and materialise a "
        "score matrix of heads x tokens x tokens (32-bit floats):\n\n"
        + _table(["site", "widths", "tokens", "heads", "bytes of the scores"], srows, "llrrr")
    )


def work_per_pixel():
    r = rrdbnet_tally().total()[2]
    c = codeformer_tally().total()[2]
    n_full, _ = da.unet_count(da.SDXL_UNET, 128, 128)
    half, _ = da.unet_count(da.SDXL_UNET, 128, 128, through_mid=True)
    sup = n_full.total_flops() + half.total_flops() + conv_flops(4, 320, 3, 128 * 128) + adaptor_tally().total()[2]
    rows = [
        ["Real-ESRGAN (RRDBNet)", "128 x 128 in, 512 x 512 out", 1, f"{r / 1e9:,.1f}", f"{r / (512 * 512) / 1e3:,.1f}"],
        ["CodeFormer", "512 x 512 in and out", 1, f"{c / 1e9:,.1f}", f"{c / (512 * 512) / 1e3:,.1f}"],
        ["SUPIR, 50 steps with guidance", "1024 x 1024 in and out", 100, f"{100 * sup / 1e9:,.1f}", f"{100 * sup / (1024 * 1024) / 1e3:,.1f}"],
    ]
    return (
        "The operations of one restoration by each of brain's three restorers (the network passes only: SUPIR's two encoder passes and "
        "its decodes are left out), and per pixel of the restored picture.\n\n"
        + _table(["restorer", "picture", "network passes", "GFLOP", "kilo-FLOP per output pixel"], rows, "llrrr")
    )
