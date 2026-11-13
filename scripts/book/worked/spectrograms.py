# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Spectrograms and the Mel Scale".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: spectrograms.<name> -->`
markers; the functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the
table beside it cannot disagree.

* a sound and the same sound played backwards, whose whole-signal spectra are equal in magnitude, and the short-time transform that
  tells them apart;
* windows: the periodic and symmetric Hann windows, the leakage of a tone between two bins under three windows, and the
  overlap-add sums that only one of the two Hann windows keeps constant;
* padding and framing: frame counts and frame positions under the three conventions in use, and the first and last frames of a
  six-sample signal under reflect and constant padding, including the four-sample frame that brain's own test pins by hand;
* the mel scale in its HTK and Slaney forms, the filter edges they give, a six-filter bank written out weight by weight, the
  difference between triangles that are linear in hertz and triangles that are linear in mel, the effect of area normalisation on a
  flat spectrum, and how many bins the lowest filters of real configurations contain;
* log compression: five conventions for the floor and the clamp on the same mel energies, what a gain does to each, and how one
  click moves the floor of the dynamic-range convention;
* one stated clip (a chirp, a steady tone and a little noise from the book's generator) pushed through variants of one front end
  that differ in a single convention, and through full re-implementations of two speech-recognition front ends;
* the table of front ends that one engine configures, and the work of computing them.

Arithmetic is the language's 64-bit floating point (correctly rounded library sin, cos, sqrt, log, log10, exp). Every discrete
Fourier transform is the definition, X[k] = sum over n = 0, 1, ..., N-1 in this order of x[n] times exp(-2 pi i k n / N), with the
products of one frame summed left to right; the printed digits are far above the rounding of that order. Mel energies are sums over
the bins b = 0, 1, ... in index order. The only random numbers are the 8000 noise draws of the clip, described below.

Swedish Embedded AB implements solutions for turning raw audio into the features a speech model was trained on, exactly. If your
team needs expertise in audio front ends for recognition, synthesis and speaker models, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math

import bookrng
import fourier as fo

PI = math.pi
_table = fo._table
_g = fo._g


def _f(v, digits=4):
    """A fixed-point number without a negative zero."""
    text = f"{v:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


# ------------------------------------------------------------------------------------------------ transform and windows

_TRIG = {}


def _trig(n):
    """cos(2 pi j / n) and sin(2 pi j / n) for j = 0 .. n-1."""
    if n not in _TRIG:
        _TRIG[n] = ([math.cos(2 * PI * j / n) for j in range(n)], [math.sin(2 * PI * j / n) for j in range(n)])
    return _TRIG[n]


def dft(x):
    """The DFT of a real list by the definition, as complex numbers for every bin k = 0 .. N-1."""
    n = len(x)
    c, s = _trig(n)
    return [complex(sum(x[j] * c[(k * j) % n] for j in range(n)), -sum(x[j] * s[(k * j) % n] for j in range(n))) for k in range(n)]


def power_bins(x):
    """|X[k]|^2 for k = 0 .. N/2 of a real list."""
    n = len(x)
    c, s = _trig(n)
    out = []
    for k in range(n // 2 + 1):
        re = im = 0.0
        for j in range(n):
            i = (k * j) % n
            re += x[j] * c[i]
            im -= x[j] * s[i]
        out.append(re * re + im * im)
    return out


def hann(n, periodic=True):
    """0.5 - 0.5 cos(2 pi i / d) with d = n (periodic) or n - 1 (symmetric)."""
    d = n if periodic else n - 1
    return [0.5 - 0.5 * math.cos(2 * PI * i / d) for i in range(n)]


def povey(n):
    """The symmetric Hann window raised to the power 0.85."""
    return [h ** 0.85 for h in hann(n, False)]


def reflect_pad(x, p):
    """p samples on each side, mirrored about the end samples without repeating them: out[p-1-i] = x[i+1], out[p+L+i] = x[L-2-i];
    an index outside the signal reads 0."""
    n = len(x)
    out = [0.0] * (n + 2 * p)
    out[p:p + n] = x
    for i in range(p):
        out[p - 1 - i] = x[i + 1] if i + 1 < n else 0.0
        out[p + n + i] = x[n - 2 - i] if n - 2 - i >= 0 else 0.0
    return out


def zero_pad(x, p):
    return [0.0] * p + list(x) + [0.0] * p


def frame_count(length, n, hop):
    return 1 + (length - n) // hop if length >= n else 0


def stft_power(padded, n, hop, window):
    """The power of every frame of an already padded signal: frame t is padded[t*hop : t*hop + n] times the window."""
    return [power_bins([padded[t * hop + i] * window[i] for i in range(n)]) for t in range(frame_count(len(padded), n, hop))]


# ------------------------------------------------------------------------------------------------ the mel scale

SLANEY_SP = 200.0 / 3.0  # hertz per mel below the knee
SLANEY_KNEE = 1000.0
SLANEY_KNEE_MEL = SLANEY_KNEE / SLANEY_SP  # 15
SLANEY_STEP = math.log(6.4) / 27.0  # log-steps per mel above the knee


def htk_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def htk_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def slaney_mel(f):
    return f / SLANEY_SP if f < SLANEY_KNEE else SLANEY_KNEE_MEL + math.log(f / SLANEY_KNEE) / SLANEY_STEP


def slaney_hz(m):
    return SLANEY_SP * m if m < SLANEY_KNEE_MEL else SLANEY_KNEE * math.exp(SLANEY_STEP * (m - SLANEY_KNEE_MEL))


SCALES = {"htk": (htk_mel, htk_hz), "slaney": (slaney_mel, slaney_hz)}


def band_edges(n_mels, fmin, fmax, scale):
    """n_mels + 2 frequencies, equally spaced on the mel scale from fmin to fmax, as hertz."""
    to_mel, to_hz = SCALES[scale]
    lo, hi = to_mel(fmin), to_mel(fmax)
    return [to_hz(lo + (hi - lo) * i / (n_mels + 1)) for i in range(n_mels + 2)]


def filterbank(rate, n_fft, n_mels, fmin, fmax, scale, norm, shape="hz"):
    """Rows of weights over the bins b = 0 .. n_fft/2, at bin frequencies b*rate/n_fft. shape 'hz': each triangle is linear in
    hertz between its edges; shape 'mel': linear in the mel value of the bin. norm multiplies a row by 2/(right - left)."""
    to_mel = SCALES[scale][0]
    e = band_edges(n_mels, fmin, fmax, scale)
    rows = []
    for m in range(n_mels):
        left, centre, right = e[m], e[m + 1], e[m + 2]
        row = []
        for b in range(n_fft // 2 + 1):
            f = b * rate / n_fft
            if shape == "hz":
                up, down = (f - left) / (centre - left), (right - f) / (right - centre)
            else:
                ml, mc, mr, mf = to_mel(left), to_mel(centre), to_mel(right), to_mel(f)
                up, down = (mf - ml) / (mc - ml), (mr - mf) / (mr - mc)
            w = max(0.0, min(up, down))
            row.append(w * 2.0 / (right - left) if norm else w)
        rows.append(row)
    return rows


def mel_energies(power, rows):
    """For each frame and each filter, the sum over b = 0, 1, ... of weight times power."""
    return [[sum(w * p for w, p in zip(row, frame)) for row in rows] for frame in power]


# ------------------------------------------------------------------------------------------------ the clip

CLIP_NAME = "spectrograms.clip"
CLIP_RATE = 16000
CLIP_LENGTH = 8000


def clip_samples(length=CLIP_LENGTH, rate=CLIP_RATE):
    """x[n] = 0.5 sin(2 pi (200 t + 3200 t^2)) + 0.1 sin(2 pi 6000 t) + 0.01 (2 u_n - 1), t = n / rate, u_n the n-th draw
    (n = 0, 1, ...) of bookrng.generator(CLIP_NAME).uniform()."""
    rng = bookrng.generator(CLIP_NAME)
    out = []
    for n in range(length):
        t = n / rate
        out.append(0.5 * math.sin(2 * PI * (200 * t + 3200 * t * t)) + 0.1 * math.sin(2 * PI * 6000 * t) + 0.01 * (2 * rng.uniform() - 1))
    return out


_CLIP = []


def clip():
    if not _CLIP:
        _CLIP.extend(clip_samples())
    return _CLIP


def clip_check():
    x = clip()
    rows = [(n, _f(x[n], 6)) for n in (0, 1, 2, 3, 4000, 7999)]
    return (f"The clip: {CLIP_LENGTH} samples at {CLIP_RATE} Hz, seed {bookrng.seed_of(CLIP_NAME)}, one `uniform()` draw per sample in "
            "order of n. Check values:\n\n" + _table(["n", "x[n]"], rows))


# ------------------------------------------------------------------------------------------------ the opening


def _two_tones(n=64, rate=8000):
    return [math.cos(2 * PI * (1000 if j < n // 2 else 2000) * j / rate) for j in range(n)]


def reversal_spectra():
    x = _two_tones()
    y = x[::-1]
    n = len(x)
    xs, ys = dft(x), dft(y)
    gap = max(abs(abs(a) - abs(b)) for a, b in zip(xs, ys))
    assert gap < 1e-12
    ks = [5, 7, 8, 9, 11, 13, 15, 16, 17, 19]
    rows = [(k, _f(k * 8000 / n, 0), _f(2 * abs(xs[k]) / n), _f(2 * abs(ys[k]) / n)) for k in ks]
    return ("x[n] = cos(2 pi 1000 n / 8000) for n = 0 .. 31 and cos(2 pi 2000 n / 8000) for n = 32 .. 63, and y[n] = x[63 - n]. "
            "Amplitude estimate 2 abs(X[k]) / 64 of the 64-point DFT of each.\n\n"
            + _table(["bin k", "Hz", "x, 1000 Hz then 2000 Hz", "y, 2000 Hz then 1000 Hz"], rows)
            + "\n\nLargest difference of abs(X[k]) and abs(Y[k]) over all 64 bins: below 1e-12.")


def reversal_frames():
    x = _two_tones()
    y = x[::-1]
    rows = []
    for t in range(5):
        px, py = power_bins(x[8 * t:8 * t + 32]), power_bins(y[8 * t:8 * t + 32])
        rows.append((t, 8 * t, _f(px[4], 1), _f(px[8], 1), _f(py[4], 1), _f(py[8], 1)))
    return ("The same two signals cut into frames of N = 32 samples at hop 8, rectangular window. With N = 32 at 8000 Hz a bin is "
            "250 Hz wide, so 1000 Hz is bin 4 and 2000 Hz is bin 8. Power abs(X_t[k])^2 in those two bins.\n\n"
            + _table(["frame t", "first sample", "x, bin 4", "x, bin 8", "y, bin 4", "y, bin 8"], rows))


def stft_frames():
    x = _two_tones()
    ks = list(range(2, 11))
    rows = []
    for t in range(5):
        p = power_bins(x[8 * t:8 * t + 32])
        rows.append([t] + [_f(p[k], 1) for k in ks])
    return ("Power abs(X_t[k])^2 of x (1000 Hz then 2000 Hz) for frames of N = 32 samples at hop H = 8, rectangular window, "
            "bins k = 2 .. 10 of 0 .. 16.\n\n"
            + _table(["t"] + [f"k={k}" for k in ks], rows))


# ------------------------------------------------------------------------------------------------ windows


def window_shapes():
    n = 8
    cols = [("rectangular", [1.0] * n), ("Hann periodic", hann(n)), ("Hann symmetric", hann(n, False)), ("symmetric Hann ^ 0.85", povey(n))]
    rows = [[i] + [_f(w[i]) for _, w in cols] for i in range(n)]
    return ("Windows of N = 8 samples.\n\n" + _table(["i"] + [c for c, _ in cols], rows))


def window_leakage():
    n = 32
    x = [math.cos(2 * PI * 4.5 * j / n) for j in range(n)]
    wins = [("rectangular", [1.0] * n), ("Hann periodic", hann(n)), ("Hann symmetric", hann(n, False))]
    spectra = []
    for _, w in wins:
        p = power_bins([a * b for a, b in zip(x, w)])
        top = max(p)
        spectra.append([10 * math.log10(v / top) for v in p])
    rows = [[k] + [_f(s[k], 1) for s in spectra] for k in range(1, 16)]
    return ("Power in decibels, 10 log10(P[k] / P_max), of a cosine with 4.5 cycles in N = 32 samples, bins 1 .. 15 (bins 0 and 16 "
            "are omitted: for the periodic Hann window they are zero up to rounding).\n\n"
            + _table(["k"] + [c for c, _ in wins], rows))


def window_overlap():
    n = 16
    length = 6 * n
    rows = []
    for hop in (8, 4, 2):
        cells = []
        for periodic in (True, False):
            w = hann(n, periodic)
            total = [0.0] * length
            for start in range(0, length - n + 1, hop):
                for i in range(n):
                    total[start + i] += w[i]
            inner = total[n:length - n]
            cells.append(f"{_f(min(inner))} to {_f(max(inner))}")
        rows.append((hop, cells[0], cells[1]))
    return ("The sum over frames of the window at hop H, N = 16, for the samples that every frame position covers (positions 16 .. "
            "79 of a signal of 96).\n\n"
            + _table(["hop H", "periodic Hann", "symmetric Hann"], rows, align="rll"))


# ------------------------------------------------------------------------------------------------ padding and frames


def _centre_pad(conv, n, hop):
    return {"none": 0, "centre": n // 2, "shifted": (n - hop) // 2}[conv]


def frame_counts():
    n, hop = 400, 160
    rows = []
    for length in (159, 160, 399, 400, 16000, 16001, 16160):
        cells = []
        for conv in ("none", "centre", "shifted"):
            p = _centre_pad(conv, n, hop)
            count = frame_count(length + 2 * p, n, hop)
            closed = {"none": frame_count(length, n, hop), "centre": 1 + length // hop, "shifted": length // hop if length >= hop else 0}[conv]
            assert count == closed
            cells.append(count)
        rows.append((f"{length:,}", *cells))
    centres = [("no padding", "160 t + 200"), ("pad N/2 = 200 each side", "160 t"), ("pad (N - H)/2 = 120 each side", "160 t + 80")]
    return ("Frames of N = 400 samples at hop H = 160 for a signal of L samples, counted from the length of the padded signal as "
            "1 + floor((padded length - N) / H) (0 if it is shorter than N).\n\n"
            + _table(["L", "no padding", "pad 200", "pad 120"], rows)
            + "\n\nCentre of frame t, in samples of the original signal, for N = 400, H = 160:\n\n"
            + _table(["padding", "centre of frame t"], centres, align="ll"))


def padding_by_hand():
    n, hop = 4, 2
    w = hann(n)
    p = reflect_pad([1.0, 3.0], 1)
    first = power_bins([a * b for a, b in zip(p, w)])
    return ("Signal (1, 3), N = 4, H = 2, periodic Hann window "
            f"({', '.join(_f(v, 1) for v in w)}), padded by (N - H)/2 = 1 on each side by reflection: the padded signal is "
            f"({', '.join(_f(v, 0) for v in p)}), one frame, whose power in bins 0, 1, 2 is "
            f"{', '.join(_f(v, 1) for v in first)}.")


def padding_edges():
    x = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    n, hop = 4, 2
    w = hann(n)
    rows = []
    for name, padded in (("reflect", reflect_pad(x, n // 2)), ("constant 0", zero_pad(x, n // 2))):
        count = frame_count(len(padded), n, hop)
        for t in (0, count - 1):
            seg = padded[t * hop:t * hop + n]
            p = power_bins([a * b for a, b in zip(seg, w)])
            rows.append((name, t, "(" + ", ".join(_f(v, 0) for v in seg) + ")", *[_f(v, 2) for v in p]))
    return ("Signal (1, 3, 5, 2, 4, 6), N = 4, H = 2, pad N/2 = 2 on each side, periodic Hann window (0, 0.5, 1, 0.5); 4 frames, "
            "of which the first and the last are listed (the two middle frames are the same under both paddings).\n\n"
            + _table(["padding", "frame t", "samples", "P[0]", "P[1]", "P[2]"], rows, align="lrlrrr"))


# ------------------------------------------------------------------------------------------------ the scales


def mel_scales():
    rows = []
    for f in (0, 100, 200, 500, 1000, 2000, 4000, 8000):
        rows.append((f, _f(htk_mel(f), 2), _f(slaney_mel(f), 2)))
    for f in (0, 100, 1000, 2000, 8000):
        assert abs(htk_hz(htk_mel(f)) - f) < 1e-9 and abs(slaney_hz(slaney_mel(f)) - f) < 1e-9
    return ("The two mel scales at the same frequencies. HTK: 2595 log10(1 + f/700). Slaney: f / (200/3) below 1000 Hz, and "
            "15 + ln(f / 1000) / (ln(6.4)/27) from 1000 Hz up. Each inverse returns f to within 1e-9.\n\n"
            + _table(["f (Hz)", "HTK mel", "Slaney mel"], rows))


def mel_edges():
    h, s = band_edges(8, 0, 8000, "htk"), band_edges(8, 0, 8000, "slaney")
    rows = [(i, _f(h[i], 1), _f(s[i], 1)) for i in range(10)]
    return ("The 10 band edges of an 8-filter bank from 0 to 8000 Hz, equally spaced in mel, as hertz. Filter m uses edges m, m+1, "
            "m+2.\n\n" + _table(["edge i", "HTK (Hz)", "Slaney (Hz)"], rows))


def filter_widths_data(n_mels=40, fmax=8000):
    """(centre, width) in hertz of each filter for both scales, for the figure."""
    out = {}
    for scale in ("htk", "slaney"):
        e = band_edges(n_mels, 0, fmax, scale)
        out[scale] = [(e[m + 1], e[m + 2] - e[m]) for m in range(n_mels)]
    return out


_SMALL = dict(rate=8000, n_fft=32, n_mels=6, fmin=0.0, fmax=4000.0)


def filterbank_small():
    e = band_edges(6, 0, 4000, "slaney")
    plain = filterbank(8000, 32, 6, 0, 4000, "slaney", False)
    rows = []
    for m in range(6):
        weights = ", ".join(f"{b}: {_f(w, 4)}" for b, w in enumerate(plain[m]) if w > 0)
        rows.append((m, _f(e[m], 1), _f(e[m + 1], 1), _f(e[m + 2], 1), weights, _f(2 / (e[m + 2] - e[m]), 5)))
    return ("A filterbank for 8000 Hz audio with N = 32 (17 bins, 250 Hz apart), 6 filters from 0 to 4000 Hz, Slaney scale. Left, "
            "centre and right edges in Hz; the nonzero weights of the unnormalized triangle as bin: weight; and the area factor "
            "2 / (right - left) in 1/Hz.\n\n"
            + _table(["m", "left", "centre", "right", "nonzero weights (bin: weight)", "2/(right-left)"], rows, align="rrrrlr"))


def white_spectrum():
    plain = filterbank(8000, 32, 6, 0, 4000, "slaney", False)
    normal = filterbank(8000, 32, 6, 0, 4000, "slaney", True)
    flat = [1.0] * 17
    a = [sum(w * p for w, p in zip(r, flat)) for r in plain]
    b = [sum(w * p for w, p in zip(r, flat)) for r in normal]
    return ("Mel energy of a flat power spectrum (P[b] = 1 for every bin) through the bank above, filter by filter.\n\n"
            + _table(["m", "unit-height triangles", "area-normalized triangles"], [(m, _f(a[m], 4), _f(b[m], 5)) for m in range(6)]))


def shape_difference():
    hz = filterbank(8000, 32, 6, 0, 4000, "htk", False, "hz")
    ml = filterbank(8000, 32, 6, 0, 4000, "htk", False, "mel")
    e = band_edges(6, 0, 4000, "htk")
    rows = [(b, _f(b * 250.0, 0), _f(hz[3][b], 4), _f(ml[3][b], 4)) for b in range(17) if hz[3][b] > 0 or ml[3][b] > 0]
    return (f"Filter 3 of a 6-filter HTK bank over 0 to 4000 Hz (N = 32, 8000 Hz): edges {', '.join(_f(v, 1) for v in e[3:6])} Hz. "
            "The weight of each bin under a triangle that is linear in hertz and under one that is linear in the mel value of the "
            "bin; both reach 1 at the centre frequency.\n\n"
            + _table(["bin b", "Hz", "linear in Hz", "linear in mel"], rows))


_BANKS = [("40-filter bank, 16 kHz, N = 400", 16000, 400, 40, 8000.0),
          ("128 filters, 16 kHz, N = 400", 16000, 400, 128, 8000.0),
          ("128 filters, 16 kHz, N = 512", 16000, 512, 128, 8000.0),
          ("128 filters, 24 kHz, N = 1024, to 12 kHz", 24000, 1024, 128, 12000.0),
          ("80 filters, 24 kHz, N = 1920, to 8 kHz", 24000, 1920, 80, 8000.0)]


def low_band():
    rows = []
    for name, rate, n_fft, n_mels, fmax in _BANKS:
        rows_w = filterbank(rate, n_fft, n_mels, 0, fmax, "slaney", True)
        e = band_edges(n_mels, 0, fmax, "slaney")
        # No bin may sit within 1e-6 Hz of an inner edge, so that the count of nonzero weights does not depend on rounding.
        gap = min(abs(b * rate / n_fft - v) for b in range(n_fft // 2 + 1) for v in e[1:-1])
        assert gap > 1e-6
        nz = [sum(1 for w in r if w > 0) for r in rows_w]
        rows.append((name, _f(rate / n_fft, 2), _f(e[2] - e[0], 1), sum(1 for v in nz if v == 0), sum(1 for v in nz if v == 1),
                     sum(1 for v in nz if v <= 2), sum(nz), n_mels * (n_fft // 2 + 1)))
    # The filters of the 128-filter bank for N = 400 whose right edge lies at or below 1000 Hz, and the bins they read between them.
    bank = filterbank(16000, 400, 128, 0, 8000, "slaney", True)
    e = band_edges(128, 0, 8000, "slaney")
    low = [m for m in range(128) if e[m + 2] <= 1000.0]
    shared = {b for m in low for b, w in enumerate(bank[m]) if w > 0}
    return ("Slaney scale, area-normalized triangles. Width of the first filter (right edge minus left edge), the number of "
            "filters with no, with exactly one and with at most two nonzero weights, and the nonzero weights of the whole bank "
            "against its size.\n\n"
            + _table(["bank", "bin spacing (Hz)", "first filter width (Hz)", "empty", "one bin", "at most two", "nonzero weights", "all weights"],
                     rows, align="lrrrrrrr")
            + f"\n\nIn the 128-filter bank with N = 400, the {len(low)} filters whose right edge is at or below 1000 Hz read "
            f"{len(shared)} different bins between them.")


# ------------------------------------------------------------------------------------------------ log compression

GUARD_LN = 1e-6
GUARD_NEMOTRON = 2.0 ** -24
FLOOR_MAGNITUDE = 1e-5
FLOOR_SINGLE = 2.0 ** -23
CLAMP_LOG10 = 1e-10
RANGE_DB = 8.0


def ln_guard(e):
    return math.log(e + GUARD_LN)


def ln_guard_small(e):
    return math.log(e + GUARD_NEMOTRON)


def ln_floor(e):
    return math.log(max(e, FLOOR_MAGNITUDE))


def ln_floor_single(e):
    return math.log(max(e, FLOOR_SINGLE))


def range_compress(e, peak_log10):
    """(max(log10(max(e, 1e-10)), peak - 8) + 4) / 4, where peak is the largest log10 energy of the whole clip."""
    return (max(math.log10(max(e, CLAMP_LOG10)), peak_log10 - RANGE_DB) + 4.0) / 4.0


def compression_table():
    peak = 2.0
    rows = []
    for e in (0.0, 1e-9, 1e-6, 1e-3, 1.0, 100.0):
        rows.append((_g(e, 3) if e else "0", _f(ln_guard(e)), _f(ln_guard_small(e)), _f(ln_floor(e)), _f(ln_floor_single(e)),
                     _f(range_compress(e, peak))))
    return ("Five compressions of the same mel energy E. A: ln(E + 1e-6). B: ln(E + 2^-24). C: ln(max(E, 1e-5)). D: ln(max(E, "
            "2^-23)). E: (max(log10(max(E, 1e-10)), g - 8) + 4) / 4 with g = 2, the largest log10 energy of the clip (the cell with "
            "E = 100).\n\n"
            + _table(["energy E", "A", "B", "C", "D", "E"], rows))


def gain_table():
    rows = []
    for e in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        # a clip whose loudest cell is 1 (g = 0) is made 20 dB louder: every energy times 100 and g = 2
        rows.append((_g(e, 3), _f(ln_guard(100 * e) - ln_guard(e)), _f(ln_guard_small(100 * e) - ln_guard_small(e)),
                     _f(ln_floor(100 * e) - ln_floor(e)), _f(ln_floor_single(100 * e) - ln_floor_single(e)),
                     _f(range_compress(100 * e, 2.0) - range_compress(e, 0.0))))
    return ("The change of each output when the signal is made 10 times larger (every energy times 100, ln 100 = 4.6052; for E "
            "the clip's largest log10 energy goes from g = 0 to g = 2).\n\n"
            + _table(["energy E", "A", "B", "C", "D", "E"], rows))


def click_floor():
    cells = [-4, -3, -2, -1, 0, 1]
    rows = []
    for v in cells:
        quiet = range_compress(10.0 ** v, 1.0)
        loud = range_compress(10.0 ** v, 6.0)
        rows.append((v, _f(quiet, 3), _f(loud, 3)))
    return ("Convention E on six cells with log10 energies -4 .. 1. Left: the clip as it is, largest log10 energy g = 1. Right: "
            "the same cells in a clip that also holds one loud click whose cell has log10 energy 6 (g = 6, floor g - 8 = -2).\n\n"
            + _table(["log10 of the energy", "output, g = 1", "output, g = 6"], rows))


# ------------------------------------------------------------------------------------------------ one front end, one change


def _log_mel(power, rows, compress):
    return [[compress(e) for e in frame] for frame in mel_energies(power, rows)]


def _preemphasis(x, a):
    return [x[0]] + [x[i] - a * x[i - 1] for i in range(1, len(x))]


def variants():
    x = clip()
    n, hop, n_mels = 400, 160, 40
    per, sym = hann(n), hann(n, False)
    base_p = stft_power(reflect_pad(x, n // 2), n, hop, per)
    slaney = filterbank(16000, n, n_mels, 0, 8000, "slaney", True)
    htk_plain = filterbank(16000, n, n_mels, 0, 8000, "htk", False)
    ln = lambda e: math.log(max(e, CLAMP_LOG10))  # noqa: E731
    base = _log_mel(base_p, slaney, ln)

    def compare(other):
        """Frames, cells that differ by more than 0.01, largest and mean absolute difference over the cells both have."""
        count = min(len(base), len(other))
        diffs = [abs(base[t][m] - other[t][m]) for t in range(count) for m in range(n_mels)]
        assert all(abs(d - 0.01) > 1e-6 for d in diffs)  # no cell sits on the counting threshold, so the count is rounding-proof
        return count, sum(1 for d in diffs if d > 0.01), max(diffs), sum(diffs) / len(diffs)

    def edge_only(other):
        """The largest difference over the first two and the last two frames, the four that the padding touches."""
        diffs = [abs(base[t][m] - other[t][m]) for t in (0, 1, len(base) - 2, len(base) - 1) for m in range(n_mels)]
        return max(diffs)

    rows = []
    out = _log_mel(stft_power(reflect_pad(x, n // 2), n, hop, sym), slaney, ln)
    rows.append(("symmetric Hann window", len(out), *compare(out)[1:]))
    out = _log_mel(stft_power(zero_pad(x, n // 2), n, hop, per), slaney, ln)
    rows.append(("constant zero padding", len(out), *compare(out)[1:]))
    out = _log_mel(base_p, htk_plain, ln)
    rows.append(("HTK scale, unit-height triangles", len(out), *compare(out)[1:]))
    mag = [[math.sqrt(p) for p in frame] for frame in base_p]
    out = _log_mel(mag, slaney, ln)
    rows.append(("magnitude in place of power", len(out), *compare(out)[1:]))
    out = _log_mel(stft_power(reflect_pad(_preemphasis(x, 0.97), n // 2), n, hop, per), slaney, ln)
    rows.append(("pre-emphasis 0.97", len(out), *compare(out)[1:]))
    out = _log_mel(stft_power(reflect_pad(x, (n - hop) // 2), n, hop, per), slaney, ln)
    rows.append(("pad 120 instead of 200", len(out), *compare(out)[1:]))
    body = [(name, frames, cells, _f(mx, 3), _f(mean, 4)) for name, frames, cells, mx, mean in rows]
    padded_edge = edge_only(_log_mel(stft_power(zero_pad(x, n // 2), n, hop, per), slaney, ln))
    return ("The clip through one baseline front end and through six variants that each change one thing. Baseline: N = 400, "
            "H = 160, periodic Hann window, reflect padding of 200 on each side (51 frames), 40 Slaney filters with area "
            "normalization from 0 to 8000 Hz, power, ln(max(E, 1e-10)). Pre-emphasis 0.97 means y[0] = x[0] and "
            "y[n] = x[n] - 0.97 x[n-1] for n >= 1 on the whole clip, before the padding. Over the frames both versions have "
            "(all 40 cells of every frame, so 40 x frames cells), each row counts the cells that differ from the baseline by "
            "more than 0.01 and gives the largest and the mean absolute difference over all of those 40 x frames cells, in nats.\n\n"
            + _table(["change", "frames", "cells differing by > 0.01", "largest difference", "mean difference"], body, align="lrrrr")
            + f"\n\nWith constant padding the largest difference over the first two and the last two frames, the four that the padding touches, is {_f(padded_edge, 3)}.")


# ------------------------------------------------------------------------------------------------ two recogniser front ends


def qwen_style(samples, target):
    """N = 400, H = 160, periodic Hann, reflect pad 200, 128 Slaney filters (area-normalized) 0 .. 8000 Hz on 16000 Hz, power,
    log10(max(E, 1e-10)), floor at the clip maximum - 8, (v + 4) / 4; the last frame is dropped. Returns [mel][frame]."""
    x = list(samples) + [0.0] * (target - len(samples))
    power = stft_power(reflect_pad(x, 200), 400, 160, hann(400))
    power = power[:-1]
    e = mel_energies(power, filterbank(16000, 400, 128, 0, 8000, "slaney", True))
    logs = [[math.log10(max(v, CLAMP_LOG10)) for v in frame] for frame in e]
    peak = max(max(f) for f in logs)
    cells = [[(max(v, peak - RANGE_DB) + 4.0) / 4.0 for v in f] for f in logs]
    return [[cells[t][m] for t in range(len(cells))] for m in range(128)]


def nemotron_style(samples):
    """Pre-emphasis 0.97, N = 512, H = 160, a symmetric Hann window of 400 centred in the 512 (offset 56), constant pad 256,
    128 Slaney filters 0 .. 8000 Hz on 16000 Hz, power, ln(E + 2^-24); frames from floor(L / 160) on are zero. [frame][mel]."""
    y = _preemphasis(samples, 0.97)
    window = [0.0] * 56 + hann(400, False) + [0.0] * 56
    power = stft_power(zero_pad(y, 256), 512, 160, window)
    rows = filterbank(16000, 512, 128, 0, 8000, "slaney", True)
    valid = len(samples) // 160
    e = mel_energies(power[:valid], rows)
    cells = [[math.log(v + GUARD_NEMOTRON) for v in f] for f in e]
    return cells + [[0.0] * 128 for _ in range(len(power) - valid)]


def recogniser_outputs():
    x = clip()
    q = qwen_style(x, 16000)
    qs = qwen_style([0.0] * 1600, 16000)
    n = nemotron_style(x)
    ns = nemotron_style([0.0] * 1600)
    q_cells = [v for row in q for v in row]
    valid = len(x) // 160
    n_cells = [v for f in n[:valid] for v in f]
    rows = [("Qwen3-ASR style", f"[128, {len(q[0])}]", _f(min(q_cells), 3), _f(max(q_cells), 3), _f(qs[0][0], 3)),
            ("Nemotron style", f"[{len(n)}, 128], last row zero", _f(min(n_cells), 3), _f(max(n_cells), 3), _f(ns[0][0], 3))]
    return ("The half-second clip through both front ends (the first padded with zeros to one whole second, as that front end does), "
            "128 filters each: output shape, smallest and largest value over the cells the encoder reads (for the first, the "
            "100 frames of the padded second; for the second, the 50 valid frames), and the value in any cell of a tenth of a second of digital "
            "silence. The Nemotron style applies the pre-emphasis y[0] = x[0], y[n] = x[n] - 0.97 x[n-1] to the whole clip "
            "first, then pads 256 zeros on each side, then cuts frames of 512 whose weights are zero except samples 56 .. 455, "
            "which carry the 400-sample symmetric Hann window (centred: 56 zeros on each side of it); the frames from "
            "floor(L / 160) on, with L the clip length, are set to zero.\n\n"
            + _table(["front end", "shape", "smallest", "largest", "value for silence"], rows, align="llrrr"))


# ------------------------------------------------------------------------------------------------ configurations and work

FRONT_ENDS = [
    # key, label, rate, n_fft, hop, n_mels, fmin, fmax, window, padding, pre-processing, mel scale and shape, spectrum, compression
    dict(label="qwen3asr", rate=16000, n_fft=400, hop=160, n_mels=128, fmin=0, fmax=8000, bank=("slaney", True, "hz"),
         window="periodic Hann 400", padding="reflect 200, last frame dropped", pre="none",
         mel="Slaney, area-normalized", spectrum="power", compression="log10, clamp 1e-10, max - 8, (v + 4)/4"),
    dict(label="nemotronasr", rate=16000, n_fft=512, hop=160, n_mels=128, fmin=0, fmax=8000, bank=("slaney", True, "hz"),
         window="symmetric Hann 400 centred in 512", padding="zeros 256", pre="pre-emphasis 0.97",
         mel="Slaney, area-normalized", spectrum="power", compression="ln(E + 2^-24)"),
    dict(label="ecapatdnn", rate=24000, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000, bank=("slaney", True, "hz"),
         window="periodic Hann 1024", padding="reflect 384", pre="none",
         mel="Slaney, area-normalized", spectrum="magnitude sqrt(p + 1e-9)", compression="ln(max(E, 1e-5))"),
    dict(label="cosyvoice prompt", rate=24000, n_fft=1920, hop=480, n_mels=80, fmin=0, fmax=8000, bank=("slaney", True, "hz"),
         window="periodic Hann 1920", padding="reflect 720", pre="none",
         mel="Slaney, area-normalized", spectrum="magnitude sqrt(p + 1e-9)", compression="ln(max(E, 1e-5))"),
    dict(label="ltxv audio", rate=16000, n_fft=1024, hop=160, n_mels=64, fmin=0, fmax=8000, bank=("slaney", True, "hz"),
         window="periodic Hann 1024", padding="reflect 512", pre="none",
         mel="Slaney, area-normalized", spectrum="magnitude", compression="ln(max(E, 1e-5))"),
    dict(label="x-vector filterbank", rate=16000, n_fft=512, hop=160, n_mels=80, fmin=20, fmax=8000, bank=("htk", False, "mel"),
         window="symmetric Hann 400, power 0.85, in 512", padding="none (frames that fit)",
         pre="mean removed per frame, pre-emphasis 0.97",
         mel="1127 ln(1 + f/700), unit height, linear in mel", spectrum="power", compression="ln(max(E, 2^-23))"),
    dict(label="evaluation", rate=24000, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000, bank=("slaney", True, "hz"),
         window="periodic Hann 1024", padding="reflect 512", pre="none",
         mel="Slaney, area-normalized", spectrum="power", compression="ln(E + 1e-6)"),
]


def front_end_geometry():
    rows = []
    for fe in FRONT_ENDS:
        fps = fe["rate"] / fe["hop"]
        rows.append((fe["label"], f"{fe['rate']:,}", f"{fe['n_fft']} / {fe['hop']}", fe["n_mels"], f"{fe['fmin']:,} to {fe['fmax']:,}",
                     _f(1000 * fe["hop"] / fe["rate"], 2), _f(fps, 2), f"{fps * fe['n_mels']:,.0f}",
                     _f(fps * fe["n_mels"] / fe["rate"], 3)))
    return ("Frame geometry of the front ends, and the number of values one second of audio becomes: frames per second rate / H "
            "times filters, against the samples per second.\n\n"
            + _table(["front end", "rate (Hz)", "N / H", "filters", "range (Hz)", "hop (milliseconds)", "frames per second", "values per second", "values per sample"],
                     rows, align="lrrrrrrrr"))


def front_end_signal_path():
    rows = [(fe["label"], fe["window"], fe["padding"], fe["pre"]) for fe in FRONT_ENDS]
    return _table(["front end", "window", "padding", "before the transform"], rows, align="llll")


def front_end_features():
    rows = [(fe["label"], fe["mel"], fe["spectrum"], fe["compression"]) for fe in FRONT_ENDS]
    return _table(["front end", "mel filters", "spectrum", "compression"], rows, align="llll")


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _fft_flops(n):
    """Real operations of one complex FFT of length n: 5 n log2 n when n is a power of two (a butterfly is one complex multiply,
    6 operations, and two complex additions, 4), and for any other n the chirp method: 3 FFTs of length m = the next power of two
    of 2n - 1 and (2n + m) complex multiplications."""
    if n & (n - 1) == 0:
        return 5 * n * int(math.log2(n))
    m = _next_pow2(2 * n - 1)
    return 3 * 5 * m * int(math.log2(m)) + 6 * (2 * n + m)


def work_counts():
    rows = []
    for fe in FRONT_ENDS:
        n_fft, n_mels = fe["n_fft"], fe["n_mels"]
        bins = n_fft // 2 + 1
        scale, norm, shape = fe["bank"]
        fft = _fft_flops(n_fft)
        power = 3 * bins
        dense = 2 * n_mels * bins
        nz = sum(1 for r in filterbank(fe["rate"], n_fft, n_mels, fe["fmin"], fe["fmax"], scale, norm, shape) for w in r if w > 0)
        sparse = 2 * nz
        fps = fe["rate"] / fe["hop"]
        rows.append((fe["label"], f"{fft:,}", f"{dense:,}", f"{sparse:,}", _f((fft + power + dense) * fps / 1e6, 1),
                     _f((fft + power + sparse) * fps / 1e6, 1)))
    return ("Real operations per frame and per second of audio. FFT: 5 N log2 N for a power of two, otherwise the chirp method of "
            "three FFTs of length m (the next power of two above 2N - 1) and 2N + m complex multiplications of 6 operations. Power: "
            "3 per bin. Mel: 2 per weight, over all weights (dense) or over the nonzero weights only (sparse); the last two columns "
            "add the FFT, the power and the mel step, times frames per second.\n\n"
            + _table(["front end", "FFT per frame", "mel dense per frame", "mel sparse per frame", "millions of operations per second, dense", "millions of operations per second, sparse"],
                     rows, align="lrrrrr"))


# ------------------------------------------------------------------------------------------------ figures

FIG_NAME = "spectrograms.figure_clip"
FIG_RATE, FIG_LENGTH, FIG_N, FIG_HOP, FIG_MELS = 8000, 3200, 128, 80, 24


def figure_signal():
    """A chirp 100 Hz to 3500 Hz over 0.4 s (phase 2 pi (100 t + 4250 t^2)), amplitude 0.5, plus 0.01 (2u - 1) noise from
    bookrng.generator(FIG_NAME), one uniform draw per sample."""
    rng = bookrng.generator(FIG_NAME)
    out = []
    for n in range(FIG_LENGTH):
        t = n / FIG_RATE
        out.append(0.5 * math.sin(2 * PI * (100 * t + 4250 * t * t)) + 0.01 * (2 * rng.uniform() - 1))
    return out


def figure_spectrograms():
    """Decibels below the largest cell (floor -80) of the linear-frequency power spectrogram [frame][bin] and of the mel
    spectrogram [frame][filter], for the same frames."""
    power = stft_power(reflect_pad(figure_signal(), FIG_N // 2), FIG_N, FIG_HOP, hann(FIG_N))
    mel = mel_energies(power, filterbank(FIG_RATE, FIG_N, FIG_MELS, 0, FIG_RATE / 2, "slaney", True))

    def to_db(frames):
        top = max(max(f) for f in frames)
        return [[max(-80.0, 10 * math.log10(max(v, 1e-30) / top)) for v in f] for f in frames]

    return to_db(power), to_db(mel)


def figure_triangles(n_mels=10, fmax=4000.0, scale="slaney"):
    """Weights of the area-normalized triangles over a fine grid of frequencies (N = 1024 at 8000 Hz) for the figure."""
    rate, n_fft = 8000, 1024
    rows = filterbank(rate, n_fft, n_mels, 0, fmax, scale, True)
    freqs = [b * rate / n_fft for b in range(n_fft // 2 + 1)]
    return freqs, rows
