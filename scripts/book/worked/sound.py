# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Sound and Sampling".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: sound.<name> -->` markers; the
functions with arguments (or a leading underscore) supply the numbers of the chapter's figures, so a picture and the table beside
it cannot disagree.

* a tone and the tone it becomes when every third sample is kept, the folding rule for any frequency, and the tone at exactly half
  the sampling rate;
* the sampling theorem as a computation: the samples of a band-limited window, evaluated between the sample times by the inverse
  DFT, and the kernel that the evaluation uses;
* 16-bit PCM: quantisation noise measured against its formula, the asymmetric write and read of one 16-bit convention, the integer
  and float sample formats, and the bytes of a stereo WAV file;
* resampling between 16 kHz and 24 kHz as one weighted sum over the input samples, with three kernels (linear interpolation, a
  Kaiser-tapered sinc, a cos^2-tapered sinc), measured on stated tones and counted in multiply-adds.

Arithmetic is the language's 64-bit floating point (correctly rounded library sin, cos, sqrt, log10) unless a block says that
32-bit values are emulated by rounding a 64-bit result to the nearest 32-bit number (`images.r32`). Sums are accumulated left to
right in index order. No example uses random numbers.

Swedish Embedded AB implements solutions for getting recorded sound into models without changing what it says. If your team needs
expertise in audio front ends and sample-rate conversion for speech and music models, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
import struct

import fourier as fo
import images as im

PI = math.pi

_table = fo._table
_g = fo._g


# ------------------------------------------------------------------------------------------------ rounding helpers


def round_half_away(x):
    """The integer nearest to x, ties away from zero (the rule of the pseudocode's round(x, 0))."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def f4(v, digits=4):
    """A fixed-point number without a negative zero."""
    text = f"{v:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def small(v, tol=1e-12):
    """A difference printed as 'below tol' when it is, so that a last digit that depends on the library's sin and cos is not shown."""
    return f"below {tol:.0e}".replace("e-0", "e-").replace("e-", "e-") if v < tol else f"{v:.1e}"


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


# ------------------------------------------------------------------------------------------------ aliasing

FS16 = 16000.0


def alias_of(f, fs):
    """The frequency in [0, fs/2] whose cosine has the same samples as the cosine at f, and the sign of the matching sine."""
    r = f % fs
    return (r, 1) if r <= fs / 2 else (fs - r, -1)


def alias_samples():
    """A 10 kHz tone sampled at 48 kHz and kept at every third sample, beside the same tone sampled at 16 kHz and a 6 kHz tone."""
    rows = []
    for n in range(12):
        kept = math.cos(2 * PI * 10000 * (3 * n) / 48000)
        direct = math.cos(2 * PI * 10000 * n / 16000)
        six = math.cos(2 * PI * 6000 * n / 16000)
        rows.append((n, 3 * n, f4(kept), f4(direct), f4(six)))
    worst = max(abs(math.cos(2 * PI * 10000 * (3 * n) / 48000) - math.cos(2 * PI * 6000 * n / 16000)) for n in range(1600))
    return ("x(t) = cos(2 pi 10000 t). Column 3: x sampled at 48 kHz, sample 3n kept. Column 4: x sampled at 16 kHz. Column 5: "
            "cos(2 pi 6000 t) sampled at 16 kHz.\n\n"
            + _table(["n", "48 kHz index 3n", "10 kHz, every third of 48 kHz", "10 kHz at 16 kHz", "6 kHz at 16 kHz"], rows)
            + f"\n\nLargest difference between columns 3 and 5 over the first 1600 values of n: {small(worst, 1e-9)}.")


def alias_waves(points=240, samples=9):
    """Data of the first figure: the 10 kHz and 6 kHz cosines over 0.5 ms and the 16 kHz sample times with their value."""
    ts = [0.5e-3 * k / (points - 1) for k in range(points)]
    ten = [math.cos(2 * PI * 10000 * t) for t in ts]
    six = [math.cos(2 * PI * 6000 * t) for t in ts]
    st = [n / FS16 for n in range(samples)]
    sv = [math.cos(2 * PI * 10000 * t) for t in st]
    return ts, ten, six, st, sv


def alias_table():
    fs = FS16
    freqs = [1000, 4000, 7500, 8000, 8500, 10000, 12000, 15000, 16000, 17000, 20000, 25000]
    rows = []
    for f in freqs:
        a, sgn = alias_of(f, fs)
        worst = max(abs(math.cos(2 * PI * f * n / fs) - math.cos(2 * PI * a * n / fs)) for n in range(32))
        sin_worst = max(abs(math.sin(2 * PI * f * n / fs) - sgn * math.sin(2 * PI * a * n / fs)) for n in range(32))
        rows.append((f, _g(f % fs), _g(a), f"{sgn:+d}", small(worst), small(sin_worst)))
    return ("Sampling rate f_s = 16000 Hz. r = f mod f_s; the apparent frequency is r if r <= f_s / 2 and f_s - r otherwise; the "
            "sine at f equals s times the sine at the apparent frequency, with s = -1 in the second case. The last two columns are "
            "the largest difference over n = 0 .. 31 between the samples of the tone at f and the predicted samples.\n\n"
            + _table(["f (Hz)", "r = f mod f_s", "apparent frequency (Hz)", "s", "cosine, largest difference",
                      "sine, largest difference"], rows))


def alias_curve(fs=FS16, top=40000.0):
    """Data of the second figure: apparent frequency against true frequency, with the corners of the folding."""
    corners = [0.0]
    k = 1
    while k * fs / 2 < top:
        corners.append(k * fs / 2)
        k += 1
    corners.append(top)
    return corners, [alias_of(f, fs)[0] for f in corners]


def nyquist_edge():
    rows = []
    for name, phi in [("0", 0.0), ("pi / 4", PI / 4), ("pi / 2", PI / 2), ("3 pi / 4", 3 * PI / 4)]:
        s = [math.cos(PI * n + phi) for n in range(8)]  # a cosine at fs/2 with phase phi, sampled at n / fs
        rows.append((name, " ".join(f4(v) for v in s[:6]), f4(math.cos(phi))))
    return ("A cosine at exactly f_s / 2, cos(2 pi (f_s / 2) n / f_s + phi) = cos(pi n + phi), for four phases.\n\n"
            + _table(["phi", "samples n = 0 .. 5", "cos(phi), the factor of (-1)^n"], rows, "lll"))


# ------------------------------------------------------------------------------------------------ the sampling theorem

N_WIN = 32
TONES = [(1000.0, 1.0, "cos", 0.0), (3000.0, 0.5, "cos", 0.7), (6500.0, 0.25, "sin", 0.0)]
EXTRA = (8500.0, 0.5, "cos", 0.0)


def _tone_value(tone, t):
    f, a, kind, ph = tone
    return a * (math.cos(2 * PI * f * t + ph) if kind == "cos" else math.sin(2 * PI * f * t + ph))


def _signal(tones, t):
    s = 0.0
    for tone in tones:
        s += _tone_value(tone, t)
    return s


def interpolate(samples, u):
    """The window's inverse DFT evaluated at the (non-integer) sample position u, keeping only the frequencies |k| < N / 2:
    x(u) = (1/N) sum_{k=-M}^{M} X[k mod N] e^{+2 pi i k u / N}, with M = N/2 - 1. (The book's DFT has the exponent
    e^{-2 pi i k n / N}, the inverse the opposite sign.)"""
    n = len(samples)
    big_x = fo._dft(samples)
    m = n // 2 - 1
    acc = 0j
    for k in range(-m, m + 1):
        acc += big_x[k % n] * complex(math.cos(2 * PI * k * u / n), math.sin(2 * PI * k * u / n))
    return (acc / n).real


def reconstruction():
    fs = FS16
    clean = [_signal(TONES, n / fs) for n in range(N_WIN)]
    dirty = [_signal(TONES + [EXTRA], n / fs) for n in range(N_WIN)]
    rows = []
    for n in range(6):
        u = n + 0.5
        t = u / fs
        rows.append((n, f"{t * 1000:.4f}", f"{_signal(TONES, t):.6f}", f"{interpolate(clean, u):.6f}",
                     f"{_signal(TONES + [EXTRA], t):.6f}", f"{interpolate(dirty, u):.6f}",
                     f"{_signal(TONES, t) + 0.5 * math.cos(2 * PI * 7500 * t):.6f}"))
    err_clean = max(abs(_signal(TONES, (n + 0.5) / fs) - interpolate(clean, n + 0.5)) for n in range(N_WIN))
    same = max(abs(a - b) for a, b in zip(dirty, [_signal(TONES + [(7500.0, 0.5, "cos", 0.0)], n / fs) for n in range(N_WIN)]))
    return ("f_s = 16000 Hz, a window of N = 32 samples, x(t) = cos(2 pi 1000 t) + 0.5 cos(2 pi 3000 t + 0.7) + "
            "0.25 sin(2 pi 6500 t), evaluated at t = (n + 0.5) / f_s by the inverse DFT of its 32 samples with the frequencies "
            "|k| <= 15 (columns 3 and 4). Columns 5 and 6: the same with an extra 0.5 cos(2 pi 8500 t), which has k = 17 > 16. "
            "Column 7: the signal with that tone replaced by 0.5 cos(2 pi 7500 t), the tone it aliases to.\n\n"
            + _table(["n", "t (ms)", "x(t) exact", "x(t) from samples", "with 8500 Hz, exact", "with 8500 Hz, from samples",
                      "clean + 7500 Hz tone"], rows)
            + f"\n\nLargest difference between exact and reconstructed over the 32 half-sample points, clean signal: {small(err_clean)}. "
              f"Largest difference between the 32 samples of the signal with the 8500 Hz tone and the samples of the signal with "
              f"the 7500 Hz tone: {small(same)}.")


def dirichlet(n, u):
    """(1/N) sum_{k=-M}^{M} cos(2 pi k u / N) with M = N/2 - 1, summed as 1 + 2 sum_{k=1}^{M} (k ascending), divided by N."""
    m = n // 2 - 1
    acc = 1.0
    for k in range(1, m + 1):
        acc += 2 * math.cos(2 * PI * k * u / n)
    return acc / n


def sinc(x):
    return 1.0 if x == 0 else math.sin(PI * x) / (PI * x)


def interpolation_kernel():
    rows = []
    for n in (8, 16, 64, 1024):
        rows.append((n, *(f"{dirichlet(n, u):.6f}" for u in (0.0, 0.25, 0.5, 1.5, 2.5))))
    rows.append(("sinc(u)", *(f"{sinc(u):.6f}" for u in (0.0, 0.25, 0.5, 1.5, 2.5))))
    return ("The weight D_N(u) = (1/N) sum over |k| <= N/2 - 1 of cos(2 pi k u / N) that a sample at distance u (in samples) "
            "gets in the evaluation between samples, and sin(pi u) / (pi u).\n\n"
            + _table(["N", "u = 0", "u = 0.25", "u = 0.5", "u = 1.5", "u = 2.5"], rows))


# ------------------------------------------------------------------------------------------------ PCM and quantisation


def quantise(x, bits):
    """Round to the nearest multiple of delta = 2^(1 - bits), the codes -2^(bits-1) .. 2^(bits-1) - 1."""
    delta = 2.0 ** (1 - bits)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = clamp(round_half_away(x / delta), lo, hi)
    return q, delta * q


def quantisation_snr():
    n = 16000
    fs = 16000.0
    f = 997.0
    rows = []
    for bits, amp in [(8, 0.9), (12, 0.9), (16, 0.9), (16, 0.01), (16, 0.0001)]:
        delta = 2.0 ** (1 - bits)
        sig = 0.0
        noise = 0.0
        for i in range(n):
            x = amp * math.sin(2 * PI * f * i / fs)
            _, y = quantise(x, bits)
            sig += x * x
            noise += (y - x) * (y - x)
        measured = 10 * math.log10(sig / noise)
        predicted = 10 * math.log10((amp * amp / 2) / (delta * delta / 12))
        rows.append((bits, f"2^-{bits - 1}", amp, f"{measured:.2f}", f"{predicted:.2f}"))
    full = [(b, f"{6.0206 * b + 1.7609:.2f}") for b in (8, 12, 16)]
    return ("x[i] = A sin(2 pi 997 i / 16000), i = 0 .. 15999, rounded to the nearest multiple of Delta = 2^(1 - b) (ties away "
            "from zero). SNR = 10 log10(sum x^2 / sum (y - x)^2), sums in index order. Predicted: 10 log10((A^2 / 2) / "
            "(Delta^2 / 12)).\n\n"
            + _table(["bits b", "Delta", "amplitude A", "measured SNR (dB)", "predicted SNR (dB)"], rows)
            + "\n\nFull-scale sine (A = 1) predicted by 6.0206 b + 1.7609 dB: "
            + ", ".join(f"{b} bits {v}" for b, v in full) + ".")


def brain_write(x):
    """The 16-bit write of one common convention: clamp to [-1, 1], multiply by 32767, round to the nearest integer."""
    return round_half_away(clamp(x, -1.0, 1.0) * 32767.0)


def brain_read(code):
    """The matching read: the integer code divided by 32768."""
    return code / 32768.0


def write_read_convention():
    rows = []
    for x in (0.0, 0.5, -0.5, 0.3, 1e-5, 1.0, -1.0, 1.5):
        c = brain_write(x)
        y = brain_read(c)
        rows.append((f"{x:g}", c, f"{y:.10f}", f"{(y - clamp(x, -1.0, 1.0)) * 32768:+.4f}"))
    ends = [(-32768, brain_read(-32768)), (32767, brain_read(32767))]
    return ("Write: code = round(clamp(x, -1, 1) * 32767), ties away from zero. Read: value = code / 32768. The last column is "
            "the error of the round trip in units of 1/32768 (one least significant bit), measured from the clamped input.\n\n"
            + _table(["x", "code", "read back", "error (LSB)"], rows)
            + "\n\nReading the extreme codes: " + ", ".join(f"{c} gives {v:.12f}" for c, v in ends)
            + f". The largest possible round-trip error, at x = 1, is {abs((brain_read(brain_write(1.0)) - 1.0) * 32768):.0f} LSB; "
              f"for any |x| <= 1 it is at most (0.5 + |x|) LSB.")


def _f32_from_int(i):
    return im.r32(float(i))


def sample_formats():
    rows = []
    # (name, bytes, little-endian bytes, integer value, decoded)
    def le(value, nbytes, signed=True):
        return (value & ((1 << (8 * nbytes)) - 1)).to_bytes(nbytes, "little")

    def hexs(b):
        return " ".join(f"{v:02x}" for v in b)

    cases = []
    for byte in (0, 128, 255):
        cases.append(("8-bit unsigned", hexs(bytes([byte])), byte, (byte - 128.0) / 128.0))
    for value in (-32768, 0, 32767):
        cases.append(("16-bit signed", hexs(le(value, 2)), value, value / 32768.0))
    for value in (-8388608, 1, 8388607):
        cases.append(("24-bit signed", hexs(le(value, 3)), value, _f32_from_int(value) / 8388608.0))
    for value in (-2147483648, 2147483647):
        cases.append(("32-bit signed", hexs(le(value, 4)), value, _f32_from_int(value) / 2147483648.0))
    for value in (0.5, -1.0):
        cases.append(("32-bit float", hexs(struct.pack("<f", value)), "-", value))
    for name, b, code, val in cases:
        rows.append((name, b, code, f"{val:.9f}"))
    # sign extension of 24-bit values by arithmetic, no shifts
    raw = [0x7FFFFF, 0x800000, 0xFFFFFF]
    ext = [v - (1 << 24) if v >= (1 << 23) else v for v in raw]
    return ("Decoded values; the integer formats are converted to 32-bit float before the division (the 32-bit division by "
            "2^31 is exact, the conversion of the integer is rounded to nearest, ties to even).\n\n"
            + _table(["format", "bytes (little-endian)", "code", "decoded"], rows, "llrr")
            + "\n\nSign extension of a 24-bit code v read as an unsigned number: v - 2^24 if v >= 2^23 else v. "
            + "; ".join(f"0x{r:06X} gives {e}" for r, e in zip(raw, ext)) + ".")


def wav_layout():
    left = [1.0, -1.0, 0.0]
    right = [-1.0, 1.0, 0.0]
    rate = 8000
    chans = 2
    data = b""
    for i in range(3):
        for ch in (left, right):
            data += struct.pack("<h", brain_write(ch[i]))
    header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + struct.pack("<H", 1)
              + struct.pack("<H", chans) + struct.pack("<I", rate) + struct.pack("<I", rate * chans * 2)
              + struct.pack("<H", chans * 2) + struct.pack("<H", 16) + b"data" + struct.pack("<I", len(data)))
    fields = [
        (0, 4, "chunk id", "RIFF"),
        (4, 4, "size of everything after this field", f"36 + {len(data)} = {36 + len(data)}"),
        (8, 4, "form type", "WAVE"),
        (12, 4, "chunk id", "'fmt '"),
        (16, 4, "size of the fmt chunk", "16"),
        (20, 2, "format tag (1 = integer PCM, 3 = float)", "1"),
        (22, 2, "channels", str(chans)),
        (24, 4, "sample rate (Hz)", str(rate)),
        (28, 4, "byte rate = rate x block align", str(rate * chans * 2)),
        (32, 2, "block align = channels x bytes per sample", str(chans * 2)),
        (34, 2, "bits per sample", "16"),
        (36, 4, "chunk id", "'data'"),
        (40, 4, "size of the data", str(len(data))),
    ]
    rows = []
    for off, size, name, val in fields:
        rows.append((off, size, " ".join(f"{b:02x}" for b in header[off:off + size]), name, val))
    frame_rows = []
    for i in range(3):
        frame_rows.append((i, f"{left[i]:g}", f"{right[i]:g}", brain_write(left[i]), brain_write(right[i]),
                           " ".join(f"{b:02x}" for b in data[4 * i:4 * i + 4])))
    planes = [[brain_read(struct.unpack_from("<h", data, 4 * i + 2 * c)[0]) for i in range(3)] for c in range(2)]
    mono = [(planes[0][i] + planes[1][i]) / 2 for i in range(3)]
    return ("Two channels, three frames, 8000 Hz, each sample written as code = round(32767 x) with ties away from zero (the rule the section after this table justifies), two bytes little-endian. The file is "
            f"{len(header) + len(data)} bytes.\n\n"
            + _table(["offset", "bytes", "hex (little-endian)", "field", "value"], rows, "rrlll")
            + "\n\n"
            + _table(["frame", "left", "right", "left code", "right code", "4 bytes"], frame_rows, "rrrrrl")
            + f"\n\nRead back as planes: left {[round(v, 6) for v in planes[0]]}, right {[round(v, 6) for v in planes[1]]}; "
              f"the mean of the two channels is {[round(v, 6) for v in mono]}.")


def data_rates():
    rows = []
    for name, rate, ch, nbytes in [("telephone, 8-bit", 8000, 1, 1), ("speech models, 16-bit integer", 16000, 1, 2),
                                   ("speech models, 32-bit float", 16000, 1, 4), ("codec models, 16-bit integer", 24000, 1, 2),
                                   ("compact disc", 44100, 2, 2), ("studio, 32-bit float", 48000, 2, 4)]:
        per_sec = rate * ch * nbytes
        rows.append((name, rate, ch, nbytes, per_sec, per_sec * 60))
    return ("Bytes of raw PCM = sample rate x channels x bytes per sample (x seconds).\n\n"
            + _table(["use", "rate (Hz)", "channels", "bytes per sample", "bytes per second", "bytes per minute"], rows, "lrrrrr"))


# ------------------------------------------------------------------------------------------------ resampling

ATTEN_DB = 80.0
HALF_ZEROS = 32


def bessel_i0(x):
    """I0(x) = sum_k ((x/2)^k / k!)^2, the series summed until a term is below 1e-16 of the sum."""
    total = 1.0
    term = 1.0
    k = 1.0
    while term > total * 1e-16:
        term *= (x / (2.0 * k)) ** 2
        total += term
        k += 1.0
    return total


def kaiser_beta(atten_db):
    if atten_db > 50:
        return 0.1102 * (atten_db - 8.7)
    if atten_db >= 21:
        return 0.5842 * (atten_db - 21) ** 0.4 + 0.07886 * (atten_db - 21)
    return 0.0


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def reduced(old_rate, new_rate):
    g = gcd(old_rate, new_rate)
    return new_rate // g, old_rate // g  # up, down


def out_length(n, old_rate, new_rate):
    up, down = reduced(old_rate, new_rate)
    return -(-n * up // down)


def linear_kernel(d):
    return max(1.0 - abs(d), 0.0)


def resample_linear(x, old_rate, new_rate):
    """Output i at input position s = i * old / new, y = x[floor s] (1 - frac) + x[floor s + 1] frac, the last sample repeated
    beyond the end; the output has round(len * new / old) samples (ties away from zero)."""
    n = len(x)
    ratio = new_rate / old_rate
    out = []
    for i in range(round_half_away(n * ratio)):
        s = i / ratio
        i0 = math.floor(s)
        frac = s - i0
        a = x[min(i0, n - 1)]
        b = x[min(i0 + 1, n - 1)]
        out.append(a + (b - a) * frac)
    return out


class KaiserSinc:
    """The kernel h(d) = c sinc(c d) I0(beta sqrt(1 - (d / hw)^2)) / I0(beta), |d| <= hw input samples, of the design with
    c = min(1, up/down), beta from 80 dB and 32 zero crossings of the sinc at the intermediate rate; d = m / up for the integer
    m = j down - i up of an (output j, input i) pair, so that |m| <= reach = 32 max(up, down)."""

    def __init__(self, old_rate, new_rate):
        self.up, self.down = reduced(old_rate, new_rate)
        self.cut = min(1.0, self.up / self.down)
        self.beta = kaiser_beta(ATTEN_DB)
        self.reach = HALF_ZEROS * max(self.up, self.down)
        self.flen = 2 * self.reach + 1
        self.hw = self.reach / self.up
        self.norm = bessel_i0(self.beta)
        self._cache = {}

    def weight_of(self, m):
        w = self._cache.get(m)
        if w is None:
            d = m / self.up
            u = d / self.hw
            w = self._cache[m] = self.cut * sinc(self.cut * d) * bessel_i0(self.beta * math.sqrt(max(1 - u * u, 0.0))) / self.norm
        return w


class HannSinc:
    """The kernel h(d) = c sinc(c d) cos^2(pi c d / 12) for |c d| < 6 (and 0 beyond), with c = 0.99 min(1, new/old); d = m / up
    for the integer m = j down - i up, and reach = ceil(6 up / c) bounds the m with a nonzero weight."""

    def __init__(self, old_rate, new_rate):
        self.up, self.down = reduced(old_rate, new_rate)
        self.cut = 0.99 * min(1.0, self.up / self.down)
        self.reach = math.ceil(6 * self.up / self.cut)
        self.hw = 6 / self.cut
        self._cache = {}

    def weight_of(self, m):
        w = self._cache.get(m)
        if w is None:
            u = self.cut * m / self.up
            w = self._cache[m] = 0.0 if abs(u) >= 6 else self.cut * sinc(u) * math.cos(PI * u / 12) ** 2
        return w


def taps_of(kernel, j, n):
    """The inputs i in 0 .. n - 1 with |j down - i up| <= reach, ascending."""
    lo = max((j * kernel.down - kernel.reach + kernel.up - 1) // kernel.up, 0)
    hi = min((j * kernel.down + kernel.reach) // kernel.up, n - 1)
    return range(lo, hi + 1)


def resample_weighted(x, kernel, old_rate, new_rate):
    n = len(x)
    out = []
    for j in range(out_length(n, old_rate, new_rate)):
        acc = 0.0
        for i in taps_of(kernel, j, n):
            acc += kernel.weight_of(j * kernel.down - i * kernel.up) * x[i]
        out.append(acc)
    return out


class PlainSinc(KaiserSinc):
    """The Kaiser design's support and cutoff with no taper at all: h(d) = c sinc(c d) for |d| <= hw."""

    def weight_of(self, m):
        w = self._cache.get(m)
        if w is None:
            d = m / self.up
            w = self._cache[m] = self.cut * sinc(self.cut * d)
        return w


def resample_kaiser(x, old_rate, new_rate):
    return resample_weighted(x, KaiserSinc(old_rate, new_rate), old_rate, new_rate)


def resample_hann(x, old_rate, new_rate):
    return resample_weighted(x, HannSinc(old_rate, new_rate), old_rate, new_rate)


METHODS = [("linear interpolation", resample_linear), ("Kaiser-tapered sinc", resample_kaiser), ("cos^2-tapered sinc", resample_hann)]


def tone(freq, rate, n, amp=1.0):
    return [amp * math.cos(2 * PI * freq * i / rate) for i in range(n)]


def amplitude_at(y, freq, rate, start, count):
    """2 |sum_{i} y[i] e^{-2 pi i freq i / rate}| / count over y[start : start + count] (a whole number of cycles of freq)."""
    re = 0.0
    im_ = 0.0
    for k in range(start, start + count):
        a = 2 * PI * freq * k / rate
        re += y[k] * math.cos(a)
        im_ -= y[k] * math.sin(a)
    return 2 * math.hypot(re, im_) / count


def design_table():
    rows = []
    for old, new in [(24000, 16000), (16000, 24000)]:
        kz = KaiserSinc(old, new)
        hn = HannSinc(old, new)
        rows.append((f"{old} to {new}", f"{kz.up} : {kz.down}", f"{kz.cut:.4f}", f"{kz.beta:.4f}", kz.flen, f"{kz.hw:g}",
                     f"{hn.cut:.4f}", f"{hn.hw:.3f}"))
    return ("Kaiser-tapered sinc: cutoff c = min(1, up/down), beta = 0.1102 (80 - 8.7), 2 x 32 x max(up, down) + 1 taps, "
            "half-width hw = 32 max(up, down) / up input samples. Cos^2-tapered sinc: cutoff c = 0.99 x min(1, new/old), "
            "half-width 6 / c input samples.\n\n"
            + _table(["conversion", "up : down", "Kaiser c", "beta", "filter taps", "hw (samples)", "cos^2 c",
                      "half-width 6 / c"], rows, "llrrrrrr"))


def linear_weights():
    rows = []
    for i in range(6):
        s = i * 1.5
        i0 = math.floor(s)
        frac = s - i0
        rows.append((i, _g(s), f"x[{i0}]" if frac == 0 else f"x[{i0}], x[{i0 + 1}]",
                     "1" if frac == 0 else f"{1 - frac:g}, {frac:g}"))
    return ("Linear interpolation from 24 kHz to 16 kHz: output i sits at input position s = 1.5 i.\n\n"
            + _table(["output i", "position s", "inputs read", "weights"], rows, "rrll"))


def down_tones():
    n_in, old, new = 4800, 24000, 16000
    x = [a + b for a, b in zip(tone(3000, old, n_in), tone(10000, old, n_in))]
    start, count = 400, 2400
    rows = []
    for name, fn in METHODS:
        y = fn(x, old, new)
        rows.append((name, len(y), f"{amplitude_at(y, 3000, new, start, count):.4f}",
                     _g(amplitude_at(y, 6000, new, start, count), 3)))
    return ("Input at 24 kHz, 4800 samples: cos(2 pi 3000 i / 24000) + cos(2 pi 10000 i / 24000), resampled to 16 kHz. "
            "Amplitude 2 |sum y e^{-2 pi i f k / 16000}| / 2400 over output samples 400 .. 2799 (2400 samples, a whole "
            "number of cycles of both frequencies); 10 kHz folds to 16 - 10 = 6 kHz.\n\n"
            + _table(["method", "output samples", "amplitude at 3 kHz", "amplitude at 6 kHz"], rows, "lrrr"))


def up_tones():
    n_in, old, new = 3200, 16000, 24000
    x = tone(3000, old, n_in)
    start, count = 384, 4032
    rows = []
    for name, fn in METHODS:
        y = fn(x, old, new)
        rows.append((name, len(y), f"{amplitude_at(y, 3000, new, start, count):.4f}",
                     _g(amplitude_at(y, 5000, new, start, count), 3), _g(amplitude_at(y, 11000, new, start, count), 3)))
    return ("Input at 16 kHz, 3200 samples: cos(2 pi 3000 i / 16000), resampled to 24 kHz. Amplitudes as in the previous table, "
            "over output samples 384 .. 4415 (4032 samples). The copies of the 3 kHz tone around 16 kHz, at 13 and 19 kHz, "
            "fold to 24 - 13 = 11 kHz and 24 - 19 = 5 kHz.\n\n"
            + _table(["method", "output samples", "amplitude at 3 kHz", "amplitude at 5 kHz", "amplitude at 11 kHz"], rows,
                     "lrrrr"))


def gain_curve(freqs=None):
    """Data of the third figure and a table: gain in dB of each method for a tone at f (24 kHz in, 16 kHz out), measured at the
    apparent output frequency."""
    old, new, n_in = 24000, 16000, 4800
    start, count = 400, 2400
    if freqs is None:
        freqs = [500 * k for k in range(1, 24) if 500 * k != 8000]
    series = {name: [] for name, _ in METHODS}
    for f in freqs:
        x = tone(f, old, n_in)
        alias = alias_of(f, new)[0]
        for name, fn in METHODS:
            y = fn(x, old, new)
            series[name].append(amplitude_at(y, alias, new, start, count))
    return freqs, series


def _db(a):
    return 20 * math.log10(max(a, 1e-6))


def taper_table():
    old, new, n_in = 24000, 16000, 4800
    start, count = 400, 2400
    rows = []
    for f in (3000, 7500, 8500, 9000, 10000, 11500):
        x = tone(f, old, n_in)
        alias = alias_of(f, new)[0]
        amps = []
        for kernel in (PlainSinc(old, new), KaiserSinc(old, new)):
            y = resample_weighted(x, kernel, old, new)
            amps.append(amplitude_at(y, alias, new, start, count))
        rows.append((f, _g(alias), f4(_db(amps[0]), 1), f4(_db(amps[1]), 1)))
    return ("The same support (193 taps at the intermediate rate) and cutoff, with and without the Kaiser taper; a tone at f, "
            "24 kHz in, 16 kHz out, gain in dB measured at the apparent frequency as in the table above.\n\n"
            + _table(["f (Hz)", "apparent (Hz)", "no taper (dB)", "Kaiser taper (dB)"], rows))


def gain_table():
    freqs, series = gain_curve([1000, 4000, 7000, 7500, 8500, 9000, 10000, 12000 - 500])
    rows = []
    for k, f in enumerate(freqs):
        rows.append((f, _g(alias_of(f, 16000)[0]), *(f4(_db(series[name][k]), 1) for name, _ in METHODS)))
    return ("A cosine at f (24 kHz in, 4800 samples) resampled to 16 kHz; gain = 20 log10 of the amplitude measured at the "
            "apparent frequency (2400 output samples from 400, as above), floored at -120 dB.\n\n"
            + _table(["f (Hz)", "apparent (Hz)", "linear (dB)", "Kaiser (dB)", "cos^2 (dB)"], rows))


def work_counts():
    rows = []
    for old, new in [(24000, 16000), (16000, 24000)]:
        n_in = old  # one second of input
        n_out = out_length(n_in, old, new)
        # linear: nonzero weights per output
        lin = 0
        for i in range(n_out):
            # output i sits at input position i * old / new: a whole position has one weight, otherwise two,
            # and a neighbour past the last input sample is not counted
            first, rem = divmod(i * old, new)
            lin += 1 if rem == 0 else (2 if first + 1 < n_in else 1)
        kz = KaiserSinc(old, new)
        hn = HannSinc(old, new)
        kz_taps = sum(len(taps_of(kz, j, n_in)) for j in range(n_out))
        hn_taps = sum(1 for j in range(n_out) for i in taps_of(hn, j, n_in) if hn.weight_of(j * hn.down - i * hn.up) != 0.0)
        stuffed = (kz.flen) * (n_in * kz.up)  # direct zero-stuffed convolution at the intermediate rate
        rows.append((f"{old} to {new}", n_out, f"{lin / n_out:.3f}", lin, f"{kz_taps / n_out:.2f}", kz_taps,
                     f"{hn_taps / n_out:.2f}", hn_taps, stuffed))
    return ("One second of input (as many samples as the old rate). Multiply-adds with a stored weight, counting only inputs that "
            "exist inside the second in every column (a tap that would reach before the first or past the last sample "
            "multiplies the zero padding and is not counted): linear interpolation counts the nonzero weights (one at a "
            "whole input position, else two), the Kaiser design the filter taps n = 0 .. 192 that an output reaches, the "
            "cos^2 design the inputs with |c d| < 6. The last column is the count of a dense convolution of the zero-stuffed "
            "signal at up x input rate with the 193 taps, before discarding all but every down-th output.\n\n"
            + _table(["conversion", "outputs", "linear per output", "linear total", "Kaiser per output", "Kaiser total",
                      "cos^2 per output", "cos^2 total", "zero-stuffed, dense"], rows, "lrrrrrrrr"))
