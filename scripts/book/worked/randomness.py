# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Randomness on a Deterministic Machine".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: randomness.<name> -->` markers. Helpers with a leading underscore
(or that take arguments) are shared with the chapter's figures, so a picture
and the table beside it cannot disagree.

The generators under study (a linear congruential generator and SplitMix64) are
implemented here from the chapter's description, on Python's unbounded
integers with explicit masking to 64 bits. The only source of "outside"
randomness is a named `bookrng` stream, used for the seeds the examples start
from and as the independent reference stream the statistical tests are
calibrated against; no module in the book carries a seed literal.

Swedish Embedded AB implements solutions for reproducible random sampling in
machine learning pipelines. If your team needs expertise in pseudo-random
generation, seeding discipline and the statistics of sampling, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math
from fractions import Fraction
from itertools import permutations

import bookrng

M64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15      # 2^64 / golden ratio, rounded to the nearest odd integer
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _hex(v):
    return f"0x{v:016x}"


def _fx(v, d=4):
    v = float(v) + 0.0
    s = f"{v:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def _seed(name):
    """A 64-bit seed for the example `name`, drawn from its frozen bookrng stream."""
    return bookrng.stream("randomness." + name).getrandbits(64)


# ------------------------------------------------------------ the generators

class SplitMix64:
    """Add a fixed odd constant to the state, then mix the new state with xor-shifts and multiplications."""

    def __init__(self, seed):
        self.state = seed & M64

    def next_u64(self):
        self.state = (self.state + GAMMA) & M64
        return _mix(self.state)

    def uniform(self):
        return (self.next_u64() >> 11) / (1 << 53)

    def integer(self, n):
        """Uniform in 0 .. n-1 by rejection (draws until the value lies above 2^64 mod n)."""
        assert 0 < n <= (1 << 64)
        threshold = (1 << 64) % n
        while True:
            r = self.next_u64()
            if r >= threshold:
                return r % n

    def normal(self):
        """Box-Muller; the sine partner of each pair is returned on the next call."""
        if self._spare is not None:
            z, self._spare = self._spare, None
            return z
        u1, u2 = self.uniform(), self.uniform()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)

    _spare = None

    def shuffle(self, items):
        """Fisher-Yates, in place; returns the list for convenience."""
        for i in range(len(items) - 1, 0, -1):
            j = self.integer(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


def _mix(z):
    z = ((z ^ (z >> 30)) * C1) & M64
    z = ((z ^ (z >> 27)) * C2) & M64
    return z ^ (z >> 31)


def _derive(seed, name):
    """The seed of the stream called `name` under the run seed `seed`: a hash of both."""
    h = _mix(seed & M64)
    for b in name.encode("utf-8"):
        h = _mix(h ^ b)
    return h


def _lcg(a, c, m, x):
    while True:
        x = (a * x + c) % m
        yield x


def _period(a, c, m, x0):
    """Length of the cycle the sequence x0, x1, ... ends up in (brute force)."""
    seen = {}
    x, i = x0, 0
    while x not in seen:
        seen[x] = i
        x = (a * x + c) % m
        i += 1
    return i - seen[x]


# ------------------------------------------------------------ the toy run

DATA = [(0.5, 1.4), (1.0, 2.1), (1.5, 4.6), (2.0, 5.8), (2.5, 7.9), (3.0, 9.3), (3.5, 10.2), (4.0, 12.4)]


def _sgd_run(seed, lr=0.01, epochs=2, rng=None):
    """Fit y = w x by one-example steps. w starts uniform in [-1, 1); each epoch visits the examples
    in a freshly shuffled order. Returns (initial w, last epoch's order, final w, final mean squared error)."""
    rng = rng or SplitMix64(seed)
    w = 2.0 * rng.uniform() - 1.0
    w0 = w
    order = []
    for _ in range(epochs):
        order = rng.shuffle(list(range(len(DATA))))
        for i in order:
            x, y = DATA[i]
            w -= lr * 2.0 * (w * x - y) * x
    loss = sum((w * x - y) ** 2 for x, y in DATA) / len(DATA)
    return w0, order, w, loss


def toy_runs():
    sa, sb = _seed("toy_runs.a"), _seed("toy_runs.b")
    rows = []
    for label, s in (("run 1", sa), ("run 2", sb), ("run 3 (seed of run 1 again)", sa)):
        w0, order, w, loss = _sgd_run(s)
        rows.append((label, _hex(s), _fx(w0), " ".join(map(str, order)), f"{w:.6f}", f"{loss:.6f}"))
    head = ["run", "seed", "initial w", "last epoch order", "final w", "final loss"]
    return ("The same eight points, learning rate 0.01, two epochs, differing only in the seed:\n\n"
            + _table(head, rows, "llrlrr"))


# ------------------------------------------------------------ the tiny LCG

def tiny_lcg():
    m, a, c = 16, 5, 3
    xs, x = [], 7
    for _ in range(18):
        xs.append(x)
        x = (a * x + c) % m
    rows = [(i, v, f"{a} * {v} + {c} = {a * v + c}", f"{(a * v + c) % m}") for i, v in enumerate(xs[:6])]
    out = _table(["step i", "state x_i", "a x_i + c", "mod 16 = x_(i+1)"], rows)
    out += f"\n\nThe first 18 states from the seed 7: {' '.join(map(str, xs))}"
    out += f"\n\nPeriod of this generator from every possible seed: {sorted({_period(a, c, m, s) for s in range(m)})}."
    return out


def tiny_lcg_periods():
    m = 16
    rows = []
    for a, c in ((5, 3), (5, 2), (4, 3), (3, 1), (9, 7), (1, 1), (13, 1), (5, 0)):
        rows.append((a, c, _period(a, c, m, 1)))
    full = [(a, c) for a in range(m) for c in range(m) if _period(a, c, m, 1) == m]
    predicted = [(a, c) for a in range(m) for c in range(m) if c % 2 == 1 and a % 4 == 1]
    out = _table(["a", "c", "period from seed 1"], rows)
    out += (f"\n\nOf the {m * m} pairs (a, c), {len(full)} reach the full period {m} from seed 1; "
            f"the pairs with c odd and a - 1 divisible by 4 number {len(predicted)}, "
            f"and the two sets are {'identical' if full == predicted else 'different'}.")
    return out


# --------------------------------------------------------------- low bits

A16, C16, M16 = 25173, 13849, 1 << 16


def low_bits():
    """Period of bit k of a full-period LCG modulo 2^16."""
    seq = [0] * (1 << 16)
    x = 1
    for i in range(1 << 16):
        seq[i] = x
        x = (A16 * x + C16) % M16
    rows = []
    for k in (0, 1, 2, 3, 7, 15):
        bits = [(v >> k) & 1 for v in seq]
        period = next(p for p in (1 << j for j in range(17)) if all(bits[i] == bits[i % p] for i in range(len(bits))))
        first = "".join(str(b) for b in bits[:16])
        rows.append((k, period, first))
    return ("First 16 values of bit k of the sequence x_0, x_1, ... of the generator with a = 25173, c = 13849, "
            "m = 65536 started at 1.\n\n" + _table(["bit k", "period of bit k", "first 16 values"], rows, "rrl"))


# --------------------------------------------------------------- lattice

def _shortest_dual(a, m):
    """Shortest nonzero (s, t) with s + a t = 0 (mod m): the pairs (x_i, x_(i+1)) lie on lines s u + t v = integer."""
    best = None
    for t in range(0, 4096):
        s = (-a * t) % m
        if s > m // 2:
            s -= m
        if s == 0 and t == 0:
            continue
        n2 = s * s + t * t
        if best is None or n2 < best[0]:
            best = (n2, s, t)
    return best[1], best[2], math.sqrt(best[0])


def lattice_lines():
    rows = []
    for m, a in ((M16, 5), (M16, 4097), (M16, 25173), (4096, 257)):
        s, t, norm = _shortest_dual(a, m)
        rows.append((f"{m:,}", a, f"({s}, {t})", f"{norm:.1f}", f"{1 / norm:.5f}"))
    return _table(["modulus m", "multiplier a", "shortest (s, t)", "length", "line spacing 1/length"], rows)


def _pairs(gen, n, bits):
    """Consecutive pairs of the top `bits` bits of each value, as ints."""
    out, prev = [], next(gen)
    for _ in range(n):
        cur = next(gen)
        out.append((prev, cur))
        prev = cur
    return out


# --------------------------------------------------------------- splitmix

def splitmix_trace():
    seed = _seed("splitmix_trace")
    state = (seed + GAMMA) & M64
    z0 = state
    z1 = ((z0 ^ (z0 >> 30)) * C1) & M64
    z2 = ((z1 ^ (z1 >> 27)) * C2) & M64
    out = z2 ^ (z2 >> 31)
    rows = [("seed", _hex(seed)), ("state + GAMMA", _hex(z0)),
            ("z ^ (z >> 30), times C1", _hex(z1)), ("z ^ (z >> 27), times C2", _hex(z2)),
            ("z ^ (z >> 31) = output", _hex(out))]
    g = SplitMix64(seed)
    nxt = [_hex(g.next_u64()) for _ in range(4)]
    assert nxt[0] == _hex(out)
    return (_table(["step", "64-bit value"], rows, "lr")
            + "\n\nThe first four outputs from that seed: " + ", ".join(f"`{v}`" for v in nxt) + ".")


def _popcount(v):
    return bin(v).count("1")


def avalanche():
    rng = bookrng.stream("randomness.avalanche")
    trials = 2000
    sm_flip, lcg_flip, sm_gap, lcg_gap = [], [], set(), set()
    a, c = 6364136223846793005, 1442695040888963407

    def sm(s):
        return SplitMix64(s).next_u64()

    def lcg(s):
        return (a * s + c) & M64

    for _ in range(trials):
        s = rng.getrandbits(64)
        k = rng.randrange(64)
        sm_flip.append(_popcount(sm(s) ^ sm(s ^ (1 << k))))
        lcg_flip.append(_popcount(lcg(s) ^ lcg(s ^ (1 << k))))
        sm_gap.add((sm((s + 1) & M64) - sm(s)) & M64)
        lcg_gap.add((lcg((s + 1) & M64) - lcg(s)) & M64)
    rows = [("SplitMix64 first output", f"{sum(sm_flip) / trials:.2f}", len(sm_gap)),
            ("one LCG step (64-bit modulus)", f"{sum(lcg_flip) / trials:.2f}", len(lcg_gap))]
    return (f"Over {trials} random seeds.\n\n"
            + _table(["generator", "average number of the 64 output bits that change when one random seed bit is flipped",
                      "distinct values of (output for s + 1) - (output for s)"], rows, "lrr"))


# --------------------------------------------------------- bucket tests

def _chi2(counts, expected):
    return sum((o - expected) ** 2 / expected for o in counts)


def _reference_chi2(cells, n, runs):
    """X^2 of `runs` experiments of `n` draws over `cells` equally likely cells, from an independent reference stream."""
    rng = bookrng.stream(f"randomness.reference.{cells}.{n}")
    out = []
    for _ in range(runs):
        counts = [0] * cells
        for _ in range(n):
            counts[rng.randrange(cells)] += 1
        out.append(_chi2(counts, n / cells))
    return out


def _mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def bucket_calibration():
    rows = []
    for cells, n, runs in ((4, 400, 3000), (16, 1600, 2000), (256, 25600, 150)):
        m, sd = _mean_sd(_reference_chi2(cells, n, runs))
        rows.append((cells, n, runs, f"{m:.1f}", cells - 1, f"{sd:.1f}", f"{math.sqrt(2 * (cells - 1)):.1f}",
                     f"{m + 5 * sd:.0f}"))
    return _table(["cells k", "draws n", "experiments", "mean of X^2", "k - 1", "spread of X^2", "sqrt(2(k - 1))", "alarm: mean + 5 spreads"], rows)


def _tests_for(name, draw, bits_total):
    """Three X^2 values for a source of `bits_total`-bit values: top 4 bits, pairs of low 2 bits, pairs of top 4 bits."""
    n1 = 1600
    c = [0] * 16
    for _ in range(n1):
        c[draw() >> (bits_total - 4)] += 1
    t1 = _chi2(c, n1 / 16)
    n2 = 1600
    c = [0] * 16
    prev = draw()
    for _ in range(n2):
        cur = draw()
        c[(prev & 3) * 4 + (cur & 3)] += 1
        prev = cur
    t2 = _chi2(c, n2 / 16)
    n3 = 25600
    c = [0] * 256
    prev = draw()
    for _ in range(n3):
        cur = draw()
        c[(prev >> (bits_total - 4)) * 16 + (cur >> (bits_total - 4))] += 1
        prev = cur
    t3 = _chi2(c, n3 / 256)
    return t1, t2, t3


def bucket_tests():
    gens = []
    sm = SplitMix64(_seed("bucket_tests.splitmix"))
    gens.append(("SplitMix64 (64 bits)", sm.next_u64, 64))
    x = [_seed("bucket_tests.nr") & 0xFFFFFFFF]

    def nr():
        x[0] = (1664525 * x[0] + 1013904223) % (1 << 32)
        return x[0]
    gens.append(("LCG a = 1664525, c = 1013904223, m = 2^32", nr, 32))
    y = [_seed("bucket_tests.good16") & 0xFFFF]

    def good16():
        y[0] = (A16 * y[0] + C16) % M16
        return y[0]
    gens.append(("LCG a = 25173, c = 13849, m = 2^16", good16, 16))
    z = [_seed("bucket_tests.bad16") & 0xFFFF]

    def bad16():
        z[0] = (4097 * z[0] + 1) % M16
        return z[0]
    gens.append(("LCG a = 4097, c = 1, m = 2^16", bad16, 16))
    rows = []
    for name, draw, bits in gens:
        t1, t2, t3 = _tests_for(name, draw, bits)
        rows.append((name, f"{t1:.1f}", f"{t2:.1f}", f"{t3:.1f}"))
    return _table(["generator", "top 4 bits, 16 cells (alarm 42)", "pairs of low 2 bits, 16 cells (alarm 42)",
                   "pairs of top 4 bits, 256 cells (alarm about 370)"], rows, "lrrr")


# ----------------------------------------------------- uniform from bits

def uniform_from_bits():
    rows = []
    for raw in (0, 1 << 63, (1 << 64) - (1 << 11), (1 << 64) - 1):
        top = (raw >> 11) / (1 << 53)
        naive = float(raw) / float(1 << 64)
        rows.append((_hex(raw), f"{top!r}", f"{naive!r}"))
    head = ["raw 64-bit value", "top 53 bits / 2^53", "float(raw) / 2^64"]
    out = _table(head, rows, "lrr")
    out += ("\n\nThe spacing of the grid of values (raw >> 11) / 2^53 is 2^-53 = "
            f"{2.0 ** -53:.3e}; the largest value is 1 - 2^-53 = {1 - 2.0 ** -53!r}. "
            f"For a 32-bit float the same recipe takes the top 24 bits: the grid is 2^-24 = {2.0 ** -24:.3e} and the largest value {1 - 2.0 ** -24!r}.")
    return out


def one_sided_range():
    rng = SplitMix64(_seed("one_sided_range"))
    lo33 = hi33 = lo32 = hi32 = None
    neg33 = neg32 = 0
    n = 100000
    a, c = 6364136223846793005, 1442695040888963407
    s = rng.next_u64()
    for _ in range(n):
        s = (a * s + c) & M64
        v33 = (s >> 33) / (1 << 31) - 1.0
        v32 = (s >> 32) / (1 << 31) - 1.0
        lo33 = v33 if lo33 is None else min(lo33, v33)
        hi33 = v33 if hi33 is None else max(hi33, v33)
        lo32 = v32 if lo32 is None else min(lo32, v32)
        hi32 = v32 if hi32 is None else max(hi32, v32)
        neg33 += v33 < 0
        neg32 += v32 < 0
    rows = [("top 31 bits (shift by 33), divided by 2^31, minus 1", f"{lo33:.6f}", f"{hi33:.6f}", f"{neg33 / n:.4f}"),
            ("top 32 bits (shift by 32), divided by 2^31, minus 1", f"{lo32:.6f}", f"{hi32:.6f}", f"{neg32 / n:.4f}")]
    return _table(["recipe for a value meant to lie in [-1, 1)", "smallest of 100,000", "largest of 100,000", "share below 0"], rows, "lrrr")


# ------------------------------------------------------ integers in 0..n

def modulo_bias_small():
    bits, n = 3, 6
    size = 1 << bits
    plain = [sum(1 for r in range(size) if r % n == k) for k in range(n)]
    threshold = size % n
    kept = [sum(1 for r in range(size) if r >= threshold and r % n == k) for k in range(n)]
    rows = [(k, f"{plain[k]}/{size}", f"{kept[k]}/{size - threshold}") for k in range(n)]
    out = _table(["result k", "probability of r mod 6 = k", "probability after rejecting r < 2"], rows, "lrr")
    out += (f"\n\nPlain remainder: the values 6 and 7 wrap around to 0 and 1, so those two results are twice as likely as the rest. "
            f"Rejecting the first {threshold} values leaves {size - threshold} that divide evenly into {n} results of {(size - threshold) // n} each; "
            f"the probability that a draw is kept is {size - threshold}/{size} = {(size - threshold) / size}, so the expected number of draws per result is {size / (size - threshold):.4f}.")
    return out


def modulo_bias_by_n():
    rows = []
    for n in (6, 1000, 10 ** 6, (1 << 32) + 1, 3 << 62):
        q = (1 << 64) // n
        rem = (1 << 64) % n
        bias = Fraction(1, q)
        keep = Fraction((1 << 64) - rem, 1 << 64)
        rows.append((f"{n:,}", f"{rem:,}", f"{float(bias):.3e}", f"{float(keep):.6f}", f"{float(1 / keep):.6f}"))
    return _table(["n", "2^64 mod n", "relative excess of the likelier results", "chance a draw is kept", "expected draws"], rows)


def rejection_integer_counts():
    bits, n, wanted = 3, 6, 60000
    rng = SplitMix64(_seed("rejection_integer_counts"))
    counts, drawn = [0] * n, 0
    plain = [0] * n
    while sum(counts) < wanted:
        r = rng.next_u64() >> (64 - bits)
        drawn += 1
        plain[r % n] += 1
        if r >= (1 << bits) % n:
            counts[r % n] += 1
    rows = [(k, plain[k], counts[k]) for k in range(n)]
    out = _table(["result", f"remainder only ({drawn:,} draws)", f"with rejection ({wanted:,} kept)"], rows)
    out += f"\n\nThe rejecting sampler used {drawn:,} draws for {wanted:,} results, {drawn / wanted:.4f} per result (the prediction is {8 / 6:.4f})."
    return out


# ------------------------------------------------- sampling by cumulative sum

def discrete_sampling():
    logits = [2.0, 1.0, 0.5, 0.0, -1.0, -2.0]
    mx = max(logits)
    ex = [math.exp(v - mx) for v in logits]
    s = sum(ex)
    p = [v / s for v in ex]
    cum, acc = [], 0.0
    for v in p:
        acc += v
        cum.append(acc)
    rng = SplitMix64(_seed("discrete_sampling"))
    n = 20000
    counts = [0] * len(p)
    for _ in range(n):
        u = rng.uniform()
        lo, hi = 0, len(cum) - 1
        while lo < hi:           # binary search for the first cum[k] > u
            mid = (lo + hi) // 2
            if cum[mid] > u:
                hi = mid
            else:
                lo = mid + 1
        counts[lo] += 1
    rows = [(k, f"{logits[k]:.1f}", f"{p[k]:.4f}", f"{cum[k]:.4f}", counts[k], f"{counts[k] / n:.4f}") for k in range(len(p))]
    out = _table(["token", "logit", "p", "cumulative p", f"draws of {n:,}", "frequency"], rows)
    return out


def inverse_cdf_exponential():
    lam = 2.0
    rows = [(f"{u:.2f}", f"{abs(-math.log(1 - u) / lam):.4f}", f"{1 - math.exp(-lam * (-math.log(1 - u) / lam)):.2f}") for u in (0.0, 0.10, 0.50, 0.90, 0.99)]
    out = _table(["u", "x = -ln(1 - u) / 2", "F(x) = 1 - e^(-2x), which equals u"], rows)
    rng = SplitMix64(_seed("inverse_cdf_exponential"))
    n = 50000
    xs = [-math.log(1.0 - rng.uniform()) / lam for _ in range(n)]
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    over1 = sum(1 for x in xs if x > 1.0) / n
    out += "\n\n" + _table(["quantity", f"{n:,} draws", "exact"], [
        ("mean", f"{mean:.4f}", f"{1 / lam:.4f}"),
        ("variance", f"{var:.4f}", f"{1 / lam ** 2:.4f}"),
        ("share above 1", f"{over1:.4f}", f"{math.exp(-lam):.4f}")])
    return out


# --------------------------------------------------------------- Gaussians

def _box_muller_pairs(n, seed_name):
    rng = SplitMix64(_seed(seed_name))
    out = []
    for _ in range(n):
        u1, u2 = rng.uniform(), rng.uniform()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        out.append((r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)))
    return out


def _gaussian_samples(n=20000):
    """The standard-normal samples of the chapter's figure and table: 2n values from n Box-Muller pairs."""
    return [z for pair in _box_muller_pairs(n, "box_muller_samples") for z in pair]


def box_muller_one_pair():
    rng = SplitMix64(_seed("box_muller_one_pair"))
    rows = []
    for _ in range(3):
        u1, u2 = rng.uniform(), rng.uniform()
        r2 = -2.0 * math.log(1.0 - u1)
        r = math.sqrt(r2)
        th = 2 * math.pi * u2
        rows.append((f"{u1:.4f}", f"{u2:.4f}", f"{r2:.4f}", f"{r:.4f}", f"{th:.4f}", f"{r * math.cos(th):.4f}", f"{r * math.sin(th):.4f}",
                     f"{(r * math.cos(th)) ** 2 + (r * math.sin(th)) ** 2:.4f}"))
    return _table(["u1", "u2", "R^2 = -2 ln(1 - u1)", "R", "angle = 2 pi u2", "z1 = R cos", "z2 = R sin", "z1^2 + z2^2"], rows)


def box_muller_check():
    pairs = _box_muller_pairs(20000, "box_muller_samples")
    zs = [z for p in pairs for z in p]
    n = len(zs)
    mean = sum(zs) / n
    var = sum((z - mean) ** 2 for z in zs) / n
    cov = sum(a * b for a, b in pairs) / len(pairs)
    rows = [("mean", f"{mean:.4f}", "0"), ("variance", f"{var:.4f}", "1"),
            ("covariance of z1 and z2", f"{cov:.4f}", "0")]
    for k in (1, 2, 3):
        rows.append((f"share with absolute value at most {k}", f"{sum(1 for z in zs if abs(z) <= k) / n:.4f}", f"{math.erf(k / math.sqrt(2)):.4f}"))
    return f"{len(pairs):,} pairs, {n:,} values.\n\n" + _table(["quantity", "measured", "exact"], rows)


def tail_reach():
    rows = []
    for label, smallest in (("53-bit uniform, 1 - u in [2^-53, 1]", 2.0 ** -53),
                            ("32-bit uniform, 1 - u in [2^-32, 1]", 2.0 ** -32),
                            ("24-bit uniform (float32 grid), 1 - u in [2^-24, 1]", 2.0 ** -24),
                            ("53-bit uniform with u clamped to at least 10^-12", 1e-12)):
        reach = math.sqrt(-2.0 * math.log(smallest))
        tail = math.erfc(reach / math.sqrt(2))
        rows.append((label, f"{reach:.3f}", f"{tail:.2e}"))
    return _table(["uniform source", "largest absolute z it can produce", "Gaussian probability beyond that"], rows, "lrr")


# ------------------------------------------------- rejection sampling (general)

def rejection_general():
    rng = SplitMix64(_seed("rejection_general"))
    f = lambda x: 6.0 * x * (1.0 - x)       # density on [0, 1]; its largest value is 1.5 at x = 1/2
    M = 1.5
    n, kept, drawn = 40000, [], 0
    while len(kept) < n:
        x = rng.uniform()
        y = M * rng.uniform()
        drawn += 1
        if y < f(x):
            kept.append(x)
    cdf = lambda x: 3 * x * x - 2 * x ** 3
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        cnt = sum(1 for x in kept if lo <= x < hi)
        rows.append((f"[{lo:.1f}, {hi:.1f})", f"{cnt / n:.4f}", f"{cdf(hi) - cdf(lo):.4f}", f"{(hi - lo) :.1f}"))
    out = _table(["interval", "share of accepted", "exact probability", "share if the proposal were kept as is"], rows)
    out += (f"\n\n{drawn:,} proposals gave {n:,} accepted values: acceptance {n / drawn:.4f} against the predicted 1/M = {1 / M:.4f}. "
            f"The accepted values have mean {sum(kept) / n:.4f} (exact 0.5000).")
    return out


def disc_and_polar():
    rng = SplitMix64(_seed("disc_and_polar"))
    tries = accepted = 0
    zs = []
    while len(zs) < 40000:
        v1, v2 = 2 * rng.uniform() - 1, 2 * rng.uniform() - 1
        tries += 1
        s = v1 * v1 + v2 * v2
        if s >= 1.0 or s == 0.0:
            continue
        accepted += 1
        k = math.sqrt(-2.0 * math.log(s) / s)
        zs.extend((v1 * k, v2 * k))
    n = len(zs)
    mean = sum(zs) / n
    var = sum((z - mean) ** 2 for z in zs) / n
    rows = [("pairs tried", f"{tries:,}", ""), ("pairs inside the unit disc", f"{accepted:,}", ""),
            ("acceptance", f"{accepted / tries:.4f}", f"pi/4 = {math.pi / 4:.4f}"),
            ("mean of the 40,000 outputs", f"{mean:.4f}", "0"), ("variance", f"{var:.4f}", "1"),
            ("share with absolute value at most 2", f"{sum(1 for z in zs if abs(z) <= 2) / n:.4f}", f"{math.erf(2 / math.sqrt(2)):.4f}")]
    return _table(["quantity", "measured", "exact"], rows, "lrr")


# ---------------------------------------------------------------- shuffles

def _wrong_shuffle_counts(n):
    """Exact number of the n^n equally likely index sequences that produce each permutation: swap a[i] with a[j], j uniform on 0..n-1."""
    dist = {tuple(range(n)): 1}
    for i in range(n):
        nxt = {}
        for perm, cnt in dist.items():
            for j in range(n):
                p = list(perm)
                p[i], p[j] = p[j], p[i]
                key = tuple(p)
                nxt[key] = nxt.get(key, 0) + cnt
        dist = nxt
    return dist


def _fisher_yates_counts(n):
    """Same for Fisher-Yates: for i = n-1 down to 1, swap a[i] with a[j], j uniform on 0..i. n! sequences in all."""
    dist = {tuple(range(n)): 1}
    for i in range(n - 1, 0, -1):
        nxt = {}
        for perm, cnt in dist.items():
            for j in range(i + 1):
                p = list(perm)
                p[i], p[j] = p[j], p[i]
                key = tuple(p)
                nxt[key] = nxt.get(key, 0) + cnt
        dist = nxt
    return dist


def wrong_shuffle_probabilities(n=3):
    """Exact probability of each permutation under the wrong shuffle, in lexicographic order (figure data)."""
    counts = _wrong_shuffle_counts(n)
    total = n ** n
    return [(p, Fraction(counts.get(p, 0), total)) for p in sorted(permutations(range(n)))]


def shuffle_n3():
    wrong = _wrong_shuffle_counts(3)
    fy = _fisher_yates_counts(3)
    rows = []
    for p in sorted(permutations(range(3))):
        rows.append((" ".join(map(str, p)), f"{wrong.get(p, 0)}/27", f"{float(Fraction(wrong.get(p, 0), 27)):.4f}",
                     f"{fy.get(p, 0)}/6", f"{float(Fraction(fy.get(p, 0), 6)):.4f}"))
    return _table(["resulting order", "wrong shuffle: paths of 27", "probability", "Fisher-Yates: paths of 6", "probability"], rows, "lrrrr")


def shuffle_bias_by_n():
    rows = []
    for n in range(3, 8):
        wrong = _wrong_shuffle_counts(n)
        fy = _fisher_yates_counts(n)
        total = n ** n
        fact = math.factorial(n)
        counts = [wrong.get(p, 0) for p in permutations(range(n))]
        tv = sum(abs(Fraction(c, total) - Fraction(1, fact)) for c in counts) / 2
        uniform_fy = all(c == 1 for c in fy.values()) and len(fy) == fact
        rows.append((n, f"{total:,}", f"{fact:,}", f"{total / fact:.2f}", f"{max(counts)}/{min(counts)}",
                     f"{max(counts) / min(counts):.2f}", f"{float(tv):.4f}", "yes" if uniform_fy else "no"))
    return _table(["n", "paths n^n", "orders n!", "paths per order", "most / fewest paths", "ratio", "total variation from uniform",
                   "Fisher-Yates: every order exactly once"], rows)


def shuffle_empirical():
    rng = SplitMix64(_seed("shuffle_empirical"))
    n = 60000
    counts = {p: 0 for p in permutations(range(3))}
    for _ in range(n):
        counts[tuple(rng.shuffle([0, 1, 2]))] += 1
    rows = [(" ".join(map(str, p)), c, f"{c / n:.4f}") for p, c in counts.items()]
    return _table(["order", f"count in {n:,} shuffles", "frequency"], rows)


# ---------------------------------------------------- seeds and streams

def shared_vs_split():
    seed = _seed("shared_vs_split")

    def run(extra_weights, split):
        if split:
            init_rng = SplitMix64(_derive(seed, "init"))
            order_rng = SplitMix64(_derive(seed, "order"))
        else:
            init_rng = order_rng = SplitMix64(seed)
        weights = [init_rng.uniform() * 2 - 1 for _ in range(4 + extra_weights)]
        order = order_rng.shuffle(list(range(6)))
        return weights, order

    rows = []
    for split in (False, True):
        for extra in (0, 1):
            w, order = run(extra, split)
            rows.append(("one shared generator" if not split else "one stream per purpose", 4 + extra,
                         " ".join(f"{v:+.3f}" for v in w[:4]), " ".join(map(str, order))))
    return _table(["design", "weights drawn", "first four weights", "data order"], rows, "lrll")


def streams_are_positions():
    seed = _seed("streams_are_positions")
    a = SplitMix64(seed)
    b = SplitMix64((seed + 3 * GAMMA) & M64)
    xs = [a.next_u64() for _ in range(6)]
    ys = [b.next_u64() for _ in range(3)]
    rows = [(i, _hex(xs[i]), _hex(ys[i - 3]) if i >= 3 else "") for i in range(6)]
    out = _table(["draw", f"stream from state s", "stream from state s + 3 GAMMA"], rows, "rll")
    out += f"\n\nThe second stream is the first one three positions later: {'yes' if ys == xs[3:6] else 'no'}."
    names = ("init", "order", "dropout", "init2")
    rows = []
    for nm in names:
        g = SplitMix64(_derive(seed, nm))
        rows.append((f"`{nm}`", _hex(_derive(seed, nm)), _hex(g.next_u64())))
    out += "\n\n" + _table(["stream name", "derived seed", "first draw"], rows, "lrr")
    return out


def hidden_luck():
    """Two arms run the same procedure; and two arms that differ by a small real change."""
    seeds = bookrng.stream("randomness.hidden_luck")
    n = 400
    pairs = [(seeds.getrandbits(64), seeds.getrandbits(64)) for _ in range(n)]
    same = [(_sgd_run(a, lr=0.01)[3], _sgd_run(b, lr=0.01)[3]) for a, b in pairs]
    diff_own = [(_sgd_run(a, lr=0.01)[3], _sgd_run(b, lr=0.0105)[3]) for a, b in pairs]
    diff = [(_sgd_run(a, lr=0.01)[3], _sgd_run(a, lr=0.0105)[3]) for a, _ in pairs]
    losses = [x for x, _ in same]
    m, sd = _mean_sd(losses)

    def summary(ps, label):
        d = [b - a for a, b in ps]
        dm, dsd = _mean_sd(d)
        wins = sum(1 for a, b in ps if b < a)
        return (label, f"{wins / len(ps):.3f}", f"{dm:+.4f}", f"{dsd / math.sqrt(len(ps)):.4f}")
    rows = [summary(same, "B = A's procedure, own seed"), summary(diff_own, "B = learning rate 0.0105, own seed"),
            summary(diff, "B = learning rate 0.0105, A's seed")]
    out = f"Final loss of the toy run over {n} seeds: mean {m:.4f}, standard deviation {sd:.4f}, smallest {min(losses):.4f}, largest {max(losses):.4f}.\n\n"
    out += _table(["comparison (400 pairs)", "share of pairs where B has the lower loss", "mean of (loss B - loss A)", "standard error of that mean"], rows, "lrrr")
    first = [(f"{a:.4f}", f"{b:.4f}", "B" if b < a else "A") for a, b in same[:5]]
    out += "\n\nThe first five identical-procedure pairs:\n\n" + _table(["loss A", "loss B", "winner"], first, "rrl")
    return out


def float_order():
    rng = SplitMix64(_seed("float_order"))
    xs = [rng.uniform() - 0.5 + 1e-3 * (i % 7) for i in range(100000)]
    fwd = 0.0
    for v in xs:
        fwd += v
    bwd = 0.0
    for v in reversed(xs):
        bwd += v

    def pairwise(a):
        if len(a) <= 2:
            return sum(a)
        h = len(a) // 2
        return pairwise(a[:h]) + pairwise(a[h:])
    pw = pairwise(xs)
    ex = math.fsum(xs)
    rows = [("left to right", repr(fwd), f"{fwd - ex:+.2e}"), ("right to left", repr(bwd), f"{bwd - ex:+.2e}"),
            ("pairwise tree", repr(pw), f"{pw - ex:+.2e}"), ("exact (fsum)", repr(ex), "0")]
    return "The same 100,000 draws, added in three orders.\n\n" + _table(["order of the additions", "sum", "difference from exact"], rows, "lrr")


def _unxorshift_right(y, k):
    x = y
    for _ in range(64 // k + 1):
        x = y ^ (x >> k)
    return x


def predict_from_output():
    seed = _seed("predict_from_output")
    g = SplitMix64(seed)
    observed = g.next_u64()
    z = _unxorshift_right(observed, 31)
    z = (z * pow(C2, -1, 1 << 64)) & M64
    z = _unxorshift_right(z, 27)
    z = (z * pow(C1, -1, 1 << 64)) & M64
    state = _unxorshift_right(z, 30)
    clone = SplitMix64(state)
    rows = []
    for i in range(3):
        actual = g.next_u64()
        guess = clone.next_u64()
        rows.append((i + 2, _hex(guess), _hex(actual), "yes" if guess == actual else "no"))
    head = (f"One observed output `{_hex(observed)}` was undone step by step (xor-shift, multiply by the inverse of C2 modulo 2^64, "
            f"xor-shift, multiply by the inverse of C1, xor-shift) to the state `{_hex(state)}`.\n\n")
    return head + _table(["draw number", "predicted from the one output", "actual", "equal"], rows, "rllr")


def seed_parse():
    cases = ["42", "0", "0xff", "0XFF", " 7 ", "18446744073709551615", "18446744073709551616", "-1", "1.5", "", "abc", "0x"]
    rows = []
    for t in cases:
        s = t.strip()
        try:
            v = int(s[2:], 16) if s[:2].lower() == "0x" else (int(s) if s.isdigit() else None)
            if v is None or not 0 <= v <= M64:
                raise ValueError
            rows.append((f"`{t}`", str(v), "accepted"))
        except ValueError:
            rows.append((f"`{t}`", "-", "refused"))
    out = _table(["text given as a seed", "value", "result"], rows, "lrl")
    big = Fraction(M64 - (1 << 63) + 1, 1 << 64)
    out += (f"\n\nThe seeds above the largest signed 64-bit integer, {(1 << 63) - 1:,}, are the share {big} of all {1 << 64:,} possible seeds. "
            f"Round trip of seeds through a signed 64-bit field:\n\n")
    rows = []
    for v in (1 << 62, (1 << 63) - 1, 1 << 63, M64):
        signed = v - (1 << 64) if v >= (1 << 63) else v
        rows.append((f"{v:,}", f"{signed:,}", "yes" if signed == v else "no"))
    out += _table(["seed", "read back as a signed 64-bit integer", "survives"], rows, "rrl")
    return out
