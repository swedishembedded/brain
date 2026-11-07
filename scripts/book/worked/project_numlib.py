# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Reference implementation and acceptance tables for the Volume I project.

Plain Python 3, standard library only, written from the project chapter's own
description. The classes and functions without a leading underscore that take
arguments are the reference library: a strided array with views, matrix
multiplication with an operation count, compensated summation and
log-sum-exp, SplitMix64 with its derived samplers, central-difference
derivatives and gradients, and mean, standard error and bootstrap interval.
Each public function without arguments returns a markdown string that the
chapter embeds between `<!-- worked: project_numlib.<name> -->` markers: the
acceptance tests, every expected value computed here by the library itself.

Randomness: the library carries its own generator (SplitMix64). The only
outside source is a named `bookrng` stream, used for the seeds the tests start
from and for the independent reference values the tests are compared with; no
module in the book carries a seed literal. The one literal state, zero, is the
edge case a generator must handle and not a choice of seed.

Swedish Embedded AB implements solutions for verifiable numerical libraries.
If your team needs expertise in numerical computing and machine learning
engineering, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math

import bookrng

M64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB
EPS = 2.0 ** -52          # machine epsilon of a 64-bit float, the gap above 1


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


def _seed(name):
    """A 64-bit seed for the test called `name`, drawn from its frozen bookrng stream."""
    return bookrng.stream("project_numlib." + name).getrandbits(64)


def _g(v):
    """A float in its shortest form that reads back to the same value."""
    return repr(float(v))


def _list(v):
    return "[" + ", ".join(str(x) for x in v) + "]"


def _flist(v):
    return "[" + ", ".join(_g(x) for x in v) + "]"


def _e(v):
    return f"{v:.3e}"


# ------------------------------------------------- milestone 1: strided arrays

def row_major_strides(shape):
    strides, run = [0] * len(shape), 1
    for k in range(len(shape) - 1, -1, -1):
        strides[k] = run
        run *= shape[k]
    return tuple(strides)


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _indices(shape):
    """All indices of `shape` in row-major order (the last index varies fastest)."""
    idx = [0] * len(shape)
    if numel(shape) == 0:
        return
    while True:
        yield tuple(idx)
        k = len(shape) - 1
        while k >= 0:
            idx[k] += 1
            if idx[k] < shape[k]:
                break
            idx[k] = 0
            k -= 1
        if k < 0:
            return


def broadcast_shape(a, b):
    r = max(len(a), len(b))
    out = [0] * r
    for k in range(1, r + 1):
        x = a[len(a) - k] if k <= len(a) else 1
        y = b[len(b) - k] if k <= len(b) else 1
        if x != y and x != 1 and y != 1:
            raise ValueError(f"shapes {list(a)} and {list(b)} do not broadcast")
        out[r - k] = y if x == 1 else x
    return tuple(out)


class NDArray:
    """Flat shared storage plus (offset, shape, strides); element (i_0 .. i_(R-1)) is
    storage[offset + sum_k i_k * strides[k]]."""

    def __init__(self, storage, shape, strides=None, offset=0):
        self.storage = storage
        self.shape = tuple(shape)
        self.strides = tuple(strides) if strides is not None else row_major_strides(self.shape)
        self.offset = offset

    def _offset_of(self, index):
        if len(index) != len(self.shape):
            raise IndexError("index rank differs from array rank")
        off = self.offset
        for k, i in enumerate(index):
            if not 0 <= i < self.shape[k]:
                raise IndexError("index out of bounds")
            off += i * self.strides[k]
        return off

    def get(self, *index):
        return self.storage[self._offset_of(index)]

    def set(self, index, value):
        self.storage[self._offset_of(index)] = value

    def transpose(self, a=-2, b=-1):
        shape, strides = list(self.shape), list(self.strides)
        shape[a], shape[b] = shape[b], shape[a]
        strides[a], strides[b] = strides[b], strides[a]
        return NDArray(self.storage, shape, strides, self.offset)

    def slice(self, start, stop, step):
        shape, strides, off = [], [], self.offset
        for k in range(len(self.shape)):
            if not (step[k] >= 1 and 0 <= start[k] <= stop[k] <= self.shape[k]):
                raise IndexError("bad slice")
            shape.append((stop[k] - start[k] + step[k] - 1) // step[k])
            strides.append(self.strides[k] * step[k])
            off += start[k] * self.strides[k]
        return NDArray(self.storage, shape, strides, off)

    def broadcast_to(self, shape):
        lead = len(shape) - len(self.shape)
        if lead < 0:
            raise ValueError("target has fewer axes")
        strides = [0] * len(shape)
        for k in range(len(self.shape)):
            if self.shape[k] == shape[lead + k]:
                strides[lead + k] = self.strides[k]
            elif self.shape[k] == 1:
                strides[lead + k] = 0
            else:
                raise ValueError("only an axis of size 1 can stretch")
        return NDArray(self.storage, shape, strides, self.offset)

    def is_contiguous(self):
        want = row_major_strides(self.shape)
        return all(d == 1 or s == w for d, s, w in zip(self.shape, self.strides, want))

    def to_list(self):
        """The elements in row-major order of this view (a copy)."""
        return [self.storage[self._offset_of(i)] for i in _indices(self.shape)]

    def copy(self):
        return NDArray(self.to_list(), self.shape)

    def reshape(self, shape):
        if numel(shape) != numel(self.shape):
            raise ValueError("element counts differ")
        base = self if self.is_contiguous() else self.copy()
        return NDArray(base.storage, shape, None, base.offset)


def add(a, b):
    """Elementwise sum; the shapes must already be equal (broadcasting is explicit)."""
    if a.shape != b.shape:
        raise ValueError(f"shapes {list(a.shape)} and {list(b.shape)} differ; broadcast first")
    return NDArray([x + y for x, y in zip(a.to_list(), b.to_list())], a.shape)


def _matrix(rows):
    m, n = len(rows), len(rows[0])
    return NDArray([v for r in rows for v in r], (m, n))


def _rows(a):
    m, n = a.shape
    flat = a.to_list()
    return [flat[i * n:(i + 1) * n] for i in range(m)]


def _view_row(name, v):
    return (name, _list(v.shape), _list(v.strides), v.offset, _list(v.to_list()),
            "yes" if v.is_contiguous() else "no")


def array_views():
    base = NDArray(list(range(12)), (3, 4))
    t = base.transpose()
    s = base.slice((1, 0), (3, 4), (1, 2))
    ts = t.slice((0, 1), (4, 3), (2, 1))
    col = NDArray([10, 20, 30], (3, 1))
    bc = col.broadcast_to((3, 4))
    rows = [_view_row("base, shape [3, 4], storage 0 .. 11", base),
            _view_row("transpose of base", t),
            _view_row("base, rows 1:3, columns 0:4 step 2", s),
            _view_row("transpose of base, rows 0:4 step 2, columns 1:3", ts),
            _view_row("[3, 1] column (10, 20, 30) broadcast to [3, 4]", bc)]
    out = _table(["view", "shape", "strides", "offset", "elements in row-major order", "contiguous"],
                 rows, "lllrll")
    shared = NDArray(list(range(12)), (3, 4))
    shared.transpose().slice((0, 1), (4, 3), (2, 1)).set((1, 1), 99)
    out += (f"\n\nWrite through a view: set element (1, 1) of the last transposed slice to 99 in a fresh "
            f"copy of base; the storage becomes {_list(shared.storage)}.")
    r = base.transpose().reshape((12,))
    out += (f"\n\nReshape of the transpose to shape [12] reads {_list(r.to_list())} and "
            f"{'shares' if r.storage is base.storage else 'does not share'} storage with base; "
            f"reshape of base itself to [2, 6] "
            f"{'shares' if base.reshape((2, 6)).storage is base.storage else 'does not share'} storage.")
    return out


def array_broadcasting():
    cases = [([3, 4], [4]), ([3, 1], [1, 4]), ([2, 1, 5], [3, 1]), ([], [5]), ([3, 4], [3]),
             ([3, 4], [2, 4]), ([1], [0])]
    rows = []
    for a, b in cases:
        try:
            res = _list(broadcast_shape(tuple(a), tuple(b)))
        except ValueError:
            res = "error"
        rows.append((_list(a), _list(b), res))
    out = _table(["shape a", "shape b", "broadcast shape"], rows, "llr")
    col = NDArray([10, 20, 30], (3, 1))
    row = NDArray([1, 2, 3, 4], (1, 4))
    shape = broadcast_shape(col.shape, row.shape)
    total = add(col.broadcast_to(shape), row.broadcast_to(shape))
    out += (f"\n\nColumn (10, 20, 30) of shape [3, 1] plus row (1, 2, 3, 4) of shape [1, 4], both broadcast "
            f"to {_list(shape)}: {_list(_rows(total))}.")
    try:
        add(col, row)
        refused = "accepted"
    except ValueError:
        refused = "refused"
    out += f"\n\nThe same sum without the explicit broadcast is {refused}."
    return out


# ------------------------------------------------ milestone 2: matrix multiply

def matmul_naive(a, b):
    """C[i, j] = sum over p of A[i, p] B[p, j], through indexed reads. Returns (C, multiply-adds)."""
    (m, k), (k2, n) = a.shape, b.shape
    if k != k2:
        raise ValueError("inner dimensions differ")
    c, ops = [], 0
    for i in range(m):
        for j in range(n):
            acc = 0
            for p in range(k):
                acc += a.get(i, p) * b.get(p, j)
                ops += 1
            c.append(acc)
    return NDArray(c, (m, n)), ops


def matmul(a, b):
    """The loop order i, p, j reading each operand through its strides: the same sums
    accumulated in a different order. Returns (C, multiply-adds)."""
    (m, k), (k2, n) = a.shape, b.shape
    if k != k2:
        raise ValueError("inner dimensions differ")
    c, ops = [0] * (m * n), 0
    for i in range(m):
        for p in range(k):
            aip = a.storage[a.offset + i * a.strides[0] + p * a.strides[1]]
            base = b.offset + p * b.strides[0]
            for j in range(n):
                c[i * n + j] += aip * b.storage[base + j * b.strides[1]]
                ops += 1
    return NDArray(c, (m, n)), ops


def matmul_cases():
    a = _matrix([[1, 2, 3], [4, 5, 6]])
    b = _matrix([[7, 8], [9, 10], [11, 12]])
    c, ops = matmul_naive(a, b)
    out = f"A = {_list(_rows(a))}, B = {_list(_rows(b))}: A B = {_list(_rows(c))}, {ops} multiply-adds."
    fa = NDArray([(3 * i + 2 * j) % 5 - 2 for i in range(4) for j in range(3)], (4, 3))
    fb = NDArray([(j + 5 * k) % 4 - 1 for j in range(3) for k in range(5)], (3, 5))
    c1, o1 = matmul_naive(fa, fb)
    c2, o2 = matmul(fa, fb)
    out += (f"\n\nGenerated inputs A[i, j] = (3 i + 2 j) mod 5 - 2 with shape [4, 3] and B[j, k] = (j + 5 k) "
            f"mod 4 - 1 with shape [3, 5]:\n\n- A = {_list(_rows(fa))}\n- B = {_list(_rows(fb))}\n"
            f"- A B = {_list(_rows(c1))}, {o1} multiply-adds\n"
            f"- both orders of the loops give the same product: {c1.to_list() == c2.to_list()}, "
            f"with {o2} multiply-adds")
    ta = NDArray(fa.to_list(), (4, 3)).transpose()           # a [3, 4] view of the same storage
    sq = NDArray([(2 * i + j) % 3 - 1 for i in range(4) for j in range(2)], (4, 2))
    d1, _ = matmul_naive(ta, sq)
    d2, _ = matmul(ta, sq)
    out += (f"\n- transpose(A) times Q, where Q[i, k] = (2 i + k) mod 3 - 1 has shape [4, 2], multiplied "
            f"through the transposed view without copying: {_list(_rows(d1))}, "
            f"identical in both loop orders: {d1.to_list() == d2.to_list()}")
    return out


def matmul_float_agreement():
    rng = SplitMix64(_seed("matmul_float"))
    a = NDArray([rng.uniform() - 0.5 for _ in range(16 * 16)], (16, 16))
    b = NDArray([rng.uniform() - 0.5 for _ in range(16 * 16)], (16, 16))
    c1, _ = matmul_naive(a, b)
    c2, _ = matmul(a, b)
    worst = max(abs(x - y) for x, y in zip(c1.to_list(), c2.to_list()))
    return (f"Two [16, 16] matrices with entries uniform in [-0.5, 0.5) from the seed {_hex(_seed('matmul_float'))} "
            f"(A first, then B, row-major). The largest difference between the two loop orders is below "
            f"1e-14: {worst < 1e-14}.")


def operation_counts():
    shapes = [(2, 3, 2), (4, 3, 5), (64, 64, 64), (128, 256, 512)]
    rows = [(f"[{m}, {k}] by [{k}, {n}]", f"{m * k * n:,}") for m, k, n in shapes]
    out = _table(["product", "multiply-adds m k n"], rows, "lr")
    p, q, r, s = 10, 100, 5, 50
    left = p * q * r + p * r * s
    right = q * r * s + p * q * s
    out += (f"\n\nChain of A [{p}, {q}], B [{q}, {r}], C [{r}, {s}]: (A B) C costs {p}*{q}*{r} + {p}*{r}*{s} = "
            f"{left:,}; A (B C) costs {q}*{r}*{s} + {p}*{q}*{s} = {right:,}; the ratio is {right / left:.1f}.")
    return out


# --------------------------------------------- milestone 3: stable reductions

def naive_sum(xs):
    s = 0.0
    for x in xs:
        s += x
    return s


def kahan_sum(xs):
    s, c = 0.0, 0.0
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def logsumexp(zs):
    if len(zs) == 0:
        raise ValueError("the log-sum-exp of nothing is undefined")
    m = max(zs)
    if m == -math.inf:
        return -math.inf
    if m == math.inf:
        return math.inf
    s = 0.0
    for z in zs:
        s += math.exp(z - m)
    return m + math.log(s)


def compensated_sums():
    n = 1_000_000
    cases = [("1.0 followed by 10^6 copies of 1e-16", [1.0] + [1e-16] * n),
             ("10^6 copies of 0.1", [0.1] * n),
             ("1.0, then 1e-8 repeated 10^6 times, then -1.0", [1.0] + [1e-8] * n + [-1.0])]
    rows = []
    for name, xs in cases:
        exact = math.fsum(xs)
        scale = math.fsum(abs(x) for x in xs)
        ns, ks = naive_sum(xs), kahan_sum(xs)
        rows.append((name, _g(exact), _g(ns), _e(abs(ns - exact) / scale), _g(ks),
                     _e(abs(ks - exact) / scale)))
    out = _table(["input", "exact sum", "left to right", "error / sum of absolute values", "compensated", "error / sum of absolute values"],
                 rows, "lrrrrr")
    out += ("\n\nThe exact sum is the correctly rounded sum of the stored 64-bit numbers. The compensated "
            "error divided by the sum of absolute values must stay below 2.3e-16 (twice the unit round-off "
            "of 1.1e-16), whatever the length.")
    return out


def logsumexp_cases():
    cases = [[1, 2, 3], [88, 88, 88], [1000, 1000, 1000], [1000, 0, -1000], [-1000, -1000, -1000],
             [-800, -801], [0, -math.inf], [-math.inf, -math.inf], [0.0]]
    rows = []
    for z in cases:
        z = [float(v) for v in z]
        try:
            naive = _g(math.log(sum(math.exp(v) for v in z)))
        except (OverflowError, ValueError):
            naive = "fails"
        rows.append((_flist(z), naive, _g(logsumexp(z))))
    out = _table(["z", "log(sum(exp(z)))", "logsumexp(z)"], rows, "lrr")
    z = [1000.0, 998.0, 990.0, 0.0, -5.0]
    out += (f"\n\nCross-entropy by lse(z) - z[t] for z = {_flist(z)}, t = 1: {_g(logsumexp(z) - z[1])}.")
    out += ("\n\nThe naive column shows only whether the formula as written survives: its failures "
            "(overflow, or the logarithm of zero) are the point. The empty list must be refused with an error.")
    return out


# ------------------------------------------------- milestone 4: the generator

def _mix(z):
    z = ((z ^ (z >> 30)) * C1) & M64
    z = ((z ^ (z >> 27)) * C2) & M64
    return z ^ (z >> 31)


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & M64
        self._spare = None

    def next_u64(self):
        self.state = (self.state + GAMMA) & M64
        return _mix(self.state)

    def uniform(self):
        return (self.next_u64() >> 11) / float(1 << 53)

    def integer(self, n):
        """Uniform in 0 .. n-1 by rejection: raw values below 2^64 mod n are refused."""
        if not 0 < n <= (1 << 64):
            raise ValueError("no integer in an empty range")
        threshold = (1 << 64) % n
        while True:
            r = self.next_u64()
            if r >= threshold:
                return r % n

    def normal(self):
        """Box-Muller; the sine partner of each pair is held for the next call."""
        if self._spare is not None:
            z, self._spare = self._spare, None
            return z
        u1, u2 = self.uniform(), self.uniform()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            j = self.integer(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


class _Scripted(SplitMix64):
    """A source that replays chosen raw 64-bit values (to test the samplers apart from the mixing)."""

    def __init__(self, raws):
        super().__init__(0)
        self.raws = list(raws)
        self.used = 0

    def next_u64(self):
        self.used += 1
        return self.raws.pop(0)


def splitmix_vectors():
    rows = []
    for label, seed in (("zero (the edge case)", 0), ("a drawn seed", _seed("vectors"))):
        r = SplitMix64(seed)
        rows.append((label, _hex(seed), " ".join(_hex(r.next_u64()) for _ in range(4))))
    out = _table(["seed", "value", "the first four next_u64 outputs"], rows, "llr")
    seed = _seed("vectors")
    r = SplitMix64(seed)
    out += f"\n\nFrom the drawn seed, uniform() x 4: {_flist([r.uniform() for _ in range(4)])}."
    r = SplitMix64(seed)
    out += f"\n\nFrom the drawn seed, integer(6) x 12: {_list([r.integer(6) for _ in range(12)])}."
    r = SplitMix64(seed)
    out += f"\n\nFrom the drawn seed, normal() x 5: {_flist([r.normal() for _ in range(5)])}."
    r = SplitMix64(seed)
    out += f"\n\nFrom the drawn seed, shuffle of 0 .. 9: {_list(r.shuffle(list(range(10))))}."
    return out


def sampler_scripts():
    big = 13835058055282163712          # 3 * 2^62: the n of chapter 17 for which a quarter of the raw values are refused
    thr = (1 << 64) % big
    raws = [0, thr - 1, thr, M64]
    s = _Scripted(raws[:3])
    first = s.integer(big)
    rows = [(f"integer({big}) on raw values 0, {thr - 1}, {thr}", f"threshold = 2^64 mod n = {thr}",
             f"returns {first} after consuming {s.used} raw values")]
    s = _Scripted([0, 1, 2, 3])
    rows.append(("integer(3) on raw values 0, 1, 2, 3", "threshold = 2^64 mod 3 = 1",
                 f"returns {s.integer(3)} after consuming {s.used} raw values"))
    s = _Scripted([M64])
    rows.append(("uniform() on raw value 2^64 - 1", "top 53 bits / 2^53", f"returns {_g(s.uniform())}"))
    s = _Scripted([1 << 63, 0])
    rows.append(("uniform() on raw values 2^63, 0", "", f"returns {_g(s.uniform())} then {_g(s.uniform())}"))
    u1, u2 = 0.5, 0.25
    s = _Scripted([int(u1 * (1 << 53)) << 11, int(u2 * (1 << 53)) << 11])
    z1 = s.normal()
    z2 = s.normal()
    rows.append(("normal() twice on the uniforms u1 = 0.5, u2 = 0.25", "radius sqrt(-2 ln 0.5), angle pi / 2",
                 f"returns {_g(z1)} then {_g(z2)} using {s.used} raw values"))
    out = _table(["call", "rule", "result"], rows, "lll")
    return out + "\n\nTolerance for the two normal values: 1e-12 absolute."


def shuffle_scripts():
    # integer(4), integer(3), integer(2) have thresholds 0, 1, 0, so the raw value 4 is kept by all three
    s = _Scripted([4, 4, 4])
    got = s.shuffle(list(range(4)))
    out = (f"shuffle of 0, 1, 2, 3 when the three calls integer(4), integer(3), integer(2) are all fed the raw "
           f"value 4 (the remainders are 0, 1, 0, so the swaps are with j = 0, 1, 0): {_list(got)}, using {s.used} raw values.")
    counts = {}
    rng = SplitMix64(_seed("shuffle_counts"))
    trials = 60000
    for _ in range(trials):
        o = tuple(rng.shuffle([0, 1, 2]))
        counts[o] = counts.get(o, 0) + 1
    rows = [(" ".join(map(str, o)), c, f"{c / trials:.4f}") for o, c in sorted(counts.items())]
    out += (f"\n\nSix orders of three items, {trials:,} shuffles from the seed {_hex(_seed('shuffle_counts'))}; "
            f"each count must lie within 5 standard deviations ({5 * math.sqrt(trials / 6 * 5 / 6):.0f}) of "
            f"{trials // 6:,}:\n\n" + _table(["order", "count", "frequency"], rows, "lrr"))
    return out


def normal_moments():
    n = 100_000
    rng = SplitMix64(_seed("moments"))
    xs = [rng.normal() for _ in range(n)]
    mean = kahan_sum(xs) / n
    var = kahan_sum([(x - mean) ** 2 for x in xs]) / (n - 1)
    within1 = sum(1 for x in xs if abs(x) <= 1) / n
    rows = [("mean", f"{mean:.6f}", "0", f"{4 / math.sqrt(n):.4f} (4 standard errors)"),
            ("variance", f"{var:.6f}", "1", f"{4 * math.sqrt(2 / n):.4f}"),
            ("share with absolute value at most 1", f"{within1:.5f}", "0.68269", "0.0066")]
    return (f"{n:,} values from normal(), seed {_hex(_seed('moments'))}. The measured column is what this "
            f"reference produced; your values must equal it to 1e-9 if your generator is bit-exact, and must "
            f"in any case lie within the stated distance of the exact value.\n\n"
            + _table(["quantity", "measured", "exact", "allowed distance"], rows, "lrrr"))


# ------------------------------------------- milestone 5: derivatives, gradients

def derivative(f, x):
    """Central difference with the step h = eps^(1/3) max(1, |x|); the step is made exactly
    representable relative to x, as chapter 15 recommends."""
    h = EPS ** (1.0 / 3.0) * max(1.0, abs(x))
    xp = x + h
    xm = x - h
    return (f(xp) - f(xm)) / (xp - xm)


def gradient(f, x):
    """Central-difference gradient of f: list -> float, one coordinate at a time."""
    g = []
    for i in range(len(x)):
        h = EPS ** (1.0 / 3.0) * max(1.0, abs(x[i]))
        up, dn = list(x), list(x)
        up[i] = x[i] + h
        dn[i] = x[i] - h
        g.append((f(up) - f(dn)) / (up[i] - dn[i]))
    return g


def close(a, b, rtol, atol):
    return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b))


def derivative_cases():
    cases = [("exp", math.exp, math.exp, 1.0), ("sin", math.sin, math.cos, 0.7),
             ("x^3", lambda x: x ** 3, lambda x: 3 * x * x, 2.0),
             ("log", math.log, lambda x: 1.0 / x, 0.01),
             ("tanh", math.tanh, lambda x: 1 - math.tanh(x) ** 2, -1.5),
             ("exp at a large point", math.exp, math.exp, 20.0)]
    rows = []
    for name, f, df, x in cases:
        d = derivative(f, x)
        h = EPS ** (1.0 / 3.0) * max(1.0, abs(x))
        exact = df(x)
        rows.append((name, _g(x), _e(h), _g(exact), _e(abs(d - exact) / abs(exact))))
    out = _table(["function", "x", "step h", "analytic derivative", "relative error of the check"],
                 rows, "lrrrr")
    out += ("\n\nThe step is eps^(1/3) max(1, |x|) with eps = 2^-52. The check passes when the relative "
            "error is below 1e-6. The log row at x = 0.01 has the largest error because the step (6.06e-6) is "
            "not small next to the distance 0.01 to the singularity of the function; the last row has the "
            "step scaled up with |x|, as the rule demands.")
    return out


# A small function that uses the whole kit: f(x) = log sum_i exp((A x)_i) + (1/2) |x|^2
_GA = [[1.0, 2.0], [-1.0, 0.5], [0.5, -2.0]]
_GX = [0.3, -0.7]


def _gf(x):
    ax = _rows(matmul(NDArray(sum(_GA, []), (3, 2)), NDArray(list(x), (2, 1)))[0])
    return logsumexp([r[0] for r in ax]) + 0.5 * sum(v * v for v in x)


def _gf_analytic(x, bug=False):
    a = NDArray(sum(_GA, []), (3, 2))
    z = [r[0] for r in _rows(matmul(a, NDArray(list(x), (2, 1)))[0])]
    lse = logsumexp(z)
    p = [math.exp(v - lse) for v in z]
    at = a.transpose()
    g = [r[0] for r in _rows(matmul(at, NDArray(p, (3, 1)))[0])]
    return g if bug else [gi + xi for gi, xi in zip(g, x)]


def gradient_check():
    num = gradient(_gf, _GX)
    ana = _gf_analytic(_GX)
    bad = _gf_analytic(_GX, bug=True)
    rows = [(f"{i}", _g(ana[i]), _g(num[i]), _e(abs(ana[i] - num[i]) / abs(ana[i]))) for i in range(2)]
    out = _table(["i", "analytic gradient", "numerical gradient", "relative error"], rows, "lrrr")
    out += (f"\n\nf(x) = log(exp(z_0) + exp(z_1) + exp(z_2)) + (1/2)(x_0^2 + x_1^2), where z = A x for "
            f"A = {_list(_GA)}, at x = {_list(_GX)}: f(x) = {_g(_gf(_GX))}. The analytic gradient is "
            f"A^T p + x, where p_i = exp(z_i - lse(z)).")
    out += (f"\n\nWith close(analytic, numerical, rtol = 1e-6, atol = 1e-8): the correct gradient "
            f"{'passes' if close(ana, num, 1e-6, 1e-8) else 'fails'}; the gradient with the + x term "
            f"forgotten (A^T p alone, {_flist(bad)}) {'passes' if close(bad, num, 1e-6, 1e-8) else 'fails'}.")
    return out


# --------------------------------------------------- milestone 6: statistics

LATENCIES = [18.9, 19.8, 22.7, 23.5, 25.9, 27.5, 28.3, 36.4, 37.2, 37.8, 38.8, 46.9, 52.3, 56.1, 57.4,
             58.5, 62.9, 67.0, 72.8, 77.3, 79.2, 81.2, 84.1, 90.0, 116.7]


def mean(xs):
    return kahan_sum(xs) / len(xs)


def sample_variance(xs):
    m = mean(xs)
    return kahan_sum([(x - m) ** 2 for x in xs]) / (len(xs) - 1)


def standard_error(xs):
    if len(xs) < 2:
        raise ValueError("a standard error needs at least two values")
    return math.sqrt(sample_variance(xs) / len(xs))


def percentile(sorted_vals, q):
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def median(xs):
    return percentile(sorted(xs), 0.5)


def bootstrap_interval(xs, statistic, resamples, level, rng):
    if len(xs) < 2 or resamples < 1000 or not 0.0 < level < 1.0:
        raise ValueError("too few values or resamples, or level outside (0, 1)")
    n = len(xs)
    rep = []
    for _ in range(resamples):
        rep.append(statistic([xs[rng.integer(n)] for _ in range(n)]))
    rep.sort()
    tail = (1.0 - level) / 2.0
    return percentile(rep, tail), percentile(rep, 1.0 - tail)


def statistics_cases():
    xs = LATENCIES
    m, se = mean(xs), standard_error(xs)
    seed = _seed("bootstrap")
    lo, hi = bootstrap_interval(xs, mean, 2000, 0.95, SplitMix64(seed))
    mlo, mhi = bootstrap_interval(xs, median, 2000, 0.95, SplitMix64(seed))
    rows = [("mean", _g(m)), ("sample variance (divisor n - 1)", _g(sample_variance(xs))),
            ("standard error of the mean", _g(se)),
            ("normal-formula 95 percent interval mean -+ 1.96 se", f"[{_g(m - 1.96 * se)}, {_g(m + 1.96 * se)}]"),
            ("median (position 0.5 (n - 1), interpolated)", _g(median(xs))),
            (f"bootstrap interval of the mean, 2000 resamples, seed {_hex(seed)}", f"[{_g(lo)}, {_g(hi)}]"),
            (f"bootstrap interval of the median, same seed", f"[{_g(mlo)}, {_g(mhi)}]")]
    out = (f"Twenty-five latencies in milliseconds: {_flist(xs)}\n\n"
           + _table(["quantity", "value"], rows, "lr"))
    ys = [3.0] * 10
    out += (f"\n\nTen identical values (3.0): standard error {_g(standard_error(ys))}, bootstrap interval "
            f"{_list(bootstrap_interval(ys, mean, 1000, 0.9, SplitMix64(seed)))}.")
    out += ("\n\nThe bootstrap draws, for each resample, n indices in order from integer(n), each used as the "
            "next value of the resample; the sorted statistics are read at positions 0.025 (R - 1) and "
            "0.975 (R - 1) with linear interpolation. Tolerance: 1e-9 for a bit-exact generator.")
    return out


# --------------------------------------------- milestone 7: the integration task

N_POINTS = 50
TRUE_SLOPE, TRUE_INTERCEPT, NOISE = 2.0, 1.0, 0.5


def make_line_data(seed):
    rng = SplitMix64(seed)
    xs = [4.0 * i / (N_POINTS - 1) for i in range(N_POINTS)]
    ys = [TRUE_SLOPE * x + TRUE_INTERCEPT + NOISE * rng.normal() for x in xs]
    return xs, ys


def line_loss(p, xs, ys):
    return mean([(p[0] * x + p[1] - y) ** 2 for x, y in zip(xs, ys)])


def line_gradient(p, xs, ys):
    r = [p[0] * x + p[1] - y for x, y in zip(xs, ys)]
    return [2.0 * mean([ri * x for ri, x in zip(r, xs)]), 2.0 * mean(r)]


def fit_line_closed(xs, ys):
    mx, my = mean(xs), mean(ys)
    sxy = kahan_sum([(x - mx) * (y - my) for x, y in zip(xs, ys)])
    sxx = kahan_sum([(x - mx) ** 2 for x in xs])
    slope = sxy / sxx
    return slope, my - slope * mx


def fit_line_steps(xs, ys, rate=0.05, steps=3000):
    p = [0.0, 0.0]
    for _ in range(steps):
        g = line_gradient(p, xs, ys)
        p = [p[0] - rate * g[0], p[1] - rate * g[1]]
    return p


def integration_run():
    seed = _seed("integration")
    xs, ys = make_line_data(seed)
    start = [0.0, 0.0]
    ana = line_gradient(start, xs, ys)
    num = gradient(lambda q: line_loss(q, xs, ys), start)
    out = (f"Data: n = {N_POINTS} points, x_i = 4 i / 49, y_i = 2 x_i + 1 + 0.5 z_i, where z_0, z_1, ... are the "
           f"first fifty normal() draws of the generator seeded with {_hex(seed)}.\n\n"
           f"- first three points: {_flist([xs[0], ys[0]])}, {_flist([xs[1], ys[1]])}, {_flist([xs[2], ys[2]])}\n"
           f"- mean of y: {_g(mean(ys))}\n"
           f"- gradient of the mean squared residual at (0, 0): analytic {_flist(ana)}, "
           f"numerical {_flist(num)}, check passes: {close(ana, num, 1e-6, 1e-8)}")
    p = fit_line_steps(xs, ys)
    q = fit_line_closed(xs, ys)
    ana = line_gradient(p, xs, ys)
    out += (f"\n- after 3000 steps of size 0.05 against the gradient from (0, 0): slope {_g(p[0])}, "
            f"intercept {_g(p[1])}, mean squared residual {_g(line_loss(p, xs, ys))}\n"
            f"- the closed-form slope and intercept (slope = sum (x - mean x)(y - mean y) / sum (x - mean x)^2): "
            f"{_g(q[0])}, {_g(q[1])}; largest difference from the stepped values {_e(max(abs(p[0] - q[0]), abs(p[1] - q[1])))} "
            f"(must be below 1e-6)\n"
            f"- gradient at the stepped values: {_flist(ana)} (both entries below 1e-6 in absolute value)")
    pairs = list(zip(xs, ys))

    def slope_of(sample):
        return fit_line_closed([s[0] for s in sample], [s[1] for s in sample])[0]

    lo, hi = bootstrap_interval(pairs, slope_of, 2000, 0.95, SplitMix64(_seed("integration.bootstrap")))
    out += (f"\n- 95 percent bootstrap interval of the slope, resampling whole (x, y) pairs, 2000 resamples, generator "
            f"seeded with {_hex(_seed('integration.bootstrap'))}: [{_g(lo)}, {_g(hi)}]; it "
            f"{'contains' if lo <= TRUE_SLOPE <= hi else 'does not contain'} the true slope 2.")
    return out
