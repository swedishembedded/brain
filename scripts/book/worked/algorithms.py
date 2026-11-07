# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Algorithms, Cost and Complexity".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: algorithms.<name> -->` markers. Functions whose names start with
an underscore (or that take arguments) are shared with the chapter's figures,
so a picture and the table beside it cannot disagree.

Every algorithm is instrumented: a counter is incremented at exactly the
operation the chapter says it counts (a comparison of two token ids, a hash
evaluation, a rank lookup), so the tables are measurements of the algorithms as
written, not evaluations of a formula. Where a formula is printed beside a
count (n(n-1)/2, n log2 n - n + 1) the two are computed separately and the
agreement is the check.

Swedish Embedded AB implements solutions for predictable cost in machine
learning pipelines. If your team needs expertise in algorithm design, data
structures and the performance of sorting, scanning and hashing on CPUs and
GPUs, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import hashlib
import heapq
import functools
import itertools
import math
import struct
from fractions import Fraction

import bookrng


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _n(v):
    """An integer with thousands separators."""
    return f"{v:,}"


def _sci(v, digits=2):
    return f"{v:.{digits}e}"


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _list(xs):
    return "[" + ", ".join(str(x) for x in xs) + "]"


# ------------------------------------------------- duplicates, three methods

def _seed_note(name):
    """The sentence every randomised example prints so that a reader can seed the same generator."""
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (the seed of the name `{name}`)."


def _distinct_draws(rng, bound, count):
    """The first `count` distinct values of repeated rng.integer(bound) draws; a repeat is discarded."""
    seen, out = set(), []
    while len(out) < count:
        v = rng.integer(bound)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


DUP_NAME = "algorithms.duplicates_growth"
DUP_VOCAB = 50_000
DUP_MAX = 4096


@functools.lru_cache(maxsize=None)
def _all_ids():
    return tuple(_distinct_draws(bookrng.generator(DUP_NAME), DUP_VOCAB, DUP_MAX))


def _distinct_ids(n):
    """The first n of the 4096 distinct token ids drawn in order as integer(50000) with repeats discarded."""
    return list(_all_ids()[:n])


def _dup_pairs(ids):
    """Compare every pair. Returns (duplicate id or None, comparisons)."""
    comparisons = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            comparisons += 1
            if ids[i] == ids[j]:
                return ids[i], comparisons
    return None, comparisons


def _merge(left, right, counter):
    """Merge two sorted lists; counter[0] counts element comparisons."""
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        counter[0] += 1
        if right[j] < left[i]:
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def _merge_sort(xs, counter):
    """Top-down merge sort, stable, counting element comparisons."""
    if len(xs) <= 1:
        return list(xs)
    mid = len(xs) // 2
    return _merge(_merge_sort(xs[:mid], counter), _merge_sort(xs[mid:], counter), counter)


def _dup_sort(ids):
    """Sort, then compare neighbours. Returns (duplicate or None, comparisons, extra memory)."""
    counter = [0]
    ordered = _merge_sort(ids, counter)
    for k in range(len(ordered) - 1):
        counter[0] += 1
        if ordered[k] == ordered[k + 1]:
            return ordered[k], counter[0], len(ids)
    return None, counter[0], len(ids)


HASH_MULTIPLIER = 2654435761          # odd, close to 2^32 / golden ratio (Knuth's multiplicative hash)


def _hash_bucket(x, bits):
    """Multiplicative hash of a 32-bit key into 2^bits buckets."""
    return ((x * HASH_MULTIPLIER) & 0xFFFFFFFF) >> (32 - bits)


class _ChainTable:
    """A hash table with separate chaining that counts key comparisons."""

    def __init__(self, bits, bucket=_hash_bucket):
        self.bits = bits
        self.bucket = bucket
        self.chains = [[] for _ in range(1 << bits)]
        self.comparisons = 0
        self.evaluations = 0

    def _chain(self, key):
        self.evaluations += 1
        return self.chains[self.bucket(key, self.bits)]

    def find(self, key):
        """Position (0-based) of key in its chain, or None. Counts one comparison per entry visited."""
        for pos, stored in enumerate(self._chain(key)):
            self.comparisons += 1
            if stored == key:
                return pos
        return None

    def insert(self, key):
        self._chain(key).append(key)


def _dup_hash(ids):
    """Remember every id seen in a hash table. Returns (duplicate or None, evaluations + comparisons, memory)."""
    bits = max(1, math.ceil(math.log2(2 * max(1, len(ids)))))
    table = _ChainTable(bits)
    for x in ids:
        if table.find(x) is not None:
            return x, table.evaluations + table.comparisons, (1 << bits) + len(ids)
        table.insert(x)
    return None, table.evaluations + table.comparisons, (1 << bits) + len(ids)


def duplicate_counts(ns):
    """Operation counts of the three methods on lists of n distinct ids (no duplicate to stop early)."""
    rows = []
    for n in ns:
        ids = _distinct_ids(n)
        pairs = _dup_pairs(ids)
        ordered = _dup_sort(ids)
        hashed = _dup_hash(ids)
        assert pairs[0] is None and ordered[0] is None and hashed[0] is None
        rows.append((n, pairs[1], ordered[1], hashed[1], ordered[2], hashed[2]))
    return rows


def duplicates_small():
    ids = [4021, 87, 15032, 9, 87, 311, 40022, 7, 311, 12, 6, 15032]
    d_pairs, c_pairs = _dup_pairs(ids)
    d_sort, c_sort, _ = _dup_sort(ids)
    d_hash, c_hash, _ = _dup_hash(ids)
    sort_counter = [0]
    _merge_sort(ids, sort_counter)
    sort_tests = c_sort - sort_counter[0]
    rows = [
        ("compare all pairs", d_pairs, c_pairs, "stops at the first equal pair it meets"),
        ("sort, then compare neighbours", d_sort, c_sort, "merge sort comparisons plus neighbour tests up to the first equal pair"),
        ("remember ids in a hash table", d_hash, c_hash, "hash evaluations plus key comparisons; stops at the first repeat"),
    ]
    note = (f"The list, in order, has n = {len(ids)} ids: {_list(ids)}. The ids that occur twice are "
            f"{sorted({x for x in ids if ids.count(x) > 1})}. The methods may report different repeated ids because "
            "they visit the list in different orders; each reported id really is repeated. "
            "The hash table has m = 32 buckets (2n = 24 rounded up to a power of two) and the hash of the chapter; "
            "the all-pairs count is the number of pairs (i, j) with i < j visited up to and including the first "
            "equal pair. The sort count is the element comparisons of the chapter's merge sort of the whole list "
            f"({c_sort - sort_tests}) plus the neighbour tests in the sorted list up to and including the first equal "
            f"pair ({sort_tests} here, not all n - 1 = {len(ids) - 1}). The hash count is one hash evaluation per "
            "lookup, plus the key comparisons along the chain, and the loop stops when a lookup finds the id; "
            "an insertion costs one more evaluation and no comparison.")
    return note + "\n\n" + _table(["method", "reports", "operations counted", "what is counted"], rows, "llrl")


def duplicates_growth():
    ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = duplicate_counts(ns)
    body = [(_n(n), _n(p), _n(s), _n(h)) for n, p, s, h, _, _ in rows]
    return ("Operations counted on n distinct token ids (no repeat, so no method can stop early). "
            f"The ids: {_n(DUP_VOCAB)} possible values 0 to {DUP_VOCAB - 1}; one list of {_n(DUP_MAX)} distinct ids is made "
            "by calling `integer(50000)` repeatedly and discarding any value already drawn, and the list for n is its "
            f"first n ids. {_seed_note(DUP_NAME)} "
            "All pairs counts one comparison per pair (i, j), i < j. Sort counts the element comparisons of the merge "
            "sort of the chapter (a merge compares only while both halves are non-empty) plus the n - 1 tests of "
            "neighbours in the sorted list. The hash set has m buckets, the smallest power of two that is at least 2n "
            "(exactly 2n here, because n is a power of two), chaining and the multiplicative hash; for each id it "
            "evaluates the hash once for the lookup, compares the id with every entry of that bucket's chain, and "
            "evaluates the hash once more to insert, which costs no comparison. The count is hash evaluations plus "
            "key comparisons.\n\n"
            + _table(["n", "all pairs: comparisons", "sort: comparisons", "hash set: evaluations + comparisons"], body))


def duplicates_memory():
    ns = [16, 256, 4096]
    rows = duplicate_counts(ns)
    body = [(_n(n), "O(1): 3 variables", _n(sm), _n(hm)) for n, _, _, _, sm, hm in rows]
    return _table(["n", "all pairs: extra cells", "merge sort: extra cells", "hash set: extra cells"], body)


# ------------------------------------------------------------ algorithm, invariant

def invariant_max():
    x = [3, 7, 2, 9, 4]
    m = x[0]
    rows = [("start", 1, m, _list(x[:1]), max(x[:1]) == m)]
    for i in range(1, len(x)):
        if x[i] > m:
            m = x[i]
        rows.append((f"after i = {i}", i + 1, m, _list(x[:i + 1]), max(x[:i + 1]) == m))
    return ("Maximum of x = " + _list(x) + ", checking the invariant after every step.\n\n"
            + _table(["step", "elements examined", "m", "x[0 .. examined]", "m = max of examined?"], rows, "lrrlr"))


# ---------------------------------------------------- counting, dominant term

def _dense_layer_ops(n, m):
    """Operation counts of one dense layer y = relu(W x + b), W: [m, n], counted in loops."""
    mult_adds = bias_adds = activations = 0
    for _i in range(m):
        for _j in range(n):
            mult_adds += 1
        bias_adds += 1
        activations += 1
    return mult_adds, bias_adds, activations


def dominant_term():
    rows = []
    for n in [4, 16, 64, 256, 1024]:
        ma, ba, ac = _dense_layer_ops(n, n)
        total = ma + ba + ac
        assert total == n * n + 2 * n
        rows.append((_n(n), _n(ma), _n(ba + ac), _n(total), f"{ma / total:.4%}"))
    return ("One dense layer with n inputs and n outputs, operations counted inside the loops.\n\n"
            + _table(["n", "multiply-adds", "bias adds + activations", "total", "share of the n^2 term"], rows))


def asymptotic_ratios():
    ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    counts = duplicate_counts(ns)
    rows, quad, lin, nlogn = [], [], [], []
    for n, p, s, h, _, _ in counts:
        a, b, c = p / n ** 2, s / (n * math.log2(n)), h / n
        quad.append(a)
        nlogn.append(b)
        lin.append(c)
        rows.append((_n(n), f"{a:.4f}", f"{b:.4f}", f"{c:.4f}"))
    rows.append(("smallest ratio", f"{min(quad):.4f}", f"{min(nlogn):.4f}", f"{min(lin):.4f}"))
    rows.append(("largest ratio", f"{max(quad):.4f}", f"{max(nlogn):.4f}", f"{max(lin):.4f}"))
    return ("Each count divided by the candidate growth rate. A column that stays between two positive constants "
            "shows that the candidate fits.\n\n"
            + _table(["n", "pairs / n^2", "sort / (n log2 n)", "hash / n"], rows))


def wrong_fits():
    ns = [16, 64, 256, 1024, 4096]
    counts = duplicate_counts(ns)
    rows = [(_n(n), f"{p / n:.1f}", f"{p / n ** 3:.5f}", f"{s / n:.2f}", f"{s / n ** 2:.5f}")
            for n, p, s, _, _, _ in counts]
    return _table(["n", "pairs / n", "pairs / n^3", "sort / n", "sort / n^2"], rows)


# ------------------------------------------------------------- growth classes

def _largest_n(cost, budget):
    """Largest integer n with cost(n) <= budget (cost increasing)."""
    lo, hi = 1, 2
    while cost(hi) <= budget:
        hi *= 2
        if hi > 10 ** 30:
            return None
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if cost(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def _human_time(seconds):
    for unit, size in (("years", 365 * 86400), ("days", 86400), ("hours", 3600), ("minutes", 60)):
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} seconds"


def growth_values():
    ns = [10, 20, 40, 80]
    cols = [("log2 n", lambda n: math.log2(n)), ("n", lambda n: n), ("n log2 n", lambda n: n * math.log2(n)),
            ("n^2", lambda n: n ** 2), ("n^3", lambda n: n ** 3), ("2^n", lambda n: 2 ** n)]
    rows = []
    for n in ns:
        cells = []
        for _, f in cols:
            v = f(n)
            cells.append(f"{v:.1f}" if v < 100 else (_n(round(v)) if v < 1e7 else _sci(v)))
        rows.append((n, *cells))
    return _table(["n"] + [c[0] for c in cols], rows)


def growth_budget():
    rate = 10 ** 9      # assumption: one billion simple operations per second
    classes = [("n", lambda n: n), ("n log2 n", lambda n: n * math.log2(n)), ("n^2", lambda n: n ** 2),
               ("n^3", lambda n: n ** 3), ("2^n", lambda n: 2 ** n)]
    rows = []
    for name, f in classes:
        cells = []
        for seconds in (1, 3600, 86400):
            best = _largest_n(f, rate * seconds)
            cells.append(_n(best) if best < 10 ** 12 else _sci(best))
        rows.append((name, *cells))
    t = [(f"n = {n}", _human_time(2 ** n / rate)) for n in (30, 50, 70, 100)]
    return (_table(["cost", "largest n in 1 second", "in 1 hour", "in 1 day"], rows)
            + "\n\nTime for 2^n operations at the same rate:\n\n" + _table(["n", "time"], t, "lr"))


def known_costs():
    rows = []
    for n in (1024, 4096):
        dft = n * n
        fft = (n // 2) * int(math.log2(n))
        matvec = n * n
        matmul = n ** 3
        rows.append((_n(n), _n(dft), _n(fft), f"{dft / fft:.0f}", _n(matvec), _n(matmul)))
    return _table(["n", "DFT multiplications n^2", "FFT multiplications (n/2) log2 n", "ratio",
                   "matrix-vector multiply-adds n^2", "matrix-matrix multiply-adds n^3"], rows)


# ------------------------------------------------------ memory: store or recompute

def _chain_backward(layers, stride):
    """Backward pass over a chain of `layers` layers when only every stride-th activation is kept.

    Layer i computes a_i from a_{i-1}; the backward step of layer i needs a_{i-1}; the output a_L is never
    stored or needed. Forward pass: a_0 is stored, then a_s, a_2s, ... (all indices below L that are multiples of
    s) as the layers run; every layer evaluation counts. Backward pass: segments [k s, min((k + 1) s, L)) from the
    last to the first; entering one, recompute a_{k s + 1} ... a_{end - 1} from the checkpoint a_{k s} (one
    evaluation each), run the backward steps of the segment's layers, and drop the segment's arrays, including
    its checkpoint a_{k s}, when its last backward step is done. The peak is the largest number of arrays held at
    once, measured after every store.
    Returns (peak number of stored activations, layer evaluations in total).
    """
    stored, evaluations, peak = set(), 0, 0
    for i in range(layers):                  # forward pass: a_0 is the input
        if i % stride == 0:
            stored.add(i)
            peak = max(peak, len(stored))
        evaluations += 1
    starts = list(range(0, layers, stride))
    for start in reversed(starts):
        end = min(start + stride, layers)
        for j in range(start + 1, end):      # recompute the segment from its checkpoint
            stored.add(j)
            evaluations += 1
            peak = max(peak, len(stored))
        for j in range(end - 1, start - 1, -1):
            assert j in stored                # the backward step of layer j+1 reads a_j
            stored.discard(j)
    assert not stored
    return peak, evaluations


def checkpoint_tradeoff():
    layers = 64
    rows = []
    for stride in (1, 2, 4, 6, 8, 16, 32, 64):
        peak, evals = _chain_backward(layers, stride)
        rows.append((stride, peak, evals, f"{evals / layers:.2f}"))
    best = min(range(1, layers + 1), key=lambda s: (_chain_backward(layers, s)[0], s))
    return (f"A chain of {layers} layers a_1 ... a_{layers}, input a_0. Stride s keeps the checkpoints a_0, a_s, a_2s, ... "
            f"during the forward pass (the output a_{layers} is never stored); entering a segment of the backward "
            "pass the arrays between its checkpoint and the next are recomputed from the checkpoint, and the segment's "
            "arrays, checkpoint included, are dropped when its backward step is finished. The last segment is shorter "
            f"when s does not divide {layers}. The peak counts the arrays held at the same moment, a_0 included.\n\n"
            + _table(["stride s", "peak stored activations", "layer evaluations", "relative to a plain forward pass"], rows)
            + f"\n\nOver every stride from 1 to {layers}, the smallest peak is {_chain_backward(layers, best)[0]} "
            f"stored activations, first reached at s = {best}.")


# ------------------------------------------------------------------ reductions

def _fold(op, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = op(acc, x)
    return acc


def _tree_reduce(op, xs):
    """Combine neighbours in pairs, layer by layer (an odd last element is carried up unchanged)."""
    layer = list(xs)
    while len(layer) > 1:
        nxt = [op(layer[2 * i], layer[2 * i + 1]) for i in range(len(layer) // 2)]
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    return layer[0]


def _argmax_op(a, b):
    """Combine (value, index) pairs; ties go to the smaller index, which makes the operator associative."""
    return b if b[0] > a[0] else a


def _strided_fold(op, xs, lanes):
    """lanes independent running folds over every lanes-th element, then combined left to right."""
    partial = [_fold(op, xs[k::lanes]) for k in range(lanes)]
    return _fold(op, partial)


def reduction_orders():
    gen_name = "algorithms.reduction_orders"
    rng = bookrng.generator(gen_name)
    xs = []
    for _ in range(4096):
        a = rng.uniform()
        k = rng.integer(7)
        xs.append(_f32((2.0 * a - 1.0) * 10.0 ** (k - 3)))
    add32 = lambda a, b: _f32(a + b)       # noqa: E731
    exact = sum(Fraction(x) for x in xs)
    rows = []
    for name, value in (("left to right", _fold(add32, xs)), ("tree (pairs, layer by layer)", _tree_reduce(add32, xs)),
                        ("4 interleaved lanes", _strided_fold(add32, xs, 4)),
                        ("32 interleaved lanes", _strided_fold(add32, xs, 32))):
        rel = abs(Fraction(value) - exact) / abs(exact)
        rows.append((name, f"{value:.4f}", f"{float(rel):.2e}"))
    ints = [rng.integer(2001) - 1000 for _ in range(1024)]
    int_vals = {_fold(lambda a, b: a + b, ints), _tree_reduce(lambda a, b: a + b, ints),
                _strided_fold(lambda a, b: a + b, ints, 32)}
    pairs = [(x, i) for i, x in enumerate(ints)]
    arg_vals = {_fold(_argmax_op, pairs), _tree_reduce(_argmax_op, pairs), _strided_fold(_argmax_op, pairs, 32)}
    mx_vals = {_fold(max, ints), _tree_reduce(max, ints), _strided_fold(max, ints, 32)}
    sub = [10, 3, 2, 1]
    out = [f"Sum of 4096 signed 32-bit numbers of mixed magnitude (exact value of the stored numbers: {float(exact):.4f}). "
           f"{_seed_note(gen_name)} Each number draws a = `uniform()` then k = `integer(7)` and is "
           "f32((2a - 1) * 10^(k - 3)); the 1024 integers that follow, from the same generator, are "
           "`integer(2001)` - 1000. The orders: left to right; the tree of the chapter (adjacent pairs layer by "
           "layer, an odd last element carried up); L interleaved lanes, meaning lane j (0 <= j < L) adds the "
           "elements j, j + L, j + 2L, ... left to right and the L lane totals are then added left to right. "
           "Every addition is rounded to 32 bits. The relative error is against the exact rational sum.\n",
           _table(["order", "32-bit sum", "relative error"], rows),
           "\nThe same orders (left to right, tree, 32 lanes) on 1024 integers in [-1000, 1000] and on an operator that is not associative:\n",
           _table(["reduction", "orders tried", "distinct results"],
                  [("integer sum", 3, len(int_vals)), ("max", 3, len(mx_vals)),
                   ("(max, index) with ties to the smaller index", 3, len(arg_vals)),
                   ("subtraction on [10, 3, 2, 1]: left to right, tree", 2,
                    len({_fold(lambda a, b: a - b, sub), _tree_reduce(lambda a, b: a - b, sub)}))], "lrr"),
           f"\nSubtraction gives {_fold(lambda a, b: a - b, sub)} left to right and "
           f"{_tree_reduce(lambda a, b: a - b, sub)} as a tree.",
           f"The largest integer is {next(iter(arg_vals))[0]}; it occurs {ints.count(next(iter(arg_vals))[0])} times, "
           f"and every order reports the first occurrence, at index {next(iter(arg_vals))[1]}."]
    return "\n".join(out)


def tree_depth():
    rows = []
    for n in (8, 1000, 1024, 10 ** 6, 2 ** 20):
        layer, levels, ops = n, 0, 0
        while layer > 1:
            ops += layer // 2
            layer = layer // 2 + layer % 2
            levels += 1
        rows.append((_n(n), _n(n - 1), levels, _n(ops)))
    return _table(["n", "left-to-right steps in sequence", "tree layers", "additions in the tree"], rows)


# ---------------------------------------------------------------- prefix sums

def _scan(xs, exclusive=False):
    out, acc = [], 0
    for x in xs:
        if exclusive:
            out.append(acc)
            acc += x
        else:
            acc += x
            out.append(acc)
    return out


SCAN_DEMO = [3, 1, 4, 1, 5, 9, 2, 6]


def _doubling_rounds(x):
    """The array after each round of the doubling scan: [(offset, array, additions)]."""
    cur, offset, rounds = list(x), 1, []
    while offset < len(x):
        cur = [cur[i] + cur[i - offset] if i >= offset else cur[i] for i in range(len(x))]
        rounds.append((offset, cur, len(x) - offset))
        offset *= 2
    return rounds


def scan_basics():
    x = SCAN_DEMO
    inc, exc = _scan(x), _scan(x, True)
    rows = [("x", *x), ("inclusive scan", *inc), ("exclusive scan", *exc)]
    out = [_table(["", *[f"i={i}" for i in range(len(x))]], rows)]
    rounds = _doubling_rounds(x)
    assert rounds[-1][1] == inc
    steps = [(f"offset {off}", *arr, adds) for off, arr, adds in rounds]
    out.append("\nThe doubling scan: each round adds the element `offset` places to the left, all at once.\n")
    out.append(_table(["after", *[f"i={i}" for i in range(len(x))], "additions"], steps))
    out.append(f"\nTotal additions: {sum(r[2] for r in rounds)} in {len(rounds)} rounds, against {len(x) - 1} additions "
               f"in {len(x) - 1} sequential steps for the left-to-right scan.")
    return "\n".join(out)


def _inverse_cdf(cdf, u):
    """Smallest index i with cdf[i] > u, by binary search. Returns (i, steps).

    The interval [lo, hi] of indices that may hold the answer starts as [0, K - 1]: the last index is known to
    qualify because cdf[K - 1] = 1 > u. One step is one loop pass, which compares cdf[mid] with u for
    mid = (lo + hi) // 2 and sets hi = mid when cdf[mid] > u, otherwise lo = mid + 1. The search stops when
    lo = hi.
    """
    lo, hi, steps = 0, len(cdf) - 1, 0
    while lo < hi:
        steps += 1
        mid = (lo + hi) // 2
        if cdf[mid] > u:
            hi = mid
        else:
            lo = mid + 1
    return lo, steps


def scan_cdf():
    p = [Fraction(10, 100), Fraction(35, 100), Fraction(5, 100), Fraction(25, 100), Fraction(15, 100), Fraction(10, 100)]
    cdf = _scan(p)
    assert cdf[-1] == 1
    rows = [(f"token {i}", f"{float(p[i]):.2f}", f"{float(cdf[i]):.2f}") for i in range(len(p))]
    draws = [Fraction(2, 100), Fraction(12, 100), Fraction(50, 100), Fraction(70, 100), Fraction(97, 100)]
    drows = [(f"{float(u):.2f}", *_inverse_cdf(cdf, u)) for u in draws]
    name = "algorithms.scan_cdf"
    rng = bookrng.generator(name)
    counts = [0] * len(p)
    cdf_f = [float(c) for c in cdf]
    trials = 20000
    for _ in range(trials):
        counts[_inverse_cdf(cdf_f, rng.uniform())[0]] += 1
    frows = [(f"token {i}", f"{float(p[i]):.2f}", _n(counts[i]), f"{counts[i] / trials:.4f}") for i in range(len(p))]
    return (_table(["token", "probability", "cumulative (inclusive scan)"], rows) + "\n\n"
            + "Binary search for the first index i with F_i > u. The search interval [lo, hi] starts as [0, K - 1] "
            "with K = 6 (the last index always qualifies, because F_(K-1) = 1 > u). One step is one pass of the loop "
            "\"while lo < hi\": mid = (lo + hi) // 2; if F_mid > u then hi = mid, else lo = mid + 1. The answer is lo "
            "when the loop ends, and the table counts the passes.\n\n"
            + _table(["u", "token drawn", "binary-search steps"], drows) + "\n\n"
            + f"{_n(trials)} draws, each `uniform()` from the book's SplitMix64 with seed "
            f"{bookrng.seed_of(name):#018x}, taken in order:\n\n"
            + _table(["token", "probability", "draws", "frequency"], frows))


def scan_offsets():
    lengths = [5, 3, 7, 2]
    offsets = _scan(lengths, True)
    total = sum(lengths)
    buf = [None] * total
    for s, (off, ln) in enumerate(zip(offsets, lengths)):
        for t in range(ln):
            buf[off + t] = f"{s}.{t}"
    rows = [(s, lengths[s], offsets[s], offsets[s] + lengths[s]) for s in range(len(lengths))]
    return (_table(["sequence", "length", "offset (exclusive scan)", "end"], rows) + "\n\n"
            + f"The packed buffer has {total} slots; slot k holds (sequence.position):\n\n"
            + "`" + " ".join(buf) + "`")


def block_scan():
    n, block = 16, 4
    x = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5]
    blocks = [x[i:i + block] for i in range(0, n, block)]
    local = [_scan(b, True) for b in blocks]
    sums = [sum(b) for b in blocks]
    base = _scan(sums, True)
    final = [v + base[k] for k, loc in enumerate(local) for v in loc]
    assert final == _scan(x, True)
    rows = [(k, _list(blocks[k]), _list(local[k]), sums[k], base[k], _list([v + base[k] for v in local[k]]))
            for k in range(len(blocks))]
    return (f"The array has {n} elements and the block size is {block}, so there are {n // block} blocks; when the length "
            "is not a multiple of the block size the last block is shorter and is scanned in the same way.\n\n"
            + _table(["block", "input", "1: scan inside the block", "block total", "2: scan of totals", "3: add it back"],
                   rows, "lllrrl")
            + f"\n\nConcatenated, the last column is {_list(final)}, the exclusive scan of the whole array.")


# --------------------------------------------------------------------- sorting

def _post_order(nodes, ident=0):
    """Node ids in the order the merges finish: the left subtree, then the right, then the node itself."""
    kids = [n[0] for n in nodes if n[1] == ident]
    order = []
    for k in kids:
        order.extend(_post_order(nodes, k))
    return order + [ident]


def _merge_tree(values):
    """Merge sort recursion as nodes: (id, parent id or None, array before, array after, comparisons here)."""
    nodes = []

    def go(xs, parent):
        ident = len(nodes)
        nodes.append([ident, parent, list(xs), None, 0])
        if len(xs) > 1:
            mid = len(xs) // 2
            left = go(xs[:mid], ident)
            right = go(xs[mid:], ident)
            counter = [0]
            nodes[ident][3] = _merge(nodes[left][3], nodes[right][3], counter)
            nodes[ident][4] = counter[0]
        else:
            nodes[ident][3] = list(xs)
        return ident

    go(list(values), None)
    return [tuple(n) for n in nodes]


MERGE_DEMO = [5, 2, 9, 1, 7, 3, 8, 4]


def merge_sort_trace():
    nodes = _merge_tree(MERGE_DEMO)
    rows = [(f"{_list(nodes[i][2])}", f"{_list(nodes[i][3])}", nodes[i][4])
            for i in _post_order(nodes) if len(nodes[i][2]) > 1]
    total = sum(n[4] for n in nodes)
    counter = [0]
    assert _merge_sort(MERGE_DEMO, counter) == sorted(MERGE_DEMO) and counter[0] == total
    return ("Merges in the order the algorithm performs them (the left half is sorted before the right half).\n\n"
            + _table(["merge of the range", "result", "comparisons"], rows, "llr")
            + f"\n\nTotal: {total} comparisons for n = {len(MERGE_DEMO)}.")


def _merge_worst(n):
    """Worst-case comparisons W(n) = W(floor(n/2)) + W(ceil(n/2)) + n - 1, W(1) = 0."""
    if n <= 1:
        return 0
    return _merge_worst(n // 2) + _merge_worst(n - n // 2) + n - 1


def _insertion_sort_comparisons(xs):
    xs, count = list(xs), 0
    for i in range(1, len(xs)):
        j = i
        while j > 0:
            count += 1
            if xs[j - 1] > xs[j]:
                xs[j - 1], xs[j] = xs[j], xs[j - 1]
                j -= 1
            else:
                break
    return count


def sorting_counts():
    name = "algorithms.sorting_counts"
    rng = bookrng.generator(name)
    rows = []
    for n in (8, 16, 64, 256, 1024, 4096):
        merge_total = insert_total = 0
        trials = 20
        for _ in range(trials):
            perm = list(range(n))
            rng.shuffle(perm)
            c = [0]
            assert _merge_sort(perm, c) == sorted(perm)
            merge_total += c[0]
            if n <= 1024:
                insert_total += _insertion_sort_comparisons(perm)
        bound = math.ceil(math.log2(math.factorial(n)))
        worst = _merge_worst(n)
        assert math.log2(math.factorial(n)) <= merge_total / trials <= worst
        rows.append((_n(n), _n(bound), f"{merge_total / trials:,.0f}", _n(worst), f"{n * math.log2(n):,.0f}",
                     f"{insert_total / trials:,.0f}" if n <= 1024 else "not run"))
    return ("Mean of 20 random permutations per row. The lower bound is the smallest number of comparisons "
            "any comparison sort needs in the worst case: ceil(log2 n!). "
            f"{_seed_note(name)} Rows in the order of the table, 20 permutations each: the permutation starts as "
            "[0, 1, ..., n - 1] and is shuffled with the book's `shuffle` (Fisher-Yates from the end, one "
            "`integer(i + 1)` per position i = n - 1 down to 1). Insertion sort inserts each element into the sorted "
            "prefix by comparing from the right and stops at the first element not larger; every such comparison "
            "counts. The same permutations feed merge sort and insertion sort.\n\n"
            + _table(["n", "lower bound ceil(log2 n!)", "merge sort (mean)", "merge sort worst W(n)", "n log2 n",
                      "insertion sort (mean)"], rows))


def decision_tree3():
    rows = []
    for perm in itertools.permutations([1, 2, 3]):
        c = [0]
        out = _merge_sort(list(perm), c)
        assert out == [1, 2, 3]
        rows.append((_list(perm), c[0]))
    counts = [r[1] for r in rows]
    table = _table(["input order", "comparisons used by merge sort"], rows, "lr")
    bounds = _table(["n", "orderings n!", "log2 n!", "ceil(log2 n!)"],
                    [(n, _n(math.factorial(n)), f"{math.log2(math.factorial(n)):.3f}", math.ceil(math.log2(math.factorial(n))))
                     for n in (2, 3, 4, 5, 8, 16)])
    return (table + f"\n\nFewest comparisons: {min(counts)}, most: {max(counts)}. There are 6 inputs, so a tree of "
            "depth 2 (at most 4 leaves) cannot separate them.\n\n" + bounds)


def radix_trace():
    keys = [170, 45, 75, 90, 802, 24, 2, 66]
    cur, out = list(keys), []
    for digit_place, name in ((1, "ones"), (10, "tens"), (100, "hundreds")):
        digits = [(k // digit_place) % 10 for k in cur]
        counts = [digits.count(d) for d in range(10)]
        offsets = _scan(counts, True)
        nxt = [None] * len(cur)
        cursor = list(offsets)
        for k, d in zip(cur, digits):
            nxt[cursor[d]] = k
            cursor[d] += 1
        out.append((f"{name} digit", _list(cur), _list(digits), _list(counts), _list(offsets), _list(nxt)))
        cur = nxt
    assert cur == sorted(keys)
    return _table(["pass", "input order", "digits", "count per digit 0..9", "exclusive scan", "output order"], out, "llllll")


def radix_vs_merge():
    name = "algorithms.radix_vs_merge"
    rng = bookrng.generator(name)
    all_keys = [rng.next_u64() >> 32 for _ in range(65536)]
    rows = []
    for n in (1024, 16384, 65536):
        keys = all_keys[:n]
        cur, ops = list(keys), 0
        for shift in (0, 8, 16, 24):
            counts = [0] * 256
            for k in cur:
                counts[(k >> shift) & 255] += 1
                ops += 1
            offsets = _scan(counts, True)
            ops += 256
            nxt = [0] * n
            for k in cur:
                d = (k >> shift) & 255
                nxt[offsets[d]] = k
                offsets[d] += 1
                ops += 1
            cur = nxt
        assert cur == sorted(keys)
        c = [0]
        _merge_sort(keys, c)
        rows.append((_n(n), _n(ops), _n(c[0]), f"{c[0] / ops:.2f}"))
    return (f"{_seed_note(name)} Each key is the top 32 bits of one raw draw (`next()` shifted "
            "right by 32); the keys for n are the first n of 65,536 drawn in order. Radix sort: four passes of 8 bits "
            "starting from the least significant, each pass making one counter increment per key, 256 scan steps and "
            "one move per key, so 4 (2n + 256) steps in all. Merge sort: comparisons of the chapter's algorithm on "
            "the same keys. The ratio column is merge-sort comparisons divided by radix-sort steps.\n\n"
            + _table(["n", "radix sort: counter increments + scan steps + moves", "merge sort: comparisons",
                      "ratio"], rows))


def radix_stability():
    pairs = [(2, "a"), (1, "b"), (2, "c"), (1, "d"), (0, "e"), (2, "f")]
    counts = [0] * 3
    for k, _ in pairs:
        counts[k] += 1
    offsets = _scan(counts, True)
    out = [None] * len(pairs)
    for k, tag in pairs:
        out[offsets[k]] = (k, tag)
        offsets[k] += 1
    return f"Input {pairs}; output of one counting pass: {out}."


# --------------------------------------------------------------------- hashing

def hash_loads():
    bits = 10
    m = 1 << bits
    name = "algorithms.hash_loads"
    keys = _distinct_draws(bookrng.generator(name), 1 << 24, 4096 + 2000)
    absent = keys[4096:]
    rows = []
    for n in (256, 512, 1024, 2048, 4096):
        stored = keys[:n]
        table = _ChainTable(bits)
        for k in stored:
            table.insert(k)
        table.comparisons = 0
        for k in stored:
            table.find(k)
        found = table.comparisons / n
        table.comparisons = 0
        for k in absent:
            table.find(k)
        missed = table.comparisons / len(absent)
        alpha = n / m
        rows.append((n, f"{alpha:.2f}", f"{found:.3f}", f"{1 + alpha / 2:.3f}", f"{missed:.3f}", f"{alpha:.3f}",
                     max(len(c) for c in table.chains)))
    return (f"A table of {_n(m)} buckets with chaining and the multiplicative hash of the chapter (b = {bits}); n random 24-bit keys. "
            f"{_seed_note(name)} Keys are `integer(2^24)` draws in order with repeats discarded: the stored keys of a "
            "row are the first n of them, and the 2,000 absent keys are the 4,097th to 6,096th distinct keys (the draws "
            "that repeat an earlier key are skipped and not numbered), the same for every row. Comparisons are counted for looking up every stored key (found) or every absent key "
            "(absent), one per chain entry visited up to and including the match; the chain is in insertion order. "
            "The prediction columns are 1 + alpha/2 and alpha.\n\n"
            + _table(["n keys", "load alpha = n/m", "comparisons to find a stored key", "prediction 1 + alpha/2",
                      "comparisons for an absent key", "prediction alpha", "longest chain"], rows))


def _mod_bucket(x, bits):
    return x % (1 << bits)


def hash_bad_function():
    bits, n = 6, 256
    keys = [64 * k for k in range(1, n + 1)]
    rows = []
    for name, fn in (("x mod 64", _mod_bucket), ("multiplicative", _hash_bucket)):
        table = _ChainTable(bits, fn)
        for k in keys:
            table.insert(k)
        table.comparisons = 0
        for k in keys:
            table.find(k)
        rows.append((name, max(len(c) for c in table.chains), sum(1 for c in table.chains if c),
                     f"{table.comparisons / n:.2f}"))
    return (f"{n} token ids, the multiples of 64 from 64 to {64 * n} (ids 64 k for k = 1 .. {n}), in a table of 64 buckets (load 4.0). "
            f"Each id is inserted in this order, then every id is looked up once; x mod 64 is the bucket of the first row, the "
            f"multiplicative hash of the chapter with b = 6 the second; comparisons are counted as in the table of loads above.\n\n"
            + _table(["hash function", "longest chain", "buckets used (of 64)", "mean comparisons per lookup"], rows))


def fingerprints():
    msg = b"token ids: 1 2 3 4 5 6 7 8"
    flipped = bytes([msg[0] ^ 1]) + msg[1:]
    other = msg[:-1] + b"9"
    d0 = hashlib.sha256(msg).digest()
    rows = []
    for label, m in (("original", msg), ("first byte, lowest bit flipped", flipped), ("last character 8 -> 9", other)):
        d = hashlib.sha256(m).digest()
        differ = sum(bin(a ^ b).count("1") for a, b in zip(d, d0))
        rows.append((label, hashlib.sha256(m).hexdigest()[:32] + "...", differ))
    name = "algorithms.fingerprints"
    rng = bookrng.generator(name)
    buf = b"".join(rng.next_u64().to_bytes(8, "big") for _ in range((1 << 20) // 8))
    whole = hashlib.sha256(buf).hexdigest()
    h = hashlib.sha256()
    for i in range(0, len(buf), 65536):
        h.update(buf[i:i + 65536])
    files = {"weights.bin": buf[:4096], "config.json": b'{"layers": 24, "width": 512}', "vocab.txt": b"a\nb\nc\n"}
    manifest = {k: hashlib.sha256(v).hexdigest()[:16] for k, v in files.items()}
    files2 = dict(files)
    files2["weights.bin"] = bytes([files["weights.bin"][0] ^ 0x80]) + files["weights.bin"][1:]
    manifest2 = {k: hashlib.sha256(v).hexdigest()[:16] for k, v in files2.items()}
    mrows = [(k, manifest[k], manifest2[k], "changed" if manifest[k] != manifest2[k] else "same") for k in files]
    return (f"The original message is the {len(msg)} ASCII bytes `{msg.decode()}`; the second message flips the lowest "
            "bit of its first byte; the third replaces the last character `8` by `9`. SHA-256 is the standard "
            "function, used as a black box.\n\n"
            + _table(["message (SHA-256 of the bytes, first 32 of 64 hex digits)", "fingerprint", "bits differing from the original (of 256)"], rows, "llr")
            + "\n\n" + f"The 1 MiB buffer is 131,072 raw draws of the book's SplitMix64 (seed {bookrng.seed_of(name):#018x}), "
            "each written as 8 bytes, most significant byte first. "
            f"Hashed whole and in 16 chunks of 64 KiB it gives the same fingerprint: {whole == h.hexdigest()}.\n\n"
            "The manifest files: weights.bin is the first 4096 bytes of that buffer, config.json is the "
            f"{len(files['config.json'])} bytes `{files['config.json'].decode()}`, vocab.txt is the "
            f"{len(files['vocab.txt'])} bytes `a`, newline, `b`, newline, `c`, newline. "
            "Then the highest bit of the first byte of weights.bin is flipped. "
            "A three-file manifest before and after one bit of weights.bin changes (first 16 hex digits shown):\n\n"
            + _table(["file", "before", "after", ""], mrows, "llll"))


# ------------------------------------------------------------ top-k selection

def _heap_topk_comparisons(xs, k):
    """k largest by a binary min-heap of size k, counting every comparison of two logits.

    While the heap has fewer than k entries a new logit is appended and sifted up (one comparison with the parent
    per level, stopping at the first parent not larger). Once full, a logit is compared with the root (one
    comparison) and, if larger, replaces it and is sifted down (per level one comparison between the two children
    when both exist, then one between the smaller child and the element, stopping when the element is not larger).
    """
    heap, count = [], 0
    for x in xs:
        if len(heap) < k:
            heap.append(x)
            i = len(heap) - 1
            while i > 0:
                parent = (i - 1) // 2
                count += 1
                if heap[i] < heap[parent]:
                    heap[i], heap[parent] = heap[parent], heap[i]
                    i = parent
                else:
                    break
            continue
        count += 1
        if not x > heap[0]:
            continue
        heap[0] = x
        i = 0
        while True:
            left = 2 * i + 1
            if left >= k:
                break
            child = left
            if left + 1 < k:
                count += 1
                if heap[left + 1] < heap[left]:
                    child = left + 1
            count += 1
            if heap[child] < heap[i]:
                heap[i], heap[child] = heap[child], heap[i]
                i = child
            else:
                break
    return count, sorted(heap, reverse=True)


def topk_costs():
    name = "algorithms.topk_costs"
    rng = bookrng.generator(name)
    vocab = 16384
    logits = [rng.normal() for _ in range(vocab)]
    rows = []
    for k in (1, 8, 64):
        c = [0]
        ordered = _merge_sort(logits, c)
        want = ordered[::-1][:k]
        passes = (vocab - 1) * k          # k scans, each vocab - 1 comparisons for the maximum of what remains
        work = list(logits)
        got = []
        for _ in range(k):
            best = max(range(len(work)), key=work.__getitem__)
            got.append(work[best])
            work[best] = float("-inf")
        heap_count, heap_vals = _heap_topk_comparisons(logits, k)
        assert got == want == heap_vals
        rows.append((k, _n(c[0]), _n(passes), _n(heap_count), _n(vocab * 4), _n(k * 8)))
    return (f"Logits of a vocabulary of {_n(vocab)} tokens: the first {_n(vocab)} `normal()` draws of the book's SplitMix64, "
            f"seed {bookrng.seed_of(name):#018x}. Comparisons to find the k largest. Sort everything: the "
            "comparisons of the chapter's merge sort. k scans: k (V - 1), each scan finds the maximum of the "
            "remaining logits with V - 1 comparisons. Heap of size k: a binary min-heap holding the k largest so far; "
            "a logit is first compared with the smallest (the root) once the heap is full and replaces it only if "
            "larger, and every comparison made while moving an element up or down the heap counts (sifting up: one "
            "comparison with the parent per level; sifting down: one comparison between the two children and one "
            "of the smaller child with the element, per level).\n\n"
            + _table(["k", "sort everything", "k scans for a maximum", "heap of size k",
                      "bytes to read back, full row (4 per logit)", "bytes to read back, k (index, value) pairs"], rows))


# ---------------------------------------------------------- BPE merge, counted

def _bpe_chain(n):
    """Merge table whose merges can only happen one at a time: symbol k joins the growing symbol on its left."""
    ranks, result = {}, {}
    acc = 0
    for k in range(n - 1):
        new = n + k
        ranks[(acc, k + 1)] = k
        result[(acc, k + 1)] = new
        acc = new
    return ranks, result


def _bpe_textbook(n, ranks, result):
    word = list(range(n))
    lookups = 0
    while True:
        best = None
        for a, b in zip(word, word[1:]):
            lookups += 1
            r = ranks.get((a, b))
            if r is not None and (best is None or r < best[0]):
                best = (r, (a, b))
        if best is None:
            return word, lookups
        out, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == best[1]:
                out.append(result[best[1]])
                i += 2
            else:
                out.append(word[i])
                i += 1
        word = out


def _bpe_heap(n, ranks, result):
    sym = list(range(n))
    prev = [i - 1 for i in range(n)]
    nxt = [i + 1 if i + 1 < n else -1 for i in range(n)]
    alive = [True] * n
    lookups = 0
    heap = []

    def push(left):
        nonlocal lookups
        right = nxt[left]
        if right != -1:
            lookups += 1
            r = ranks.get((sym[left], sym[right]))
            if r is not None:
                heapq.heappush(heap, (r, left, right, sym[left], sym[right]))

    for i in range(n - 1):
        push(i)
    while heap:
        _, left, right, sl, sr = heapq.heappop(heap)
        if not (alive[left] and alive[right] and nxt[left] == right and sym[left] == sl and sym[right] == sr):
            continue                                  # a stale entry: one of its symbols has changed
        sym[left] = result[(sl, sr)]
        alive[right] = False
        nxt[left] = nxt[right]
        if nxt[right] != -1:
            prev[nxt[right]] = left
        if prev[left] != -1:
            push(prev[left])
        push(left)
    out, i = [], 0
    while i != -1:
        out.append(sym[i])
        i = nxt[i]
    return out, lookups


def bpe_scan_counts():
    demo_ranks, demo_result = _bpe_chain(4)
    demo = "; ".join(f"{pair} -> {demo_result[pair]} (rank {r})" for pair, r in sorted(demo_ranks.items(), key=lambda t: t[1]))
    rows = []
    for n in (16, 64, 256, 1024):
        ranks, result = _bpe_chain(n)
        slow, a = _bpe_textbook(n, ranks, result)
        fast, b = _bpe_heap(n, ranks, result)
        assert slow == fast and len(slow) == 1
        rows.append((_n(n), _n(a), _n(n * (n - 1) // 2), _n(b)))
    return ("The piece is the n symbols 0, 1, ..., n - 1. The merge table has n - 1 entries: for k = 0 .. n - 2 the pair "
            "(A_k, k + 1) has rank k and merges into the new symbol A_(k+1) = n + k, where A_0 = 0 (so only the leftmost "
            f"symbol ever merges, one merge at a time). For n = 4 the table is: {demo}. "
            "A pair lookup is one read of this table for one ordered pair of adjacent symbols (found or not). "
            "Textbook loop: each round looks up every adjacent pair of the current piece, merges the pair of lowest rank "
            "(every non-overlapping occurrence, left to right), and stops after a round in which no pair is found; it "
            "ends with one symbol. Heap + linked list: the pieces are a doubly linked list of symbols; one lookup for each "
            "of the n - 1 initial pairs, then after each merge one lookup for the new pair with the left neighbour (if any) "
            "and one for the new pair with the right neighbour (if any); a found pair is pushed on a min-heap "
            "ordered by (rank, position); popping the heap and discarding stale entries costs no lookup.\n\n"
            + _table(["symbols n", "textbook loop: pair lookups", "n(n-1)/2", "heap + linked list: pair lookups"], rows))
