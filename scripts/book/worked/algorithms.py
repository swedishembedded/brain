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

def _distinct_ids(n, vocab, name):
    """n distinct token ids from range(vocab), in a frozen random order."""
    return bookrng.stream(name).sample(range(vocab), n)


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


def duplicate_counts(ns, vocab=50_000):
    """Operation counts of the three methods on lists of n distinct ids (no duplicate to stop early)."""
    rows = []
    for n in ns:
        ids = _distinct_ids(n, vocab, f"algorithms.duplicates_growth.{n}")
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
    rows = [
        ("compare all pairs", d_pairs, c_pairs, "stops at the first equal pair it meets"),
        ("sort, then compare neighbours", d_sort, c_sort, "merge sort comparisons plus n - 1 neighbour tests"),
        ("remember ids in a hash table", d_hash, c_hash, "hash evaluations plus key comparisons; stops at the first repeat"),
    ]
    note = (f"The list has n = {len(ids)} ids; the ids that occur twice are "
            f"{sorted({x for x in ids if ids.count(x) > 1})}. The methods may report different repeated ids because "
            "they visit the list in different orders; each reported id really is repeated.")
    return note + "\n\n" + _table(["method", "reports", "operations counted", "what is counted"], rows, "llrl")


def duplicates_growth():
    ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = duplicate_counts(ns)
    body = [(_n(n), _n(p), _n(s), _n(h)) for n, p, s, h, _, _ in rows]
    return ("Operations counted on n distinct token ids (no repeat, so no method can stop early).\n\n"
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

    Layer i computes a_i from a_{i-1}; the backward step of layer i needs a_{i-1}.
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
    return (f"A chain of {layers} layers; stride s keeps every s-th activation and recomputes the rest.\n\n"
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
    rng = bookrng.stream("algorithms.reduction_orders")
    xs = [_f32(rng.uniform(-1.0, 1.0) * 10 ** rng.randint(-3, 3)) for _ in range(4096)]
    add32 = lambda a, b: _f32(a + b)       # noqa: E731
    exact = sum(Fraction(x) for x in xs)
    rows = []
    for name, value in (("left to right", _fold(add32, xs)), ("tree (pairs, layer by layer)", _tree_reduce(add32, xs)),
                        ("4 interleaved lanes", _strided_fold(add32, xs, 4)),
                        ("32 interleaved lanes", _strided_fold(add32, xs, 32))):
        rel = abs(Fraction(value) - exact) / abs(exact)
        rows.append((name, f"{value:.4f}", f"{float(rel):.2e}"))
    ints = [rng.randint(-1000, 1000) for _ in range(1024)]
    int_vals = {_fold(lambda a, b: a + b, ints), _tree_reduce(lambda a, b: a + b, ints),
                _strided_fold(lambda a, b: a + b, ints, 32)}
    pairs = [(x, i) for i, x in enumerate(ints)]
    arg_vals = {_fold(_argmax_op, pairs), _tree_reduce(_argmax_op, pairs), _strided_fold(_argmax_op, pairs, 32)}
    mx_vals = {_fold(max, ints), _tree_reduce(max, ints), _strided_fold(max, ints, 32)}
    sub = [10, 3, 2, 1]
    out = [f"Sum of 4096 signed 32-bit numbers of mixed magnitude (exact value of the stored numbers: {float(exact):.4f}).\n",
           _table(["order", "32-bit sum", "relative error"], rows),
           "\nThe same three orders on 1024 integers in [-1000, 1000] and on an operator that is not associative:\n",
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
    """Smallest index i with cdf[i] > u, by binary search. Returns (i, steps)."""
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
    rng = bookrng.stream("algorithms.scan_cdf")
    counts = [0] * len(p)
    cdf_f = [float(c) for c in cdf]
    cdf_f[-1] = 1.0 + 1e-9
    trials = 20000
    for _ in range(trials):
        counts[_inverse_cdf(cdf_f, rng.random())[0]] += 1
    frows = [(f"token {i}", f"{float(p[i]):.2f}", _n(counts[i]), f"{counts[i] / trials:.4f}") for i in range(len(p))]
    return (_table(["token", "probability", "cumulative (inclusive scan)"], rows) + "\n\n"
            + _table(["u", "token drawn", "binary-search steps"], drows) + "\n\n"
            + f"{_n(trials)} draws with a frozen uniform stream:\n\n"
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
    return (_table(["block", "input", "1: scan inside the block", "block total", "2: scan of totals", "3: add it back"],
                   rows, "lllrrl")
            + f"\n\nConcatenated, the last column is {_list(final)}, the exclusive scan of the whole array.")


# --------------------------------------------------------------------- sorting

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
    rows = [(f"{_list(n[2])}", f"{_list(n[3])}", n[4]) for n in reversed(nodes) if len(n[2]) > 1]
    total = sum(n[4] for n in nodes)
    counter = [0]
    assert _merge_sort(MERGE_DEMO, counter) == sorted(MERGE_DEMO) and counter[0] == total
    return (_table(["merge of the range", "result", "comparisons"], rows, "llr")
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
    rng = bookrng.stream("algorithms.sorting_counts")
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
            "any comparison sort needs in the worst case: ceil(log2 n!).\n\n"
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
    rng = bookrng.stream("algorithms.radix_vs_merge")
    rows = []
    for n in (1024, 16384, 65536):
        keys = [rng.getrandbits(32) for _ in range(n)]
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
    return _table(["n", "radix sort: counter increments + scan steps + moves", "merge sort: comparisons",
                   "ratio"], rows)


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
    rows = []
    for n in (256, 512, 1024, 2048, 4096):
        rng = bookrng.stream(f"algorithms.hash_loads.{n}")
        keys = rng.sample(range(1 << 24), n + 2000)
        stored, absent = keys[:n], keys[n:]
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
    return (f"A table of {_n(m)} buckets with chaining and a multiplicative hash; n random 24-bit keys.\n\n"
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
    return (f"{n} token ids, all multiples of 64, in a table of 64 buckets (load 4.0).\n\n"
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
    buf = bookrng.stream("algorithms.fingerprints").randbytes(1 << 20)
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
    return (_table(["message (SHA-256 of the bytes, first 32 of 64 hex digits)", "fingerprint", "bits differing from the original (of 256)"], rows, "llr")
            + "\n\n" + f"A 1 MiB buffer hashed whole and in 16 chunks of 64 KiB gives the same fingerprint: {whole == h.hexdigest()}.\n\n"
            + "A three-file manifest before and after one bit of weights.bin changes (first 16 hex digits shown):\n\n"
            + _table(["file", "before", "after", ""], mrows, "llll"))


# ------------------------------------------------------------ top-k selection

def _heap_topk_comparisons(xs, k):
    """k largest by a min-heap of size k; counts value comparisons (heapq compares tuples of floats)."""
    class Counted:
        count = 0
        __slots__ = ("v",)

        def __init__(self, v):
            self.v = v

        def __lt__(self, other):
            Counted.count += 1
            return self.v < other.v

    heap = []
    for x in xs:
        item = Counted(x)
        if len(heap) < k:
            heapq.heappush(heap, item)
        else:
            Counted.count += 1
            if x > heap[0].v:
                heapq.heapreplace(heap, item)
    return Counted.count, sorted((c.v for c in heap), reverse=True)


def topk_costs():
    rng = bookrng.stream("algorithms.topk_costs")
    vocab = 16384
    logits = [rng.gauss(0.0, 1.0) for _ in range(vocab)]
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
    return (f"Logits of a vocabulary of {_n(vocab)} tokens (frozen random normal numbers). Comparisons to find the k largest.\n\n"
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
    rows = []
    for n in (16, 64, 256, 1024):
        ranks, result = _bpe_chain(n)
        slow, a = _bpe_textbook(n, ranks, result)
        fast, b = _bpe_heap(n, ranks, result)
        assert slow == fast and len(slow) == 1
        rows.append((_n(n), _n(a), _n(n * (n - 1) // 2), _n(b)))
    return _table(["symbols n", "textbook loop: pair lookups", "n(n-1)/2", "heap + linked list: pair lookups"], rows)
