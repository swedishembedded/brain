# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The CPU: Cores, Caches and Vectors".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: cpu.<name> -->` markers. Helpers with a leading underscore, or
that take arguments, are shared with the chapter's figures so a picture and
the table beside it cannot disagree.

Nothing here measures a machine. The chapter states speed as work: cache lines
fetched, instructions issued, length of a dependency chain, bytes moved. Those
are counted by small models whose rules are printed beside each table: a cache
simulator (`Cache`: line size, number of sets, ways, least-recently-used
replacement), a list scheduler for a dependency graph of additions, a model of
which core owns a cache line, and the arithmetic of splitting a loop among
workers. Where a cost in cycles appears it is the model's output from the
stated assumptions, labelled as a model, never an observation.

Swedish Embedded AB implements solutions for making numerical code use the
cache hierarchy, vector units and cores of the machines it runs on. If your
team needs expertise in CPU performance work for machine learning inference
and training, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import struct
from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache

import bookrng

LINE = 64        # bytes per cache line (the book's stated model)
ELEM = 4         # bytes per f32 element
PER_LINE = LINE // ELEM

# --------------------------------------------------------------- formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _n(v):
    return f"{v:,}"


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _seed_note(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (the seed of the name `{name}`)."


# ------------------------------------------------------------ cache model


class Cache:
    """A set-associative cache with least-recently-used replacement.

    `size` bytes in lines of `line` bytes, `ways` lines per set, so
    `size // line // ways` sets. Line number L = address // line lives in set
    L mod sets. An access to a line that is in its set is a hit and makes the
    line the most recently used; otherwise it is a miss: the line is fetched,
    and if the set already holds `ways` lines the least recently used one is
    evicted. `ways = size // line` gives one set: a fully associative cache.
    Reads and writes are treated alike (a write to a missing line fetches it).
    """

    def __init__(self, size=32768, line=LINE, ways=8):
        self.line = line
        self.ways = ways
        self.nsets = size // line // ways
        self.sets = [OrderedDict() for _ in range(self.nsets)]
        self.accesses = 0
        self.misses = 0

    def access(self, addr):
        self.accesses += 1
        ln = addr // self.line
        st = self.sets[ln % self.nsets]
        if ln in st:
            st.move_to_end(ln)
            return True
        self.misses += 1
        if len(st) >= self.ways:
            st.popitem(last=False)
        st[ln] = True
        return False

    def contents(self):
        """The resident lines of a one-set cache, least recently used first."""
        return list(self.sets[0])


def _full(size):
    """Constructor arguments of a fully associative cache of `size` bytes."""
    return {"size": size, "ways": size // LINE}


@lru_cache(maxsize=None)
def traversal_misses(n, order, size=32768, ways=8):
    """Lines fetched when an n-by-n row-major f32 matrix is read element by element."""
    cache = Cache(size=size, ways=ways)
    if order == "row":
        pairs = ((i, j) for i in range(n) for j in range(n))
    else:
        pairs = ((i, j) for j in range(n) for i in range(n))
    for i, j in pairs:
        cache.access((i * n + j) * ELEM)
    return cache.misses


# ------------------------------------------------- 1. the problem, by hand

def lru_trace():
    n, cap = 16, 4
    cache = Cache(**_full(cap * LINE))
    rows = []
    for k, (i, j) in enumerate((i, 0) for i in range(10)):
        addr = (i * n + j) * ELEM
        hit = cache.access(addr)
        rows.append((k, f"A[{i}, {j}]", addr, addr // LINE, "hit" if hit else "miss",
                     ", ".join(str(x) for x in cache.contents())))
    return ("A 16 by 16 matrix of 4-byte numbers, row-major from address 0 (a row is exactly one 64-byte line, so "
            "element A[i, j] is at address 4(16i + j) and is in line i). A cache of 4 lines, fully associative, "
            "least recently used first. The first ten reads of column 0, read down the column:\n\n"
            + _table(["read", "element", "address", "line", "result", "lines held afterwards (oldest first)"], rows,
                     "lrrrll"))


def lru_cliff():
    n = 16
    rows = []
    for cap in (4, 8, 15, 16, 17, 32):
        col = traversal_misses(n, "col", **_full(cap * LINE))
        row = traversal_misses(n, "row", **_full(cap * LINE))
        rows.append((cap, row, col, f"{col / row:.1f}"))
    return ("A 16 by 16 matrix of 4-byte numbers (256 reads, 16 lines), cache fully associative with least "
            "recently used replacement, all reads. Lines fetched when the matrix is read by rows and by columns:\n\n"
            + _table(["cache size in lines", "by rows", "by columns", "columns / rows"], rows))


# ---------------------------------------------- 2. matrix size, two models

SIZES = (16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024)


def _traversal_rows():
    """(n, row-order fetches, column-order fetches fully associative, column-order fetches 8-way)."""
    out = []
    for n in SIZES:
        out.append((n, traversal_misses(n, "row"), traversal_misses(n, "col", **_full(32768)),
                    traversal_misses(n, "col")))
    return out


def traversal_sizes():
    rows = []
    for n, r, cf, c8 in _traversal_rows():
        rows.append((n, _n(n * n), _n(n * n * ELEM // 1024) + " KiB", _n(r), _n(cf), _n(c8),
                     f"{cf / r:.1f}", f"{c8 / r:.1f}"))
    return ("An n by n row-major matrix of 4-byte numbers, every element read once, row by row or column by column. "
            "Cache model: 32 KiB (512 lines of 64 bytes), least recently used replacement; \"full\" is fully "
            "associative (one set of 512 ways), \"8-way\" has 64 sets of 8 ways. Lines fetched:\n\n"
            + _table(["n", "elements", "matrix size", "rows", "columns, full", "columns, 8-way", "full / rows",
                      "8-way / rows"], rows))


def hierarchy_cost_model():
    """Expected cycles per element read for a matrix that lives in L3, under the typical latencies below."""
    n = 1024
    l1_hit = HIERARCHY[1][3]
    l3 = HIERARCHY[3][3]
    r = traversal_misses(n, "row")
    c = traversal_misses(n, "col")
    elems = n * n
    rows = []
    for name, m in (("by rows", r), ("by columns", c)):
        rate = m / elems
        rows.append((name, _n(m), f"{rate:.4f}", f"{l1_hit + rate * (l3 - l1_hit):.1f}"))
    ratio = (l1_hit + (c / elems) * (l3 - l1_hit)) / (l1_hit + (r / elems) * (l3 - l1_hit))
    return ("A MODEL, not a measurement. A 1024 by 1024 matrix of 4-byte numbers (4 MiB) that sits in L3, read once "
            f"in each order, 8-way 32 KiB cache. Each read costs {l1_hit} cycles when its line is in L1, and a line "
            f"fetched from L3 adds the difference {l3 - l1_hit} cycles (the typical latencies of the hierarchy table; "
            "no prefetching, no overlap between fetches). Expected cycles per element = "
            f"{l1_hit} + (lines fetched / elements) * {l3 - l1_hit}:\n\n"
            + _table(["order", "lines fetched", "fetches per element", "modelled cycles per element"], rows)
            + f"\n\nModelled ratio: {ratio:.1f}.")


# ------------------------------------------------------- 3. the hierarchy

# (name, typical capacity in bytes, "who shares it", typical latency in cycles, note); typical orders of
# magnitude of a recent desktop or server core, chosen for round arithmetic, not measured on any machine.
HIERARCHY = (
    ("registers", 640, "one core", 1, "named in the instruction"),
    ("L1 data cache", 32 * 1024, "one core", 4, "split from the instruction cache"),
    ("L2 cache", 1024 * 1024, "one core", 14, "unified"),
    ("L3 cache", 32 * 1024 * 1024, "all cores", 45, "shared"),
    ("DRAM", 16 * 1024 ** 3, "all cores", 250, "off the chip"),
)


def _size(b):
    for unit, name in ((1024 ** 3, "GiB"), (1024 ** 2, "MiB"), (1024, "KiB")):
        if b >= unit:
            return f"{b // unit} {name}"
    return f"{b} B"


def hierarchy_table():
    rows = []
    for name, cap, who, lat, _ in HIERARCHY:
        lines = "-" if name == "registers" else _n(cap // LINE)
        rows.append((name, _size(cap), lines, who, lat, f"{lat / HIERARCHY[1][3]:.2f}"))
    return ("TYPICAL orders of magnitude for one core of a recent server or desktop chip, rounded for arithmetic; "
            "they vary by chip and are not measurements of any machine. The register row counts "
            "16 vector registers of 32 bytes (512 bytes) and 16 scalar registers of 8 bytes (128 bytes): 640 bytes.\n\n"
            + _table(["level", "typical capacity", "lines of 64 bytes", "shared by", "typical latency, cycles",
                      "relative to L1"], rows, "lrrlrr"))


# --------------------------------------------------- 4. dependency chains

def _schedule(ops, latency, issue):
    """List-schedule additions. ops[i] = indices of the ops whose results op i needs.

    Each cycle issues up to `issue` ready ops, lowest index first; an op issued in cycle c delivers its
    result at the start of cycle c + latency. Returns the cycle at which the last result is available.
    """
    done = [None] * len(ops)
    cycle, finished, last = 0, 0, 0
    pending = list(range(len(ops)))
    while pending:
        slots, rest = issue, []
        for i in pending:
            if slots and all(done[d] is not None and done[d] <= cycle for d in ops[i]):
                done[i] = cycle + latency
                slots -= 1
                last = max(last, done[i])
            else:
                rest.append(i)
        pending = rest
        cycle += 1
    return last


def _sum_program(n, lanes):
    """The additions of summing n numbers into `lanes` totals, then adding the totals in layers, in program order.

    Element i belongs to total i mod `lanes`. The first addition of a total adds its first two elements and each
    later one adds the next element of the total to the running value, so a total with m elements is a chain of
    m - 1 additions; a total with one element needs none. The totals are then combined in layers: adjacent pairs
    (0 with 1, 2 with 3, ...) are added, an odd one out moves to the next layer unchanged, and the next layer pairs
    the results the same way, until one value is left. Program order is every addition of total 0, then of total 1,
    and so on, then the layers from the first to the last, left to right. Each addition is a pair of operands,
    ("x", i) for input element i or ("o", k) for the result of addition k.
    """
    prog, layer = [], []
    for lane in range(lanes):
        prev = ("x", lane)
        for i in range(lane + lanes, n, lanes):
            prog.append((prev, ("x", i)))
            prev = ("o", len(prog) - 1)
        layer.append(prev)
    while len(layer) > 1:
        nxt = []
        for a, b in zip(layer[::2], layer[1::2]):
            prog.append((a, b))
            nxt.append(("o", len(prog) - 1))
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    return prog


def _sum_graph(n, lanes):
    """The dependencies of `_sum_program`: ops[k] = indices of the additions whose results addition k needs.

    Returns (ops, longest chain of additions). Operands are assumed to be in registers already.
    """
    ops = [[v for kind, v in pair if kind == "o"] for pair in _sum_program(n, lanes)]
    depth = [0] * len(ops)
    for i, deps in enumerate(ops):
        depth[i] = 1 + max((depth[d] for d in deps), default=0)
    return ops, max(depth, default=0)


def _label(operand):
    return f"{operand[0]}{operand[1]}"


def _program_text(n, lanes):
    return "; ".join(f"o{k} = {_label(a)} + {_label(b)}" for k, (a, b) in enumerate(_sum_program(n, lanes)))


# What a reader needs to reproduce the schedule from the page; shared by both tables.
_SCHEDULE_RULE = (
    "Program: element i belongs to total i mod L; the first addition of a total adds its first two elements, each later "
    "one adds the next element of that total to its running value, so a total of m elements is a chain of m - 1 "
    "additions; then the L totals are combined in layers, adding adjacent pairs (total 0 with 1, 2 with 3, ...) and then "
    "adjacent pairs of those results, until one value is left. Program order is all the additions of total 0, then of "
    "total 1, and so on, then the layers from first to last, left to right. Schedule: cycles are numbered 0, 1, 2, ...; "
    "in each cycle the not yet issued additions are scanned in program order and the first two whose operands are "
    "available at the start of that cycle are issued (an addition that is not ready does not block the ones after "
    "it); an addition issued in cycle c delivers its result at the start of cycle c + 4. The modelled cycles are the "
    "cycle at which the last result is available; the dependent chain counts additions along the longest path."
)


def dependency_small():
    n, lat, iss = 8, 4, 2
    rows = []
    progs = []
    for lanes in (1, 2, 4, 8):
        ops, chain = _sum_graph(n, lanes)
        rows.append((lanes, len(ops), chain, _schedule(ops, lat, iss)))
        progs.append((lanes, _program_text(n, lanes)))
    listing = "\n".join(f"- L = {lanes}: {text}" for lanes, text in progs)
    return (f"Summing n = {n} numbers into L accumulators, then adding the L totals as a tree. Model: an addition "
            f"takes {lat} cycles from issue to result, the core can start {iss} additions per cycle, operands are "
            f"already in registers. {_SCHEDULE_RULE} Here x0 ... x7 are the inputs and o0, o1, ... are the results of "
            "the additions in program order; the programs are:\n\n" + listing + "\n\n"
            + _table(["L", "additions", "longest chain of dependent additions", "modelled cycles to the sum"], rows))


def dependency_chain():
    n, lat, iss = 1024, 4, 2
    rows = []
    for lanes in (1, 2, 4, 8, 16, 32):
        ops, chain = _sum_graph(n, lanes)
        cyc = _schedule(ops, lat, iss)
        ideal = math.ceil(len(ops) / iss)
        rows.append((lanes, len(ops), chain, cyc, ideal, f"{ideal / cyc:.3f}"))
    return (f"The same model with n = {n}: an addition takes {lat} cycles from issue to result, the core starts "
            f"{iss} additions per cycle, so {lat * iss} independent additions are needed to keep it busy. A MODEL "
            f"of a core, not a measurement of one. {_SCHEDULE_RULE} The issue-limited cycles are the number of "
            "additions divided by 2, rounded up; the last column is that divided by the modelled cycles.\n\n"
            + _table(["L", "additions", "longest dependent chain", "modelled cycles", "issue-limited cycles",
                      "fraction of the issue rate"], rows))


# ---------------------------------------------------------- 5. vectors

def simd_instructions():
    n = 1024
    rows = []
    for name, lanes in (("scalar", 1), ("SSE (128-bit)", 4), ("AVX2 (256-bit)", 8), ("AVX-512 (512-bit)", 16)):
        body = 3 * (n // lanes)               # two loads and one fused multiply-add per step
        tail = 2 * int(math.log2(lanes))      # one shuffle and one add per halving of the lanes
        rows.append((name, lanes, 4 * lanes, _n(n // lanes), _n(body), tail, _n(body + tail),
                     f"{(3 * n) / (body + tail):.2f}"))
    return (f"A dot product of two f32 arrays of n = {n} elements, one fused multiply-add per step. Counting model: "
            "each step issues two loads and one multiply-add (3 instructions) for `lanes` elements; the horizontal "
            "sum at the end costs one shuffle and one add per halving of the lanes; loop control and the "
            "address arithmetic are not counted.\n\n"
            + _table(["width", "lanes", "bytes per load", "steps", "instructions in the loop", "horizontal sum",
                      "total", "scalar total / total"], rows))


def hsum_trace():
    name = "cpu.hsum_trace"
    rng = bookrng.generator(name)
    n = 20
    x = [_f32(2.0 * rng.uniform() - 1.0) for _ in range(n)]
    y = [_f32(2.0 * rng.uniform() - 1.0) for _ in range(n)]
    lanes = [0.0] * 8
    for i in range(16):
        lanes[i % 8] = _f32(lanes[i % 8] + _f32(x[i] * y[i]))
    s4 = [_f32(lanes[i] + lanes[i + 4]) for i in range(4)]
    s2 = [_f32(s4[0] + s4[1]), _f32(s4[2] + s4[3])]
    s1 = _f32(s2[0] + s2[1])
    total = s1
    for i in range(16, n):
        total = _f32(total + _f32(x[i] * y[i]))
    ref = 0.0
    for i in range(n):
        ref = _f32(ref + _f32(x[i] * y[i]))
    exact = sum(Fraction(a) * Fraction(b) for a, b in zip(x, y))
    rows = [("lane accumulators " + str(j), f"{lanes[j]:.9g}") for j in range(8)]
    rows += [(f"lanes j and j + 4: s[{j}]", f"{s4[j]:.9g}") for j in range(4)]
    rows += [("s[0] + s[1]", f"{s2[0]:.9g}"), ("s[2] + s[3]", f"{s2[1]:.9g}"), ("horizontal sum", f"{s1:.9g}"),
             ("after the 4 tail elements (16 .. 19), one at a time", f"{total:.9g}"),
             ("left to right, one accumulator", f"{ref:.9g}"),
             ("exact sum of the exact products of the stored x and y", f"{float(exact):.9g}")]
    return (f"{_seed_note(name)} n = {n}. Draw order: the 20 values of x, each `2 * uniform() - 1` rounded to f32, "
            "then the 20 values of y the same way. Every product and every sum is rounded to f32 (no fused "
            "multiply-add). Lane j of the 8 accumulators adds the products j, j + 8 (elements 0..15 only). The "
            "horizontal sum is s[j] = lane j + lane j + 4, then (s[0] + s[1]) + (s[2] + s[3]); the last four "
            "products are then added to it one at a time. The last row is the exact (unrounded) products of the stored f32 "
            "inputs summed in exact arithmetic, then shown to 9 digits.\n\n" + _table(["step", "value"], rows))


def _hsum(acc):
    """The horizontal sum of w f32 lanes (w a power of 2): while more than 4 lanes remain, add lane j + h to
    lane j for h = half the lanes; then (a0 + a1) + (a2 + a3), or the one or two lanes that remain."""
    while len(acc) > 4:
        h = len(acc) // 2
        acc = [_f32(acc[j] + acc[j + h]) for j in range(h)]
    if len(acc) == 4:
        return _f32(_f32(acc[0] + acc[1]) + _f32(acc[2] + acc[3]))
    return _f32(acc[0] + acc[1]) if len(acc) == 2 else acc[0]


def simd_sum_order():
    name = "cpu.simd_sum_order"
    rng = bookrng.generator(name)
    n = 4096
    x = [_f32(2.0 * rng.uniform() - 1.0) for _ in range(n)]
    y = [_f32(2.0 * rng.uniform() - 1.0) for _ in range(n)]
    prods = [_f32(a * b) for a, b in zip(x, y)]
    exact = sum(Fraction(a) * Fraction(b) for a, b in zip(x, y))
    exact_rounded_products = sum(Fraction(p) for p in prods)

    def lanes_sum(w):
        acc = [0.0] * w
        for i, p in enumerate(prods):
            acc[i % w] = _f32(acc[i % w] + p)
        return _hsum(acc)

    rows = []
    results = set()
    for w in (1, 4, 8, 16, 32):
        v = lanes_sum(w)
        results.add(v)
        rel = abs(Fraction(v) - exact) / abs(exact)
        rows.append((w, f"{v:.9g}", f"{float(rel):.2e}"))
    return (f"{_seed_note(name)} n = {n}. Draw order: the 4096 values of x, each `2 * uniform() - 1` rounded to f32, "
            "then the 4096 of y. The products are rounded to f32 and then added in f32, one addition at a time. With "
            "w lanes, lane j adds the products j, j + w, j + 2w, ... in order, and the lanes are then combined by "
            "the horizontal sum (while more than 4 lanes remain, lane j + h is added to lane j for h = half "
            "the lanes; then (a0 + a1) + (a2 + a3)); w = 1 is the plain left-to-right sum. The reference is the exact "
            "sum, in exact arithmetic, of the exact (unrounded) products of the stored f32 inputs: "
            f"{float(exact):.9g}. (The exact sum of the f32-rounded products is {float(exact_rounded_products):.9g}; "
            "the two differ by the rounding of the products, which no order of additions can remove.) The error column "
            "is |f32 sum - reference| / |reference|.\n\n"
            + _table(["lanes w", "f32 sum", "relative error against the reference"], rows)
            + f"\n\n{len(results)} distinct results from 5 orders.")


def alignment():
    rows = []
    for start in (0, 4, 16, 32, 60):
        row = [start]
        for width in (32, 64):
            loads = 1024 * ELEM // width
            split = sum(1 for k in range(loads) if (start + k * width) % LINE + width > LINE)
            row.append(f"{split} of {loads}")
        rows.append(tuple(row))
    return ("Loads of consecutive vectors from an array of 1024 f32 (4096 bytes) that starts `offset` bytes after a "
            "64-byte line boundary. A load is split when its bytes lie in two lines.\n\n"
            + _table(["offset in bytes", "32-byte loads split", "64-byte loads split"], rows))


# ------------------------------------------------------- 6. tiling

def _transpose_misses(n, tile, size, ways, pitch=None):
    """Lines fetched copying A[i][j] to B[j][i] (n by n, row pitch `pitch` elements), visiting tile by tile."""
    pitch = pitch or n
    cache = Cache(size=size, ways=ways)
    base_b = n * pitch * ELEM
    for bi in range(0, n, tile):
        for bj in range(0, n, tile):
            for i in range(bi, min(bi + tile, n)):
                for j in range(bj, min(bj + tile, n)):
                    cache.access((i * pitch + j) * ELEM)           # read A[i][j]
                    cache.access(base_b + (j * pitch + i) * ELEM)  # write B[j][i]
    return cache.misses


def tiled_transpose():
    n, size = 256, 8192
    lines = size // LINE
    rows = []
    for tile in (1, 2, 4, 8, 16, 32, 64, 128):
        f = _transpose_misses(n, tile, size, lines)
        w = _transpose_misses(n, tile, size, 8)
        p = _transpose_misses(n, tile, size, 8, pitch=n + PER_LINE)
        rows.append((tile, _n(f), _n(w), _n(p)))
    ideal = 2 * n * n // PER_LINE
    return (f"B[j][i] = A[i][j] for a {n} by {n} matrix of 4-byte numbers (256 KiB each; B starts right after A), "
            f"visited tile by tile: tiles of T by T elements in row-major order of tiles, each tile in row-major "
            f"order. Cache model: {size // 1024} KiB ({lines} lines), least recently used; \"full\" is fully "
            f"associative, \"8-way\" has {lines // 8} sets of 8 ways, \"8-way, padded\" lays both matrices out "
            f"with rows of {n + PER_LINE} elements instead of {n} (the data is the same). Lines fetched; "
            f"the least possible is every line once, 2 * {n} * {n} / {PER_LINE} = {_n(ideal)}.\n\n"
            + _table(["T", "full", "8-way", "8-way, padded"], rows))


def _visit_order(n, tile):
    """Row-major order of the cells of an n by n grid, visiting tile by tile; returns {(i, j): step}."""
    order, step = {}, 0
    for bi in range(0, n, tile):
        for bj in range(0, n, tile):
            for i in range(bi, min(bi + tile, n)):
                for j in range(bj, min(bj + tile, n)):
                    order[(i, j)] = step
                    step += 1
    return order


def commit_transpose():
    tk, hd, jt = 4096, 64, 16
    results = []
    for label, ways in (("fully associative", 512), ("8-way", 8)):
        for variant in ("naive", "tiled"):
            cache = Cache(size=32768, ways=ways)
            dst = tk * hd * ELEM
            if variant == "naive":
                for j in range(tk):
                    for d in range(hd):
                        cache.access((j * hd + d) * ELEM)           # read V[j][d]
                        cache.access(dst + (d * tk + j) * ELEM)     # write VT[d][j]
            else:
                for j0 in range(0, tk, jt):
                    for j in range(j0, j0 + jt):                   # read the tile's source rows
                        for d in range(hd):
                            cache.access((j * hd + d) * ELEM)
                    for d in range(hd):                            # write d-major: 16 consecutive elements
                        for j in range(j0, j0 + jt):
                            cache.access(dst + (d * tk + j) * ELEM)
            results.append((label, variant, cache.misses))
    rows = [(a, b, _n(m)) for a, b, m in results]
    lo = tk * hd * ELEM // LINE
    return (f"VT[d][j] = V[j][d] with j in 0 .. {tk}, d in 0 .. {hd}: V is {tk} by {hd} (row of {hd} elements, 256 B), VT "
            f"is {hd} by {tk} (row of {tk} elements, 16 KiB), VT starts right after V. \"naive\" is j outer, d "
            f"inner, one element at a time. \"tiled\" reads {jt} rows of V, then writes VT d by d, {jt} consecutive "
            f"elements (one whole line) each; the {jt * hd * ELEM // 1024} KiB staging buffer is not counted. "
            f"Cache: 32 KiB, 64-byte lines, least recently used. Every line of V and VT once is "
            f"2 * {tk} * {hd} / {PER_LINE} = {_n(2 * lo)} fetches.\n\n"
            + _table(["cache", "loop", "lines fetched"], rows, "llr"))


# ------------------------------------------------------- 7. threads

def parallel_schedules():
    n, workers, grain = 1000, 8, 16
    cost = [i + 1 for i in range(n)]
    total = sum(cost)

    def block():
        per = [0] * workers
        for t in range(workers):
            lo, hi = t * n // workers, (t + 1) * n // workers
            per[t] = sum(cost[lo:hi])
        return per

    def cyclic():
        per = [0] * workers
        for i in range(n):
            per[i % workers] += cost[i]
        return per

    def dynamic(g):
        per = [0] * workers
        for lo in range(0, n, g):
            t = per.index(min(per))               # the worker that is free first takes the next chunk
            per[t] += sum(cost[lo:lo + g])
        return per

    rows = []
    for name, per in (("one block of 125 per worker", block()), ("round robin, one iteration at a time", cyclic()),
                      ("chunks of 1, next chunk to the least loaded worker", dynamic(1)),
                      (f"chunks of {grain}, next chunk to the least loaded worker", dynamic(grain)),
                      ("chunks of 125, next chunk to the least loaded worker", dynamic(125))):
        rows.append((name, _n(max(per)), _n(min(per)), f"{total / (workers * max(per)):.3f}"))
    return (f"{n} iterations, iteration i costs i + 1 units of work (total {_n(total)}), {workers} workers, so a "
            f"perfectly balanced split gives each worker {total / workers:,.1f}. The makespan is the largest "
            "load; efficiency = total work / (workers * makespan). Dynamic scheduling hands out the chunks in order of "
            "index, each to the worker with the smallest load so far.\n\n"
            + _table(["split", "largest load", "smallest load", "efficiency"], rows))


def dispatch_grain():
    floor = 16 * 1024
    rows = []
    for threads in (8, 144):
        for total in (2048, 100_000, 1_048_576, 42_000_000):
            chunk = max(-(-total // (8 * threads)), floor)
            tasks = -(-total // chunk)
            rows.append((threads, _n(total), _n(chunk), _n(tasks) if chunk < total else "0 (runs on the caller)"))
    return (f"The splitting rule: a loop of N invocations on T threads is cut into chunks of "
            f"max(ceil(N / 8T), {floor}) invocations (about 8 chunks per thread, but never fewer than {_n(floor)} "
            "per chunk); one chunk or fewer means no tasks at all. An invocation is one run of the loop body for one index. "
            "The tasks column is ceil(N / chunk size) when that is more than one, and 0 otherwise.\n\n"
            + _table(["threads T", "invocations N", "chunk size", "tasks"], rows))


def false_sharing():
    writes, workers = 1000, 4
    rows = []
    for layout, slot in (("adjacent 8-byte counters", 8), ("one counter per line", LINE)):
        row = [layout, math.ceil(workers * slot / LINE)]
        for burst in (1, 8, 100, 1000):
            owner, transfers, done = None, 0, [0] * workers
            line_of = lambda t: (t * slot) // LINE   # noqa: E731
            owners = {}
            while min(done) < writes:
                for t in range(workers):
                    for _ in range(min(burst, writes - done[t])):
                        ln = line_of(t)
                        if owners.get(ln) not in (None, t):
                            transfers += 1
                        owners[ln] = t
                        done[t] += 1
            row.append(_n(transfers))
        rows.append(tuple(row))
    local = workers - 1
    return (f"{workers} workers each add 1 to their own 8-byte counter {_n(writes)} times. A line may be written "
            "only by the core that owns it; a write by another core takes the line away (one invalidation, one "
            "transfer) and the line then belongs to the writer. Counter t sits at byte 8t (adjacent) or 64t "
            "(padded). The schedule is `burst` writes by worker 0, then `burst` by worker 1, and so on round "
            "after round: burst 1 is the worst case, burst 1000 is no overlap at all. Line transfers:\n\n"
            + _table(["layout", "lines holding counters", "burst 1", "burst 8", "burst 100", "burst 1000"], rows)
            + f"\n\nWith each worker keeping its count in a register and writing it once at the end, there are at "
            f"most {local} transfers whatever the schedule.")


# --------------------------------------------------------- 8. work counts

def work_counts():
    n = 1024
    rows = []
    for name, flops, byts in (
        ("dot product of two vectors of n", 2 * n, 2 * n * ELEM),
        ("elementwise sum c = a + b, vectors of n", n, 3 * n * ELEM),
        ("matrix-vector product, [n, n] by [n]", 2 * n * n, (n * n + 2 * n) * ELEM),
        ("matrix product, [n, n] by [n, n]", 2 * n ** 3, 3 * n * n * ELEM),
    ):
        rows.append((name, _n(flops), _n(byts), f"{flops / byts:.3f}"))
    return (f"n = {n}, 4-byte numbers. FLOPs count each multiplication and each addition (a fused multiply-add is "
            "two). Bytes are the least that must cross the boundary of the core's cache: every input read once, "
            "every output written once.\n\n"
            + _table(["computation", "FLOPs", "bytes at least", "FLOPs per byte"], rows))
