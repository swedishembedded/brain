# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Parallel Reductions, Scans and Sorting".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: reductions.<name> -->` markers. Helpers with a leading
underscore, or that take arguments, are shared with the chapter's figures so a
picture and the table beside it cannot disagree.

Nothing here measures a machine. Speed is work: additions, dependent steps
(depth), barriers, memory words and the cycles of the shared-memory bank model.
The kernels are run by the kernel simulator of Chapter 5
(`compute_kernels.launch`), occupancy by the model of Chapter 4
(`gpu.occupancy`), the banks by the model of Chapter 6
(`matmul_fast.bank_degree`) and the roofline by the machines of Chapter 3
(`roofline.MACHINES`). Every sum in float32 rounds each addition to the nearest
float32 (ties to even) exactly as Volume I, Chapter 15 specifies; a sum of two
float32 numbers computed in binary64 and then rounded to float32 is that
rounding, because 53 >= 2 x 24 + 2. The randomness is drawn from the book's
SplitMix64 (`bookrng.generator`), one stream per example, named in each block.

Swedish Embedded AB implements solutions for fast and reproducible reductions,
scans and sorts on GPUs. If your team needs expertise in parallel algorithms
for machine learning kernels, you can procure our services by sending an email
to info@swedishembedded.com.
"""
import math
import struct
from fractions import Fraction

import bookrng
import compute_kernels as ck
import gpu
import matmul_fast as mf
import roofline as rf

# ------------------------------------------------------------------ parameters

F32 = 4                      # bytes per fp32 number and per u32 word
U = 2.0 ** -24               # unit roundoff of float32 (Volume I, Chapter 15)
LANES = gpu.LANES            # 10,240 lanes of the model GPU (Chapter 4)
SUBGROUP = ck.SUBGROUP       # 32
N_CU = gpu.N_CU              # 80 compute units
X16 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]   # the problem's sixteen numbers
X8 = [3, 1, 4, 1, 5, 9, 2, 6]                            # the scan example of Volume I, Chapter 16
REDUCE_WG = 256              # invocations of the chapter's reduction workgroup
REDUCE_REGS = 24             # registers per invocation of the reduction kernel (a parameter)
SCAN_BLOCK = 256             # elements one scan block holds (a parameter of the chapter)
_F32 = struct.Struct("<f")

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


def _v(x):
    """A float that is a whole number prints without a fraction."""
    return str(int(x)) if float(x) == int(x) else f"{x:g}"


def _vs(xs):
    return "[" + ", ".join(_v(x) for x in xs) + "]"


def _ceil_div(a, b):
    return -(-a // b)


def _yes(flag):
    return "yes" if flag else "no"


def _f32(x):
    return _F32.unpack(_F32.pack(x))[0]


def _bits(x):
    return f"{struct.unpack('<I', _F32.pack(x))[0]:08x}"


def _ibits(x):
    return struct.unpack("<I", _F32.pack(x))[0]


def _seed_note(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (the seed of the name `{name}`)."


def _lg(n):
    """log2 of a power of two, exactly."""
    assert n > 0 and n & (n - 1) == 0
    return n.bit_length() - 1


def _clog2(n):
    return (n - 1).bit_length()


# ------------------------------------------------------ 1. the sixteen numbers


def first_sixteen():
    n = len(X16)
    total = sum(X16)
    rows, vals, stride, adds = [], list(X16), n // 2, 0
    rows.append(("0", "-", "-", _vs(vals), 0))
    step = 0
    while stride >= 1:
        step += 1
        for l in range(stride):
            vals[l] = vals[l] + vals[l + stride]
        adds += stride
        rows.append((step, stride, f"0 .. {stride - 1}", _vs(vals[:stride]), adds))
        stride //= 2
    one_cell = []
    acc = 0
    for g in range(n):
        acc += X16[g]
        one_cell.append(acc)
    return (f"Sixteen numbers x = {_vs(X16)}, one per invocation, sum {total}. A tree step with stride s makes invocation l < s add "
            f"the value of invocation l + s to its own; the table lists the values the active invocations hold after each step "
            f"(the others are not used again).\n\n"
            + _table(["step", "stride s", "invocations adding", "values after the step", "additions so far"], rows)
            + f"\n\nThe other way: all {n} invocations update one cell that starts at 0. The cell serves one update per step, so "
              f"after step j it holds the sum of the first j numbers in the order the hardware chose; in order 0 .. 15 it holds "
              f"{_vs(one_cell)} after steps 1 .. {n}.\n\n"
            + _table(["way", "dependent steps (depth)", "additions (work)"],
                     [("one cell, every invocation updates it", n, n), ("tree", _lg(n), n - 1)]))


# ------------------------------------------------------------ 2. work and depth


def _brent(w, d, p):
    """(lower bound, greedy upper bound) on the steps of w operations of depth d on p workers."""
    return max(_ceil_div(w, p), d), _ceil_div(w - d, p) + d


def work_depth():
    rows = []
    for n in (16, 1024, 1 << 20):
        p = min(n, LANES)
        lo, hi = _brent(n - 1, _clog2(n), p)
        rows.append((_n(n), "one invocation walks the array", _n(n - 1), _n(n - 1), _n(n - 1)))
        rows.append((_n(n), "every invocation updates one cell", _n(n), _n(n), _n(n)))
        rows.append((_n(n), "tree", _n(n - 1), _clog2(n), f"{_n(lo)} to {_n(hi)}"))
    return (f"The three ways to sum n numbers, counted. Work = additions; depth = the longest chain of additions that wait for one "
            f"another; steps = how long n_wk workers need, n_wk = min(n, {_n(LANES)}) (the lanes of the model GPU of Chapter 4). For a "
            f"computation of work N_work and depth N_dep the steps are at least max(ceil(N_work / n_wk), N_dep), and a greedy schedule "
            f"needs at most ceil((N_work - N_dep) / n_wk) + N_dep; the tree row prints both. The cell serves one update per step "
            f"whatever n_wk is.\n\n"
            + _table(["n", "way", "work", "depth", "steps on n_wk workers"], rows, "rlrrr"))


# ----------------------------------------------- 3. the tree in one workgroup

STRIDES_UP = lambda n: [1 << k for k in range(_lg(n))]          # noqa: E731  1, 2, 4, ...
STRIDES_DOWN = lambda n: [n >> (k + 1) for k in range(_lg(n))]  # noqa: E731  n/2, ..., 1


def _tree_phases(n_wg, variant, trace=None, round32=False):
    """Phases of the workgroup reduction kernel: load, one phase per step, store.

    variant: "interleaved" (invocation l with l mod 2s = 0 adds s[l + s] to s[l], s = 1, 2, 4, ...),
             "strided" (invocation l adds s[2 s l + s] to s[2 s l] while 2 s l < n_wg, s = 1, 2, 4, ...),
             "sequential" (invocation l < s adds s[l + s] to s[l], s = n_wg / 2, ..., 1).
    `trace`, if given, receives for each step a dict lane -> (read index, second read index) of workgroup 0."""
    def load(inv, p):
        g = inv.gid_x
        inv.shared_store(inv.l, inv.load("x", g) if g < p["n"] else 0.0)

    def make_step(k, s):
        def step(inv, p):
            l = inv.l
            if variant == "interleaved":
                i = l if l % (2 * s) == 0 else None
            elif variant == "strided":
                i = 2 * s * l if 2 * s * l < n_wg else None
            else:
                i = l if l < s else None
            if i is None:
                return
            if trace is not None and inv.wx == 0:
                trace[k][l] = (i, i + s)
            v = inv.shared_load(i) + inv.shared_load(i + s)
            inv.shared_store(i, _f32(v) if round32 else v)
        return step

    def store(inv, p):
        if inv.l == 0:
            inv.store("part", inv.wx, inv.shared_load(0))

    strides = STRIDES_DOWN(n_wg) if variant == "sequential" else STRIDES_UP(n_wg)
    return [load] + [make_step(k, s) for k, s in enumerate(strides)] + [store], strides


def _run_tree(values, n_wg, variant, *, order="asc", trace=None, snapshots=None, round32=False):
    """Run the workgroup reduction over ceil(len / n_wg) workgroups; returns (partials, barriers)."""
    n = len(values)
    groups = _ceil_div(n, n_wg)
    mem = ck.Memory()
    bx = mem.alloc("x", values, "read")
    bp = mem.alloc("part", [ck.UNWRITTEN] * groups)
    mem.alloc("tail", [0.0] * 8)
    phases, _ = _tree_phases(n_wg, variant, trace, round32)
    ck.launch(mem, phases, groups, n_wg, {"n": n}, {"x": bx, "part": bp}, shared_words=n_wg, order=order,
              snapshots=snapshots)
    return mem.values("part"), len(phases) - 1


def tree_kernel_trace():
    n = len(X16)
    snaps = []
    part, barriers = _run_tree(X16, n, "sequential", snapshots=snaps)
    desc, _ = _run_tree(X16, n, "sequential", order="desc")
    rows = [("after the load", _vs(snaps[0][2]))]
    for (wg, k, sh), s in zip(snaps[1:], STRIDES_DOWN(n)):
        rows.append((f"after the step of stride {s}", _vs(sh)))
    ragged, _ = _run_tree(X16[:13], n, "sequential")
    return (f"The workgroup kernel `reduce_tree` of the chapter (sequential addressing) run by the simulator of Chapter 5 on "
            f"x = {_vs(X16)} with one workgroup of {n} invocations. The table prints the {n} slots of the shared array "
            f"at each barrier; slots from index s up are not touched by the step of stride s, so they keep their old values.\n\n"
            + _table(["moment", "shared array s[0 .. 15]"], rows, "ll")
            + f"\n\nThe store phase writes part[0] = {_v(part[0])} (the sum of x is {sum(X16)}). Barriers between phases: {barriers}. "
              f"Running the invocations of every phase in descending instead of ascending order gives {_v(desc[0])}: "
              f"same result, {_yes(desc == part)}. With only the first 13 numbers (N = 13, the three missing slots are loaded "
              f"as 0, the identity of the sum) part[0] = {_v(ragged[0])} against the direct sum {sum(X16[:13])}.")


# ----------------------------------------- 4. the three addressing patterns


def _step_stats(active, n_wg):
    """Subgroup accounting of one step. `active` maps lane -> (read, read + stride)."""
    sg = SUBGROUP
    issuing = divergent = cycles = 0
    worst = 1
    for g0 in range(0, n_wg, sg):
        lanes = [l for l in range(g0, min(g0 + sg, n_wg)) if l in active]
        if not lanes:
            continue
        issuing += 1
        divergent += len(lanes) < sg
        first = [active[l][0] for l in lanes]
        second = [active[l][1] for l in lanes]
        degs = [mf.bank_degree(first), mf.bank_degree(second), mf.bank_degree(first)]   # read, read, write
        cycles += sum(degs)
        worst = max(worst, *degs)
    return issuing, divergent, cycles, worst


def _variant_stats(n_wg, variant):
    trace = [dict() for _ in range(_lg(n_wg))]
    part, barriers = _run_tree([1.0] * n_wg, n_wg, variant, trace=trace)
    assert part == [float(n_wg)]
    return [_step_stats(t, n_wg) + (len(t),) for t in trace], barriers


VARIANTS = (("interleaved", "A: interleaved, l mod 2s = 0"), ("strided", "B: interleaved, index 2sl"),
            ("sequential", "C: sequential, l < s"))


def addressing_steps():
    n_wg = REDUCE_WG
    stats = {v: _variant_stats(n_wg, v)[0] for v, _ in VARIANTS}
    rows = []
    for k in range(_lg(n_wg)):
        up, down = STRIDES_UP(n_wg)[k], STRIDES_DOWN(n_wg)[k]
        active = stats["interleaved"][k][4]
        cells = [f"{stats[v][k][0]} / {stats[v][k][1]} / {stats[v][k][3]}" for v, _ in VARIANTS]
        rows.append((k + 1, f"{up} (A, B), {down} (C)", active, str(Fraction(active, n_wg)), *cells))
    return (f"One workgroup of {n_wg} invocations ({n_wg // SUBGROUP} subgroups of {SUBGROUP}) reduces {n_wg} numbers; the three kernels "
            f"differ only in which invocation does the addition of a step. The simulator ran each and logged the shared-memory "
            f"addresses of workgroup 0. Per step and variant the cell shows: subgroups issuing the addition (some lane active) / of them "
            f"divergent (some lane idle) / worst bank-conflict degree of the step's three shared accesses "
            f"(two reads and a write, the banks of Chapter 6: 32 banks of 4-byte words, word a in bank a mod 32).\n\n"
            + _table(["step", "stride s", "active invocations", "share of the workgroup active"]
                     + [name for _, name in VARIANTS], rows, "rlrrlll"))


def addressing_totals():
    n_wg = REDUCE_WG
    rows = []
    for v, name in VARIANTS:
        st, barriers = _variant_stats(n_wg, v)
        issuing = sum(s[0] for s in st)
        divergent = sum(s[1] for s in st)
        cycles = sum(s[2] for s in st)
        active = sum(s[4] for s in st)
        rows.append((name, barriers, issuing, divergent, _n(issuing * SUBGROUP),
                     f"{100 * active / (issuing * SUBGROUP):.1f}", cycles, 3 * issuing))
    return (f"Totals over the {_lg(n_wg)} steps for a workgroup of {n_wg}. Issue slots = subgroup-steps issuing the addition x "
            f"{SUBGROUP} lanes; lane utilization = additions ({n_wg - 1}) / issue slots; shared cycles = sum over issuing subgroup-steps "
            f"of the bank degrees of the two reads and the write; the last column is the same count with no conflicts. The barriers "
            f"are those between the {_lg(n_wg) + 2} phases (load, {_lg(n_wg)} steps, store).\n\n"
            + _table(["variant", "barriers", "subgroup-steps issuing", "of them divergent", "issue slots",
                      "lane utilization (%)", "shared cycles", "cycles without conflicts"], rows))


# ------------------------------------------- 5. beyond one workgroup: two passes


def _resident_workgroups():
    per_cu, _, _ = gpu.occupancy(REDUCE_WG, REDUCE_REGS, REDUCE_WG * F32)
    return per_cu, per_cu * N_CU


def single_dispatch_limit():
    per_cu, total = _resident_workgroups()
    rows = []
    for e in (16, 18, 20, 22, 24):
        n = 1 << e
        wgs = _ceil_div(n, REDUCE_WG)
        rows.append((f"2^{e}", _n(wgs), _n(total), _yes(wgs <= total), _ceil_div(wgs, total)))
    return (f"A reduction kernel of {REDUCE_WG} invocations, {REDUCE_REGS} registers per invocation and {_n(REDUCE_WG * F32)} bytes of shared "
            f"memory per workgroup fills {per_cu} workgroups per compute unit by the model of Chapter 4 (`gpu.occupancy`: limited by "
            f"invocation slots, {_n(gpu.MAX_THREADS_PER_CU)} // {REDUCE_WG}), so the {N_CU} compute units hold {_n(total)} workgroups at once. "
            f"A barrier across the whole grid is only safe if every workgroup of the grid is resident while it waits. With one element "
            f"per invocation:\n\n"
            + _table(["n", "workgroups", "resident at once", "all resident (a grid barrier could work)", "waves of workgroups"], rows))


def two_pass():
    n, wg = 1 << 24, REDUCE_WG
    _, resident = _resident_workgroups()
    rows = []
    for f in (1, 4, 16, 64, 128):
        wgs = n // (wg * f)
        # pass 1: every invocation adds f numbers, a tree adds the wg partials; pass 2: one workgroup of wg invocations
        # adds ceil(wgs / wg) partials each and a tree adds the wg results
        per2 = _ceil_div(wgs, wg)
        adds = (n - n // f) + wgs * (wg - 1) + wg * (per2 - 1) + (wg - 1)
        depth = (f - 1) + _lg(wg) + (per2 - 1) + _lg(wg)
        bytes1 = F32 * (n + wgs)
        bytes2 = F32 * (wgs + 1)
        total = bytes1 + bytes2
        rows.append((f, _n(wgs), _ceil_div(wgs, resident), _n(adds), depth, _n(total), f"{100 * (total / (F32 * n) - 1):.3f}"))
    flop_per_byte = (n - 1) / (F32 * n)
    peak, bw = rf.MACHINES["GPU"]
    return (f"The sum of n = 2^24 = {_n(n)} fp32 numbers in two dispatches. Pass 1: {wg} invocations per workgroup, each adds n_el numbers "
            f"spaced a whole grid apart (a grid-stride loop, n_el - 1 additions), then the workgroup's tree adds the {wg} results "
            f"({wg - 1} additions) and writes one partial. Pass 2: one workgroup of {wg} invocations adds the partials, each "
            f"invocation ceil(partials / {wg}) of them, then a tree adds the {wg} results. 'Depth' counts the chain of dependent "
            f"additions of both passes; 'waves' is ceil(workgroups / {_n(resident)} resident); bytes = 4 x (words read + words "
            f"written) with each word counted once per dispatch: pass 1 reads n and writes one partial per workgroup, pass 2 reads "
            f"the partials and writes 1.\n\n"
            + _table(["n_el", "workgroups of pass 1", "waves", "additions in all", "depth", "bytes moved in all",
                      "bytes above the minimum 4n (%)"], rows)
            + f"\n\nEvery row does n - 1 = {_n(n - 1)} additions on 4n = {_n(F32 * n)} bytes of input: {flop_per_byte:.4f} operations per "
              f"byte, against the ridge point {peak / bw:.0f} of the model GPU (Chapter 3), so a fraction {flop_per_byte / (peak / bw):.4f} "
              f"of its peak is all a reduction can use: its cost is its bytes.")


# ------------------------------------------------- 6. the order of the additions


def _seq(xs):
    """Left fold in float32, starting from the first element."""
    acc = xs[0]
    for x in xs[1:]:
        acc = _f32(acc + x)
    return acc


def _tree_adjacent(xs):
    """Adjacent pairs, layer by layer (the tree of Volume I, Chapter 16); len(xs) is a power of two."""
    v = list(xs)
    assert v and len(v) & (len(v) - 1) == 0
    while len(v) > 1:
        v = [_f32(v[i] + v[i + 1]) for i in range(0, len(v), 2)]
    return v[0]


def _tree_halves(xs):
    """Sequential addressing: slot l adds slot l + s, s = len / 2, ..., 1; len(xs) is a power of two."""
    v = list(xs)
    s = len(v) // 2
    assert v and len(v) & (len(v) - 1) == 0
    while s:
        v = [_f32(v[l] + v[l + s]) for l in range(s)]
        s //= 2
    return v[0]


def _strided(xs, lanes):
    """Invocation j adds xs[j], xs[j + lanes], ... left to right, then a sequential-addressing tree adds the lanes."""
    acc = []
    for j in range(lanes):
        part = xs[j::lanes]
        acc.append(_seq(part) if part else 0.0)
    return _tree_halves(acc)


def _exact_scaled(xs):
    """The exact sum of float32 numbers as an integer multiple of 2^-149."""
    return sum(int(x * 2.0 ** 149) for x in xs)


def _err_u(computed, exact_scaled):
    """|computed - exact| / exact in units of the unit roundoff u = 2^-24."""
    ex = Fraction(exact_scaled, 1 << 149)
    return float(abs(Fraction(computed) - ex) / ex / U)


ERR_NAME = "reductions.summation_error"
ERR_NMAX = 1 << 18
ERR_REPS = 6
ERR_LANES = 256
CONST_01 = _f32(0.1)


def _prefix_errors(xs, sizes):
    """Relative error in units of u of the left fold of the first n numbers, for every n in `sizes`, in one pass."""
    out, acc, exact, want = {}, 0.0, 0, set(sizes)
    for i, x in enumerate(xs):
        acc = x if i == 0 else _f32(acc + x)
        exact += int(x * 2.0 ** 149)
        if i + 1 in want:
            out[i + 1] = _err_u(acc, exact)
    return out


def error_series(lo=3, hi=18):
    """Rows (n, left fold, tree, lanes, constant left fold, constant tree), in units of u, for n = 2^lo .. 2^hi.
    The first three columns are means over ERR_REPS independent vectors of uniform numbers."""
    sizes = [1 << e for e in range(lo, hi + 1)]
    g = bookrng.generator(ERR_NAME)
    sums = {n: [0.0, 0.0, 0.0] for n in sizes}
    for _ in range(ERR_REPS):
        xs = [_f32(g.uniform()) for _ in range(1 << hi)]
        seq = _prefix_errors(xs, sizes)
        for n in sizes:
            data = xs[:n]
            ex = _exact_scaled(data)
            sums[n][0] += seq[n]
            sums[n][1] += _err_u(_tree_adjacent(data), ex)
            sums[n][2] += _err_u(_strided(data, ERR_LANES), ex)
    const = [CONST_01] * (1 << hi)
    cseq = _prefix_errors(const, sizes)
    rows = []
    for n in sizes:
        ex = _exact_scaled(const[:n])
        rows.append((n, *(v / ERR_REPS for v in sums[n]), cseq[n], _err_u(_tree_adjacent(const[:n]), ex)))
    return rows


def summation_error():
    rows = []
    for n, a, b, c, d, e in error_series(4, 18)[::2]:
        rows.append((_n(n), _n(n - 1), _clog2(n), f"{a:.3g}", f"{b:.3g}", f"{c:.3g}", f"{d:.3g}", f"{e:.3g}"))
    return (f"Relative error |computed - exact| / exact of a float32 sum, in units of u = 2^-24, against the bounds (n - 1) u of the left "
            f"fold and ceil(log2 n) u of the tree (Volume I, Chapter 15). Every addition is rounded to float32. 'Uniform' data: "
            f"{ERR_REPS} vectors of {_n(ERR_NMAX)} numbers x_i = float32(`uniform()`), drawn vector after vector in order from one "
            f"generator; the first n of each vector are summed and the table prints the mean error over the {ERR_REPS} vectors. "
            f"{_seed_note(ERR_NAME)} 'Constant' data: n copies of float32(0.1) = {CONST_01!r}. 'Left fold' adds x_0, x_1, ... in "
            f"order; 'tree' adds adjacent pairs layer by layer; 'lanes' is the arrangement of a workgroup: invocation j (of "
            f"{ERR_LANES}) adds x_j, x_(j+{ERR_LANES}), ... left to right, then a sequential-addressing tree adds the {ERR_LANES} "
            f"results. The exact sum is computed in integer arithmetic.\n\n"
            + _table(["n", "bound (n - 1)", "bound ceil(log2 n)", "uniform: left fold", "uniform: tree", "uniform: lanes",
                      "constant: left fold", "constant: tree"], rows))


def tree_shapes():
    name, trials, n = "reductions.tree_shapes", 200, 256
    g = bookrng.generator(name)
    cnt = {"A vs C": 0, "A vs left fold": 0, "C vs left fold": 0}
    err = {"left fold": 0.0, "A (and B)": 0.0, "C": 0.0}
    agree = True
    for t in range(trials):
        xs = [_f32(g.uniform()) for _ in range(n)]
        a, c, s = _tree_adjacent(xs), _tree_halves(xs), _seq(xs)
        cnt["A vs C"] += a != c
        cnt["A vs left fold"] += a != s
        cnt["C vs left fold"] += c != s
        ex = _exact_scaled(xs)
        err["left fold"] += _err_u(s, ex)
        err["A (and B)"] += _err_u(a, ex)
        err["C"] += _err_u(c, ex)
        if t == 0:
            for variant, want in (("interleaved", a), ("strided", a), ("sequential", c)):
                part, _ = _run_tree(xs, n, variant, round32=True)
                agree = agree and part[0] == want
    rows = [(k, f"{v} of {trials}") for k, v in cnt.items()]
    rows2 = [(k, f"{v / trials:.3g}") for k, v in err.items()]
    return (f"{trials} vectors of {n} numbers, x = float32(`uniform()`), vector by vector in order. {_seed_note(name)} Each vector is "
            f"summed by the left fold, by tree A (pairs of neighbours: kernels A and B of the previous blocks) and by tree C "
            f"(slot l adds slot l + s, s = 128, ..., 1), all in float32. The simulator of Chapter 5, running kernels A, B and C with "
            f"every addition rounded to float32, returns the same bits as the direct functions on vector 0: {_yes(agree)}.\n\n"
            + _table(["pair of results", "bitwise different on"], rows, "lr")
            + "\n\nMean relative error against the exact sum, in units of u = 2^-24:\n\n"
            + _table(["order", "mean error / u"], rows2, "lr"))


def _ulps_from(value, ref):
    return _ibits(value) - _ibits(ref)


def _two_pass_f32(xs, wg, n_wg, f, wg2=64):
    stride = n_wg * wg
    parts = []
    for w in range(n_wg):
        lane = []
        for t in range(wg):
            i0 = w * wg + t
            acc = xs[i0]
            for k in range(1, f):
                acc = _f32(acc + xs[i0 + k * stride])
            lane.append(acc)
        parts.append(_tree_halves(lane))
    lane2 = []
    for t in range(wg2):
        mine = parts[t::wg2]
        lane2.append(_seq(mine) if mine else 0.0)
    return _tree_halves(lane2)


SHAPES = ((256, 64, 4), (64, 128, 8), (256, 16, 16))


def launch_shape_bits():
    name, n = "reductions.launch_shape", 1 << 16
    g = bookrng.generator(name)
    xs = [_f32(g.normal()) for _ in range(n)]
    ex = _exact_scaled(xs)
    ref = mf._round_f32_exact(Fraction(ex, 1 << 149))
    oname = "reductions.atomic_orders"
    og = bookrng.generator(oname)
    rows = []

    def add(label, val, again):
        rows.append((label, f"{val:.9g}", _bits(val), f"{_ulps_from(val, ref):+d}", _yes(again == val)))

    add("left fold", _seq(xs), _seq(xs))
    add("tree of adjacent pairs", _tree_adjacent(xs), _tree_adjacent(xs))
    for wg, n_wg, f in SHAPES:
        add(f"two passes: workgroup {wg}, {n_wg} workgroups, n_el = {f}", _two_pass_f32(xs, wg, n_wg, f),
            _two_pass_f32(xs, wg, n_wg, f))
    for k in range(3):
        perm = og.shuffle(list(range(n)))
        acc = 0.0
        for i in perm:
            acc = _f32(acc + xs[i])
        rows.append((f"one cell, update order = shuffle {k + 1}", f"{acc:.9g}", _bits(acc), f"{_ulps_from(acc, ref):+d}", "-"))
    return (f"The same n = {_n(n)} numbers x_i = float32(`normal()`), drawn in order from the generator of the name `{name}`, summed "
            f"in different orders, every addition rounded to float32. {_seed_note(name)} The exact sum rounded to float32 is "
            f"{ref:.9g} (bits {_bits(ref)}); 'ulps' is the distance of a result from it in float32 steps. 'Two passes' is the "
            f"kernel of the chapter: pass 1 has n_wg workgroups of the given size, invocation t of workgroup w adds the n_el numbers "
            f"x[(k n_wg + w) size + t], k = 0 .. n_el - 1, left to right, and a sequential-addressing tree adds the workgroup; "
            f"pass 2 is one workgroup of 64 invocations, invocation t adding partials t, t + 64, ... left to right (missing ones "
            f"count as 0), then the same tree. The three 'one cell' rows model an unordered accumulate: the updates arrive in the "
            f"order of a shuffle of 0 .. n - 1 (the book's Fisher-Yates), three shuffles drawn one after another from the "
            f"generator of the name `{oname}` (seed {bookrng.seed_of(oname):#018x}). 'Same on rerun' repeats the whole computation.\n\n"
            + _table(["order", "result", "bits", "ulps from exact", "same on rerun"], rows, "lrrrr"))


def argmax_ties():
    x = [2, 9, 3, 1, 9, 4, 0, 5, 6, 2, 1, 7]
    lanes = 4

    def lower_wins(a, b):          # a pair is (value, index); the larger value wins, a tie goes to the lower index
        return b if (b[0] > a[0] or (b[0] == a[0] and b[1] < a[1])) else a

    def left_wins(a, b):           # keep a unless b is strictly larger
        return b if b[0] > a[0] else a

    seq = (x[0], 0)
    for i in range(1, len(x)):
        seq = lower_wins(seq, (x[i], i))

    def lane_best(t, owned):
        best = (x[owned[0]], owned[0])
        for i in owned[1:]:
            best = lower_wins(best, (x[i], i))
        return best

    def tree(parts, comb):
        v = list(parts)
        s = len(v) // 2
        while s:
            v = [comb(v[l], v[l + s]) for l in range(s)]
            s //= 2
        return v[0]

    strided = [lane_best(t, list(range(t, len(x), lanes))) for t in range(lanes)]
    chunk = len(x) // lanes
    chunked = [lane_best(t, list(range(t * chunk, (t + 1) * chunk))) for t in range(lanes)]
    pair = lambda p: f"({p[0]}, {p[1]})"  # noqa: E731
    rows = [("left to right scan, lower index wins ties", pair(seq)),
            (f"{lanes} invocations own indices t, t + {lanes}, ...; tree; 'keep the left on a tie'", pair(tree(strided, left_wins))),
            (f"{lanes} invocations own indices t, t + {lanes}, ...; tree; 'lower index wins a tie'", pair(tree(strided, lower_wins))),
            (f"{lanes} invocations own runs of {chunk} (indices {chunk} t .. {chunk} t + {chunk - 1}); tree; 'keep the left on a tie'",
             pair(tree(chunked, left_wins)))]
    parts = lambda ps: ", ".join(pair(p) for p in ps)  # noqa: E731
    return (f"x = {_vs(x)} (N = {len(x)}); the maximum 9 occurs at indices 1 and 4. A pair is (value, index); the sequential scan keeps the "
            f"first maximum. Partial results of the {lanes} invocations, strided ownership: {parts(strided)}; run ownership: "
            f"{parts(chunked)}. Trees use sequential addressing (invocation l combines its pair with that of l + s, s = "
            f"{lanes // 2}, ..., 1, the left operand being invocation l).\n\n"
            + _table(["method", "(value, index) returned"], rows, "ll"))


# ------------------------------------------------------------------ 7. scans


def _scan_load(inv, p):
    inv.shared_store(inv.l, inv.load("x", inv.gid_x))


def _hillis_steele_phases(n, counter):
    """Inclusive scan, one workgroup of n invocations. Each step reads into a register, waits, then writes (two barriers)."""
    regs = {}

    def make(d):
        def read(inv, p):
            v = inv.shared_load(inv.l)
            if inv.l >= d:
                v += inv.shared_load(inv.l - d)
                counter[0] += 1
            regs[(inv.wx, inv.l)] = v

        def write(inv, p):
            inv.shared_store(inv.l, regs[(inv.wx, inv.l)])
        return read, write

    phases, d = [_scan_load], 1
    while d < n:
        phases.extend(make(d))
        d *= 2
    phases.append(lambda inv, p: inv.store("y", inv.gid_x, inv.shared_load(inv.l)))
    return phases


def _blelloch_phases(n, counter):
    """Exclusive scan, one workgroup of n/2 busy invocations (n invocations load): up-sweep, clear the root, down-sweep."""
    def up(d):
        def step(inv, p):
            if inv.l < n // (2 * d):
                i = 2 * d * (inv.l + 1) - 1
                inv.shared_store(i, inv.shared_load(i) + inv.shared_load(i - d))
                counter[0] += 1
        return step

    def clear(inv, p):
        if inv.l == 0:
            inv.shared_store(n - 1, 0.0)

    def down(d):
        def step(inv, p):
            if inv.l < n // (2 * d):
                i = 2 * d * (inv.l + 1) - 1
                t = inv.shared_load(i - d)
                inv.shared_store(i - d, inv.shared_load(i))
                inv.shared_store(i, inv.shared_load(i) + t)
                counter[0] += 1
        return step

    ups = [up(1 << k) for k in range(_lg(n))]
    downs = [down(n >> (k + 1)) for k in range(_lg(n))]
    return [_scan_load] + ups + [clear] + downs + [lambda inv, p: inv.store("y", inv.gid_x, inv.shared_load(inv.l))]


def _run_scan(values, kind, snapshots=None):
    """Run a scan kernel in one workgroup; returns (output, additions, barriers)."""
    n = len(values)
    counter = [0]
    phases = (_hillis_steele_phases if kind == "hillis" else _blelloch_phases)(n, counter)
    mem = ck.Memory()
    bx = mem.alloc("x", values, "read")
    by = mem.alloc("y", [ck.UNWRITTEN] * n)
    mem.alloc("tail", [0.0] * 8)
    ck.launch(mem, phases, 1, n, {}, {"x": bx, "y": by}, shared_words=n, snapshots=snapshots)
    return mem.values("y"), counter[0], len(phases) - 1


def _seq_scan(xs):
    inc, acc = [], 0
    for x in xs:
        acc += x
        inc.append(acc)
    return inc, [0] + inc[:-1]


def scan_trace_hillis():
    n = len(X8)
    inc, _ = _seq_scan(X8)
    snaps = []
    hs, adds, bars = _run_scan(X8, "hillis", snaps)
    rows = [(lab, _vs(arr)) for lab, arr in scan_snapshots(X8, "hillis")]
    return (f"Input x = {_vs(X8)} (Volume I, Chapter 16), one workgroup of {n} invocations, run by the simulator of Chapter 5. "
            f"Sequential inclusive scan {_vs(inc)}. The shared array at each step (every step reads, waits at a barrier, "
            f"writes, waits again):\n\n"
            + _table(["moment", "shared array"], rows, "ll")
            + f"\n\nOutput {_vs(hs)}; equals the sequential inclusive scan: {_yes(hs == inc)}. Additions {adds}, barriers {bars}.")


def scan_trace_blelloch():
    n = len(X8)
    _, exc = _seq_scan(X8)
    bl, adds, bars = _run_scan(X8, "blelloch")
    rows = [(lab, _vs(arr)) for lab, arr in scan_snapshots(X8, "blelloch")]
    return (f"The same input x = {_vs(X8)}, one workgroup of {n} invocations (at most {n // 2} of them add in any step). Sequential "
            f"exclusive scan {_vs(exc)}. The shared array at each barrier:\n\n"
            + _table(["moment", "shared array"], rows, "ll")
            + f"\n\nOutput {_vs(bl)}; equals the sequential exclusive scan: {_yes(bl == exc)}. Additions {adds}, barriers {bars}.")


def scan_costs():
    name = "reductions.scan_check"
    g = bookrng.generator(name)
    rows = []
    for n in (8, 16, 64, 256):
        xs = [g.integer(100) for _ in range(n)]
        inc, exc = _seq_scan(xs)
        hs, ha, hb = _run_scan(xs, "hillis")
        bl, ba, bb = _run_scan(xs, "blelloch")
        rows.append((n, _yes(hs == inc), ha, n * _lg(n) - n + 1, hb, _yes(bl == exc), ba, 2 * (n - 1), bb))
    big = []
    for e in (10, 20):
        n = 1 << e
        p = min(n, LANES)
        for label, w, d in (("sequential", n - 1, n - 1), ("Hillis-Steele", n * e - n + 1, e), ("Blelloch", 2 * (n - 1), 2 * e)):
            big.append((_n(n), label, _n(w), _n(d), _n(max(_ceil_div(w, p), d))))
    return (f"Both scans in one workgroup of n invocations on seeded integers, checked against the sequential scan (inclusive for "
            f"Hillis-Steele, exclusive for Blelloch). {_seed_note(name)} Each size draws its n values with `integer(100)` in "
            f"order, the sizes one after another. 'Additions' are counted by the simulator; the formula columns are n log2 n - n + 1 "
            f"and 2 (n - 1).\n\n"
            + _table(["n", "Hillis-Steele equal", "additions", "formula", "barriers", "Blelloch equal", "additions", "formula",
                      "barriers"], rows)
            + f"\n\nWork, depth and the least number of steps on n_wk = min(n, {_n(LANES)}) workers, max(ceil(N_work / n_wk), N_dep), by "
            f"the formulas:\n\n"
            + _table(["n", "algorithm", "work N_work", "depth N_dep", "steps on n_wk workers (at least)"], big, "llrrr"))


def _scan_plan(n, block=SCAN_BLOCK):
    """Sizes scanned at each level of the multi-block scan: n, ceil(n / block), ... until one block remains."""
    sizes = [n]
    while _ceil_div(sizes[-1], block) > 1:
        sizes.append(_ceil_div(sizes[-1], block))
    return sizes


def _scan_words(n, block=SCAN_BLOCK):
    """Words read and written by the multi-block scan, each word counted once per dispatch."""
    sizes = _scan_plan(n, block)
    words = 0
    for j, nj in enumerate(sizes):
        nb = _ceil_div(nj, block)
        words += nj + (nj + nb)                      # block scan: read the run, write the scanned run and the block total
        if j < len(sizes) - 1:
            words += (nj + nb) + nj                  # add-back: read the run and its offset, write the run
    return words


def block_scan_dispatches():
    rows = []
    for e in (16, 20, 24):
        n = 1 << e
        sizes = _scan_plan(n)
        chain = sum(min(SCAN_BLOCK, s) - 1 for s in sizes) + (len(sizes) - 1)
        words = _scan_words(n)
        rows.append((f"2^{e}", " -> ".join(_n(s) for s in sizes), 2 * len(sizes) - 1, _n(_ceil_div(n, SCAN_BLOCK)), chain,
                     _n(words), f"{words / (2 * n):.3f}"))
    return (f"The exclusive scan of n u32 words in blocks of {SCAN_BLOCK}: a dispatch with one invocation per block scans its block "
            f"serially in place and writes the block's total; the totals are scanned the same way, recursively, until they fit one "
            f"block; then one dispatch per level above the first adds the scanned totals back. 'Levels' lists the number of words "
            f"scanned at each level. Dispatches = levels block scans + (levels - 1) add-backs. 'Chain' = the longest chain of "
            f"dependent additions (each block scan adds up to {SCAN_BLOCK - 1} times in sequence, each add-back once). Words are "
            f"counted once per dispatch (block scan: read n_j, write n_j + totals; add-back: read n_j + totals, write n_j); the "
            f"minimum for a scan is 2n (read every word, write every word).\n\n"
            + _table(["n", "levels (words scanned)", "dispatches", "invocations in the first dispatch", "chain",
                      "words moved", "ratio to the minimum 2n"], rows, "lllrrrr"))


# ------------------------------------------------------------- 8. radix sort

RS_NAME = "reductions.radix_trace"
RS_N, RS_KEYBITS, RS_DIGIT, RS_CHUNK = 16, 6, 3, 4


def _radix_pass(mem, bufs, n, shift, digit_bits, chunk, counts):
    """One pass of the sort as three dispatches (histogram, scan, scatter); `counts` accumulates loads and stores."""
    radix = 1 << digit_bits
    c = _ceil_div(n, chunk)

    def tally(kind, k=1):
        counts[kind] += k

    def hist(inv, p):
        if inv.gid_x >= c:
            inv.ret()
        local = [0] * radix
        for i in range(inv.gid_x * chunk, min((inv.gid_x + 1) * chunk, n)):
            local[(int(inv.load("keys_in", i)) >> shift) & (radix - 1)] += 1
            tally("load")
        for d in range(radix):
            inv.store("hist", d * c + inv.gid_x, local[d])
            tally("store")

    def scan(inv, p):
        if inv.gid_x != 0:
            inv.ret()
        running = 0
        for i in range(radix * c):
            v = int(inv.load("hist", i))
            inv.store("hist", i, running)
            tally("load")
            tally("store")
            running += v
        inv.store("total", 0, running)
        tally("store")

    def scatter(inv, p):
        if inv.gid_x >= c:
            inv.ret()
        off = []
        for d in range(radix):
            off.append(int(inv.load("hist", d * c + inv.gid_x)))
            tally("load")
        for i in range(inv.gid_x * chunk, min((inv.gid_x + 1) * chunk, n)):
            k = int(inv.load("keys_in", i))
            v = inv.load("vals_in", i)
            tally("load", 2)
            d = (k >> shift) & (radix - 1)
            inv.store("keys_out", off[d], k)
            inv.store("vals_out", off[d], v)
            tally("store", 2)
            off[d] += 1

    snap = {}
    for kernel in (hist, scan, scatter):
        ck.launch(mem, [kernel], 1, c, {}, bufs)
        if kernel is hist:
            snap["hist"] = [int(v) for v in mem.values("hist")]
        if kernel is scan:
            snap["scan"] = [int(v) for v in mem.values("hist")]
    return snap


def _radix_sort_sim(keys, key_bits, digit_bits, chunk):
    """Stable LSD radix sort of (key, original index) pairs on the simulator; returns passes' snapshots and counts."""
    n = len(keys)
    radix = 1 << digit_bits
    c = _ceil_div(n, chunk)
    mem = ck.Memory()
    ka = mem.alloc("ka", keys)
    va = mem.alloc("va", list(range(n)))
    kb = mem.alloc("kb", [ck.UNWRITTEN] * n)
    vb = mem.alloc("vb", [ck.UNWRITTEN] * n)
    h = mem.alloc("hist", [0] * (radix * c))
    t = mem.alloc("total", [0])
    mem.alloc("tail", [0.0] * 8)
    cur, other = ("ka", "va"), ("kb", "vb")
    counts = {"load": 0, "store": 0}
    passes = []
    for shift in range(0, key_bits, digit_bits):
        bufs = {"keys_in": mem.buffers[cur[0]], "vals_in": mem.buffers[cur[1]], "keys_out": mem.buffers[other[0]],
                "vals_out": mem.buffers[other[1]], "hist": h, "total": t}
        snap = _radix_pass(mem, bufs, n, shift, digit_bits, chunk, counts)
        cur, other = other, cur
        snap["keys"] = [int(v) for v in mem.values(cur[0])]
        snap["vals"] = [int(v) for v in mem.values(cur[1])]
        passes.append(snap)
    return passes, counts


def radix_trace():
    g = bookrng.generator(RS_NAME)
    n, kb, db, ch = RS_N, RS_KEYBITS, RS_DIGIT, RS_CHUNK
    keys = [g.integer(1 << kb) for _ in range(n)]
    radix, c = 1 << db, _ceil_div(n, ch)
    passes, counts = _radix_sort_sim(keys, kb, db, ch)
    first = passes[0]
    rows = []
    for d in range(radix):
        rows.append((d, *(first["hist"][d * c + j] for j in range(c)), *(first["scan"][d * c + j] for j in range(c))))
    pair = lambda k, v: f"{k}:{v}"  # noqa: E731
    stage = [("input", [pair(k, i) for i, k in enumerate(keys)])]
    for p, snap in enumerate(passes):
        stage.append((f"after pass {p} (digit {p}, bits {db * p} .. {db * p + db - 1})",
                      [pair(k, v) for k, v in zip(snap["keys"], snap["vals"])]))
    final = passes[-1]
    want = sorted(range(n), key=lambda i: (keys[i], i))
    stable = final["vals"] == want
    h_words = radix * c
    formula = len(passes) * (5 * n + 2 * h_words + _scan_words(h_words))
    got = counts["load"] + counts["store"]
    return (f"{n} keys of {kb} bits (0 .. {(1 << kb) - 1}), each paired with its input position; digits of {db} bits ({radix} values), "
            f"{len(passes)} passes from the least significant digit; chunks of {ch} keys, so {c} chunks and one invocation per chunk. "
            f"{_seed_note(RS_NAME)} The keys are {n} calls of `integer({1 << kb})` in order: {_vs(keys)}. Pass 0 histogram: "
            f"entry (d, j) is the number of keys of chunk j with digit d, stored at hist[{c} d + j] (digit-major); the scan of the "
            f"whole array is the position where chunk j writes its first key with digit d.\n\n"
            + _table(["digit d"] + [f"count, chunk {j}" for j in range(c)] + [f"start, chunk {j}" for j in range(c)], rows)
            + "\n\nThe pairs key:position after each pass:\n\n"
            + _table(["moment", "pairs"], [(a, " ".join(b)) for a, b in stage], "ll")
            + f"\n\nThe keys of the last pass are sorted: {_yes(final['keys'] == sorted(keys))}; equal keys keep their input order "
              f"(positions equal the stable order): {_yes(stable)}. Words moved by the simulated dispatches: {counts['load']} loads + "
              f"{counts['store']} stores = {got}; the formula 5n + 2 N_hist + S(N_hist) per pass with n = {n}, N_hist = {h_words} histogram words and "
              f"S(N_hist) = {_scan_words(h_words)} words of the scan gives {formula}: {_yes(got == formula)}.")


def _sort_pass_words(n, digit_bits, chunk):
    """Words moved by one pass of a pair sort: histogram (read keys, write N_hist), scan of N_hist, scatter (read keys, values
    and N_hist, write keys and values): 5n + 2 N_hist + S(N_hist) with N_hist = 2^digit_bits n / chunk."""
    h = (1 << digit_bits) * _ceil_div(n, chunk)
    return 5 * n + 2 * h + _scan_words(h), h


def radix_costs():
    n = 1 << 20
    rows = []
    for b in (1, 2, 4, 6, 8, 11, 16):
        passes = _ceil_div(32, b)
        per, h = _sort_pass_words(n, b, 1024)
        rows.append((b, passes, _n(h), str(Fraction(h, n)), _n(per), _n(passes * per), f"{passes * per / n:.1f}"))
    n2 = 1 << 24
    rows2 = []
    for chunk in (256, 512, 1024, 2048, 4096, 8192):
        per, h = _sort_pass_words(n2, 8, chunk)
        c = _ceil_div(n2, chunk)
        rows2.append((_n(chunk), _n(c), str(Fraction(h, n2)), f"{4 * per / n2:.2f}", f"{min(1.0, c / LANES):.2f}"))
    return (f"The words a radix sort of 32-bit keys with 32-bit payloads moves, counted by the rule of the trace above: one pass moves "
            f"5n + 2 N_hist + S(N_hist) words, N_hist = 2^b_dg n / L_ch histogram words for digits of b_dg bits and chunks of L_ch keys, "
            f"S(N_hist) the words of the {SCAN_BLOCK}-element block scan of N_hist words (previous block).\n\nFirst, n = 2^20 keys "
            f"and L_ch = 1,024 keys per chunk (1,024 chunks), the digit width varying; passes = ceil(32 / b_dg):\n\n"
            + _table(["b_dg", "passes", "N_hist", "N_hist / n", "words per pass", "words in all", "in all / n"], rows)
            + f"\n\nThen n = 2^24 keys, b_dg = 8 (four passes), the chunk length L_ch varying; 'lane fill' is min(1, chunks / "
              f"{_n(LANES)}) with {_n(LANES)} lanes:\n\n"
            + _table(["L_ch", "chunks n_ch (invocations)", "N_hist / n", "words in all / n", "lane fill"], rows2))


def topk_vs_sort():
    v = 1 << 17
    per, h = _sort_pass_words(v, 8, 256)
    sort_words = 4 * per
    n_wg = 256
    round_words = v + 2 * n_wg + 2 * n_wg + 1       # read V, write (value, index) pairs; read them, write the winner
    rows = []
    for k in (1, 2, 4, 8, 16, 32, 64):
        rows.append((k, _n(k * round_words), f"{k * round_words / v:.1f}", f"{sort_words / v:.1f}",
                     "rounds" if k * round_words < sort_words else "sort"))
    breakeven = next(k for k in range(1, 1000) if k * round_words >= sort_words)
    return (f"Selecting the k largest of V = 2^17 = {_n(v)} logits. Sort: the radix sort of the previous block, 32-bit keys, four passes "
            f"of b_dg = 8, chunks of L_ch = 256 keys (so N_hist = {_n(h)} histogram words), {_n(sort_words)} words in all. Rounds: k two-pass "
            f"argmax reductions, each reading V words, writing {n_wg} (value, index) pairs, reading them and writing the winner "
            f"({_n(round_words)} words a round; masking the winner is one word and not counted). The cheaper way is the one with "
            f"fewer words; the rounds stop being cheaper from k = {breakeven}.\n\n"
            + _table(["k", "rounds: words", "rounds / V", "sort / V", "fewer words"], rows, "rrrrl"))


def tree_structure(values, variant):
    """The tree a workgroup reduction builds: nested (value, left, right, index) tuples, leaves (value, None, None, index).
    Slot i combines with slot i + s in the order of `variant` ("interleaved" or "sequential"), as the kernels do."""
    n = len(values)
    slots = [(v, None, None, i) for i, v in enumerate(values)]
    strides = STRIDES_DOWN(n) if variant == "sequential" else STRIDES_UP(n)
    for s in strides:
        for l in range(n):
            if (variant == "sequential" and l < s) or (variant != "sequential" and l % (2 * s) == 0):
                a, b = slots[l], slots[l + s]
                slots[l] = (a[0] + b[0], a, b, None)
    return slots[0]


def scan_snapshots(values, kind):
    """(label, shared array) at every barrier of a scan kernel, for the figure; the array before the first step is the input."""
    snaps = []
    _run_scan(values, kind, snaps)
    n = len(values)
    if kind == "hillis":
        labels = ["input"] + [f"stride {d}" for d in STRIDES_UP(n)]
        picked = [snaps[0]] + [s for s in snaps if s[1] % 2 == 0 and s[1] > 0]
    else:
        labels = ["input"] + [f"up-sweep s = {d}" for d in STRIDES_UP(n)] + ["root cleared"] \
            + [f"down-sweep s = {d}" for d in STRIDES_DOWN(n)]
        picked = snaps
    return [(lab, list(sh)) for lab, (_, _, sh) in zip(labels, picked)]
