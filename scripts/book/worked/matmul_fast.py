# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Making Matrix Multiplication Fast".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: matmul_fast.<name> -->` markers. Helpers with a leading
underscore, or that take arguments, are shared with the chapter's figures so a
picture and the table beside it cannot disagree.

Nothing here measures a machine. Speed is work: floating-point operations and
bytes. The kernels are run by the kernel simulator of Chapter 5
(`compute_kernels.launch`), the cache by the cache simulator of Chapter 2
(`cpu.Cache`), occupancy by the model of Chapter 4 (`gpu.occupancy`) and the
roofline by the machines of Chapter 3 (`roofline.MACHINES`). Three parameters
are this chapter's own and are labelled as parameters wherever they are used:
the bandwidth of shared memory, the fixed registers of a kernel and the ratio
of the matrix unit's peak to the vector peak. The only randomness is in
`summation_orders` and `fragment_product`, drawn from the book's SplitMix64.

Swedish Embedded AB implements solutions for making dense linear algebra reach
the arithmetic and memory limits of the accelerators a model runs on. If your
team needs expertise in GEMM kernel design for machine learning inference and
training, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import struct
from fractions import Fraction

import bookrng
import compute_kernels as ck
import cpu
import gpu
import roofline as rf

# ------------------------------------------------------------------ parameters

F32 = 4                                  # bytes per fp32 number
F16 = 2                                  # bytes per fp16 number
N_BIG = 1024                             # the size of the chapter's running product
F_PK = rf.MACHINES["GPU"][0]             # GFLOP/s, Chapter 3 (the rounded 40,000)
BW = rf.MACHINES["GPU"][1]               # GB/s, Chapter 3
CPU_PK, CPU_BW = rf.MACHINES["CPU"]
SHARED_BYTES_PER_CYCLE = 128             # PARAMETER: 32 banks of 4 bytes, per compute unit per cycle
BASE_REGS = 32                           # PARAMETER: registers a kernel needs besides its operands
TENSOR_RATIO = 8                         # PARAMETER: matrix-unit peak over the vector peak
F_TC = F_PK * TENSOR_RATIO               # GFLOP/s, the matrix unit's peak (a parameter, not a product)
BANKS = 32                               # shared-memory banks of 4 bytes
MAX_REGS = 255                           # registers per invocation a kernel can name
WG = 256                                 # invocations of the register-blocked workgroup (16 x 16)
BK = 8                                   # k-chunk of the register-blocked kernel
PORTABLE_INVOCATIONS = 256               # Chapter 4: the portable workgroup limit
PORTABLE_SHARED = 16384                  # Chapter 4: the portable shared-memory limit (bytes)
NEED_RESIDENT = 500                      # Chapter 4: resident invocations per compute unit to hide latency
SHARED_BW_TOTAL = gpu.N_CU * SHARED_BYTES_PER_CYCLE * gpu.CLOCK_HZ / 1e9   # GB/s over the whole device
FRAG = (16, 16, 16)                      # the fragment shape (m_f, n_f, k_f) of the chapter

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


def _g(v, digits=2):
    return f"{v:,.{digits}f}"


def _ceil_div(a, b):
    return -(-a // b)


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f16(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]


def _seed_note(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (the seed of the name `{name}`)."


def _matrix_rows(m, label):
    return _table([label] + [f"col {j}" for j in range(len(m[0]))],
                  [(f"row {i}", *row) for i, row in enumerate(m)])


# ------------------------------------------------------ 1. the three-line loop


def _small_a(i, k):
    return (7 * i + 3 * k) % 11 - 5


def _small_b(k, j):
    return (5 * k + 2 * j) % 13 - 6


def naive_small():
    m = n = k = 4
    a = [[_small_a(i, p) for p in range(k)] for i in range(m)]
    b = [[_small_b(p, j) for j in range(n)] for p in range(k)]
    c = [[0] * n for _ in range(m)]
    loads = 0
    for i in range(m):
        for j in range(n):
            acc = 0
            for p in range(k):
                acc += a[i][p] * b[p][j]
                loads += 2
            c[i][j] = acc
    terms = " + ".join(f"({a[1][p]})({b[p][2]})" for p in range(k))
    flops = 2 * m * n * k
    rows = [("multiply-adds m n k", m * n * k), ("FLOPs 2 m n k", flops),
            ("loads, 2 k per output", loads), ("distinct numbers in A and B", m * k + k * n),
            ("loads per distinct number", loads // (m * k + k * n)), ("stores", m * n)]
    return (f"A[i, k] = (7 i + 3 k) mod 11 - 5 and B[k, j] = (5 k + 2 j) mod 13 - 6, both [4, 4], integers so that every "
            f"sum is exact. The loop is `for i, for j, acc = 0, for k: acc = acc + A[i, k] B[k, j]`, then `C[i, j] = acc`.\n\n"
            f"A:\n\n{_matrix_rows(a, 'A')}\n\nB:\n\n{_matrix_rows(b, 'B')}\n\nC = A B:\n\n{_matrix_rows(c, 'C')}\n\n"
            f"One output: C[1, 2] = {terms} = {c[1][2]}.\n\n" + _table(["count", "value"], rows))


def _naive_intensity(n):
    """FLOP per byte of the loop that reloads a row and a column for every output."""
    return Fraction(2 * n ** 3, F32 * (2 * n ** 3 + n * n))


def naive_counts():
    rows = []
    for n in (4, 64, 256, 1024, 4096):
        flops = 2 * n ** 3
        loads = 2 * n ** 3
        byts = F32 * (loads + n * n)
        low = 3 * F32 * n * n
        inten = flops / byts
        gpu_att = min(F_PK, inten * BW)
        rows.append((_n(n), _n(flops), _n(loads), _n(byts), f"{inten:.4f}", _n(low), f"{flops / low:.2f}",
                     f"{gpu_att:,.1f}", f"{gpu_att / F_PK:.4f}"))
    return ("C = A B, all sides n, fp32, one invocation per output with a K-loop: loads 2 n^3, stores n^2, bytes 4 (2 n^3 + n^2) "
            "if every load goes to main memory; FLOPs 2 n^3. The lower bound reads A and B once and writes C once: 12 n^2 bytes. "
            "GPU of Chapters 3 and 4: F_pk = 40,000 (GFLOP/s), BW = 800 (GB/s), attainable = min(F_pk, intensity x BW).\n\n"
            + _table(["n", "FLOPs", "loads", "bytes", "FLOP per byte", "bytes, lower bound", "FLOP per byte, bound",
                      "GPU attainable (GFLOP/s)", "fraction of peak"], rows))


# ------------------------------------------------- 2. loop order and the cache


def _misses(task):
    n, nest, ways, tile = task
    size = 32768
    cache = cpu.Cache(size=size, ways=ways if ways else size // cpu.LINE)
    e = F32
    a0, b0, c0 = 0, n * n * e, 2 * n * n * e
    acc = cache.access
    if nest == "ijk":
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    acc(a0 + (i * n + k) * e)
                    acc(b0 + (k * n + j) * e)
                acc(c0 + (i * n + j) * e)
    elif nest == "ikj":
        for i in range(n):
            for k in range(n):
                acc(a0 + (i * n + k) * e)
                for j in range(n):
                    acc(b0 + (k * n + j) * e)
                    acc(c0 + (i * n + j) * e)
    else:
        for ii in range(0, n, tile):
            for jj in range(0, n, tile):
                for kk in range(0, n, tile):
                    for i in range(ii, ii + tile):
                        for j in range(jj, jj + tile):
                            for k in range(kk, kk + tile):
                                acc(a0 + (i * n + k) * e)
                                acc(b0 + (k * n + j) * e)
                            acc(c0 + (i * n + j) * e)
    return cache.misses


LOOP_SIZES = (64, 96, 128)
LOOP_NESTS = (("ijk", "i, j, k (the dot-product order)"), ("ikj", "i, k, j (row of B at a time)"),
              ("blk", "ijk inside 16 x 16 x 16 blocks"))
BLOCK = 16


def loop_order_table():
    """{(n, nest, ways): line misses}, ways 8 or 0 for fully associative (one process; the result is cached by worked.py)."""
    return {(n, nest, w): _misses((n, nest, w, BLOCK)) for n in LOOP_SIZES for nest, _ in LOOP_NESTS for w in (8, 0)}


def loop_order_misses():
    res = loop_order_table()
    rows = []
    for n in LOOP_SIZES:
        flops = 2 * n ** 3
        compulsory = 3 * n * n * F32 // cpu.LINE
        for nest, desc in LOOP_NESTS:
            m8, mf = res[(n, nest, 8)], res[(n, nest, 0)]
            rows.append((n, desc, _n(compulsory), _n(m8), _n(mf), f"{flops / (cpu.LINE * m8):.3f}",
                         f"{flops / (cpu.LINE * mf):.3f}"))
    return ("C = A B, n by n fp32 row-major matrices at addresses 0, 4 n^2 and 8 n^2 (so each starts on a line boundary), the "
            "cache of Chapter 2: 32 KiB, 64-byte lines, least recently used, 8-way (64 sets) or fully associative (512 ways). "
            "Counted accesses: ijk reads A[i, k] and B[k, j] in the inner loop and touches C[i, j] once per (i, j); ikj reads A[i, k] once per "
            "(i, k) and then B[k, j] and C[i, j] in the inner loop; the blocked nest runs ijk inside blocks of 16 x 16 x 16 with the block "
            "loops in order i, j, k and touches C[i, j] once per block. 'Intensity' = 2 n^3 / (64 x line misses): FLOPs per byte of "
            "lines fetched. Compulsory = 3 n^2 x 4 / 64 lines, the least any order can fetch.\n\n"
            + _table(["n", "loop order", "compulsory lines", "line misses, 8-way", "line misses, full",
                      "FLOP per byte, 8-way", "FLOP per byte, full"], rows, "rlrrrrr"))


# ----------------------------------------- 3. register blocking on the CPU core

CPU_VECTOR_REGS = 16          # ymm registers of the vector core in Chapter 2
CPU_FMA_LATENCY = 5           # cycles
CPU_FMA_PORTS = 2             # fused multiply-adds started per cycle


def _cpu_block(rm, rn):
    """(registers, FMAs per k-step, vector loads per k-step) of an rm x rn block of vector accumulators:
    rm x rn accumulators, rm vectors of A, one temporary for the current vector of B."""
    return rm * rn + rm + 1, rm * rn, rm + rn


def cpu_register_block():
    rows = []
    best = None
    cands = []
    for rm in range(1, 9):
        for rn in range(1, 9):
            regs, fma, loads = _cpu_block(rm, rn)
            if regs <= CPU_VECTOR_REGS:
                cands.append((fma / loads, rm, rn, regs, fma, loads))
    best = max(cands)
    need = CPU_FMA_LATENCY * CPU_FMA_PORTS
    for ratio, rm, rn, regs, fma, loads in sorted(cands, key=lambda t: (-t[0], t[1])):
        if (rm, rn) in ((1, 1), (1, 4), (2, 2), (2, 6), (3, 4), (4, 3), (4, 2)) or (rm, rn) == best[1:3]:
            rows.append((f"{rm} x {rn}", regs, fma, loads, f"{ratio:.3f}", "yes" if fma >= need else "no"))
    return (f"A core with {CPU_VECTOR_REGS} vector registers, a fused multiply-add of latency {CPU_FMA_LATENCY} cycles started {CPU_FMA_PORTS} per cycle "
            f"(Chapter 2), so {need} independent accumulators keep both ports busy. A block of r_m x r_n vector accumulators needs r_m r_n "
            "accumulators, r_m vectors of A and one temporary for the vector of B just loaded: r_m r_n + r_m + 1 registers; one k-step does r_m r_n fused "
            f"multiply-adds and r_m + r_n vector loads. Blocks that fit in {CPU_VECTOR_REGS} registers (selected rows; the best fit by FMAs per load is "
            f"{best[1]} x {best[2]}):\n\n"
            + _table(["block r_m x r_n", "registers", "FMAs per k-step", "loads per k-step", "FMAs per load",
                      f"at least {need} accumulators"], rows))


# ------------------------------------------ 4. the generic blocked kernel, run


def _blocked_phases(q, r, bk, counters):
    """The register-blocked kernel (tiled when r = 1) as phases of Chapter 5's simulator.

    Workgroup of q x q invocations; workgroup tile T x T with T = q r; k-chunk bk. The A chunk is a_sh[row * bk + kk]
    (T rows of bk), the B chunk is b_sh[kk * T + col]. Each invocation owns the r x r block of rows ty r .. ty r + r - 1 and
    columns tx r .. tx r + r - 1 and keeps its accumulators in `regs` (its registers).
    """
    big = q * r
    n_wg = q * q
    regs = {}

    def load_phase(c):
        def phase(inv, p):
            m, n, k = p["M"], p["N"], p["K"]
            for e in range(_ceil_div(big * bk, n_wg)):
                idx = inv.l + e * n_wg
                if idx >= big * bk:
                    continue
                row, kk = divmod(idx, bk)                   # element (row, kk) of the A chunk
                gr, gk = inv.wy * big + row, c * bk + kk
                v = 0.0
                if gr < m and gk < k:
                    v = inv.load("A", gr * k + gk, site=("A", e))
                    counters["global"] += 1
                inv.shared_store(idx, v)
                kk2, col = divmod(idx, big)                 # element (kk2, col) of the B chunk
                gk2, gc = c * bk + kk2, inv.wx * big + col
                v = 0.0
                if gk2 < k and gc < n:
                    v = inv.load("B", gk2 * n + gc, site=("B", e))
                    counters["global"] += 1
                inv.shared_store(big * bk + idx, v)
        return phase

    def compute_phase(c, last):
        def phase(inv, p):
            m, n = p["M"], p["N"]
            ty, tx = divmod(inv.l, q)
            key = (inv.wx, inv.wy, inv.l)
            acc = regs.setdefault(key, [[0.0] * r for _ in range(r)])
            for kk in range(bk):
                a_reg = [inv.shared_load((ty * r + i) * bk + kk) for i in range(r)]
                b_reg = [inv.shared_load(big * bk + kk * big + tx * r + j) for j in range(r)]
                counters["shared"] += 2 * r
                for i in range(r):
                    for j in range(r):
                        acc[i][j] += a_reg[i] * b_reg[j]
                        counters["fma"] += 1
            if last:
                for i in range(r):
                    for j in range(r):
                        gr, gc = inv.wy * big + ty * r + i, inv.wx * big + tx * r + j
                        if gr < m and gc < n:
                            inv.store("C", gr * n + gc, acc[i][j], site=("C", i, j))
        return phase

    def build(k_total):
        steps = _ceil_div(k_total, bk)
        phases = []
        for c in range(steps):
            phases.append(load_phase(c))
            phases.append(compute_phase(c, c == steps - 1))
        return phases

    return build


def _naive_phase(counters):
    def phase(inv, p):
        m, n, k = p["M"], p["N"], p["K"]
        g = inv.gid_x
        if g >= m * n:
            inv.ret()
        i, j = divmod(g, n)
        acc = 0.0
        for kk in range(k):
            acc += inv.load("A", i * k + kk) * inv.load("B", kk * n + j)
            counters["global"] += 2
            counters["fma"] += 1
        inv.store("C", g, acc)
    return [phase]


def _run_gemm(m, n, k, kind, q=1, r=1, bk=1, wg_naive=16):
    """Run a kernel on the simulator over integer matrices; returns (counters, load blocks, store blocks, equals direct)."""
    a = [float(_small_a(i, p)) for i in range(m) for p in range(k)]
    b = [float(_small_b(p, j)) for p in range(k) for j in range(n)]
    mem = ck.Memory()
    ba = mem.alloc("A", a, "read")
    bb = mem.alloc("B", b, "read")
    bc = mem.alloc("C", [ck.UNWRITTEN] * (m * n))
    counters = {"global": 0, "shared": 0, "fma": 0}
    params = {"M": m, "N": n, "K": k}
    if kind == "naive":
        stats = ck.launch(mem, _naive_phase(counters), _ceil_div(m * n, wg_naive), wg_naive, params,
                          {"A": ba, "B": bb, "C": bc})
    else:
        big = q * r
        phases = _blocked_phases(q, r, bk, counters)(k)
        stats = ck.launch(mem, phases, (_ceil_div(n, big), _ceil_div(m, big)), q * q, params,
                          {"A": ba, "B": bb, "C": bc}, shared_words=2 * big * bk)
    want = [float(sum(_small_a(i, p) * _small_b(p, j) for p in range(k))) for i in range(m) for j in range(n)]
    return counters, stats, mem.values("C") == want


def tiled_loads(m, n, k, big):
    """Global loads of a workgroup tile of side `big`: every A element once per column tile, every B element once per row tile."""
    return k * (m * _ceil_div(n, big) + n * _ceil_div(m, big))


def tiled_trace():
    rows = []
    cases = [("naive, 16 invocations per workgroup", 16, 16, 16, "naive", 0, 0, 0),
             ("tiled, T = 4 (16 invocations)", 16, 16, 16, "tiled", 4, 1, 4),
             ("tiled, T = 8 (64 invocations)", 16, 16, 16, "tiled", 8, 1, 8),
             ("naive, 16 invocations per workgroup", 6, 10, 7, "naive", 0, 0, 0),
             ("tiled, T = 4 (16 invocations)", 6, 10, 7, "tiled", 4, 1, 4)]
    for name, m, n, k, kind, q, r, bk in cases:
        c, st, ok = _run_gemm(m, n, k, kind, q, r, bk)
        formula = 2 * m * n * k if kind == "naive" else tiled_loads(m, n, k, q * r)
        rows.append((f"[{m}, {k}] by [{k}, {n}]", name, _n(c["global"]), _n(formula), _n(st["load"]),
                     "yes" if ok else "NO"))
    return ("Integer matrices A[i, k] = (7 i + 3 k) mod 11 - 5 and B[k, j] = (5 k + 2 j) mod 13 - 6 (as in `matmul_fast.naive_small`), run on "
            "the kernel simulator of Chapter 5 (32-lane subgroups, 32-byte blocks). Loads are counted by the kernel as it executes them; "
            "an invocation that falls outside the matrices loads nothing. Formula: naive 2 m n k; tiled k (m ceil(n / T) + n ceil(m / T)). "
            "'Load blocks' are the simulator's 32-byte transactions for loads, counted as follows. A memory instruction is one "
            "load statement of the kernel at one step of its loops: for the naive kernel the load of A and the load of B at one k; "
            "for the tiled kernel the load of A and the load of B at one step e of the cooperative loop in one k-chunk of one "
            "workgroup. The invocations of a subgroup (32 consecutive invocations of a workgroup; a workgroup of 16 is one "
            "partial subgroup) that reach an instruction issue it together, and one that has returned or whose element lies "
            "outside the matrix does not issue it, so a skipped load never moves the later loads of that invocation into "
            "another instruction. An instruction costs one transaction per distinct 32-byte block (8 consecutive words, "
            "aligned to a multiple of 8 words) that its addresses touch. A, B and C sit in one flat memory of 4-byte words, in "
            "that order, each buffer starting at the first multiple of 8 words at or after the end of the previous one: for "
            "[6, 7] by [7, 10], A is words 0 to 41, B words 48 to 117 and C words 120 to 179 (for [16, 16] by [16, 16] "
            "all three are multiples of 8 anyway).\n\n"
            + _table(["product", "kernel", "global loads, counted", "global loads, formula", "load blocks", "equals direct"],
                     rows, "llrrrl"))


def blocked_trace():
    rows = []
    n = 8
    m = k = n
    for name, kind, q, r, bk in (("naive (one output per invocation)", "naive", 0, 0, 0),
                                 ("tiled: q = 4, r = 1", "tiled", 4, 1, 2),
                                 ("register-blocked: q = 2, r = 2", "blocked", 2, 2, 2),
                                 ("register-blocked: q = 1, r = 4", "blocked", 1, 4, 2)):
        c, st, ok = _run_gemm(m, n, k, kind, q, r, bk)
        big = q * r
        rows.append((name, big if big else "-", q * q if q else 16, r * r if r else 1, _n(c["global"]), _n(c["shared"]),
                     _n(c["fma"]), "yes" if ok else "NO"))
    return (f"The product of two [{n}, {n}] integer matrices (as in `matmul_fast.naive_small`) on the simulator, workgroup tile T = q r = 4 for the "
            "last three rows, k-chunk 2. Every count is taken from the running kernel: global loads of A and B, shared-memory reads, fused multiply-adds "
            f"(each is 1 of the {n}^3 = {_n(n ** 3)} that the product needs). The global loads depend on T only; the shared reads fall as 1 / r.\n\n"
            + _table(["kernel", "workgroup tile T", "invocations per workgroup", "outputs per invocation r^2",
                      "global loads", "shared reads", "FMAs", "equals direct"], rows, "lrrrrrrl"))


# ---------------------------------------------------- 5. choosing the tile size


def _loads(n, big):
    return 2 * n ** 3 // big


def _bytes_f32(n, big):
    return F32 * (_loads(n, big) + n * n)


def _intensity_f32(n, big):
    return Fraction(2 * n ** 3, _bytes_f32(n, big))


def tile_size_table():
    n = N_BIG
    flops = 2 * n ** 3
    rows = []
    for t in (2, 4, 8, 16, 32):
        inv = t * t
        shared = 2 * t * t * F32
        res, occ, why = gpu.occupancy(inv, 40, shared) if inv <= gpu.MAX_THREADS_PER_CU else (0, 0.0, "-")
        byts = _bytes_f32(n, t)
        inten = flops / byts
        att = min(F_PK, inten * BW)
        ok = inv <= PORTABLE_INVOCATIONS and shared <= PORTABLE_SHARED
        rows.append((t, _n(inv), _n(shared), "yes" if ok else "no", res, f"{occ * 100:.1f}", why, _n(_loads(n, t)),
                     f"{inten:.2f}", f"{att:,.0f}", f"{att / F_PK:.3f}"))
    return (f"C = A B, n = {n}, fp32, workgroup of T x T invocations computing a T x T tile with T x T tiles of A and B in shared memory "
            "(2 T^2 numbers, 8 T^2 bytes); global loads 2 n^3 / T; bytes 4 (loads + n^2); 40 registers per invocation as in Chapter 4's "
            "table; occupancy by the model of Chapter 4 (65,536 registers and 65,536 bytes of shared memory per compute unit, 2,048 resident invocations, 16 resident workgroups). "
            "Portable: at most 256 invocations and 16 KiB of shared memory. Attainable = min(40,000, intensity x 800), in (GFLOP/s).\n\n"
            + _table(["T", "invocations", "shared bytes", "portable", "resident workgroups", "occupancy (%)", "limited by",
                      "global loads", "FLOP per byte", "attainable (GFLOP/s)", "fraction of peak"], rows, "rrrlrrlrrrr"))


def loads_curve(n=N_BIG, sides=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)):
    """(side of the workgroup tile, global loads) for the figure and its table."""
    return [(b, 2 * n ** 3 // b) for b in sides]


# --------------------------------------------------------- 6. shared-memory banks


def bank_degree(addresses):
    """Cycles one subgroup's shared-memory access takes: the most distinct words that fall in one bank (the same word is a broadcast)."""
    per_bank = {}
    for a in addresses:
        per_bank.setdefault(a % BANKS, set()).add(a)
    return max(len(v) for v in per_bank.values())


def bank_strides():
    rows = []
    for stride in (1, 2, 4, 8, 16, 32, 33):
        addrs = [stride * t for t in range(32)]
        rows.append((stride, len({a % BANKS for a in addrs}), bank_degree(addrs)))
    return rows


def bank_conflicts():
    out = ["Shared memory of the model: 32 banks of 4-byte words; word a lives in bank a mod 32; the 32 lanes of a subgroup issue one access and it "
           "takes as many cycles as the most distinct words in one bank (lanes reading the same word are served together). 'Degree' is that count.\n",
           "Lane t reads word `stride x t` (a column of a row-major tile):\n",
           _table(["stride (words)", "banks used by 32 lanes", "degree"], bank_strides()), ""]
    # the transpose of Chapter 5: 32 lanes of a 16 x 16 workgroup read tile[lx * T + ly] with T words per row, or T + 1 padded
    t = 16
    rows = []
    for stride in (t, t + 1):
        addrs = [(l % t) * stride + (l // t) for l in range(32)]
        rows.append((f"tile[lx x {stride} + ly], lanes l = 0 .. 31", bank_degree(addrs)))
    # the register-tiled kernels of brain: the B-tile read and the staging store
    reads = [("B-tile read, thread tx owns 8 adjacent columns: Bs[kk x 128 + 8 tx + j], tx = 0 .. 15, j fixed",
              [8 * tx + 3 for tx in range(16)]),
             ("B-tile read, interleaved columns: Bs[kk x 129 + tx + 16 j], tx = 0 .. 15, j fixed",
              [129 * 5 + tx + 16 * 3 for tx in range(16)])]
    for name, addrs in reads:
        rows.append((name, bank_degree(addrs)))
    for stride, name in ((128, "no padding"), (129, "padded by one word"), (132, "padded by one quad of four words")):
        addrs = [(idx % 8) * stride + idx // 8 for idx in range(32)]       # As[kk x stride + r], r = idx div 8, kk = idx mod 8
        rows.append((f"staging store As[kk x {stride} + r], lanes idx = 0 .. 31 with r = idx div 8, kk = idx mod 8 ({name})",
                     bank_degree(addrs)))
    out.append("The accesses of the tiles in this chapter (32 lanes unless the line says 16):\n")
    out.append(_table(["access", "degree"], rows, "lr"))
    return "\n".join(out)


# ---------------------------------------------------------- 7. register blocking


def _regs(r):
    return BASE_REGS + r * r + 2 * r


def register_block_rows(n=N_BIG):
    flops = 2 * n ** 3
    rows = []
    for r in (1, 2, 4, 8, 16):
        big = 16 * r
        regs = _regs(r)
        shared = 2 * big * BK * F32
        if regs > MAX_REGS:
            rows.append(dict(r=r, big=big, regs=regs, shared=shared, ok=False))
            continue
        res, occ, why = gpu.occupancy(WG, regs, shared)
        byts = _bytes_f32(n, big)
        inten = flops / byts
        global_roof = inten * BW
        shared_roof = SHARED_BW_TOTAL / (4.0 / r)       # GFLOP/s the shared memory can feed: bandwidth / (bytes per FLOP)
        att = min(F_PK, global_roof, shared_roof)
        limit = "arithmetic peak" if att == F_PK else ("global memory" if att == global_roof else "shared memory")
        rows.append(dict(r=r, big=big, regs=regs, shared=shared, ok=True, resident=res * WG, occ=occ, why=why,
                         loads=_loads(n, big), inten=inten, shared_per_fma=2 / r, global_roof=global_roof,
                         shared_roof=shared_roof, att=att, limit=limit, hides=res * WG >= NEED_RESIDENT))
    return rows


def register_blocking():
    rows = []
    for d in register_block_rows():
        if not d["ok"]:
            rows.append((d["r"], d["big"], d["regs"], _n(d["shared"]), "does not fit", "-", "-", "-", "-", "-", "-", "-", "-"))
            continue
        rows.append((d["r"], d["big"], d["regs"], _n(d["shared"]), f"{d['resident']:,}", f"{d['occ'] * 100:.1f}",
                     "yes" if d["hides"] else "no", _n(d["loads"]), f"{d['inten']:.2f}", f"{d['shared_per_fma']:.3f}",
                     f"{d['global_roof']:,.0f}", f"{d['shared_roof']:,.0f}", f"{d['att']:,.0f} ({d['limit']})"))
    return (f"C = A B, n = {N_BIG}, fp32, workgroup of {WG} invocations (16 x 16), each computing an r x r block, so the workgroup tile is T = 16 r on a side; "
            f"k-chunk {BK}, shared memory (T + T) x {BK} x 4 bytes. PARAMETERS of this chapter: registers per invocation = {BASE_REGS} + r^2 + 2 r "
            f"({BASE_REGS} for indices and loop counters, r^2 accumulators, 2 r operands); shared memory delivers {SHARED_BYTES_PER_CYCLE} bytes per cycle per compute unit "
            f"({_g(SHARED_BW_TOTAL, 0)} (GB/s) over the {gpu.N_CU} compute units at 2 GHz). Shared roof = shared bandwidth / (4 / r bytes per FLOP); "
            f"global roof = intensity x 800; both in (GFLOP/s); the arithmetic peak is {_n(int(F_PK))}. Occupancy by the model of Chapter 4; "
            f"'enough' = at least {NEED_RESIDENT} resident invocations per compute unit.\n\n"
            + _table(["r", "tile T", "registers per invocation", "shared bytes", "resident invocations", "occupancy (percent)", "enough",
                      "global loads", "FLOP per byte", "shared reads per FMA", "global roof", "shared roof", "attainable (limit)"],
                     rows))


def tile_fill():
    rows = []
    n = 2560
    for m in (1, 8, 32, 77, 128, 129, 512):
        wgs = _ceil_div(m, 128) * _ceil_div(n, 128)
        used = m * n / (_ceil_div(m, 128) * 128 * _ceil_div(n, 128) * 128)
        mf = _ceil_div(m, FRAG[0]) * FRAG[0]
        rows.append((m, wgs, f"{used * 100:.1f}", mf, f"{m / mf * 100:.1f}"))
    return ("Output [m, 2560] from a 128 x 128 workgroup tile (n = 2560 is exactly 20 tiles wide) and from 16 x 16 fragments (rows padded to a multiple "
            "of 16; 2560 needs no padding). Share of the computed outputs that are real.\n\n"
            + _table(["m", "128 x 128 workgroups", "real share of tile outputs (%)", "m padded to 16", "real share of fragment rows (%)"], rows))


# ---------------------------------------------------------- 8. the roofline climb


def climb_rows(n=N_BIG):
    """The kernels of the chapter as points on the roofline of the model GPU: (name, intensity, attainable, peak, kind)."""
    flops = 2 * n ** 3
    pts = []
    inten = Fraction(2 * n ** 3, F32 * (2 * n ** 3 + n * n))
    pts.append(("naive", float(inten), min(F_PK, float(inten) * BW), F_PK))
    for name, big in (("tiled, T = 16", 16), ("tiled, T = 32", 32), ("register-blocked, T = 64", 64),
                      ("register-blocked, T = 128", 128)):
        i = flops / _bytes_f32(n, big)
        pts.append((name, i, min(F_PK, i * BW), F_PK))
    # matrix unit: fp16 operands, fp32 output, tile B = 128 and 256
    for big in (128, 256):
        byts = F16 * _loads(n, big) + F32 * n * n
        i = flops / byts
        pts.append((f"matrix unit, fp16, T = {big}", i, min(F_TC, i * BW), F_TC))
    return pts


def roofline_climb():
    rows = []
    for name, inten, att, peak in climb_rows():
        rows.append((name, f"{inten:.2f}", f"{att:,.0f}", f"{peak:,.0f}", f"{att / peak:.4f}",
                     "compute" if inten >= peak / BW else "bandwidth"))
    return (f"C = A B, n = {N_BIG}, on the model GPU: arithmetic peak {_n(int(F_PK))} (GFLOP/s) (Chapter 3), bandwidth {int(BW)} (GB/s), ridge {F_PK / BW:.0f} FLOP per byte; "
            f"the matrix unit's peak is {TENSOR_RATIO} times the vector peak, {_n(int(F_TC))} (GFLOP/s), a PARAMETER of this chapter, with ridge {F_TC / BW:.0f}. "
            "Bytes: fp32 operands 4 (2 n^3 / T + n^2); matrix-unit rows read fp16 operands (2 bytes) and write the fp32 output. Attainable = min(peak, intensity x 800) "
            "with global traffic only, the model's roofline of Chapter 3; shared memory is not the limit in any row.\n\n"
            + _table(["kernel", "FLOP per byte", "attainable (GFLOP/s)", "peak of its units (GFLOP/s)", "fraction of that peak",
                      "side of its ridge"], rows, "lrrrrl"))


def tile_for_ridge():
    rows = []
    for name, peak, operand in (("vector units, fp32 operands", F_PK, F32), ("matrix unit, fp16 operands", F_TC, F16)):
        ridge = peak / BW
        # large n: bytes per FLOP = operand bytes x 2 n^3 / B / (2 n^3) = operand / B, so intensity = B / operand
        need = ridge * operand
        pow2 = 1 << math.ceil(math.log2(need))
        rows.append((name, f"{ridge:.0f}", operand, f"{need:.0f}", pow2))
    return ("For large n the global traffic is 2 n^3 / T operand reads, so intensity -> T / (operand bytes) FLOP per byte (the writes of C vanish). To reach a "
            "ridge the workgroup tile must have side T = ridge x operand bytes.\n\n"
            + _table(["units", "ridge (FLOP per byte)", "operand bytes", "tile side T to reach it", "next power of two"], rows))


# ---------------------------------------------------------- 9. matrix fragments


def _round_f32_exact(x):
    """The float32 nearest to the exact rational x (ties to even), as a float. Normal range only."""
    x = Fraction(x)
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a < Fraction(2) ** e:
        e -= 1
    elif a >= Fraction(2) ** (e + 1):
        e += 1
    assert -126 <= e <= 127
    q = Fraction(2) ** (e - 23)
    scaled = a / q
    fl = scaled.numerator // scaled.denominator
    rem = scaled - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    return sign * float(Fraction(fl) * q)


def _ulp(x):
    """Spacing of float32 numbers at |x| (normal range)."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 23)


def fragment_mad(a, b, c):
    """D = C + A B for a fragment: products exact, the sum of the k products and the accumulator rounded once to float32.
    This is the book's model; real units differ in how they align and truncate the sum, and vendors differ."""
    rows, kk, cols = len(a), len(b), len(b[0])
    out = []
    for i in range(rows):
        out.append([_round_f32_exact(Fraction(c[i][j]) + sum(Fraction(a[i][p]) * Fraction(b[p][j]) for p in range(kk)))
                    for j in range(cols)])
    return out


def fragment_gemm(a, b, m, n, k, frag=FRAG):
    """C = A B by fragments (pseudocode `matmul_coop` of the chapter); one workgroup of one subgroup per output fragment.
    Returns (C, counts) with counts of fragment loads, mads, stores and numbers loaded."""
    fm, fn, fk = frag
    require = m % fm == 0 and n % fn == 0 and k % fk == 0
    assert require
    c = [[0.0] * n for _ in range(m)]
    counts = {"loads": 0, "mads": 0, "stores": 0, "numbers": 0}
    for wy in range(m // fm):
        for wx in range(n // fn):
            acc = [[0.0] * fn for _ in range(fm)]
            for k0 in range(0, k, fk):
                af = [[a[wy * fm + i][k0 + p] for p in range(fk)] for i in range(fm)]
                bf = [[b[k0 + p][wx * fn + j] for j in range(fn)] for p in range(fk)]
                counts["loads"] += 2
                counts["numbers"] += fm * fk + fk * fn
                acc = fragment_mad(af, bf, acc)
                counts["mads"] += 1
            for i in range(fm):
                for j in range(fn):
                    c[wy * fm + i][wx * fn + j] = acc[i][j]
            counts["stores"] += 1
    return c, counts


FRAG_NAME = "matmul_fast.fragment_product"
FRAG_SIDE = 32


def fragment_product():
    g = bookrng.generator(FRAG_NAME)
    side = FRAG_SIDE
    a32 = [[_f32(2.0 * g.uniform() - 1.0) for _ in range(side)] for _ in range(side)]
    b32 = [[_f32(2.0 * g.uniform() - 1.0) for _ in range(side)] for _ in range(side)]
    a16 = [[_f16(x) for x in row] for row in a32]
    b16 = [[_f16(x) for x in row] for row in b32]
    d, cnt = fragment_gemm(a16, b16, side, side, side)

    def exact(a, b, i, j):
        return sum(Fraction(a[i][p]) * Fraction(b[p][j]) for p in range(side))

    err_acc, err_in, err_f32 = [], [], []
    for i in range(side):
        for j in range(side):
            ref16 = exact(a16, b16, i, j)
            ref32 = exact(a32, b32, i, j)
            err_acc.append(abs(Fraction(d[i][j]) - ref16))
            err_in.append(abs(Fraction(d[i][j]) - ref32))
            seq = 0.0
            for p in range(side):
                seq = _f32(seq + _f32(a32[i][p] * b32[p][j]))
            err_f32.append(abs(Fraction(seq) - ref32))
    fm, fn, fk = FRAG
    flops_frag = 2 * fm * fn * fk
    flops = 2 * side ** 3
    sg = gpu.SUBGROUP
    rows = [("fragment shape m_f x n_f x k_f", f"{fm} x {fn} x {fk}"),
            ("FLOPs of one fragment multiply-add 2 m_f n_f k_f", _n(flops_frag)),
            (f"fragment multiply-adds for [{side}, {side}] by [{side}, {side}]", cnt["mads"]),
            ("FLOPs of the product 2 n^3 = fragment multiply-adds x FLOPs each", _n(flops)),
            (f"subgroup instructions of vector fused multiply-adds ({sg} lanes) for the same product", _n(side ** 3 // sg)),
            ("accumulator registers per lane, m_f n_f / 32", fm * fn // sg),
            (f"numbers loaded from A and B = fragment multiply-adds x (m_f k_f + k_f n_f) = {cnt['mads']} x ({fm * fk} + {fk * fn})",
             _n(cnt["numbers"])),
            ("largest |D - exact product of the fp16 inputs| (accumulation only)", f"{float(max(err_acc)):.3e}"),
            ("largest |D - exact product of the original fp32 inputs| (inputs rounded to fp16)", f"{float(max(err_in)):.3e}"),
            ("largest |sequential fp32 loop - exact product of the fp32 inputs|", f"{float(max(err_f32)):.3e}")]
    return (f"A and B are [{side}, {side}] with entries `2 uniform() - 1` rounded to float32 and then to fp16 (round to nearest, ties to even) for the fragment path. "
            f"{_seed_note(FRAG_NAME)} Draw order: the {side * side} entries of A row by row, then the {side * side} of B row by row. A fragment multiply-add forms "
            "the products exactly and rounds the sum with the accumulator once to float32 (the book's model of a fragment unit). The product is cut into "
            f"{fm} x {fn} output fragments and {side // fk} steps along k.\n\n" + _table(["quantity", "value"], rows, "lr"))


# ------------------------------------------------------ 10. the order of the sum


SUM_NAME = "matmul_fast.summation_orders"
SUM_SIDE, SUM_K = 4, 1024


def _dot_variants(av, bv):
    """The same dot product of fp32 vectors summed in seven ways; every product and every sum is rounded to float32 unless said."""
    k = len(av)
    prods = [_f32(av[p] * bv[p]) for p in range(k)]
    out = {}
    acc = 0.0
    for p in range(k):
        acc = _f32(acc + prods[p])
    out["sequential"] = acc
    # tiled: a carried accumulator across 16-wide tiles, tile loops outermost, ascending k inside each
    acc = 0.0
    for t0 in range(0, k, 16):
        for p in range(t0, t0 + 16):
            acc = _f32(acc + prods[p])
    out["tiled, accumulator carried"] = acc
    acc = 0.0
    for p in range(k):
        acc = _round_f32_exact(Fraction(acc) + Fraction(av[p]) * Fraction(bv[p]))
    out["fused multiply-add"] = acc
    parts = []
    for s0 in range(0, k, 256):
        part = 0.0
        for p in range(s0, s0 + 256):
            part = _f32(part + prods[p])
        parts.append(part)
    acc = 0.0
    for part in parts:
        acc = _f32(acc + part)
    out["split-k, 4 slices"] = acc
    acc = 0.0
    for t0 in range(0, k, 16):
        part = 0.0
        for p in range(t0, t0 + 16):
            part = _f32(part + prods[p])
        acc = _f32(acc + part)
    out["tile sums added to the total"] = acc
    lanes = [0.0] * 8
    for p in range(k):
        lanes[p % 8] = _f32(lanes[p % 8] + prods[p])
    s = [_f32(lanes[j] + lanes[j + 4]) for j in range(4)]
    out["8 vector lanes"] = _f32(_f32(s[0] + s[1]) + _f32(s[2] + s[3]))
    acc = 0.0
    a16 = [_f16(x) for x in av]
    b16 = [_f16(x) for x in bv]
    for t0 in range(0, k, 16):
        acc = _round_f32_exact(Fraction(acc) + sum(Fraction(a16[p]) * Fraction(b16[p]) for p in range(t0, t0 + 16)))
    out["fragments, fp16 inputs"] = acc
    return out


VARIANTS = ("sequential", "tiled, accumulator carried", "fused multiply-add", "split-k, 4 slices",
            "tile sums added to the total", "8 vector lanes", "fragments, fp16 inputs")


def _bits(x):
    return f"{struct.unpack('<I', struct.pack('<f', x))[0]:08x}"


def summation_orders():
    g = bookrng.generator(SUM_NAME)
    side, k = SUM_SIDE, SUM_K
    a = [[_f32(2.0 * g.uniform() - 1.0) for _ in range(k)] for _ in range(side)]
    b = [[_f32(2.0 * g.uniform() - 1.0) for _ in range(side)] for _ in range(k)]
    results = {v: [] for v in VARIANTS}
    exact, scale = [], []
    for i in range(side):
        for j in range(side):
            col = [b[p][j] for p in range(k)]
            res = _dot_variants(a[i], col)
            for v in VARIANTS:
                results[v].append(res[v])
            exact.append(sum(Fraction(a[i][p]) * Fraction(col[p]) for p in range(k)))
            scale.append(sum(abs(Fraction(a[i][p]) * Fraction(col[p])) for p in range(k)))
    u = Fraction(1, 2 ** 24)
    base = results["sequential"]
    rows = []
    for v in VARIANTS:
        differ = sum(1 for x, y in zip(results[v], base) if x != y)
        err = [float(abs(Fraction(x) - e) / (u * sc)) for x, e, sc in zip(results[v], exact, scale)]
        rows.append((v, f"{differ} of {side * side}", f"{max(err):.3f}", f"{sum(err) / len(err):.3f}"))
    one = []
    for v in VARIANTS:
        x = results[v][0]
        one.append((v, f"{x:.9g}", _bits(x), f"{(x - base[0]) / _ulp(base[0]):+.0f}"))
    return (f"C = A B with A [{side}, {k}] and B [{k}, {side}], entries `2 uniform() - 1` rounded to float32. {_seed_note(SUM_NAME)} Draw order: the {side * k} entries "
            f"of A row by row, then the {k * side} entries of B row by row. Every product and every sum is a correctly rounded float32 operation (no fused "
            "multiply-add) unless the row says otherwise. With p = 0 .. 1023 the index of the sum and fl(.) a rounding to float32, "
            "the seven orders are: sequential, acc = fl(acc + fl(a_p b_p)) for p ascending from acc = 0; tiled with the "
            "accumulator carried, the same additions in the same order written as 64 tiles of 16 consecutive p (p ascending inside a tile, "
            "one accumulator for the whole sum); fused multiply-add, acc = fl(acc + a_p b_p) with the product exact (one rounding "
            "per term); split-k with 4 slices, the four slices of 256 consecutive p are each summed sequentially from 0 "
            "(rounded products) and the four slice sums are then added sequentially, slice 0 first, from acc = 0; tile sums "
            "added to the total, each of the 64 tiles of 16 consecutive p is summed sequentially from 0 and the tile sums are "
            "then added sequentially to acc, tile 0 first; 8 vector lanes, lane j = fl(lane j + fl(a_p b_p)) for the p with p mod 8 = j "
            "in ascending order, then s_j = fl(lane j + lane j + 4) for j = 0 .. 3 and the result fl(fl(s_0 + s_1) + fl(s_2 + s_3)); "
            "fragments with fp16 inputs, a and b first rounded to fp16, then for each tile of 16 consecutive p the 16 products "
            "are formed exactly, added exactly to acc, and the sum is rounded once to float32. 'Differ' counts the outputs whose float32 value differs from the "
            "sequential order. Error is |result - exact sum of the float32 inputs|, with the exact sum computed in rational arithmetic, divided by u S, where "
            "u = 2^-24 and S = the sum of |A[i, p] B[p, j]| (Volume I's bound for a sum of K terms is (K - 1) u S = 1,023 u S).\n\n"
            + _table(["order of the sum", "differ from sequential", "largest error / (u S)", "mean error / (u S)"], rows)
            + "\n\nThe output C[0, 0] under each order: its value to 9 significant digits, its float32 bit pattern in hexadecimal, and its distance from the "
              "sequential result in units of the float32 spacing there (ulp).\n\n"
            + _table(["order of the sum", "C[0, 0]", "bits", "ulp from sequential"], one, "lrrr"))


def sequential_tiled_identical():
    """True when the carried-accumulator tiled order reproduces the sequential order bit for bit on the seeded example (used by the text)."""
    g = bookrng.generator(SUM_NAME)
    side, k = SUM_SIDE, SUM_K
    a = [[_f32(2.0 * g.uniform() - 1.0) for _ in range(k)] for _ in range(side)]
    b = [[_f32(2.0 * g.uniform() - 1.0) for _ in range(side)] for _ in range(k)]
    for i in range(side):
        for j in range(side):
            res = _dot_variants(a[i], [b[p][j] for p in range(k)])
            if res["sequential"] != res["tiled, accumulator carried"]:
                return False
    return True
