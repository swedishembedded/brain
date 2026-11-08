# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "The GPU: Thousands of Small Workers".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: gpu.<name> -->` markers. Helpers with a leading underscore, or
that take arguments, are shared with the chapter's figures so a picture and
the table beside it cannot disagree.

Nothing here measures a machine. The GPU is a set of round PARAMETERS invented
for the book (compute units, lanes, a register file, shared memory, a memory
latency in cycles, a transaction size), consistent with the illustrative GPU of
Chapter 3 up to rounding. Everything else is counting: work items, rounds of
lanes, bytes in flight, issue slots of a subgroup that disagrees with itself,
resident workgroups, memory transactions of 32 bytes, bytes over a bus. Where a
time appears it is the model's output from the stated parameters, never an
observation. The only randomness is one stream of uniform numbers from the
book's own generator (`bookrng.generator("gpu.divergence")`).

Swedish Embedded AB implements solutions for making machine learning kernels
fill a GPU and stream its memory. If your team needs expertise in GPU
performance analysis for machine learning inference and training, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng

# ------------------------------------------------------------------ parameters

N_CU = 80                    # compute units
LANES_PER_CU = 128           # lanes (fp32 units) per compute unit
CLOCK_HZ = 2.0e9             # cycles per second
LANES = N_CU * LANES_PER_CU  # 10,240
SUBGROUP = 32                # invocations that execute one instruction together
REGS_PER_CU = 65_536         # 32-bit registers in one compute unit's register file
SHARED_PER_CU = 65_536       # bytes of shared memory in one compute unit
MAX_THREADS_PER_CU = 2_048   # invocations resident on one compute unit at most
MAX_WG_PER_CU = 16           # workgroups resident on one compute unit at most
MEM_LATENCY_CYCLES = 400     # cycles from a load's issue to its data
SECTOR = 32                  # bytes per memory transaction
F32 = 4                      # bytes per fp32 number
BW = 800e9                   # bytes/s, device memory (Chapter 3)
BUS = 32e9                   # bytes/s, host to device bus (parameter)
F_PK_CH3 = 40_000e9          # FLOP/s, the rounded peak of Chapter 3
CPU_LANES_PER_CORE = 16      # 2 vector units of 8 lanes (Chapter 3)
CPU_CORES = 10
CPU_CLOCK_HZ = 2.5e9
CPU_BW = 80e9

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


def _ms(seconds):
    return f"{seconds * 1e3:,.4f}"


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


# ------------------------------------------------------------------- machine


def machine():
    rows = [
        ("compute units", _n(N_CU), "80"),
        ("lanes per compute unit", _n(LANES_PER_CU), "-"),
        ("lanes in all", _n(LANES), "about 10,000"),
        ("clock (GHz)", f"{CLOCK_HZ / 1e9:.1f}", "2"),
        ("peak fp32 (GFLOP/s) = lanes x 2 x clock", _n(round(LANES * 2 * CLOCK_HZ / 1e9)), "40,000"),
        ("memory bandwidth BW (GB/s)", _n(round(BW / 1e9)), "800"),
        ("subgroup size (invocations)", str(SUBGROUP), "-"),
        ("registers per compute unit (32-bit)", _n(REGS_PER_CU), "-"),
        ("shared memory per compute unit (bytes)", _n(SHARED_PER_CU), "-"),
        ("resident invocations per compute unit, at most", _n(MAX_THREADS_PER_CU), "-"),
        ("resident workgroups per compute unit, at most", str(MAX_WG_PER_CU), "-"),
        ("memory latency (cycles)", str(MEM_LATENCY_CYCLES), "-"),
        ("memory transaction (bytes)", str(SECTOR), "-"),
        ("host-device bus (GB/s)", _n(round(BUS / 1e9)), "-"),
    ]
    return ("Model parameters, round numbers invented for the book (not any product). The last column is the value the "
            "roofline chapter rounded to; the two differ by the rounding of 10,240 lanes to 10,000.\n\n"
            + _table(["quantity", "this chapter", "Chapter 3"], rows))


# ------------------------------------------------------------ the vector add


def vector_add():
    n = 1 << 24
    byts = 3 * F32 * n
    rows = []
    for name, lanes, clock, bw in (
        ("one CPU core", CPU_LANES_PER_CORE, CPU_CLOCK_HZ, CPU_BW),
        ("the 10-core CPU", CPU_CORES * CPU_LANES_PER_CORE, CPU_CLOCK_HZ, CPU_BW),
        ("the GPU", LANES, CLOCK_HZ, BW),
    ):
        rounds = -(-n // lanes)
        rows.append((name, _n(lanes), _n(rounds), _ms(rounds / clock), _ms(byts / bw)))
    return (f"c = a + b with n = 2^24 = {_n(n)} fp32 numbers. Work: {_n(n)} additions and {_n(byts)} bytes "
            f"(12 per element, {byts // 2 ** 20} MiB). A round is every lane doing one addition; rounds = ceil(n / lanes). "
            "Arithmetic time = rounds / clock; memory time = bytes / bandwidth with the whole machine's bandwidth "
            "(80 and 800, in GB/s). Both are model outputs in milliseconds.\n\n"
            + _table(["machine", "lanes", "rounds", "arithmetic (ms)", "memory (ms)"], rows))


def latency_hiding():
    latency_s = MEM_LATENCY_CYCLES / CLOCK_HZ
    in_flight = BW * latency_s
    rows = []
    for loads in (1, 2, 4, 8):
        need = math.ceil(in_flight / (loads * F32))
        per_cu = math.ceil(need / N_CU)
        rows.append((loads, _n(need), _n(per_cu), _pct(per_cu / MAX_THREADS_PER_CU)))
    return (f"BW = 800 (GB/s), latency = {MEM_LATENCY_CYCLES} cycles at 2 GHz = {latency_s * 1e9:.0f} ns, so "
            f"bytes in flight = BW x latency = {_n(round(in_flight))} bytes = {_n(round(in_flight / SECTOR))} "
            "transactions of 32 bytes. Each invocation keeps `loads` independent 4-byte loads outstanding; "
            "every byte requested is used (perfect coalescing). Invocations needed = ceil(bytes in flight / (4 loads)); "
            "per compute unit = ceil(needed / 80); the last column is that as a percentage of the 2,048 resident "
            "invocations a compute unit can hold.\n\n"
            + _table(["loads in flight per invocation", "invocations needed", "per compute unit",
                      "% of a compute unit's 2,048"], rows))


# --------------------------------------------------------------- divergence

DIV_N = 1024
DIV_COST_TEST, DIV_COST_A, DIV_COST_B = 1, 10, 6


def _divergence_rows(flags_list):
    rows = []
    for name, flags in flags_list:
        mixed = issues = useful = 0
        for g in range(DIV_N // SUBGROUP):
            grp = flags[g * SUBGROUP:(g + 1) * SUBGROUP]
            a = sum(grp)
            has_a, has_b = a > 0, a < SUBGROUP
            mixed += has_a and has_b
            issues += DIV_COST_TEST + (DIV_COST_A if has_a else 0) + (DIV_COST_B if has_b else 0)
            useful += SUBGROUP * DIV_COST_TEST + a * DIV_COST_A + (SUBGROUP - a) * DIV_COST_B
        slots = issues * SUBGROUP
        rows.append((name, sum(flags), mixed, _n(issues), _n(useful), _n(slots), _pct(useful / slots)))
    return rows


def divergence():
    name = "gpu.divergence"
    gen = bookrng.generator(name)
    u = [gen.uniform() for _ in range(DIV_N)]
    flags_half = [x < 0.5 for x in u]
    flags_rare = [x < 0.05 for x in u]
    sorted_rare = sorted(flags_rare, reverse=True)
    cases = [
        ("path A when i is even", [i % 2 == 0 for i in range(DIV_N)]),
        ("path A when (i // 32) is even", [(i // SUBGROUP) % 2 == 0 for i in range(DIV_N)]),
        ("path A when u_i < 0.5", flags_half),
        ("path A when u_i < 0.05", flags_rare),
        ("same flags as the row above, A elements first", sorted_rare),
    ]
    rows = _divergence_rows(cases)
    p_mixed = lambda p: 1 - p ** SUBGROUP - (1 - p) ** SUBGROUP  # noqa: E731
    return (f"{DIV_N} invocations i = 0 .. {DIV_N - 1} in {DIV_N // SUBGROUP} subgroups of {SUBGROUP}. Every "
            f"invocation runs the test ({DIV_COST_TEST} instruction), then path A ({DIV_COST_A} instructions) or path "
            f"B ({DIV_COST_B}). A subgroup issues the test once, path A if any lane takes it, path B if any lane takes it. "
            "Issue slots = issued instructions x 32 lanes; useful slots = the instructions each lane needs itself; "
            "utilization = useful / slots. "
            f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (the seed of the name `{name}`); "
            f"u_0 .. u_{DIV_N - 1} are {DIV_N} calls of `uniform` in index order, the same draws for the two random rows.\n\n"
            + _table(["rule", "invocations on A", "subgroups with both paths", "instructions issued",
                      "useful lane-instructions", "issue slots", "utilization (%)"], rows)
            + "\n\nProbability that a subgroup of 32 disagrees with itself when each lane takes A independently with "
              "probability p: 1 - p^32 - (1 - p)^32 = "
              + ", ".join(f"{p_mixed(p):.4f} (p = {p})" for p in (0.5, 0.05, 0.01, 0.001)) + ".")


def divergence_flags():
    """The 1,024 uniform draws of `divergence`, in index order (for the figure)."""
    gen = bookrng.generator("gpu.divergence")
    return [gen.uniform() for _ in range(DIV_N)]


def divergence_measured(p):
    """Utilization of the 32 subgroups when invocation i takes path A iff u_i < p."""
    flags = [x < p for x in divergence_flags()]
    row = _divergence_rows([("", flags)])[0]
    return float(row[-1]) / 100


def divergence_curve(p):
    """Expected utilization of a subgroup whose lanes take A independently with probability p."""
    q = 1 - p
    pa = 1 - q ** SUBGROUP          # some lane takes A
    pb = 1 - p ** SUBGROUP          # some lane takes B
    issues = DIV_COST_TEST + DIV_COST_A * pa + DIV_COST_B * pb
    useful = DIV_COST_TEST + p * DIV_COST_A + q * DIV_COST_B
    return useful / issues


# ------------------------------------------------------------------- the grid

GRID_N, GRID_WG = 1000, 64


def grid_shape(n=GRID_N, wg=GRID_WG, sg=SUBGROUP):
    groups = -(-n // wg)
    return groups, groups * wg, groups * wg - n, wg // sg


def grid_indices():
    groups, threads, idle, sgs = grid_shape()
    rows = []
    for g in (0, 31, 32, 63, 64, 999, 1000, 1023):
        rows.append((g, g // GRID_WG, g % GRID_WG, (g % GRID_WG) // SUBGROUP, g % SUBGROUP,
                     "idle (g >= n)" if g >= GRID_N else "works"))
    return (f"n = {GRID_N} elements, workgroup size 64, subgroup size 32. Workgroups = ceil(n / 64) = {groups}; "
            f"invocations dispatched = {groups} x 64 = {_n(threads)}; idle = {idle}; subgroups per workgroup = "
            f"{sgs}. For global index g: workgroup = g // 64, local index = g mod 64, subgroup = (g mod 64) // 32, "
            "lane = g mod 32.\n\n"
            + _table(["global index g", "workgroup", "local index", "subgroup in workgroup", "lane", "role"], rows,
                     "rrrrrl"))


def dispatch_units():
    rows_n, wg = 512, 64
    right = rows_n
    wrong = -(-rows_n // wg)
    return (f"A kernel in which one workgroup of {wg} invocations normalizes one row (the shape of a layer "
            f"normalization over {rows_n} rows). The dispatch must count workgroups, one per row. If the caller "
            f"passes the row count as an invocation count, the grid holds ceil({rows_n} / {wg}) workgroups.\n\n"
            + _table(["dispatch", "workgroups", "invocations", "rows written", "rows never written"],
                     [("one workgroup per row", _n(right), _n(right * wg), _n(right), "0"),
                      ("row count taken as invocations", _n(wrong), _n(wrong * wg), _n(wrong), _n(rows_n - wrong))]))


# ----------------------------------------------------- shared memory and barriers

STENCIL_X = [3, 1, 4, 1, 5, 9, 2, 6]
STENCIL_STALE = 99


def _stencil(schedule):
    """One workgroup of len(STENCIL_X) invocations; thread t owns output t.

    Shared array s has len + 2 slots: s[0] is the left halo (0), s[t + 1] holds x[t], s[len + 1] is the right halo
    (0). Thread t writes s[t + 1] (thread 0 also writes s[0], the last thread also s[len + 1]) and then reads
    s[t], s[t + 1], s[t + 2]. Slots start holding STENCIL_STALE, leftover from an earlier workgroup.
    """
    n = len(STENCIL_X)
    s = [STENCIL_STALE] * (n + 2)
    out = [None] * n

    def write(t):
        s[t + 1] = STENCIL_X[t]
        if t == 0:
            s[0] = 0
        if t == n - 1:
            s[n + 1] = 0

    def read(t):
        out[t] = s[t] + s[t + 1] + s[t + 2]

    if schedule == "barrier":
        for t in range(n):
            write(t)
        for t in range(n):      # after the barrier
            read(t)
    else:                       # each thread runs to completion before the next starts
        for t in schedule:
            write(t)
            read(t)
    return out


def stencil_barrier():
    n = len(STENCIL_X)
    good = _stencil("barrier")
    ordered = _stencil(list(range(n)))
    backward = _stencil(list(range(n - 1, -1, -1)))
    rows = [("all writes, barrier, all reads", good),
            ("no barrier, threads run 0, 1, ..., 7 in turn", ordered),
            ("no barrier, threads run 7, 6, ..., 0 in turn", backward)]
    body = _table(["schedule"] + [f"y{t}" for t in range(n)], [(a, *b) for a, *b in
                  [(r[0], *r[1]) for r in rows]], "l" + "r" * n)
    w = 64
    return (f"x = {STENCIL_X}, y_t = x_(t-1) + x_t + x_(t+1) with x_(-1) = x_8 = 0, one workgroup of {n} invocations. "
            f"Shared slots start as {STENCIL_STALE} (leftover). The sums of the first row are the correct answer.\n\n"
            + body
            + f"\n\nGlobal reads for a workgroup of {w}: without staging 3 x {w} = {3 * w}; staging in shared memory "
              f"{w} + 2 = {w + 2}. For the workgroup of {n}: {3 * n} against {n + 2}.")


def tile_reuse():
    n = 1024
    flops = 2 * n ** 3
    rows = []
    for t in (1, 4, 8, 16, 32):
        loads = 2 * n ** 3 // t
        byts = F32 * (loads + n * n)
        shared = 2 * t * t * F32
        rows.append((t, _n(t * t), _n(shared), _n(loads), _n(byts), f"{flops / byts:.2f}",
                     "yes" if t * t <= 256 and shared <= 16384 else "no"))
    return (f"C = A B with n = {n}, fp32, one workgroup per T x T tile of C; for each of n/T steps along the shared "
            "dimension it loads one T x T tile of A and one of B into shared memory (2 T^2 numbers, 8 T^2 bytes), "
            "so global loads = (n/T)^2 x (n/T) x 2 T^2 = 2 n^3 / T. Bytes = 4 x (loads + n^2 stores of C). "
            f"N_flop = 2 n^3 = {_n(flops)}. 'Portable' asks for at most 256 invocations and 16 KiB of shared memory "
            "per workgroup (the baseline of the WebGPU floor).\n\n"
            + _table(["T", "invocations per workgroup", "shared bytes per workgroup", "global loads",
                      "bytes moved", "FLOP per byte", "portable"], rows))


# ---------------------------------------------------------------- occupancy


def occupancy(wg, regs, shared):
    """(resident workgroups, occupancy fraction, limiting resource) of one compute unit.

    Limits: registers (REGS_PER_CU // (regs x wg)), shared memory (SHARED_PER_CU // shared when shared > 0),
    resident invocations (MAX_THREADS_PER_CU // wg) and the workgroup cap. Ties are reported in that order.
    """
    limits = [("registers", REGS_PER_CU // (regs * wg)),
              ("shared memory", SHARED_PER_CU // shared if shared > 0 else MAX_WG_PER_CU),
              ("invocation slots", MAX_THREADS_PER_CU // wg),
              ("workgroup cap", MAX_WG_PER_CU)]
    best = min(v for _, v in limits)
    why = next(name for name, v in limits if v == best)
    return best, best * wg / MAX_THREADS_PER_CU, why


OCC_KERNELS = (
    ("vector add", 256, 8, 0),
    ("row normalization, 64 invocations", 64, 24, 512),
    ("matmul tile T = 16", 256, 40, 2048),
    ("matmul tile T = 32", 1024, 40, 8192),
    ("register-heavy kernel", 256, 128, 0),
    ("shared-memory-heavy kernel", 256, 32, 24576),
)


def occupancy_table():
    rows = []
    for name, wg, regs, shared in OCC_KERNELS:
        res, occ, why = occupancy(wg, regs, shared)
        rows.append((name, wg, regs, _n(shared), res, _n(res * wg), _pct(occ), why))
    return ("Per compute unit: 65,536 registers, 65,536 bytes of shared memory, 2,048 resident invocations, 16 "
            "resident workgroups. Resident workgroups = the smallest of 65,536 // (registers x size), "
            "65,536 // shared bytes, 2,048 // size and 16; occupancy = resident invocations / 2,048. "
            "A model: real hardware allocates registers in blocks.\n\n"
            + _table(["kernel", "workgroup size", "registers per invocation", "shared bytes per workgroup",
                      "resident workgroups", "resident invocations", "occupancy (%)", "limited by"], rows,
                     "lrrrrrrl"))


def register_cliff():
    need = math.ceil(BW * MEM_LATENCY_CYCLES / CLOCK_HZ / F32 / N_CU)
    rows = []
    for regs in (64, 96, 128, 129, 160, 255):
        cells = [regs]
        for wg in (64, 256):
            res, occ, _ = occupancy(wg, regs, 0)
            cells.append(f"{res * wg} ({'enough' if res * wg >= need else 'too few'})")
        rows.append(tuple(cells))
    return (f"Resident invocations per compute unit as the registers per invocation grow, no shared memory. One "
            f"4-byte load in flight per invocation needs {need} resident invocations per compute unit (the first "
            "row of the latency table). The cliff after 128 registers is the integer division 65,536 // (r x size).\n\n"
            + _table(["registers per invocation", "size 64", "size 256"], rows))


# --------------------------------------------------------------- coalescing


def sectors(addresses, size=F32):
    """Distinct 32-byte transactions that cover the byte ranges [a, a + size) for a in `addresses`."""
    touched = set()
    for a in addresses:
        touched.update(range(a // SECTOR, (a + size - 1) // SECTOR + 1))
    return len(touched)


def lane_addresses(stride_elems, start_elem=0, lanes=SUBGROUP):
    return [(start_elem + t * stride_elems) * F32 for t in range(lanes)]


def coalescing_strides():
    rows = []
    cases = [("1", 1, 0), ("1, starting at element 1", 1, 1),
             ("2", 2, 0), ("3", 3, 0), ("4", 4, 0), ("8", 8, 0), ("16", 16, 0), ("1,024", 1024, 0)]
    for name, stride, start in cases:
        s = sectors(lane_addresses(stride, start))
        moved = s * SECTOR
        useful = SUBGROUP * F32
        rows.append((name, s, moved, _pct(useful / moved), f"{BW / 1e9 * useful / moved:,.1f}"))
    return ("One load instruction of a subgroup of 32 invocations; lane t reads the 4-byte number at element "
            "start + t x stride of an array that begins on a 32-byte boundary. Transactions = number of distinct "
            "32-byte aligned blocks touched. Useful bytes = 32 x 4 = 128. The last column is the model's streaming "
            "bandwidth, 800 (GB/s) times the useful fraction.\n\n"
            + _table(["stride (elements)", "transactions", "bytes moved", "useful (%)", "effective GB/s"], rows))


def matrix_access():
    n = 1024
    loads = n * n // SUBGROUP
    row_s = sectors(lane_addresses(1))
    col_s = sectors(lane_addresses(n))
    tile_both = 2 * (n * n * F32 // SECTOR)
    naive = loads * row_s + loads * col_s
    rows = [
        ("read all of A, lanes along a row", _n(loads), row_s, _n(loads * row_s), f"{loads * row_s * SECTOR / 2 ** 20:.0f}"),
        ("read all of A, lanes down a column", _n(loads), col_s, _n(loads * col_s), f"{loads * col_s * SECTOR / 2 ** 20:.0f}"),
        ("transpose, rows read and columns written", _n(2 * loads), "-", _n(naive), f"{naive * SECTOR / 2 ** 20:.0f}"),
        ("transpose through a 32 x 32 tile in shared memory", _n(2 * loads), "-", _n(tile_both),
         f"{tile_both * SECTOR / 2 ** 20:.0f}"),
    ]
    return (f"A is [{n}, {n}] fp32 in row-major layout, {n * n * F32 // 2 ** 20} MiB, starting on a 32-byte "
            f"boundary: {_n(loads)} subgroup loads cover it once. Lane t of a subgroup reads A[i, j0 + t] (row) or "
            "A[i0 + t, j] (column); the transactions are counted per load instruction, as if no cache held the "
            "blocks between instructions (the count of requests, an upper bound on the bytes from memory). "
            "A transpose does the same number of loads and stores.\n\n"
            + _table(["access", "subgroup instructions", "transactions each", "transactions", "MiB moved"], rows))


def _layernorm_old(d, rows=SUBGROUP):
    """Row per invocation: at step c, lane t reads x[t x d + c]. Returns transactions for `rows` rows."""
    total = 0
    for c in range(d):
        total += sectors([(t * d + c) * F32 for t in range(rows)])
    return total


def _layernorm_new(d, rows=SUBGROUP, wg=64):
    """Workgroup per row (wg invocations striding by wg): subgroup s, step k, lane t reads x[row x d + k x wg + s x 32 + t]."""
    total = 0
    for row in range(rows):
        for k in range(-(-d // wg)):
            for s in range(wg // SUBGROUP):
                cols = [k * wg + s * SUBGROUP + t for t in range(SUBGROUP) if k * wg + s * SUBGROUP + t < d]
                if cols:
                    total += sectors([(row * d + c) * F32 for c in cols])
    return total


def layernorm_access():
    rows = []
    for d in (768, 1000, 1024, 3072, 770):
        old, new = _layernorm_old(d), _layernorm_new(d)
        ideal = -(-SUBGROUP * d * F32 // SECTOR)
        rows.append((d, _n(old), _n(new), _n(ideal), f"{old / new:.2f}"))
    return ("Reading 32 rows of a [rows, d] fp32 matrix once. 'Row per invocation': lane t owns row t and reads "
            "column c at step c (d steps). 'Workgroup per row': 64 invocations read one row, invocation u reading "
            "columns u, u + 64, ... (two subgroups, lane t of subgroup b reading column 64 k + 32 b + t at step k; "
            "lanes past d idle). Transactions are counted per subgroup load as in the previous table; "
            "the row is a multiple of 32 bytes only when 4 d is. Ideal = 32 d x 4 / 32.\n\n"
            + _table(["d", "row per invocation", "workgroup per row", "ideal", "first / second"], rows))


# ------------------------------------------------------- host and device


def transfers():
    n = 1 << 24
    byts = 3 * F32 * n
    t_bus = byts / BUS
    t_dev = byts / BW
    rows = [("a and b up, c down over the bus", _n(byts), _ms(t_bus)),
            ("the addition itself, memory bound on the device", _n(byts), _ms(t_dev)),
            ("bus time / device time", "-", f"{t_bus / t_dev:.1f}")]
    return (f"c = a + b of Section 1 with the data starting on the host: bytes over the bus = {_n(byts)} (a and b "
            "up, c back), bus 32 (GB/s), device memory 800 (GB/s). Times are bytes / rate, a model.\n\n"
            + _table(["step", "bytes", "milliseconds"], rows))


def bus_break_even():
    ridge = F_PK_CH3 / BUS
    rows = []
    for n in (1024, 4096, 8192, 16384):
        flops = 2 * n ** 3
        byts = 3 * F32 * n * n
        rows.append((_n(n), f"{flops / byts:,.1f}", _ms(byts / BUS), _ms(flops / F_PK_CH3),
                     "arithmetic" if flops / F_PK_CH3 > byts / BUS else "bus"))
    return (f"C = A B for [n, n] fp32 matrices that start on the host: A and B go up, C comes back, "
            f"{_n(3 * F32)}n^2 bytes over a 32 (GB/s) bus; the arithmetic runs at the 40,000 (GFLOP/s) peak of "
            f"Chapter 3. Intensity per bus byte = 2 n^3 / (12 n^2) = n / 6; the break-even is "
            f"F_pk / bus = {ridge:,.0f} FLOP per bus byte, so n = {6 * ridge:,.0f}.\n\n"
            + _table(["n", "FLOP per bus byte", "bus (ms)", "arithmetic at peak (ms)", "longer"], rows,
                     "rrrrl"))
