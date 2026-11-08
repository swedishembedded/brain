# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Writing Compute Kernels".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: compute_kernels.<name> -->` markers. Helpers with a leading
underscore, or that take arguments, are shared with the chapter's figures so a
picture and the table beside it cannot disagree.

The heart of the module is a SIMULATOR of the kernel model of the chapter. A
kernel is written as the program of ONE invocation; the simulator runs that
program for every invocation of every workgroup of a dispatch grid, in plain
Python, in an order the caller chooses. A kernel with a barrier is written as a
list of phases: the simulator runs a phase for every invocation of a workgroup
before it starts the next, and reports a fault when only some invocations of a
workgroup reach the barrier. Memory is one flat list of fp32 words with named
buffers laid out in it, so a write past the end of a buffer lands on whatever
lies next, and every load and store is recorded so that the memory
transactions of a subgroup (32-byte blocks, as in Chapter 4) can be counted.

Nothing here measures a machine. Everything printed is counting: invocations,
transactions of 32 bytes, bytes over a bus, synchronization points. Where a time
appears it is the model's output from a stated bus rate. There is no
randomness: every input is written out in the chapter.

Swedish Embedded AB implements solutions for writing GPU compute kernels that
are correct at the edges of their data and that agree with their host about the
layout of every buffer. If your team needs expertise in GPU kernel design and
verification for machine learning, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import struct

# ------------------------------------------------------------------ parameters

SUBGROUP = 32       # invocations that issue one memory instruction together (Chapter 4)
SECTOR = 32         # bytes per memory transaction (Chapter 4)
F32 = 4             # bytes per fp32 number
BUS = 32e9          # bytes/s, host to device bus of Chapter 4 (parameter)
SHARED_FILL = 99.0  # what a shared-memory slot holds before anything writes it
PAD = -777.0        # what a padding word between buffers holds
UNWRITTEN = 1000.0  # what an output buffer holds before the kernel runs

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


def _ceil_div(a, b):
    return -(-a // b)


def _yes(flag):
    return "yes" if flag else "no"


# ------------------------------------------------------------- the simulator


class Fault(Exception):
    """A device fault: an access outside all of memory, or a store to a read-only buffer."""


class BarrierDivergence(Exception):
    """A barrier that only some invocations of a workgroup reach."""

    def __init__(self, workgroup, arrived, of):
        super().__init__(f"workgroup {workgroup}: {arrived} of {of} invocations reach the barrier")
        self.workgroup, self.arrived, self.of = workgroup, arrived, of


class _Returned(Exception):
    """Control-flow signal of `return`: the invocation does no more work."""


class Buffer:
    def __init__(self, name, base, n, access):
        self.name, self.base, self.n, self.access = name, base, n, access


class Memory:
    """Flat device memory of fp32 words with named buffers laid out in it."""

    def __init__(self):
        self.words = []
        self.buffers = {}

    def alloc(self, name, values, access="read_write", align=8):
        """Append a buffer whose first word is a multiple of `align` words
        (8 words = 32 bytes, a transaction boundary). `align=1` packs it right
        after the previous buffer."""
        while len(self.words) % align:
            self.words.append(PAD)
        buf = Buffer(name, len(self.words), len(values), access)
        self.words.extend(float(v) for v in values)
        self.buffers[name] = buf
        return buf

    def values(self, name):
        b = self.buffers[name]
        return self.words[b.base:b.base + b.n]


class Invocation:
    """What one running copy of a kernel can see: its indices, its buffers and
    the shared memory of its workgroup."""

    def __init__(self, mem, bufs, policy, log, shared, wg_size, wx, wy, nx, ny, l):
        self._mem, self._bufs, self._policy, self._log, self._shared = mem, bufs, policy, log, shared
        self.l, self.wx, self.wy, self.nwg_x, self.nwg_y, self.wg_size = l, wx, wy, nx, ny, wg_size
        self.gid_x = wx * wg_size + l   # global_invocation_id.x
        self.gid_y = wy                 # global_invocation_id.y (one invocation per y-group)

    def _address(self, buf, i):
        if self._policy == "clamp":                 # the language's robust-access guarantee
            i = min(max(i, 0), buf.n - 1)
        a = buf.base + i
        if not 0 <= a < len(self._mem.words):
            raise Fault(f"access to word {a} of {len(self._mem.words)}")
        return a

    def load(self, name, i):
        buf = self._bufs[name]
        a = self._address(buf, i)
        self._log.append(("load", name, a))
        return self._mem.words[a]

    def store(self, name, i, value):
        buf = self._bufs[name]
        if buf.access != "read_write":
            raise Fault(f"store to read-only buffer {name}")
        a = self._address(buf, i)
        self._log.append(("store", name, a))
        self._mem.words[a] = float(value)

    def shared_load(self, i):
        return self._shared[i]

    def shared_store(self, i, value):
        self._shared[i] = float(value)

    def ret(self):
        raise _Returned


def _count_transactions(logs, wg_size):
    """Transactions of one phase of one workgroup. The j-th access of the
    invocations of a subgroup is one memory instruction (invocations that
    returned or skipped it do not issue it); it costs one transaction per
    distinct 32-byte block its addresses touch."""
    counts = {"load": 0, "store": 0}
    for s in range(_ceil_div(wg_size, SUBGROUP)):
        lanes = logs[s * SUBGROUP:(s + 1) * SUBGROUP]
        for j in range(max((len(x) for x in lanes), default=0)):
            blocks = {}
            for lane in lanes:
                if j < len(lane):
                    kind, name, a = lane[j]
                    blocks.setdefault((kind, name), set()).add(a * F32 // SECTOR)
            for (kind, _), s_ in blocks.items():
                counts[kind] += len(s_)
    return counts


def launch(mem, phases, groups, wg_size, params, bufs, *, shared_words=0, policy="raw", order="asc",
           collect=None, snapshots=None):
    """Run a kernel (a list of phases; a barrier lies between consecutive
    phases) over a grid of `groups` workgroups (an int, or an (x, y) pair),
    each of `wg_size` invocations. Returns the transactions counted.

    Workgroups run one after another in ascending order and the invocations of
    a workgroup in `order` ("asc" or "desc") inside each phase: a kernel whose
    result changes with `order` has a race. A barrier that only some
    invocations of a workgroup reach raises BarrierDivergence, or, if
    `collect` is a list, is appended to it and the workgroup is abandoned."""
    nx, ny = groups if isinstance(groups, tuple) else (groups, 1)
    stats = {"load": 0, "store": 0}
    for wy in range(ny):
        for wx in range(nx):
            shared = [SHARED_FILL] * shared_words
            alive = [True] * wg_size
            for k, phase in enumerate(phases):
                logs = [[] for _ in range(wg_size)]
                lanes = range(wg_size) if order == "asc" else range(wg_size - 1, -1, -1)
                for l in lanes:
                    if not alive[l]:
                        continue
                    inv = Invocation(mem, bufs, policy, logs[l], shared, wg_size, wx, wy, nx, ny, l)
                    try:
                        phase(inv, params)
                    except _Returned:
                        alive[l] = False
                c = _count_transactions(logs, wg_size)
                stats["load"] += c["load"]
                stats["store"] += c["store"]
                if k < len(phases) - 1:
                    arrived = sum(alive)
                    if 0 < arrived < wg_size:
                        fault = BarrierDivergence((wx, wy), arrived, wg_size)
                        if collect is None:
                            raise fault
                        collect.append(fault)
                        break
                    if snapshots is not None:
                        snapshots.append(((wx, wy), k, list(shared)))
    return stats


# ------------------------------------------------------------------- kernels
# Each kernel is a list of phases; a phase is the program of one invocation.


def _axpb(guard=True):
    """y[g] = scale * x[g] + shift, one invocation per element."""
    def phase(inv, p):
        g = inv.gid_x
        if guard and g >= p["n"]:
            inv.ret()
        inv.store("y", g, p["scale"] * inv.load("x", g) + p["shift"])
    return [phase]


def _axpb_flat(inv, p):
    """The same kernel with the 2D-safe flat index of brain's kernels."""
    g = inv.gid_y * (inv.nwg_x * inv.wg_size) + inv.gid_x
    if g >= p["n"]:
        inv.ret()
    inv.store("y", g, p["scale"] * inv.load("x", g) + p["shift"])


def _axpb_x_only(inv, p):
    """The flat index forgotten: correct only while the grid has one row."""
    g = inv.gid_x
    if g >= p["n"]:
        inv.ret()
    inv.store("y", g, p["scale"] * inv.load("x", g) + p["shift"])


def _view_gather():
    """y[g] = x[offset + i s0 + j s1] for g = i d1 + j: materialise a strided view."""
    def phase(inv, p):
        g = inv.gid_x
        if g >= p["d0"] * p["d1"]:
            inv.ret()
        i, j = g // p["d1"], g % p["d1"]
        inv.store("y", g, inv.load("x", p["offset"] + i * p["s0"] + j * p["s1"]))
    return [phase]


def _tiled_transpose(tile, variant="predicated"):
    """Transpose a [m, n] matrix to [n, m] through a tile x tile shared tile.
    Workgroup (wx, wy) owns the tile at rows wy*tile.. and columns wx*tile.. of x.
    variant: "predicated" (the guard sits around the loads and stores, the barrier
    is reached by all), "early_return" (the guard returns before the barrier),
    "no_barrier" (both steps in one phase)."""
    def load_step(inv, p):
        ly, lx = inv.l // tile, inv.l % tile
        r, c = inv.wy * tile + ly, inv.wx * tile + lx
        if r < p["rows"] and c < p["cols"]:
            inv.shared_store(ly * tile + lx, inv.load("x", r * p["cols"] + c))

    def early_load(inv, p):
        ly, lx = inv.l // tile, inv.l % tile
        r, c = inv.wy * tile + ly, inv.wx * tile + lx
        if r >= p["rows"] or c >= p["cols"]:
            inv.ret()
        inv.shared_store(ly * tile + lx, inv.load("x", r * p["cols"] + c))

    def store_step(inv, p):
        ly, lx = inv.l // tile, inv.l % tile
        out_row, out_col = inv.wx * tile + ly, inv.wy * tile + lx   # y[out_row, out_col] = x[out_col, out_row]
        if out_row < p["cols"] and out_col < p["rows"]:
            inv.store("y", out_row * p["rows"] + out_col, inv.shared_load(lx * tile + ly))

    if variant == "predicated":
        return [load_step, store_step]
    if variant == "early_return":
        return [early_load, store_step]
    return [lambda inv, p: (load_step(inv, p), store_step(inv, p))]


# ---------------------------------------------------- 1. the first kernel

AXPB_N, AXPB_WG, AXPB_SCALE, AXPB_SHIFT = 10, 4, 2.0, 1.0


def _axpb_input(n=AXPB_N):
    return [float(i - 3) for i in range(n)]


def _axpb_expected(x, scale=AXPB_SCALE, shift=AXPB_SHIFT):
    return [scale * v + shift for v in x]


def _run_axpb(n, groups, *, guard=True, policy="raw", neighbour_align=None, wg=AXPB_WG):
    """Run y = scale x + shift; returns (memory, stats). A neighbour buffer z of
    4 words follows y when `neighbour_align` is given (1 = packed right behind
    y, 8 = at the next 32-byte boundary)."""
    x = _axpb_input(n)
    mem = Memory()
    bx = mem.alloc("x", x, "read")
    by = mem.alloc("y", [UNWRITTEN] * n)
    if neighbour_align is not None:
        mem.alloc("z", [500.0 + i for i in range(4)], align=neighbour_align)
    mem.alloc("tail", [0.0] * 8, align=8)   # keeps an unchecked write inside memory
    params = {"n": n, "scale": AXPB_SCALE, "shift": AXPB_SHIFT}
    launch(mem, _axpb(guard), groups, wg, params, {"x": bx, "y": by}, policy=policy)
    return mem


def axpb_trace():
    n, wg = AXPB_N, AXPB_WG
    groups = _ceil_div(n, wg)
    x = _axpb_input()
    mem = _run_axpb(n, groups)
    y = mem.values("y")
    assert y == _axpb_expected(x)
    rows = []
    for g in range(groups * wg):
        w, l = g // wg, g % wg
        if g < n:
            rows.append((g, w, l, "yes", f"x[{g}] = {_v(x[g])}", f"y[{g}] = {_v(2 * x[g] + 1)}"))
        else:
            rows.append((g, w, l, "no", "-", "-"))
    return (f"y_g = 2 x_g + 1 for N = {n}, x = [{', '.join(_v(v) for v in x)}], workgroup size {wg}. Workgroups = "
            f"ceil({n} / {wg}) = {groups}; invocations launched = {groups * wg}. Each invocation computes its global "
            f"index g = w {wg} + l and runs the guard g < N. The simulator ran all {groups * wg} invocations; "
            f"y = [{', '.join(_v(v) for v in y)}], equal to the direct computation 2 x + 1: "
            f"{_yes(y == _axpb_expected(x))}.\n\n"
            + _table(["g", "workgroup w", "local l", "g < N", "reads", "writes"], rows,
                     align="rrrlll"))


# --------------------------------------------- 2. the grid size, and its errors


def grid_failures():
    n, wg = AXPB_N, AXPB_WG
    expected = _axpb_expected(_axpb_input(n))
    cases = [
        ("ceil(N / wg) groups, with the guard", n, _ceil_div(n, wg), True, "raw", None),
        ("floor(N / wg) groups, with the guard", n, n // wg, True, "raw", None),
        ("ceil, no guard, z packed behind y", n, _ceil_div(n, wg), False, "raw", 1),
        ("ceil, no guard, z at the next 32-byte boundary", n, _ceil_div(n, wg), False, "raw", 8),
        ("ceil, no guard, accesses clamped to the buffer", n, _ceil_div(n, wg), False, "clamp", 1),
        ("N = 12, floor(N / wg) groups, with the guard", 12, 12 // wg, True, "raw", None),
    ]
    rows = []
    for name, nn, groups, guard, policy, align in cases:
        mem = _run_axpb(nn, groups, guard=guard, policy=policy, neighbour_align=align)
        y = mem.values("y")
        want = _axpb_expected(_axpb_input(nn))
        ok = sum(1 for a, b in zip(y, want) if a == b)
        unwritten = sum(1 for a in y if a == UNWRITTEN)
        z_changed = (sum(1 for k, a in enumerate(mem.values("z")) if a != 500.0 + k)
                     if "z" in mem.buffers else 0)
        pad_changed = sum(1 for a in mem.words[mem.buffers["y"].base + nn:
                                               mem.buffers["z"].base if "z" in mem.buffers else 0] if a != PAD)
        rows.append((name, nn, groups, groups * wg, f"{ok} of {nn}", unwritten, z_changed if align else "-",
                     pad_changed if align else "-"))
    return (f"The kernel of the previous block, y = 2 x + 1 with workgroup size {wg}, launched with a wrong grid or "
            f"without its guard. y starts at {_v(UNWRITTEN)} (a value the kernel never writes); the neighbour buffer z has 4 words "
            f"500, 501, 502, 503; memory words between buffers hold {_v(PAD)}. 'clamped' is the language's robust-access "
            f"rule: an index past the end is replaced by the last valid one. Last two columns: words of z changed, "
            f"padding words changed.\n\n"
            + _table(["launch", "N", "workgroups", "invocations", "y correct", "y never written",
                      "z changed", "padding changed"], rows, align="lrrrrrrr"))


# ------------------------------------- 3. strided access: materialising a view


VIEW_STORAGE = [0, 10, 20, 30, 40, 50, 60]
VIEW_OFFSET, VIEW_SHAPE, VIEW_STRIDES = 1, (2, 3), (1, 2)


def _gather_view(storage, offset, shape, strides, wg=4):
    mem = Memory()
    bx = mem.alloc("x", storage, "read")
    by = mem.alloc("y", [UNWRITTEN] * (shape[0] * shape[1]))
    mem.alloc("tail", [0.0] * 8)
    params = {"d0": shape[0], "d1": shape[1], "s0": strides[0], "s1": strides[1], "offset": offset}
    launch(mem, _view_gather(), _ceil_div(shape[0] * shape[1], wg), wg, params, {"x": bx, "y": by})
    return mem.values("y")


def view_gather():
    storage, offset, shape, strides = VIEW_STORAGE, VIEW_OFFSET, VIEW_SHAPE, VIEW_STRIDES
    wg = 4
    total = shape[0] * shape[1]
    y = _gather_view(storage, offset, shape, strides, wg)
    direct = [storage[offset + i * strides[0] + j * strides[1]] for i in range(shape[0]) for j in range(shape[1])]
    assert y == [float(v) for v in direct] == [10.0, 30.0, 50.0, 20.0, 40.0, 60.0]
    rows = []
    for g in range(_ceil_div(total, wg) * wg):
        if g < total:
            i, j = g // shape[1], g % shape[1]
            rows.append((g, f"({i}, {j})", f"{offset} + {i} x {strides[0]} + {j} x {strides[1]} = "
                         f"{offset + i * strides[0] + j * strides[1]}",
                         _v(y[g])))
        else:
            rows.append((g, "-", f"idle (g >= {total})", "-"))
    return (f"x = {storage}, view of shape {list(shape)}, strides {strides}, offset {offset}, as in Volume I, Chapter 5. "
            f"Output element g = i d1 + j reads x[offset + i s0 + j s1] and writes y[g]. Workgroup size {wg}, "
            f"ceil({total} / {wg}) = {_ceil_div(total, wg)} workgroups. y = [{', '.join(_v(v) for v in y)}], equal to "
            f"the view read directly: {_yes(y == [float(v) for v in direct])}.\n\n"
            + _table(["g", "(i, j)", "offset into x", "y[g]"], rows, align="rlll"))


# --------------------------------- 4. transposes, and the transactions counted

TR_SIDE, TR_TILE = 128, 16


def _transpose_runs(side=TR_SIDE, tile=TR_TILE):
    """(name, stats, equals-direct) for the copy, the naive transpose and the
    tiled transpose of a [side, side] matrix with a[i, j] = i side + j."""
    m = n = side
    a = [float(i * n + j) for i in range(m) for j in range(n)]
    direct_t = [a[j * n + i] for i in range(n) for j in range(m)]
    out = []
    for name, shape, strides in (("copy, strides (n, 1)", (m, n), (n, 1)),
                                 ("transpose, one invocation per output element, strides (1, n)", (n, m), (1, n))):
        mem = Memory()
        bx = mem.alloc("x", a, "read")
        by = mem.alloc("y", [UNWRITTEN] * (m * n))
        params = {"d0": shape[0], "d1": shape[1], "s0": strides[0], "s1": strides[1], "offset": 0}
        st = launch(mem, _view_gather(), _ceil_div(m * n, 64), 64, params, {"x": bx, "y": by})
        out.append((name, st, mem.values("y") == (a if strides == (n, 1) else direct_t), 64, _ceil_div(m * n, 64) * 64))
    mem = Memory()
    bx = mem.alloc("x", a, "read")
    by = mem.alloc("y", [UNWRITTEN] * (m * n))
    st = launch(mem, _tiled_transpose(tile), (_ceil_div(n, tile), _ceil_div(m, tile)), tile * tile,
                {"rows": m, "cols": n}, {"x": bx, "y": by}, shared_words=tile * tile)
    out.append((f"transpose through a {tile} x {tile} tile in shared memory", st, mem.values("y") == direct_t,
                tile * tile, _ceil_div(n, tile) * _ceil_div(m, tile) * tile * tile))
    return out


def _transactions_model(kind, count):
    """Closed form for N = count elements read once and written once, 32-lane
    subgroups, 32-byte blocks, rows of at least 8 numbers."""
    instr = count // SUBGROUP
    full = SUBGROUP * F32 // SECTOR   # blocks of a fully coalesced instruction: 4
    if kind == "copy" or kind == "tiled":
        return instr * full, instr * full
    return instr * SUBGROUP, instr * full     # naive transpose: one block per lane on the strided side


def transpose_traffic():
    side = TR_SIDE
    runs = _transpose_runs(side)
    count = side * side
    kinds = ("copy", "naive", "tiled")
    rows = []
    for (name, st, ok, wg, invocations), kind in zip(runs, kinds):
        assert ok
        assert (st["load"], st["store"]) == _transactions_model(kind, count)
        total = st["load"] + st["store"]
        rows.append((name, _n(invocations), _n(st["load"]), _n(st["store"]), _n(total), _n(total * SECTOR),
                     f"{100 * 2 * F32 * count / (total * SECTOR):.1f}", _yes(ok)))
    big = 1024
    big_rows = []
    for (name, _, _, _, _), kind in zip(runs, kinds):
        ld, stt = _transactions_model(kind, big * big)
        big_rows.append((name, _n(ld), _n(stt), _n(ld + stt), f"{(ld + stt) * SECTOR / 2 ** 20:g}"))
    return (f"A [{side}, {side}] fp32 matrix a[i, j] = {side} i + j, 32-lane subgroups, 32-byte blocks, every buffer "
            f"starting on a block boundary. Each row of a table is a simulated dispatch whose loads and stores were "
            f"recorded and counted per subgroup instruction; 'equals direct' compares the simulated output with the "
            f"transpose (or copy) computed directly. Useful bytes = 2 x 4 x {_n(count)} = {_n(2 * F32 * count)}.\n\n"
            + _table(["kernel", "invocations", "load blocks", "store blocks", "blocks", "bytes moved", "useful %",
                      "equals direct"], rows, align="lrrrrrrl")
            + f"\n\nThe counts follow N/4 (copy, tiled) and N + N/8 (naive) for N elements, a formula that equals the "
              f"simulation above. Applied to N = {big} x {big} = {_n(big * big)}:\n\n"
            + _table(["kernel", "load blocks", "store blocks", "blocks", "MiB moved"], big_rows))


def strided_toy():
    """Data of the index-mapping figure: a [3, 8] matrix transposed to [8, 3]."""
    m, n = 3, 8
    reads = [j * n + i for i in range(n) for j in range(m)]   # output g = i m + j reads x[j n + i]
    return m, n, reads


# --------------------------------------- 5. shared memory and a barrier

TILE_M, TILE_N, TILE_T = 6, 10, 4


def _tile_run(m, n, tile, variant, order="asc"):
    a = [float(10 * i + j) for i in range(m) for j in range(n)]
    mem = Memory()
    bx = mem.alloc("x", a, "read")
    by = mem.alloc("y", [UNWRITTEN] * (m * n))
    faults, snaps = [], []
    launch(mem, _tiled_transpose(tile, variant), (_ceil_div(n, tile), _ceil_div(m, tile)), tile * tile,
           {"rows": m, "cols": n}, {"x": bx, "y": by}, shared_words=tile * tile, order=order, collect=faults,
           snapshots=snaps)
    want = [a[j * n + i] for i in range(n) for j in range(m)]
    return mem.values("y"), want, faults, snaps, a


def tile_trace():
    m, n, t = TILE_M, TILE_N, TILE_T
    y, want, faults, snaps, a = _tile_run(m, n, t, "predicated")
    assert y == want and not faults
    snap = next(s for (w, k, s) in snaps if w == (2, 1) and k == 0)
    rows_of = lambda flat: [flat[r * t:(r + 1) * t] for r in range(t)]
    tile_rows = [(r, *[_v(v) for v in row]) for r, row in enumerate(rows_of(snap))]
    out_rows = [(8 + r, *[_v(y[(8 + r) * m + 4 + c]) for c in range(2)]) for r in range(2)]
    return (f"x is [{m}, {n}] with x[i, j] = 10 i + j, tile {t} x {t}, workgroups = ceil({n} / {t}) x ceil({m} / {t}) = "
            f"{_ceil_div(n, t)} x {_ceil_div(m, t)} = {_ceil_div(n, t) * _ceil_div(m, t)} of {t * t} invocations; the shared tile "
            f"starts as {_v(SHARED_FILL)} in every slot. Workgroup (2, 1) owns rows 4 to 5 and columns 8 to 9 of x, "
            f"a ragged tile (2 of {t} rows and 2 of {t} columns exist). Its shared tile at the barrier (invocations that "
            f"skip a load leave {_v(SHARED_FILL)}):\n\n"
            + _table(["tile row"] + [f"col {c}" for c in range(t)], tile_rows)
            + f"\n\nand the 2 x 2 block of y = x^T it writes after the barrier (rows 8 to 9, columns 4 to 5):\n\n"
            + _table(["y row", "col 4", "col 5"], out_rows)
            + f"\n\nThe whole simulated y equals the direct transpose of x: {_yes(y == want)}.")


def barrier_rule():
    rows = []
    for (m, n) in ((6, 10), (8, 12)):
        for variant, label in (("predicated", "guard around the load and store, barrier reached by all"),
                               ("early_return", "guard returns before the barrier")):
            y, want, faults, _, _ = _tile_run(m, n, 4, variant)
            groups = _ceil_div(n, 4) * _ceil_div(m, 4)
            rows.append((f"[{m}, {n}]", label, groups, len(faults),
                         (_yes(y == want) if not faults else "undefined"),
                         (", ".join(f"({f.workgroup[0]}, {f.workgroup[1]}): {f.arrived} of {f.of}" for f in faults)
                          or "-")))
    return ("The tiled transpose of the previous block (tile 4 x 4, 16 invocations per workgroup) with the guard written "
            "two ways, on a matrix whose sides are not multiples of the tile and on one whose sides are. A workgroup "
            "'faults' when only some of its invocations reach the barrier; its result is then undefined. The last column "
            "lists the faulting workgroups (x, y) and how many invocations arrived.\n\n"
            + _table(["x", "guard", "workgroups", "faulting", "y equals direct", "arrived at the barrier"], rows,
                     align="llrrll"))


def missing_barrier():
    m, n, t = TILE_M, TILE_N, TILE_T
    rows = []
    for variant, label in (("predicated", "load phase, barrier, store phase"),
                           ("no_barrier", "load and store in one phase, no barrier")):
        counts = []
        for order in ("asc", "desc"):
            y, want, _, _, _ = _tile_run(m, n, t, variant, order)
            counts.append(sum(1 for a, b in zip(y, want) if a != b))
        rows.append((label, f"{counts[0]} of {m * n}", f"{counts[1]} of {m * n}"))
    return (f"The same [{m}, {n}] transpose with and without the barrier. Inside each phase the invocations of a workgroup "
            f"run in the order 0, 1, ..., {t * t - 1} (ascending) or {t * t - 1}, ..., 0 (descending); both orders are legal "
            f"on a device. Entries of y that differ from the direct transpose:\n\n"
            + _table(["kernel", "ascending order", "descending order"], rows))


# ---------------------------------------------- 6. parameter layout in bytes

# name: (alignment, size) in bytes of WGSL types made of 4-byte scalars
TYPES = {"u32": (4, 4), "i32": (4, 4), "f32": (4, 4),
         "vec2<u32>": (8, 8), "vec2<f32>": (8, 8),
         "vec3<u32>": (16, 12), "vec3<f32>": (16, 12),
         "vec4<u32>": (16, 16), "vec4<f32>": (16, 16)}

# the example parameter block of the chapter (WGSL type of each field, in declaration order)
EXAMPLE_FIELDS = [("n", "u32"), ("scale", "f32"), ("origin", "vec3<f32>"), ("flag", "u32"),
                  ("dims", "vec2<u32>"), ("gain", "f32")]
EXAMPLE_VALUES = {"n": 1000, "scale": 0.5, "origin": (1.0, 2.0, 3.0), "flag": 7, "dims": (32, 16), "gain": 2.5}


def _round_up(v, a):
    return _ceil_div(v, a) * a


def _layout(fields, packed=False):
    """Offsets of a struct's fields. With packed=True every field is aligned to
    4 bytes (a host struct of 4-byte scalars and arrays of them); otherwise to
    its WGSL alignment. Returns (rows, struct alignment, struct size)."""
    off, rows, struct_align = 0, [], 1
    for name, ty in fields:
        align, size = TYPES[ty]
        if packed:
            align = 4
        start = _round_up(off, align)
        rows.append({"name": name, "type": ty, "align": align, "offset": start, "size": size, "gap": start - off})
        off, struct_align = start + size, max(struct_align, align)
    return rows, struct_align, _round_up(off, struct_align)


def _encode(rows, values, total):
    """Bytes of the values at the offsets of a layout, little-endian, zero-padded to `total`."""
    buf = bytearray(total)
    for r in rows:
        v = values[r["name"]]
        v = v if isinstance(v, tuple) else (v,)
        code = "<" + ("f" if "f32" in r["type"] else "I") * len(v)
        struct.pack_into(code, buf, r["offset"], *v)
    return bytes(buf)


def _decode(rows, data):
    out = {}
    for r in rows:
        count = {"vec2": 2, "vec3": 3, "vec4": 4}.get(r["type"][:4], 1)
        code = "<" + ("f" if "f32" in r["type"] else "I") * count
        v = struct.unpack_from(code, data, r["offset"])
        out[r["name"]] = v if count > 1 else v[0]
    return out


def _fmt_value(v):
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt_value(x) for x in v) + ")"
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def parameter_layout_rows():
    """Data for the figure and the tables: (kernel rows, size, host rows, host size)."""
    krows, kalign, ksize = _layout(EXAMPLE_FIELDS)
    hrows, _, hsize = _layout(EXAMPLE_FIELDS, packed=True)
    return krows, kalign, ksize, hrows, hsize


def struct_layout():
    krows, kalign, ksize, _, _ = parameter_layout_rows()
    rows = []
    for r in krows:
        rows.append((r["name"], f"`{r['type']}`", r["align"], r["size"], r["gap"] or "-", r["offset"],
                     f"{r['offset']} to {r['offset'] + r['size'] - 1}"))
    end = krows[-1]["offset"] + krows[-1]["size"]
    payload = sum(r["size"] for r in krows)
    return ("The parameter block `struct Params { n: u32, scale: f32, origin: vec3<f32>, flag: u32, dims: vec2<u32>, "
            "gain: f32 }`, laid out by the rules: a field starts at the next multiple of its alignment after the end of "
            "the previous field; a scalar has alignment and size 4, a vec2 of scalars 8 and 8, a vec3 alignment 16 and size "
            "12, a vec4 16 and 16; the struct's alignment is the largest field alignment and its size is the end of the last "
            "field rounded up to it. 'gap' is the padding before the field in bytes.\n\n"
            + _table(["field", "type", "alignment", "size", "gap before", "offset", "bytes"], rows, align="llrrrrl")
            + f"\n\nThe last field ends at byte {end}; the struct alignment is {kalign}, so its size is "
              f"ceil({end} / {kalign}) x {kalign} = {ksize} bytes. Of those, {payload} bytes hold values and "
              f"{ksize - payload} are padding ({sum(r['gap'] for r in krows)} before fields, {ksize - end} at the end).")


def host_disagreement():
    krows, _, ksize, hrows, hsize = parameter_layout_rows()
    sent = _encode(hrows, EXAMPLE_VALUES, ksize)    # the host fills 36 bytes, the buffer is padded to 48
    seen = _decode(krows, sent)
    host_off = {r["name"]: r["offset"] for r in hrows}
    rows = []
    for r in krows:
        want = EXAMPLE_VALUES[r["name"]]
        got = seen[r["name"]]
        rows.append((r["name"], host_off[r["name"]], r["offset"], _fmt_value(want), _fmt_value(got),
                     _yes(got == want)))
    return (f"The host (a C-style struct of 4-byte scalars and arrays of them, every field aligned to 4) writes the values "
            f"n = 1000, scale = 0.5, origin = (1, 2, 3), flag = 7, dims = (32, 16), gain = 2.5 at its own offsets; the struct "
            f"is {hsize} bytes and the buffer is padded with zeros to the {ksize} bytes the kernel's block occupies. The "
            f"kernel reads each field at ITS offset and type. Floats print with 6 significant digits; "
            f"a u32 that is read as an f32 is a tiny subnormal number.\n\n"
            + _table(["field", "host offset", "kernel offset", "host wrote", "kernel reads", "agree"], rows,
                     align="lrrlll"))


def flat_blocks():
    """The parameter blocks of flat scalar kernels: words, struct size, buffer bytes."""
    blocks = [("add", ["total"]), ("axpy", ["n", "s"]), ("clip_coef_wg", ["n_parts", "max_norm", "extra_scale"]),
              ("swap_axes12_vec", ["a0", "a1", "a2", "d"]),
              ("scale_add", ["seq_len", "d_model", "n_experts", "e_idx", "accumulate"])]
    rows = []
    for name, fields in blocks:
        words = len(fields)
        rows.append((name, words, 4 * words, _round_up(4 * words, 16), _round_up(4 * words, 16) // 4 - words))
    return ("Flat blocks of 4-byte scalars have alignment 4, so the struct is 4 x (number of fields) bytes, and the "
            "buffer that holds it is rounded up to a multiple of 16 bytes and filled with zeros. A list of words shorter "
            "than the block therefore still fills the buffer; the missing trailing fields read 0.\n\n"
            + _table(["kernel", "fields", "struct bytes", "buffer bytes", "zero words after the fields"], rows))


def binding_offsets():
    """Per-head sub-buffers bound at an offset: raw stride against a stride padded to 64 words."""
    length, hd, heads, align_bytes = 200, 4, 4, 256
    raw = length * hd
    padded = _round_up(raw, align_bytes // F32)
    rows = []
    for h in range(heads):
        r_off, p_off = h * raw * F32, h * padded * F32
        rows.append((h, _n(r_off), r_off % align_bytes, _yes(r_off % align_bytes == 0), _n(p_off),
                     p_off % align_bytes, _yes(p_off % align_bytes == 0)))
    return (f"Four per-head slices of a packed buffer, {length} positions of {hd} fp32 numbers each, so a slice is "
            f"{length} x {hd} = {raw} words. Binding a sub-range of a buffer requires its offset to be a multiple of "
            f"{align_bytes} bytes ({align_bytes // F32} words). Raw stride {raw} words = {raw * F32} bytes; padded stride = "
            f"ceil({raw} / {align_bytes // F32}) x {align_bytes // F32} = {padded} words = {padded * F32} bytes, "
            f"{padded - raw} words ({100 * (padded - raw) / raw:.2f} percent) of padding per slice.\n\n"
            + _table(["head", "raw offset (bytes)", "mod 256", "aligned", "padded offset (bytes)", "mod 256", "aligned"],
                     rows))


# ------------------------------------------- 7. the host side: dispatch and read-back


class _HostDevice:
    """The host's view of a device: buffers, uploads, an in-order queue of
    dispatches, and read-backs that wait for the queue. Counts what crosses the
    bus and how often the host waits."""

    def __init__(self, n, wg):
        self.n, self.wg = n, wg
        self.mem = Memory()
        self.queue = []
        self.upload_bytes = self.download_bytes = self.syncs = self.dispatches = 0
        self.pending_at_sync = []

    def create(self, name, access="read_write"):
        self.mem.alloc(name, [0.0] * self.n, access)

    def upload(self, name, values):
        b = self.mem.buffers[name]
        self.mem.words[b.base:b.base + b.n] = [float(v) for v in values]
        self.upload_bytes += F32 * len(values)

    def upload_params(self, words):
        self.upload_bytes += _round_up(F32 * words, 16)     # the uniform buffer is padded to 16 bytes

    def dispatch(self, src, dst, scale, shift):
        """Enqueue y = scale x + shift from buffer src to buffer dst; the host does not wait."""
        self.queue.append((src, dst, scale, shift))
        self.dispatches += 1
        self.upload_params(3)

    def read(self, name):
        """Wait for every queued dispatch, then copy the buffer to the host."""
        self.syncs += 1
        self.pending_at_sync.append(len(self.queue))
        for src, dst, scale, shift in self.queue:
            bufs = {"x": self.mem.buffers[src], "y": self.mem.buffers[dst]}
            launch(self.mem, _axpb(True), _ceil_div(self.n, self.wg), self.wg,
                   {"n": self.n, "scale": scale, "shift": shift}, bufs)
        self.queue = []
        v = self.mem.values(name)
        self.download_bytes += F32 * len(v)
        return v


CHAIN = ((2.0, 1.0), (0.5, -2.0), (3.0, 0.5))      # (scale, shift) of three kernels run one after another


def _chain_direct(x):
    for scale, shift in CHAIN:
        x = [scale * v + shift for v in x]
    return x


def _run_chain(resident, n=AXPB_N, wg=AXPB_WG):
    dev = _HostDevice(n, wg)
    names = ["b0", "b1", "b2", "b3"]
    for nm in names:
        dev.create(nm)
    x = _axpb_input(n)
    dev.upload("b0", x)
    if resident:
        for k, (scale, shift) in enumerate(CHAIN):
            dev.dispatch(names[k], names[k + 1], scale, shift)
        y = dev.read(names[3])
    else:
        cur = x
        for k, (scale, shift) in enumerate(CHAIN):
            if k:
                dev.upload("b0", cur)
            dev.dispatch("b0", "b1", scale, shift)
            cur = dev.read("b1")
        y = cur
    return dev, y


def host_program():
    n = AXPB_N
    x = _axpb_input(n)
    want = _chain_direct(x)
    rows = []
    for resident, label in ((False, "each result read back and uploaded again"),
                            (True, "buffers stay on the device, one read-back")):
        dev, y = _run_chain(resident)
        assert y == want
        rows.append((label, dev.dispatches, dev.syncs, ", ".join(str(p) for p in dev.pending_at_sync),
                     _n(dev.upload_bytes), _n(dev.download_bytes), _yes(y == want)))
    return (f"Three kernels y = scale x + shift with (scale, shift) = {CHAIN[0]}, {CHAIN[1]}, {CHAIN[2]} on x = "
            f"[{', '.join(_v(v) for v in x)}] (N = {n}, workgroup size {AXPB_WG}), run by a host program that creates "
            f"buffers, uploads, enqueues dispatches and reads back. Each dispatch uploads its 3-word parameter block "
            f"into a uniform buffer padded to 16 bytes. A read-back first runs every dispatch still in the queue "
            f"('pending when the host waits'). Final y = [{', '.join(_v(v) for v in want)}].\n\n"
            + _table(["design", "dispatches", "read-backs (waits)", "pending when the host waits", "bytes up",
                      "bytes down", "y equals direct"], rows, align="lrrlrrl"))


def bus_traffic():
    n = 1 << 24
    k = len(CHAIN)
    rows = []
    for label, up_vecs, down_vecs, waits in (
            ("each result read back and uploaded again", k, k, k),
            ("buffers stay on the device, one read-back", 1, 1, 1)):
        up = up_vecs * F32 * n + k * 16
        down = down_vecs * F32 * n
        ms = (up + down) / BUS * 1e3
        rows.append((label, _n(up), _n(down), waits, f"{ms:.4f}"))
    return (f"The two designs of the previous block for N = 2^24 = {_n(n)} fp32 numbers and {k} kernels: bytes up = "
            f"(vectors uploaded) x 4N + {k} parameter blocks of 16 bytes, bytes down = (vectors read back) x 4N. The "
            f"last column is the model time of those bytes over the {BUS / 1e9:.0f} (GB/s) bus of Chapter 4.\n\n"
            + _table(["design", "bytes up", "bytes down", "waits", "bus time (ms)"], rows))


def dispatch_figure_numbers():
    """Numbers drawn in the dispatch-path figure for n = 10, workgroup size 4."""
    n, wg = AXPB_N, AXPB_WG
    return {"n": n, "wg": wg, "groups": _ceil_div(n, wg), "vec_bytes": F32 * n, "params_bytes": _round_up(F32 * 3, 16)}


# ------------------------------------------------- 8. a grid of two dimensions


def grid_dims(threads, wg, max_dim):
    groups = max(_ceil_div(threads, wg), 1)
    if groups <= max_dim:
        return groups, 1
    return max_dim, _ceil_div(groups, max_dim)


def grid_fold():
    wg, small_max = 4, 3
    rows = []
    n = 30
    gx, gy = grid_dims(n, wg, small_max)
    for label, kernel in (("index g = gid.x only", _axpb_x_only), ("index g = (w_y n_x + w_x) n_wg + l", _axpb_flat)):
        mem = Memory()
        x = _axpb_input(n)
        bx = mem.alloc("x", x, "read")
        by = mem.alloc("y", [UNWRITTEN] * n)
        launch(mem, [kernel], (gx, gy), wg, {"n": n, "scale": 2.0, "shift": 1.0}, {"x": bx, "y": by})
        y = mem.values("y")
        ok = sum(1 for a, b in zip(y, _axpb_expected(x)) if a == b)
        rows.append((label, f"({gx}, {gy})", gx * gy * wg, f"{ok} of {n}", n - ok))
    small = _table(["kernel", "grid (n_x, n_y)", "invocations", "y correct", "never written"], rows, align="lrrrr")
    big_n, big_wg, big_max = 1024 * 1024 * 3 * 3, 64, 65535
    bx_, by_ = grid_dims(big_n, big_wg, big_max)
    covered = big_max * big_wg
    big = [("elements N = 1024 x 1024 x 3 x 3", _n(big_n)),
           ("workgroups needed = ceil(N / 64)", _n(_ceil_div(big_n, big_wg))),
           ("grid (n_x, n_y), n_x at most 65,535", f"({_n(bx_)}, {by_})"),
           ("workgroups launched", _n(bx_ * by_)),
           ("invocations launched", _n(bx_ * by_ * big_wg)),
           ("idle (fail the guard)", _n(bx_ * by_ * big_wg - big_n)),
           ("elements a gid.x-only kernel can reach = 65,535 x 64", _n(covered)),
           ("elements it never writes", _n(big_n - covered)),
           ("fraction never written", f"{100 * (big_n - covered) / big_n:.1f} percent")]
    return (f"A dispatch limited to {small_max} workgroups per dimension, y = 2 x + 1 for N = {n} with workgroup size {wg}: "
            f"ceil({n} / {wg}) = {_ceil_div(n, wg)} workgroups do not fit one row, so the grid is "
            f"({gx}, {_ceil_div(_ceil_div(n, wg), gx)}) = {gx * gy} workgroups. (n_x = workgroups along x, w_x and w_y = "
            f"the workgroup's coordinates.)\n\n"
            + small
            + "\n\nThe real limit of 65,535 workgroups per dimension with workgroup size 64, applied to the weight-gradient "
              "tensor of a 3 x 3 convolution with 1024 input and 1024 output channels:\n\n"
            + _table(["quantity", "value"], big, align="lr"))
