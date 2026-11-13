# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Reference implementation and acceptance tables for the Volume IV project.

Plain Python 3, standard library only, written from the project chapter's own
description. The first half of the module is the reference engine: a strided
tensor with a float64 matrix-multiply oracle, a backend interface with a CPU
backend that runs kernels written as the program of one invocation through the
workgroup-and-barrier executor of Chapter 5 (`compute_kernels.launch`), the
kernels themselves (naive, tiled, register-blocked, with an optional epilogue
and a deliberately broken variant; row-sum and two-pass reductions; a
group-quantized matrix product), a device configuration that changes the
workgroup tile, the order of the additions and the contraction of multiply-adds,
the parity test with its derived tolerance, and the selection pipeline
(capability filter, reference gate, seeded model clock). The second half
builds the acceptance tables: each public function without arguments returns a
markdown string that the chapter embeds between
`<!-- worked: project_tensor_engine.<name> -->` markers, every expected value
computed here by the reference itself.

The engine reuses, by import, the pieces earlier chapters built: the kernel
simulator (Chapter 5), the float32 arithmetic and fused multiply-add of
Chapter 9 (`backends`), the GELU of Chapter 8 (`fusion`), the model devices and
cost model of Chapter 10 (`kernel_selection`), the clock and the warm-up rule
of Chapter 11 (`profiling`). Nothing here measures a machine: every cost is a
count (loads, stores, FLOPs, bytes) or the output of a seeded model clock, and
the model clock is labelled as such wherever it is used.

Layout of numbers: every matrix is a flat list in row-major order; a float32
value is held in a Python float that is exactly a float32. A hash of a result is
SHA-256 over the little-endian binary32 (or binary64, as stated) bytes of the
values in row-major order, written as lowercase hexadecimal and cut to 16 digits.

Randomness: every generator is the book's SplitMix64 and every seed comes from
`bookrng.seed_of("project_tensor_engine.<name>")`, printed with its draw order.

Swedish Embedded AB implements solutions for tensor engines that run one set of
kernels on every device a client owns and prove that the devices agree. If your
team needs expertise in compute kernels, backend parity and kernel selection
for machine learning inference, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import hashlib
import math
import struct
from bisect import bisect_right
from functools import lru_cache

import backends as bk
import bookrng
import compute_kernels as ck
import fusion
import kernel_selection as ks
import profiling as pf

NAME = "project_tensor_engine"
U32 = 2.0 ** -24            # unit roundoff of float32 (Volume I, Chapter 15)
U64 = 2.0 ** -53            # unit roundoff of float64
WORD = 4                    # bytes per float32 word


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _n(v):
    return f"{v:,}"


def _name(n):
    return n if "." in n else f"{NAME}.{n}"


def _gen(n):
    return bookrng.generator(_name(n))


def _seed(n):
    s = bookrng.seed_of(_name(n))
    return f"seed {s:#018x} ({s})"


def _yes(flag):
    return "yes" if flag else "no"


def _ceil_div(a, b):
    return -(-a // b)


f32 = bk._f32
fma32 = bk._fma


def digest(values, fmt="f"):
    """SHA-256 over the little-endian binary32 ('f') or binary64 ('d') bytes of the values, row-major; first 16 hex digits."""
    data = struct.pack(f"<{len(values)}{fmt}", *values)
    return hashlib.sha256(data).hexdigest()[:16]


def int_matrix(gen, rows, cols):
    """Small integers -4 .. 4, `gen.integer(9) - 4`, row-major: every product and sum below 2^24 is exact."""
    return [float(gen.integer(9) - 4) for _ in range(rows * cols)]


def real_matrix(gen, rows, cols):
    """float32 entries `2 uniform() - 1` rounded to float32, row-major."""
    return [f32(2.0 * gen.uniform() - 1.0) for _ in range(rows * cols)]


# ============================================================ 1. the tensor


class Tensor:
    """Flat storage plus shape, strides and an offset: element (i, j) is storage[offset + i s0 + j s1]."""

    def __init__(self, storage, shape, strides=None, offset=0):
        self.storage, self.shape, self.offset = storage, tuple(shape), offset
        if strides is None:
            strides, acc = [], 1
            for d in reversed(self.shape):
                strides.append(acc)
                acc *= d
            strides = tuple(reversed(strides))
        self.strides = tuple(strides)

    def offset_of(self, *idx):
        assert all(0 <= i < d for i, d in zip(idx, self.shape))
        return self.offset + sum(i * s for i, s in zip(idx, self.strides))

    def get(self, *idx):
        return self.storage[self.offset_of(*idx)]

    def t(self):
        """The transpose is a view: the same storage with the strides exchanged."""
        return Tensor(self.storage, self.shape[::-1], self.strides[::-1], self.offset)

    def rows(self, lo, hi):
        """Rows lo .. hi - 1 as a view: the offset moves, nothing is copied."""
        return Tensor(self.storage, (hi - lo,) + self.shape[1:], self.strides, self.offset + lo * self.strides[0])

    def flat(self):
        """The elements in row-major order of the shape."""
        m, n = self.shape
        return [self.get(i, j) for i in range(m) for j in range(n)]


def matmul_oracle(a, b):
    """C = A B for two Tensors, every product and sum in float64, a left fold over k. Returns a flat row-major list."""
    (m, k), (k2, n) = a.shape, b.shape
    assert k == k2
    out = []
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a.get(i, p) * b.get(p, j)
            out.append(acc)
    return out


def abs_sums(a, b):
    """S[i, j] = sum_p |a[i, p] b[p, j]|, the scale of every error bound (float64)."""
    (m, k), (_, n) = a.shape, b.shape
    return [math.fsum(abs(a.get(i, p) * b.get(p, j)) for p in range(k)) for i in range(m) for j in range(n)]


# ============================================== 2. memory, kernels, backend


class _CountedWords(list):
    """Device words that count every single-word load and store (a slice is a host read and is not counted)."""

    def __init__(self):
        super().__init__()
        self.owner, self.load, self.store = None, {}, {}    # owner(address) -> buffer name, set by the backend per dispatch

    def arm(self, owner):
        self.owner, self.load, self.store = owner, {}, {}

    def __getitem__(self, i):
        if isinstance(i, int) and self.owner is not None:
            o = self.owner(i)
            self.load[o] = self.load.get(o, 0) + 1
        return super().__getitem__(i)

    def __setitem__(self, i, v):
        if isinstance(i, int) and self.owner is not None:
            o = self.owner(i)
            self.store[o] = self.store.get(o, 0) + 1
        super().__setitem__(i, v)


class CountedMemory(ck.Memory):
    def __init__(self):
        super().__init__()
        self.words = _CountedWords()


class Kernel:
    """A compiled kernel: the arguments it binds (in order), its workgroup size, its shared words and a builder that
    returns the list of phases (a barrier lies between consecutive phases) for a given parameter block."""

    def __init__(self, name, args, wg, shared_words, build, barriers=True):
        self.name, self.args, self.wg = name, tuple(args), wg
        self.shared_words, self.build, self.uses_barriers = shared_words, build, barriers


class CpuBackend:
    """The seven operations of Chapter 9 over the kernel simulator of Chapter 5. A buffer is a named region of one flat
    device memory (32-byte aligned, as the allocator of Chapter 5 places them); `dispatch` runs the kernel's phases with
    the workgroup-and-barrier executor and records what the launch loaded and stored."""

    def __init__(self, config):
        self.config = config
        self.mem = CountedMemory()
        self.sizes, self.count = {}, 0
        self.log = []

    def capabilities(self):
        return dict(max_workgroup=256, shared_bytes=16384, subgroup=0, max_barriers=math.inf)

    def compile(self, kernel):
        c = self.capabilities()
        if kernel.wg > c["max_workgroup"]:
            raise ValueError(f"kernel {kernel.name}: workgroup {kernel.wg} exceeds max_workgroup {c['max_workgroup']}")
        return kernel

    def create_buffer(self, words):
        self.count += 1
        name = f"b{self.count}"
        self.mem.alloc(name, [ck.UNWRITTEN] * words, "read_write")
        self.sizes[name] = words
        return name

    def upload(self, buf, data):
        b = self.mem.buffers[buf]
        assert len(data) <= b.n
        for i, v in enumerate(data):
            list.__setitem__(self.mem.words, b.base + i, float(v))

    def read_back(self, buf, count):
        return self.mem.values(buf)[:count]

    def release(self, buf):
        self.sizes.pop(buf, None)           # the arena only grows: the words stay, the handle is gone

    def dispatch(self, kernel, buffers, params, n_grp):
        bufs = dict(zip(kernel.args, (self.mem.buffers[b] for b in buffers)))
        phases = kernel.build(params)
        bases = sorted((b.base, b.base + b.n, kernel_arg) for kernel_arg, b in bufs.items())
        starts = [s for s, _, _ in bases]

        def owner(addr):
            j = bisect_right(starts, addr) - 1
            if j >= 0 and addr < bases[j][1]:
                return bases[j][2]
            return "outside"

        self.mem.words.arm(owner)
        stats = ck.launch(self.mem, phases, n_grp, kernel.wg, params, bufs, shared_words=kernel.shared_words(params))
        rec = dict(kernel=kernel.name, n_grp=n_grp, wg=kernel.wg, invocations=n_grp * kernel.wg,
                   loads=dict(self.mem.words.load), stores=dict(self.mem.words.store),
                   load_blocks=stats["load"], store_blocks=stats["store"], phases=len(phases))
        self.mem.words.arm(None)
        self.log.append(rec)
        return rec


class DeviceConfig:
    """What differs between two devices that run the same kernels: the workgroup tile, the number of partial
    accumulators the additions of a dot product are dealt over (`lanes`), whether a product followed by an addition is
    contracted into one fused multiply-add, and the workgroup size of the reductions."""

    def __init__(self, name, tile, lanes, contract, reduce_wg, wg_naive):
        self.name, self.tile, self.lanes, self.contract = name, tile, lanes, contract
        self.reduce_wg, self.wg_naive = reduce_wg, wg_naive


DEVICE_A = DeviceConfig("A", tile=4, lanes=1, contract=False, reduce_wg=8, wg_naive=16)
DEVICE_B = DeviceConfig("B", tile=8, lanes=4, contract=True, reduce_wg=16, wg_naive=32)


def mad(p, acc, a, b):
    """acc + a b in float32: two roundings, or one when the device contracts."""
    return fma32(a, b, acc) if p["contract"] else f32(acc + f32(a * b))


def fold_lanes(accs):
    """Combine the partial accumulators left to right (one accumulator: the value itself, no operation)."""
    s = accs[0]
    for x in accs[1:]:
        s = f32(s + x)
    return s


def _a_at(p, i, kk):
    return p["oa"] + i * p["sa0"] + kk * p["sa1"]


def _b_at(p, kk, j):
    return p["ob"] + kk * p["sb0"] + j * p["sb1"]


# ----------------------------------------------------------------- matmul kernels


def matmul_naive(wg):
    """One output per invocation (Chapter 6): g = w wg + l, i = g div n, j = g mod n."""
    def phase(inv, p):
        m, n, k = p["m"], p["n"], p["k"]
        g = inv.gid_x
        if g >= m * n:
            inv.ret()
        i, j = divmod(g, n)
        accs = [0.0] * p["lanes"]
        for kk in range(k):
            ln = kk % p["lanes"]
            accs[ln] = mad(p, accs[ln], inv.load("A", _a_at(p, i, kk)), inv.load("B", _b_at(p, kk, j)))
        inv.store("C", g, fold_lanes(accs))
    return Kernel("matmul_naive", ("A", "B", "C"), wg, lambda p: 0, lambda p: [phase], barriers=False)


def gelu_epilogue(u):
    """GELU of Chapter 8 (tanh form), every operation rounded to float32 one at a time."""
    return fusion._gelu_f32(u)


def matmul_blocked(q, r, bk_, guard_k=True, epilogue=False, name=None):
    """The register-blocked kernel of Chapter 6: a workgroup of q x q invocations computes a tile of side T = q r, each
    invocation an r x r block in registers, the k loop in steps of bk_ with a barrier between the load and the compute
    phase of every step. With r = 1 and bk_ = q it is the tiled kernel of Chapter 6. `guard_k=False` removes the
    bound check of the k edge from the two tile loads (the planted defect of milestone 5). `epilogue=True` adds the
    bias and GELU of Chapter 8 before the store."""
    big, n_wg = q * r, q * q
    args = ("A", "B", "bias", "C") if epilogue else ("A", "B", "C")

    def build(p):
        m, n, k, lanes = p["m"], p["n"], p["k"], p["lanes"]
        steps = _ceil_div(k, bk_)
        regs = {}

        def load_phase(c):
            def phase(inv, p_):
                wy, wx = divmod(inv.wx, p_["nx"])
                for e in range(_ceil_div(big * bk_, n_wg)):
                    idx = inv.l + e * n_wg
                    if idx >= big * bk_:
                        continue
                    row, kk = divmod(idx, bk_)
                    gr, gk = wy * big + row, c * bk_ + kk
                    v = 0.0
                    if gr < m and (gk < k or not guard_k):
                        v = inv.load("A", _a_at(p_, gr, gk))
                    inv.shared_store(idx, v)
                    kk2, col = divmod(idx, big)
                    gk2, gc = c * bk_ + kk2, wx * big + col
                    v = 0.0
                    if gc < n and (gk2 < k or not guard_k):
                        v = inv.load("B", _b_at(p_, gk2, gc))
                    inv.shared_store(big * bk_ + idx, v)
            return phase

        def compute_phase(c, last):
            def phase(inv, p_):
                wy, wx = divmod(inv.wx, p_["nx"])
                ty, tx = divmod(inv.l, q)
                acc = regs.setdefault((inv.wx, inv.l), [[[0.0] * r for _ in range(r)] for _ in range(lanes)])
                for kk in range(bk_):
                    ln = (c * bk_ + kk) % lanes
                    a_reg = [inv.shared_load((ty * r + i) * bk_ + kk) for i in range(r)]
                    b_reg = [inv.shared_load(big * bk_ + kk * big + tx * r + j) for j in range(r)]
                    for i in range(r):
                        for j in range(r):
                            acc[ln][i][j] = mad(p_, acc[ln][i][j], a_reg[i], b_reg[j])
                if last:
                    for i in range(r):
                        for j in range(r):
                            gr, gc = wy * big + ty * r + i, wx * big + tx * r + j
                            if gr < m and gc < n:
                                v = fold_lanes([acc[ln][i][j] for ln in range(lanes)])
                                if epilogue:
                                    v = gelu_epilogue(f32(v + inv.load("bias", gc)))
                                inv.store("C", gr * n + gc, v)
            return phase

        phases = []
        for c in range(steps):
            phases.append(load_phase(c))
            phases.append(compute_phase(c, c == steps - 1))
        return phases

    nm = name or ("matmul_tiled" if r == 1 else "matmul_blocked") + ("_epilogue" if epilogue else "") + ("" if guard_k else "_nok")
    return Kernel(nm, args, n_wg, lambda p: 2 * big * bk_, build)


# ------------------------------------------------------- elementwise kernels


def k_bias_add(wg):
    def phase(inv, p):
        g = inv.gid_x
        if g >= p["total"]:
            inv.ret()
        inv.store("u", g, f32(inv.load("z", g) + inv.load("bias", g % p["n"])))
    return Kernel("bias_add", ("z", "bias", "u"), wg, lambda p: 0, lambda p: [phase], barriers=False)


def k_gelu(wg):
    def phase(inv, p):
        g = inv.gid_x
        if g >= p["total"]:
            inv.ret()
        inv.store("gl", g, gelu_epilogue(inv.load("u", g)))
    return Kernel("gelu", ("u", "gl"), wg, lambda p: 0, lambda p: [phase], barriers=False)


# ------------------------------------------------------------- reductions


def _tree_phases(wg, store):
    """The tree of Chapter 7 as phases: s[l] holds each invocation's value; steps of stride wg/2 .. 1; then the store."""
    phases = []
    stride = wg // 2
    while stride >= 1:
        def step(inv, p, s=stride):
            if inv.l < s:
                inv.shared_store(inv.l, f32(inv.shared_load(inv.l) + inv.shared_load(inv.l + s)))
        phases.append(step)
        stride //= 2

    def fin(inv, p):
        if inv.l == 0:
            store(inv, inv.shared_load(0))
    phases.append(fin)
    return phases


def reduce_rows(wg):
    """Row sums: workgroup w sums row w of an [m, n] array, each invocation adding the elements l, l + wg, ... of the row
    (the grid-stride loop of Chapter 7 with the row as the grid), then the tree."""
    def build(p):
        def load(inv, p_):
            row, acc, j = inv.wx, 0.0, inv.l
            while j < p_["n"]:
                acc = f32(acc + inv.load("x", row * p_["n"] + j))
                j += wg
            inv.shared_store(inv.l, acc)
        return [load] + _tree_phases(wg, lambda inv, v: inv.store("out", inv.wx, v))
    return Kernel("reduce_rows", ("x", "out"), wg, lambda p: wg, build)


def reduce_pass(wg):
    """One pass of the two-pass sum: a grid-stride loop over N values, then the tree, one partial per workgroup."""
    def build(p):
        def load(inv, p_):
            acc, i = 0.0, inv.wx * wg + inv.l
            while i < p_["N"]:
                acc = f32(acc + inv.load("x", i))
                i += p_["n_grp"] * wg
            inv.shared_store(inv.l, acc)
        return [load] + _tree_phases(wg, lambda inv, v: inv.store("out", inv.wx, v))
    return Kernel("reduce_pass", ("x", "out"), wg, lambda p: wg, build)


def chain_depth_pass(n_in, n_grp, wg):
    """Rounding steps on the longest path of one pass: the loop of an invocation adds ceil(N / (n_grp wg)) numbers to a
    zero accumulator (the first addition is exact), then log2 wg levels of the tree."""
    n_el = _ceil_div(n_in, n_grp * wg)
    return max(n_el - 1, 0) + int(math.log2(wg))


# ------------------------------------------------------ quantized product


def matmul_quantized(wg, factored=True):
    """C = A W with W quantized group-wise along k (Chapter 12): codes Q[j * k + t] and scales S[j * (k / G) + g] for column j.
    Factored: the group's dot product of activations with integer codes, then one multiplication by the scale per group.
    Dequantize first: each weight is scale * code before the multiply-add. One output per invocation."""
    def phase(inv, p):
        m, n, k, G = p["m"], p["n"], p["k"], p["G"]
        g = inv.gid_x
        if g >= m * n:
            inv.ret()
        i, j = divmod(g, n)
        total = 0.0
        for grp in range(k // G):
            scale = inv.load("S", j * (k // G) + grp)
            inner = 0.0
            for t in range(G):
                kk = grp * G + t
                a, q = inv.load("A", i * k + kk), inv.load("Q", j * k + kk)
                inner = mad(p, inner, a, q) if factored else mad(p, inner, a, f32(scale * q))
            total = mad(p, total, scale, inner) if factored else f32(total + inner)
        inv.store("C", g, total)
    return Kernel("matmul_q" + ("" if factored else "_dequant_first"), ("A", "Q", "S", "C"), wg, lambda p: 0,
                  lambda p: [phase], barriers=False)


def quantize_groups(w, G, nbits):
    """Symmetric group-wise quantization of Chapter 12 with a float32 scale: (scales, codes)."""
    assert len(w) % G == 0 and 2 <= nbits <= 8
    qmax = 2 ** (nbits - 1) - 1
    scales, codes = [], []
    for g in range(len(w) // G):
        grp = w[g * G:(g + 1) * G]
        amax = max(abs(x) for x in grp)
        s = f32(amax / qmax) if amax > 0 else 0.0
        scales.append(s)
        codes.extend(0 if s == 0 else max(-qmax, min(qmax, round(x / s))) for x in grp)
    return scales, codes


# ------------------------------------------------------------- the engine


class Engine:
    """The host side: upload tensors, pick the grid, dispatch, read back. `kind` names the matmul kernel."""

    def __init__(self, config=DEVICE_A):
        self.config = config
        self.be = CpuBackend(config)

    def _params(self, m, n, k, a, b, extra=None):
        p = dict(m=m, n=n, k=k, oa=a.offset, sa0=a.strides[0], sa1=a.strides[1],
                 ob=b.offset, sb0=b.strides[0], sb1=b.strides[1], lanes=self.config.lanes,
                 contract=self.config.contract, nx=1)
        p.update(extra or {})
        return p

    def matmul(self, kind, a, b, guard_k=True, tile=None, bias=None, epilogue=False):
        """kind: 'naive', 'tiled' (r = 1, T = q), 'blocked' (given as (q, r, bk)). Returns (flat C, dispatch record)."""
        (m, k), (_, n) = a.shape, b.shape
        be = self.be
        ba, bb = be.create_buffer(len(a.storage)), be.create_buffer(len(b.storage))
        bc = be.create_buffer(m * n)
        be.upload(ba, a.storage)
        be.upload(bb, b.storage)
        if kind == "naive":
            kern = be.compile(matmul_naive(self.config.wg_naive))
            n_grp, p = _ceil_div(m * n, kern.wg), self._params(m, n, k, a, b)
        else:
            q, r, bk_ = (tile, 1, tile) if kind == "tiled" else kind
            kern = be.compile(matmul_blocked(q, r, bk_, guard_k=guard_k, epilogue=epilogue))
            big = q * r
            nx, ny = _ceil_div(n, big), _ceil_div(m, big)
            n_grp, p = nx * ny, self._params(m, n, k, a, b, dict(nx=nx))
        bufs = [ba, bb, bc]
        if epilogue:
            bbias = be.create_buffer(n)
            be.upload(bbias, bias)
            bufs = [ba, bb, bbias, bc]
        rec = be.dispatch(kern, bufs, p, n_grp)
        out = be.read_back(bc, m * n)
        for h in bufs:
            be.release(h)
        return out, rec


# =========================================================== 3. the tolerance


def gamma(k, u):
    """The error factor of k roundings, k u / (1 - k u) (Chapter 9)."""
    assert k * u < 1
    return k * u / (1.0 - k * u)


def k_rd_matmul(k, lanes):
    """Roundings on the longest chain of a dot product of k terms whose additions are dealt over `lanes` accumulators
    (term t goes to accumulator t mod lanes, each a left fold) and combined left to right: ceil(k / lanes) for the
    longest accumulator plus lanes - 1 for the combination."""
    return _ceil_div(k, lanes) + lanes - 1


def error_report(got, ref, s, k_rd, k_oracle):
    """(largest |got - ref| / (u S), largest error / tolerance, number outside tolerance, number not bit-identical to ref's float32)."""
    worst_u, worst_t, bad = 0.0, 0.0, 0
    for g_, r_, s_ in zip(got, ref, s):
        e = abs(g_ - r_)
        tol = (gamma(k_rd, U32) + gamma(k_oracle, U64)) * s_
        worst_u = max(worst_u, e / (U32 * s_)) if s_ > 0 else worst_u
        ratio = e / tol if tol > 0 else (0.0 if e == 0 else math.inf)
        worst_t = max(worst_t, ratio)
        bad += ratio > 1.0 or not math.isfinite(g_)
    return worst_u, worst_t, bad


# ============================================================ M1: the tensor


def tensor_views():
    store = [float(v) for v in range(12)]
    a = Tensor(store, (3, 4))
    at = a.t()
    r = a.rows(1, 3)
    rt = r.t()
    rows = []
    for label, t in (("A", a), ("A transposed", at), ("A rows 1 to 2", r), ("A rows 1 to 2, transposed", rt)):
        idx = (1, 2) if t.shape[0] > 1 and t.shape[1] > 2 else (0, 1)
        rows.append((label, list(t.shape), list(t.strides), t.offset, f"{idx}", t.offset_of(*idx), int(t.get(*idx)),
                     _yes(t.flat() == [t.get(i, j) for i in range(t.shape[0]) for j in range(t.shape[1])])))
    # A view of A^T in a product equals the product with the transpose copied out.
    gen = _gen("tensor_views")
    x = Tensor(real_matrix(gen, 4, 5), (4, 5))
    w = Tensor(real_matrix(gen, 3, 5), (3, 5))              # stored as [n, k]; the product uses its transpose
    via_view = matmul_oracle(x, w.t())
    wt_copy = Tensor([w.get(j, p) for p in range(5) for j in range(3)], (5, 3))
    via_copy = matmul_oracle(x, wt_copy)
    return (f"Storage = [0, 1, ..., 11] (twelve float32 words), A = shape [3, 4] with the row-major strides (4, 1) and offset 0. A transpose is the same storage with the "
            f"shape and the strides exchanged; a row slice moves the offset and keeps the strides. Element (i, j) of a view is storage[offset + i s0 + j s1]; "
            f"the table reads the element at the index shown.\n\n"
            + _table(["view", "shape", "strides", "offset", "index", "storage position", "element", "flat() equals element by element"], rows, "lllrlrrl")
            + f"\n\nA view in a product. {_seed('tensor_views')}. Draw order: the 20 entries of X (shape [4, 5]) then the 15 entries of W (shape [3, 5], stored "
            f"[n, k]), each `2 uniform() - 1` rounded to float32. The product X W^T taken through the transposed view of W has the digest "
            f"{digest(via_view, 'd')} (binary64, row-major, 12 values); the same product with W^T copied out first has "
            f"{digest(via_copy, 'd')}; identical: {_yes(via_view == via_copy)}.")


# cases of the oracle and the exact device tests: (m, n, k)
EXACT_SHAPES = [(2, 2, 3), (5, 7, 3), (8, 8, 8), (13, 11, 9), (17, 19, 23)]


def _exact_case(gen, m, n, k):
    return Tensor(int_matrix(gen, m, k), (m, k)), Tensor(int_matrix(gen, k, n), (k, n))


def oracle_cases():
    gen = _gen("exact_cases")
    cases = [(m, n, k, *_exact_case(gen, m, n, k)) for (m, n, k) in EXACT_SHAPES]
    m, n, k, a, b = cases[0]
    c = matmul_oracle(a, b)
    rows = []
    for (m_, n_, k_, a_, b_) in cases:
        c_ = matmul_oracle(a_, b_)
        rows.append((f"{m_} x {n_} x {k_}", _n(2 * m_ * n_ * k_), digest(a_.flat(), "d"), digest(b_.flat(), "d"), digest(c_, "d")))
    return (f"Integer matrices for the oracle and the exact device tests. {_seed('exact_cases')}. Draw order: the shapes (m, n, k) in the order of the table; for each, the "
            f"m k entries of A row-major, then the k n entries of B row-major, each `integer(9) - 4` (so in -4 .. 4). The smallest case, m = {m}, n = {n}, k = {k}: "
            f"A = {[[int(x) for x in a.flat()[i * k:(i + 1) * k]] for i in range(m)]}, B = {[[int(x) for x in b.flat()[i * n:(i + 1) * n]] for i in range(k)]}, "
            f"so C = A B = {[[int(x) for x in c[i * n:(i + 1) * n]] for i in range(m)]}. 'FLOPs' is 2 m n k. Digests are SHA-256 over binary64 little-endian "
            f"bytes, row-major, first 16 hex digits.\n\n"
            + _table(["m x n x k", "FLOPs", "digest of A", "digest of B", "digest of C (oracle)"], rows, "lrlll"))


# ============================================================ M2: the CPU backend


def backend_exact():
    gen = _gen("exact_cases")
    eng = Engine(DEVICE_A)
    rows = []
    small = None
    for (m, n, k) in EXACT_SHAPES:
        a, b = _exact_case(gen, m, n, k)
        ref = matmul_oracle(a, b)
        out, rec = eng.matmul("naive", a, b)
        if small is None:
            small = (m, n, k, out, rec)
        rows.append((f"{m} x {n} x {k}", rec["n_grp"], rec["invocations"], _n(rec["loads"].get("A", 0)), _n(rec["loads"].get("B", 0)),
                     _n(rec["stores"].get("C", 0)), rec["load_blocks"], rec["store_blocks"], _n(2 * m * n * k),
                     _yes(out == ref)))
    return ("The naive kernel of Chapter 6 (one output per invocation, workgroups of 16, the guard `g >= m n` returns) on the CPU backend, device A "
            "(left fold, no contraction), on the cases of `project_tensor_engine.oracle_cases` (same seed, same draws). Buffers are allocated in the order A, B, C, each starting at a "
            "multiple of 8 words (32 bytes) with -777 in the padding words; C starts as 1000. 'loads' and 'stores' count single words as the executor saw the kernel "
            "touch memory; 'blocks' are the 32-byte transactions of Chapter 5 (32-lane subgroups, one instruction per j-th access, one transaction per distinct block). "
            "'Expected' is 2 m n k loads and m n stores.\n\n"
            + _table(["m x n x k", "workgroups", "invocations", "loads of A", "loads of B", "stores of C", "load blocks", "store blocks",
                      "2 m n k", "C equals the oracle"], rows))


# ============================================================ M3: tiled and blocked kernels

TRAFFIC_SHAPES = [(13, 10, 7), (20, 18, 9)]


def tiled_loads(m, n, k, big):
    """k (m ceil(n / T) + n ceil(m / T)): every element of A once per column tile, every element of B once per row tile."""
    return k * (m * _ceil_div(n, big) + n * _ceil_div(m, big))


KERNELS_M3 = [("naive, 16 per workgroup", "naive", None, 1), ("tiled, T = 4 (16 invocations)", "tiled", 4, 4),
              ("blocked, q = 4, r = 2, T = 8 (16 invocations)", (4, 2, 4), None, 8)]


def tiled_traffic():
    rows = []
    for (m, n, k) in TRAFFIC_SHAPES:
        gen = _gen(f"traffic.{m}x{n}x{k}")
        a, b = Tensor(int_matrix(gen, m, k), (m, k)), Tensor(int_matrix(gen, k, n), (k, n))
        ref = matmul_oracle(a, b)
        for label, kind, tile, big in KERNELS_M3:
            out, rec = Engine(DEVICE_A).matmul(kind, a, b, tile=tile)
            loads = rec["loads"].get("A", 0) + rec["loads"].get("B", 0)
            formula = 2 * m * n * k if kind == "naive" else tiled_loads(m, n, k, big)
            assert loads == formula and out == ref
            rows.append((f"{m} x {n} x {k}", label, rec["n_grp"], rec["invocations"], _n(rec["loads"].get("A", 0)),
                         _n(rec["loads"].get("B", 0)), _n(loads), _n(formula), _n(rec["stores"].get("C", 0)), rec["load_blocks"],
                         rec["store_blocks"], _yes(out == ref)))
    return ("Ragged products on the CPU backend, device A, integer matrices (`integer(9) - 4`; "
            + "; ".join(f"{_seed(f'traffic.{m}x{n}x{k}')} for {m} x {n} x {k}" for (m, n, k) in TRAFFIC_SHAPES)
            + "; draw order: the m k entries of A, then the k n entries of B, row-major). "
            "The tiled kernel is the blocked kernel with r = 1 and step depth bk = T; the blocked kernel uses q = 4 invocations per side, r = 2, step depth 4. Elements "
            "outside the matrices are not loaded (the guards sit around the loads). Expected loads: naive 2 m n k; tiled and blocked k (m ceil(n / T) + n ceil(m / T)), "
            "T the side of the workgroup tile. Stores are m n for every kernel.\n\n"
            + _table(["m x n x k", "kernel", "workgroups", "invocations", "loads of A", "loads of B", "loads in all", "formula", "stores of C",
                      "load blocks", "store blocks", "C equals the oracle"], rows, "llrrrrrrrrrl"))


def summation_order_identity():
    m, n, k = 13, 10, 37
    gen = _gen("order_identity")
    a, b = Tensor(real_matrix(gen, m, k), (m, k)), Tensor(real_matrix(gen, k, n), (k, n))
    rows = []
    hashes = []
    for label, kind, tile, _ in KERNELS_M3:
        out, rec = Engine(DEVICE_A).matmul(kind, a, b, tile=tile)
        hashes.append(digest(out, "f"))
        rows.append((label, hashes[-1], rec["loads"]["A"] + rec["loads"]["B"]))
    ref = matmul_oracle(a, b)
    s = abs_sums(a, b)
    out0, _ = Engine(DEVICE_A).matmul("naive", a, b)
    wu, wt, bad = error_report(out0, ref, s, k_rd_matmul(k, 1), k)
    return (f"Real-valued ragged product, m = {m}, n = {n}, k = {k} (k is not a multiple of any step depth used). {_seed('order_identity')}. Draw order: the {m * k} entries of A, then the "
            f"{k * n} entries of B, row-major, each `2 uniform() - 1` rounded to float32. Device A: every multiply-add is two float32 operations (the product rounded, then the sum), "
            "each output a left fold over k. Digest: SHA-256 of the binary32 little-endian bytes of C, row-major, first 16 hex digits.\n\n"
            + _table(["kernel", "digest of C", "global loads"], rows, "lrr")
            + f"\n\nAll three digests equal: {_yes(len(set(hashes)) == 1)}. Tiling changes where the numbers are read from, not the order in which each output adds its terms. "
            f"Against the float64 oracle: largest |C - oracle| / (u S) = {wu:.3f}, derived tolerance {gamma(k, U32) / U32:.1f} u S (k_rd = {k}); outside tolerance: {bad}.")


# ============================================================ M4: reductions and the epilogue


def _run_pass(eng, kern, x_words, n_out, params, n_grp):
    be = eng.be
    bx, bo = be.create_buffer(len(x_words)), be.create_buffer(n_out)
    be.upload(bx, x_words)
    rec = be.dispatch(be.compile(kern), [bx, bo], params, n_grp)
    out = be.read_back(bo, n_out)
    be.release(bx)
    be.release(bo)
    return out, rec


def row_sums(cfg, x, m, n):
    eng = Engine(cfg)
    out, rec = _run_pass(eng, reduce_rows(cfg.reduce_wg), x, m, dict(n=n), m)
    return out, rec


def global_sum(cfg, x, n_grp):
    """Two dispatches: pass 1 over n_grp workgroups leaves one partial each; pass 2 is one workgroup over the partials."""
    eng = Engine(cfg)
    wg, N = cfg.reduce_wg, len(x)
    part, rec1 = _run_pass(eng, reduce_pass(wg), x, n_grp, dict(N=N, n_grp=n_grp), n_grp)
    tot, rec2 = _run_pass(eng, reduce_pass(wg), part, 1, dict(N=n_grp, n_grp=1), 1)
    k_rd = chain_depth_pass(N, n_grp, wg) + chain_depth_pass(n_grp, 1, wg)
    return tot[0], (rec1, rec2), k_rd


RED_M, RED_N, RED_TOTAL, RED_GRP = 6, 37, 1000, 5


def reductions():
    gen = _gen("reductions")
    x = real_matrix(gen, RED_M, RED_N)
    y = real_matrix(gen, 1, RED_TOTAL)
    rows = []
    for cfg in (DEVICE_A, DEVICE_B):
        out, rec = row_sums(cfg, x, RED_M, RED_N)
        wg = cfg.reduce_wg
        k_rd = chain_depth_pass(RED_N, 1, wg)
        worst = 0.0
        bad = 0
        for i in range(RED_M):
            row = x[i * RED_N:(i + 1) * RED_N]
            ref, s = 0.0, math.fsum(abs(v) for v in row)
            for v in row:
                ref += v
            tol = (gamma(k_rd, U32) + gamma(RED_N - 1, U64)) * s
            worst = max(worst, abs(out[i] - ref) / (U32 * s))
            bad += abs(out[i] - ref) > tol
        rows.append((f"row sums, device {cfg.name}", f"{wg}", f"{RED_M} x {RED_N}", 1, rec["loads"]["x"], rec["stores"]["out"],
                     digest(out, "f"), k_rd, f"{gamma(k_rd, U32) / U32:.2f}", f"{worst:.3f}", bad))
        tot, recs, k_rd2 = global_sum(cfg, y, RED_GRP)
        ref, s = 0.0, math.fsum(abs(v) for v in y)
        for v in y:
            ref += v
        tol = (gamma(k_rd2, U32) + gamma(RED_TOTAL - 1, U64)) * s
        loads = sum(r["loads"]["x"] for r in recs)
        stores = sum(r["stores"]["out"] for r in recs)
        rows.append((f"global sum, device {cfg.name}", f"{wg}", f"{RED_TOTAL}", 2, loads, stores,
                     f"{bk._bits(tot):#010x}", k_rd2, f"{gamma(k_rd2, U32) / U32:.2f}", f"{abs(tot - ref) / (U32 * s):.3f}",
                     int(abs(tot - ref) > tol)))
    # bytes of the two-pass sum against the one-pass minimum
    tot, recs, _ = global_sum(DEVICE_A, y, RED_GRP)
    words = sum(sum(r["loads"].values()) + sum(r["stores"].values()) for r in recs)
    return (f"Row sums of a [{RED_M}, {RED_N}] array X (one workgroup per row, each invocation adds the elements l, l + wg, ... of its row, then the tree of Chapter 7) and the "
            f"two-pass sum of {RED_TOTAL} numbers Y (pass 1 over {RED_GRP} workgroups with a grid-stride loop, pass 2 one workgroup over the {RED_GRP} partials). {_seed('reductions')}. "
            f"Draw order: the {RED_M * RED_N} entries of X row-major, then the {RED_TOTAL} entries of Y, each `2 uniform() - 1` rounded to float32. Device A reduces with "
            f"workgroups of 8, device B with workgroups of 16. 'k_rd' counts the roundings on the longest chain (an invocation's loop adds ceil(N / (n_grp wg)) numbers to a zero "
            f"accumulator, the first addition being exact, then log2 wg levels of the tree; the global sum adds the chains of its two passes). 'Tolerance' is "
            f"gamma(k_rd, u) in units of u S with u = 2^-24, S = sum |x|, plus the float64 oracle's own gamma(N - 1, 2^-53) S (a left fold in float64 is the oracle); 'error' is the "
            f"largest |sum - oracle| / (u S). Digests of the row sums are SHA-256 over binary32 little-endian bytes (first 16 hex digits); a single total is printed as its 32 bits.\n\n"
            + _table(["reduction", "wg", "input", "dispatches", "loads", "stores", "digest or bits", "k_rd", "tolerance (u S)", "error (u S)", "outside tolerance"],
                     rows, "llrrrrlrrrr")
            + f"\n\nThe two-pass sum of device A moves {words} words ({_n(WORD * words)} bytes) through memory for {RED_TOTAL} numbers: "
            f"{_n(WORD * RED_TOTAL)} bytes is the minimum, so the partials cost {100 * (words - RED_TOTAL) / RED_TOTAL:.1f} percent more.")


EPI_SHAPE = (13, 10, 9)


def epilogue_bytes():
    m, n, k = EPI_SHAPE
    gen = _gen("epilogue")
    a, b = Tensor(real_matrix(gen, m, k), (m, k)), Tensor(real_matrix(gen, k, n), (k, n))
    bias = [f32(0.5 * gen.normal()) for _ in range(n)]
    cfg = DEVICE_A
    eng = Engine(cfg)
    be = eng.be
    # unfused: matmul, bias_add, gelu
    z, rec_mm = eng.matmul("tiled", a, b, tile=4)
    bz, bb, bu = be.create_buffer(m * n), be.create_buffer(n), be.create_buffer(m * n)
    be.upload(bz, z)
    be.upload(bb, bias)
    r2 = be.dispatch(be.compile(k_bias_add(16)), [bz, bb, bu], dict(total=m * n, n=n), _ceil_div(m * n, 16))
    u = be.read_back(bu, m * n)
    bg = be.create_buffer(m * n)
    r3 = be.dispatch(be.compile(k_gelu(16)), [bu, bg], dict(total=m * n), _ceil_div(m * n, 16))
    out_un = be.read_back(bg, m * n)
    # fused: the epilogue inside the tiled matmul
    out_fu, rec_fu = Engine(cfg).matmul("tiled", a, b, tile=4, bias=bias, epilogue=True)
    want = [fusion._gelu_f32(f32(v + bias[i % n])) for i, v in enumerate(z)]

    def words(rs):
        return sum(sum(r["loads"].values()) + sum(r["stores"].values()) for r in rs)

    def blocks(rs):
        return sum(r["load_blocks"] + r["store_blocks"] for r in rs)

    rows = [("matmul, bias_add, gelu", 3, words([rec_mm, r2, r3]), _n(WORD * words([rec_mm, r2, r3])), blocks([rec_mm, r2, r3]), digest(out_un, "f")),
            ("matmul with the epilogue", 1, words([rec_fu]), _n(WORD * words([rec_fu])), blocks([rec_fu]), digest(out_fu, "f"))]
    saved = words([rec_mm, r2, r3]) - words([rec_fu])
    assert saved == 4 * m * n, (saved, m, n)
    return (f"The epilogue of Chapter 8, bias then GELU (the nine operations of Chapter 8 in the order listed there, every one rounded to float32, the constants sqrt(2 / pi) and 0.044715 rounded to float32, tanh the library tanh of the float32 argument rounded to float32) on the output of the tiled kernel (T = 4), "
            f"m = {m}, n = {n}, k = {k}. {_seed('epilogue')}. Draw order: the {m * k} entries of A, the {k * n} entries of B (each `2 uniform() - 1` rounded to float32), then the {n} "
            f"bias entries `0.5 normal()` rounded to float32. 'Words' is every single-word load and store the executor counted, over all dispatches; bytes are 4 times that; "
            f"'blocks' are 32-byte transactions. The two separate kernels are elementwise, one element per invocation, workgroups of 16 and ceil(m n / 16) workgroups, the guard `g >= m n` returns: "
            f"bias_add stores u[g] = f32(z[g] + bias[g mod n]) and gelu stores the GELU of u[g]; the matmul is the tiled kernel's own launch. Every buffer starts on a 32-byte boundary, so the order of allocation does not change a count. "
            f"Digest: SHA-256 over binary32 little-endian bytes of the result, first 16 hex digits.\n\n"
            + _table(["design", "dispatches", "words through memory", "bytes", "transaction blocks", "digest of the result"], rows, "lrrrrl")
            + f"\n\nWords saved: {saved}, which is 4 m n: the intermediate z and the intermediate u are each stored once and loaded once. The results are bit-identical: "
            f"{_yes(out_un == out_fu)}; both equal the direct evaluation of the formula: {_yes(out_fu == want)}.")


# ============================================================ M5: parity

PARITY_SHAPES = [(13, 10, 7), (21, 18, 100), (33, 17, 200)]


def parity_run(kind, cfg, a, b, guard_k=True):
    tile = cfg.tile if kind == "tiled" else None
    return Engine(cfg).matmul(kind, a, b, guard_k=guard_k, tile=tile)


@lru_cache(maxsize=None)
def _parity_case(m, n, k, kind):
    """Both devices on one seeded case: ({device: (flat C, k_rd)}, float64 oracle, S)."""
    gen = _gen(f"parity.{m}x{n}x{k}")
    a, b = Tensor(real_matrix(gen, m, k), (m, k)), Tensor(real_matrix(gen, k, n), (k, n))
    outs = {cfg.name: (parity_run(kind, cfg, a, b)[0], k_rd_matmul(k, cfg.lanes)) for cfg in (DEVICE_A, DEVICE_B)}
    return outs, matmul_oracle(a, b), abs_sums(a, b)


def parity_devices():
    rows_dev, rows = [], []
    for cfg in (DEVICE_A, DEVICE_B):
        rows_dev.append((cfg.name, f"{cfg.tile} x {cfg.tile} ({cfg.tile * cfg.tile} invocations)", cfg.wg_naive, cfg.lanes,
                         "fused multiply-add" if cfg.contract else "product, then sum", cfg.reduce_wg))
    for (m, n, k) in PARITY_SHAPES:
        for kind in ("naive", "tiled"):
            res, ref, s = _parity_case(m, n, k, kind)
            outs = {}
            for cfg in (DEVICE_A, DEVICE_B):
                out, kr = res[cfg.name]
                outs[cfg.name] = out
                wu, wt, bad = error_report(out, ref, s, kr, k)
                rows.append((f"{m} x {n} x {k}", kind, cfg.name, kr, f"{(gamma(kr, U32) + gamma(k, U64)) / U32:.2f}", f"{wu:.3f}",
                             f"{wt:.4f}", bad, digest(out, "f")))
            ka, kb = k_rd_matmul(k, DEVICE_A.lanes), k_rd_matmul(k, DEVICE_B.lanes)
            pair = (gamma(ka, U32) + gamma(kb, U32))
            same = sum(1 for x, y in zip(outs["A"], outs["B"]) if x == y)
            worst = max(abs(x - y) / (U32 * s_) for x, y, s_ in zip(outs["A"], outs["B"], s))
            wr = max(abs(x - y) / (pair * s_) for x, y, s_ in zip(outs["A"], outs["B"], s))
            bad = sum(abs(x - y) > pair * s_ for x, y, s_ in zip(outs["A"], outs["B"], s))
            rows.append((f"{m} x {n} x {k}", kind, "A vs B", f"{ka} + {kb}", f"{pair / U32:.2f}", f"{worst:.3f}", f"{wr:.4f}", bad,
                         f"{same} of {m * n} equal"))
    return ("Two device configurations of the CPU backend (the same kernels, different compilers).\n\n"
            + _table(["device", "tile of the tiled kernel", "naive workgroup", "accumulators (lanes)", "multiply-add", "reduction workgroup"], rows_dev, "lllllr")
            + "\n\nThe lane rule: term t of a dot product goes to accumulator t mod lanes, each accumulator is a left fold, and the accumulators are added left to right at the end; "
            "the longest chain has k_rd = ceil(k / lanes) + lanes - 1 roundings. The tolerance of a device against the float64 oracle (a left fold in float64) is "
            "(gamma(k_rd, 2^-24) + gamma(k, 2^-53)) S per output, gamma(j, u) = j u / (1 - j u), S = sum |a b|; the tolerance of two devices against each other is "
            "(gamma(k_rd,A, 2^-24) + gamma(k_rd,B, 2^-24)) S. Errors are in units of u S (u = 2^-24); 'worst ratio' is the largest error over the tolerance of its own entry. "
            + "; ".join(f"{_seed(f'parity.{m}x{n}x{k}')} for {m} x {n} x {k}" for (m, n, k) in PARITY_SHAPES)
            + ". Draw order: the m k entries of A, then the k n entries of B, row-major, each `2 uniform() - 1` rounded to float32. Digests: binary32 little-endian, first 16 hex digits.\n\n"
            + _table(["m x n x k", "kernel", "device", "k_rd", "tolerance (u S)", "largest error (u S)", "worst ratio", "outside tolerance", "digest or agreement"], rows, "lllrrrrrl"))


BROKEN_SHAPES = [(8, 8, 8), (12, 8, 16), (13, 10, 7), (9, 6, 5), (16, 16, 20)]


def parity_check_shape(kind, cfg, shape, guard_k=True):
    """The parity test of the project for one candidate and one shape: against the float64 oracle with the derived tolerance.
    Returns (outputs outside tolerance or not finite, largest error over tolerance, fault text or None)."""
    m, n, k = shape
    gen = _gen(f"broken.{m}x{n}x{k}")
    a, b = Tensor(real_matrix(gen, m, k), (m, k)), Tensor(real_matrix(gen, k, n), (k, n))
    ref, s = matmul_oracle(a, b), abs_sums(a, b)
    try:
        out, _ = parity_run(kind, cfg, a, b, guard_k)
    except ck.Fault as e:
        return m * n, math.inf, str(e)
    wu, wt, bad = error_report(out, ref, s, k_rd_matmul(k, cfg.lanes), k)
    return bad, wt, None


def broken_kernel():
    rows = []
    for (m, n, k) in BROKEN_SHAPES:
        row = [f"{m} x {n} x {k}", _yes(k % 4 == 0)]
        for guard in (True, False):
            bad, wt, fault = parity_check_shape("tiled", DEVICE_A, (m, n, k), guard)
            verdict = "PASS" if bad == 0 else ("FAIL (fault)" if fault else "FAIL")
            row += [f"{bad} of {m * n}", "above 1e6" if wt > 1e6 else f"{wt:.3f}", verdict]
        rows.append(tuple(row))
    return ("The tiled kernel (T = 4, step depth 4) with and without the bound check of the k edge on its two tile loads (`ka < K` and `kb < K` removed; the checks on rows and "
            "columns remain), tested on device A against the float64 oracle with the derived tolerance (k_rd = k). Real-valued inputs, `2 uniform() - 1` rounded to float32, "
            "the m k entries of A then the k n entries of B, row-major, from the generator of the name `project_tensor_engine.broken.<m>x<n>x<k>` "
            + "; ".join(f"{_seed(f'broken.{m}x{n}x{k}')} for {m} x {n} x {k}" for (m, n, k) in BROKEN_SHAPES)
            + ". Buffers A, B, C are allocated in this order at 32-byte boundaries with -777 in the padding and 1000 in C before the launch, and the workgroups run in ascending "
            "order, so that the words an unguarded load reads past the end of its buffer are defined. 'Outside' counts outputs beyond the tolerance or not finite, out of m n; "
            "'worst ratio' is the largest error over the tolerance (printed as 'above 1e6' when it is: its value depends on the words the load happened to read).\n\n"
            + _table(["m x n x k", "k a multiple of 4", "guarded: outside", "guarded: worst ratio", "guarded: verdict",
                      "unguarded: outside", "unguarded: worst ratio", "unguarded: verdict"], rows, "llrrlrrl"))


# ============================================================ M6: group-wise quantization

Q_M, Q_K, Q_N, Q_G = 6, 128, 24, 32


def _quantized_case():
    gen = _gen("quantized")
    w = [0.05 * gen.normal() for _ in range(Q_K * Q_N)]
    w = [f32(v) for v in w]
    x = [f32(gen.normal()) for _ in range(Q_M * Q_K)]
    return w, x


@lru_cache(maxsize=None)
def _quant_rows():
    w, x = _quantized_case()
    a = Tensor(x, (Q_M, Q_K))
    wt = Tensor(w, (Q_K, Q_N))
    ref_fp = matmul_oracle(a, wt)
    rows, rows2 = [], []
    ref_pow = sum(v * v for v in ref_fp) / len(ref_fp)
    for nbits in (8, 4):
        cols = [[w[kk * Q_N + j] for kk in range(Q_K)] for j in range(Q_N)]
        S, Q = [], []
        for col in cols:
            s_, q_ = quantize_groups(col, Q_G, nbits)
            S.extend(s_)
            Q.extend(q_)
        w_dq = [S[j * (Q_K // Q_G) + kk // Q_G] * Q[j * Q_K + kk] for kk in range(Q_K) for j in range(Q_N)]
        w_err = sum((p - q) ** 2 for p, q in zip(w, w_dq)) / len(w)
        w_sqnr = 10 * math.log10((sum(v * v for v in w) / len(w)) / w_err)
        ref_dq = matmul_oracle(a, Tensor(w_dq, (Q_K, Q_N)))
        s_dq = abs_sums(a, Tensor(w_dq, (Q_K, Q_N)))
        for factored in (True, False):
            eng = Engine(DEVICE_A)
            be = eng.be
            kern = be.compile(matmul_quantized(16, factored))
            ba, bq, bs, bc = (be.create_buffer(len(x)), be.create_buffer(len(Q)), be.create_buffer(len(S)), be.create_buffer(Q_M * Q_N))
            be.upload(ba, x)
            be.upload(bq, Q)
            be.upload(bs, S)
            rec = be.dispatch(kern, [ba, bq, bs, bc], dict(m=Q_M, n=Q_N, k=Q_K, G=Q_G, contract=False), _ceil_div(Q_M * Q_N, 16))
            out = be.read_back(bc, Q_M * Q_N)
            k_rd = Q_G + Q_K // Q_G if factored else Q_K + 1
            wu, wt_, bad = error_report(out, ref_dq, s_dq, k_rd, Q_K)
            err_fp = sum((p - q) ** 2 for p, q in zip(out, ref_fp)) / len(out)
            rows.append((f"int{nbits}", "factored" if factored else "dequantize first", f"{nbits + 32 / Q_G:g}",
                         f"{(nbits + 32 / Q_G) / 8:.4f}", f"{w_sqnr:.2f}", f"{10 * math.log10(ref_pow / err_fp):.2f}",
                         f"{100 * math.sqrt(err_fp / ref_pow):.3f}", k_rd, f"{wt_:.4f}", bad, digest(out, "f")))
            if factored:
                rows2.append((f"int{nbits}", _n(rec["loads"].get("A", 0)), _n(rec["loads"].get("Q", 0)), _n(rec["loads"].get("S", 0)),
                              _n(rec["stores"].get("C", 0)), _n(WORD * Q_K * Q_N), _n(round(Q_K * Q_N * (nbits + 32 / Q_G) / 8))))
    return rows, rows2


def quantized_matmul():
    rows, rows2 = _quant_rows()
    return (f"The product C = A W, A of shape [{Q_M}, {Q_K}] (activations), W of shape [{Q_K}, {Q_N}] (weights), W quantized along k in groups of G = {Q_G} for each column "
            f"(symmetric codes -qmax .. qmax, qmax = 2^(bits - 1) - 1, scale = amax / qmax stored as float32, code = clamp(round_even(w / scale), -qmax, qmax)). {_seed('quantized')}. "
            f"Draw order: the {Q_K * Q_N} weights `0.05 normal()` rounded to float32, row-major over [k, n], then the {Q_M * Q_K} activations `normal()` rounded to float32, row-major. "
            f"Device A. 'Bits per weight' counts the {32}-bit scale of every group: bits + 32 / G. 'Weight SQNR' is 10 log10 (mean w^2 / MSE of the decoded weights); "
            "'output SQNR' and 'relative error' compare the kernel's C with the float64 product of the original weights (the error to expect from quantizing); "
            "'worst ratio' compares the kernel's C with the float64 product of the decoded weights scale x code (the float32 arithmetic alone) over a tolerance "
            "(gamma(k_rd, 2^-24) + gamma(k, 2^-53)) S with S = sum |a w'|, k_rd = G + k / G for the factored form (the group's chain, the scale's product, the additions across groups) "
            "and k + 1 = 129 for dequantizing first (a safe bound that ignores the grouping: the product with the rounded weight scale x code adds one rounding to a chain of k; "
            "the longest chain the kernel really has is the decode rounding, G additions in a group and k / G additions across groups, 1 + G + k / G = 37 roundings, so this bound is loose by a factor of about 3.5 and the ratios of its rows are small for that reason). "
            "The dequantize-first kernel is the loop of the factored one with two lines changed: in the group loop `inner = inner + A[i * K + grp * G + t] * (scale * Q[j * K + grp * G + t])` (the product scale x code is rounded to float32 first, the activation times it is a second rounding) "
            "and, after the group loop, `total = total + inner` with no multiplication by the scale (it is already inside the products); `inner` is reset to 0 for every group, `total` starts at 0, and each sum is a left fold from t = 0 and grp = 0 in float32 (device A: no contraction). "
            "Digest: binary32 little-endian bytes of C, first 16 hex digits.\n\n"
            + _table(["weights", "form", "bits per weight", "bytes per weight", "weight SQNR (dB)", "output SQNR (dB)", "relative error (%)", "k_rd",
                      "worst ratio", "outside", "digest"], rows, "llrrrrrrrrl")
            + "\n\nWhat the factored kernel reads (executor words) and the stored size of the weights:\n\n"
            + _table(["weights", "loads of A", "loads of Q", "loads of S", "stores of C", "bytes of W in float32", "bytes of W stored"], rows2))


# ============================================================ M7: selection and honest measurement

GPU_UNIT, CPU_UNIT = ks.DEVICES["GPU"], ks.DEVICES["CPU"]
GATE_CFG = DeviceConfig("gate", tile=4, lanes=1, contract=False, reduce_wg=8, wg_naive=64)
# name, the engine kernel, the tile side T it computes, its cost-model entry of Chapter 10, the model's price factor
CATALOGUE = (
    ("naive", dict(kind="naive", tile=8, model="naive", price=1.0, guard=True)),
    ("tiled", dict(kind=(8, 4, 8), tile=32, model="tiled", price=1.0, guard=True)),
    ("reg", dict(kind=(16, 8, 8), tile=128, model="reg", price=1.0, guard=True)),
    ("reg-nok", dict(kind=(16, 8, 8), tile=128, model="reg", price=0.95, guard=False)),
)
CAND = dict(CATALOGUE)


def device_record(dev):
    """The capabilities() record of Chapter 9 for a model unit of Chapter 10."""
    return dict(max_workgroup=dev["max_wg"], shared_bytes=dev["shared_kib"] * 1024, subgroup=32 if dev["workgroup_barriers"] else 0,
                max_barriers=math.inf if dev["workgroup_barriers"] else 0)


def declaration(name):
    """What a candidate needs, read from its kernel: barriers, shared bytes, workgroup size (a naive kernel's size is free)."""
    c = CAND[name]
    if c["kind"] == "naive":
        return dict(barriers=0, shared_bytes=0, workgroup=1)
    q, r, bk_ = c["kind"]
    kern = matmul_blocked(q, r, bk_)
    return dict(barriers=1, shared_bytes=WORD * kern.shared_words({}), workgroup=kern.wg)


def refusal(record, name):
    """Why a device whose capabilities() is `record` cannot run the candidate, or None. Barriers first, as in Chapter 10."""
    need = declaration(name)
    if need["barriers"] > record["max_barriers"]:
        return "no workgroup barriers"
    if need["shared_bytes"] > record["shared_bytes"]:
        return "workgroup memory"
    if need["workgroup"] > record["max_workgroup"]:
        return "workgroup size"
    return None


def selection_capabilities():
    rows = []
    for name, _ in CATALOGUE:
        need = declaration(name)
        row = [name, need["barriers"] and "yes" or "no", _n(need["shared_bytes"]), need["workgroup"] if name != "naive" else "any"]
        for dev in (CPU_UNIT, GPU_UNIT):
            why = refusal(device_record(dev), name)
            row.append("runs" if why is None else f"refused: {why}")
        rows.append(tuple(row))
    recs = [("CPU unit", device_record(CPU_UNIT)), ("GPU unit", device_record(GPU_UNIT))]
    rrows = [(nm, r["max_workgroup"], _n(r["shared_bytes"]), r["subgroup"], "unbounded" if r["max_barriers"] == math.inf else r["max_barriers"]) for nm, r in recs]
    return ("The two model units of Chapter 10 (`kernel_selection.DEVICES`) as the `capabilities()` record of Chapter 9; the CPU unit's compiler cannot run a "
            "barrier kernel (Chapter 9), so its record says 0 barriers.\n\n"
            + _table(["unit", "max_workgroup", "shared_bytes", "subgroup", "max_barriers"], rrows, "lrrrr")
            + "\n\nThe catalogue. A candidate's declaration is read from its kernel: barriers (yes if it has any), bytes of shared memory (2 T bk words of 4 bytes), "
            "and its workgroup size (q x q invocations). 'tiled' is the blocked kernel with q = 8, r = 4, bk = 8 (T = 32), 'reg' with q = 16, r = 8, bk = 8 (T = 128), "
            "'naive' one output per invocation; 'reg-nok' is 'reg' with the bound check of the k edge removed from its two tile loads (the defect of milestone 5). "
            "The filter reads the record, never the name of the device.\n\n"
            + _table(["candidate", "barriers", "shared bytes", "workgroup", "CPU unit", "GPU unit"], rows, "lrrrll")
            + "\n\nOn the CPU unit the filter leaves only the naive kernel: with one candidate there is nothing to choose between, so the engine runs it without a gate or a measurement.")


GATE_SHAPES_OF = lambda t: [(t, 16, 16), (t + 1, 17, 19)]   # noqa: E731  a whole-tile shape and a ragged one


def gate_candidate(name):
    """The reference gate of Chapter 10 on the engine: run the candidate's kernel on a whole-tile shape and a ragged shape and require
    every output within the derived tolerance of the float64 oracle. Returns ([(shape, outside, worst ratio)], admitted)."""
    c = CAND[name]
    out = []
    for (m, n, k) in GATE_SHAPES_OF(c["tile"]):
        gen = _gen(f"gate.{m}x{n}x{k}")
        a, b = Tensor(real_matrix(gen, m, k), (m, k)), Tensor(real_matrix(gen, k, n), (k, n))
        ref, s = matmul_oracle(a, b), abs_sums(a, b)
        try:
            got, _ = Engine(GATE_CFG).matmul(c["kind"], a, b, guard_k=c["guard"])
            _, wt, bad = error_report(got, ref, s, k_rd_matmul(k, 1), k)
        except ck.Fault:
            bad, wt = m * n, math.inf
        out.append(((m, n, k), bad, wt))
    return out, all(bad == 0 for _, bad, _ in out)


@lru_cache(maxsize=None)
def _gates():
    return {name: gate_candidate(name) for name, _ in CATALOGUE}


def selection_gate():
    rows = []
    for name, _ in CATALOGUE:
        res, ok = _gates()[name]
        for (m, n, k), bad, wt in res:
            rows.append((name, f"{m} x {n} x {k}", f"{bad} of {m * n}", "above 1e6" if wt > 1e6 else f"{wt:.4f}", "admitted" if bad == 0 else "REFUSED"))
        rows.append((name, "gate verdict", "", "", "ADMITTED" if ok else "REFUSED"))
    seeds = "; ".join(f"{_seed(f'gate.{m}x{n}x{k}')} for {m} x {n} x {k}" for (m, n, k) in sorted({s for nm, _ in CATALOGUE for s, _, _ in _gates()[nm][0]}))
    return ("The reference gate of Chapter 10 run on the engine: each candidate's kernel on the whole-tile shape (T, 16, 16) and the ragged shape (T + 1, 17, 19) of its tile T "
            "(T = 8 for the naive kernel, 32 for tiled, 128 for reg and reg-nok), device configuration A with one accumulator and no contraction (k_rd = k), real-valued inputs "
            f"`2 uniform() - 1` rounded to float32 (the m k entries of A, then the k n entries of B, row-major; {seeds}). The same seeded inputs are used by every candidate of a shape. "
            "A candidate is admitted when every output of both shapes is within (gamma(k, 2^-24) + gamma(k, 2^-53)) S of the float64 oracle and finite.\n\n"
            + _table(["candidate", "shape", "outside tolerance", "worst ratio", "verdict"], rows, "llrrl"))


REQUESTS = [(1, 2560, 2048), (48, 2560, 2048), (200, 2560, 2050), (512, 2560, 2048)]
RUNS = 300
MIN_STEADY = 264          # 4 x 66 = 4 x 8 cv^2 / 0.05^2 (Chapter 11): a gap of 5 percent is four standard errors


def _useful(m, n, k):
    return 2 * m * n * k, 4 * (m * k + k * n + m * n)


def _model_floor_us(dev, name, m, n, k):
    """The noise-free cost of a candidate on the unit, in microseconds of the model: Chapter 10's cost in equivalent bytes over the bandwidth."""
    c = CAND[name]
    return c["price"] * ks._cost(dev, ks.CANDIDATES[c["model"]], m, n, k) / (dev["bw"] * 1e3)


def _series(gen, floor_us):
    """RUNS dispatches timed by device stamps: floor x ((1 + 0.8 exp(-i / 5)) f_i + 20 on run 0), f_i the noise factor of Chapter 10 (three uniforms)."""
    return [floor_us * ((1.0 + pf.WARM_A * math.exp(-i / pf.WARM_TAU)) * pf._factor(gen) + (pf.WARM_COMPILE if i == 0 else 0.0)) for i in range(RUNS)]


def _faster(a, b, z=2.0):
    """b is faster than a only when the difference of the means exceeds z standard errors of the difference (Chapter 11)."""
    se = math.sqrt(pf._sd(a) ** 2 / len(a) + pf._sd(b) ** 2 / len(b))
    return pf._mean(a) - pf._mean(b) > z * se, (pf._mean(a) - pf._mean(b)) / se


@lru_cache(maxsize=None)
def _measure_all():
    """For each request, each candidate the GPU unit can run (catalogue order): its warm-up end, steady runs and statistics."""
    gen = _gen("clock")
    out = {}
    for (m, n, k) in REQUESTS:
        res = {}
        for name, _ in CATALOGUE:
            if refusal(device_record(GPU_UNIT), name):
                continue
            floor = _model_floor_us(GPU_UNIT, name, m, n, k)
            t = _series(gen, floor)
            end, _ = pf._warm_end([x / floor for x in t])
            steady = t[end:]
            assert len(steady) >= MIN_STEADY
            res[name] = dict(end=end, steady=steady, mean=pf._mean(steady), sd=pf._sd(steady), median=pf._median(steady), floor=floor)
        out[(m, n, k)] = res
    return out


def select_for(shape):
    """The engine's choice for a request on the GPU unit: the admitted candidate with the lowest mean, and the alone-by-measurement winner."""
    res = _measure_all()[shape]
    admitted = [nm for nm in res if _gates()[nm][1]]
    best = min(admitted, key=lambda nm: res[nm]["mean"])
    alone = min(res, key=lambda nm: res[nm]["mean"])
    return best, alone, admitted


def selection_report():
    m_ = _measure_all()
    rows, notes = [], []
    for shape in REQUESTS:
        m, n, k = shape
        fl, by = _useful(m, n, k)
        bound = pf._bound_us(fl, by)
        res = m_[shape]
        best, alone, admitted = select_for(shape)
        order = sorted(admitted, key=lambda nm: res[nm]["mean"])
        runner = order[1]
        sep, margin = _faster(res[runner]["steady"], res[best]["steady"])
        for name in res:
            c = CAND[name]
            cnt = ks._counts(ks.CANDIDATES[c["model"]], m, n, k)
            r = res[name]
            se = r["sd"] / math.sqrt(len(r["steady"]))
            tag = ("chosen" if name == best else ("admitted" if name in admitted else "REFUSED by the gate"))
            rows.append((f"{m} x {n} x {k}", name, _n(cnt["flop_pad"]), _n(cnt["n_byte"]), r["end"], len(r["steady"]), f"{r['mean']:.2f}", f"{se:.2f}",
                         f"{r['median']:.2f}", f"{bound / r['median']:.3f}", tag))
        gap = "-"
        if alone != best:
            _, gap_se = _faster(res[best]["steady"], res[alone]["steady"])
            gap = f"{alone} by {gap_se:.1f}"
        notes.append((f"{m} x {n} x {k}", _n(fl), _n(by), f"{bound:.2f}", alone, gap, best, f"{margin:.1f}"))
    dev = GPU_UNIT
    ridge = dev["flops"] / dev["bw"]
    prm = []
    for nm in ("naive", "tiled", "reg"):
        c = ks.CANDIDATES[nm]
        prm.append((nm, f"{c['bm']} x {c['bn']}", c["n_wg"], f"{c['rb'][0]} x {c['rb'][1]}", c["ell"], f"{ks._efficiency(c):.4f}"))
    params = (f"The cost model is that of Chapter 10 with the unit's TRUE rates, not the measured ones: peak {dev['flops']:,.0f} (GFLOP/s), bandwidth {dev['bw']:.0f} (GB/s), so the ridge is "
              f"{ridge:.4f} FLOP per byte; {dev['lanes']:,} lanes; memory latency {dev['lat_ns']:.0f} ns, so n_need = {ks._loads_needed(dev):,.0f} loads in flight; one dispatch alone costs "
              f"L = {dev['launch_us']:g} microseconds x {dev['bw']:.0f} (GB/s) = {ks._launch_bytes(dev):,.0f} bytes. Candidates of the model (tile T_m x T_n, invocations per workgroup, register block, loads in flight n_fl, issue efficiency e):\n\n"
              + _table(["candidate", "tile", "invocations per workgroup", "register block", "n_fl", "e"], prm, "lrrrrr")
              + f"\n\nFor a shape (m, n, k): t_m = ceil(m / T_m), t_n = ceil(n / T_n); N_byte = 4 (m k t_n + k n t_m + m n); executed FLOPs 2 (t_m T_m)(t_n T_n) k for tiled and reg, 2 m n k for naive; "
              "invocations t_m t_n n_wg for tiled and reg, and for naive m n rounded UP to a multiple of 64 (T_m = T_n = 1); o_cmp = min(1, invocations / lanes); o_mem = min(1, invocations x n_fl / n_need); "
              "cost in equivalent bytes B_dev = L + max(executed FLOPs / (ridge e o_cmp), N_byte / o_mem). The time of a launch in microseconds is this cost divided by the TRUE bandwidth, "
              f"B_dev / ({dev['bw']:.0f} x 1000) (bytes per microsecond), times 0.95 for reg-nok. The fraction of the roofline uses the MEASURED roofline: the roofline time of the problem is "
              "max(2 m n k / 34,469 (GFLOP/s), 4 (m k + k n + m n) / 646 (GB/s)), divided by the median.\n\n")
    return ("The GPU unit of Chapter 10 with its measured roofline, 34,469 (GFLOP/s) and 646 (GB/s), and the model clock of Chapters 10 and 11. " + params
            + "A run of a "
            "candidate is its noise-free cost (the cost in equivalent bytes of Chapter 10 over the bandwidth of the unit, times 0.95 for reg-nok, which the model grants for the "
            "checks it dropped) multiplied by the noise factor (1 + 0.04 u0)(1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1) and by the warm-up transient 1 + 0.8 exp(-i / 5), plus 20 "
            "noise-free costs on run 0. Everything below is the output of this model, not a measurement of a machine. Protocol: "
            f"{RUNS} dispatches per candidate timed by the device's own stamps; the warm-up ends at the first run that starts 5 consecutive runs each within 5 percent of the minimum of "
            "runs 150 to 299, and those runs are discarded; the rest (at least 264 = 4 x 66, the runs for a 5 percent gap at four standard errors, Chapter 11) give the mean, its "
            f"standard error s / sqrt(n) and the median; the winner is the admitted candidate with the lowest mean. {_seed('clock')}. Draw order: requests in the order of the table, "
            "candidates in catalogue order (those the unit can run), runs 0 .. 299, three uniforms each (u0, u1, u2). 'FLOPs executed' and 'bytes requested' are the counts of Chapter 10 "
            "for the candidate's tile (padded tiles included). Times are in microseconds of the model clock; the fraction is the roofline time of the problem on the measured roofline "
            "over the median (a fraction near 1 is the roofline).\n\n"
            + _table(["m x n x k", "candidate", "FLOPs executed", "bytes requested", "warm-up ends at run", "steady runs", "mean", "standard error", "median",
                      "fraction of roofline", "status"], rows, "llrrrrrrrrl")
            + "\n\nThe problem itself, and what measurement alone would have picked:\n\n"
            + _table(["m x n x k", "useful FLOPs 2 m n k", "floor bytes 4 (m k + k n + m n)", "roofline time (microseconds)", "fastest by measurement alone",
                      "its lead over the choice (standard errors of the difference)", "engine's choice", "choice over the admitted runner-up (standard errors of the difference)"],
                     notes, "lrrrlllr"))


def acceptance_fingerprint():
    gen = _gen("exact_cases")
    a, b = _exact_case(gen, *EXACT_SHAPES[0])
    small = digest(matmul_oracle(a, b), "d")
    gen = _gen("exact_cases")
    cases = [(s, *_exact_case(gen, *s)) for s in EXACT_SHAPES]
    (m, n, k), ca, cb = cases[3][0], cases[3][1], cases[3][2]
    out, rec = Engine(DEVICE_A).matmul("naive", ca, cb)
    gen = _gen("traffic.13x10x7")
    ta, tb = Tensor(int_matrix(gen, 13, 7), (13, 7)), Tensor(int_matrix(gen, 7, 10), (7, 10))
    ld = {}
    for label, kind, tile, big in KERNELS_M3:
        _, r = Engine(DEVICE_A).matmul(kind, ta, tb, tile=tile)
        ld[label] = r["loads"]["A"] + r["loads"]["B"]
    y = real_matrix(_gen("reductions"), 1, RED_M * RED_N + RED_TOTAL)[RED_M * RED_N:]
    ta_, _, _ = global_sum(DEVICE_A, y, RED_GRP)
    tb_, _, _ = global_sum(DEVICE_B, y, RED_GRP)
    res, _, _ = _parity_case(21, 18, 100, "tiled")
    same = sum(1 for p, q in zip(res["A"][0], res["B"][0]) if p == q)
    bad, _, _ = parity_check_shape("tiled", DEVICE_A, (13, 10, 7), guard_k=False)
    pick = {s: select_for(s) for s in REQUESTS}
    qrows, _ = _quant_rows()
    r512 = _measure_all()[REQUESTS[3]][pick[REQUESTS[3]][0]]
    frac512 = pf._bound_us(*_useful(*REQUESTS[3])) / r512["median"]
    rows = [
        (1, "digest of the oracle product C, 2 x 2 x 3 (binary64)", small, "exact"),
        (2, "naive kernel, 13 x 11 x 9: loads of A and B, stores of C", f"{_n(rec['loads']['A'] + rec['loads']['B'])}, {rec['stores']['C']}", "exact"),
        (3, "loads at 13 x 10 x 7: naive, tiled T = 4, blocked T = 8", ", ".join(_n(v) for v in ld.values()), "exact"),
        (4, "two-pass sum of Y: float32 bits on device A and on device B", f"{bk._bits(ta_):#010x}, {bk._bits(tb_):#010x}", "exact"),
        (5, "parity 21 x 18 x 100, tiled: entries equal on A and B", f"{same} of {18 * 21}", "exact"),
        (5, "unguarded tiled kernel at 13 x 10 x 7: outputs outside tolerance", f"{bad} of 130", "exact"),
        (6, "int4 and int8, G = 32: output SQNR (dB) of the factored kernel", f"{qrows[2][5]}, {qrows[0][5]}", "0.01"),
        (7, "choice for 1 x 2560 x 2048, 48 x 2560 x 2048, 200 x 2560 x 2050, 512 x 2560 x 2048", ", ".join(pick[s][0] for s in REQUESTS), "exact"),
        (7, "fraction of the measured roofline of the choice at 512 x 2560 x 2048 (median of the steady runs)", f"{frac512:.3f}", "0.002"),
    ]
    return _table(["milestone", "quantity", "value", "tolerance"], rows, "llll")
