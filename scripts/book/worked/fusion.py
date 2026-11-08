# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Kernel Fusion and Memory Traffic".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: fusion.<name> -->` markers. Helpers with a leading underscore, or
that take arguments, are shared with the chapter's figures so a picture and the
table beside it cannot disagree.

Nothing here measures a machine. The model GPU is the one of Chapters 3 and 4
(ridge point 50 FLOP per byte). Everything printed is counting: FLOPs (a
multiplication, addition, subtraction, division, square root or tanh is one),
bytes that cross between memory and the arithmetic (every distinct operand a
kernel reads once, every result it writes once, as in Chapter 3), dispatches and
submissions. The only quantity that is not a count is the COST of a design, which
is stated in bytes: the larger of its bytes and its FLOPs divided by the ridge
point (the time of Chapter 3's roofline multiplied by the bandwidth), plus the
fixed costs of starting work, two labelled parameters. The kernel simulator is
Chapter 5's. Randomness comes from the book's own generator
(`bookrng.generator("fusion.<example>")`) and is printed with its seed.

Swedish Embedded AB implements solutions for removing the memory round trips
between the small kernels of a machine learning workload. If your team needs
expertise in kernel fusion and memory-traffic analysis for GPU inference and
training, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import struct
from fractions import Fraction

import bookrng
import compute_kernels as ck
import gpu
import number_formats as nf
import roofline as rf

# ------------------------------------------------------------------ parameters

F32 = 4                                   # bytes per fp32 number
RIDGE = rf.ridge("GPU")                   # 50 FLOP per byte (Chapter 3)
BW = rf.MACHINES["GPU"][1] * 1e9          # bytes/s (Chapter 3)
N_ROWS, D_MODEL = 2048, 1024              # the activation [n, d] of the running chain
T_SUB = 4e-6                              # seconds the host-side submission costs (parameter)
T_DSP = 1e-6                              # seconds the device spends starting one recorded dispatch (parameter)
L_SUB = round(T_SUB * BW)                 # the same costs in bytes of memory traffic: 3,200,000
L_DSP = round(T_DSP * BW)                 # 800,000
L_LONE = L_SUB + L_DSP                    # a dispatch submitted alone: Chapter 3's 5 microseconds
assert L_LONE == round(rf.LAUNCH_SECONDS * BW)
GELU_K = struct.unpack("<f", struct.pack("<f", math.sqrt(2.0 / math.pi)))[0]
GELU_C = struct.unpack("<f", struct.pack("<f", 0.044715))[0]
EPS = 1e-5

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


def _m(v, digits=2):
    """Millions of bytes, for costs."""
    return f"{v / 1e6:,.{digits}f}"


def _yes(flag):
    return "yes" if flag else "no"


def _seed_note(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (the seed of the name `{name}`)."


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _cost(flops, nbytes):
    """Cost in bytes of one kernel on the model GPU: the larger of its bytes and
    the bytes the roofline's compute time is worth, FLOPs / ridge."""
    return max(nbytes, flops / RIDGE)


# ----------------------------------------------- 1. the chain and its accounting
# y = layer_norm(x + gelu(x W + b)) on an activation x of shape [n, d], W of shape [d, d].


def _chain(n=N_ROWS, d=D_MODEL):
    """The five kernels: (name, FLOPs, arrays read, array written), and array sizes in bytes."""
    sizes = {"x": F32 * n * d, "W": F32 * d * d, "b": F32 * d, "z": F32 * n * d, "u": F32 * n * d,
             "g": F32 * n * d, "r": F32 * n * d, "gamma": F32 * d, "beta": F32 * d, "y": F32 * n * d}
    kernels = [
        ("matmul z = x W", 2 * n * d * d, ["x", "W"], "z"),
        ("bias u = z + b", n * d, ["z", "b"], "u"),
        ("gelu g = gelu_tanh(u)", 9 * n * d, ["u"], "g"),
        ("residual r = g + x", n * d, ["g", "x"], "r"),
        ("layer norm y", n * (7 * d + 3), ["r", "gamma", "beta"], "y"),
    ]
    return kernels, sizes


def _group_bytes(group, kernels, sizes):
    """(read, written) bytes of the dispatch that runs kernels `group` (indices) together: the operands read
    that no kernel of the group produces, each once, and the results some kernel outside the group (or the
    caller, for y) still needs."""
    made = {kernels[i][3] for i in group}
    reads = {a for i in group for a in kernels[i][2] if a not in made}
    outside = {a for j, k in enumerate(kernels) if j not in group for a in k[2]}
    writes = {a for a in made if a in outside or a == "y"}
    return sum(sizes[a] for a in reads), sum(sizes[a] for a in writes)


def _group_flops(group, kernels):
    return sum(kernels[i][1] for i in group)


def _bound(flops, nbytes):
    return "bandwidth" if flops / nbytes < RIDGE else "compute"


LEVELS = (
    ("five kernels, one per operation", [[0], [1], [2], [3], [4]]),
    ("matmul with bias, gelu and residual as its epilogue, then layer norm", [[0, 1, 2, 3], [4]]),
    ("everything in one kernel", [[0, 1, 2, 3, 4]]),
)


def chain_traffic():
    kernels, sizes = _chain()
    rows = []
    tf = tr = tw = 0
    for i, (name, flops, _, _) in enumerate(kernels):
        r, w = _group_bytes([i], kernels, sizes)
        tf, tr, tw = tf + flops, tr + r, tw + w
        rows.append((name, _n(flops), _n(r), _n(w), f"{flops / (r + w):.3f}", _bound(flops, r + w)))
    rows.append(("total", _n(tf), _n(tr), _n(tw), f"{tf / (tr + tw):.3f}", ""))
    n, d = N_ROWS, D_MODEL
    return (f"x is [n, d] = [{n}, {d}] in float32 ({_n(F32 * n * d)} bytes), W is [d, d] ({_n(F32 * d * d)} bytes), b, gamma and beta are "
            f"[d]. Each kernel reads every distinct operand once and writes its result once. FLOPs: matmul 2 n d^2; bias n d; "
            f"gelu_tanh 9 n d (u u, (u u) u, 0.044715 times that, u plus it, sqrt(2/pi) times the sum, tanh counted as one, 1 plus it, "
            f"0.5 u, their product); residual n d; layer norm n (7 d + 3) (Volume III, Chapter 10). Intensity is FLOPs divided by "
            f"bytes read plus written; the ridge point of the model GPU is {RIDGE:g}.\n\n"
            + _table(["kernel", "FLOPs", "bytes read", "bytes written", "FLOP per byte", "bound"], rows, "lrrrrl"))


def fusion_levels():
    kernels, sizes = _chain()
    flops = sum(k[1] for k in kernels)
    floor_r = sum(sizes[a] for a in ("x", "W", "b", "gamma", "beta"))
    floor_w = sizes["y"]
    base = None
    rows = []
    for label, groups in LEVELS:
        r = sum(_group_bytes(g, kernels, sizes)[0] for g in groups)
        w = sum(_group_bytes(g, kernels, sizes)[1] for g in groups)
        cost = sum(_cost(_group_flops(g, kernels), sum(_group_bytes(g, kernels, sizes))) for g in groups)
        total = cost + len(groups) * L_LONE
        if base is None:
            base = (r + w, cost, total)
        rows.append((label, len(groups), _n(r), _n(w), _n(r + w), f"{base[0] / (r + w):.3f}", _m(cost), f"{base[1] / cost:.3f}",
                     _m(total), f"{base[2] / total:.3f}"))
    rows.append(("floor: x, W, b, gamma, beta in, y out", "-", _n(floor_r), _n(floor_w), _n(floor_r + floor_w),
                 f"{base[0] / (floor_r + floor_w):.3f}", "", "", "", ""))
    return (f"The chain of the previous block, run as three designs. A group of kernels run together reads the operands no member produces "
            f"(each once, x once although two members use it) and writes the results a later dispatch or the caller needs. FLOPs are "
            f"{_n(flops)} in every design. Cost is in millions of bytes: the larger of a dispatch's bytes and its FLOPs / {RIDGE:g}, summed over the "
            f"dispatches; 'with starts' adds {_n(L_LONE)} per dispatch (Chapter 3's 5 microsecond start at 800 (GB/s)). 'x' columns divide the "
            f"first design's value by the row's.\n\n"
            + _table(["design", "dispatches", "bytes read", "bytes written", "total bytes", "bytes x", "cost (M)", "cost x",
                      "with starts (M)", "starts x"], rows, "lrrrrrrrrr"))


def tanh_sensitivity():
    """The gelu kernel's intensity when tanh is counted as 1, 8 and 20 FLOPs."""
    n, d = N_ROWS, D_MODEL
    nbytes = 2 * F32 * n * d
    rows = []
    for t in (1, 8, 20):
        flops = (8 + t) * n * d
        rows.append((t, f"{8 + t}", _n(flops), f"{flops / nbytes:.3f}", _bound(flops, nbytes)))
    return ("The gelu kernel of the chain reads u and writes g: " + _n(nbytes) + " bytes. It does 8 ordinary operations and one tanh per "
            "element; the table counts the tanh at 1, 8 and 20 FLOPs.\n\n"
            + _table(["FLOPs counted for tanh", "FLOPs per element", "FLOPs", "FLOP per byte", "bound"], rows))


# ------------------------------------------- 2. the fused elementwise chain, simulated

SIM_ROWS, SIM_D, SIM_WG = 8, 8, 32


def _gelu_f32(u):
    """gelu_tanh evaluated one operation at a time in float32 (tanh: the library tanh of the float32 argument, rounded to float32)."""
    a = _f32(u * u)
    b = _f32(a * u)
    c = _f32(GELU_C * b)
    s = _f32(u + c)
    e = _f32(GELU_K * s)
    t = _f32(math.tanh(e))
    f = _f32(1.0 + t)
    h = _f32(0.5 * u)
    return _f32(h * f)


def _sim_inputs():
    gen = bookrng.generator("fusion.chain_sim")
    n = SIM_ROWS * SIM_D
    z = [_f32(gen.normal()) for _ in range(n)]
    b = [_f32(0.5 * gen.normal()) for _ in range(SIM_D)]
    xr = [_f32(gen.normal()) for _ in range(n)]
    return z, b, xr


def _direct_chain(z, b, xr):
    return [_f32(_gelu_f32(_f32(z[i] + b[i % SIM_D])) + xr[i]) for i in range(len(z))]


def _k_bias(inv, p):
    g = inv.gid_x
    if g >= p["n"]:
        inv.ret()
    inv.store("u", g, _f32(inv.load("z", g) + inv.load("b", g % p["d"])))


def _k_gelu(inv, p):
    g = inv.gid_x
    if g >= p["n"]:
        inv.ret()
    inv.store("gl", g, _gelu_f32(inv.load("u", g)))


def _k_resid(inv, p):
    g = inv.gid_x
    if g >= p["n"]:
        inv.ret()
    inv.store("r", g, _f32(inv.load("gl", g) + inv.load("xr", g)))


def _k_fused(inv, p):
    g = inv.gid_x
    if g >= p["n"]:
        inv.ret()
    u = _f32(inv.load("z", g) + inv.load("b", g % p["d"]))     # lives in a register
    inv.store("r", g, _f32(_gelu_f32(u) + inv.load("xr", g)))


def _sim_run(fused):
    z, b, xr = _sim_inputs()
    n = len(z)
    mem = ck.Memory()
    bufs = {"z": mem.alloc("z", z, "read"), "b": mem.alloc("b", b, "read"), "xr": mem.alloc("xr", xr, "read"),
            "u": mem.alloc("u", [0.0] * n), "gl": mem.alloc("gl", [0.0] * n), "r": mem.alloc("r", [0.0] * n)}
    params = {"n": n, "d": SIM_D}
    groups = -(-n // SIM_WG)
    stats = []
    for phase in ([_k_fused] if fused else [_k_bias, _k_gelu, _k_resid]):
        stats.append(ck.launch(mem, [phase], groups, SIM_WG, params, bufs))
    return mem.values("r"), stats


def chain_sim():
    z, b, xr = _sim_inputs()
    want = _direct_chain(z, b, xr)
    rows = []
    outs = {}
    for fused, label in ((False, "three kernels: bias, gelu, residual"), (True, "one fused kernel")):
        out, stats = _sim_run(fused)
        outs[fused] = out
        ld = sum(s["load"] for s in stats)
        st = sum(s["store"] for s in stats)
        same = sum(1 for a, c in zip(out, want) if _f32_bits(a) == _f32_bits(c))
        rows.append((label, len(stats), ld, st, _n(32 * (ld + st)), f"{same} of {len(want)}"))
    n = len(z)
    shown = ", ".join(f"{v:.6f}" for v in want[:6])
    return (f"The epilogue of the chain on a [{SIM_ROWS}, {SIM_D}] activation ({n} elements, workgroup size {SIM_WG}), run on the kernel simulator of Chapter 5: "
            f"r[i] = f32(gelu_tanh(f32(z[i] + b[i mod {SIM_D}])) + xr[i]), every operation rounded to float32 one at a time "
            f"(`gelu_tanh` as in the previous block, constants rounded to float32, tanh the library tanh of the float32 argument rounded to float32). "
            f"{_seed_note('fusion.chain_sim')} Draws in order: z[0..{n - 1}] = normal(), b[0..{SIM_D - 1}] = 0.5 normal(), "
            f"xr[0..{n - 1}] = normal(), each rounded to float32. First six results: [{shown}]. Transactions are the simulator's "
            f"32-byte blocks (each workgroup reads b once). 'bit-identical' compares float32 bit patterns with the direct evaluation.\n\n"
            + _table(["design", "dispatches", "load transactions", "store transactions", "bytes moved", "bit-identical to direct"], rows, "lrrrrl"))


# ------------------------------------------------ 3. when the row must stay in one place


def _row_owner(n, d, rows_per_wg, base_regs=24, wg=256):
    """The fused matmul + layer norm kernel in which a workgroup owns rows_per_wg whole rows of the output.
    Returns (accumulators per workgroup, registers per invocation, resident workgroups, occupancy %, bytes, FLOPs)."""
    acc = rows_per_wg * d
    regs = base_regs + acc // wg
    if regs > 255:
        res, occ = None, None
    else:
        res, occ, _ = gpu.occupancy(wg, regs, 0)
    row_blocks = -(-n // rows_per_wg)
    nbytes = F32 * (n * d + d * d * row_blocks + d + 2 * d + n * d)       # x once, W once per row block, b, gamma, beta, y
    flops = 2 * n * d * d + n * d + 9 * n * d + n * d + n * (7 * d + 3)
    return acc, regs, res, occ, nbytes, flops


def row_owner():
    n = N_ROWS
    rows = []
    for d in (1024, 8192):
        kernels, sizes = _chain(n, d)
        two = sum(sum(_group_bytes(g, kernels, sizes)) for g in LEVELS[1][1])
        for r in (1, 4, 16, 32, 64):
            acc, regs, res, occ, nbytes, flops = _row_owner(n, d, r)
            regs_s = f"{regs}" if regs <= 255 else f"{regs} (spills)"
            res_s = "-" if res is None else str(res)
            occ_s = "-" if occ is None else f"{100 * occ:.1f}"
            rows.append((d, r, _n(acc), regs_s, res_s, occ_s, _n(nbytes), f"{flops / nbytes:.1f}", _bound(flops, nbytes),
                         f"{nbytes / two:.2f}"))
    return (f"The whole chain in one kernel in which a workgroup of 256 invocations owns n_rw whole rows of the output, so that the layer norm "
            f"can finish in it. n = {n}. The workgroup holds n_rw d accumulators, shared by its 256 invocations on top of 24 registers each; resident "
            f"workgroups follow the occupancy rule of Chapter 4 (65,536 registers, 2,048 invocations, 16 workgroups per compute unit; "
            f"at most 255 registers per invocation). Bytes count the loads the workgroups issue (Chapter 6's rule): x once, W once per "
            f"row block (ceil(n / n_rw) times), b, gamma, beta, and y written. The last column divides by the bytes of the two-dispatch "
            f"design of the previous section (epilogue fusion, then layer norm).\n\n"
            + _table(["d", "n_rw", "accumulators", "registers per invocation", "resident workgroups", "occupancy (%)", "bytes",
                      "FLOP per byte", "bound", "bytes / two-dispatch bytes"], rows))


def redundant_partials():
    big = 1 << 24
    rows = []
    for n_el in (256, 128, 64, 16):
        wg_n = big // (256 * n_el)
        three_bytes = F32 * (big + wg_n) + F32 * (wg_n + 1) + F32 * (big + 1 + big)
        two_bytes = F32 * (big + wg_n) + F32 * (wg_n * wg_n + big + big)
        three_cost = three_bytes + 3 * L_DSP
        two_cost = two_bytes + 2 * L_DSP
        rows.append((n_el, _n(wg_n), 3, _n(three_bytes), _m(three_cost), 2, _n(two_bytes), _m(two_cost),
                     "two dispatches" if two_cost < three_cost else "three dispatches"))
    return (f"y[i] = x[i] / s with s the sum of all x, N = 2^24 = {_n(big)} floats, workgroups of 256 invocations each adding n_el numbers, so "
            f"n_wg = N / (256 n_el) workgroups. Three dispatches: partial sums (reads 4 N, writes 4 n_wg), the total by one workgroup (reads 4 n_wg, "
            f"writes 4), the division (reads 4 N + 4, writes 4 N). Two dispatches: partial sums, then a division in which every workgroup first adds "
            f"all n_wg partials itself (n_wg workgroups each reading 4 n_wg bytes). Cost in millions of bytes inside one recorded submission: bytes plus "
            f"{_n(L_DSP)} per dispatch. The cost of the submission itself, {_n(L_SUB)} bytes, is the same for both designs and is left out of the "
            f"'cost' columns (add {L_SUB / 1e6:.1f} to every cost to include it; it changes no comparison). The extra reads of the two-dispatch design, "
            f"4 n_wg^2 bytes, equal the dispatch they save when n_wg = sqrt({_n(L_DSP)} / 4) = {math.sqrt(L_DSP / 4):.1f}; "
            f"if the dispatches were submitted alone ({_n(L_LONE)} each) it would be n_wg = {math.sqrt(L_LONE / 4):.1f}.\n\n"
            + _table(["n_el", "n_wg", "dispatches", "bytes", "cost without submission (M)", "dispatches", "bytes",
                       "cost without submission (M)", "cheaper"],
                     rows, "rrrrrrrrl"))


# ------------------------------------------------------ 4. recompute or store

SCORE_D = 64


def recompute_store():
    n, d = N_ROWS, D_MODEL
    store = 2 * F32          # a write in the forward pass and a read in the backward pass, per element
    cands = [
        ("u = z + b, from z", 1),
        ("r = g + x, from g and x", 1),
        ("g = gelu_tanh(u), from u", 9),
        ("y = layer norm(r), from r and 2 numbers per row", 7 + 3 / d),
        ("z = x W, from x and W", 2 * d),
        (f"score s = q . k, head width {SCORE_D}", 2 * SCORE_D),
    ]
    rows = []
    for name, f in cands:
        rc = f / RIDGE
        rows.append((name, f"{f:.4g}", f"{rc:.3f}", store, "recompute" if rc < store else "store"))
    return (f"For one element of an intermediate tensor the backward pass needs again: storing costs {store} bytes (4 written, 4 read back). "
            f"Recomputing from tensors already in memory costs the FLOPs of the producing operation per element, worth FLOPs / {RIDGE:g} bytes "
            f"on the model GPU (the ridge point of Chapter 3). The rule picks the smaller.\n\n"
            + _table(["intermediate", "FLOPs per element", "recompute cost (bytes)", "store cost (bytes)", "choice"], rows, "lrrrl"))


def variant_count():
    axes = [("activation: identity, relu, silu, gelu", 4), ("bias: yes or no", 2), ("residual: yes or no", 2),
            ("output format: float32, float16, bfloat16", 3), ("normalization: none, layer, RMS", 3)]
    prod = 1
    ssum = 0
    rows = []
    for name, k in axes:
        prod *= k
        ssum += k
        rows.append((name, k, _n(prod), _n(ssum)))
    return ("Choices a fused matmul epilogue might offer. 'Product' is the number of kernels to write and compile when every combination is its own kernel; "
            "'sum' is the number of codes a kernel that reads one uniform integer per axis must understand.\n\n"
            + _table(["axis", "choices", "product so far", "sum so far"], rows))


# ---------------------------------------------- 5. launch overhead and command recording


def launch_costs(n_el, k):
    """(separate submissions, one recorded submission, fused) cost in bytes of a chain of k unary kernels on n_el floats."""
    per = 2 * F32 * n_el
    return (k * (L_LONE + per), L_SUB + k * L_DSP + k * per, L_LONE + per)


def launch_curve(n_el=1 << 16, k_max=64):
    ks = list(range(1, k_max + 1))
    cols = [launch_costs(n_el, k) for k in ks]
    return ks, [c[0] for c in cols], [c[1] for c in cols], [c[2] for c in cols]


def launch_overhead():
    k = 16
    rows = []
    for e in (12, 16, 20, 24):
        n_el = 1 << e
        a, r, f = launch_costs(n_el, k)
        launch_share = k * L_LONE / a
        rows.append((f"2^{e}", _n(2 * F32 * n_el), _m(a), f"{100 * launch_share:.1f}", _m(r), f"{a / r:.2f}", _m(f), f"{a / f:.2f}"))
    return (f"A chain of k = {k} elementwise kernels, each reading and writing N floats (8 N bytes), on the model GPU. Parameters: submission {_n(L_SUB)} bytes "
            f"({T_SUB * 1e6:g} microseconds at 800 (GB/s)), starting one dispatch {_n(L_DSP)} bytes ({T_DSP * 1e6:g} microsecond); a dispatch submitted alone pays both "
            f"({_n(L_LONE)}, Chapter 3's 5 microseconds). Separate: k (submission + dispatch + 8 N). Recorded: one submission + k dispatches + k 8 N. "
            f"Fused: one submission + one dispatch + 8 N (the intermediates never leave registers). Costs in millions of bytes.\n\n"
            + _table(["N", "bytes per kernel", "separate (M)", "starts' share (%)", "recorded (M)", "separate / recorded",
                      "fused (M)", "separate / fused"], rows))


class _Host:
    """A host with a queue, to count submissions. Commands are Chapter 5's y = scale x + shift kernels on buffers b0 .. b3."""

    def __init__(self, n=ck.AXPB_N, wg=ck.AXPB_WG):
        self.n, self.wg = n, wg
        self.mem = ck.Memory()
        self.bufs = {nm: self.mem.alloc(nm, [0.0] * n) for nm in ("b0", "b1", "b2", "b3")}
        self.recording = None
        self.queue = []
        self.submissions = self.dispatches = self.waits = 0

    def upload(self, name, values):
        b = self.bufs[name]
        self.mem.words[b.base:b.base + b.n] = [float(v) for v in values]

    def begin_commands(self):
        assert self.recording is None
        self.recording = []

    def record(self, src, dst, scale, shift):
        self.recording.append((src, dst, scale, shift))

    def submit(self):
        self.queue.append(self.recording)
        self.submissions += 1
        self.dispatches += len(self.recording)
        self.recording = None

    def dispatch(self, src, dst, scale, shift):
        self.begin_commands()
        self.record(src, dst, scale, shift)
        self.submit()

    def read_back(self, name):
        assert self.recording is None, "commands recorded but never submitted"
        self.waits += 1
        for cmds in self.queue:
            for src, dst, scale, shift in cmds:
                ck.launch(self.mem, ck._axpb(True), -(-self.n // self.wg), self.wg,
                          {"n": self.n, "scale": scale, "shift": shift}, {"x": self.bufs[src], "y": self.bufs[dst]})
        self.queue = []
        return self.mem.values(name)


def _program(kind):
    h = _Host()
    h.upload("b0", ck._axpb_input(h.n))
    c = ck.CHAIN
    if kind == "alone":
        for i in range(3):
            h.dispatch(f"b{i}", f"b{i + 1}", *c[i])
        y = h.read_back("b3")
    elif kind == "recorded":
        h.begin_commands()
        for i in range(3):
            h.record(f"b{i}", f"b{i + 1}", *c[i])
        h.submit()
        y = h.read_back("b3")
    else:
        h.begin_commands()
        for i in range(2):
            h.record(f"b{i}", f"b{i + 1}", *c[i])
        h.submit()
        h.read_back("b2")                 # the host looks at the middle result
        h.begin_commands()
        h.record("b2", "b3", *c[2])
        h.submit()
        y = h.read_back("b3")
    return h, y


def command_recording():
    want = ck._chain_direct(ck._axpb_input(ck.AXPB_N))
    rows = []
    for kind, label in (("alone", "each dispatch submitted alone"), ("recorded", "three dispatches recorded, one submission"),
                        ("broken", "recorded, but the host reads the middle result")):
        h, y = _program(kind)
        launch = h.submissions * L_SUB + h.dispatches * L_DSP
        rows.append((label, h.dispatches, h.submissions, h.waits, _n(launch), _yes(y == want)))
    return (f"Chapter 5's three kernels y = scale x + shift with (scale, shift) = {ck.CHAIN[0]}, {ck.CHAIN[1]}, {ck.CHAIN[2]} on x = "
            f"[{', '.join(ck._v(v) for v in ck._axpb_input(ck.AXPB_N))}] (N = {ck.AXPB_N}, workgroup size {ck.AXPB_WG}), buffers resident on the device, "
            f"written as host programs with the operations of the chapter. 'Start cost' is submissions x {_n(L_SUB)} + dispatches x {_n(L_DSP)} bytes. "
            f"Final y = [{', '.join(ck._v(v) for v in want)}].\n\n"
            + _table(["host program", "dispatches", "submissions", "read-backs", "start cost (bytes)", "y equals direct"], rows, "lrrrrl"))


# --------------------------------------------------------- 6. numerics of fusion

PREC_N, PREC_D = 4096, 64


def _f16(x):
    return nf.quantize(x, "fp16")


def _gelu_f64(u):
    a = u * u
    b = a * u
    return (0.5 * u) * (1.0 + math.tanh(GELU_K_D * (u + GELU_C_D * b)))


GELU_K_D, GELU_C_D = math.sqrt(2.0 / math.pi), 0.044715


def _precision_run(tanh=math.tanh):
    gen = bookrng.generator("fusion.precision")
    z = [_f16(2.0 * gen.normal()) for _ in range(PREC_N)]
    b = [_f16(0.5 * gen.normal()) for _ in range(PREC_D)]
    xr = [_f16(gen.normal()) for _ in range(PREC_N)]

    def gelu32(u):
        a = _f32(u * u)
        bb = _f32(a * u)
        c = _f32(GELU_C * bb)
        s = _f32(u + c)
        e = _f32(GELU_K * s)
        t = _f32(tanh(e))
        return _f32(_f32(0.5 * u) * _f32(1.0 + t))

    unf, fus, ref, exact = [], [], [], []
    for i in range(PREC_N):
        u16 = _f16(_f32(z[i] + b[i % PREC_D]))                    # stored as float16 between the kernels
        g16 = _f16(gelu32(u16))
        unf.append(_f16(_f32(g16 + xr[i])))
        u = _f32(z[i] + b[i % PREC_D])                            # kept as float32 in a register
        fus.append(_f16(_f32(gelu32(u) + xr[i])))
        u64 = z[i] + b[i % PREC_D]
        v = _gelu_f64(u64) + xr[i]
        exact.append(v)
        ref.append(_f16(v))
    return z, b, xr, unf, fus, ref, exact


def precision():
    _, _, _, unf, fus, ref, exact = _precision_run()
    rows = []
    for label, out in (("three kernels, float16 buffers between them", unf), ("one kernel, float32 in registers, float16 result", fus)):
        diff_ref = sum(1 for a, c in zip(out, ref) if a != c)
        worst = max(abs(a - e) for a, e in zip(out, exact))
        mae = sum(abs(a - e) for a, e in zip(out, exact)) / len(out)
        rows.append((label, diff_ref, f"{worst:.3e}", f"{mae:.3e}"))
    differ = sum(1 for a, c in zip(unf, fus) if a != c)
    return (f"The epilogue chain r = gelu_tanh(z + b) + xr on N = {PREC_N} elements with float16 inputs and a float16 result. {_seed_note('fusion.precision')} "
            f"Draws in order: z[0..{PREC_N - 1}] = 2 normal(), b[0..{PREC_D - 1}] = 0.5 normal() (b[i mod {PREC_D}] multiplies element i), "
            f"xr[0..{PREC_N - 1}] = normal(), each rounded to float16 (ties to even). Every arithmetic operation is rounded to float32 one at a time; "
            f"the unfused design also rounds u and g to float16 where they are stored. The reference evaluates the chain in float64 (library tanh) "
            f"and rounds once to float16; errors are against the unrounded float64 value. "
            f"The two designs differ in {differ} of {PREC_N} results.\n\n"
            + _table(["design", "results differing from the reference", "largest absolute error", "mean absolute error"], rows, "lrrr"))


def fma_rounding():
    gen = bookrng.generator("fusion.fma")
    n = 4096
    a = [_f32(gen.normal()) for _ in range(n)]
    b = [_f32(gen.normal()) for _ in range(n)]
    c = [_f32(gen.normal()) for _ in range(n)]
    steps = {}
    for x, y, z in zip(a, b, c):
        two = _f32(_f32(x * y) + z)
        one = _round_fraction_f32(Fraction(x) * Fraction(y) + Fraction(z))
        d = abs(_f32_order(one) - _f32_order(two))
        steps[d] = steps.get(d, 0) + 1
    rows = [("identical", steps.get(0, 0)), ("1 step apart", steps.get(1, 0)), ("2 steps apart", steps.get(2, 0)),
            ("3 or more steps apart", sum(v for k, v in steps.items() if k >= 3))]
    return (f"x y + z for N = {n} float32 triples. {_seed_note('fusion.fma')} Draws in order: all x (normal()), then all y, then all z, each rounded to float32. "
            f"'Two roundings' computes f32(f32(x y) + z); 'one rounding' forms x y + z exactly and rounds once to float32 (a fused multiply-add instruction). "
            f"A step is the distance between neighbouring float32 numbers.\n\n"
            + _table(["the two results are", "triples"], rows, "lr"))


def _f32_order(x):
    """The position of a float32 number among all of them: neighbours differ by 1."""
    bits = _f32_bits(x)
    return -(bits & 0x7FFFFFFF) if bits >> 31 else bits


def _round_fraction_f32(q):
    """The float32 nearest to the rational q, ties to even."""
    if q == 0:
        return 0.0
    sign = -1.0 if q < 0 else 1.0
    a = abs(Fraction(q))
    e = nf.exponent_of(a)
    e = max(e, -126)
    m = nf.round_even(a / Fraction(2) ** (e - 23))
    return sign * math.ldexp(m, e - 23)


# -------------------------------------------------------------- 7. attention bytes


def _attention(n, d=SCORE_D):
    """(unfused FLOPs, unfused bytes, fused FLOPs, fused bytes) of one head, float32, scores [n, n]."""
    flops = 4 * n * n * d + n * (5 * n - 2)
    unf = F32 * (2 * n * d + n * n) + F32 * (2 * n * n) + F32 * (n * n + 2 * n * d)
    fus = F32 * 4 * n * d
    return flops, unf, fus


def attention_bytes():
    rows = []
    for n in (512, 4096, 32768):
        flops, unf, fus = _attention(n)
        rows.append((_n(n), f"{F32 * n * n / 2 ** 20:,.1f}", _n(unf), _n(fus), f"{unf / fus:.1f}", f"{flops / unf:.1f}", f"{flops / fus:.1f}"))
    return (f"One attention head in float32, head width d = {SCORE_D}, n positions, scores S = Q K^T of shape [n, n]. Unfused: QK^T reads Q and K and writes S; softmax "
            f"reads S and writes P (same size); P V reads P and V and writes the output. Bytes = 4 (2 n d + n^2) + 4 (2 n^2) + 4 (n^2 + 2 n d) = 16 n^2 + 16 n d. "
            f"Fused: Q, K, V in, output out: 16 n d. FLOPs 4 n^2 d for the two products plus n (5 n - 2) for the row softmaxes "
            f"(maximum, subtraction, exp, sum, division). Ridge point {RIDGE:g}.\n\n"
            + _table(["n", "S in MiB", "unfused bytes", "fused bytes", "ratio", "FLOP per byte, unfused", "FLOP per byte, fused"], rows))


# ------------------------------------------------------ 8. layout of the figures


def figure_numbers():
    kernels, sizes = _chain()
    out = []
    for label, groups in LEVELS:
        r = sum(_group_bytes(g, kernels, sizes)[0] for g in groups)
        w = sum(_group_bytes(g, kernels, sizes)[1] for g in groups)
        out.append((label, len(groups), r, w))
    boundary = 2 * sizes["z"]        # one intermediate written and read back
    return out, boundary, sizes["z"]
