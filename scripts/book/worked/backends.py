# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "One Program, Many Devices".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: backends.<name> -->` markers. Helpers with a leading underscore,
or that take arguments, are shared with the chapter's figures so a picture and
the table beside it cannot disagree.

Nothing here measures a machine and nothing here is a real vendor's compiler.
The "devices" are small models of the three ways a kernel reaches hardware:

  * the invocation model of Chapter 5 (a program for one invocation, run for
    every invocation of a grid, in any order);
  * a CUDA-like model (blocks of threads, every index clamped to its buffer, the
    uniform block a list of 32-bit words);
  * a CPU loop nest (a workgroup is a loop over its invocations, a barrier ends a
    loop).

Everything printed is a count, a bit pattern or an error in units of the unit
roundoff u = 2^-24, so it is reproducible digit for digit in any language that
rounds to nearest even in float32 and has exact rational arithmetic. The few
parameters of the startup model (kernels, bytes, compiler steps per byte) are
labelled as parameters. Randomness comes from the book's own generator
(`bookrng.generator("backends.<example>")`) and is printed with its seed.

Swedish Embedded AB implements solutions for running one set of compute
kernels on every device a client owns and proving that the devices agree. If
your team needs expertise in shader translation, kernel compilation and
cross-backend numerical parity, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import hashlib
import math
import struct
from fractions import Fraction

import bookrng

# ------------------------------------------------------------------ parameters

U = 2.0 ** -24                 # unit roundoff of float32 (Volume I, Chapter 15)
N_AXPB, WG_AXPB = 10, 4        # the kernel of Chapter 5: y = scale x + shift over 10 numbers, workgroups of 4
SCALE, SHIFT = 2.0, 1.0
DOT_SIZES = (16, 64, 256, 1024, 4096)   # lengths of the seeded dot products
DOT_VECTORS = 40                        # seeded vector pairs per length

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


def _seed_note(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (the seed of the name `{name}`)."


def _yes(flag):
    return "yes" if flag else "no"


def _g(v, digits=3):
    return f"{v:.{digits}g}"


# ------------------------------------------------------- float32 arithmetic


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _from_bits(a):
    return struct.unpack("<f", struct.pack("<I", a & 0xFFFFFFFF))[0]


def _round_f32(q):
    """The float32 nearest to the rational q, ties to even (normal range)."""
    q = Fraction(q)
    if q == 0:
        return 0.0
    sign = -1.0 if q < 0 else 1.0
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    # a / 2^e lies in [1, 2); keep 24 significant bits
    scaled = a / Fraction(2) ** (e - 23)
    m = scaled.numerator // scaled.denominator
    rem = scaled - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m % 2 == 1):
        m += 1
    return sign * math.ldexp(m, e - 23)


def _fma(a, b, c):
    """a b + c with one rounding to float32."""
    return _round_f32(Fraction(a) * Fraction(b) + Fraction(c))


def _order(x):
    """Position of a float32 among all of them (neighbours differ by 1)."""
    b = _bits(x)
    return -(b & 0x7FFFFFFF) if b >> 31 else b


def _ulps(a, b):
    return abs(_order(a) - _order(b))


# ============================================================ 1. three paths
# One kernel, written once as the description below, and the three programs the
# three kinds of device get from it.

UNIFORM_FIELDS = [("n", "u32", "unsigned int"), ("scale", "f32", "float"), ("shift", "f32", "float")]


def _wgsl_source():
    fields = "\n".join(f"    {n}: {t}," for n, t, _ in UNIFORM_FIELDS)
    return (f"struct Params {{\n{fields}\n}};\n\n"
            "@group(0) @binding(0) var<uniform> p: Params;\n"
            "@group(0) @binding(1) var<storage, read>       x: array<f32>;\n"
            "@group(0) @binding(2) var<storage, read_write> y: array<f32>;\n\n"
            f"@compute @workgroup_size({WG_AXPB})\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
            "    let g = gid.x;\n"
            "    if (g >= p.n) { return; }\n"
            "    y[g] = p.scale * x[g] + p.shift;\n"
            "}\n")


def _cuda_source():
    """The shape the translator gives the kernel: uniform words read at the byte offsets of the
    WGSL layout, one pointer and one length per storage binding, every index clamped."""
    off = {name: 4 * i for i, (name, _, _) in enumerate(UNIFORM_FIELDS)}
    rd = {"u32": "__brain_uu32", "f32": "__brain_uf32"}
    ty = {name: t for name, t, _ in UNIFORM_FIELDS}

    def p(name):
        return f"{rd[ty[name]]}(__params, {off[name]}u)"
    return ("extern \"C\" __global__ void brain_scale_shift(const unsigned int* __params,\n"
            "    float* __b1, float* __b2, unsigned long long __n1, unsigned long long __n2)\n"
            "{\n"
            "  const size_t __nel1 = __n1 / 1u;\n"
            "  const size_t __nel2 = __n2 / 1u;\n"
            f"  const unsigned int __gid_x = blockIdx.x * {WG_AXPB}u + threadIdx.x;\n"
            f"  if (__gid_x >= {p('n')}) {{\n"
            "    return;\n"
            "  }\n"
            f"  __b2[__brain_clamp(__gid_x, __nel2)] = {p('scale')} * __b1[__brain_clamp(__gid_x, __nel1)] + {p('shift')};\n"
            "}\n")


def _cpu_listing():
    return ("function scale_shift_cpu(x: array<f32>[N], y: inout array<f32>[N], scale: f32, shift: f32, N: int, n_grp: int):\n"
            "    for w in 0 .. n_grp:                  # workgroups: the loop that is shared out over the cores\n"
            f"        for l in 0 .. {WG_AXPB}:                      # the invocations of one workgroup, in order\n"
            f"            g = w * {WG_AXPB} + l\n"
            "            if g >= N: continue           # `return` ends this invocation only\n"
            "            y[g] = scale * x[g] + shift\n")


def _x_input(n=N_AXPB):
    return [float(i - 3) for i in range(n)]


def _run_invocation_model(x, order):
    """Chapter 5's model: the body of one invocation, run for every invocation in the given order."""
    n_grp = -(-len(x) // WG_AXPB)
    y = [None] * len(x)
    ran = 0
    for inv in order:
        w, l = divmod(inv, WG_AXPB)
        g = w * WG_AXPB + l
        ran += 1
        if g >= len(x):
            continue
        y[g] = SCALE * x[g] + SHIFT
    return y, n_grp, ran


def _run_cuda_model(x, params):
    """Blocks of threads (run last block first), the uniform block a list of 32-bit words, every index clamped."""
    n = params[0]
    scale, shift = _from_bits(params[1]), _from_bits(params[2])
    nel = len(x)
    y = [None] * nel
    clamp = lambda i, m: i if i < m else (m - 1 if m else 0)  # noqa: E731
    n_blocks = -(-nel // WG_AXPB)
    ran = 0
    for block in range(n_blocks - 1, -1, -1):
        for thread in range(WG_AXPB):
            gid = block * WG_AXPB + thread
            ran += 1
            if gid >= n:
                continue
            y[clamp(gid, nel)] = scale * x[clamp(gid, nel)] + shift
    return y, n_blocks, ran


def _run_cpu_model(x):
    """The loop nest of the listing."""
    n_grp = -(-len(x) // WG_AXPB)
    y = [None] * len(x)
    body = 0
    for w in range(n_grp):
        for l in range(WG_AXPB):
            g = w * WG_AXPB + l
            body += 1
            if g >= len(x):
                continue
            y[g] = SCALE * x[g] + SHIFT
    return y, n_grp, body


def three_paths():
    x = _x_input()
    direct = [SCALE * v + SHIFT for v in x]
    params = [N_AXPB, _bits(SCALE), _bits(SHIFT)]
    gen = bookrng.generator("backends.three_paths")
    order = gen.shuffle(list(range(-(-N_AXPB // WG_AXPB) * WG_AXPB)))
    ya, ga, ra = _run_invocation_model(x, order)
    yb, gb, rb = _run_cuda_model(x, params)
    yc, gc, rc = _run_cpu_model(x)
    wgsl, cuda, cpu = _wgsl_source(), _cuda_source(), _cpu_listing()
    rows = [("invocation model, shuffled order", "text compiled by the driver", ga, ra, _yes(ya == direct)),
            ("CUDA-like model, last block first", "translated C++ text", gb, rb, _yes(yb == direct)),
            ("CPU loop nest", "loops, no launch", gc, rc, _yes(yc == direct))]
    mapping = [
        ("workgroup size", f"`@workgroup_size({WG_AXPB})`", f"`blockDim` = {WG_AXPB} at launch, `blockIdx.x * {WG_AXPB}u`", f"inner loop bound {WG_AXPB}"),
        ("global index g", "`gid.x`", "`blockIdx.x * 4u + threadIdx.x`", "`g = w * 4 + l`"),
        ("uniform p.n", "`p.n` (u32 at byte 0)", "`__brain_uu32(__params, 0u)`", "the argument `N`"),
        ("uniform p.scale", "`p.scale` (f32 at byte 4)", "`__brain_uf32(__params, 4u)`, bit pattern reread as f32", "the argument `scale`"),
        ("x[g] and y[g]", "`x[g]`, `y[g]`", "`__b1[__brain_clamp(g, __nel1)]`", "`x[g]`, `y[g]`"),
        ("`return`", "`return;`", "`return;`", "`continue`"),
    ]
    return (f"The kernel y = scale x + shift of Chapter 5, scale = {SCALE:g}, shift = {SHIFT:g}, N = {N_AXPB}, x = {[int(v) for v in x]}, workgroup size {WG_AXPB}. "
            f"The uniform block is the list of 32-bit words [{params[0]}, {params[1]:#010x}, {params[2]:#010x}] (n, then the bit patterns of scale and shift). "
            f"{_seed_note('backends.three_paths')} Draw: one Fisher-Yates shuffle (the book's) of the {len(order)} invocation numbers 0 .. {len(order) - 1}, giving the order {order} "
            f"(invocation number = w * {WG_AXPB} + l).\n\n"
            "The WGSL source:\n\n```wgsl\n" + wgsl + "```\n\n"
            "The same kernel as the translator gives it to a CUDA device (names and the clamp are the emitter's; the temporaries it introduces for sub-expressions are left out):\n\n"
            "```cuda\n" + cuda + "```\n\n"
            "The same kernel as a CPU loop nest, in the pseudocode of the book:\n\n"
            "```text\n" + cpu + "```\n\n"
            + _table(["construct", "WGSL", "translated for CUDA", "CPU loop nest"], mapping, "llll") + "\n\n"
            + _table(["execution", "program form", "workgroups or blocks", "times the body ran", "y equals scale x + shift"],
                     rows, "llrrl") + "\n\n"
            f"y = {[int(v) for v in yc]}.")


# ============================================ 2. artifacts to keep consistent


def maintained_artifacts():
    d = 3
    rows = []
    for k in (10, 100, 1000):
        rows.append((_n(k), _n(k * d), _n(k + d), _n(k * d * (d - 1) // 2)))
    return (f"D = {d} devices (a CPU, a Vulkan GPU, a CUDA GPU) and K kernels. 'Per-device copies': one hand-written source per kernel per device, K D files. "
            f"'One source and translators': K sources and D translators, K + D artifacts. A comparison of two devices on one kernel is one cell of the parity matrix, "
            f"K D (D - 1) / 2 cells in both designs: translation does not remove the testing, it removes the copies that can drift and moves the repair of a defect "
            f"from K D files to D translators.\n\n"
            + _table(["K", "files, per-device copies", "artifacts, one source and translators", "parity cells, either design"], rows))


# ===================================================== 3. what must survive


def _shift_rows():
    rows = []
    a = 1
    for s in (0, 1, 31, 32, 33, 63):
        masked = (a << (s % 32)) & 0xFFFFFFFF
        clamped = (a << s) & 0xFFFFFFFF if s < 32 else 0
        rows.append((s, f"{masked:#010x}", f"{clamped:#010x}", _yes(masked == clamped)))
    return rows


def _wrap32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def preserved_semantics():
    wrap_rows = []
    for a, b, op in ((2147483647, 1, "+"), (-2147483648, -1, "+"), (65536, 65536, "*"), (100000, 100000, "*")):
        exact = a + b if op == "+" else a * b
        wrap_rows.append((f"{a} {op} {b}", _n(exact), _n(_wrap32(exact)), _yes(exact == _wrap32(exact))))
    x = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    n = len(x)
    pol_rows = []
    for policy, neighbour in (("unchecked", 500.0), ("unchecked", -777.0), ("clamp to the last element", None), ("return zero", None)):
        if policy == "unchecked":
            last = neighbour
        elif policy.startswith("clamp"):
            last = x[min(n, n - 1)]
        else:
            last = 0.0
        pol_rows.append((policy, "-" if neighbour is None else f"{neighbour:g}", f"{last:g}"))
    return ("**Shift counts.** The unsigned 32-bit word 1 shifted left by s. A machine that masks the count takes s mod 32; one that clamps gives 0 for s >= 32. WGSL calls the range s >= 32 "
            "indeterminate, so a translation must pick the reference's rule and write it out (here: mask).\n\n"
            + _table(["s", "1 << (s mod 32), masked", "clamped at the word width", "agree"], _shift_rows(), "rrrl") + "\n\n"
            "**Signed overflow.** WGSL's i32 arithmetic wraps modulo 2^32. The table is the wrapped value next to the exact one (C++ calls the overflow undefined, which is why a translator "
            "does the arithmetic on the unsigned bit pattern and reads the result back as signed).\n\n"
            + _table(["expression", "exact integer", "i32, wrapped", "same"], wrap_rows, "lrrl") + "\n\n"
            f"**Bounds.** The kernel y[g] = x[g + 1] for g = 0 .. {n - 1}, x = {[int(v) for v in x]}, whose last invocation reads one element past the buffer (the pattern of a "
            "ragged tile that masks the result afterwards). y[5] under the three rules a language may choose; the word after the buffer is the first word of whatever buffer "
            "the allocator placed next.\n\n"
            + _table(["rule for an index past the end", "word after the buffer", "y[5]"], pol_rows, "lrr"))


# ====================================== 4. a workgroup as loops on the CPU

WG = 4
SENTINEL = -1.0   # what an output slot holds when nobody wrote it


def _gpu_reference(x, n, groups, n_real, out_len):
    """The semantics WGSL promises: every workgroup starts with zeroed shared memory, every invocation of a
    workgroup finishes the first phase before any starts the second, and a returned invocation stays returned."""
    out = [SENTINEL] * out_len
    for w in range(groups):
        shared = [0.0] * WG
        alive = [True] * WG
        for l in range(WG):
            if w >= n_real:
                alive[l] = False                     # the workgroup-uniform guard: return
                continue
            g = w * WG + l
            if g < n:
                shared[l] = x[g]
        for l in range(WG):
            if alive[l] and l == 0:
                out[w] = shared[0] + shared[1] + shared[2] + shared[3]
    return out


def _cpu_loops(x, n, groups, n_real, out_len, *, return_scope, zero_shared):
    """The CPU compilation: per workgroup, loop 1 over l runs the code before the barrier, loop 2 the code after.
    return_scope: 'invocation' (a returned invocation skips loop 2 too, kept in the alive mask) or 'first-loop'.
    zero_shared: whether the shared array is cleared at the start of every workgroup. The array is ONE allocation
    reused by all workgroups (a stack slot)."""
    out = [SENTINEL] * out_len
    shared = [0.0] * WG
    for w in range(groups):
        if zero_shared:
            shared = [0.0] * WG
        alive = [True] * WG
        for l in range(WG):                          # loop 1: before the barrier
            if w >= n_real:
                alive[l] = False
                continue
            g = w * WG + l
            if g < n:
                shared[l] = x[g]
        for l in range(WG):                          # loop 2: after the barrier
            if return_scope == "invocation" and not alive[l]:
                continue
            if l == 0:
                out[w] = shared[0] + shared[1] + shared[2] + shared[3]
    return out


def _phase_run(mode):
    x = [float(i + 1) for i in range(10)]
    args = (x, 10, 5, 3, 5)
    if mode == "gpu":
        return _gpu_reference(*args)
    scope, zero = {"loops": ("invocation", True), "first-loop": ("first-loop", True), "stale": ("invocation", False),
                   "both": ("first-loop", False)}[mode]
    return _cpu_loops(*args, return_scope=scope, zero_shared=zero)


def _local_across_barrier(x, mode):
    """a = x[g]; s[l] = 2 a; barrier; out[g] = s[(l + 1) mod 4] + a. The value a is a local that lives across the barrier.
    mode: 'gpu' (the reference), 'naive' (a local re-created in loop 2, so it reads 0), 'promoted' (a becomes a per-workgroup array)."""
    n = len(x)
    out = [0.0] * n
    for w in range(n // WG):
        shared = [0.0] * WG
        keep = [0.0] * WG
        for l in range(WG):
            a = x[w * WG + l]
            shared[l] = 2 * a
            keep[l] = a
        for l in range(WG):
            a = {"naive": 0.0}.get(mode, keep[l])
            out[w * WG + l] = shared[(l + 1) % WG] + a
    return out


def cpu_phases():
    x = [float(i + 1) for i in range(10)]
    ref = _phase_run("gpu")
    names = [("gpu", "GPU semantics: barrier is a rendezvous, shared memory zeroed per workgroup, a return is permanent"),
             ("loops", "CPU loops, return kept in a mask across both loops, shared memory cleared per workgroup"),
             ("first-loop", "CPU loops, a return only skips the rest of the first loop"),
             ("stale", "CPU loops, shared memory reused without clearing"),
             ("both", "CPU loops, both mistakes")]
    rows = []
    for mode, label in names:
        out = _phase_run(mode)
        rows.append((label, [int(v) for v in out], _yes(out == ref)))
    xs = [float(i + 1) for i in range(8)]
    ref2 = _local_across_barrier(xs, "gpu")
    rows2 = [("GPU semantics", [int(v) for v in ref2], "yes"),
             ("CPU loops, the local re-created in loop 2", [int(v) for v in _local_across_barrier(xs, "naive")],
              _yes(_local_across_barrier(xs, "naive") == ref2)),
             ("CPU loops, the local promoted to a per-workgroup array", [int(v) for v in _local_across_barrier(xs, "promoted")],
              _yes(_local_across_barrier(xs, "promoted") == ref2))]
    return (f"The kernel `wg_sum` with workgroups of {WG}: each invocation with g < N stores x[g] in its slot of a shared array of {WG}; after the barrier invocation 0 stores the sum of the "
            f"{WG} slots in out[w]. The grid is padded: 5 workgroups are launched for N = 10 (3 are real), and `if w >= 3: return` (a test every invocation of a workgroup passes or fails together) "
            f"stands before the barrier. x = {[int(v) for v in x]}; an out slot nobody writes holds {int(SENTINEL)}. WGSL zeroes workgroup memory at the start of each workgroup, so the "
            f"last real workgroup, which fills 2 of its 4 slots, sums two zeros.\n\n"
            + _table(["execution", "out[0 .. 4]", "equals GPU semantics"], rows, "llr") + "\n\n"
            f"A local variable that lives across the barrier. Kernel: a = x[g]; s[l] = 2 a; barrier; out[g] = s[(l + 1) mod {WG}] + a, over x = {[int(v) for v in xs]} in 2 workgroups of {WG}.\n\n"
            + _table(["execution", "out[0 .. 7]", "equals GPU semantics"], rows2, "llr"))


def phase_figure_numbers():
    """What the figure of the barrier split prints: shared memory after loop 1 of workgroup 1, the group sums."""
    x = [float(i + 1) for i in range(10)]
    w = 1
    shared = [x[w * WG + l] for l in range(WG)]
    return {"x": x, "shared": shared, "sum": sum(shared), "wg": WG}


# ========================================== 5. JIT and AOT: keys and work


def _length_prefixed_key(fields):
    h = hashlib.sha256()
    for f in fields:
        b = f.encode("utf-8")
        h.update(struct.pack("<Q", len(b)))
        h.update(b)
    return h.hexdigest()


def _plain_key(fields):
    h = hashlib.sha256()
    for f in fields:
        h.update(f.encode("utf-8"))
    return h.hexdigest()


def _jit_fields(source, entry, target, compiler, flags, defines, abi):
    return [source, entry, target, compiler, flags, defines, f"abi{abi}"]


def cache_key():
    src = _cuda_source()
    base = _jit_fields(src, "brain_scale_shift", "sm_61", "nvrtc 12.9", "--fmad=false", "", 1)
    k0 = _length_prefixed_key(base)
    variants = [
        ("the same inputs again", base),
        ("one comment line appended to the source", [src + "// shift 1.5\n"] + base[1:]),
        ("another entry name", [base[0], "brain_other"] + base[2:]),
        ("another device architecture, sm_70", base[:2] + ["sm_70"] + base[3:]),
        ("another compiler version, nvrtc 13.0", base[:3] + ["nvrtc 13.0"] + base[4:]),
        ("a flag changed, --fmad=true", base[:4] + ["--fmad=true"] + base[5:]),
        ("a specialization macro, -DTILE=16", base[:5] + ["-DTILE=16"] + base[6:]),
        ("another launch ABI version", base[:6] + ["abi2"]),
    ]
    rows = [(label, _length_prefixed_key(f)[:12], _yes(_length_prefixed_key(f) == k0)) for label, f in variants]
    aot_fields = [src, "brain_scale_shift", "sm_61", "--gpu-architecture=sm_61 --fmad=false", "abi1"]
    pair_a, pair_b = ["ab", "c"], ["a", "bc"]
    coll = [("fields ('ab', 'c')", _plain_key(pair_a)[:12], _length_prefixed_key(pair_a)[:12]),
            ("fields ('a', 'bc')", _plain_key(pair_b)[:12], _length_prefixed_key(pair_b)[:12]),
            ("same key", _yes(_plain_key(pair_a) == _plain_key(pair_b)), _yes(_length_prefixed_key(pair_a) == _length_prefixed_key(pair_b)))]
    return (f"The key of one compiled kernel is the SHA-256 digest (first 12 hexadecimal digits shown) of its inputs, each written as an 8-byte little-endian length and then its UTF-8 bytes, in the "
            f"order: source text ({len(src.encode('utf-8'))} bytes, the CUDA text of `backends.three_paths`), entry name, device architecture, compiler version, flags, specialization macros, "
            f"launch ABI. Base inputs: entry `brain_scale_shift`, `sm_61`, `nvrtc 12.9`, `--fmad=false`, no macros, `abi1`.\n\n"
            + _table(["inputs", "key", "same key as the base"], rows, "llr") + "\n\n"
            f"An ahead-of-time image is looked up by a key without the compiler version (no compiler is present where it is loaded; the builder's version is recorded beside the image): the fields "
            f"source, entry, architecture, flags and ABI give the key {_length_prefixed_key(aot_fields)[:12]}.\n\n"
            "Why each field is length-prefixed: without the prefix two different splits of the same characters hash alike.\n\n"
            + _table(["inputs", "digest without lengths", "digest with lengths"], coll, "lrr"))


# the startup model: parameters, all labelled
N_KER, N_USED = 100, 12            # kernels registered by a program, kernels one run actually dispatches
S_SRC, S_IMG, S_PTX = 2400, 6000, 3000     # bytes of a kernel's source, of one compiled image, of its portable form
C_CMP = 500                        # compiler steps per source byte (a parameter of the model)


def _startup_rows():
    rows = []
    designs = (("compile every registered kernel at construction, empty cache", N_KER, True, "a compiler"),
               ("compile every registered kernel at construction, warm cache", N_KER, False, "a compiler on some earlier run"),
               ("compile a kernel at its first dispatch, empty cache", N_USED, True, "a compiler"),
               ("compile a kernel at its first dispatch, warm cache", N_USED, False, "a compiler on some earlier run"),
               ("load a prebuilt image for the device (ahead of time)", N_USED, False, "none"))
    for label, kernels, cold, need in designs:
        steps = C_CMP * S_SRC * kernels if cold else 0
        read = kernels * S_SRC                                  # the source is read to form the key
        read += 0 if cold else kernels * S_IMG                  # a hit reads the image back
        write = kernels * S_IMG if cold else 0                  # a miss stores the image
        rows.append((label, kernels, steps, read, write, need))
    return rows


def startup_work():
    rows = _startup_rows()
    body = [(r[0], _n(r[1]), _n(r[2]), _n(r[3]), _n(r[4]), r[5]) for r in rows]
    lane12 = (6, 1)
    lane13 = (3, 1)
    size12 = N_KER * (lane12[0] * S_IMG + lane12[1] * S_PTX)
    size13 = N_KER * (lane13[0] * S_IMG + lane13[1] * S_PTX)
    return (f"Parameters of the model (not measurements): N_ker = {N_KER} kernels registered, N_used = {N_USED} of them dispatched by a run, source s_src = {_n(S_SRC)} bytes, "
            f"one compiled image s_img = {_n(S_IMG)} bytes, a portable (PTX-like) form {_n(S_PTX)} bytes, a compiler spends c_cmp = {C_CMP} steps per source byte. Work of one start, in compiler steps "
            f"(c_cmp s_src for each kernel compiled), bytes read (each source is read to form its key; a hit also reads its image) and bytes written (a miss stores its image).\n\n"
            + _table(["design of the start", "kernels handled", "compiler steps", "bytes read", "bytes written", "toolchain needed at the target"], body, "lrrrrl") + "\n\n"
            f"Disk for prebuilt images: N_ker (N_tgt s_img + s_ptx) bytes for N_tgt targets plus one portable form: {_n(size12)} bytes for the six targets of the older toolkit lane "
            f"({_n(size12 // N_KER)} bytes per kernel) and {_n(size13)} for the three of the newer one ({_n(size13 // N_KER)} per kernel).")


LANES = {
    "toolkit 12": ([(6, 1), (7, 0), (8, 0), (8, 6), (8, 9), (9, 0)], (6, 1)),
    "toolkit 13": ([(9, 0), (10, 0), (12, 0)], (9, 0)),
}
DEVICES = [(6, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (8, 7), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)]


def _image_for(cc, cubins, ptx):
    """The loader's order: the cubin for the device's own architecture, then the cubins of lower minors of the same major,
    then portable code whose virtual architecture is not above the device's (the driver compiles it)."""
    for minor in range(cc[1], -1, -1):
        if (cc[0], minor) in cubins:
            return f"cubin sm_{cc[0]}{minor}" + ("" if minor == cc[1] else " (lower minor)")
    if ptx <= cc:
        return f"portable form, driver compiles (compute_{ptx[0]}{ptx[1]})"
    return "none"


def aot_coverage():
    rows = []
    for cc in DEVICES:
        row = [f"{cc[0]}.{cc[1]}"]
        for lane in LANES.values():
            row.append(_image_for(cc, *lane))
        rows.append(row)
    return ("Which prebuilt image a device gets, by the loader's rule: its own architecture, else a lower minor of the same major, else portable code whose virtual architecture is not above the "
            "device's. Cubins of the older toolkit lane: " + ", ".join(f"sm_{a}{b}" for a, b in LANES["toolkit 12"][0]) + f" and portable compute_{LANES['toolkit 12'][1][0]}{LANES['toolkit 12'][1][1]}; "
            "of the newer lane: " + ", ".join(f"sm_{a}{b}" for a, b in LANES["toolkit 13"][0]) + f" and portable compute_{LANES['toolkit 13'][1][0]}{LANES['toolkit 13'][1][1]}. "
            "'none' means a run needs the compiler at the device. Whether the driver accepts a lower-minor image is the driver's decision; the rule is the loader's order of attempts.\n\n"
            + _table(["device compute capability", "toolkit 12 images", "toolkit 13 images"], rows, "lll"))


# ============================================ 6. parity: the dot product


def _dot_data(n, count, gen):
    pairs = []
    for _ in range(count):
        xs = [_f32(2 * gen.uniform() - 1) for _ in range(n)]
        ys = [_f32(2 * gen.uniform() - 1) for _ in range(n)]
        pairs.append((xs, ys))
    return pairs


def _dot_left(xs, ys):
    acc = 0.0
    for a, b in zip(xs, ys):
        acc = _f32(acc + _f32(a * b))
    return acc


def _dot_left_fma(xs, ys):
    acc = 0.0
    for a, b in zip(xs, ys):
        acc = _fma(a, b, acc)
    return acc


def _dot_tree(xs, ys):
    vals = [_f32(a * b) for a, b in zip(xs, ys)]
    while len(vals) > 1:
        vals = [_f32(vals[i] + vals[i + 1]) for i in range(0, len(vals), 2)]
    return vals[0]


def _dot_exact(xs, ys):
    """The exact sum of the stored products: each product of two float32 numbers is exact in double precision,
    and math.fsum adds exactly (one rounding at the end, to double)."""
    return math.fsum(a * b for a, b in zip(xs, ys))


def _abs_sum(xs, ys):
    return math.fsum(abs(a * b) for a, b in zip(xs, ys))


def _bound_units(k):
    """The bound |computed - exact| <= k u / (1 - k u) S, as a multiple of u S: k / (1 - k u)."""
    return k / (1 - k * U)


_DOT_CACHE = {}


def _dot_stats():
    """For each length: per profile the largest error in units of u S, and the pairwise differences."""
    if _DOT_CACHE:
        return _DOT_CACHE["stats"]
    gen = bookrng.generator("backends.dot_parity")
    stats = []
    detail = {}
    for n in DOT_SIZES:
        data = _dot_data(n, DOT_VECTORS, gen)
        k_tree = 1 + math.ceil(math.log2(n))
        errs = {"L": [], "F": [], "T": []}
        same = {"LF": 0, "LT": 0, "FT": 0}
        diff = {"LF": 0.0, "LT": 0.0, "FT": 0.0}
        s_sum = 0.0
        records = []
        for xs, ys in data:
            exact = _dot_exact(xs, ys)
            s = _abs_sum(xs, ys)
            s_sum += s
            r = {"L": _dot_left(xs, ys), "F": _dot_left_fma(xs, ys), "T": _dot_tree(xs, ys)}
            records.append((xs, ys, exact, s, r))
            for p in "LFT":
                errs[p].append(abs(r[p] - exact) / (U * s))
            for pair in ("LF", "LT", "FT"):
                a, b = r[pair[0]], r[pair[1]]
                same[pair] += (a == b)
                diff[pair] = max(diff[pair], abs(a - b) / (U * s))
        stats.append({"n": n, "S": s_sum / len(data), "k": {"L": n, "F": n, "T": k_tree},
                      "err": {p: max(v) for p, v in errs.items()}, "same": same, "diff": diff,
                      "mean_err": {p: sum(v) / len(v) for p, v in errs.items()}})
        detail[n] = records
    _DOT_CACHE["stats"] = stats
    _DOT_CACHE["detail"] = detail
    return stats


def dot_parity():
    stats = _dot_stats()
    rows = []
    rows2 = []
    for st in stats:
        n = st["n"]
        b = {p: _bound_units(st["k"][p]) for p in "LFT"}
        rows.append((_n(n), _g(st["S"]),
                     _g(st["err"]["L"]), _g(b["L"]), _g(st["err"]["F"]), _g(b["F"]), _g(st["err"]["T"]), _g(b["T"])))
        pb = b["L"] + b["T"]
        rows2.append((_n(n), f"{st['same']['LF']} of {DOT_VECTORS}", f"{st['same']['LT']} of {DOT_VECTORS}", f"{st['same']['FT']} of {DOT_VECTORS}",
                      _g(st["diff"]["LF"]), _g(st["diff"]["LT"]), _g(pb), _yes(st["diff"]["LT"] <= pb and st["diff"]["LF"] <= 2 * b["L"])))
    return (f"The dot product of two vectors of n float32 numbers in [-1, 1), {DOT_VECTORS} seeded pairs per length n. {_seed_note('backends.dot_parity')} Draws in order, pair after pair, "
            f"lengths 16, 64, 256, 1024, 4096 in turn: the n numbers of x (each `2 uniform() - 1` rounded to float32), then the n of y. Three float32 programs: L, the left fold "
            "(each product rounded, then added in order); F, the left fold with a fused multiply-add per term (one rounding per term); T, each product rounded, then an adjacent-pair tree. "
            "The exact value is the sum of the stored products, each exact in double precision, added exactly (`math.fsum`). S = sum |x_i y_i| (mean over the pairs). Errors are |computed - exact| / (u S), u = 2^-24, the largest of the "
            f"{DOT_VECTORS} pairs; 'bound' is k / (1 - k u), the bound k u S / (1 - k u) in units of u S, with k the roundings on the longest path of the program (n for L and F; 1 + ceil(log2 n) for T).\n\n"
            + _table(["n", "S", "L: largest error", "L: bound", "F: largest error", "F: bound", "T: largest error", "T: bound"], rows) + "\n\n"
            "Agreement of the programs with one another on the same inputs. 'Pair bound' is the sum of the two bounds of L and T; the last column asks whether the observed differences "
            "stay within the sum of the two programs' bounds.\n\n"
            + _table(["n", "L = F bitwise", "L = T bitwise", "F = T bitwise", "largest |L - F| / (u S)", "largest |L - T| / (u S)", "pair bound (L, T)", "within the bounds"], rows2))


def parity_series():
    """For the figure: per n the largest errors and the bounds."""
    out = []
    for st in _dot_stats():
        out.append({"n": st["n"], "L": st["err"]["L"], "T": st["err"]["T"], "diff": st["diff"]["LT"],
                    "bL": _bound_units(st["k"]["L"]), "bT": _bound_units(st["k"]["T"]),
                    "mean_L": st["mean_err"]["L"], "mean_T": st["mean_err"]["T"]})
    return out


def _decimal(q, digits=22):
    """The rational q as a decimal string with `digits` significant digits (rounded half even)."""
    from decimal import Decimal, getcontext
    getcontext().prec = digits
    return str(Decimal(q.numerator) / Decimal(q.denominator))


def fma_example():
    gen = bookrng.generator("backends.fma_example")
    draws = 0
    while True:
        x, y, z = (_f32(2 * gen.uniform() - 1) for _ in range(3))
        draws += 1
        two = _f32(_f32(x * y) + z)
        one = _fma(x, y, z)
        if one != two:
            break
    exact = Fraction(x) * Fraction(y) + Fraction(z)
    prod = Fraction(x) * Fraction(y)
    rows = [("x", f"{x!r}", f"{_bits(x):#010x}"), ("y", f"{y!r}", f"{_bits(y):#010x}"), ("z", f"{z!r}", f"{_bits(z):#010x}"),
            ("x y, exact", _decimal(prod), "-"),
            ("f32(x y), the product rounded", f"{_f32(x * y)!r}", f"{_bits(_f32(x * y)):#010x}"),
            ("x y + z, exact", _decimal(exact), "-"),
            ("two roundings: f32(f32(x y) + z)", f"{two!r}", f"{_bits(two):#010x}"),
            ("one rounding: f32(x y + z)", f"{one!r}", f"{_bits(one):#010x}")]
    return (f"The first triple whose two evaluations differ. {_seed_note('backends.fma_example')} Draws in order: for each triple x, y, z, each `2 uniform() - 1` rounded to float32; "
            f"the triple is number {draws} drawn. The value column prints each float32 number exactly as the double-precision number that holds it (its shortest identifying decimal); 'exact' rows are the exact rational values to 22 significant digits.\n\n"
            + _table(["quantity", "value", "bits"], rows, "lll") + "\n\n"
            f"The two results are {_ulps(one, two)} float32 step{'s' if _ulps(one, two) != 1 else ''} apart.")


# ================================================ 7. parity: a transcendental


def exp_parity():
    gen = bookrng.generator("backends.exp_parity")
    count = 4000
    xs = [_f32(40 * gen.uniform() - 20) for _ in range(count)]
    log2e = _f32(math.log2(math.e))
    bands = [(0, 1), (1, 4), (4, 8), (8, 16), (16, 20.001)]
    rows = []
    for lo, hi in bands:
        sel = [x for x in xs if lo <= abs(x) < hi]
        worst = 0.0
        total = 0.0
        max_ulp = 0
        same = 0
        for x in sel:
            ref = _f32(math.exp(x))
            t = _f32(x * log2e)
            fast = _f32(2.0 ** t)
            rel = abs(fast - ref) / ref / U
            worst = max(worst, rel)
            total += rel
            max_ulp = max(max_ulp, _ulps(ref, fast))
            same += (ref == fast)
        rows.append((f"{lo:g} to {min(hi, 20):g}", len(sel), same, max_ulp, _g(total / len(sel)), _g(worst), _g(2 * min(hi, 20) + 2)))
    return (f"exp(x) for {count} seeded x. {_seed_note('backends.exp_parity')} Draws in order: x_i = `40 uniform() - 20` rounded to float32. The reference rounds the double-precision library "
            "exp(x) to float32. The 'fast' path computes t = x L in float32 with L = float32(log2 e) = " + f"{log2e!r}" + ", then 2^t in double precision rounded to float32 (the shape of a "
            "device exp built on a base-2 instruction). Relative difference |fast - ref| / ref in units of u; 'ulps' is the distance in float32 steps; the bound is (2 |x| + 2) u at the upper end of the band: "
            "t carries a relative error of about 2 u (one rounding of L, one of the product), which 2^t multiplies by ln 2 |t| = |x|, plus one final rounding on each side.\n\n"
            + _table(["|x| in", "values", "bitwise equal", "largest ulps", "mean difference / u", "largest difference / u", "bound / u"], rows))


# ========================================== 8. a test that can see a defect


def defect_detection():
    _dot_stats()
    detail = _DOT_CACHE["detail"]
    rows = []
    for n in (1024, 4096):
        tols = {
            "fixed 1e-6": lambda s: 1e-6,
            "loose, 1e-3 S": lambda s: 1e-3 * s,
            "derived, tree, k = 1 + ceil(log2 n)": lambda s: _bound_units(1 + math.ceil(math.log2(n))) * U * s,
            "derived, left fold, k = n": lambda s: _bound_units(n) * U * s,
        }
        for label, tol in tols.items():
            false_alarm = detected = 0
            for xs, ys, exact, s, r in detail[n]:
                bad = _dot_tree(xs[:-1] + [0.0], ys[:-1] + [0.0])
                t = tol(s)
                false_alarm += abs(r["T"] - exact) > t
                detected += abs(bad - exact) > t
            rows.append((_n(n), label, f"{false_alarm} of {DOT_VECTORS}", f"{detected} of {DOT_VECTORS}"))
    return (f"The device under test runs program T of `backends.dot_parity` on the same {DOT_VECTORS} seeded pairs. The defect is the under-dispatch of Chapter 5: the last product x_(n-1) y_(n-1) "
            "is never added (replaced by 0 in the tree). A test compares the device's result with the exact value and flags it when the difference exceeds its tolerance; the tolerances are a fixed "
            "absolute 1e-6, 1e-3 S, and two derived ones: k u S / (1 - k u) for the tree (k = 1 + ceil(log2 n)) and for a left fold (k = n). 'False alarms' count correct results flagged; "
            "'detected' counts defective results flagged.\n\n"
            + _table(["n", "tolerance", "false alarms (correct device)", "defect detected"], rows, "llrr"))


# ============================================== 9. the parity matrix


def _exp_ref(x):
    return _f32(math.exp(x))


def _exp_fast(x):
    t = _f32(x * _f32(math.log2(math.e)))
    return _f32(2.0 ** t)


def _transpose_demo(m=6, n=10):
    a = [[float(10 * i + j) for j in range(n)] for i in range(m)]
    return [[a[i][j] for i in range(m)] for j in range(n)]


def parity_matrix():
    gen = bookrng.generator("backends.parity_matrix")
    n_el = 4096
    xs = [_f32(2 * gen.uniform() - 1) for _ in range(n_el)]
    a_coef, b_coef = _f32(1.7), _f32(-0.3)
    # kernel 1: y = a x + b, the reference rounds the product, the contracting device does not
    ref1 = [_f32(_f32(a_coef * v) + b_coef) for v in xs]
    con1 = [_fma(a_coef, v, b_coef) for v in xs]
    # kernel 2: a transpose moves values and rounds nothing
    ref2 = _transpose_demo()
    con2 = _transpose_demo()
    # kernel 3: sum of 4 slots (adds only, order fixed by the kernel)
    sl = [_f32(2 * gen.uniform() - 1) for _ in range(n_el)]
    ref3 = [_f32(_f32(sl[i] + sl[i + 1]) + _f32(sl[i + 2] + sl[i + 3])) for i in range(0, n_el, 4)]
    con3 = list(ref3)
    # kernel 4: 64-term dot product, tree of adjacent pairs, one product per leaf
    pa = [_f32(2 * gen.uniform() - 1) for _ in range(n_el)]
    pb = [_f32(2 * gen.uniform() - 1) for _ in range(n_el)]
    ref4, con4 = [], []
    for i in range(0, n_el, 64):
        ref4.append(_dot_tree(pa[i:i + 64], pb[i:i + 64]))
        leaves = [_fma(pa[i + j + 1], pb[i + j + 1], _f32(pa[i + j] * pb[i + j])) for j in range(0, 64, 2)]
        while len(leaves) > 1:
            leaves = [_f32(leaves[j] + leaves[j + 1]) for j in range(0, len(leaves), 2)]
        con4.append(leaves[0])
    # kernel 5: exp
    ex = [_f32(16 * gen.uniform() - 8) for _ in range(n_el)]
    ref5 = [_exp_ref(v) for v in ex]
    fast5 = [_exp_fast(v) for v in ex]

    def cell(ref, other, scale=None):
        flat_r = [v for row in ref for v in row] if isinstance(ref[0], list) else ref
        flat_o = [v for row in other for v in row] if isinstance(other[0], list) else other
        diff = sum(1 for p, q in zip(flat_r, flat_o) if p != q)
        if diff == 0:
            return f"identical ({len(flat_r)} of {len(flat_r)})"
        worst = max(_ulps(p, q) for p, q in zip(flat_r, flat_o))
        text = f"{diff} of {len(flat_r)} differ, at most {worst} ulp"
        if scale is not None:
            text += f", at most {max(abs(p - q) / (U * sc) for p, q, sc in zip(flat_r, flat_o, scale)):.2f} u S"
        return text

    scale1 = [abs(a_coef * v) + abs(b_coef) for v in xs]
    scale4 = [math.fsum(abs(pa[i + j] * pb[i + j]) for j in range(64)) for i in range(0, n_el, 64)]
    scale5 = [abs(v) for v in ref5]

    rows = [
        ("y = a x + b, 4096 values", cell(ref1, ref1), cell(ref1, con1, scale1), "not applicable"),
        ("transpose of a [6, 10] matrix", cell(ref2, ref2), cell(ref2, con2), "not applicable"),
        ("sum of 4 slots, 1024 outputs", cell(ref3, ref3), cell(ref3, con3), "not applicable"),
        ("dot product of 64 terms, tree, 64 outputs", cell(ref4, ref4), cell(ref4, con4, scale4), "not applicable"),
        ("exp(x), x in [-8, 8), 4096 values", cell(ref5, ref5), "not applicable", cell(ref5, fast5, scale5)),
    ]
    return (f"Three modelled arithmetic profiles on the same seeded inputs. {_seed_note('backends.parity_matrix')} Draws in order: 4096 x for y = a x + b (`2 uniform() - 1` rounded to float32; a = 1.7 and "
            f"b = -0.3 rounded to float32); 4096 slot values; 4096 pa then 4096 pb for the dot products; 4096 x for exp (`16 uniform() - 8`, rounded to float32). "
            "Profile P0 (the reference, and what a CPU loop nest and a translator that forbids contraction compute): every operation rounds to float32 in the order the kernel states, exp is the "
            "double-precision library value rounded to float32. Profile P1 (a device that contracts a product followed by an add into a fused multiply-add where the kernel has one). Profile P2 (a device "
            "whose exp goes through a base-2 instruction, as in `backends.exp_parity`). Each cell compares a profile with P0 on bit patterns; where values differ it also prints the largest difference in units of u S, with S = |a x| + |b| for y = a x + b, the sum of the absolute values of the 64 products for a dot product and |exp(x)| for exp.\n\n"
            + _table(["kernel", "P0 (reference)", "P1: contracts", "P2: fast exp"], rows, "llll"))
