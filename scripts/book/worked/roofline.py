# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Memory Bandwidth, Compute and the Roofline".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: roofline.<name> -->` markers. Helpers with a leading underscore,
or that take arguments, are shared with the chapter's figures so a picture and
the table beside it cannot disagree.

Nothing here measures a machine. The two machines are sets of round PARAMETERS
(a peak arithmetic rate and a memory bandwidth), invented for the book and not
claims about any product. Everything else is counting: FLOPs (a multiplication
or an addition is one, a fused multiply-add is two), bytes that must cross the
boundary between main memory and the arithmetic, and the two lower bounds on
time that follow from the parameters. Where a time or a tokens-per-second
figure appears it is the model's output from the stated parameters, never an
observation. No example here uses randomness.

Swedish Embedded AB implements solutions for reaching the memory and compute
roofline of whatever accelerator a model lands on. If your team needs expertise
in performance analysis of machine learning inference and training, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math

# ------------------------------------------------------------------ parameters

# name: (peak arithmetic rate in GFLOP/s, memory bandwidth in GB/s). Round, illustrative.
MACHINES = {"CPU": (800.0, 80.0), "GPU": (40000.0, 800.0)}
F32 = 4                      # bytes per fp32 number
DECODE_PARAMS = 7_000_000_000   # the dense model of chapter 1
# bytes per parameter of the weight formats of chapter 1 (scales charged in full)
FORMATS = (("fp32", 4.0), ("fp16", 2.0), ("int8, f32 scale per 32", 1.125), ("int4, f32 scale per 32", 0.625))
LAUNCH_SECONDS = 5e-6        # the model's fixed cost of starting one GPU dispatch (parameter)

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


# ------------------------------------------------------------------- the model


def ridge(machine):
    """FLOP per byte at which the two bounds meet: peak / bandwidth."""
    peak, bw = MACHINES[machine]
    return peak / bw


def attainable(machine, intensity):
    """Attainable GFLOP/s at a given intensity: min(peak, intensity * bandwidth)."""
    peak, bw = MACHINES[machine]
    return min(peak, intensity * bw)


def bound_of(machine, intensity):
    return "bandwidth" if intensity < ridge(machine) else "compute"


# Counts of the three operations of the chapter. Each returns (FLOPs, bytes).


def vector_add(n):
    """c = a + b for fp32 vectors of n: n additions; read a and b, write c."""
    return n, 3 * F32 * n


def gemv(m, k, weight_bytes):
    """y = W x, W of shape [m, k] stored at `weight_bytes` per entry, x and y in fp32."""
    return 2 * m * k, weight_bytes * m * k + F32 * (k + m)


def gemm(m, k, n, weight_bytes=F32):
    """C = A B, A [m, k] activations in fp32, B [k, n] weights at `weight_bytes`, C in fp32."""
    return 2 * m * k * n, F32 * m * k + weight_bytes * k * n + F32 * m * n


def _operations():
    n = 1 << 20
    side = 4096
    return [
        ("c = a + b, vectors of 1,048,576", *vector_add(n)),
        ("y = W x, [4096, 4096], fp32 weights", *gemv(side, side, 4)),
        ("y = W x, [4096, 4096], int8 weights", *gemv(side, side, 1)),
        ("C = A B, [4096, 4096] by [4096, 4096], fp32", *gemm(side, side, side)),
    ]


# ---------------------------------------------------------------- the problem


def decode_two_times():
    peak, bw = MACHINES["GPU"]
    p = DECODE_PARAMS
    flops, nbytes = 2 * p, 2 * p
    t_arith = flops / (peak * 1e9)
    t_mem = nbytes / (bw * 1e9)
    rows = [
        ("FLOPs for one token (2 per weight)", _n(flops)),
        ("bytes of weights read for one token (2 per weight)", _n(nbytes)),
        ("seconds the arithmetic needs at the peak rate", f"{t_arith:.6f}"),
        ("seconds the weights need to arrive at the bandwidth", f"{t_mem:.6f}"),
        ("the second divided by the first", f"{t_mem / t_arith:.1f}"),
    ]
    return (
        "One token of a dense model with P = 7,000,000,000 weights stored as fp16 (2 bytes), batch size 1: every weight is read once and used in one "
        "multiplication and one addition. The machine is the illustrative GPU of the next section: F_pk = 40,000 (GFLOP/s) and BW = 800 (GB/s). "
        "Both times are the work divided by the stated rate, a model and not a measurement.\n\n"
        + _table(["quantity", "value"], rows, "lr")
    )


# -------------------------------------------------------------------- machines


def machines():
    rows = []
    for name, (peak, bw) in MACHINES.items():
        rows.append((name, _g(peak, 0), _g(bw, 0), _g(peak / bw, 1)))
    return (
        "Model parameters, round numbers invented for the book (not any product). Peak is the rate of fp32 arithmetic in GFLOP/s (a multiply and an add "
        "count 2); bandwidth is bytes from main memory per second in GB/s (1 GB = 10^9 bytes); the ridge is the peak divided by the bandwidth.\n\n"
        + _table(["machine", "peak F_pk (GFLOP/s)", "bandwidth BW (GB/s)", "ridge F_pk / BW (FLOP/byte)"], rows)
    )


# ------------------------------------------------------------ three operations


def three_operations():
    rows = []
    for name, flops, nbytes in _operations():
        rows.append((name, _n(flops), _n(nbytes), f"{flops / nbytes:.4f}"))
    return (
        "FLOPs count each multiplication and each addition as one. Bytes are the least that must cross between main memory and the arithmetic: every input "
        "read once, every output written once; vectors and matrices of activations are fp32 (4 bytes); weights are fp32 or int8 (1 byte, scales ignored) as "
        "stated. The matrix product moves A, B and C once each.\n\n"
        + _table(["operation", "FLOPs", "bytes", "intensity (FLOP/byte)"], rows)
    )


def roofline_positions():
    rows = []
    for name, flops, nbytes in _operations():
        i = flops / nbytes
        cells = [name, f"{i:.4f}"]
        for m in MACHINES:
            a = attainable(m, i)
            cells += [_g(a, 1), f"{a / MACHINES[m][0]:.4f}", bound_of(m, i)]
        rows.append(cells)
    return (
        "Attainable GFLOP/s = min(F_pk, intensity x BW) on the two machines of `roofline.machines` (CPU: F_pk = 800, BW = 80, ridge 10; GPU: F_pk = 40,000, BW = 800, ridge 50, in GFLOP/s and GB/s)"
        ", for the four operations of `roofline.three_operations`. The fraction is of the machine's peak arithmetic rate.\n\n"
        + _table(["operation", "intensity", "CPU attainable", "CPU fraction of peak", "CPU bound", "GPU attainable", "GPU fraction of peak", "GPU bound"], rows,
                 "lrrrlrrl")
    )


# ------------------------------------------------------------ size and growth


def _gemm_square_intensity(n):
    flops, nbytes = gemm(n, n, n)
    return flops / nbytes


def intensity_by_size():
    sizes = (16, 64, 256, 1024, 4096, 16384)
    rows = []
    for n in sizes:
        f, b = gemm(n, n, n)
        rows.append((_n(n), _n(f), _n(b), f"{f / b:.3f}", f"{n / 6:.3f}"))
    first = {m: min(n for n in range(1, 100000) if _gemm_square_intensity(n) >= ridge(m)) for m in MACHINES}
    return (
        "C = A B for square fp32 matrices of side n: FLOPs 2 n^3, bytes 4 (3 n^2) = 12 n^2 (A, B read once, C written once). The last column is the closed form n / 6.\n\n"
        + _table(["n", "FLOPs", "bytes", "intensity", "n / 6"], rows)
        + "\n\nThe smallest side n for which the square product reaches the ridge: "
        + f"CPU (ridge {ridge('CPU'):.0f}) n = {first['CPU']}; GPU (ridge {ridge('GPU'):.0f}) n = {first['GPU']}."
    )


# ------------------------------------------------------------------- decoding


def decode_per_token():
    p = DECODE_PARAMS
    rows = []
    for name, b in FORMATS:
        nbytes = p * b
        i = 2 * p / nbytes
        row = [name, f"{b:.4f}", _g(nbytes / 1e9, 3), f"{i:.2f}"]
        for m in MACHINES:
            peak, bw = MACHINES[m]
            row += [f"{attainable(m, i) / peak:.4f}", _g(bw * 1e9 / nbytes, 1)]
        rows.append(row)
    return (
        "A dense model of P = 7,000,000,000 weights at batch size 1: one token is 2 P FLOPs and reads every weight once, so the bytes per token are P b for b bytes "
        "per weight (scales charged, as in Volume IV, Chapter 1) and the intensity is 2 / b. Activations and the cache of earlier tokens are ignored. Arithmetic is fp32 "
        "at the machine's peak whatever the storage format. The last column of each machine is the roofline's bound on tokens per second, BW / (P b): a model, not a measurement. GB = 10^9 bytes.\n\n"
        + _table(["weights stored as", "bytes per weight", "GB read per token", "intensity", "CPU fraction of peak", "CPU tokens/s bound", "GPU fraction of peak", "GPU tokens/s bound"], rows,
                 "lrrrrrrr")
    )


def _layer_intensity(batch, side, b):
    """[batch, side] by [side, side] with weights at b bytes, activations in and out in fp32."""
    flops = 2 * batch * side * side
    nbytes = b * side * side + F32 * batch * (side + side)
    return flops / nbytes


def _smallest_batch(side, b, machine):
    r = ridge(machine)
    return next(B for B in range(1, 100000) if _layer_intensity(B, side, b) >= r)


def batch_ladder():
    side = 4096
    fmts = (("fp32", 4.0), ("int8", 1.125), ("int4", 0.625))
    rows = []
    for B in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        row = [str(B)]
        for _, b in fmts:
            row.append(f"{_layer_intensity(B, side, b):.2f}")
        for _, b in fmts:
            i = _layer_intensity(B, side, b)
            row.append(f"{attainable('GPU', i) / MACHINES['GPU'][0]:.4f}")
        rows.append(row)
    first = []
    for name, b in fmts:
        first.append((name, f"{b:.3f}", _smallest_batch(side, b, "CPU"), _smallest_batch(side, b, "GPU")))
    return (
        "One layer [B, 4096] by [4096, 4096]: B rows of activations (fp32) times a weight matrix stored at b bytes per weight (fp32 b = 4; int8 and int4 with an f32 scale per 32 weights, "
        "b = 1.125 and 0.625). FLOPs 2 B (4096)^2; bytes b (4096)^2 + 4 B (4096 + 4096). Percent of the GPU's peak at the attainable rate min(F_pk, intensity x BW).\n\n"
        + _table(["B", "I fp32", "I int8", "I int4", "GPU fraction of peak fp32", "GPU fraction of peak int8", "GPU fraction of peak int4"], rows)
        + "\n\nThe smallest batch B whose intensity reaches the ridge:\n\n"
        + _table(["weights", "bytes per weight b", "CPU (ridge 10)", "GPU (ridge 50)"], first)
    )


def whole_model_batch():
    p = DECODE_PARAMS
    peak, bw = MACHINES["GPU"]
    fmts = (("fp16", 2.0), ("int4, f32 scale per 32", 0.625))

    def tokens_per_s(batch, b):
        t = max(2 * p * batch / (peak * 1e9), p * b / (bw * 1e9))
        return batch / t

    rows = []
    for B in (1, 4, 16, 32, 64, 128, 256):
        cells = [str(B)]
        for _, b in fmts:
            i = 2 * B / b
            cells += [f"{i:.1f}", bound_of("GPU", i), _g(tokens_per_s(B, b), 0)]
        cells.append(f"{tokens_per_s(B, fmts[1][1]) / tokens_per_s(B, fmts[0][1]):.2f}")
        rows.append(cells)
    ceil = {name: math.ceil(ridge("GPU") * b / 2) for name, b in fmts}
    return (
        "The 7,000,000,000-weight model on the GPU (F_pk = 40,000, BW = 800, in GFLOP/s and GB/s), batch B tokens at once: FLOPs 2 P B, weight bytes P b (read once for the whole batch; activations and "
        "the cache of earlier tokens ignored), intensity 2 B / b, tokens per second = B / max(2 P B / F_pk, P b / BW) from the roofline's two bounds (a model).\n\n"
        + _table(["B", "I fp16", "fp16 bound", "fp16 tokens/s", "I int4", "int4 bound", "int4 tokens/s", "int4 / fp16"], rows, "rrlrrlrr")
        + f"\n\nSmallest B reaching the ridge, ceil(ridge x b / 2): fp16 {ceil['fp16']}, int4 {ceil['int4, f32 scale per 32']}."
    )


# ------------------------------------------------------- training operations


def training_operations():
    def layer(B, D, O):
        return 2 * B * D * O, F32 * (B * D + O * D + B * O)

    items = []
    for B in (64, 8192):
        f, b = layer(B, 784, 256)
        items.append((f"linear layer forward, B = {B}, D = 784, O = 256", f, b))
    items.append(("linear layer backward dX or dW, B = 64", *layer(64, 784, 256)))
    items.append(("AdamW update, per parameter", 16, 28))
    items.append(("SGD update, per parameter", 2, 12))
    rows = []
    for name, f, b in items:
        i = f / b
        cells = [name, _n(f), _n(b), f"{i:.2f}"]
        for m in MACHINES:
            cells += [f"{attainable(m, i) / MACHINES[m][0]:.4f}", bound_of(m, i)]
        rows.append(cells)
    return (
        "FLOPs and bytes as counted in Volume III, Chapters 5 and 7 (fp32; a layer reads X, W and writes Y once; an update reads w, g and the state and writes w and the state). "
        "Percent of peak at min(F_pk, intensity x BW) on the two machines of `roofline.machines`.\n\n"
        + _table(["operation", "FLOPs", "bytes", "intensity", "CPU fraction of peak", "CPU bound", "GPU fraction of peak", "GPU bound"], rows, "lrrrrlrl")
    )


# ----------------------------------------------------------------- data reuse


def _tile_row(n, t):
    ws = F32 * (t * t + 2 * t)
    nbytes = F32 * (2 * n ** 3 // t + n * n)
    return ws, nbytes, 2 * n ** 3 / nbytes


def _level(ws):
    for name, cap in (("L1 (32 KiB)", 32 * 1024), ("L2 (1 MiB)", 1 << 20), ("L3 (32 MiB)", 32 << 20)):
        if ws <= cap:
            return name
    return "none"


def tiling_traffic():
    n = 4096
    rows = []
    for t in (1, 8, 64, 256, 1024, 2048):
        ws, nbytes, i = _tile_row(n, t)
        rows.append((str(t), _n(ws), _level(ws), _n(nbytes), f"{i:.2f}",
                     f"{attainable('CPU', i) / MACHINES['CPU'][0]:.4f}",
                     f"{attainable('GPU', i) / MACHINES['GPU'][0]:.4f}"))
    f, b = gemm(n, n, n)
    rows.append(("whole matrices", "-", "-", _n(b), f"{f / b:.2f}",
                 f"{attainable('CPU', f / b) / MACHINES['CPU'][0]:.4f}",
                 f"{attainable('GPU', f / b) / MACHINES['GPU'][0]:.4f}"))
    return (
        "C = A B for n = 4096, fp32. The cache holds a T by T tile of C and, for each k, the T numbers of A and the T numbers of B that update it: a working set of 4 (T^2 + 2 T) bytes. "
        "Main-memory traffic is 4 (2 n^3 / T + n^2) bytes (A and B re-read for every tile, C written once); FLOPs 2 n^3. Caches of Volume IV, Chapter 2: L1 32 KiB, L2 1 MiB, L3 32 MiB. "
        "The last two columns are percent of peak at min(F_pk, intensity x BW) for the two machines of `roofline.machines`; the model gives each machine the same cache, to isolate the effect of reuse.\n\n"
        + _table(["T", "working set (bytes)", "smallest level that holds it", "DRAM bytes", "intensity", "CPU fraction of peak", "GPU fraction of peak"], rows,
                 "rrlrrrr")
    )


# ---------------------------------------------------------- achieved fraction


# Invented stopwatch readings, seconds, for one hypothetical run per machine and operation.
READINGS = {
    ("GEMV", "CPU"): 1.00e-3, ("GEMV", "GPU"): 0.30e-3,
    ("GEMM", "CPU"): 0.40, ("GEMM", "GPU"): 5.0e-3,
}


def achieved_fraction():
    ops = {"GEMV": gemv(4096, 4096, 4), "GEMM": gemm(4096, 4096, 4096)}
    rows = []
    for op in ("GEMV", "GEMM"):
        flops, nbytes = ops[op]
        i = flops / nbytes
        for m in MACHINES:
            peak, bw = MACHINES[m]
            t = READINGS[(op, m)]
            rate = flops / t / 1e9
            bound = attainable(m, i)
            rows.append((op, m, f"{t * 1e3:.2f}", _g(rate, 1), _g(nbytes / t / 1e9, 1), _g(bound, 1),
                         f"{rate / bound:.4f}", f"{rate / peak:.4f}"))
    return (
        "HYPOTHETICAL stopwatch readings, invented for the example and not measurements: fp32 y = W x with W [4096, 4096] (33,554,432 FLOPs, 67,141,632 bytes, intensity 0.4998) and "
        "fp32 C = A B with all sides 4096 (137,438,953,472 FLOPs, 201,326,592 bytes, intensity 682.67). The machines are those of `roofline.machines`. Achieved rate = FLOPs / time; "
        "achieved bandwidth = bytes / time; roofline bound = min(F_pk, intensity x BW); fraction of the roofline = achieved rate / bound.\n\n"
        + _table(["operation", "machine", "reading (ms)", "achieved GFLOP/s", "achieved GB/s", "roofline bound (GFLOP/s)", "fraction of the roofline", "fraction of peak compute"], rows,
                 "llrrrrrr")
    )


# ----------------------------------------------------------------- the limits


def small_problems():
    peak, bw = MACHINES["GPU"]
    rows = []
    for k in (10, 14, 18, 20, 22, 24, 26):
        n = 1 << k
        nbytes = 12 * n
        t = nbytes / (bw * 1e9)
        eff = t / (t + LAUNCH_SECONDS)
        rows.append((f"2^{k}", _n(n), _n(nbytes), f"{t * 1e6:.2f}", f"{(t + LAUNCH_SECONDS) * 1e6:.2f}", f"{eff:.4f}"))
    half = LAUNCH_SECONDS * bw * 1e9 / 12
    return (
        "c = a + b on the GPU of `roofline.machines` (BW = 800, in GB/s). Roofline time = 12 n bytes / BW. The model adds a fixed cost of 5 microseconds to start the dispatch (a parameter, not a measurement), "
        "so the fraction of the roofline reached is roofline time / (roofline time + 5 microseconds).\n\n"
        + _table(["n", "elements", "bytes", "roofline time (microseconds)", "with the 5 microsecond start (microseconds)", "fraction of the roofline"], rows)
        + f"\n\nThe fraction is one half when the roofline time equals the start cost: n = 5e-6 x 800e9 / 12 = {half:,.0f} elements."
    )
