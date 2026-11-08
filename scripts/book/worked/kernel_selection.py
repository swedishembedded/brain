# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Choosing the Fastest Correct Kernel".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: kernel_selection.<name> -->` markers. Helpers with a leading
underscore, or that take arguments, are shared with the chapter's figures so a
picture and the table beside it cannot disagree.

Nothing here times a real machine. The "device" is a small cost model of the
model GPU of Chapters 3 and 4 (and the model CPU of Chapter 3), and a "run" of
a kernel on it is the model's cost times a seeded noise factor. Costs are in
EQUIVALENT BYTES (Chapter 8): the roofline time multiplied by the bandwidth,
so a cost carries no seconds. The few constants of the model (the efficiency
of an instruction mix, the loads an invocation keeps in flight, the cost of a
launch) are labelled parameters and printed by the chapter. The matrix
products of the reference gate are real arithmetic: float32 rounding after
every operation, an exact reference, the tolerance of Chapter 9.

Randomness comes from the book's own generator
(`bookrng.generator("kernel_selection.<example>")`), is printed with its seed,
and is drawn in an order the chapter states.

Swedish Embedded AB implements solutions for GPU kernels that choose
themselves on hardware nobody tuned them for, and prove they are right
before they are fast. If your team needs expertise in kernel selection and
autotuning for machine learning inference, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import struct

import bookrng

# ---------------------------------------------------------------- formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _n(v):
    return f"{v:,}"


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (the seed of the name `{name}`)."


def _f32(x):
    """The float32 nearest to x (ties to even), as a Python float."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---------------------------------------------------------------- the devices

# Round numbers, parameters of the book. `spec_*` are the datasheet-style
# numbers of Chapters 3 and 4. `flops` (GFLOP/s) and `bw` (GB/s) are what the
# silicon of the one unit used here really reaches: the values the calibration
# of `calibration` recovers. The CPU launches (wakes its thread pool) faster
# than the GPU; both are parameters.
DEVICES = {
    "GPU": dict(spec_flops=40000.0, spec_bw=800.0, flops=36000.0, bw=664.0, lanes=10240, lat_ns=200.0, launch_us=5.0,
                workgroup_barriers=True, shared_kib=16, max_wg=256, matrix_unit=True),
    "CPU": dict(spec_flops=800.0, spec_bw=80.0, flops=640.0, bw=60.0, lanes=160, lat_ns=100.0, launch_us=2.0,
                workgroup_barriers=False, shared_kib=0, max_wg=1, matrix_unit=False),
}


def _ridge(dev):
    return dev["flops"] / dev["bw"]


def _launch_bytes(dev):
    """The cost of starting one dispatch alone, in equivalent bytes: t0 BW."""
    return dev["launch_us"] * 1e-6 * dev["bw"] * 1e9


def _loads_needed(dev):
    """Invocations with one 4-byte load in flight that keep the memory busy: BW t_lat / 4 (Chapter 4)."""
    return dev["bw"] * 1e9 * dev["lat_ns"] * 1e-9 / 4.0


# --------------------------------------------------------------- the candidates

# name, tile BM x BN (outputs one workgroup owns), invocations per workgroup,
# register block per invocation (rows x cols), loads in flight per invocation,
# workgroup memory in KiB. The issue efficiency e is the count of Chapter 6:
# useful fused multiply-adds divided by all instructions of one step of k, with
# 2 instructions of loop overhead.
CANDIDATES = {
    "naive": dict(bm=1, bn=1, n_wg=64, rb=(1, 1), ell=2, shared_kib=0, barriers=False),
    "tiled": dict(bm=32, bn=32, n_wg=64, rb=(4, 4), ell=8, shared_kib=2, barriers=True),
    "reg": dict(bm=128, bn=128, n_wg=256, rb=(8, 8), ell=8, shared_kib=8, barriers=True),
}


def _efficiency(c):
    """FMAs of one step of k over all instructions of it: loads from the two slabs, the FMAs, 2 of overhead."""
    r, s = c["rb"]
    if c["bm"] == 1:
        return 1.0 / (1 + 2 + 2)          # one FMA, two loads from memory, loop overhead
    return r * s / (r * s + r + s + 2)     # r + s reads of the slabs for r s FMAs


def _counts(c, m, n, k):
    """The Chapter 6 counts of one candidate on C[m, n] = A[m, k] B[k, n]."""
    tm, tn = -(-m // c["bm"]), -(-n // c["bn"])
    if c["bm"] == 1:
        flop_pad = 2 * m * n * k
        n_inv = -(-(m * n) // c["n_wg"]) * c["n_wg"]
    else:
        flop_pad = 2 * (tm * c["bm"]) * (tn * c["bn"]) * k
        n_inv = tm * tn * c["n_wg"]
    n_byte = 4 * (m * k * tn + k * n * tm + m * n)       # operand requests, one pass per tile; C written once
    return dict(tm=tm, tn=tn, flop_pad=flop_pad, n_byte=n_byte, n_inv=n_inv)


def _floor(dev, m, n, k):
    """The roofline cost of the problem itself: every operand once, 2 m n k FLOPs, whatever the kernel."""
    return max(4 * (m * k + k * n + m * n), 2 * m * n * k / _ridge(dev))


def _predicted(dev, c, m, n, k):
    """The roofline model's cost of a candidate (Chapters 3 and 8): its own bytes against the useful FLOPs."""
    return max(_counts(c, m, n, k)["n_byte"], 2 * m * n * k / _ridge(dev))


def _cost(dev, c, m, n, k):
    """The cost on the device, noise-free: the launch, then the slower of the padded arithmetic done by the
    invocations that exist and the requested bytes moved by the loads in flight."""
    q = _counts(c, m, n, k)
    o_cmp = min(1.0, q["n_inv"] / dev["lanes"])
    o_mem = min(1.0, q["n_inv"] * c["ell"] / _loads_needed(dev))
    return _launch_bytes(dev) + max(q["flop_pad"] / _ridge(dev) / (_efficiency(c) * o_cmp), q["n_byte"] / o_mem)


# the defective candidate of the reference gate: the register-blocked kernel
# with its row guard removed, so it computes only whole tiles of rows.
NOTAIL = dict(CANDIDATES["reg"])


def _cost_of(dev, name, m, n, k):
    if name == "reg-notail":
        rows = NOTAIL["bm"] * (m // NOTAIL["bm"])
        return _cost(dev, NOTAIL, rows, n, k) if rows > 0 else _launch_bytes(dev)
    return _cost(dev, CANDIDATES[name], m, n, k)


# ------------------------------------------------------------------ the noise

SPIKE_P = 0.2          # one run in five is hit by contention
SPIKE_MEAN = 0.25      # a spike lengthens the run by 25 percent on average
JITTER = 0.04          # every run is lengthened by a uniform [0, 4 percent)


def _run_factor(gen):
    """One run's factor over the device's floor: exactly three uniform draws, u0 u1 u2.
    factor = (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2 else 1). A run never beats the floor."""
    u0, u1, u2 = gen.uniform(), gen.uniform(), gen.uniform()
    spike = -SPIKE_MEAN * math.log(1.0 - u2) if u1 < SPIKE_P else 0.0
    return (1.0 + JITTER * u0) * (1.0 + spike)


def _min_of(gen, floor, runs):
    return min(floor * _run_factor(gen) for _ in range(runs))


def _batch_estimate(gen, floor, batches=5, per_batch=4):
    """The estimate of the chapter's protocol: the mean of `batches` batch minima of `per_batch` launches, its
    standard error s / sqrt(batches), and the device work spent (the sum of the launches' costs)."""
    vals, spent = [], 0.0
    for _ in range(batches):
        runs = [floor * _run_factor(gen) for _ in range(per_batch)]
        vals.append(min(runs))
        spent += sum(runs)
    mean = sum(vals) / batches
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (batches - 1))
    return mean, sd / math.sqrt(batches), spent


CAP_FACTOR = 2.0       # a launch is abandoned once it has cost this many times the best estimate so far


def _capped_estimate(gen, floor, cap, batches=5, per_batch=4):
    """Like `_batch_estimate`, for a challenger: every launch is abandoned at `cap` (its cost is then `cap`), and
    the candidate is dropped when the whole first batch was abandoned. Returns (mean, se, spent, launches), with
    mean = None for a dropped candidate."""
    vals, spent, launches = [], 0.0, 0
    for b in range(batches):
        runs = [min(floor * _run_factor(gen), cap) for _ in range(per_batch)]
        spent += sum(runs)
        launches += per_batch
        if b == 0 and min(runs) >= cap:
            return None, None, spent, launches
        vals.append(min(runs))
    mean = sum(vals) / batches
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (batches - 1))
    return mean, sd / math.sqrt(batches), spent, launches


def _race(gen, pref, cost_of):
    """Measure the candidates `pref` (the model's preference order, best guess first) and choose. The first is
    the incumbent and gets the full 20 launches; each later one is a challenger that gets 20 launches capped at
    twice the best estimate so far, and displaces the best only when it is better by more than 2 SE of the
    difference. Returns (choice, spent, launches, dropped)."""
    best, est, spent, launches, dropped = None, {}, 0.0, 0, []
    for c in pref:
        floor = cost_of(c)
        if best is None:
            mu, se, w = _batch_estimate(gen, floor)
            est[c], best = (mu, se), c
            spent, launches = spent + w, launches + 20
            continue
        mu, se, w, nl = _capped_estimate(gen, floor, CAP_FACTOR * est[best][0])
        spent, launches = spent + w, launches + nl
        if mu is None:
            dropped.append(c)
            continue
        est[c] = (mu, se)
        if est[best][0] - mu > 2.0 * math.sqrt(est[best][1] ** 2 + se ** 2):
            best = c
    return best, spent, launches, dropped


# ------------------------------------------------------- the ranking (problem)

SHAPES = [(1, 2560, 2048), (4, 2560, 2048), (8, 2560, 2048), (16, 2560, 2048), (32, 2560, 2048), (48, 2560, 2048),
          (64, 2560, 2048), (77, 2560, 2048), (128, 2560, 2048), (129, 2560, 2048), (256, 2560, 2048),
          (512, 2560, 2048), (1024, 130, 2048), (512, 100, 4096), (4096, 1, 4096), (1, 100, 100)]
RANK_RUNS = 8
ORDER = ("naive", "tiled", "reg")


def _ranking_rows():
    gen = bookrng.generator("kernel_selection.ranking")
    dev = DEVICES["GPU"]
    rows = []
    for (m, n, k) in SHAPES:
        floor = _floor(dev, m, n, k)
        pred = {c: floor / _predicted(dev, CANDIDATES[c], m, n, k) for c in ORDER}
        meas = {c: floor / _min_of(gen, _cost(dev, CANDIDATES[c], m, n, k), RANK_RUNS) for c in ORDER}
        pick_model = max(ORDER, key=lambda c: (pred[c], -ORDER.index(c)))
        pick_meas = max(ORDER, key=lambda c: (meas[c], -ORDER.index(c)))
        rows.append(dict(shape=(m, n, k), pred=pred, meas=meas, pick_model=pick_model, pick_meas=pick_meas,
                         lost=1.0 - meas[pick_model] / meas[pick_meas]))
    return rows


def _ranking_text(rows):
    return (f"{_seed_line('kernel_selection.ranking')} Draw order: for each shape in the order of the table, for each "
            f"candidate in the order naive, tiled, reg, {RANK_RUNS} runs of three uniforms each (u0, u1, u2).")


def model_vs_device():
    rows = _ranking_rows()
    pick = [r for r in rows if r["shape"][0] in (1, 32, 129, 512) and r["shape"][1] == 2560 and r["shape"][2] == 2048]
    pick += [r for r in rows if r["shape"] == (512, 100, 4096)]
    body = []
    for r in pick:
        m, n, k = r["shape"]
        body.append((f"{m} x {n} x {k}", r["pick_model"], f"{r['pred'][r['pick_model']]:.3f}",
                     r["pick_meas"], f"{r['meas'][r['pick_meas']]:.3f}", f"{r['meas'][r['pick_model']]:.3f}",
                     f"{r['lost']:.3f}"))
    return (
        "The model GPU unit of the next sections: peak 36,000 (GFLOP/s), bandwidth 664 (GB/s), so the ridge is "
        "36,000 / 664 = 54.22 FLOP per byte. Three candidates, C[m, n] = A[m, k] B[k, n] in fp32. 'Predicted' is "
        "the roofline model of Chapter 3: the cost of the candidate is max(its bytes, 2 m n k / ridge), and the "
        "fraction is the cost of the problem's own floor (every operand once) divided by it. 'Device' is the model "
        "device of this chapter (formulas below): the minimum of 8 seeded runs, as a fraction of the same floor. "
        "'Lost' is 1 minus (device fraction of the model's pick over device fraction of the device's pick).\n\n"
        + _ranking_text(rows) + "\n\n"
        + _table(["m x n x k", "model's pick", "its predicted fraction", "device's pick", "its device fraction",
                  "device fraction of the model's pick", "lost"], body, "lrrrrrr")
    )


def ranking_table():
    rows = _ranking_rows()
    body = []
    for r in rows:
        m, n, k = r["shape"]
        p, d = r["pred"], r["meas"]
        body.append((f"{m} x {n} x {k}",
                     f"{p['naive']:.3f} / {d['naive']:.3f}", f"{p['tiled']:.3f} / {d['tiled']:.3f}",
                     f"{p['reg']:.3f} / {d['reg']:.3f}", r["pick_model"], r["pick_meas"], f"{r['lost']:.3f}"))
    wrong = sum(1 for r in rows if r["pick_model"] != r["pick_meas"])
    return (
        "Each cell is predicted fraction / device fraction of the problem's roofline floor, the device fraction "
        "being the minimum of 8 seeded runs. Same model GPU unit, same runs as `kernel_selection.model_vs_device`.\n\n"
        + _ranking_text(rows) + "\n\n"
        + _table(["m x n x k", "naive", "tiled", "reg", "model's pick", "device's pick", "lost"], body, "lrrrllr")
        + f"\n\nThe model's pick differs from the device's on {wrong} of the {len(rows)} shapes."
    )


def candidate_table():
    rows = []
    for name, c in CANDIDATES.items():
        r, s = c["rb"]
        rows.append((name, f"{c['bm']} x {c['bn']}", c["n_wg"], f"{r} x {s}", c["ell"], f"{_efficiency(c):.4f}"))
    return (
        "Three implementations of C = A B. 'Tile' is the block of outputs one workgroup owns (the naive kernel owns "
        "one output per invocation). 'Loads in flight' is the independent 4-byte loads an invocation has outstanding "
        "before it must wait: 2 for the naive kernel (the two operands of one step of k), and the staging loads of "
        "a step for the tiled kernels (a 32 x 8 slab of A and an 8 x 32 slab of B over 64 invocations is 8; a "
        "128 x 8 and an 8 x 128 over 256 is 8). The issue efficiency e = r s / (r s + r + s + 2) counts the fused "
        "multiply-adds of one step of k against the reads of the slabs and 2 instructions of loop overhead; the naive "
        "kernel does 1 multiply-add against 2 loads from memory and the same overhead, 1 / 5.\n\n"
        + _table(["candidate", "tile", "invocations per workgroup", "register block", "loads in flight",
                  "issue efficiency e"], rows, "lrrrrr")
    )


def cost_model_example():
    dev = DEVICES["GPU"]
    m, n, k = 77, 2560, 2048
    rows = []
    for name in ORDER:
        c = CANDIDATES[name]
        q = _counts(c, m, n, k)
        o_cmp = min(1.0, q["n_inv"] / dev["lanes"])
        o_mem = min(1.0, q["n_inv"] * c["ell"] / _loads_needed(dev))
        rows.append((name, f"{q['tm']} x {q['tn']}", _n(q["n_inv"]), _n(q["n_byte"]), _n(q["flop_pad"]),
                     f"{o_cmp:.4f}", f"{o_mem:.4f}", _n(round(_predicted(dev, c, m, n, k))),
                     _n(round(_cost(dev, c, m, n, k)))))
    return (
        f"The shape m = {m}, n = {n}, k = {k} on the model GPU unit. Ridge {_ridge(dev):.4f}; the start of one "
        f"dispatch alone costs {_launch_bytes(dev):,.0f} bytes (5 microseconds times a bandwidth of 664 (GB/s)); the invocations "
        f"that keep the memory busy at one load in flight each: {_loads_needed(dev):,.0f}; lanes {dev['lanes']:,}. "
        f"The floor of the problem is {_n(round(_floor(dev, m, n, k)))} equivalent bytes.\n\n"
        + _table(["candidate", "tile grid", "invocations", "bytes requested", "FLOPs executed", "o_cmp", "o_mem",
                  "predicted cost", "device cost"], rows, "lrrrrrrrr")
    )


# ------------------------------------------------------------- the capabilities

# What a candidate needs, in the vocabulary of DeviceCaps (Chapter 4 and 9).
# `reg-notail` is the defective candidate of the gate section; coop16 is the
# matrix-unit candidate of Chapter 6 (an fp16-input contract).
DECLARATIONS = [
    ("naive", "nothing", "any shape"),
    ("tiled", "workgroup barriers, 2 KiB workgroup memory", "any shape (row and column guards)"),
    ("reg", "workgroup barriers, 8 KiB workgroup memory, workgroups of 256", "any shape (row and column guards)"),
    ("coop16", "matrix unit with (f16, f16, f32), subgroups", "m, n, k all multiples of 16; inputs already fp16"),
]


def _needs_met(dev, name):
    """Why a candidate is refused on a device, or None when the device declares everything it needs."""
    if name == "naive":
        return None
    if name in ("tiled", "reg", "reg-notail"):
        c = CANDIDATES["tiled"] if name == "tiled" else CANDIDATES["reg"]
        if not dev["workgroup_barriers"]:
            return "no workgroup barriers"
        if dev["shared_kib"] < c["shared_kib"]:
            return "workgroup memory"
        if dev["max_wg"] < c["n_wg"]:
            return "workgroup size"
        return None
    if name == "coop16":
        return None if dev["matrix_unit"] else "no matrix unit"
    raise KeyError(name)


def capability_filter():
    rows = []
    for name, needs, valid in DECLARATIONS:
        row = [name, needs, valid]
        for dname in ("CPU", "GPU"):
            why = _needs_met(DEVICES[dname], name)
            row.append("runs" if why is None else f"refused: {why}")
        rows.append(tuple(row))
    cpu, gpu = DEVICES["CPU"], DEVICES["GPU"]
    m, n, k = 512, 2560, 2048
    pred = {c: _floor(cpu, m, n, k) / _predicted(cpu, CANDIDATES[c], m, n, k) for c in ORDER}
    inten = {c: 2 * m * n * k / _counts(CANDIDATES[c], m, n, k)["n_byte"] for c in ORDER}
    pr = [(c, f"{inten[c]:.2f}", f"{pred[c]:.3f}", "yes" if _needs_met(cpu, c) is None else "no") for c in ORDER]
    shapes = []
    for sh in [(64, 64, 64), (77, 2560, 2048), (512, 2560, 2048), (48, 2560, 2048)]:
        ok = all(x % 16 == 0 for x in sh)
        shapes.append((f"{sh[0]} x {sh[1]} x {sh[2]}", "valid" if ok else "invalid (a dimension is not a multiple of 16)"))
    return (
        "The device side is the record of Chapter 9 (queried, an unknown capability is absent): the CPU unit "
        "declares no workgroup barriers (its compiler runs a workgroup as loops and cannot execute a barrier "
        "kernel correctly, Chapter 9) and no matrix unit; the GPU unit declares barriers, 16 KiB of workgroup "
        "memory, workgroups of up to 256 and a (f16, f16, f32) matrix unit.\n\n"
        + _table(["candidate", "needs", "valid for", "CPU unit", "GPU unit"], rows, "lllll")
        + "\n\nThe filter is a correctness gate, not a preference. What the roofline model alone would pick on the CPU "
        "unit (ridge 10.67), for m = 512, n = 2560, k = 2048, if the filter did not exist:\n\n"
        + _table(["candidate", "FLOP per byte requested", "predicted fraction of the floor", "executable on the CPU unit"],
                 pr, "lrrr")
        + "\n\nThe shape condition of coop16 on four shapes:\n\n" + _table(["m x n x k", "coop16"], shapes, "ll")
    )


# --------------------------------------------------------------- the roofline

MIN_FACTOR = 100.0     # a probe run must cost at least this many launches
PROBE_REPS = 3         # runs per attempt; the minimum is taken
PROBE_THREADS = 1 << 20
PROBE_FLOPS_PER_ITER = 16
PROBE_BW_ELEMS = 1 << 26
PROBE_BW_BYTES_PER_ELEM = 12


def _probe(dev, gen, kind, cold=1.0):
    """The calibration loop of the chapter, on the model device. Returns (rate, attempts, work) where work is the
    FLOPs (kind 'flops') or bytes (kind 'bw') all attempts executed together."""
    t_launch = dev["launch_us"] * 1e-6
    rate = (dev["flops"] if kind == "flops" else dev["bw"]) * 1e9 * cold
    trip = 256 if kind == "flops" else 1
    attempts, work = 0, 0.0
    while True:
        unit = PROBE_THREADS * PROBE_FLOPS_PER_ITER if kind == "flops" else PROBE_BW_ELEMS * PROBE_BW_BYTES_PER_ELEM
        amount = unit * trip
        times = []
        for _ in range(PROBE_REPS):
            times.append((t_launch + amount / rate) * _run_factor(gen))
        attempts += 1
        work += amount * PROBE_REPS
        t = min(times)
        if t >= MIN_FACTOR * t_launch:
            return amount / t / 1e9, attempts, work
        want = math.ceil(trip * MIN_FACTOR * t_launch / t)
        trip = max(want, 2 * trip)


def _calibrate(dname):
    gen = bookrng.generator(f"kernel_selection.calibration.{dname}")
    dev = DEVICES[dname]
    f, fa, fw = _probe(dev, gen, "flops")
    b, ba, bwk = _probe(dev, gen, "bw")
    return dict(flops=f, bw=b, f_attempts=fa, b_attempts=ba, f_work=fw, b_work=bwk)


def calibration():
    rows = []
    for dname in ("CPU", "GPU"):
        dev = DEVICES[dname]
        c = _calibrate(dname)
        rows.append((dname, _n(round(dev["spec_flops"])), _n(round(c["flops"])), f"{100 * c['flops'] / dev['spec_flops']:.1f}",
                     _n(round(dev["spec_bw"])), _n(round(c["bw"])), f"{100 * c['bw'] / dev['spec_bw']:.1f}",
                     f"{dev['spec_flops'] / dev['spec_bw']:.2f}", f"{c['flops'] / c['bw']:.2f}"))
    work = []
    for dname in ("CPU", "GPU"):
        c = _calibrate(dname)
        work.append((dname, c["f_attempts"], f"{c['f_work'] / 1e9:,.1f}", c["b_attempts"], f"{c['b_work'] / 1e9:,.1f}"))
    return (
        "Two probes, as described above. Compute probe: 2^20 invocations, each a chain of 8 independent fused "
        "multiply-adds repeated `trip` times (16 FLOPs per step), starting at trip = 256. Bandwidth probe: "
        "c = c + s a over 2^26 elements (12 bytes per element, 805,306,368 bytes per pass), repeated `trip` "
        "passes, starting at 1. Each attempt runs 3 times and keeps the minimum; an attempt is accepted when its "
        "cost is at least 100 launches, otherwise trip becomes max(2 trip, ceil(trip x 100 launches / cost)). "
        "A run costs (launch + work / true rate) x its noise factor (three uniforms, as in the ranking). The "
        "hidden true rates are those of the units in `DEVICES` (GPU 36,000 and 664; CPU 640 and 60, in GFLOP/s and GB/s), "
        "launches are 5 and 2 microseconds. The rates are reported as the work of the accepted attempt divided by "
        "its time. " + "Generators: the book's SplitMix64, seeds " + ", ".join(f"{bookrng.seed_of('kernel_selection.calibration.' + d):#018x} "
        f"(`kernel_selection.calibration.{d}`)" for d in ("CPU", "GPU")) + ". Draw "
        "order: the compute probe's runs, then the bandwidth probe's, 3 uniforms per run, attempts in order.\n\n"
        + _table(["device", "spec F (GFLOP/s)", "measured F", "% of spec", "spec BW (GB/s)", "measured BW",
                  "% of spec", "spec ridge", "measured ridge"], rows, "lrrrrrrrr")
        + "\n\nThe work the probes executed, all attempts together:\n\n"
        + _table(["device", "compute attempts", "GFLOP executed", "bandwidth attempts", "GB moved"], work, "lrrrr")
    )


def fraction_of_roofs():
    dev = DEVICES["GPU"]
    cal = _calibrate("GPU")
    cold = _calibrate_cold()
    rows_by_shape = {r["shape"]: r for r in _ranking_rows()}
    rows = []
    for (m, n, k), name in [((1, 2560, 2048), "tiled"), ((512, 2560, 2048), "reg"), ((4096, 1, 4096), "tiled")]:
        phi = rows_by_shape[(m, n, k)]["meas"][name]            # fraction of the unit's own floor, from the ranking table
        flops, nbytes = 2 * m * n * k, 4 * (m * k + k * n + m * n)

        def bound(F, BW):
            return max(flops / (F * 1e9), nbytes / (BW * 1e9))

        t = bound(dev["flops"], dev["bw"]) / phi                 # seconds of the best of 8 runs
        rows.append((f"{name}, {m} x {n} x {k}", f"{phi:.3f}", f"{bound(dev['spec_flops'], dev['spec_bw']) / t:.3f}",
                     f"{bound(cal['flops'], cal['bw']) / t:.3f}", f"{bound(cold['flops'], cal['bw']) / t:.3f}"))
    return (
        "The same best-of-8 runs as `kernel_selection.ranking_table` (no new draws), graded against four rooflines: the "
        "unit's true rates (36,000 and 664: the fractions of the ranking table), the specification (40,000 and 800), "
        f"the calibration of `kernel_selection.calibration` on this unit ({cal['flops']:,.0f} and {cal['bw']:,.0f}), and a "
        f"calibration whose compute probe ran on an idle device and measured {cold['flops']:,.0f} (its bandwidth is the "
        "calibrated one). "
        "The bound of a row is max(2 m n k / F, 4 (m k + k n + m n) / BW). The cold probe is "
        "`kernel_selection.calibration.GPU.cold` (seed " + f"{bookrng.seed_of('kernel_selection.calibration.GPU.cold'):#018x}"
        + "): the compute probe of the calibration, with the device's true compute rate multiplied by 0.4.\n\n"
        + _table(["kernel, shape", "of the true roofline", "of the specification", "of the measured roofline",
                  "of the cold-probe roofline"], rows, "lrrrr")
    )


def _calibrate_cold():
    gen = bookrng.generator("kernel_selection.calibration.GPU.cold")
    f = _probe(DEVICES["GPU"], gen, "flops", 0.4)[0]
    return dict(flops=f)


# ---------------------------------------------------------------- the gate

BM_GATE = CANDIDATES["reg"]["bm"]
U32 = 2.0 ** -24


def _positions(m, n):
    rows = [0, m // 2, max(m - 1, 0)]
    cols = [0, 1, 15, 16, 17, n // 3, n // 2, max(n - 17, 0), max(n - 1, 0)]
    out = []
    for r in rows:
        for c in cols:
            if r < m and c < n and (r, c) not in out:
                out.append((r, c))
    return out


def _dot_left(a, b):
    acc = 0.0
    for x, y in zip(a, b):
        acc = _f32(acc + _f32(x * y))
    return acc


def _dot_tree(a, b):
    v = [_f32(x * y) for x, y in zip(a, b)]
    while len(v) > 1:
        if len(v) % 2:
            v.append(0.0)
        v = [_f32(v[i] + v[i + 1]) for i in range(0, len(v), 2)]
    return v[0]


def _k_rd(variant, k):
    if variant == "reg-tree":
        return 1 + math.ceil(math.log2(k))
    return k


def _gate_shape(gen, m, n, k, variants):
    """Run the reference gate for one shape: build the sampled rows of A and columns of B (draws), then check
    every variant at the sampled positions against the exact reference and the derived tolerance."""
    pos = _positions(m, n)
    rows_i = sorted({r for r, _ in pos})
    cols_j = sorted({c for _, c in pos})
    arows = {r: [_f32(2.0 * gen.uniform() - 1.0) for _ in range(k)] for r in rows_i}
    bcols = {c: [_f32(2.0 * gen.uniform() - 1.0) for _ in range(k)] for c in cols_j}
    out = {}
    left = {}                                   # the left-fold value at a position, shared by the candidates that use it

    def stored(v, r, c):
        """What candidate v stores at C[r, c]."""
        if v == "reg-notail" and r >= BM_GATE * (m // BM_GATE):
            return 0.0                          # the row was never computed; the output holds its initial zero
        if v == "reg-tree":
            return _dot_tree(arows[r], bcols[c])
        if (r, c) not in left:
            left[(r, c)] = _dot_left(arows[r], bcols[c])
        return left[(r, c)]

    for v in variants:
        bad, worst = 0, 0.0
        for (r, c) in pos:
            a, b = arows[r], bcols[c]
            ref = math.fsum(x * y for x, y in zip(a, b))
            s = math.fsum(abs(x * y) for x, y in zip(a, b))
            kr = _k_rd(v, k)
            tol = kr * U32 * s / (1.0 - kr * U32)
            err = abs(stored(v, r, c) - ref)
            ratio = err / tol if tol > 0 else (0.0 if err == 0 else math.inf)
            worst = max(worst, ratio)
            bad += ratio > 1.0
        out[v] = (len(pos), bad, worst)
    return out


GATE_VARIANTS = ("reg", "reg-tree", "reg-notail")
GATE_SHAPES = [(128, 2560, 2048), (129, 2560, 2048), (200, 2560, 2048), (256, 2560, 2048)]


def gate_table():
    gen = bookrng.generator("kernel_selection.gate")
    rows = []
    for (m, n, k) in GATE_SHAPES:
        res = _gate_shape(gen, m, n, k, GATE_VARIANTS)
        for v in GATE_VARIANTS:
            tot, bad, worst = res[v]
            rows.append((f"{m} x {n} x {k}", v, tot, bad, f"{worst:.3g}", "admitted" if bad == 0 else "REFUSED"))
    return (
        "The gate of the text: for each shape, the 27 output positions of rows {0, m div 2, m - 1} and columns "
        "{0, 1, 15, 16, 17, n div 3, n div 2, n - 17, n - 1} (duplicates removed), entries 2 uniform() - 1 rounded to "
        "float32, the products and sums of a candidate each rounded to float32 (no fused multiply-add). reg is a left "
        "fold over k; reg-tree adds the products in a tree of adjacent pairs; reg-notail is reg with its row guard "
        "removed, so it computes only the 128 x (m div 128) rows of whole tiles and leaves the other rows at their "
        "initial zero. The reference of a position is math.fsum of the exact products (a correctly rounded double "
        "sum); the tolerance is k_rd u S / (1 - k_rd u) with u = 2^-24, S = sum of |a b|, k_rd = k = 2048 for a left "
        "fold and 1 + log2(2048) = 12 for the tree. 'Worst' is the largest error divided by the tolerance. "
        + _seed_line("kernel_selection.gate") + " Draw order: per shape in the order of the table, the entries of "
        "the sampled rows of A in increasing row order (k values each), then the entries of the sampled columns of B "
        "in increasing column order (k values each).\n\n"
        + _table(["m x n x k", "candidate", "positions checked", "outside tolerance", "worst error / tolerance", "verdict"],
                 rows, "llrrrl")
    )


def fast_and_wrong():
    dev = DEVICES["GPU"]
    gen = bookrng.generator("kernel_selection.fast_and_wrong")
    rows = []
    for m in (129, 200, 255):
        n, k = 2560, 2048
        floor = _floor(dev, m, n, k)
        fr = {}
        for name in ("tiled", "reg", "reg-notail"):
            fr[name] = floor / _min_of(gen, _cost_of(dev, name, m, n, k), 8)
        best = max(fr, key=fr.get)
        rows.append((f"{m} x {n} x {k}", f"{fr['tiled']:.3f}", f"{fr['reg']:.3f}", f"{fr['reg-notail']:.3f}", best))
    return (
        "Device fractions of the floor, minimum of 8 seeded runs, for three shapes of the bucket 129 .. 256. The "
        "cost of reg-notail is the cost of reg on its 128 x (m div 128) rows. "
        + _seed_line("kernel_selection.fast_and_wrong") + " Draw order: per shape, per candidate in the order tiled, "
        "reg, reg-notail, 8 runs of 3 uniforms each.\n\n"
        + _table(["m x n x k", "tiled", "reg", "reg-notail", "fastest by measurement alone"], rows, "lrrrl")
    )


# --------------------------------------------------------------- statistics

TRIALS = 2000
LAUNCHES = 20


def _procedures(a_runs, b_runs):
    """The four ways to decide whether the challenger B displaces the incumbent A, each from the same 20 launches
    of each. Returns the four booleans (B chosen)."""
    out = []
    mean = lambda v: sum(v) / len(v)   # noqa: E731
    med = lambda v: sorted(v)[len(v) // 2 - 1] / 2 + sorted(v)[len(v) // 2] / 2   # noqa: E731
    out.append(mean(b_runs) < mean(a_runs))
    out.append(med(b_runs) < med(a_runs))
    out.append(min(b_runs) < min(a_runs))

    def stats(runs):
        mins = [min(runs[i:i + 4]) for i in range(0, LAUNCHES, 4)]
        mu = mean(mins)
        sd = math.sqrt(sum((x - mu) ** 2 for x in mins) / (len(mins) - 1))
        return mu, sd / math.sqrt(len(mins))

    (ma, sa), (mb, sb) = stats(a_runs), stats(b_runs)
    out.append(ma - mb > 2.0 * math.sqrt(sa * sa + sb * sb))
    return out


def noise_runs():
    gen = bookrng.generator("kernel_selection.noise_runs")
    rows = []
    a = [_run_factor(gen) for _ in range(8)]
    b = [0.97 * _run_factor(gen) for _ in range(8)]
    for i in range(8):
        rows.append((i, f"{a[i]:.4f}", f"{b[i]:.4f}"))
    return (
        "One pair of candidates in the units of the incumbent's floor: A has floor 1.00, B has floor 0.97. Each run "
        "is floor x (1 + 0.04 u0) x (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1). " + _seed_line("kernel_selection.noise_runs")
        + " Draw order: the 8 runs of A, then the 8 runs of B, 3 uniforms each.\n\n"
        + _table(["run", "A (floor 1.00)", "B (floor 0.97)"], rows, "lrr")
        + f"\n\nMinimum of the 8: A {min(a):.4f}, B {min(b):.4f}. Mean of the 8: A {sum(a) / 8:.4f}, B {sum(b) / 8:.4f}. "
        f"Median of the 8: A {sorted(a)[3] / 2 + sorted(a)[4] / 2:.4f}, B {sorted(b)[3] / 2 + sorted(b)[4] / 2:.4f}. "
        "The mean ranks the two the wrong way round on these 8 runs only if a spike hit B and not A; the minimum "
        "stays within the jitter of the floor."
    )


def estimator_simulation():
    gen = bookrng.generator("kernel_selection.estimators")
    ratios = (1.00, 0.99, 0.97, 0.90)
    rows = []
    for rho in ratios:
        wins = [0, 0, 0, 0]
        for _ in range(TRIALS):
            a = [_run_factor(gen) for _ in range(LAUNCHES)]
            b = [rho * _run_factor(gen) for _ in range(LAUNCHES)]
            for i, w in enumerate(_procedures(a, b)):
                wins[i] += w
        rows.append((f"{rho:.2f}",) + tuple(f"{w / TRIALS:.3f}" for w in wins))
    return (
        f"{TRIALS:,} trials per row. In a trial, the incumbent A (floor 1) and the challenger B (floor = the ratio) "
        f"each run {LAUNCHES} launches, A's first and then B's, 3 uniforms each (the noise of `kernel_selection.noise_runs`). "
        "The four procedures see the same 20 launches of each: 'mean' picks B when its mean is lower; 'median' when "
        "its median is lower; 'minimum' when its minimum is lower; 'batch-minimum, 2 SE' splits the 20 launches in "
        "5 batches of 4, takes the minimum of each batch, and picks B only when the difference of the two means of "
        "the 5 batch minima exceeds twice sqrt(SE_A^2 + SE_B^2), SE = sd / sqrt(5) with divisor 4. The entries are "
        "the share of trials in which B was chosen. " + _seed_line("kernel_selection.estimators")
        + "\n\n"
        + _table(["floor of B / floor of A", "mean", "median", "minimum", "batch-minimum, 2 SE"], rows, "lrrrr")
    )


# -------------------------------------------------------------------- the cache


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


SOURCE_REG = "kernel reg: tile 128x128, block 8x8, step 8, barriers 2, guard rows yes, guard cols yes"
SOURCE_REG_EDITED = SOURCE_REG.replace("step 8", "step 4")


def bucket_of(m):
    """Row bucket: the smallest power of two not below m (m >= 1)."""
    return 1 << (m - 1).bit_length()


def _key(device, driver, source, m, n, k):
    return f"{device}|{driver}|{fnv1a64(source.encode('utf-8')):016x}|M<={bucket_of(m)},N={n},K={k}"


def cache_keys():
    vec = [("", fnv1a64(b"")), ("a", fnv1a64(b"a")), ("foobar", fnv1a64(b"foobar"))]
    base = dict(device="gpu-unit7", driver="drv-535.1", source=SOURCE_REG, m=77, n=2560, k=2048)
    store = {_key(**base): "reg"}
    probes = [
        ("the same request", base),
        ("m = 100 (same bucket 128)", dict(base, m=100)),
        ("m = 129 (bucket 256)", dict(base, m=129)),
        ("another device", dict(base, device="gpu-unit8")),
        ("a driver update", dict(base, driver="drv-535.2")),
        ("an edited kernel source", dict(base, source=SOURCE_REG_EDITED)),
        ("n = 2561", dict(base, n=2561)),
    ]
    rows = []
    for label, p in probes:
        key = _key(**p)
        rows.append((label, key, "hit" if key in store else "miss"))
    return (
        "FNV-1a over bytes, 64 bits: h = 0xcbf29ce484222325; for each byte b: h = h xor b, then h = h x 0x100000001b3 "
        "(mod 2^64). Test values: " + ", ".join(f"FNV-1a(\"{s}\") = {h:#018x}" for s, h in vec) + ".\n\n"
        f"The stored entry has the key `{_key(**base)}` and the value reg. The source text is "
        f"`{SOURCE_REG}` (its edited form changes `step 8` to `step 4`). A request is looked up under its own key:\n\n"
        + _table(["request", "key", "lookup"], rows, "llr")
    )


# the bucket trade-off -----------------------------------------------------

WORKLOAD_REQUESTS = 400
WORKLOAD_N, WORKLOAD_K = 1024, 2048


def _workload():
    gen = bookrng.generator("kernel_selection.workload")
    return [math.ceil(2.0 ** (9.0 * gen.uniform())) for _ in range(WORKLOAD_REQUESTS)]


def _true_best(dev, m):
    costs = {c: _cost(dev, CANDIDATES[c], m, WORKLOAD_N, WORKLOAD_K) for c in ORDER}
    return min(costs, key=costs.get), costs


def _choose_at(dev, gen, rep):
    """Measure the three correct candidates at the representative shape (`_race`). Returns (choice, device work)."""
    pref = sorted(ORDER, key=lambda c: -_predicted_neg(dev, c, rep))
    best, spent, _, _ = _race(gen, pref, lambda c: _cost(dev, CANDIDATES[c], rep, WORKLOAD_N, WORKLOAD_K))
    return best, spent


def _predicted_neg(dev, c, m):
    """Higher is better: the negative of the roofline model's predicted cost (ties keep the order of ORDER reversed)."""
    return -_predicted(dev, CANDIDATES[c], m, WORKLOAD_N, WORKLOAD_K) - 1e-9 * ORDER.index(c)


SCHEMES = [
    ("one bucket per m", lambda m: m),
    ("powers of two", bucket_of),
    ("three buckets: 32, 128, 512", lambda m: 32 if m <= 32 else 128 if m <= 128 else 512),
    ("one bucket: 512", lambda m: 512),
]


def _bucket_results():
    dev = DEVICES["GPU"]
    ms = _workload()
    base_best = {m: _true_best(dev, m) for m in set(ms)}
    ideal = sum(base_best[m][1][base_best[m][0]] for m in ms)
    static = sum(base_best[m][1]["reg"] for m in ms)
    out = []
    for label, f in SCHEMES:
        gen = bookrng.generator(f"kernel_selection.buckets.{label}")
        chosen, spent_total = {}, 0.0
        for m in ms:                            # lazy: a bucket is measured at its first request
            b = f(m)
            if b not in chosen:
                chosen[b], spent = _choose_at(dev, gen, b)
                spent_total += spent
        served = sum(base_best[m][1][chosen[f(m)]] for m in ms)
        out.append(dict(label=label, buckets=len(chosen), spent=spent_total, served=served))
    return dict(ms=ms, ideal=ideal, static=static, schemes=out)


def bucket_tradeoff():
    r = _bucket_results()
    rows = []
    for s in r["schemes"]:
        excess = s["served"] - r["ideal"]
        rows.append((s["label"], s["buckets"], f"{s['spent'] / 1e9:.2f}", f"{100 * excess / r['ideal']:.2f}",
                     f"{excess / 1e9:.2f}"))
    static_excess = r["static"] - r["ideal"]
    return (
        f"A workload of {WORKLOAD_REQUESTS} requests with n = {WORKLOAD_N} and k = {WORKLOAD_K}, m = ceil(2^(9 u)) for a "
        "uniform draw u per request (so m runs from 1 to 512 with equal weight on every doubling). "
        + _seed_line("kernel_selection.workload") + " Draw order: one uniform per request, in order. A bucket is "
        "measured the first time a request falls in it, at its largest member (the powers of two: 1, 2, 4, ..., 512), with "
        "the chapter's protocol (20 launches of each of the three correct candidates, a 2 SE rule over the model's "
        "preference order). Measurement draws: `kernel_selection.buckets.<scheme>`, candidates in the model's "
        "preference order, 20 launches of 3 uniforms each. Costs are in billions (10^9) of equivalent bytes on the model GPU unit. 'Excess' is the work of "
        f"serving the {WORKLOAD_REQUESTS} requests with the buckets' choices minus the work with the best candidate for each "
        f"exact m ({r['ideal'] / 1e9:.2f}).\n\n"
        + _table(["scheme", "buckets measured", "measurement work", "excess over the best, %", "excess work"], rows, "lrrrr")
        + f"\n\nAlways using reg (the model's pick) costs an excess of {static_excess / 1e9:.2f}, "
        f"{100 * static_excess / r['ideal']:.2f} percent."
    )


def amortization():
    r = _bucket_results()
    rows = []
    static_excess = r["static"] - r["ideal"]
    per_run_saving = {}
    for s in r["schemes"]:
        per_run_saving[s["label"]] = r["static"] - s["served"]
    for s in r["schemes"]:
        sv = per_run_saving[s["label"]]
        be = f"{s['spent'] / sv:.2f}" if sv > 0 else "never"
        shares = [f"{100 * s['spent'] / (n * s['served'] + s['spent']):.2f}" for n in (1, 10, 100)]
        rows.append((s["label"], f"{s['spent'] / 1e9:.2f}", f"{sv / 1e9:.2f}", be) + tuple(shares))
    return (
        "Same workload and measurements as `kernel_selection.bucket_tradeoff`. 'Saved per run' is the work of one "
        "run of the 400 requests under always-reg minus the work under the scheme's choices; 'runs to break even' is "
        "the measurement work divided by it. The last three columns are the measurement's share of all the work "
        "(requests plus the one-time measurement) after 1, 10 and 100 runs of the workload, the cache being kept "
        "between runs. Billions of equivalent bytes.\n\n"
        + _table(["scheme", "measurement work", "saved per run", "runs to break even", "share after 1 run, %",
                  "after 10 runs, %", "after 100 runs, %"], rows, "lrrrrrr")
        + f"\n\nThe best possible saving per run (the exact-m choice) is {static_excess / 1e9:.2f}."
    )


# ------------------------------------------------------------- the pipeline

PIPE_DEVICE = "GPU"
PIPE_CATALOGUE = ("reg", "tiled", "naive", "reg-notail")
REFERENCE = "naive"      # the portable baseline: correct by definition, never gated, never absent


def _pipeline_run():
    """The whole selection for four requests: two on the GPU unit around a source edit, one on the CPU unit."""
    gen = bookrng.generator("kernel_selection.pipeline")
    store = {}
    log = []
    n, k = WORKLOAD_N, WORKLOAD_K

    def request(label, m, source_reg=SOURCE_REG, device="GPU", driver="drv-535.1"):
        d = DEVICES[device]
        key = _key(f"{device.lower()}-unit7", driver, source_reg, m, n, k)
        if key in store:
            log.append(dict(label=label, m=m, status="hit", choice=store[key]))
            return
        cap = [c for c in PIPE_CATALOGUE if _needs_met(d, c) is None]
        entry = dict(label=label, m=m, status="miss", n_cat=len(PIPE_CATALOGUE), n_cap=len(cap), gate_launches=0,
                     launches=0, dropped=[], work=0.0)
        if len(cap) == 1:                                   # nothing to choose between: the reference runs
            entry.update(n_gate=1, choice=cap[0])
            store[key] = cap[0]
            log.append(entry)
            return
        hi, lo = bucket_of(m), bucket_of(m) // 2
        gated, spent = [REFERENCE], 0.0
        for c in cap:
            if c == REFERENCE:
                continue
            ok = True
            for gm in (hi, lo + 1):
                variant = c if c == "reg-notail" else "reg"
                ok = ok and _gate_shape(gen, gm, n, k, (variant,))[variant][1] == 0
                spent += _cost_of(d, c, gm, n, k)
                entry["gate_launches"] += 1
            if ok:
                gated.append(c)
        pref = sorted(gated, key=lambda c: -_predicted_neg(d, c, hi))
        best, w, launches, dropped = _race(gen, pref, lambda c: _cost_of(d, c, hi, n, k))
        entry.update(n_gate=len(gated), launches=launches, dropped=dropped, work=spent + w, choice=best)
        store[key] = best
        log.append(entry)

    request("1. first request, GPU unit", 200)
    request("2. same bucket", 230)
    request("3. after editing the source of reg", 230, source_reg=SOURCE_REG_EDITED)
    request("4. the first request on the CPU unit", 200, device="CPU")
    return log


def pipeline_trace():
    log = _pipeline_run()
    rows = []
    for e in log:
        if e["status"] == "hit":
            rows.append((e["label"], e["m"], "-", "-", "-", "-", "-", e["choice"], "hit"))
        else:
            dropped = ("dropped: " + ", ".join(e["dropped"])) if e["dropped"] else "none dropped"
            rows.append((e["label"], e["m"], e["n_cap"], e["n_gate"], e["gate_launches"],
                         f"{e['launches']} ({dropped})" if e["launches"] else 0, f"{e['work'] / 1e9:.2f}", e["choice"], "miss"))
    return (
        "Catalogue, in the model's preference order: reg, tiled, naive, reg-notail (the register-blocked kernel "
        "without its row guard); naive is the reference. Request shape m x 1024 x 2048, rows bucketed to powers of "
        "two (m = 200 and 230 both fall in the bucket 256, which is measured at its largest member, 256). The gate "
        "checks each candidate except the reference on the bucket's two extreme shapes, 256 and 129, with the sampled "
        "positions of `kernel_selection.gate_table` (n = 1024 here), and charges the device one launch at each; the "
        "measurement is the race of the text (20 launches for the incumbent, challengers capped at twice the best "
        "estimate and dropped when their first batch of 4 is all abandoned). A request with one candidate left "
        "after the filter is not gated or measured. The key is `device|driver|digest of reg's source|bucket,n,k`. "
        + _seed_line("kernel_selection.pipeline") + " Draw order: for each miss, the gate's entries for each "
        "candidate that passed the filter and is not the reference (shape 256 then 129; the sampled rows of A, then the "
        "sampled columns of B), then the race's launches in the model's preference order (3 uniforms per launch). "
        "Work is in billions of equivalent bytes (gate launches plus measurement launches).\n\n"
        + _table(["step", "m", "after the capability filter", "after the gate", "gate launches", "measurement launches",
                  "work", "choice", "cache"], rows, "lrrrrrrll")
    )


def pipeline_numbers():
    """For the figure: the first request of `pipeline_trace` on the GPU unit and on the CPU unit."""
    log = _pipeline_run()
    return log[0], log[3]


# -------------------------------------------- series for the figures

FIG_MS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 129, 192, 256, 384, 512]


def figure_series():
    """Predicted and device fractions of the floor against m (n = 2560, k = 2048), and the cost ratio reg / tiled."""
    dev = DEVICES["GPU"]
    gen = bookrng.generator("kernel_selection.figure")
    n, k = 2560, 2048
    pred = {c: [] for c in ORDER}
    meas = {c: [] for c in ORDER}
    ratio_pred, ratio_meas = [], []
    for m in FIG_MS:
        floor = _floor(dev, m, n, k)
        runs = {c: _min_of(gen, _cost(dev, CANDIDATES[c], m, n, k), 8) for c in ORDER}
        for c in ORDER:
            pred[c].append(floor / _predicted(dev, CANDIDATES[c], m, n, k))
            meas[c].append(floor / runs[c])
        ratio_pred.append(_predicted(dev, CANDIDATES["reg"], m, n, k) / _predicted(dev, CANDIDATES["tiled"], m, n, k))
        ratio_meas.append(runs["reg"] / runs["tiled"])
    return dict(ms=FIG_MS, pred=pred, meas=meas, ratio_pred=ratio_pred, ratio_meas=ratio_meas)


def crossover(n=2560, k=2048):
    """The smallest m at which the noise-free device cost of reg is below tiled's, for output width n and depth k."""
    dev = DEVICES["GPU"]
    for m in range(1, 1025):
        if _cost(dev, CANDIDATES["reg"], m, n, k) < _cost(dev, CANDIDATES["tiled"], m, n, k):
            return m
    return None


def crossover_table():
    rows = []
    for (n, k) in [(2560, 2048), (1024, 2048), (256, 2048), (130, 2048)]:
        mc = crossover(n, k)
        rows.append((f"{n} x {k}", mc if mc else "none up to 1024"))
    return (
        "The smallest m at which the register-blocked kernel's noise-free device cost is below the tiled kernel's "
        "(no randomness), by output width n and depth k, on the model GPU unit.\n\n"
        + _table(["n x k", "crossover m"], rows, "lr")
    )
