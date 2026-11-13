# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Measuring Performance Honestly".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: profiling.<name> -->` markers. Helpers with a leading underscore,
or that take arguments, are shared with the chapter's figures so a picture and
the table beside it cannot disagree.

Nothing here times a real machine. The "device" is the model GPU unit of
Chapters 3, 4 and 10 and the "clock" is a model too: a run of a kernel is its
noise-free cost (microseconds, from the roofline of Chapter 3 and a hidden
efficiency) times the seeded noise factor of Chapter 10, plus the host-side
costs of Chapter 8 (a submission, the start of a dispatch) and the cost of the
wait itself. Every number is a property of that model and of its stated
parameters; none is an observation of hardware.

Randomness comes from the book's own generator
(`bookrng.generator("profiling.<example>")`), is printed with its seed, and is
drawn in an order the chapter states.

Swedish Embedded AB implements solutions for measuring machine learning
workloads so that the numbers hold on the customer's hardware. If your team
needs expertise in performance measurement of GPU inference and training, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import math

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
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`, defined on the pseudocode page)."


def _median(v):
    s = sorted(v)
    m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0


def _mean(v):
    return sum(v) / len(v)


def _sd(v):
    mu = _mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))


# ---------------------------------------------------------------- the model

# The model GPU unit of Chapter 10: its hidden true rates (what the silicon
# reaches) and the measured roofline Chapter 10's calibration recovered, which
# is the denominator of every fraction in this chapter.
F_TRUE, BW_TRUE = 36000.0, 664.0      # (GFLOP/s), (GB/s)
F_MS, BW_MS = 34469.0, 646.0          # the measured roofline of `kernel_selection.calibration`
RIDGE_MS = F_MS / BW_MS

# Host-side and wait costs, microseconds. T_SUB and L_DSP are Chapter 8's (a dispatch submitted alone pays both,
# Chapter 3's 5 microseconds). T_WAKE is this chapter's one new parameter: after the device has finished, the host still
# needs this long to notice (the fence signal, the scheduler, the return from the call).
T_SUB, L_DSP = 4.0, 1.0
T_ENQ = T_SUB + L_DSP
ENQ_JITTER = 0.20                     # the host's enqueue cost is T_ENQ (1 + 0.20 u)
T_WAKE = 40.0
WAKE_JITTER = 0.25                    # the wait costs T_WAKE (1 + 0.25 u)

# The noise of Chapter 10: factor = (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1).
SPIKE_P, SPIKE_MEAN, JITTER = 0.2, 0.25, 0.04
FACTOR_MEAN = (1.0 + JITTER / 2.0) * (1.0 + SPIKE_P * SPIKE_MEAN)       # 1.02 * 1.05 = 1.071


def _factor(gen):
    """One device run's factor over its floor: exactly three uniform draws u0 u1 u2. A run never beats the floor."""
    u0, u1, u2 = gen.uniform(), gen.uniform(), gen.uniform()
    spike = -SPIKE_MEAN * math.log(1.0 - u2) if u1 < SPIKE_P else 0.0
    return (1.0 + JITTER * u0) * (1.0 + spike)


def _factor_sd():
    """The standard deviation of the factor, exactly: factor = a m with a = 1 + 0.04 U and m = 1 + 0.25 X on a spike
    (X exponential with mean 1, probability 0.2) and 1 otherwise, a and m independent."""
    ea, va = 1.0 + JITTER / 2.0, JITTER ** 2 / 12.0
    em = 1.0 + SPIKE_P * SPIKE_MEAN
    em2 = (1.0 - SPIKE_P) + SPIKE_P * (1.0 + 2.0 * SPIKE_MEAN + 2.0 * SPIKE_MEAN ** 2)
    var = (va + ea * ea) * em2 - (ea * em) ** 2
    return math.sqrt(var)


def _work_us(flops, nbytes):
    """The roofline time of Chapter 3 on the device's TRUE rates, microseconds: max(FLOPs / F, bytes / BW)."""
    return max(flops / (F_TRUE * 1e3), nbytes / (BW_TRUE * 1e3))


def _bound_us(flops, nbytes):
    """The roofline time on the MEASURED roofline, the floor a fraction is taken against."""
    return max(flops / (F_MS * 1e3), nbytes / (BW_MS * 1e3))


def _floor_us(spec, clock=1.0):
    """A kernel's noise-free device time as a timestamp pair between dispatches reads it: the start of the dispatch plus
    the work at the kernel's hidden efficiency (`clock` < 1 slows the work, a device that has not ramped up)."""
    return L_DSP + _work_us(spec["flops"], spec["bytes"]) / (spec["eff"] * clock)


def _runs(gen, spec, n, clock=1.0):
    """n runs of one dispatch with a wait after it: a list of (enqueue, wait, device) times in microseconds.
    Exactly five uniform draws per run, in this order: the enqueue jitter, the wait jitter, then the factor's u0 u1 u2."""
    floor = _floor_us(spec, clock)
    out = []
    for _ in range(n):
        enq = T_ENQ * (1.0 + ENQ_JITTER * gen.uniform())
        wake = T_WAKE * (1.0 + WAKE_JITTER * gen.uniform())
        out.append((enq, wake, floor * _factor(gen)))
    return out


def _frac(spec_flops, spec_bytes, t_us):
    """The achieved fraction of the measured roofline of Chapter 3: roofline time over the time taken."""
    return _bound_us(spec_flops, spec_bytes) / t_us


def _mm(m, n, k):
    return 2 * m * n * k, 4 * (m * k + k * n + m * n)


def _add_counts(n):
    return n, 12 * n


# ---------------------------------------------------------------- the timer lie

ADD_1M = dict(name="add, 2^20 floats", flops=1 << 20, bytes=12 << 20, eff=0.85)
ADD_64M = dict(name="add, 2^26 floats", flops=1 << 26, bytes=12 << 26, eff=0.85)
_MM_F, _MM_B = _mm(512, 2560, 2048)
MATMUL = dict(name="matmul, 512 x 2560 x 2048", flops=_MM_F, bytes=_MM_B, eff=0.55)
LIE_RUNS = 11


def _timers(runs):
    """The three timers around one dispatch: the clock stopped when the host regains control after enqueueing
    (enqueue only), after the wait (enqueue + device + wait), and the device's own stamps (device only)."""
    return (_median([e for e, _, _ in runs]), _median([e + d + w for e, w, d in runs]), _median([d for _, _, d in runs]))


def enqueue_lie():
    name = "profiling.enqueue_lie"
    gen = bookrng.generator(name)
    rows = []
    for spec in (ADD_1M, ADD_64M, MATMUL):
        t_enq, t_sync, t_dev = _timers(_runs(gen, spec, LIE_RUNS))
        floor = _bound_us(spec["flops"], spec["bytes"])
        for label, t in (("enqueue only (clock stopped when the call returns)", t_enq),
                         ("enqueue and wait (clock stopped after the wait)", t_sync),
                         ("device stamps (the dispatch's own start and end)", t_dev)):
            rows.append((spec["name"], f"{floor:,.2f}", label, f"{t:,.2f}", f"{floor / t:,.3f}"))
    lines = [
        "The model of this chapter, all parameters labelled and none a measurement. Device: the model GPU unit of Chapter 10, "
        "true rates 36,000 (GFLOP/s) and 664 (GB/s), graded against its measured roofline 34,469 (GFLOP/s) and 646 (GB/s). "
        "A dispatch submitted alone costs the host t_sub + t_dsp = 4 + 1 = 5 microseconds of enqueue, times (1 + 0.20 u); "
        "the wait costs 40 microseconds after the device has finished, times (1 + 0.25 u); the device runs for t_dsp = 1 "
        "microsecond plus the roofline time of the kernel at its hidden efficiency (0.85 for the additions c = a + b, 0.55 for "
        "the matrix product), times the noise factor of Chapter 10, (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1), "
        "with ln the natural logarithm. Bytes and FLOPs: the addition of N floats does N FLOPs and moves 12 N bytes, the "
        "product C[m, n] = A[m, k] B[k, n] in fp32 does 2 m n k FLOPs and moves 4 (m k + k n + m n) bytes.",
        "",
        f"{_seed_line(name)} Draw order: for each kernel in the order of the table, {LIE_RUNS} runs; each run draws five uniforms "
        "in the order enqueue jitter, wait jitter, u0, u1, u2. Each timer reports the median of the 11 runs. 'Roofline time' is "
        "the bound of Chapter 3 on the measured roofline, max(FLOPs / 34,469 (GFLOP/s), bytes / 646 (GB/s)); the fraction is that "
        "bound divided by the time read.",
        "",
        _table(["kernel", "roofline time (microseconds)", "timer", "time read (microseconds)", "fraction of the measured roofline"],
               rows, "lrlrr"),
    ]
    return "\n".join(lines)


SIZES_LOG2 = (12, 14, 16, 18, 20, 22, 24, 26)


def wait_cost():
    name = "profiling.wait_cost"
    gen = bookrng.generator(name)
    rows = []
    for e in SIZES_LOG2:
        nel = 1 << e
        spec = dict(flops=nel, bytes=12 * nel, eff=0.85)
        runs = _runs(gen, spec, LIE_RUNS)
        _, t_sync, t_dev = _timers(runs)
        rows.append((f"2^{e}", _n(12 * nel), f"{t_dev:,.2f}", f"{t_sync:,.2f}", f"{t_sync / t_dev:,.2f}",
                     f"{_frac(spec['flops'], spec['bytes'], t_dev):.3f}", f"{_frac(spec['flops'], spec['bytes'], t_sync):.3f}"))
    lines = [
        "The addition c = a + b on N floats (N FLOPs, 12 N bytes), hidden efficiency 0.85, the model and the timers of "
        "`profiling.enqueue_lie`. Each row: 11 runs, medians. 'Device stamps' read only the device's time; 'enqueue and wait' "
        "reads the enqueue, the device and the wait. "
        f"{_seed_line(name)} Draw order: the sizes of the table in order, 11 runs each, five uniforms per run as before.",
        "",
        _table(["N", "bytes", "device stamps (microseconds)", "enqueue and wait (microseconds)", "wait reading / stamps",
                "fraction by stamps", "fraction by enqueue and wait"], rows),
    ]
    return "\n".join(lines)


def timeline_numbers():
    """The noise-free times of the addition of 2^20 floats for the timeline figure: the host's enqueue, the device's time and
    the wait, in microseconds."""
    return dict(enq=T_ENQ, dev=_floor_us(ADD_1M), wake=T_WAKE)


# ---------------------------------------------------------------- warmup

WARM_N = 200
WARM_A = 0.8           # the clock ramp: the first run is 1.8 times the floor, decaying with ...
WARM_TAU = 5.0         # ... a time constant of 5 runs
WARM_COMPILE = 20.0    # the first run also builds its pipelines: 20 floors extra
WARM_TOL = 0.05
WARM_K = 5


def _warm_series(n=WARM_N):
    """Run times in units of the steady floor: t_i = (1 + 0.8 exp(-i / 5)) f_i + 20 [i = 0], f_i the noise factor of
    Chapter 10 (three uniforms per run)."""
    gen = bookrng.generator("profiling.warmup_series")
    return [(1.0 + WARM_A * math.exp(-i / WARM_TAU)) * _factor(gen) + (WARM_COMPILE if i == 0 else 0.0) for i in range(n)]


def _warm_end(t, k=WARM_K, tol=WARM_TOL):
    """The end of the warm-up: the floor is the minimum of the second half of the series, and the warm-up ends at the first
    run that starts k consecutive runs each within tol of it."""
    ref = min(t[len(t) // 2:])
    limit = (1.0 + tol) * ref
    for i in range(len(t) - k + 1):
        if all(x <= limit for x in t[i:i + k]):
            return i, ref
    return len(t), ref


def warmup_runs():
    t = _warm_series()
    end, ref = _warm_end(t)
    limit = (1.0 + WARM_TOL) * ref
    rows = []
    for i in range(24):
        trans = 1.0 + WARM_A * math.exp(-i / WARM_TAU)
        rows.append((i, f"{t[i]:.4f}", f"{trans:.4f}", "yes" if t[i] <= limit else "no", "<- warm-up ends" if i == end else ""))
    name = "profiling.warmup_series"
    lines = [
        "A series of 200 runs of one pass, in units of its steady floor. Run i costs (1 + 0.8 e^(-i / 5)) f_i, plus 20 on run "
        "0 (the pipelines are built on the first call), where e is the base of the natural logarithm, exp the library "
        "function, and f_i the noise factor of Chapter 10: (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1). "
        f"{_seed_line(name)} Draw order: runs 0 to 199, three uniforms each (u0, u1, u2). The floor is estimated as the "
        f"minimum of runs 100 to 199, {ref:.4f}; a run is 'within tolerance' when it is at most 1.05 times that "
        f"({limit:.4f}); the warm-up ends at the first run that starts 5 consecutive runs each within tolerance. "
        "'Transient' is the noise-free factor 1 + 0.8 e^(-i / 5) (without the compile cost).",
        "",
        _table(["run i", "time (floors)", "transient", "within tolerance", ""], rows, "rrrll"),
        "",
        f"The warm-up ends at run {end}.",
    ]
    return "\n".join(lines)


def warmup_series():
    t = _warm_series()
    end, ref = _warm_end(t)
    name = "profiling.warmup_series"
    all_, after = t, t[end:]
    rows = []
    for label, fn in (("mean", _mean), ("median", _median), ("minimum", min), ("standard deviation", _sd)):
        rows.append((label, f"{fn(all_):.4f}", f"{fn(after):.4f}"))
    se_all, se_after = _sd(all_) / math.sqrt(len(all_)), _sd(after) / math.sqrt(len(after))
    rows.append(("standard error of the mean, s / sqrt(n)", f"{se_all:.4f}", f"{se_after:.4f}"))
    rows.append(("runs", len(all_), len(after)))
    bias = _mean(all_) / _mean(after) - 1.0
    lines = [
        # perf-number: hex prefix of a printed seed, not a multiplier
        f"The 200 runs of `profiling.warmup_runs` (the same series, the same seed 0x{bookrng.seed_of(name):016x}). Left: all 200 runs. Right: runs "
        f"{end} to 199, after the warm-up detected by the rule of the text. Units: floors. The noise factor's expected value "
        f"is (1 + 0.04 / 2) (1 + 0.2 x 0.25) = {FACTOR_MEAN:.3f}, so a steady mean near that is the right answer.",
        "",
        _table(["statistic", "all 200 runs", f"runs {end} to 199"], rows),
        "",
        f"The mean over all runs is {bias:.4f} above the mean after the warm-up (a bias of {100 * bias:.1f} percent); the "
        f"median is {_median(all_) / _median(after) - 1.0:.4f} above and the minimum is the same.",
    ]
    return "\n".join(lines)


def warmup_bias():
    t = _warm_series()
    end, _ = _warm_end(t)
    steady = _mean(t[end:])
    rows = []
    for n in (5, 10, 20, 50, 100, 200):
        expected = (WARM_A * sum(math.exp(-i / WARM_TAU) for i in range(n)) + WARM_COMPILE / FACTOR_MEAN) / n
        rows.append((n, f"{_mean(t[:n]):.4f}", f"{_mean(t[:n]) / steady - 1.0:.4f}", f"{expected:.4f}"))
    lines = [
        f"The same series, the mean of the first n runs against the steady mean {steady:.4f} (runs {end} to 199). 'Expected bias' "
        "is what the warm-up adds to the mean of n runs on average, relative to the steady mean: "
        "(0.8 times the sum over i < n of e^(-i / 5), plus 20 / 1.071) / n, where 1.071 is the noise factor's mean; the "
        "measured column differs from it by the noise of one series.",
        "",
        _table(["n", "mean of the first n runs", "measured: relative to the steady mean, minus 1", "expected bias"], rows),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- repeated runs

MC_DRAWS = 100000


def noise_moments():
    name = "profiling.noise_moments"
    gen = bookrng.generator(name)
    v = [_factor(gen) for _ in range(MC_DRAWS)]
    mu, sd = _mean(v), _sd(v)
    cv = _factor_sd() / FACTOR_MEAN
    rows = [
        ("mean of the factor", f"{FACTOR_MEAN:.4f}", f"{mu:.4f}"),
        ("standard deviation of the factor", f"{_factor_sd():.4f}", f"{sd:.4f}"),
        ("relative standard deviation cv = sd / mean", f"{cv:.4f}", f"{sd / mu:.4f}"),
        ("median of the factor", "-", f"{_median(v):.4f}"),
        ("minimum of the 100,000 draws", "1 (the floor)", f"{min(v):.4f}"),
    ]
    lines = [
        "The noise factor of Chapter 10, (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1). Analytic: with a = 1 + 0.04 U and "
        "m = 1 + 0.25 X on a spike (X exponential with mean 1, probability 0.2), E[a] = 1.02, Var[a] = 0.04^2 / 12, E[m] = 1.05, "
        "E[m^2] = 0.8 + 0.2 (1 + 0.5 + 0.125) = 1.125, and the factor is a m with a and m independent. "
        f"{_seed_line(name)} Draw order: 100,000 factors, three uniforms each (u0, u1, u2). Sample standard deviation with "
        "divisor n - 1.",
        "",
        _table(["quantity", "analytic", "sample of 100,000"], rows),
    ]
    return "\n".join(lines)


REL_GAPS = (0.05, 0.02, 0.01, 0.005)
SIM_TRIALS = 300


def _c_rel():
    return _factor_sd() / FACTOR_MEAN


def repetitions_needed():
    name = "profiling.repetitions"
    gen = bookrng.generator(name)
    c = _c_rel()
    rows = []
    for gap in REL_GAPS:
        n2 = math.ceil(8.0 * c * c / (gap * gap))
        row = [f"{gap:.3f}", _n(n2), _n(4 * n2)]
        if gap >= 0.01:
            for n in (n2, 4 * n2):
                hit = 0
                for _ in range(SIM_TRIALS):
                    a = [_factor(gen) for _ in range(n)]
                    b = [(1.0 - gap) * _factor(gen) for _ in range(n)]
                    se = math.sqrt(_sd(a) ** 2 / n + _sd(b) ** 2 / n)
                    if _mean(a) - _mean(b) > 2.0 * se:
                        hit += 1
                row.append(f"{hit / SIM_TRIALS:.3f}")
        else:
            row += ["not simulated", "not simulated"]
        rows.append(tuple(row))
    lines = [
        f"Two variants, A with floor 1 and B with floor 1 - gap, each timed n times with the noise of Chapter 10 (relative standard "
        f"deviation cv = {c:.6f}, printed as {c:.4f} in the table above). The runs needed so that the gap equals twice the standard error of the "
        "difference, 2 cv sqrt(2 / n) = gap, are n = 8 cv^2 / gap^2 (rounded up), and four times as many when the gap is to equal "
        f"four standard errors. The last two columns are the share of {SIM_TRIALS} trials in which the mean of A exceeded the mean "
        "of B by more than 2 sqrt(s_A^2 / n + s_B^2 / n), with s the sample standard deviation (divisor n - 1). "
        f"{_seed_line(name)} Draw order: for each gap in the order of the table, for n = n_2 and then n = 4 n_2, for each trial, "
        "A's n factors and then B's n factors, three uniforms each; B's run is (1 - gap) times its factor. The 0.005 row is "
        "not simulated. Each n_2 is the ceiling of its own formula, and n for four standard errors is 4 n_2 of the same row; the n_2 of the 0.005 row (6,573) is computed from the formula directly, "
        "so it is not 4 times the n_2 of the 0.01 row (4 x 1,644 = 6,576, because 1,644 was itself rounded up).",
        "",
        _table(["gap", "n (gap = 2 SE)", "n (gap = 4 SE)", "flagged at n_2", "flagged at 4 n_2"], rows),
    ]
    return "\n".join(lines)


BEST_KS = (1, 3, 10, 30, 100, 300)
BEST_TRIALS = 1000


def best_of_k():
    name = "profiling.best_of_k"
    gen = bookrng.generator(name)
    rows = []
    for k in BEST_KS:
        best_sum = typ_sum = 0.0
        for _ in range(BEST_TRIALS):
            v = [_factor(gen) for _ in range(k)]
            best_sum += min(v)
            typ_sum += _mean(v)
        best, typ = best_sum / BEST_TRIALS, typ_sum / BEST_TRIALS
        rows.append((k, f"{best:.4f}", f"{typ:.4f}", f"{typ / best - 1.0:.4f}"))
    lines = [
        "Reporting the best of k runs. The device has floor 1 and the noise of Chapter 10. For each k, 1000 trials of k runs; "
        "'best' is the mean over trials of the minimum of the k runs, 'typical' the mean over trials of the mean of the k runs. "
        f"{_seed_line(name)} Draw order: for each k in the order of the table, each of the 1000 trials in turn, its k factors, "
        "three uniforms each.",
        "",
        _table(["k", "best of k", "typical (mean of k)", "typical / best - 1"], rows),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- the profile

M_TOK, D_MODEL, D_FF = 512, 2048, 6144
REPLAYS = 15


def _pass_spec():
    """One layer of a transformer for 512 tokens, ten dispatches in dependency order. FLOPs and bytes by the counting rules of
    the chapter (fp32, 4 bytes, each operand moved once)."""
    m, d, ff = M_TOK, D_MODEL, D_FF
    mmf = lambda n, k: _mm(m, n, k)                    # noqa: E731
    rms = (4 * m * d, 4 * (2 * m * d + d))
    attn = (4 * m * m * d, 4 * 4 * m * d)
    silu = (5 * m * ff, 4 * 3 * m * ff)
    add = (m * d, 12 * m * d)
    mk = lambda name, kind, fb, eff: dict(name=name, kind=kind, flops=fb[0], bytes=fb[1], eff=eff)   # noqa: E731
    return [
        mk("rmsnorm_1", "rmsnorm", rms, 0.55),
        mk("matmul_qkv", "matmul", mmf(3 * d, d), 0.62),
        mk("attention", "attention", attn, 0.30),
        mk("matmul_out", "matmul", mmf(d, d), 0.58),
        mk("add_1", "add", add, 0.85),
        mk("rmsnorm_2", "rmsnorm", rms, 0.55),
        mk("matmul_gateup", "matmul", mmf(2 * ff, d), 0.66),
        mk("silu_mul", "silu_mul", silu, 0.80),
        mk("matmul_down", "matmul", mmf(d, ff), 0.60),
        mk("add_2", "add", add, 0.85),
    ]


PASS = _pass_spec()


def _profile_times():
    """Per-dispatch device time: the minimum over 15 replays of the same pass (Another process only lengthens a run, so
    the minimum is the least contaminated). Three uniforms per dispatch per replay, replay by replay, dispatches in order."""
    gen = bookrng.generator("profiling.kernel_profile")
    best = [math.inf] * len(PASS)
    for _ in range(REPLAYS):
        for i, s in enumerate(PASS):
            best[i] = min(best[i], _floor_us(s) * _factor(gen))
    return best


def _pass_floor(times):
    """The pass as one submission with one wait: the submission, the dispatches back to back, the wait (noise-free)."""
    return T_SUB + sum(times) + T_WAKE


def _fraction(s, t):
    return _frac(s["flops"], s["bytes"], t)


def kernel_profile():
    name = "profiling.kernel_profile"
    times = _profile_times()
    total = sum(times)
    rows = []
    for i, (s, t) in enumerate(zip(PASS, times)):
        inten = s["flops"] / s["bytes"]
        rows.append((i, s["name"], _n(s["flops"]), _n(s["bytes"]), f"{inten:,.2f}", "compute" if inten >= RIDGE_MS else "bandwidth",
                     f"{t:,.2f}", f"{t / total:.3f}", f"{_fraction(s, t):.3f}"))
    lines = [
        f"One layer for m = {M_TOK} tokens, model width d = {D_MODEL}, feed-forward width {D_FF} = 3 d, fp32. Ten dispatches in "
        "dependency order. Counts: rmsnorm 4 m d FLOPs and 4 (2 m d + d) bytes; a matrix product [m, k] times [k, n] 2 m n k "
        "FLOPs and 4 (m k + k n + m n) bytes; attention (fused, scores never leave the chip) 4 m^2 d FLOPs and 16 m d bytes; "
        "silu_mul 5 m (feed-forward width) FLOPs and 12 m (feed-forward width) bytes; add m d FLOPs and 12 m d bytes. The device "
        "is the model unit of `profiling.enqueue_lie`; the hidden efficiencies are rmsnorm 0.55, matmul_qkv 0.62, attention 0.30, "
        "matmul_out 0.58, add 0.85, matmul_gateup 0.66, silu_mul 0.80, matmul_down 0.60. A dispatch's time is its device stamp "
        "pair (t_dsp = 1 microsecond plus the roofline time on the true rates at the efficiency) times the noise factor of "
        "Chapter 10, and the table keeps the minimum over 15 replays of the pass. "
        f"{_seed_line(name)} Draw order: replay 0 to 14, dispatches 0 to 9 in each, three uniforms (u0, u1, u2). "
        f"'Bound' compares the intensity with the measured ridge 34,469 / 646 = {RIDGE_MS:.2f} FLOP per byte; the fraction is the "
        "bound on the measured roofline over the time; the share is of the sum of the ten times.",
        "",
        _table(["#", "dispatch", "FLOPs", "bytes", "FLOP per byte", "bound", "device time (microseconds)", "share", "fraction of the roofline"],
               rows, "rlrrrlrrr"),
        "",
        f"Sum of the ten device times: {total:,.2f} microseconds.",
    ]
    return "\n".join(lines)


def profile_by_kind():
    times = _profile_times()
    total = sum(times)
    kinds = []
    for s in PASS:
        if s["kind"] not in kinds:
            kinds.append(s["kind"])
    rows = []
    for k in kinds:
        idx = [i for i, s in enumerate(PASS) if s["kind"] == k]
        t = sum(times[i] for i in idx)
        fl = sum(PASS[i]["flops"] for i in idx)
        by = sum(PASS[i]["bytes"] for i in idx)
        bound = sum(_bound_us(PASS[i]["flops"], PASS[i]["bytes"]) for i in idx)
        rows.append((k, len(idx), _n(fl), _n(by), f"{t:,.2f}", f"{t / total:.3f}", f"{bound / t:.3f}"))
    lines = [
        "The same ten times grouped by kernel kind, as brain's profiler groups them. The fraction of a kind is the sum of its "
        "dispatches' roofline times (on the measured roofline) over the sum of their device times, so a long dispatch weighs more.",
        "",
        _table(["kind", "calls", "FLOPs", "bytes", "device time (microseconds)", "share", "fraction of the roofline"], rows, "lrrrrrr"),
    ]
    return "\n".join(lines)


def sum_vs_pass():
    times = _profile_times()
    total = sum(times)
    one_sub = T_SUB + total + T_WAKE
    drained = sum(T_ENQ + t + T_WAKE for t in times)
    a, b = 600.0, 400.0
    over = T_SUB + max(a, b) + T_WAKE
    rows = [
        ("ten dependent dispatches, one submission, one wait", f"{total:,.2f}", f"{one_sub:,.2f}", f"{one_sub - total:+,.2f}"),
        ("the same ten, each submitted alone and waited for", f"{total:,.2f}", f"{drained:,.2f}", f"{drained - total:+,.2f}"),
        ("two independent dispatches of 600 and 400, overlapped", f"{a + b:,.2f}", f"{over:,.2f}", f"{over - (a + b):+,.2f}"),
    ]
    lines = [
        "Per-dispatch device stamps against the time of the pass, noise-free. Rows 1 and 2 use the ten minimum times of "
        "`profiling.kernel_profile`. One submission with one wait costs t_sub + sum of the device times + t_wait = 4 + sum + 40 "
        "microseconds; a dispatch submitted alone and waited for costs t_enq + its time + t_wait = 5 + time + 40. Row 3: two "
        "dispatches with no dependence between them, each using at most half the device's lanes, so they run at the same time "
        "(a model assumption); their stamps are 600 and 400 microseconds, and the pass lasts the longer plus the host's 4 + 40.",
        "",
        _table(["case", "sum of the device stamps (microseconds)", "time of the pass (microseconds)", "pass minus sum"], rows, "lrrr"),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- the whole pass

PASS_RUNS = 300
SPLIT_SPEEDUP = 0.85          # variant B: the output projection's work takes 0.85 of its time
EXTRA_DISPATCHES = 3          # variant B: three more small dispatches (zero the partials, reduce, scale), 2 microseconds of work each
EXTRA_WORK_US = 2.0
EXTRA_SYNCS = 2               # variant B: two points at which the host must read a result: a submission and a wait each


def _pass_times(gen, variant):
    """One pass of the layer, a synchronized timing: (the submission(s), the dispatches' device times, the wait(s)).
    Draws per pass: three uniforms per dispatch in order, then one per wait."""
    t = 0.0
    n_waits = 1 + (EXTRA_SYNCS if variant == "B" else 0)
    for s in PASS:
        base = _floor_us(s)
        if variant == "B" and s["name"] == "matmul_out":
            base = L_DSP + _work_us(s["flops"], s["bytes"]) / s["eff"] * SPLIT_SPEEDUP
        t += base * _factor(gen)
    if variant == "B":
        for _ in range(EXTRA_DISPATCHES):
            t += (L_DSP + EXTRA_WORK_US) * _factor(gen)
    t += T_SUB * n_waits
    for _ in range(n_waits):
        t += T_WAKE * (1.0 + WAKE_JITTER * gen.uniform())
    return t


PASS_RUNS_SMALL = 300
PASS_RUNS = 1200


def whole_pass_change():
    name = "profiling.whole_pass"
    gen = bookrng.generator(name)
    a = [_pass_times(gen, "A") for _ in range(PASS_RUNS)]
    b = [_pass_times(gen, "B") for _ in range(PASS_RUNS)]
    mo = next(s for s in PASS if s["name"] == "matmul_out")
    k_a = _floor_us(mo)
    k_b = L_DSP + _work_us(mo["flops"], mo["bytes"]) / mo["eff"] * SPLIT_SPEEDUP
    rows = []
    stats = {}
    for n in (PASS_RUNS_SMALL, PASS_RUNS):
        xa, xb = a[:n], b[:n]
        ma, mb = _mean(xa), _mean(xb)
        sa, sb = _sd(xa) / math.sqrt(n), _sd(xb) / math.sqrt(n)
        diff, se = mb - ma, math.sqrt(sa * sa + sb * sb)
        stats[n] = (ma, diff, se)
        rows.append((n, f"{ma:,.2f}", f"{sa:,.2f}", f"{mb:,.2f}", f"{sb:,.2f}", f"{diff:+,.2f}", f"{se:,.2f}", f"{diff / se:+.2f}"))
    ma, diff, _ = stats[PASS_RUNS]
    c_pass = _sd(a) / _mean(a)
    n2 = math.ceil(8.0 * c_pass ** 2 / (diff / ma) ** 2)
    lines = [
        "Changing the output projection (`matmul_out`) of the layer of `profiling.kernel_profile`. Variant B runs the product "
        f"with the start of the dispatch unchanged (t_dsp = {L_DSP:g} microsecond) and the rest of its time, the work at the hidden efficiency, multiplied by {SPLIT_SPEEDUP} "
        f"(by its device stamp: {L_DSP:g} + {SPLIT_SPEEDUP} x ({k_a:,.2f} - {L_DSP:g}) = {k_b:,.2f} against {k_a:,.2f} microseconds, noise-free) but "
        f"needs {EXTRA_DISPATCHES} more dispatches of t_dsp + {EXTRA_WORK_US:g} = {L_DSP + EXTRA_WORK_US:g} microseconds each and "
        f"{EXTRA_SYNCS} points where the host must wait for a result, each costing a submission (4) and a wait "
        "(40 microseconds, times (1 + 0.25 u)). A pass is timed with a wait at its end: the sum over its dispatches of the "
        "noise-free time times the noise factor, plus 4 per submission and one wait per submission. "
        f"{PASS_RUNS} passes of each variant are drawn; the first {PASS_RUNS_SMALL} are one row and all {PASS_RUNS} the other. "
        "The mean is the cost a user pays. "
        f"{_seed_line(name)} Draw order: the {PASS_RUNS} passes of A, then the {PASS_RUNS} passes of B; in a pass, the dispatches "
        "in order (A: the ten of the layer; B: the same with `matmul_out` replaced, then the three extra), three uniforms each "
        "(u0, u1, u2), then one uniform for each wait. Standard error s / sqrt(n) with divisor n - 1; the standard error of the "
        "difference is sqrt(SE_A^2 + SE_B^2).",
        "",
        _table(["passes per variant", "mean A (microseconds)", "SE A", "mean B (microseconds)", "SE B", "B minus A", "SE of the difference",
                "difference / SE"], rows),
        "",
        f"By its stamp the product is {100 * (1 - k_b / k_a):.1f} percent faster in B; the pass is {100 * diff / ma:+.2f} percent "
        f"({PASS_RUNS} passes). The relative standard deviation of one pass of A is cv = {c_pass:.4f}, so the formula "
        f"n = 8 cv^2 / gap^2 with gap = {diff / ma:.4f} asks for {n2:,} passes to bring the gap to twice its standard error.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- Amdahl's law


def amdahl(p, g):
    """The overall speedup when a part with share p of the time is made g times as fast: 1 / ((1 - p) + p / g)."""
    return 1.0 / ((1.0 - p) + p / g)


def _shares():
    times = _profile_times()
    total = _pass_floor(times)
    parts = {}
    for s, t in zip(PASS, times):
        parts.setdefault(s["kind"], []).append((s, t))
    return times, total, parts


def matmul_share():
    times, total, parts = _shares()
    return sum(t for _, t in parts["matmul"]) / total


def kind_roof_g(kind):
    """The factor by which a kind's time falls if each of its dispatches reached the measured roofline."""
    times, _, parts = _shares()
    items = parts[kind]
    return sum(t for _, t in items) / sum(_bound_us(s['flops'], s['bytes']) for s, _ in items)


def amdahl_table():
    times, total, parts = _shares()
    host = T_SUB + T_WAKE
    rows = []
    for k, items in parts.items():
        t = sum(x for _, x in items)
        p = t / total
        roof = sum(_bound_us(s["flops"], s["bytes"]) for s, _ in items)
        g_roof = t / roof
        rows.append((k, f"{t:,.2f}", f"{p:.4f}", f"{amdahl(p, 2.0):.4f}", f"{1.0 / (1.0 - p):.4f}", (f"{p / (1.0 - p):.2f}" if p >= 0.5 else "-"),
                     f"{g_roof:.3f}", f"{amdahl(p, g_roof):.4f}"))
    rows.append(("submission and wait", f"{host:,.2f}", f"{host / total:.4f}", "-", f"{1.0 / (1.0 - host / total):.4f}", "-", "-", "-"))
    roof_all = host + sum(_bound_us(s["flops"], s["bytes"]) for s in PASS)
    lines = [
        f"The profile of `profiling.kernel_profile` and the pass it makes: {total:,.2f} microseconds = 4 (submission) + the sum of "
        f"the ten device times {sum(times):,.2f} + 40 (wait). For each kind, p is its time over the pass; A(2) is the speedup if "
        "its time is halved, 1 / (1 - p) the ceiling if it took no time at all, p / (1 - p) the factor g at which the speedup "
        "reaches half of the ceiling (the improved part then takes as long as all the rest; none exists for p below one half), 'g to the "
        "roofline' the factor by which the kind's time falls if every dispatch reached the measured roofline (the kind's time over "
        "the sum of its roofline times), and the last column the speedup of that.",
        "",
        _table(["part", "time (microseconds)", "p", "A(p, 2)", "ceiling 1 / (1 - p)", "g for half the ceiling", "g to the roofline", "A at that g"],
               rows),
        "",
        f"Every dispatch at the measured roofline and the host's costs unchanged: the pass would take {roof_all:,.2f} microseconds, "
        f"a speedup of {total / roof_all:.4f}.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- other lies

COLD_CLOCK = 0.4


def lies_catalogue():
    name = "profiling.lies"
    gen = bookrng.generator(name)
    big = ADD_64M
    small = dict(flops=1 << 14, bytes=12 << 14, eff=0.85)
    mm = MATMUL
    rows = []

    def add(label, saw, t, flops, nbytes):
        f = _frac(flops, nbytes, t)
        rows.append((label, saw, f"{t:,.2f}", f"{f:,.3f}", "yes" if f > 1.05 else "no"))

    # 1 honest: device stamps of a synchronized run
    runs = _runs(gen, big, LIE_RUNS)
    add("honest: production size, device stamps", "the device's time", _median([d for _, _, d in runs]), big["flops"], big["bytes"])
    # 2 enqueue only
    runs = _runs(gen, big, LIE_RUNS)
    add("clock stopped at enqueue", "the host queueing the work", _median([e for e, _, _ in runs]), big["flops"], big["bytes"])
    # 3 dead code: the result is never read, the kernel body is removed; the wait still happens
    empty = dict(flops=0, bytes=0, eff=1.0)
    runs = _runs(gen, empty, LIE_RUNS)
    add("result unused, kernel removed", "an empty dispatch and the wait", _median([e + d + w for e, w, d in runs]), big["flops"], big["bytes"])
    # 4 cached: the library returns the previous result for the same inputs, no dispatch at all
    lookup = [3.0 * (1.0 + ENQ_JITTER * gen.uniform()) for _ in range(LIE_RUNS)]
    add("same call on same inputs, result cached", "a table lookup on the host", _median(lookup), big["flops"], big["bytes"])
    # 5 wrong size: a benchmark at 2^14 floats, timed with a wait
    runs = _runs(gen, small, LIE_RUNS)
    add("benchmark at 2^14 floats, enqueue and wait", "the wait, mostly", _median([e + d + w for e, w, d in runs]), small["flops"], small["bytes"])
    # 6 cold clock: the same kernel while the clock has not ramped up
    runs = _runs(gen, big, LIE_RUNS, clock=COLD_CLOCK)
    add("production size, device not yet warmed up", "the device at 0.4 of its clock", _median([d for _, _, d in runs]), big["flops"], big["bytes"])
    # 7 matmul against a roof probed cold
    runs = _runs(gen, mm, LIE_RUNS)
    t_mm = _median([d for _, _, d in runs])
    cold_bound = max(mm["flops"] / (F_MS * COLD_CLOCK * 1e3), mm["bytes"] / (BW_MS * 1e3))
    rows.append(("matmul against a roof probed cold", "the device's time, against 0.4 of the arithmetic roof", f"{t_mm:,.2f}",
                 f"{cold_bound / t_mm:,.3f}", "yes" if cold_bound / t_mm > 1.05 else "no"))
    # 8 best of 100
    best = min(_floor_us(big) * _factor(gen) for _ in range(100))
    add("best of 100 runs, reported as the result", "the luckiest run", best, big["flops"], big["bytes"])
    lines = [
        "A catalogue of the ways to read a wrong time, on the model of `profiling.enqueue_lie`. The kernel is the addition c = a + b "
        "on 2^26 floats (67,108,864 FLOPs, 805,306,368 bytes, hidden efficiency 0.85), except where the row says otherwise; each "
        "row is the median of 11 runs (the best of 100 in the last row). 'Removed' means the kernel body is gone, so the dispatch "
        "costs only t_dsp = 1 microsecond on the device; 'cached' costs 3 microseconds on the host (times (1 + 0.20 u)) and no "
        "dispatch; 'not yet warmed up' multiplies the work time by 1 / 0.4; the roof probed cold is the arithmetic roof times 0.4. "
        "The fraction is the roofline time of the row's own work over the time read; 'caught' means the fraction is above 1.05, the "
        "threshold at which brain's profiler prints an error instead of a percentage (Chapter 3). "
        f"{_seed_line(name)} Draw order: rows in the order of the table; a row of runs draws five uniforms per run (enqueue jitter, "
        "wait jitter, u0, u1, u2); the cached row draws one uniform per run; the best-of-100 row draws three per run (u0, u1, u2).",
        "",
        _table(["situation", "what the timer saw", "time read (microseconds)", "fraction of the roofline", "caught by the roof test"], rows, "llrrl"),
    ]
    return "\n".join(lines)
