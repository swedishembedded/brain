# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Mixed-Precision Training".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: mixed_precision.<name> -->` markers. Helpers with a leading
underscore, or that take arguments, are shared with the chapter's figures so a
picture and the table beside it cannot disagree.

The number formats come from `number_formats` (imported, never copied): its
`quantize` rounds a value to bf16, fp16, E4M3 or E5M2 with exact rational
arithmetic. The two roundings that this chapter adds are written here from
its description: `rn_bf16` (nearest, ties to even) and `sr_bf16` (rounding up
with probability equal to the fraction of the gap covered), both on the bf16
grid of a normal number, whose spacing is 2^(e - 7) for 2^e <= |x| < 2^(e+1).

Swedish Embedded AB implements solutions for memory-bounded low-precision
training for its clients. If your team needs expertise in training large
models in 16 and 8 bits, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from fractions import Fraction
from functools import lru_cache

import bookrng
import number_formats as nf
import stable

_table = nf._table
_num = nf._num


# ------------------------------------------------------------------ the two roundings on the bf16 grid

def gap_bf16(x):
    """Spacing of the bf16 numbers around a normal x: 2^(e - 7) with 2^e <= |x| < 2^(e + 1)."""
    return math.ldexp(1.0, math.frexp(abs(x))[1] - 1 - 7)


def rn_bf16(x):
    """x rounded to the nearest bf16 number, ties to the even code (Python's round on the exact quotient x / gap)."""
    g = gap_bf16(x)
    return round(x / g) * g


def sr_bf16(x, u):
    """x rounded to a neighbouring bf16 number: the one away from zero when u < phi, where phi is the fraction of
    the gap that |x| covers above the lower neighbour, otherwise the one toward zero. u is a uniform draw in [0, 1)."""
    g = gap_bf16(x)
    a = abs(x)
    lo = math.floor(a / g) * g
    phi = (a - lo) / g
    r = lo + g if u < phi else lo
    return math.copysign(r, x)


def _self_check():
    """rn_bf16 agrees with the exact rounding of number_formats on a spread of values."""
    rng = bookrng.generator("mixed_precision.self_check")
    for _ in range(3000):
        x = rng.normal() * 10.0 ** (4 * rng.uniform() - 2)
        assert rn_bf16(x) == nf.quantize(x, "bf16"), x
    for k in range(1, 400):
        x = 1.0 + k / 997.0
        assert rn_bf16(x) == nf.quantize(x, "bf16"), x


# ------------------------------------------------------------------ opening: a stream of small updates

UPDATE_STREAM_N = 1000
UPDATE_STREAM_STEP = 0.001
UPDATE_STREAM_RUNS = 8


def update_stream():
    n, u = UPDATE_STREAM_N, UPDATE_STREAM_STEP
    # bf16 weight, nearest even
    w, moved = 1.0, 0
    for _ in range(n):
        new = rn_bf16(w + u)
        moved += new != w
        w = new
    rows = [["bf16 weight, nearest even", "-", _num(w, 6), moved]]
    # fp32 master, bf16 view
    m = 1.0
    for _ in range(n):
        m = nf.f32(m + u)
    rows.append(["fp32 weight (the master)", "-", _num(m, 6), "-"])
    rows.append(["bf16 view of that master", "-", _num(rn_bf16(m), 6), "-"])
    # stochastic rounding, eight runs from one generator
    rng = bookrng.generator("mixed_precision.update_stream")
    finals = []
    for r in range(UPDATE_STREAM_RUNS):
        w, moved = 1.0, 0
        for _ in range(n):
            new = sr_bf16(w + u, rng.uniform())
            moved += new != w
            w = new
        finals.append(w)
        rows.append([f"bf16 weight, stochastic, run {r}", r, _num(w, 6), moved])
    mean = sum(finals) / len(finals)
    seed = bookrng.seed_of("mixed_precision.update_stream")
    head = (f"A weight starts at 1 and receives {n} updates of {u}, each applied as w = round(w + {u}) where the sum is formed in exact "
            "arithmetic (float64 here) and rounded once. 'moved' counts the updates that changed the stored weight. The stochastic rows "
            f"come from one generator seeded {seed} (seed_of 'mixed_precision.update_stream') that runs across the eight runs, run after "
            f"run, one uniform() per update in order; run r therefore starts where run r - 1 stopped. The fp32 row adds {u} "
            "rounded to fp32 at every step (ties to even). The mean of the eight stochastic finals is "
            f"{_num(mean, 6)}.\n\n")
    return head + _table(["storage and rounding", "run", "weight after 1000 updates", "updates that moved it"], rows, "lrrr")


# ------------------------------------------------------------------ a weight that cannot reach its target

STALL_TARGET = 3.1
STALL_LR = 0.1
STALL_STEPS = 300
STALL_START = 1.0
STALL_RUNS = 8


@lru_cache(maxsize=None)
def _stall_runs():
    """Trajectories (working weight after each step) of the four rules on the loss (w - t)^2 / 2, and of eight stochastic runs."""
    t, eta, T = STALL_TARGET, STALL_LR, STALL_STEPS
    out = {}
    w, tr = STALL_START, []
    for _ in range(T):
        w = nf.f32(w - eta * (w - t))
        tr.append(w)
    out["f32"] = (tr, tr)
    w, tr = STALL_START, []
    for _ in range(T):
        w = rn_bf16(w - eta * (w - t))
        tr.append(w)
    out["rne"] = (tr, tr)
    theta, view, tr, mas = STALL_START, STALL_START, [], []
    for _ in range(T):
        view = rn_bf16(theta)
        theta = nf.f32(theta - eta * (view - t))
        tr.append(rn_bf16(theta))
        mas.append(theta)
    out["master"] = (tr, mas)
    rng = bookrng.generator("mixed_precision.stall")
    runs = []
    for _ in range(STALL_RUNS):
        w, tr = STALL_START, []
        for _ in range(T):
            w = sr_bf16(w - eta * (w - t), rng.uniform())
            tr.append(w)
        runs.append(tr)
    out["sr"] = runs
    return out


def stall():
    t, eta = STALL_TARGET, STALL_LR
    runs = _stall_runs()
    marks = (10, 30, 100, 300)
    rows = []
    for key, name in [("f32", "fp32 weight"), ("rne", "bf16 weight, nearest even"), ("master", "fp32 master, bf16 working copy")]:
        tr, mas = runs[key]
        rows.append([name] + [_num(tr[k - 1], 7) for k in marks] + [_num(abs(tr[-1] - t), 3), _num(abs(mas[-1] - t), 3)])
    gap = gap_bf16(3.0)
    radius = gap / (2 * eta)
    sr = runs["sr"]
    srows = []
    for r, tr in enumerate(sr):
        tail = sum(tr[-100:]) / 100
        srows.append([r, _num(tr[-1], 7), _num(tail, 5), _num(abs(tail - t), 2)])
    mean_tail = sum(sum(tr[-100:]) / 100 for tr in sr) / len(sr)
    seed = bookrng.seed_of("mixed_precision.stall")
    head = (f"The loss is (w - t)^2 / 2 with target t = {t}, so the gradient is w - t; plain gradient descent with rate eta = {eta} "
            f"from w = {STALL_START}, {STALL_STEPS} steps. Every step forms w - eta * (g) in exact arithmetic (float64) and rounds once to "
            "the storage format; fp32 rounds to fp32, ties to even. The master row evaluates the gradient at the bf16 working copy "
            "round(theta), updates the fp32 master theta, and shows the working copy round(theta). The last two columns are |w - t| after "
            "step 300 for the stored weight (the working copy in the master row) and for the master itself. "
            f"Gap of the bf16 numbers around 3: {_num(gap, 5)}; an update is absorbed when it is below half a gap, eta |w - t| < "
            f"{_num(gap / 2, 5)}, that is |w - t| < {_num(radius, 5)}.\n\n")
    t1 = _table(["rule", "after step 10", "after step 30", "after step 100", "after step 300", "|w - t| at 300", "|master - t| at 300"], rows)
    shead = (f"\n\nStochastic rounding of the same descent, eight runs from one generator seeded {seed} (seed_of 'mixed_precision.stall') that "
             "runs across the runs, one uniform() per step in order. 'mean of last 100' is the average weight over steps 201 to 300. "
             f"The mean of the eight averages is {_num(mean_tail, 5)}.\n\n")
    t2 = _table(["run", "w after step 300", "mean of last 100", "mean of last 100 minus t, absolute"], srows)
    return head + t1 + shead + t2


# ------------------------------------------------------------------ fp16 gradients, with and without a loss scale

ADAM_D = [10.0 ** (-1.5 * i) for i in range(6)]
ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS, ADAM_T = 0.01, 0.9, 0.999, 1e-8, 300
GROW_AFTER = 20


def _adam_run(mode, S0=1.0, nudge=0.0, unscale=True):
    """AdamW (no weight decay) on L = sum_i d_i (w_i - 1)^2 / 2 from w = 0, the gradient stored in fp16 after multiplying by S.

    mode 'f32': gradient kept in fp32, no scale. 'fixed': fp16 with the scale S0 throughout. 'dynamic': S starts at S0, is
    halved (and the step skipped) when any scaled gradient is not finite, and is doubled after GROW_AFTER consecutive applied
    steps. A skipped step changes nothing, the Adam step counter and the moments included."""
    n = len(ADAM_D)
    w = [0.0] * n
    w[0] += nudge
    m, v = [0.0] * n, [0.0] * n
    S, good, k, skipped, events = S0, 0, 0, 0, []
    for step in range(ADAM_T):
        g = [d * (wi - 1.0) for d, wi in zip(ADAM_D, w)]
        if mode == "f32":
            gu = g
        else:
            gs = [nf.quantize(S * x, "fp16") for x in g]
            if not all(math.isfinite(x) for x in gs):
                skipped += 1
                if mode == "dynamic":
                    S, good = S / 2, 0
                    events.append((step, S, "halved, step skipped"))
                continue
            gu = [x / S for x in gs] if unscale else gs
            if mode == "dynamic":
                good += 1
                if good == GROW_AFTER:
                    S, good = S * 2, 0
                    events.append((step, S, "doubled"))
        k += 1
        for i in range(n):
            m[i] = ADAM_B1 * m[i] + (1 - ADAM_B1) * gu[i]
            v[i] = ADAM_B2 * v[i] + (1 - ADAM_B2) * gu[i] ** 2
            mh, vh = m[i] / (1 - ADAM_B1 ** k), v[i] / (1 - ADAM_B2 ** k)
            w[i] -= ADAM_LR * mh / (math.sqrt(vh) + ADAM_EPS)
    return [abs(wi - 1.0) for wi in w], skipped, S, events


_ADAM_CASES = [
    ("fp32 gradients", dict(mode="f32")),
    ("fp16, no scale", dict(mode="fixed", S0=1.0)),
    ("fp16, S = 2^12", dict(mode="fixed", S0=2.0 ** 12)),
    ("fp16, S = 2^12, not divided back", dict(mode="fixed", S0=2.0 ** 12, unscale=False)),
    ("fp16, S = 2^16", dict(mode="fixed", S0=2.0 ** 16)),
    ("fp16, dynamic from 2^16", dict(mode="dynamic", S0=2.0 ** 16)),
]


def _adam_stable(case, i):
    kw = case
    base = _adam_run(**kw)[0][i]
    ws = [_adam_run(nudge=d, **kw)[0][i] for d in stable.NUDGES]
    return stable.text(base, ws, lambda x: _num(x, 3))


def loss_scale_adam():
    cols = []
    for name, kw in _ADAM_CASES:
        cols.append([_adam_stable(kw, i) for i in range(len(ADAM_D))])
    rows = []
    for i, d in enumerate(ADAM_D):
        rows.append([i, f"{d:.3g}"] + [c[i] for c in cols])
    skips = ["steps skipped"] + [""] + [str(_adam_run(**kw)[1]) for _, kw in _ADAM_CASES]
    rows.append(skips)
    head = (f"Six weights with curvature d_i = 10^(-1.5 i), i = 0 .. 5, on L = sum_i d_i (w_i - 1)^2 / 2, all starting at w_i = 0, so "
            f"g_i = d_i (w_i - 1); AdamW with rate {ADAM_LR}, beta1 = {ADAM_B1}, beta2 = {ADAM_B2}, epsilon = {ADAM_EPS:g}, no weight decay, "
            f"bias-corrected with the count of applied steps, {ADAM_T} steps. Only the gradient's storage differs: the scaled gradient "
            "S g_i is rounded once to fp16 (ties to even; a value of 65520 or more is infinity), divided by S, and fed to the update in "
            "float64. A step whose scaled gradient has a non-finite entry is skipped whole. The dynamic scale starts at 2^16, halves on "
            f"a skipped step and doubles after {GROW_AFTER} applied steps in a row. The cells are |w_i - 1| after the last step. "
            + stable.RULE + " In the nudged runs the first weight starts at the stated tiny amount instead of 0.\n\n")
    return head + _table(["i", "d_i"] + [n for n, _ in _ADAM_CASES], rows)


def scale_trace():
    _, skipped, S, events = _adam_run(mode="dynamic", S0=2.0 ** 16)
    rows = [[k + 1, st, f"2^{int(math.log2(sc))}", what] for k, (st, sc, what) in enumerate(events)]
    shown = rows[:10]
    doubles = sum(1 for _, _, _, what in rows if what == "doubled")
    head = (f"The dynamic run of the previous table, listed event by event (a step is numbered from 0). The scale started at 2^16. "
            f"Over the {ADAM_T} steps it halved {len(events) - doubles} times (each with a skipped step), doubled {doubles} times, "
            f"skipped {skipped} steps and ended at 2^{int(math.log2(S))}. The first ten events:\n\n")
    return head + _table(["event", "at step", "scale after it", "what happened"], shown, "rrrl")


# ------------------------------------------------------------------ stochastic rounding on the bit pattern

SR_BIT_VALUES = [1.001, 1.0 + 2.0 ** -8, 1.0, -1.001, 3.1, 0.001]


def _fp32_sr_by_bits(b, r):
    """The bf16 pattern (as 16 bits) from the fp32 pattern b by adding r (0 .. 65535) and keeping the top 16 bits."""
    return ((b + r) & 0xFFFFFFFF) >> 16


def _bf16_bits_value(h):
    return nf.bits_f32(h << 16)


def sr_bit_rule():
    rows = []
    for x in SR_BIT_VALUES:
        xf = nf.f32(x)
        b = nf.f32_bits(xf)
        low = b & 0xFFFF
        carries = 0
        total = Fraction(0)
        for r in range(65536):
            h = _fp32_sr_by_bits(b, r)
            carries += h != (b >> 16)
            total += Fraction(_bf16_bits_value(h))
        mean = total / 65536
        exact = Fraction(xf)
        rows.append([f"{x:g}", f"0x{b:08X}", f"0x{low:04X} = {low}", carries, f"{low}/65536 = {low / 65536:.6f}", "yes" if mean == exact else "no"])
    head = ("Each value is first rounded to fp32 (ties to even), then written as its 32-bit pattern. All 65536 values r = 0 .. 65535 are "
            "added to the pattern as an unsigned integer, the top 16 bits are kept as the bf16 pattern, and the result is compared with "
            "the pattern without the addition. 'carried' counts the r for which the bf16 pattern changed. 'mean equals x' compares "
            "the exact mean of the 65536 bf16 results with the fp32 value, in exact rational arithmetic.\n\n")
    return head + _table(["x", "fp32 pattern", "dropped 16 bits", "r that carried", "carry probability", "mean equals x"], rows, "lrrrrl")


SR_STAT_X = 1.001
SR_STAT_DRAWS = 100_000


def sr_statistics():
    x = nf.f32(SR_STAT_X)
    g = gap_bf16(x)
    lo = math.floor(x / g) * g
    phi = (x - lo) / g
    rng = bookrng.generator("mixed_precision.sr_statistics")
    ups = 0
    total = 0.0
    sq = 0.0
    for _ in range(SR_STAT_DRAWS):
        y = sr_bf16(x, rng.uniform())
        ups += y > lo
        total += y
        sq += (y - x) ** 2
    mean = total / SR_STAT_DRAWS
    var_meas = sq / SR_STAT_DRAWS
    var_pred = phi * (1 - phi) * g * g
    near = rn_bf16(x)
    seed = bookrng.seed_of("mixed_precision.sr_statistics")
    rows = [
        ["lower neighbour of x", _num(lo, 8)],
        ["upper neighbour of x (gap above)", f"{_num(lo + g, 8)} ({_num(g, 5)})"],
        ["phi = (x - lower) / gap", _num(phi, 6)],
        ["nearest even rounding of x", _num(near, 8)],
        ["share of draws rounded up (phi predicts it)", _num(ups / SR_STAT_DRAWS, 4)],
        ["mean of the stochastic results", _num(mean, 7)],
        ["mean minus x", _num(mean - x, 3)],
        ["standard error of that mean, sqrt(phi (1 - phi) / draws) gap", _num(math.sqrt(phi * (1 - phi) / SR_STAT_DRAWS) * g, 3)],
        ["mean of (result - x)^2 over the draws", _num(var_meas, 4)],
        ["phi (1 - phi) gap^2, the predicted value", _num(var_pred, 4)],
        ["(nearest - x)^2 for nearest even rounding", _num((near - x) ** 2, 4)],
    ]
    head = (f"x = {SR_STAT_X} rounded to fp32 is {_num(x, 9)}. {SR_STAT_DRAWS} draws u_i = uniform() from the generator seeded {seed} "
            "(seed_of 'mixed_precision.sr_statistics'), in order, each giving one stochastic rounding of x to bf16.\n\n")
    return head + _table(["quantity", "value"], rows, "lr")


# ------------------------------------------------------------------ a draw that depends on the element, not the order

def mix32(x):
    """Scrambling of a 32-bit word in five operations: xor with the word shifted right 16, multiply by 0x7FEB352D (mod 2^32), xor with
    the word shifted right 15, multiply by 0x846CA68B (mod 2^32), xor with the word shifted right 16."""
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    return x ^ (x >> 16)


def counter_uniform(key, i):
    """The draw of element i under a 32-bit key: the top 24 bits of mix32(mix32(key xor i) xor 0x9E3779B9), over 2^24."""
    return (mix32(mix32((key ^ i) & 0xFFFFFFFF) ^ 0x9E3779B9) >> 8) / 2.0 ** 24


COUNTER_N = 12
COUNTER_KEY = 0x1234ABCD


def _sr_vector(vals, tiles, draw):
    """Round the vector vals to bf16 stochastically, one tile of the listed (start, length) at a time, in the listed order."""
    out = [0.0] * len(vals)
    for start, n in tiles:
        for j in range(start, start + n):
            out[j] = sr_bf16(vals[j], draw(j))
    return out


def counter_draws():
    vals = [1.0 + 0.001 * (j + 1) for j in range(COUNTER_N)]
    whole = [(0, COUNTER_N)]
    cut5 = [(0, 5), (5, 5), (10, 2)]
    cut5_rev = list(reversed(cut5))
    cut3 = [(j, 3) for j in range(0, COUNTER_N, 3)]
    ref = _sr_vector(vals, whole, lambda j: counter_uniform(COUNTER_KEY, j))
    rows = []
    for name, tiles in [("one tile of 12", whole), ("tiles of 5, 5, 2 in order", cut5), ("tiles of 5, 5, 2 in reverse order", cut5_rev),
                        ("tiles of 3 in order", cut3)]:
        a = _sr_vector(vals, tiles, lambda j: counter_uniform(COUNTER_KEY, j))
        rng = bookrng.generator("mixed_precision.counter_draws")
        b = _sr_vector(vals, tiles, lambda j: rng.uniform())
        ref_stream = _sr_vector(vals, whole, (lambda r: (lambda j: r.uniform()))(bookrng.generator("mixed_precision.counter_draws")))
        rows.append([name, sum(x != y for x, y in zip(a, ref)), sum(x != y for x, y in zip(b, ref_stream))])
    seed = bookrng.seed_of("mixed_precision.counter_draws")
    head = (f"Twelve values x_j = 1 + 0.001 (j + 1), j = 0 .. 11, are rounded to bf16 stochastically. The counter draw is u(j) = "
            f"counter_uniform({COUNTER_KEY:#x}, j) (mix32 twice, as defined above); the stream draw takes the next uniform() "
            f"of a generator seeded {seed} (seed_of 'mixed_precision.counter_draws'), created afresh for each row, in the order the "
            "tiles are processed. Each row counts the elements whose rounded value differs from the same rule applied in one tile of 12 "
            "in order.\n\n")
    return head + _table(["order of work", "elements that differ, counter draw", "elements that differ, stream draw"], rows, "lrr")


# ------------------------------------------------------------------ fp8 products

FP8_M, FP8_K, FP8_N = 32, 64, 16
E4M3_MAX, E5M2_MAX = 448.0, 57344.0


def _quant_tensor(A, f, top):
    """Every entry divided by s = amax / top, rounded to format f (saturating), times s: one scale for the whole matrix."""
    amax = max(abs(x) for r in A for x in r)
    s = amax / top
    return [[s * nf.quantize(x / s, f, True) for x in r] for r in A]


def _quant_blocks(A, f, top, b=16):
    """The same with one scale per block of b consecutive entries of a row."""
    out = []
    for r in A:
        o = []
        for j in range(0, len(r), b):
            c = r[j:j + b]
            s = max(abs(x) for x in c) / top
            o += [s * nf.quantize(x / s, f, True) if s > 0 else 0.0 for x in c]
        out.append(o)
    return out


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def fp8_operands():
    rng = bookrng.generator("mixed_precision.fp8_operands")
    X0 = [[rng.normal() for _ in range(FP8_K)] for _ in range(FP8_M)]
    W = [[0.1 * rng.normal() for _ in range(FP8_N)] for _ in range(FP8_K)]
    bf = lambda A: [[rn_bf16(x) if x else 0.0 for x in r] for r in A]
    rows = []
    for decades in (0, 3, 6, 9):
        X = [[x * 10.0 ** (-decades * i / (FP8_M - 1)) for x in r] for i, r in enumerate(X0)]
        Y = _matmul(X, W)

        def row_error(Q):
            tot = 0.0
            for r1, r2 in zip(Q, Y):
                tot += math.sqrt(sum((p - q) ** 2 for p, q in zip(r1, r2)) / sum(q * q for q in r2))
            return tot / FP8_M

        rows.append([decades,
                     f"{row_error(_matmul(bf(X), bf(W))):.4f}",
                     f"{row_error(_matmul(_quant_tensor(X, 'E4M3', E4M3_MAX), _quant_tensor(W, 'E4M3', E4M3_MAX))):.4f}",
                     f"{row_error(_matmul(_quant_blocks(X, 'E4M3', E4M3_MAX), _quant_tensor(W, 'E4M3', E4M3_MAX))):.4f}",
                     f"{row_error(_matmul(_quant_tensor(X, 'E5M2', E5M2_MAX), _quant_tensor(W, 'E5M2', E5M2_MAX))):.4f}"])
    seed = bookrng.seed_of("mixed_precision.fp8_operands")
    head = (f"Y = X W with X of shape [{FP8_M}, {FP8_K}] and W of shape [{FP8_K}, {FP8_N}]. From the generator seeded {seed} (seed_of "
            f"'mixed_precision.fp8_operands'): the {FP8_M * FP8_K} entries of X as normal() in row-major order, then the {FP8_K * FP8_N} "
            "entries of W in row-major order as 0.1 * normal(). Row i of X is then multiplied by 10^(-a i / 31) for the stated number of "
            "decades a, so the rows of X span a decades of magnitude. The reference Y is formed in float64 from the unrounded matrices; "
            "the rounded operands are multiplied and summed in float64 too (a wide accumulator). The cell is the mean over the 32 rows "
            "of ||row of the rounded product - row of Y|| / ||row of Y|| (2-norm, mean divisor 32). bf16 rounds each entry to nearest "
            "even. 'E4M3, one scale per tensor' divides each matrix by s = amax / 448, rounds to E4M3 (nearest even, saturating) and "
            "multiplies back; 'E4M3, X in blocks of 16' gives X one scale per 16 consecutive entries of a row (W keeps one per tensor); "
            "'E5M2' is the tensor rule with 57344. Errors are printed to 4 decimals.\n\n")
    return head + _table(["decades spanned by the rows of X", "bf16", "E4M3, one scale per tensor", "E4M3, X in blocks of 16",
                          "E5M2, one scale per tensor"], rows)


GRAD_N = 5000
GRAD_DECADES = 6


def fp8_gradients():
    rng = bookrng.generator("mixed_precision.fp8_gradients")
    g = [10.0 ** (-GRAD_DECADES * rng.uniform()) for _ in range(GRAD_N)]
    amax = max(g)
    rows = []
    for f, top in (("E4M3", E4M3_MAX), ("E5M2", E5M2_MAX)):
        s = amax / top
        q = [s * nf.quantize(x / s, f, True) for x in g]
        zero = sum(1 for y in q if y == 0)
        mn = math.ldexp(1.0, 1 - nf.bias(f))
        sub = sum(1 for y in q if 0 < abs(y / s) < mn)
        err_all = sum(1.0 if y == 0 else abs(y - x) / x for x, y in zip(g, q)) / len(g)
        rows.append([f, _num(top, 6), f"{100 * zero / GRAD_N:.2f}%", f"{100 * sub / GRAD_N:.2f}%", f"{err_all:.4f}", f"{sum(q) / sum(g) - 1:+.4f}"])
    seed = bookrng.seed_of("mixed_precision.fp8_gradients")
    head = (f"{GRAD_N} stand-in gradient magnitudes g_i = 10^(-{GRAD_DECADES} u_i), u_i = uniform() drawn in order from the generator seeded "
            f"{seed} (seed_of 'mixed_precision.fp8_gradients'), so they span {GRAD_DECADES} decades below the largest. Each is divided by "
            "s = amax / (largest finite number of the format), rounded once to the format (nearest even, saturating) and multiplied back. "
            "'zero' is the share stored as 0, 'subnormal' the share stored below the smallest normal number of the format, 'mean "
            "relative error' the mean of |stored - g| / g with a zero counted as 1, and 'error of the sum' the relative error of the "
            "sum of all stored values against the sum of all g.\n\n")
    return head + _table(["format", "amax maps to", "zero", "subnormal", "mean relative error", "error of the sum"], rows)


DELAY_N = 1000
DELAY_WINDOW = 4
DELAY_STEPS = 16


def delayed_scale():
    rng = bookrng.generator("mixed_precision.delayed_scale")
    hist, sigma, rows = [], 1.0, []
    for step in range(DELAY_STEPS):
        sigma *= 1.05
        if step == 10:
            sigma *= 4
        x = [sigma * rng.normal() for _ in range(DELAY_N)]
        amax = max(abs(v) for v in x)
        s_now = amax / E4M3_MAX
        s_old = (max(hist[-DELAY_WINDOW:]) if hist else amax) / E4M3_MAX

        def stats(s):
            clipped = sum(1 for v in x if abs(v) / s >= 464.0)
            q = [s * nf.quantize(v / s, "E4M3", True) for v in x]
            return clipped, math.sqrt(sum((a - b) ** 2 for a, b in zip(x, q)) / sum(a * a for a in x))

        c0, e0 = stats(s_now)
        c1, e1 = stats(s_old)
        rows.append([step, _num(sigma, 4), _num(amax, 4), _num(s_old * E4M3_MAX, 4), c1, f"{e1:.4f}", c0, f"{e0:.4f}"])
        hist.append(amax)
    seed = bookrng.seed_of("mixed_precision.delayed_scale")
    head = (f"One tensor of {DELAY_N} entries per step, x = sigma_k * normal() drawn in order from the generator seeded {seed} (seed_of "
            f"'mixed_precision.delayed_scale') for steps k = 0 .. {DELAY_STEPS - 1}; sigma_k is 1.05 times sigma_(k-1) from sigma_(-1) = 1, "
            "and at step 10 it is multiplied by a further 4. The current scale divides amax of this step's tensor by 448; the "
            f"delayed scale divides the largest amax of the previous {DELAY_WINDOW} steps by 448 (step 0 uses its own amax). 'clipped' "
            "counts the entries with |x| / s >= 464 (the midpoint between 448 and the next code, so a saturating conversion changes "
            "them); 'rms error' is the root of the sum of squared errors over the sum of squares of the entries after rounding to E4M3 "
            "(nearest even, saturating) and scaling back.\n\n")
    return head + _table(["step", "sigma", "amax of this step", "amax the delayed scale uses", "clipped, delayed", "rms error, delayed",
                          "clipped, current", "rms error, current"], rows)


# ------------------------------------------------------------------ the bill of a policy

# value dtype bytes, gradient dtype bytes, state per element per slot, master bytes (the formats of paramstore::precision)
def _state_bytes(fmt, numel, block=None):
    if fmt == "f32":
        return 4 * numel
    if fmt == "bf16":
        return 2 * numel
    return numel + 4 * -(-numel // block)  # int8 codes plus one f32 scale per block


MANIFEST = [
    ("embed.weight", 16_384_000),
    ("blk.0.attn.weight", 1_048_576),
    ("blk.0.attn_norm.weight", 4096),
    ("blk.0.mlp.weight", 2_097_152),
    ("out_norm.weight", 512),
    ("blk.0.attn.bias", 512),
]
FP32_PREC = dict(value="f32", grad="f32", state="f32", master=None, sr=False)
BIG_PREC = dict(value="bf16", grad="bf16", state="int8:64", master="f32", sr=True)
POLICY_RULES = [("name contains 'norm'", lambda n, c: "norm" in n, FP32_PREC),
                ("at least 1024 elements", lambda n, c: c >= 1024, BIG_PREC)]


def _resolve(name, numel):
    for i, (_, match, prec) in enumerate(POLICY_RULES):
        if match(name, numel):
            return i, prec
    return None, FP32_PREC


def _param_bytes(prec, numel, slots=2):
    v = {"f32": 4, "bf16": 2}[prec["value"]] * numel
    g = {"f32": 4, "bf16": 2}[prec["grad"]] * numel
    st = prec["state"]
    s = slots * (_state_bytes(st, numel) if st in ("f32", "bf16") else _state_bytes("int8", numel, int(st.split(":")[1])))
    m = 0 if prec["master"] is None else _state_bytes(prec["master"], numel)
    return v, g, s, m


def precision_policy():
    rows, tot, tot32 = [], [0, 0, 0, 0], 0
    for name, numel in MANIFEST:
        i, prec = _resolve(name, numel)
        v, g, s, m = _param_bytes(prec, numel)
        for k, x in enumerate((v, g, s, m)):
            tot[k] += x
        tot32 += 16 * numel
        rows.append([name, f"{numel:,}", "default" if i is None else f"rule {i}", f"{v:,}", f"{g:,}", f"{s:,}", f"{m:,}", f"{v + g + s + m:,}"])
    rows.append(["total", f"{sum(n for _, n in MANIFEST):,}", "", f"{tot[0]:,}", f"{tot[1]:,}", f"{tot[2]:,}", f"{tot[3]:,}", f"{sum(tot):,}"])
    head = ("Six tensors resolved by an ordered list of two rules, the first that matches deciding: rule 0 (name contains 'norm') gives "
            "fp32 value, fp32 gradient, fp32 moments and no master; rule 1 (at least 1024 elements) gives a bf16 value, a bf16 gradient, "
            "8-bit moments in blocks of 64 elements (one byte per element plus one fp32 scale, 4 bytes, per block, the last block "
            "counted whole), an fp32 master and stochastic rounding; the default is fp32 everything with no master. AdamW has two "
            "moment slots. Bytes are physical, scales included. 'moments' is both slots together.\n\n")
    foot = (f"\n\nWith every tensor in fp32 and no master the same manifest costs 16 bytes per element, {tot32:,} bytes in all; the policy "
            f"above costs {sum(tot):,}, which is {100 * sum(tot) / tot32:.2f}% of that.")
    return head + _table(["tensor", "elements", "rule", "value", "gradient", "moments", "master", "total"], rows) + foot


if __name__ == "__main__":
    _self_check()
    for fn in (update_stream, stall, loss_scale_adam, scale_trace, sr_bit_rule, sr_statistics, counter_draws, fp8_operands,
               fp8_gradients, delayed_scale, precision_policy):
        print(fn(), "\n")
