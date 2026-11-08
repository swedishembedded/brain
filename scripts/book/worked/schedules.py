# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Learning-Rate Schedules and Gradient Clipping".

Plain Python 3 on floats (IEEE double precision), written from the chapter's
own description of AdamW, of the schedules and of the clip, and independent of
any engine code. Every public function returns a markdown string that the
chapter embeds between `<!-- worked: schedules.<name> -->` markers; the same
module supplies the curves of the chapter's figures.

The network, the data and the starting weights are those of Chapter 7 (the
`optimizers` module builds them from the generator named `optimizers.mlp`);
this module trains that network with its own AdamW, with a learning rate that
is a function of the step, an optional clip of the gradient and an optional
outlier batch, on batches drawn from its own generators.

Counting rule used for work: one multiplication, addition, subtraction,
division or square root is one FLOP. Bytes are those read from and written to
memory, every array in 32-bit floats unless a row says otherwise.

Swedish Embedded AB implements solutions for stable training of neural
networks. If your team needs expertise in learning-rate schedules, gradient
clipping and loss-curve diagnosis, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
import struct

import bookrng
import optimizers as opt
import stable

# ------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _sci(v, d=2):
    return f"{v:.{d}e}"


def _seed(name):
    s = bookrng.seed_of(name)
    return f"0x{s:016x}"


# ------------------------------------------------------------------ the schedules
#
# A step index t is 0-based: step t uses the rate eta(t). The warmup is W steps long, the run T steps long.


def warmup_rate(t, peak, W):
    """Linear warmup: peak (t + 1) / W for t < W, so that step W - 1 is the first at the peak."""
    return peak * (t + 1) / W


def constant(peak):
    return lambda t: peak


def warmup_constant(peak, W):
    return lambda t: warmup_rate(t, peak, W) if t < W else peak


def step_decay(peak, W, S, gamma, floor):
    """Warmup, then multiply by gamma after every S steps, never below the floor."""
    def f(t):
        if t < W:
            return warmup_rate(t, peak, W)
        return max(floor, peak * gamma ** ((t - W) // S))
    return f


def exp_decay(peak, W, gamma, floor):
    def f(t):
        if t < W:
            return warmup_rate(t, peak, W)
        return max(floor, peak * gamma ** (t - W))
    return f


def inv_sqrt(peak, W, floor):
    """Warmup, then peak sqrt(W / (t + 1)): the rate falls like one over the root of the step count."""
    def f(t):
        if t < W:
            return warmup_rate(t, peak, W)
        return max(floor, peak * math.sqrt(W / (t + 1)))
    return f


def linear_decay(peak, floor, W, T):
    """Warmup W, then a straight line from the peak to the floor at step T (held at the floor after)."""
    def f(t):
        if t < W:
            return warmup_rate(t, peak, W)
        u = min(1.0, (t - W) / (T - W))
        return peak - (peak - floor) * u
    return f


def cosine(peak, floor, W, T, H=0):
    """Warmup W, hold H at the peak, then half a cosine from peak to floor ending at step T (held at the floor after)."""
    start = W + H

    def f(t):
        if t < W:
            return warmup_rate(t, peak, W)
        if t < start:
            return peak
        u = min(1.0, (t - start) / (T - start))
        return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * u))
    return f


# ------------------------------------------------------------------ AdamW on the Chapter 7 network

LAMBDA = 0.01  # decoupled weight decay of every run in this chapter
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
RUN_STEPS = 600
BATCH = opt.MLP_B


def _problem():
    return opt.mlp_problem()


_PROBLEM = {}


def problem():
    if not _PROBLEM:
        _PROBLEM["p"] = _problem()
    return _PROBLEM["p"]


def batch_lists(stream, steps):
    """Index lists of `steps` batches: shuffle 0 .. 199, cut into batches of 20, reshuffle at the end of each epoch."""
    rng = bookrng.generator(stream)
    order = list(range(opt.MLP_N))
    out = []
    while len(out) < steps:
        rng.shuffle(order)
        for s in range(0, opt.MLP_N, BATCH):
            out.append(order[s:s + BATCH])
    return out[:steps]


def l2(v):
    return math.sqrt(sum(x * x for x in v))


class Run:
    """Everything one training run records.

    `losses` maps a step count s (0 .. steps) to the mean loss over all 200 examples after s updates (recorded for
    s <= dense and then every `every` steps); `norms[t]` is the gradient norm of step t before any clip; `clipped[t]`
    says whether the clip changed that gradient; `update_rms[t]` is the root mean square over the 33 parameters of the
    Adam ratio mhat / (sqrt(vhat) + eps) of step t (the step in units of the learning rate)."""

    def __init__(self):
        self.losses, self.norms, self.clipped, self.update_rms = {}, [], [], []
        self._args, self._twins = None, None

    @property
    def twins(self):
        """The same run with the first starting weight changed by each of `stable.NUDGES`: a quantity that differs
        between the runs is chaotic, and its digits cannot be reproduced by another implementation."""
        if self._twins is None:
            schedule, stream, steps, clip, outlier, dense, every = self._args
            self._twins = [train(schedule, stream, steps, clip, outlier, dense, every, nudge=d) for d in stable.NUDGES]
        return self._twins


def train(schedule, stream="schedules.batches", steps=RUN_STEPS, clip=None, outlier=None, dense=60, every=10, nudge=0.0):
    """AdamW (beta1 0.9, beta2 0.999, eps 1e-8, decoupled decay LAMBDA) with learning rate schedule(t) at step t.

    clip: None, or tau: when ||g|| > tau the gradient is multiplied by tau / (||g|| + 1e-6) (global-norm clip).
    outlier: None, or (step, factor): that step's gradient is multiplied by factor before anything else."""
    net, xs, ys, theta = problem()
    theta = list(theta)
    theta[0] += nudge
    n = len(theta)
    m, v = [0.0] * n, [0.0] * n
    run = Run()
    run._args = (schedule, stream, steps, clip, outlier, dense, every)
    run.losses[0] = net.loss(theta, xs, ys)
    for t, idx in enumerate(batch_lists(stream, steps)):
        g = net.gradient(theta, [xs[i] for i in idx], [ys[i] for i in idx])
        if outlier is not None and t == outlier[0]:
            g = [outlier[1] * x for x in g]
        norm = l2(g)
        run.norms.append(norm)
        if clip is not None and norm > clip:
            scale = clip / (norm + 1e-6)
            g = [scale * x for x in g]
            run.clipped.append(True)
        else:
            run.clipped.append(False)
        lr = schedule(t)
        s = t + 1
        bc1, bc2 = 1 - BETA1 ** s, 1 - BETA2 ** s
        ss = 0.0
        for i in range(n):
            m[i] = BETA1 * m[i] + (1 - BETA1) * g[i]
            v[i] = BETA2 * v[i] + (1 - BETA2) * g[i] * g[i]
            ratio = (m[i] / bc1) / (math.sqrt(v[i] / bc2) + EPS)
            ss += ratio * ratio
            theta[i] -= lr * LAMBDA * theta[i] + lr * ratio
        run.update_rms.append(math.sqrt(ss / n))
        if s <= dense or s % every == 0:
            run.losses[s] = net.loss(theta, xs, ys)
    return run


def cell(f, runs, digits=4):
    """The mean over `runs` of the quantity f(run), to `digits` decimals, with only the digits that survive in the same
    mean over every set of nudged twins of the runs (see `stable.text`)."""
    v = sum(f(r) for r in runs) / len(runs)
    ws = [sum(f(r.twins[k]) for r in runs) / len(runs) for k in range(len(stable.NUDGES))]
    return stable.text(v, ws, lambda x: _fx(x, digits))


def span_cell(f, runs, digits=4):
    """'lowest to highest' of f over the runs, each end written with `cell`'s rule, or 'unstable' when any run is."""
    parts = [cell(f, [r], digits) for r in runs]
    if "unstable" in parts:
        return "unstable"
    vs = [f(r) for r in runs]
    return f"{parts[vs.index(min(vs))]} to {parts[vs.index(max(vs))]}"


def worst(run, lo, hi):
    """The largest recorded loss after s updates for lo <= s <= hi."""
    return max(v for s, v in run.losses.items() if lo <= s <= hi)


def tail(run, steps=RUN_STEPS):
    """The score of a run: the mean loss at s = steps - 50, steps - 40, ..., steps."""
    pts = [steps - 50 + 10 * k for k in range(6)]
    return sum(run.losses[s] for s in pts) / len(pts)


ORDERS = ["schedules.batches"] + [f"schedules.batches.{k}" for k in range(1, 5)]


def _unstable_note():
    return "Rounding noise grows into any difference in a chaotic run. " + stable.RULE


def _setup_text():
    net, xs, ys, theta = problem()
    return (f"The network, data and starting weights are those of Chapter 7 (generator `optimizers.mlp`, seed {_seed('optimizers.mlp')}; "
            f"starting loss {_fx(net.loss(theta, xs, ys), 4)}). AdamW with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and decoupled decay "
            f"lambda = {LAMBDA:g}; {RUN_STEPS} steps of batch size {BATCH}; batches from the generator `schedules.batches` "
            f"(seed {_seed('schedules.batches')}): shuffle the list 0 .. {opt.MLP_N - 1}, cut it into consecutive batches of {BATCH}, "
            "reshuffle at the end of each epoch.")


# ------------------------------------------------------------------ 1. a constant rate


PEAK_HIGH = 3.0       # the peak of the warmup table: thirty times the best constant
PEAK_COMPARE = 1.0    # the peak of the schedule comparison: ten times the best constant
PEAK_TUNED = 0.1
CLEAN_PEAK = 0.3
CONSTANT_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


def constant_rates():
    rows = []
    for lr in CONSTANT_GRID:
        r = train(constant(lr))
        rows.append([f"{lr:g}", cell(lambda q: q.losses[2], [r]), cell(lambda q: worst(q, 1, 50), [r]), cell(lambda q: q.losses[100], [r]),
                     cell(lambda q: worst(q, 301, RUN_STEPS), [r]), cell(tail, [r])])
    head = (_setup_text() + " Entries: the mean loss over all 200 examples after 2 updates; the largest loss over steps 1 to 50 (every step "
            "recorded); the loss after 100 updates; the largest loss over steps 301 to 600 (recorded every 10 steps: steps 310, 320, ..., 600); the mean loss at steps 550, 560, ..., 600 (the tail). "
            + _unstable_note() + "\n\n")
    return head + _table(["constant eta", "loss at step 2", "worst, steps 1-50", "step 100", "worst, steps 301-600", "tail"], rows)


# ------------------------------------------------------------------ 2. why the early steps are large


ES_K, ES_T = 4000, 300
ES_TIMES = (1, 2, 3, 5, 10, 20, 50, 100, 300)


def adam_early_steps():
    """For a coordinate whose gradient is pure noise, g_t ~ N(0, 1) independently, the RMS of the Adam ratio after t steps."""
    name = "schedules.early_steps"
    rng = bookrng.generator(name)
    ssq = {t: 0.0 for t in ES_TIMES}
    spread_sum = {t: [0.0, 0.0] for t in ES_TIMES}
    for _ in range(ES_K):
        m = v = 0.0
        for t in range(1, ES_T + 1):
            g = rng.normal()
            m = BETA1 * m + (1 - BETA1) * g
            v = BETA2 * v + (1 - BETA2) * g * g
            if t in ssq:
                root = math.sqrt(v / (1 - BETA2 ** t))
                u = (m / (1 - BETA1 ** t)) / (root + EPS)
                ssq[t] += u * u
                spread_sum[t][0] += root
                spread_sum[t][1] += root * root
    rows = []
    for t in ES_TIMES:
        analytic = math.sqrt((1 - BETA1) / (1 + BETA1) * (1 - BETA1 ** (2 * t))) / (1 - BETA1 ** t)
        mean_root = spread_sum[t][0] / ES_K
        var_root = spread_sum[t][1] / ES_K - mean_root ** 2
        limit0 = math.sqrt((1 - BETA1) / (1 + BETA1))
        rows.append([t, _fx(analytic, 4), _fx(math.sqrt(ssq[t] / ES_K), 4), _fx(math.sqrt(max(var_root, 0.0)) / mean_root, 3),
                     _fx(limit0 / analytic, 4), _fx(min(1.0, t / 50), 4)])
    limit = math.sqrt((1 - BETA1) / (1 + BETA1))
    head = (f"{ES_K} independent coordinates, each with gradients g_1, g_2, ... drawn from the standard normal distribution by the generator `{name}` "
            f"(seed {_seed(name)}): coordinate by coordinate, {ES_T} calls of normal() each, in time order. Adam with beta1 = 0.9, beta2 = 0.999, "
            "epsilon = 1e-8 and bias correction. Columns: the exact RMS of the corrected first moment, sqrt((1 - beta1)/(1 + beta1) (1 - beta1^(2t))) / (1 - beta1^t) "
            "(unit gradient variance); the measured RMS of the whole Adam ratio mhat / (sqrt(vhat) + epsilon) over the coordinates (its sampling error is about 1.2%); "
            "the measured relative spread (standard deviation with divisor 4000, the number of coordinates, over the mean) of sqrt(vhat) across the coordinates (a few percent of sampling error); the learning rate, as a fraction of its final value, that would keep the typical step at its "
            "steady size (the limit divided by the first column); and, for comparison, the linear warmup t/50. "
            f"The limit of the first column for large t is sqrt((1 - beta1)/(1 + beta1)) = {_fx(limit, 4)}.\n\n")
    return head + _table(["t", "RMS of mhat (exact)", "RMS of the Adam ratio (measured)", "spread of sqrt(vhat) (measured)", "rate that keeps the step steady", "linear ramp t/50"], rows)


LAM_Q, S2_Q = 1.0, 2.0


def noise_vs_rate():
    """SGD on one weight with curvature lam and minibatch noise variance s2: stationary error variance and steps to forget the start."""
    rows = []
    for eta in (0.5, 0.25, 0.125, 0.0625, 0.03125):
        v = eta * S2_Q / (LAM_Q * (2 - eta * LAM_Q))
        k = math.log(100) / -math.log(1 - eta * LAM_Q)
        rows.append([f"{eta:g}", _fx(v, 4), _fx(math.sqrt(v), 4), f"{k:.1f}"])
    head = ("SGD on one weight with a quadratic loss of curvature lambda = 1 and minibatch noise of variance s^2 = 2 per step (the setting of Chapter 7's stationary variance). "
            "Columns: the stationary mean square error V = eta s^2 / (lambda (2 - eta lambda)); its square root; the number of steps in which the average error shrinks by a factor 100, "
            "ln(100) / (-ln(1 - eta lambda)).\n\n")
    return head + _table(["eta", "V", "sqrt(V)", "steps to forget the start (factor 100)"], rows)


# ------------------------------------------------------------------ 3. warmup


WARMUPS = (0, 10, 30, 60, 120)


def warmup_lengths():
    rows = []
    for W in WARMUPS:
        sched = warmup_constant(PEAK_HIGH, W) if W else constant(PEAK_HIGH)
        r = train(sched)
        rows.append([W if W else "none"] + [cell(lambda q, s=s: q.losses[s], [r], 3) for s in (1, 2, 3, 5, 10, 20, 50)]
                    + [cell(lambda q: worst(q, 1, 50), [r], 3), cell(lambda q: q.losses[100], [r], 3)])
    head = (_setup_text() + f" The learning rate is eta(t) = {PEAK_HIGH:g} (t + 1) / W for t < W and {PEAK_HIGH:g} afterwards; W = none means {PEAK_HIGH:g} from the first step. "
            "Entries: the mean loss over all 200 examples after the stated number of updates; 'worst' is the largest of the losses after 1, 2, ..., 50 updates. " + _unstable_note() + "\n\n")
    return head + _table(["warmup W", "step 1", "step 2", "step 3", "step 5", "step 10", "step 20", "step 50", "worst, 1-50", "step 100"], rows)


# ------------------------------------------------------------------ 4. the shapes of a schedule

SHAPE_T, SHAPE_W = 600, 30
SHAPE_PEAK, SHAPE_FLOOR = 0.3, 0.003


def shape_functions():
    p, f, W, T = SHAPE_PEAK, SHAPE_FLOOR, SHAPE_W, SHAPE_T
    return [
        ("constant", constant(p)),
        ("step decay (x 0.3 every 150)", step_decay(p, W, 150, 0.3, f)),
        ("exponential (0.99 per step)", exp_decay(p, W, 0.99, f)),
        ("inverse square root", inv_sqrt(p, W, f)),
        ("cosine to the floor", cosine(p, f, W, T)),
        ("warmup, hold 300, cosine", cosine(p, f, W, T, H=300)),
    ]


def shape_table():
    shapes = shape_functions()
    marks = (0, 29, 30, 150, 300, 450, 599)
    rows = []
    for name, fn in shapes:
        rates = [fn(t) for t in range(SHAPE_T)]
        rows.append([name] + [f"{fn(t):.5f}" for t in marks] + [f"{sum(rates) / SHAPE_T:.5f}"])
    head = (f"Peak eta_max = {SHAPE_PEAK:g}, floor eta_min = {SHAPE_FLOOR:g}, warmup W = {SHAPE_W} steps for every row but the first (eta(t) = {SHAPE_PEAK:g} (t + 1)/{SHAPE_W} for t < {SHAPE_W}), "
            f"total T = {SHAPE_T} steps, t counted from 0. After the warmup: step decay {SHAPE_PEAK:g} * 0.3^floor((t - 30)/150); exponential {SHAPE_PEAK:g} * 0.99^(t - 30); "
            f"inverse square root {SHAPE_PEAK:g} sqrt(30/(t + 1)); cosine eta_min + (eta_max - eta_min)(1 + cos(pi u))/2 with u = (t - 30)/(T - 30) "
            "(and, in the last row, u = (t - 330)/(T - 330) after a hold of 300 steps at the peak); every decaying row is floored at eta_min. "
            "The first row, 'constant', is the constant rate 0.3 with no warmup (0.3 at t = 0, mean 0.3); the other rows use the warmup. The last column is the mean rate over the 600 steps.\n\n")
    return head + _table(["schedule"] + [f"t = {t}" for t in marks] + ["mean rate"], rows)


def cosine_values():
    u_values = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    rows = []
    for u in u_values:
        c = 0.5 * (1 + math.cos(math.pi * u))
        slope = -0.5 * math.pi * math.sin(math.pi * u)
        rows.append([f"{u:g}", _fx(c, 4), _fx(1 - u, 4), _fx(slope, 4)])
    head = "The decay factor c(u) = (1 + cos(pi u))/2 of the cosine schedule, a straight line 1 - u for comparison, and the slope dc/du = -(pi/2) sin(pi u).\n\n"
    return head + _table(["u", "c(u)", "linear 1 - u", "slope dc/du"], rows)


def floor_waste():
    """How many steps of a T-step cooldown run at a rate below a hundredth of the peak, for a few floors."""
    T = 600
    rows = []
    for floor_ratio in (0.0, 0.01, 0.1):
        fn = cosine(1.0, floor_ratio, 30, T)
        low = sum(1 for t in range(30, T) if fn(t) < 0.01)
        tiny = sum(1 for t in range(30, T) if fn(t) < 0.05)
        rows.append([f"{floor_ratio:g}", f"{fn(T - 1):.6f}", low, tiny])
    head = ("Cosine schedule with peak 1, warmup 30, T = 600, as a function of the floor (a fraction of the peak). Columns: the rate of the last step; "
            "how many of the 570 decay steps use a rate below 0.01 (below a hundredth of the peak); how many use a rate below 0.05.\n\n")
    return head + _table(["floor", "rate at step 599", "steps below 0.01", "steps below 0.05"], rows)


# ------------------------------------------------------------------ 5. a budget


def budget_independence():
    p, f, W, C = 1.0, 0.1, 40, 100
    rows = []
    for T in (400, 800):
        plain = cosine(p, f, W, T)
        wsd = cosine(p, f, W, T, H=T - W - C)
        rows.append([f"{T}", "cosine"] + [f"{plain(t):.4f}" for t in (100, 200, 300, 399)])
        rows.append([f"{T}", "hold, then cooldown of 100"] + [f"{wsd(t):.4f}" for t in (100, 200, 300, 399)])
    head = ("Peak 1, floor 0.1, warmup 40 steps. 'cosine' decays from step 40 to the floor at step T; 'hold, then cooldown of 100' holds the peak until step T - 100 and then "
            "runs the same cosine over its last 100 steps. Rows: the rate at steps 100, 200, 300 and 399 of a run planned for T = 400 steps and of one planned for T = 800.\n\n")
    return head + _table(["planned T", "shape", "t = 100", "t = 200", "t = 300", "t = 399"], rows)


def warmup_longer_than_run():
    peak, W, T = 2e-4, 100, 40
    fn = warmup_constant(peak, W)
    rates = [fn(t) for t in range(T)]
    fixed = max(1, T // 20)
    fn2 = cosine(peak, peak * 0.1, fixed, T)
    rates2 = [fn2(t) for t in range(T)]
    rows = [
        ["warmup 100, run of 40 steps", f"{rates[7]:.3e}", f"{rates[T - 1]:.3e}", f"{max(rates) / peak:.2f}", f"{sum(rates) / T / peak:.3f}"],
        [f"warmup {fixed} (a twentieth of the run), cosine to a tenth", f"{rates2[7]:.3e}", f"{rates2[T - 1]:.3e}", f"{max(rates2) / peak:.2f}", f"{sum(rates2) / T / peak:.3f}"],
    ]
    head = ("A fine-tuning run of 40 steps with peak rate 2e-4. First row: a linear warmup of 100 steps (rate 2e-4 (t + 1)/100 for t < 100). Second row: warmup of "
            "max(1, 40 // 20) = 2 steps, then a cosine over the remaining steps to a tenth of the peak. Columns: the rate at step index 7 (the eighth update), the rate of the last "
            "update, the largest rate reached as a fraction of the peak, and the mean rate over the 40 updates as a fraction of the peak.\n\n")
    return head + _table(["schedule", "rate, 8th update", "rate, last update", "largest / peak", "mean / peak"], rows)


# ------------------------------------------------------------------ 6. comparing schedules over five batch orders


def _comparison_cases():
    p, f = PEAK_COMPARE, PEAK_COMPARE / 100
    return [
        ("constant 0.1 (the best constant)", constant(PEAK_TUNED)),
        ("constant 1", constant(p)),
        ("warmup 60, then constant 1", warmup_constant(p, 60)),
        ("cosine from step 0 (no warmup)", _cosine_no_warmup(p, f, RUN_STEPS)),
        ("warmup 60, linear to 0.01", linear_decay(p, f, 60, RUN_STEPS)),
        ("warmup 60, cosine to 0.01", cosine(p, f, 60, RUN_STEPS)),
    ]


def _cosine_no_warmup(peak, floor, T):
    return lambda t: floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t / T)))


_COMPARE = {}


def comparison_results():
    if not _COMPARE:
        for name, sched in _comparison_cases():
            _COMPARE[name] = [train(sched, stream=s) for s in ORDERS]
    return _COMPARE


def schedule_comparison():
    res = comparison_results()
    rows = []
    for name, runs in res.items():
        rows.append([name, cell(lambda q: worst(q, 1, 50), runs, 3), cell(lambda q: worst(q, 301, RUN_STEPS), runs, 3), cell(tail, runs), span_cell(tail, runs)])
    seeds = ", ".join(_seed(s) for s in ORDERS)
    head = (_setup_text() + f" The same run on five batch orders (generators `schedules.batches`, `schedules.batches.1` .. `schedules.batches.4`; seeds {seeds}). "
            "Peak 1 (ten times the rate that is best for a constant), floor 0.01, T = 600. Entries are means over the five orders: the largest loss over steps 1 to 50; "
            "the largest loss over steps 301 to 600 (recorded every 10 steps); the tail (mean loss at steps 550, 560, ..., 600); and the range of the tail over the orders. " + _unstable_note() + "\n\n")
    return head + _table(["schedule", "worst, steps 1-50", "worst, steps 301-600", "tail (mean of 5)", "tail range"], rows)


PEAKS = (0.03, 0.1, 0.3, 1.0, 3.0)


def peak_robustness():
    rows = []
    for peak in PEAKS:
        const = [train(constant(peak), stream=s) for s in ORDERS]
        sched = [train(cosine(peak, peak / 100, 60, RUN_STEPS), stream=s) for s in ORDERS]
        rows.append([f"{peak:g}", cell(tail, const), cell(lambda q: worst(q, 1, 50), const, 3),
                     cell(tail, sched), cell(lambda q: worst(q, 1, 50), sched, 3)])
    head = (_setup_text() + " The five batch orders of the schedule comparison. For each peak rate: a constant rate, and a linear warmup over 60 steps followed by a cosine to one hundredth of the peak "
            "at step 600. Entries are means over the five orders of the tail (mean loss at steps 550, 560, ..., 600) and of the largest loss over steps 1 to 50. " + _unstable_note() + "\n\n")
    return head + _table(["peak", "constant: tail", "constant: worst 1-50", "warmup + cosine: tail", "warmup + cosine: worst 1-50"], rows)


def mean_curve(name, steps):
    """Mean loss over the five orders at each of `steps` for a comparison case (used by the figures)."""
    runs = comparison_results()[name]
    return [(s, sum(r.losses[s] for r in runs) / len(runs)) for s in steps]


BUDGETS = (300, 450, 600)
COOLDOWN = 100


def budget_runs():
    rows = []
    peak, floor, W = CLEAN_PEAK, CLEAN_PEAK / 100, 30
    for B in BUDGETS:
        for label, sched in (("cosine over the whole run", cosine(peak, floor, W, B)),
                             (f"hold, then cooldown of {COOLDOWN}", cosine(peak, floor, W, B, H=B - W - COOLDOWN))):
            runs = [train(sched, stream=s, steps=B) for s in ORDERS]
            rows.append([B, label, cell(lambda q, B=B: tail(q, B), runs), span_cell(lambda q, B=B: tail(q, B), runs)])
    head = (_setup_text() + f" Peak {CLEAN_PEAK:g}, floor {CLEAN_PEAK / 100:g}, warmup 30 steps; five batch orders as in the schedule comparison. Each run is trained for its own budget B: "
            f"'cosine' decays from step 30 to the floor at step B; 'hold, then cooldown' holds the peak until step B - {COOLDOWN} and then runs the cosine over the last {COOLDOWN} steps. "
            "Entries: the tail (mean loss at the six steps B - 50, B - 40, ..., B), mean and range over the five orders. " + _unstable_note() + "\n\n")
    return head + _table(["budget B", "shape", "tail (mean of 5)", "tail range"], rows, align="rlrr")


# ------------------------------------------------------------------ 7. clipping: geometry


def clip_elementwise(g, c):
    return [max(-c, min(c, x)) for x in g]


def clip_global(g, tau):
    n = l2(g)
    s = min(1.0, tau / (n + 1e-6))
    return [s * x for x in g]


def _angle_deg(v):
    return math.degrees(math.atan2(v[1], v[0]))


def _cosine_sim(a, b):
    return sum(x * y for x, y in zip(a, b)) / (l2(a) * l2(b))


CLIP_VECTORS = [(4.0, 1.0), (3.0, 3.0), (0.6, 0.3), (-5.0, 2.0)]


def clip_geometry():
    rows = []
    for g in CLIP_VECTORS:
        e, k = clip_elementwise(g, 1.0), clip_global(g, 1.0)
        rows.append([f"({g[0]:g}, {g[1]:g})", _fx(l2(g), 4), f"({e[0]:.4f}, {e[1]:.4f})", _fx(l2(e), 4), _fx(_cosine_sim(g, e), 4), f"({k[0]:.4f}, {k[1]:.4f})", _fx(l2(k), 4), _fx(_cosine_sim(g, k), 4)])
    head = ("Two clips with threshold 1 on four gradients in two dimensions. Per-element: each coordinate is replaced by max(-1, min(1, g_i)). Global: g is multiplied by "
            "min(1, 1/(||g|| + 10^-6)). 'cos' is the cosine of the angle between the original and the clipped vector.\n\n")
    return head + _table(["g", "||g||", "per-element", "its norm", "cos", "global", "its norm", "cos"], rows)


TENSORS = {"layer 1 (weights)": [3.0, 4.0], "layer 2 (weights)": [0.6, 0.8], "bias": [0.0, 0.0, 0.2]}


def clip_tensors():
    names = list(TENSORS)
    flat = [x for n in names for x in TENSORS[n]]
    tau = 1.0
    total = l2(flat)
    k = min(1.0, tau / (total + 1e-6))
    per_tensor = {n: [min(1.0, tau / (l2(TENSORS[n]) + 1e-6)) * x for x in TENSORS[n]] for n in names}
    flat_per = [x for n in names for x in per_tensor[n]]
    flat_global = [k * x for x in flat]
    rows = []
    for n in names:
        gn = l2(TENSORS[n])
        rows.append([n, str(TENSORS[n]), _fx(gn, 4), _fx(min(1.0, tau / (gn + 1e-6)), 4), "[" + ", ".join(f"{x:.4f}" for x in per_tensor[n]) + "]",
                     "[" + ", ".join(f"{k * x:.4f}" for x in TENSORS[n]) + "]"])
    head = (f"Three tensors of one model's gradient, threshold tau = {tau:g}. 'per tensor' clips each tensor to norm {tau:g} on its own; 'global' computes the norm over all seven numbers at once "
            f"(sum of squares {sum(x * x for x in flat):.2f}, norm {total:.4f}, one factor {k:.4f}).\n\n")
    tail_text = (f"\n\nCosine between the concatenated original and the concatenated clipped gradient: per tensor {_cosine_sim(flat, flat_per):.4f}, global {_cosine_sim(flat, flat_global):.4f}. "
                 f"Norm after the clip: per tensor {l2(flat_per):.4f}, global {l2(flat_global):.4f}.")
    return head + _table(["tensor", "gradient", "norm", "own factor", "per tensor", "global"], rows, align="lrrrrr") + tail_text


# ------------------------------------------------------------------ 8. clipping and Adam on one weight


def adam_one_weight(grads, clip=None, lr=1.0):
    m = v = 0.0
    rows = []
    for t, g in enumerate(grads, start=1):
        gc = g
        if clip is not None and abs(g) > clip:
            gc = clip * (1 if g > 0 else -1)
        m = BETA1 * m + (1 - BETA1) * gc
        v = BETA2 * v + (1 - BETA2) * gc * gc
        mh, vh = m / (1 - BETA1 ** t), v / (1 - BETA2 ** t)
        rows.append((t, g, gc, mh, math.sqrt(vh), mh / (math.sqrt(vh) + EPS)))
    return rows


def clip_adam_one_weight():
    grads = [1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    plain = adam_one_weight(grads)
    clipped = adam_one_weight(grads, clip=2.0)
    rows = []
    for a, b in zip(plain, clipped):
        rows.append([a[0], f"{a[1]:g}", _fx(a[3], 4), _fx(a[4], 4), _fx(a[5], 4), f"{b[2]:g}", _fx(b[3], 4), _fx(b[4], 4), _fx(b[5], 4)])
    head = ("One weight, gradients 1, 1, 1, 1, 100, 1, 1, 1, 1, 1 (the fifth is the outlier). Adam with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, bias correction; "
            "'step' is mhat / (sqrt(vhat) + epsilon), the update in units of the learning rate. Right-hand block: the same with the gradient clipped to [-2, 2] "
            "(for one weight, the global-norm clip with threshold 2 is the same operation).\n\n")
    return head + _table(["t", "g", "mhat", "sqrt(vhat)", "step", "clipped g", "mhat", "sqrt(vhat)", "step"], rows)


# ------------------------------------------------------------------ 9. choosing the threshold from norm statistics


def percentile(a, q):
    v = sorted(a)
    pos = q * (len(v) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])


THRESHOLDS = (0.1, 0.2, 0.4, 1.0, 5.0)
OUT_STEP, OUT_FACTOR = 300, 100.0
CLEAN_SCHEDULE = cosine(CLEAN_PEAK, CLEAN_PEAK / 100, 30, RUN_STEPS)
_CLEAN = {}


def clean_run(stream="schedules.batches"):
    if stream not in _CLEAN:
        _CLEAN[stream] = train(CLEAN_SCHEDULE, stream=stream)
    return _CLEAN[stream]


def gradient_norm_statistics():
    r = clean_run()
    n = r.norms
    qs = [("minimum", 0.0), ("median", 0.5), ("90th percentile", 0.9), ("99th percentile", 0.99), ("maximum", 1.0)]
    rows = [[name, _fx(percentile(n, q), 4)] for name, q in qs]
    first = [f"{x:.3f}" for x in n[:5]]
    ranges = []
    for lo, hi in ((0, 100), (100, 300), (300, 600)):
        seg = n[lo:hi]
        ranges.append([f"steps {lo} to {hi - 1}", _fx(percentile(seg, 0.5), 4), _fx(max(seg), 4)])
    head = (_setup_text() + f" Learning rate: warmup 30 steps to {CLEAN_PEAK:g}, then a cosine to {CLEAN_PEAK / 100:g} at step {RUN_STEPS}. No clip. The norm is the Euclidean norm of "
            f"the 33-component minibatch gradient of each step ({RUN_STEPS} values; the first five are {', '.join(first)}).\n\n")
    return head + _table(["statistic", "gradient norm"], rows) + "\n\n" + _table(["part of the run", "median", "maximum"], ranges)



def clip_threshold_table():
    base = [clean_run(s) for s in ORDERS]
    out_runs = [train(CLEAN_SCHEDULE, stream=s, outlier=(OUT_STEP, OUT_FACTOR), dense=700) for s in ORDERS]
    rows = [["no clip", "0.0%", cell(tail, base), cell(lambda q: worst(q, 301, 400), out_runs, 3), cell(tail, out_runs)]]
    for tau in THRESHOLDS:
        runs = [train(CLEAN_SCHEDULE, stream=s, clip=tau) for s in ORDERS]
        frac = sum(sum(r.clipped) for r in runs) / (5 * RUN_STEPS)
        outs = [train(CLEAN_SCHEDULE, stream=s, clip=tau, outlier=(OUT_STEP, OUT_FACTOR), dense=700) for s in ORDERS]
        rows.append([f"{tau:g}", f"{100 * frac:.1f}%", cell(tail, runs), cell(lambda q: worst(q, 301, 400), outs, 3), cell(tail, outs)])
    head = ("The run of the previous tables (warmup 30 steps to 0.3, cosine to 0.003, 600 steps) on the five batch orders of the schedule comparison, with the global-norm clip "
            f"at each threshold tau, once as it is and once with the outlier of the next table (the gradient of step index {OUT_STEP} multiplied by {OUT_FACTOR:g}). "
            "Columns: the share of the 3000 steps of the clean runs in which the clip changed the gradient; the tail (mean loss at steps 550, 560, ..., 600) of the clean runs; "
            "the largest loss over steps 301 to 400 of the runs with the outlier; their tail. All loss entries are means of the five orders. " + _unstable_note() + "\n\n")
    return head + _table(["tau", "steps clipped (clean)", "tail (clean)", "worst, 301-400 (outlier)", "tail (outlier)"], rows)


# ------------------------------------------------------------------ 10. an outlier batch


def outlier_batch():
    cases = [("no clip", None), ("clip tau = 1", 1.0), ("clip tau = 0.2", 0.2)]
    marks = (299, 300, 301, 305, 320, 350, 400, 600)
    rows, extra = [], []
    for name, tau in cases:
        r = train(CLEAN_SCHEDULE, clip=tau, outlier=(OUT_STEP, OUT_FACTOR), dense=700)
        rows.append([name] + [cell(lambda q, s=s: q.losses[s], [r]) for s in marks] + [cell(lambda q: worst(q, 301, 400), [r], 3)])

        def mean_step(lo, hi):
            return cell(lambda q: sum(q.update_rms[lo:hi]) / len(q.update_rms[lo:hi]), [r])
        extra.append([name, _fx(r.norms[OUT_STEP], 3), mean_step(OUT_STEP - 100, OUT_STEP), _fx(r.update_rms[OUT_STEP], 4),
                      mean_step(OUT_STEP + 1, OUT_STEP + 21), mean_step(OUT_STEP + 51, OUT_STEP + 151), mean_step(OUT_STEP + 201, RUN_STEPS)])
    clean = clean_run()
    head = (_setup_text() + f" Learning rate: warmup 30 steps to {CLEAN_PEAK:g}, cosine to {CLEAN_PEAK / 100:g} at step {RUN_STEPS}. At step index {OUT_STEP} (the {OUT_STEP + 1}st update) the minibatch "
            f"gradient is multiplied by {OUT_FACTOR:g}, as if one batch had carried a loss weight of {OUT_FACTOR:g}; nothing else differs between the rows. "
            f"The same run without an outlier has a gradient norm of {_fx(clean.norms[OUT_STEP], 3)} at that step and a loss of {_fx(clean.losses[600], 4)} at step 600. "
            "Entries: the mean loss over all 200 examples after the stated number of updates; the last column is the largest loss over steps 301 to 400. " + _unstable_note() + "\n\n")
    t1 = _table(["clip", "step 299", "step 300", "step 301", "step 305", "step 320", "step 350", "step 400", "step 600", "worst, 301-400"], rows)
    t2 = _table(["clip", "norm at the outlier", "steps 200-299", "at the outlier", "steps 301-320", "steps 351-450", "steps 501-599"], extra)
    return (head + t1 + "\n\nThe size of Adam's step in units of the learning rate: the root mean square over the 33 parameters of mhat / (sqrt(vhat) + epsilon), "
            "averaged over the stated steps (step indices 0 .. 599; the last window, 501 to 599, has 99 steps because the run has no step 600).\n\n" + t2)


def outlier_curves(clip):
    r = train(CLEAN_SCHEDULE, clip=clip, outlier=(OUT_STEP, OUT_FACTOR), dense=700)
    return [(s, r.losses[s]) for s in sorted(r.losses)]


# ------------------------------------------------------------------ 11. what the clip costs


def clip_work():
    P = 1_000_000_000
    rows = [
        ("sum of squares of all gradients", 2 * P, 4 * P, "read g; 1 multiply and 1 add per element"),
        ("scale by the coefficient, own pass", P, 8 * P, "read g, write g"),
        ("scale folded into the optimizer's read of g", P, 0, "one more multiply in a pass that already reads g"),
        ("AdamW update itself (Chapter 7)", 16 * P, 28 * P, "reads w, g, m, v; writes w, m, v"),
    ]
    body = []
    for name, fl, by, note in rows:
        body.append([name, f"{fl:,}", f"{by:,}", note])
    own = 3 * P / (16 * P)
    own_bytes = (4 * P + 8 * P) / (28 * P)
    fold_bytes = 4 * P / (28 * P)
    head = ("Work for a model of P = 1,000,000,000 parameters in 32-bit floats, per optimizer step. The sum of squares is a reduction (the final sum of a few hundred partials is negligible).\n\n")
    tail_text = (f"\n\nRelative to the AdamW update (16P FLOPs, 28P bytes): the norm and the scaling in its own pass add (2 + 1)P FLOPs and (4 + 8)P bytes, which is {own:.4f} of the update's FLOPs and "
                 f"{own_bytes:.4f} of its bytes. With the scale folded into the optimizer's read of g, the norm pass is the only extra traffic, {fold_bytes:.4f} of the update's bytes, "
                 f"and the extra FLOPs are the same {own:.4f} of the update's.")
    return head + _table(["operation", "FLOPs", "bytes", "what it does"], body, align="lrrl") + tail_text


# ------------------------------------------------------------------ 12. summing squares in 32-bit floats


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


NORM_N, NORM_CHUNK = 262_144, 1024


def _below_floor(rel, floor=1e-12):
    """A relative error at the level of 64-bit rounding noise is printed as a bound: its digits depend on the summation order."""
    return f"below {floor:g}" if rel < floor else f"{rel:.2e}"


def norm_accumulation():
    name = "schedules.norm_sum"
    rng = bookrng.generator(name)
    g = [_f32(0.01 * rng.normal()) for _ in range(NORM_N)]
    exact = math.fsum(x * x for x in g)
    serial = 0.0
    for x in g:
        serial = _f32(serial + _f32(x * x))
    partial_sum_f64 = 0.0
    for s in range(0, NORM_N, NORM_CHUNK):
        acc = 0.0
        for x in g[s:s + NORM_CHUNK]:
            acc = _f32(acc + _f32(x * x))
        partial_sum_f64 += acc
    f64_serial = 0.0
    for x in g:
        f64_serial += x * x
    rows = [
        ["exact (compensated sum of the 32-bit squares' double-precision values)", f"{exact:.9e}", "0"],
        ["one 32-bit accumulator, in order", f"{serial:.9e}", f"{abs(serial - exact) / exact:.2e}"],
        [f"chunks of {NORM_CHUNK} in 32 bits, chunk sums added in 64 bits", f"{partial_sum_f64:.9e}", f"{abs(partial_sum_f64 - exact) / exact:.2e}"],
        ["one 64-bit accumulator, in order", f"{f64_serial:.9e}", _below_floor(abs(f64_serial - exact) / exact)],
    ]
    head = (f"The sum of squares of N = {NORM_N:,} gradient entries: entry i is the 32-bit rounding of 0.01 * z_i, with z_i from normal() of the generator `{name}` (seed {_seed(name)}), "
            "i = 0 .. N - 1 in order. In the two 32-bit rows every operation (the square, each addition inside a chunk or inside the single accumulator) is rounded to the nearest 32-bit float. "
            "In the 'exact' and 'one 64-bit accumulator' rows the square of a 32-bit entry is formed in double precision (it is exact there) and summed in double precision in order of index, with Python's compensated fsum for the exact row; "
            "the chunk sums of the third row are added in double precision in order of chunk. The last row's relative error is rounding noise of that double-precision sum and is printed as a bound.\n\n")
    return head + _table(["method", "sum of squares", "relative error"], rows, align="lrr")


# ------------------------------------------------------------------ curves for the figures


def warmup_curves(lengths=(0, 10, 30, 60), upto=60):
    """{W: [(step, loss)]} for the peak-3 runs of the warmup table, steps 0 .. upto (every step)."""
    out = {}
    for W in lengths:
        r = train(warmup_constant(PEAK_HIGH, W) if W else constant(PEAK_HIGH))
        out[W] = [(s, r.losses[s]) for s in range(0, upto + 1)]
    return out
