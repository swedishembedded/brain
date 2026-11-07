# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Regularization in Neural Networks".

Plain Python 3 on floats (IEEE double precision), standard library only, written from the
chapter's own description and independent of any engine code. Each public function returns a
markdown string that the chapter embeds between `<!-- worked: regularization_nn.<name> -->`
markers. The same module supplies the data of the chapter's figures.

One task runs through the chapter. A curve y = sin(2 pi x) is observed at 16 random points
x in [0, 1) with Gaussian noise of standard deviation 0.3; the model is a network with one
hidden layer of 64 tanh units (193 parameters), trained by full-batch AdamW. Every method of
the chapter is a switch of the one training routine `_run`, and every number of the
comparisons comes from running it on 8 seeds. A seed fixes the 16 training points, the
initial weights and the stream of masks and augmentations, so two methods with the same seed
see the same data and start from the same weights (a paired comparison). The validation set
(200 points) and the test set (1000 points) are the same for every seed.

The runs are independent, so they are spread over the machine's cores; each one is a pure
function of its specification and the results do not depend on the number of workers.

Swedish Embedded AB implements solutions for making neural networks generalize beyond their
training data. If your team needs expertise in regularization and honest model selection, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import bookrng

# ---------------------------------------------------------------- the task and the recipe

HIDDEN = 64                       # tanh units in the one hidden layer
PARAMS = 3 * HIDDEN + 1           # w (64), b (64), v (64), c (1)
N_TRAIN = 16
N_VAL = 200
N_TEST = 1000
NOISE_SD = 0.3                    # standard deviation of the label noise; the noise floor is its square
STEPS = 3000
LR = 0.01
BETA1, BETA2, EPS_DEN = 0.9, 0.999, 1e-8
EVAL_EVERY = 10                   # steps between two evaluations on the validation set
SEEDS = tuple(range(8))
SQRT12 = math.sqrt(12.0)          # 1 / sd of Uniform[0, 1): inputs are u = (x - 0.5) * sqrt(12)
HELD_VAL_NAME = "regularization_nn.validation"
HELD_TEST_NAME = "regularization_nn.test"

WD_GRID = (0.1, 0.3, 1.0, 3.0)                  # decoupled weight decay lambda
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)              # lambda of a penalty added to the gradient (coupled)
DROP_GRID = (0.1, 0.3, 0.5)
NOISE_GRID = (0.03, 0.1, 0.2)                   # standard deviation of the additive noise on x
GAIN_SPREAD = 0.1                               # multiplicative gain 1 + 0.1 (2u - 1)
N_SWEEP = (8, 16, 32, 64, 128)


def _seed_line(name):
    s = bookrng.seed_of(name)
    return f"`{name}` seed 0x{s:016x}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _mean(v):
    return sum(v) / len(v)


def _sd(v):
    m = _mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def _truth(x):
    return math.sin(2.0 * math.pi * x)


def _sample(n, g):
    """n pairs: for each i, x = uniform() then e = normal(); y = sin(2 pi x) + 0.3 e."""
    xs, ys = [], []
    for _ in range(n):
        x = g.uniform()
        e = g.normal()
        xs.append(x)
        ys.append(_truth(x) + NOISE_SD * e)
    return xs, ys


@lru_cache(maxsize=None)
def _heldout():
    vx, vy = _sample(N_VAL, bookrng.generator(HELD_VAL_NAME))
    tx, ty = _sample(N_TEST, bookrng.generator(HELD_TEST_NAME))
    return vx, vy, tx, ty


def _training_set(seed, n):
    return _sample(n, bookrng.generator(f"regularization_nn.train.{seed}"))


def _initial(seed):
    """w_j (64 normals), then b_j (64 normals), then v_j = normal() / sqrt(64); c = 0."""
    g = bookrng.generator(f"regularization_nn.init.{seed}")
    w = [g.normal() for _ in range(HIDDEN)]
    b = [g.normal() for _ in range(HIDDEN)]
    v = [g.normal() / math.sqrt(HIDDEN) for _ in range(HIDDEN)]
    return w + b + v + [0.0]


def _predict(theta, xs):
    """Eval-mode output of the network at each x (no masks, no augmentation)."""
    w, b, v, c = theta[:HIDDEN], theta[HIDDEN:2 * HIDDEN], theta[2 * HIDDEN:3 * HIDDEN], theta[3 * HIDDEN]
    tanh = math.tanh
    out = []
    for x in xs:
        u = (x - 0.5) * SQRT12
        s = c
        for j in range(HIDDEN):
            s += v[j] * tanh(w[j] * u + b[j])
        out.append(s)
    return out


def _mse(theta, xs, ys):
    p = _predict(theta, xs)
    return sum((a - b) ** 2 for a, b in zip(p, ys)) / len(ys)


def _steepest_slope(theta, points=501):
    """Largest |f(x1) - f(x0)| / (x1 - x0) between neighbouring points of a grid of `points` on [0, 1]."""
    xs = [i / (points - 1) for i in range(points)]
    f = _predict(theta, xs)
    return max(abs(f[i + 1] - f[i]) for i in range(points - 1)) * (points - 1)


def _norm(theta):
    return math.sqrt(sum(a * a for a in theta))


# --------------------------------------------------------------- one training run

def _augment(kind, amount, x, noise):
    """The input actually shown to the network for the training point x (label unchanged)."""
    if kind == "noise":
        return x + amount * noise.normal()
    if kind == "gain":
        return x * (1.0 + amount * (2.0 * noise.uniform() - 1.0))
    if kind == "shift":
        return x + 0.5 if noise.uniform() < 0.5 else x
    return x


def _run_impl(spec):
    """Train one network. spec = (seed, n_train, wd, coupled, p_drop, aug_kind, aug_amount, keep).

    Full-batch AdamW, 3000 steps. At every 10th step the clean training loss, the validation loss
    and the norm of the parameters are recorded. Draw order of the stream
    `regularization_nn.noise.<seed>`: for each step, for each training example in order, first the
    draw of the augmentation (if any), then 64 uniform() draws, one per hidden unit, for the
    dropout mask (if p_drop > 0); a unit is dropped when its draw is below p_drop."""
    seed, n_train, wd, coupled, p_drop, aug_kind, aug_amount, keep = spec[:8]
    nudge = spec[8] if len(spec) > 8 else 0.0      # test of chaos: the first hidden weight changed by this amount
    vx, vy, tx, ty = _heldout()
    xs, ys = _training_set(seed, n_train)
    theta = _initial(seed)
    theta[0] += nudge
    noise = bookrng.generator(f"regularization_nn.noise.{seed}")
    m1 = [0.0] * PARAMS
    m2 = [0.0] * PARAMS
    n = n_train
    scale_keep = 1.0 / (1.0 - p_drop)
    tanh = math.tanh
    history = []
    snapshots = []
    best_val, best_step, best_theta = float("inf"), 0, None
    rng_h = range(HIDDEN)
    for t in range(1, STEPS + 1):
        w, b, v, c = theta[:HIDDEN], theta[HIDDEN:2 * HIDDEN], theta[2 * HIDDEN:3 * HIDDEN], theta[3 * HIDDEN]
        gw = [0.0] * HIDDEN
        gb = [0.0] * HIDDEN
        gv = [0.0] * HIDDEN
        gc = 0.0
        for x0, y in zip(xs, ys):
            x = _augment(aug_kind, aug_amount, x0, noise)
            u = (x - 0.5) * SQRT12
            h = [tanh(w[j] * u + b[j]) for j in rng_h]
            if p_drop > 0.0:
                mk = [scale_keep if noise.uniform() >= p_drop else 0.0 for _ in rng_h]
            else:
                mk = None
            if mk is None:
                o = c + sum(v[j] * h[j] for j in rng_h)
            else:
                o = c + sum(v[j] * h[j] * mk[j] for j in rng_h)
            d = 2.0 * (o - y) / n                      # dL/do for L = mean (o - y)^2
            gc += d
            for j in rng_h:
                f = 1.0 if mk is None else mk[j]
                gv[j] += d * h[j] * f
                if f != 0.0:
                    dz = d * v[j] * f * (1.0 - h[j] * h[j])
                    gw[j] += dz * u
                    gb[j] += dz
        g = gw + gb + gv + [gc]
        if coupled and wd > 0.0:
            g = [gi + wd * th for gi, th in zip(g, theta)]
        bc1 = 1.0 - BETA1 ** t
        bc2 = 1.0 - BETA2 ** t
        for i in range(PARAMS):
            m1[i] = BETA1 * m1[i] + (1.0 - BETA1) * g[i]
            m2[i] = BETA2 * m2[i] + (1.0 - BETA2) * g[i] * g[i]
            if not coupled:
                theta[i] -= LR * wd * theta[i]
            theta[i] -= LR * (m1[i] / bc1) / (math.sqrt(m2[i] / bc2) + EPS_DEN)
        if t % EVAL_EVERY == 0:
            val = _mse(theta, vx, vy)
            history.append((t, _mse(theta, xs, ys), val, _norm(theta)))
            if val < best_val:                      # strict improvement
                best_val, best_step, best_theta = val, t, list(theta)
            if keep:
                snapshots.append(list(theta))
    return {
        "history": history,
        "theta": theta,
        "final_val": history[-1][2],
        "final_test": _mse(theta, tx, ty),
        "final_train": history[-1][1],
        "final_norm": history[-1][3],
        "best_val": best_val,
        "best_step": best_step,
        "best_test": _mse(best_theta, tx, ty),
        "best_norm": _norm(best_theta),
        "snapshots": snapshots,
    }


_RESULTS = {}


def _run_many(specs):
    """Results for every spec, in order; runs not yet computed are spread over the cores."""
    todo = [s for s in dict.fromkeys(specs) if s not in _RESULTS]
    if todo:
        # the largest training sets first, so that the pool is not left waiting for them
        todo.sort(key=lambda s: -s[1])
        # At most four workers: the book is built on shared workstations, and
        # results do not depend on the worker count.
        workers = max(1, min(len(todo), os.cpu_count() or 1, 4))
        if workers == 1:
            outs = [_run_impl(s) for s in todo]
        else:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as ex:
                outs = list(ex.map(_run_impl, todo, chunksize=1))
        _RESULTS.update(zip(todo, outs))
    return [_RESULTS[s] for s in specs]


def _spec(seed, n=N_TRAIN, wd=0.0, coupled=False, drop=0.0, aug=("none", 0.0), keep=False):
    return (seed, n, wd, coupled, drop, aug[0], aug[1], keep)


def _runs(n=N_TRAIN, wd=0.0, coupled=False, drop=0.0, aug=("none", 0.0), keep=False):
    return _run_many([_spec(s, n, wd, coupled, drop, aug, keep) for s in SEEDS])


def _fmt(v, d=3):
    return f"{v + 0.0:.{d}f}"


NUDGE = 1e-12   # the first hidden weight of every network changed by this amount: a printed number that moves under it is chaotic


def _twins(n=N_TRAIN, wd=0.0, coupled=False, drop=0.0, aug=("none", 0.0), keep=False):
    """The same runs as `_runs` with the first hidden weight nudged, to test which printed digits are reproducible."""
    return _run_many([_spec(s, n, wd, coupled, drop, aug, keep) + (NUDGE,) for s in SEEDS])


def _fmtm(v, w, d=3):
    """v to d decimals if the nudged value w agrees with it to within half a unit of the last digit; otherwise 'about'
    v to the fewest decimals on which the two agree ('unstable' when they differ even in the integer part)."""
    if abs(v - w) <= 0.5 * 10 ** (-d):
        return f"{v + 0.0:.{d}f}"
    for k in range(d - 1, -1, -1):
        if f"{v + 0.0:.{k}f}" == f"{w + 0.0:.{k}f}":
            return f"about {v + 0.0:.{k}f}"
    return "unstable"


def _m(f, a, b, d=3):
    """Mean over the seeds of f(run), printed with `_fmtm` against the same mean over the nudged runs."""
    return _fmtm(_mean([f(x) for x in a]), _mean([f(x) for x in b]), d)


def _signed(v, w, d=3):
    """_fmtm with an explicit sign."""
    t = _fmtm(v, w, d)
    return t if t == "unstable" else (t.replace("about ", "about +") if t.startswith("about") and not t.startswith("about -") else (("+" + t) if not t.startswith(("-", "about")) else t))


def _paired_text(a, b, at, bt, d=3):
    """('+0.012 (SE 0.004)', '+3.1', '7 of 8') of the paired comparison a - b, each printed against the nudged runs at, bt."""
    m, se, t, wins = _paired(a, b)
    m2, se2, t2, wins2 = _paired(at, bt)
    return (f"{_signed(m, m2, d)} (SE {_fmtm(se, se2, d)})", _signed(t, t2, 1), f"{wins} of 8" if wins == wins2 else "unstable")


def _paired(a, b):
    """Mean difference a - b over the seeds, its standard error, t, and the number of seeds with a < b."""
    d = [x - y for x, y in zip(a, b)]
    m = _mean(d)
    se = _sd(d) / math.sqrt(len(d))
    return m, se, (m / se if se > 0 else float("inf")), sum(1 for x in d if x < 0)


# ----------------------------------------------------------------------- the chapter's blocks

def task_setup():
    """The task, the network, the recipe and the generator streams."""
    vx, vy, tx, ty = _heldout()
    xs, ys = _training_set(0, N_TRAIN)
    rows = [(i, _fmt(x, 4), _fmt(_truth(x), 4), _fmt(y, 4)) for i, (x, y) in enumerate(zip(xs, ys))]
    out = [
        f"Task: x ~ uniform on [0, 1), y = sin(2 pi x) + {NOISE_SD} * normal(); the noise floor of the mean squared error is {NOISE_SD ** 2:.2f}. "
        f"Network: u = (x - 0.5) * sqrt(12); {HIDDEN} hidden units tanh(w_j u + b_j); output c + sum_j v_j h_j. "
        f"Parameters: {HIDDEN} + {HIDDEN} + {HIDDEN} + 1 = {PARAMS}, against {N_TRAIN} training examples ({PARAMS / N_TRAIN:.1f} per example).",
        "",
        f"Loss: mean squared error over the {N_TRAIN} training points. Optimizer: AdamW, full batch, learning rate {LR}, "
        f"beta1 {BETA1}, beta2 {BETA2}, epsilon {EPS_DEN:g}, {STEPS} steps, no schedule; the validation loss ({N_VAL} points) is computed every {EVAL_EVERY} steps.",
        "",
        "Generators (the book's SplitMix64; k is the seed number 0 .. 7; a generator named `regularization_nn.<stream>.k` is seeded with the seed printed in the table below, so each of the eight seeds has its own seed for each of the three streams):",
        "",
        "- `train.k`: for i = 0 .. 15, x_i = uniform() then e_i = normal(); y_i = sin(2 pi x_i) + 0.3 e_i (for a training set of N > 16 points the same rule continues to i = N - 1).",
        "- `init.k`: w_j = normal() for j = 0 .. 63, then b_j = normal() for j = 0 .. 63, then v_j = normal() / 8 for j = 0 .. 63; c = 0.",
        "- `noise.k`: masks and augmentations (draw order in the examples that use them).",
        "",
        _table(["k", "train.k", "init.k", "noise.k"],
               [(k, *[f"0x{bookrng.seed_of(f'regularization_nn.{st}.{k}'):016x}" for st in ("train", "init", "noise")]) for k in SEEDS], "rlll"),
        "",
        f"- {_seed_line(HELD_VAL_NAME)}: {N_VAL} validation points, drawn like the training points. {_seed_line(HELD_TEST_NAME)}: {N_TEST} test points, same rule.",
        "",
        "Training points of seed 0:",
        "",
        _table(["i", "x_i", "sin(2 pi x_i)", "y_i"], rows),
    ]
    return "\n".join(out)


def memorisation():
    """The unregularized network on seed 0: the curves, and the same picture over the 8 seeds."""
    r = _runs(keep=False)
    h = r[0]["history"]
    marks = (10, 50, 100, 200, 400, 600, 1000, 1500, 2000, 3000)
    rt = _run_many([_spec(0) + (NUDGE,)])[0]["history"]
    rows = []
    for (t, tr, va, nm), (_, tr2, va2, nm2) in zip(h, rt):
        if t in marks:
            rows.append((t, _fmtm(tr, tr2, 4), _fmtm(va, va2, 4), _fmtm(nm, nm2, 1)))
    out = [
        f"Seed 0, no regularization (AdamW with weight decay 0). Training loss is the mean squared error on the {N_TRAIN} training points in eval mode; "
        f"validation is the mean squared error on the {N_VAL} validation points; the noise floor is {NOISE_SD ** 2:.2f}; the norm of the parameters is the Euclidean norm of the {PARAMS} parameters "
        "(w, b, v and c together). The late rows come from a chaotic run: a change of 1e-12 in one initial weight changes the printed digits there, so an entry is printed in full only if it survives that change "
        "and as 'about x' (the digits that survive) otherwise; the same rule applies to every number of this chapter's tables that comes from training.",
        "",
        _table(["step", "training loss", "validation loss", "norm of the parameters"], rows),
        "",
    ]
    bs = [x["best_step"] for x in r]
    rtw = _twins()
    out.append(
        f"Over the 8 seeds: the validation loss is lowest at step {_mean(bs):.0f} on average (range {min(bs)} to {max(bs)}), where it is "
        f"{_m(lambda x: x['best_val'], r, rtw)}; at step {STEPS} it is {_m(lambda x: x['final_val'], r, rtw)} on average "
        f"(range {_fmtm(min(x['final_val'] for x in r), min(x['final_val'] for x in rtw))} to {_fmtm(max(x['final_val'] for x in r), max(x['final_val'] for x in rtw))}), while the training loss is "
        f"{_m(lambda x: x['final_train'], r, rtw, 4)}."
    )
    i_stop, i_best = _stop_point(h, 10)
    out.append(
        f"Seed 0 alone: the lowest validation loss is {h[i_best][2]:.4f}, at step {h[i_best][0]}; a patience of 10 evaluations (100 steps) fires at step {h[i_stop][0]}."
    )
    return "\n".join(out)


# ------------------------------------------------------------------------ weight decay

def _prior_sd(lam):
    """The Gaussian prior on each parameter that a penalty (lam / 2) ||theta||^2 added to the mean
    squared error stands for. Volume II, chapter 4: -log posterior = (1 / (2 s^2)) sum r^2 +
    (1 / (2 sw^2)) ||theta||^2; multiplying by 2 s^2 / N gives mean r^2 + (s^2 / (N sw^2)) ||theta||^2,
    so lam / 2 = s^2 / (N sw^2) and sw^2 = 2 s^2 / (N lam)."""
    return math.sqrt(2.0 * NOISE_SD ** 2 / (N_TRAIN * lam))


def weight_decay_sweep():
    """Decoupled weight decay: the strength, what it does to a weight the data never touches, and the result."""
    rows = []
    for lam in (0.0,) + WD_GRID:
        r, rt = _runs(wd=lam), _twins(wd=lam)
        shrink = (1.0 - LR * lam) ** STEPS
        sd = "no prior" if lam == 0 else _fmt(_prior_sd(lam), 3)
        rows.append((
            f"{lam:g}", f"{LR * lam:.4f}", f"{shrink:.2e}" if lam else "1", sd,
            _m(lambda x: x["final_norm"], r, rt, 1),
            _m(lambda x: _steepest_slope(x["theta"]), r, rt, 1),
            _m(lambda x: x["final_train"], r, rt, 4),
            _m(lambda x: x["final_val"], r, rt, 3),
            _m(lambda x: x["final_test"], r, rt, 3),
        ))
    out = [
        f"Decoupled weight decay (AdamW, theta <- theta - eta lam theta - eta mhat / (sqrt(vhat) + eps)), applied to all {PARAMS} parameters, "
        f"eta = {LR}, {STEPS} steps; every row is the mean over the 8 seeds of the same 8 trainings that differ only in lam. "
        f"'eta lam' is the fraction of every weight removed per step; 'shrink' = (1 - eta lam)^{STEPS} is what is left of a weight the data never touches; "
        f"the prior sd is sqrt(2 * 0.09 / (16 lam)) (the Gaussian prior whose MAP estimate is this penalty, Volume II chapter 4). "
        f"'Steepest slope' is the largest |f(x1) - f(x0)| / (x1 - x0) between neighbouring points of the grid x = 0, 0.002, ..., 1 (the true curve's is 2 pi = 6.3). Training loss is eval-mode on the 16 training points; validation has {N_VAL} points, test {N_TEST}; the noise floor is 0.09.",
        "",
        _table(["lam", "eta lam", "shrink", "prior sd", "norm of the parameters", "steepest slope", "training loss", "validation loss", "test loss"], rows),
    ]
    return "\n".join(out)


def coupled_vs_decoupled():
    """The same penalty put inside Adam's gradient (coupled) and applied to the weights (decoupled)."""
    rows = []
    for name, coupled, grid in (("decoupled (AdamW)", False, WD_GRID), ("in the gradient (Adam + L2)", True, L2_GRID)):
        for lam in grid:
            r, rt = _runs(wd=lam, coupled=coupled), _twins(wd=lam, coupled=coupled)
            rows.append((
                name, f"{lam:g}",
                _m(lambda x: x["final_norm"], r, rt, 1),
                _m(lambda x: x["final_train"], r, rt, 4),
                _m(lambda x: x["final_val"], r, rt, 3),
            ))
    out = [
        "The same network and seeds, with the penalty (lam / 2) ||theta||^2 either applied to the weights directly (decoupled) or added to the gradient that Adam then rescales "
        "(g <- g + lam theta, no decay term). Means over the 8 seeds, 3000 steps, validation loss on the 200 validation points.",
        "",
        _table(["how the penalty enters", "lam", "norm of the parameters", "training loss", "validation loss"], rows, "llrrr"),
    ]
    return "\n".join(out)


# ------------------------------------------------------------------------------ dropout

def _mask(g, n, p):
    """n masks entries: unit j is kept (1) when uniform() >= p, dropped (0) otherwise. One draw per unit, in order."""
    return [1 if g.uniform() >= p else 0 for _ in range(n)]


def dropout_expectation():
    """Why the kept activations are divided by 1 - p: the mean and the variance of a masked activation."""
    a = 2.0
    n = 100000
    g = bookrng.generator("regularization_nn.expectation")
    rows = []
    for p in (0.1, 0.25, 0.5, 0.75):
        scale = 1.0 / (1.0 - p)
        raw = [a * m for m in _mask(g, n, p)]
        sc = [r * scale for r in raw]
        mr, ms = _mean(raw), _mean(sc)
        vs = sum((x - ms) ** 2 for x in sc) / n
        rows.append((f"{p:g}", f"{scale:.4f}", _fmt(mr, 4), _fmt((1 - p) * a, 4), _fmt(ms, 4), _fmt(a, 4), _fmt(vs, 4), _fmt(a * a * p / (1 - p), 4)))
    out = [
        f"One activation a = {a}, dropped with probability p. For each p in the order of the table, {n} masks: {n} uniform() draws from one generator "
        f"({_seed_line('regularization_nn.expectation')}); the activation is dropped when the draw is below p. "
        "'raw' is the activation times the mask (0 or a); 'scaled' is the raw value times 1 / (1 - p). Variances use divisor n. "
        "The values in the columns 'expected' are the formulas of the text.",
        "",
        _table(["p", "1 / (1 - p)", "mean of raw", "expected (1 - p) a", "mean of scaled", "expected a", "variance of scaled", "expected a^2 p / (1 - p)"], rows),
    ]
    return "\n".join(out)


DROPOUT_A = (1.0, -0.5, 2.0, 0.5, -1.5, 0.25)


def dropout_mask_data():
    """Three masks over six activations at p = 0.5, for the table and the figure."""
    g = bookrng.generator("regularization_nn.three-masks")
    p = 0.5
    masks = []
    for _ in range(3):
        draws = [g.uniform() for _ in DROPOUT_A]
        masks.append((draws, [1 if d >= p else 0 for d in draws]))
    return DROPOUT_A, p, masks


def dropout_masks():
    a, p, masks = dropout_mask_data()
    scale = 1.0 / (1.0 - p)
    rows = [("eval mode", "-", "-", "(" + ", ".join(f"{x:g}" for x in a) + ")", f"{sum(a):g}")]
    for k, (draws, m) in enumerate(masks):
        out_v = [x * mj * scale for x, mj in zip(a, m)]
        rows.append((f"mask {k + 1}", "(" + ", ".join(f"{d:.3f}" for d in draws) + ")", "(" + ", ".join(str(x) for x in m) + ")",
                     "(" + ", ".join(f"{x + 0.0:g}" for x in out_v) + ")", f"{sum(out_v):g}"))
    return "\n".join([
        f"Six activations a = (1, -0.5, 2, 0.5, -1.5, 0.25) and p = 0.5, so the scale is 1 / (1 - p) = 2. Each mask is six uniform() draws in the order of the units "
        f"({_seed_line('regularization_nn.three-masks')}); a unit is dropped (mask 0) when its draw is below 0.5. The output of the layer is a_j m_j / (1 - p).",
        "",
        _table(["", "draws", "mask m", "output a_j m_j / (1 - p)", "sum"], rows, "lllll"),
    ])


ENS_H = (1.0, -0.5, 2.0, 0.5, -1.5, 0.25)
ENS_V = (0.5, 1.0, -0.25, 2.0, 0.4, -1.0)
ENS_C = 0.1


def dropout_ensemble():
    """All 2^6 masks of a six-unit layer: the average of the sub-networks is the eval-mode network."""
    p = 0.5
    scale = 1.0 / (1.0 - p)
    outs, tans = [], []
    for code in range(64):
        m = [(code >> j) & 1 for j in range(6)]
        s = ENS_C + sum(v * h * mj * scale for v, h, mj in zip(ENS_V, ENS_H, m))
        outs.append(s)
        tans.append(math.tanh(s))
    ev = ENS_C + sum(v * h for v, h in zip(ENS_V, ENS_H))
    mean = _mean(outs)
    var = sum((s - mean) ** 2 for s in outs) / 64
    rows = [
        ("linear output c + sum_j v_j h_j m_j / (1 - p)", _fmt(mean, 6), _fmt(ev, 6), f"{abs(mean - ev):.1e}", _fmt(min(outs), 3), _fmt(max(outs), 3)),
        ("tanh of that output", _fmt(_mean(tans), 6), _fmt(math.tanh(ev), 6), f"{abs(_mean(tans) - math.tanh(ev)):.1e}", _fmt(min(tans), 3), _fmt(max(tans), 3)),
    ]
    return "\n".join([
        "A layer of six units with fixed activations h = (1, -0.5, 2, 0.5, -1.5, 0.25), output weights v = (0.5, 1, -0.25, 2, 0.4, -1), bias c = 0.1 and p = 0.5. "
        "All 2^6 = 64 masks are enumerated, each with probability (1/2)^6, so the 'mean over masks' is exact; 'eval' is the output with no mask and no scaling. "
        f"The standard deviation of the linear output over the masks is {math.sqrt(var):.4f}.",
        "",
        _table(["quantity", "mean over the 64 masks", "eval mode", "difference", "smallest", "largest"], rows, "lrrrrr"),
    ])


def _forward_small(w, b, v, c, u, mask):
    h = [math.tanh(wj * u + bj) for wj, bj in zip(w, b)]
    o = c + sum(vj * hj * mj for vj, hj, mj in zip(v, h, mask))
    return h, o


def _grad_small(w, b, v, c, u, y, forward_mask, backward_mask):
    """Backward pass of L = (o - y)^2 for the small net. The output o comes from `forward_mask`; the
    backward pass multiplies by `backward_mask` (both with the 1 / (1 - p) already in the kept entries).
    The correct backward pass passes the same mask twice."""
    h, o = _forward_small(w, b, v, c, u, forward_mask)
    d = 2.0 * (o - y)
    gv = [d * hj * mj for hj, mj in zip(h, backward_mask)]
    dh = [d * vj * mj for vj, mj in zip(v, backward_mask)]
    dz = [dhj * (1.0 - hj * hj) for dhj, hj in zip(dh, h)]
    return [dzj * u for dzj in dz], dz, gv, d


def dropout_backward():
    """The gradient of one example with a fixed mask: formulas against central differences."""
    g = bookrng.generator("regularization_nn.dropout-gradient")
    n = 4
    w = [g.normal() for _ in range(n)]
    b = [g.normal() for _ in range(n)]
    v = [g.normal() for _ in range(n)]
    c = 0.1
    x, y, p = 0.7, 0.3, 0.25
    u = (x - 0.5) * SQRT12
    scale = 1.0 / (1.0 - p)
    keep1 = _mask(g, n, p)
    keep2 = _mask(g, n, p)
    mask = [k * scale for k in keep1]
    other = [k * scale for k in keep2]
    gw, gb, gv, gc = _grad_small(w, b, v, c, u, y, mask, mask)
    gw2, gb2, gv2, gc2 = _grad_small(w, b, v, c, u, y, mask, other)

    def loss(params):
        _, o = _forward_small(params[:n], params[n:2 * n], params[2 * n:3 * n], params[3 * n], u, mask)
        return (o - y) ** 2

    theta = w + b + v + [c]
    analytic = gw + gb + gv + [gc]
    wrong = gw2 + gb2 + gv2 + [gc2]
    eps = 1e-6
    rows = []
    names = [f"w{j}" for j in range(n)] + [f"b{j}" for j in range(n)] + [f"v{j}" for j in range(n)] + ["c"]
    for i, name in enumerate(names):
        hi = list(theta)
        lo = list(theta)
        hi[i] += eps
        lo[i] -= eps
        num = (loss(hi) - loss(lo)) / (2 * eps)
        unit = i % n
        dropped = "" if i == 3 * n else (" (unit dropped)" if mask[unit] == 0 else "")
        rows.append((name + dropped, _fmt(analytic[i], 6), _fmt(num, 6), f"{abs(analytic[i] - num):.1e}", _fmt(wrong[i], 6)))
    return "\n".join([
        f"A network of {n} hidden units, one example x = {x} (u = (x - 0.5) sqrt(12) = {u:.4f}), y = {y}, L = (o - y)^2, p = {p} so the scale is {scale:.4f}. "
        f"Parameters from {_seed_line('regularization_nn.dropout-gradient')}: w_j (4 normal() draws), then b_j (4), then v_j (4); c = {c}. Then the mask used in the forward pass: "
        f"4 uniform() draws (kept when the draw is at least p); then a second mask of 4 draws, used only in the last column, where the forward pass uses the first mask and the backward pass the second. "
        f"Forward mask m = {keep1} (times the scale); second mask = {keep2}. Central differences with step {eps:g} on the loss with the forward mask held fixed.",
        "",
        _table(["parameter", "backward, mask reused", "central difference", "difference", "backward with a fresh mask"], rows, "lrrrr"),
    ])


def _stochastic(theta, x, p, g):
    """Train-mode output at x: one fresh mask over the 64 hidden units, kept units scaled by 1 / (1 - p)."""
    w, b, v, c = theta[:HIDDEN], theta[HIDDEN:2 * HIDDEN], theta[2 * HIDDEN:3 * HIDDEN], theta[3 * HIDDEN]
    u = (x - 0.5) * SQRT12
    s = c
    for j in range(HIDDEN):
        if g.uniform() >= p:
            s += v[j] * math.tanh(w[j] * u + b[j]) / (1.0 - p)
    return s


def train_vs_eval():
    """The dropout network of seed 0 evaluated at one input in eval mode and in train mode."""
    p = 0.3
    x = 0.75
    pair = []
    for spec in (_spec(0, drop=p), _spec(0, drop=p) + (NUDGE,)):
        theta = _run_many([spec])[0]["theta"]
        ev = _predict(theta, [x])[0]
        g = bookrng.generator("regularization_nn.predict")
        five = [_stochastic(theta, x, p, g) for _ in range(5)]
        many = [_stochastic(theta, x, p, g) for _ in range(2000)]
        pair.append((five, ev, _mean(many), _sd(many)))
    (five, ev, mm, sd), (five2, ev2, mm2, sd2) = pair
    rows = [(f"train mode, mask {i + 1}", _fmtm(s, s2, 4)) for i, (s, s2) in enumerate(zip(five, five2))]
    rows.append(("eval mode (no mask, no scaling)", _fmtm(ev, ev2, 4)))
    rows.append(("mean of 2000 further train-mode outputs", _fmtm(mm, mm2, 4)))
    rows.append(("standard deviation of those 2000", _fmtm(sd, sd2, 4)))
    rows.append(("true value sin(2 pi x)", _fmt(_truth(x), 4)))
    return "\n".join([
        f"The network of seed 0 trained with dropout p = {p} on its hidden layer (3000 steps, otherwise the recipe of the chapter), evaluated at x = {x}. "
        f"Each train-mode output uses 64 fresh uniform() draws, one per unit, from {_seed_line('regularization_nn.predict')}: the five rows, then 2000 more.",
        "",
        _table(["output", "value"], rows, "lr"),
    ])


def dropout_sweep():
    rows = []
    for p in (0.0,) + DROP_GRID:
        r, rt = _runs(drop=p), _twins(drop=p)
        rows.append((f"{p:g}", _m(lambda x: x["final_norm"], r, rt, 1), _m(lambda x: x["final_train"], r, rt, 4),
                     _m(lambda x: x["final_val"], r, rt, 3), _m(lambda x: x["best_val"], r, rt, 3),
                     f"{_mean([x['best_step'] for x in r]):.0f}"))
    return "\n".join([
        "Dropout on the 64 hidden units, p as listed, the recipe of the chapter, means over the 8 seeds. Training loss is eval mode on the 16 training points; "
        "'best validation' is the lowest validation loss met during the run (checked every 10 steps), 'at step' the step where it was met.",
        "",
        _table(["p", "norm of the parameters", "training loss", "validation loss at step 3000", "best validation", "at step"], rows),
    ])


def dropout_by_size():
    rows = []
    for n in N_SWEEP:
        a, at = _runs(n=n), _twins(n=n)
        d, dt = _runs(n=n, drop=0.3), _twins(n=n, drop=0.3)
        ta = [x["final_test"] for x in a]
        td = [x["final_test"] for x in d]
        diff, tt, wins = _paired_text(td, ta, [x["final_test"] for x in dt], [x["final_test"] for x in at])
        rows.append((n, f"{PARAMS / n:.1f}", _m(lambda x: x["final_train"], a, at, 3), _m(lambda x: x["final_test"], a, at, 3), _m(lambda x: x["final_test"], d, dt, 3), diff, tt, wins))
    return "\n".join([
        "The same network and recipe with N training examples (N = 16 is the chapter's task; for the others the training points are the first N pairs of a stream "
        "`regularization_nn.train.k` drawn by the same rule), without dropout and with p = 0.3, 3000 steps, test loss on the 1000 test points, means over the 8 seeds. "
        "The difference is dropout minus none for each seed; 'wins' counts the seeds where dropout has the lower test loss.",
        "",
        _table(["N", "parameters per example", "training loss, none", "test loss, none", "test loss, p = 0.3", "mean difference", "paired t", "wins"], rows),
    ])


# ----------------------------------------------------------------------------- augmentation

def augmentation_ridge():
    """Noise added to the inputs of a linear model is a ridge penalty: the exact formula against sampled copies."""
    g = bookrng.generator("regularization_nn.ridge")
    n, d = 12, 3
    w_true = (1.0, -2.0, 0.5)
    xs = [[g.normal() for _ in range(d)] for _ in range(n)]
    ys = [sum(wi * xi for wi, xi in zip(w_true, x)) + 0.5 * g.normal() for x in xs]

    def solve(A, rhs):
        m = len(A)
        M = [row[:] + [rhs[i]] for i, row in enumerate(A)]
        for c in range(m):
            piv = max(range(c, m), key=lambda r: abs(M[r][c]))
            M[c], M[piv] = M[piv], M[c]
            for r in range(c + 1, m):
                f = M[r][c] / M[c][c]
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
        sol = [0.0] * m
        for r in range(m - 1, -1, -1):
            sol[r] = (M[r][m] - sum(M[r][k] * sol[k] for k in range(r + 1, m))) / M[r][r]
        return sol

    def normal_equations(rows, targets, ridge):
        A = [[sum(r[i] * r[j] for r in rows) + (ridge if i == j else 0.0) for j in range(d)] for i in range(d)]
        rhs = [sum(r[i] * t for r, t in zip(rows, targets)) for i in range(d)]
        return solve(A, rhs)

    out_rows = []
    s_list = (0.3, 0.6)
    for s in s_list:
        exact = normal_equations(xs, ys, n * s * s)
        row = [f"{s:g}", f"{n * s * s:.3f}", "(" + ", ".join(_fmt(w, 4) for w in exact) + ")"]
        for k in (10, 1000):
            rows_aug, targets = [], []
            for x, y in zip(xs, ys):
                for _ in range(k):
                    rows_aug.append([xi + s * g.normal() for xi in x])
                    targets.append(y)
            w = normal_equations(rows_aug, targets, 0.0)
            row.append("(" + ", ".join(_fmt(wi, 4) for wi in w) + ")")
            row.append(f"{max(abs(a - b) for a, b in zip(w, exact)):.4f}")
        out_rows.append(tuple(row))
    plain = normal_equations(xs, ys, 0.0)
    return "\n".join([
        f"A linear model y = w . x with {d} features on {n} examples, features and noise from one generator ({_seed_line('regularization_nn.ridge')}): "
        f"first all {n} feature rows, row by row (3 normal() draws per row, 36 draws in all), and only then the {n} label noises (one normal() each, in the order of the rows), so that y_i = 1 x_i0 - 2 x_i1 + 0.5 x_i2 + 0.5 e_i with e_i the i-th noise draw. Least squares without any augmentation: w = ({', '.join(_fmt(w, 4) for w in plain)}). "
        "'Ridge' solves (X^T X + N s^2 I) w = X^T y. 'K copies' replaces each example by K noisy copies x + s * (three normal() draws), with the same label, and solves ordinary least squares "
        "on the 12 K rows (the draws continue the same generator: first all rows of s = 0.3 with K = 10, then K = 1000, then s = 0.6 with K = 10, then K = 1000; within one setting example after example, K copies of an example one after the other, each copy three normal() draws, one per feature). "
        "'Max difference' is the largest difference of an entry from the ridge solution.",
        "",
        _table(["noise sd s", "ridge strength N s^2", "ridge w", "K = 10 copies: w", "max difference", "K = 1000 copies: w", "max difference"], out_rows, "lrlrrrr"),
    ])


def _drift(kind, amount, g, n=20000):
    """Root mean square change of the noiseless target when the augmentation moves the input: sqrt(E (f(x') - f(x))^2)."""
    total = 0.0
    for _ in range(n):
        x = g.uniform()
        xp = _augment(kind, amount, x, g)
        total += (_truth(xp) - _truth(x)) ** 2
    return math.sqrt(total / n)


AUGMENTATIONS = (
    ("none", ("none", 0.0)),
    ("noise sd 0.03", ("noise", 0.03)),
    ("noise sd 0.1", ("noise", 0.1)),
    ("noise sd 0.2", ("noise", 0.2)),
    ("gain 1 +- 0.1", ("gain", GAIN_SPREAD)),
    ("shift by 0.5, half the time", ("shift", 0.5)),
)


def augmentation_table():
    g = bookrng.generator("regularization_nn.drift")
    base, baset = _runs(), _twins()
    tb = [x["final_test"] for x in base]
    tbt = [x["final_test"] for x in baset]
    rows = []
    for name, aug in AUGMENTATIONS:
        r, rt = _runs(aug=aug), _twins(aug=aug)
        drift = 0.0 if aug[0] == "none" else _drift(aug[0], aug[1], g)
        t_ = [x["final_test"] for x in r]
        diff, tt, _ = _paired_text(t_, tb, [x["final_test"] for x in rt], tbt)
        rows.append((name, _fmt(drift, 3), _m(lambda x: x["final_train"], r, rt, 4), _m(lambda x: x["final_val"], r, rt, 3),
                     _m(lambda x: x["final_test"], r, rt, 3), "-" if aug[0] == "none" else diff.split(" (")[0], "-" if aug[0] == "none" else tt))
    return "\n".join([
        "Augmentation by changing the input of a training example, with its label kept, a fresh draw at every step for every example (online). Draw order of the stream "
        "`regularization_nn.noise.k`: for each step, for each example in order, one draw, then (when dropout is on, which it is not here) the 64 mask draws. 'noise' adds sd * normal() to x; "
        "'gain' multiplies x by 1 + 0.1 (2 uniform() - 1); 'shift' adds 0.5 to x when uniform() < 0.5. 'Label drift' is sqrt(E (sin(2 pi x') - sin(2 pi x))^2), the change "
        f"in the true label that the augmentation causes, over 20000 points x = uniform() with the augmentation drawn from {_seed_line('regularization_nn.drift')} (one generator, rows in the order of the table, "
        "per point first x then the augmentation draw). Training loss is eval mode on the unaugmented training points. Means over 8 seeds; the difference is the test loss minus that of no augmentation, per seed, 3000 steps.",
        "",
        _table(["augmentation", "label drift", "training loss", "validation loss", "test loss", "mean difference in test loss", "paired t"], rows),
    ])


# ---------------------------------------------------------------------------- early stopping

PATIENCES = (1, 2, 5, 10, 20, 50)


def _stop_point(history, patience):
    """Index of the evaluation at which a patience rule stops, and of the best evaluation before it.

    The rule: keep the lowest validation loss met so far (a strict improvement resets the counter);
    stop at the first evaluation where `patience` evaluations in a row have not improved on the best."""
    best, best_i, since = float("inf"), 0, 0
    for i, (_, _, val, _) in enumerate(history):
        if val < best:
            best, best_i, since = val, i, 0
        else:
            since += 1
            if since >= patience:
                return i, best_i
    return len(history) - 1, best_i


def patience_rules():
    """Stopping rules replayed on the recorded validation curves of the runs without and with dropout."""
    _, _, tx, ty = _heldout()
    out_rows = []
    for label, kw in (("no regularization", {}), ("dropout p = 0.3", {"drop": 0.3})):
        runs, twins = _runs(keep=True, **kw), _twins(keep=True, **kw)
        full_test = _m(lambda x: x["final_test"], runs, twins)
        oracle = _m(lambda x: x["best_test"], runs, twins)
        for pat in PATIENCES:
            res = []
            for group in (runs, twins):
                stops, restored, at_stop, bests = [], [], [], []
                for x in group:
                    i_stop, i_best = _stop_point(x["history"], pat)
                    stops.append(x["history"][i_stop][0])
                    bests.append(x["history"][i_best][0])
                    restored.append(_mse(x["snapshots"][i_best], tx, ty))
                    at_stop.append(_mse(x["snapshots"][i_stop], tx, ty))
                res.append((_mean(stops), _mean(bests), _mean(restored), _mean(at_stop)))
            (st, be, re_, ats), (st2, be2, re2, ats2) = res
            step = lambda a, b: f"{a:.0f}" if round(a) == round(b) else f"about {a:.0f}"   # noqa: E731
            out_rows.append((label, pat, step(st, st2), step(be, be2), _fmtm(re_, re2, 3), _fmtm(ats, ats2, 3), full_test, oracle))
    return "\n".join([
        f"The recorded validation losses (one every {EVAL_EVERY} steps) of the 8 runs of each setting, replayed with a patience rule: keep the lowest validation loss so far "
        "(only a strictly lower value counts as an improvement) and stop at the first evaluation at which `patience` evaluations in a row have not improved on it. "
        "'Stop' is the mean step at which the rule fires, 'best' the mean step of the checkpoint it keeps; 'test at best' is the mean test loss of that restored checkpoint, "
        "'test at stop' of the weights at the moment of stopping (no restore). 'Test at 3000' is the same runs never stopped, and 'oracle' the test loss at the best-validation "
        "checkpoint over all 3000 steps (patience without limit).",
        "",
        _table(["setting", "patience (evaluations)", "stop (step)", "best (step)", "test at best", "test at stop", "test at 3000", "oracle"], out_rows, "lrrrrrrr"),
    ])


def _quadratic_problem():
    """Two correlated features, 20 rows: x1 = normal(), x2 = 0.9 x1 + 0.3 normal(), y = x1 - 0.5 x2 + 0.5 normal()."""
    g = bookrng.generator("regularization_nn.quadratic")
    n = 20
    rows = []
    for _ in range(n):
        x1 = g.normal()
        x2 = 0.9 * x1 + 0.3 * g.normal()
        y = x1 - 0.5 * x2 + 0.5 * g.normal()
        rows.append((x1, x2, y))
    h11 = sum(r[0] * r[0] for r in rows) / n
    h12 = sum(r[0] * r[1] for r in rows) / n
    h22 = sum(r[1] * r[1] for r in rows) / n
    b1 = sum(r[0] * r[2] for r in rows) / n
    b2 = sum(r[1] * r[2] for r in rows) / n
    return n, (h11, h12, h22), (b1, b2)


QUAD_ETA = 0.25


def _solve2(h, b, alpha):
    """(H + alpha I) w = b for the 2 x 2 symmetric H = (h11, h12, h22)."""
    a, bb, d = h[0] + alpha, h[1], h[2] + alpha
    det = a * d - bb * bb
    return ((d * b[0] - bb * b[1]) / det, (a * b[1] - bb * b[0]) / det)


def _eigen2(h):
    tr, det = h[0] + h[2], h[0] * h[2] - h[1] * h[1]
    disc = math.sqrt(tr * tr / 4 - det)
    return tr / 2 + disc, tr / 2 - disc


def quadratic_paths():
    """Gradient descent from zero and ridge regression on the same quadratic: the two paths."""
    n, h, b = _quadratic_problem()
    mu_hi, mu_lo = _eigen2(h)
    w_star = _solve2(h, b, 0.0)
    ts = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
    w = [0.0, 0.0]
    rows = []
    t_done = 0
    for t in ts:
        while t_done < t:
            g0 = h[0] * w[0] + h[1] * w[1] - b[0]
            g1 = h[1] * w[0] + h[2] * w[1] - b[1]
            w = [w[0] - QUAD_ETA * g0, w[1] - QUAD_ETA * g1]
            t_done += 1
        alpha = 1.0 / (QUAD_ETA * t)
        r = _solve2(h, b, alpha)
        nrm = lambda v: math.hypot(v[0], v[1])  # noqa: E731
        diff = math.hypot(w[0] - r[0], w[1] - r[1])
        rows.append((t, f"{alpha:.4f}", f"({w[0]:.4f}, {w[1]:.4f})", _fmt(nrm(w), 4), f"({r[0]:.4f}, {r[1]:.4f})", _fmt(nrm(r), 4), _fmt(diff, 4)))
    return "\n".join([
        f"Twenty rows with two correlated features from {_seed_line('regularization_nn.quadratic')}: for i = 0 .. 19, x1 = normal(), then x2 = 0.9 x1 + 0.3 normal(), "
        "then y = x1 - 0.5 x2 + 0.5 normal(). Loss (1/2N) sum (w . x - y)^2 with N = 20, so the Hessian is H = X^T X / N and the gradient H w - b with b = X^T y / N. "
        f"H = [[{h[0]:.4f}, {h[1]:.4f}], [{h[1]:.4f}, {h[2]:.4f}]] with eigenvalues {mu_hi:.4f} and {mu_lo:.4f}; the least squares solution is w* = ({w_star[0]:.4f}, {w_star[1]:.4f}). "
        f"Gradient descent starts at w = (0, 0) with eta = {QUAD_ETA} and is read after t steps. Ridge regression minimises the loss plus (lam / 2) ||w||^2, "
        "that is (H + lam I) w = b, at lam = 1 / (eta t).",
        "",
        _table(["t", "lam = 1 / (eta t)", "gradient descent w", "norm", "ridge w", "norm", "distance between them"], rows, "rrlrlrr"),
    ])


def quadratic_filters():
    """Per eigen-direction: the fraction of the least squares coefficient that each method keeps."""
    n, h, b = _quadratic_problem()
    mu = _eigen2(h)
    rows = []
    for t in (1, 5, 20, 100, 500):
        alpha = 1.0 / (QUAD_ETA * t)
        row = [t, f"{alpha:.4f}"]
        for m in mu:
            row.append(_fmt(1.0 - (1.0 - QUAD_ETA * m) ** t, 4))
            row.append(_fmt(m / (m + alpha), 4))
        rows.append(tuple(row))
    return "\n".join([
        f"The same problem. Along the eigenvector of H with curvature c (the eigenvalue; H and eta of the previous example), gradient descent from zero has kept the fraction 1 - (1 - eta c)^t of the least squares coefficient "
        f"after t steps, and ridge regression with lam = 1 / (eta t) keeps c / (c + lam). The two curvatures are {mu[0]:.4f} and {mu[1]:.4f}.",
        "",
        _table(["t", "lam = 1 / (eta t)", f"gradient descent, c = {mu[0]:.3f}", "ridge", f"gradient descent, c = {mu[1]:.3f}", "ridge"], rows),
    ])


def path_data():
    """The two paths of the figure: gradient descent at t = 0 .. 1000, and the ridge path for lam from 20 down to 0."""
    n, h, b = _quadratic_problem()
    w = [0.0, 0.0]
    gd = [tuple(w)]
    for t in range(1, 1001):
        g0 = h[0] * w[0] + h[1] * w[1] - b[0]
        g1 = h[1] * w[0] + h[2] * w[1] - b[1]
        w = [w[0] - QUAD_ETA * g0, w[1] - QUAD_ETA * g1]
        gd.append(tuple(w))
    ridge = [(a, _solve2(h, b, a)) for a in [20.0 * (0.9 ** k) for k in range(0, 200)] + [0.0]]
    marks = [(t, _solve2(h, b, 1.0 / (QUAD_ETA * t))) for t in (1, 5, 20, 100, 500)]
    return gd, ridge, marks, _solve2(h, b, 0.0)


# ------------------------------------------------------------------------ label smoothing

def label_smoothing_gap():
    """The logit gap that minimises the cross-entropy against a smoothed target (chapter 3), found by bisection."""
    c = 4
    rows = []
    for eps in (0.0, 0.05, 0.1, 0.2):
        t_true = 1.0 - eps + eps / c
        if eps == 0.0:
            rows.append(("0", "1", "never (the gap grows without limit)", "-", "0.0000"))
            continue
        lo, hi = 0.0, 40.0
        for _ in range(200):                      # dL/dgap = p_true(gap) - t_true is increasing in gap
            mid = 0.5 * (lo + hi)
            p_true = math.exp(mid) / (math.exp(mid) + (c - 1))
            if p_true < t_true:
                lo = mid
            else:
                hi = mid
        gap = 0.5 * (lo + hi)
        closed = math.log((c * (1.0 - eps) + eps) / eps)
        p_other = eps / c
        loss = -(t_true * math.log(t_true) + (c - 1) * p_other * math.log(p_other))
        rows.append((f"{eps:g}", _fmt(t_true, 4), _fmt(gap, 4), _fmt(closed, 4), _fmt(loss, 4)))
    return "\n".join([
        "Four classes. The logits are (g, 0, 0, 0) for a correct example with a gap g; the target is (1 - eps) one-hot + eps / 4 on every class. The cross-entropy as a function of g is minimised "
        "where the predicted probability of the true class equals the target, found here by 200 bisection steps on d(loss)/dg = p_true(g) - t_true over [0, 40]; "
        "the closed form is g = ln((4 (1 - eps) + eps) / eps). The last column is the minimum loss (the entropy of the target).",
        "",
        _table(["eps", "target of the true class", "best gap (bisection)", "best gap (closed form)", "minimum loss"], rows),
    ])


# --------------------------------------------------------------------------- comparison

def _chosen():
    """The setting of each method with the lowest mean validation loss at step 3000, among its grid."""
    wd = min(WD_GRID, key=lambda lam: _mean([x["final_val"] for x in _runs(wd=lam)]))
    dp = min(DROP_GRID, key=lambda p: _mean([x["final_val"] for x in _runs(drop=p)]))
    ag = min(NOISE_GRID, key=lambda s: _mean([x["final_val"] for x in _runs(aug=("noise", s))]))
    return wd, dp, ag


def _methods():
    wd, dp, ag = _chosen()
    aug = ("noise", ag)
    return [
        ("none", "-", {}),
        ("weight decay", f"lam {wd:g}", {"wd": wd}),
        ("dropout", f"p {dp:g}", {"drop": dp}),
        ("augmentation", f"noise sd {ag:g}", {"aug": aug}),
        ("weight decay + dropout", f"lam {wd:g}, p {dp:g}", {"wd": wd, "drop": dp}),
        ("weight decay + augmentation", f"lam {wd:g}, noise sd {ag:g}", {"wd": wd, "aug": aug}),
        ("dropout + augmentation", f"p {dp:g}, noise sd {ag:g}", {"drop": dp, "aug": aug}),
        ("all three", f"lam {wd:g}, p {dp:g}, noise sd {ag:g}", {"wd": wd, "drop": dp, "aug": aug}),
    ]


def comparison_data():
    """(name, setting, runs) for every method of the comparison, in table order."""
    return [(name, setting, _runs(**kw)) for name, setting, kw in _methods()]


def method_comparison():
    data = comparison_data()
    wd, dp, ag = _chosen()
    twin_data = [(name, _twins(**kw)) for name, _, kw in _methods()]
    none, none_t = data[0][2], twin_data[0][1]
    tn = [x["final_test"] for x in none]
    te = [x["best_test"] for x in none]
    tnt = [x["final_test"] for x in none_t]
    tet = [x["best_test"] for x in none_t]
    rows_a, rows_b = [], []
    for (name, setting, r), (_, rt) in zip(data, twin_data):
        t_final = [x["final_test"] for x in r]
        t_es = [x["best_test"] for x in r]
        d1, tt1, w1 = _paired_text(t_final, tn, [x["final_test"] for x in rt], tnt)
        rows_a.append((name, setting, _m(lambda x: x["final_val"], r, rt), _m(lambda x: x["final_test"], r, rt),
                       _fmtm(_sd(t_final), _sd([x["final_test"] for x in rt]), 3),
                       "-" if name == "none" else d1, "-" if name == "none" else tt1,
                       "-" if name == "none" else w1, _m(lambda x: x["final_norm"], r, rt, 1)))
        d2, tt2, w2 = _paired_text(t_es, te, [x["best_test"] for x in rt], tet)
        rows_b.append((name, _m(lambda x: x["best_val"], r, rt), f"{_mean([x['best_step'] for x in r]):.0f}", _m(lambda x: x["best_test"], r, rt),
                       "-" if name == "none" else d2, "-" if name == "none" else tt2,
                       "-" if name == "none" else w2))
    return "\n".join([
        "Settings: the value of each method's grid with the lowest mean validation loss at step 3000 (weight decay lam in "
        f"{WD_GRID}, dropout p in {DROP_GRID}, noise sd in {NOISE_GRID}); the combinations use those values. "
        f"The chosen values are lam = {wd:g}, p = {dp:g}, noise sd = {ag:g}. Every row is the same 8 seeds (same training points, same initial weights); "
        "test loss is on the 1000 test points, which were not used to choose anything. 'Test (sd over seeds)' is the standard deviation with divisor 7 of the eight test losses. 'Difference' is the method's test loss minus that of 'none' per seed: mean (standard error = sd / sqrt(8), sd with divisor 7), "
        "paired t with 7 degrees of freedom (|t| above 2.36 is significant at 5%), and the number of seeds where the method is better. The noise floor of the test loss is 0.09.",
        "",
        "Trained for 3000 steps and taken as they are:",
        "",
        _table(["method", "setting", "validation", "test (mean)", "test (sd over seeds)", "difference to none (SE)", "paired t", "better in", "norm"], rows_a, "llrrrrrrr"),
        "",
        "The same runs, each stopped at its own best validation step (the checkpoint with the lowest validation loss, as the early-stopping rule of this chapter would keep). "
        f"The difference is to the unregularized network stopped at its own best step (test loss {_mean(te):.3f}, the first row), per seed.",
        "",
        _table(["method", "best validation", "at step", "test (mean)", "difference to none, early stopped (SE)", "paired t", "better in"], rows_b, "lrrrrrr"),
    ])


# --------------------------------------------------------------------------------- costs

def regularizer_costs():
    """What each method adds to the work and the memory of one training step of a dense hidden layer."""
    b, k, m = 256, 1024, 1024
    elems = b * m
    layer_fwd = 2 * b * k * m
    layer_bwd = 4 * b * k * m
    layer_bytes_fwd = 4 * (b * k + k * m + b * m)
    layer_bytes_bwd = 4 * (b * m + b * k + k * m) + 4 * (b * k + k * m)
    drop_fwd = 2 * elems                       # one comparison and one multiplication per element
    drop_bwd = elems                           # one multiplication per element
    drop_bytes_fwd = elems * (4 + 4 + 1)       # read x, write y, write a one-byte mask
    drop_bytes_bwd = elems * (4 + 1 + 4)       # read dy, read the mask, write dx
    params = k * m
    wd_flops = 2 * params
    adamw_flops = 16 * params
    v, every = 2048, 100
    val_flops = 2 * v * k * m
    train_flops = layer_fwd + layer_bwd
    rows = [
        ("hidden layer forward", f"{layer_fwd:,}", "0", f"{layer_bytes_fwd:,}", "the activations kept for backward"),
        ("hidden layer backward", f"{layer_bwd:,}", "0", f"{layer_bytes_bwd:,}", "-"),
        ("dropout forward", f"{drop_fwd:,}", f"{elems:,}", f"{drop_bytes_fwd:,}", f"mask: {elems:,} B (bool), {4 * elems:,} B (f32), {elems // 8:,} B (bits), 0 B (replayed)"),
        ("dropout backward", f"{drop_bwd:,}", "0 (or " + f"{elems:,}" + " if the mask is replayed)", f"{drop_bytes_bwd:,}", "-"),
        ("weight decay, per step", f"{wd_flops:,}", "0", "0 extra (fused in the optimizer step)", "none"),
        ("AdamW update altogether", f"{adamw_flops:,}", "0", f"{28 * params:,}", f"{8 * params:,} B of moments"),
        ("one validation pass of 2048 examples", f"{val_flops:,}", "0", "-", "a copy of the best weights: " + f"{4 * params:,} B"),
    ]
    ratios = [
        f"Dropout forward arithmetic / layer forward arithmetic = {drop_fwd / layer_fwd:.5f}; dropout forward bytes / layer forward bytes = {drop_bytes_fwd / layer_bytes_fwd:.3f}.",
        f"Weight decay arithmetic / AdamW update arithmetic = {wd_flops / adamw_flops:.4f} (2 of 16 FLOPs per parameter).",
        f"Early stopping with one validation pass of {v} examples every {every} steps adds {val_flops / every / train_flops:.4f} of the training arithmetic (forward and backward of the layer).",
    ]
    return "\n".join([
        f"One dense hidden layer with B = {b} examples, D = {k} inputs and H = {m} outputs, 32-bit floats. FLOPs: a multiply-add counts 2, so the layer forward is 2 B D H and its backward "
        "(the gradient of the weights and of the input) 4 B D H. Dropout forward: one comparison and one multiplication per output element; backward: one multiplication. "
        "Bytes moved: the layer forward reads x and W and writes y; the backward reads dy, x and W and writes dx and dW; dropout forward reads x, writes y and a one-byte mask, "
        "backward reads dy and the mask and writes dx. Weight decay: 2 FLOPs per parameter (the 'decay 2' of the AdamW count of chapter 7, whose update is 16 FLOPs and 28 bytes of traffic per parameter).",
        "",
        _table(["item", "FLOPs", "random draws", "bytes moved", "memory held"], rows, "lrrrl"),
        "",
        *ratios,
    ])
