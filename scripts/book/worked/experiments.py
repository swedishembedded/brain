# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Experiments, Baselines and Claims".

Plain Python 3, standard library only, written from the chapter's own
description. One small world (four noisy sensor readings, a binary label whose
rule is known, see `dataset_recipe`) carries every experiment: a claim that a
new preprocessing step raises accuracy, a ladder of baselines, run-to-run
variance over many seeds, the effect of unequal tuning effort, the difference
between an effect and its significance as the number of examples grows, a 2 x 2
ablation with an interaction, the interval a result deserves, and four ways of
manufacturing a win from a method that has none. Each public function without a
leading underscore that takes no arguments returns a markdown string that the
chapter embeds between `<!-- worked: experiments.<name> -->` markers.
Underscore functions and the constants are shared with the chapter's figures,
so a picture and the table beside it cannot disagree.

Swedish Embedded AB implements solutions for experiments whose results a
customer can believe and repeat. If your team needs expertise in experiment
design, baselines and statistical reporting for machine learning, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache

import bookrng

# ----------------------------------------------------------------- the world

OFFSET = (2.0, 2.0, 0.0, 0.0)       # added to the four readings: the sensors have a resting level
TRUE_W = (1.2, -0.8, 0.5, 0.0)      # weights of the centred readings in the true logit
TRUE_INTERACTION = 0.6              # weight of the product of the first two centred readings
N_TRAIN, N_VAL, N_TEST = 200, 300, 1000
DATA_NAME = "experiments.dataset"
CEILING_NAME = "experiments.ceiling"
CEILING_N = 20000
SEEDS_NAME = "experiments.seeds"
N_RUNS = 100                       # paired runs; seeds N_RUNS .. 2 N_RUNS - 1 are the unpaired partners
LR, EPOCHS, INIT_SD = 0.03, 3, 0.5  # the recipe every experiment uses unless it says otherwise
CLAIM_POINTS = 1.2                  # the claimed gain, in points of accuracy

PIPELINES = ("plain", "interaction", "centring", "both")  # the second and third are the two parts of "both"
PIPELINE_LABELS = {"plain": "plain", "interaction": "interaction only", "centring": "centring only",
                   "both": "centring + interaction"}


# ----------------------------------------------------------------- helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _generator_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} ({bookrng.seed_of(name)})."


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _mean(v):
    return sum(v) / len(v)


def _sd(v):
    """Sample standard deviation, divisor n - 1."""
    m = _mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def _pts(v, d=1, sign=False):
    """A fraction shown as points (percent) with a fixed number of decimals."""
    s = f"{100.0 * v:+.{d}f}" if sign else f"{100.0 * v:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def _derive(run_seed, name):
    """derive_seed of Volume I, chapter 17: a chain of the SplitMix64 mixing function over the bytes of the name."""
    h = bookrng._mix(run_seed)
    for b in name.encode("utf-8"):
        h = bookrng._mix(h ^ b)
    return h


def _stream(run_seed, name):
    return bookrng.SplitMix64(_derive(run_seed, name))


def _t_multiplier(df, level=0.95):
    """The multiplier holding `level` of a Student t density with df degrees of freedom: integrate the density, bisect."""
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))

    def pdf(t):
        return c * (1 + t * t / df) ** (-(df + 1) / 2)

    def central(z, steps=4000):
        h = z / steps
        s = pdf(0) + pdf(z)
        s += 4 * sum(pdf((2 * i - 1) * h) for i in range(1, steps // 2 + 1))
        s += 2 * sum(pdf(2 * i * h) for i in range(1, steps // 2))
        return 2 * s * h / 3

    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if central(mid) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@lru_cache(maxsize=None)
def _t95(df):
    return _t_multiplier(df)


def _interval(values):
    """Mean, sample sd, standard error and the 95 percent t interval of a list of per-seed numbers."""
    n = len(values)
    m, s = _mean(values), _sd(values)
    se = s / math.sqrt(n)
    half = _t95(n - 1) * se
    return m, s, se, m - half, m + half


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _wald(p, n):
    se = math.sqrt(p * (1 - p) / n)
    return 1.96 * se


# ----------------------------------------------------------------- data

def _draw_rows(rng, n):
    """n rows in order: four rng.normal() (the centred readings), then one rng.uniform() for the label.
    A row is (raw readings, label, true logit)."""
    rows = []
    for _ in range(n):
        x = [rng.normal() for _ in range(4)]
        z = sum(TRUE_W[j] * x[j] for j in range(4)) + TRUE_INTERACTION * x[0] * x[1]
        y = 1 if rng.uniform() < _sigmoid(z) else 0
        rows.append(([x[j] + OFFSET[j] for j in range(4)], y, z))
    return rows


@lru_cache(maxsize=None)
def _dataset():
    rng = bookrng.generator(DATA_NAME)
    return _draw_rows(rng, N_TRAIN), _draw_rows(rng, N_VAL), _draw_rows(rng, N_TEST)


def _train_means(train):
    return [_mean([r[0][j] for r in train]) for j in range(4)]


def _features(raw, means, pipeline):
    centre = pipeline in ("centring", "both")
    inter = pipeline in ("interaction", "both")
    x = [raw[j] - means[j] for j in range(4)] if centre else list(raw)
    return x + [x[0] * x[1]] if inter else x


def _featurise(rows, means, pipeline):
    return [_features(r[0], means, pipeline) for r in rows], [r[1] for r in rows]


# ----------------------------------------------------------------- the learner

def _score(w, b, f):
    s = 0.0
    for j in range(len(f)):
        s += w[j] * f[j]
    return b + s


def _sgd(F, y, lr, epochs, init_rng, order_rng, init_sd=INIT_SD):
    """Logistic regression by stochastic gradient descent, one example per step.

    The bias is drawn first, then the weights in feature order, each init_sd * init_rng.normal(). The index list
    0 .. N-1 is shuffled in place by order_rng at the start of every epoch (continuing from the previous order)."""
    d = len(F[0])
    b = init_sd * init_rng.normal()
    w = [init_sd * init_rng.normal() for _ in range(d)]
    idx = list(range(len(F)))
    for _ in range(epochs):
        order_rng.shuffle(idx)
        for i in idx:
            f = F[i]
            g = _sigmoid(_score(w, b, f)) - y[i]
            for j in range(d):
                w[j] = w[j] - lr * (g * f[j])
            b = b - lr * g
    return w, b


def _fit_scores(pipeline, run_seed, lr, epochs, data):
    """Train on data = (train, val, test) rows; return the test and validation scores (logits)."""
    train, val, test = data
    means = _train_means(train)
    Ftr, ytr = _featurise(train, means, pipeline)
    w, b = _sgd(Ftr, ytr, lr, epochs, _stream(run_seed, "init"), _stream(run_seed, "order"))
    Fva, _ = _featurise(val, means, pipeline)
    Fte, _ = _featurise(test, means, pipeline)
    return [_score(w, b, f) for f in Fte], [_score(w, b, f) for f in Fva]


@lru_cache(maxsize=None)
def _fixed_scores(pipeline, run_seed, lr=LR, epochs=EPOCHS):
    """Test-set scores of one run on the fixed benchmark dataset."""
    return tuple(_fit_scores(pipeline, run_seed, lr, epochs, _dataset())[0])


def _accuracy(scores, rows):
    return sum(1 for z, r in zip(scores, rows) if (z >= 0) == (r[1] == 1)) / len(rows)


def _correct(scores, rows):
    return [1 if (z >= 0) == (r[1] == 1) else 0 for z, r in zip(scores, rows)]


@lru_cache(maxsize=None)
def _run_seeds(n=2 * N_RUNS):
    rng = bookrng.generator(SEEDS_NAME)
    return tuple(rng.next_u64() for _ in range(n))


@lru_cache(maxsize=None)
def _seed_accuracies(pipeline, first_run=0):
    """Test accuracy of the fixed dataset's run for each of the N_RUNS seeds (the partner runs start at first_run > 0)."""
    test = _dataset()[2]
    seeds = _run_seeds()[first_run:first_run + N_RUNS]
    return tuple(_accuracy(_fixed_scores(pipeline, s), test) for s in seeds)


@lru_cache(maxsize=None)
def _seed_correct(pipeline):
    test = _dataset()[2]
    return tuple(tuple(_correct(_fixed_scores(pipeline, s), test)) for s in _run_seeds()[:N_RUNS])


# ----------------------------------------------------------------- stronger learners

def _solve(a, b):
    """Gaussian elimination with partial pivoting."""
    n = len(a)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            k = m[r][c] / m[c][c]
            for j in range(c, n + 1):
                m[r][j] -= k * m[c][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def _newton_logistic(F, y, l2, steps=30):
    """Logistic regression with an L2 penalty on the weights (not the bias), fitted to convergence by Newton's method."""
    d = len(F[0]) + 1
    theta = [0.0] * d                      # weights then bias
    for _ in range(steps):
        grad = [0.0] * d
        hess = [[0.0] * d for _ in range(d)]
        for f, t in zip(F, y):
            a = f + [1.0]
            p = _sigmoid(sum(theta[j] * a[j] for j in range(d)))
            s = p * (1 - p)
            for i in range(d):
                grad[i] += (p - t) * a[i]
                for j in range(d):
                    hess[i][j] += s * a[i] * a[j]
        for i in range(d - 1):
            grad[i] += l2 * theta[i]
            hess[i][i] += l2
        hess[d - 1][d - 1] += 1e-9
        step = _solve(hess, grad)
        theta = [theta[i] - step[i] for i in range(d)]
        if max(abs(v) for v in step) < 1e-10:
            break
    return theta[:-1], theta[-1]


def _quadratic(raw, means):
    x = [raw[j] - means[j] for j in range(4)]
    return x + [x[i] * x[j] for i in range(4) for j in range(i, 4)]


def _log_loss(scores, rows):
    s = 0.0
    for z, r in zip(scores, rows):
        s += (max(z, 0.0) + math.log1p(math.exp(-abs(z)))) - (z if r[1] == 1 else 0.0)
    return s / len(rows)


def _knn_accuracies(train, test, ks):
    """Accuracy of k nearest neighbours on `test` for every k in ks (standardised readings, Euclidean distance)."""
    means = _train_means(train)
    sds = [math.sqrt(_mean([(r[0][j] - means[j]) ** 2 for r in train])) for j in range(4)]

    def std(r):
        return [(r[0][j] - means[j]) / sds[j] for j in range(4)]
    S = [std(r) for r in train]
    ok = {k: 0 for k in ks}
    for r in test:
        q = std(r)
        d = sorted(range(len(S)), key=lambda i: (sum((S[i][j] - q[j]) ** 2 for j in range(4)), i))
        for k in ks:
            votes = sum(train[i][1] for i in d[:k])
            ok[k] += (1 if 2 * votes > k else 0) == r[1]
    return {k: ok[k] / len(test) for k in ks}


def _naive_bayes_accuracy(train, test):
    stats = {}
    for c in (0, 1):
        rows = [r[0] for r in train if r[1] == c]
        mu = [_mean([r[j] for r in rows]) for j in range(4)]
        var = [_mean([(r[j] - mu[j]) ** 2 for r in rows]) + 1e-9 for j in range(4)]
        stats[c] = (len(rows) / len(train), mu, var)

    def logp(c, x):
        prior, mu, var = stats[c]
        return math.log(prior) + sum(-0.5 * math.log(2 * math.pi * var[j]) - (x[j] - mu[j]) ** 2 / (2 * var[j]) for j in range(4))
    ok = 0
    for r in test:
        pred = 1 if logp(1, r[0]) > logp(0, r[0]) else 0
        ok += pred == r[1]
    return ok / len(test)


# ===================================================================== examples

def dataset_recipe():
    train, val, test = _dataset()
    first = [(i + 1, *(f"{v:.2f}" for v in r[0]), r[1]) for i, r in enumerate(train[:5])]
    shares = [(name, len(rows), f"{_mean([r[1] for r in rows]):.3f}") for name, rows in
              (("training", train), ("validation", val), ("test", test))]
    means = _train_means(train)
    out = (f"{_generator_line(DATA_NAME)} One generator, rows drawn in order: {N_TRAIN} training rows, then {N_VAL} validation rows, "
           f"then {N_TEST} test rows. A row is four calls of `rng.normal()` (x1, x2, x3, x4, in that order), then one `rng.uniform()` u. "
           "The logit is z = 1.2 x1 - 0.8 x2 + 0.5 x3 + 0.6 x1 x2 (x4 has weight 0), the label is 1 when u < sigmoid(z) and 0 otherwise, "
           f"and the reading a classifier sees is x plus the resting level ({OFFSET[0]:g}, {OFFSET[1]:g}, {OFFSET[2]:g}, {OFFSET[3]:g}).\n\n"
           "The first five training rows (readings rounded to 2 decimals):\n\n"
           + _table(["n", "reading 1", "reading 2", "reading 3", "reading 4", "label"], first)
           + "\n\nShare of label 1 in each part:\n\n" + _table(["part", "rows", "share of label 1"], shares)
           + "\n\nAverage of each reading over the training rows (used for centring): "
           + ", ".join(f"{m:.3f}" for m in means) + ".")
    return out


def _claim_rows(n=5):
    rows = []
    test = _dataset()[2]
    for k, s in enumerate(_run_seeds()[:n]):
        a = _accuracy(_fixed_scores("plain", s), test)
        b = _accuracy(_fixed_scores("both", s), test)
        rows.append((k, s, a, b))
    return rows


def one_seed_claims():
    rows = [(k, f"{s:#018x}", _pts(a), _pts(b), _pts(b - a, 1, True)) for k, s, a, b in _claim_rows()]
    return (f"{_generator_line(SEEDS_NAME)} The run seed of run k is the raw draw number k (`next()`, counting from 0) of this generator. "
            f"Each run trains on the {N_TRAIN} training rows with learning rate {LR:g}, {EPOCHS} epochs, one example per step, and scores the same "
            f"{N_TEST} test rows. Both pipelines use the same run seed, so they start from the same initial weights and visit the examples in the same order.\n\n"
            + _table(["run", "run seed", "plain: test accuracy %", "centring + interaction: test accuracy %", "difference (points)"], rows, "rrrrr"))


@lru_cache(maxsize=None)
def _converged_scores():
    """Test scores of logistic regression fitted to convergence on the raw readings."""
    train, _, test = _dataset()
    means = _train_means(train)
    Ftr, ytr = _featurise(train, means, "plain")
    Fte, _ = _featurise(test, means, "plain")
    w, b = _newton_logistic(Ftr, ytr, 0.0)
    return tuple(_score(w, b, f) for f in Fte)


@lru_cache(maxsize=None)
def _previous_best():
    """The strongest ordinary method in the family: all products of the centred readings, the L2 penalty chosen on validation."""
    train, val, test = _dataset()
    means = _train_means(train)
    ytr = [r[1] for r in train]
    Qtr = [_quadratic(r[0], means) for r in train]
    Qva = [_quadratic(r[0], means) for r in val]
    Qte = [_quadratic(r[0], means) for r in test]
    fits = {l: _newton_logistic(Qtr, ytr, l) for l in (0.0, 0.1, 1.0, 10.0)}
    best_l = max(fits, key=lambda l: (_accuracy([_score(*fits[l], f) for f in Qva], val), -l))
    return best_l, tuple(_score(*fits[best_l], f) for f in Qte)


@lru_cache(maxsize=None)
def _ceiling_accuracy():
    """Accuracy of the true rule (predict 1 when the true logit is not below 0) on CEILING_N fresh rows."""
    rows = _draw_rows(bookrng.generator(CEILING_NAME), CEILING_N)
    return _mean([1.0 if (r[2] >= 0) == (r[1] == 1) else 0.0 for r in rows])


@lru_cache(maxsize=None)
def _ladder_data():
    """Rows of the baseline ladder: (short label, long label, accuracy, rows scored, log loss or None)."""
    train, val, test = _dataset()
    n = len(test)
    rows = []
    ytr = [r[1] for r in train]
    share = _mean(ytr)
    majority = 1 if share > 0.5 else 0
    acc_major = _mean([1.0 if r[1] == majority else 0.0 for r in test])
    ll_const = -(sum(math.log(share) if r[1] == 1 else math.log(1 - share) for r in test)) / n
    rows.append(("majority class", "majority class (the training share of label 1 is %.3f)" % share, acc_major, n, ll_const))
    rows.append(("naive Bayes", "naive Bayes (Gaussian)", _naive_bayes_accuracy(train, test), n, None))
    ks = (1, 5, 15, 31, 61, 101)
    val_acc = _knn_accuracies(train, val, ks)
    kbest = max(ks, key=lambda k: (val_acc[k], -k))
    rows.append((f"k-NN (k = {kbest})", f"k nearest neighbours, k = {kbest} chosen on validation", _knn_accuracies(train, test, (kbest,))[kbest], n, None))
    z = _converged_scores()
    rows.append(("converged logistic regression", "logistic regression to convergence, raw readings", _accuracy(z, test), n, _log_loss(z, test)))
    for short, label, pipe in (("lab baseline (SGD)", "lab baseline: logistic regression by SGD, plain", "plain"),
                               ("candidate (SGD)", "candidate: SGD with centring + interaction", "both")):
        runs = [_fixed_scores(pipe, sd) for sd in _run_seeds()[:N_RUNS]]
        rows.append((short, label + f" (mean of {N_RUNS} seeds)", _mean([_accuracy(r, test) for r in runs]), n,
                     _mean([_log_loss(r, test) for r in runs])))
    best_l, zq = _previous_best()
    rows.append(("previous best", f"previous best: logistic regression on all products, L2 {best_l:g} chosen on validation", _accuracy(zq, test), n, _log_loss(zq, test)))
    rows.append(("true rule, these rows", "the rule that generated the labels, on these 1000 rows", _mean([1.0 if (r[2] >= 0) == (r[1] == 1) else 0.0 for r in test]), n, None))
    rows.append(("ceiling", f"the same rule on {CEILING_N:,} fresh rows (the ceiling)", _ceiling_accuracy(), CEILING_N, None))
    return rows


def baseline_ladder():
    rows = []
    for _, label, acc, rows_n, loss in _ladder_data():
        h = _wald(acc, rows_n)
        rows.append((label, _pts(acc), f"{_pts(acc - h)} to {_pts(acc + h)}", "-" if loss is None else f"{loss:.3f}"))
    return ("Every row is scored on the same 1000 test rows; the interval is the Wald interval of the accuracy (Volume I, chapter 12). "
            "Log loss is the mean negative log-likelihood in nats of the predicted probability (Volume II, chapter 4), shown where the model outputs one. "
            f"The two SGD rows average the {N_RUNS} paired seeds of the seed table (accuracy and log loss are each averaged over the seeds; the interval is the test-sample interval of the average accuracy). "
            f"The interval of the last row uses its {CEILING_N:,} rows. Convergence: Newton's method on the logistic loss, at most 30 steps, stopping when the largest "
            "change of a parameter is below 1e-10. k nearest neighbours: readings standardised by the training mean and standard deviation (divisor N), "
            "Euclidean distance, k from (1, 5, 15, 31, 61, 101), the best validation accuracy, the smaller k on a tie. The previous best uses the four centred readings "
            "and their ten products x_i x_j with i <= j, with the L2 penalty from (0, 0.1, 1, 10) that has the best validation accuracy (the smaller on a tie).\n\n"
            + _table(["method", "test accuracy %", "95% interval (accuracy %)", "log loss"], rows, "lrrr"))


def product_correlations():
    train = _dataset()[0]
    means = _train_means(train)
    cols = {
        "x1 (reading 1)": [r[0][0] for r in train],
        "x2 (reading 2)": [r[0][1] for r in train],
        "x1 x2 (product of the raw readings)": [r[0][0] * r[0][1] for r in train],
        "x1 - mean (centred reading 1)": [r[0][0] - means[0] for r in train],
        "x2 - mean (centred reading 2)": [r[0][1] - means[1] for r in train],
        "(x1 - mean)(x2 - mean) (product of the centred readings)": [(r[0][0] - means[0]) * (r[0][1] - means[1]) for r in train],
    }
    names = list(cols)
    rows = [("raw product against raw reading 1", _correlation(cols[names[2]], cols[names[0]])),
            ("raw product against raw reading 2", _correlation(cols[names[2]], cols[names[1]])),
            ("centred product against centred reading 1", _correlation(cols[names[5]], cols[names[3]])),
            ("centred product against centred reading 2", _correlation(cols[names[5]], cols[names[4]]))]
    return ("Correlation coefficient over the 200 training rows (the training means are those of the dataset table).\n\n"
            + _table(["pair of features", "correlation"], [(a, f"{b:+.3f}") for a, b in rows], "lr"))


# ----------------------------------------------------------------- unequal tuning effort

TUNE_NAME = "experiments.tuning"
TUNE_REPS = 100
TUNE_LR_BASELINE = 0.1
TUNE_LR_LOW, TUNE_LR_RANGE = 0.003, 100.0   # a try draws lr = 0.003 * 100^u
TUNE_TRIES = 16
TUNE_PROTOCOLS = ((1, 1), (1, 4), (1, 16), (4, 4), (16, 16))   # (tries for A, tries for B)


@lru_cache(maxsize=None)
def _tuning_results():
    rng = bookrng.generator(TUNE_NAME)
    out = []
    for _ in range(TUNE_REPS):
        data = (_draw_rows(rng, N_TRAIN), _draw_rows(rng, N_VAL), _draw_rows(rng, N_TEST))
        _, val, test = data

        def search(first_lr=None):
            """Returns the test accuracy of the best-on-validation try after 1, 4 and 16 tries (nested prefixes)."""
            best = None
            res = {}
            for t in range(TUNE_TRIES):
                lr = first_lr if (t == 0 and first_lr is not None) else TUNE_LR_LOW * TUNE_LR_RANGE ** rng.uniform()
                seed = rng.next_u64()
                zt, zv = _fit_scores("plain", seed, lr, EPOCHS, data)
                va = _accuracy(zv, val)
                if best is None or va > best[0]:
                    best = (va, _accuracy(zt, test))
                if t + 1 in (1, 4, 16):
                    res[t + 1] = best[1]
            return res
        a_fixed = _accuracy(_fit_scores("plain", rng.next_u64(), TUNE_LR_BASELINE, EPOCHS, data)[0], test)
        b = search()
        a = search()
        out.append((a_fixed, a, b))
    return out


def tuning_effort():
    res = _tuning_results()
    rows = []
    for ka, kb in TUNE_PROTOCOLS:
        if ka == 1:
            diffs = [b[kb] - a_fixed for a_fixed, _, b in res]
            who = f"A: its fixed recipe (lr {TUNE_LR_BASELINE:g})"
        else:
            diffs = [b[kb] - a[ka] for _, a, b in res]
            who = f"A: {ka} tries"
        m, sd, se, lo, hi = _interval(diffs)
        ahead = sum(1 for d in diffs if d >= CLAIM_POINTS / 100) / len(diffs)
        rows.append((who, f"B: {kb} tr{'y' if kb == 1 else 'ies'}", _pts(m, 2, True), f"{100 * se:.2f}", f"{100 * lo:+.2f} to {100 * hi:+.2f}", f"{ahead:.2f}"))
    base = _mean([a for a, _, _ in res])
    return (f"{_generator_line(TUNE_NAME)} {TUNE_REPS} repetitions from one generator. A repetition draws {N_TRAIN} training, {N_VAL} validation and {N_TEST} test rows "
            f"(recipe of the dataset), then one `next()` for A's fixed run (lr {TUNE_LR_BASELINE:g}, {EPOCHS} epochs), then B's {TUNE_TRIES} tries, then A's {TUNE_TRIES} tries; "
            "a try is u = `uniform()` (learning rate 0.003 x 100^u), then a run seed = `next()`. All runs use the plain pipeline, so A and B are the same method. "
            "A search keeps the try with the best validation accuracy (the earlier try on a tie) and reports its test accuracy; "
            "'k tries' means the first k tries of that search. The difference is B minus A in points of test accuracy; "
            "the interval is the 95% t interval over the repetitions; the last column is the share of repetitions in which B's reported accuracy is at least 1.2 points above A's.\n\n"
            f"Mean test accuracy of A's fixed recipe over the repetitions: {_pts(base)}%.\n\n"
            + _table(["A gets", "B gets", "mean B - A (points)", "standard error", "95% interval (points)", "share with B ahead by 1.2 or more"], rows, "llrrrr"))


# ----------------------------------------------------------------- seeds

def _spread_stats(v):
    m, s, se, lo, hi = _interval(v)
    return m, s, min(v), max(v)


def seed_spread():
    pa, pb = _seed_accuracies("plain"), _seed_accuracies("both")
    ub = _seed_accuracies("both", N_RUNS)
    rows = []
    for name, v in (("plain", pa), ("centring + interaction", pb)):
        m, s, lo, hi = _spread_stats(v)
        rows.append((name, _pts(m, 2), _pts(s, 2), _pts(lo, 1), _pts(hi, 1), _pts(hi - lo, 1)))
    paired = [b - a for a, b in zip(pa, pb)]
    unpaired = [b - a for a, b in zip(pa, ub)]
    cmp_rows = []
    for name, d in (("one run each, different seeds", unpaired), ("one run each, the same seed", paired)):
        m, s, se, lo, hi = _interval(d)
        cmp_rows.append((name, _pts(m, 2, True), _pts(s, 2),
                         f"{sum(1 for x in d if x > 0) / len(d):.2f}", f"{sum(1 for x in d if x < 0) / len(d):.2f}",
                         f"{sum(1 for x in d if x >= CLAIM_POINTS / 100) / len(d):.2f}"))
    corr = _correlation(pa, pb)
    return (f"{_generator_line(SEEDS_NAME)} Run seed k is the raw draw k of this generator (`next()`, counting from 0); the {N_RUNS} paired runs use draws 0 .. {N_RUNS - 1} "
            f"for both pipelines, the partner runs of the second comparison use draws {N_RUNS} .. {2 * N_RUNS - 1} for the candidate. Same data, same recipe "
            f"(learning rate {LR:g}, {EPOCHS} epochs); the test set is the same {N_TEST} rows every time, so every difference below comes from the seed (initial weights and data order) alone. "
            "Standard deviations use the divisor n - 1.\n\n"
            + _table(["pipeline", f"mean test accuracy % over {N_RUNS} seeds", "standard deviation (points)", "lowest %", "highest %", "range (points)"], rows, "lrrrrr")
            + f"\n\nCorrelation over the seeds between the two pipelines' accuracies at the same seed: {corr:.3f}.\n\n"
            + _table(["comparison (100 pairs)", "mean of B - A (points)", "standard deviation of B - A (points)", "share of pairs with B ahead",
                      "share with A ahead", "share with B ahead by 1.2 or more"], cmp_rows, "lrrrrr"))


def _correlation(a, b):
    ma, mb = _mean(a), _mean(b)
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return sab / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


# ----------------------------------------------------------------- effect size and significance

EFFECT_NAME = "experiments.effect_vs_n"
EFFECT_NS = (50, 200, 1000, 5000, 20000, 100000)
# probabilities of (both right, only A right, only B right, both wrong) for a question
EFFECT_SCENARIOS = (("tiny effect", (0.700, 0.030, 0.033, 0.237)), ("large effect", (0.700, 0.040, 0.100, 0.160)))


@lru_cache(maxsize=None)
def _effect_rows():
    rng = bookrng.generator(EFFECT_NAME)
    out = {}
    for name, (p11, p10, p01, p00) in EFFECT_SCENARIOS:
        rows = []
        for n in EFFECT_NS:
            c10 = c01 = 0
            for _ in range(n):
                u = rng.uniform()
                if u < p11:
                    pass
                elif u < p11 + p10:
                    c10 += 1
                elif u < p11 + p10 + p01:
                    c01 += 1
            dbar = (c01 - c10) / n
            sumsq = c01 + c10
            var = (sumsq - n * dbar * dbar) / (n - 1)
            se = math.sqrt(var / n)
            z = dbar / se if se > 0 else 0.0
            p = 2 * (1 - _phi(abs(z)))
            sd_d = math.sqrt(var)
            rows.append((n, c10, c01, dbar, se, dbar - 1.96 * se, dbar + 1.96 * se, z, p, dbar / sd_d if sd_d > 0 else 0.0))
        out[name] = rows
    return out


def effect_vs_n():
    res = _effect_rows()
    parts = [f"{_generator_line(EFFECT_NAME)} One generator, the tiny-effect scenario first and then the large-effect one; within a scenario the sizes in the order of the table, "
             "each size a fresh sample. A question is one `rng.uniform()` u: both right if u is below p11, only A right if below p11 + p10, only B right if below p11 + p10 + p01, "
             "else both wrong. D = +1 when only B is right, -1 when only A is. D-bar is the mean of D, SE is s_D / sqrt(n) (s_D with divisor n - 1), z = D-bar / SE, "
             "p is the two-sided Gaussian area outside |z|, the interval is D-bar +- 1.96 SE, and the last column is D-bar / s_D (the standardised effect)."]
    for name, (p11, p10, p01, p00) in EFFECT_SCENARIOS:
        true = p01 - p10
        var = (p01 + p10) - true * true
        dz = true / math.sqrt(var)
        need = (1.96 / dz) ** 2
        parts.append(f"\n\nScenario '{name}': p11 = {p11}, p10 = {p10}, p01 = {p01}, p00 = {p00}. True gain of B over A: {100 * true:+.2f} points; "
                     f"true standardised effect {dz:.4f}; the n at which the expected z reaches 1.96 is (1.96 / {dz:.4f})^2 = {need:,.0f}.\n\n")
        rows = [(f"{n:,}", c10, c01, _pts(d, 2, True), f"{100 * se:.2f}", f"{100 * lo:+.2f} to {100 * hi:+.2f}", f"{z:.2f}", f"{p:.4f}", f"{s:.4f}")
                for n, c10, c01, d, se, lo, hi, z, p, s in res[name]]
        parts.append(_table(["n", "only A right", "only B right", "gain D-bar (points)", "SE (points)", "95% interval (points)", "z", "p", "D-bar / s_D"],
                            rows, "rrrrrrrrr"))
    return "".join(parts)


REPEAT_NAME = "experiments.effect_repeated"
REPEAT_NS = (50, 200, 1000)
REPEAT_REPS = 300


def _paired_z(c10, c01, n):
    """The z statistic of the mean of D (+1 for only B right, -1 for only A right) over n questions, divisor n - 1."""
    dbar = (c01 - c10) / n
    var = (c01 + c10 - n * dbar * dbar) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    return dbar, (dbar / se if se > 0 else 0.0)


@lru_cache(maxsize=None)
def _repeated_rows():
    rng = bookrng.generator(REPEAT_NAME)
    out = []
    for name, (p11, p10, p01, p00) in EFFECT_SCENARIOS:
        true = p01 - p10
        dz = true / math.sqrt((p01 + p10) - true * true)
        for n in REPEAT_NS:
            gains, sig = [], []
            for _ in range(REPEAT_REPS):
                c10 = c01 = 0
                for _ in range(n):
                    u = rng.uniform()
                    if u < p11:
                        continue
                    if u < p11 + p10:
                        c10 += 1
                    elif u < p11 + p10 + p01:
                        c01 += 1
                dbar, z = _paired_z(c10, c01, n)
                gains.append(dbar)
                if abs(z) > 1.96 and dbar > 0:
                    sig.append(dbar)
            share = len(sig) / len(gains)
            out.append((name, n, dz * math.sqrt(n), share, _mean(gains), _mean(sig) if sig else None, true))
    return out


def significance_repeated():
    rows = [(name, f"{n:,}", f"{ez:.2f}", f"{share:.2f}", _pts(mg, 2, True), "-" if sg is None else _pts(sg, 2, True), _pts(true, 2, True))
            for name, n, ez, share, mg, sg, true in _repeated_rows()]
    return (f"{_generator_line(REPEAT_NAME)} The two scenarios of the previous table, the tiny effect first; within a scenario the sizes n = 50, 200, 1000 in turn, "
            f"each {REPEAT_REPS} repeated experiments in a row, an experiment being n questions drawn as before (one `rng.uniform()` each). 'Expected z' is (true standardised effect) x sqrt(n). "
            "'Significant' means z > 1.96 (a two-sided p below 0.05 with B ahead).\n\n"
            + _table(["scenario", "n", "expected z", f"share of {REPEAT_REPS} experiments significant", "mean gain over all (points)",
                      "mean gain over the significant ones (points)", "true gain (points)"], rows, "llrrrrr"))


# ----------------------------------------------------------------- ablation

def _cell_stats():
    acc = {p: _seed_accuracies(p) for p in PIPELINES}
    return acc


def _diff_stats(a, b):
    """Paired difference b - a over seeds: mean, standard error, 95% interval, all in points."""
    d = [y - x for x, y in zip(a, b)]
    m, s, se, lo, hi = _interval(d)
    return m, se, lo, hi


def ablation_table():
    acc = _cell_stats()
    rows = []
    for p in PIPELINES:
        m, s, se, lo, hi = _interval(list(acc[p]))
        rows.append((PIPELINE_LABELS[p], "yes" if p in ("centring", "both") else "no", "yes" if p in ("interaction", "both") else "no",
                     _pts(m, 2), f"{100 * se:.2f}", f"{100 * lo:.2f} to {100 * hi:.2f}"))
    p, i, c, b = (acc[k] for k in PIPELINES)
    contrasts = [
        ("add the interaction to the plain pipeline", p, i),
        ("add the interaction to the centred pipeline", c, b),
        ("add centring to the plain pipeline", p, c),
        ("add centring to the pipeline with the interaction", i, b),
        ("remove the interaction from the full pipeline", b, c),
        ("remove centring from the full pipeline", b, i),
        ("remove both from the full pipeline", b, p),
    ]
    crows = []
    for name, x, y in contrasts:
        m, se, lo, hi = _diff_stats(x, y)
        crows.append((name, _pts(m, 2, True), f"{100 * se:.2f}", f"{100 * lo:+.2f} to {100 * hi:+.2f}"))
    inter = [(bb - ii) - (cc - pp) for pp, ii, cc, bb in zip(p, i, c, b)]
    m, s, se, lo, hi = _interval(inter)
    drop_i = _mean([y - x for x, y in zip(c, b)])
    drop_c = _mean([y - x for x, y in zip(i, b)])
    total = _mean([y - x for x, y in zip(p, b)])
    return (f"The {N_RUNS} paired runs of the seed table: the four pipelines are trained with the same {N_RUNS} run seeds and scored on the same {N_TEST} test rows. "
            "Standard error = sd (divisor n - 1) / sqrt(100); intervals are 95% t intervals (99 degrees of freedom, multiplier "
            f"{_t95(99):.3f}) of the per-seed values.\n\n"
            + _table(["pipeline", "centring", "interaction", "mean test accuracy %", "standard error (points)", "95% interval (accuracy %)"], rows, "lllrrr")
            + "\n\nPaired differences over the seeds (points of accuracy):\n\n"
            + _table(["change", "mean difference (points)", "standard error", "95% interval"], crows, "lrrr")
            + f"\n\nInteraction contrast per seed, (full - centring only) - (interaction only - plain): mean {_pts(m, 2, True)} points, standard error {100 * se:.2f}, "
            f"95% interval {100 * lo:+.2f} to {100 * hi:+.2f}. Sum of the two removal losses {_pts(drop_i + drop_c, 2)} points against the loss of removing both {_pts(total, 2)} points."
            )


# ----------------------------------------------------------------- four ways of manufacturing a win

FAIL_NAME = "experiments.failures"
FAIL_REPS = 60
FAIL_RUNS = 10
FAIL_LONG_EPOCHS = 4 * EPOCHS


def _auc(scores, rows):
    pos = [z for z, r in zip(scores, rows) if r[1] == 1]
    neg = [z for z, r in zip(scores, rows) if r[1] == 0]
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def _f1(scores, rows):
    tp = sum(1 for z, r in zip(scores, rows) if z >= 0 and r[1] == 1)
    fp = sum(1 for z, r in zip(scores, rows) if z >= 0 and r[1] == 0)
    fn = sum(1 for z, r in zip(scores, rows) if z < 0 and r[1] == 1)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


@lru_cache(maxsize=None)
def _failure_results():
    rng = bookrng.generator(FAIL_NAME)
    out = []
    for _ in range(FAIL_REPS):
        data = (_draw_rows(rng, N_TRAIN), _draw_rows(rng, N_VAL), _draw_rows(rng, N_TEST))
        test = data[2]
        a_seeds = [rng.next_u64() for _ in range(FAIL_RUNS)]
        b_seeds = [rng.next_u64() for _ in range(FAIL_RUNS)]
        long_seed = rng.next_u64()
        za = [_fit_scores("plain", s, LR, EPOCHS, data)[0] for s in a_seeds]
        zb = [_fit_scores("plain", s, LR, EPOCHS, data)[0] for s in b_seeds]
        acc_a = [_accuracy(z, test) for z in za]
        acc_b = [_accuracy(z, test) for z in zb]
        zl = _fit_scores("plain", long_seed, LR, FAIL_LONG_EPOCHS, data)[0]
        out.append(dict(
            honest=acc_b[0] - acc_a[0],
            best_seed=max(acc_b) - acc_a[0],
            both_best=max(acc_b) - max(acc_a),
            metrics=max(100 * (acc_b[0] - acc_a[0]), 100 * (_f1(zb[0], test) - _f1(za[0], test)), 100 * (_auc(zb[0], test) - _auc(za[0], test))) / 100,
            long_epochs=_accuracy(zl, test) - acc_a[0],
        ))
    return out


def failure_modes():
    res = _failure_results()
    spec = (("one run each, accuracy fixed in advance", "honest"),
            ("B reports its best of 10 seeds, A its first", "best_seed"),
            ("both report their best of 10 seeds", "both_best"),
            ("B reports the best of accuracy, F1 and AUC", "metrics"),
            (f"B trains {FAIL_LONG_EPOCHS // EPOCHS} times as long ({FAIL_LONG_EPOCHS} epochs against {EPOCHS})", "long_epochs"))
    rows = []
    for name, key in spec:
        d = [r[key] for r in res]
        m, s, se, lo, hi = _interval(d)
        rows.append((name, _pts(m, 2, True), f"{100 * se:.2f}", f"{sum(1 for x in d if x >= CLAIM_POINTS / 100) / len(d):.2f}"))
    return (f"{_generator_line(FAIL_NAME)} {FAIL_REPS} repetitions from one generator. A repetition draws {N_TRAIN} training, {N_VAL} validation and {N_TEST} test rows, "
            f"then {FAIL_RUNS} run seeds for A and {FAIL_RUNS} for B (one `next()` each), then one more `next()` for the long run of the last row. A and B are the same method: the plain pipeline, learning rate {LR:g}, "
            f"{EPOCHS} epochs, except in the last row, where B uses {FAIL_LONG_EPOCHS} epochs. 'Best' means the highest test value. In the metric row the claimed gain is the largest of the three "
            "differences (accuracy, F1 of label 1 at the threshold 0, area under the ROC curve), each in points, for the first seed of each method. "
            "Standard error = sd (divisor n - 1) / sqrt(60). The last column is the share of repetitions in which the reported gain is at least 1.2 points.\n\n"
            + _table(["protocol", "mean reported gain of B over A (points)", "standard error", "share of repetitions with gain >= 1.2"], rows, "lrrr"))


# ----------------------------------------------------------------- reporting

def _seed_average_correct(pipeline):
    c = _seed_correct(pipeline)
    n = len(c[0])
    return [sum(run[i] for run in c) / len(c) for i in range(n)]


def report_card():
    test = _dataset()[2]
    conv = _correct(_converged_scores(), test)
    prev = _correct(_previous_best()[1], test)
    cand = _seed_average_correct("both")
    plain = _seed_average_correct("plain")
    seeds_plain, seeds_both = _seed_accuracies("plain"), _seed_accuracies("both")
    rows = []
    for name, per_example, per_seed in (
            ("plain pipeline (same SGD recipe)", plain, [b_ - a_ for a_, b_ in zip(seeds_plain, seeds_both)]),
            ("logistic regression to convergence, raw readings", conv, None),
            ("previous best (all products, L2 by validation)", prev, None)):
        d = [x - y for x, y in zip(cand, per_example)]
        n = len(d)
        dbar = _mean(d)
        se = _sd(d) / math.sqrt(n)
        test_part = f"{100 * (dbar - 1.96 * se):+.2f} to {100 * (dbar + 1.96 * se):+.2f}"
        if per_seed is not None:
            m, s, sse, lo, hi = _interval(per_seed)
            seed_part = f"{100 * lo:+.2f} to {100 * hi:+.2f}"
        else:
            seed_part = "-"
        rows.append((name, _pts(dbar, 2, True), seed_part, test_part))
    return ("Candidate: centring + interaction, SGD, learning rate 0.03, 3 epochs, 100 seeds (the paired runs of the seed table), test set of 1000 rows, tuned on nothing. "
            "For each baseline the per-example difference D = (candidate's share of the 100 seeds that classify the example correctly) - (the baseline's correct, 0 or 1; for the "
            "plain SGD baseline also its share of seeds). Gain = mean of D in points. 'Test-sample interval' is gain +- 1.96 SE with SE = s_D / sqrt(1000) "
            "(divisor n - 1): the uncertainty from having only 1000 test rows, for the seed-averaged models. 'Seed interval' is the 95% t interval of the "
            "per-seed gain over the 100 paired seeds: the uncertainty from the training randomness, on these 1000 rows.\n\n"
            + _table(["baseline", "gain of the candidate (points)", "seed interval (points)", "test-sample interval (points)"], rows, "lrrr"))
