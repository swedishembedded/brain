# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Generalization, Overfitting and Honest Evaluation".

Plain Python 3, standard library only, written from the chapter's own
description: polynomial fits of rising degree to the sensor source of the
learning-problem chapter (training error falls, fresh error turns up), the
inflation of a score by choosing among many models on the test set, a
train/validation/test split used to choose the strength of the penalty of the
likelihood chapter, the bias-variance decomposition computed over many
training sets, k-fold and leave-one-out cross-validation, four leaks (duplicate
rows, a statistic computed with the labels of the whole dataset, future
information, one subject on both sides of a split), covariate shift and label
shift, and learning curves. Each public function without a leading underscore
that takes no arguments returns a markdown string that the chapter embeds
between `<!-- worked: generalization.<name> -->` markers. Underscore functions
and the constants are shared with the chapter's figures, so a picture and the
table beside it cannot disagree.

Swedish Embedded AB implements solutions for evaluating machine learning
models honestly before they reach production. If your team needs expertise in
validation design, leakage audits and distribution shift, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng

# ------------------------------------------------------------------ the source

LAW = (-6.0, 50.0, -8.0)        # temperature = LAW[0] + LAW[1] v + LAW[2] v^2 + noise
NOISE_SD = 1.2
V_LO, V_HI = 0.4, 2.8
X_CENTRE, X_HALF = 1.6, 1.2     # features use x = (v - 1.6) / 1.2, which maps [0.4, 2.8] to [-1, 1]
MAX_DEGREE = 9

TRAIN_NAME = "generalization.sensor_train"
FRESH_NAME = "generalization.sensor_fresh"
N_TRAIN, N_FRESH = 20, 2000

# ------------------------------------------------------------------ experiment constants

GUESS_NAME = "generalization.test_set_looks"
GUESS_N, GUESS_REPS, GUESS_FRESH = 50, 50, 2000
GUESS_CANDIDATES = (1, 10, 100, 1000)   # the numbers C of candidates
GUESS_THRESHOLD = 33            # 33 of 50 correct is one-sided p = 0.016 for a single coin-flipper

SPLIT_NAME = "generalization.split_data"
SPLIT_SIZES = (20, 10, 10)      # training, validation, test rows
LAMBDAS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SEARCH_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 10.0)  # the repeated experiments leave out the near-unpenalised fits (see the chapter)
OPTIMISM_NAME = "generalization.selection_optimism"
OPTIMISM_REPS, OPTIMISM_FRESH = 1000, 1000

BV_DEGREE_NAME = "generalization.bias_variance"
BV_RIDGE_NAME = "generalization.bias_variance_ridge"
BV_N, BV_SETS, BV_GRID = 20, 1000, 41
BV_DEGREES = tuple(range(7))
BV_LAMBDAS = (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0, 100.0)

CV_NAME = "generalization.cv_folds"
CV_N, CV_K = 30, 5
CV_REPS_NAME = "generalization.cv_vs_split"
CV_REPS = 1000

LEAK_DUP_NAME, LEAK_DUP_ITEMS, LEAK_DUP_REPS = "generalization.leak_duplicates", 100, 200
LEAK_SEL_NAME, LEAK_SEL_ROWS, LEAK_SEL_FEATURES, LEAK_SEL_KEEP, LEAK_SEL_REPS = "generalization.leak_selection", 40, 200, 10, 100
LEAK_TIME_NAME, LEAK_TIME_STEPS, LEAK_TIME_TRAIN, LEAK_TIME_REPS = "generalization.leak_time", 300, 210, 100
LEAK_GRP_NAME, LEAK_GRP_SUBJECTS, LEAK_GRP_ROWS, LEAK_GRP_REPS = "generalization.leak_groups", 30, 8, 30
LEAK_GRP_SPREAD = 0.1

SHIFT_NAME = "generalization.shift"
SHIFT_TRAIN_N, SHIFT_HELD_N, SHIFT_PROD_N = 40, 1000, 1000
SHIFT_TRAIN_HI = 1.8
SHIFT_DEGREES = (1, 2, 5)
LABEL_SHIFT_MEANS = (0.0, 2.0)  # the two class-conditional Gaussians, both with standard deviation 1
LABEL_SHIFT_PRIORS = (0.5, 0.2, 0.1, 0.02)

CURVE_NAME = "generalization.learning_curves"
CURVE_SIZES = (20, 40, 80, 160, 320, 640)
CURVE_DEGREES = (1, 2, 6)
CURVE_SETS = 100


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


HELD_GAUSSIAN = (" A normal() call that finds a held value returns it without drawing, and a held value left at the end of "
                 "a repetition is carried into the next one (the generator is never reset).")


def _generator_line(name, normal=False):
    """The seed sentence of a block; `normal` adds the rule about the Gaussian value held by Box-Muller."""
    seed = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed})." + (HELD_GAUSSIAN if normal else "")


def _fresh_rows_phrase(count=N_FRESH):
    """How a later block names the fresh rows of the first table."""
    which = f"the {count} fresh rows" if count == N_FRESH else f"the first {count} of the {N_FRESH} fresh rows"
    return f"{which} of the first table (seed {bookrng.seed_of(FRESH_NAME):#018x})"


def _num(x):
    """Three decimals, or four significant digits in the form m x 10^e once the value reaches 10000."""
    if abs(x) < 10000:
        return f"{x:.3f}"
    m, e = f"{x:.3e}".split("e")
    return f"{m} x 10^{int(e)}"


def truth(v):
    return LAW[0] + LAW[1] * v + LAW[2] * v * v


def _draw(g, n, lo=V_LO, hi=V_HI):
    """n rows (v, y): v = lo + (hi - lo) * uniform(), then z = normal(), y = truth(v) + NOISE_SD * z."""
    rows = []
    for _ in range(n):
        v = lo + (hi - lo) * g.uniform()
        z = g.normal()
        rows.append((v, truth(v) + NOISE_SD * z))
    return rows


def _features(v, d=MAX_DEGREE):
    x = (v - X_CENTRE) / X_HALF
    out, p = [], 1.0
    for _ in range(d + 1):
        out.append(p)
        p *= x
    return out


def _solve(a, b):
    """A w = b by Gaussian elimination with partial pivoting."""
    n = len(a)
    m = [list(r) + [bb] for r, bb in zip(a, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        for r in range(c + 1, n):
            k = m[r][c] / m[c][c]
            for j in range(c, n + 1):
                m[r][j] -= k * m[c][j]
    w = [0.0] * n
    for i in range(n - 1, -1, -1):
        w[i] = (m[i][n] - sum(m[i][j] * w[j] for j in range(i + 1, n))) / m[i][i]
    return w


def _gram(feats, ys):
    """X^T X and X^T y for the full feature list (all MAX_DEGREE + 1 columns)."""
    d = len(feats[0])
    a = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for row, y in zip(feats, ys):
        for i in range(d):
            ri = row[i]
            b[i] += ri * y
            ai = a[i]
            for j in range(i, d):
                ai[j] += ri * row[j]
    for i in range(d):
        for j in range(i):
            a[i][j] = a[j][i]
    return a, b


def _fit_gram(a, b, degree, lam=0.0):
    """Minimiser of 1/2 sum (y - w.x)^2 + lam/2 ||w||^2 over the columns 1, x, ..., x^degree."""
    n = degree + 1
    return _solve([[a[i][j] + (lam if i == j else 0.0) for j in range(n)] for i in range(n)], b[:n])


def _fit(rows, degree, lam=0.0):
    a, b = _gram([_features(v) for v, _ in rows], [y for _, y in rows])
    return _fit_gram(a, b, degree, lam)


def _predict(w, feat):
    return sum(wj * fj for wj, fj in zip(w, feat))


def _mse(w, feats, ys):
    return sum((_predict(w, f) - y) ** 2 for f, y in zip(feats, ys)) / len(ys)


def _rmse_rows(w, rows):
    return math.sqrt(_mse(w, [_features(v) for v, _ in rows], [y for _, y in rows]))


def _train_rows():
    return _draw(bookrng.generator(TRAIN_NAME), N_TRAIN)


def _fresh_rows():
    return _draw(bookrng.generator(FRESH_NAME), N_FRESH)


def _degree_sweep_data():
    rows, fresh = _train_rows(), _fresh_rows()
    out = []
    a, b = _gram([_features(v) for v, _ in rows], [y for _, y in rows])
    ff = [_features(v) for v, _ in fresh]
    fy = [y for _, y in fresh]
    tf = [_features(v) for v, _ in rows]
    ty = [y for _, y in rows]
    for d in range(MAX_DEGREE + 1):
        w = _fit_gram(a, b, d)
        out.append((d, math.sqrt(_mse(w, tf, ty)), math.sqrt(_mse(w, ff, fy))))
    return out


def _split_data():
    rows = _draw(bookrng.generator(SPLIT_NAME), sum(SPLIT_SIZES))
    n1, n2, _ = SPLIT_SIZES
    return rows[:n1], rows[n1:n1 + n2], rows[n1 + n2:]


def _grid(lo, hi, steps):
    return [lo + (hi - lo) * i / steps for i in range(steps + 1)]


# ------------------------------------------------------------------ the source and the first U-curve

def sensor_source():
    """The sensor source as the previous chapters wrote it, with this chapter's seeds and the 20 training rows."""
    rows = _train_rows()
    head = (
        f"The source: voltage v uniform on [{V_LO}, {V_HI}], temperature y = {LAW[0]:g} + {LAW[1]:g} v - {-LAW[2]:g} v^2 + "
        f"eps, where the noise is eps = {NOISE_SD} * (a standard Gaussian sample). A row is two draws in this order: "
        f"v = {V_LO} + {V_HI - V_LO:.1f} * rng.uniform(), then eps = {NOISE_SD} * rng.normal(); v and y are kept unrounded. "
        f"Each dataset is drawn from its own generator (seeds below), rows in turn, and rng.normal() follows the rule of the "
        f"randomness chapter: a call that finds a held value returns it without drawing. Every model of this chapter uses the "
        f"feature x = (v - {X_CENTRE}) / {X_HALF}, which maps the voltage range to [-1, 1], and the columns 1, x, x^2, ..., "
        f"x^d. Wherever a later example scores on 'fresh' rows, these are the {N_FRESH} fresh rows of the second line of the "
        f"table below, in order.\n\n"
        + _table(["dataset", "generator seed", "rows", "use"],
                 [("training rows", f"{bookrng.seed_of(TRAIN_NAME):#018x}", N_TRAIN, "the rows every model below is fitted to"),
                  ("fresh rows", f"{bookrng.seed_of(FRESH_NAME):#018x}", N_FRESH, "scoring only, never used to choose anything")],
                 "llrl") + "\n\n")
    body = _table(["n", "voltage v (V)", "temperature y (deg C)"],
                  [(i + 1, f"{v:.4f}", f"{y:.3f}") for i, (v, y) in enumerate(rows)])
    return head + body


def degree_sweep():
    """Polynomial fits of degree 0 to 9 to the 20 training rows: training error against fresh error."""
    rows = []
    for d, tr, fr in _degree_sweep_data():
        rows.append((d, d + 1, f"{tr:.3f}", f"{fr:.3f}"))
    head = (f"Least squares fits of degree d to the {N_TRAIN} training rows (the normal equations X^T X w = X^T y, "
            f"solved by Gaussian elimination with partial pivoting, solve(A, y, true), in the features x of the previous "
            f"table). RMSE = square root of the mean squared difference between "
            f"prediction and label, in degrees Celsius, on the {N_TRAIN} training rows and on the {N_FRESH} fresh rows "
            f"(labels included, so even a perfect model scores about {NOISE_SD}).\n\n")
    return head + _table(["degree d", "parameters", "training RMSE", "fresh RMSE"], rows)


# ------------------------------------------------------------------ the test set must be touched once

def _binom_cdf(n):
    pmf = [math.comb(n, j) / 2 ** n for j in range(n + 1)]
    cdf, acc = [], 0.0
    for p in pmf:
        acc += p
        cdf.append(acc)
    return cdf


def look_at_test_set():
    """Candidates that only guess, evaluated on a test set of 50 questions: the best of C looks better as C grows."""
    n = GUESS_N
    cdf = _binom_cdf(n)
    rows = []
    g = bookrng.generator(GUESS_NAME)
    for c in GUESS_CANDIDATES:
        best_sum, fresh_sum = 0.0, 0.0
        for _ in range(GUESS_REPS):
            best = -1
            for _ in range(c):
                correct = sum(1 for _ in range(n) if g.uniform() < 0.5)
                best = max(best, correct)
            best_sum += best / n
            fresh_sum += sum(1 for _ in range(GUESS_FRESH) if g.uniform() < 0.5) / GUESS_FRESH
        exact = sum(m / n * (cdf[m] ** c - (cdf[m - 1] if m else 0.0) ** c) for m in range(n + 1))
        p_hit = 1.0 - cdf[GUESS_THRESHOLD - 1] ** c
        rows.append((c, f"{best_sum / GUESS_REPS:.3f}", f"{exact:.3f}", f"{p_hit:.3f}", f"{fresh_sum / GUESS_REPS:.3f}"))
    single = 1.0 - cdf[GUESS_THRESHOLD - 1]
    head = (f"{_generator_line(GUESS_NAME)}\nEach candidate answers each of {n} test questions correctly with probability 0.5, "
            f"independently of every other answer and candidate: one rng.uniform() per answer, correct when it is below 0.5. "
            f"One generator serves the whole table and it is never reset. Draw order: for each C in {GUESS_CANDIDATES} in turn "
            f"(all repetitions of one C before the next C), {GUESS_REPS} repetitions in turn; one repetition draws, in this "
            f"order, the {n} answers of candidate 1, then those of candidate 2, and so on to candidate C, then {GUESS_FRESH} "
            f"more answers of the chosen candidate on fresh questions. The chosen candidate is the one with the most correct "
            f"answers (the first on a tie). The two simulated columns are means over the {GUESS_REPS} repetitions. "
            f"'Exact' is E[best]/{n}, and 'P(best >= {GUESS_THRESHOLD}/{n})' is 1 - F({GUESS_THRESHOLD - 1})^C, where F(m) = "
            f"sum over j = 0 .. m of ({n} choose j) / 2^{n} is the cumulative distribution function of the number of correct answers "
            f"of one candidate; the best of C independent candidates is at most m with probability F(m)^C, so "
            f"E[best] = sum over m of m (F(m)^C - F(m-1)^C) with F(-1) = 0 (one candidate alone reaches {GUESS_THRESHOLD} with "
            f"probability {single:.4f}).\n\n")
    return head + _table(["candidates C", "best test accuracy (simulated)", "best test accuracy (exact)",
                          f"P(best >= {GUESS_THRESHOLD}/{n})", "accuracy of the chosen one on fresh questions"], rows)


# ------------------------------------------------------------------ train, validation, test

def three_way_split():
    """Degree-9 ridge fits on 20 rows for a grid of penalties, scored on training, validation, test and fresh rows."""
    tr, va, te = _split_data()
    fresh = _fresh_rows()
    a, b = _gram([_features(v) for v, _ in tr], [y for _, y in tr])
    sets = [([_features(v) for v, _ in s], [y for _, y in s]) for s in (tr, va, te, fresh)]
    rows = []
    for lam in LAMBDAS:
        w = _fit_gram(a, b, MAX_DEGREE, lam)
        scores = [math.sqrt(_mse(w, f, y)) for f, y in sets]
        rows.append((f"{lam:g}", *[f"{s:.3f}" for s in scores]))
    head = (f"{_generator_line(SPLIT_NAME, True)}\n{sum(SPLIT_SIZES)} rows of the source in one generator, two draws per row; rows 1 to "
            f"{SPLIT_SIZES[0]} are the training set, the next {SPLIT_SIZES[1]} the validation set, the last {SPLIT_SIZES[2]} the "
            f"test set. Degree 9 (ten columns), minimise 1/2 sum (y - w.x)^2 + lambda/2 ||w||^2 on the training rows "
            f"(every weight penalised, the intercept included, as in the likelihood chapter): the normal equations "
            f"(X^T X + lambda I) w = X^T y, solved with solve(A, y, true). RMSE in degrees Celsius; fresh = "
            f"{_fresh_rows_phrase()}.\n\n")
    return head + _table(["lambda", "training RMSE", "validation RMSE", "test RMSE", "fresh RMSE"], rows)


def selection_optimism():
    """Repeating the split of the previous example: what each way of reporting a score claims, against the truth."""
    fresh = _fresh_rows()[:OPTIMISM_FRESH]
    ff = [_features(v) for v, _ in fresh]
    fy = [y for _, y in fresh]
    g = bookrng.generator(OPTIMISM_NAME)
    n1, n2, n3 = SPLIT_SIZES
    gaps = {key: [] for key in ("train", "val", "test", "test_pick")}
    sums = {key: [0.0, 0.0] for key in gaps}
    for _ in range(OPTIMISM_REPS):
        rows = _draw(g, n1 + n2 + n3)
        tr, va, te = rows[:n1], rows[n1:n1 + n2], rows[n1 + n2:]
        a, b = _gram([_features(v) for v, _ in tr], [y for _, y in tr])
        sets = [([_features(v) for v, _ in s], [y for _, y in s]) for s in (tr, va, te)]
        table = []
        for lam in SEARCH_LAMBDAS:
            w = _fit_gram(a, b, MAX_DEGREE, lam)
            table.append(([_mse(w, f, y) for f, y in sets], _mse(w, ff, fy)))
        pick_train = min(range(len(table)), key=lambda i: table[i][0][0])
        pick_val = min(range(len(table)), key=lambda i: table[i][0][1])
        pick_test = min(range(len(table)), key=lambda i: table[i][0][2])
        for key, pick, col in (("train", pick_train, 0), ("val", pick_val, 1), ("test", pick_val, 2), ("test_pick", pick_test, 2)):
            sums[key][0] += table[pick][0][col]
            sums[key][1] += table[pick][1]
            gaps[key].append(table[pick][0][col] - table[pick][1])
    labels = {"train": ("training MSE", "training MSE"),
              "val": ("validation MSE", "validation MSE"),
              "test": ("validation MSE", "test MSE, read once"),
              "test_pick": ("test MSE", "test MSE, read for every lambda")}
    out = []
    for key in ("train", "val", "test", "test_pick"):
        rep, fr = sums[key][0] / OPTIMISM_REPS, sums[key][1] / OPTIMISM_REPS
        se = _sd(gaps[key]) / math.sqrt(OPTIMISM_REPS)
        out.append((labels[key][0], labels[key][1], f"{rep:.3f}", f"{fr:.3f}", f"{rep - fr:+.3f}", f"{se:.3f}"))
    head = (f"{_generator_line(OPTIMISM_NAME, True)}\n{OPTIMISM_REPS} repetitions of the split of the previous example "
            f"({n1} training, {n2} validation, {n3} test rows; each repetition draws its {n1 + n2 + n3} rows in turn from "
            f"the one generator, two draws per row). Candidates: degree 9, lambda in {SEARCH_LAMBDAS}. Squared errors "
            f"(MSE) are averaged over the repetitions; 'true' is the MSE of the chosen fit on "
            f"{_fresh_rows_phrase(OPTIMISM_FRESH)}. Each row picks the lambda with the smallest value of the score named in its "
            f"first column (the first on a tie), fits it to that repetition's {n1} training rows, and reports the score named in "
            f"its second column. The last column is the standard error of the mean of (reported - true) over the "
            f"repetitions: the sample standard deviation (divisor n - 1) of the {OPTIMISM_REPS} values of (reported - true), "
            f"divided by sqrt({OPTIMISM_REPS}).\n\n")
    return head + _table(["lambda chosen by", "score reported", "mean reported", "mean true (fresh) MSE of the chosen fit",
                          "reported - true", "standard error"], out, "llrrrr")


# ------------------------------------------------------------------ bias and variance

def _bv_stats(preds, truth_vals, noise_draws=None):
    """preds[i][s]: prediction at grid point i by the fit to training set s. Returns (bias2, variance, identity gap)."""
    sets = len(preds[0])
    b2 = var = ident = 0.0
    for i, tv in enumerate(truth_vals):
        p = preds[i]
        m = sum(p) / sets
        b2 += (m - tv) ** 2
        var += sum((q - m) ** 2 for q in p) / sets
        ident += sum((q - tv) ** 2 for q in p) / sets
    k = len(truth_vals)
    return b2 / k, var / k, ident / k


def _bv_simulation(name, configs):
    """configs: list of (degree, lambda). One pass over BV_SETS training sets; returns per config
    (bias2, variance, identity gap, measured MSE on fresh noisy labels)."""
    g = bookrng.generator(name)
    grid = [V_LO + (V_HI - V_LO) * i / (BV_GRID - 1) for i in range(BV_GRID)]
    gf = [_features(v) for v in grid]
    tv = [truth(v) for v in grid]
    preds = [[[] for _ in grid] for _ in configs]
    meas = [0.0] * len(configs)
    for _ in range(BV_SETS):
        rows = _draw(g, BV_N)
        a, b = _gram([_features(v) for v, _ in rows], [y for _, y in rows])
        labels = [t + NOISE_SD * g.normal() for t in tv]
        for c, (d, lam) in enumerate(configs):
            w = _fit_gram(a, b, d, lam)
            for i, f in enumerate(gf):
                p = _predict(w, f)
                preds[c][i].append(p)
                meas[c] += (p - labels[i]) ** 2
    out = []
    for c in range(len(configs)):
        b2, var, ident = _bv_stats(preds[c], tv)
        out.append((b2, var, ident, meas[c] / (BV_SETS * BV_GRID)))
    return out


def _bv_head(name, what):
    return (f"{_generator_line(name, True)}\n{BV_SETS} training sets of {BV_N} rows. Draw order: for each training set in turn, "
            f"{BV_N} rows (two draws each, as in the source), then {BV_GRID} draws of normal() e_i, i = 0 .. {BV_GRID - 1}, "
            f"which make the label of one fresh row at each grid point, truth(v_i) + {NOISE_SD} * e_i. The {BV_GRID} grid "
            f"points are v_i = {V_LO} + 2.4 i / {BV_GRID - 1}, i = 0 .. {BV_GRID - 1}. {what} Every training set is fitted at "
            f"every setting of the table before the next set is drawn, so all rows of the table see the same {BV_SETS} "
            f"training sets. Bias^2 is the grid average of (mean prediction - true value)^2, where the mean is over the "
            f"{BV_SETS} fits; variance is the grid average of the population variance (divisor {BV_SETS}, not {BV_SETS - 1}) "
            f"of the predictions; noise = {NOISE_SD}^2 = {NOISE_SD ** 2:.2f}. 'Measured MSE on fresh labels' is the mean, "
            f"over sets and grid points, of (prediction - fresh label)^2.\n\n")


def bias_variance_degree():
    """Bias squared, variance and total expected error of polynomial fits by degree, over many training sets."""
    stats = _bv_simulation(BV_DEGREE_NAME, [(d, 0.0) for d in BV_DEGREES])
    rows = []
    worst = 0.0
    for d, (b2, var, ident, meas) in zip(BV_DEGREES, stats):
        worst = max(worst, abs(ident - (b2 + var)))
        rows.append((d, d + 1, _num(b2), _num(var), f"{NOISE_SD ** 2:.3f}", _num(b2 + var + NOISE_SD ** 2), _num(meas)))
    head = _bv_head(BV_DEGREE_NAME, "Least squares fits of degree d.")
    tail = (f"\n\nThe mean over sets and grid of (prediction - true value)^2 equals bias^2 + variance in every row to within "
            f"10^-9 (the identity is exact; any gap is rounding): {'yes' if worst < 1e-9 else 'NO'}.")
    return head + _table(["degree d", "parameters", "bias^2", "variance", "noise", "bias^2 + variance + noise",
                          "measured MSE on fresh labels"], rows) + tail


def bias_variance_ridge():
    """The same decomposition for a fixed degree-9 model as the penalty lambda grows."""
    stats = _bv_simulation(BV_RIDGE_NAME, [(MAX_DEGREE, lam) for lam in BV_LAMBDAS])
    rows = []
    for lam, (b2, var, ident, meas) in zip(BV_LAMBDAS, stats):
        rows.append((f"{lam:g}", _num(b2), _num(var), _num(b2 + var + NOISE_SD ** 2), _num(meas)))
    head = _bv_head(BV_RIDGE_NAME, "Degree 9 fits minimising 1/2 sum (y - w.x)^2 + lambda/2 ||w||^2.")
    return head + _table(["lambda", "bias^2", "variance", "bias^2 + variance + noise", "measured MSE on fresh labels"], rows)


# ------------------------------------------------------------------ cross-validation

def _fold_assignment(g, n, k):
    """Shuffle 0 .. n-1 with the book's shuffle; the row at shuffled position p goes to fold p mod k."""
    order = list(range(n))
    g.shuffle(order)
    fold = [0] * n
    for p, row in enumerate(order):
        fold[row] = p % k
    return fold


def cv_folds():
    """The fold assignment of the cross-validation example: which rows belong to which fold."""
    fold = _fold_assignment(bookrng.generator(CV_NAME), CV_N, CV_K)
    rows = [(j + 1, ", ".join(str(i + 1) for i in range(CV_N) if fold[i] == j)) for j in range(CV_K)]
    head = (f"{_generator_line(CV_NAME)}\nThe order 0, 1, ..., {CV_N - 1} is shuffled by the book's shuffle (Fisher-Yates, "
            f"{CV_N - 1} integer draws); the row at shuffled position p goes to fold p mod {CV_K}. Rows are numbered from 1 "
            f"in the table.\n\n")
    return head + _table(["fold", f"rows (of the {CV_N} development rows)"], rows, "rl")


def _folds_of_dev():
    return _fold_assignment(bookrng.generator(CV_NAME), CV_N, CV_K)


def _cv_scores(dev, fold, k, lams):
    """Per-fold MSE for each lambda: fit degree 9 on the other folds, score on the held-out fold."""
    feats = [_features(v) for v, _ in dev]
    ys = [y for _, y in dev]
    out = {lam: [] for lam in lams}
    for j in range(k):
        tr = [i for i in range(len(dev)) if fold[i] != j]
        va = [i for i in range(len(dev)) if fold[i] == j]
        a, b = _gram([feats[i] for i in tr], [ys[i] for i in tr])
        for lam in lams:
            w = _fit_gram(a, b, MAX_DEGREE, lam)
            out[lam].append(_mse(w, [feats[i] for i in va], [ys[i] for i in va]))
    return out


def cross_validation():
    """5-fold and leave-one-out cross-validation of the penalty on 30 development rows."""
    tr, va, te = _split_data()
    dev = tr + va
    fold = _folds_of_dev()
    scores = _cv_scores(dev, fold, CV_K, LAMBDAS)
    loo = _cv_scores(dev, list(range(CV_N)), CV_N, LAMBDAS)
    fresh = _fresh_rows()
    ff = [_features(v) for v, _ in fresh]
    fy = [y for _, y in fresh]
    a, b = _gram([_features(v) for v, _ in dev], [y for _, y in dev])
    rows = []
    for lam in LAMBDAS:
        s = scores[lam]
        cv = sum(s) / CV_K
        loo_mean = sum(loo[lam]) / CV_N
        w = _fit_gram(a, b, MAX_DEGREE, lam)
        rows.append((f"{lam:g}", *[_num(x) for x in s], _num(cv), _num(loo_mean), _num(_mse(w, ff, fy))))
    best_cv = min(LAMBDAS, key=lambda lam: sum(scores[lam]) / CV_K)
    best_loo = min(LAMBDAS, key=lambda lam: sum(loo[lam]) / CV_N)
    head = (f"The development rows are the {SPLIT_SIZES[0]} training and {SPLIT_SIZES[1]} validation rows of the earlier "
            f"split ({CV_N} rows), folds as in the previous table. For each fold: fit degree 9 with the penalty lambda to the "
            f"other {CV_N - CV_N // CV_K} rows (the normal equations (X^T X + lambda I) w = X^T y of those rows), take the MSE on "
            f"the {CV_N // CV_K} held-out rows. 'CV' is the mean of the "
            f"{CV_K} fold MSEs; 'LOO' is leave-one-out ({CV_N} fits, one row held out each time, mean squared error over the "
            f"rows; fold j of the leave-one-out run is row j alone); 'fresh' is the MSE on {_fresh_rows_phrase()} of the fit to all "
            f"{CV_N} rows. The rows are the first {CV_N} of the {sum(SPLIT_SIZES)} rows of the three-way example (seed "
            f"{bookrng.seed_of(SPLIT_NAME):#018x}).\n\n")
    tail = f"\n\nSmallest CV: lambda = {best_cv:g}. Smallest LOO: lambda = {best_loo:g}."
    return head + _table(["lambda", *[f"fold {j + 1}" for j in range(CV_K)], "CV", "LOO", "fresh (all 30 rows)"], rows) + tail


def cv_vs_split():
    """Repeating the choice of lambda with a single split and with 5-fold cross-validation on many development sets."""
    fresh = _fresh_rows()[:OPTIMISM_FRESH]
    ff = [_features(v) for v, _ in fresh]
    fy = [y for _, y in fresh]
    g = bookrng.generator(CV_REPS_NAME)
    res = {"split": [], "cv": []}
    for _ in range(CV_REPS):
        dev = _draw(g, CV_N)
        feats = [_features(v) for v, _ in dev]
        ys = [y for _, y in dev]
        # single split: first 20 rows fit, last 10 score
        a, b = _gram(feats[:20], ys[:20])
        split_scores = [_mse(_fit_gram(a, b, MAX_DEGREE, lam), feats[20:], ys[20:]) for lam in SEARCH_LAMBDAS]
        # 5-fold with a fresh shuffle from the same generator
        fold = _fold_assignment(g, CV_N, CV_K)
        cv_all = _cv_scores(dev, fold, CV_K, SEARCH_LAMBDAS)
        cv_scores = [sum(cv_all[lam]) / CV_K for lam in SEARCH_LAMBDAS]
        a_all, b_all = _gram(feats, ys)
        for key, sc in (("split", split_scores), ("cv", cv_scores)):
            i = min(range(len(sc)), key=lambda t: sc[t])
            w = _fit_gram(a_all, b_all, MAX_DEGREE, SEARCH_LAMBDAS[i])
            res[key].append((sc[i], _mse(w, ff, fy)))
    rows = []
    for key, label, fits in (("split", "single split, 20 fit + 10 score", len(SEARCH_LAMBDAS) + 1),
                             ("cv", f"{CV_K}-fold cross-validation", CV_K * len(SEARCH_LAMBDAS) + 1)):
        est = [e for e, _ in res[key]]
        tru = [t for _, t in res[key]]
        gap = [e - t for e, t in res[key]]
        mean = lambda x: sum(x) / len(x)
        sd = lambda x: math.sqrt(sum((v - mean(x)) ** 2 for v in x) / (len(x) - 1))
        median = sorted(tru)[len(tru) // 2 - 1:len(tru) // 2 + 1]
        rows.append((label, fits, f"{mean(est):.3f}", f"{mean(tru):.3f}", f"{sum(median) / 2:.3f}",
                     f"{sd(gap):.3f}"))
    head = (f"{_generator_line(CV_REPS_NAME, True)}\n{CV_REPS} development sets of {CV_N} rows from the one generator. Draw "
            f"order: for each set in turn, its {CV_N} rows (two draws each), then the shuffle of the 5-fold assignment "
            f"({CV_N - 1} integer draws, folds as in the earlier fold example). The single split fits the first 20 rows and "
            f"scores the last 10 (it draws nothing). Both methods choose lambda from {SEARCH_LAMBDAS} (degree 9) as the "
            f"smallest score (the first on a tie); the chosen lambda is then refitted on all {CV_N} rows, which is the "
            f"last fit counted in the second column. 'estimate' is the score the method reports for its choice (MSE on the "
            f"10 scored rows, or the mean of the 5 fold MSEs), 'true' the MSE of the refitted model on "
            f"{_fresh_rows_phrase(OPTIMISM_FRESH)}. The median is the mean of the two middle values of the {CV_REPS} true "
            f"MSEs (the 500th and 501st in ascending order). The last column is the sample standard deviation (divisor "
            f"n - 1) over the sets of (estimate - true).\n\n")
    return head + _table(["method", "fits per set, refit included", "mean estimate", "mean true MSE", "median true MSE", "sd of (estimate - true)"],
                         rows, "lrrrrr")


# ------------------------------------------------------------------ leaks

def _mean(x):
    return sum(x) / len(x)


def _sd(x):
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def leak_duplicates():
    """The same record on both sides of a split: a lookup model scores far above chance on noise."""
    g = bookrng.generator(LEAK_DUP_NAME)
    n = LEAK_DUP_ITEMS
    leaky, clean = [], []
    for _ in range(LEAK_DUP_REPS):
        items = [(g.uniform(), 1 if g.uniform() < 0.5 else 0) for _ in range(n)]
        records = [it for it in items for _ in range(2)]   # adjacent copies: record 2i and 2i+1 are item i
        g.shuffle(records)
        leaky.append(_nn_accuracy(records[:n], records[n:]))
        order = list(range(n))
        g.shuffle(order)
        train = [items[i] for i in order[:n // 2] for _ in range(2)]
        test = [items[i] for i in order[n // 2:] for _ in range(2)]
        clean.append(_nn_accuracy(train, test))
    expect = 0.5 + 0.5 * n / (2 * n - 1)
    head = (f"{_generator_line(LEAK_DUP_NAME)}\n{n} items, each a feature x = rng.uniform() followed by a label 1 if "
            f"rng.uniform() < 0.5 else 0 (item 0 first), so the label is a coin flip unrelated to x. Each item is recorded twice: "
            f"the list of {2 * n} records is item 0, item 0, item 1, item 1, ..., item {n - 1}, item {n - 1} (the two copies "
            f"of an item are adjacent, record 2i and record 2i + 1 are item i, positions numbered from 0). "
            f"{LEAK_DUP_REPS} repetitions; per repetition the draws are the {n} items in turn, then the shuffle of that "
            f"list of {2 * n} records (leaky protocol: the first {n} shuffled records train, the other {n} test), then, "
            f"starting again from the list 0, 1, ..., {n - 1} (not from any earlier shuffle), the shuffle of the {n} item "
            f"numbers (clean protocol: the first {n // 2} shuffled items, both copies each and in shuffled order, train; "
            f"the other {n // 2} test). The model answers a test record with the label of the training record whose x is "
            f"nearest (the first in training-list order on a tie). Accuracy is averaged over the test records, and sd "
            f"is the sample standard deviation (divisor n - 1) over the {LEAK_DUP_REPS} repetitions.\n\n")
    tail = f"\n\nExpected accuracy of the leaky protocol: 1/2 + 1/2 * {n}/{2 * n - 1} = {expect:.4f}; of the clean protocol: 0.5."
    return head + _table(["protocol", "mean accuracy", "sd over repetitions"],
                         [("random split of the records", f"{_mean(leaky):.3f}", f"{_sd(leaky):.3f}"),
                          ("split by item", f"{_mean(clean):.3f}", f"{_sd(clean):.3f}")], "lrr") + tail


def _nn_accuracy(train, test):
    hits = 0
    for x, y in test:
        best = min(range(len(train)), key=lambda i: abs(train[i][0] - x))
        hits += 1 if train[best][1] == y else 0
    return hits / len(test)


def _correlation_pick(x, y, keep):
    """Indices of the `keep` features with the largest |mean of class 1 - mean of class 0| (first on a tie)."""
    d = len(x[0])
    ones = [r for r, t in zip(x, y) if t == 1]
    zeros = [r for r, t in zip(x, y) if t == 0]
    score = [abs(_mean([r[j] for r in ones]) - _mean([r[j] for r in zeros])) for j in range(d)]
    return sorted(range(d), key=lambda j: (-score[j], j))[:keep]


def _centroid_accuracy(xtr, ytr, xte, yte, cols):
    c1 = [_mean([r[j] for r, t in zip(xtr, ytr) if t == 1]) for j in cols]
    c0 = [_mean([r[j] for r, t in zip(xtr, ytr) if t == 0]) for j in cols]
    hits = 0
    for r, t in zip(xte, yte):
        d1 = sum((r[j] - c) ** 2 for j, c in zip(cols, c1))
        d0 = sum((r[j] - c) ** 2 for j, c in zip(cols, c0))
        pred = 1 if d1 < d0 else 0
        hits += 1 if pred == t else 0
    return hits / len(yte)


def leak_selection():
    """Choosing features with the labels of the whole dataset before splitting: noise looks predictive."""
    g = bookrng.generator(LEAK_SEL_NAME)
    n, d, keep = LEAK_SEL_ROWS, LEAK_SEL_FEATURES, LEAK_SEL_KEEP
    y = [i % 2 for i in range(n)]
    ntr = n // 2
    leaky, clean = [], []
    for _ in range(LEAK_SEL_REPS):
        x = [[g.normal() for _ in range(d)] for _ in range(n)]
        cols = _correlation_pick(x, y, keep)
        leaky.append(_centroid_accuracy(x[:ntr], y[:ntr], x[ntr:], y[ntr:], cols))
        cols = _correlation_pick(x[:ntr], y[:ntr], keep)
        clean.append(_centroid_accuracy(x[:ntr], y[:ntr], x[ntr:], y[ntr:], cols))
    head = (f"{_generator_line(LEAK_SEL_NAME, True)}\n{n} rows, {d} features, every feature value an independent normal(): there is "
            f"nothing to learn. Row i has label i mod 2 (labels 0, 1, 0, 1, ...), so each half has {ntr // 2} of each class. "
            f"{LEAK_SEL_REPS} repetitions; each draws the {n} x {d} matrix row by row. Rows 1 to {ntr} train, rows {ntr + 1} to "
            f"{n} test. The model keeps the {keep} features with the largest |mean of class 1 - mean of class 0| (means over the "
            f"rows used for choosing; ties go to the lower feature number), then classifies a test row by the nearer of the two "
            f"class centroids, each the mean of the {ntr} training rows' kept features of its class (squared distance over the "
            f"kept features, class 0 on a tie). Leaky protocol: the features are chosen from all {n} rows; clean protocol: "
            f"from the {ntr} training rows only; the centroids use the training rows in both. The means are printed with four "
            f"decimals because the clean mean is an exact multiple of 1/{LEAK_SEL_REPS * ntr}, and sd is the sample "
            f"standard deviation (divisor n - 1) over the {LEAK_SEL_REPS} repetitions.\n\n")
    return head + _table(["where the features were chosen", "mean test accuracy", "sd over repetitions"],
                         [(f"all {n} rows, labels included", f"{_mean(leaky):.4f}", f"{_sd(leaky):.4f}"),
                          (f"the {ntr} training rows only", f"{_mean(clean):.4f}", f"{_sd(clean):.4f}")], "lrr")


def leak_time():
    """A random walk predicted by the nearest training time: a random split flatters it, a chronological split does not."""
    g = bookrng.generator(LEAK_TIME_NAME)
    n, ntr = LEAK_TIME_STEPS, LEAK_TIME_TRAIN
    leaky, clean = [], []
    for _ in range(LEAK_TIME_REPS):
        s, level = [], 0.0
        for _ in range(n):
            level += g.normal()
            s.append(level)
        order = list(range(n))
        g.shuffle(order)
        for train_t, test_t, out in ((sorted(order[:ntr]), sorted(order[ntr:]), leaky),
                                     (list(range(ntr)), list(range(ntr, n)), clean)):
            err = 0.0
            for t in test_t:
                nearest = min(train_t, key=lambda u: (abs(u - t), u))
                err += (s[nearest] - s[t]) ** 2
            out.append(math.sqrt(err / len(test_t)))
    head = (f"{_generator_line(LEAK_TIME_NAME, True)}\n{LEAK_TIME_REPS} random walks of {n} steps: s_0 = normal(), then "
            f"s_t = s_(t-1) + normal(), t = 1 .. {n - 1}. Per walk the draws are the {n} steps, then the shuffle of the times "
            f"0 .. {n - 1} (random protocol: the first {ntr} shuffled times train, the other {n - ntr} test). Chronological "
            f"protocol: times 0 .. {ntr - 1} train, {ntr} .. {n - 1} test. The model predicts s_t by the value at the nearest "
            f"training time (the earlier one on a tie). RMSE over the test times of one walk; the table gives the mean and the sample "
            f"standard deviation (divisor n - 1) over the {LEAK_TIME_REPS} walks.\n\n")
    return head + _table(["protocol", "mean RMSE", "sd over walks"],
                         [("random split of the times", f"{_mean(leaky):.3f}", f"{_sd(leaky):.3f}"),
                          ("chronological split", f"{_mean(clean):.3f}", f"{_sd(clean):.3f}")], "lrr")


def leak_groups():
    """Subjects with an identifying fingerprint: splitting rows lets the model recognise the subject, not the pattern."""
    g = bookrng.generator(LEAK_GRP_NAME)
    ns, nr = LEAK_GRP_SUBJECTS, LEAK_GRP_ROWS
    row_acc, grp_acc = [], []
    for _ in range(LEAK_GRP_REPS):
        rows, subject = [], []
        for s in range(ns):
            fp = (g.normal(), g.normal())
            label = 1 if g.uniform() < 0.5 else 0
            for _ in range(nr):
                rows.append(((fp[0] + LEAK_GRP_SPREAD * g.normal(), fp[1] + LEAK_GRP_SPREAD * g.normal()), label))
                subject.append(s)
        n = len(rows)
        order = list(range(n))
        g.shuffle(order)
        row_acc.append(_nn2_accuracy([rows[i] for i in order[:n // 2]], [rows[i] for i in order[n // 2:]]))
        subjects = list(range(ns))
        g.shuffle(subjects)
        train_subjects = set(subjects[:ns // 2])
        grp_acc.append(_nn2_accuracy([r for r, s in zip(rows, subject) if s in train_subjects],
                                     [r for r, s in zip(rows, subject) if s not in train_subjects]))
    head = (f"{_generator_line(LEAK_GRP_NAME, True)}\n{ns} subjects with {nr} rows each. Per subject, in turn: a fingerprint of two "
            f"normal() draws, a label (1 if rng.uniform() < 0.5 else 0) shared by all its rows, then {nr} rows, each "
            f"(fingerprint_1 + {LEAK_GRP_SPREAD} * normal(), fingerprint_2 + {LEAK_GRP_SPREAD} * normal()). The label is a coin "
            f"flip per subject, so nothing about it is predictable from a new subject's features. {LEAK_GRP_REPS} repetitions; "
            f"after the {ns} subjects, the shuffle of the {ns * nr} row numbers (row protocol: first half trains) and then the "
            f"shuffle of the {ns} subject numbers (group protocol: the first {ns // 2} subjects, all their rows, train). The "
            f"model answers with the label of the nearest training row in the plane (squared Euclidean distance; the first in "
            f"training-list order on a tie; the training list is in shuffled order for the row protocol and in original row "
            f"order, subject after subject, for the group protocol). sd is the sample standard deviation (divisor n - 1) "
            f"over the {LEAK_GRP_REPS} repetitions.\n\n")
    return head + _table(["protocol", "mean accuracy", "sd over repetitions"],
                         [("random split of the rows", f"{_mean(row_acc):.3f}", f"{_sd(row_acc):.3f}"),
                          ("split by subject", f"{_mean(grp_acc):.3f}", f"{_sd(grp_acc):.3f}")], "lrr")


def _nn2_accuracy(train, test):
    hits = 0
    for (x1, x2), y in test:
        best = min(range(len(train)), key=lambda i: (train[i][0][0] - x1) ** 2 + (train[i][0][1] - x2) ** 2)
        hits += 1 if train[best][1] == y else 0
    return hits / len(test)


# ------------------------------------------------------------------ distribution shift

def _shift_data():
    g = bookrng.generator(SHIFT_NAME)
    train = _draw(g, SHIFT_TRAIN_N, V_LO, SHIFT_TRAIN_HI)
    held = _draw(g, SHIFT_HELD_N, V_LO, SHIFT_TRAIN_HI)
    prod = _draw(g, SHIFT_PROD_N, SHIFT_TRAIN_HI, V_HI)
    return train, held, prod


def covariate_shift():
    """Fits to low voltages, scored on held-out low voltages and on high voltages."""
    train, held, prod = _shift_data()
    a, b = _gram([_features(v) for v, _ in train], [y for _, y in train])
    rows = []
    for d in SHIFT_DEGREES:
        w = _fit_gram(a, b, d)
        rows.append((d, d + 1, f"{_rmse_rows(w, train):.3f}", f"{_rmse_rows(w, held):.3f}", _num(_rmse_rows(w, prod))))
    head = (f"{_generator_line(SHIFT_NAME, True)}\nThree sets from the source, drawn in this order from one generator, two draws per row: "
            f"{SHIFT_TRAIN_N} training rows with v = {V_LO} + {SHIFT_TRAIN_HI - V_LO:.1f} * rng.uniform(); {SHIFT_HELD_N} held-out "
            f"rows from the same range; then {SHIFT_PROD_N} 'production' rows with v = {SHIFT_TRAIN_HI} + "
            f"{V_HI - SHIFT_TRAIN_HI:.1f} * rng.uniform(). Rows are as in the source (v, then the noise draw), with the features x "
            f"of the first table. Least squares fits of degree d to the training rows (normal equations, solve(A, y, true)); "
            f"RMSE in degrees Celsius.\n\n")
    return head + _table(["degree d", "parameters", "training RMSE", "held-out RMSE (same range)", "production RMSE (higher voltages)"], rows)


def shift_detection():
    """Comparing the feature of the production rows with the feature of the training rows."""
    train, held, prod = _shift_data()
    cols = {}
    for name, rows in (("training", train), ("held-out", held), ("production", prod)):
        vs = [v for v, _ in rows]
        cols[name] = vs
    out = []
    for name, vs in cols.items():
        m, s = _mean(vs), _sd(vs)
        lo, hi = min(vs), max(vs)
        outside = sum(1 for v in vs if v < min(cols["training"]) or v > max(cols["training"])) / len(vs)
        out.append((name, len(vs), f"{m:.3f}", f"{s:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{outside:.3f}"))
    tr, ho, pr = cols["training"], cols["held-out"], cols["production"]

    def welch(x, y):
        return (_mean(y) - _mean(x)) / math.sqrt(_sd(x) ** 2 / len(x) + _sd(y) ** 2 / len(y))
    tail = (f"\n\nDifference of means in standard errors, (mean of second - mean of first) / sqrt(s1^2/n1 + s2^2/n2), with s the sample standard deviation: "
            f"training to held-out {welch(tr, ho):+.2f}; training to production {welch(tr, pr):+.2f}.")
    head = ("The voltage feature v of the three sets of the previous example. sd v is the sample standard deviation (divisor "
            "n - 1). 'Outside' is the fraction of the set's rows with v below the smallest or above the largest training "
            "voltage (a value equal to the smallest or largest is inside).\n\n")
    return head + _table(["set", "rows", "mean v", "sd v", "min v", "max v", "outside training range"], out) + tail


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def label_shift():
    """Two Gaussian classes with a fixed rule and a deployment in which the classes are rarer or commoner."""
    m0, m1 = LABEL_SHIFT_MEANS
    mid = (m0 + m1) / 2

    def error(threshold, prior1):
        # predict class 1 when x > threshold
        return (1 - prior1) * (1 - _phi(threshold - m0)) + prior1 * _phi(threshold - m1)

    rows = []
    for p1 in LABEL_SHIFT_PRIORS:
        best = mid + math.log((1 - p1) / p1) / (m1 - m0)
        rows.append((f"{p1:g}", f"{error(mid, p1):.4f}", f"{p1:.4f}", f"{best:.4f}", f"{error(best, p1):.4f}"))
    head = (f"Class 0: x ~ N({m0:g}, 1). Class 1: x ~ N({m1:g}, 1). The rule trained on equal classes is 'predict class 1 when "
            f"x > {mid:g}'. The error of the rule 'predict class 1 when x > x0' is (1 - p1) P(x > x0 | class 0) + p1 P(x <= x0 | class 1), "
            f"computed with the Gaussian cumulative distribution function Phi(z) = (1 + erf(z / sqrt(2))) / 2. The best "
            f"threshold for a class-1 share p1 is x0 = {mid:g} + ln((1 - p1)/p1) / {m1 - m0:g}. Always answering class 0 has "
            f"error p1.\n\n")
    return head + _table(["share of class 1 in use, p1", f"error of the rule x > {mid:g}", "error of always answering class 0", "best threshold", "error at the best threshold"], rows)


# ------------------------------------------------------------------ more data or more model

def learning_curves():
    """Excess error (bias^2 + variance) of a line, a parabola and a degree-6 polynomial as the training set grows."""
    g = bookrng.generator(CURVE_NAME)
    grid = [V_LO + (V_HI - V_LO) * i / (BV_GRID - 1) for i in range(BV_GRID)]
    gf = [_features(v) for v in grid]
    tv = [truth(v) for v in grid]
    rows = []
    for n in CURVE_SIZES:
        tot = {d: 0.0 for d in CURVE_DEGREES}
        for _ in range(CURVE_SETS):
            data = _draw(g, n)
            a, b = _gram([_features(v) for v, _ in data], [y for _, y in data])
            for d in CURVE_DEGREES:
                w = _fit_gram(a, b, d)
                tot[d] += sum((_predict(w, f) - t) ** 2 for f, t in zip(gf, tv)) / len(tv)
        rows.append((n, *[_num(tot[d] / CURVE_SETS) for d in CURVE_DEGREES]))
    head = (f"{_generator_line(CURVE_NAME, True)}\nFor each N in {CURVE_SIZES} in turn, {CURVE_SETS} training sets of N rows in "
            f"turn from the one generator (two draws per row; the generator is not reset between sizes). Every set is fitted at "
            f"every degree. Least squares fits of degree {CURVE_DEGREES}; the entry is the mean over the sets "
            f"of the grid average of (prediction - true value)^2 on the {BV_GRID} grid points v = {V_LO} + 2.4 i / {BV_GRID - 1}. "
            f"This is the excess error: the expected squared error on fresh labels is this plus {NOISE_SD ** 2:.2f}.\n\n")
    return head + _table(["N", *[f"degree {d}" for d in CURVE_DEGREES]], rows)
