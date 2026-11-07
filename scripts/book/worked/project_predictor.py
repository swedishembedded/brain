# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Reference implementation and acceptance tables for the Volume II project.

Plain Python 3, standard library only, written from the project chapter's own
description. The functions that take arguments are the reference predictor:
a pump-measurement source with a hidden law, group-aware splits and folds with
a leakage probe, trivial and simple baselines, linear regression by the normal
equations and by gradient descent, an L2-penalised logistic classifier with the
penalty chosen by grouped cross-validation, the evaluation metrics of the
metrics chapter (confusion counts, ROC area two ways, reliability bins, ECE,
temperature scaling, a cost-based threshold), k nearest neighbours, a cluster
bootstrap, a multi-seed study and an ablation. Each public function without
arguments returns a markdown string that the chapter embeds between
`<!-- worked: project_predictor.<name> -->` markers: the acceptance tests, every
expected value computed here by the reference itself.

Randomness: every generator is the book's SplitMix64 and every seed comes from
`bookrng.seed_of("project_predictor.<name>")`, printed with its draw order. The
multi-seed study draws its per-run seeds as raw values of one master generator.

Swedish Embedded AB implements solutions for predictors whose claims can be
checked. If your team needs expertise in honest model evaluation and
calibration, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from functools import lru_cache
from operator import mul

import bookrng

NAME = "project_predictor"

# ------------------------------------------------------------- the source
G_SITE_A, G_SITE_B, ROWS_PER_PUMP, D = 300, 60, 4, 5
WITHIN_SD = 0.2          # spread of a pump's rows around the pump's fingerprint
SHIFT_B = 2.0            # added to feature 0 at site B
SD_U, SD_V, SD_EPS = 1.5, 0.4, 0.5
LOGIT = (-0.5, (2.0, -1.2, 0.8, 0.0, 0.0), -0.8)   # intercept, slopes, coefficient of x0^2
TARGET = (3.0, (2.0, -1.0, 0.5, 0.0, 0.0))         # intercept, slopes
N_TRAIN_PUMPS, N_VAL_PUMPS, N_TEST_PUMPS = 180, 60, 60
K_FOLDS = 5
LAMBDAS = (0.0, 0.003, 0.01, 0.03, 0.1)
KS = (1, 5, 15, 31, 61, 121)
ETA_LOG, STEPS_LOG = 2.0, 100        # logistic gradient descent
ETA_LIN, STEPS_LIN = 0.3, 100        # linear regression gradient descent
C_FP, C_FN = 1.0, 5.0                # the cost of a false alarm and of a missed failure
BOOT_RESAMPLES = 1000
N_RUNS = 16
N_PROBES = 20
RUN_BOOT_RESAMPLES = 2000
FEATURES = ("load", "temperature", "vibration", "age", "humidity")


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _seed(name):
    return bookrng.seed_of(f"{NAME}.{name}")


def _gen(name):
    return bookrng.generator(f"{NAME}.{name}")


def _seed_line(name):
    s = _seed(name)
    return f"seed {s:#018x} ({s})"


def _f(v, digits=4):
    return f"{v:.{digits}f}"


def _fs(v, digits=4):
    return f"{v:+.{digits}f}"


def sigmoid(z):
    """The two-branch form: the exponent is never positive."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def mean(v):
    return sum(v) / len(v)


def sd(v):
    """Standard deviation with divisor n - 1."""
    m = mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def percentile(sorted_values, q):
    """Linear interpolation at position q (R - 1) of an ascending list."""
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


# ----------------------------------------------------------- the data source

class Data:
    """Rows 0 .. N-1 in generation order; row n belongs to pump n // ROWS_PER_PUMP."""

    def __init__(self):
        rng = _gen("data")
        self.x, self.c, self.t, self.pump = [], [], [], []
        self.u, self.v = [], []
        b0, bs, bq = LOGIT
        t0, ts = TARGET
        for g in range(G_SITE_A + G_SITE_B):
            a = [rng.normal() for _ in range(D)]
            u = SD_U * rng.normal()
            v = SD_V * rng.normal()
            self.u.append(u)
            self.v.append(v)
            for r in range(ROWS_PER_PUMP):
                x = [a[j] + WITHIN_SD * rng.normal() for j in range(D)]
                if g >= G_SITE_A:
                    x[0] += SHIFT_B
                eta = b0 + sum(bs[j] * x[j] for j in range(D)) + bq * x[0] * x[0] + u
                uni = rng.uniform()
                self.c.append(1 if uni < sigmoid(eta) else 0)
                eps = rng.normal()
                self.t.append(t0 + sum(ts[j] * x[j] for j in range(D)) + v + SD_EPS * eps)
                if g == 0 and r == 0:
                    self.trace = {"a": a, "u": u, "v": v, "eta": eta, "uniform": uni, "eps": eps}
                self.x.append(x)
                self.pump.append(g)
        self.n = len(self.x)
        self.n_site_a = G_SITE_A * ROWS_PER_PUMP

    def rows_of(self, pumps):
        """Row numbers of the given pumps, in ascending row order."""
        wanted = set(pumps)
        return [n for n in range(self.n) if self.pump[n] in wanted]


@lru_cache(maxsize=None)
def data():
    return Data()


# ------------------------------------------------------- splits and folds

def fold_assignment(n, k, rng):
    require = 2 <= k <= n
    if not require:
        raise ValueError("need at least two folds and no more folds than rows")
    order = list(range(n))
    rng.shuffle(order)
    fold = [0] * n
    for position, item in enumerate(order):
        fold[item] = position % k
    return fold


def group_fold_assignment(group, n_groups, k, rng):
    group_fold = fold_assignment(n_groups, k, rng)
    return [group_fold[g] for g in group]


def split_pumps(rng):
    """(train, validation, test) pump numbers of site A, each ascending."""
    order = list(range(G_SITE_A))
    rng.shuffle(order)
    a, b = N_TRAIN_PUMPS, N_TRAIN_PUMPS + N_VAL_PUMPS
    return sorted(order[:a]), sorted(order[a:b]), sorted(order[b:])


class Split:
    """The honest split of one run: pump sets, row sets and the grouped folds of the training rows."""

    def __init__(self, rng, fold_rng=None):
        self.pumps = split_pumps(rng)
        d = data()
        self.train, self.val, self.test = (d.rows_of(p) for p in self.pumps)
        self.slice_b = [n for n in range(d.n) if d.pump[n] >= G_SITE_A]
        local = {g: i for i, g in enumerate(self.pumps[0])}
        self.fold_of_pump = fold_assignment(len(self.pumps[0]), K_FOLDS, fold_rng or rng)
        self.fold = [self.fold_of_pump[local[d.pump[n]]] for n in self.train]


@lru_cache(maxsize=None)
def main_split():
    rng = _gen("split")
    return Split(rng, _gen("folds"))


# ----------------------------------------------------- standardisation

def fit_standardiser(rows):
    n, d = len(rows), len(rows[0])
    m = [sum(r[j] for r in rows) / n for j in range(d)]
    s = [math.sqrt(sum((r[j] - m[j]) ** 2 for r in rows) / n) for j in range(d)]
    return m, s


def standardise(rows, m, s):
    return [[(r[j] - m[j]) / s[j] for j in range(len(m))] for r in rows]


def _columns(Z):
    return [list(c) for c in zip(*Z)]


# --------------------------------------------------- linear regression

def solve(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("singular system")
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c2 in range(col, n + 1):
                M[r][c2] -= f * M[col][c2]
    w = [0.0] * n
    for i in range(n - 1, -1, -1):
        w[i] = (M[i][n] - sum(M[i][j] * w[j] for j in range(i + 1, n))) / M[i][i]
    return w


def design(Z):
    return [[1.0] + list(z) for z in Z]


def linreg_normal(Z, y):
    X = design(Z)
    p = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(p)] for i in range(p)]
    b = [sum(r[i] * t for r, t in zip(X, y)) for i in range(p)]
    return solve(A, b)


def mse_loss_grad(w, X, y):
    n = len(X)
    res = [sum(map(mul, w, r)) - t for r, t in zip(X, y)]
    loss = sum(e * e for e in res) / n
    grad = [2.0 * sum(e * r[j] for e, r in zip(res, X)) / n for j in range(len(w))]
    return loss, grad


def linreg_gd(Z, y, eta=ETA_LIN, steps=STEPS_LIN):
    X = design(Z)
    w = [0.0] * len(X[0])
    for _ in range(steps):
        _, g = mse_loss_grad(w, X, y)
        w = [wi - eta * gi for wi, gi in zip(w, g)]
    return w


def predict_linear(w, Z):
    return [w[0] + sum(map(mul, w[1:], z)) for z in Z]


def rmse(pred, y):
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y))


def r_squared(pred, y, reference_mean):
    """1 - SSE / SST, with the total sum of squares taken around `reference_mean` (the training mean)."""
    sse = sum((p - t) ** 2 for p, t in zip(pred, y))
    sst = sum((t - reference_mean) ** 2 for t in y)
    return 1.0 - sse / sst


def numerical_gradient(f, w, h=1e-5):
    g = []
    for j in range(len(w)):
        up = list(w)
        dn = list(w)
        up[j] += h
        dn[j] -= h
        g.append((f(up) - f(dn)) / (2.0 * h))
    return g


def close(a, b, rtol, atol):
    return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b))


# ------------------------------------------------------ logistic regression

def logistic_loss(w, Z, y, lam):
    """Mean binary cross-entropy from the logits plus lam/2 times the squared slopes (the intercept w[0] is free)."""
    n = len(Z)
    total = 0.0
    for z, t in zip(Z, y):
        e = w[0] + sum(map(mul, w[1:], z))
        total += softplus(e) - t * e
    return total / n + 0.5 * lam * sum(a * a for a in w[1:])


def logistic_grad(w, cols, y, lam):
    """Gradient of logistic_loss; `cols` holds the standardised features by column (cols[j][n])."""
    n = len(y)
    e = [w[0]] * n
    for wj, col in zip(w[1:], cols):
        e = [a + wj * b for a, b in zip(e, col)]
    r = [sigmoid(a) - t for a, t in zip(e, y)]
    return [sum(r) / n] + [sum(map(mul, r, col)) / n + lam * wj for col, wj in zip(cols, w[1:])]


def fit_logistic(Z, y, lam, eta=ETA_LOG, steps=STEPS_LOG):
    cols = _columns(Z)
    w = [0.0] * (len(cols) + 1)
    for _ in range(steps):
        g = logistic_grad(w, cols, y, lam)
        w = [wi - eta * gi for wi, gi in zip(w, g)]
    return w


def logits(w, Z):
    return [w[0] + sum(map(mul, w[1:], z)) for z in Z]


def norm(v):
    return math.sqrt(sum(a * a for a in v))


# --------------------------------------------------------------- metrics

def log_loss_from_logits(z, y):
    return sum(softplus(a) - t * a for a, t in zip(z, y)) / len(y)


def log_loss_from_probs(p, y):
    return -sum(math.log(q) if t == 1 else math.log(1.0 - q) for q, t in zip(p, y)) / len(y)


def accuracy(pred, y):
    return sum(1 for p, t in zip(pred, y) if p == t) / len(y)


def confusion(pred, y):
    tp = sum(1 for p, t in zip(pred, y) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, y) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, y) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(pred, y) if p == 0 and t == 0)
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return prec, rec, f1


def auc_rank(score, y):
    """(sum of the positives' ranks - n+ (n+ + 1) / 2) / (n+ n-); ranks 1-based ascending, ties share their average."""
    n = len(score)
    order = sorted(range(n), key=lambda i: score[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rank[order[k]] = (i + j + 2) / 2.0
        i = j + 1
    pos = sum(y)
    neg = n - pos
    return (sum(r for r, t in zip(rank, y) if t == 1) - pos * (pos + 1) / 2.0) / (pos * neg)


def roc_points(score, y):
    """(FPR, TPR) from the threshold above every score downward; tied scores move together."""
    pos = sum(y)
    neg = len(y) - pos
    order = sorted(range(len(y)), key=lambda i: -score[i])
    pts = [(0.0, 0.0)]
    tp = fp = 0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and score[order[j]] == score[order[i]]:
            if y[order[j]] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        pts.append((fp / neg, tp / pos))
        i = j
    return pts


def auc_trapezoid(score, y):
    pts = roc_points(score, y)
    return sum((x1 - x0) * (y0 + y1) / 2.0 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))


def reliability_bins(p, y, n_bins=10):
    """(count, mean p, observed share) for equal-width bins of p; p = 1 falls in the last bin."""
    cnt, sp, sy = [0] * n_bins, [0.0] * n_bins, [0] * n_bins
    for q, t in zip(p, y):
        k = min(int(q * n_bins), n_bins - 1)
        cnt[k] += 1
        sp[k] += q
        sy[k] += t
    return [(cnt[k], sp[k] / cnt[k] if cnt[k] else None, sy[k] / cnt[k] if cnt[k] else None) for k in range(n_bins)]


def ece(p, y, n_bins=10):
    n = len(y)
    return sum(c / n * abs(f - m) for c, m, f in reliability_bins(p, y, n_bins) if c)


def fit_scale(z, y):
    """The scale a > 0 minimising the cross-entropy of sigmoid(a z): bracket by doubling, then bisection."""
    def slope(a):
        return sum((sigmoid(a * v) - t) * v for v, t in zip(z, y)) / len(y)
    if not slope(0.0) < 0.0:
        raise ValueError("the logits do not rank the positives above the negatives")
    lo, hi = 0.0, 1.0
    for _ in range(60):
        if slope(hi) > 0.0:
            break
        lo, hi = hi, 2.0 * hi
    else:
        raise ValueError("no finite scale: the logits separate the labels")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if slope(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cost_threshold():
    return C_FP / (C_FP + C_FN)


def cost_per_row(pred, y):
    tp, fp, fn, tn = confusion(pred, y)
    return (C_FP * fp + C_FN * fn) / len(y)


# --------------------------------------------------- nearest neighbours

def knn_probabilities(Ztr, ytr, Zq, ks):
    """For each k in ks, the smoothed share (n1 + 1) / (k + 2) of label 1 among the k nearest training rows of each
    query, Euclidean distance, equal distances ordered by row number. Returns {k: [probability per query]}."""
    out = {k: [] for k in ks}
    for q in Zq:
        dist = [math.dist(z, q) for z in Ztr]
        order = sorted(range(len(Ztr)), key=lambda i: dist[i])      # sorted() is stable
        run, idx = 0, 0
        for k in sorted(ks):
            while idx < k:
                run += ytr[order[idx]]
                idx += 1
            out[k].append((run + 1.0) / (k + 2.0))
    return out


def one_nn_labels(Ztr, ytr, Zq):
    labels = []
    for q in Zq:
        best, best_d = 0, None
        for i, z in enumerate(Ztr):
            dd = math.dist(z, q)
            if best_d is None or dd < best_d:
                best, best_d = i, dd
        labels.append(ytr[best])
    return labels


# ----------------------------------------------------- fitted pipelines

def _xs(rows):
    d = data()
    return [d.x[n] for n in rows]


def _ys(rows, which="fail"):
    d = data()
    src = d.c if which == "fail" else d.t
    return [src[n] for n in rows]


def cv_logistic(rows, fold, lams=LAMBDAS):
    """{lam: [log loss of fold j, ...]}: each fold is scored by a model fitted on the other folds,
    with the standardiser fitted on those other folds too."""
    out = {lam: [] for lam in lams}
    for j in range(K_FOLDS):
        tr = [n for n, f in zip(rows, fold) if f != j]
        ho = [n for n, f in zip(rows, fold) if f == j]
        m, s = fit_standardiser(_xs(tr))
        Ztr, Zho = standardise(_xs(tr), m, s), standardise(_xs(ho), m, s)
        ytr, yho = _ys(tr), _ys(ho)
        for lam in lams:
            w = fit_logistic(Ztr, ytr, lam)
            out[lam].append(log_loss_from_logits(logits(w, Zho), yho))
    return out


def cv_knn(rows, fold, ks=KS):
    out = {k: [] for k in ks}
    for j in range(K_FOLDS):
        tr = [n for n, f in zip(rows, fold) if f != j]
        ho = [n for n, f in zip(rows, fold) if f == j]
        m, s = fit_standardiser(_xs(tr))
        probs = knn_probabilities(standardise(_xs(tr), m, s), _ys(tr), standardise(_xs(ho), m, s), ks)
        yho = _ys(ho)
        for k in ks:
            out[k].append(log_loss_from_probs(probs[k], yho))
    return out


def best_of(table):
    """The key with the smallest mean fold loss; on a tie the later (larger) key."""
    best, best_v = None, None
    for key, folds in table.items():
        v = mean(folds)
        if best_v is None or v <= best_v:
            best, best_v = key, v
    return best


class Fitted:
    """The logistic model of one run: standardiser, penalty and weights fitted on the training rows."""

    def __init__(self, split, lam):
        self.lam = lam
        self.m, self.s = fit_standardiser(_xs(split.train))
        self.w = fit_logistic(standardise(_xs(split.train), self.m, self.s), _ys(split.train), lam)

    def logits(self, rows):
        return logits(self.w, standardise(_xs(rows), self.m, self.s))


@lru_cache(maxsize=None)
def main_cv():
    sp = main_split()
    return cv_logistic(sp.train, sp.fold)


@lru_cache(maxsize=None)
def main_model():
    sp = main_split()
    return Fitted(sp, best_of(main_cv()))


@lru_cache(maxsize=None)
def main_cv_knn():
    sp = main_split()
    return cv_knn(sp.train, sp.fold)


def knn_scores(split, k, rows):
    m, s = fit_standardiser(_xs(split.train))
    return knn_probabilities(standardise(_xs(split.train), m, s), _ys(split.train),
                             standardise(_xs(rows), m, s), (k,))[k]


# ---------------------------------------------------------- the bootstrap

def cluster_bootstrap(rows_by_pump, statistic, rng, resamples=BOOT_RESAMPLES):
    """Replicates of statistic(list of rows) over resamples of whole pumps, each pump drawn with rng.integer(G)."""
    g = len(rows_by_pump)
    reps = []
    for _ in range(resamples):
        rows = []
        for _ in range(g):
            rows.extend(rows_by_pump[rng.integer(g)])
        reps.append(statistic(rows))
    return reps


def interval(reps, level=0.95):
    s = sorted(reps)
    a = (1.0 - level) / 2.0
    return percentile(s, a), percentile(s, 1.0 - a)


def rows_by_pump(rows):
    d = data()
    groups = {}
    for n in rows:
        groups.setdefault(d.pump[n], []).append(n)
    return [groups[g] for g in sorted(groups)]


# ======================================================================
# Worked blocks. Each returns markdown for the chapter.
# ======================================================================

def dataset_fingerprint():
    d = data()
    rows = [0, 1, 2, 3, 4, 1199, 1200, 1439]
    body = []
    for n in rows:
        body.append((n, d.pump[n], *(f"{v:+.6f}" for v in d.x[n]), d.c[n], f"{d.t[n]:+.6f}"))
    out = (f"Generator: the book's SplitMix64, {_seed_line('data')}; the draws are those of `generate` in the order printed in "
           "the chapter. Selected rows (x0 .. x4, then fail and wear):\n\n")
    out += _table(["row", "pump", *[f"x{j}" for j in range(D)], "fail", "wear"], body)
    sums = [f"{sum(d.x[n][j] for n in range(d.n)):+.6f}" for j in range(D)]
    pos_a = sum(d.c[: d.n_site_a])
    pos_b = sum(d.c[d.n_site_a:])
    tr = d.trace
    out += ("\n\nTrace of the first draws (pump 0, row 0): profile = (" + ", ".join(f"{v:+.6f}" for v in tr["a"])
            + f"), pump effects effect_fail = {tr['u']:+.6f} and effect_wear = {tr['v']:+.6f} (the standard deviations 1.5 and 0.4 already applied), "
            f"true logit e* = {tr['eta']:+.6f}, label draw {tr['uniform']:.6f}, target noise draw {tr['eps']:+.6f} "
            "(before the factor 0.5).")
    out += "\n\n" + _table(
        ["quantity", "value"],
        [("rows, site A (pumps 0 to 299) and site B (pumps 300 to 359)", f"{d.n_site_a} and {d.n - d.n_site_a}"),
         ("sum over all rows of x0, x1, x2, x3, x4", ", ".join(sums)),
         ("rows with fail = 1, site A and site B", f"{pos_a} and {pos_b}"),
         ("sum of wear over all rows", f"{sum(d.t):+.6f}"),
         ("sum of the pump effects effect_fail (pumps 0 to 359) and effect_wear", f"{sum(d.u):+.6f} and {sum(d.v):+.6f}")],
        "lr")
    return out


def split_and_probe():
    d, sp = data(), main_split()
    pumps = sp.pumps
    rows = [("training", len(pumps[0]), len(sp.train), sum(_ys(sp.train)), pumps[0][:6]),
            ("validation", len(pumps[1]), len(sp.val), sum(_ys(sp.val)), pumps[1][:6]),
            ("test", len(pumps[2]), len(sp.test), sum(_ys(sp.test)), pumps[2][:6]),
            ("shift slice (site B)", G_SITE_B, len(sp.slice_b), sum(_ys(sp.slice_b)), list(range(G_SITE_A, G_SITE_A + 6)))]
    out = (f"Generator for the split: the book's SplitMix64, {_seed_line('split')}. The pump numbers 0 .. 299 are shuffled "
           f"(299 integer draws, the book's shuffle); the first 180 shuffled numbers train, the next 60 validate, the last 60 "
           f"test; each set is then sorted ascending. Rows of a set are those of its pumps in ascending row number.\n\n"
           + _table(["set", "pumps", "rows", "rows with fail = 1", "first six pumps"],
                    [(a, b, c, e, ", ".join(str(x) for x in f)) for a, b, c, e, f in rows], "lrrrl"))
    shared = (set(pumps[0]) & set(pumps[1]), set(pumps[0]) & set(pumps[2]), set(pumps[1]) & set(pumps[2]))
    out += ("\n\nPumps shared between training and validation, training and test, validation and test: "
            + ", ".join(str(len(s)) for s in shared) + ".")
    # folds
    counts = [sum(1 for f in sp.fold_of_pump if f == j) for j in range(K_FOLDS)]
    row_counts = [sum(1 for f in sp.fold if f == j) for j in range(K_FOLDS)]
    out += (f"\n\nFolds of the 180 training pumps: the book's SplitMix64, {_seed_line('folds')}; the training pumps, "
            f"numbered 0 .. 179 in ascending pump order, are shuffled (179 integer draws) and the pump at shuffled "
            f"position p goes to fold p mod 5. Pumps per fold: {counts}. Rows per fold: {row_counts}. "
            f"First eight training pumps and their folds: "
            + ", ".join(f"{g}: {sp.fold_of_pump[i]}" for i, g in enumerate(pumps[0][:8])) + ".")
    # probe: repeated, because one draw of 240 scored rows is noisy
    master = _gen("probe")
    accs = {"rows": [], "pumps": []}
    for _ in range(N_PROBES):
        rng = bookrng.SplitMix64(master.next_u64())
        order = list(range(d.n_site_a))
        rng.shuffle(order)
        row_train, row_test = order[:720], order[720:960]
        pump_order = list(range(G_SITE_A))
        rng.shuffle(pump_order)
        pump_train = d.rows_of(pump_order[:N_TRAIN_PUMPS])
        pump_test = d.rows_of(pump_order[N_TRAIN_PUMPS + N_VAL_PUMPS:])
        for tag, tr, te in (("rows", row_train, row_test), ("pumps", pump_train, pump_test)):
            m, s = fit_standardiser(_xs(tr))
            lab = one_nn_labels(standardise(_xs(tr), m, s), _ys(tr), standardise(_xs(te), m, s))
            accs[tag].append(accuracy(lab, _ys(te)))
    out += ("\n\nLeakage probe. A model that can only memorise: label each held-out row with the label of its nearest training "
            "row (Euclidean distance on the training-standardised features, the standardiser fitted on the training rows of that protocol; "
            "on a tie the training row listed first wins; continuous data have no ties, so this never matters here). One draw of 240 scored rows is "
            f"noisy, so the probe is repeated {N_PROBES} times. Master generator: {_seed_line('probe')}; repetition r uses the "
            "r-th raw value (rng.next(), counting from 0) as the seed of its own SplitMix64, which draws, in order, the shuffle "
            "of the 1200 site A row numbers (1199 integer draws; random-row protocol: the first 720 shuffled rows train, listed in "
            "shuffled order, and the next 240 are scored) and then the shuffle of the 300 site A pump numbers (299 integer "
            "draws; pump protocol: the rows of the first 180 shuffled pumps train, listed in ascending row number, and the rows "
            "of the last 60 shuffled pumps, in ascending row number, are scored). The sd over repetitions has the divisor "
            f"{N_PROBES} - 1.\n\n"
            + _table(["protocol", "mean accuracy of the nearest-row label", "sd over repetitions", "lowest", "highest"],
                     [("random rows", _f(mean(accs["rows"])), _f(sd(accs["rows"])), _f(min(accs["rows"])), _f(max(accs["rows"]))),
                      ("whole pumps", _f(mean(accs["pumps"])), _f(sd(accs["pumps"])), _f(min(accs["pumps"])), _f(max(accs["pumps"])))],
                     "lrrrr"))
    return out


def _standardised_main():
    sp = main_split()
    m, s = fit_standardiser(_xs(sp.train))
    return sp, m, s


def standardiser_check():
    sp, m, s = _standardised_main()
    Ztr = standardise(_xs(sp.train), m, s)
    Zte = standardise(_xs(sp.test), m, s)
    rows = []
    for j in range(D):
        rows.append((f"x{j}", _f(m[j], 6), _f(s[j], 6), "yes" if abs(mean([z[j] for z in Ztr])) < 1e-12 else "NO",
                     _f(math.sqrt(sum(z[j] ** 2 for z in Ztr) / len(Ztr)), 6), _f(mean([z[j] for z in Zte]), 6)))
    return ("The standardiser is fitted on the 720 training rows (mean, and standard deviation with divisor N) and "
            "applied unchanged to the test rows. The mean of the training z is zero up to round-off (about 1e-17, whose digits depend "
            "on the order of summation), so the table tests it against 1e-12.\n\n"
            + _table(["feature", "training mean", "training sd", "abs(mean of training z) below 1e-12", "rms of training z", "mean of test z"], rows))


# ------------------------------------------------------------- baselines

def _best_feature(rows, which):
    """Index of the feature with the largest absolute correlation (training rows) with the outcome."""
    y = _ys(rows, which)
    X = _xs(rows)
    my = mean(y)
    best, best_r = 0, -1.0
    for j in range(D):
        col = [x[j] for x in X]
        mx = mean(col)
        sxy = sum((a - mx) * (b - my) for a, b in zip(col, y))
        sxx = sum((a - mx) ** 2 for a in col)
        syy = sum((b - my) ** 2 for b in y)
        r = abs(sxy) / math.sqrt(sxx * syy)
        if r > best_r:
            best, best_r = j, r
    return best, best_r


def _bin_table(rows, j):
    """The simple classifier: four bins of feature j cut at the midpoints of the sorted training values at positions
    N/4, N/2 and 3N/4 (cut q = (v[qN/4 - 1] + v[qN/4]) / 2); each bin predicts (n1 + 1) / (n + 2) from its training rows."""
    X = sorted(x[j] for x in _xs(rows))
    n = len(X)
    cuts = [(X[q * n // 4 - 1] + X[q * n // 4]) / 2.0 for q in (1, 2, 3)]
    y = _ys(rows)
    cnt, pos = [0] * 4, [0] * 4
    for x, t in zip(_xs(rows), y):
        b = sum(1 for c in cuts if x[j] > c)
        cnt[b] += 1
        pos[b] += t
    return cuts, [(pos[b] + 1.0) / (cnt[b] + 2.0) for b in range(4)], cnt


def _bin_predict(table, rows, j):
    cuts, probs, _ = table
    return [probs[sum(1 for c in cuts if x[j] > c)] for x in _xs(rows)]


def baselines():
    sp = main_split()
    ytr, yva = _ys(sp.train), _ys(sp.val)
    ttr, tva = _ys(sp.train, "wear"), _ys(sp.val, "wear")
    pbar = mean(ytr)
    tbar = mean(ttr)
    jc, rc = _best_feature(sp.train, "fail")
    jt, rt = _best_feature(sp.train, "wear")
    table = _bin_table(sp.train, jc)
    pbin = _bin_predict(table, sp.val, jc)
    t = cost_threshold()
    # the single-feature line, closed form
    xs = [x[jt] for x in _xs(sp.train)]
    mx = mean(xs)
    slope = sum((a - mx) * (b - tbar) for a, b in zip(xs, ttr)) / sum((a - mx) ** 2 for a in xs)
    icpt = tbar - slope * mx
    line_pred = [icpt + slope * x[jt] for x in _xs(sp.val)]
    const = [pbar] * len(yva)
    out = ("Classification, 180 training pumps (720 rows) against the 240 validation rows. The constant model predicts the "
           f"training share of fail = 1 for every row: p = {pbar:.6f}. The simple model is a four-bin frequency table on the one "
           f"feature with the largest absolute correlation with fail over the training rows (x{jc}, {FEATURES[jc]}, "
           f"|correlation| {rc:.4f}). Its cuts are {', '.join(f'{c:.4f}' for c in table[0])}; a row goes to bin 0 to 3 by "
           "the number of cuts it exceeds. Decisions are 'predict 1 when p >= 1/6' (the cost threshold of milestone 5).\n\n")
    out += _table(["bin", "training rows", "prediction p"], [(b, table[2][b], f"{table[1][b]:.6f}") for b in range(4)])
    out += "\n\n"
    out += _table(
        ["model", "log loss", "ROC AUC", "accuracy at 0.5", "cost per row at p >= 1/6"],
        [("constant probability", _f(log_loss_from_probs(const, yva)), _f(auc_rank(const, yva)),
          _f(accuracy([1 if q >= 0.5 else 0 for q in const], yva)), _f(cost_per_row([1 if q >= t else 0 for q in const], yva))),
         (f"four-bin table on x{jc}", _f(log_loss_from_probs(pbin, yva)), _f(auc_rank(pbin, yva)),
          _f(accuracy([1 if q >= 0.5 else 0 for q in pbin], yva)), _f(cost_per_row([1 if q >= t else 0 for q in pbin], yva)))],
        "lrrrr")
    out += (f"\n\nCost uses {C_FP:g} per false alarm and {C_FN:g} per missed failure. A model that always raises the alarm "
            "(the constant model at this threshold) has the cost per row of the share of rows with fail = 0.\n\n"
            "Regression, the same rows. The constant model predicts the training mean of wear. The simple model is the "
            f"least-squares line on the one feature with the largest absolute correlation with wear (x{jt}, {FEATURES[jt]}, "
            f"|correlation| {rt:.4f}).\n\n")
    out += _table(
        ["model", "RMSE", "R^2 (around the training mean)"],
        [(f"constant {tbar:.6f}", _f(rmse([tbar] * len(tva), tva)), _f(r_squared([tbar] * len(tva), tva, tbar))),
         (f"line on x{jt}: wear = {icpt:.4f} + {slope:.4f} x{jt}", _f(rmse(line_pred, tva)), _f(r_squared(line_pred, tva, tbar)))],
        "lrr")
    return out


# ------------------------------------------------- milestone 3: regression

@lru_cache(maxsize=None)
def _regression():
    sp, m, s = _standardised_main()
    Ztr = standardise(_xs(sp.train), m, s)
    ttr = _ys(sp.train, "wear")
    wn = linreg_normal(Ztr, ttr)
    wg = linreg_gd(Ztr, ttr)
    return sp, m, s, Ztr, ttr, wn, wg


def regression_fit():
    sp, m, s, Ztr, ttr, wn, wg = _regression()
    X = design(Ztr)
    # the gradient check at a fixed non-trivial point
    w0 = [0.5, -0.25, 0.1, 0.3, -0.2, 0.05]
    sub, tsub = X[:200], ttr[:200]
    ana = mse_loss_grad(w0, sub, tsub)[1]
    num = numerical_gradient(lambda w: mse_loss_grad(w, sub, tsub)[0], w0)
    wrong = list(ana)
    wrong[1] = 0.0
    out = ("Training rows standardised as in the previous table; design matrix [1, z] of shape [720, 6]; loss "
           "mean((w.[1, z] - wear)^2). Normal equations (X^T X) w = X^T wear solved by Gaussian elimination with partial pivoting; "
           f"gradient descent from w = 0 with eta = {ETA_LIN:g} and {STEPS_LIN} steps.\n\n")
    names = ["intercept"] + [f"z{j}" for j in range(D)]
    out += _table(["weight", "normal equations", "gradient descent"],
                  [(n_, f"{a:.10f}", f"{b:.10f}") for n_, a, b in zip(names, wn, wg)])
    _, gfin = mse_loss_grad(wg, X, ttr)
    out += (f"\n\nThe two solutions agree to 1e-8 in every weight: {close(wg, wn, 0.0, 1e-8)}. The gradient norm at the "
            f"gradient-descent solution is below 1e-8: {norm(gfin) < 1e-8}. Loss at both: {mse_loss_grad(wn, X, ttr)[0]:.10f}.")
    out += ("\n\nGradient check at w = (0.5, -0.25, 0.1, 0.3, -0.2, 0.05) on the first 200 training rows, central differences "
            "with h = 1e-5, close with rtol = 1e-6 and atol = 1e-8:\n\n")
    out += _table(["component", "analytic", "numerical"],
                  [(n_, f"{a:.8f}", f"{b:.8f}") for n_, a, b in zip(names, ana, num)])
    out += (f"\n\nCheck passes: {close(num, ana, 1e-6, 1e-8)}. The same check against a gradient whose z0 component is "
            f"replaced by 0 passes: {close(num, wrong, 1e-6, 1e-8)}.")
    # raw units
    raw_slope = [wn[1 + j] / s[j] for j in range(D)]
    raw_icpt = wn[0] - sum(wn[1 + j] * m[j] / s[j] for j in range(D))
    law = [TARGET[0]] + list(TARGET[1])
    out += ("\n\nMapped back to the raw features (slope w_j / sd_j, intercept w_0 - sum w_j mean_j / sd_j), beside the "
            "coefficients of the law that generated wear:\n\n")
    out += _table(["coefficient", "fitted", "law"],
                  [("intercept", f"{raw_icpt:.4f}", f"{law[0]:.4f}")]
                  + [(f"x{j} ({FEATURES[j]})", f"{raw_slope[j]:+.4f}", f"{law[1 + j]:+.4f}") for j in range(D)])
    # validation scores
    Zva = standardise(_xs(sp.val), m, s)
    tva = _ys(sp.val, "wear")
    tbar = mean(ttr)
    pred = predict_linear(wn, Zva)
    floor = math.sqrt(SD_V ** 2 + SD_EPS ** 2)
    out += (f"\n\nOn the 240 validation rows: RMSE {rmse(pred, tva):.4f}, R^2 {r_squared(pred, tva, tbar):.4f} "
            f"(around the training mean {tbar:.4f}). The noise floor of the law for a new pump is "
            f"sqrt({SD_V}^2 + {SD_EPS}^2) = {floor:.4f}.")
    return out


# --------------------------------------------- milestone 4: logistic regression

def logistic_gradient_check():
    sp, m, s = _standardised_main()
    Z = standardise(_xs(sp.train), m, s)[:200]
    y = _ys(sp.train)[:200]
    w0 = [-0.3, 0.2, -0.1, 0.05, 0.0, 0.15]
    lam = 0.1
    cols = _columns(Z)
    ana = logistic_grad(w0, cols, y, lam)
    num = numerical_gradient(lambda w: logistic_loss(w, Z, y, lam), w0)
    nopen = logistic_grad(w0, cols, y, 0.0)
    names = ["intercept"] + [f"z{j}" for j in range(D)]
    out = ("First 200 training rows, standardised with the training statistics; w = (-0.3, 0.2, -0.1, 0.05, 0.0, 0.15); "
           "lambda = 0.1; loss = mean(softplus(e) - y e) + (lambda / 2) sum of the squared slopes; central differences with "
           "h = 1e-5, rtol = 1e-6, atol = 1e-8.\n\n")
    out += _table(["component", "analytic", "numerical"],
                  [(n_, f"{a:.8f}", f"{b:.8f}") for n_, a, b in zip(names, ana, num)])
    out += (f"\n\nCheck passes: {close(num, ana, 1e-6, 1e-8)}. With the penalty term left out of the gradient "
            f"(lambda = 0 in the analytic gradient only) the check passes: {close(num, nopen, 1e-6, 1e-8)}.")
    return out


def largest_eigenvalue(A, iterations=300):
    """Power iteration from the all-ones vector for a symmetric positive semi-definite matrix."""
    n = len(A)
    v = [1.0] * n
    lam = 0.0
    for _ in range(iterations):
        u = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        lam = math.sqrt(sum(a * a for a in u))
        v = [a / lam for a in u]
    return lam


def logistic_step_size():
    sp, m, s = _standardised_main()
    X = design(standardise(_xs(sp.train), m, s))
    n, p = len(X), len(X[0])
    A = [[sum(r[i] * r[j] for r in X) / n for j in range(p)] for i in range(p)]
    lam_max = largest_eigenvalue(A)
    curv = lam_max / 4.0
    rows = []
    for lam in LAMBDAS + (0.3, 1.0):
        limit = 2.0 / (curv + lam)
        rows.append((f"{lam:g}", "yes" if lam in LAMBDAS else "no", f"{1.0 / (curv + lam):.4f}", f"{limit:.4f}",
                     "yes" if ETA_LOG < limit else "no (outside the guarantee)"))
    out = (f"The matrix of the design [1, z] over the 720 training rows, A = X^T X / N (6 by 6); largest eigenvalue by power "
           f"iteration (300 iterations from the all-ones vector; each iteration forms A v, takes its Euclidean norm as the "
           f"eigenvalue estimate and divides A v by it to get the next v; the estimate of the last iteration is used): "
           f"{lam_max:.6f}. The curvature bound of the unpenalised loss is lambda_cap = lambda_max / 4 = {curv:.6f}; the penalty "
           f"adds lambda to it. Gradient descent with step eta is stable when eta < 2 / (lambda_cap + lambda); the step "
           f"1 / (lambda_cap + lambda) is the classical choice. The first five penalties are the grid of the cross-validation; "
           f"0.3 and 1 are added to show where the guarantee ends.\n\n")
    out += _table(["lambda", "in the grid", "1 / (lambda_cap + lambda)", "stability limit 2 / (lambda_cap + lambda)", f"eta = {ETA_LOG:g} stable?"],
                  rows, "llrrl")
    return out


def logistic_cv():
    sp = main_split()
    cv = main_cv()
    best = best_of(cv)
    model = main_model()
    rows = []
    for lam in LAMBDAS:
        f = cv[lam]
        rows.append((f"{lam:g}" + (" (chosen)" if lam == best else ""), *[_f(v) for v in f], _f(mean(f)), _f(sd(f))))
    out = ("Five folds of whole pumps (the fold assignment printed with the split). For each fold: the standardiser and "
           f"the model are fitted on the other four folds (gradient descent from w = 0, eta = {ETA_LOG:g}, {STEPS_LOG} steps) "
           "and the log loss of the held-out fold is recorded. The chosen lambda has the smallest mean (the larger lambda on a "
           "tie); sd uses the divisor 4.\n\n")
    out += _table(["lambda", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "sd"], rows)
    Ztr = standardise(_xs(sp.train), model.m, model.s)
    g = logistic_grad(model.w, _columns(Ztr), _ys(sp.train), model.lam)
    yva = _ys(sp.val)
    zva = model.logits(sp.val)
    out += (f"\n\nRefit on all 720 training rows with lambda = {best:g}: weights (intercept, then z0 .. z4) "
            + ", ".join(f"{v:.6f}" for v in model.w) + f". Gradient norm below 1e-6: {norm(g) < 1e-6}. Training log loss "
            f"{log_loss_from_logits(model.logits(sp.train), _ys(sp.train)):.4f}; validation log loss "
            f"{log_loss_from_logits(zva, yva):.4f}; validation ROC AUC {auc_rank(zva, yva):.4f}.")
    return out


# ------------------------------------------------- milestone 5: evaluation

FIX_SCORE = [0.9, 0.8, 0.8, 0.7, 0.55, 0.5, 0.5, 0.4, 0.3, 0.2, 0.15, 0.05]
FIX_LABEL = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
FIX_Z = [-2.0, -1.2, -0.4, 0.3, 0.9, 1.6, 2.4, -0.8, 1.1, 3.1, -3.0, 0.1]
FIX_ZY = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]


def metric_fixture():
    s, y = FIX_SCORE, FIX_LABEL
    pairs = 0.0
    ps = [a for a, t in zip(s, y) if t == 1]
    ns = [a for a, t in zip(s, y) if t == 0]
    for p in ps:
        for q in ns:
            pairs += 1.0 if p > q else (0.5 if p == q else 0.0)
    out = ("Twelve scored items (score, label): "
           + ", ".join(f"({a:g}, {t})" for a, t in zip(s, y)) + ".\n\n")
    out += _table(["quantity", "value"],
                  [("ROC AUC, rank statistic", f"{auc_rank(s, y):.10f}"),
                   ("ROC AUC, trapezoid on the ROC points", f"{auc_trapezoid(s, y):.10f}"),
                   ("ROC AUC, pairs counted (a tie counts 1/2): sum / (n+ n-)", f"{pairs:g} / {len(ps) * len(ns)} = {pairs / (len(ps) * len(ns)):.10f}")],
                  "lr")
    out += "\n\nROC points (FPR, TPR), tied scores moving together: " + ", ".join(f"({a:.4f}, {b:.4f})" for a, b in roc_points(s, y)) + "."
    rows = []
    for t in (0.5, 0.3, cost_threshold()):
        pred = [1 if v >= t else 0 for v in s]
        tp, fp, fn, tn = confusion(pred, y)
        prec, rec, f1 = prf(tp, fp, fn)
        rows.append((f"{t:.6f}", tp, fp, fn, tn, _f(accuracy(pred, y)), _f(prec), _f(rec), _f(f1), _f(cost_per_row(pred, y))))
    out += ("\n\nDecisions: predict 1 when score >= t. Cost uses 1 per false alarm and 5 per missed failure; t = 1/6 is the cost "
            "threshold c_FP / (c_FP + c_FN).\n\n")
    out += _table(["t", "TP", "FP", "FN", "TN", "accuracy", "precision", "recall", "F1", "cost per item"], rows)
    return out


def calibration_fixture():
    z, y = FIX_Z, FIX_ZY
    a = fit_scale(z, y)
    p0 = [sigmoid(v) for v in z]
    p1 = [sigmoid(a * v) for v in z]
    out = ("Twelve logits and labels: " + ", ".join(f"({v:g}, {t})" for v, t in zip(z, y)) + ".\n\n")
    out += f"Fitted scale a (bisection as in the metrics chapter): {a:.10f}; temperature 1/a = {1.0 / a:.10f}.\n\n"
    rows = []
    for name, p in (("raw, a = 1", p0), (f"scaled, a = {a:.4f}", p1)):
        rows.append((name, f"{log_loss_from_logits([v * (a if name[0] == 's' else 1.0) for v in z], y):.10f}", f"{ece(p, y):.10f}",
                     f"{ece(p, y, 5):.10f}", f"{auc_rank(p, y):.10f}"))
    out += _table(["probabilities", "log loss", "ECE, 10 bins", "ECE, 5 bins", "ROC AUC"], rows)
    out += "\n\nReliability bins of the raw probabilities (10 equal-width bins; empty bins omitted):\n\n"
    rows = []
    for k, (c, mp, fr) in enumerate(reliability_bins(p0, y)):
        if c:
            rows.append((f"{k / 10:.1f} to {(k + 1) / 10:.1f}", c, f"{mp:.6f}", f"{fr:.6f}"))
    out += _table(["bin of p", "items", "mean p", "observed share"], rows)
    return out


def validation_report():
    sp = main_split()
    model = main_model()
    zv, yv = model.logits(sp.val), _ys(sp.val)
    a = fit_scale(zv, yv)
    pv = [sigmoid(v) for v in zv]
    ps = [sigmoid(a * v) for v in zv]
    t = cost_threshold()
    out = (f"Validation rows (240, 60 pumps), logistic model with lambda = {model.lam:g}. Temperature scaling: the single scale a "
           f"is fitted on these rows by bisection, a = {a:.6f} (temperature {1.0 / a:.6f}).\n\n")
    rows = []
    for name, p, z in (("raw", pv, zv), ("scaled", ps, [a * v for v in zv])):
        rows.append((name, _f(log_loss_from_logits(z, yv)), _f(ece(p, yv)), _f(auc_rank(p, yv)), _f(auc_trapezoid(p, yv))))
    out += _table(["probabilities", "log loss", "ECE, 10 bins", "ROC AUC, rank", "ROC AUC, trapezoid"], rows)
    out += "\n\nReliability of the raw probabilities on the validation rows (10 equal-width bins, empty bins omitted):\n\n"
    rows = []
    for k, (c, mp, fr) in enumerate(reliability_bins(pv, yv)):
        if c:
            rows.append((f"{k / 10:.1f} to {(k + 1) / 10:.1f}", c, f"{mp:.4f}", f"{fr:.4f}"))
    out += _table(["bin of p", "rows", "mean p", "observed share"], rows)
    out += f"\n\nDecisions at the cost threshold t = {t:.6f} on the scaled probabilities, and at 0.5 for comparison:\n\n"
    rows = []
    for name, thr in (("0.5", 0.5), (f"{t:.6f}", t)):
        pred = [1 if q >= thr else 0 for q in ps]
        tp, fp, fn, tn = confusion(pred, yv)
        prec, rec, f1 = prf(tp, fp, fn)
        rows.append((name, tp, fp, fn, tn, _f(accuracy(pred, yv)), _f(prec), _f(rec), _f(f1), _f(cost_per_row(pred, yv))))
    out += _table(["t", "TP", "FP", "FN", "TN", "accuracy", "precision", "recall", "F1", "cost per row"], rows)
    return out


# ------------------------------------------------ milestone 6: k-NN

def knn_comparison():
    sp = main_split()
    cv = main_cv_knn()
    best_k = best_of(cv)
    rows = []
    for k in KS:
        f = cv[k]
        rows.append((f"{k}" + (" (chosen)" if k == best_k else ""), *[_f(v) for v in f], _f(mean(f)), _f(sd(f))))
    out = ("The same five grouped folds. For each fold the standardiser is fitted on the other four, each held-out row gets "
           "the probability (n1 + 1) / (k + 2) from its k nearest rows (Euclidean distance, equal distances in row order), and "
           "the log loss of the fold is recorded.\n\n")
    out += _table(["k", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "sd"], rows)
    model = main_model()
    yv = _ys(sp.val)
    zv = model.logits(sp.val)
    pk = knn_scores(sp, best_k, sp.val)
    pl = [sigmoid(v) for v in zv]
    out += f"\n\nOn the 240 validation rows, models fitted on all 720 training rows:\n\n"
    out += _table(["model", "log loss", "ROC AUC", "ECE, 10 bins"],
                  [(f"logistic, lambda = {model.lam:g}", _f(log_loss_from_logits(zv, yv)), _f(auc_rank(zv, yv)), _f(ece(pl, yv))),
                   (f"k nearest neighbours, k = {best_k}", _f(log_loss_from_probs(pk, yv)), _f(auc_rank(pk, yv)), _f(ece(pk, yv)))])
    return out


# ------------------------------------------------ milestone 7: the report

def _test_scores():
    sp = main_split()
    model = main_model()
    zv, yv = model.logits(sp.val), _ys(sp.val)
    a = fit_scale(zv, yv)
    k = best_of(main_cv_knn())
    zt = model.logits(sp.test)
    return sp, model, a, k, zt, knn_scores(sp, k, sp.test)


def test_report():
    sp, model, a, k, zt, pk = _test_scores()
    d = data()
    test = sp.test
    yt = _ys(test)
    pbar = mean(_ys(sp.train))
    jc, _ = _best_feature(sp.train, "fail")
    pbin = _bin_predict(_bin_table(sp.train, jc), test, jc)
    pl = [sigmoid(v) for v in zt]
    ps = [sigmoid(a * v) for v in zt]
    t = cost_threshold()
    out = ("The test rows (240 rows, 60 pumps), read once, with every choice already made on training and validation rows: "
           f"lambda = {model.lam:g}, k = {k}, scale a = {a:.6f}, decision threshold {t:.6f} on the scaled probability "
           "(and on the other probabilities as they are).\n\n")
    ent = lambda q: [q] * len(yt)
    models = [
        ("constant probability", ent(pbar), [1 if pbar >= t else 0] * len(yt)),
        (f"four-bin table on x{jc}", pbin, [1 if q >= t else 0 for q in pbin]),
        ("logistic, raw", pl, [1 if q >= t else 0 for q in pl]),
        ("logistic, temperature scaled", ps, [1 if q >= t else 0 for q in ps]),
        (f"k nearest neighbours, k = {k}", pk, [1 if q >= t else 0 for q in pk]),
    ]
    rows = []
    for name, p, pred in models:
        score = p if p is not None else [float(v) for v in pred]
        ll = _f(log_loss_from_probs(p, yt)) if p is not None else "-"
        e = _f(ece(p, yt)) if p is not None else "-"
        tp, fp, fn, tn = confusion(pred, yt)
        _, rec, f1 = prf(tp, fp, fn)
        rows.append((name, ll, _f(auc_rank(score, yt)), e, _f(f1), _f(rec), _f(cost_per_row(pred, yt))))
    out += _table(["model", "log loss", "ROC AUC", "ECE, 10 bins", "F1 at t", "recall at t", "cost per row at t"], rows)

    # cluster bootstrap, paired
    groups = rows_by_pump(test)
    pos_of = {n: i for i, n in enumerate(test)}
    def stat(fn):
        return lambda rows_: fn([pos_of[n] for n in rows_])
    rng = _gen("bootstrap")
    zl = {n: zt[i] for i, n in enumerate(test)}
    pk_of = {n: pk[i] for i, n in enumerate(test)}
    y_of = {n: d.c[n] for n in test}
    reps = {"auc_l": [], "auc_k": [], "ll_l": [], "ll_k": [], "ll_c": [], "ece_l": []}
    g = len(groups)
    for _ in range(BOOT_RESAMPLES):
        rows_ = []
        for _ in range(g):
            rows_.extend(groups[rng.integer(g)])
        yy = [y_of[n] for n in rows_]
        zz = [zl[n] for n in rows_]
        kk = [pk_of[n] for n in rows_]
        reps["auc_l"].append(auc_rank(zz, yy))
        reps["auc_k"].append(auc_rank(kk, yy))
        reps["ll_l"].append(log_loss_from_logits(zz, yy))
        reps["ll_k"].append(log_loss_from_probs(kk, yy))
        reps["ll_c"].append(log_loss_from_probs([pbar] * len(yy), yy))
        reps["ece_l"].append(ece([sigmoid(v) for v in zz], yy))
    def row(name, est, rp):
        lo, hi = interval(rp)
        return (name, _f(est), _f(sd(rp)), _f(lo), _f(hi))
    est_l = log_loss_from_logits(zt, yt)
    est_k = log_loss_from_probs(pk, yt)
    est_c = log_loss_from_probs(ent(pbar), yt)
    rows = [row("logistic, ROC AUC", auc_rank(zt, yt), reps["auc_l"]),
            row("logistic, log loss", est_l, reps["ll_l"]),
            row("logistic, ECE", ece(pl, yt), reps["ece_l"]),
            row("log loss, logistic minus constant", est_l - est_c, [a_ - b_ for a_, b_ in zip(reps["ll_l"], reps["ll_c"])]),
            row("log loss, logistic minus k-NN", est_l - est_k, [a_ - b_ for a_, b_ in zip(reps["ll_l"], reps["ll_k"])]),
            row("ROC AUC, logistic minus k-NN", auc_rank(zt, yt) - auc_rank(pk, yt),
                [a_ - b_ for a_, b_ in zip(reps["auc_l"], reps["auc_k"])])]
    out += ("\n\nCluster bootstrap over the 60 test pumps: " + _seed_line("bootstrap") +
            f". One resample is 60 calls of rng.integer(60), in order, each the index (in ascending pump order) of a pump whose "
            f"four rows are all taken, the pumps in draw order and the rows of a pump ascending; {BOOT_RESAMPLES} resamples from one "
            "generator, all six quantities of the table and the quantities they are differences of recomputed on the same "
            "resample (so the differences are paired). Interval ends are the `percentile` of the replicates at q = 0.025 and "
            "q = 0.975 (linear interpolation at position q (number of replicates - 1)); sd is the standard deviation of the "
            f"replicates with divisor {BOOT_RESAMPLES} - 1.\n\n")
    out += _table(["quantity", "estimate", "sd of replicates", "95 percent lower", "95 percent upper"], rows)
    return out


def regression_test():
    sp, m, s, Ztr, ttr, wn, wg = _regression()
    test = sp.test
    tte = _ys(test, "wear")
    tbar = mean(ttr)
    jt, _ = _best_feature(sp.train, "wear")
    xs = [x[jt] for x in _xs(sp.train)]
    mx = mean(xs)
    slope = sum((a - mx) * (b - tbar) for a, b in zip(xs, ttr)) / sum((a - mx) ** 2 for a in xs)
    icpt = tbar - slope * mx
    pred_full = predict_linear(wn, standardise(_xs(test), m, s))
    pred_line = [icpt + slope * x[jt] for x in _xs(test)]
    pred_const = [tbar] * len(test)
    groups = rows_by_pump(test)
    pos = {n: i for i, n in enumerate(test)}
    rng = _gen("bootstrap_regression")
    reps = {"full": [], "line": [], "diff": []}
    for _ in range(BOOT_RESAMPLES):
        rows_ = []
        for _ in range(len(groups)):
            rows_.extend(groups[rng.integer(len(groups))])
        idx = [pos[n] for n in rows_]
        y_ = [tte[i] for i in idx]
        rf = rmse([pred_full[i] for i in idx], y_)
        rl = rmse([pred_line[i] for i in idx], y_)
        reps["full"].append(rf)
        reps["line"].append(rl)
        reps["diff"].append(rf - rl)
    def row(name, pred, key=None):
        r = rmse(pred, tte)
        if key is None:
            return (name, _f(r), _f(r_squared(pred, tte, tbar)), "-", "-")
        lo, hi = interval(reps[key])
        return (name, _f(r), _f(r_squared(pred, tte, tbar)), _f(lo), _f(hi))
    out = ("The 240 test rows, read once, with the weights of milestone 3 (fitted on the training rows only). Interval: the "
           "cluster bootstrap over the 60 test pumps as in the classification report (" + _seed_line("bootstrap_regression")
           + f", {BOOT_RESAMPLES} resamples, 60 calls of rng.integer(60) each, taken as in the classification report; percentile "
           "interval of the RMSE at q = 0.025 and 0.975).\n\n")
    out += _table(["model", "test RMSE", "R^2 (around the training mean)", "RMSE 95 percent lower", "RMSE 95 percent upper"],
                  [row("constant (training mean)", pred_const),
                   row(f"line on x{jt}", pred_line, "line"),
                   row("linear regression on x0 .. x4", pred_full, "full")], "lrrrr")
    lo, hi = interval(reps["diff"])
    out += (f"\n\nRMSE of the linear regression minus RMSE of the line, same resamples: {rmse(pred_full, tte) - rmse(pred_line, tte):+.4f}, "
            f"95 percent interval {lo:+.4f} to {hi:+.4f}.")
    return out


# ---------------------------------------------- the multi-seed study

@lru_cache(maxsize=None)
def _seed_runs():
    master = _gen("seeds")
    seeds = [master.next_u64() for _ in range(N_RUNS)]
    d = data()
    runs = []
    for s in seeds:
        rng = bookrng.SplitMix64(s)
        sp = Split(rng)                                   # shuffle of 300 pumps, then the grouped folds (179 draws)
        row_fold = fold_assignment(len(sp.train), K_FOLDS, rng)   # then the row folds (719 draws)
        yte, ysl = _ys(sp.test), _ys(sp.slice_b)
        pbar = mean(_ys(sp.train))
        rec = {"seed": s, "pbar": pbar}
        for tag, folds in (("group", sp.fold), ("row", row_fold)):
            cvl = cv_logistic(sp.train, folds)
            lam = best_of(cvl)
            cvk = cv_knn(sp.train, folds)
            k = best_of(cvk)
            model = Fitted(sp, lam)
            zt = model.logits(sp.test)
            pk = knn_scores(sp, k, sp.test)
            rec[tag] = {"lam": lam, "k": k, "cv_log": mean(cvl[lam]), "cv_knn": mean(cvk[k]),
                        "test_log": log_loss_from_logits(zt, yte), "test_knn": log_loss_from_probs(pk, yte),
                        "auc_log": auc_rank(zt, yte), "auc_knn": auc_rank(pk, yte)}
            if tag == "group":
                a_scale = fit_scale(model.logits(sp.val), _ys(sp.val))
                rec["test_scaled"] = log_loss_from_logits([a_scale * v for v in zt], yte)
                rec["scale"] = a_scale
                zs = model.logits(sp.slice_b)
                rec["slice_log"] = log_loss_from_logits(zs, ysl)
                rec["slice_auc"] = auc_rank(zs, ysl)
                rec["slice_knn"] = log_loss_from_probs(knn_scores(sp, k, sp.slice_b), ysl)
                rec["const_test"] = log_loss_from_probs([pbar] * len(yte), yte)
                rec["const_slice"] = log_loss_from_probs([pbar] * len(ysl), ysl)
        runs.append(rec)
    return runs


def _boot_mean_interval(values, name):
    rng = _gen(name)
    n = len(values)
    reps = []
    for _ in range(RUN_BOOT_RESAMPLES):
        reps.append(sum(values[rng.integer(n)] for _ in range(n)) / n)
    reps.sort()
    return interval(reps)


def seed_study():
    runs = _seed_runs()
    master = _gen("seeds")
    out = (f"{N_RUNS} runs. Master generator: {_seed_line('seeds')}; run s uses the s-th raw value (rng.next(), counting from 0) "
           "as the seed of its own SplitMix64, which draws, in order: the shuffle of the 300 site A pump numbers (299 integer "
           "draws; 180 pumps train, 60 validate, 60 test, each set sorted ascending), the shuffle of the 180 training pumps for the "
           "grouped folds (179 integer draws; the pumps numbered 0 .. 179 in ascending pump order, the pump at shuffled position p "
           "goes to fold p mod 5), and the shuffle of the 720 training rows for the row folds used only by the ablation below "
           "(719 integer draws; the rows numbered 0 .. 719 in ascending row number, the row at shuffled position p goes to "
           "fold p mod 5). The 719 draws are made in every run, whether or not the ablation is read. In each run the penalty "
           "lambda and the neighbour count k are chosen by grouped five-fold cross-validation on the mean fold log loss (the "
           "smallest mean; the larger value on a tie, for lambda and for k alike), the models are refitted on the 720 training "
           "rows and scored on the 240 test rows and on the 240 rows of site B. The scale a of 'scaled' is fitted by the "
           "doubling-and-bisection routine of milestone 5 on the 240 validation rows of that run, and the 'constant' model is "
           "the share of ones among that run's 720 training rows.\n\n")
    rows = []
    for i, r in enumerate(runs):
        g = r["group"]
        rows.append((i, f"{r['seed']:#018x}", f"{g['lam']:g}", g["k"], _f(g["test_log"]), _f(g["test_knn"]), _f(r["const_test"]),
                     _f(g["auc_log"]), _f(g["auc_knn"]), _f(r["slice_log"]), _f(r["slice_auc"])))
    out += _table(["run", "seed", "lambda", "k", "test log loss, logistic", "k-NN", "constant", "test AUC, logistic", "k-NN",
                   "site B log loss", "site B AUC"], rows, "lllrrrrrrrr")

    def summary(name, vals, boot_name):
        lo, hi = _boot_mean_interval(vals, boot_name)
        return (name, _fs(mean(vals)) if "minus" in name else _f(mean(vals)), _f(sd(vals)), _f(lo), _f(hi),
                f"{_seed(boot_name):#018x}")

    col = lambda key: [r["group"][key] for r in runs]
    rows = [summary("test log loss, logistic", col("test_log"), "seeds.ll"),
            summary("test log loss, scaled minus raw logistic", [r["test_scaled"] - r["group"]["test_log"] for r in runs], "seeds.sc"),
            summary("test log loss, logistic minus constant", [a - r["const_test"] for a, r in zip(col("test_log"), runs)], "seeds.llc"),
            summary("test log loss, logistic minus k-NN", [a - b for a, b in zip(col("test_log"), col("test_knn"))], "seeds.llk"),
            summary("test ROC AUC, logistic", col("auc_log"), "seeds.auc"),
            summary("test ROC AUC, logistic minus k-NN", [a - b for a, b in zip(col("auc_log"), col("auc_knn"))], "seeds.auck"),
            summary("site B log loss, logistic", [r["slice_log"] for r in runs], "seeds.sl"),
            summary("site B log loss, logistic minus constant", [r["slice_log"] - r["const_slice"] for r in runs], "seeds.slc"),
            summary("site B ROC AUC, logistic", [r["slice_auc"] for r in runs], "seeds.sauc")]
    out += (f"\n\nOver the {N_RUNS} runs: mean, standard deviation (divisor n - 1) and a bootstrap 95 percent interval of the mean "
            f"(resample the {N_RUNS} runs with replacement, {RUN_BOOT_RESAMPLES} resamples, interval ends at the 2.5th and "
            "97.5th percentiles of the replicated means; each row has its own generator, started at the seed in the last column; a resample is "
            f"{N_RUNS} calls of rng.integer({N_RUNS}), each the index of the next run to include).\n\n")
    out += _table(["quantity", "mean", "sd", "95 percent lower", "95 percent upper", "bootstrap seed"], rows, "lrrrrl")
    wins = sum(1 for a, b in zip(col("test_log"), col("test_knn")) if a < b)
    out += f"\n\nRuns in which the logistic model has the smaller test log loss than k-NN: {wins} of {N_RUNS}."
    return out


def ablation_cv():
    runs = _seed_runs()
    rows = []
    for name, key in (("grouped folds (by pump)", "group"), ("random row folds", "row")):
        for model, cv_key, test_key in (("logistic", "cv_log", "test_log"), ("k-NN", "cv_knn", "test_knn")):
            cv = [r[key][cv_key] for r in runs]
            te = [r[key][test_key] for r in runs]
            gap = [a - b for a, b in zip(cv, te)]
            lo, hi = _boot_mean_interval(gap, f"ablation.{key}.{model}")
            rows.append((name, model, _f(mean(cv)), _f(mean(te)), _fs(mean(gap)), _f(lo), _f(hi),
                         f"{_seed(f'ablation.{key}.{model}'):#018x}"))
    out = (f"The {N_RUNS} runs of the seed study. The design choice: how the folds that select lambda (logistic) and k (k-NN) "
           "are drawn (grouped folds: the folds of pumps of the seed study; random row folds: the 720 training rows dealt by the "
           "row-fold shuffle of the seed study). The chosen value is the one with the smallest mean fold log loss over the five "
           "folds of the drawn kind (the larger value on a tie). 'CV estimate' is that mean fold log loss of the chosen value, the number a report would claim; 'test' is "
           "the log loss of the refitted model on the 240 test rows of the same run, from pumps never seen; gap = CV estimate "
           "minus test. The interval is a bootstrap 95 percent interval of the mean gap over the runs (2000 resamples as in the "
           "seed study, each row with the generator seed of the last column).\n\n")
    out += _table(["folds", "model", "mean CV estimate", "mean test log loss", "mean gap", "95 percent lower", "95 percent upper",
                   "bootstrap seed"], rows, "llrrrrrl")
    g, r = runs, runs
    for model, cv_key, test_key in (("logistic", "cv_log", "test_log"), ("k-NN", "cv_knn", "test_knn")):
        diff = [a["row"][test_key] - a["group"][test_key] for a in runs]
        lo, hi = _boot_mean_interval(diff, f"ablation.test.{model}")
        out += (f"\n\nTest log loss of the {model} model chosen with row folds minus the one chosen with grouped folds, over the "
                f"runs: mean {_fs(mean(diff), 5)}, 95 percent interval {_fs(lo, 5)} to {_fs(hi, 5)} (bootstrap seed "
                f"{_seed(f'ablation.test.{model}'):#018x})"
                + (f"; chosen lambda differs in {sum(1 for a in runs if a['row']['lam'] != a['group']['lam'])} runs."
                   if model == "logistic" else
                   f"; chosen k differs in {sum(1 for a in runs if a['row']['k'] != a['group']['k'])} runs."))
    return out


# ------------------------------------------------ the shift slice

def shift_slice():
    sp, model, a, k, zt, pk = _test_scores()
    d = data()
    yt = _ys(sp.test)
    sl = sp.slice_b
    ysl = _ys(sl)
    zs = model.logits(sl)
    pbar = mean(_ys(sp.train))
    rows = []
    for name, rws, y, z in (("test (site A)", sp.test, yt, zt), ("shift slice (site B)", sl, ysl, zs)):
        p = [sigmoid(v) for v in z]
        pk_ = knn_scores(sp, k, rws)
        rows.append((name, len(rws), _f(sum(y) / len(y)), _f(mean(p)), _f(log_loss_from_logits(z, y)),
                     _f(log_loss_from_probs([pbar] * len(y), y)), _f(log_loss_from_probs(pk_, y)),
                     _f(auc_rank(z, y)), _f(auc_rank(pk_, y)), _f(ece(p, y))))
    out = ("The model of the report (logistic, no recalibration) on the test rows and on the 240 rows of site B, which "
           "no earlier step looked at.\n\n")
    out += _table(["rows", "n", "observed share fail = 1", "mean predicted p", "log loss, logistic", "log loss, constant",
                   "log loss, k-NN", "ROC AUC, logistic", "ROC AUC, k-NN", "ECE, logistic"], rows)
    # reliability of the slice
    ps = [sigmoid(v) for v in zs]
    out += "\n\nReliability of the logistic probabilities on the shift slice and on the test rows (10 equal-width bins; empty bins omitted):\n\n"
    bt = reliability_bins([sigmoid(v) for v in zt], yt)
    bs = reliability_bins(ps, ysl)
    rws = []
    for i in range(10):
        ct, mt, ft = bt[i]
        cs, ms, fs = bs[i]
        if ct or cs:
            rws.append((f"{i / 10:.1f} to {(i + 1) / 10:.1f}", ct, "-" if not ct else _f(mt), "-" if not ct else _f(ft),
                        cs, "-" if not cs else _f(ms), "-" if not cs else _f(fs)))
    out += _table(["bin of p", "test rows", "mean p", "observed", "site B rows", "mean p", "observed"], rws)
    # feature shift
    Ztr = standardise(_xs(sp.train), model.m, model.s)
    Zs = standardise(_xs(sl), model.m, model.s)
    out += "\n\nWhere the slice differs: the mean of each standardised feature (training statistics) over the training rows and over the slice.\n\n"
    out += _table(["feature", "training rows", "site B rows"],
                  [(f"x{j} ({FEATURES[j]})", _f(mean([z[j] for z in Ztr])), _f(mean([z[j] for z in Zs]))) for j in range(D)])
    # share of slice rows outside training range of x0
    lo0 = min(x[0] for x in _xs(sp.train))
    hi0 = max(x[0] for x in _xs(sp.train))
    outside = sum(1 for x in _xs(sl) if x[0] > hi0 or x[0] < lo0)
    above = sum(1 for x in _xs(sl) if x[0] > 2.0)
    train_above = sum(1 for x in _xs(sp.train) if x[0] > 2.0)
    out += (f"\n\nRange of the raw (unstandardised) x0 over the training rows: {lo0:.4f} to {hi0:.4f}; site B rows outside it: "
            f"{outside} of {len(sl)}. Rows with raw x0 > 2: {train_above} of {len(sp.train)} training rows, {above} of "
            f"{len(sl)} site B rows.")
    return out


# ------------------------------------------------ the acceptance fingerprint

def acceptance_fingerprint():
    sp, model, a, k, zt, pk = _test_scores()
    d = data()
    yt = _ys(sp.test)
    cv = main_cv()
    wn = _regression()[5]
    rows = [
        ("1", "sum of x0 over all 1440 rows", f"{sum(x[0] for x in d.x):+.6f}", "5e-7"),
        ("1", "rows with fail = 1 over all 1440 rows", sum(d.c), "exact"),
        ("1", "training pumps, first six", ", ".join(str(g) for g in sp.pumps[0][:6]), "exact"),
        ("3", "normal-equation weights, intercept and z0", f"{wn[0]:.8f}, {wn[1]:.8f}", "1e-8"),
        ("4", "chosen lambda; mean CV log loss at lambda = 0", f"{best_of(cv):g}; {mean(cv[0.0]):.6f}", "exact; 1e-6"),
        ("4", "logistic weights, intercept and z0", f"{model.w[0]:.8f}, {model.w[1]:.8f}", "1e-6"),
        ("5", "scale a fitted on the validation rows", f"{a:.8f}", "1e-6"),
        ("6", "chosen k", k, "exact"),
        ("7", "test RMSE, linear regression", f"{rmse(predict_linear(wn, standardise(_xs(sp.test), *fit_standardiser(_xs(sp.train)))), _ys(sp.test, 'wear')):.6f}", "1e-6"),
        ("7", "test ROC AUC, logistic", f"{auc_rank(zt, yt):.6f}", "1e-6"),
        ("7", "test log loss, logistic", f"{log_loss_from_logits(zt, yt):.6f}", "1e-6"),
    ]
    return _table(["milestone", "quantity", "value", "tolerance"], rows, "llrl")
