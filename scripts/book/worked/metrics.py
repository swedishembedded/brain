# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Measuring Models: Metrics and Calibration".

Plain Python 3, standard library only, written from the chapter's own
description: the confusion matrix and accuracy under class imbalance,
precision and recall as conditional probabilities, a threshold sweep, the
F1 score and its harmonic mean, the ROC curve and its area by the trapezoid
rule and as a pairwise (rank) probability, the dependence of the precision-
recall curve (and the independence of the ROC curve) on the share of
positives, reliability bins and the expected calibration error, temperature
scaling, regression and language-model metrics, the choice of a threshold
from the costs of the two errors, and a bootstrap interval for a metric.
Each public function without a leading underscore that takes no arguments
returns a markdown string that the chapter embeds between
`<!-- worked: metrics.<name> -->` markers. Underscore functions and the
constants are shared with the chapter's figures, so a picture and the table
beside it cannot disagree.

The data are one imbalanced screening problem. A hidden law (written here,
and only here, because the book plays the part of nature) decides from twenty
measurements whether a part is defective; only two of the twenty matter. A
logistic model is fitted to a small training set, which is exactly the
situation in which it becomes overconfident.

Swedish Embedded AB implements solutions for deciding whether a classifier is
good enough to act on, and saying so honestly. If your team needs expertise in
evaluation metrics, calibration and threshold selection, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache

import bookrng

# ------------------------------------------------------------- the source

SCREEN_NAME = "metrics.screen"
D = 20                       # measurements per part
LAW = (-4.5, 2.5, 1.8)       # P(defective) = sigmoid(LAW[0] + LAW[1] x1 + LAW[2] x2)
N_TRAIN, N_VAL, N_TEST = 100, 1000, 3000
STEPS, ETA = 3000, 0.5       # plain gradient descent from w = 0

BOOT_NAME = "metrics.bootstrap_intervals"
BOOT_RESAMPLES = 1000
BOOT_LEVEL = 0.95

COST_FP, COST_FN = 1.0, 10.0   # cost of a good part rejected, and of a defective part passed


# ---------------------------------------------------------------- helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _generator_line(name):
    seed = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed})."


def sigmoid(z):
    """The two-branch form: the exponent is never positive."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def softplus(z):
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def logit(p):
    return math.log(p / (1.0 - p))


def _percentile(sorted_values, q):
    """Linear interpolation at position q (R - 1) of an ascending list."""
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


# ------------------------------------------------------------ the screen

def _draw(rng, n):
    rows = []
    for _ in range(n):
        x = [rng.normal() for _ in range(D)]
        u = rng.uniform()
        y = 1 if u < sigmoid(LAW[0] + LAW[1] * x[0] + LAW[2] * x[1]) else 0
        rows.append((x, y))
    return rows


@lru_cache(maxsize=None)
def _splits():
    """(train, validation, test): lists of (x, y), drawn in that order from one generator."""
    rng = bookrng.generator(SCREEN_NAME)
    train = _draw(rng, N_TRAIN)
    val = _draw(rng, N_VAL)
    test = _draw(rng, N_TEST)
    return train, val, test


def _logit_of(w, x):
    return w[0] + sum(wj * xj for wj, xj in zip(w[1:], x))


@lru_cache(maxsize=None)
def _fit():
    """Weights (intercept first) of the logistic model after STEPS plain gradient steps from zero."""
    train, _, _ = _splits()
    n = len(train)
    w = [0.0] * (D + 1)
    for _ in range(STEPS):
        g = [0.0] * (D + 1)
        for x, y in train:
            r = sigmoid(_logit_of(w, x)) - y
            g[0] += r
            for j in range(D):
                g[j + 1] += r * x[j]
        w = [wi - ETA * gi / n for wi, gi in zip(w, g)]
    return tuple(w)


@lru_cache(maxsize=None)
def _scores():
    """(z_val, y_val, z_test, y_test, z_train, y_train): the logits z = w.x and the labels."""
    train, val, test = _splits()
    w = _fit()
    f = lambda rows: ([_logit_of(w, x) for x, _ in rows], [y for _, y in rows])
    return (*f(val), *f(test), *f(train))


# --------------------------------------------------------------- metrics

def _confusion(pred, y):
    """(TP, FP, FN, TN) of boolean predictions against 0/1 labels."""
    tp = sum(1 for p, t in zip(pred, y) if p and t == 1)
    fp = sum(1 for p, t in zip(pred, y) if p and t == 0)
    fn = sum(1 for p, t in zip(pred, y) if not p and t == 1)
    tn = sum(1 for p, t in zip(pred, y) if not p and t == 0)
    return tp, fp, fn, tn


def _at_prob(z, y, t):
    """Confusion counts when a part is called defective when sigmoid(z) >= t."""
    return _confusion([sigmoid(v) >= t for v in z], y)


def _prf(tp, fp, fn, tn):
    """(precision, recall, F1, accuracy); None where a denominator is zero."""
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return prec, rec, f1, (tp + tn) / (tp + fp + fn + tn)


def _fmt(v, digits=4):
    return "undefined" if v is None else f"{v:.{digits}f}"


def _roc_points(score, y):
    """ROC points (FPR, TPR) from the threshold above every score downward; tied scores move together."""
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


def _trapezoid(pts):
    return sum((x1 - x0) * (y0 + y1) / 2.0 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))


def _auc_pairs(score, y, tie=0.5):
    """Share of (positive, negative) pairs ranked correctly; a tie scores `tie`."""
    ps = [s for s, t in zip(score, y) if t == 1]
    ns = [s for s, t in zip(score, y) if t == 0]
    total = 0.0
    for p in ps:
        for q in ns:
            total += 1.0 if p > q else (tie if p == q else 0.0)
    return total / (len(ps) * len(ns))


def _auc_ranksum(score, y):
    """(R_pos - P (P + 1) / 2) / (P N), ranks 1-based ascending, tied scores sharing their average rank."""
    n = len(score)
    order = sorted(range(n), key=lambda i: score[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rank[order[k]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = sum(y)
    neg = n - pos
    r_pos = sum(r for r, t in zip(rank, y) if t == 1)
    return (r_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _average_precision(score, y, neg_weight=1.0):
    """Sum over distinct thresholds of (recall step) x (precision there); a negative counts `neg_weight`."""
    pos = sum(y)
    order = sorted(range(len(y)), key=lambda i: -score[i])
    tp = fp = 0.0
    prev_recall = ap = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and score[order[j]] == score[order[i]]:
            if y[order[j]] == 1:
                tp += 1
            else:
                fp += neg_weight
            j += 1
        recall = tp / pos
        ap += (recall - prev_recall) * tp / (tp + fp)
        prev_recall = recall
        i = j
    return ap


def _pr_points(score, y):
    """(recall, precision) after each distinct threshold, from the highest downward."""
    pos = sum(y)
    order = sorted(range(len(y)), key=lambda i: -score[i])
    pts = []
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
        pts.append((tp / pos, tp / (tp + fp)))
        i = j
    return pts


def _nll(z, y, a=1.0, b=0.0):
    """Mean binary cross-entropy of sigmoid(a z + b) against labels, from the logit."""
    return sum(softplus(a * v + b) - t * (a * v + b) for v, t in zip(z, y)) / len(y)


def _brier(p, y):
    return sum((q - t) ** 2 for q, t in zip(p, y)) / len(y)


def _bins(p, y, n_bins):
    """Equal-width bins of the predicted probability: (lo, hi, count, mean p, observed frequency)."""
    cnt = [0] * n_bins
    sp = [0.0] * n_bins
    sy = [0] * n_bins
    for q, t in zip(p, y):
        k = min(int(q * n_bins), n_bins - 1)
        cnt[k] += 1
        sp[k] += q
        sy[k] += t
    return [(k / n_bins, (k + 1) / n_bins, cnt[k], sp[k] / cnt[k] if cnt[k] else None,
             sy[k] / cnt[k] if cnt[k] else None) for k in range(n_bins)]


def _ece(p, y, n_bins=10):
    n = len(y)
    return sum(c / n * abs(f - m) for _, _, c, m, f in _bins(p, y, n_bins) if c)


def _ece_equal_mass(p, y, n_bins=10):
    n = len(y)
    order = sorted(range(n), key=lambda i: p[i])
    total = 0.0
    for b in range(n_bins):
        idx = order[b * n // n_bins:(b + 1) * n // n_bins]
        if idx:
            total += len(idx) / n * abs(sum(y[i] for i in idx) / len(idx) - sum(p[i] for i in idx) / len(idx))
    return total


def _fit_scale(z, y):
    """The scale a > 0 minimising the cross-entropy of sigmoid(a z): bisection on the (increasing) slope."""
    def slope(a):
        return sum((sigmoid(a * v) - t) * v for v, t in zip(z, y)) / len(y)
    if slope(0.0) >= 0.0:
        raise ValueError("the logits do not rank the positives above the negatives: no positive scale helps")
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


def _fit_shift_scale(z, y):
    """(a, b) minimising the cross-entropy of sigmoid(a z + b): Newton's method from (1, 0), halving a step that does not lower the loss."""
    a, b = 1.0, 0.0
    n = len(y)
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for v, t in zip(z, y):
            p = sigmoid(a * v + b)
            r, s = p - t, p * (1.0 - p)
            ga += r * v
            gb += r
            haa += s * v * v
            hab += s * v
            hbb += s
        ga, gb, haa, hab, hbb = ga / n, gb / n, haa / n, hab / n, hbb / n
        det = haa * hbb - hab * hab
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        base = _nll(z, y, a, b)
        step = 1.0
        while step > 1e-6 and _nll(z, y, a - step * da, b - step * db) > base:
            step /= 2.0
        a, b = a - step * da, b - step * db
        if abs(step * da) + abs(step * db) < 1e-12:
            break
    return a, b


# --------------------------------------------------- accuracy and imbalance

def accuracy_under_imbalance():
    """100000 parts, 1% defective: a detector with 90% recall and 10% false alarms against 'never alarm'."""
    n, prev, tpr, fpr = 100_000, 0.01, 0.90, 0.10
    pos = round(n * prev)
    neg = n - pos
    tp = round(pos * tpr)
    fp = round(neg * fpr)
    fn, tn = pos - tp, neg - fp
    rows_conf = [("defective", pos, tp, fn), ("good", neg, fp, tn), ("total", n, tp + fp, fn + tn)]
    out = [f"{n} parts, {prev:.0%} defective ({pos}); the detector alarms on {tpr:.0%} of defective parts and on "
           f"{fpr:.0%} of good ones, counts rounded to whole parts.", ""]
    out.append(_table(["", "parts", "alarm (predicted defective)", "no alarm"], rows_conf))
    out.append("")
    p_a, r_a, f_a, acc_a = _prf(tp, fp, fn, tn)
    p_n, r_n, f_n, acc_n = _prf(0, 0, pos, neg)
    rows = [("detector", f"{acc_a:.4f}", _fmt(p_a), _fmt(r_a), _fmt(f_a)),
            ("never alarm", f"{acc_n:.4f}", _fmt(p_n), _fmt(r_n), _fmt(f_n))]
    out.append(_table(["rule", "accuracy", "precision", "recall", "F1"], rows))
    bayes = r_a * prev / (r_a * prev + fpr * (1 - prev))
    out.append("")
    out.append(f"Precision by Bayes' rule, recall x prevalence / (recall x prevalence + false alarm rate x (1 - prevalence)) "
               f"= {bayes:.4f}. F1 is computed from the counts, 2 TP / (2 TP + FP + FN), which is defined whenever there is a "
               f"defective part or an alarm: for 'never alarm' it is 0 / {2 * 0 + 0 + pos} = 0 although its precision is undefined.")
    return "\n".join(out)


def screen_data():
    """The screening data set: sizes, positives, and the model fitted to the training rows."""
    train, val, test = _splits()
    w = _fit()
    z_val, y_val, z_test, y_test, z_train, y_train = _scores()
    head = (_generator_line(SCREEN_NAME) + "\n"
            f"Draw order, part by part: x1, ..., x{D} = rng.normal() in that order (so ten Box-Muller pairs, none held over), "
            f"then u = rng.uniform(); y = 1 (defective) when u < sigmoid({LAW[0]} + {LAW[1]} x1 + {LAW[2]} x2). "
            f"{N_TRAIN} training parts, then {N_VAL} validation parts, then {N_TEST} test parts, all from this one generator.\n\n")
    rows = []
    for name, ys in (("training", y_train), ("validation", y_val), ("test", y_test)):
        rows.append((name, len(ys), sum(ys), f"{sum(ys) / len(ys):.4f}"))
    t1 = _table(["split", "parts", "defective", "fraction defective"], rows)
    norm = math.sqrt(sum(v * v for v in w))
    fit = (f"\n\nModel: logistic regression with an intercept and all {D} measurements (no penalty), weights set to zero, "
           f"{STEPS} plain gradient steps of size {ETA} on the mean cross-entropy of the {N_TRAIN} training parts. "
           f"The norm of the {D + 1} weights is {norm:.4f}; the weight of x1 is {w[1]:.4f}, of x2 is {w[2]:.4f}, "
           f"and the largest weight of the other {D - 2} measurements is {max(abs(v) for v in w[3:]):.4f} in absolute value. "
           f"Training cross-entropy {_nll(z_train, y_train):.4f}, validation cross-entropy {_nll(z_val, y_val):.4f}.")
    return head + t1 + fit


# ---------------------------------------------------- precision and recall

def threshold_sweep():
    """Precision, recall, F1 and accuracy on the validation parts as the probability threshold moves."""
    z, y, *_ = _scores()
    pos = sum(y)
    rows = []
    for t in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999):
        tp, fp, fn, tn = _at_prob(z, y, t)
        p, r, f, acc = _prf(tp, fp, fn, tn)
        rows.append((f"{t}", f"{logit(t):+.3f}", tp, fp, fn, tn, _fmt(p), _fmt(r), _fmt(f), f"{acc:.4f}"))
    head = (f"The {len(y)} validation parts ({pos} defective). A part is called defective when sigmoid(z) >= t.\n\n")
    return head + _table(["t", "logit(t)", "TP", "FP", "FN", "TN", "precision", "recall", "F1", "accuracy"], rows)


def f1_means():
    """Arithmetic mean, harmonic mean (F1) and minimum of precision and recall for six pairs."""
    pairs = [(0.9, 0.9), (0.5, 0.5), (1.0, 0.5), (1.0, 0.1), (1.0, 0.01), (0.2, 0.8)]
    rows = []
    for p, r in pairs:
        rows.append((f"{p}", f"{r}", f"{(p + r) / 2:.4f}", f"{2 * p * r / (p + r):.4f}", f"{min(p, r)}"))
    t1 = _table(["precision", "recall", "arithmetic mean", "harmonic mean F1", "min"], rows)
    z, y, *_ = _scores()
    tp, fp, fn, tn = _at_prob(z, y, 0.5)
    p, r, f, acc = _prf(tp, fp, fn, tn)
    check = (f"\n\nValidation parts at t = 0.5: TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}. "
             f"F1 from the two ratios 2 P R / (P + R) = {2 * p * r / (p + r):.6f}; from the counts 2 TP / (2 TP + FP + FN) = "
             f"{f:.6f}.")
    big = 1_000_000
    p2, r2, f2, acc2 = _prf(tp, fp, fn, tn + big)
    tail = (f"\nAdd {big} more good parts, all correctly passed (TN becomes {tn + big}): precision {_fmt(p2)}, recall {_fmt(r2)}, "
            f"F1 {_fmt(f2)} are unchanged, accuracy goes from {acc:.4f} to {acc2:.6f}.")
    return t1 + check + tail


# ------------------------------------------------------------ the ROC curve

_HAND_SCORES = [(0.95, 1), (0.85, 1), (0.80, 0), (0.70, 1), (0.60, 1), (0.60, 0), (0.40, 0), (0.30, 0)]


def roc_by_hand():
    """Eight scored parts, one tie: the ROC points, the trapezoid area and the pair count."""
    s = [a for a, _ in _HAND_SCORES]
    y = [b for _, b in _HAND_SCORES]
    pts = _roc_points(s, y)
    thr = sorted(set(s), reverse=True)
    rows = [("above all", 0, 0, "0.00", "0.00")]
    for t, (fpr, tpr) in zip(thr, pts[1:]):
        tp = round(tpr * sum(y))
        fp = round(fpr * (len(y) - sum(y)))
        rows.append((f"{t:.2f}", tp, fp, f"{fpr:.2f}", f"{tpr:.2f}"))
    seg = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg.append(f"({x1 - x0:.2f} x {(y0 + y1) / 2:.3f}) = {(x1 - x0) * (y0 + y1) / 2:.5f}")
    head = ("Scores and labels, highest first: " + ", ".join(f"{a:.2f}{'+' if b else '-'}" for a, b in _HAND_SCORES) +
            ". '+' is defective (4 parts), '-' is good (4 parts). A part is called defective when its score is at or "
            "above the threshold; tied scores cross it together.\n\n")
    t1 = _table(["threshold", "TP", "FP", "FPR", "TPR"], rows)
    area = _trapezoid(pts)
    pos_s = [a for a, b in _HAND_SCORES if b == 1]
    neg_s = [a for a, b in _HAND_SCORES if b == 0]
    greater = sum(1 for p in pos_s for q in neg_s if p > q)
    equal = sum(1 for p in pos_s for q in neg_s if p == q)
    tail = ("\n\nTrapezoids, width x average height: " + "; ".join(seg) + f".\nArea = {area:.5f}.\n"
            f"Pairs: {len(pos_s)} x {len(neg_s)} = {len(pos_s) * len(neg_s)}; positive above negative in {greater}, tied in {equal}, "
            f"below in {len(pos_s) * len(neg_s) - greater - equal}; (greater + equal / 2) / pairs = "
            f"{(greater + equal / 2) / (len(pos_s) * len(neg_s)):.5f}.\n"
            f"Rank sum: {_auc_ranksum(s, y):.5f}.")
    return head + t1 + tail


def auc_three_ways():
    """The test-set AUC by trapezoid, by counting pairs and by ranks; with and without tied scores."""
    _, _, z, y, *_ = _scores()
    rows = []
    for name, sc in (("logits as computed", z), ("logits rounded to 1 decimal (ties)", [round(v, 1) for v in z])):
        trap = _trapezoid(_roc_points(sc, y))
        half = _auc_pairs(sc, y)
        rank = _auc_ranksum(sc, y)
        ties = sum(1 for p, t in zip(sc, y) if t == 1 for q, u in zip(sc, y) if u == 0 and p == q)
        rows.append((name, ties, f"{trap:.8f}", f"{half:.8f}", f"{rank:.8f}",
                     f"{_auc_pairs(sc, y, 0.0):.8f}", f"{_auc_pairs(sc, y, 1.0):.8f}"))
    pos = sum(y)
    head = (f"The {len(y)} test parts ({pos} defective, {len(y) - pos} good, {pos * (len(y) - pos)} pairs). "
            "Ties counted as half in the first three columns; counted as 0 and as 1 in the last two.\n\n")
    return head + _table(["scores", "tied pairs", "trapezoid", "pairs, tie = 1/2", "rank sum", "pairs, tie = 0", "pairs, tie = 1"], rows)


def pr_vs_prevalence():
    """Same scores, other shares of positives: ROC AUC does not move, the PR curve does."""
    _, _, z, y, *_ = _scores()
    pos = sum(y)
    neg = len(y) - pos
    pts = _roc_points(z, y)
    k = next(i for i, (_, tpr) in enumerate(pts) if tpr >= 0.8)
    fpr, tpr = pts[k]
    # The threshold of that ROC point: the k-th distinct score from the top.
    thr = sorted(set(z), reverse=True)[k - 1]
    auc = _trapezoid(pts)
    rows = []
    for prev in (0.5, 0.2, pos / len(y), 0.05, 0.01, 0.001):
        w = (1 - prev) / prev * pos / neg          # weight of each good part so positives make up `prev`
        ap = _average_precision(z, y, w)
        prec = tpr * prev / (tpr * prev + fpr * (1 - prev))
        # weighted counts at the threshold agree with the formula
        tp, fp, _, _ = _confusion([v >= thr for v in z], y)
        prec_w = tp / (tp + w * fp)
        assert abs(prec - prec_w) < 1e-12
        label = f"{prev:.4f}" + (" (the test set)" if abs(prev - pos / len(y)) < 1e-12 else "")
        rows.append((label, f"{auc:.4f}", f"{ap:.4f}", f"{prec:.4f}"))
    head = (f"The {len(y)} test parts. Each good part is given the weight (1 - xi)/xi x {pos}/{neg} (each defective part has weight 1) so that the defective parts "
            f"make up a share xi (the base rate) of the weighted total. The operating point is the first threshold at which recall "
            f"reaches 0.8: logit {thr:.4f}, recall {tpr:.4f}, false positive rate {fpr:.4f}.\n\n")
    return head + _table(["base rate xi", "ROC AUC", "average precision", "precision at the operating point"], rows)


# -------------------------------------------------------------- calibration

def reliability_raw():
    """Reliability bins of the raw probabilities on the test parts, with ECE and the other summary scores."""
    _, _, z, y, ztr, ytr = _scores()
    p = [sigmoid(v) for v in z]
    n = len(y)
    rows = []
    for lo, hi, c, m, f in _bins(p, y, 10):
        rows.append((f"{lo:.1f} to {hi:.1f}", c, _fmt(m, 4) if c else "-", _fmt(f, 4) if c else "-",
                     f"{abs(f - m):.4f}" if c else "-"))
    extreme = sum(1 for q in p if q < 0.01 or q > 0.99)
    w = _fit()
    tail = (f"\n\nECE (10 equal-width bins) = {_ece(p, y):.4f}. Mean predicted probability {sum(p) / n:.4f}, "
            f"observed share of defective parts {sum(y) / n:.4f}. "
            f"Parts with p < 0.01 or p > 0.99: {extreme} of {n} (share {extreme / n:.4f}). "
            f"Cross-entropy {_nll(z, y):.4f}, squared-error probability score {_brier(p, y):.4f}. "
            f"For comparison, the predictions on the {len(ytr)} training parts have cross-entropy {_nll(ztr, ytr):.4f}.")
    head = f"The {n} test parts, 10 equal-width bins of the predicted probability of 'defective'.\n\n"
    return head + _table(["bin of p", "parts", "mean p", "observed share", "gap"], rows) + tail


def temperature_fit():
    """Fit the single scale a of sigmoid(a z) on the validation parts: the valley, the minimiser, what changes."""
    zv, yv, zt, yt, *_ = _scores()
    a = _fit_scale(zv, yv)
    grid = [0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    rows = [(f"{g}", f"{1 / g:.3f}", f"{_nll(zv, yv, g):.4f}") for g in grid]
    rows.append((f"{a:.4f} (fitted)", f"{1 / a:.3f}", f"{_nll(zv, yv, a):.4f}"))
    slope = sum((sigmoid(a * v) - t) * v for v, t in zip(zv, yv)) / len(yv)
    t1 = _table(["scale a", "temperature 1/a", "validation cross-entropy"], rows)
    head = (f"The {len(yv)} validation parts. The model is sigmoid(a z) with z the logit of the fitted model of "
            "`metrics.screen_data`; a is found by bisection on the slope of the cross-entropy.\n\n")
    tail = (f"\n\nSlope at the fitted a: absolute value below 1e-12 ({'yes' if abs(slope) < 1e-12 else 'NO'}; the exact residue is "
            f"rounding noise and depends on the summation order). Temperature = 1/a = {1 / a:.4f}.")
    # what changes and what does not, on the test parts
    pr = [sigmoid(v) for v in zt]
    ps = [sigmoid(a * v) for v in zt]
    acc = lambda pp: sum(1 for q, t in zip(pp, yt) if (q >= 0.5) == (t == 1)) / len(yt)
    inv = [("cross-entropy", f"{_nll(zt, yt):.4f}", f"{_nll(zt, yt, a):.4f}"),
           ("squared-error probability score", f"{_brier(pr, yt):.4f}", f"{_brier(ps, yt):.4f}"),
           ("ECE, 10 equal-width bins", f"{_ece(pr, yt):.4f}", f"{_ece(ps, yt):.4f}"),
           ("accuracy at t = 0.5", f"{acc(pr):.4f}", f"{acc(ps):.4f}"),
           ("ROC AUC", f"{_trapezoid(_roc_points(pr, yt)):.6f}", f"{_trapezoid(_roc_points(ps, yt)):.6f}")]
    t2 = "\n\nOn the test parts:\n\n" + _table(["quantity", "before (a = 1)", "after (a fitted)"], inv)
    return head + t1 + tail + t2


def reliability_scaled():
    """Reliability bins after temperature scaling, and the effect of also fitting a shift."""
    zv, yv, zt, yt, *_ = _scores()
    a = _fit_scale(zv, yv)
    a2, b2 = _fit_shift_scale(zv, yv)
    ps = [sigmoid(a * v) for v in zt]
    p2 = [sigmoid(a2 * v + b2) for v in zt]
    n = len(yt)
    rows = []
    for (lo, hi, c, m, f), (_, _, c2, m2, f2) in zip(_bins(ps, yt, 10), _bins(p2, yt, 10)):
        rows.append((f"{lo:.1f} to {hi:.1f}", c, _fmt(m) if c else "-", _fmt(f) if c else "-",
                     c2, _fmt(m2) if c2 else "-", _fmt(f2) if c2 else "-"))
    t1 = _table(["bin of p", "parts (scale)", "mean p", "observed", "parts (scale+shift)", "mean p", "observed"], rows)
    tail = (f"\n\nScale only: a = {a:.4f}, ECE = {_ece(ps, yt):.4f}, cross-entropy {_nll(zt, yt, a):.4f}. "
            f"Scale and shift, sigmoid(a z + b) fitted on the validation parts by Newton's method with step halving "
            f"(`fit_scale_shift` of the chapter: start (1, 0), at most 100 iterations, stop when |da| + |db| < 1e-12): "
            f"a = {a2:.4f}, b = {b2:.4f}, ECE = {_ece(p2, yt):.4f}, cross-entropy {_nll(zt, yt, a2, b2):.4f}. "
            f"Mean predicted probability: scale only {sum(ps) / n:.4f}, scale and shift {sum(p2) / n:.4f}, "
            f"observed {sum(yt) / n:.4f}.")
    return f"The {n} test parts, a and b fitted on the validation parts.\n\n" + t1 + tail


def ece_binning():
    """ECE under several binning rules: the model, its recalibration, and the true probabilities as a reference."""
    zv, yv, zt, yt, *_ = _scores()
    a2, b2 = _fit_shift_scale(zv, yv)
    _, _, test = _splits()
    p_raw = [sigmoid(v) for v in zt]
    p_fix = [sigmoid(a2 * v + b2) for v in zt]
    p_true = [sigmoid(LAW[0] + LAW[1] * x[0] + LAW[2] * x[1]) for x, _ in test]
    small = 300
    rules = (("5 equal-width bins", lambda p, y: _ece(p, y, 5)),
             ("10 equal-width bins", lambda p, y: _ece(p, y, 10)),
             ("20 equal-width bins", lambda p, y: _ece(p, y, 20)),
             ("10 equal-mass bins", lambda p, y: _ece_equal_mass(p, y, 10)),
             ("30 equal-mass bins", lambda p, y: _ece_equal_mass(p, y, 30)),
             ("7 equal-mass bins (7 does not divide 3000 or 300)", lambda p, y: _ece_equal_mass(p, y, 7)))
    rows = []
    for name, fn in rules:
        rows.append((name, f"{fn(p_raw, yt):.4f}", f"{fn(p_fix, yt):.4f}", f"{fn(p_true, yt):.4f}",
                     f"{fn(p_true[:small], yt[:small]):.4f}"))
    head = (f"The {len(yt)} test parts. 'Recalibrated' is sigmoid(a z + b) with a, b fitted on the validation parts "
            f"(`metrics.reliability_scaled`). 'True law' uses the hidden probability sigmoid({LAW[0]} + {LAW[1]} x1 + {LAW[2]} x2) "
            f"of each part, which is perfectly calibrated by construction; the last column uses only the first {small} test parts "
            "(in the order drawn). Equal-width bins follow the `ece` pseudocode of the chapter. Equal-mass bins: sort the N parts by "
            "predicted probability ascending (parts with equal probability keep their original order), and let bin b = 0 .. B - 1 hold "
            "the sorted positions floor(b N / B) to floor((b + 1) N / B) - 1, so bin sizes differ by at most one; each bin is "
            "weighted by its own count, n_b / N. Four decimals printed.\n\n")
    return head + _table(["binning rule", "model, raw", "model, recalibrated", "true law, 3000 parts", f"true law, {small} parts"], rows)


# ------------------------------------------------------ regression, language

def regression_metrics():
    """MSE, MAE and R^2 for three prediction vectors, one with a single large error; the predictions are printed."""
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    sets = [("small errors", [2.5, 3.5, 6.5, 7.5, 10.5]),
            ("one large error", [2.0, 4.0, 6.0, 8.0, 20.0]),
            ("always the mean", [6.0] * 5)]
    mean = sum(y) / len(y)
    sst = sum((v - mean) ** 2 for v in y)
    rows = []
    for name, pred in sets:
        sse = sum((p - t) ** 2 for p, t in zip(pred, y))
        mae = sum(abs(p - t) for p, t in zip(pred, y)) / len(y)
        rows.append((name, "[" + ", ".join(f"{p:.1f}" for p in pred) + "]",
                     f"{sse / len(y):.4f}", f"{mae:.4f}", f"{1 - sse / sst:.4f}"))
    head = (f"Labels y = [{', '.join(f'{v:.1f}' for v in y)}], mean {mean:.1f}, SST = sum of (y - mean)^2 = {sst:.1f}. "
            "MSE = SSE / 5, MAE = mean of |prediction - y|, R^2 = 1 - SSE/SST, with SSE = sum of (prediction - y)^2; "
            "no rounding before the last step, four decimals printed.\n\n")
    return head + _table(["predictions", "predicted values", "MSE", "MAE", "R^2"], rows, "llrrr")


def perplexity_vs_exact_match():
    """Two models on 5 answers of 4 tokens: the one with better perplexity has worse exact match."""
    n_q, n_t = 5, 4
    # per token: (probability of the correct token, is the correct token the model's top choice?)
    a = [[(0.45, True)] * n_t for _ in range(n_q)]
    b = [[(0.95, True)] * n_t for _ in range(n_q)]
    b[2][1] = (0.02, False)
    rows = []
    for name, m in (("A: hedges, always right", a), ("B: sharp, one slip", b)):
        toks = [t for q in m for t in q]
        ce = -sum(math.log(p) for p, _ in toks) / len(toks)
        tok_acc = sum(1 for _, ok in toks if ok) / len(toks)
        em = sum(1 for q in m if all(ok for _, ok in q)) / len(m)
        rows.append((name, f"{ce:.4f}", f"{math.exp(ce):.4f}", f"{tok_acc:.4f}", f"{em:.4f}"))
    head = (f"{n_q} test answers of {n_t} tokens. Model A gives the correct token probability 0.45 at all {n_q * n_t} positions "
            "and it is its top choice every time. Model B gives it 0.95 at 19 positions and 0.02 at the second token of the "
            "third answer, where another token is the top choice. An answer is an exact match when all its tokens are the "
            "top choice.\n\n")
    t1 = _table(["model", "mean cross-entropy (nats)", "perplexity", "token accuracy", "exact match"], rows)
    lens = (1, 4, 10, 20, 50)
    rows2 = [(f"{acc}",) + tuple(f"{acc ** length:.4f}" for length in lens) for acc in (0.99, 0.95, 0.90)]
    t2 = ("\n\nIf each token is right independently with probability a, an answer of L tokens is right with probability a^L:\n\n" +
          _table(["token accuracy a"] + [f"L = {length}" for length in lens], rows2))
    return head + t1 + t2


# ------------------------------------------------ choosing and reporting

def _cost_per_part(pred, y):
    tp, fp, fn, tn = _confusion(pred, y)
    return (COST_FP * fp + COST_FN * fn) / len(y)


def cost_threshold():
    """A defective part passed costs 10, a good part rejected costs 1: where to cut, and what it costs on the test parts."""
    zv, yv, zt, yt, *_ = _scores()
    a = _fit_scale(zv, yv)
    a2, b2 = _fit_shift_scale(zv, yv)
    t_star = COST_FP / (COST_FP + COST_FN)
    zstar = logit(t_star)
    # the cut on z found by searching the validation cost over a grid of cuts
    grid = [-15.0 + 0.05 * k for k in range(601)]
    best = min(grid, key=lambda c: (_cost_per_part([v >= c for v in zv], yv), c))
    best_test = min(grid, key=lambda c: (_cost_per_part([v >= c for v in zt], yt), c))
    # (name, cut on z) for each rule; None means flag nothing, -infinity flag everything
    strategies = [
        ("flag nothing", None),
        ("flag everything", -math.inf),
        ("raw probability, t = 0.5", 0.0),
        (f"raw probability, t = {t_star:.4f}", zstar),
        (f"scaled probability, t = {t_star:.4f}", zstar / a),
        (f"scaled and shifted probability, t = {t_star:.4f}", (zstar - b2) / a2),
        ("cut searched on the validation parts", best),
        ("for reference: best cut on the test parts", best_test),
    ]
    rows = []
    for name, cut in strategies:
        pv = [False] * len(zv) if cut is None else [v >= cut for v in zv]
        pt = [False] * len(zt) if cut is None else [v >= cut for v in zt]
        _, ct_fp, ct_fn, _ = _confusion(pt, yt)
        shown = "-" if cut is None or cut == -math.inf else f"{cut:+.3f}"
        rows.append((name, shown, f"{_cost_per_part(pv, yv):.4f}", f"{_cost_per_part(pt, yt):.4f}", ct_fp, ct_fn))
    head = (f"Cost of a good part rejected (false positive) {COST_FP:g}, of a defective part passed (false negative) {COST_FN:g}. "
            f"A part is flagged when its cut is met on the logit z. For a calibrated probability p, flagging pays when "
            f"p x {COST_FN:g} > (1 - p) x {COST_FP:g}, that is p > {COST_FP:g}/({COST_FP:g} + {COST_FN:g}) = {t_star:.4f}, "
            f"logit {zstar:+.4f}. The scaled probability is sigmoid(a z) with a = {a:.4f}, so its cut on z is "
            f"{zstar:+.4f}/{a:.4f}; the scaled and shifted one is sigmoid(a z + b) with a = {a2:.4f}, b = {b2:.4f} "
            f"(`metrics.reliability_scaled`), so its cut is ({zstar:+.4f} - ({b2:.4f}))/{a2:.4f}. A searched cut is the "
            "lowest of the cuts c_k = -15 + 0.05 k, k = 0 .. 600 (computed by that multiplication; a part is flagged when z >= c_k) "
            "with the smallest cost on the parts named. Costs are per part, four decimals printed.\n\n")
    return head + _table(["rule", "cut on z", "cost per part, validation", "cost per part, test", "FP on test", "FN on test"], rows)


def _resample_stats(z, y, idx, cut):
    zs = [z[i] for i in idx]
    ys = [y[i] for i in idx]
    tp, fp, fn, tn = _confusion([v >= cut for v in zs], ys)
    return (_auc_ranksum(zs, ys), (tp + tn) / len(ys), 2 * tp / (2 * tp + fp + fn))


def bootstrap_intervals():
    """Percentile bootstrap intervals for AUC, accuracy and F1 on the test parts, beside the formula for accuracy."""
    zv, yv, zt, yt, *_ = _scores()
    a = _fit_scale(zv, yv)
    # the cut is chosen on the validation parts: the lowest cut with the largest F1
    def f1_at(c):
        tp, fp, fn, _ = _confusion([v >= c for v in zv], yv)
        return 2 * tp / (2 * tp + fp + fn)
    grid = [-15.0 + 0.05 * k for k in range(601)]
    best = min(grid, key=lambda c: (-f1_at(c), c))
    rng = bookrng.generator(BOOT_NAME)
    n = len(yt)
    reps = []
    for _ in range(BOOT_RESAMPLES):
        idx = [rng.integer(n) for _ in range(n)]
        reps.append(_resample_stats(zt, yt, idx, best))
    est = _resample_stats(zt, yt, list(range(n)), best)
    tail = (1 - BOOT_LEVEL) / 2
    rows = []
    for k, name in enumerate(("ROC AUC", "accuracy", "F1")):
        col = sorted(r[k] for r in reps)
        mean = sum(col) / len(col)
        sd = math.sqrt(sum((v - mean) ** 2 for v in col) / len(col))
        rows.append((name, f"{est[k]:.4f}", f"{sd:.4f}", f"{_percentile(col, tail):.4f}", f"{_percentile(col, 1 - tail):.4f}"))
    acc = est[1]
    se = math.sqrt(acc * (1 - acc) / n)
    head = (_generator_line(BOOT_NAME) + "\n"
            f"The {n} test parts ({sum(yt)} defective). The cut on z is chosen on the validation parts (the lowest of the cuts "
            f"c_k = -15 + 0.05 k, k = 0 .. 600, computed by that multiplication, with the largest F1 on the validation parts, a part "
            f"being flagged when z >= c_k; on equal F1 the lowest cut): z >= {best:+.2f}, "
            f"which is held fixed. The estimate column is the metric on the original {n} test parts in their given order. {BOOT_RESAMPLES} resamples drawn one after the other from this one generator; a resample is {n} calls of rng.integer({n}), each the index of a "
            f"test part (rows in the order drawn). Interval ends are the 2.5th and 97.5th percentiles of the sorted replicates by "
            f"linear interpolation at position q x ({BOOT_RESAMPLES} - 1), q = 0.025 and 0.975, in the ascending list of the {BOOT_RESAMPLES} replicates; the standard error is the standard deviation of the replicates "
            f"(divisor {BOOT_RESAMPLES}).\n\n")
    tail_txt = (f"\n\nFormula for the accuracy, with p the estimate above and n = {n}: sqrt(p (1 - p) / n) = {se:.4f}, interval p -/+ 1.959964 x that = "
                f"{acc - 1.959964 * se:.4f} to {acc + 1.959964 * se:.4f}.")
    return head + _table(["metric", "estimate", "standard error", "95% lower", "95% upper"], rows) + tail_txt


# ------------------------------------------------------------ in brain

def survival_auc_example():
    """The four-subject example of crates/survival/src/auc.rs, as a pair table and as a trapezoid."""
    risk = [("subject 0", 0.9, 1), ("subject 1", 0.4, 1), ("subject 2", 0.6, 0), ("subject 3", 0.1, 0)]
    pos = [(n, r) for n, r, t in risk if t == 1]
    neg = [(n, r) for n, r, t in risk if t == 0]
    rows = []
    total = 0.0
    for pn, pr in pos:
        for nn, nr in neg:
            win = 1.0 if pr > nr else (0.5 if pr == nr else 0.0)
            total += win
            rows.append((f"{pn} ({pr}) vs {nn} ({nr})", f"{win:g}"))
    s = [r for _, r, _ in risk]
    y = [t for _, _, t in risk]
    area = _trapezoid(_roc_points(s, y))
    tied = _trapezoid(_roc_points([0.5] * 4, y))
    tail = (f"\n\nShare of pairs: {total:g} of {len(pos) * len(neg)} = {total / (len(pos) * len(neg)):.4f}. "
            f"Trapezoid area of the ROC curve: {area:.4f}. With all four scores equal to 0.5 the curve is the diagonal "
            f"and both counts give {tied:.4f} ({_auc_pairs([0.5] * 4, y):.4f} by pairs).")
    return _table(["pair (case vs control)", "credit"], rows, "lr") + tail
