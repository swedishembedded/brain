# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Scoring Rules and Calibrated Decisions".

Plain Python 3, standard library only, written from the chapter's own description: the expected loss of every
report a forecaster can make under four scoring rules (and the minimiser of each), a log score floored from
below and the report it rewards, the Brier decomposition into reliability, resolution and uncertainty with the
remainder that appears when forecasts vary inside a bin, the reliability bins and the expected calibration
error, recalibration by a scale and a shift of the logit and by isotonic regression fitted on a fit set, on
the test set itself, and judged on 20000 fresh draws, a cost matrix with a deferral action, the value of
information of a test, the threshold chosen by cost against the one chosen by accuracy, the risk-coverage
curve of an abstaining classifier, the Brier scores over thresholds that add up to the CRPS, and the
gradients of the cross-entropy and the Brier score on a saturated prediction. Each public function without a
leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: calibrated_decisions.<name> -->` markers. Underscore functions and the constants are shared
with the chapter's figures, so a picture and the table beside it cannot disagree.

Swedish Embedded AB implements solutions for probabilities that can be acted on. If your team needs
expertise in calibrated forecasting and cost-sensitive decisions, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
from bisect import bisect_right
from functools import lru_cache

import bookrng
from metrics import _fit_shift_scale
from logistic_regression import sigmoid
from time_series import _table, HELD_GAUSSIAN

# ------------------------------------------------------------------ constants

DATA_NAME = "calibrated_decisions.forecaster"
N_FIT, N_TEST, N_FRESH = 2000, 300, 20000
ETA = (-0.8, 1.5)                 # the true logit of example i: eta = -0.8 + 1.5 * s, s the first draw
RAW = (2.0, 0.5, 0.6)             # the model's logit: z = 2 eta + 0.5 eta |eta| + 0.6
N_BINS = 10
FOCAL_GAMMA = 2.0
P_GAME = 0.7
LOG_FLOOR = math.log(1e-4)        # ln(1e-4) = -9.2103..., the lower clamp of the floored log score
FLOOR_P = 5e-5
COST_FP, COST_FN, COST_DEFER = 1.0, 10.0, 0.3
ABSTAIN_COST = 0.2                # cost of abstaining; a wrong answer costs 1, a right one 0
RATIOS = (1, 2, 5, 10, 20)        # cost of a miss over cost of a false alarm
COVERAGES = tuple(k / 10 for k in range(1, 11))


def _fmt(x, d=4):
    if x is None:
        return "none"
    if math.isinf(x):
        return "inf"
    t = f"{x:.{d}f}"
    return t[1:] if t.startswith("-") and float(t) == 0.0 else t


def _seed_line():
    seed = bookrng.seed_of(DATA_NAME)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed})." + HELD_GAUSSIAN


# ------------------------------------------------------------------ the data

@lru_cache(maxsize=None)
def _data():
    """(fit, test, fresh): each a list of (eta, p, y, z) with p the true probability and z the model's logit.

    One generator; example i of the fit set, then of the test set, then of the fresh set draws s = normal() and
    u = uniform(); eta = -0.8 + 1.5 * s, p = sigmoid(eta), y = 1 when u < p."""
    g = bookrng.generator(DATA_NAME)
    out = []
    for n in (N_FIT, N_TEST, N_FRESH):
        rows = []
        for _ in range(n):
            s = g.normal()
            u = g.uniform()
            eta = ETA[0] + ETA[1] * s
            p = sigmoid(eta)
            z = RAW[0] * eta + RAW[1] * eta * abs(eta) + RAW[2]
            rows.append((eta, p, 1 if u < p else 0, z))
        out.append(rows)
    return tuple(out)


def _cols(rows):
    return ([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])


def _brier(p, y):
    return sum((q - t) ** 2 for q, t in zip(p, y)) / len(y)


def _logloss(p, y):
    return sum(-math.log(q if t else 1.0 - q) for q, t in zip(p, y)) / len(y)


def _bin_of(q, n_bins=N_BINS):
    return min(int(q * n_bins), n_bins - 1)     # bin k holds k/10 <= q < (k+1)/10; q = 1 goes to the last bin


def _bins(p, y, n_bins=N_BINS):
    """(count, mean forecast, observed frequency) of each equal-width bin; empty bins are None."""
    cnt, sp, sy = [0] * n_bins, [0.0] * n_bins, [0] * n_bins
    for q, t in zip(p, y):
        k = _bin_of(q, n_bins)
        cnt[k] += 1
        sp[k] += q
        sy[k] += t
    return [(cnt[k], sp[k] / cnt[k], sy[k] / cnt[k]) if cnt[k] else None for k in range(n_bins)]


def _ece(p, y, n_bins=N_BINS):
    n = len(y)
    return sum(b[0] / n * abs(b[2] - b[1]) for b in _bins(p, y, n_bins) if b)


def _decompose(p, y, n_bins=N_BINS):
    """(BS, REL, RES, UNC, remainder): the Brier score, the three Murphy terms over equal-width bins and
    BS - (REL - RES + UNC), which is the within-bin term and is 0 when forecasts are constant inside a bin."""
    n = len(y)
    ybar = sum(y) / n
    rel = res = 0.0
    for b in _bins(p, y, n_bins):
        if b:
            rel += b[0] / n * (b[1] - b[2]) ** 2
            res += b[0] / n * (b[2] - ybar) ** 2
    unc = ybar * (1.0 - ybar)
    bs = _brier(p, y)
    return bs, rel, res, unc, bs - (rel - res + unc)


# ---------------------------------------------------------- isotonic regression

def _pav(scores, y):
    """Pool adjacent violators on the examples sorted by ascending score (equal scores keep input order).

    Returns (sorted_scores, fitted) where fitted[i] is the value of the block that holds sorted position i."""
    order = sorted(range(len(y)), key=lambda i: scores[i])
    xs = [scores[i] for i in order]
    blocks = []                                   # each block: [sum of labels, count]
    for i in order:
        blocks.append([float(y[i]), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    fitted = []
    for s, c in blocks:
        fitted.extend([s / c] * c)
    return xs, fitted


def _isotonic(scores, y):
    """A predictor z -> the fitted value at the largest training score not above z (the first block's value
    below every training score)."""
    xs, fitted = _pav(scores, y)

    def predict(z):
        return fitted[max(bisect_right(xs, z) - 1, 0)]
    return predict, len(set(fitted))


def _platt(z, y):
    a, b = _fit_shift_scale(z, y)
    return (lambda v: sigmoid(a * v + b)), a, b


# ------------------------------------------------------------ the report game

def _loss_rules():
    """name -> loss(q, y) of reporting the probability q of the event when the event has value y."""
    def side(q, y):
        return 0.0 if (q >= 0.5) == (y == 1) else 1.0

    def absolute(q, y):
        return abs(y - q)

    def brier(q, y):
        return (y - q) ** 2

    def log(q, y):
        pr = q if y else 1.0 - q
        return math.inf if pr <= 0.0 else -math.log(pr)

    def focal(q, y):
        pr = q if y else 1.0 - q
        return math.inf if pr <= 0.0 else (1.0 - pr) ** FOCAL_GAMMA * -math.log(pr)
    return (("side of 1/2", side), ("absolute", absolute), ("Brier", brier), ("log", log), ("focal 2", focal))


def _expected(loss, q, p=P_GAME):
    a, b = loss(q, 1), loss(q, 0)
    return (0.0 if p == 0 else p * a) + (0.0 if p == 1 else (1 - p) * b)


def _minimiser(loss, p=P_GAME):
    """The report on the grid 0, 0.001, ..., 1 with the smallest expected loss; the first of equals."""
    best, bq = math.inf, None
    for k in range(1001):
        q = k / 1000
        v = _expected(loss, q, p)
        if v < best - 1e-12:
            best, bq = v, q
    return bq


def report_game():
    rules = _loss_rules()
    rows = []
    for k in range(11):
        q = k / 10
        rows.append([f"{q:.1f}"] + [_fmt(_expected(f, q)) for _, f in rules])
    rows.append(["best report on the 0.001 grid"] + [f"{_minimiser(f):.3f}" for _, f in rules])
    head = ["report q"] + [name for name, _ in rules]
    return ("The event has probability p = 0.7. The forecaster's report q is scored by each loss with outcome y = 1 "
            "(probability 0.7) or y = 0; the cells are the expected loss 0.7 L(q, 1) + 0.3 L(q, 0). 'side of 1/2' "
            "is 0 when q >= 0.5 and y = 1 or q < 0.5 and y = 0, else 1; 'absolute' is |y - q|; 'Brier' is (y - q)^2; "
            "'log' is -ln of the probability the report gave to the outcome; 'focal 2' is (1 - r)^2 (-ln r) for the "
            "probability r the report gave to the outcome. Four decimals; inf is an outcome the report called "
            "impossible. The last row scans q = k/1000 for k = 0 .. 1000 and keeps the first q whose expected loss is "
            "smaller than every earlier one by more than 1e-12.\n\n" + _table(head, rows))


def _floor_reward(q, p, w_sph):
    """Expected reward of the report (1 - q, q) of a binary event with probability p: the log score floored at
    ln(1e-4) (after a clamp of the probability at 1e-12) plus w_sph times the spherical score."""
    norm = math.sqrt((1 - q) ** 2 + q ** 2)
    tot = 0.0
    for prob, t in ((1 - p, (1.0, 0.0)), (p, (0.0, 1.0))):
        rep = (1 - q, q)
        log = sum(ti * max(math.log(max(ri, 1e-12)), LOG_FLOOR) for ti, ri in zip(t, rep))
        sph = sum(ti * ri for ti, ri in zip(t, rep)) / norm
        tot += prob * (log + w_sph * sph)
    return tot


def floored_log():
    rows = []
    for q in (0.0, 1e-5, FLOOR_P, 1e-4, 2e-4, 1e-3):
        rows.append([f"{q:.0e}" if q else "0", f"{_floor_reward(q, FLOOR_P, 0.0):.6f}",
                     f"{_floor_reward(q, FLOOR_P, 0.5):.6f}"])
    # the maximiser of each over q = k * 1e-6, k = 0 .. 2000
    best = []
    for w in (0.0, 0.5):
        bv, bq = -math.inf, None
        for k in range(2001):
            v = _floor_reward(k * 1e-6, FLOOR_P, w)
            if v > bv + 1e-15:
                bv, bq = v, k * 1e-6
        best.append(bq)
    rows.append(["best q on the 1e-6 grid", f"{best[0]:.6f}", f"{best[1]:.6f}"])
    lo, hi = 1e-4, 1e-3                  # the probability at which reporting 0 stops beating reporting p, by bisection
    for _ in range(80):
        mid = (lo + hi) / 2
        if _floor_reward(0.0, mid, 0.0) > _floor_reward(mid, mid, 0.0):
            lo = mid
        else:
            hi = mid
    return ("A binary event of probability p = 5e-5 (the honest report) and a reward that is the log score "
            "ln(max(r, 1e-12)) floored below at ln(1e-4) = -9.2103, r the probability the report gave the outcome; "
            "its expected value for the report (1 - q, q) is in the second column. The third column adds 0.5 times "
            "the spherical score (the dot product of the report with the one-hot outcome, divided by the length of the "
            "report). Six decimals; the last row scans q = k 1e-6 for k = 0 .. 2000 and keeps the first q whose reward "
            "exceeds every earlier one by more than 1e-15.\n\n"
            + _table(["report q", "floored log reward", "floored log + 0.5 spherical"], rows)
            + f"\n\nFor the floored log reward alone, reporting 0 beats reporting the true probability p for every p below "
              f"{lo:.2e} (found by 80 bisections of [1e-4, 1e-3], comparing the two rewards).")


# ---------------------------------------------------------- Brier decomposition

def _forecasters():
    """name -> list of forecast probabilities on the fresh set (and on the fit set for the base rate)."""
    fit, _, fresh = _data()
    p, y, z = _cols(fresh)
    fz, fy = _cols(fit)[2], _cols(fit)[1]
    pf, _, _ = _platt(fz, fy)
    base = sum(fy) / len(fy)
    return {
        "true probability": p,
        "raw model": [sigmoid(v) for v in z],
        "model after scale and shift": [pf(v) for v in z],
        "true probability, bin centre": [(_bin_of(q) + 0.5) / N_BINS for q in p],
        "base rate of the fit set": [base] * len(p),
    }


def decomposition():
    fit, _, fresh = _data()
    y = _cols(fresh)[1]
    rows = []
    for name, f in _forecasters().items():
        bs, rel, res, unc, rem = _decompose(f, y)
        rows.append([name, _fmt(bs), _fmt(rel), _fmt(res), _fmt(unc), _fmt(rem, 5), _fmt(_ece(f, y)), _fmt(_logloss(f, y))])
    ybar = sum(y) / len(y)
    return (_seed_line() + " Example i of the fit set (2000 examples), then of the test set (300), then of the fresh set "
            "(20000), in that order from one stream, draws s = normal(), then u = uniform(); eta = -0.8 + 1.5 * s, "
            "p = sigmoid(eta), and the label is y = 1 when u < p. The model's logit is z = 2 eta + 0.5 eta |eta| + 0.6. "
            "The table is on the fresh set (share of labels 1: " + f"{ybar:.4f}" + "). Equal-width bins: bin k holds "
            "k/10 <= q < (k+1)/10, and q = 1 goes to bin 9. REL = sum over nonempty bins of (n_k/n)(mean q - mean y)^2, "
            "RES = sum of (n_k/n)(mean y in the bin - mean y)^2, UNC = ybar (1 - ybar); 'remainder' is BS - (REL - RES + UNC). "
            "'Scale and shift' is sigmoid(a z + b) with (a, b) fitted on the fit set by Newton's method on the "
            "cross-entropy (`metrics._fit_shift_scale`). The base rate is the share of labels 1 in the fit set. "
            "ECE is the 10-bin expected calibration error of the probability of the event; log loss is the mean "
            "-ln of the probability given to the label.\n\n"
            + _table(["forecaster", "BS", "REL", "RES", "UNC", "remainder", "ECE", "log loss"], rows))


def fit_values():
    fit, _, _ = _data()
    fz, fy = _cols(fit)[2], _cols(fit)[1]
    _, a, b = _platt(fz, fy)
    _, k = _isotonic(fz, fy)
    return a, b, k, sum(fy) / len(fy)


# ------------------------------------------------------------- recalibration

def recalibration():
    fit, test, fresh = _data()
    fz, fy = _cols(fit)[2], _cols(fit)[1]
    tp, ty, tz = _cols(test)
    rp, ry, rz = _cols(fresh)
    assert len(set(fz)) == len(fz) and len(set(tz)) == len(tz)       # no equal scores: ties never matter here
    methods = []
    methods.append(("raw model", lambda v: sigmoid(v), "-"))
    pa, a, b = _platt(fz, fy)
    methods.append(("scale and shift, fitted on the fit set", pa, "fit"))
    ia, ka = _isotonic(fz, fy)
    _, fitted = _pav(fz, fy)
    methods.append(("isotonic, fitted on the fit set", ia, "fit"))
    pt, at, bt = _platt(tz, ty)
    methods.append(("scale and shift, fitted on the test set", pt, "test"))
    it, kt = _isotonic(tz, ty)
    methods.append(("isotonic, fitted on the test set", it, "test"))
    rows = []
    for name, f, src in methods:
        fp = [f(v) for v in tz]
        gp = [f(v) for v in rz]
        rows.append([name, src, _fmt(_brier(fp, ty)), _fmt(_ece(fp, ty)), _fmt(_brier(gp, ry)), _fmt(_ece(gp, ry))])
    rows.append(["true probability", "-", _fmt(_brier(tp, ty)), _fmt(_ece(tp, ty)), _fmt(_brier(rp, ry)), _fmt(_ece(rp, ry))])
    iso_fresh = [ia(v) for v in rz]
    sure = [(q, t) for q, t in zip(iso_fresh, ry) if q in (0.0, 1.0)]
    wrong = sum(1 for q, t in sure if q != t)
    return ("Data as in `calibrated_decisions.decomposition`. The scale-and-shift fit on the fit set is a = "
            f"{a:.4f}, b = {b:.4f}; the isotonic fit on the fit set has {ka} distinct values, on the test set {kt}. "
            "Isotonic regression pools adjacent violators of the labels sorted by ascending model logit z; the "
            "fitted function at a logit z is the value of the block that holds the largest training logit not above "
            f"z, and the first block's value below all of them; the fit-set values run from {min(fitted):.4f} to {max(fitted):.4f}. The model logits of the fit and test sets are all "
            "different, so no tie rule was used. Brier score and 10-bin ECE on the 300 test examples and on the "
            "20000 fresh examples; four decimals.\n\n"
            + _table(["recalibrator", "fitted on", "test BS", "test ECE", "fresh BS", "fresh ECE"], rows)
            + f"\n\nThe isotonic fit on the fit set gives exactly 0 or exactly 1 to {len(sure)} of the 20000 fresh examples, and "
              f"{wrong} of those carry the other label.")


HAND_FORECASTS = ((0.2, (1, 0, 0, 0)), (0.9, (1, 1, 1, 1, 0, 1)))


def decomposition_by_hand():
    rows, n = [], sum(len(ys) for _, ys in HAND_FORECASTS)
    allp = [p for p, ys in HAND_FORECASTS for _ in ys]
    ally = [t for _, ys in HAND_FORECASTS for t in ys]
    ybar = sum(ally) / n
    rel = res = 0.0
    for p, ys in HAND_FORECASTS:
        f = sum(ys) / len(ys)
        rel += len(ys) / n * (p - f) ** 2
        res += len(ys) / n * (f - ybar) ** 2
        rows.append([f"{p:.1f}", str(len(ys)), "".join(str(t) for t in ys), f"{f:.4f}"])
    unc = ybar * (1 - ybar)
    bs = _brier(allp, ally)
    return ("Ten forecasts: four of probability 0.2 and six of probability 0.9, with the outcomes listed (1 = the event happened).\n\n"
            + _table(["forecast", "count", "outcomes", "share of labels 1"], rows, "rrll")
            + f"\n\nBrier score {bs:.4f}; share of labels 1 {ybar:.1f}, uncertainty {unc:.4f}; reliability {rel:.6f}; resolution "
              f"{res:.6f}; reliability - resolution + uncertainty = {rel - res + unc:.4f}.")


def reliability_table():
    _, _, fresh = _data()
    p, y, z = _cols(fresh)
    q = [sigmoid(v) for v in z]
    n = len(y)
    rows, tot = [], 0.0
    for k, b in enumerate(_bins(q, y)):
        if b:
            gap = abs(b[2] - b[1])
            tot += b[0] / n * gap
            rows.append([f"{k / N_BINS:.1f} to {(k + 1) / N_BINS:.1f}", str(b[0]), _fmt(b[1]), _fmt(b[2]), _fmt(gap)])
    return ("The raw model on the fresh set of `calibrated_decisions.decomposition`: the 10 equal-width bins of its probability "
            "(bin k holds k/10 <= q < (k+1)/10), the number of examples, their mean forecast, the share of them with label 1, "
            "and the absolute gap. Four decimals.\n\n"
            + _table(["bin", "count", "mean forecast", "share of labels 1", "gap"], rows)
            + f"\n\nECE = sum of (count / {n}) x gap = {tot:.4f}.")


# ------------------------------------------------------------- costs and thresholds

def _expected_costs(p, cfp=COST_FP, cfn=COST_FN, cd=COST_DEFER):
    """(flag, pass, defer): expected cost of each action when the event has probability p."""
    return ((1.0 - p) * cfp, p * cfn, cd)


def _action(p, cfp=COST_FP, cfn=COST_FN, cd=COST_DEFER):
    c = _expected_costs(p, cfp, cfn, cd)
    best = min(range(3), key=lambda a: (c[a], a))      # the first of equals
    return best, c[best]


def device_world():
    """The prior 0.2 for a fault and a test with P(+ | fault) = 0.8, P(+ | healthy) = 0.1; block costs 1 on a healthy
    device, release costs 10 on a faulty one."""
    prior, sens, fpr = 0.2, 0.8, 0.1
    pp = prior * sens + (1 - prior) * fpr
    post_pos = prior * sens / pp
    post_neg = prior * (1 - sens) / (1 - pp)
    rows = []

    def plain(p, cfp=COST_FP, cfn=COST_FN):
        return min((1 - p) * cfp, p * cfn)
    for label, p in (("before the test", prior), ("test positive", post_pos), ("test negative", post_neg)):
        blk, rel = (1 - p) * COST_FP, p * COST_FN
        rows.append([label, f"{p:.4f}", _fmt(blk), _fmt(rel), "block" if blk <= rel else "release", _fmt(min(blk, rel))])
    before = plain(prior)
    after = pp * plain(post_pos) + (1 - pp) * plain(post_neg)
    voi = before - after - 0.05
    flip = [((1 - prior) * cfp, prior * COST_FN) for cfp in (1.0, 4.0)]
    text = (f"Costs: blocking a healthy device 1, releasing a faulty one 10, a correct action 0. Expected cost of an action = "
            f"sum over outcomes of probability x cost. Prior P(fault) = 0.2, P(+ | fault) = 0.8, P(+ | healthy) = 0.1, "
            f"so P(+) = {pp:.2f}. Four decimals.\n\n"
            + _table(["belief", "P(fault)", "block", "release", "action", "risk"], rows, "lrrrlr"))
    text += (f"\n\nThe risk before the test is {before:.2f}; after it, in expectation, {pp:.2f} x {plain(post_pos):.4f} + "
             f"{1 - pp:.2f} x {plain(post_neg):.4f} = {after:.2f}. With a query cost of 0.05 the value of information is "
             f"{before:.2f} - {after:.2f} - 0.05 = {voi:.2f}. If blocking a healthy device cost 4 instead of 1, the prior 0.2 "
             f"would give block {flip[1][0]:.1f} against release {flip[1][1]:.1f}: the action changes from "
             f"{'block' if flip[0][0] <= flip[0][1] else 'release'} to {'block' if flip[1][0] <= flip[1][1] else 'release'} "
             f"while the probability stays 0.2; the boundary moves from 1/11 = {1 / 11:.4f} to 4/14 = {4 / 14:.4f}.")
    return text


def deferral_regions():
    """Block, release or defer as a function of P(fault) when deferring costs a fixed 0.3."""
    rows = []
    for p in (0.01, 0.02, 0.05, 0.0909, 0.2, 0.5, 0.65, 0.75, 0.9):
        blk, rel, dfr = (1 - p) * COST_FP, p * COST_FN, COST_DEFER
        best = min(blk, rel, dfr)
        act = "block" if best == blk else ("release" if best == rel else "defer")
        rows.append([f"{p:.4f}", _fmt(blk), _fmt(rel), _fmt(dfr), act])
    lo, hi = COST_DEFER / COST_FN, 1.0 - COST_DEFER / COST_FP
    return (f"Blocking a healthy device costs 1, releasing a faulty one 10, deferring (a manual inspection) 0.3 whatever the "
            f"truth. Expected cost of each action at a probability p of a fault; four decimals; ties do not occur at these "
            f"values.\n\n"
            + _table(["P(fault)", "block", "release", "defer", "cheapest"], rows, "lrrrl")
            + f"\n\nRelease is cheapest for p <= 0.3 / 10 = {lo:.2f}, block for p >= 1 - 0.3 / 1 = {hi:.2f}, deferring in "
              f"between; the plain boundary 1/11 = {1 / 11:.4f} lies inside the deferral band.")


def _accuracy_threshold(p, y):
    """The cut c_k = k/100, k = 0 .. 100, with the highest accuracy of 'flag when p >= c_k' (first of equals)."""
    best, bc = -1, None
    for k in range(101):
        c = k / 100
        acc = sum(1 for q, t in zip(p, y) if (q >= c) == (t == 1))
        if acc > best:
            best, bc = acc, c
    return bc


def _mean_cost(p, y, t, cfp, cfn):
    tot = 0.0
    for q, lab in zip(p, y):
        flag = q >= t
        if flag and not lab:
            tot += cfp
        elif not flag and lab:
            tot += cfn
    return tot / len(y)


def _best_cut(r=10.0):
    """(threshold, cost) of the cheapest cut on the grid 0.01 .. 0.99 on the fresh set, for each set of probabilities."""
    series, grid = _cost_by_threshold(r)
    out = {}
    for name, ys in series.items():
        m = min(ys)
        out[name] = (grid[ys.index(m)], m)
    return out


def threshold_choice():
    fit, _, fresh = _data()
    fz, fy = _cols(fit)[2], _cols(fit)[1]
    rp, ry, rz = _cols(fresh)
    raw_fit = [sigmoid(v) for v in fz]
    raw = [sigmoid(v) for v in rz]
    pf, _, _ = _platt(fz, fy)
    cal = [pf(v) for v in rz]
    t_acc = _accuracy_threshold(raw_fit, fy)
    t_acc_cal = _accuracy_threshold([pf(v) for v in fz], fy)
    rows = []
    best = _best_cut()
    for r in RATIOS:
        ts = 1.0 / (1.0 + r)
        rows.append([str(r), f"{ts:.4f}",
                     _fmt(_mean_cost(raw, ry, t_acc, 1.0, float(r))),
                     _fmt(_mean_cost(raw, ry, ts, 1.0, float(r))),
                     _fmt(_mean_cost(cal, ry, t_acc_cal, 1.0, float(r))),
                     _fmt(_mean_cost(cal, ry, ts, 1.0, float(r))),
                     _fmt(_mean_cost(rp, ry, ts, 1.0, float(r)))])
    return (f"Fresh set of `calibrated_decisions.decomposition` (20000 examples). A false alarm (block, label 0) costs 1 and a "
            f"miss (release, label 1) costs r; block when the probability is at least the threshold. 'accuracy cut' is the "
            f"cut c = k/100, k = 0 .. 100, that maximises the accuracy of the raw model on the fit set (the first of equals): "
            f"{t_acc:.2f} on the raw probabilities and {t_acc_cal:.2f} on the probabilities recalibrated by scale and shift on the fit "
            f"set (the same fit as in `calibrated_decisions.decomposition`). t* = 1/(1 + r) is applied to the "
            f"raw probabilities, to the recalibrated ones, and to the true "
            f"probabilities (the least cost any rule on the true probabilities can have, in expectation). Cells are the mean cost per "
            f"example, four decimals.\n\n"
            + _table(["r", "t*", "raw, accuracy cut", "raw, t*", "recalibrated, accuracy cut", "recalibrated, t*",
                      "true probability, t*"], rows)
            + "\n\nFor reference only, since it looks at the fresh labels: at r = 10 the cheapest cut on the grid 0.01 .. 0.99 "
              f"is {best['raw model'][0]:.2f} for the raw probabilities (cost {best['raw model'][1]:.4f}) and "
              f"{best['recalibrated'][0]:.2f} for the recalibrated ones (cost {best['recalibrated'][1]:.4f}).")


def quoted_cost():
    """The cost a rule predicts for itself (mean over examples of the cheapest expected cost) against what it pays."""
    fit, _, fresh = _data()
    fz, fy = _cols(fit)[2], _cols(fit)[1]
    rp, ry, rz = _cols(fresh)
    pf, _, _ = _platt(fz, fy)
    r = 10.0
    rows = []
    for name, probs in (("raw model", [sigmoid(v) for v in rz]), ("recalibrated", [pf(v) for v in rz]), ("true probability", rp)):
        quoted = sum(min((1 - q) * 1.0, q * r) for q in probs) / len(probs)
        paid = _mean_cost(probs, ry, 1.0 / (1.0 + r), 1.0, r)
        rows.append([name, _fmt(quoted), _fmt(paid), _fmt(paid - quoted)])
    return ("Fresh set, cost of a miss 10 and of a false alarm 1, block when p >= 1/11. 'quoted' is the mean over the "
            "examples of min((1 - p), 10 p), the expected cost the forecaster's own probability promises; 'paid' is the mean "
            "cost actually incurred. Four decimals.\n\n"
            + _table(["probabilities used", "quoted", "paid", "paid - quoted"], rows))


# ------------------------------------------------------------- selective prediction

def _risk_curve(conf, err):
    """Examples by descending confidence (equal confidences keep input order); (coverage, risk) after each prefix."""
    order = sorted(range(len(err)), key=lambda i: -conf[i])
    out, wrong = [], 0
    for k, i in enumerate(order, 1):
        wrong += err[i]
        out.append((k / len(err), wrong / k))
    return out, order


def _aurc(curve):
    return sum(r for _, r in curve) / len(curve)


def _failure_auroc(conf, err):
    """Share of (error, correct) pairs in which the error has the lower confidence; a tie counts one half."""
    bad = [conf[i] for i in range(len(err)) if err[i]]
    good = [conf[i] for i in range(len(err)) if not err[i]]
    wins = 0.0
    sg = sorted(good)
    # for each error: good examples with higher confidence count 1, equal ones 1/2 (sorted + bisect)
    from bisect import bisect_left
    for c in bad:
        lo, hi = bisect_left(sg, c), bisect_right(sg, c)
        wins += (len(sg) - hi) + 0.5 * (hi - lo)
    return wins / (len(bad) * len(good))


def _scorers():
    fit, _, fresh = _data()
    fz, fy = _cols(fit)[2], _cols(fit)[1]
    rp, ry, rz = _cols(fresh)
    pf, _, _ = _platt(fz, fy)
    out = {}
    for name, probs in (("true probability", rp), ("recalibrated", [pf(v) for v in rz]), ("raw model", [sigmoid(v) for v in rz])):
        err = [1 if (q >= 0.5) != (t == 1) else 0 for q, t in zip(probs, ry)]
        out[name] = (probs, [max(q, 1 - q) for q in probs], err)
    return out


def selective_prediction():
    rows = []
    sc = _scorers()
    curves = {n: _risk_curve(c, e)[0] for n, (_, c, e) in sc.items()}
    for cov in COVERAGES:
        row = [f"{cov:.1f}"]
        for n in sc:
            k = round(cov * N_FRESH)
            row.append(_fmt(curves[n][k - 1][1]))
        rows.append(row)
    foot = []
    for n, (_, c, e) in sc.items():
        foot.append(f"{n}: area under the curve {_aurc(curves[n]):.4f}, failure AUROC {_failure_auroc(c, e):.4f}")
    return ("Fresh set (20000 examples). The answer is 1 (the event) when p >= 0.5, an error is an answer that differs from "
            "the label, and the confidence is max(p, 1 - p). Examples are sorted by descending confidence (equal "
            "confidences keep their order in the data) and the risk at coverage c is the error rate of the first 20000 c; "
            "four decimals.\n\n"
            + _table(["coverage", "true probability", "recalibrated", "raw model"], rows)
            + "\n\nThe area under the curve is the mean over k = 1 .. 20000 of the error rate of the k most confident examples; "
              "the failure AUROC is the share of (error, correct answer) pairs in which the error has the lower confidence, "
              "a tie counting one half. " + "; ".join(foot) + ".")


def abstention_cost():
    sc = _scorers()
    probs, conf, err = sc["recalibrated"]
    n = len(err)
    curve, order = _risk_curve(conf, err)
    rows, best = [], (math.inf, None)
    for cov in COVERAGES:
        k = round(cov * n)
        cost = (curve[k - 1][1] * k + ABSTAIN_COST * (n - k)) / n
        rows.append([f"{cov:.1f}", _fmt(cost)])
        if cost < best[0]:
            best = (cost, cov)
    rules = []
    for name, (pr, _, e) in sc.items():
        answered = [i for i in range(n) if min(pr[i], 1 - pr[i]) <= ABSTAIN_COST]
        paid = (sum(e[i] for i in answered) + ABSTAIN_COST * (n - len(answered))) / n
        quoted = sum(min(min(q, 1 - q), ABSTAIN_COST) for q in pr) / n
        rules.append([name, f"{len(answered) / n:.4f}", _fmt(quoted), _fmt(paid)])
    return ("Fresh set, recalibrated probabilities. A wrong answer costs 1, a right one 0, abstaining costs 0.2. Cost per example "
            "when the first 20000 c examples by descending confidence are answered and the rest are not. Four decimals.\n\n"
            + _table(["coverage", "cost per example"], rows)
            + f"\n\nThe cheapest coverage on this grid is {best[1]:.1f} at {best[0]:.4f}. The rule that answers when "
              f"min(p, 1 - p) <= 0.2 needs no ranking at all; applied to each set of probabilities it gives:\n\n"
            + _table(["probabilities", "share answered", "quoted cost", "paid cost"], rules)
            + "\n\n'Quoted' is the mean of min(min(p, 1 - p), 0.2), the cost the probabilities promise; 'paid' is the cost incurred.")


# ------------------------------------------------------------- CRPS as a sum of Brier scores

CRPS_PROBS = (0.1, 0.4, 0.3, 0.2)
CRPS_OUTCOME = 2


def crps_from_brier():
    probs, y = CRPS_PROBS, CRPS_OUTCOME
    cdf, acc = [], 0.0
    for q in probs:
        acc += q
        cdf.append(acc)
    rows, total = [], 0.0
    for k in range(len(probs) - 1):
        ev = 1 if y <= k else 0
        bs = (cdf[k] - ev) ** 2
        total += bs
        rows.append([f"outcome <= {k}", f"{cdf[k]:.1f}", str(ev), f"{bs:.4f}"])
    pair = sum(probs[i] * probs[j] * abs(i - j) for i in range(len(probs)) for j in range(len(probs)))
    direct = sum(probs[i] * abs(i - y) for i in range(len(probs))) - 0.5 * pair
    rows.append(["sum", "", "", f"{total:.4f}"])
    return (f"A forecast of the integer outcome 0, 1, 2, 3 with probabilities 0.1, 0.4, 0.3, 0.2; the outcome is {y}. The "
            f"event 'outcome <= k' has probability F(k) (the running total of the probabilities) and the value 1 when "
            f"y <= k. Its Brier score is (F(k) - value)^2; the thresholds k = 0, 1, 2 are the ones between outcomes "
            f"(k = 3 contributes 0).\n\n"
            + _table(["event", "forecast F(k)", "value", "Brier score"], rows)
            + f"\n\nThe CRPS from the definition E|X - y| - E|X - X'|/2 (X and X' independent draws from the forecast) is "
              f"{direct:.4f}.")


# ------------------------------------------------------------- gradients on a saturated prediction

def saturated_gradients():
    rows = []
    for z in (0.0, -2.0, -4.0, -6.0, -10.0, -15.0):
        p = sigmoid(z)
        ce = p - 1.0                           # label 1: d(-ln p)/dz
        br = 2 * (p - 1.0) * p * (1.0 - p)     # d(p - 1)^2/dz
        rows.append([f"{z:.0f}", f"{p:.3e}", f"{-math.log(p):.4f}", f"{(p - 1.0) ** 2:.4f}", f"{ce:.4f}", f"{br:.3e}"])
    return ("One example with label 1 and a model whose logit is z, so that p = sigmoid(z). Loss and derivative with respect "
            "to z of the cross-entropy -ln p, which is p - 1, and of the one-class Brier score (p - 1)^2, which is "
            "2 (p - 1) p (1 - p). The table prints p and the Brier derivative in scientific notation, the rest to four decimals.\n\n"
            + _table(["z", "p", "cross-entropy", "Brier", "d cross-entropy / dz", "d Brier / dz"], rows))


# ------------------------------------------------------------- soft targets

SOFT_ROWS = (((0.8, 0.2), "no evidence"), ((1 / 3, 2 / 3), "positive test"), ((18 / 19, 1 / 19), "negative test"))


def soft_scores():
    """Mean soft cross-entropy, KL and squared distance of a report to the oracle's distributions (each used once)."""
    reports = {
        "the oracle itself": [q for q, _ in SOFT_ROWS],
        "hedged toward the middle": [(0.7, 0.3), (0.5, 0.5), (0.8, 0.2)],
        "collapsed to the likelier label": [(0.98, 0.02), (0.02, 0.98), (0.98, 0.02)],
    }
    rows = []
    for name, rep in reports.items():
        ce = kl = sq = hard = 0.0
        for (q, _), p in zip(SOFT_ROWS, rep):
            hard += -math.log(max(p[0] if q[0] >= q[1] else p[1], 1e-12))
            ce += sum(-qi * math.log(max(pi, 1e-12)) for qi, pi in zip(q, p))
            kl += sum(qi * math.log(qi / max(pi, 1e-12)) for qi, pi in zip(q, p) if qi > 0)
            sq += sum((pi - qi) ** 2 for qi, pi in zip(q, p))
        n = len(SOFT_ROWS)
        rows.append([name, _fmt(hard / n), _fmt(ce / n), _fmt(kl / n), _fmt(sq / n)])
    ent = sum(-sum(qi * math.log(qi) for qi in q) for q, _ in SOFT_ROWS) / len(SOFT_ROWS)
    return (f"Three oracle distributions over (healthy, faulty): (0.8, 0.2), (1/3, 2/3), (18/19, 1/19), one report for each. "
            f"Cells are the mean over the three of sum_i q_i (-ln p_i) (soft cross-entropy), sum_i q_i ln(q_i / p_i) (KL) and "
            f"sum_i (p_i - q_i)^2, four decimals; the mean entropy of the oracle is {ent:.4f}. 'Likelier label' is the mean "
            f"-ln of the probability the report gives to the likelier outcome of the oracle (healthy, faulty, healthy), "
            f"the score of a collapsed target.\n\n"
            + _table(["report", "likelier label", "soft cross-entropy", "KL", "squared distance"], rows))


# ------------------------------------------------------------- series for the figures

def _excess_curve(rule, qs, p=P_GAME):
    """Expected loss of reporting q minus the expected loss of the best report on the 0.001 grid, for the rule `rule`."""
    f = dict(_loss_rules())[rule]
    best = _expected(f, _minimiser(f, p), p)
    return [_expected(f, q, p) - best for q in qs]


def _reliability_series():
    """name -> (mean forecast, share of labels 1) of the nonempty bins on the fresh set, for three forecasters."""
    fit, _, fresh = _data()
    fz, fy = _cols(fit)[2], _cols(fit)[1]
    _, ry, rz = _cols(fresh)
    pf, _, _ = _platt(fz, fy)
    isof, _ = _isotonic(fz, fy)
    out = {}
    for name, f in (("raw model", sigmoid), ("scale and shift, fit set", pf), ("isotonic, fit set", isof)):
        bins = [b for b in _bins([f(v) for v in rz], ry) if b]
        out[name] = ([b[1] for b in bins], [b[2] for b in bins])
    return out


def _cost_by_threshold(r=10.0, grid=tuple(k / 100 for k in range(1, 100))):
    """name -> mean cost on the fresh set of 'block when p >= t' for t in the grid, with misses costing r."""
    fit, _, fresh = _data()
    fz, fy = _cols(fit)[2], _cols(fit)[1]
    _, ry, rz = _cols(fresh)
    pf, _, _ = _platt(fz, fy)
    raw = [sigmoid(v) for v in rz]
    cal = [pf(v) for v in rz]
    return {"raw model": [_mean_cost(raw, ry, t, 1.0, r) for t in grid],
            "recalibrated": [_mean_cost(cal, ry, t, 1.0, r) for t in grid]}, grid


def _risk_curves():
    """name -> (coverage, risk) after every prefix, for each scorer, plus the 'perfect ranking' (every error last) and the
    'no ranking' (the overall error rate of the recalibrated classifier) references."""
    sc = _scorers()
    out = {n: _risk_curve(c, e)[0] for n, (_, c, e) in sc.items()}
    err = sc["recalibrated"][2]
    n_err = sum(err)
    n = len(err)
    out["perfect ranking"] = [(k / n, max(0.0, (k - (n - n_err)) / k)) for k in range(1, n + 1)]
    out["no ranking"] = [(k / n, n_err / n) for k in range(1, n + 1)]
    return out


# ------------------------------------------------------------- a score as an average over thresholds

THRESHOLD_PANELS = 1000


def scores_as_threshold_averages():
    """For a report q and a label y: the loss of acting at q when the threshold is t averaged over t in (0, 1) by the
    midpoint rule on 1000 panels, with weight 1 (Brier / 2) and weight 1 / (t (1 - t)) (log score)."""
    n = THRESHOLD_PANELS
    rows = []
    for q, y in ((0.3, 1), (0.3, 0), (0.9, 1), (0.9, 0), (0.02, 1)):
        uni = log = 0.0
        for j in range(n):
            t = (j + 0.5) / n
            act = q >= t
            loss = t * (1 if act and y == 0 else 0) + (1 - t) * (1 if (not act) and y == 1 else 0)
            uni += loss / n
            log += loss / (t * (1 - t)) / n
        pr = q if y else 1 - q
        rows.append([f"{q:.2f}", str(y), f"{uni:.5f}", f"{(1 - pr) ** 2 / 2:.5f}", f"{log:.4f}", f"{-math.log(pr):.4f}"])
    return (f"A report q is acted on at a threshold t as 'act if q >= t'. The loss is t when it acts and the label is 0 and 1 - t "
            f"when it does not act and the label is 1, else 0. The midpoint rule at t = (j + 0.5)/{n}, j = 0 .. {n - 1}, "
            f"averages the loss over t (weight 1) and the loss divided by t (1 - t) (weight 1/(t(1-t))). Five decimals "
            f"for the first pair, four for the second.\n\n"
            + _table(["q", "y", "mean loss over t", "(y - q)^2 / 2", "weighted mean loss", "-ln(prob. of the label)"], rows))
