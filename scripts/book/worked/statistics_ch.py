# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Statistics: Learning Facts from Samples".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: statistics_ch.<name> -->` markers. Helpers with a leading
underscore (or that take arguments) are shared with the chapter's figures, so a
picture and the table beside it cannot disagree. Small tables are computed
exactly (fractions or enumeration), areas under the bell with the error
function, the Student t multiplier by integrating its density, and every
simulation draws from a named `bookrng` stream. The module is named
`statistics_ch` because `statistics` is a standard-library module.

Swedish Embedded AB implements solutions for deciding, from finite test sets,
whether one model is really better than another. If your team needs expertise
in statistical evaluation of machine learning systems, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
from fractions import Fraction
from functools import lru_cache
from itertools import product

import bookrng

# The running example. Two models answer the same N questions. Each scenario is
# the 2 x 2 table (both right, only A right, only B right, both wrong).
N_Q = 500
SCENARIOS = {1: (332, 20, 24, 124), 2: (338, 14, 32, 116)}
P_NULL = 0.708          # pooled accuracy of scenario 1: (352 + 356) / 1000


# ------------------------------------------------------------------ helpers

def _table(headers, rows, align=None):
    """A markdown table; `align` is a string of 'l'/'r' per column (default: first l, rest r)."""
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fx(v, d=4):
    v = float(v) + 0.0
    s = f"{v:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def _pct(v, d=1):
    return _fx(100 * v, d)


def _mean(v):
    return sum(v) / len(v)


def _sample_var(v):
    """The unbiased estimator: squared deviations divided by n - 1."""
    m = _mean(v)
    return sum((a - m) ** 2 for a in v) / (len(v) - 1)


def _cdf(z):
    """Area under the standard Gaussian density to the left of z (Chapter 11)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _p_two_sided(z):
    return 2.0 * (1.0 - _cdf(abs(z)))


def _z_central(area):
    """The z with the standard Gaussian holding `area` within [-z, z], by bisection."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 2 * _cdf(mid) - 1 < area:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


Z95 = _z_central(0.95)


def _binom(rng, n, p):
    """Number of successes among n independent trials of probability p."""
    return sum(rng.random() < p for _ in range(n))


def _tail_ge(n, k):
    """Exact P(Binomial(n, 1/2) >= k)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def _sign_p_two_sided(n, k):
    """Two-sided sign test: twice the smaller tail, at most 1."""
    if n == 0:
        return 1.0
    return min(1.0, 2 * min(_tail_ge(n, k), _tail_ge(n, n - k)))


def _scenario_vectors(s):
    """Per-question 0/1 correctness of A and B reproducing scenario s's 2 x 2 table."""
    both, a_only, b_only, neither = SCENARIOS[s]
    a = [1] * both + [1] * a_only + [0] * b_only + [0] * neither
    b = [1] * both + [0] * a_only + [1] * b_only + [0] * neither
    return a, b


def _paired_summary(s):
    """Mean difference B - A, its standard error from the per-question differences, and z."""
    a, b = _scenario_vectors(s)
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    se = math.sqrt(_sample_var(d) / n)
    return _mean(d), se, _mean(d) / se


def _unpaired_se(s):
    a, b = _scenario_vectors(s)
    pa, pb = _mean(a), _mean(b)
    return math.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))


# ------------------------------------------------------------- the problem

@lru_cache(maxsize=None)
def _null_runs():
    """2000 experiments in which both models truly answer right with probability P_NULL."""
    rng = bookrng.stream("statistics.chance_gaps")
    return tuple((_binom(rng, N_Q, P_NULL), _binom(rng, N_Q, P_NULL)) for _ in range(2000))


def chance_gaps():
    runs = _null_runs()
    rows = [(i + 1, a, _pct(a / N_Q), b, _pct(b / N_Q), f"{100 * (b - a) / N_Q:+.1f}") for i, (a, b) in enumerate(runs[:8])]
    out = [_table(["run", "A right", "A %", "B right", "B %", "B - A (points)"], rows)]
    ahead = sum(1 for a, b in runs if b - a >= 4) / len(runs)
    behind = sum(1 for a, b in runs if a - b >= 4) / len(runs)
    out.append("")
    out.append(_table(["over all 2000 runs of two equal models", "share of runs"],
                      [("B ahead by 4 questions (0.8 points) or more", _fx(ahead)),
                       ("A ahead by 4 questions or more", _fx(behind)),
                       ("one of them ahead by 4 or more", _fx(ahead + behind))]))
    return "\n".join(out)


# ---------------------------------------------------------------- estimators

def variance_divisors():
    sigma2 = Fraction(35, 12)
    rows, srows = [], []
    for n in (2, 3):
        tot_n = tot_n1 = Fraction(0)
        tot_s = 0.0
        count = 0
        for roll in product(range(1, 7), repeat=n):
            m = Fraction(sum(roll), n)
            ss = sum((x - m) ** 2 for x in roll)
            tot_n += ss / n
            tot_n1 += ss / (n - 1)
            tot_s += math.sqrt(float(ss / (n - 1)))
            count += 1
        for label, tot in (("n", tot_n), ("n - 1", tot_n1)):
            avg = tot / count
            rows.append((n, label, _fx(float(avg)), _fx(float(sigma2)), _fx(float(avg / sigma2))))
        srows.append((n, _fx(tot_s / count), _fx(math.sqrt(float(sigma2)))))
    return (_table(["sample size n", "divisor", "average over all 6^n samples", "true variance 35/12", "average / true"], rows)
            + "\n\n" + _table(["sample size n", "average of s (divisor n - 1)", "true standard deviation"], srows))


def variance_divisor_simulation():
    rows = []
    for n in (2, 5, 10, 50):
        rng = bookrng.stream(f"statistics.variance_divisor.{n}")
        sum_n = sum_n1 = 0.0
        reps = 20000
        for _ in range(reps):
            x = [int(rng.random() * 6) + 1 for _ in range(n)]
            m = _mean(x)
            ss = sum((v - m) ** 2 for v in x)
            sum_n += ss / n
            sum_n1 += ss / (n - 1)
        sigma2 = 35 / 12
        rows.append((n, _fx(sum_n / reps), _fx(sum_n1 / reps), _fx(sigma2), _fx((n - 1) / n * sigma2)))
    return _table(["n", "average, divisor n", "average, divisor n - 1", "true variance", "(n - 1)/n x true variance"], rows)


# ------------------------------------------------- the sample mean and its SE

@lru_cache(maxsize=None)
def sample_means(n, p=0.7, repeats=1000):
    """`repeats` accuracies, each over n independent questions answered right with probability p."""
    rng = bookrng.stream(f"statistics.sampling_distribution.{n}")
    return tuple(_binom(rng, n, p) / n for _ in range(repeats))


SAMPLE_SIZES = (25, 100, 400, 1600)


def sampling_distribution():
    rows = []
    for n in SAMPLE_SIZES:
        m = sample_means(n)
        sd = math.sqrt(sum((v - _mean(m)) ** 2 for v in m) / len(m))
        pred = math.sqrt(0.7 * 0.3 / n)
        rows.append((n, _fx(_mean(m)), _fx(sd), _fx(pred), _fx(sd / pred, 3)))
    return _table(["questions n", "average of 1000 accuracies", "spread of the 1000 accuracies", "sqrt(p(1-p)/n)", "ratio"], rows)


def standard_errors():
    p = 352 / 500
    rows = [(n, _fx(math.sqrt(p * (1 - p) / n)), _fx(Z95 * math.sqrt(p * (1 - p) / n) * 100, 2)) for n in (100, 500, 2000, 10000)]
    need = math.ceil((Z95 / 0.01) ** 2 * p * (1 - p))
    note = (f"Accuracy {_fx(p, 3)}. Questions needed for a 95% half-width of 1 point: "
            f"({_fx(Z95, 3)} / 0.01)^2 x {_fx(p, 3)} x {_fx(1 - p, 3)} = {need}.")
    return _table(["questions n", "standard error", "95% half-width (points)"], rows) + "\n\n" + note


# ---------------------------------------------------------------- z-scores

AREA = [62, 85, 120, 74, 150, 95, 58, 108]     # square metres
AGE = [12, 40, 7, 25, 3, 33, 55, 18]           # years


def _scale(col):
    m = _mean(col)
    return m, math.sqrt(_sample_var(col))


def feature_scales():
    rows = [(i + 1, AREA[i], AGE[i]) for i in range(len(AREA))]
    ma, sa = _scale(AREA)
    mg, sg = _scale(AGE)
    out = [_table(["house", "area (m^2)", "age (years)"], rows), ""]
    out.append(_table(["feature", "mean", "standard deviation (divisor n - 1)"],
                      [("area", _fx(ma, 2), _fx(sa, 2)), ("age", _fx(mg, 2), _fx(sg, 2))]))
    da, dg = AREA[0] - AREA[1], AGE[0] - AGE[1]
    out.append("")
    out.append(_table(["raw features, houses 1 and 2", "squared difference", "share of squared distance"],
                      [("area: 62 - 85", da * da, _fx(da * da / (da * da + dg * dg), 3)),
                       ("age: 12 - 40", dg * dg, _fx(dg * dg / (da * da + dg * dg), 3)),
                       ("distance", _fx(math.sqrt(da * da + dg * dg), 2), "")]))
    return "\n".join(out)


def feature_zscores():
    ma, sa = _scale(AREA)
    mg, sg = _scale(AGE)
    za = [(x - ma) / sa for x in AREA]
    zg = [(x - mg) / sg for x in AGE]
    rows = [(i + 1, _fx(za[i], 3), _fx(zg[i], 3)) for i in range(len(AREA))]
    out = [_table(["house", "z of area", "z of age"], rows), ""]
    out.append(_table(["check", "area", "age"],
                      [("mean of z", _fx(_mean(za), 4), _fx(_mean(zg), 4)),
                       ("standard deviation of z (divisor n - 1)", _fx(math.sqrt(_sample_var(za)), 4), _fx(math.sqrt(_sample_var(zg)), 4))]))
    da, dg = za[0] - za[1], zg[0] - zg[1]
    out.append("")
    out.append(_table(["standardised features, houses 1 and 2", "squared difference", "share of squared distance"],
                      [("area", _fx(da * da, 3), _fx(da * da / (da * da + dg * dg), 3)),
                       ("age", _fx(dg * dg, 3), _fx(dg * dg / (da * da + dg * dg), 3)),
                       ("distance", _fx(math.sqrt(da * da + dg * dg), 3), "")]))
    new_area, new_age = 100, 20
    out.append("")
    out.append(_table(["new house, scaled with the eight-house statistics", "raw", "z"],
                      [("area", new_area, _fx((new_area - ma) / sa, 3)), ("age", new_age, _fx((new_age - mg) / sg, 3))]))
    ratio = [(n, _fx(math.sqrt(n / (n - 1)), 4)) for n in (8, 100, 10000)]
    out.append("")
    out.append(_table(["n", "ratio of the two standard deviations, sqrt(n/(n-1))"], ratio))
    return "\n".join(out)


# ------------------------------------------------------ confidence intervals

def central_areas():
    rows = [(z, _fx(2 * _cdf(z) - 1, 4)) for z in (1.0, 1.645, 1.96, 2.576, 3.0)]
    return _table(["z", "area within -z and z"], rows)


def accuracy_intervals():
    rows = []
    for name, k in (("A", 352), ("B", 356)):
        p = k / N_Q
        se = math.sqrt(p * (1 - p) / N_Q)
        rows.append((name, k, _pct(p), _fx(se, 4), _pct(p - Z95 * se), _pct(p + Z95 * se)))
    return _table(["model", "right of 500", "accuracy %", "standard error", "95% lower %", "95% upper %"], rows)


def _wald(k, n):
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return p - Z95 * se, p + Z95 * se


@lru_cache(maxsize=None)
def coverage_intervals(p, n, count, tag):
    """`count` intervals from independent experiments of n questions at true accuracy p."""
    rng = bookrng.stream(f"statistics.coverage.{tag}")
    out = []
    for _ in range(count):
        lo, hi = _wald(_binom(rng, n, p), n)
        out.append((lo, hi))
    return tuple(out)


def coverage_table():
    rows = []
    for p in (0.7, 0.1):
        for n in (20, 100, 500):
            ints = coverage_intervals(p, n, 2000, f"table.{p}.{n}")
            hit = sum(lo <= p <= hi for lo, hi in ints) / len(ints)
            flat = sum(hi - lo == 0 for lo, hi in ints) / len(ints)
            rows.append((p, n, _fx(hit, 3), _fx(flat, 3)))
    return _table(["true accuracy", "questions n", "share of 2000 intervals containing it", "share with zero width"], rows)


FIG_P, FIG_N, FIG_COUNT = 0.7, 100, 100


def _figure_intervals():
    return coverage_intervals(FIG_P, FIG_N, FIG_COUNT, "figure.a")


def coverage_figure_summary():
    ints = _figure_intervals()
    miss = sum(not (lo <= FIG_P <= hi) for lo, hi in ints)
    return _table(["experiments", "questions each", "true accuracy", "intervals that missed it"],
                  [(FIG_COUNT, FIG_N, FIG_P, miss)])


def _t_multiplier(df):
    """The z-like multiplier holding 95% of a Student t density: integrate the density, bisect."""
    def pdf(t):
        c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
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
        if central(mid) < 0.95:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_multipliers():
    rows = [(df, _fx(_t_multiplier(df), 3)) for df in (1, 2, 5, 10, 30)]
    rows.append(("infinite (Gaussian)", _fx(Z95, 3)))
    return _table(["degrees of freedom (n - 1)", "multiplier for 95%"], rows)


# --------------------------------------------------------- hypothesis tests

def unpaired_test():
    k_a, k_b = 352, 356
    pa, pb = k_a / N_Q, k_b / N_Q
    pool = (k_a + k_b) / (2 * N_Q)
    se0 = math.sqrt(2 * pool * (1 - pool) / N_Q)
    z = (pb - pa) / se0
    runs = _null_runs()
    sim = sum(1 for a, b in runs if abs(b - a) >= k_b - k_a) / len(runs)
    rows = [("accuracy A, B", f"{_fx(pa, 3)}, {_fx(pb, 3)}"),
            ("difference B - A", _fx(pb - pa, 4)),
            ("pooled accuracy (the null world)", _fx(pool, 3)),
            ("standard error of the difference under the null", _fx(se0, 5)),
            ("test statistic z", _fx(z, 3)),
            ("two-sided p-value from the Gaussian", _fx(_p_two_sided(z), 3)),
            ("share of 2000 simulated null experiments with |gap| >= 4 questions", _fx(sim, 3))]
    return _table(["quantity", "value"], rows, "ll")


def test_error_rates():
    rng = bookrng.stream("statistics.test_error_rates")
    reps = 1000
    rows = []
    for label, p_a, p_b in (("both truly 0.700", 0.700, 0.700), ("B truly 3 points better", 0.700, 0.730)):
        rej = 0
        for _ in range(reps):
            ka, kb = _binom(rng, N_Q, p_a), _binom(rng, N_Q, p_b)
            pool = (ka + kb) / (2 * N_Q)
            if pool in (0.0, 1.0):
                continue
            z = (kb - ka) / N_Q / math.sqrt(2 * pool * (1 - pool) / N_Q)
            rej += abs(z) > Z95
        rows.append((label, N_Q, _fx(rej / reps, 3)))
    pw = []
    for n in (500, 2000, 8000):
        se_n = math.sqrt(2 * 0.7 * 0.3 / n)
        shift = 0.03 / se_n
        power = (1 - _cdf(Z95 - shift)) + _cdf(-Z95 - shift)
        pw.append((n, _fx(se_n, 4), _fx(power, 3)))
    return (_table(["truth", "questions", "share of 1000 experiments declared different"], rows, "lrr")
            + "\n\n" + _table(["questions n", "standard error of the difference", "chance of detecting a 3-point gain"], pw))


def false_discoveries():
    ideas, share_null, alpha, power = 1000, 0.9, 0.05, 0.5
    null = round(ideas * share_null)
    real = ideas - null
    false_pos = null * alpha
    true_pos = real * power
    rows = [("ideas tested", ideas), ("no better than the baseline (90 percent)", null), ("really better (10 percent)", real),
            ("false positives: 900 x 0.05", _fx(false_pos, 0)), ("true positives: 100 x 0.5", _fx(true_pos, 0)),
            ("significant results", _fx(false_pos + true_pos, 0)),
            ("share of significant results that are false", _fx(false_pos / (false_pos + true_pos), 3))]
    return _table(["step", "count"], rows, "lr")


# -------------------------------------------------------------- paired tests

def paired_standard_errors():
    rows = []
    for s in (1, 2):
        a, b = _scenario_vectors(s)
        n = len(a)
        pa, pb = _mean(a), _mean(b)
        va, vb = pa * (1 - pa), pb * (1 - pb)
        cov = sum(x * y for x, y in zip(a, b)) / n - pa * pb
        d = [y - x for x, y in zip(a, b)]
        vd_direct = sum(v * v for v in d) / n - _mean(d) ** 2
        vd_formula = va + vb - 2 * cov
        rho = cov / math.sqrt(va * vb)
        mean_d, se_pair, _ = _paired_summary(s)
        rows.append((s, _fx(pa, 3), _fx(pb, 3), _fx(cov, 4), _fx(rho, 3), _fx(va + vb, 4), _fx(vd_formula, 4), _fx(vd_direct, 4),
                     _fx(_unpaired_se(s), 4), _fx(se_pair, 4)))
    return _table(["scenario", "acc A", "acc B", "Cov[A, B]", "correlation", "Var A + Var B", "Var A + Var B - 2 Cov",
                   "Var of per-question difference", "SE if unpaired", "SE paired"], rows)


def three_tests():
    rows = []
    for s in (1, 2):
        both, a_only, b_only, neither = SCENARIOS[s]
        a, b = _scenario_vectors(s)
        diff = _mean(b) - _mean(a)
        z_un = diff / _unpaired_se(s)
        _, _, z_pair = _paired_summary(s)
        n_disc = a_only + b_only
        rows.append((s, f"{_pct(_mean(a))} / {_pct(_mean(b))}", _fx(diff, 3),
                     _fx(z_un, 2), _fx(_p_two_sided(z_un), 3),
                     _fx(z_pair, 2), _fx(_p_two_sided(z_pair), 4),
                     f"{b_only} of {n_disc}", _fx(_tail_ge(n_disc, b_only), 4), _fx(_sign_p_two_sided(n_disc, b_only), 4)))
    return _table(["scenario", "accuracy A / B %", "B - A", "unpaired z", "unpaired p", "paired z", "paired p",
                   "B wins of discordant", "sign test one-sided p", "sign test two-sided p"], rows)


def table_two_by_two():
    rows = []
    for s in (1, 2):
        both, a_only, b_only, neither = SCENARIOS[s]
        rows.append((s, both, a_only, b_only, neither, both + a_only, both + b_only, a_only + b_only))
    return _table(["scenario", "both right", "only A right", "only B right", "both wrong", "A right", "B right", "discordant"], rows)


def pairing_vs_correlation():
    n, p = N_Q, 0.7
    rows = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        v = p * (1 - p)
        pmf = [p * p + rho * v, v * (1 - rho), v * (1 - rho), (1 - p) ** 2 + rho * v]   # both, A only, B only, neither
        rng = bookrng.stream(f"statistics.pairing.{rho}")
        reps = 1000
        diffs = []
        for _ in range(reps):
            net = 0
            for _ in range(n):
                u = rng.random()
                if u < pmf[0]:
                    continue
                if u < pmf[0] + pmf[1]:
                    net -= 1
                elif u < pmf[0] + pmf[1] + pmf[2]:
                    net += 1
            diffs.append(net / n)
        m = _mean(diffs)
        sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / reps)
        rows.append((rho, _fx(pmf[1] + pmf[2], 3), _fx(math.sqrt(2 * v * (1 - rho) / n), 4), _fx(sd, 4)))
    return _table(["correlation of the two models' answers", "share of questions where they disagree", "SE of the gap, formula", "spread of 1000 simulated gaps"], rows)


def sign_test_floor():
    rows = []
    for n in (3, 4, 5, 6, 8, 10, 20, 44, 100):
        k = next((k for k in range(n + 1) if _tail_ge(n, k) <= 0.05), None)
        if k is None:
            rows.append((n, "none", "", ""))
        else:
            rows.append((n, k, _fx(_tail_ge(n, k), 4), _fx(k / n, 3)))
    return _table(["discordant pairs n", "smallest k with p <= 0.05", "p at that k", "share of wins needed"], rows)


# ---------------------------------------------------------------- bootstrap

@lru_cache(maxsize=None)
def bootstrap_replicates(s, resamples=2000):
    """Mean per-question difference B - A on each resample of the questions, with replacement."""
    a, b = _scenario_vectors(s)
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    rng = bookrng.stream(f"statistics.bootstrap.{s}")
    out = []
    for _ in range(resamples):
        r = rng.random
        out.append(sum(d[int(r() * n)] for _ in range(n)) / n)
    return tuple(out)


def _percentile(sorted_vals, q):
    """Value at fraction q of a sorted list, by linear interpolation between neighbours."""
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_intervals():
    rows = []
    for s in (1, 2):
        mean_d, se, _ = _paired_summary(s)
        reps = sorted(bootstrap_replicates(s))
        lo, hi = _percentile(reps, 0.025), _percentile(reps, 0.975)
        boot_se = math.sqrt(sum((x - _mean(reps)) ** 2 for x in reps) / len(reps))
        rows.append((s, "paired, Gaussian", _fx(mean_d, 4), _fx(se, 4), _fx(mean_d - Z95 * se, 4), _fx(mean_d + Z95 * se, 4)))
        rows.append((s, "bootstrap, 2000 resamples", _fx(mean_d, 4), _fx(boot_se, 4), _fx(lo, 4), _fx(hi, 4)))
    return _table(["scenario", "method", "B - A", "standard error", "95% lower", "95% upper"], rows, "llrrrr")


def _latencies():
    rng = bookrng.stream("statistics.latency_sample")
    return [round(math.exp(rng.gauss(4.0, 0.6)), 1) for _ in range(25)]


def _median(v):
    s = sorted(v)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def bootstrap_median():
    x = _latencies()
    n = len(x)
    rng = bookrng.stream("statistics.bootstrap_median")
    meds, means = [], []
    for _ in range(2000):
        r = [x[int(rng.random() * n)] for _ in range(n)]
        meds.append(_median(r))
        means.append(_mean(r))
    meds.sort()
    means.sort()
    se_mean = math.sqrt(_sample_var(x) / n)
    out = ["Sorted latencies (ms): " + ", ".join(f"{v:.1f}" for v in sorted(x)), ""]
    out.append(_table(["statistic", "estimate (ms)", "lower end of 95 percent interval", "upper end", "how"],
                      [("mean", _fx(_mean(x), 1), _fx(_mean(x) - Z95 * se_mean, 1), _fx(_mean(x) + Z95 * se_mean, 1), "standard error s/sqrt(n)"),
                       ("mean", _fx(_mean(x), 1), _fx(_percentile(means, 0.025), 1), _fx(_percentile(means, 0.975), 1), "bootstrap"),
                       ("median", _fx(_median(x), 1), _fx(_percentile(meds, 0.025), 1), _fx(_percentile(meds, 0.975), 1), "bootstrap")],
                      "lrrrl"))
    return "\n".join(out)


CL_SEQ, CL_FRAMES = 15, 16


def _cluster_data(rng):
    """15 sequences of 16 frames; each sequence has its own success probability."""
    return [[1 if rng.random() < p else 0 for _ in range(CL_FRAMES)] for p in (rng.uniform(0.3, 0.9) for _ in range(CL_SEQ))]


def cluster_bootstrap():
    truth_rng = bookrng.stream("statistics.cluster_truth")
    overall = []
    for _ in range(2000):
        data = _cluster_data(truth_rng)
        overall.append(sum(map(sum, data)) / (CL_SEQ * CL_FRAMES))
    m = _mean(overall)
    true_sd = math.sqrt(sum((v - m) ** 2 for v in overall) / len(overall))
    data = _cluster_data(bookrng.stream("statistics.cluster_data"))
    frames = [v for seq in data for v in seq]
    rng = bookrng.stream("statistics.cluster_resample")
    nf = len(frames)
    naive, clustered = [], []
    for _ in range(1000):
        naive.append(sum(frames[int(rng.random() * nf)] for _ in range(nf)) / nf)
        clustered.append(sum(sum(data[int(rng.random() * CL_SEQ)]) for _ in range(CL_SEQ)) / nf)

    def sd(v):
        mm = _mean(v)
        return math.sqrt(sum((x - mm) ** 2 for x in v) / len(v))
    rows = [("spread of the accuracy over 2000 fresh experiments (the target)", _fx(true_sd, 4), ""),
            ("bootstrap resampling frames", _fx(sd(naive), 4), _fx(sd(naive) / true_sd, 2)),
            ("bootstrap resampling whole sequences", _fx(sd(clustered), 4), _fx(sd(clustered) / true_sd, 2))]
    return _table(["method", "standard error of the accuracy", "ratio to the target"], rows, "lrr")


# ------------------------------------------------------ multiple comparisons

def family_error():
    rows = []
    z_cache = {}
    for m in (1, 2, 5, 10, 20, 50):
        alpha_each = 0.05 / m
        z_cache[m] = _z_central(1 - alpha_each)
        rows.append((m, _fx(1 - 0.95 ** m, 3), _fx(alpha_each, 4), _fx(z_cache[m], 2), _fx(1 - (1 - alpha_each) ** m, 4)))
    return _table(["tests m", "chance of at least one false positive at 0.05 each", "Bonferroni level 0.05/m", "z needed", "chance of at least one at that level"], rows)


def best_of_twenty():
    m, runs, p = 20, 400, 0.7
    rng = bookrng.stream("statistics.best_of_twenty")
    tests = hits = bonf_hits = 0
    any_sig = any_bonf = 0
    best_gaps = []
    z_bonf = _z_central(1 - 0.05 / m)
    for _ in range(runs):
        base = [rng.random() < p for _ in range(N_Q)]
        best, sig, bon = -1.0, False, False
        for _ in range(m):
            co = bo = 0
            for bi in base:
                ci = rng.random() < p
                if ci and not bi:
                    co += 1
                elif bi and not ci:
                    bo += 1
            gap = (co - bo) / N_Q
            var = (co + bo - (co - bo) ** 2 / N_Q) / (N_Q - 1)
            z = gap / math.sqrt(var / N_Q) if var > 0 else 0.0
            tests += 1
            hits += abs(z) > Z95
            sig = sig or abs(z) > Z95
            bon = bon or abs(z) > z_bonf
            best = max(best, gap)
        any_sig += sig
        any_bonf += bon
        best_gaps.append(best)
    rows = [("a single test: share declared different", _fx(hits / tests, 3), "0.050"),
            ("20 tests, any declared different at 0.05", _fx(any_sig / runs, 3), _fx(1 - 0.95 ** m, 3)),
            (f"20 tests, any declared different at 0.05/20 (z > {_fx(z_bonf, 2)})", _fx(any_bonf / runs, 3), _fx(1 - (1 - 0.0025) ** m, 3)),
            ("average gap of the best-looking of 20 configurations (points)", _fx(100 * _mean(best_gaps), 2), "0 (true gap)")]
    return _table(["400 repeats of: baseline and 20 configurations, all truly 0.70", "simulated", "independent-tests formula"], rows, "lrr")


def holm_example():
    ps = [0.001, 0.012, 0.018, 0.040, 0.300]
    m = len(ps)
    running = 0.0
    rows = []
    for rank, p in enumerate(sorted(ps)):
        mult = m - rank
        running = max(running, min(1.0, mult * p))
        rows.append((rank + 1, _fx(p, 3), mult, _fx(min(1.0, m * p), 3), "yes" if m * p <= 0.05 else "no",
                     _fx(running, 3), "yes" if running <= 0.05 else "no"))
    return _table(["rank", "p-value", "Holm multiplier m - rank + 1", "Bonferroni p x m", "significant (Bonferroni)",
                   "Holm adjusted", "significant (Holm)"], rows)


# --------------------------------------------------------- where brain uses it

def seed_ablation():
    a = [0.135, 0.161, 0.250]
    b = [0.313, 0.276, 0.193]
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    se = math.sqrt(_sample_var(d) / n)
    mult = _t_multiplier(2)
    rows = [("per-seed difference B - A", ", ".join(_fx(v, 3) for v in d)),
            ("mean difference", _fx(_mean(d), 4)),
            ("sample standard deviation (n - 1)", _fx(math.sqrt(_sample_var(d)), 4)),
            ("standard error", _fx(se, 4)),
            ("95 percent interval with the Gaussian 1.960", f"{_fx(_mean(d) - Z95 * se, 3)} to {_fx(_mean(d) + Z95 * se, 3)}"),
            (f"95 percent interval with the 2-degree-of-freedom multiplier {_fx(mult, 3)}",
             f"{_fx(_mean(d) - mult * se, 3)} to {_fx(_mean(d) + mult * se, 3)}"),
            ("best possible one-sided sign-test p with 3 pairs", _fx(_tail_ge(3, 3), 3))]
    return _table(["three seeds, final score of arm B minus arm A", "value"], rows, "ll")


def block_drop_arithmetic():
    se_max = math.sqrt(0.25 / 48)
    out = [_table(["quantity", "value"],
                  [("largest standard error of a mean of 48 pass/fail probes: sqrt(0.5 x 0.5 / 48)", _fx(se_max, 4)),
                   ("a block drop of 0.20 in standard errors", _fx(0.20 / se_max, 2))], "lr"), ""]
    rows = []
    for k in (5, 10, 15, 20, 50):
        pooled = Fraction(30, 100) / k
        rows.append((k, _fx(float(pooled), 4), "yes" if pooled <= Fraction(2, 100) else "no"))
    out.append(_table(["earlier cycles k", "pooled mean drop when one cycle falls by 0.30", "within the 0.02 budget (the pooled bar is silent)"], rows))
    return "\n".join(out)
