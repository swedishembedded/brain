# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Survival Analysis and Competing Risks".

Plain Python 3, standard library only, written from the chapter's own description: what dropping or
mis-labelling censored rows does to a mean lifetime and to a probability of failure, the Kaplan-Meier
estimator with its tie rule, Greenwood variance and Nelson-Aalen cumulative hazard worked by hand and on a
seeded cohort, hazards and cumulative hazards of three piecewise-constant shapes, a discrete-time hazard
model fitted by logistic likelihood on person-period rows (and shown to reproduce Kaplan-Meier when it has
one intercept per interval), cause-specific hazards with the cumulative incidence of the Aalen-Johansen
estimator against the naive one minus Kaplan-Meier, a count with no exposure behind it, the concordance
index by hand and on a cohort, the inverse-probability-of-censoring-weighted Brier score against the Brier
score an oracle computes from the uncensored lifetimes, and the closed-form survival and cumulative
incidence of a piecewise-constant hazard table at any horizon, and the answers to two exercises on the opening cohort. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: survival_analysis.<name> -->` markers. Underscore functions and the constants are shared with
the chapter's figures, so a picture and the table beside it cannot disagree.

Swedish Embedded AB implements solutions for validating risk models whose outcomes arrive late, censored
and competing. If your team needs expertise in time-to-event prediction, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import statistics
from functools import lru_cache

import bookrng
from logistic_regression import _solve, sigmoid
from time_series import _table

Z975 = statistics.NormalDist().inv_cdf(0.975)   # 1.959964, the library inverse of the Gaussian distribution function

# ------------------------------------------------------------------ constants

COHORT_NAME = "survival_analysis.cohort"
COHORT_N, COHORT_RATE, STUDY_END = 400, 0.1, 24.0   # units, true failure rate per month, months from the start to the day the log is read

DISCRETE_NAME = "survival_analysis.discrete"
DISCRETE_N, DISCRETE_J = 400, 8
DISCRETE_A0, DISCRETE_A_STEP, DISCRETE_B = -2.6, 0.15, 0.8   # alpha_j = -2.6 + 0.15 (j - 1); beta = 0.8
NEWTON_STEPS = 10

COMPETE_NAME = "survival_analysis.compete"
COMPETE_N, COMPETE_RATES, COMPETE_CENSOR = 500, (0.06, 0.10), (6.0, 24.0)

SCORE_NAME = "survival_analysis.scores"
SCORE_N, SCORE_BASE, SCORE_BETA = 300, 0.08, (0.7, -0.5)
# the censoring scenarios: C = lo + (hi - lo) v, and the truncation time t* of the concordance
SCORE_SCENARIOS = (((3.0, 15.0), 12.0), ((2.0, 8.0), 6.0))
SCORE_TRUNC = 12.0
REPLICATES_NAME, REPLICATES_N, REPLICATES = "survival_analysis.replicates", 200, 200
SCORE_HORIZONS = (3.0, 6.0, 9.0, 12.0)
SCORE_GRID = tuple(float(t) for t in range(1, 13))

# the small cohort of the hand-worked Kaplan-Meier table: (time, 1 = event, 0 = censored)
HAND = ((1, 1), (2, 1), (2, 0), (3, 1), (3, 1), (4, 0), (5, 1), (6, 0), (6, 0), (7, 1), (8, 0), (9, 0))
# the six subjects of the concordance example: (time, 1 = event, risk)
PAIRS = ((2, 1, 0.8), (3, 1, 0.5), (3, 0, 0.6), (5, 1, 0.5), (6, 0, 0.2), (8, 1, 0.1))

# three piecewise-constant hazards on the pieces [0, 2), [2, 6), [6, infinity)
SHAPE_EDGES = (0.0, 2.0, 6.0)
SHAPES = (("falling", (0.30, 0.10, 0.05)), ("constant", (0.10, 0.10, 0.10)), ("rising", (0.04, 0.10, 0.25)))

# hazards per piece of the table of one subject served at any horizon: pieces [0, 2), [2, 5), [5, 10]
KNOTS = (0.0, 2.0, 5.0, 10.0)
CODES = (("cause a (absorbing)", True, (0.05, 0.10, 0.20)),
         ("cause b (absorbing)", True, (0.02, 0.04, 0.08)),
         ("diagnosis (not absorbing)", False, (0.10, 0.10, 0.05)))


# ------------------------------------------------------------------ helpers


def _fx(v, d=3):
    return f"{v:.{d}f}"


def _fs(v, d=3):
    return f"{v:+.{d}f}"


def _gen_line(name, draws):
    seed = bookrng.seed_of(name)
    return f"Generator: the book's SplitMix64, seed {seed:#018x} ({seed}). {draws}"


def _exp_draw(g, rate):
    """The inverse-transform exponential of Volume I, chapter 17: -ln(1 - u) / rate."""
    return -math.log(1.0 - g.uniform()) / rate


# ------------------------------------------------------------------ the estimators


def _km_table(times, events):
    """Rows (t, n, d, c, S, greenwood_sum) at each distinct time, ascending.

    n counts the subjects with time >= t, so a subject censored at t is at risk of an event at t (events
    happen before censorings at a tie); S is the product of 1 - d / n; the Greenwood sum is the running
    total of d / (n (n - d)), which is left unchanged by a time with no event."""
    rows, s, gw = [], 1.0, 0.0
    for t in sorted(set(times)):
        n = sum(1 for x in times if x >= t)
        d = sum(1 for x, e in zip(times, events) if x == t and e)
        c = sum(1 for x, e in zip(times, events) if x == t and not e)
        if d:
            s *= 1.0 - d / n
            if n > d:
                gw += d / (n * (n - d))
        rows.append((t, n, d, c, s, gw))
    return rows


def _km_at(rows, t):
    s = 1.0
    for r in rows:
        if r[0] > t:
            break
        s = r[4]
    return s


def _se_at(rows, t):
    s, gw = 1.0, 0.0
    for r in rows:
        if r[0] > t:
            break
        s, gw = r[4], r[5]
    return s * math.sqrt(gw)


def _rmst(rows, tstar):
    """Area under the Kaplan-Meier step function from 0 to tstar."""
    area, prev_t, prev_s = 0.0, 0.0, 1.0
    for r in rows:
        if r[0] >= tstar:
            break
        area += prev_s * (r[0] - prev_t)
        prev_t, prev_s = r[0], r[4]
    return area + prev_s * (tstar - prev_t)


def _aalen_johansen(times, causes, k, n_causes):
    """Cumulative incidence of cause k as a step function [(t, F_k)], ascending in t.

    causes[i] is 0 for a censored subject and 1 .. n_causes for the cause of an event. At each distinct time,
    F_k grows by S(t-) d_k / n and S becomes S (1 - d / n) with d the events of every cause."""
    out, s, f = [], 1.0, 0.0
    for t in sorted(set(times)):
        n = sum(1 for x in times if x >= t)
        d_k = sum(1 for x, c in zip(times, causes) if x == t and c == k)
        d = sum(1 for x, c in zip(times, causes) if x == t and c > 0)
        f += s * d_k / n
        s *= 1.0 - d / n
        out.append((t, f))
    return out


def _step_at(steps, t, start=0.0):
    v = start
    for x, y in steps:
        if x > t:
            break
        v = y
    return v


# ------------------------------------------------------------------ the censoring cohort


@lru_cache(maxsize=None)
def _cohort():
    """(observed time, event flag, true lifetime) of COHORT_N units, drawn in increasing unit index.

    Unit i draws entry = 24 u (u = uniform()) and then the lifetime -ln(1 - u) / 0.1 (a second uniform()). The
    log is read at month 24, so the unit is followed for 24 - entry months: observed = min(lifetime, 24 - entry),
    and the event flag is 1 when the lifetime is the smaller (or equal)."""
    g = bookrng.generator(COHORT_NAME)
    rows = []
    for _ in range(COHORT_N):
        entry = STUDY_END * g.uniform()
        life = _exp_draw(g, COHORT_RATE)
        follow = STUDY_END - entry
        rows.append((min(life, follow), 1 if life <= follow else 0, life))
    return tuple(rows)


def _cohort_spec():
    return _gen_line(COHORT_NAME, "Unit i = 0 .. 399 in order draws entry = 24 u from one uniform() and then a lifetime "
                     "T = -ln(1 - u) / 0.1 months from a second one; the log is read at month 24, so the unit is followed for 24 - entry "
                     "months, its observed time is min(T, 24 - entry), and it is an event when T is not larger than 24 - entry and censored "
                     "otherwise.")


def censoring_bias():
    rows = _cohort()
    times = [r[0] for r in rows]
    events = [r[1] for r in rows]
    lives = [r[2] for r in rows]
    n, d = len(rows), sum(events)
    failed_times = [t for t, e in zip(times, events) if e]
    exposure = sum(times)
    mean_rows = (("true lifetimes of all 400 units (unobservable: most are longer than the log shows)", sum(lives) / n),
                 ("mean observed time of the units that failed (censored rows dropped)", sum(failed_times) / d),
                 ("mean observed time of all units (censored times treated as failures)", exposure / n),
                 ("total observed time divided by the number of failures", exposure / d))
    t1 = _table(["estimate of the mean lifetime", "months"], [(a, _fx(b, 2)) for a, b in mean_rows], "lr")
    km = _km_table(times, events)
    horizon = 12.0
    cens_before = sum(1 for t, e in zip(times, events) if not e and t < horizon)
    failed_by = sum(1 for t, e in zip(times, events) if e and t <= horizon)
    kept = n - cens_before
    p_rows = (("true failure probability by month 12, 1 - exp(-1.2)", 1.0 - math.exp(-COHORT_RATE * horizon)),
              ("fraction of the 400 lifetimes that are at most 12 (unobservable)", sum(1 for x in lives if x <= horizon) / n),
              ("failures by month 12 over all 400 units (censored count as survivors)", failed_by / n),
              (f"failures by month 12 over the {kept} units not censored before month 12 (drop them)", failed_by / kept),
              ("one minus the Kaplan-Meier estimate at month 12", 1.0 - _km_at(km, horizon)))
    t2 = _table(["estimate of P(T <= 12)", "value"], [(a, _fx(b, 4)) for a, b in p_rows], "lr")
    head = (_cohort_spec() + f" Of the 400 units {d} fail inside the window and {n - d} are censored; {cens_before} are censored "
            f"before month 12. Mean lifetime of the model that generated them: 1 / 0.1 = 10 months.\n\n")
    return head + t1 + "\n\n" + t2


# ------------------------------------------------------------------ Kaplan-Meier by hand


def km_by_hand():
    times = [t for t, _ in HAND]
    events = [e for _, e in HAND]
    rows = _km_table(times, events)
    out, h = [], 0.0
    for t, n, d, c, s, gw in rows:
        h += d / n
        se = s * math.sqrt(gw)
        lo, hi = max(0.0, s - Z975 * se), min(1.0, s + Z975 * se)
        out.append((t, n, d, c, _fx(1.0 - d / n, 4), _fx(s, 4), _fx(gw, 4), _fx(se, 4), f"[{lo:.3f}, {hi:.3f}]", _fx(h, 4),
                    _fx(-math.log(s), 4)))
    head = ("The twelve subjects as (time, event) with event 1 for a failure and 0 for a censoring: "
            + ", ".join(f"({t}, {e})" for t, e in HAND)
            + ". At each distinct time t: n is the number of subjects with time at least t (a subject censored at t is still at risk at "
              "t), d the failures at t, c the censorings at t; the factor is 1 - d / n, S the running product, W the running sum of "
              "d / (n (n - d)), SE = S sqrt(W), the band S +- 1.959964 SE clipped to [0, 1], H the running sum of d / n and -ln S "
              "beside it. Times with no failure change none of S, W or H.\n\n")
    t = _table(["t", "n", "d", "c", "1 - d/n", "S(t)", "W", "SE", "95% band", "H(t)", "-ln S(t)"], out)
    med = next((r[0] for r in rows if r[4] <= 0.5), None)
    tail = (f"\n\nThe median survival time is the first time at which S is at most 0.5: t = {med}. The restricted mean survival time up to "
            f"t* = 9, the area under the step function from 0 to 9, is {_rmst(rows, 9.0):.4f}.")
    return head + t + tail


def km_cohort():
    rows = _cohort()
    times = [r[0] for r in rows]
    events = [r[1] for r in rows]
    km = _km_table(times, events)
    lives = [r[2] for r in rows]
    out = []
    for t in (3.0, 6.0, 12.0, 18.0, 24.0):
        s = _km_at(km, t)
        se = _se_at(km, t)
        naive = sum(1 for e, x in zip(events, times) if e and x > t) / sum(events)   # share of failed units' times above t
        out.append((t, _fx(math.exp(-COHORT_RATE * t), 4), _fx(sum(1 for x in lives if x > t) / len(lives), 4), _fx(s, 4), _fx(se, 4),
                    f"[{max(0.0, s - Z975 * se):.4f}, {min(1.0, s + Z975 * se):.4f}]", _fx(naive, 4)))
    med = next(r[0] for r in km if r[4] <= 0.5)
    head = (_cohort_spec() + " The estimator is the one of the previous table (events before censorings at a tie) with the Greenwood "
            "standard error and the clipped 95 percent band. Last column: the share of the units that failed whose failure time is "
            "above t, which is what dropping every censored row leaves.\n\n")
    tab = _table(["t", "true exp(-0.1 t)", "share of the 400 lifetimes above t", "Kaplan-Meier", "SE", "95% band",
                  "censored rows dropped"], out)
    tail = (f"\n\nMedian of the Kaplan-Meier curve: {med:.4f} months (true median of the model: ln 2 / 0.1 = {math.log(2) / COHORT_RATE:.4f}). "
            f"Restricted mean up to 24 months: Kaplan-Meier {_rmst(km, 24.0):.4f}, true {10 * (1 - math.exp(-2.4)):.4f}.")
    return head + tab + tail


def km_curve_points():
    """Dense points of the cohort's Kaplan-Meier step function and of the censored-rows-dropped curve, for the figure."""
    rows = _cohort()
    times = [r[0] for r in rows]
    events = [r[1] for r in rows]
    km = _km_table(times, events)
    xs, ys, prev = [0.0], [1.0], 1.0
    for r in km:
        xs += [r[0], r[0]]
        ys += [prev, r[4]]
        prev = r[4]
    xs.append(STUDY_END)
    ys.append(prev)
    failed = sorted(t for t, e in zip(times, events) if e)
    dx, dy, k = [0.0], [1.0], 0
    for t in failed:
        k += 1
        dx += [t, t]
        dy += [1.0 - (k - 1) / len(failed), 1.0 - k / len(failed)]
    dx.append(STUDY_END)
    dy.append(dy[-1])
    return (xs, ys), (dx, dy)


# ------------------------------------------------------------------ hazards and cumulative hazards


def _pw_cum_hazard(hazards, t):
    total = 0.0
    for p, h in enumerate(hazards):
        lo = SHAPE_EDGES[p]
        hi = SHAPE_EDGES[p + 1] if p + 1 < len(SHAPE_EDGES) else math.inf
        total += h * max(0.0, min(t, hi) - lo)
    return total


def _pw_hazard(hazards, t):
    p = sum(1 for e in SHAPE_EDGES[1:] if t >= e)
    return hazards[p]


def _pw_mean(hazards):
    """Mean lifetime = integral of S, piece by piece: S(start) (1 - exp(-h w)) / h, the last piece to infinity."""
    total, s = 0.0, 1.0
    for p, h in enumerate(hazards):
        if p + 1 < len(hazards):
            w = SHAPE_EDGES[p + 1] - SHAPE_EDGES[p]
            total += s * (1.0 - math.exp(-h * w)) / h
            s *= math.exp(-h * w)
        else:
            total += s / h
    return total


def hazard_shapes():
    rows = []
    for name, hz in SHAPES:
        for t in (1.0, 4.0, 8.0):
            lam = _pw_hazard(hz, t)
            cum = _pw_cum_hazard(hz, t)
            s = math.exp(-cum)
            step = 1.0 - math.exp(-(_pw_cum_hazard(hz, t + 1.0) - cum))
            rows.append((name, t, _fx(lam, 2), _fx(cum, 3), _fx(s, 4), _fx(lam * s, 4), _fx(step, 4)))
    head = ("Three hazards, each constant on the pieces [0, 2), [2, 6) and [6, infinity) with the values falling 0.30, 0.10, 0.05; "
            "constant 0.10, 0.10, 0.10; rising 0.04, 0.10, 0.25 (per month). lambda(t) is the hazard at t, Lambda(t) = the sum over "
            "pieces of the hazard times the length of the piece below t, S = exp(-Lambda), f = lambda S, and the last column is the "
            "probability of failing in (t, t + 1] given survival to t, 1 - exp(-(Lambda(t + 1) - Lambda(t))).\n\n")
    t = _table(["shape", "t", "lambda(t)", "Lambda(t)", "S(t)", "f(t)", "P(fail in (t, t+1] | T > t)"], rows, "lrrrrrr")
    means = ", ".join(f"{name} {_pw_mean(hz):.3f}" for name, hz in SHAPES)
    return head + t + f"\n\nMean lifetime, the area under S, in months: {means}."


# ------------------------------------------------------------------ discrete-time hazards on person-period rows


@lru_cache(maxsize=None)
def _discrete_cohort():
    """(x, last interval observed, event flag) per subject.

    Subject i draws, in order: u = uniform() (x = 1 when u < 0.5), c = 1 + integer(8) (observed through the end
    of interval c), then eight uniform() draws u_1 .. u_8; the event interval is the first j with
    u_j < sigmoid(alpha_j + 0.8 x), alpha_j = -2.6 + 0.15 (j - 1). The event counts when j <= c."""
    g = bookrng.generator(DISCRETE_NAME)
    rows = []
    for _ in range(DISCRETE_N):
        x = 1 if g.uniform() < 0.5 else 0
        c = 1 + g.integer(DISCRETE_J)
        ev = None
        for j in range(1, DISCRETE_J + 1):
            u = g.uniform()
            if ev is None and u < sigmoid(DISCRETE_A0 + DISCRETE_A_STEP * (j - 1) + DISCRETE_B * x):
                ev = j
        if ev is not None and ev <= c:
            rows.append((x, ev, 1))
        else:
            rows.append((x, c, 0))
    return tuple(rows)


def _person_period(cohort):
    """Rows (j, x, y): subject (x, last, event) gives rows j = 1 .. last with y = 0, and y = 1 on the last row if event."""
    out = []
    for x, last, ev in cohort:
        for j in range(1, last + 1):
            out.append((j, x, 1 if (ev and j == last) else 0))
    return out


def _newton_logistic(rows, use_x):
    """Maximum likelihood of the logistic model with one intercept per interval (and beta x when use_x), by Newton's
    method from all-zero parameters, NEWTON_STEPS steps. Sums over rows in row order. Returns (theta, covariance
    diagonal, largest absolute gradient entry at the end)."""
    dim = DISCRETE_J + (1 if use_x else 0)
    theta = [0.0] * dim

    def feats(j, x):
        v = [0.0] * dim
        v[j - 1] = 1.0
        if use_x:
            v[DISCRETE_J] = float(x)
        return v

    grad = [0.0] * dim
    hess = [[0.0] * dim for _ in range(dim)]
    for it in range(NEWTON_STEPS + 1):
        grad = [0.0] * dim
        hess = [[0.0] * dim for _ in range(dim)]
        for j, x, y in rows:
            f = feats(j, x)
            p = sigmoid(sum(a * b for a, b in zip(theta, f)))
            w = p * (1.0 - p)
            for a in range(dim):
                if f[a] == 0.0:
                    continue
                grad[a] += (y - p) * f[a]
                for b in range(dim):
                    hess[a][b] += w * f[a] * f[b]
        if it == NEWTON_STEPS:
            break
        step = _solve(hess, grad)
        theta = [a + b for a, b in zip(theta, step)]
    var = [_solve(hess, [1.0 if i == k else 0.0 for i in range(dim)])[k] for k in range(dim)]
    return theta, var, max(abs(v) for v in grad)


def _discrete_fits():
    rows = _person_period(_discrete_cohort())
    fits = _newton_logistic(rows, False), _newton_logistic(rows, True)
    assert all(f[2] < 1e-9 for f in fits), "Newton did not converge"
    return rows, fits[0], fits[1]


def discrete_hazard():
    cohort = _discrete_cohort()
    rows, (th0, _, g0), (th1, var1, g1) = _discrete_fits()
    n_ev = sum(c[2] for c in cohort)
    assert max(g0, g1) < 1e-9, "Newton's method has not converged"
    head = (_gen_line(DISCRETE_NAME, "Subject i = 0 .. 399 in order draws u = uniform() (x = 1 when u < 0.5), then c = 1 + integer(8) "
                      "(the subject is observed through the end of interval c), then eight uniform() draws u_1 .. u_8; the event "
                      "interval is the first j with u_j < sigmoid(alpha_j + 0.8 x), alpha_j = -2.6 + 0.15 (j - 1), and the event counts "
                      "when j <= c, otherwise the subject is censored after interval c. ")
            + (f"The {len(cohort)} subjects give {len(rows)} person-period rows (j, x, y): a subject observed through interval L gives the "
               f"rows j = 1 .. L with y = 0, except that the row j = L has y = 1 when the event happened in interval L. There are {n_ev} events. "
               f"The model is logit p_j = alpha_j (one intercept per interval) or alpha_j + beta x, fitted by Newton's method from "
               f"all-zero parameters, {NEWTON_STEPS} steps; for both models the largest gradient entry after them is below 1e-9.\n\n"))
    counts = []
    for j in range(1, DISCRETE_J + 1):
        n_j = sum(1 for r in rows if r[0] == j)
        d_j = sum(1 for r in rows if r[0] == j and r[2])
        n0 = sum(1 for r in rows if r[0] == j and r[1] == 0)
        n1 = n_j - n0
        p_true0 = sigmoid(DISCRETE_A0 + DISCRETE_A_STEP * (j - 1))
        counts.append((j, n_j, d_j, _fx(d_j / n_j, 4), _fx(sigmoid(th0[j - 1]), 4), _fx(p_true0, 4),
                       _fx(sigmoid(th1[j - 1]), 4), _fx(sigmoid(th1[j - 1] + th1[DISCRETE_J]), 4)))
    dev = max(abs(sigmoid(th0[j]) - sum(1 for r in rows if r[0] == j + 1 and r[2]) / sum(1 for r in rows if r[0] == j + 1))
              for j in range(DISCRETE_J))
    t1 = _table(["interval j", "at risk n_j", "events d_j", "d_j / n_j", "p_j fitted, intercepts only", "p_j true at x = 0",
                 "p_j fitted at x = 0", "p_j fitted at x = 1"], counts)
    beta, se = th1[DISCRETE_J], math.sqrt(var1[DISCRETE_J])
    surv = []
    for x in (0, 1):
        s = 1.0
        for j in range(1, DISCRETE_J + 1):
            s *= 1.0 - sigmoid(th1[j - 1] + th1[DISCRETE_J] * x)
        surv.append(s)
    km_by_x = []
    for x in (0, 1):
        sub = [(last, ev) for xx, last, ev in cohort if xx == x]
        rws = _km_table([float(a) for a, _ in sub], [b for _, b in sub])
        km_by_x.append(_km_at(rws, float(DISCRETE_J)))
    true_s = []
    for x in (0, 1):
        s = 1.0
        for j in range(1, DISCRETE_J + 1):
            s *= 1.0 - sigmoid(DISCRETE_A0 + DISCRETE_A_STEP * (j - 1) + DISCRETE_B * x)
        true_s.append(s)
    assert dev < 1e-9, "the intercept-only fit must reproduce d_j / n_j"
    tail = (f"\n\nWith intercepts only, the largest gap between the fitted hazard sigmoid(alpha_j) and the count ratio d_j / n_j is "
            f"below 1e-9. With the covariate: beta = {beta:.4f} with standard error {se:.4f} (the square root of the diagonal entry of the "
            f"inverse of the Hessian), true 0.8; exp(beta) = {math.exp(beta):.3f}. Probability of surviving all eight intervals, "
            f"x = 0 and x = 1: true {true_s[0]:.4f} and {true_s[1]:.4f}; the product of the fitted 1 - p_j, {surv[0]:.4f} and {surv[1]:.4f}; "
            f"Kaplan-Meier within each group (interval index as the time), {km_by_x[0]:.4f} and {km_by_x[1]:.4f}.")
    return head + t1 + tail


# ------------------------------------------------------------------ competing risks


@lru_cache(maxsize=None)
def _compete_cohort():
    """(observed time, 0 for censored or 1 / 2 for the cause of the event, true time) per subject.

    Subject i draws u1, u2, u3 = three uniform() in this order: T = -ln(1 - u1) / 0.16, cause 1 if u2 < 0.375 and
    cause 2 otherwise, censoring time C = 6 + 18 u3. The event is seen when T <= C."""
    g = bookrng.generator(COMPETE_NAME)
    total = sum(COMPETE_RATES)
    rows = []
    for _ in range(COMPETE_N):
        t = _exp_draw(g, total)
        cause = 1 if g.uniform() < COMPETE_RATES[0] / total else 2
        c = COMPETE_CENSOR[0] + (COMPETE_CENSOR[1] - COMPETE_CENSOR[0]) * g.uniform()
        rows.append(((t, cause) if t <= c else (c, 0)) + (t,))
    return tuple(rows)


def _compete_spec():
    return _gen_line(COMPETE_NAME, "Subject i = 0 .. 499 in order draws three uniform() values u1, u2, u3: T = -ln(1 - u1) / 0.16 months, "
                     "the cause is 1 (cause a) when u2 < 0.375 = 0.06 / 0.16 and 2 (cause b) otherwise, and the censoring time is "
                     "C = 6 + 18 u3. The subject is observed at min(T, C), with its cause if T <= C and censored otherwise.")


def _compete_curves():
    rows = _compete_cohort()
    times = [r[0] for r in rows]
    causes = [r[1] for r in rows]
    af_a = _aalen_johansen(times, causes, 1, 2)
    af_b = _aalen_johansen(times, causes, 2, 2)
    km_a = _km_table(times, [c == 1 for c in causes])
    km_b = _km_table(times, [c == 2 for c in causes])
    km_all = _km_table(times, [c > 0 for c in causes])
    return af_a, af_b, km_a, km_b, km_all


def _true_cif(k, t):
    lam = sum(COMPETE_RATES)
    return COMPETE_RATES[k] / lam * (1.0 - math.exp(-lam * t))


def competing_risks():
    rows = _compete_cohort()
    af_a, af_b, km_a, km_b, km_all = _compete_curves()
    d = [sum(1 for r in rows if r[1] == k) for k in (0, 1, 2)]
    last = max(rows, key=lambda r: r[0])
    out = []
    for t in (3.0, 6.0, 12.0, 18.0, 24.0):
        fa, fb = _step_at(af_a, t), _step_at(af_b, t)
        na, nb = 1.0 - _km_at(km_a, t), 1.0 - _km_at(km_b, t)
        s = _km_at(km_all, t)
        out.append((t, sum(1 for r in rows if r[0] >= t), _fx(_true_cif(0, t), 4), _fx(fa, 4), _fx(na, 4), _fx(_true_cif(1, t), 4),
                    _fx(fb, 4), _fx(nb, 4), _fx(fa + fb + s, 4), _fx(na + nb, 4)))
    head = (_compete_spec() + f" Of the 500 subjects {d[1]} have cause a, {d[2]} cause b and {d[0]} are censored. True cumulative "
            f"incidences: F_k(t) = (rate_k / 0.16)(1 - exp(-0.16 t)) with rates 0.06 and 0.10. 'AJ' is the Aalen-Johansen estimate "
            f"(at each distinct time F_k grows by S(t-) d_k / n, then S is multiplied by 1 - d / n with d the events of both causes); "
            f"'naive' is one minus the Kaplan-Meier estimate that treats the other cause as a censoring. Last two columns: "
            f"F_a + F_b + S for the AJ estimates (S the Kaplan-Meier estimate for either cause) and the sum of the two naive "
            f"estimates. The second column is the number of subjects whose time is at least t. The largest observed time is "
            f"{last[0]:.2f} months, {'a failure from cause ' + 'ab'[last[1] - 1] if last[1] else 'a censoring'}.\n\n")
    tab = _table(["t", "at risk", "true F_a", "AJ F_a", "naive F_a", "true F_b", "AJ F_b", "naive F_b", "AJ: F_a + F_b + S", "naive: F_a + F_b"], out)
    return head + tab


def competing_curve_points():
    """Step points of the true, Aalen-Johansen and naive curves of cause a on 0 .. 24, for the figure."""
    af_a, _, km_a, _, _ = _compete_curves()
    grid = [i * 0.25 for i in range(0, 97)]
    return (grid, [_true_cif(0, t) for t in grid], [_step_at(af_a, t) for t in grid], [1.0 - _km_at(km_a, t) for t in grid])


def cause_specific_hazards():
    rows = _compete_cohort()
    exposure = sum(r[0] for r in rows)
    d = [sum(1 for r in rows if r[1] == k) for k in (1, 2)]
    est = [x / exposure for x in d]
    tab = [("cause a", d[0], _fx(exposure, 2), _fx(est[0], 4), "0.0600"), ("cause b", d[1], _fx(exposure, 2), _fx(est[1], 4), "0.1000")]
    lam = sum(est)
    t12 = est[0] / lam * (1.0 - math.exp(-lam * 12.0))
    af_a = _compete_curves()[0]
    head = (_compete_spec() + " For each cause the hazard of a constant-rate model is the number of its events divided by the total "
            "time the subjects were under observation (every subject contributes its observed time, whatever ended it).\n\n")
    t = _table(["cause", "events d_k", "total exposure E", "d_k / E", "true rate"], tab, "lrrrr")
    lam = sum(est)
    t12 = est[0] / lam * (1.0 - math.exp(-lam * 12.0))
    af_a = _compete_curves()[0]
    rate_a, rate_b = COMPETE_RATES
    doubled = rate_a / (rate_a + 2 * rate_b) * (1.0 - math.exp(-(rate_a + 2 * rate_b) * 12.0))
    tail = (f"\n\nFrom the two constant hazards the cumulative incidence of cause a at month 12 is (lambda_a / (lambda_a + lambda_b))"
            f"(1 - exp(-(lambda_a + lambda_b) 12)) = {t12:.4f}; the Aalen-Johansen estimate there is {_step_at(af_a, 12.0):.4f} and the "
            f"truth {_true_cif(0, 12.0):.4f}. If the true rate of cause b doubled to 0.20 and the rate of cause a stayed 0.06, the same "
            f"formula with the true rates gives {doubled:.4f} for the truth at month 12: the hazard of cause a is unchanged and its "
            f"cumulative incidence falls.")
    return head + t + tail


def count_without_exposure():
    nll_rows = []
    for e in (1.0, 0.1, 0.01, 0.001):
        lam_hat = 1.0 / e
        nll_rows.append((e, _fx(lam_hat, 0), _fx(lam_hat * e - math.log(lam_hat), 3)))
    t = _table(["exposure E (one event)", "minimizer d / E", "negative log-likelihood at it"], nll_rows, "lrr")
    return ("The Poisson negative log-likelihood of one event (d = 1) over an exposure E, lambda E - ln lambda, is minimized at "
            "lambda = 1 / E, where it equals 1 + ln E:\n\n" + t)


# ------------------------------------------------------------------ concordance


def concordance_by_hand():
    out, num, den = [], 0.0, 0
    for i, (ti, ei, ri) in enumerate(PAIRS):
        if not ei:
            continue
        for j, (tj, ej, rj) in enumerate(PAIRS):
            if j == i:
                continue
            comparable = tj > ti or (tj == ti and not ej)
            if not comparable:
                continue
            credit = 1.0 if ri > rj else (0.5 if ri == rj else 0.0)
            num += credit
            den += 1
            out.append((i, j, f"{ti} (event)", f"{tj} ({'event' if ej else 'censored'})", ri, rj, credit))
    head = ("Six subjects as (time, event, risk score): "
            + ", ".join(f"{i}: ({t}, {e}, {r})" for i, (t, e, r) in enumerate(PAIRS))
            + ". A pair (i, j) is comparable when subject i had an event and either T_j > T_i or T_j = T_i with j censored. It is "
              "concordant (credit 1) when the risk of i is higher, tied (credit 1/2) when equal, discordant (0) otherwise. Two events at "
              "the same time are not compared.\n\n")
    t = _table(["i", "j", "time of i", "time of j", "risk i", "risk j", "credit"], out, "rrllrrr")
    return head + t + f"\n\nComparable pairs {den}, credit {num}, concordance {num / den:.4f}."


@lru_cache(maxsize=None)
def _scores_cohort():
    """(x1, x2, x3, true lifetime, v) per subject, v the uniform draw that a scenario turns into a censoring time.

    Subject i draws, in order, three normal() values x1, x2, x3 then two uniform() values u, v: the linear predictor is
    0.7 x1 - 0.5 x2, T = -ln(1 - u) / (0.08 exp(lp)). x3 is noise that plays no part in T."""
    g = bookrng.generator(SCORE_NAME)
    rows = []
    for _ in range(SCORE_N):
        x1, x2, x3 = g.normal(), g.normal(), g.normal()
        lp = SCORE_BETA[0] * x1 + SCORE_BETA[1] * x2
        life = _exp_draw(g, SCORE_BASE * math.exp(lp))
        rows.append((x1, x2, x3, life, g.uniform()))
    return tuple(rows)


def _scores_spec():
    return _gen_line(SCORE_NAME, "Subject i = 0 .. 299 in order draws three normal() values x1, x2, x3 (a normal() call that finds a "
                     "held value returns it without drawing), then two uniform() values u and v: lp = 0.7 x1 - 0.5 x2, "
                     "T = -ln(1 - u) / (0.08 exp(lp)) months and the censoring time is C = 3 + 12 v (a second scenario, used once below, "
                     "takes C = 2 + 6 v from the same v). The subject is observed at min(T, C), as an event when "
                     "T <= C. x3 plays no part in T.")


def _observed(scenario=0):
    lo, hi = SCORE_SCENARIOS[scenario][0]
    rows = _scores_cohort()
    cens = [lo + (hi - lo) * r[4] for r in rows]
    t = [min(r[3], c) for r, c in zip(rows, cens)]
    e = [1 if r[3] <= c else 0 for r, c in zip(rows, cens)]
    return t, e


def _censoring_curve(times, events):
    """G(t) = P(C > t) as a Kaplan-Meier step function in which the censorings are the events."""
    return _km_table(times, [0 if e else 1 for e in events])


def _concordance(risk, times, events, full_times=None, tstar=None, weights=None):
    """Sum of credit and of comparable pairs. Case i needs events[i] and (tstar is None or times[i] < tstar);
    comparable j needs times[j] > times[i] (no ties in time in these cohorts); weight(i) multiplies a case's pairs."""
    num = den = 0.0
    n = len(times)
    for i in range(n):
        if not events[i] or (tstar is not None and times[i] >= tstar):
            continue
        w = 1.0 if weights is None else weights(i)
        for j in range(n):
            if times[j] > times[i]:
                credit = 1.0 if risk[i] > risk[j] else (0.5 if risk[i] == risk[j] else 0.0)
                num += w * credit
                den += w
    return num / den


def _models():
    rows = _scores_cohort()
    lp_true = [SCORE_BETA[0] * r[0] + SCORE_BETA[1] * r[1] for r in rows]
    lp_x1 = [SCORE_BETA[0] * r[0] for r in rows]
    return (("both risk factors, true score 0.7 x1 - 0.5 x2", lp_true), ("x1 only, score 0.7 x1", lp_x1),
            ("noise x3", [r[2] for r in rows]), ("no covariates (every subject the same)", [0.0] * len(rows)))


def concordance_cohort():
    rows = _scores_cohort()
    lives = [r[3] for r in rows]
    parts = []
    for k, ((lo, hi), tstar) in enumerate(SCORE_SCENARIOS):
        times, events = _observed(k)
        g = _censoring_curve(times, events)
        out = []
        for name, risk in _models():
            harrell = _concordance(risk, times, events)
            uno = _concordance(risk, times, events, tstar=tstar, weights=lambda i: 1.0 / _km_at(g, times[i]) ** 2)
            oracle = _concordance(risk, lives, [1] * len(rows), tstar=tstar)
            out.append((name, _fx(harrell, 4), _fx(uno, 4), _fx(oracle, 4)))
        n_ev = sum(events)
        parts.append(f"Censoring C = {lo:g} + {hi - lo:g} v, t* = {tstar:g}: {n_ev} events and {len(rows) - n_ev} censored, "
                     f"G(t*) = {_km_at(g, tstar):.4f}.\n\n"
                     + _table(["risk score", "Harrell", f"Uno, t* = {tstar:g}", f"from the true lifetimes, t* = {tstar:g}"], out))
    head = (_scores_spec() + " Harrell's concordance compares every pair (i, j) with an event for i and T_j > T_i (observed times). "
            "Uno's concordance keeps the cases with observed time below t* and weights each case by 1 / G(T_i)^2, with G the "
            "Kaplan-Meier estimate of the censoring distribution (the censorings are the events of that estimate). The last column "
            "applies the definition to the true lifetimes, which nobody observes (every subject is a case when T is below t*, and "
            "the pairs are those with T_j > T_i). Tied risks count one half.\n\n")
    return head + "\n\n".join(parts)


def concordance_replicates():
    g = bookrng.generator(REPLICATES_NAME)
    results = [[] for _ in SCORE_SCENARIOS]
    for _ in range(REPLICATES):
        lps, lives, vs = [], [], []
        for _i in range(REPLICATES_N):
            x1, x2, _x3 = g.normal(), g.normal(), g.normal()
            lp = SCORE_BETA[0] * x1 + SCORE_BETA[1] * x2
            lps.append(lp)
            lives.append(_exp_draw(g, SCORE_BASE * math.exp(lp)))
            vs.append(g.uniform())
        for k, ((lo, hi), tstar) in enumerate(SCORE_SCENARIOS):
            cens = [lo + (hi - lo) * v for v in vs]
            times = [min(a, c) for a, c in zip(lives, cens)]
            events = [1 if a <= c else 0 for a, c in zip(lives, cens)]
            gc = _censoring_curve(times, events)
            harrell = _concordance(lps, times, events)
            uno = _concordance(lps, times, events, tstar=tstar, weights=lambda i: 1.0 / _km_at(gc, times[i]) ** 2)
            oracle = _concordance(lps, lives, [1] * REPLICATES_N, tstar=tstar)
            results[k].append((harrell, uno, oracle))
    rows = []
    for k, ((lo, hi), tstar) in enumerate(SCORE_SCENARIOS):
        res = results[k]
        mean = lambda xs: sum(xs) / len(xs)
        sd = lambda xs: math.sqrt(sum((x - mean(xs)) ** 2 for x in xs) / (len(xs) - 1))
        for idx, name in ((0, "Harrell"), (1, "Uno")):
            diff = [r[idx] - r[2] for r in res]
            rows.append((f"C = {lo:g} + {hi - lo:g} v, t* = {tstar:g}", name, _fx(mean([r[idx] for r in res]), 4), _fx(sd([r[idx] for r in res]), 4),
                         _fx(mean([r[2] for r in res]), 4), _fx(mean(diff), 4), _fx(sd(diff) / math.sqrt(len(diff)), 4)))
    head = (_gen_line(REPLICATES_NAME, f"{REPLICATES} replicates in order; each replicate draws {REPLICATES_N} subjects in order, every subject "
                      "three normal() values x1, x2, x3, then u = uniform() for T = -ln(1 - u) / (0.08 exp(0.7 x1 - 0.5 x2)) and v = uniform() "
                      "for the censoring time of the scenario (the generator is never reset, a held normal() value is carried on); both scenarios of a replicate use "
                      "the same 200 lifetimes and the same v, so C = 3 + 12 v and C = 2 + 6 v censor the same subjects with different times. ")
            + "The risk score is the true linear predictor 0.7 x1 - 0.5 x2. For each replicate: Harrell's and Uno's concordance on the "
              "censored data and the oracle concordance on the true lifetimes (cases with T below t*). Columns: the mean and the "
              "standard deviation over the replicates of the estimate, the mean of the oracle value, and the mean of the difference to "
              "the oracle with its standard error (standard deviation, divisor n - 1, over the square root of 200).\n\n")
    return head + _table(["scenario", "estimator", "mean", "sd over replicates", "mean of oracle", "mean difference to oracle", "SE of that mean"], rows, "llrrrrr")


# ------------------------------------------------------------------ the Brier score under censoring


def _surv_matrix(kind):
    """Predicted survival of every subject on the grid of SCORE_GRID, for a model named kind."""
    rows = _scores_cohort()
    out = []
    if kind == "true":
        lp = [SCORE_BETA[0] * r[0] + SCORE_BETA[1] * r[1] for r in rows]
    elif kind == "x1":
        lp = [SCORE_BETA[0] * r[0] for r in rows]
    else:
        times, events = _observed()
        km = _km_table(times, events)
        return lambda i, t: _km_at(km, t)
    return lambda i, t: math.exp(-SCORE_BASE * math.exp(lp[i]) * t)


def _brier_ipcw(surv, times, events, g, t):
    """Mean over all subjects: an event by t gives S_i(t)^2 / G(T_i); a subject with time above t gives
    (1 - S_i(t))^2 / G(t); a subject censored by t gives 0."""
    total = 0.0
    gt = _km_at(g, t)
    for i, (x, e) in enumerate(zip(times, events)):
        s = surv(i, t)
        if x <= t:
            if e:
                total += s * s / _km_at(g, x)
        else:
            total += (1.0 - s) ** 2 / gt
    return total / len(times)


def _brier_oracle(surv, lives, t):
    return sum(((1.0 if x <= t else 0.0) - (1.0 - surv(i, t))) ** 2 for i, x in enumerate(lives)) / len(lives)


def brier_curves():
    """The IPCW Brier score of the three models on SCORE_GRID, for the figure: [(name, [score per t])]."""
    times, events = _observed()
    g = _censoring_curve(times, events)
    return [(name, [_brier_ipcw(_surv_matrix(kind), times, events, g, t) for t in SCORE_GRID])
            for name, kind in (("both risk factors", "true"), ("x1 only", "x1"), ("Kaplan-Meier, no covariates", "km"))]


def brier_under_censoring():
    rows = _scores_cohort()
    times, events = _observed()
    g = _censoring_curve(times, events)
    lives = [r[3] for r in rows]
    s_true = _surv_matrix("true")
    out = []
    for t in SCORE_HORIZONS:
        keep = sum(1 for x, e in zip(times, events) if not (x <= t and not e))
        # censored subjects counted as survivors, or dropped
        lab = [(1.0 if (x <= t and e) else 0.0, 1.0 - s_true(i, t)) for i, (x, e) in enumerate(zip(times, events))]
        naive_keep = sum((y - f) ** 2 for y, f in lab) / len(lab)
        kept = [(y, f) for (y, f), x, e in zip(lab, times, events) if not (x <= t and not e)]
        naive_drop = sum((y - f) ** 2 for y, f in kept) / len(kept)
        out.append((t, _fx(_brier_oracle(s_true, lives, t), 4), _fx(naive_keep, 4), _fx(naive_drop, 4),
                    _fx(_brier_ipcw(s_true, times, events, g, t), 4), keep))
    t1 = _table(["horizon t", "from the true lifetimes", "censored counted as survivors", "censored by t dropped",
                 "inverse-probability-of-censoring weighted", "subjects kept by dropping"], out)
    out2 = []
    for name, kind in (("both risk factors (true model)", "true"), ("x1 only", "x1"), ("Kaplan-Meier, no covariates", "km")):
        s = _surv_matrix(kind)
        bs = [_brier_ipcw(s, times, events, g, t) for t in SCORE_GRID]
        ibs = sum(0.5 * (bs[k] + bs[k + 1]) * (SCORE_GRID[k + 1] - SCORE_GRID[k]) for k in range(len(bs) - 1)) / (SCORE_GRID[-1] - SCORE_GRID[0])
        ora = [_brier_oracle(s, lives, t) for t in SCORE_GRID]
        ibs_o = sum(0.5 * (ora[k] + ora[k + 1]) * (SCORE_GRID[k + 1] - SCORE_GRID[k]) for k in range(len(ora) - 1)) / (SCORE_GRID[-1] - SCORE_GRID[0])
        cells = [name] + [_fx(_brier_ipcw(s, times, events, g, t), 4) for t in SCORE_HORIZONS] + [_fx(ibs, 4), _fx(ibs_o, 4)]
        out2.append(tuple(cells))
    t2 = _table(["model, inverse-probability weighted"] + [f"t = {t:g}" for t in SCORE_HORIZONS]
                + ["integrated, t = 1 .. 12", "integrated, true lifetimes"], out2, "lrrrrrr")
    head = (_scores_spec() + " The predicted survival of subject i for the true model is exp(-0.08 exp(lp_i) t); for 'x1 only' the same "
            "with lp = 0.7 x1; for the Kaplan-Meier model every subject has the Kaplan-Meier estimate of the observed data. The score at "
            "horizon t is the mean over all 300 subjects of the squared distance between the predicted probability of failing by t and what "
            "happened. With G the Kaplan-Meier estimate of the censoring distribution: a subject that failed by t contributes "
            "S_i(t)^2 / G(T_i), a subject whose observed time exceeds t contributes (1 - S_i(t))^2 / G(t), and a subject censored by t "
            "contributes 0. The integrated score is the trapezoid rule over t = 1, 2, ..., 12 divided by 11. Below the tables, the skill "
            "1 - (score) / (Kaplan-Meier score) at t = 12 is computed from the unrounded scores.\n\n")
    at12 = {kind: _brier_ipcw(_surv_matrix(kind), times, events, g, 12.0) for kind in ("true", "x1", "km")}
    skill = ("\n\nSkill at t = 12 against the Kaplan-Meier model, from the unrounded scores: both risk factors "
             f"{1.0 - at12['true'] / at12['km']:.3f}, x1 only {1.0 - at12['x1'] / at12['km']:.3f}.")
    return head + t1 + "\n\n" + t2 + skill


# ------------------------------------------------------------------ closed-form curves of a hazard table


def _overlap(p, t):
    return max(0.0, min(t, KNOTS[p + 1]) - KNOTS[p])


def _survival_closed(t):
    cum = 0.0
    for p in range(len(KNOTS) - 1):
        cum += sum(h[p] for _, absorbing, h in CODES if absorbing) * _overlap(p, t)
    return math.exp(-cum)


def _cif_closed(code, t):
    _, absorbing, haz = CODES[code]
    free, cif = 1.0, 0.0
    for p in range(len(KNOTS) - 1):
        dt = _overlap(p, t)
        if dt <= 0.0:
            break
        lam = sum(h[p] for _, a, h in CODES if a) + (0.0 if absorbing else haz[p])
        if lam > 0.0:
            leave = 1.0 - math.exp(-lam * dt)
            cif += haz[p] / lam * free * leave
            free *= 1.0 - leave
    return cif


def _cif_numeric(code, t, step=0.0005):
    """Midpoint rule for the integral of lambda_k(s) S_k(s) ds, with S_k the probability of no absorbing event and,
    for a non-absorbing code, no occurrence of the code itself; the step divides every piece exactly."""
    _, absorbing, haz = CODES[code]
    total, s_free = 0.0, 1.0
    for p in range(len(KNOTS) - 1):
        dt = _overlap(p, t)
        if dt <= 0.0:
            break
        lam = sum(h[p] for _, a, h in CODES if a) + (0.0 if absorbing else haz[p])
        m = int(round(dt / step))
        w = dt / m
        for i in range(m):
            s_mid = s_free * math.exp(-lam * (i + 0.5) * w)
            total += haz[p] * s_mid * w
        s_free *= math.exp(-lam * dt)
    return total


def horizon_curves():
    times = (1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 12.0)
    out, worst = [], 0.0
    for t in times:
        s = _survival_closed(t)
        fa, fb, fd = _cif_closed(0, t), _cif_closed(1, t), _cif_closed(2, t)
        worst = max(worst, max(abs(fa - _cif_numeric(0, t)), abs(fb - _cif_numeric(1, t)), abs(fd - _cif_numeric(2, t))))
        out.append((t, _fx(s, 4), _fx(fa, 4), _fx(fb, 4), _fx(s + fa + fb, 4), _fx(fd, 4)))
    hz = _table(["code", "absorbing", "[0, 2)", "[2, 5)", "[5, 10]"], [(n, "yes" if a else "no", *(_fx(x, 2) for x in h)) for n, a, h in CODES], "llrrr")
    head = ("One subject's table of hazards on the pieces [0, 2), [2, 5) and [5, 10] (knots 0, 2, 5, 10), per year:\n\n" + hz +
            "\n\nIn piece p let L_p be the sum of the hazards of the absorbing codes (plus the code's own hazard when the code is not "
            "absorbing) and w_p(t) the time of the piece below t. S(t) = exp(-sum over pieces of (sum of absorbing hazards) w_p(t)); "
            "the cumulative incidence of code k is the sum over pieces of (lambda_kp / L_p) times the probability of no competing event "
            "at the start of the piece times (1 - exp(-L_p w_p(t))); past the last knot the curves keep their value at 10. "
            "'Diagnosis' is the probability of the diagnosis before t and before any death.\n\n")
    tab = _table(["t", "S(t)", "F_a(t)", "F_b(t)", "S + F_a + F_b", "diagnosis before t, alive"], out)
    assert worst < 1e-6, "closed form and midpoint rule disagree"
    return head + tab + ("\n\nAgainst the midpoint rule on the integral of lambda_k S_k with step 0.0005 years, every one of the 21 "
                         "closed-form incidences above agrees to below 1e-6.")


# ------------------------------------------------------------------ the answers to two exercises


def censoring_exercises():
    """Exercise 11 (informative censoring) and exercise 12 (piecewise-exponential fit) on the opening cohort."""
    rows = _cohort()
    times = [r[0] for r in rows]
    events = [r[1] for r in rows]
    t2, e2, moved = list(times), list(events), 0
    for i, (t, e) in enumerate(zip(times, events)):
        if i % 2 == 0 and e and t < 12.0:
            t2[i], e2[i], moved = t / 2.0, 0, moved + 1

    def median(km):
        return next(r[0] for r in km if r[4] <= 0.5)

    km, km2 = _km_table(times, events), _km_table(t2, e2)
    knots = (0.0, 3.0, 6.0, 12.0, 24.0)
    pieces = []
    for a, b in zip(knots, knots[1:]):
        d = sum(1 for t, e in zip(times, events) if e and a < t <= b)
        exp_p = sum(max(0.0, min(t, b) - a) for t in times)
        pieces.append((f"({a:g}, {b:g}]", d, exp_p, d / exp_p))
    s12 = math.exp(-sum(lam * max(0.0, min(12.0, b) - a) for (_, _, _, lam), a, b in zip(pieces, knots, knots[1:])))
    tab = _table(["piece (months)", "failures d_p", "exposure E_p (months)", "d_p / E_p (per month)"],
                 [(p, d, _fx(e, 2), _fx(lam, 4)) for p, d, e, lam in pieces])
    return (f"Exercise 11, on the cohort of `survival_analysis.censoring_bias`: the rule turns {moved} failures into censorings, "
            f"leaving {sum(e2)} failures. Kaplan-Meier median {median(km):.2f} months before and {median(km2):.2f} after; survival at "
            f"month 12 {_km_at(km, 12.0):.4f} before and {_km_at(km2, 12.0):.4f} after.\n\n"
            f"Exercise 12, the same cohort with no features: total exposure {sum(times):.2f} months and {sum(events)} failures.\n\n" + tab +
            f"\n\nImplied S(12) = exp(-(3 lambda_1 + 3 lambda_2 + 6 lambda_3)) = {s12:.4f}, against the Kaplan-Meier "
            f"{_km_at(km, 12.0):.4f} with standard error {_se_at(km, 12.0):.4f}.")
