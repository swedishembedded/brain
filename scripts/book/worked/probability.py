# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Probability: Reasoning Under Uncertainty".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: probability.<name> -->` markers. Helpers with a leading
underscore (or that take arguments) are shared with the chapter's figures, so a
picture and the table beside it cannot disagree. Counting problems are done by
exact enumeration with `fractions.Fraction`; every simulation draws from a named
`bookrng` stream.

Swedish Embedded AB implements solutions for decisions that must hold up under
uncertainty. If your team needs expertise in probabilistic reasoning for machine
learning evaluation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
from fractions import Fraction
from itertools import product

import bookrng


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
    v = float(v) + 0.0  # turns -0.0 into 0.0
    return f"{v:.{d}f}"


def _simpson(f, a, b, n=2000):
    """Area under f on [a, b] (Chapter 9's Simpson rule); n must be even."""
    h = (b - a) / n
    s = f(a) + f(b)
    s += 4 * sum(f(a + (2 * i - 1) * h) for i in range(1, n // 2 + 1))
    s += 2 * sum(f(a + 2 * i * h) for i in range(1, n // 2))
    return s * h / 3


# -------------------------------------------- the spam filter of the opening

# 1000 emails, counted by (what the filter said, what the email really was).
SPAM_COUNTS = {
    ("flagged", "spam"): 180,
    ("not flagged", "spam"): 20,
    ("flagged", "ham"): 40,
    ("not flagged", "ham"): 760,
}
VERDICTS = ("flagged", "not flagged")
KINDS = ("spam", "ham")


def spam_joint_probabilities():
    """The joint distribution of (verdict, kind) as exact fractions of 1000."""
    total = sum(SPAM_COUNTS.values())
    return {k: Fraction(v, total) for k, v in SPAM_COUNTS.items()}


def spam_counts():
    rows = []
    for v in VERDICTS:
        row = [SPAM_COUNTS[(v, k)] for k in KINDS]
        rows.append([v] + row + [sum(row)])
    col = [sum(SPAM_COUNTS[(v, k)] for v in VERDICTS) for k in KINDS]
    rows.append(["total"] + col + [sum(col)])
    return _table(["filter said", "spam", "ham", "total"], rows)


def spam_joint():
    """The counts divided by 1000, with the row and column sums (the marginals)."""
    p = spam_joint_probabilities()
    rows = []
    for v in VERDICTS:
        rows.append([v] + [_fx(p[(v, k)], 3) for k in KINDS] + [_fx(sum(p[(v, k)] for k in KINDS), 3)])
    rows.append(["P(kind)"] + [_fx(sum(p[(v, k)] for v in VERDICTS), 3) for k in KINDS] + ["1.000"])
    return _table(["", "spam", "ham", "P(verdict)"], rows)


def spam_conditionals():
    """Conditional probabilities from the counts, and the product rule checked both ways."""
    c = SPAM_COUNTS
    n = sum(c.values())
    flagged = c[("flagged", "spam")] + c[("flagged", "ham")]
    spam = c[("flagged", "spam")] + c[("not flagged", "spam")]
    nf = n - flagged
    rows = [
        ["P(spam)", f"{spam} / {n}", _fx(Fraction(spam, n), 4)],
        ["P(flagged)", f"{flagged} / {n}", _fx(Fraction(flagged, n), 4)],
        ["P(spam and flagged)", f"{c[('flagged', 'spam')]} / {n}", _fx(Fraction(c[('flagged', 'spam')], n), 4)],
        ["P(spam given flagged)", f"{c[('flagged', 'spam')]} / {flagged}", _fx(Fraction(c[("flagged", "spam")], flagged), 4)],
        ["P(flagged given spam)", f"{c[('flagged', 'spam')]} / {spam}", _fx(Fraction(c[("flagged", "spam")], spam), 4)],
        ["P(spam given not flagged)", f"{c[('not flagged', 'spam')]} / {nf}", _fx(Fraction(c[("not flagged", "spam")], nf), 4)],
        ["P(flagged given ham)", f"{c[('flagged', 'ham')]} / {n - spam}", _fx(Fraction(c[("flagged", "ham")], n - spam), 4)],
    ]
    p_both = Fraction(c[("flagged", "spam")], n)
    via_flagged = Fraction(flagged, n) * Fraction(c[("flagged", "spam")], flagged)
    via_spam = Fraction(spam, n) * Fraction(c[("flagged", "spam")], spam)
    out = _table(["quantity", "count ratio", "value"], rows, "llr")
    out += (f"\n\nProduct rule: P(flagged) P(spam given flagged) = {_fx(via_flagged, 4)}, "
            f"P(spam) P(flagged given spam) = {_fx(via_spam, 4)}, both equal P(spam and flagged) = {_fx(p_both, 4)}.")
    return out


def spam_condition_on_verdict():
    """Condition on the verdict: keep one row of the table and divide by its sum."""
    c = SPAM_COUNTS
    rows = []
    for v in VERDICTS:
        row_sum = c[(v, "spam")] + c[(v, "ham")]
        rows.append([v, f"{c[(v, 'spam')]} / {row_sum}", f"{c[(v, 'ham')]} / {row_sum}",
                     _fx(Fraction(c[(v, 'spam')], row_sum), 4), _fx(Fraction(c[(v, 'ham')], row_sum), 4),
                     _fx(Fraction(c[(v, 'spam')] + c[(v, 'ham')], row_sum), 1)])
    return _table(["given", "spam count", "ham count", "P(spam given row)", "P(ham given row)", "sum"], rows, "lrrrrr")


def spam_cost_expectation():
    """Accuracy is the expectation of an indicator; a cost-weighted loss is another expectation."""
    p = spam_joint_probabilities()
    correct = {("flagged", "spam"): 1, ("not flagged", "ham"): 1, ("flagged", "ham"): 0, ("not flagged", "spam"): 0}
    cost = {("flagged", "spam"): 0, ("not flagged", "ham"): 0, ("flagged", "ham"): 10, ("not flagged", "spam"): 1}
    rows = []
    for k in [("flagged", "spam"), ("not flagged", "spam"), ("flagged", "ham"), ("not flagged", "ham")]:
        rows.append([f"{k[0]}, {k[1]}", _fx(p[k], 3), correct[k], cost[k], _fx(correct[k] * p[k], 3), _fx(cost[k] * p[k], 3)])
    e_correct = sum(correct[k] * p[k] for k in p)
    e_cost = sum(cost[k] * p[k] for k in p)
    rows.append(["expectation", "1.000", "", "", _fx(e_correct, 3), _fx(e_cost, 3)])
    return _table(["outcome", "p", "correct", "cost", "correct x p", "cost x p"], rows, "lrrrrr")


# 1000 images, counted by (true class, predicted class) for a three-class classifier.
CLASSES = ("cat", "dog", "bird")
CLASS_COUNTS = {
    ("cat", "cat"): 270, ("cat", "dog"): 40, ("cat", "bird"): 20,
    ("dog", "cat"): 30, ("dog", "dog"): 300, ("dog", "bird"): 10,
    ("bird", "cat"): 15, ("bird", "dog"): 15, ("bird", "bird"): 300,
}


def classifier_joint_matrix():
    """Rows: true class; columns: predicted class; entries: joint probabilities (exact fractions)."""
    total = sum(CLASS_COUNTS.values())
    return [[Fraction(CLASS_COUNTS[(t, q)], total) for q in CLASSES] for t in CLASSES]


def classifier_joint():
    """The joint distribution of (true class, predicted class) with both marginals."""
    m = classifier_joint_matrix()
    rows = [[t] + [_fx(v, 3) for v in m[i]] + [_fx(sum(m[i]), 3)] for i, t in enumerate(CLASSES)]
    rows.append(["P(predicted)"] + [_fx(sum(m[i][j] for i in range(3)), 3) for j in range(3)] + [_fx(sum(map(sum, m)), 3)])
    out = _table(["true class"] + list(CLASSES) + ["P(true)"], rows)
    acc = sum(m[i][i] for i in range(3))
    recall = ", ".join(f"{t} {_fx(m[i][i] / sum(m[i]), 3)}" for i, t in enumerate(CLASSES))
    out += f"\n\nDiagonal (correct) mass: {_fx(acc, 3)}. P(predicted = true given true): {recall}."
    return out


# --------------------------------------------------- frequencies and counting

def _generator_line(name):
    """The sentence that makes a simulated table reproducible: generator, seed, and the rule for what it draws."""
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} ({bookrng.seed_of(name)})."

def coin_frequencies():
    """Relative frequency of heads in a growing run of fair-coin flips."""
    name = "probability.coin_frequencies"
    rng = bookrng.generator(name)
    checkpoints = [10, 100, 1000, 10000, 100000]
    heads, flips, rows = 0, 0, []
    for stop in checkpoints:
        while flips < stop:
            heads += rng.uniform() < 0.5
            flips += 1
        rows.append([f"{flips:,}", heads, _fx(heads / flips, 4), _fx(abs(heads / flips - 0.5), 4)])
    return (f"{_generator_line(name)} Flip k is the k-th call of `rng.uniform()` and counts as heads when it is "
            "below 0.5; one run of 100,000 flips, read off at the flip counts listed.\n\n"
            + _table(["flips", "heads", "heads / flips", "distance from 0.5"], rows))


def two_dice_sums():
    """The 36 outcomes of two dice grouped by their sum."""
    counts = {s: 0 for s in range(2, 13)}
    for a, b in product(range(1, 7), repeat=2):
        counts[a + b] += 1
    rows = [[s, c, f"{c}/36", _fx(Fraction(c, 36), 4), "#" * c] for s, c in counts.items()]
    rows.append(["total", sum(counts.values()), "36/36", _fx(Fraction(sum(counts.values()), 36), 4), ""])
    return _table(["sum", "outcomes", "fraction", "probability", "tally"], rows, "lrrrl")


def dice_events():
    """Events are sets of outcomes; their probabilities come from counting the sets."""
    omega = set(product(range(1, 7), repeat=2))
    a = {o for o in omega if o[0] + o[1] == 7}
    b = {o for o in omega if o[0] % 2 == 0}
    c = {o for o in omega if o[0] + o[1] >= 10}
    n = len(omega)
    rows = [
        ["A: sum is 7", len(a), _fx(Fraction(len(a), n), 4)],
        ["B: first die is even", len(b), _fx(Fraction(len(b), n), 4)],
        ["C: sum is at least 10", len(c), _fx(Fraction(len(c), n), 4)],
        ["A and B (intersection)", len(a & b), _fx(Fraction(len(a & b), n), 4)],
        ["A or B (union)", len(a | b), _fx(Fraction(len(a | b), n), 4)],
        ["P(A) + P(B) - P(A and B)", "", _fx(Fraction(len(a) + len(b) - len(a & b), n), 4)],
        ["not C (complement)", len(omega - c), _fx(Fraction(len(omega - c), n), 4)],
        ["1 - P(C)", "", _fx(1 - Fraction(len(c), n), 4)],
    ]
    return _table(["event", "outcomes", "probability"], rows)


def three_coins():
    """A random variable is a function: X maps each of 8 outcomes to its number of heads."""
    outcomes = list(product("TH", repeat=3))
    rows = [["".join(o), o.count("H")] for o in outcomes]
    table = _table(["outcome", "X = heads"], rows, "lr")
    counts = {}
    for o in outcomes:
        counts.setdefault(o.count("H"), []).append("".join(o))
    prow = [[x, ", ".join(v), len(v), f"{len(v)}/8", _fx(Fraction(len(v), 8), 3)] for x, v in sorted(counts.items())]
    dist = _table(["x", "outcomes with X = x", "count", "fraction", "P(X = x)"], prow, "llrrr")
    return table + "\n\n" + dist


# ----------------------------------------------------------------- densities

def _density(x):
    """A density on [0, 1]: f(x) = 6 x (1 - x), zero elsewhere."""
    return 6.0 * x * (1.0 - x) if 0.0 <= x <= 1.0 else 0.0


def _cdf(x):
    """Exact area of the density up to x, 3 x^2 - 2 x^3, used to check the numerical areas."""
    x = min(max(x, 0.0), 1.0)
    return 3 * x * x - 2 * x ** 3


def density_probability(lo=0.2, hi=0.5):
    """P(lo <= X <= hi) for the density of the chapter, by Simpson's rule."""
    return _simpson(_density, lo, hi)


def density_area():
    rows = []
    for lo, hi in [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (0.0, 1.0), (0.3, 0.3)]:
        area = 0.0 if lo == hi else _simpson(_density, lo, hi)
        rows.append([f"{lo:g} to {hi:g}", _fx(area, 6), _fx(_cdf(hi) - _cdf(lo), 6)])
    out = _table(["interval", "area under f (Simpson)", "exact 3 x^2 - 2 x^3"], rows)
    out += f"\n\nLargest value of the density, f(0.5) = {_density(0.5):g}."
    # Probability of a thin slice next to the density value: area is about f times width.
    thin = []
    for w in [0.1, 0.01, 0.001]:
        a = _simpson(_density, 0.5, 0.5 + w)
        thin.append([f"{w:g}", _fx(a, 6), _fx(_density(0.5) * w, 6), _fx(a / w, 4)])
    out += "\n\n" + _table(["slice [0.5, 0.5 + w], w", "P(slice)", "f(0.5) x w", "P(slice) / w"], thin)
    return out


def density_exceeds_one():
    """A uniform density on [0, w] is 1/w: narrower intervals force taller densities, area stays 1."""
    rows = []
    for w in [1.0, 0.5, 0.1, 0.01]:
        h = 1.0 / w
        area = _simpson(lambda x: h if 0 <= x <= w else 0.0, 0.0, w)
        rows.append([f"{w:g}", _fx(h, 1), _fx(area, 4)])
    return _table(["interval length w", "density height 1/w", "area"], rows)


def density_moments():
    """Expectation and variance of the density f(x) = 6 x (1 - x), as integrals."""
    e = _simpson(lambda x: x * _density(x), 0.0, 1.0)
    e2 = _simpson(lambda x: x * x * _density(x), 0.0, 1.0)
    var_direct = _simpson(lambda x: (x - e) ** 2 * _density(x), 0.0, 1.0)
    rows = [
        ["E[X] = integral of x f(x)", _fx(e, 6)],
        ["E[X^2] = integral of x^2 f(x)", _fx(e2, 6)],
        ["E[X^2] - E[X]^2", _fx(e2 - e * e, 6)],
        ["integral of (x - E[X])^2 f(x)", _fx(var_direct, 6)],
    ]
    return _table(["quantity", "value"], rows, "lr")


# ------------------------------------------------------------- independence

def independence_checks():
    """Test P(A and B) = P(A) P(B) on events of two fair dice."""
    omega = set(product(range(1, 7), repeat=2))
    n = len(omega)

    def prob(ev):
        return Fraction(len(ev), n)

    def ev(pred):
        return {o for o in omega if pred(o)}

    pairs = [
        ("first die is 6", "second die is 6", ev(lambda o: o[0] == 6), ev(lambda o: o[1] == 6)),
        ("first die is 3", "sum is 7", ev(lambda o: o[0] == 3), ev(lambda o: sum(o) == 7)),
        ("first die is 3", "sum is 8", ev(lambda o: o[0] == 3), ev(lambda o: sum(o) == 8)),
        ("first die is 1", "first die is 2", ev(lambda o: o[0] == 1), ev(lambda o: o[0] == 2)),
        ("first die even", "second die even", ev(lambda o: o[0] % 2 == 0), ev(lambda o: o[1] % 2 == 0)),
        ("first die even", "sum even", ev(lambda o: o[0] % 2 == 0), ev(lambda o: sum(o) % 2 == 0)),
    ]
    rows = []
    for na, nb, a, b in pairs:
        both, prod_ = prob(a & b), prob(a) * prob(b)
        rows.append([f"{na} / {nb}", _fx(prob(a), 4), _fx(prob(b), 4), _fx(both, 4), _fx(prod_, 4),
                     "independent" if both == prod_ else "dependent",
                     "yes" if not (a & b) else "no"])
    return _table(["events A / B", "P(A)", "P(B)", "P(A and B)", "P(A) P(B)", "verdict", "mutually exclusive"],
                  rows, "lrrrrll")


def spam_independence():
    """Are the filter verdict and the true kind independent? Compare the joint with the product of marginals."""
    p = spam_joint_probabilities()
    rows = []
    for v in VERDICTS:
        for k in KINDS:
            pv = sum(p[(v, kk)] for kk in KINDS)
            pk = sum(p[(vv, k)] for vv in VERDICTS)
            rows.append([f"{v}, {k}", _fx(p[(v, k)], 3), _fx(pv * pk, 4), _fx(p[(v, k)] / (pv * pk), 2)])
    return _table(["cell", "joint p", "product of marginals", "ratio"], rows)


# --------------------------------------------------------------- Bayes' rule

def _posterior(prior, sensitivity, false_alarm):
    """P(condition | positive) by Bayes' rule."""
    evidence = sensitivity * prior + false_alarm * (1 - prior)
    return sensitivity * prior / evidence


def bayes_rare_defect():
    """100,000 parts, 1 percent defective, a detector that is 90 percent sensitive and has a 10 percent false-alarm rate."""
    n, prior, sens, fa = 100_000, 0.01, 0.9, 0.1
    bad = round(n * prior)
    good = n - bad
    tp, fn = round(bad * sens), bad - round(bad * sens)
    fp = round(good * fa)
    tn = good - fp
    rows = [
        ["defective", bad, tp, fn],
        ["good", good, fp, tn],
        ["total", n, tp + fp, fn + tn],
    ]
    out = _table(["", "parts", "alarm", "no alarm"], rows)
    out += (f"\n\nOf the {tp + fp} alarms, {tp} are real: P(defective given alarm) = {tp}/{tp + fp} = "
            f"{_fx(tp / (tp + fp), 4)}. Of the {fn + tn} silences, {tn} are good: "
            f"P(good given no alarm) = {_fx(tn / (fn + tn), 4)}.")
    out += f"\nBayes' rule directly: {_fx(_posterior(prior, sens, fa), 4)}."
    return out


def bayes_base_rate_sweep():
    """The same detector (sensitivity 0.9, false-alarm rate 0.1) at different base rates."""
    rows = []
    for prior in [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]:
        post = _posterior(prior, 0.9, 0.1)
        rows.append([f"{prior:g}", _fx(0.9 * prior + 0.1 * (1 - prior), 4), _fx(post, 4),
                     f"{(1 - post) / post:.1f}"])
    return _table(["base rate P(defective)", "P(alarm)", "P(defective given alarm)", "false alarms per real one"], rows)


def bayes_two_detectors():
    """Two detectors that err independently given the truth: each positive turns the posterior into the next prior."""
    p = 0.01
    rows = [["before any alarm", _fx(p, 4)]]
    for i in (1, 2, 3):
        p = _posterior(p, 0.9, 0.1)
        rows.append([f"after alarm {i} (detector {i})", _fx(p, 4)])
    return ("Prior 0.01; each detector has sensitivity 0.9 and false-alarm rate 0.1, and they err independently given the truth.\n\n"
            + _table(["state", "P(defective)"], rows, "lr"))


def bayes_spam_two_ways():
    """P(spam given flagged) from counting and from Bayes' rule with the likelihood and the base rate."""
    p = spam_joint_probabilities()
    p_spam = p[("flagged", "spam")] + p[("not flagged", "spam")]
    p_flag_given_spam = p[("flagged", "spam")] / p_spam
    p_flag_given_ham = p[("flagged", "ham")] / (1 - p_spam)
    evidence = p_flag_given_spam * p_spam + p_flag_given_ham * (1 - p_spam)
    rows = [
        ["P(spam)", _fx(p_spam, 4)],
        ["P(flagged given spam)", _fx(p_flag_given_spam, 4)],
        ["P(flagged given ham)", _fx(p_flag_given_ham, 4)],
        ["P(flagged) = sum over kinds", _fx(evidence, 4)],
        ["Bayes: 0.9 x 0.2 / P(flagged)", _fx(p_flag_given_spam * p_spam / evidence, 4)],
    ]
    return _table(["quantity", "value"], rows, "lr")


# --------------------------------------------------------------- expectation

def _expect(pmf, g=lambda x: x):
    return sum(Fraction(p) * g(x) for x, p in pmf.items())


def _die():
    return {x: Fraction(1, 6) for x in range(1, 7)}


def _sum_pmf():
    pmf = {}
    for a, b in product(range(1, 7), repeat=2):
        pmf[a + b] = pmf.get(a + b, 0) + Fraction(1, 36)
    return pmf


def expectation_by_table():
    """E[X] as the sum of value times probability, for one die and for the sum of two."""
    rows = [[x, f"{p}", _fx(x * p, 4)] for x, p in _die().items()]
    rows.append(["E[one die]", "", _fx(_expect(_die()), 4)])
    one = _table(["x", "p(x)", "x p(x)"], rows)
    sum_rows = [[x, f"{p}", _fx(x * p, 4)] for x, p in sorted(_sum_pmf().items())]
    sum_rows.append(["E[sum]", "", _fx(_expect(_sum_pmf()), 4)])
    two = _table(["s", "p(s)", "s p(s)"], sum_rows)
    return one + "\n\n" + two


def running_average(n, name):
    """Running average of n rolls of the two-dice sum from the generator `name`; returns the list of averages.

    A roll is `rng.integer(6) + 1` for the first die, then the same for the second.
    """
    rng = bookrng.generator(name)
    total, out = 0, []
    for i in range(1, n + 1):
        total += rng.integer(6) + 1 + rng.integer(6) + 1
        out.append(total / i)
    return out


def long_run_average():
    """The sample average of the two-dice sum approaches the expectation 7 as rolls accumulate."""
    name = "probability.long_run_average"
    avg = running_average(100_000, name)
    rows = []
    for n in [10, 100, 1000, 10_000, 100_000]:
        rows.append([f"{n:,}", _fx(avg[n - 1], 4), _fx(abs(avg[n - 1] - 7), 4)])
    return (f"{_generator_line(name)} Roll k uses two calls of `rng.integer(6)`, each plus 1: first the first die, then "
            "the second; the sum of the two is the roll. One run of 100,000 rolls, read off at the roll counts listed.\n\n"
            + _table(["rolls", "average of the sum", "distance from 7"], rows))


def dataset_mean_is_expectation():
    """The mean of N per-example losses is the expectation under the distribution that gives each example weight 1/N."""
    losses = [0, 1, 1, 4, 9]
    mean = Fraction(sum(losses), len(losses))
    pmf = {}
    for v in losses:
        pmf[v] = pmf.get(v, 0) + Fraction(1, len(losses))
    rows = [[v, f"{p}", _fx(v * p, 2)] for v, p in sorted(pmf.items())]
    rows.append(["E[loss]", "", _fx(_expect(pmf), 2)])
    return (f"Losses of five examples: {losses}; their mean is {_fx(mean, 2)}.\n\n"
            + _table(["loss value", "probability", "value x probability"], rows))


def linearity_check():
    """E[X + Y] = E[X] + E[Y] by enumeration of the 36 outcomes, with independent and dependent pairs."""
    omega = list(product(range(1, 7), repeat=2))
    w = Fraction(1, 36)

    def e(g):
        return sum(w * g(o) for o in omega)

    x = lambda o: o[0]
    y = lambda o: o[1]
    s = lambda o: o[0] + o[1]
    rows = [
        ["X = first die, Y = second die", "independent",
         _fx(e(x), 4), _fx(e(y), 4), _fx(e(lambda o: x(o) + y(o)), 4), _fx(e(lambda o: x(o) * y(o)), 4), _fx(e(x) * e(y), 4)],
        ["X = first die, Y = sum of both", "dependent",
         _fx(e(x), 4), _fx(e(s), 4), _fx(e(lambda o: x(o) + s(o)), 4), _fx(e(lambda o: x(o) * s(o)), 4), _fx(e(x) * e(s), 4)],
        ["X = first die, Y = 7 - first die", "dependent",
         _fx(e(x), 4), _fx(e(lambda o: 7 - x(o)), 4), _fx(e(lambda o: x(o) + 7 - x(o)), 4),
         _fx(e(lambda o: x(o) * (7 - x(o))), 4), _fx(e(x) * e(lambda o: 7 - x(o)), 4)],
    ]
    return _table(["pair", "relation", "E[X]", "E[Y]", "E[X + Y]", "E[X Y]", "E[X] E[Y]"], rows, "llrrrrr")


# ------------------------------------------------------------------ variance

def variance_die():
    """Var[X] by its definition and by the shortcut E[X^2] - E[X]^2, for one die."""
    pmf = _die()
    mean = _expect(pmf)
    rows = [[x, _fx(x - mean, 2), _fx((x - mean) ** 2, 4), _fx(Fraction(1, 6) * (x - mean) ** 2, 4)] for x in pmf]
    var_def = _expect(pmf, lambda x: (x - mean) ** 2)
    e2 = _expect(pmf, lambda x: x * x)
    rows.append(["Var by definition", "", "", _fx(var_def, 4)])
    out = _table(["x", "x - E[X]", "(x - E[X])^2", "p x (x - E[X])^2"], rows)
    out += (f"\n\nE[X^2] = {e2} = {_fx(e2, 4)}, E[X]^2 = {_fx(mean ** 2, 4)}, "
            f"E[X^2] - E[X]^2 = {_fx(e2 - mean ** 2, 4)} = {e2 - mean ** 2}. "
            f"Standard deviation sqrt(Var) = {_fx(float(var_def) ** 0.5, 4)}.")
    return out


def variance_scaling():
    """Adding a constant leaves the variance alone; multiplying by a scales it by a squared."""
    pmf = _die()

    def var(g):
        m = _expect(pmf, g)
        return _expect(pmf, lambda x: (g(x) - m) ** 2)

    rows = [["X", _fx(var(lambda x: x), 4)], ["X + 3", _fx(var(lambda x: x + 3), 4)],
            ["2 X", _fx(var(lambda x: 2 * x), 4)], ["2 X + 3", _fx(var(lambda x: 2 * x + 3), 4)],
            ["-X", _fx(var(lambda x: -x), 4)]]
    return _table(["variable", "variance"], rows, "lr")


def variance_of_sums():
    """Var[X + Y] against Var[X] + Var[Y] for independent, equal and opposite pairs of dice."""
    omega = list(product(range(1, 7), repeat=2))
    w = Fraction(1, 36)

    def e(g):
        return sum(w * g(o) for o in omega)

    def var(g):
        m = e(g)
        return e(lambda o: g(o) ** 2) - m * m

    cases = [
        ("independent: X = first, Y = second", lambda o: o[0], lambda o: o[1]),
        ("equal: X = first, Y = first", lambda o: o[0], lambda o: o[0]),
        ("opposite: X = first, Y = 7 - first", lambda o: o[0], lambda o: 7 - o[0]),
        ("overlapping: X = first, Y = sum of both", lambda o: o[0], lambda o: o[0] + o[1]),
    ]
    rows = []
    for name, f, g in cases:
        v_sum = var(lambda o: f(o) + g(o))
        gap = v_sum - var(f) - var(g)
        cross = 2 * (e(lambda o: f(o) * g(o)) - e(f) * e(g))
        rows.append([name, _fx(var(f), 4), _fx(var(g), 4), _fx(var(f) + var(g), 4), _fx(v_sum, 4), _fx(gap, 4), _fx(cross, 4)])
    return _table(["pair", "Var[X]", "Var[Y]", "sum of the two", "Var[X + Y]", "gap", "2(E[XY] - E[X]E[Y])"],
                  rows, "lrrrrrr")


def mean_variance_shrinks():
    """The variance of the mean of n die rolls is Var[X] / n: simulate 4000 means for each n."""
    name = "probability.mean_variance_shrinks"
    rng = bookrng.generator(name)
    var_x = 35 / 12
    rows = []
    for n in [1, 4, 16, 64, 256]:
        means = [sum(rng.integer(6) + 1 for _ in range(n)) / n for _ in range(4000)]
        m = sum(means) / len(means)
        v = sum((a - m) ** 2 for a in means) / len(means)
        rows.append([n, _fx(m, 4), _fx(v, 4), _fx(var_x / n, 4), _fx(v ** 0.5, 4)])
    return (f"{_generator_line(name)} One generator for the whole table. A roll is `rng.integer(6) + 1`. Rows in the order "
            "n = 1, 4, 16, 64, 256; within a row 4000 means in turn, each the average of its n rolls drawn in turn. "
            "Variance of the means: average squared deviation from their average (divisor 4000).\n\n"
            + _table(["rolls per mean n", "average of the means", "variance of the means", "Var[X] / n", "std of the means"], rows))


def sampling_a_table():
    """Drawing from a probability table with one uniform number (cumulative sums), then counting the draws."""
    probs = [("a", Fraction(1, 2)), ("b", Fraction(3, 10)), ("c", Fraction(1, 5))]
    name = "probability.sampling_a_table"
    rng = bookrng.generator(name)
    counts = {k: 0 for k, _ in probs}
    n = 20_000
    for _ in range(n):
        u = rng.uniform()
        acc = 0.0
        for k, p in probs:
            acc += float(p)
            if u < acc:
                counts[k] += 1
                break
        else:
            counts[probs[-1][0]] += 1
    rows = []
    cum = Fraction(0)
    for k, p in probs:
        cum += p
        rows.append([k, _fx(p, 2), _fx(cum, 2), counts[k], _fx(counts[k] / n, 4)])
    return (f"{_generator_line(name)} Each of the {n:,} draws is one `rng.uniform()` passed to the `sample_discrete` "
            "rule (first symbol whose cumulative probability exceeds u).\n\n"
            + _table(["symbol", "p", "cumulative p", f"draws of {n:,}", "frequency"], rows))
