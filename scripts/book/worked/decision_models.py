# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Decision Models in Practice".

Plain Python 3, standard library only, written from the chapter's own description. One small world carries the chapter: a payments
desk that must route each incoming request to one of four teams, or hand it to a person, in a fixed amount of arithmetic.

* A request has five words and four numbers. The words become a sentence vector by the rule of Volume V, Chapter 14 (the mean of
  the word vectors, scaled to length 1), where the word vectors are a frozen stand-in for a trained encoder: synonyms share a
  concept vector plus noise, so two paraphrases lie close together. The numbers are scaled by the mean and the standard deviation of
  the training requests. The state vector is the ten numbers, text first.
* The decision head is a softmax over four routes, `z = W x + b`, trained by full-batch gradient descent on the mean cross-entropy.
  Three heads are trained (text only, numbers only, both) to show what each half of the state contributes.
* The head's probabilities are calibrated by one temperature fitted on a validation split, as in Volume II, Chapter 6, and the
  decision rule on top of them is the one of Volume XI, Chapter 6: act when the expected cost of a wrong route is at most the cost
  of handing the request to a person.
* The rest is what a deployed model needs: an offline estimate of a new policy from the log of an old one, the audit record of one
  decision and its replay, the arithmetic of one decision against a large model's, and the small computations behind the three
  details of brain's `decide` crate that the chapter discusses (a yes/no question that cannot be a softmax over two options, a
  confidence that does not depend on the number of options, and a reward that has to be a proper scoring rule).

Every number learned from data goes through `stable` (the rule of `value_learning`): the training is repeated with the first
initial weight of the head changed by each nudge of `stable.py`, and a digit is printed only if the run and its four copies share it.
Each public function without a leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: decision_models.<name> -->` markers. Underscore functions and the constants are shared with the chapter's figures, so a
picture and the table beside it cannot disagree.

Swedish Embedded AB implements solutions for decisions that must be fast, priced and explainable. If your team needs expertise in
small decision models inside applications, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import struct
from functools import lru_cache

import bookrng
import calibrated_decisions as cd
import stable
from time_series import _table, HELD_GAUSSIAN

# ------------------------------------------------------------------ the world

WORLD = "decision_models.world"
INIT = "decision_models.init"
LOGGING = "decision_models.logging"
M, K = 6, 4                                   # sentence-vector length, number of routes
ROUTES = ("card", "payment", "account", "dispute")
FAMILIES = (("card", "lost", "stolen", "replacement"),
            ("payment", "declined", "failed", "charge"),
            ("account", "address", "password", "login"),
            ("refund", "dispute", "chargeback", "unauthorised"))
FILLER = ("please", "help", "my", "today")
PARTNER = (2, 3, 0, 1)                        # the family a request of route k is most often confused with
PRIOR = (0.35, 0.30, 0.20, 0.15)
OWN, ALSO = 0.65, 0.85                        # slot rule: u < 0.65 own family, u < 0.85 partner family, otherwise filler
WORD_NOISE = 0.5
LOG_AMOUNT = (3.0, 3.6, 2.6, 4.4)             # mean of ln(amount) by route; the standard deviation is 0.8
P_DEVICE = (0.1, 0.2, 0.15, 0.6)
FEATURES = ("ln amount", "account age (months)", "new device", "contacts in 30 days")
SPLITS = (("train", 120), ("validation", 500), ("test", 1000), ("log", 2000))
STEPS, RATE = 1500, 0.5
HEADS = (("text", M), ("numbers", 4), ("both", M + 4))
FIRST_WEIGHT_SCALE = 0.1
C_ESCALATE = 1.0
C_WRONG = (2.0, 4.0, 10.0)
N_BINS = cd.N_BINS


def _fmt(x, d=4):
    return cd._fmt(x, d)


def _seed_line(name, normal=False):
    seed = bookrng.seed_of(name)
    return f"`{name}` (seed {seed:#018x}, {seed})" + ("." + HELD_GAUSSIAN if normal else "")


@lru_cache(maxsize=None)
def _world():
    """(concept vectors, word vectors, the four splits). One generator for everything, in the order the chapter states."""
    g = bookrng.generator(WORLD)
    concept = [[g.normal() for _ in range(M)] for _ in range(K)]
    words = {}
    for k in range(K):
        for w in FAMILIES[k]:
            words[w] = [concept[k][j] + WORD_NOISE * g.normal() for j in range(M)]
    for w in FILLER:
        words[w] = [WORD_NOISE * g.normal() for _ in range(M)]
    splits = {}
    for name, n in SPLITS:
        splits[name] = [_draw(g, words) for _ in range(n)]
    return concept, words, splits


def _draw(g, words):
    u = g.uniform()
    y, acc = K - 1, 0.0
    for k in range(K):
        acc += PRIOR[k]
        if u < acc:
            y = k
            break
    text = []
    for _ in range(3):
        v = g.uniform()
        fam = FAMILIES[y] if v < OWN else (FAMILIES[PARTNER[y]] if v < ALSO else FILLER)
        text.append(fam[g.integer(4)])
    for _ in range(2):
        text.append(FILLER[g.integer(4)])
    amount = LOG_AMOUNT[y] + 0.8 * g.normal()
    age = 1 + g.integer(60)
    device = 1 if g.uniform() < P_DEVICE[y] else 0
    contacts = g.integer(3)
    if y == 3:
        contacts += g.integer(3)
    return {"y": y, "words": text, "raw": [amount, float(age), float(device), float(contacts)], "u": _sentence(words, text)}


def _sentence(words, text):
    """The mean of the word vectors, left to right, scaled to length 1."""
    s = [0.0] * M
    for w in text:
        for j in range(M):
            s[j] += words[w][j]
    s = [v / len(text) for v in s]
    n = math.sqrt(sum(v * v for v in s))
    return [v / n for v in s]


def _scaling():
    """Mean and standard deviation (divisor n) of each numeric field over the training requests."""
    rows = _world()[2]["train"]
    n = len(rows)
    mean = [sum(r["raw"][j] for r in rows) / n for j in range(4)]
    sd = [math.sqrt(sum((r["raw"][j] - mean[j]) ** 2 for r in rows) / n) for j in range(4)]
    return mean, sd


def _state(row, head="both"):
    mean, sd = _scaling()
    num = [(row["raw"][j] - mean[j]) / sd[j] for j in range(4)]
    return {"text": row["u"], "numbers": num, "both": row["u"] + num}[head]


def _argmax(p):
    best = 0
    for i, v in enumerate(p):
        if v > p[best]:
            best = i
    return best


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _logits(W, b, x):
    out = []
    for k in range(K):
        s = 0.0
        for j in range(len(x)):
            s += W[k][j] * x[j]
        out.append(s + b[k])
    return out


# ------------------------------------------------------------------ training, repeated under the nudges

def _init(head):
    g = bookrng.generator(f"{INIT}.{head}")
    d = dict(HEADS)[head]
    return [[FIRST_WEIGHT_SCALE * g.normal() for _ in range(d)] for _ in range(K)]


def _train(head, nudge):
    rows = _world()[2]["train"]
    xs = [_state(r, head) for r in rows]
    ys = [r["y"] for r in rows]
    d = len(xs[0])
    W = _init(head)
    W[0][0] += nudge
    b = [0.0] * K
    n = len(rows)
    losses = {}
    for step in range(STEPS + 1):
        gW = [[0.0] * d for _ in range(K)]
        gb = [0.0] * K
        loss = 0.0
        for x, y in zip(xs, ys):
            p = _softmax(_logits(W, b, x))
            loss -= math.log(p[y])
            for k in range(K):
                e = p[k] - (1.0 if k == y else 0.0)
                gb[k] += e
                for j in range(d):
                    gW[k][j] += e * x[j]
        if step in (0, 10, 100, 500, STEPS):
            losses[step] = loss / n
        if step == STEPS:
            break
        for k in range(K):
            b[k] -= RATE * gb[k] / n
            for j in range(d):
                W[k][j] -= RATE * gW[k][j] / n
    return {"W": W, "b": b, "loss": losses}


def _fit_temperature(z_rows, ys):
    """The scale a = 1/T minimising the mean cross-entropy of softmax(a z): double an upper end (at most 60 times) until the slope is
    positive, then halve the bracket 80 times (Volume II, Chapter 6, on the multiclass logit vector)."""
    def slope(a):
        s = 0.0
        for z, y in zip(z_rows, ys):
            p = _softmax([a * v for v in z])
            s += sum(p[k] * z[k] for k in range(K)) - z[y]
        return s / len(ys)
    if slope(0.0) >= 0.0:
        raise ValueError("the logits do not favour the right routes on average")
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
    return 1.0 / ((lo + hi) / 2.0)


def _scored(run, head, split, temperature):
    rows = _world()[2][split]
    out = []
    for r in rows:
        z = _logits(run["W"], run["b"], _state(r, head))
        out.append(_softmax([v / temperature for v in z]))
    return out


def _measure(run, head):
    """Everything a table needs from one trained head: the temperature, and the scores of every split."""
    w = _world()[2]
    yv = [r["y"] for r in w["validation"]]
    zv = [_logits(run["W"], run["b"], _state(r, head)) for r in w["validation"]]
    t = _fit_temperature(zv, yv)
    return {"T": t, "raw": {s: _scored(run, head, s, 1.0) for s in ("train", "validation", "test")},
            "cal": {s: _scored(run, head, s, t) for s in ("validation", "test")}}


@lru_cache(maxsize=None)
def _runs(head):
    """The run and its four nudged copies, each with its measurements."""
    out = []
    for nudge in (0.0,) + stable.NUDGES:
        run = _train(head, nudge)
        run.update(_measure(run, head))
        out.append(run)
    return out


def _st(head, getter, full):
    runs = _runs(head)
    return stable.text(getter(runs[0]), [getter(r) for r in runs[1:]], full)


def _f4(x):
    return f"{x:.4f}"


def _f3(x):
    return f"{x:.3f}"


def _accuracy(P, rows):
    return sum(1 for p, r in zip(P, rows) if _argmax(p) == r["y"]) / len(rows)


def _nll(P, rows):
    return sum(-math.log(p[r["y"]]) for p, r in zip(P, rows)) / len(rows)


def _brier(P, rows):
    """Mean over requests of the sum over routes of (p_k - 1[k = label])^2: divisor n, not n K."""
    return sum(sum((p[k] - (1.0 if k == r["y"] else 0.0)) ** 2 for k in range(K)) for p, r in zip(P, rows)) / len(rows)


def _top_ece(P, rows):
    """Top-label ECE: the pairs (p_max, 1 if the argmax is right else 0) go through the binary ECE of Volume XI, 10 equal-width bins."""
    conf = [max(p) for p in P]
    hit = [1 if _argmax(p) == r["y"] else 0 for p, r in zip(P, rows)]
    return cd._ece(conf, hit)


# ------------------------------------------------------------------ the world, printed

def world_rule():
    concept, words, splits = _world()
    fam_rows = [[ROUTES[k], ", ".join(FAMILIES[k]), ", ".join(_fmt(v, 3) for v in concept[k])] for k in range(K)]
    first = splits["train"][:5]
    rows = []
    for i, r in enumerate(first):
        rows.append([i, ROUTES[r["y"]], " ".join(r["words"]), ", ".join(_fmt(v, 2) for v in r["raw"])])
    counts = [[name, n] + [sum(1 for r in splits[name] if r["y"] == k) for k in range(K)] for name, n in SPLITS]
    text = ("The generator is " + _seed_line(WORLD, True) + " One generator runs across the whole world, in this order. "
            "First the four concept vectors, 6 `normal()` calls each, in route order. Then the word vectors: for each family in route "
            "order and each of its four words in the listed order, coordinate j is the concept coordinate plus 0.5 times one `normal()`; "
            "then the four filler words, each coordinate 0.5 times one `normal()`. Then the splits, train, validation, test and log, one "
            "request after another. One request draws, in this order: `u = uniform()` for the route (the first route whose running "
            "prior sum 0.35, 0.65, 0.85, 1.0 exceeds `u`); for each of three word slots `v = uniform()` (own family if `v < 0.65`, the "
            "partner family 2, 3, 0, 1 of the routes 0, 1, 2, 3 if `v < 0.85`, a filler otherwise) and then `integer(4)` for the word "
            "of that family; for each of two filler slots `integer(4)`; then `normal()` for the amount, `integer(60)` for the age, "
            "`uniform()` for the new-device flag (1 if below 0.1, 0.2, 0.15, 0.6 for the routes 0 to 3), `integer(3)` for the contacts, "
            "and for route 3 a second `integer(3)` added to them. ln amount is 3.0, 3.6, 2.6, 4.4 for the routes 0 to 3 plus 0.8 times "
            "the `normal()`; the age is `1 + integer(60)` months.\n\n"
            + _table(["route", "family words", "concept vector (3 decimals)"], fam_rows, "lll")
            + "\n\nThe filler words are " + ", ".join(FILLER) + ". The first five training requests (the numbers are ln amount, age, "
            "new device, contacts):\n\n" + _table(["request", "route", "words", "numbers"], rows, "rlll")
            + "\n\nRequests per route in each split:\n\n"
            + _table(["split", "requests"] + list(ROUTES), counts, "lrrrrr"))
    return text


def state_vector():
    concept, words, splits = _world()
    mean, sd = _scaling()
    r = splits["train"][0]
    vec = [[_fmt(v, 3) for v in words[w]] for w in r["words"]]
    s = [sum(words[w][j] for w in r["words"]) / len(r["words"]) for j in range(M)]
    n = math.sqrt(sum(v * v for v in s))
    rows = [[w] + vec[i] for i, w in enumerate(r["words"])]
    rows.append(["mean of the five"] + [_fmt(v, 4) for v in s])
    rows.append([f"divided by its length {n:.4f}"] + [_fmt(v, 4) for v in r["u"]])
    x = _state(r)
    srows = [[FEATURES[j], _fmt(r["raw"][j], 4), _fmt(mean[j], 4), _fmt(sd[j], 4), _fmt(x[M + j], 4)] for j in range(4)]
    return ("Request 0 of the training split: `" + " ".join(r["words"]) + "`.\n\n"
            + _table(["word"] + [f"v{j}" for j in range(M)], rows, "l" + "r" * M)
            + "\n\nThe four numbers are scaled by the mean and the standard deviation (divisor n = 120) of the training requests:\n\n"
            + _table(["field", "value", "training mean", "training sd", "scaled"], srows, "lrrrr")
            + "\n\nThe state vector is the six numbers of the last line of the first table followed by the four scaled numbers: "
            + "(" + ", ".join(_fmt(v, 4) for v in x) + ").")


# ------------------------------------------------------------------ the head

def head_parameters():
    seeds = [[h, _seed_line(f"{INIT}.{h}"), d] for h, d in HEADS]
    W0 = _init("both")
    runs = _runs("both")
    cols_t = [f"u{j}" for j in range(M)]
    cols_n = ["amount", "age", "device", "contacts"]

    def table(W, b, fmt, stab=None):
        t1 = _table(["route"] + cols_t + ["bias"],
                    [[ROUTES[k]] + [fmt(k, j) for j in range(M)] + [fmt(k, "b")] for k in range(K)], "l" + "r" * (M + 1))
        t2 = _table(["route"] + cols_n, [[ROUTES[k]] + [fmt(k, M + j) for j in range(4)] for k in range(K)], "lrrrr")
        return t1 + "\n\n" + t2

    init_text = table(W0, [0.0] * K, lambda k, j: "0" if j == "b" else _fmt(W0[k][j], 4))
    final_text = table(None, None, lambda k, j: _st("both", (lambda r, k=k, j=j: r["b"][k] if j == "b" else r["W"][k][j]), _f4))
    loss = [[str(s), _st("both", lambda r, s=s: r["loss"][s], _f4)] for s in (0, 10, 100, 500, STEPS)]
    return ("Three heads, three initial draws (`W` row by row, route by route, `0.1 * normal()` per entry, bias 0):\n\n"
            + _table(["head", "generator", "inputs"], seeds, "llr")
            + "\n\nInitial parameters of the head on the whole state (44 numbers; the draws of the first row are the first six of the first "
            "table, then the four of the second):\n\n" + init_text
            + f"\n\nTraining: {STEPS} steps of full-batch gradient descent with step size {RATE} on the mean cross-entropy of the 120 training "
            "requests (the gradient of a row is `p - onehot(y)` times the input, summed over the rows in order and divided by 120, "
            "the bias gradient is the same without the input). Mean training cross-entropy at some steps (step 0 is the initial "
            "head):\n\n" + _table(["step", "loss"], loss, "rr")
            + "\n\nParameters after step 1500. Digits are printed only where the run and its four copies with the first initial weight changed "
            "by +1e-12, -1e-12, +3e-12 and -3e-12 agree; the objective is convex, so a copy is pulled back to the same path and all "
            "digits agree:\n\n" + final_text)


# ------------------------------------------------------------------ intent classification, measured

def _baselines():
    w = _world()[2]
    counts = [sum(1 for r in w["train"] if r["y"] == k) for k in range(K)]
    major = _argmax([float(c) for c in counts])
    test = w["test"]
    return major, sum(1 for r in test if r["y"] == major) / len(test), 1.0 / K


def intent_heads():
    w = _world()[2]
    major, major_acc, chance = _baselines()
    rows = []
    for head, _ in HEADS:
        row = [head]
        for split in ("train", "validation", "test"):
            rows_s = w[split]
            row.append(_st(head, lambda r, s=split, rs=rows_s: _accuracy(r["raw"][s], rs), _f4))
        for split in ("train", "test"):
            rows_s = w[split]
            row.append(_st(head, lambda r, s=split, rs=rows_s: _nll(r["raw"][s], rs), _f4))
        rows.append(row)
    conf = [[0] * K for _ in range(K)]
    runs = _runs("both")
    for p, r in zip(runs[0]["raw"]["test"], w["test"]):
        conf[r["y"]][_argmax(p)] += 1
    stable_conf = all(
        sum(1 for p, r in zip(run["raw"]["test"], w["test"]) if r["y"] == a and _argmax(p) == b) == conf[a][b]
        for run in runs[1:] for a in range(K) for b in range(K))
    crow = [[ROUTES[a]] + conf[a] + [f"{conf[a][a] / sum(conf[a]):.4f}"] for a in range(K)]
    return ("Accuracy is the share of requests whose argmax (the first of equal probabilities) is the true route; cross-entropy is the "
            "mean of -ln of the probability of the true route; four decimals, raw probabilities (temperature 1).\n\n"
            + _table(["head", "accuracy, train", "accuracy, validation", "accuracy, test", "cross-entropy, train", "cross-entropy, test"], rows)
            + f"\n\nThe majority route of the training split is {ROUTES[major]}; always answering it gives {major_acc:.4f} on the test split, and "
            f"a uniform guess is right with probability {chance:.2f}.\n\nConfusion matrix of the head on the whole state, test split (rows true "
            "route, columns predicted route, then the share of the row that is on the diagonal):\n\n"
            + _table(["true \\ predicted"] + list(ROUTES) + ["row recall"], crow, "l" + "r" * (K + 1))
            + ("\n\nThe counts are the same in all five runs." if stable_conf else "\n\nThe counts differ between the runs."))


# ------------------------------------------------------------------ calibration

def _bins_of(P, rows):
    conf = [max(p) for p in P]
    hit = [1 if _argmax(p) == r["y"] else 0 for p, r in zip(P, rows)]
    return cd._bins(conf, hit)


def _reliability_series():
    """(raw bins, calibrated bins) of the head on the whole state, test split, from the unnudged run."""
    test = _world()[2]["test"]
    run = _runs("both")[0]
    return _bins_of(run["raw"]["test"], test), _bins_of(run["cal"]["test"], test)


def calibration():
    w = _world()[2]
    test = w["test"]
    t = _st("both", lambda r: r["T"], _f4)
    rows = []
    for label, key in (("raw, T = 1", "raw"), ("scaled, T fitted", "cal")):
        rows.append([label,
                     _st("both", lambda r, k=key: _nll(r[k]["test"], test), _f4),
                     _st("both", lambda r, k=key: _brier(r[k]["test"], test), _f4),
                     _st("both", lambda r, k=key: _top_ece(r[k]["test"], test), _f4),
                     _st("both", lambda r, k=key: sum(max(p) for p in r[k]["test"]) / len(test), _f4),
                     _st("both", lambda r, k=key: _accuracy(r[k]["test"], test), _f4)])
    runs = _runs("both")
    val_before = _st("both", lambda r: _nll(r["raw"]["validation"], w["validation"]), _f4)
    val_after = _st("both", lambda r: _nll(r["cal"]["validation"], w["validation"]), _f4)
    return (f"The temperature is fitted on the 500 validation requests by minimising the mean cross-entropy of softmax(z / T): T = {t}; "
            f"on the validation split the mean cross-entropy goes from {val_before} to {val_after}. The scores below are on the 1000 test requests "
            "(natural logarithms; Brier is the mean over requests of the sum over the four routes of the squared difference to the one-hot "
            "label; ECE takes the pair (largest probability, whether the argmax is right) through 10 equal-width bins, bin i holding "
            "i/10 <= p < (i+1)/10 and 1.0 in the last, the count-weighted mean of the absolute gap, as in Volume XI, Chapter 6; "
            "'mean p_max' is the mean of the largest probability, to be compared with the accuracy).\n\n"
            + _table(["probabilities", "cross-entropy", "Brier", "ECE", "mean p_max", "accuracy"], rows)
            + "\n\nDividing every logit by one number cannot change which logit is largest, so the accuracy is the same in both rows.")


def reliability_bins():
    raw, cal = _reliability_series()
    rows = []
    for k in range(N_BINS):
        r, c = raw[k], cal[k]
        rows.append([f"{k / N_BINS:.1f} to {(k + 1) / N_BINS:.1f}",
                     r[0] if r else 0, _fmt(r[1], 3) if r else "none", _fmt(r[2], 3) if r else "none",
                     c[0] if c else 0, _fmt(c[1], 3) if c else "none", _fmt(c[2], 3) if c else "none"])
    return ("Test split, unnudged run (counts and means of a bin are what the figure plots); bin = range of the largest probability; "
            "'accuracy' is the share of the bin's requests whose argmax is right; three decimals.\n\n"
            + _table(["bin", "raw count", "raw mean p_max", "raw accuracy", "scaled count", "scaled mean p_max", "scaled accuracy"], rows, "lrrrrrr"))


# ------------------------------------------------------------------ the decision rule

def _decide(p, c_wrong, c_esc=C_ESCALATE):
    """Act on the argmax when 1 - p_max <= c_esc / c_wrong, that is when p_max >= 1 - c_esc / c_wrong (not strict: a tie acts), else escalate."""
    return _argmax(p) if max(p) >= 1.0 - c_esc / c_wrong else None


def _rule_stats(P, rows, c_wrong):
    n = len(rows)
    answered = right = 0
    cost = 0.0
    for p, r in zip(P, rows):
        a = _decide(p, c_wrong)
        if a is None:
            cost += C_ESCALATE
        else:
            answered += 1
            if a == r["y"]:
                right += 1
            else:
                cost += c_wrong
    quoted = sum(min((1.0 - max(p)) * c_wrong, C_ESCALATE) for p in P) / n
    return {"coverage": answered / n, "selective": right / answered if answered else 0.0, "paid": cost / n, "quoted": quoted}


def decision_rule():
    test = _world()[2]["test"]
    rows = []
    for c in C_WRONG:
        t = 1.0 - C_ESCALATE / c
        for label, key in (("raw", "raw"), ("scaled", "cal")):
            vals = [_st("both", lambda r, k=key, c=c, f=f: _rule_stats(r[k]["test"], test, c)[f], _f4)
                    for f in ("coverage", "selective", "quoted", "paid")]
            rows.append([f"{c:g}", f"{t:.4f}", label] + vals)
    base = _runs("both")[0]["raw"]["test"]
    wrong = sum(1 for p, r in zip(base, test) if _argmax(p) != r["y"]) / len(test)
    ref = [[f"{c:g}", f"{wrong * c:.4f}", f"{C_ESCALATE:.4f}"] for c in C_WRONG]
    run = _runs("both")[0]
    moved = []
    for c in C_WRONG:
        held = [(p, r) for p, q, r in zip(run["raw"]["test"], run["cal"]["test"], test)
                if _decide(p, c) is not None and _decide(q, c) is None]
        right = sum(1 for p, r in held if _argmax(p) == r["y"])
        moved.append([f"{c:g}", str(len(held)), str(right), f"{(len(held) - right) * c:g}", f"{len(held) * C_ESCALATE:g}"])
    def _on_threshold(r, c):
        t = round(1.0 - C_ESCALATE / c, 2)
        return sum(1 for p in r["cal"]["test"] if round(max(p), 2) == t)
    ties = [[f"{c:g}", f"{1.0 - C_ESCALATE / c:.2f}", _st("both", lambda r, c=c: _on_threshold(r, c), lambda x: f"{x:.0f}")] for c in C_WRONG]
    return ("A wrong route costs c, a right route 0 and handing the request to a person costs 1 whatever the truth. The rule acts when "
            "p_max >= 1 - 1/c (a tie acts) and otherwise escalates. 'coverage' is the share of requests the rule acts on; 'selective "
            "accuracy' the share of those whose route is right (an escalated request is in neither the numerator nor the denominator); "
            "'quoted' the mean over all requests of min((1 - p_max) c, 1), the cost the probabilities promise; 'paid' the mean cost "
            "actually incurred, an escalation counted as 1. Test split, four decimals.\n\n"
            + _table(["c", "threshold", "probabilities", "coverage", "selective accuracy", "quoted", "paid"], rows, "rrlrrrr")
            + "\n\nThe two constant policies on the same requests: always act on the argmax costs (1 - accuracy) c, always escalate costs 1:\n\n"
            + _table(["c", "always act", "always escalate"], ref)
            + "\n\nThe requests the temperature takes out of the routed set (the raw probabilities act, the scaled ones escalate; no request moves the "
            "other way, since T > 1 only lowers p_max), unnudged run: how many, how many of them the raw argmax routes right, what routing "
            "them costs (the wrong ones at c) and what escalating them costs (1 each).\n\n"
            + _table(["c", "requests", "routed right", "cost if routed", "cost if escalated"], moved, "rrrrr")
            + "\n\nThe tie convention in numbers: the scaled probabilities rounded to two decimals put p_max exactly on the threshold for this many of the 1000 test "
            "requests. All of them act under the rule above and would escalate under a strict inequality, so the coverage of that variant is lower by this count over 1000 "
            "(for rounded probabilities).\n\n"
            + _table(["c", "threshold", "requests with p_max on it"], ties, "rrr"))


# ------------------------------------------------------------------ a cost matrix that is not argmax

COST_MATRIX = ((0.0, 2.0, 2.0, 2.0),          # route to card: cost when the truth is card, payment, account, dispute
               (2.0, 0.0, 2.0, 2.0),          # route to payment
               (2.0, 2.0, 0.0, 2.0),          # route to account
               (10.0, 10.0, 10.0, 0.0),       # route to dispute: it is the costly team to disturb, so a wrong one is dear
               (1.5, 1.5, 1.5, 1.5))          # escalate
ACTIONS = ROUTES + ("escalate",)


def _bayes(p):
    costs = [sum(p[y] * COST_MATRIX[a][y] for y in range(K)) for a in range(len(COST_MATRIX))]
    best = 0
    for a, c in enumerate(costs):
        if c < costs[best]:
            best = a
    return best, costs


def cost_matrix():
    test = _world()[2]["test"]
    run = _runs("both")[0]
    P = run["cal"]["test"]
    differ = [i for i, p in enumerate(P) if _bayes(p)[0] != _argmax(p) and _bayes(p)[0] != 4]
    esc = sum(1 for p in P if _bayes(p)[0] == 4)
    mism = sum(1 for p in P if _bayes(p)[0] != _argmax(p))
    mat = [[ACTIONS[a]] + [f"{c:g}" for c in COST_MATRIX[a]] for a in range(len(COST_MATRIX))]
    i = differ[0] if differ else None
    ex = ""
    if i is not None:
        p = P[i]
        best, costs = _bayes(p)
        ex = (f"\n\nTest request {i} (`{' '.join(test[i]['words'])}`, true route {ROUTES[test[i]['y']]}) has the scaled probabilities "
              + ", ".join(f"{ROUTES[k]} {p[k]:.4f}" for k in range(K)) + ". The expected cost of each action:\n\n"
              + _table(["action"] + ["expected cost"], [[ACTIONS[a], f"{costs[a]:.4f}"] for a in range(5)], "lr")
              + f"\n\nThe argmax is {ROUTES[_argmax(p)]}; the cheapest action is {ACTIONS[best]}.")
    return ("Five actions against four truths; the entry in row a and column y is the cost of action a when the request is truly of route y "
            "(the same rule as a threshold, argmin over actions of the sum over truths of probability times cost; a tie goes to the first "
            "action in the order below).\n\n" + _table(["action"] + list(ROUTES), mat, "lrrrr")
            + f"\n\nOn the 1000 test requests with the scaled probabilities, the cheapest action differs from the argmax on {mism} requests, "
            f"{esc} of them escalations, and the other {mism - esc} route to a team that is not the most probable one." + ex)


# ------------------------------------------------------------------ logged outcomes

@lru_cache(maxsize=None)
def _logged():
    """The log: for each of the 2000 log requests, the route the old policy chose and its propensity.

    The old policy follows the numbers-only head's argmax a0 with probability 0.7 and otherwise picks a route uniformly at random
    (one `uniform()` to decide, one `integer(4)` when random): P(a | s) = 0.7 [a = a0] + 0.3 / 4."""
    g = bookrng.generator(LOGGING)
    run = _runs("numbers")[0]
    out = []
    for r in _world()[2]["log"]:
        a0 = _argmax(_softmax(_logits(run["W"], run["b"], _state(r, "numbers"))))
        a = a0 if g.uniform() < 0.7 else g.integer(K)
        prop = 0.7 * (1.0 if a == a0 else 0.0) + 0.3 / K
        out.append((a, prop, 1.0 if a == r["y"] else 0.0))
    return out


def _estimates(run, lo, hi, target="head"):
    """The four estimates of the new policy's value from the log requests lo .. hi - 1 (lo inclusive, hi exclusive), and the true value."""
    log = _logged()
    rows = _world()[2]["log"]
    n = hi - lo
    if "new_route" not in run:
        run["new_route"] = [_argmax(_softmax(_logits(run["W"], run["b"], _state(x, "both")))) for x in rows]
    routes = run["new_route"] if target == "head" else [3] * len(rows)
    ent = list(zip(routes[lo:hi], log[lo:hi], rows[lo:hi]))
    old = sum(l[2] for _, l, _ in ent) / n
    truth = sum(1.0 for a, _, x in ent if a == x["y"]) / n
    agree = [l for a, l, _ in ent if a == l[0]]
    ipw = sum(l[2] / l[1] for l in agree) / n
    snips = sum(l[2] / l[1] for l in agree) / sum(1.0 / l[1] for l in agree)
    naive = sum(l[2] for l in agree) / len(agree)
    return {"old": old, "truth": truth, "ipw": ipw, "snips": snips, "naive": naive}


BLOCK = 100


def _block_stats(run, key, target="head"):
    """(mean, standard deviation with divisor = number of blocks) of one estimate over the blocks of BLOCK consecutive log requests."""
    cache = run.setdefault("blocks", {})
    if (target, key) not in cache:
        n_all = len(_world()[2]["log"])
        vals = [_estimates(run, i, i + BLOCK, target)[key] for i in range(0, n_all, BLOCK)]
        mean = sum(vals) / len(vals)
        cache[(target, key)] = (mean, math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)))
    return cache[(target, key)]


def logged_outcomes():
    n_all = len(_world()[2]["log"])
    names = (("old policy (mean logged outcome)", "old"), ("true value", "truth"), ("inverse propensity", "ipw"),
             ("self-normalised inverse propensity", "snips"), ("logged agreements only, no weights", "naive"))
    rows = []
    for target, tlabel in (("head", "new policy = the head's argmax"), ("dispute", "new policy = always route to dispute")):
        for label, key in names[1:]:
            whole = _st("both", lambda r, k=key, t=target: _estimates(r, 0, n_all, t)[k], _f4)
            mean = _st("both", lambda r, k=key, t=target: _block_stats(r, k, t)[0], _f4)
            sd = _st("both", lambda r, k=key, t=target: _block_stats(r, k, t)[1], _f4)
            rows.append([tlabel, label, whole, mean, sd])
    old = _st("both", lambda r: _estimates(r, 0, n_all)["old"], _f4)
    return (f"The log is the {n_all} log requests, each routed by the old policy (generator {_seed_line(LOGGING)}; for each request in order one "
            "`uniform()`, below 0.7 the numbers-only head's argmax, otherwise one `integer(4)` for a uniformly random route), with the "
            "outcome 1 when the route was right and 0 otherwise and the propensity P(a | s) = 0.7 [a = old argmax] + 0.075 of the route it "
            "chose. Two target policies are valued: the argmax of the head on the whole state, and the constant policy that routes everything "
            "to dispute. The true value of a policy is the share of the requests it would route right, which only a simulation that keeps "
            "the labels can print. With w = outcome / propensity on the requests "
            "where the new policy's route is the logged one (and nothing elsewhere), 'inverse propensity' is the sum of w over all n requests "
            "divided by n, the self-normalised form divides the sum of w by the sum of 1 / propensity over the same requests, and the last "
            f"row is the mean outcome over those requests alone. The first column uses all {n_all} requests; the other two split the log into "
            f"{n_all // BLOCK} blocks of {BLOCK} consecutive requests, compute each estimate on every block, and give the mean and the "
            f"standard deviation (divisor {n_all // BLOCK}) of the {n_all // BLOCK} values. Four decimals.\n\n"
            + _table(["target policy", "estimate of its value", f"all {n_all}", f"mean, blocks of {BLOCK}", f"sd, blocks of {BLOCK}"], rows, "llrrr")
            + f"\n\nFor reference the old policy's own mean logged outcome over all {n_all} requests is {old}.")


# ------------------------------------------------------------------ one decision, audited

def _digest(run):
    """SHA-256 of the 44 parameters, W row-major then b, each as the little-endian 32-bit integer round(1e4 w), as a hex string."""
    vals = [w for row in run["W"] for w in row] + list(run["b"])
    data = b"".join(struct.pack("<i", int(math.floor(abs(v) * 1e4 + 0.5)) * (1 if v >= 0 else -1)) for v in vals)
    import hashlib
    return hashlib.sha256(data).hexdigest()


def audit_record():
    w = _world()[2]
    test = w["test"]
    runs = _runs("both")
    run = runs[0]
    T = run["T"]
    c = 4.0
    esc = next(i for i, p in enumerate(run["cal"]["test"]) if _decide(p, c) is None)
    act = next(i for i, p in enumerate(run["cal"]["test"]) if _decide(p, c) is not None)
    digs = {_digest(r)[:16] for r in runs}
    out = []
    for i in (act, esc):
        r = test[i]
        x = _state(r)
        z = _logits(run["W"], run["b"], x)
        p = _softmax([v / T for v in z])
        a = _decide(p, c)
        out.append((i, r, x, z, p, a))
    rec = []
    for i, r, x, z, p, a in out:
        rec.append([i, " ".join(r["words"]), ", ".join(_fmt(v, 2) for v in r["raw"]),
                    ", ".join(_fmt(v, 4) for v in z), ", ".join(_fmt(v, 4) for v in p), f"{max(p):.4f}",
                    "route to " + ROUTES[a] if a is not None else "escalate"])
    # replay: the decision recomputed from the logged state vector alone, and with the test split in another order
    i, r, x, z, p, a = out[0]
    again = _softmax([v / T for v in _logits(run["W"], run["b"], x)])
    g = bookrng.generator("decision_models.replay")
    order = list(range(len(test)))
    g.shuffle(order)
    base = run["cal"]["test"]
    shuffled = {}
    for idx in order:
        shuffled[idx] = _softmax([v / T for v in _logits(run["W"], run["b"], _state(test[idx]))])
    diff = sum(1 for idx in range(len(test)) if shuffled[idx] != base[idx])
    # explanation of the first decision: why the chosen route over the runner-up
    top = _argmax(p)
    second = max((k for k in range(K) if k != top), key=lambda k: (p[k], -k))
    groups = [("sentence vector (6 numbers)", range(0, M))] + [(FEATURES[j], [M + j]) for j in range(4)]
    contrib = []
    total = 0.0
    for name, idx in groups:
        s = 0.0
        for j in idx:
            s += (run["W"][top][j] - run["W"][second][j]) * x[j]
        contrib.append([name, _fmt(s, 4)])
        total += s
    bias = run["b"][top] - run["b"][second]
    contrib.append(["bias", _fmt(bias, 4)])
    total += bias
    contrib.append(["sum", _fmt(total, 4)])
    return ("The decision rule is the one of the previous block at c = 4 (act when p_max >= 0.75), the probabilities are those of the head on "
            f"the whole state divided by the fitted temperature T = {T:.4f}. The model identity is the SHA-256 of the 44 parameters (W row by "
            "row, then b), each written as the little-endian 32-bit integer round(1e4 w) (ties away from zero); first 16 hex digits "
            f"{_digest(run)[:16]}" + (", the same in all five runs." if len(digs) == 1 else f", which differs between the runs ({len(digs)} values)."
            ) + "\n\nTwo test requests, the first acted on and the first escalated:\n\n"
            + _table(["request", "words", "numbers", "logits z", "probabilities", "p_max", "action"], rec, "rllllrl")
            + f"\n\nReplay. The probabilities of request {i} recomputed from its logged state vector alone differ from the first computation by "
            f"{max(abs(a - b) for a, b in zip(p, again)):.1f} in every entry. The 1000 test requests scored again in the order of one "
            "`shuffle` of the indices 0 to 999 (generator `decision_models.replay`, seed "
            f"{bookrng.seed_of('decision_models.replay'):#018x}) give probabilities that differ from the first pass on {diff} requests (a "
            "request's answer does not depend on the others).\n\n"
            f"Why {ROUTES[top]} rather than {ROUTES[second]} for request {i}: the logit difference z[{ROUTES[top]}] - z[{ROUTES[second]}] "
            "splits exactly into one contribution per input group (the weight difference times the input, summed in input order) and the "
            "bias difference:\n\n" + _table(["input group", "contribution to the logit difference"], contrib, "lr"))


# ------------------------------------------------------------------ what a decision costs

D, L, HEADS_N, FF, VOCAB, POS, TYPES = 384, 6, 12, 1536, 30522, 512, 2


def _encoder_params():
    emb = VOCAB * D + POS * D + TYPES * D + 2 * D
    layer = (3 * D * D + 3 * D) + (D * D + D) + 2 * D + (D * FF + FF) + (FF * D + D) + 2 * D
    return emb, layer, emb + L * layer


def _head_params():
    return (D * D + D) + (2 * D * D + 2 * D) + 2 * D + (D + 1)


def _encode_flops(n):
    """FLOPs of the six layers over one span of n tokens: the four projections and the two feed-forward matrices at 2 FLOPs per
    multiply-add, plus the attention scores and the weighted sum of values (4 n^2 d). Layer norms, GELU and softmax are not counted."""
    per_token = 2 * (3 * D * D + D * D + 2 * D * FF)
    return L * (n * per_token + 4 * n * n * D)


def _head_flops(rows, options):
    """Key and value projection of the state rows once, query projection and attention per option, and the score."""
    return 2 * rows * D * 2 * D + options * (2 * D * D + 4 * rows * D + 2 * D)


def cost_per_decision():
    emb, layer, total = _encoder_params()
    hp = _head_params()
    state, slot, options = 32, 8, 4
    big, prompt, shared, per_q = 7e9, 200, 160, 40
    rows = []
    for q in (1, 10):
        enc = _encode_flops(state) + q * options * _encode_flops(slot)
        head = _head_flops(state, q * options)
        small = enc + head
        large_sep = q * 2 * big * prompt
        large_shared = 2 * big * (shared + q * per_q)
        rows.append([q, f"{enc:.3e}", f"{head:.3e}", f"{small:.3e}", f"{large_sep:.3e}", f"{large_shared:.3e}",
                     f"{large_shared / small:,.0f}"])
    ptab = [["token, position, type embeddings and their layer norm", f"{emb:,}"], ["one layer", f"{layer:,}"],
            [f"{L} layers", f"{L * layer:,}"], ["encoder total", f"{total:,}"], ["cross-attention head", f"{hp:,}"]]
    sb = 4 * (total + hp)
    return ("The encoder is the 6-layer, 384-wide, 12-head, 1536-wide-feed-forward, 30522-word encoder that `crates/decide` reads from a released "
            "checkpoint (token table 30522 x 384, position table 512 x 384, type table 2 x 384, one layer norm; per layer a fused query-key-value "
            "matrix 384 -> 1152, an output matrix 384 -> 384, two layer norms and the feed-forward pair 384 -> 1536 -> 384, each with a bias). "
            "The cross-attention head is a query matrix 384 x 384 with a bias (147,840), a fused key-value matrix 384 -> 768 with a bias (295,680), "
            "one layer norm of 384 (gain and bias, 768) and a score vector of 384 with a bias (385). Parameter counts:\n\n" + _table(["part", "numbers"], ptab, "lr")
            + f"\n\nIn 32-bit floats the weights are {sb:,} bytes. A span of n tokens costs, per layer, n x 2(3 d^2 + d^2 + 2 d f) FLOPs for the "
            "matrix products and 4 n^2 d for the attention, with d = 384 and f = 1536; the head costs 2 R d (2d) for the keys and values of the R "
            "state rows once, and per option 2 d^2 for the query, 4 R d for the attention and 2 d for the score. Assumptions: a state of "
            f"R = {state} tokens, {options} options of {slot} tokens each (every option is its own span), Q questions that share the state. "
            f"The large model is a decoder with {big:.0e} parameters at 2 FLOPs per parameter per prompt token and 2 bytes per weight; its prompt "
            f"is {prompt} tokens ({shared} of state, {per_q} of instruction and options); 'shared' reads the state once for all Q questions, "
            "'separate' reads the whole prompt for each. Attention of the large model is not counted, which favours it.\n\n"
            + _table(["Q", "small encoder", "small head", "small total", "large, separate", "large, shared", "large shared / small"], rows)
            + f"\n\nBytes of weights read once per forward pass at batch 1: small {sb / 1e6:,.1f} MB, large {2 * big / 1e9:,.0f} GB, "
            f"a ratio of {2 * big / sb:,.0f}.")


# ------------------------------------------------------------------ the details of brain's decide crate

def common_mode():
    cos = 0.989
    sin = math.sqrt(1.0 - cos * cos)
    qy, qn = (1.0, 0.0), (cos, sin)
    states = (("obvious close", (0.60, 0.10)), ("obvious loss", (0.35, 0.10)), ("undecided", (0.10, 0.05)))
    rows = []
    ys, ds = [], []
    for name, c in states:
        sy = qy[0] * c[0] + qy[1] * c[1]
        sn = qn[0] * c[0] + qn[1] * c[1]
        ys.append(sy)
        ds.append(sy - sn)
        rows.append([name, f"({c[0]}, {c[1]})", _fmt(sy), _fmt(sn), _fmt(sy - sn)])
    gap = math.sqrt(2.0 - 2.0 * cos)
    return ("Two option queries of length 1 at cosine 0.989: q_yes = (1, 0) and q_no = (0.989, sqrt(1 - 0.989^2)). The context vector c of a state "
            "is what the attention hands back (chosen by hand, three states); an option's score is the dot product of its query with c. Four decimals.\n\n"
            + _table(["state", "context c", "score yes", "score no", "yes - no"], rows, "llrrr")
            + f"\n\nSpread (largest minus smallest) of the yes score over the three states: {max(ys) - min(ys):.4f}; of the difference: "
            f"{max(ds) - min(ds):.4f}. The difference of two unit queries at cosine 0.989 is (q_yes - q_no) . c, and |q_yes - q_no| = sqrt(2 - 2 x 0.989) = "
            f"{gap:.4f}, so the difference can never be more than {gap:.4f} times the length of c, while one score alone can be as large as c itself.")


def confidence_measure():
    rows = []
    for k in (2, 5, 77, 255):
        uni = [1.0 / k] * k
        one = [1.0] + [0.0] * (k - 1)
        half = [0.5] + [0.5 / (k - 1)] * (k - 1)

        def conf(p):
            h = sum(-v * math.log(v) for v in p if v > 0.0)
            return min(1.0, max(0.0, 1.0 - h / math.log(k)))
        rows.append([k, f"{max(uni):.4f}", f"{conf(uni):.4f}", f"{max(one):.4f}", f"{conf(one):.4f}", f"{max(half):.4f}", f"{conf(half):.4f}"])
    return ("confidence(p) = 1 - H(p) / ln k with H the entropy in nats and 0 ln 0 = 0, clamped to [0, 1]; four decimals. Three distributions over k options: "
            "uniform, one-hot, and one option at 0.5 with the rest sharing 0.5 equally.\n\n"
            + _table(["k", "uniform p_max", "uniform confidence", "one-hot p_max", "one-hot confidence", "half p_max", "half confidence"], rows))


def reward_direction():
    rows = []
    for q in (0.7, 0.2):
        for p in (0.3, 0.5, 0.7, 0.9, 0.99):
            hit = (2 * q - 1) * p * (1 - p)
            prop = 2 * (q - p) * p * (1 - p)
            rows.append([f"{q:.1f}", f"{p:.2f}", _fmt(hit), _fmt(prop)])
    ends = []
    for q in (0.7, 0.2):
        for name, rule in (("hit", lambda q, p: (2 * q - 1) * p * (1 - p)), ("proper", lambda q, p: 2 * (q - p) * p * (1 - p))):
            z = 0.0
            for _ in range(20000):
                p = 1.0 / (1.0 + math.exp(-z))
                z += 1.0 * rule(q, p)
            ends.append([f"{q:.1f}", name, f"{1.0 / (1.0 + math.exp(-z)):.4f}"])
    return ("A state whose outcome has probability q of 1; the policy commits the probability p = sigmoid(z) and, in the first reading, "
            "samples a call (1 with probability p) and is paid 1 if the call equals the outcome. Its expected reward is p q + (1 - p)(1 - q), "
            "whose slope with respect to z is (2q - 1) p (1 - p). In the second reading it is paid 1 - (p - y)^2, whose expectation over "
            "y has the slope 2 (q - p) p (1 - p). The ascent direction in z (four decimals):\n\n"
            + _table(["q", "p", "hit-rate reward", "proper-score reward"], rows)
            + "\n\nThe value of p after 20000 steps of z <- z + 1.0 x direction from z = 0 (p = 0.5):\n\n"
            + _table(["q", "reward", "p after 20000 steps"], ends, "rlr"))
