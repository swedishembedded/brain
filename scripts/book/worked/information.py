# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Information, Entropy and Surprise".

Plain Python 3, standard library only, written from the chapter's own
description. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between
`<!-- worked: information.<name> -->` markers. Helpers with a leading
underscore (or that take arguments) are shared with the chapter's figures, so a
picture and the table beside it cannot disagree. Code constructions use exact
`fractions.Fraction` arithmetic; every simulation draws from a named `bookrng`
generator.

Swedish Embedded AB implements solutions for measuring how well a model's
beliefs match the data. If your team needs expertise in information-theoretic
evaluation of machine learning models, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import heapq
import math
from fractions import Fraction

import bookrng
import probability as pr

INF = float("inf")


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
    if v == INF:
        return "inf"
    v = float(v) + 0.0  # turns -0.0 into 0.0
    return f"{v:.{d}f}"


def _softmax(scores):
    m = max(scores)
    e = [math.exp(z - m) for z in scores]
    s = sum(e)
    return [v / s for v in e]


def surprise(p, base=2.0):
    """Information content -log p of an event of probability p (inf for p = 0)."""
    if p == 0:
        return INF
    return -math.log(p) / math.log(base)


def entropy(p, base=2.0):
    """Expected surprise sum p log(1/p), with 0 log 0 taken as 0."""
    return sum(pi * surprise(pi, base) for pi in p if pi > 0)


def cross_entropy(p, q, base=2.0):
    """Expected surprise under q of data drawn from p (inf if q misses an outcome p allows)."""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            total += pi * surprise(qi, base)
    return total


def kl_divergence(p, q, base=2.0):
    """sum p log(p / q), computed directly (not as a difference of two entropies)."""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi == 0:
                return INF
            total += pi * math.log(pi / qi) / math.log(base)
    return total


# The five-word next-token distribution of the distributions chapter.
NEXT_WORDS = ["mat", "floor", "sofa", "moon", "the"]
NEXT_SCORES = [3.0, 2.0, 1.0, -1.0, 0.0]


# The three contexts of the next-token examples: name and the scores (logits) for NEXT_WORDS.
NEXT_CONTEXTS = [
    ("confident context", [6.0, 1.0, 0.0, -2.0, -1.0]),
    ("Chapter 11's context", NEXT_SCORES),
    ("undecided context", [0.4, 0.3, 0.2, -0.1, 0.0]),
]


def next_token_distribution():
    return _softmax(NEXT_SCORES)


# Four weather symbols used for codes and cross-entropy.
WEATHER = ["sun", "cloud", "rain", "snow"]
WEATHER_P = [Fraction(4, 10), Fraction(3, 10), Fraction(2, 10), Fraction(1, 10)]


# ------------------------------------------------- guessing games, surprise

def twenty_questions():
    """How many yes/no questions pin down one of n equally likely items."""
    rows = []
    for n in (2, 8, 20, 1000, 2 ** 20):
        rows.append([f"{n:,}", math.ceil(math.log2(n)), _fx(math.log2(n), 4)])
    return _table(["items n", "questions needed (ceil of log_2 n)", "log_2 n"], rows)


def eight_items_halving():
    """Identify the secret item among eight by always halving the candidates."""
    items = list("ABCDEFGH")
    secret = "F"
    cand = list(items)
    rows = []
    q = 0
    while len(cand) > 1:
        q += 1
        half = cand[: len(cand) // 2]
        other = cand[len(cand) // 2:]
        answer = "yes" if secret in half else "no"
        question = f"is it {half[0]}?" if len(half) == 1 else f"is it in {{{', '.join(half)}}}?"
        cand = half if answer == "yes" else other
        rows.append([q, question, answer, "{" + ", ".join(cand) + "}"])
    return _table(["question", "asked", "answer", "candidates left"], rows, "llll")


def skewed_guessing():
    """Asking the likeliest item first, when the items are not equally likely."""
    probs = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)]
    names = ["a", "b", "c", "d"]
    # Ask 'is it a?', then 'is it b?', then 'is it c?'; the third answer settles c versus d.
    asked = [1, 2, 3, 3]
    rows = [[n, str(p), k, _fx(float(p) * k, 3)] for n, p, k in zip(names, probs, asked)]
    expected = sum(float(p) * k for p, k in zip(probs, asked))
    rows.append(["expected", "1", "", _fx(expected, 3)])
    return _table(["item", "probability", "questions to identify it", "probability x questions"], rows)


def surprise_table():
    """Surprise -log p of events from certain to very rare."""
    rows = []
    for label, p in (("certain", 1.0), ("fair coin lands heads", 0.5), ("a die shows 1 or 2", 1 / 3),
                     ("a fair die shows 6", 1 / 6), ("a one-in-ten event", 0.1),
                     ("a one-in-a-hundred event", 0.01), ("a one-in-a-million event", 1e-6)):
        rows.append([label, _fx(p, 6).rstrip("0").rstrip("."), _fx(surprise(p, 2), 4), _fx(surprise(p, math.e), 4)])
    return _table(["event", "probability p", "surprise in bits, -log_2 p", "surprise in nats, -ln p"], rows)


def surprise_adds():
    """Independent events: probabilities multiply, surprises add."""
    p_heads, p_six = 0.5, 1 / 6
    both = p_heads * p_six
    rows = [
        ["coin lands heads", _fx(p_heads, 4), _fx(surprise(p_heads), 4)],
        ["die shows 6", _fx(p_six, 4), _fx(surprise(p_six), 4)],
        ["sum of the two surprises", "", _fx(surprise(p_heads) + surprise(p_six), 4)],
        ["both together (p = 1/12)", _fx(both, 4), _fx(surprise(both), 4)],
    ]
    return _table(["event", "probability", "surprise in bits"], rows)


def bits_and_nats():
    """The same quantity measured in bits and in nats."""
    ln2 = math.log(2)
    rows = [
        ["1 bit in nats", _fx(ln2, 4)],
        ["1 nat in bits", _fx(1 / ln2, 4)],
        ["a loss of 2.3026 nats in bits", _fx(2.3026 / ln2, 4)],
        ["a loss of 3.0 bits in nats", _fx(3.0 * ln2, 4)],
        ["error of reading nats as bits (1/ln 2 - 1)", _fx(1 / ln2 - 1, 4)],
    ]
    return _table(["quantity", "value"], rows, "lr")


# ------------------------------------------------------------------- entropy

def entropy_examples():
    """Entropy of some small distributions, in bits and in nats."""
    p_next = next_token_distribution()
    loaded = [0.5] + [0.1] * 5
    cases = [
        ("certain outcome (1, 0)", [1.0, 0.0]),
        ("fair coin", [0.5, 0.5]),
        ("coin with P(heads) = 0.9", [0.9, 0.1]),
        ("fair die", [1 / 6] * 6),
        ("die with P(6) = 0.5, others 0.1", loaded),
        ("eight equally likely items", [1 / 8] * 8),
        ("five-word next-token distribution", p_next),
    ]
    rows = []
    for name, p in cases:
        rows.append([name, len(p), _fx(entropy(p, 2), 4), _fx(entropy(p, math.e), 4),
                     _fx(math.log2(len(p)), 4)])
    return _table(["distribution", "outcomes", "H in bits", "H in nats", "log_2 of outcome count"], rows)


def binary_entropy_table():
    """Entropy of a coin as its bias moves from 0 to 1/2."""
    rows = []
    for p in (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        rows.append([_fx(p, 2), _fx(surprise(p), 4), _fx(surprise(1 - p), 4), _fx(entropy([p, 1 - p]), 4)])
    return _table(["P(heads) = p", "surprise of heads", "surprise of tails", "H(p) in bits"], rows)


UNIFORM_NAME = "information.uniform_is_largest"
GIBBS_NAME = "information.gibbs_random"


def uniform_is_largest():
    """Over four outcomes, entropy is largest when the outcomes are equally likely."""
    rows = []
    for name, p in (("uniform", [0.25] * 4),
                    ("(0.4, 0.3, 0.2, 0.1)", [0.4, 0.3, 0.2, 0.1]),
                    ("(0.7, 0.1, 0.1, 0.1)", [0.7, 0.1, 0.1, 0.1]),
                    ("(0.97, 0.01, 0.01, 0.01)", [0.97, 0.01, 0.01, 0.01]),
                    ("(0.25, 0.25, 0.5, 0)", [0.25, 0.25, 0.5, 0.0])):
        rows.append([name, _fx(entropy(p), 4)])
    rng = bookrng.generator(UNIFORM_NAME)
    best, total, n = 0.0, 0.0, 10000
    for _ in range(n):
        w = [-math.log(1.0 - rng.uniform()) for _ in range(4)]
        s = sum(w)
        h = entropy([x / s for x in w])
        best = max(best, h)
        total += h
    rows.append([f"best of {n:,} random distributions", _fx(best, 4)])
    rows.append([f"mean of those {n:,}", _fx(total / n, 4)])
    return _table(["distribution over 4 outcomes", "H in bits"], rows, "lr")


def next_token_entropy():
    """The same five words under three different contexts: confident, as before, undecided."""
    rows = []
    for name, z in NEXT_CONTEXTS:
        p = _softmax(z)
        rows.append([name, "(" + ", ".join(f"{v:g}" for v in z) + ")"] + [_fx(v, 5) for v in p]
                    + [_fx(entropy(p), 4)])
    rows.append(["uniform guess", "(0, 0, 0, 0, 0)"] + ["0.20000"] * 5 + [_fx(math.log2(5), 4)])
    return _table(["context", "scores z"] + NEXT_WORDS + ["H in bits"], rows, "llrrrrrr")


# ---------------------------------------------------------- prefix codes

def huffman_codes(probs, names):
    """Huffman's construction on exact fractions. Returns (codes, merge steps).

    Repeatedly merge the two least likely nodes; the first one popped becomes
    the 0 branch, the second the 1 branch. Ties are broken by creation order.
    """
    heap = [(Fraction(p), i, n) for i, (p, n) in enumerate(zip(probs, names))]
    heapq.heapify(heap)
    counter = len(heap)
    steps = []
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = (a[0] + b[0], counter, (a[2], b[2]))
        steps.append((a[0], b[0], merged[0]))
        counter += 1
        heapq.heappush(heap, merged)
    codes = {}

    def walk(node, prefix):
        if isinstance(node, tuple):
            walk(node[0], prefix + "0")
            walk(node[1], prefix + "1")
        else:
            codes[node] = prefix or "0"

    walk(heap[0][2], "")
    return codes, steps


def code_dyadic():
    """When every probability is a power of 1/2, the code length is exactly -log_2 p."""
    names = ["a", "b", "c", "d"]
    probs = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)]
    codes, _ = huffman_codes(probs, names)
    rows, avg = [], 0.0
    for n, p in zip(names, probs):
        length = len(codes[n])
        avg += float(p) * length
        rows.append([n, str(p), codes[n], length, _fx(surprise(float(p)), 3), _fx(float(p) * length, 3)])
    rows.append(["average", "1", "", "", "", _fx(avg, 3)])
    h = entropy([float(p) for p in probs])
    return _table(["symbol", "probability", "code", "length", "-log_2 p", "p x length"], rows, "lrlrrr") \
        + f"\n\nEntropy H = {_fx(h, 3)} bits; average code length = {_fx(avg, 3)} bits."


def code_weather():
    """Huffman's construction for (0.4, 0.3, 0.2, 0.1): the merges, the code, its average length."""
    codes, steps = huffman_codes(WEATHER_P, WEATHER)
    merge_rows = [[i + 1, _fx(float(a), 1), _fx(float(b), 1), _fx(float(s), 1)] for i, (a, b, s) in enumerate(steps)]
    out = _table(["merge", "least likely", "next least likely", "new node"], merge_rows)
    rows, avg = [], 0.0
    for n, p in zip(WEATHER, WEATHER_P):
        length = len(codes[n])
        avg += float(p) * length
        rows.append([n, _fx(float(p), 1), codes[n], length, _fx(surprise(float(p)), 3), _fx(float(p) * length, 2)])
    rows.append(["average", "1.0", "", "", "", _fx(avg, 2)])
    h = entropy([float(p) for p in WEATHER_P])
    kraft = sum(Fraction(1, 2 ** len(c)) for c in codes.values())
    out += "\n\n" + _table(["symbol", "probability", "code", "length", "-log_2 p", "p x length"], rows, "lrlrrr")
    out += (f"\n\nEntropy H = {_fx(h, 4)} bits; average code length = {_fx(avg, 2)} bits; "
            f"fixed 2-bit code = 2.00 bits; sum of 2^-length = {kraft}.")
    return out


def encode_message():
    """Ten days of weather, encoded with the code above and decoded back."""
    codes, _ = huffman_codes(WEATHER_P, WEATHER)
    message = ["sun"] * 4 + ["cloud"] * 3 + ["rain"] * 2 + ["snow"]
    bits = "".join(codes[s] for s in message)
    inverse = {c: s for s, c in codes.items()}
    decoded, cur = [], ""
    for b in bits:
        cur += b
        if cur in inverse:
            decoded.append(inverse[cur])
            cur = ""
    rows = [
        ["message", " ".join(message)],
        ["encoded", bits],
        ["bits used", f"{len(bits)} (fixed 2-bit code: {2 * len(message)})"],
        ["decoded equals message", str(decoded == message and cur == "")],
    ]
    return _table(["", ""], rows, "ll")


# ------------------------------------------------- cross-entropy and KL

CE_P = [0.4, 0.3, 0.2, 0.1]
CE_MODELS = [
    ("q = p (the right belief)", [0.4, 0.3, 0.2, 0.1]),
    ("q uniform", [0.25] * 4),
    ("q = (0.7, 0.1, 0.1, 0.1)", [0.7, 0.1, 0.1, 0.1]),
    ("q reversed, (0.1, 0.2, 0.3, 0.4)", [0.1, 0.2, 0.3, 0.4]),
]


def cross_entropy_breakdown():
    """Per-symbol cost of coding weather drawn from p with a code designed for q = reversed p."""
    q = CE_MODELS[3][1]
    rows = []
    for n, pi, qi in zip(WEATHER, CE_P, q):
        rows.append([n, _fx(pi, 1), _fx(qi, 1), _fx(surprise(qi), 4), _fx(pi * surprise(qi), 4),
                     _fx(surprise(pi), 4), _fx(pi * surprise(pi), 4)])
    rows.append(["total", "1.0", "1.0", "", _fx(cross_entropy(CE_P, q), 4), "", _fx(entropy(CE_P), 4)])
    return _table(["symbol", "p", "q", "-log_2 q", "p x (-log_2 q)", "-log_2 p", "p x (-log_2 p)"], rows)


def cross_entropy_models():
    """H(p, q) for several q, with the excess over H(p) and the same excess computed as sum p log(p/q)."""
    h = entropy(CE_P)
    rows = []
    for name, q in CE_MODELS:
        ce = cross_entropy(CE_P, q)
        rows.append([name, _fx(ce, 4), _fx(h, 4), _fx(ce - h, 4), _fx(kl_divergence(CE_P, q), 4)])
    return _table(["belief q", "H(p, q)", "H(p)", "H(p, q) - H(p)", "KL(p || q) from its formula"], rows)


def log_below_tangent():
    """ln x <= x - 1, with equality only at x = 1."""
    rows = []
    for x in (0.1, 0.25, 0.5, 0.9, 1.0, 1.1, 2.0, 4.0, 10.0):
        rows.append([_fx(x, 2), _fx(math.log(x), 4), _fx(x - 1, 4), _fx(x - 1 - math.log(x), 4)])
    return _table(["x", "ln x", "x - 1", "(x - 1) - ln x"], rows)


def gibbs_random():
    """KL(p || q) over many random pairs of distributions on five outcomes is never negative."""
    rng = bookrng.generator(GIBBS_NAME)

    def draw(sharp):
        w = [(-math.log(1.0 - rng.uniform())) ** sharp for _ in range(5)]
        s = sum(w)
        return [x / s for x in w]

    rows = []
    for sharp in (1.0, 2.0, 4.0):
        vals = []
        for _ in range(5000):
            p, q = draw(sharp), draw(sharp)
            vals.append(kl_divergence(p, q))
        rows.append([sharp, len(vals), _fx(min(vals), 4), _fx(sum(vals) / len(vals), 4), _fx(max(vals), 4),
                     sum(1 for v in vals if v < 0)])
    p = draw(1.0)
    return _table(["sharpening exponent", "pairs", "smallest KL", "mean KL", "largest KL", "negative values"], rows) \
        + f"\n\nKL(p || p) for one random p: {_fx(kl_divergence(p, p), 4)}."


def kl_asymmetry():
    """The two directions of KL for a peaked P and a uniform Q, and for a case with a zero."""
    rows = []
    for name, p, q in (
        ("P = (0.97, 0.01, 0.01, 0.01), Q uniform", [0.97, 0.01, 0.01, 0.01], [0.25] * 4),
        ("P = (0.5, 0.5, 0), Q = (0.5, 0.25, 0.25)", [0.5, 0.5, 0.0], [0.5, 0.25, 0.25]),
    ):
        rows.append([name, _fx(kl_divergence(p, q), 4), _fx(kl_divergence(q, p), 4)])
    return _table(["pair", "KL(P || Q) in bits", "KL(Q || P) in bits"], rows, "lrr")


KL_P = [0.97, 0.01, 0.01, 0.01]
KL_Q = [0.25] * 4


def entropy_below_log_n():
    """H(p) = log_2 n - KL(p || uniform): the entropy gap to the maximum is a divergence."""
    rows = []
    for name, p in (("(0.4, 0.3, 0.2, 0.1)", [0.4, 0.3, 0.2, 0.1]),
                    ("(0.7, 0.1, 0.1, 0.1)", [0.7, 0.1, 0.1, 0.1]),
                    ("uniform", [0.25] * 4)):
        d = kl_divergence(p, [0.25] * 4)
        rows.append([name, _fx(entropy(p), 4), _fx(d, 4), _fx(2 - d, 4)])
    return _table(["p", "H(p)", "KL(p || uniform)", "2 - KL(p || uniform)"], rows)


# --------------------------------------------- the loss on observed labels

LABEL_PROBS = [  # model output for five examples over three classes
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.3, 0.3, 0.4],
    [0.05, 0.05, 0.9],
    [0.6, 0.3, 0.1],
]
LABELS = [0, 1, 2, 2, 1]


def label_cross_entropy():
    """The average negative log-probability of the observed labels, and the same number as H(p_data, q)."""
    rows = []
    losses = []
    for i, (q, y) in enumerate(zip(LABEL_PROBS, LABELS)):
        onehot = [1.0 if k == y else 0.0 for k in range(3)]
        loss = -math.log(q[y])
        assert abs(loss - cross_entropy(onehot, q, math.e)) < 1e-12
        losses.append(loss)
        rows.append([i + 1, y, "(" + ", ".join(_fx(v, 2) for v in q) + ")", _fx(q[y], 2), _fx(loss, 4)])
    mean = sum(losses) / len(losses)
    out = _table(["example", "label", "model probabilities", "probability of the label", "-ln p(label)"], rows)
    out += f"\n\nMean over the five examples: {_fx(mean, 4)} nats = {_fx(mean / math.log(2), 4)} bits."
    return out


def label_cross_entropy_sensitivity():
    """Which examples dominate the average: the cost of probability given to the true label."""
    rows = []
    for p in (0.9, 0.5, 0.1, 0.01, 0.001):
        rows.append([_fx(p, 3), _fx(-math.log(p), 4), _fx(surprise(p), 4)])
    return _table(["probability of the true label", "-ln p in nats", "-log_2 p in bits"], rows)


# ----------------------------------------------------------------- perplexity

def perplexity_uniform():
    """A model that spreads its belief evenly over k choices has perplexity k."""
    rows = []
    for k in (1, 2, 6, 50000):
        ce = math.log(k)
        rows.append([f"{k:,}", _fx(ce, 4), _fx(ce / math.log(2), 4), _fx(math.exp(ce), 2)])
    return _table(["k equally likely tokens", "cross-entropy in nats", "in bits", "perplexity exp(CE)"], rows)


def perplexity_sentence():
    """Perplexity of a six-token sentence from the probabilities a model gave each token."""
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    probs = [0.20, 0.05, 0.10, 0.50, 0.40, 0.25]
    nll = [-math.log(p) for p in probs]
    rows = [[t, _fx(p, 2), _fx(v, 4)] for t, p, v in zip(tokens, probs, nll)]
    mean = sum(nll) / len(nll)
    ppl = math.exp(mean)
    prod = math.prod(probs)
    out = _table(["token", "probability the model gave it", "-ln p"], rows)
    out += (f"\n\nMean -ln p = {_fx(mean, 4)} nats; perplexity exp(mean) = {_fx(ppl, 4)}.\n"
            f"Probability of the whole sentence = {prod:.3e}; (1 / probability)^(1/6) = {_fx(prod ** (-1 / 6), 4)}.\n"
            f"In bits: mean = {_fx(mean / math.log(2), 4)}, 2^mean = {_fx(2 ** (mean / math.log(2)), 4)}.")
    return out


def perplexity_effective_choices():
    """exp(H) of several distributions: the number of equally likely choices that matches its uncertainty."""
    cases = [
        ("fair die", [1 / 6] * 6),
        ("coin with P(heads) = 0.9", [0.9, 0.1]),
        ("die with P(6) = 0.5, others 0.1", [0.5] + [0.1] * 5),
        ("five-word next-token distribution", next_token_distribution()),
        ("confident context", _softmax(NEXT_CONTEXTS[0][1])),
    ]
    rows = []
    for name, p in cases:
        rows.append([name, len(p), _fx(entropy(p, math.e), 4), _fx(math.exp(entropy(p, math.e)), 3)])
    return _table(["distribution", "outcomes", "H in nats", "exp(H) = effective choices"], rows)


def perplexity_of_a_wrong_model():
    """Perplexity of cross-entropy H(p, q): data from the weather distribution, four beliefs."""
    rows = []
    for name, q in CE_MODELS:
        ce = cross_entropy(CE_P, q, math.e)
        rows.append([name, _fx(ce, 4), _fx(math.exp(ce), 3)])
    return _table(["belief q", "H(p, q) in nats", "perplexity"], rows)


# -------------------------------------------------------- mutual information

def _spam_joint():
    p = pr.spam_joint_probabilities()
    return {k: float(v) for k, v in p.items()}


def mutual_information_table():
    """I(verdict; kind) from the spam table: sum of p(x, y) log2 [p(x, y) / (p(x) p(y))]."""
    p = _spam_joint()
    px = {v: sum(p[(v, k)] for k in pr.KINDS) for v in pr.VERDICTS}
    py = {k: sum(p[(v, k)] for v in pr.VERDICTS) for k in pr.KINDS}
    rows, total = [], 0.0
    for v in pr.VERDICTS:
        for k in pr.KINDS:
            ratio = p[(v, k)] / (px[v] * py[k])
            term = p[(v, k)] * math.log2(ratio)
            total += term
            rows.append([f"{v}, {k}", _fx(p[(v, k)], 3), _fx(px[v] * py[k], 3), _fx(ratio, 3),
                         _fx(math.log2(ratio), 3), _fx(term, 4)])
    rows.append(["I(verdict; kind)", "", "", "", "", _fx(total, 4)])
    return _table(["cell", "p(x, y)", "p(x) p(y)", "ratio", "log_2 ratio", "p(x, y) x log_2 ratio"], rows)


def mutual_information_entropies():
    """The same number as a reduction of entropy: H(kind) - H(kind given verdict)."""
    p = _spam_joint()
    px = {v: sum(p[(v, k)] for k in pr.KINDS) for v in pr.VERDICTS}
    py = {k: sum(p[(v, k)] for v in pr.VERDICTS) for k in pr.KINDS}
    h_kind = entropy([py[k] for k in pr.KINDS])
    rows = [["before seeing the verdict", "-", "-", _fx(h_kind, 4)]]
    h_cond = 0.0
    for v in pr.VERDICTS:
        cond = [p[(v, k)] / px[v] for k in pr.KINDS]
        hv = entropy(cond)
        h_cond += px[v] * hv
        rows.append([f"verdict = {v}", _fx(px[v], 3), "(" + ", ".join(_fx(c, 3) for c in cond) + ")", _fx(hv, 4)])
    rows.append(["average after the verdict", "1.000", "", _fx(h_cond, 4)])
    h_joint = entropy(list(p.values()))
    h_verdict = entropy([px[v] for v in pr.VERDICTS])
    out = _table(["situation", "probability", "(P(spam), P(ham)) there", "entropy of kind, bits"], rows)
    out += (f"\n\nReduction: {_fx(h_kind, 4)} - {_fx(h_cond, 4)} = {_fx(h_kind - h_cond, 4)} bits.\n"
            f"Joint entropy H(verdict, kind) = {_fx(h_joint, 4)}; H(verdict) + H(kind) - H(verdict, kind) = "
            f"{_fx(h_verdict, 4)} + {_fx(h_kind, 4)} - {_fx(h_joint, 4)} = {_fx(h_verdict + h_kind - h_joint, 4)} bits.")
    return out


def mutual_information_extremes():
    """Mutual information of an independent pair, a perfect copy, and a half-informative pair."""
    def mi(joint):
        px = [sum(r) for r in joint]
        py = [sum(c) for c in zip(*joint)]
        return sum(joint[i][j] * math.log2(joint[i][j] / (px[i] * py[j]))
                   for i in range(len(joint)) for j in range(len(joint[0])) if joint[i][j] > 0)

    cases = [
        ("independent fair bits", [[0.25, 0.25], [0.25, 0.25]]),
        ("a perfect copy of a fair bit", [[0.5, 0.0], [0.0, 0.5]]),
        ("a copy that is wrong 10% of the time", [[0.45, 0.05], [0.05, 0.45]]),
        ("a perfect copy of a 20/80 bit", [[0.2, 0.0], [0.0, 0.8]]),
    ]
    rows = []
    for name, j in cases:
        px = [sum(r) for r in j]
        rows.append([name, _fx(entropy(px), 4), _fx(mi(j), 4)])
    return _table(["pair (X, Y)", "H(X)", "I(X; Y)"], rows)
