# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Language Modeling": predicting the next token.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own
description and independent of any engine code. Each public function without a leading underscore that takes no
arguments returns a markdown string that the chapter embeds between `<!-- worked: language_modeling.<name> -->`
markers; the helpers with arguments or a leading underscore are shared with the chapter's figures, so a picture and
the table beside it cannot disagree.

* a toy language with a long-range rule (the verb agrees with the subject, not with the noun nearest to it), a
  seeded training, development and test corpus, and the exact probability and entropy of the process behind them;
* the chain rule on a four-token sentence, multiplied out two ways;
* n-gram models of order 1 to 5 by counting, their maximum-likelihood probabilities, the zero-count problem, add-alpha
  smoothing and interpolation with back-off, perplexity on held-out text, and the sparsity of n-grams;
* text sampled from the models with the book's generator;
* a neural next-token model (embeddings of a window of four tokens, one tanh layer, a softmax over the tokens)
  trained by full-batch gradient descent, compared with the counting models, its work counted in FLOPs;
* teacher forcing as a table of training pairs, the shift of the targets by one, and the bug of not shifting.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed.

Swedish Embedded AB implements solutions for building and evaluating language models from first principles. If your
team needs expertise in next-token training objectives, perplexity and the data pipelines behind them, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math
from fractions import Fraction
from functools import lru_cache

import bookrng
import stable

# --------------------------------------------------------------------------------------------- formatting


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(x, nd=4):
    if x == math.inf:
        return "inf"
    s = f"{x:.{nd}f}"
    return f"{0.0:.{nd}f}" if float(s) == 0 else s


def _int(n):
    return f"{n:,}"


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


# --------------------------------------------------------------------------------------------- the toy language

# Ids 0 to 17 are the tokens a model predicts; id 18, `<s>`, is only ever read (it pads the start of a sentence).
TOKENS = ["the", "cat", "dog", "fox", "cats", "dogs", "foxes", "near", "sees", "chases", "see", "chase",
          "big", "small", "ball", "bone", "fish", ".", "<s>"]
ID = {t: i for i, t in enumerate(TOKENS)}
V = 18            # tokens a model predicts (ids 0 to 17)
V_IN = 19         # tokens a model reads (adds `<s>`)
BOS = 18
NOUNS = ["cat", "dog", "fox", "cats", "dogs", "foxes"]
VERBS = [("sees", "chases"), ("see", "chase")]    # singular forms, plural forms
ADJS = [None, "big", "small"]
OBJECTS = ["ball", "bone", "fish"]
CTX = 4           # the longest context any model here reads: four tokens


def _sentence(rng):
    """One sentence of the toy language as a list of token strings, ending in `.`.

    Draw order: subject = integer(6); has_pp = integer(2); if has_pp, attractor = integer(6); verb = integer(2);
    adjective = integer(3); object = integer(3).
    """
    subj = rng.integer(6)
    out = ["the", NOUNS[subj]]
    if rng.integer(2) == 1:
        out += ["near", "the", NOUNS[rng.integer(6)]]
    out.append(VERBS[0 if subj < 3 else 1][rng.integer(2)])
    adj = ADJS[rng.integer(3)]
    if adj:
        out.append(adj)
    out.append(OBJECTS[rng.integer(3)])
    out.append(".")
    return out


def _corpus(name, n):
    rng = bookrng.generator(name)
    return [[ID[t] for t in _sentence(rng)] for _ in range(n)]


NAME_TRAIN, NAME_DEV, NAME_TEST = "language_modeling.train", "language_modeling.dev", "language_modeling.test"
N_TRAIN, N_DEV, N_TEST = 40, 20, 30


@lru_cache(maxsize=None)
def train_set():
    return _corpus(NAME_TRAIN, N_TRAIN)


@lru_cache(maxsize=None)
def dev_set():
    return _corpus(NAME_DEV, N_DEV)


@lru_cache(maxsize=None)
def test_set():
    return _corpus(NAME_TEST, N_TEST)


def _words(sentence):
    return " ".join(TOKENS[i] for i in sentence)


def _positions(sentence, ctx=CTX):
    """The training pairs of one sentence: (window of the `ctx` tokens before the target, target id), position by
    position, the start padded with `<s>`."""
    padded = [BOS] * ctx + sentence
    return [(tuple(padded[j:j + ctx]), padded[j + ctx]) for j in range(len(sentence))]


def _n_predicted(sentences):
    return sum(len(s) for s in sentences)


def _grammar_accepts(sentence):
    """Is `sentence` (token ids) a sentence of the toy language, with the verb agreeing with the subject?"""
    w = [TOKENS[i] for i in sentence]
    if len(w) < 5 or w[0] != "the" or w[1] not in NOUNS or w[-1] != "." or w[-2] not in OBJECTS:
        return False
    subj = NOUNS.index(w[1])
    k = 2
    if w[k:k + 1] == ["near"]:
        if w[k + 1:k + 2] != ["the"] or w[k + 2:k + 3] == [] or w[k + 2] not in NOUNS:
            return False
        k += 3
    if k >= len(w) or w[k] not in VERBS[0 if subj < 3 else 1]:
        return False
    k += 1
    rest = w[k:-2]
    return rest in ([], ["big"], ["small"])


@lru_cache(maxsize=None)
def _all_sentences():
    """Every sentence of the language with its exact probability (Fractions), by enumeration: 756 sentences."""
    out = []
    for s in range(6):
        for pp in range(1 + 6):                 # 0 = no phrase, 1..6 = the phrase with attractor noun pp - 1
            for v in range(2):
                for a in range(3):
                    for o in range(3):
                        w = ["the", NOUNS[s]]
                        p = Fraction(1, 6)
                        if pp == 0:
                            p *= Fraction(1, 2)
                        else:
                            w += ["near", "the", NOUNS[pp - 1]]
                            p *= Fraction(1, 2) * Fraction(1, 6)
                        w.append(VERBS[0 if s < 3 else 1][v])
                        if ADJS[a]:
                            w.append(ADJS[a])
                        w += [OBJECTS[o], "."]
                        p *= Fraction(1, 2) * Fraction(1, 3) * Fraction(1, 3)
                        out.append(([ID[t] for t in w], p))
    return out


@lru_cache(maxsize=None)
def _process_stats():
    """Exact entropy per sentence (nats), mean tokens per sentence, and their ratio, the loss floor per token."""
    sents = _all_sentences()
    assert sum(p for _, p in sents) == 1
    h = -sum(float(p) * math.log(float(p)) for _, p in sents)
    length = sum(float(p) * len(s) for s, p in sents)
    return h, length, h / length


@lru_cache(maxsize=None)
def _prob_table():
    return {tuple(s): p for s, p in _all_sentences()}


def _true_prob(sentence):
    """Probability of a sentence under the process itself (Fraction); 0 if it is not a sentence of the language."""
    return _prob_table().get(tuple(sentence), Fraction(0))


def corpus():
    h, length, floor = _process_stats()
    lines = [
        _seed_line(NAME_TRAIN) + f" Dev: seed {bookrng.seed_of(NAME_DEV):#018x} (`{NAME_DEV}`), test: seed "
        f"{bookrng.seed_of(NAME_TEST):#018x} (`{NAME_TEST}`).",
        "",
        "The 18 tokens a model predicts have ids 0 to 17 in this order: " + ", ".join(
            f"{i} `{t}`" for i, t in enumerate(TOKENS[:V])) + ". Id 18, `<s>`, only ever pads the start of a sentence "
        "and is never predicted.",
        "A sentence is `the`, a subject noun, optionally `near the` and a second noun, a verb, optionally `big` or "
        "`small`, an object noun and `.`. Draw order for each sentence: `integer(6)` picks the subject (cat, dog, fox, "
        "cats, dogs, foxes); `integer(2)` is 1 when the phrase `near the <noun>` is present, and then `integer(6)` "
        "picks its noun; `integer(2)` picks the verb among the two that agree in number with the *subject* "
        "(`sees`/`chases` for the first three nouns, `see`/`chase` for the others); `integer(3)` picks no adjective, "
        "`big` or `small`; `integer(3)` picks the object (ball, bone, fish). The training, development and test "
        f"sets are {N_TRAIN}, {N_DEV} and {N_TEST} sentences from three generators.",
        "",
        f"The training corpus ({_int(_n_predicted(train_set()))} tokens to predict, one `.` ending each sentence):",
        "",
    ]
    for i, s in enumerate(train_set()):
        lines.append(f"    {i + 1:2d}. {_words(s)}")
    lines += [
        "",
        f"There are 756 sentences in the language. Enumerating them gives the process's own numbers: entropy "
        f"{_f(h)} nats per sentence, {_f(length)} tokens per sentence on average, so no model can have a mean loss "
        f"below {_f(h)} / {_f(length)} = {_f(floor)} nats per token on text from this process in the long run "
        f"(perplexity {_f(math.exp(floor))}). The training set has {_f(_n_predicted(train_set()) / N_TRAIN, 3)} tokens "
        f"per sentence, the development set {_f(_n_predicted(dev_set()) / N_DEV, 3)}, the test set "
        f"{_f(_n_predicted(test_set()) / N_TEST, 3)}.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- n-gram models


class NGram:
    """Counts of the training positions: `c[m][(context of m tokens, target)]` and `ctx[m][context of m tokens]`
    for every context length m = 0 .. CTX. The context is the last m tokens before the target, with `<s>` before
    the start of a sentence."""

    def __init__(self, sentences):
        self.joint = [dict() for _ in range(CTX + 1)]
        self.ctx = [dict() for _ in range(CTX + 1)]
        for s in sentences:
            for window, y in _positions(s):
                for m in range(CTX + 1):
                    h = window[CTX - m:]
                    self.joint[m][(h, y)] = self.joint[m].get((h, y), 0) + 1
                    self.ctx[m][h] = self.ctx[m].get(h, 0) + 1

    def c(self, h, y):
        return self.joint[len(h)].get((h, y), 0)

    def c_ctx(self, h):
        return self.ctx[len(h)].get(h, 0)


@lru_cache(maxsize=None)
def ngram_train():
    return NGram(train_set())


def p_mle(model, h, y):
    n = model.c_ctx(h)
    return model.c(h, y) / n if n else 0.0


def p_addalpha(model, h, y, alpha):
    return (model.c(h, y) + alpha) / (model.c_ctx(h) + alpha * V)


EPS_UNIFORM = 0.01


def p_interp(model, h, y, ratio):
    """Interpolation with back-off. h has m tokens (the model's order is m + 1). For each context length j = 0..m
    whose context has been seen, the component is the maximum-likelihood estimate with the last j tokens of h; the
    components get weights ratio^j normalised over those present (an unseen context hands its weight to the
    others); the result is mixed with the uniform 1/V at weight EPS_UNIFORM."""
    num = den = 0.0
    for j in range(len(h) + 1):
        hj = h[len(h) - j:]
        if model.c_ctx(hj):
            w = ratio ** j
            num += w * model.c(hj, y) / model.c_ctx(hj)
            den += w
    return EPS_UNIFORM / V + (1 - EPS_UNIFORM) * num / den


def _hist(window, order):
    return tuple(window[CTX - (order - 1):]) if order > 1 else ()


def mean_nll(prob, sentences, order):
    """Mean -ln p over every predicted token of the sentences (inf if any has probability 0)."""
    tot = 0.0
    n = 0
    for s in sentences:
        for window, y in _positions(s):
            p = prob(_hist(window, order), y)
            if p <= 0:
                return math.inf
            tot -= math.log(p)
            n += 1
    return tot / n


def ppl(nll):
    return math.exp(nll) if nll != math.inf else math.inf


# --------------------------------------------------------------------------------------------- the neural model

D_EMB, H_HID = 6, 12
MLP_NAME = "language_modeling.mlp"


class Mlp:
    """Embeddings of the last CTX tokens (table E of shape [19, D]), concatenated, one tanh layer, a softmax over the
    18 predicted tokens. Parameters are kept as flat lists of rows."""

    def __init__(self, rng):
        self.E = [[0.5 * rng.normal() for _ in range(D_EMB)] for _ in range(V_IN)]
        fan = CTX * D_EMB
        self.W1 = [[rng.normal() / math.sqrt(fan) for _ in range(fan)] for _ in range(H_HID)]
        self.b1 = [0.0] * H_HID
        self.W2 = [[rng.normal() / math.sqrt(H_HID) for _ in range(H_HID)] for _ in range(V)]
        self.b2 = [0.0] * V

    def n_params(self):
        return V_IN * D_EMB + H_HID * CTX * D_EMB + H_HID + V * H_HID + V

    def forward(self, window):
        x = [v for tok in window for v in self.E[tok]]
        a = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        z = [sum(w * ai for w, ai in zip(row, a)) + b for row, b in zip(self.W2, self.b2)]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return x, a, [v / s for v in e]

    def probs(self, window):
        return self.forward(window)[2]


def _batch(sentences):
    return [pair for s in sentences for pair in _positions(s)]


def _mlp_loss(model, pairs):
    return -sum(math.log(model.probs(w)[y]) for w, y in pairs) / len(pairs)


def _mlp_step(model, pairs, lr):
    """One full-batch gradient step on the mean cross-entropy. Returns the loss before the step."""
    n = len(pairs)
    gE = [[0.0] * D_EMB for _ in range(V_IN)]
    gW1 = [[0.0] * (CTX * D_EMB) for _ in range(H_HID)]
    gb1 = [0.0] * H_HID
    gW2 = [[0.0] * H_HID for _ in range(V)]
    gb2 = [0.0] * V
    loss = 0.0
    for window, y in pairs:
        x, a, p = model.forward(window)
        loss -= math.log(p[y])
        dz = [pj / n for pj in p]
        dz[y] -= 1.0 / n
        da = [0.0] * H_HID
        for j in range(V):
            row, g = model.W2[j], gW2[j]
            d = dz[j]
            gb2[j] += d
            for i in range(H_HID):
                g[i] += d * a[i]
                da[i] += d * row[i]
        dx = [0.0] * (CTX * D_EMB)
        for i in range(H_HID):
            dpre = da[i] * (1.0 - a[i] * a[i])
            gb1[i] += dpre
            row, g = model.W1[i], gW1[i]
            for k in range(CTX * D_EMB):
                g[k] += dpre * x[k]
                dx[k] += dpre * row[k]
        for pos, tok in enumerate(window):
            ge = gE[tok]
            for c in range(D_EMB):
                ge[c] += dx[pos * D_EMB + c]
    for tab, g in ((model.E, gE), (model.W1, gW1), (model.W2, gW2)):
        for r, gr in zip(tab, g):
            for c in range(len(r)):
                r[c] -= lr * gr[c]
    for vec, g in ((model.b1, gb1), (model.b2, gb2)):
        for c in range(len(vec)):
            vec[c] -= lr * g[c]
    return loss / n


MLP_STEPS, MLP_LR, MLP_EVERY = 500, 1.0, 10


def _copy_mlp(m):
    c = Mlp.__new__(Mlp)
    c.E = [r[:] for r in m.E]
    c.W1 = [r[:] for r in m.W1]
    c.b1 = m.b1[:]
    c.W2 = [r[:] for r in m.W2]
    c.b2 = m.b2[:]
    return c


@lru_cache(maxsize=None)
def train_mlp(n_sentences=N_TRAIN, nudge=0.0):
    """Train on the first `n_sentences` sentences of the training generator, full batch, MLP_STEPS steps of plain
    gradient descent at MLP_LR, recording train and dev loss every MLP_EVERY steps; keep the parameters of the
    recorded step with the lowest development loss (early stopping). `nudge` is added to the first embedding weight: the
    test of chaos of `stable.text`."""
    model = Mlp(bookrng.generator(MLP_NAME))
    model.E[0][0] += nudge
    tr = _batch(_corpus(NAME_TRAIN, n_sentences))
    dv = _batch(dev_set())
    curve, best = [], (math.inf, 0, None)
    for step in range(MLP_STEPS + 1):
        if step % MLP_EVERY == 0:
            lt, ld = _mlp_loss(model, tr), _mlp_loss(model, dv)
            curve.append((step, lt, ld))
            if ld < best[0]:
                best = (ld, step, _copy_mlp(model))
        if step < MLP_STEPS:
            _mlp_step(model, tr, MLP_LR)
    return {"model": best[2], "step": best[1], "curve": curve, "n_pairs": len(tr)}


def _mlp_prob(model):
    return lambda window, y: model.probs(window)[y]


def _mlp_test_nll(model, sentences):
    pairs = _batch(sentences)
    return _mlp_loss(model, pairs)


# --------------------------------------------------------------------------------------------- model selection

ALPHA_GRID = (1.0, 0.3, 0.1, 0.03, 0.01)
RATIO_GRID = (1.0, 2.0, 4.0, 8.0)


@lru_cache(maxsize=None)
def ngram_for(n_sentences):
    return NGram(_corpus(NAME_TRAIN, n_sentences))


@lru_cache(maxsize=None)
def best_smoothing(n_sentences, order):
    """(alpha, ratio) chosen on the development set for an n-gram model of this order trained on n_sentences."""
    m = ngram_for(n_sentences)
    dev = dev_set()
    alpha = min(ALPHA_GRID, key=lambda a: mean_nll(lambda h, y: p_addalpha(m, h, y, a), dev, order))
    r = min(RATIO_GRID, key=lambda r: mean_nll(lambda h, y: p_interp(m, h, y, r), dev, order))
    return alpha, r


def ngram_test_ppls(n_sentences, order):
    """Test perplexity of the order-`order` model trained on n_sentences: (MLE, add-alpha, interpolated)."""
    m = ngram_for(n_sentences)
    alpha, r = best_smoothing(n_sentences, order)
    t = test_set()
    return (ppl(mean_nll(lambda h, y: p_mle(m, h, y), t, order)),
            ppl(mean_nll(lambda h, y: p_addalpha(m, h, y, alpha), t, order)),
            ppl(mean_nll(lambda h, y: p_interp(m, h, y, r), t, order)))


def ngram_test_ppls_dev_row(n_sentences, order):
    m = ngram_for(n_sentences)
    alpha, r = best_smoothing(n_sentences, order)
    d = dev_set()
    return (ppl(mean_nll(lambda h, y: p_mle(m, h, y), d, order)),
            ppl(mean_nll(lambda h, y: p_addalpha(m, h, y, alpha), d, order)),
            ppl(mean_nll(lambda h, y: p_interp(m, h, y, r), d, order)))


def mlp_ppls(n_sentences):
    """(dev, test) perplexity of the early-stopped network trained on n_sentences."""
    m = train_mlp(n_sentences)["model"]
    return ppl(_mlp_loss(m, _batch(dev_set()))), ppl(_mlp_loss(m, _batch(test_set())))


def ppl_by_order_data():
    """For the figure: test perplexity against n for the add-alpha and interpolated models trained on the printed
    corpus, the network's test perplexity, and the process floor."""
    rows = [ngram_test_ppls(N_TRAIN, n) for n in range(1, CTX + 2)]
    return {"orders": list(range(1, CTX + 2)), "mle": [r[0] for r in rows], "addk": [r[1] for r in rows],
            "interp": [r[2] for r in rows], "mlp": mlp_ppls(N_TRAIN)[1], "floor": math.exp(_process_stats()[2])}


# --------------------------------------------------------------------------------------------- blocks: the chain rule


def _prefix_prob(prefix):
    """Probability that a sentence of the language starts with `prefix` (a tuple of ids), exactly."""
    n = len(prefix)
    return sum((p for s, p in _all_sentences() if tuple(s[:n]) == tuple(prefix)), Fraction(0))


def chain_rule():
    sent = [ID[t] for t in ["the", "cat", "sees", "fish", "."]]
    rows, running, logsum = [], Fraction(1), 0.0
    for t in range(len(sent)):
        cond = _prefix_prob(sent[:t + 1]) / _prefix_prob(sent[:t])
        running *= cond
        logsum -= math.log(float(cond))
        rows.append([t + 1, f"`{TOKENS[sent[t]]}`", "`" + " ".join(TOKENS[i] for i in sent[:t]) + "`" if t else "(nothing)",
                     str(cond), str(running)])
    direct = Fraction(1, 6) * Fraction(1, 2) * Fraction(1, 2) * Fraction(1, 3) * Fraction(1, 3)
    return "\n".join([
        "Probabilities are exact fractions. The probability of a prefix is the sum of the probabilities of the "
        "756 sentences that begin with it; the conditional is the probability of the longer prefix divided by "
        "that of the shorter.",
        "",
        _table(["position", "token", "given", "p(token \\| given)", "product so far"], rows, "rrlrr"),
        "",
        f"The four tokens `the cat sees fish` have probability {_prefix_prob(sent[:4])}, and adding `.` costs a factor "
        f"{_prefix_prob(sent) / _prefix_prob(sent[:4])}. From the generating recipe directly, subject cat (1/6), no "
        f"phrase (1/2), verb sees (1/2), no adjective (1/3), object fish (1/3): {direct}. "
        f"The same number: {direct == _true_prob(sent)}. Sum of -ln p over the five factors: {_f(logsum)} "
        f"= ln {int(1 / direct)}.",
    ])


# --------------------------------------------------------------------------------------------- blocks: counting


def _ctx_str(h):
    return "`" + " ".join(TOKENS[i] for i in h) + "`" if h else "(empty)"


def ngram_counts():
    m = ngram_train()
    n_pred = _n_predicted(train_set())
    uni = [[f"`{TOKENS[i]}`", m.c((), i), _f(m.c((), i) / n_pred)] for i in range(V)]
    out = [
        f"Counts over the {n_pred} predicted positions of the training corpus. The unigram count of a token is how "
        f"often it is the target; its estimate is that count divided by {n_pred}.",
        "",
        _table(["token", "count", "estimate"], uni),
        "",
        "The estimates for what follows contexts that end in `cat`, with one, two, three and four tokens of "
        "context (`<s>` pads the start; every `cat` here is preceded by `the`):",
        "",
    ]
    conts = [ID[t] for t in ["near", "sees", "chases", "see", "chase"]]
    rows = []
    ctxs = [(ID["cat"],), (ID["the"], ID["cat"]), (ID["near"], ID["the"], ID["cat"]),
            (ID["foxes"], ID["near"], ID["the"], ID["cat"])]
    for h in ctxs:
        n = m.c_ctx(h)
        rows.append([_ctx_str(h), n] + [f"{m.c(h, y)} ({_f(p_mle(m, h, y), 3)})" for y in conts])
    out.append(_table(["context", "c(context)"] + [f"`{TOKENS[y]}`" for y in conts], rows, "lr" + "r" * len(conts)))
    out += ["", "Each cell is the count c(context, token) with the estimate c(context, token) / c(context) in "
            "parentheses; the other tokens never follow these contexts in the training corpus."]
    return "\n".join(out)


def _first_test_sentence_with_unseen_trigram():
    m = ngram_train()
    for s in test_set():
        ok3 = all(p_mle(m, _hist(w, 3), y) > 0 for w, y in _positions(s))
        ok2 = all(p_mle(m, _hist(w, 2), y) > 0 for w, y in _positions(s))
        if ok2 and not ok3:
            return s
    return test_set()[0]


def _pp(p, n):
    return _f(p ** (-1 / n)) if p > 0 else "inf"


def heldout_sentence():
    m = ngram_train()
    s = _first_test_sentence_with_unseen_trigram()
    rows, prod = [], [1.0, 1.0, 1.0]
    for t, (w, y) in enumerate(_positions(s)):
        cells = []
        for j, order in enumerate((1, 2, 3)):
            h = _hist(w, order)
            p = p_mle(m, h, y)
            prod[j] *= p
            cells.append(f"{m.c(h, y)}/{m.c_ctx(h)} = {_f(p)}")
        rows.append([t + 1, f"`{TOKENS[y]}`"] + cells)
    n = len(s)
    return "\n".join([
        f"The held-out sentence is the first sentence of the test set that every bigram estimate can score but some "
        f"trigram estimate cannot: `{_words(s)}` ({n} predicted tokens). Maximum-likelihood estimates, count of the "
        "token after the context divided by count of the context:",
        "",
        _table(["position", "token", "unigram", "bigram", "trigram"], rows),
        "",
        f"Probability of the sentence (product of the column): unigram {prod[0]:.4e}, bigram {prod[1]:.4e}, "
        f"trigram {prod[2]:.4e}. Perplexity (probability to the power -1/{n}): unigram {_pp(prod[0], n)}, "
        f"bigram {_pp(prod[1], n)}, trigram {_pp(prod[2], n)}.",
    ])


def heldout_repair():
    m = ngram_train()
    s = _first_test_sentence_with_unseen_trigram()
    n = len(s)
    alpha, r = best_smoothing(N_TRAIN, 3)
    rows, pr_ = [], [1.0, 1.0, 1.0]
    for t, (w, y) in enumerate(_positions(s)):
        h = _hist(w, 3)
        ps = [p_mle(m, h, y), p_addalpha(m, h, y, alpha), p_interp(m, h, y, r)]
        for j in range(3):
            pr_[j] *= ps[j]
        rows.append([t + 1, f"`{TOKENS[y]}`", f"{m.c(h, y)}/{m.c_ctx(h)}"] + [_f(p) for p in ps])
    return "\n".join([
        f"The same sentence, `{_words(s)}`, under the trigram model repaired two ways, with alpha = {alpha} and "
        f"r_ip = {r:g} (both chosen on the development set, table above):",
        "",
        _table(["position", "token", "c/c trigram", "MLE", f"add-alpha, alpha = {alpha}", f"interpolated, r_ip = {r:g}"],
               rows),
        "",
        f"Probability of the sentence: MLE {pr_[0]:.4e}, add-alpha {pr_[1]:.4e}, interpolated {pr_[2]:.4e}. "
        f"Perplexity: MLE {_pp(pr_[0], n)}, add-alpha {_pp(pr_[1], n)}, interpolated {_pp(pr_[2], n)}.",
    ])


def smoothing():
    m = ngram_train()
    orders = (2, 3, 5)
    best = {n: best_smoothing(N_TRAIN, n) for n in orders}
    rows = []
    for alpha in ALPHA_GRID:
        cells = []
        for n in orders:
            v = _f(ppl(mean_nll(lambda h, y: p_addalpha(m, h, y, alpha), dev_set(), n)), 3)
            cells.append(v + (" *" if best[n][0] == alpha else ""))
        rows.append([f"add-alpha, alpha = {alpha}"] + cells)
    for r in RATIO_GRID:
        cells = []
        for n in orders:
            v = _f(ppl(mean_nll(lambda h, y: p_interp(m, h, y, r), dev_set(), n)), 3)
            cells.append(v + (" *" if best[n][1] == r else ""))
        rows.append([f"interpolated, r_ip = {r:g}"] + cells)
    mle = [_f(ppl(mean_nll(lambda h, y: p_mle(m, h, y), dev_set(), n)), 3) for n in orders]
    rows.insert(0, ["maximum likelihood"] + mle)
    return "\n".join([
        f"Perplexity on the {N_DEV}-sentence development set ({_int(_n_predicted(dev_set()))} predicted tokens) of the "
        "models of order 2, 3 and 5 trained on the printed corpus. The star marks the setting with the lowest "
        f"development perplexity for that order. The uniform weight of the interpolated models is {EPS_UNIFORM}. "
        f"The model predicts V - 1 = {V} tokens (`<s>` is never predicted).",
        "",
        _table(["estimate", "order 2", "order 3", "order 5"], rows),
    ])


def perplexity_by_order():
    rows = []
    for n in range(1, CTX + 2):
        alpha, r = best_smoothing(N_TRAIN, n)
        mle, ak, ip = ngram_test_ppls(N_TRAIN, n)
        rows.append([n, _f(mle, 3), f"{_f(ak, 3)} (alpha = {alpha})", f"{_f(ip, 3)} (ratio {r:g})"])
    t = test_set()
    h, length, floor = _process_stats()
    nll_true = -sum(math.log(float(_true_prob(s))) for s in t) / _n_predicted(t)
    return "\n".join([
        f"Test perplexity ({N_TEST} sentences, {_int(_n_predicted(t))} predicted tokens) of the n-gram models trained on "
        "the printed corpus; alpha and the ratio are the development-set choices of the previous table.",
        "",
        _table(["order n", "maximum likelihood", "add-alpha", "interpolated"], rows),
        "",
        f"For comparison, the process that generated the text, scored with its own exact probabilities, has perplexity "
        f"{_f(math.exp(nll_true), 3)} on this test set (the long-run floor is {_f(math.exp(floor), 3)}); a model that "
        f"ignores the context entirely and guesses uniformly among the {V} tokens has perplexity {V}.",
    ])


def sparsity():
    m = ngram_for(N_TRAIN)
    lang = set()
    lang_ctx = set()
    for s, _ in _all_sentences():
        for w, y in _positions(s):
            for n in range(1, CTX + 2):
                lang.add((n, w[CTX - (n - 1):], y))
    rows = []
    t_pairs = _batch(test_set())
    for n in range(1, CTX + 2):
        poss = V_IN ** (n - 1) * V
        in_lang = sum(1 for x in lang if x[0] == n)
        in_train = len(m.joint[n - 1])
        unseen_ngram = sum(1 for w, y in t_pairs if m.c(_hist(w, n), y) == 0)
        unseen_ctx = sum(1 for w, y in t_pairs if m.c_ctx(_hist(w, n)) == 0)
        rows.append([n, _int(poss), in_lang, in_train, f"{100 * unseen_ngram / len(t_pairs):.1f}%",
                     f"{100 * unseen_ctx / len(t_pairs):.1f}%"])
    return "\n".join([
        f"An n-gram is a context of n - 1 tokens and the token after it (`<s>` may fill the context at the start). "
        f"Possible: V^(n-1) = {V_IN}^(n-1) contexts times the V - 1 = {V} tokens that can follow. 'In the language' counts the distinct n-grams of the "
        f"756 sentences, 'in training' those of the {N_TRAIN} printed sentences. The last two columns are shares of the "
        f"{len(t_pairs)} test positions whose n-gram, or whose context alone, never occurs in training.",
        "",
        _table(["n", "possible", "in the language", "in training", "test n-grams unseen", "test contexts unseen"], rows),
    ])


# --------------------------------------------------------------------------------------------- blocks: sampling

N_SAMPLES, MAX_LEN = 200, 12


def _sample_sentence(rng, dist):
    """Draw tokens one at a time from dist(window) (a list of V probabilities) until `.` or MAX_LEN tokens.
    The draw is one `rng.uniform()` per token: the token is the first id whose cumulative probability exceeds it
    (the last id if rounding leaves none)."""
    out, window, windows = [], (BOS,) * CTX, []
    while len(out) < MAX_LEN:
        windows.append(window)
        p = dist(window)
        u, acc, tok = rng.uniform(), 0.0, V - 1
        for i in range(V):
            acc += p[i]
            if u < acc:
                tok = i
                break
        out.append(tok)
        window = window[1:] + (tok,)
        if tok == ID["."]:
            break
    return out, windows


def _models_for_sampling():
    """(name, label, distribution function of a window, training sentences) for the models sampled in the chapter."""
    m = ngram_for(N_TRAIN)
    out = []
    for order in (2, 5):
        alpha, r = best_smoothing(N_TRAIN, order)
        out.append((f"ngram{order}", f"order {order} interpolated, r_ip = {r:g}",
                    (lambda o, r: lambda w: [p_interp(m, _hist(w, o), y, r) for y in range(V)])(order, r), N_TRAIN))
    out.append(("net40", f"network, {N_TRAIN} sentences", train_mlp(N_TRAIN)["model"].probs, N_TRAIN))
    out.append(("net160", "network, 160 sentences", train_mlp(160)["model"].probs, 160))
    return out


@lru_cache(maxsize=None)
def _samples():
    res = []
    for key, label, dist, n_train in _models_for_sampling():
        known = ngram_for(n_train)
        rng = bookrng.generator(f"language_modeling.sample.{key}")
        sents, unseen, positions = [], 0, 0
        for _ in range(N_SAMPLES):
            s, windows = _sample_sentence(rng, dist)
            sents.append(s)
            for w in windows:
                positions += 1
                unseen += known.c_ctx(w) == 0
        t_pairs = _batch(test_set())
        real = sum(1 for w, y in t_pairs if known.c_ctx(w) == 0) / len(t_pairs)
        res.append((key, label, sents, unseen / positions, real))
    return res


def sampling():
    rows, ex = [], []
    for key, label, sents, unseen, real in _samples():
        with_pp = [s for s in sents if ID["near"] in s]
        without = [s for s in sents if ID["near"] not in s]
        ok = lambda ss: sum(1 for s in ss if _grammar_accepts(s))
        rows.append([label, f"{100 * ok(sents) / len(sents):.1f}%", f"{ok(without)}/{len(without)}",
                     f"{ok(with_pp)}/{len(with_pp)}", f"{100 * unseen:.1f}%", f"{100 * real:.1f}%"])
        ex.append(f"- {label}: " + " | ".join(_words(s) for s in sents[:3]))
    seeds = ", ".join(f"{bookrng.seed_of('language_modeling.sample.' + key):#018x} (`language_modeling.sample.{key}`, {lab})"
                      for key, lab, _, _, _ in _samples())
    return "\n".join([
        f"Each model draws {N_SAMPLES} sentences. Generator: the book's SplitMix64, one generator per model, with the seeds "
        f"{seeds}. A sentence starts from four `<s>` tokens; "
        "for each token one `uniform()` is drawn and the token is the first id (0 to 17) whose cumulative probability "
        f"exceeds it; the sentence ends at `.` or after {MAX_LEN} tokens. The models are those of the previous "
        "tables (counts from the printed corpus, interpolation settings chosen on the development set), and the networks "
        "of the training blocks, stopped at their lowest development perplexity.",
        "",
        "The first three sentences of each model:",
        "",
        "\n".join(ex),
        "",
        "'Accepted' is the share of sentences that the grammar accepts: the right shape and a verb that agrees with the "
        "subject. The next two columns split the sentences, grammatical or not, by whether they contain the token `near` anywhere (with phrase) or not (without phrase), and give the accepted sentences over all sentences of each kind. "
        "'Unseen contexts' is the share of the positions whose last four tokens never occurred in the model's training "
        "text, among the sampled positions and among the positions of the real test text.",
        "",
        _table(["model", "accepted", "without phrase", "with phrase", "unseen, samples", "unseen, test text"], rows, "lrrrrr"),
    ])


# --------------------------------------------------------------------------------------------- blocks: the objective


def _logsoftmax(z):
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return [v - lse for v in z]


def nll_identity():
    m = ngram_for(N_TRAIN)
    alpha, r = best_smoothing(N_TRAIN, 3)
    t = test_set()

    def sentence_nll(s):
        return [-math.log(p_interp(m, _hist(w, 3), y, r)) for w, y in _positions(s)]

    s = t[0]
    nll = sentence_nll(s)
    n = len(nll)
    prob = 1.0
    for w, y in _positions(s):
        prob *= p_interp(m, _hist(w, 3), y, r)
    rows = [[i + 1, f"`{TOKENS[y]}`", _f(math.exp(-c), 4), _f(c, 4)] for i, ((w, y), c) in enumerate(zip(_positions(s), nll))]
    per_sentence, tot, cnt = [], 0.0, 0
    for s2 in t:
        c = sentence_nll(s2)
        per_sentence.append(math.exp(sum(c) / len(c)))
        tot += sum(c)
        cnt += len(c)
    return "\n".join([
        f"The trigram model interpolated with ratio {r:g} (trained on the printed corpus), on the first test sentence "
        f"`{_words(s)}`. Its probability of each token and the cross-entropy of that position, -ln p:",
        "",
        _table(["position", "token", "p(token \\| context)", "-ln p"], rows),
        "",
        f"Mean over the {n} positions: {_f(sum(nll) / n, 6)}. Product of the {n} probabilities: {prob:.6e}, so "
        f"-ln P(sentence) / {n} = {_f(-math.log(prob) / n, 6)}. Perplexity exp(mean) = {_f(math.exp(sum(nll) / n), 4)} "
        f"= P(sentence)^(-1/{n}) = {_f(prob ** (-1 / n), 4)}.",
        "",
        f"On the whole test set the perplexity is exp(total -ln p / total tokens) = {_f(math.exp(tot / cnt), 4)}. "
        f"The mean of the {len(t)} sentence perplexities is {_f(sum(per_sentence) / len(t), 4)}, a different number "
        "because every sentence counts once whatever its length.",
    ])


def teacher_forcing():
    sent = [ID[t] for t in ["the", "cat", "sees", "fish", "."]]
    s = [BOS] + sent
    rows = []
    for t in range(len(sent)):
        win = tuple(([BOS] * CTX + sent)[t:t + CTX])
        rows.append([t, f"`{TOKENS[s[t]]}`", f"`{TOKENS[s[t + 1]]}`",
                     "`" + " ".join(TOKENS[i] for i in win) + "`"])
    n_all = _n_predicted(train_set())
    return "\n".join([
        "The sentence `the cat sees fish .` with `<s>` in front is the sequence s_0 ... s_5. The inputs are x_t = s_t and "
        "the targets y_t = s_(t+1) for t = 0 ... 4. The last column is the window of the four tokens up to and "
        "including x_t that the network of this chapter reads, padded on the left with `<s>`:",
        "",
        _table(["t", "input x_t", "target y_t", "window read"], rows, "rlll"),
        "",
        f"One pass over this sentence yields {len(sent)} training pairs, one for each of its tokens. The "
        f"printed corpus of {N_TRAIN} sentences yields {_int(n_all)}.",
    ])


def _leak_pairs(sentences):
    """The bug: the window is shifted one position too far, so its last token is the target itself."""
    out = []
    for s in sentences:
        pad = [BOS] * CTX + s
        out += [(tuple(pad[j + 1:j + 1 + CTX]), pad[j + CTX]) for j in range(len(s))]
    return out


@lru_cache(maxsize=None)
def _leaky_run():
    model = Mlp(bookrng.generator(MLP_NAME))
    tr, dv = _leak_pairs(train_set()), _leak_pairs(dev_set())
    honest = _batch(dev_set())
    steps = (0, 25, 50, 100, 200)
    rows = []
    for step in range(max(steps) + 1):
        if step in steps:
            rows.append((step, math.exp(_mlp_loss(model, tr)), math.exp(_mlp_loss(model, dv)),
                         math.exp(_mlp_loss(model, honest))))
        _mlp_step(model, tr, MLP_LR)
    return model, rows


def shift_bug():
    model, rows = _leaky_run()
    floor = math.exp(_process_stats()[2])
    good = train_mlp(N_TRAIN)["model"]
    # which positions' outputs change when the token at index u is replaced?
    sent = [ID[t] for t in ["the", "cat", "sees", "fish", "."]]
    changed = {}
    for label, m, pairs_fn in (("correct window", good, _positions), ("shifted window", model, _leak_pairs)):
        base = [m.probs(w) for w, _ in pairs_fn([sent] if pairs_fn is _leak_pairs else sent)]
        alt = list(sent)
        alt[2] = ID["see"]                                    # replace `sees` (the token at index 2 of the sentence)
        new = [m.probs(w) for w, _ in pairs_fn([alt] if pairs_fn is _leak_pairs else alt)]
        changed[label] = [t for t in range(len(sent)) if max(abs(a - b) for a, b in zip(base[t], new[t])) > 1e-12]
    rows_t = [[s, _f(a, 3), _f(b, 3), _f(c, 1)] for s, a, b, c in rows]
    return "\n".join([
        f"The network of the chapter (same start, same {MLP_LR} learning rate, full batch, trained on the printed "
        "corpus) with the windows shifted one position too far: the window for target s_(t+1) is s_(t-2) ... s_(t+1), "
        "so its last token is the answer. Perplexity of the training pairs, of the development pairs built the same "
        "wrong way, and of the development text with correct windows (the evaluation the model is for):",
        "",
        _table(["step", "train, shifted", "dev, shifted", "dev, correct windows"], rows_t),
        "",
        f"The process floor is {_f(floor, 3)}. A perplexity of 1.05 on text with {_f(floor, 2)} of irreducible "
        "uncertainty is not a good model but a leak.",
        "",
        "Replace the token `sees` (index 2) of `the cat sees fish .` by `see` and count the positions (the rows t of "
        "the teacher-forcing table, 0 to 4) whose output distribution changes:",
        "",
        _table(["model", "positions whose output changes"],
               [[k, ", ".join(map(str, v))] for k, v in changed.items()], "ll"),
    ])


# --------------------------------------------------------------------------------------------- blocks: the network


@lru_cache(maxsize=None)
def _nudged_curves():
    """The training curves of the same run started with its first embedding weight changed by each of `stable.NUDGES`."""
    return [{step: (lt, ld) for step, lt, ld in train_mlp(N_TRAIN, d)["curve"]} for d in stable.NUDGES]


def mlp_training():
    r = train_mlp(N_TRAIN)
    model = r["model"]
    cp = {step: (lt, ld) for step, lt, ld in r["curve"]}
    nudged = _nudged_curves()

    def shown(step, which):
        """The perplexity at `step` with only the digits that survive the nudged runs (3 decimals when all agree)."""
        return stable.text(math.exp(cp[step][which]), [math.exp(c[step][which]) for c in nudged], lambda x: _f(x, 3))
    steps = [s for s in (0, 10, 20, 50, 100, 200, 300, 500) if s in cp]
    if r["step"] not in steps:
        steps.append(r["step"])
        steps.sort()
    rows = [[s, shown(s, 0), shown(s, 1), "<- lowest dev" if s == r["step"] else ""]
            for s in steps]
    test_nll = _mlp_loss(model, _batch(test_set()))
    return "\n".join([
        _seed_line(MLP_NAME),
        f"Model: the last {CTX} tokens (ids 0 to 18, `<s>` padding the start) are looked up in an embedding table of "
        f"V = {V_IN} rows and D = {D_EMB} columns and concatenated into a vector of {CTX * D_EMB} numbers; a layer "
        f"of H = {H_HID} tanh units (the correctly rounded library `tanh`) with weights W1 of shape [{H_HID}, "
        f"{CTX * D_EMB}] and bias b1; and an output layer W2 of shape [{V}, {H_HID}] with bias b2 whose softmax is "
        f"the distribution over the V - 1 = {V} predicted tokens. Loss: the mean of -ln p(target) over all training pairs. "
        f"Start: E entries `0.5 * rng.normal()` row by row (the {V_IN} x {D_EMB} entries first), then W1 entries "
        f"`rng.normal() / sqrt({CTX * D_EMB})` row by row, then W2 entries `rng.normal() / sqrt({H_HID})` row by row; "
        f"both biases start at 0. That is {model.n_params()} parameters. Training: full-batch gradient descent, "
        f"learning rate {MLP_LR}, {MLP_STEPS} steps on the {r['n_pairs']} training pairs of the printed corpus, "
        f"dev perplexity recorded every {MLP_EVERY} steps; the parameters of the recorded step with the lowest dev "
        "perplexity are kept (early stopping). " + stable.RULE + " Rounding noise is amplified over the 500 steps, so the last "
        "rows may lose digits.",
        "",
        _table(["step", "train perplexity", "dev perplexity", ""], rows, "rrrl"),
        "",
        f"The kept network (step {r['step']}) has test perplexity {_f(math.exp(test_nll), 3)}.",
    ])


def comparison():
    t = test_set()
    uni = ngram_test_ppls(N_TRAIN, 1)[1]
    rows = [["uniform guess over 18 tokens", "0", f"{V}"]]
    rows.append(["unigram, add-alpha", "18", _f(uni, 3)])
    for n in (2, 3, 5):
        alpha, r = best_smoothing(N_TRAIN, n)
        mle, ak, ip = ngram_test_ppls(N_TRAIN, n)
        stored = len(ngram_for(N_TRAIN).joint[n - 1])
        rows.append([f"order {n}, add-alpha (alpha = {alpha})", _int(stored), _f(ak, 3)])
        rows.append([f"order {n}, interpolated (ratio {r:g})", _int(stored), _f(ip, 3)])
    net = train_mlp(N_TRAIN)
    rows.append([f"network, window of {CTX}, {N_TRAIN} sentences (step {net['step']})", _int(net["model"].n_params()),
                 _f(mlp_ppls(N_TRAIN)[1], 3)])
    nll_true = -sum(math.log(float(_true_prob(s))) for s in t) / _n_predicted(t)
    rows.append(["the process itself", "-", _f(math.exp(nll_true), 3)])
    return "\n".join([
        f"Test perplexity on the {N_TEST} test sentences ({_int(_n_predicted(t))} predicted tokens) of every model "
        f"trained on the {N_TRAIN} printed sentences. 'Stored' is the number of distinct (context, token) entries a "
        "counting model keeps, or the number of parameters of the network.",
        "",
        _table(["model", "stored", "test perplexity"], rows),
    ])


SIZES = (20, 40, 80, 160, 320)


def data_size():
    rows = []
    for n in SIZES:
        tr = train_mlp(n)
        pairs = tr["n_pairs"]
        o2, o3, o5 = (ngram_test_ppls(n, o)[1 if o == 2 else 2] for o in (2, 3, 5))
        rows.append([n, _int(pairs), _f(o2, 3), _f(o3, 3), _f(o5, 3), f"{_f(mlp_ppls(n)[1], 3)} (step {tr['step']})"])
    return "\n".join([
        f"The training set is the first n sentences of the printed generator's stream (the {N_TRAIN} printed sentences "
        "are the first 40 of the 320; draws are made in the order already given, so a larger set extends a smaller "
        "one). Everything else is as before: the settings of the counting models are chosen again on the development "
        f"set for each size (order 2 uses add-alpha, orders 3 and 5 interpolation), the network starts from the same "
        f"seed, trains for {MLP_STEPS} steps at learning rate {MLP_LR} and is stopped at its lowest development "
        "perplexity. Test perplexity, same test set:",
        "",
        _table(["sentences", "training pairs", "order 2", "order 3", "order 5", "network"], rows),
        "",
        f"The process floor is {_f(math.exp(_process_stats()[2]), 3)} in the long run.",
    ])


CONTRAST_CONTEXTS = [["the", "cat"], ["the", "dogs"], ["the", "cat", "near", "the", "dogs"],
                     ["the", "dogs", "near", "the", "cat"]]


def _singular_share(dist, words):
    ids = [ID[w] for w in words]
    window = tuple(([BOS] * CTX + ids)[-CTX:])
    p = dist(window)
    sing = p[ID["sees"]] + p[ID["chases"]]
    plur = p[ID["see"]] + p[ID["chase"]]
    return sing / (sing + plur)


def _process_singular_share(words):
    ids = [ID[w] for w in words]
    sing = sum(_prefix_prob(ids + [ID[v]]) for v in ("sees", "chases"))
    plur = sum(_prefix_prob(ids + [ID[v]]) for v in ("see", "chase"))
    return float(sing / (sing + plur))


def contrast():
    models = _models_for_sampling()
    rows = []
    for words in CONTRAST_CONTEXTS:
        cells = [_f(_process_singular_share(words), 3)] + [_f(_singular_share(d, words), 3) for _, _, d, _ in models]
        rows.append(["`" + " ".join(words) + "`"] + cells)
    return "\n".join([
        "The probability of the singular verbs `sees` and `chases`, as a share of the probability of all four verbs, "
        "after four contexts, for the models of the sampling block (the process column is exact, from the "
        "enumeration of the 756 sentences):",
        "",
        _table(["context", "process"] + [lab for _, lab, _, _ in models], rows),
    ])


def work():
    p_dense = H_HID * CTX * D_EMB + V * H_HID
    fwd = 2 * p_dense
    per_token = 3 * fwd
    rows = []
    for n in SIZES:
        tr = train_mlp(n)
        pairs = tr["n_pairs"]
        rows.append([n, _int(pairs), _int(pairs * MLP_STEPS), _int(per_token * pairs * MLP_STEPS), tr["step"],
                     _int(per_token * pairs * tr["step"]), _int(pairs * (CTX + 1))])
    head_macs = 151936 * 1024
    return "\n".join([
        f"Multiply-adds of the forward pass of one position: the first layer {H_HID} x {CTX * D_EMB} = "
        f"{H_HID * CTX * D_EMB} and the output layer {V} x {H_HID} = {V * H_HID}, in all {p_dense}, which is "
        f"{fwd} FLOPs (two per multiply-add; the {CTX} embedding lookups, the {H_HID} tanh and the {V} exponentials "
        f"of the softmax are not counted); the output layer is {V * H_HID} of the {p_dense}. The backward pass "
        f"of a layer costs two products of the forward's size (one for the weights, one for the input), so a training "
        f"position costs 3 x {fwd} = {_int(per_token)} FLOPs = 6 x {p_dense}. A training run is that times the positions of "
        f"the set times the steps; the counting models need {CTX + 1} additions to a table for each position, once.",
        "",
        _table(["sentences", "training pairs", f"positions seen in {MLP_STEPS} steps", f"training FLOPs, {MLP_STEPS} steps",
                "kept step", "FLOPs to the kept step", "counting additions"], rows),
        "",
        f"With a vocabulary of 151,936 tokens and a model width of 1,024 (the Qwen3 0.6B shape of Chapter 3) the output "
        f"layer alone is 151,936 x 1,024 = {_int(head_macs)} multiply-adds, {_int(2 * head_macs)} FLOPs forward, for every "
        f"position. A model with P weights in dense layers trained on D positions needs about 6 P D FLOPs: P = 10^9 and "
        f"D = 10^12 give {6 * 10**9 * 10**12:.1e}.",
    ])


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def noun_geometry():
    sing = [ID[w] for w in ("cat", "dog", "fox")]
    plur = [ID[w] for w in ("cats", "dogs", "foxes")]
    groups = [("two singular nouns", [(a, b) for i, a in enumerate(sing) for b in sing[i + 1:]]),
              ("two plural nouns", [(a, b) for i, a in enumerate(plur) for b in plur[i + 1:]]),
              ("a singular and a plural noun", [(a, b) for a in sing for b in plur])]
    rows = []
    for label, pairs in groups:
        cells = []
        for n in (N_TRAIN, 160):
            E = train_mlp(n)["model"].E
            cells.append(_f(sum(_cos(E[a], E[b]) for a, b in pairs) / len(pairs), 3))
        rows.append([f"{label} ({len(pairs)} pairs)"] + cells)
    return "\n".join([
        "Mean cosine similarity of the rows of the embedding table (the cosine of Volume I, Chapter 3) for the six "
        "nouns cat, dog, fox and cats, dogs, foxes, over the pairs of each kind, for the early-stopped networks "
        f"trained on {N_TRAIN} and on 160 sentences:",
        "",
        _table(["pairs of", f"{N_TRAIN} sentences", "160 sentences"], rows),
    ])
