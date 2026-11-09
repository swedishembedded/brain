# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Decoding: Turning Probabilities into Text".

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. Each public function without a leading underscore that takes no arguments returns a
markdown string that the chapter embeds between `<!-- worked: decoding.<name> -->` markers; the helpers with arguments
or a leading underscore are shared with the chapter's figures, so a picture and the table beside it cannot disagree.

The model whose probabilities are decoded is not trained here: it is the counting model of Chapter 4 (the order-2
and order-4 interpolated models over the 40 printed training sentences), imported from `language_modeling`, so the
whole module runs in well under a second. The chapter's own subject is what is done with the probabilities:

* the distribution after one word, sampling from all of it, and the greedy loop;
* temperature (entropy against temperature, with the derivative checked), top-k, top-p and min-p cuts;
* the order of the processing steps, shown numerically, and the quality of 200 sampled sentences under several settings;
* the repetition penalty (the sign rule, its dependence on the offset of the logits) and the additive penalties;
* log-probabilities (underflow of a product, a per-token report, the sampled distribution against the model's);
* beam search by hand on a three-token table and on the toy language, with and without length normalisation;
* seeded sampling: shared against per-request random streams, and the order in which a draw walks the tokens;
* what choosing a token costs, counted in operations and bytes against the forward pass.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints its seeds.

Swedish Embedded AB implements solutions for controlling what a language model says, from the sampler to the log-
probabilities it reports. If your team needs expertise in decoding strategies for generative models, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import math
import struct
from functools import lru_cache

import bookrng
import language_modeling as lm

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
    if x == -math.inf:
        return "-inf"
    s = f"{x:.{nd}f}"
    return f"{0.0:.{nd}f}" if float(s) == 0 else s


def _int(n):
    return f"{n:,}"


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _words(ids):
    return " ".join(lm.TOKENS[i] for i in ids)


# --------------------------------------------------------------------------------------------- the model

V = lm.V                                # tokens a model predicts: ids 0 .. 17
TOKENS = lm.TOKENS
ID = lm.ID
DOT = ID["."]
THE = ID["the"]
OFFSET = 4.0                            # logits are ln p + OFFSET; any constant gives the same softmax
MAX_LEN = 12                            # a sampled sentence stops at the full stop or after 12 tokens


@lru_cache(maxsize=None)
def _counts():
    return lm.ngram_for(lm.N_TRAIN)


def _ratio(order):
    return lm.best_smoothing(lm.N_TRAIN, order)[1]


@lru_cache(maxsize=None)
def _dist_cached(order, window):
    m = _counts()
    r = _ratio(order)
    return tuple(lm.p_interp(m, lm._hist(window, order), y, r) for y in range(V))


def probs(order, ctx):
    """The model's next-token probabilities after the tokens `ctx` (a list of ids; the start is padded with `<s>`)."""
    w = (lm.BOS,) * lm.CTX
    for t in ctx[-lm.CTX:]:
        w = w[1:] + (t,)
    return list(_dist_cached(order, w))


def logits(p):
    return [math.log(x) + OFFSET for x in p]


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def argmax(z):
    """Index of the largest entry; the first on a tie."""
    return max(range(len(z)), key=lambda i: (z[i], -i))


def order_desc(p):
    """Token ids by decreasing probability; ties keep the lower id first (a stable argsort)."""
    return sorted(range(len(p)), key=lambda i: (-p[i], i))


NOUNS = tuple(ID[w] for w in lm.NOUNS)
ALLOWED = {THE: set(NOUNS), ID["near"]: {THE}}   # what the language allows after `the` and after `near`


# --------------------------------------------------------------------------------------------- the pipeline


def cut(p, k=0, top_p=1.0, min_p=0.0):
    """Truncate a distribution `p` (a list of probabilities): top-k, then top-p on the renormalised k, then min-p on the
    renormalised nucleus. Returns {token: renormalised probability}. k = 0, top_p >= 1 and min_p = 0 disable a cut."""
    idx = order_desc(p)
    if k > 0:
        idx = idx[:k]
    tot = sum(p[i] for i in idx)
    q = {i: p[i] / tot for i in idx}
    if 0 < top_p < 1:
        acc, keep = 0.0, []
        for i in idx:
            keep.append(i)
            acc += q[i]
            if acc >= top_p:
                break
        idx = keep
        tot = sum(q[i] for i in idx)
        q = {i: q[i] / tot for i in idx}
    if min_p > 0:
        top = max(q.values())
        idx = [i for i in idx if q[i] >= min_p * top]
        tot = sum(q[i] for i in idx)
        q = {i: q[i] / tot for i in idx}
    return q


def repetition(z, seen, theta):
    """The repetition penalty on logits z: a token in `seen` has a positive logit divided and a negative one multiplied by theta."""
    return [(x / theta if x > 0 else x * theta) if i in seen else x for i, x in enumerate(z)]


def additive(z, counts, pres, freq):
    """Presence and frequency penalties: pres + freq * count subtracted from the logit of a token generated count > 0 times."""
    return [x - (pres + freq * counts.get(i, 0)) if counts.get(i, 0) > 0 else x for i, x in enumerate(z)]


def choose_dist(z, *, tau=1.0, k=0, top_p=1.0, min_p=0.0):
    """The distribution a token is drawn from after the penalties: {token: probability}. tau = 0 is greedy (one token)."""
    if tau <= 0:
        return {argmax(z): 1.0}
    return cut(softmax([x / tau for x in z]), k, top_p, min_p)


def draw(q, u):
    """The token a uniform number u in [0, 1) selects from {token: probability}: walk the tokens in increasing id,
    return the first whose cumulative probability exceeds u (the last one if rounding leaves none)."""
    acc, last = 0.0, None
    for i in sorted(q):
        acc += q[i]
        last = i
        if u < acc:
            return i
    return last


def draw_by_rank(q, u):
    """The same draw with the tokens walked from the most probable to the least (ties by id)."""
    acc, last = 0.0, None
    for i in sorted(q, key=lambda i: (-q[i], i)):
        acc += q[i]
        last = i
        if u < acc:
            return i
    return last


def sample_text(order, rng, n_tokens, *, stop=False, **kw):
    """n_tokens tokens (or until the full stop when stop is true): one uniform per token, none at tau = 0."""
    ctx = []
    while len(ctx) < n_tokens:
        z = logits(probs(order, ctx))
        q = choose_dist(z, **kw)
        t = argmax(z) if kw.get("tau", 1.0) <= 0 else draw(q, rng.uniform())
        ctx.append(t)
        if stop and t == DOT:
            break
    return ctx


def _sentences(ctx):
    out, cur = [], []
    for t in ctx:
        cur.append(t)
        if t == DOT:
            out.append(cur)
            cur = []
    return out, cur


def _logprob(order, ctx):
    """ln p of the token list `ctx` under the model, started from `<s>`: the sum of the log conditional probabilities."""
    return sum(math.log(probs(order, ctx[:i])[t]) for i, t in enumerate(ctx))


# --------------------------------------------------------------------------------------------- block 1: the distribution


def toy_distribution():
    p = probs(2, [THE])
    z = logits(p)
    rows, cum = [], 0.0
    for rank, i in enumerate(order_desc(p), 1):
        cum += p[i]
        rows.append((rank, f"`{TOKENS[i]}`", _f(z[i], 3), _f(p[i], 4), _f(cum, 4)))
    near = probs(2, [ID["near"]])
    top_near = ", ".join(f"`{TOKENS[i]}` {_f(near[i], 4)}" for i in order_desc(near)[:3])
    junk = sum(p[i] for i in range(V) if i not in NOUNS)
    return (
        "The model: the interpolated bigram model of Chapter 4 (counts from the 40 printed training sentences, interpolation ratio "
        f"{_ratio(2):g}, a uniform floor of {lm.EPS_UNIFORM} shared over the 18 tokens), which reads the one token before the one it predicts. "
        "Its logits are $z_i = \\ln p_i + 4$ (the 4 is arbitrary: the softmax of $\\mathbf{z}$ is $p$ for any constant). The context is the single word `the`. "
        "Ties in probability keep the lower id first (`cats` has id 4, `dogs` id 5).\n\n"
        + _table(["rank", "token", "logit $z$", "$p$", "cumulative $p$"], rows, "rlrrr")
        + f"\n\nThe six nouns hold {_f(1 - junk, 4)} of the probability and the other twelve tokens {_f(junk, 4)}. "
        f"After the word `near` the three largest are {top_near}."
    )


# --------------------------------------------------------------------------------------------- block 2: sampling it all


def sampling_junk():
    name = "decoding.junk"
    p = probs(2, [THE])
    rng = bookrng.generator(name)
    n = 2000
    us, hits = [], [0] * V
    for j in range(n):
        u = rng.uniform()
        t = draw({i: p[i] for i in range(V)}, u)
        hits[t] += 1
        if j < 8:
            us.append((j + 1, _f(u, 4), f"`{TOKENS[t]}`"))
    junk = sum(h for i, h in enumerate(hits) if i not in NOUNS)
    expect = sum(p[i] for i in range(V) if i not in NOUNS)
    by = ", ".join(f"`{TOKENS[i]}` {hits[i]}" for i in order_desc(p)[:6])
    return (
        f"Generator: the book's SplitMix64, seed {_seed(name)} (`seed_of` of the name `{name}`). Each of the {n:,} draws is one `uniform()`; the "
        "token is the first id, in increasing id order, whose cumulative probability exceeds it (the walk of Volume I, Chapter 17). The "
        "distribution is the one after `the` of the table above.\n\n"
        + _table(["draw", "$u$", "token"], us, "rrl")
        + f"\n\nOver {n:,} draws: {by}; the other twelve tokens together {junk} times, {_f(100 * junk / n, 2)} percent against the "
        f"{_f(100 * expect, 2)} percent the distribution assigns them. If twenty tokens were drawn from this distribution one after another, "
        f"the probability that every one is a noun would be ${_f(1 - expect, 4)}^{{20}} = {_f((1 - expect) ** 20, 4)}$."
    )


# --------------------------------------------------------------------------------------------- block 3: greedy


def _greedy_ids(order, n):
    ctx = []
    for _ in range(n):
        ctx.append(argmax(probs(order, ctx)))
    return ctx


def greedy_run():
    n = 30
    g = _greedy_ids(2, n)
    steps = []
    for j in range(7):
        p = probs(2, g[:j])
        top = order_desc(p)
        steps.append((j + 1, f"`{TOKENS[g[j - 1]]}`" if j else "`<s>`", f"`{TOKENS[top[0]]}`", _f(p[top[0]], 4), f"`{TOKENS[top[1]]}`", _f(p[top[1]], 4)))
    gp = _logprob(2, g)
    name = "decoding.docs"
    rng = bookrng.generator(name)
    docs = []
    for d in range(3):
        s = sample_text(2, rng, n)
        docs.append((d + 1, _words(s), _f(_logprob(2, s) / n, 4)))
    # the shortest period of the greedy text
    period = next(q for q in range(1, n) if all(g[i] == g[i + q] for i in range(n - q)))
    margin = min(sorted(probs(2, g[:j]), reverse=True)[0] - sorted(probs(2, g[:j]), reverse=True)[1] for j in range(n))
    assert margin > 1e-3
    return (
        f"Greedy decoding of the bigram model of the previous block from the start marker, {n} tokens, no stop at the full stop (the model has never seen "
        "a word after a full stop, so its counts for that context are empty and it falls back to word frequencies; the most frequent word is `the`).\n\n"
        f"`{_words(g)}`\n\n"
        + _table(["step", "previous token", "largest", "$p$", "second", "$p$"], steps, "rlrrlr")
        + f"\n\nThe text repeats with period {period} from the first token. Mean log-probability per token of the greedy text: {_f(gp / n, 4)}. "
        f"The smallest gap between the two largest probabilities over the {n} steps is {_f(margin, 4)}, so no step is a near tie.\n\n"
        f"For contrast, three {n}-token texts sampled from the full distribution at every step. Generator: seed {_seed(name)} (`seed_of` of `{name}`), "
        f"{n} draws per text, one `uniform()` each, the three texts one after another from the same generator.\n\n"
        + _table(["text", "tokens", "mean log-probability per token"], docs, "rlr")
    )


# --------------------------------------------------------------------------------------------- block 4: temperature

TAUS = (0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)


def temperature_table():
    z = logits(probs(2, [THE]))
    rows = []
    for tau in TAUS + (100.0,):
        q = softmax([x / tau for x in z])
        h = entropy(q)
        rows.append((_f(tau, 2), _f(max(q), 4), _f(sum(q[i] for i in NOUNS), 4), _f(1 - sum(q[i] for i in NOUNS), 4), _f(h, 4), _f(math.exp(h), 2)))
    # the derivative of the entropy with respect to tau against Var(z) / tau^3, at tau = 1 and 2
    checks = []
    for tau in (1.0, 2.0):
        e = 1e-5
        fd = (entropy(softmax([x / (tau + e) for x in z])) - entropy(softmax([x / (tau - e) for x in z]))) / (2 * e)
        q = softmax([x / tau for x in z])
        mean = sum(a * b for a, b in zip(q, z))
        var = sum(a * (b - mean) ** 2 for a, b in zip(q, z))
        checks.append((_f(tau, 1), _f(fd, 6), _f(var / tau ** 3, 6)))
    return (
        "Logits of the distribution after `the` (the table above) divided by $\\tau$, then the softmax. 'Non-nouns' is the probability of the twelve tokens the "
        f"language does not allow after `the`; 'effective tokens' is $e^{{H}}$ with $H$ the entropy in nats. The uniform distribution has $H = \\ln 18 = {_f(math.log(18), 4)}$ "
        "and 18 effective tokens. Tolerance of every number: half a unit in the last digit.\n\n"
        + _table(["$\\tau$", "largest $p$", "six nouns", "non-nouns", "entropy $H$", "effective tokens"], rows)
        + "\n\nThe derivative of the entropy with respect to $\\tau$ by a central difference with step $10^{-5}$ against the formula "
        "$\\mathrm{Var}_{q}(z)/\\tau^3$ of the text:\n\n"
        + _table(["$\\tau$", "central difference", "$\\mathrm{Var}_q(z)/\\tau^3$"], checks)
    )


# --------------------------------------------------------------------------------------------- block 5: truncation

CONTEXTS = (("the", [THE]), ("near", [ID["near"]]))


def temperature_curves(taus=(0.5, 1.0, 2.0)):
    """(ranks, {tau: probabilities}) of the distribution after `the`, tokens in order of decreasing probability at tau = 1."""
    z = logits(probs(2, [THE]))
    order = order_desc(softmax(z))
    return list(range(1, V + 1)), {tau: [softmax([x / tau for x in z])[i] for i in order] for tau in taus}


def cumulative_curves():
    """{context word: cumulative probability of the 1, 2, ..., 18 most probable tokens} after `the` and after `near`."""
    out = {}
    for word, ctx in CONTEXTS:
        p = probs(2, ctx)
        acc, cum = 0.0, []
        for i in order_desc(p):
            acc += p[i]
            cum.append(acc)
        out[word] = cum
    return out


def truncation_table():
    rows = []
    settings = [("top-k 1", dict(k=1)), ("top-k 2", dict(k=2)), ("top-k 4", dict(k=4)), ("top-k 6", dict(k=6)), ("top-k 8", dict(k=8)),
                ("top-p 0.5", dict(top_p=0.5)), ("top-p 0.8", dict(top_p=0.8)), ("top-p 0.9", dict(top_p=0.9)), ("top-p 0.95", dict(top_p=0.95)),
                ("min-p 0.1", dict(min_p=0.1)), ("min-p 0.3", dict(min_p=0.3))]
    for label, kw in settings:
        cells = []
        for _, ctx in CONTEXTS:
            p = probs(2, ctx)
            allowed = ALLOWED[ctx[0]]
            q = cut(p, **kw)
            kept_mass = sum(p[i] for i in q)
            bad = sum(1 for i in q if i not in allowed)
            cells += [str(len(q)), _f(kept_mass, 4), str(bad)]
        rows.append((label, *cells))
    # the tie at the fifth and sixth place after `the`
    p = probs(2, [THE])
    tied = [TOKENS[i] for i in order_desc(p)[4:6]]
    # top-p over only the 8 best candidates, against top-p over all 18
    p8 = {i: p[i] for i in order_desc(p)[:8]}
    tot = sum(p8.values())
    cand = cut([p8.get(i, 0.0) for i in range(V)], top_p=0.95)
    full = cut(p, top_p=0.95)
    return (
        "Top-k, top-p and min-p cut the distributions after `the` and after `near`. Settings are applied alone. 'kept' is the number of tokens left, 'mass' the "
        "probability they held before renormalizing, and 'bad' how many of them the language does not allow (after `the` only the six nouns are allowed, "
        "after `near` only `the`). The nucleus rule is: the shortest prefix of the tokens in order of decreasing probability whose cumulative probability "
        "is at least $p_{\\mathrm{top}}$, with the cumulative sum of the renormalized probabilities; min-p keeps the tokens with probability at least $r_{\\min}$ times the largest.\n\n"
        + _table(["setting", "`the`: kept", "mass", "bad", "`near`: kept", "mass", "bad"], rows)
        + f"\n\nAfter `the` the fifth and sixth places are tied (`{tied[0]}` and `{tied[1]}`, both {_f(p[order_desc(p)[4]], 4)}), so a top-k of 5 keeps `{tied[0]}` and drops `{tied[1]}` "
        "by the lower-id rule. Top-p 0.95 over the full distribution keeps "
        f"{len(full)} tokens; over a list of only the 8 best candidates (renormalized from a mass of {_f(tot, 4)}) it keeps {len(cand)}."
    )


# --------------------------------------------------------------------------------------------- block 6: the order of steps


def filter_order():
    p = probs(2, [THE])
    z = logits(p)
    tau, k, pt = 1.5, 10, 0.9
    q_t = softmax([x / tau for x in z])

    def describe(q):
        top = order_desc([q.get(i, 0.0) for i in range(V)])[:1][0]
        bad = sum(v for i, v in q.items() if i not in NOUNS)
        return len(q), _f(q[top], 4), _f(bad, 4)

    pipelines = []
    # standard: temperature, top-k, top-p
    pipelines.append(("temperature, then top-k, then top-p", describe(cut(q_t, k=k, top_p=pt))))
    # top-p on the unheated distribution, then the temperature acts on what is left
    kept = cut(p, k=k, top_p=pt)
    zt = [z[i] / tau for i in kept]
    e = softmax(zt)
    pipelines.append(("top-k, then top-p, then temperature", describe({i: e[j] for j, i in enumerate(kept)})))
    # top-p first, then top-k
    kept = cut(q_t, top_p=pt)
    lst = [i for i in order_desc([kept.get(i, 0.0) for i in range(V)]) if i in kept][:k]
    tot = sum(kept[i] for i in lst)
    pipelines.append(("temperature, then top-p, then top-k", describe({i: kept[i] / tot for i in lst})))
    # no temperature, top-k then top-p
    pipelines.append(("top-k, then top-p (no temperature)", describe(cut(p, k=k, top_p=pt))))
    rows = [(a, str(b[0]), b[1], b[2]) for a, b in pipelines]
    # the renormalization identity: softmax of the kept logits equals the kept probabilities renormalized
    kept = order_desc(p)[:6]
    a = softmax([z[i] for i in kept])
    tot = sum(p[i] for i in kept)
    b = [p[i] / tot for i in kept]
    ident = max(abs(x - y) for x, y in zip(a, b))
    # penalties against the temperature: seen token `fox` (and `dog`)
    seen = {ID["fox"], ID["dog"]}
    tau2 = 0.5
    rep_a = softmax([x / tau2 for x in repetition(z, seen, 1.5)])
    rep_b = softmax([x / tau2 for x in z])
    rep_b = softmax(repetition([x / tau2 for x in z], seen, 1.5))
    add_a = softmax([x / tau2 for x in additive(z, {i: 1 for i in seen}, 1.0, 0.0)])
    add_b = softmax(additive([x / tau2 for x in z], {i: 1 for i in seen}, 1.0, 0.0))
    fox = ID["fox"]
    pen_rows = [
        ("repetition penalty 1.5, before the division", _f(rep_a[fox], 4)),
        ("repetition penalty 1.5, after the division", _f(rep_b[fox], 4)),
        ("presence penalty 1.0, before the division", _f(add_a[fox], 4)),
        ("presence penalty 1.0, after the division", _f(add_b[fox], 4)),
    ]
    return (
        f"Distribution after `the` (the table of the first block), temperature $\\tau = {tau}$, top-k $k = {k}$, top-p $p_{{\\mathrm{{top}}}} = {pt}$ (cumulative rule of the previous block). "
        "'tokens' is how many remain, 'largest' the probability of the most likely, and 'non-nouns' the total probability on tokens the language does not allow after `the`. "
        "The line without a temperature is the same cuts on the model's own distribution.\n\n"
        + _table(["order of the steps", "tokens", "largest $p$", "non-nouns"], rows, "lrrr")
        + f"\n\nThe softmax of the logits of the six best tokens and the probabilities of those six renormalized agree to {ident:.1e} (largest absolute difference).\n\n"
        "The probability of `fox` after `the` when `fox` and `dog` have been seen, with the penalty applied to the logits before or after they are divided by "
        f"$\\tau = {tau2}$ (the logits $z$ above, penalties on `fox` and `dog`):\n\n"
        + _table(["penalty and place", "$p$(`fox`)"], pen_rows, "lr")
    )


# --------------------------------------------------------------------------------------------- block 7: quality of settings

QUALITY = [
    ("full distribution", "full", dict()),
    ("temperature 0.7", "t07", dict(tau=0.7)),
    ("temperature 1.5", "t15", dict(tau=1.5)),
    ("top-k 4", "k4", dict(k=4)),
    ("top-p 0.9", "p09", dict(top_p=0.9)),
    ("min-p 0.1", "m01", dict(min_p=0.1)),
    ("temperature 0.7, top-p 0.9", "t07p09", dict(tau=0.7, top_p=0.9)),
    ("temperature 1.5, top-p 0.9", "t15p09", dict(tau=1.5, top_p=0.9)),
]
N_SENT = 200


@lru_cache(maxsize=None)
def _quality_rows():
    out = []
    for label, key, kw in QUALITY:
        name = f"decoding.quality.{key}"
        rng = bookrng.generator(name)
        sents = [sample_text(4, rng, MAX_LEN, stop=True, **kw) for _ in range(N_SENT)]
        acc = sum(lm._grammar_accepts(s) for s in sents)
        distinct = len({tuple(s) for s in sents})
        surprisal = sum(-_logprob(4, s) for s in sents) / sum(len(s) for s in sents)
        out.append((label, name, acc, distinct, surprisal))
    g = _greedy_sentence(4)
    out.append(("greedy (one sentence)", "", int(lm._grammar_accepts(g)), 1, -_logprob(4, g) / len(g)))
    return out


def _greedy_sentence(order):
    ctx = []
    while len(ctx) < MAX_LEN:
        ctx.append(argmax(probs(order, ctx)))
        if ctx[-1] == DOT:
            break
    return ctx


def decoding_quality():
    rows = _quality_rows()
    seeds = "; ".join(f"`{n}` {_seed(n)}" for _, n, _, _, _ in rows if n)
    body = [(lab, f"{acc}/{N_SENT if n else 1}", _f(100 * acc / (N_SENT if n else 1), 1), str(dist), _f(s, 4)) for lab, n, acc, dist, s in rows]
    g = _greedy_sentence(4)
    return (
        f"Each setting draws {N_SENT} sentences from the order-4 interpolated model of Chapter 4 (interpolation ratio {_ratio(4):g}), starting from `<s>`, ending at the full stop or after {MAX_LEN} tokens; "
        "one `uniform()` per token, the token chosen by the id-order walk over the tokens left after the cuts (cuts applied in the order temperature, top-k, top-p, min-p, each renormalizing). "
        f"One generator per setting, seeds: {seeds}. 'accepted' is the share the grammar of Chapter 4 accepts (right shape, verb agreeing with the subject); 'distinct' counts different sentences; "
        "'surprisal' is the mean of $-\\ln p$ per token under the model itself (no cut, temperature 1) over all tokens of the sentences.\n\n"
        + _table(["setting", "accepted", "percent", "distinct", "surprisal"], body)
        + f"\n\nThe greedy sentence is `{_words(g)}`."
    )


# --------------------------------------------------------------------------------------------- block 8: repetition penalty


def _greedy_penalised(offset, theta, n):
    ctx, seen, margins = [], set(), []
    for _ in range(n):
        z = [math.log(x) + offset for x in probs(2, ctx)]
        z = repetition(z, seen, theta)
        order = order_desc(z)
        margins.append(z[order[0]] - z[order[1]])
        t = argmax(z)
        ctx.append(t)
        seen.add(t)
    assert all(m == 0.0 or m > 1e-9 for m in margins)    # an exact tie (cats and dogs) goes to the lower id; no near tie
    return ctx


def repetition_rule():
    z = [3.0, 1.0, -1.0, -3.0]
    theta = 2.0
    naive = [x / theta for x in z]
    right = repetition(z, {0, 1, 2, 3}, theta)
    rows = [(f"{a:g}", f"{b:g}", f"{c:g}") for a, b, c in zip(z, naive, right)]
    # the same distribution under three offsets of the logits
    n = 30
    rows2, texts = [], {}
    for offset in (0.0, 4.0, 8.0):
        for th in (1.0, 1.3, 1.5, 2.0):
            ids = _greedy_penalised(offset, th, n)
            sents, rest = _sentences(ids)
            ok = sum(lm._grammar_accepts(s) for s in sents)
            rows2.append((f"{offset:g}", f"{th:g}", f"{ok}/{len(sents)}", str(len({t for t in ids}))))
            texts[(offset, th)] = ids
    return (
        "Logits $(3, 1, -1, -3)$ of four tokens that have all been seen, penalty $\\theta_{\\mathrm{rep}} = 2$. 'divide all' is the first rule one might write; 'sign rule' divides a positive logit and multiplies a negative one.\n\n"
        + _table(["logit $z$", "divide all", "sign rule"], rows)
        + f"\n\nGreedy decoding of the bigram model of the first block for {n} tokens, no stop at the full stop, with the repetition penalty on every token seen so far "
        "(the context starts empty; the first token has nothing to be penalised for). The logits are $\\ln p + c$ for the offset $c$ in the first column, which gives the same probabilities for every $c$. "
        "'accepted' counts the complete sentences the grammar accepts out of the complete sentences in the text; 'distinct' is the number of different tokens used. "
        "At every step the two largest penalized logits are either exactly equal (`cats` and `dogs`, which the lower-id rule settles) or more than $10^{-9}$ apart.\n\n"
        + _table(["offset $c$", "$\\theta_{\\mathrm{rep}}$", "accepted", "distinct tokens"], rows2, "rrrr")
        + "\n\nTwo of these texts in full.\n\n"
        + f"- offset 0, $\\theta_{{\\mathrm{{rep}}}} = 1.5$: `{_words(texts[(0.0, 1.5)])}`\n"
        + f"- offset 8, $\\theta_{{\\mathrm{{rep}}}} = 1.5$: `{_words(texts[(8.0, 1.5)])}`"
    )


def additive_penalties():
    z = [2.4, 2.0, 0.5, -1.0]
    counts = {0: 3, 1: 1}
    rows = []
    for pres, freq in ((0.0, 0.0), (1.0, 0.0), (0.0, 0.5), (0.5, 0.5)):
        zz = additive(z, counts, pres, freq)
        rows.append((f"{pres:g}", f"{freq:g}", *(_f(x, 2) for x in zz), _f(softmax(zz)[0], 4)))
    return (
        "Four tokens with logits $(2.4, 2.0, 0.5, -1.0)$; token 0 has been generated 3 times, token 1 once, tokens 2 and 3 never. The penalty subtracted from a token generated $c > 0$ times is "
        "$\\beta_{\\mathrm{pres}} + \\beta_{\\mathrm{frq}}\\, c$.\n\n"
        + _table(["$\\beta_{\\mathrm{pres}}$", "$\\beta_{\\mathrm{frq}}$", "$z_0$", "$z_1$", "$z_2$", "$z_3$", "$p_0$"], rows, "rrrrrrr")
    )


# --------------------------------------------------------------------------------------------- block 9: log-probabilities


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def logprob_underflow():
    # smallest n at which the product of n probabilities 0.1 is exactly 0, in float32 and float64
    def first_zero(rounder):
        prod, n = 1.0, 0
        while prod != 0.0:
            prod = rounder(prod * rounder(0.1))
            n += 1
        return n

    n32, n64 = first_zero(_f32), first_zero(lambda x: x)
    prod40 = 1.0
    for _ in range(40):
        prod40 = _f32(prod40 * _f32(0.1))
    total = sum(math.log(0.1) for _ in range(400))
    return (
        "A text of $n$ tokens each given probability 0.1 by the model: the product $0.1^n$ against the sum $n \\ln 0.1$, with each product rounded to the stated format "
        "(`float32` is IEEE binary32 with subnormals; `float64` is the double of the rest of the book).\n\n"
        + _table(["quantity", "value"], [
            ("float32 product after 40 tokens", f"{prod40:.3e} (below the smallest normal float32, 1.18e-38: a subnormal)"),
            ("float32 product is exactly 0 from token", str(n32)),
            ("float64 product is exactly 0 from token", str(n64)),
            ("sum of 400 log-probabilities, float64", _f(total, 3)),
        ], "lr")
    )


def logprob_report():
    name = "decoding.report"
    rng = bookrng.generator(name)
    tau, pt = 0.7, 0.9
    ctx, rows, tot_raw, tot_samp = [], [], 0.0, 0.0
    while len(ctx) < MAX_LEN:
        p = probs(4, ctx)
        z = logits(p)
        q = choose_dist(z, tau=tau, top_p=pt)
        t = draw(q, rng.uniform())
        top = [i for i in order_desc(p) if i != t][:2]
        alts = ", ".join(f"`{TOKENS[i]}` {_f(math.log(p[i]), 3)}" for i in top)
        rows.append((len(ctx) + 1, f"`{TOKENS[t]}`", _f(math.log(p[t]), 4), _f(math.log(q[t]), 4), alts))
        tot_raw += math.log(p[t])
        tot_samp += math.log(q[t])
        ctx.append(t)
        if t == DOT:
            break
    n = len(ctx)
    return (
        f"One sentence sampled from the order-4 model with $\\tau = {tau}$ and top-p {pt} (cuts in the order temperature, top-p). Generator: seed {_seed(name)} (`seed_of` of `{name}`), one `uniform()` per token, "
        "id-order walk. 'model $\\ln p$' is the log-probability the model itself gives the sampled token (temperature 1, no cut); 'sampler $\\ln q$' is the log-probability under the "
        "distribution the token was actually drawn from; 'next best' lists the two most probable other tokens by model $\\ln p$.\n\n"
        + _table(["position", "token", "model $\\ln p$", "sampler $\\ln q$", "next best, model $\\ln p$"], rows, "rlrrl")
        + f"\n\n`{_words(ctx)}`: sum of model $\\ln p$ = {_f(tot_raw, 4)} (probability {math.exp(tot_raw):.4f}, mean {_f(tot_raw / n, 4)} per token over {n} tokens); "
        f"sum of sampler $\\ln q$ = {_f(tot_samp, 4)}."
    )


# --------------------------------------------------------------------------------------------- block 10: beam search

# The three-token table of the hand example: first-token probabilities and a transition table over {a, b, c}.
LETTERS = "abc"
PI = (0.5, 0.4, 0.1)
TRANS = ((0.1, 0.4, 0.5), (0.1, 0.7, 0.2), (0.6, 0.1, 0.3))
STEPS = 3


def _seq_p(s):
    p = PI[s[0]]
    for a, b in zip(s, s[1:]):
        p *= TRANS[a][b]
    return p


def beam_run(width):
    """Beam search on the table: after each step the `width` best prefixes by joint probability (ties by the smaller sequence).
    Returns the list of (candidates sorted best first, kept) per step."""
    kept = [(PI[i], (i,)) for i in range(3)]
    kept.sort(key=lambda c: -c[0])
    history = [(list(kept), kept[:width])]
    kept = kept[:width]
    for _ in range(STEPS - 1):
        cand = [(p * TRANS[s[-1]][y], s + (y,)) for p, s in kept for y in range(3)]
        cand.sort(key=lambda c: -c[0])
        history.append((cand, cand[:width]))
        kept = cand[:width]
    return history


def _name(s):
    return "".join(LETTERS[i] for i in s)


def beam_hand():
    greedy = [max(range(3), key=lambda i: (PI[i], -i))]
    for _ in range(STEPS - 1):
        greedy.append(max(range(3), key=lambda i: (TRANS[greedy[-1]][i], -i)))
    hist = beam_run(2)
    allseq = sorted(((_seq_p(s), s) for s in _all()), key=lambda c: -c[0])
    rows = []
    for step, (cand, kept) in enumerate(hist, 1):
        ks = {s for _, s in kept}
        rows.append((step, ", ".join(f"`{_name(s)}` {p:.4f}{'' if s in ks else ' (dropped)'}" for p, s in cand)))
    return (
        "A language with three tokens $a, b, c$ and sequences of exactly three. First-token probabilities $(0.5, 0.4, 0.1)$; the probability of the next token after each token "
        "(rows are the previous token, columns the next one):\n\n"
        + _table(["after", "$a$", "$b$", "$c$"], [(LETTERS[i], *(f"{x:g}" for x in TRANS[i])) for i in range(3)], "lrrr")
        + "\n\nThe joint probability of a prefix is the product of the probabilities along it. Beam search with width 2 keeps, after each step, the two prefixes of largest joint probability "
        "(a tie would keep the candidate generated first, beam by beam and token by token; there is none here). Candidates of each step, best first:\n\n"
        + _table(["step", "candidates (joint probability)"], rows, "rl")
        + f"\n\nGreedy decoding gives `{_name(greedy)}` with probability {_seq_p(greedy):.4f}. Beam search of width 2 returns `{_name(hist[-1][1][0][1])}` with probability {hist[-1][1][0][0]:.4f}. "
        f"The best of all 27 sequences is `{_name(allseq[0][1])}` with probability {allseq[0][0]:.4f}, and the second best `{_name(allseq[1][1])}` with {allseq[1][0]:.4f}."
    )


def _all():
    out = [()]
    for _ in range(STEPS):
        out = [s + (y,) for s in out for y in range(3)]
    return out


def beam_dot():
    """Graphviz source of the width-2 search: kept prefixes solid, dropped candidates dashed grey, the winner in red."""
    hist = beam_run(2)
    lines = ["digraph beam {", "rankdir=LR; nodesep=0.12; ranksep=0.5;", 'node [shape=box, style="rounded", fontsize=11, margin="0.06,0.03"];',
             'root [label="start", shape=ellipse];']
    kept_prev = {(): "root"}
    win = hist[-1][1][0][1]
    for step, (cand, kept) in enumerate(hist, 1):
        kset = {s for _, s in kept}
        for p, s in cand:
            nid = "n_" + _name(s)
            parent = kept_prev.get(s[:-1])
            if parent is None:
                continue
            label = f"{_name(s)}\\n{p:.4f}"
            if s in kset:
                color = "#c53030" if s == win[: len(s)] else "#2b6cb0"
                lines.append(f'{nid} [label="{label}", color="{color}", penwidth=2];')
                lines.append(f'{parent} -> {nid} [label="{LETTERS[s[-1]]}", color="{color}", penwidth=2, fontsize=10];')
            else:
                lines.append(f'{nid} [label="{label}", color="#a0aec0", fontcolor="#718096", style="rounded,dashed"];')
                lines.append(f'{parent} -> {nid} [label="{LETTERS[s[-1]]}", color="#a0aec0", fontcolor="#718096", style=dashed, fontsize=10];')
        kept_prev = {s: "n_" + _name(s) for _, s in kept}
    lines.append("}")
    return "\n".join(lines)


def _beam_language(order, width, maxlen=MAX_LEN):
    """Beam search on the toy language: each step extends every active prefix by every token, keeps the `width` best extensions by total
    log-probability (ties: the extension generated first), moves those that end in the full stop to the finished list and continues with the others.
    Stops when `width` hypotheses are finished, the beam is empty or `maxlen` tokens are reached."""
    active, done = [(0.0, ())], []
    for _ in range(maxlen):
        cand = []
        for lp, ctx in active:
            p = probs(order, list(ctx))
            cand += [(lp + math.log(p[y]), ctx + (y,)) for y in range(V)]
        cand.sort(key=lambda c: -c[0])
        top = cand[:width]
        active = [c for c in top if c[1][-1] != DOT]
        done += [c for c in top if c[1][-1] == DOT]
        if len(done) >= width or not active:
            break
    return done, active


def beam_language():
    rows = []
    for width in (1, 2, 4, 8):
        done, active = _beam_language(4, width)
        raw = sorted(done, key=lambda d: -d[0])
        norm = sorted(done, key=lambda d: -d[0] / len(d[1]))
        rows.append((width, str(len(done)), f"`{_words(raw[0][1])}`", _f(raw[0][0], 3), f"`{_words(norm[0][1])}`", _f(norm[0][0] / len(norm[0][1]), 3)))
    done8, _ = _beam_language(4, 8)
    top4 = sorted(done8, key=lambda d: -d[0] / len(d[1]))[:4]
    lines = [f"- `{_words(c)}` ({len(c)} tokens, total {_f(lp, 3)}, mean {_f(lp / len(c), 3)})" for lp, c in top4]
    done2, act2 = _beam_language(2, 4)
    best_act = sorted(act2, key=lambda d: -d[0])[0]
    fin2 = ", ".join(f"`{_words(c)}`" for _, c in done2)
    g = _greedy_sentence(4)
    sample_row = next(r for r in _quality_rows() if r[0] == "full distribution")
    best_mean = sorted(done8, key=lambda d: -d[0] / len(d[1]))[0]
    return (
        "Beam search on the order-4 model of Chapter 4 (interpolation ratio "
        f"{_ratio(4):g}), started from `<s>`, at most {MAX_LEN} tokens; the rule is the one in the text: the `width` best extensions by total log-probability, those ending in the full stop leave "
        "the beam, and the search ends when `width` hypotheses have finished. For each width, the finished hypothesis with the largest total log-probability and the one with the largest "
        "mean log-probability per token (length normalization with exponent 1).\n\n"
        + _table(["width", "finished", "best by total", "total $\\ln p$", "best by mean", "mean $\\ln p$"], rows, "rrlrlr")
        + "\n\nThe four best finished hypotheses at width 8 by mean log-probability:\n\n" + "\n".join(lines)
        + f"\n\nThe order-2 model of the first block with width 4 and the same rule finishes {len(done2)} hypothesis ({fin2}) and still has "
        f"{len(act2)} unfinished beams after {MAX_LEN} tokens; the best of them is `{_words(best_act[1])}` at total {_f(best_act[0], 3)}.\n\n"
        f"Mean surprisal per token under the model: the best sentence by mean of width 8, {_f(-best_mean[0] / len(best_mean[1]), 4)}; the greedy sentence, {_f(-_logprob(4, g) / len(g), 4)}; "
        f"the {N_SENT} sentences sampled from the full distribution in the earlier table, {_f(sample_row[4], 4)}."
    )


# --------------------------------------------------------------------------------------------- block 11: determinism


SECOND_ORDER = ("B", "C", "A")


def determinism():
    n_tok = 8
    names = ("A", "B", "C")

    def run_shared(order_of_requests):
        rng = bookrng.generator("decoding.shared")
        ctxs = {r: [] for r in names}
        for _ in range(n_tok):
            for r in order_of_requests:
                z = logits(probs(2, ctxs[r]))
                ctxs[r].append(draw(choose_dist(z), rng.uniform()))
        return ctxs

    def run_own(order_of_requests):
        rngs = {r: bookrng.generator(f"decoding.request.{r}") for r in names}
        ctxs = {r: [] for r in names}
        for _ in range(n_tok):
            for r in order_of_requests:
                z = logits(probs(2, ctxs[r]))
                ctxs[r].append(draw(choose_dist(z), rngs[r].uniform()))
        return ctxs

    rows = []
    for label, fn in (("one shared generator", run_shared), ("one generator per request", run_own)):
        a = fn(names)
        b = fn(SECOND_ORDER)
        for r in names:
            rows.append((label, r, f"`{_words(a[r])}`", f"`{_words(b[r])}`", "same" if a[r] == b[r] else "different"))
    # the walk order
    p = probs(2, [THE])
    rng = bookrng.generator("decoding.walk")
    walk = []
    for j in range(8):
        u = rng.uniform()
        q = {i: p[i] for i in range(V)}
        walk.append((j + 1, _f(u, 4), f"`{TOKENS[draw(q, u)]}`", f"`{TOKENS[draw_by_rank(q, u)]}`"))
    differ = sum(1 for w in walk if w[2] != w[3])
    # a constant seed for every request
    consts = []
    for r in range(3):
        rng = bookrng.generator("decoding.constant")
        ctx = []
        for _ in range(n_tok):
            ctx.append(draw(choose_dist(logits(probs(2, ctx))), rng.uniform()))
        consts.append(f"`{_words(ctx)}`")
    return (
        f"Three requests A, B, C each sample {n_tok} tokens from the bigram model of the first block, started from `<s>`, one `uniform()` per token, id-order walk, temperature 1 and no cut. "
        "The three advance in lock step (a token for the first request of the list, then for the second, then for the third, and again). Shared: one generator of seed "
        f"{_seed('decoding.shared')} (`decoding.shared`). Own: one generator each, seeds {_seed('decoding.request.A')} (`decoding.request.A`), {_seed('decoding.request.B')} (`decoding.request.B`), "
        f"{_seed('decoding.request.C')} (`decoding.request.C`). The requests are served in the order A, B, C, and in the order B, C, A (every request then has a different place in the round).\n\n"
        + _table(["generators", "request", "order A, B, C", "order B, C, A", ""], rows, "llllc")
        + f"\n\nThe same uniform numbers (seed {_seed('decoding.walk')}, `decoding.walk`) applied to the distribution after `the`, walking the tokens in increasing id order or from the most probable to the least:\n\n"
        + _table(["draw", "$u$", "id-order walk", "walk by probability"], walk, "rrll")
        + f"\n\nThe two walks pick different tokens for {differ} of the 8 numbers.\n\n"
        f"Three requests that all use the seed of `decoding.constant` ({_seed('decoding.constant')}) produce: {consts[0]}; {consts[1]}; {consts[2]}."
    )


# --------------------------------------------------------------------------------------------- block 12: cost


def _np(D, L, F, vocab):
    """Parameters that take part in a matrix product: L blocks of 4 D^2 (attention) + 2 D F (plain feed-forward), plus the tied
    embedding table once (Chapter 10)."""
    return L * (4 * D * D + 2 * D * F) + vocab * D


def sampling_cost():
    models = [("GPT-2 small", 768, 12, 3072, 50257, 1024), ("a larger decoder", 4096, 32, 16384, 128000, 4096)]
    rows = []
    for label, D, L, F, vocab, ctx in models:
        npar = _np(D, L, F, vocab)
        fwd = 2 * npar + 4 * L * ctx * D
        ops = {
            "argmax": vocab - 1,
            "softmax and draw": 3 * vocab + vocab,
            "top-k by heap, k = 40 (up to)": int(vocab * math.log2(40)),
            "top-p by full sort": int(vocab * math.log2(vocab)),
        }
        for name, c in ops.items():
            rows.append((label, name, _int(c), _f(100 * c / fwd, 3)))
        rows.append((label, "forward pass (2 N_p + 4 L T D)", _int(fwd), "100"))
    rb = []
    for label, D, L, F, vocab, ctx in models:
        npar = _np(D, L, F, vocab)
        rows_b = 64
        rb.append((label, _int(4 * npar), _int(4 * vocab), _int(rows_b * 4 * vocab), _int(rows_b * 32 * 8), _int(rows_b * 4)))
    n, ln = 8000, 1000
    return (
        "Operations to choose one token, against the operations of the forward pass that produced its logits. Counts: argmax is $V - 1$ comparisons; softmax and the walk are about $4V$ "
        "(an exponential, a sum, a division and a cumulative addition per token); a heap that keeps the $k$ best costs $V \\log_2 k$ comparisons; a full comparison sort $V \\log_2 V$. "
        "The forward pass is $2N_{\\mathrm{p}} + 4LTD$ FLOPs per token at context $T$ ($N_{\\mathrm{p}} = L(4D^2 + 2DF) + VD$, Chapter 10; the larger decoder has $D = 4096$, $L = 32$, "
        "$F = 4D$, $V = 128{,}000$ and $T = 4096$; GPT-2 small has $T = 1024$). Tolerance: counts are rounded down to whole operations.\n\n"
        + _table(["model", "step", "operations", "percent of the forward pass"], rows, "llrr")
        + "\n\nBytes that cross between the card and the host per decoding step of 64 sequences, against the bytes of weights the forward pass reads (32-bit values): "
        "the whole row of logits of every sequence, or a list of the 32 best (id, logit) pairs of eight bytes each, or one 4-byte id.\n\n"
        + _table(["model", "weights read, bytes", "one logits row", "64 logits rows", "64 lists of 32 pairs", "64 ids"], rb, "lrrrrr")
        + f"\n\nRebuilding a seen-token set from the whole context at every generated token touches $n$ tokens per step: for a prompt of {n:,} tokens and {ln:,} generated tokens, "
        f"{_int(sum(n + j for j in range(ln)))} token visits; folding in each new token touches {_int(ln)}."
    )
