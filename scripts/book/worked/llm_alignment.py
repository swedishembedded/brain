# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Reinforcement and Preference Learning for Language Models".

Plain Python 3, standard library only, written from the chapter's own description. Two small worlds carry the chapter.

The first is the chat conversation of `tokenization` (its tokenizer and its template are imported, never copied) scored by a
context-free bigram language model counted from that chapter's corpus: it is a real causal language model small enough that
the masked cross-entropy of supervised fine-tuning, the three conventions for its divisor and its gradient can all be printed.

The second is one prompt with five complete answers. A policy is a softmax over the five answers (a language model's
distribution over completions, collapsed to the few that matter), the reference policy is the model that supervised
fine-tuning left behind, and a labeller has a reward for every answer. Everything the chapter says about preference data,
the Bradley-Terry model, a reward model, the KL-regularised objective and its closed-form optimum, direct preference
optimisation, group-relative advantages, verifiable rewards and their failure is computed exactly on those five numbers or
sampled from them with the book's generator, so that a result can be compared with the exact answer.

Each public function without a leading underscore that takes no arguments returns a markdown string that the chapter embeds
between `<!-- worked: llm_alignment.<name> -->` markers. Underscore functions and the constants are shared with the chapter's
figures, so a picture and the table beside it cannot disagree. Every number learned from samples goes through `stable` (the rule
of `value_learning`): the run is repeated with the first logit of the reference policy changed by each of the nudges of
`stable.py`, and a digit is printed only if the run and its four copies share it.

Swedish Embedded AB implements solutions for aligning language models with human preferences and with checkers that cannot be
argued with. If your team needs expertise in preference optimisation and reinforcement learning from verifiable rewards, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng
import policy_gradients as pg
import stable
import tokenization as tk
import value_learning as vl

# ------------------------------------------------------------------ the five-answer world

ANSWERS = ("42", "The sum is 42.", "43", "I am not sure.", "40 41 42 43 44 45")
K = len(ANSWERS)
PI_REF = (0.15, 0.10, 0.30, 0.20, 0.25)               # the policy supervised fine-tuning left behind
THETA_REF = tuple(math.log(p) for p in PI_REF)         # its logits, up to an additive constant
R_TRUE = (2.0, 2.5, 0.0, 0.5, -1.0)                    # the labeller's reward of each answer
EXACT_CHECK = (1.0, 1.0, 0.0, 0.0, 0.0)                # the checker that reads the last number (answers 0 and 1 are right)
WEAK_CHECK = (1.0, 1.0, 0.0, 0.0, 1.0)                 # the checker that accepts the answer if 42 appears anywhere in the text

PAIR_N, PAIR_BETA = 16, 0.5                            # labelled pairs, and the beta of the examples that use them
RM_LR, RM_STEPS = PAIR_BETA ** 2 * 4.0, 100            # reward-model step size (= eta_dpo * beta^2, see dpo_is_reward_modelling)
DPO_LR = 4.0
BETAS = (10.0, 2.0, 1.0, 0.5, 0.25, 0.1)

GROUP, GRPO_STEPS, GRPO_RUNS, GRPO_LR = 8, 150, 10, 0.5
GRPO_CHECKS = (0, 10, 25, 50, 100, 150)

# ------------------------------------------------------------------ small helpers


def _softmax(z):
    return pg._softmax(z)


def _logsumexp(z):
    m = max(z)
    acc = 0.0
    for x in z:
        acc += math.exp(x - m)
    return m + math.log(acc)


def _logsoftmax(z):
    lse = _logsumexp(z)
    return [x - lse for x in z]


def _kl(p, q):
    """KL(p || q) in nats, the sum over the answers in order of p ln(p / q), with 0 ln 0 = 0."""
    acc = 0.0
    for a, b in zip(p, q):
        if a > 0.0:
            acc += a * math.log(a / b)
    return acc


def _dot(a, b):
    acc = 0.0
    for x, y in zip(a, b):
        acc += x * y
    return acc


def _sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _fmt(x, nd=4):
    return f"{x:.{nd}f}"


def _row(xs, nd=4):
    return "(" + ", ".join(f"{x:.{nd}f}" for x in xs) + ")"


def _table(headers, rows, align=None):
    return vl._table(headers, rows, align)


def _seed_line(name):
    return (f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} "
            f"(`seed_of` of the name `{name}`, defined on the pseudocode page).")


def _mean_sd(vals):
    return pg._mean_sd(vals)


RULE_TEXT = ("Every number learned from samples is printed with the digits that the run and four copies of it share: the copies "
             "start with the first logit of the reference policy, theta_ref[0], changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the "
             "nudges of stable.py), and a digit is printed only if all five agree on it ('about x' keeps the leading digits that "
             "survive, 'unstable' means that none does). The copies use the same generator and so draw the same numbers; they "
             "differ only if a nudge flips a comparison. A run is a deterministic computation, so another implementation "
             "reproduces the printed digits.")


def _pi_star(r, beta):
    """The closed-form optimum of the KL-regularised objective: pi_ref(y) exp(r(y) / beta) / Z, as a stable softmax."""
    return _softmax([t + x / beta for t, x in zip(THETA_REF, r)])


def _objective(pi, r, beta):
    """J_beta(pi) = sum_y pi(y) r(y) - beta KL(pi || pi_ref)."""
    return _dot(pi, r) - beta * _kl(pi, PI_REF)


# ------------------------------------------------------------------ supervised fine-tuning on a chat

SFT_ALPHA = 0.1                                       # smoothing of the bigram counts
SFT_LR = 2.0                                          # step size of the gradient steps of sft_step
CONV_A = (("system", "the baker sells bread"), ("user", "the cat sat on the bread"), ("assistant", "the baker sat on the cat"))
CONV_B = (("system", "the baker sells bread"), ("user", "the baker sold bread"), ("assistant", "the baker sells bread"))


class _Chat:
    """One rendered conversation: its ids, a role-and-part label per id and the bigram model that scores it."""

    def __init__(self, tok, messages):
        start, end = tok.special_id["<|im_start|>"], tok.special_id["<|im_end|>"]
        nl = tok.encode("\n", recognise=False)
        self.ids, self.part = [], []
        for role, content in messages:
            head = [start] + tok.encode(role + "\n", recognise=False)
            body = tok.encode(content, recognise=False)
            for i in head:
                self.ids.append(i)
                self.part.append(f"{role} header")
            for i in body:
                self.ids.append(i)
                self.part.append(f"{role} text")
            self.ids.append(end)
            self.part.append(f"{role} END")
            for i in nl:
                self.ids.append(i)
                self.part.append(f"{role} newline")
        assert self.ids == tok.encode(tk.chatml(list(messages)))

    def targets(self):
        """The target positions: position t has input ids[t] and target ids[t + 1], t = 0 .. len - 2."""
        return list(range(len(self.ids) - 1))


def _bigram(tok):
    """The pretrained model: counts of adjacent ids in the chapter's corpus, logit z[prev][next] = ln(count + alpha)."""
    v = tok.first_special + len(tk.SPECIALS)
    ids = tok.encode(tk.CORPUS)
    count = [[0] * v for _ in range(v)]
    for a, b in zip(ids, ids[1:]):
        count[a][b] += 1
    return [[math.log(count[a][b] + SFT_ALPHA) for b in range(v)] for a in range(v)], v


def _supervised(chat, convention):
    """The set of target positions a convention supervises, by the part of the token being predicted (the token t + 1)."""
    kinds = {
        "text": ("assistant text", "assistant END"),
        "newline": ("assistant text", "assistant END", "assistant newline"),
        "header": ("assistant header", "assistant text", "assistant END", "assistant newline"),
        "all": None,
    }[convention]
    return [t for t in chat.targets() if kinds is None or chat.part[t + 1] in kinds]


def _nll(z, chat, t):
    return -_logsoftmax(z[chat.ids[t]])[chat.ids[t + 1]]


def sft_masked_loss():
    tok = tk.ToyTokenizer()
    z, v = _bigram(tok)
    a, b = _Chat(tok, CONV_A), _Chat(tok, CONV_B)
    rows = []
    for t in _supervised(a, "newline"):
        lp = _logsoftmax(z[a.ids[t]])[a.ids[t + 1]]
        rows.append([t, tok.name(a.ids[t]), tok.name(a.ids[t + 1]), a.part[t + 1], _fmt(math.exp(lp)), _fmt(-lp)])
    first = _table(["t", "input", "target", "part of the turn", "p(target)", "-ln p"], rows, "rlllrr")
    conv = []
    for label, key in (("assistant text and END (this chapter)", "text"),
                       ("and the line feed after END (the form `answers_as_asked` trains)", "newline"),
                       ("and the assistant's header (the mask of a whole conversation)", "header"),
                       ("every target, prompt included", "all")):
        sup = _supervised(a, key)
        total = 0.0
        for t in sup:
            total += _nll(z, a, t)
        conv.append([label, len(sup), _fmt(total, 4), _fmt(total / len(sup), 4)])
    ta = _supervised(a, "text")
    tb = _supervised(b, "text")
    sa = sum(_nll(z, a, t) for t in ta)
    sb = sum(_nll(z, b, t) for t in tb)
    pooled = (sa + sb) / (len(ta) + len(tb))
    per_ex = (sa / len(ta) + sb / len(tb)) / 2
    two = _table(["divisor", "value", "weight of one token of A", "weight of one token of B"], [
        ["supervised tokens of the whole batch", _fmt(pooled), _fmt(1.0 / (len(ta) + len(tb)), 4), _fmt(1.0 / (len(ta) + len(tb)), 4)],
        ["supervised tokens of each example, then the mean of the examples", _fmt(per_ex),
         _fmt(1.0 / (2 * len(ta)), 4), _fmt(1.0 / (2 * len(tb)), 4)]], "lrrr")
    return (
        f"The pretrained model is a bigram: for the {v} ids of the tokenization chapter's toy tokenizer (326 ordinary ids, then "
        f"START = 326, END = 327, EOT = 328) it counts the adjacent id pairs of that chapter's corpus ({len(tok.encode(tk.CORPUS))} "
        f"ids, no special tokens) and sets the logit of the pair (a, b) to ln(count(a, b) + {SFT_ALPHA}), so p(b | a) is the softmax of "
        f"row a, in the stable form. Conversation A is the three messages of that chapter ((system, `{CONV_A[0][1]}`), (user, "
        f"`{CONV_A[1][1]}`), (assistant, `{CONV_A[2][1]}`)) in its ChatML template, {len(a.ids)} ids; position t has the input id "
        f"ids[t] and the target ids[t + 1], t = 0 .. {len(a.ids) - 2}, so there are {len(a.targets())} targets and the first id is "
        f"never one. The ids come from the tokenizer of Volume V, Chapter 2 (`tokenization.ToyTokenizer`: byte-level BPE with 70 "
        f"merges trained on that corpus, pre-tokenized as there, ids 0 .. 325, then the three special ids); conversation B, "
        f"used for the two divisors below, is (system, `{CONV_B[0][1]}`), (user, `{CONV_B[1][1]}`), (assistant, "
        f"`{CONV_B[2][1]}`) in the same template. In the table START, END and the pieces are written as in that chapter ('_' is a space, <0A> a line feed). "
        f"Chosen by this chapter's rule, the supervised targets are the assistant's text and END (7 targets); the line feed after "
        f"END is listed below them.\n\n{first}\n\n"
        f"Loss of the same model on conversation A under four conventions for which targets count (sum of -ln p over the counted "
        f"targets in position order; the mean divides by their number):\n\n"
        + _table(["counted targets", "number", "sum", "mean"], conv, "lrrr")
        + f"\n\nTwo conversations in one batch: B is ((system, `{CONV_B[0][1]}`), (user, `{CONV_B[1][1]}`), (assistant, "
        f"`{CONV_B[2][1]}`)), {len(tb)} supervised targets against {len(ta)} for A, summed losses {_fmt(sa)} and {_fmt(sb)}. The same "
        f"summed losses give two means:\n\n{two}")


def sft_step():
    tok = tk.ToyTokenizer()
    z, v = _bigram(tok)
    a = _Chat(tok, CONV_A)
    sup = _supervised(a, "text")
    n = len(sup)
    t0 = sup[0]
    p = _softmax(z[a.ids[t0]])
    g_target = (p[a.ids[t0 + 1]] - 1.0) / n
    others = sorted(((p[i] / n, i) for i in range(v) if i != a.ids[t0 + 1]), reverse=True)[:2]
    grad_sum = (p[a.ids[t0 + 1]] - 1.0) / n + sum(p[i] / n for i in range(v) if i != a.ids[t0 + 1])

    def loss_of(zz, positions):
        total = 0.0
        for t in positions:
            total += _nll(zz, a, t)
        return total / len(positions)

    def step(zz):
        grad = {}
        for t in sup:
            row = _softmax(zz[a.ids[t]])
            g = grad.setdefault(a.ids[t], [0.0] * v)
            for i in range(v):
                g[i] += (row[i] - (1.0 if i == a.ids[t + 1] else 0.0)) / n
        return [[x - SFT_LR * grad[r][i] if r in grad else x for i, x in enumerate(zz[r])] for r in range(v)]

    rest = [t for t in a.targets() if t not in sup]
    rows = []
    zz = z
    for k in range(0, 21):
        if k in (0, 1, 2, 5, 10, 20):
            rows.append([k, _fmt(loss_of(zz, sup)), _fmt(loss_of(zz, rest)),
                         _fmt(math.exp(_logsoftmax(zz[a.ids[sup[-1]]])[a.ids[sup[-1] + 1]]))])
        zz = step(zz)
    untouched = all(zz[r] == z[r] for r in range(v) if r not in {a.ids[t] for t in sup})
    return (
        f"The model and conversation A of the previous block. The loss is the mean of -ln p(target) over the n = {n} supervised "
        f"targets (divisor {n}); every other position contributes nothing to the value or the gradient. At the first supervised "
        f"position (t = {t0}, input {tok.name(a.ids[t0])}, target {tok.name(a.ids[t0 + 1])}) the gradient of the loss with respect to "
        f"the {v} logits of that position's row is (p - e_target) / {n}: its target entry is {g_target:.4f} (p(target) = "
        f"{p[a.ids[t0 + 1]]:.4f}) and the two largest other entries are {others[0][0]:.4f} (the id {tok.name(others[0][1])}) and "
        f"{others[1][0]:.4f} (the id {tok.name(others[1][1])}); the {v} entries add to {'a number of absolute value below 1e-12' if abs(grad_sum) < 1e-12 else grad_sum}, which is 0 up to rounding. "
        f"Gradient descent on the logit table with the step {SFT_LR}: every step computes the gradient of the masked mean over "
        f"all {n} targets from the table as it stands (the rows of the table that no supervised position reads, "
        f"{'all of them' if untouched else 'not all'} unchanged after 20 steps, get no gradient), and subtracts {SFT_LR} times it. "
        f"'Supervised' is the loss being minimised; 'other targets' is the mean -ln p over the {len(rest)} targets the mask "
        f"leaves out, computed by the same model and never trained on; 'p(END after the last word)' is the probability of the closing id after the "
        f"last word of the answer.\n\n"
        + _table(["steps", "supervised loss", "other targets", "p(END after the last word)"], rows, "rrrr"))


# ------------------------------------------------------------------ preference data, Bradley-Terry and a reward model

PAIRS_NAME = "llm_alignment.pairs"


def _pairs():
    """PAIR_N labelled pairs (chosen, rejected, i, j, u). Per pair, in this order: i = integer(5), j = integer(4) with j
    raised by one when j >= i (so that j != i), then u = uniform(); the labeller picks i when u < sigmoid(r(i) - r(j)),
    otherwise j."""
    gen = bookrng.generator(PAIRS_NAME)
    out = []
    for _ in range(PAIR_N):
        i = gen.integer(K)
        j = gen.integer(K - 1)
        if j >= i:
            j += 1
        u = gen.uniform()
        wins = u < _sigmoid(R_TRUE[i] - R_TRUE[j])
        out.append((i if wins else j, j if wins else i, i, j, u))
    return out


def _bt_loss(s, pairs):
    """The mean Bradley-Terry loss of the scores s: (1/N) sum_pairs softplus(-(s[chosen] - s[rejected]))."""
    total = 0.0
    for (c, r, _i, _j, _u) in pairs:
        total += _softplus(-(s[c] - s[r]))
    return total / len(pairs)


def _bt_gradient(s, pairs):
    """d loss / d s: for every pair the chosen answer gets -sigmoid(-m) / N and the rejected one +sigmoid(-m) / N, m the margin."""
    g = [0.0] * K
    for (c, r, _i, _j, _u) in pairs:
        w = _sigmoid(-(s[c] - s[r])) / len(pairs)
        g[c] -= w
        g[r] += w
    return g


def _fit_scores(pairs, lr=RM_LR, steps=RM_STEPS, checks=()):
    """Full-batch gradient descent on the Bradley-Terry loss from s = 0; the scores after each of `checks` steps."""
    s = [0.0] * K
    saved = {}
    for step in range(steps + 1):
        if step in checks:
            saved[step] = list(s)
        if step == steps:
            break
        g = _bt_gradient(s, pairs)
        s = [x - lr * d for x, d in zip(s, g)]
    return s, saved


def bradley_terry_pairs():
    pairs = _pairs()
    bt = [[_sigmoid(R_TRUE[i] - R_TRUE[j]) for j in range(K)] for i in range(K)]
    matrix = _table(["P(row preferred to column)"] + [f"a{j}" for j in range(K)],
                    [[f"a{i}"] + [("-" if i == j else _fmt(bt[i][j], 3)) for j in range(K)] for i in range(K)], "l" + "r" * K)
    rows = [[n, f"a{i}", f"a{j}", _fmt(u, 6), _fmt(_sigmoid(R_TRUE[i] - R_TRUE[j]), 4), f"a{c}", "yes" if R_TRUE[c] >= R_TRUE[r] else "NO"]
            for n, (c, r, i, j, u) in enumerate(pairs)]
    table = _table(["pair", "i", "j", "u", "P(i preferred)", "chosen", "agrees with the reward"], rows, "rllrrlr")
    upsets = sum(1 for (c, r, _i, _j, _u) in pairs if R_TRUE[c] < R_TRUE[r])
    names = "; ".join(f"a{k} = `{ANSWERS[k]}`" for k in range(K))
    return (
        f"One prompt (`17 + 25 =`), five complete answers: {names}. The labeller's reward of the answers is "
        f"r = {_row(R_TRUE, 1)} (a model never sees it, only choices made by it). The Bradley-Terry probability that a labeller "
        f"prefers answer i to answer j is sigmoid(r_i - r_j):\n\n{matrix}\n\n"
        f"{_seed_line(PAIRS_NAME)} Each of the {PAIR_N} pairs draws, in this order, i = integer({K}), j = integer({K - 1}) (raised by 1 "
        f"when j >= i, so that j is another answer) and u = uniform(); the labeller chooses i when u < sigmoid(r_i - r_j) and j "
        f"otherwise. The generator runs once across the {PAIR_N} pairs.\n\n{table}\n\n"
        f"{upsets} of the {PAIR_N} labels disagree with the reward (the labeller chose the answer with the lower reward), which is what "
        f"a probability of preferring that is not 1 means.")


def reward_model():
    pairs = _pairs()
    checks = (0, 10, 30, RM_STEPS)
    s, saved = _fit_scores(pairs, checks=checks)
    rows = [[step, _row(saved[step], 3), _fmt(_bt_loss(saved[step], pairs), 4),
             f"{sum(1 for (c, r, *_x) in pairs if saved[step][c] > saved[step][r])} of {PAIR_N}"] for step in checks]
    shifted = [x + 5.0 for x in s]
    order = sorted(range(K), key=lambda k: (-s[k], k))
    return (
        f"The pairs of the previous block. The reward model is a table of one score per answer, s = (s_0 .. s_{K - 1}), the "
        f"simplest model that can express any ranking (a real one is a network that reads the text). Pair n with chosen answer c "
        f"and rejected answer d has margin m = s_c - s_d and loss softplus(-m) = -ln sigmoid(m); the loss is the mean over the "
        f"{PAIR_N} pairs (divisor {PAIR_N}), its gradient gives the chosen answer -sigmoid(-m) / {PAIR_N} and the rejected one "
        f"+sigmoid(-m) / {PAIR_N} (summed over the pairs). Full-batch gradient descent from s = 0 with the step {RM_LR}, "
        f"{RM_STEPS} steps; no weight decay and no early stopping other than the step count.\n\n"
        + _table(["step", "scores s", "mean loss", "pairs with a positive margin in the labelled direction"], rows, "rlrr")
        + f"\n\nAt step 0 the loss is ln 2 = {math.log(2):.4f}. After {RM_STEPS} steps the loss is {_fmt(_bt_loss(s, pairs), 4)}; adding 5 to "
        f"every score leaves it at {_fmt(_bt_loss(shifted, pairs), 4)}, because only differences of scores enter. The ranking the "
        f"model learned is a{' > a'.join(str(k) for k in order)}; the labeller's own is a{' > a'.join(str(k) for k in sorted(range(K), key=lambda k: (-R_TRUE[k], k)))} (rewards {', '.join(f'{x:g}' for x in sorted(R_TRUE, reverse=True))}).")


# ------------------------------------------------------------------ the KL-regularised objective and its optimum

ASCENT_LR, ASCENT_STEPS = 1.0, 20000


def _objective_gradient(theta, r, beta):
    """The exact gradient of J_beta(pi_theta) with respect to the logits: pi_j (f_j - sum_k pi_k f_k) with
    f_k = r_k - beta ln(pi_k / pi_ref_k)."""
    pi = _softmax(theta)
    f = [r[k] - beta * math.log(pi[k] / PI_REF[k]) for k in range(K)]
    mean = _dot(pi, f)
    return [pi[j] * (f[j] - mean) for j in range(K)]


def _ascend(r, beta, steps=ASCENT_STEPS, lr=ASCENT_LR):
    theta = list(THETA_REF)
    for _ in range(steps):
        g = _objective_gradient(theta, r, beta)
        theta = [t + lr * d for t, d in zip(theta, g)]
    return _softmax(theta)


def kl_optimum():
    rows = []
    for beta in BETAS:
        pi = _pi_star(R_TRUE, beta)
        z = sum(PI_REF[k] * math.exp((R_TRUE[k] - max(R_TRUE)) / beta) for k in range(K))
        log_z = max(R_TRUE) / beta + math.log(z)
        rows.append([f"{beta:g}", *(_fmt(x, 3) for x in pi), _fmt(_dot(pi, R_TRUE), 3), _fmt(_kl(pi, PI_REF), 3),
                     _fmt(_objective(pi, R_TRUE, beta), 4), _fmt(beta * log_z, 4)])
    rows.append(["infinity", *(_fmt(x, 3) for x in PI_REF), _fmt(_dot(PI_REF, R_TRUE), 3), "0.000", _fmt(_dot(PI_REF, R_TRUE), 4), _fmt(_dot(PI_REF, R_TRUE), 4)])
    table = _table(["beta", "pi*(a0)", "pi*(a1)", "pi*(a2)", "pi*(a3)", "pi*(a4)", "E r", "KL of pi* from pi_ref", "J_beta(pi*)", "beta ln Z"],
                   rows, "r" * 10)
    check = []
    for beta in (2.0, 1.0, 0.5):
        pi = _ascend(R_TRUE, beta)
        star = _pi_star(R_TRUE, beta)
        check.append([f"{beta:g}", _fmt(max(abs(a - b) for a, b in zip(pi, star)), 6),
                      _fmt(abs(_objective(star, R_TRUE, beta) - _objective(pi, R_TRUE, beta)), 9)])
    return (
        f"The five answers of the preference block, the reference policy pi_ref = {_row(PI_REF, 2)} (the policy supervised "
        f"fine-tuning left behind; its logits are theta_ref = ln pi_ref) and the reward r = {_row(R_TRUE, 1)}. For each beta the "
        f"closed-form optimum is pi*(y) = pi_ref(y) exp(r(y) / beta) / Z with Z = sum_y pi_ref(y) exp(r(y) / beta), computed as "
        f"a stable softmax of ln pi_ref(y) + r(y) / beta. 'E r' is sum_y pi*(y) r(y), the KL is in nats (natural logarithm, "
        f"direction pi* then pi_ref, sums in answer order), J_beta(pi) = E r - beta KL(pi || pi_ref), and the last column is the "
        f"number beta ln Z that the derivation says J_beta(pi*) equals (the last row, beta = infinity, is pi_ref itself).\n\n{table}\n\n"
        f"The exact gradient of J_beta(theta) with respect to the five logits is pi_j (f_j - sum_k pi_k f_k) with "
        f"f_k = r_k - beta ln(pi_k / pi_ref_k). Gradient ascent from theta_ref with the step {ASCENT_LR:g}, {ASCENT_STEPS:,} steps, "
        f"reaches:\n\n"
        + _table(["beta", "largest abs(pi - pi*)", "abs(J_beta(pi*) - J_beta(pi))"], check, "rrr"))


def _kl_matched_best(r, kl_target):
    """The largest true expected reward over the optima of r's own family at the KL budget kl_target: the beta whose
    optimum has that KL by bisection on ln beta (the KL falls as beta rises), 200 halvings."""
    lo, hi = math.log(1e-3), math.log(1e4)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _kl(_pi_star(r, math.exp(mid)), PI_REF) > kl_target:
            lo = mid
        else:
            hi = mid
    pi = _pi_star(r, math.exp(0.5 * (lo + hi)))
    return _dot(pi, R_TRUE)


def learned_reward_cost():
    pairs = _pairs()
    s, _ = _fit_scores(pairs)
    rows = []
    for beta in (10.0, 3.0, 2.0, 1.0, 0.5, 0.25):
        pi = _pi_star(s, beta)
        kl = _kl(pi, PI_REF)
        best = _kl_matched_best(R_TRUE, kl)
        got = _dot(pi, R_TRUE)
        rows.append([f"{beta:g}", _row(pi, 3), _fmt(kl, 3), _fmt(_dot(pi, s), 3), _fmt(got, 3), _fmt(best, 3), _fmt(best - got, 3)])
    return (
        f"The reward model of the previous blocks, s = {_row(s, 3)} (a table of scores after {RM_STEPS} steps), used as the reward "
        f"of the KL-regularised objective: for each beta the optimum pi*_s(y) = pi_ref(y) exp(s(y) / beta) / Z. 'E s' is the "
        f"reward the model believes the policy earns, 'E r' the reward the labeller's r = {_row(R_TRUE, 1)} gives it, and 'best "
        f"E r at that KL' is the largest E r among the optima of r itself (beta found by bisection on ln beta over [ln 0.001, ln 10000], 200 halvings: the KL of the optimum falls as beta rises, so a midpoint whose KL exceeds the target becomes the new lower end and any other midpoint the new upper end, and beta is the final midpoint) "
        f"that spend the same KL(pi || pi_ref); the last column is the shortfall, the price of the reward model's errors at that "
        f"distance from the reference.\n\n"
        + _table(["beta", "pi*_s", "KL of pi from pi_ref", "E s", "E r", "best E r at that KL", "shortfall"], rows, "rlrrrrr"))


# ------------------------------------------------------------------ direct preference optimisation


def _implicit_reward(theta, beta):
    """beta ln(pi_theta(y) / pi_ref(y)) for every answer (the implicit reward of DPO)."""
    lp = _logsoftmax(theta)
    return [beta * (lp[k] - math.log(PI_REF[k])) for k in range(K)]


def _dpo_loss(theta, pairs, beta):
    """The mean DPO loss over (chosen, rejected, ...) pairs: softplus(-u), u = beta ((ln pi - ln pi_ref)(chosen) - (the same)(rejected))."""
    rew = _implicit_reward(theta, beta)
    total = 0.0
    for (c, r, *_rest) in pairs:
        total += _softplus(-(rew[c] - rew[r]))
    return total / len(pairs)


def _dpo_gradient(theta, pairs, beta):
    """d loss / d theta: per pair -sigmoid(-u) beta (e_chosen - e_rejected), summed and divided by the number of pairs. The
    log-normaliser of the softmax cancels inside u, so the gradient has no term for it."""
    rew = _implicit_reward(theta, beta)
    g = [0.0] * K
    for (c, r, *_rest) in pairs:
        w = _sigmoid(-(rew[c] - rew[r])) * beta / len(pairs)
        g[c] -= w
        g[r] += w
    return g


def dpo_implicit_reward():
    beta = PAIR_BETA
    pi = _pi_star(R_TRUE, beta)
    rew = [beta * math.log(pi[k] / PI_REF[k]) for k in range(K)]
    shift = [R_TRUE[k] - rew[k] for k in range(K)]
    z = sum(PI_REF[k] * math.exp((R_TRUE[k] - max(R_TRUE)) / beta) for k in range(K))
    log_z = max(R_TRUE) / beta + math.log(z)
    rows = [[f"a{k}", _fmt(R_TRUE[k], 1), _fmt(PI_REF[k], 3), _fmt(pi[k], 4), _fmt(rew[k], 4), _fmt(shift[k], 4)] for k in range(K)]
    return (
        f"The optimum pi* of the KL-regularised objective for beta = {beta:g} and the labeller's reward r, with pi_ref = {_row(PI_REF, 2)}. "
        f"The implicit reward of a policy is beta ln(pi(y) / pi_ref(y)); at pi* it differs from r(y) by the same number for every "
        f"answer, which the derivation says is beta ln Z = {beta * log_z:.4f}.\n\n"
        + _table(["answer", "r", "pi_ref", "pi*", "beta ln(pi*/pi_ref)", "r minus the implicit reward"], rows, "lrrrrr"))


DPO_PAIR = (0, 2)                                       # chosen a0 `42`, rejected a2 `43`
DPO_OFFSET = (0.4, 0.0, -0.4, 0.0, 0.0)                 # theta = theta_ref + offset
DPO_STEP = 1.0


def dpo_pair_gradient():
    beta = PAIR_BETA
    c, r = DPO_PAIR
    theta = [t + o for t, o in zip(THETA_REF, DPO_OFFSET)]
    pairs = [(c, r)]
    pi = _softmax(theta)
    rew = _implicit_reward(theta, beta)
    u = rew[c] - rew[r]
    grad = _dpo_gradient(theta, pairs, beta)
    h, worst, fd = 1e-6, 0.0, []
    for k in range(K):
        up = list(theta)
        up[k] += h
        dn = list(theta)
        dn[k] -= h
        d = (_dpo_loss(up, pairs, beta) - _dpo_loss(dn, pairs, beta)) / (2 * h)
        fd.append(d)
        worst = max(worst, abs(d - grad[k]))
    after = [t - DPO_STEP * g for t, g in zip(theta, grad)]
    pi2 = _softmax(after)
    rew2 = _implicit_reward(after, beta)
    u2 = rew2[c] - rew2[r]
    at_ref = _dpo_loss(list(THETA_REF), pairs, beta)
    rows = [[f"a{k}", _fmt(PI_REF[k], 4), _fmt(pi[k], 4), _fmt(rew[k], 4), _fmt(grad[k], 5), _fmt(fd[k], 5), _fmt(pi2[k], 4)] for k in range(K)]
    return (
        f"One pair: chosen a{c} (`{ANSWERS[c]}`), rejected a{r} (`{ANSWERS[r]}`), beta = {beta:g}, the policy at theta = theta_ref + "
        f"{_row(DPO_OFFSET, 1)} (so that the policy has already moved: at theta = theta_ref the margin is 0 and the loss ln 2 = "
        f"{at_ref:.4f}). The margin is u = (implicit reward of the chosen answer) - (implicit reward of the rejected answer), the "
        f"implicit reward of an answer being beta ln(pi_theta / pi_ref); the loss is softplus(-u) and its gradient with "
        f"respect to the logits is -beta sigmoid(-u) (e_chosen - e_rejected). 'Central difference' is (L(theta + h e_k) - "
        f"L(theta - h e_k)) / (2h) with h = 1e-06 on the loss itself. 'After one step' is the policy softmax(theta - {DPO_STEP:g} "
        f"* gradient).\n\n"
        + _table(["answer", "pi_ref", "pi_theta", "implicit reward", "gradient", "central difference", "pi after one step"], rows, "lrrrrrr")
        + f"\n\nu = {u:.4f}, loss = softplus(-u) = {_softplus(-u):.4f}, sigmoid(-u) = {_sigmoid(-u):.4f}, so the gradient on the "
        f"chosen logit is -{beta:g} * {_sigmoid(-u):.4f} = {grad[c]:.5f}. The largest difference between the two gradient columns is "
        f"{worst:.1e}, below the criterion 1e-7. After the step the margin is {u2:.4f} (it was {u:.4f}) and the probability of the "
        f"chosen answer {pi2[c]:.4f} (it was {pi[c]:.4f}) while that of the rejected answer is {pi2[r]:.4f} (it was {pi[r]:.4f}).")


def _population_pairs():
    """All 20 ordered pairs (i, j) with i != j, each with the weight 1/20 times the Bradley-Terry probability that i is chosen."""
    out = []
    for i in range(K):
        for j in range(K):
            if i != j:
                out.append((i, j, _sigmoid(R_TRUE[i] - R_TRUE[j]) / (K * (K - 1))))
    return out


def _population_gradient(theta, beta):
    rew = _implicit_reward(theta, beta)
    g = [0.0] * K
    for (c, r, w) in _population_pairs():
        wt = w * _sigmoid(-(rew[c] - rew[r])) * beta
        g[c] -= wt
        g[r] += wt
    return g


POP_LR, POP_STEPS, POP_CHECKS = 20.0, 100, (0, 10, 30, 60, 100)


def dpo_population():
    beta = PAIR_BETA
    star = _pi_star(R_TRUE, beta)
    theta = list(THETA_REF)
    rows = []
    for step in range(POP_STEPS + 1):
        if step in POP_CHECKS:
            pi = _softmax(theta)
            rew = _implicit_reward(theta, beta)
            gap = max(abs(a - b) for a, b in zip(pi, star))
            rows.append([step, _row(pi, 4), f"{gap:.1e}", f"{abs(_kl(star, pi)):.1e}"])
        if step == POP_STEPS:
            break
        g = _population_gradient(theta, beta)
        theta = [t - POP_LR * d for t, d in zip(theta, g)]
    return (
        f"The population version of the preference data: every ordered pair (i, j) with i different from j of the five answers "
        f"(20 of them), with the weight 1/20 times the probability sigmoid(r_i - r_j) that the labeller chooses i over j, "
        f"r = {_row(R_TRUE, 1)}; the loss is the weighted sum of softplus(-u) over the 20 pairs (u as before, beta = {beta:g}). "
        f"Gradient descent from theta_ref with the step {POP_LR:g}. The target is the optimum pi* of the KL-regularised objective "
        f"for the same beta and the reward r, pi* = {_row(star, 4)}.\n\n"
        + _table(["step", "pi_theta", "largest abs(pi - pi*)", "KL of pi* from pi_theta"], rows, "rlrr"))


def dpo_is_reward_modelling():
    pairs = _pairs()
    beta = PAIR_BETA
    checks = (0, 10, 30, RM_STEPS)
    _s, saved = _fit_scores(pairs, checks=checks)
    theta = list(THETA_REF)
    rows = []
    shared = []
    worst_all = 0.0
    for step in range(RM_STEPS + 1):
        if step in checks:
            star = _pi_star(saved[step], beta)
            pi = _softmax(theta)
            rew = _implicit_reward(theta, beta)
            gap_pi = max(abs(a - b) for a, b in zip(pi, star))
            gap_s = max(abs((rew[k] - rew[0]) - (saved[step][k] - saved[step][0])) for k in range(K))
            worst_all = max(worst_all, gap_pi, gap_s)
            rows.append([step, _row(saved[step], 3), _row(pi, 4), f"{gap_pi:.1e}", f"{gap_s:.1e}"])
            shared.append([step, f"{pi[0]:.4f}", f"{pi[2]:.4f}", f"{rew[0] - rew[2]:.4f}"])
        if step == RM_STEPS:
            break
        g = _dpo_gradient(theta, pairs, beta)
        theta = [t - DPO_LR * d for t, d in zip(theta, g)]
    return (
        f"The {PAIR_N} pairs of the preference block. Route 1 (reward model, then optimum): scores s from 0 by gradient descent on the "
        f"mean Bradley-Terry loss with the step {RM_LR:g}, then pi*_s = softmax(theta_ref + s / beta) with beta = {beta:g}. Route 2 "
        f"(DPO): theta from theta_ref by gradient descent on the mean DPO loss over the same pairs with the step {DPO_LR:g}, beta = "
        f"{beta:g}, which is {DPO_LR:g} * beta^2 = {DPO_LR * beta ** 2:g} times the gradient of the loss in the scores. "
        f"'Policy gap' is the largest absolute difference between the DPO policy pi_theta and pi*_s at the same step; 'reward gap' "
        f"the largest difference between beta ln(pi_theta/pi_ref) and s after both are shifted to be 0 on a0.\n\n"
        + _table(["step", "scores s of route 1", "policy pi_theta of route 2", "policy gap", "reward gap"], rows, "rllrr")
        + f"\n\nThe largest of all the gaps is {worst_all:.1e}, below the criterion 1e-9.\n\n"
        + f"Pair 8 of the preference block has chosen a0 and rejected a2. The same route-2 policy at those steps, with the margin "
        f"u = beta ln(pi_theta/pi_ref) of a0 minus that of a2 (beta = {beta:g}):\n\n"
        + _table(["step", "pi_theta(a0), chosen", "pi_theta(a2), rejected", "margin u"], shared, "rrrr"))


# ------------------------------------------------------------------ reinforcement learning from a reward with a KL penalty

KLD_OFFSET = (0.8, 0.3, -0.5, -0.2, 0.4)               # theta = theta_ref + offset for the gradient comparison
FD_STEP = 1e-6


def _kl_gradients(theta):
    """The five gradients of kl_direction: finite differences of the two KLs, and the exact expectations over the sampled
    answer y ~ pi of the three per-sample gradient estimators the sections discuss."""
    pi = _softmax(theta)

    def fd(f):
        out = []
        for k in range(K):
            up, dn = list(theta), list(theta)
            up[k] += FD_STEP
            dn[k] -= FD_STEP
            out.append((f(up) - f(dn)) / (2 * FD_STEP))
        return out

    rev = fd(lambda t: _kl(_softmax(t), PI_REF))
    fwd = fd(lambda t: _kl(PI_REF, _softmax(t)))
    k1, k3, shaped = [0.0] * K, [0.0] * K, [0.0] * K
    for y in range(K):
        score = [(1.0 if k == y else 0.0) - pi[k] for k in range(K)]            # e_y - pi, the gradient of ln pi(y)
        ratio = PI_REF[y] / pi[y]
        for k in range(K):
            k1[k] += pi[y] * 1.0 * score[k]                                       # sample-fixed gradient of ln(pi/pi_ref)
            k3[k] += pi[y] * (1.0 - ratio) * score[k]                             # sample-fixed gradient of ratio - 1 - ln ratio, ratio = pi_ref/pi
            shaped[k] += pi[y] * math.log(pi[y] / PI_REF[y]) * score[k]           # score times the penalty folded into the reward
    return rev, fwd, k1, k3, shaped


def _ascend_forward(r, beta, steps=200000, lr=0.5):
    """Gradient ascent on E r - beta KL(pi_ref || pi), the objective whose gradient the sample-fixed k3 loss follows: the
    exact gradient is pi_j (r_j - E r) - beta (pi_j - pi_ref_j)."""
    theta = list(THETA_REF)
    for _ in range(steps):
        pi = _softmax(theta)
        j_mean = _dot(pi, r)
        g = [pi[j] * (r[j] - j_mean) - beta * (pi[j] - PI_REF[j]) for j in range(K)]
        theta = [t + lr * d for t, d in zip(theta, g)]
        if max(abs(x) for x in g) < 1e-13:
            break
    return _softmax(theta)


def kl_gradient_direction():
    theta = [t + o for t, o in zip(THETA_REF, KLD_OFFSET)]
    rev, fwd, k1, k3, shaped = _kl_gradients(theta)
    names = ["gradient of KL(pi_theta, pi_ref), finite difference", "gradient of KL(pi_ref, pi_theta), finite difference",
             "mean over y ~ pi of the k1 term with y held fixed", "mean over y ~ pi of the k3 term with y held fixed",
             "mean over y ~ pi of (ln(pi/pi_ref)) times the score"]
    rows = [[n, *(_fmt(x, 5) for x in g)] for n, g in zip(names, (rev, fwd, k1, k3, shaped))]
    d_fwd = max(abs(a - b) for a, b in zip(k3, fwd))
    d_rev = max(abs(a - b) for a, b in zip(shaped, rev))
    pi = _softmax(theta)
    beta = PAIR_BETA
    star = _pi_star(R_TRUE, beta)
    settled = _ascend_forward(R_TRUE, beta)
    rows2 = [["pi* (KL in the reward; the closed form)", *(_fmt(x, 3) for x in star), _fmt(_dot(star, R_TRUE), 3), _fmt(_kl(star, PI_REF), 3)],
             ["where ascent with the k3 term settles", *(_fmt(x, 3) for x in settled), _fmt(_dot(settled, R_TRUE), 3), _fmt(_kl(settled, PI_REF), 3)]]
    return (
        f"The five answers of the KL-regularised objective, pi_ref = {_row(PI_REF, 2)}, at theta = theta_ref + {_row(KLD_OFFSET, 1)}, "
        f"where pi = {_row(pi, 4)}. The rows are gradients with respect to the five logits. The finite differences use the step "
        f"{FD_STEP:g} on the exact KLs (nats, summed in answer order). For the sampled-answer rows, the gradient of the per-sample "
        f"term with y held fixed is written with the score e_y - pi (the gradient of ln pi(y)): the k1 term ln(pi(y)/pi_ref(y)) has "
        f"the gradient (e_y - pi); the k3 term q - 1 - ln q with q = pi_ref(y)/pi(y) has the gradient (1 - q)(e_y - pi); and the "
        f"penalty folded into the reward multiplies the score by the number ln(pi(y)/pi_ref(y)) without differentiating it. Each "
        f"is averaged exactly over the five answers with the weights pi(y) (no sampling).\n\n"
        + _table(["quantity", "d/d theta_0", "d/d theta_1", "d/d theta_2", "d/d theta_3", "d/d theta_4"], rows, "lrrrrr")
        + f"\n\nThe k3 row equals the forward-KL row (largest difference {d_fwd:.1e}) and the folded row equals the reverse-KL row "
        f"(largest difference {d_rev:.1e}); the criterion for both is 1e-8. The k1 row is {_row(k1, 1)} up to rounding "
        f"(largest entry {max(abs(x) for x in k1):.1e}).\n\n"
        f"Where the two penalties lead, for beta = {beta:g} and the reward r = {_row(R_TRUE, 1)}: the policy at which the gradient "
        f"of E r - beta KL(pi || pi_ref) vanishes is the closed form pi*; the policy at which the gradient of E r - beta KL(pi_ref || "
        f"pi) vanishes was found by gradient ascent from theta_ref with the step 0.5 until every component was below 1e-13.\n\n"
        + _table(["policy", "pi(a0)", "pi(a1)", "pi(a2)", "pi(a3)", "pi(a4)", "E r", "KL of pi from pi_ref"], rows2, "lrrrrrrr"))


RLHF_BETA, RLHF_BATCH, RLHF_EPOCHS, RLHF_ETA, RLHF_EPS, RLHF_ITERS, RLHF_RUNS = 0.5, 64, 4, 0.5, 0.2, 200, 10
RLHF_CHECKS = (0, 10, 25, 50, 100, 200)
RLHF_VARIANTS = (("reward", "KL folded into the reward"), ("loss", "k3 term in the loss"), ("old", "k3 term against pi_old"))


def _rlhf_names(variant):
    return [f"llm_alignment.rlhf.{variant}.run{i}" for i in range(RLHF_RUNS)]


def _rlhf_run(variant, name, nudge):
    """One PPO run on the five answers: per iteration RLHF_BATCH answers y ~ pi_old (one uniform() each, inverse CDF), then
    RLHF_EPOCHS clipped steps on the batch. Returns {iteration: (J_beta(pi), KL(pi* || pi))} at the start of the iteration."""
    gen = bookrng.generator(name)
    beta = RLHF_BETA
    theta = list(THETA_REF)
    theta[0] += nudge
    star = _pi_star(R_TRUE, beta)
    out = {}
    for it in range(RLHF_ITERS + 1):
        pi_old = _softmax(theta)
        if it in RLHF_CHECKS:
            out[it] = (_objective(pi_old, R_TRUE, beta), _kl(star, pi_old))
        if it == RLHF_ITERS:
            break
        ys = [pg._sample_index(pi_old, gen) for _ in range(RLHF_BATCH)]
        if variant == "reward":
            rew = [R_TRUE[y] - beta * math.log(pi_old[y] / PI_REF[y]) for y in ys]
        else:
            rew = [R_TRUE[y] for y in ys]
        mean = 0.0
        for x in rew:
            mean += x
        mean /= RLHF_BATCH
        adv = [x - mean for x in rew]
        for _ in range(RLHF_EPOCHS):
            pi = _softmax(theta)
            grad = [0.0] * K
            for y, a in zip(ys, adv):
                rho = pi[y] / pi_old[y]
                if rho * a <= pg._clip(rho, RLHF_EPS) * a:
                    w = a * rho
                else:
                    w = 0.0
                if variant == "loss":
                    w += beta * (PI_REF[y] / pi[y] - 1.0)
                elif variant == "old":
                    w += beta * (pi_old[y] / pi[y] - 1.0)
                for k in range(K):
                    grad[k] += w * ((1.0 if k == y else 0.0) - pi[k]) / RLHF_BATCH
            theta = [t + RLHF_ETA * d for t, d in zip(theta, grad)]
    return out


_RLHF_CACHE = {}


def _rlhf_all():
    if not _RLHF_CACHE:
        for variant, _label in RLHF_VARIANTS:
            _RLHF_CACHE[variant] = [[_rlhf_run(variant, name, nudge) for name in _rlhf_names(variant)] for nudge in vl.VARIANTS]
    return _RLHF_CACHE


def rlhf_runs():
    runs = _rlhf_all()
    star = _pi_star(R_TRUE, RLHF_BETA)
    rows = []
    for it in RLHF_CHECKS:
        row = [it]
        for variant, _label in RLHF_VARIANTS:
            per_variant = runs[variant]
            for which in (0, 1):
                means = []
                for runs_v in per_variant:
                    acc = 0.0
                    for res in runs_v:
                        acc += res[it][which]
                    means.append(acc / RLHF_RUNS)
                row.append(vl._cell(means, 3 if which == 0 else 4))
        rows.append(row)
    seeds = _table(["run", "KL folded into the reward", "k3 term in the loss", "k3 term against pi_old"],
                   [[i] + [f"{bookrng.seed_of(_rlhf_names(v)[i]):#018x}" for v, _l in RLHF_VARIANTS] for i in range(RLHF_RUNS)], "rrrr")
    return (
        f"The five answers, pi_ref = {_row(PI_REF, 2)}, the labeller's reward r = {_row(R_TRUE, 1)} used as the reward model (so that "
        f"the only error is the algorithm's), beta = {RLHF_BETA:g}. The target is the closed-form optimum pi* = {_row(star, 4)}, with "
        f"J_beta(pi*) = {_fmt(_objective(star, R_TRUE, RLHF_BETA), 4)}. Each run starts from theta_ref and draws, per iteration, "
        f"{RLHF_BATCH} answers y_i from pi_old (one uniform() each, inverse CDF over pi_old in answer order, the generator of the run "
        f"continuing), forms the reward of each, subtracts the batch mean (summed in order, divisor {RLHF_BATCH}) to get the advantage "
        f"A_i, and takes {RLHF_EPOCHS} epochs of the clipped surrogate of the policy-gradient chapter on the whole batch (one minibatch, "
        f"in drawing order) with the step {RLHF_ETA:g} and the clip {RLHF_EPS:g}: the gradient on theta is the mean over the batch of w_i "
        f"(e_(y_i) - pi_theta) with w_i = A_i rho_i for a sample that the clip leaves (raw term not above clipped term) and 0 "
        f"otherwise, rho_i = pi_theta(y_i) / pi_old(y_i). Variant 'KL folded into the reward': the reward of a sample is "
        f"r(y) - beta ln(pi_old(y) / pi_ref(y)). Variant 'k3 term in the loss': the reward is r(y), and every sample adds "
        f"beta (pi_ref(y) / pi_theta(y) - 1) to its w_i, the weight of the gradient of the k3 term of the previous block with the "
        f"current policy in the denominator, so the KL is never in the advantage. Variant 'k3 term against pi_old' (the check of "
        f"the exercise on this block) is the previous one with pi_old(y) in the place of pi_ref(y), so that the penalty compares the "
        f"policy with the one that sampled the batch and not with the reference. {RLHF_RUNS} runs per variant, every run restarting from its "
        f"own seed (seeds below). 'J' is J_beta(pi) = E r - beta KL(pi || pi_ref) of the policy at the start of the iteration and "
        f"'KL to pi*' is KL(pi* || pi); each cell is the mean over the {RLHF_RUNS} runs (summed in run order).\n\n"
        + _table(["iteration", "J, KL in the reward", "KL to pi*, KL in the reward", "J, k3 in the loss", "KL to pi*, k3 in the loss",
                  "J, k3 against pi_old", "KL to pi*, k3 against pi_old"], rows, "rrrrrrr")
        + f"\n\n{RULE_TEXT}\n\n{seeds}")


# ------------------------------------------------------------------ group-relative advantages

ADV_EPS = 1e-4                                         # added to the standard deviation of a z-score
ADV_FLAT = 1e-6                                        # a group whose standard deviation is below this carries no comparison
ADV_GROUPS = ((1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), (0.2, 0.9, 0.4, 0.7, 0.1), (0.6, 0.5), (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.7,))


def _group_advantages(rewards, zscore):
    """The group-relative advantages: a group of one gets 1.0; a group whose population standard deviation is below
    ADV_FLAT gets 0.0 for every member; otherwise r - mean, divided by (std + ADV_EPS) when zscore."""
    n = len(rewards)
    if n <= 1:
        return [1.0] * n
    mean = 0.0
    for r in rewards:
        mean += r
    mean /= n
    var = 0.0
    for r in rewards:
        var += (r - mean) * (r - mean)
    std = math.sqrt(var / n)
    if std < ADV_FLAT:
        return [0.0] * n
    return [(r - mean) / (std + ADV_EPS) if zscore else r - mean for r in rewards]


def group_advantages():
    rows = []
    for rw in ADV_GROUPS:
        n = len(rw)
        mean = sum(rw) / n
        std = math.sqrt(sum((r - mean) ** 2 for r in rw) / n)
        sample_std = math.sqrt(sum((r - mean) ** 2 for r in rw) / (n - 1)) if n > 1 else float("nan")
        rows.append([_row(rw, 1), n, _fmt(mean, 3), _fmt(std, 4), "-" if n == 1 else _fmt(sample_std, 4),
                     _row(_group_advantages(rw, False), 3), _row(_group_advantages(rw, True), 3)])
    return (
        f"Seven groups of rewards for the completions of one prompt. For a group of n_g rewards the mean is the sum in order divided by "
        f"n_g, the standard deviation is the population one, the square root of (the sum of squared deviations from the mean, "
        f"in order) / n_g (the sample standard deviation, divisor n_g - 1, is printed for comparison and is not used). The "
        f"mean-centred advantage of a completion is r - mean; the z-score advantage is (r - mean) / (std + {ADV_EPS:g}). A group of "
        f"one completion has advantage 1.0, and a group of two or more whose standard deviation is below {ADV_FLAT:g} has "
        f"advantage 0.0 for every member under both rules.\n\n"
        + _table(["rewards", "n_g", "mean", "std (population)", "std (sample)", "mean-centred", "z-score"], rows, "lrrrrll"))


GS_GROUPS, GS_PROBS = (2, 4, 8, 16), (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98)


def _binomial(n, k, p):
    return math.comb(n, k) * p ** k * (1.0 - p) ** (n - k)


def _expected_group_gradient(n, p, zscore):
    """The exact expected gradient, with respect to the logit theta of a success probability p = sigmoid(theta), of the group
    estimate (1/n) sum_i A_i score_i, score = 1 - p for a success and -p for a failure, rewards 1 and 0, summed over the
    k = 0 .. n successes with their binomial probabilities."""
    total = 0.0
    for k in range(1, n):
        rewards = [1.0] * k + [0.0] * (n - k)
        adv = _group_advantages(rewards, zscore)
        g = (k * adv[0] * (1.0 - p) + (n - k) * adv[-1] * (-p)) / n
        total += _binomial(n, k, p) * g
    return total


def group_gradient_scale():
    rows = []
    for p in GS_PROBS:
        true = p * (1.0 - p)
        row = [f"{p:g}", _fmt(true, 4)]
        for n in GS_GROUPS:
            mc = _expected_group_gradient(n, p, False)
            row.append(f"{mc / true:.4f}")
        for n in (4, 8):
            zs = _expected_group_gradient(n, p, True)
            row.append(f"{zs / true:.3f}")
        for n in (4, 8):
            row.append(f"{p ** n + (1.0 - p) ** n:.3f}")
        rows.append(row)
    headers = (["p", "p(1 - p)"] + [f"mean-centred, n_g = {n}" for n in GS_GROUPS] + ["z-score, n_g = 4", "z-score, n_g = 8"]
               + ["flat group, n_g = 4", "flat group, n_g = 8"])
    return (
        f"A policy with one logit theta whose success probability is p = sigmoid(theta); a completion earns reward 1 with probability p and 0 "
        f"otherwise, so the expected reward is J = p and the exact gradient dJ/dtheta is p (1 - p). A group of n_g completions has k "
        f"successes with the binomial probability C(n_g, k) p^k (1 - p)^(n_g - k), k = 0 .. n_g; the estimate of the group is (1 / n_g) "
        f"times the sum over its completions of the advantage times the score d ln pi / d theta (which is 1 - p for a success and "
        f"-p for a failure), with the advantages of the previous block (a flat group, k = 0 or k = n_g, contributes 0). The "
        f"expected estimate is summed exactly over k = 0 .. n_g. The cells are the expected estimate divided by p (1 - p), "
        f"the factor by which the group estimate scales the true gradient; the last two columns are the probability "
        f"p^n_g + (1 - p)^n_g that a group is flat.\n\n"
        + _table(headers, rows, "r" * len(headers)))


# ------------------------------------------------------------------ verifiable rewards and their failure


def _last_number(text):
    """The last maximal run of digits, '.', ',' and '/' in the text that contains a digit (a leading '-' is kept), or None."""
    best = None
    i, n = 0, len(text)
    while i < n:
        if text[i].isdigit():
            j = i
            while j < n and (text[j].isdigit() or text[j] in ".,/"):
                j += 1
            best = text[i - 1:j] if i > 0 and text[i - 1] == "-" else text[i:j]
            i = j
        else:
            i += 1
    return best.rstrip(".,/") if best else None


def _rational(text):
    """An exact fraction from an integer, a decimal, a/b or \\frac{a}{b}, or None. Thousands commas are removed from a
    group of the form 1,234; anything else that does not parse is None."""
    from fractions import Fraction
    s = text.strip().strip("$").strip().rstrip(".")
    if s.startswith("\\frac{") and s.endswith("}") and "}{" in s:
        a, b = s[len("\\frac{"):-1].split("}{", 1)
        return _rational(a + "/" + b)
    s = s.replace(",", "") if all(len(g) == 3 for g in s.split(",")[1:]) else s
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return Fraction(a) / Fraction(b)
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _extract(completion):
    """The answer of a completion: the contents of the last \\boxed{...}, otherwise the last number."""
    at = completion.rfind("\\boxed{")
    if at >= 0:
        depth, j = 1, at + len("\\boxed{")
        start = j
        while j < len(completion) and depth:
            depth += {"{": 1, "}": -1}.get(completion[j], 0)
            j += 1
        return completion[start:j - 1] if depth == 0 else None
    return _last_number(completion)


def exact_check(completion, expected="42"):
    """1.0 when the extracted answer and the expected answer are the same exact rational number, otherwise 0.0."""
    got = _extract(completion)
    want = _rational(expected)
    return 1.0 if got is not None and want is not None and _rational(got) == want else 0.0


def weak_check(completion, expected="42"):
    """1.0 when the expected text occurs anywhere in the completion, otherwise 0.0."""
    return 1.0 if expected in completion else 0.0


VERIFIER_CASES = ("42", "The sum is 42.", "42.0", "84/2", "\\boxed{\\frac{84}{2}}", "\\boxed{42} (checked: 6 * 7)", "43", "I am not sure.",
                  "40 41 42 43 44 45", "42 or 43", "1,042", "1042", "")


def verifier_cases():
    rows = []
    for c in VERIFIER_CASES:
        ext = _extract(c)
        rows.append([f"`{c}`" if c else "(empty)", "none" if ext is None else f"`{ext}`", _fmt(exact_check(c), 1), _fmt(weak_check(c), 1)])
    consistent = all(exact_check(a) == v for a, v in zip(ANSWERS, EXACT_CHECK)) and all(weak_check(a) == v for a, v in zip(ANSWERS, WEAK_CHECK))
    return (
        "Two checkers for the prompt `17 + 25 =` whose expected answer is 42. The exact checker takes the contents of the last "
        "`\\boxed{...}` of the completion, or when there is none its last number (the last run of digits, '.', ',' and '/' that "
        "holds a digit, with a '-' directly before it kept and trailing punctuation dropped), reads both it and the expected "
        "answer as an exact rational (an integer, a decimal, `a/b` or `\\frac{a}{b}`, a thousands comma in a group of the form "
        "1,234 removed, anything else unreadable), and rewards 1 only when the two numbers are equal. The weak checker rewards 1 when "
        "the text 42 occurs anywhere in the completion.\n\n"
        + _table(["completion", "extracted answer", "exact checker", "weak checker"], rows, "llrr")
        + f"\n\nOn the five answers of the chapter the two checkers give the reward vectors {_row(EXACT_CHECK, 0)} and {_row(WEAK_CHECK, 0)} "
        f"(a0 .. a4), which is what the constants of the module hold: {consistent}.")


# ------------------------------------------------------------------ GRPO on the five answers, with the exact and the weak checker

GRPO_METHODS = (("exact", EXACT_CHECK, "exact checker"), ("weak", WEAK_CHECK, "weak checker"))


def _grpo_names(method):
    return [f"llm_alignment.grpo.{method}.run{i}" for i in range(GRPO_RUNS)]


def _grpo_run(check, name, nudge):
    """One GRPO run on the five answers: per step GROUP answers y_i ~ pi (one uniform() each, inverse CDF), rewards check[y_i],
    mean-centred advantages (0 for a flat group), and one step theta += lr (1/G) sum_i A_i (e_(y_i) - pi), one epoch per group
    so that the ratio is 1 and the clip never binds. Returns the exact (weak reward, exact accuracy, pi(a4)) at every step."""
    gen = bookrng.generator(name)
    theta = list(THETA_REF)
    theta[0] += nudge
    curve = []
    for step in range(GRPO_STEPS + 1):
        pi = _softmax(theta)
        curve.append((_dot(pi, WEAK_CHECK), _dot(pi, EXACT_CHECK), pi[4]))
        if step == GRPO_STEPS:
            break
        ys = [pg._sample_index(pi, gen) for _ in range(GROUP)]
        adv = _group_advantages([check[y] for y in ys], False)
        grad = [0.0] * K
        for y, a in zip(ys, adv):
            for k in range(K):
                grad[k] += a * ((1.0 if k == y else 0.0) - pi[k]) / GROUP
        theta = [t + GRPO_LR * d for t, d in zip(theta, grad)]
    return curve


_GRPO_CACHE = {}


def _grpo_all():
    if not _GRPO_CACHE:
        for method, check, _label in GRPO_METHODS:
            _GRPO_CACHE[method] = [[_grpo_run(check, name, nudge) for name in _grpo_names(method)] for nudge in vl.VARIANTS]
    return _GRPO_CACHE


def _grpo_mean(method, step, which, variant=0):
    acc = 0.0
    for curve in _grpo_all()[method][variant]:
        acc += curve[step][which]
    return acc / GRPO_RUNS


def grpo_checker_runs():
    runs = _grpo_all()
    rows = []
    for step in GRPO_CHECKS:
        row = [step]
        for method, _check, _label in GRPO_METHODS:
            for which in (0, 1, 2):
                row.append(vl._cell([_grpo_mean(method, step, which, v) for v in range(len(vl.VARIANTS))], 3))
        rows.append(row)
    seeds = _table(["run", "exact checker", "weak checker"],
                   [[i] + [f"{bookrng.seed_of(_grpo_names(m)[i]):#018x}" for m, _c, _l in GRPO_METHODS] for i in range(GRPO_RUNS)], "rrr")
    headers = ["step"] + [f"{label}: {what}" for _m, _c, label in GRPO_METHODS for what in ("weak reward", "exact accuracy", "pi(a4)")]
    return (
        f"The five answers, pi_ref = {_row(PI_REF, 2)} (theta from theta_ref = ln pi_ref), group size n_g = {GROUP}. Each step draws "
        f"{GROUP} answers from the current policy (one uniform() each, inverse CDF in answer order, the generator of the run "
        f"continuing), gives each the reward of the checker (the exact checker rewards a0 and a1, the weak one a0, a1 and a4), forms "
        f"mean-centred advantages (0 for a group whose rewards are all equal), and moves theta by {GRPO_LR:g} times the mean over the "
        f"group of A_i (e_(y_i) - pi); one update per group, so that the importance ratio is 1 and the clip never binds, and no KL "
        f"term. {GRPO_RUNS} runs per checker, every run restarting from its own seed (seeds below). The cells are the mean over "
        f"the {GRPO_RUNS} runs of three exact properties of the policy at the start of the step (not of the sampled group): "
        f"'weak reward' = pi(a0) + pi(a1) + pi(a4), 'exact accuracy' = pi(a0) + pi(a1), and pi(a4), the probability of the answer "
        f"that lists many numbers (summed in answer order).\n\n"
        + _table(headers, rows, "r" * len(headers)) + f"\n\n{RULE_TEXT}\n\n{seeds}")


# ------------------------------------------------------------------ the KL penalty, token by token

TOKEN_PREFIXES = ("", "0", "1", "00", "01", "10", "11")
TOKEN_P_REF = (0.50, 0.40, 0.60, 0.30, 0.50, 0.50, 0.70)    # pi_ref(next token = 1 | prefix), in the order of TOKEN_PREFIXES
TOKEN_P_POL = (0.65, 0.50, 0.75, 0.20, 0.60, 0.45, 0.85)    # pi(next token = 1 | prefix)
TOKEN_LEN = 3


def _tok_prob(table, prefix, token):
    p1 = table[TOKEN_PREFIXES.index(prefix)]
    return p1 if token == "1" else 1.0 - p1


def _sequences():
    out = [""]
    for _ in range(TOKEN_LEN):
        out = [s + t for s in out for t in "01"]
    return out


def token_kl_estimators():
    seqs = _sequences()
    rows, kl, e1, e3, s1, s3 = [], 0.0, 0.0, 0.0, 0.0, 0.0
    acc = []
    for sq in seqs:
        pol, ref, k1, k3 = 1.0, 1.0, 0.0, 0.0
        for t in range(TOKEN_LEN):
            a, b = _tok_prob(TOKEN_P_POL, sq[:t], sq[t]), _tok_prob(TOKEN_P_REF, sq[:t], sq[t])
            pol *= a
            ref *= b
            rho = b / a
            k1 += math.log(a / b)
            k3 += rho - 1.0 - math.log(rho)
        acc.append((sq, pol, ref, k1, k3))
        kl += pol * math.log(pol / ref)
    for (_sq, pol, _ref, k1, k3) in acc:
        e1 += pol * k1
        e3 += pol * k3
    for (_sq, pol, _ref, k1, k3) in acc:
        s1 += pol * (k1 - e1) ** 2
        s3 += pol * (k3 - e3) ** 2
    for (sq, pol, ref, k1, k3) in acc:
        rows.append([sq, _fmt(pol, 4), _fmt(ref, 4), _fmt(k1, 4), _fmt(k3, 4)])
    chain, total = [], 0.0
    for pre in TOKEN_PREFIXES:
        prob = 1.0
        for t in range(len(pre)):
            prob *= _tok_prob(TOKEN_P_POL, pre[:t], pre[t])
        a, b = _tok_prob(TOKEN_P_POL, pre, "1"), _tok_prob(TOKEN_P_REF, pre, "1")
        local = a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
        total += prob * local
        chain.append([f"`{pre}`" if pre else "(empty)", _fmt(prob, 4), _fmt(TOKEN_P_REF[TOKEN_PREFIXES.index(pre)], 2),
                      _fmt(TOKEN_P_POL[TOKEN_PREFIXES.index(pre)], 2), _fmt(local, 4), _fmt(prob * local, 4)])
    return (
        f"A language model over the two tokens 0 and 1 that writes exactly {TOKEN_LEN} tokens (no end token), so there are "
        f"{2 ** TOKEN_LEN} sequences. The reference pi_ref and the policy pi are given by the probability of the token 1 after each "
        f"of the 7 prefixes (the probability of the token 0 is the complement):\n\n"
        + _table(["prefix", "P(prefix under pi)", "pi_ref(1 after prefix)", "pi(1 after prefix)", "KL of pi from pi_ref after the prefix", "weighted by the prefix"], chain, "lrrrrr")
        + f"\n\nThe weighted column adds to {total:.4f}, the exact KL(pi || pi_ref) of the whole sequence ({kl:.4f}; sums over the "
        f"sequences in the order below, natural logarithms): the chain rule of the KL, the KL after each prefix averaged over the "
        f"prefixes that pi reaches. For a sequence y_1 y_2 y_3 drawn from pi, the estimator k1 sums ln(pi(y_t | y_<t) / pi_ref(y_t | "
        f"y_<t)) over the three tokens, and k3 sums rho_t - 1 - ln rho_t with rho_t = pi_ref(y_t | y_<t) / pi(y_t | y_<t); the "
        f"means and variances over pi's eight sequences are exact (the variance has divisor 1, an expectation).\n\n"
        + _table(["sequence", "pi(sequence)", "pi_ref(sequence)", "k1 summed", "k3 summed"], rows, "lrrrr")
        + f"\n\nMean of k1: {e1:.4f}, mean of k3: {e3:.4f}, exact KL {kl:.4f}. Standard deviation of k1: {math.sqrt(s1):.4f}, of k3: "
        f"{math.sqrt(s3):.4f}. The smallest k1 is {min(a[3] for a in acc):.4f} and the smallest k3 is {min(a[4] for a in acc):.4f}. A loss that "
        f"averages the per-token terms over the {TOKEN_LEN} tokens instead of summing them has the mean {e3 / TOKEN_LEN:.4f}, "
        f"{TOKEN_LEN} times smaller.")


# ------------------------------------------------------------------ what the methods cost

COST_P, COST_PR, COST_PV, COST_TP, COST_TC, COST_G = 10 ** 9, 10 ** 9, 10 ** 9, 128, 512, 8


def _sci(x):
    return f"{x:.3e}"


def alignment_costs():
    p, pr, pv, tp, tc, g = COST_P, COST_PR, COST_PV, COST_TP, COST_TC, COST_G
    t = tp + tc
    rows = [
        ["supervised fine-tuning, one example of T tokens, one step", "6 P T", _sci(6 * p * t)],
        ["sampling G completions (prefill once, decode each)", "2 P (T_p + G T_c)", _sci(2 * p * (tp + g * tc))],
        ["reference log-probabilities of G completions", "2 P T G", _sci(2 * p * t * g)],
        ["reward model on G completions", "2 P_r T G", _sci(2 * pr * t * g)],
        ["critic of PPO, one epoch on G completions", "6 P_v T G", _sci(6 * pv * t * g)],
        ["policy update on G completions, one epoch (PPO and GRPO)", "6 P T G", _sci(6 * p * t * g)],
        ["DPO, reference log-probabilities of one pair, once", "2 P (2 T)", _sci(2 * p * 2 * t)],
        ["DPO, update on one pair, one epoch", "6 P (2 T)", _sci(6 * p * 2 * t)],
    ]
    ppo = 2 * p * (tp + g * tc) + 2 * p * t * g + 2 * pr * t * g + 6 * pv * t * g + 6 * p * t * g
    grpo = 2 * p * (tp + g * tc) + 2 * p * t * g + 6 * p * t * g
    dpo = 2 * p * 2 * t + 6 * p * 2 * t
    totals = [["RLHF with PPO (sampling, reference, reward model, critic, policy; one epoch)", _sci(ppo)],
              ["GRPO with a program as the reward (sampling, reference, policy; one epoch)", _sci(grpo)],
              [f"DPO on {g} pairs (the reference scored once for each pair, then the update; one epoch)", _sci(g * dpo)]]
    mem = [["RLHF with PPO", "policy, reference, reward model, critic", 2 * (p + p + pr + pv)],
           ["GRPO", "policy, reference", 2 * (p + p)],
           ["DPO", "policy, reference", 2 * (p + p)]]
    return (
        f"The convention of the chapter on training a GPT: a forward pass costs 2 FLOPs per parameter per token and the backward pass "
        f"4, so a step on T tokens of a model of P parameters costs 6 P T, and a token decoded with a key-value cache about 2 P "
        f"(attention over the context, a few percent, is left out). Setting: P = P_r = P_v = {p:.0e}, a prompt of T_p = {tp} tokens, "
        f"a completion of T_c = {tc} tokens, so T = T_p + T_c = {t}, and G = {g} completions per prompt. One FLOP is one "
        f"multiplication or addition.\n\n"
        + _table(["piece", "FLOPs", "value for the setting"], rows, "llr")
        + "\n\nSums for one prompt with its G completions (or G pairs for DPO):\n\n"
        + _table(["method", "FLOPs"], totals, "lr")
        + f"\n\nBytes of weights held, two bytes per parameter (bf16), optimiser state and activations not counted:\n\n"
        + _table(["method", "models held", "bytes"], [[a, b, f"{c:.1e}"] for a, b, c in mem], "llr"))
