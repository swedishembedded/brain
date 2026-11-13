# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the Volume XII project "An Agent That Improves from Verified Feedback".

Plain Python 3, standard library only, written from the project's own description. The libraries of the earlier chapters are
imported, never copied: the softmax, the inverse-CDF draw and the clip of the clipped surrogate (`policy_gradients`, Chapter 3),
the group-relative advantage (`llm_alignment`, Chapter 4) and the one-sided Gaussian point of the paired lower bound
(`continual_learning`, Chapter 7). What this module writes is what makes them one agent:

* a family of tasks (shift every digit of a five-digit input by its position plus one, modulo 3) with an exact verifier, a weak
  verifier that compares every digit modulo 2, and a difficulty knob, the number of output digits that must all be right;
* a policy of 240 numbers that reads the whole input and emits the digits one after another, trained by GRPO with the clipped
  surrogate on groups of 8 verified answers, with a curriculum that raises the level when the success on a probe bank passes a
  threshold;
* a probe bank spanning the five levels and an audit bank, both disjoint from the training pool by construction;
* snapshots every 10 steps that a promotion gate (a paired lower bound with a non-inferiority margin on every level) admits or
  refuses, a registry of promoted versions, an audit after each promotion and a rollback.

Arithmetic is 64-bit floating point; every random input comes from `bookrng.generator(name)`. Every number that comes out of a
training run goes through `stable`: the run is repeated with the first stored number changed by each of the nudges of
`stable.py`, and a digit is printed only if the run and its four copies agree on it.

Swedish Embedded AB implements solutions for training agents on verifiable rewards and for deciding, with a stated test, which
version may be served. If your team needs expertise in reinforcement learning from verified feedback, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import hashlib
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import continual_learning as cl
import llm_alignment as la
import policy_gradients as pg
import stable
import value_learning as vl

# ------------------------------------------------------------------ the world, the policy and the loop

RADIX = 3                               # a digit is 0, 1 or 2
N_IN = 5                                # digits of an input
LEVELS = 5                              # levels 1 .. 5: the number of output digits that must all be right
FEAT = 1 + N_IN * RADIX                 # features of an input: a constant and one indicator per (position, digit)
N_PARAM = LEVELS * RADIX * FEAT         # 5 * 3 * 16 = 240
N_INPUTS = RADIX ** N_IN                # 243 distinct inputs
POOL, PROBES, AUDIT = 24, 48, 48        # training inputs, probe inputs and audit inputs (disjoint slices of one shuffle)
TASKS_PER_STEP, GROUP = 4, 8            # B tasks, n_g answers each
EPOCHS, ETA, EPS = 2, 2.0, 0.2          # epochs per batch, step size, half-width of the clip band
TAU = 0.6                               # the curriculum threshold
STEPS, EVERY = 300, 10                  # training steps and steps between snapshots
DELTA = 0.02                            # the non-inferiority margin of the gate
Z = cl.Z_ONE_SIDED                      # 1.644854, the one-sided 0.95 point of the Gaussian
SPLIT = "project_agent.split"
DEMO = "project_agent.demo"
STEP_DEMO = "project_agent.step"
BUG = "project_agent.bug"
RUNS = 5
BUG_STEPS, BUG_AT = 20, 100             # the faulty-generator candidate: steps trained, and the snapshot it is trained from
VARIANTS = vl.VARIANTS                  # the run itself, then four copies with the first stored number nudged
CONFIGS = ("spec", "weak", "train", "fixed")


def _run_name(seed):
    return f"project_agent.run{seed}"


# ------------------------------------------------------------------ tasks and verifiers

def digits(m):
    """The five digits of the input number m, least significant first."""
    return [(m // RADIX ** i) % RADIX for i in range(N_IN)]


def target(x):
    """The target digit j of an input is (x_j + j + 1) mod 3."""
    return [(x[j] + j + 1) % RADIX for j in range(N_IN)]


def verify_exact(out, t, level):
    """Reward 1 when the answer has `level` digits and every one equals the target digit, else 0."""
    return 1.0 if len(out) == level and all(out[j] == t[j] for j in range(level)) else 0.0


def verify_weak(out, t, level):
    """The planted verifier: reward 1 when the answer has `level` digits and each has the parity of the target digit."""
    return 1.0 if len(out) == level and all(out[j] % 2 == t[j] % 2 for j in range(level)) else 0.0


def _split():
    order = bookrng.generator(SPLIT).shuffle(list(range(N_INPUTS)))
    return order[:PROBES], order[PROBES:PROBES + AUDIT], order[PROBES + AUDIT:PROBES + AUDIT + POOL]


PROBE_IN, AUDIT_IN, POOL_IN = _split()


# ------------------------------------------------------------------ the policy

def logits(theta, x, j):
    """The three logits of output position j: the constant weight plus the weight of each (position, digit) indicator of x."""
    out = []
    for c in range(RADIX):
        base = (j * RADIX + c) * FEAT
        z = theta[base]
        for i in range(N_IN):
            z += theta[base + 1 + RADIX * i + x[i]]
        out.append(z)
    return out


def probs(theta, x, j):
    return pg._softmax(logits(theta, x, j))


def success(theta, x, level):
    """The probability that the policy emits the first `level` target digits: the product of the target digits' probabilities."""
    t = target(x)
    p = 1.0
    for j in range(level):
        p *= probs(theta, x, j)[t[j]]
    return p


def bank_scores(theta, inputs):
    """The success probability of every input at every level: scores[level - 1][k]."""
    return [[success(theta, digits(m), level) for m in inputs] for level in range(1, LEVELS + 1)]


def probe_success(theta, inputs, level):
    acc = 0.0
    for m in inputs:
        acc += success(theta, digits(m), level)
    return acc / len(inputs)


def sample_answer(theta, x, level, rng):
    return [pg._sample_index(probs(theta, x, j), rng) for j in range(level)]


# ------------------------------------------------------------------ one GRPO step with the clipped surrogate

def train_step(theta, level, rng, verify=verify_exact, record=None, ascent=1.0):
    """B tasks drawn from the pool, n_g answers each, group advantages, EPOCHS epochs of the clipped surrogate.

    Returns (theta, mean reward of the step's samples, number of flat groups). `record`, when given, collects the batch."""
    tasks = [POOL_IN[rng.integer(POOL)] for _ in range(TASKS_PER_STEP)]
    batch = []
    rewards_all = []
    flat = 0
    for m in tasks:
        x = digits(m)
        t = target(x)
        answers = [sample_answer(theta, x, level, rng) for _ in range(GROUP)]
        rewards = [verify(a, t, level) for a in answers]
        adv = la._group_advantages(rewards, False)
        if all(a == 0.0 for a in adv):
            flat += 1
        rewards_all += rewards
        for a, r, A in zip(answers, rewards, adv):
            batch.append((x, a, A))
        if record is not None:
            record["groups"].append((m, t[:level], answers, rewards, adv))
    old = [[probs(theta, x, j)[a[j]] for j in range(level)] for x, a, _A in batch]
    n_tok = len(batch) * level
    for e in range(EPOCHS):
        grad = [0.0] * N_PARAM
        clipped = 0
        lo, hi = 1.0, 1.0
        before = list(theta)
        for (x, a, A), p_old in zip(batch, old):
            if A == 0.0:
                continue
            for j in range(level):
                pi = probs(theta, x, j)
                rho = pi[a[j]] / p_old[j]
                lo, hi = min(lo, rho), max(hi, rho)
                raw = rho * A
                if raw > pg._clip(rho, EPS) * A:
                    clipped += 1
                    continue
                w = A * rho / n_tok
                for c in range(RADIX):
                    g = ((1.0 if c == a[j] else 0.0) - pi[c]) * w
                    base = (j * RADIX + c) * FEAT
                    grad[base] += g
                    for i in range(N_IN):
                        grad[base + 1 + RADIX * i + x[i]] += g
        theta = [t_ + ascent * ETA * g for t_, g in zip(theta, grad)]
        if record is not None:
            record["epochs"].append({"clipped": clipped, "rho_min": lo, "rho_max": hi,
                                     "delta_l1": sum(abs(a - b) for a, b in zip(theta, before)), "n_tok": n_tok})
    reward = 0.0
    for r in rewards_all:
        reward += r
    return theta, reward / len(rewards_all), flat


# ------------------------------------------------------------------ the training run with its curriculum and snapshots

class Run:
    """One training run: the per-step record, the advances of the curriculum and the snapshots every EVERY steps."""

    def __init__(self):
        self.init = None
        self.theta = None
        self.levels = []          # the level used at step s (index s - 1)
        self.train = []           # the mean reward of the samples of step s
        self.flat = []            # the number of flat groups of step s
        self.probe = []           # the probe success at the level of step s, after the update of step s
        self.advances = []        # (step, level left, probe success after the step, training success of the step)
        self.snaps = []           # (step, level at the snapshot, theta, probe scores [level][k])


def run(config, seed, nudge):
    """The run of `config` ("spec", "weak", "train" or "fixed") from the generator of `seed`, the first number nudged."""
    rng = bookrng.generator(_run_name(seed))
    theta = [0.0] * N_PARAM
    theta[0] += nudge
    r_init = list(theta)
    verify = verify_weak if config == "weak" else verify_exact
    level = LEVELS if config == "fixed" else 1
    r = Run()
    r.init = r_init
    for step in range(1, STEPS + 1):
        theta, train, flat = train_step(theta, level, rng, verify)
        ps = probe_success(theta, PROBE_IN, level)
        r.levels.append(level)
        r.train.append(train)
        r.flat.append(flat)
        r.probe.append(ps)
        signal = train if config == "train" else ps
        if config != "fixed" and level < LEVELS and signal >= TAU:
            r.advances.append((step, level, ps, train))
            level += 1
        if step % EVERY == 0:
            r.snaps.append((step, level, list(theta), bank_scores(theta, PROBE_IN)))
    r.theta = theta
    return r


def _run_one(args):
    return run(*args)


@lru_cache(maxsize=None)
def runs(config, seed):
    """The run and its four nudged copies."""
    args = [(config, seed, nudge) for nudge in VARIANTS]
    with get_context("fork").Pool(min(4, len(args))) as pool:
        return pool.map(_run_one, args)


# ------------------------------------------------------------------ the gate, the registry, the audit

def paired_lower(cand, inc):
    """(mean difference, paired standard error, one-sided 0.95 lower bound) of two score lists over the same probes."""
    n = len(cand)
    d = [c - i for c, i in zip(cand, inc)]
    m = sum(d) / n
    s2 = sum((v - m) * (v - m) for v in d) / (n - 1)
    se = math.sqrt(s2 / n)
    return m, se, m - Z * se


def unpaired_lower(cand, inc):
    """The same, with the standard error of two independent means: sqrt(var(cand) / n + var(inc) / n), divisor n - 1."""
    n = len(cand)
    mc, mi = sum(cand) / n, sum(inc) / n
    vc = sum((v - mc) * (v - mc) for v in cand) / (n - 1)
    vi = sum((v - mi) * (v - mi) for v in inc) / (n - 1)
    se = math.sqrt(vc / n + vi / n)
    return mc - mi, se, mc - mi - Z * se


def non_inferior(sc_c, sc_i, bound=paired_lower):
    """The first level (1 .. 5) whose lower bound is not above -DELTA, or 0 when every level is non-inferior."""
    for level in range(1, LEVELS + 1):
        if not bound(sc_c[level - 1], sc_i[level - 1])[2] > -DELTA:
            return level
    return 0


def gate(sc_c, sc_i, level, bound=paired_lower):
    """(decision, cause) of a snapshot at `level` against the incumbent: non-inferior on every level, then a gain on `level`."""
    bad = non_inferior(sc_c, sc_i, bound)
    if bad:
        return "reject", f"level {bad} regressed"
    if not bound(sc_c[level - 1], sc_i[level - 1])[2] > 0.0:
        return "reject", "gain not shown"
    return "promote", None


def serve(run_, bound=paired_lower):
    """Walk the snapshots in order: gate each against the incumbent, audit each promotion, roll back one that fails the audit.

    Returns one record per snapshot and the registry (list of (step, theta, probe scores)) with the index served at the end."""
    zero = run_.init                     # version 0 is the untrained policy the run started from
    registry = [(0, zero, bank_scores(zero, PROBE_IN), bank_scores(zero, AUDIT_IN))]
    served = 0
    records = []
    for step, level, theta, scores in run_.snaps:
        decision, cause = gate(scores, registry[served][2], level, bound)
        lag = sum(scores[level - 1]) / PROBES - sum(registry[served][2][level - 1]) / PROBES
        rec = {"step": step, "level": level, "decision": decision, "cause": cause, "lag": lag, "served_before": served,
               "gain": bound(scores[level - 1], registry[served][2][level - 1]), "audit": None, "rolled_back": False}
        if decision == "promote":
            audit = bank_scores(theta, AUDIT_IN)
            rec["audit"] = non_inferior(audit, registry[served][3])
            registry.append((step, theta, scores, audit))
            if rec["audit"] == 0:
                served = len(registry) - 1
            else:
                rec["rolled_back"] = True
        records.append(rec)
    return records, registry, served


# ------------------------------------------------------------------ formatting

def _table(headers, rows, align=None):
    return vl._table(headers, rows, align)


def _cell(vals, nd):
    return vl._cell(vals, nd)


def _int_cell(vals):
    return str(vals[0]) if all(v == vals[0] for v in vals) else "unstable"


def _text_cell(vals):
    return str(vals[0]) if all(v == vals[0] for v in vals) else "unstable"


def _seed_line(name):
    return f"`{name}` (seed {bookrng.seed_of(name):#018x}, {bookrng.seed_of(name)})"


def _digits_text(v):
    return "".join(str(d) for d in v)


RULE_TEXT = ("Every number learned from samples is printed with the digits that the run and four copies of it share: the copies "
             "start with the first stored number of the policy, theta[0], changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the nudges "
             "of stable.py), and a digit is printed only if all five agree on it ('about x' keeps the leading digits that survive, "
             "'unstable' means that none does; a count or a decision is printed only if all five agree on it). The copies use the "
             "same generator and so draw the same numbers; they differ only if a nudge flips a comparison. A run is a deterministic "
             "computation, so another implementation reproduces the printed digits.")


# ------------------------------------------------------------------ milestone 1: the tasks and the verifiers

def tasks():
    probes = _digits_text
    rows = []
    for label, ms in (("pool", POOL_IN[:3]), ("probe", PROBE_IN[:3]), ("audit", AUDIT_IN[:2])):
        for m in ms:
            x = digits(m)
            rows.append([label, m, probes(x), probes(target(x))])
    split = _table(["slice", "size", "first entries (the shuffled order)", "digest of all its numbers"],
                   [["probes", PROBES, ", ".join(map(str, PROBE_IN[:6])), _digest_numbers(PROBE_IN)],
                    ["audit", AUDIT, ", ".join(map(str, AUDIT_IN[:6])), _digest_numbers(AUDIT_IN)],
                    ["pool", POOL, ", ".join(map(str, POOL_IN[:6])), _digest_numbers(POOL_IN)]], "lrll")
    table = _table(["slice", "input m", "digits x_0 .. x_4", "target t_0 .. t_4"], rows, "lrll")
    return (
        f"The family: an input is a number m in 0 .. {N_INPUTS - 1} and its five digits are x_i = floor(m / 3^i) mod 3 for i = 0 .. 4, "
        f"least significant first. The target digit j is t_j = (x_j + j + 1) mod 3 (the planted generator bug of milestone 8 uses "
        f"(x_j + j) mod 3). The generator {_seed_line(SPLIT)} shuffles the list 0, 1, ..., {N_INPUTS - 1} with the book's shuffle "
        f"(i from {N_INPUTS - 1} down to 1, j = integer(i + 1), swap positions i and j); the first {PROBES} numbers of the shuffled "
        f"list are the probe inputs, the next {AUDIT} the audit inputs and the next {POOL} the training pool, and the remaining "
        f"{N_INPUTS - PROBES - AUDIT - POOL} are not used. The pool in order: {', '.join(map(str, POOL_IN))}. The digest of a slice "
        f"is the SHA-256 of its numbers written in decimal and joined by a comma, in the order of the slice (first 16 hex digits).\n\n"
        + split + "\n\n" + table)


def _digest_numbers(ms):
    return hashlib.sha256(",".join(str(m) for m in ms).encode("utf-8")).hexdigest()[:16]


def verifier_cases():
    m = PROBE_IN[0]
    x = digits(m)
    t = target(x)
    level = 4
    t4 = t[:level]

    def alter(j, delta):
        out = list(t4)
        out[j] = (out[j] + delta) % RADIX
        return out
    cases = [("the target", list(t4)),
             ("digit 0 changed by +1 (the parity changes unless the digit goes 2 to 0)", alter(0, 1)),
             ("digit 1 changed by +2 (the parity is kept only when the digit goes 0 to 2)", alter(1, 2)),
             ("digit 3 changed by +1", alter(3, 1)),
             ("the first three digits only", list(t4[:3])),
             ("the input digits unchanged", list(x[:level]))]
    rows = []
    for label, out in cases:
        rows.append([label, _digits_text(out), f"{verify_exact(out, t, level):.0f}", f"{verify_weak(out, t, level):.0f}"])
    return (
        f"The input m = {m} (the first probe input), digits {_digits_text(x)}, target {_digits_text(t)}, at level {level}: the policy must "
        f"emit the first {level} target digits, {_digits_text(t4)}. The exact verifier rewards 1 when all {level} output digits equal the target digits "
        f"and 0 otherwise (an answer of any other length is 0, as in the row of three digits). The weak verifier rewards 1 when every output digit has the same "
        f"parity as the target digit (digit mod 2); it accepts every answer the exact verifier accepts and others besides.\n\n"
        + _table(["output", "digits", "exact reward", "weak reward"], rows, "lrrr"))


# ------------------------------------------------------------------ milestone 2: the policy

def _check_theta():
    return [((37 * k) % 101 - 50) / 500.0 for k in range(N_PARAM)]


def _log_prob(theta, x, level, answer):
    s = 0.0
    for j in range(level):
        s += math.log(probs(theta, x, j)[answer[j]])
    return s


def _analytic_gradient(theta, x, level, answer):
    g = [0.0] * N_PARAM
    for j in range(level):
        pi = probs(theta, x, j)
        for c in range(RADIX):
            d = (1.0 if c == answer[j] else 0.0) - pi[c]
            base = (j * RADIX + c) * FEAT
            g[base] += d
            for i in range(N_IN):
                g[base + 1 + RADIX * i + x[i]] += d
    return g


def policy():
    zero = [0.0] * N_PARAM
    m = POOL_IN[0]
    x = digits(m)
    rows = []
    for level in range(1, LEVELS + 1):
        rows.append([level, f"{success(zero, x, level):.6f}", f"{3.0 ** -level:.6f}", f"{probe_success(zero, PROBE_IN, level):.6f}"])
    theta = _check_theta()
    level, answer = 3, [2, 0, 1]
    g = _analytic_gradient(theta, x, level, answer)
    h = 1e-6
    worst = 0.0
    for k in range(N_PARAM):
        up, dn = list(theta), list(theta)
        up[k] += h
        dn[k] -= h
        fd = (_log_prob(up, x, level, answer) - _log_prob(dn, x, level, answer)) / (2 * h)
        worst = max(worst, abs(fd - g[k]) / max(1.0, abs(g[k])))
    touched = sum(1 for v in g if v != 0.0)
    rng = bookrng.generator(DEMO)
    us = []
    draws = []
    for _ in range(6):
        u = rng.uniform()
        us.append(u)
        draws.append(0 if u < 1 / 3 else (1 if u < 2 / 3 else 2))
    p = probs(zero, x, 0)
    return (
        f"The policy has {N_PARAM} = {LEVELS} positions x {RADIX} digits x {FEAT} features numbers, all {N_PARAM} initialised to 0.0: "
        f"theta[(j * 3 + c) * 16 + f] is the weight of feature f for digit c at output position j, and the logit of digit c at position j "
        f"is theta[(j*3+c)*16] + sum over i = 0 .. 4 of theta[(j*3+c)*16 + 1 + 3i + x_i] (the sum accumulated in increasing i, the constant first). "
        f"The distribution of position j is the stable softmax of its three logits. At theta = 0 every position is uniform over the "
        f"three digits, so the probability of the first `level` target digits is 3^-level.\n\n"
        + _table(["level", "success of input m = " + str(m), "3^-level", "mean over the 48 probe inputs"], rows, "rrrr")
        + f"\n\nA digit is drawn by the inverse CDF of Chapter 3: one `uniform()` from the generator, the first digit whose running "
        f"sum of probabilities exceeds it. At theta = 0 the first six draws of the generator {_seed_line(DEMO)} are "
        f"{', '.join(f'{u:.6f}' for u in us)}, which give the digits {', '.join(map(str, draws))} (the thresholds are 1/3 and 2/3; "
        f"the position-0 probabilities of input {m} are {', '.join(f'{v:.6f}' for v in p)}).\n\n"
        f"Gradient check. For the starting point theta_k = ((37 k) mod 101 - 50) / 500 for k = 0 .. {N_PARAM - 1}, the input m = {m}, level 3 and "
        f"the answer 2, 0, 1, the gradient of ln pi(answer | x) = the sum over positions of ln pi_j(answer_j) has {touched} nonzero "
        f"entries (3 positions x 3 digits x 6 features, the constant and the five indicators that are 1); the largest "
        f"relative difference |fd - g| / max(1, |g|) between the analytic gradient g and a central finite difference fd with step 1e-6, over all {N_PARAM} numbers, "
        f"is {'below 1e-6' if worst < 1e-6 else 'NOT below 1e-6'}.")


# ------------------------------------------------------------------ milestone 3: one step

def one_step():
    theta = [0.0] * N_PARAM
    rng = bookrng.generator(STEP_DEMO)
    level = 2
    rec = {"groups": [], "epochs": []}
    new, train, flat = train_step(theta, level, rng, verify_exact, rec)
    rows = []
    for m, t, answers, rewards, adv in rec["groups"]:
        k = int(sum(rewards))
        a_right = f"{1 - k / GROUP:+.3f}" if 0 < k < GROUP else "0"
        a_wrong = f"{-k / GROUP:+.3f}" if 0 < k < GROUP else "0"
        rows.append([m, _digits_text(digits(m)), _digits_text(t), " ".join(_digits_text(a) for a in answers), k, a_right, a_wrong])
    eps = rec["epochs"]
    epoch_rows = [[e + 1, f"{v['rho_min']:.4f}", f"{v['rho_max']:.4f}", v["clipped"], f"{v['delta_l1']:.6f}"] for e, v in enumerate(eps)]
    before = probe_success(theta, PROBE_IN, level)
    after = probe_success(new, PROBE_IN, level)
    return (
        f"One step from theta = 0 at level {level}. The generator is {_seed_line(STEP_DEMO)}, used as the training run uses it: first "
        f"B = {TASKS_PER_STEP} draws `integer(24)` choose the pool entries (with replacement), then for each task in order and each of its "
        f"{GROUP} answers in order, level = {level} draws `uniform()`, one per output position. An answer is the digits that come out. "
        f"The group advantage is r - mean(r) (0 for a group whose rewards are all equal); every token of an answer carries the advantage "
        f"of its answer. The loss divides by the number of tokens of the whole batch, n_tok = B x n_g x level = "
        f"{TASKS_PER_STEP * GROUP * level} (flat groups included). Two epochs, step size {ETA:g}, clip band 1 +- {EPS}.\n\n"
        + _table(["task m", "digits", "target (level 2)", f"the {GROUP} answers", "right", "advantage of a right answer",
                  "advantage of a wrong one"], rows, "rlllrrr")
        + f"\n\nThe mean reward of the step is {train:.4f} and {flat} of the {TASKS_PER_STEP} groups are flat. The two epochs, with the "
        f"importance ratio of every token with a nonzero advantage against the policy at the start of the step (1.0 exactly in epoch 1), the "
        f"tokens whose surrogate is clipped, and the L1 change of the {N_PARAM} numbers:\n\n"
        + _table(["epoch", "smallest ratio", "largest ratio", "tokens clipped", "L1 change of theta"], epoch_rows, "rrrrr")
        + f"\n\nThe mean probe success at level {level} goes from {before:.6f} (= 1/9) to {after:.6f}.")


# ------------------------------------------------------------------ milestone 4: the probe bank

def _bank_lines():
    return [f"{level}|{m}|{_digits_text(target(digits(m))[:level])}" for level in range(1, LEVELS + 1) for m in PROBE_IN]


def probe_bank():
    lines = _bank_lines()
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]
    flipped = list(lines)
    first = flipped[0].split("|")
    first[2] = str((int(first[2]) + 1) % RADIX)
    flipped[0] = "|".join(first)
    digest2 = hashlib.sha256("\n".join(flipped).encode("utf-8")).hexdigest()[:16]
    zero = [0.0] * N_PARAM
    rows = [[lv, PROBES, f"{probe_success(zero, PROBE_IN, lv):.6f}", f"{probe_success(zero, AUDIT_IN, lv):.6f}"] for lv in range(1, LEVELS + 1)]
    return (
        f"The probe bank has {LEVELS} x {PROBES} = {LEVELS * PROBES} probes: for level 1 .. 5 and each probe input in the order of the shuffled "
        f"list, one line `level|m|target digits of the first `level` positions` (for example `{lines[0]}` and `{lines[-1]}`). Its identity is the SHA-256 "
        f"of the {len(lines)} lines joined by a line feed (no final one), first 16 hex digits: {digest}; with the target digit of the first line "
        f"increased by 1 (mod 3) it is {digest2}. The audit bank is the same construction over the audit inputs. At theta = 0 the success of every "
        f"probe is exactly 3^-level, so each bank's mean is too:\n\n"
        + _table(["level", "probes", "mean success of the probe bank at theta = 0", "of the audit bank"], rows, "rrrr"))


# ------------------------------------------------------------------ milestone 5: the curriculum run

def _levels_after(r):
    return r.levels[-1] + (1 if r.advances and r.advances[-1][0] == STEPS else 0)


def _flat_after(r, step, window=10):
    """Flat groups among the groups of the `window` steps after `step`."""
    return sum(r.flat[step:step + window])


def curriculum_run():
    rs = runs("spec", 0)
    fx = runs("fixed", 0)
    rows = []
    for k in range(LEVELS - 1):
        step = [r.advances[k][0] for r in rs]
        rows.append([k + 1, _int_cell(step), _cell([r.advances[k][2] for r in rs], 4), _cell([r.advances[k][3] for r in rs], 4),
                     _int_cell([_flat_after(r, r.advances[k][0]) for r in rs])])
    marks = (10, 30, 60, 90, 120, 150, 200, 250, 300)
    traj = []
    for step in marks:
        idx = step // EVERY - 1
        traj.append([step, _int_cell([r.snaps[idx][1] for r in rs])]
                    + [_cell([r.snaps[idx][3][lv][0:PROBES] and sum(r.snaps[idx][3][lv]) / PROBES for r in rs], 3) for lv in range(LEVELS)])
    fixed_row = ["no curriculum (level 5 from step 1), step 300", "5"] + [_cell([sum(r.snaps[-1][3][lv]) / PROBES for r in fx], 3) for lv in range(LEVELS)]
    pool_row = ["the policy of step 300 on the 24 pool inputs", "5"] + [_cell([probe_success(r.theta, POOL_IN, lv + 1) for r in rs], 3) for lv in range(LEVELS)]
    traj.append(pool_row)
    traj.append(fixed_row)
    flat_total = _int_cell([sum(r.flat) for r in rs])
    flat_fixed = _int_cell([sum(r.flat) for r in fx])
    return (
        f"The headline run: generator {_seed_line(_run_name(0))}, the policy of milestone 2 from theta = 0 (theta[0] nudged in the four copies), the loop of "
        f"milestones 3 to 5 for {STEPS} steps. Each step draws, in this order, {TASKS_PER_STEP} pool entries (`integer(24)` each) and then, task by "
        f"task, answer by answer and position by position, one `uniform()` per output digit, from one generator that is never restarted within the run. "
        f"The level starts at 1. After the update of step s the mean success of the {PROBES} probe inputs at the level used in step s is computed "
        f"(summed in probe order); when it is at least {TAU:g} (non-strict) and the level is below {LEVELS} the level is raised by 1 for step s + 1. "
        f"'Training success' is the mean reward of the {TASKS_PER_STEP * GROUP} answers sampled in step s. 'Flat groups' counts the groups with no "
        f"gradient (all rewards equal) among the {TASKS_PER_STEP} groups of each of the 10 steps after the advance (of 40).\n\n"
        + _table(["level left", "step of the advance", "probe success after that step", "training success of that step", "flat groups in the next 10 steps"],
                 rows, "rrrrr")
        + f"\n\nThe policy as it stands at the end of selected steps (the mean success of the probe bank at each level, over the {PROBES} probe inputs). The row before "
        f"the last is the same policy as the row of step 300, scored on the 24 pool inputs it trained on instead of the probe inputs. The last row "
        f"is the control without a curriculum: the same run from the same generator, every step at level {LEVELS}.\n\n"
        + _table(["step", "level of the next step", "probe level 1", "level 2", "level 3", "level 4", "level 5"], traj, "rrrrrrr")
        + f"\n\nFlat groups over the {STEPS} steps ({STEPS * TASKS_PER_STEP} groups): {flat_total} with the curriculum, {flat_fixed} without it.\n\n" + RULE_TEXT)


# ------------------------------------------------------------------ milestone 6: the gate, the registry and the audit

def _decisions(rs, bound):
    return [serve(r, bound) for r in rs]


def gate_run():
    rs = runs("spec", 0)
    paired = _decisions(rs, paired_lower)
    unpaired = _decisions(rs, unpaired_lower)
    rows = []
    for k in range(STEPS // EVERY):
        step = (k + 1) * EVERY
        lv = _int_cell([r.snaps[k][1] for r in rs])
        gain = [p[0][k]["gain"] for p in paired]
        rows.append([step, lv, _cell([g[0] for g in gain], 4), _cell([g[1] for g in gain], 4), _cell([g[2] for g in gain], 4),
                     _text_cell([p[0][k]["decision"] + (": " + p[0][k]["cause"] if p[0][k]["cause"] else "") for p in paired]),
                     _text_cell([u[0][k]["decision"] + (": " + u[0][k]["cause"] if u[0][k]["cause"] else "") for u in unpaired])])
    n_p = _int_cell([sum(1 for rec in p[0] if rec["decision"] == "promote") for p in paired])
    n_u = _int_cell([sum(1 for rec in u[0] if rec["decision"] == "promote") for u in unpaired])
    fails = _int_cell([sum(1 for rec in p[0] if rec["audit"]) for p in paired])
    return (
        f"The 30 snapshots of the headline run (every {EVERY} steps, after the update and the curriculum check of that step), each judged against the "
        f"version being served. Per probe the score is the success probability of the probe's input at the level (milestone 2); for each level "
        f"the paired differences d_k (candidate minus served, over the {PROBES} probes in order), their mean, the paired standard error sqrt(s^2 / {PROBES}) with "
        f"s^2 the variance of the d_k with divisor {PROBES - 1}, and the lower bound mean - {Z:.6f} x SE (strict comparisons: a bound exactly at the bar fails). "
        f"The gate checks, in this order: every level 1 .. 5 has a lower bound above -{DELTA:g} (the first level that does not is named), then the level the "
        f"trainer will use next has a lower bound above 0. A promoted candidate becomes the served version, after the audit of milestone 7. The "
        f"'unpaired' column is the same gate with the standard error of two independent means, sqrt(var(candidate) / {PROBES} + var(served) / {PROBES}), "
        f"which is the gate of mistake 3. Gain columns are the paired difference at the trainer's next level.\n\n"
        + _table(["step", "level", "mean gain at that level", "paired SE", "lower bound", "paired gate", "unpaired gate"], rows, "rrrrrll")
        + f"\n\nPromotions of the {STEPS // EVERY} snapshots: {n_p} with the paired gate, {n_u} with the unpaired one. Audit failures after a promotion (paired gate): {fails}.\n\n" + RULE_TEXT)


def _first_split(paired, unpaired, rs):
    """The first snapshot that the paired gate promotes and the unpaired gate rejects, in the run itself."""
    for k in range(STEPS // EVERY):
        if paired[0][k]["decision"] == "promote" and unpaired[0][k]["decision"] == "reject":
            return k
    return None


def two_errors():
    rs = runs("spec", 0)
    paired = _decisions(rs, paired_lower)
    unpaired = _decisions(rs, unpaired_lower)
    k = _first_split(paired[0], unpaired[0], rs)
    r0 = rs[0]
    step, level = r0.snaps[k][0], r0.snaps[k][1]
    # the version served when that snapshot was judged: the last promoted version before it
    reg = paired[0][1]
    inc = [v for v in reg if v[0] < step][-1]
    rows = []
    for lv in range(1, LEVELS + 1):
        c = [r.snaps[k][3][lv - 1] for r in rs]
        i = [serve_inc[2][lv - 1] for serve_inc in [inc]] * len(rs)
        rows.append([lv, _cell([paired_lower(r.snaps[k][3][lv - 1], inc[2][lv - 1])[0] for r in rs], 4),
                     _cell([paired_lower(r.snaps[k][3][lv - 1], inc[2][lv - 1])[1] for r in rs], 4),
                     _cell([paired_lower(r.snaps[k][3][lv - 1], inc[2][lv - 1])[2] for r in rs], 4),
                     _cell([unpaired_lower(r.snaps[k][3][lv - 1], inc[2][lv - 1])[1] for r in rs], 4),
                     _cell([unpaired_lower(r.snaps[k][3][lv - 1], inc[2][lv - 1])[2] for r in rs], 4)])
    return (
        f"The first snapshot of the headline run that the paired gate promotes and the unpaired gate refuses: step {step}, judged against version "
        f"{reg.index(inc)} (promoted at step {inc[0]}). For each level, the mean paired difference, the paired standard error, the paired lower bound, "
        f"the standard error of two independent means and the lower bound that it gives (the bar is -{DELTA:g}).\n\n"
        + _table(["level", "mean difference", "paired SE", "paired lower bound", "unpaired SE", "unpaired lower bound"], rows, "rrrrrr")
        + "\n\n" + RULE_TEXT)


# ------------------------------------------------------------------ milestone 7: versions, audit and rollback

def _mean_scores(scores, level):
    return sum(scores[level - 1]) / len(scores[level - 1])


def versions():
    rs = runs("spec", 0)
    ps = [serve(r) for r in rs]
    n_versions = len(ps[0][1])
    picks = [v for v in (0, 1, 2, 4, 8, 12, 16, 20, 24) if v < n_versions] + [n_versions - 1]
    rows = []
    for v in picks:
        rows.append([v, _int_cell([p[1][v][0] for p in ps])]
                    + [_cell([_mean_scores(p[1][v][2], lv) for p in ps], 3) for lv in range(1, LEVELS + 1)]
                    + [_cell([_mean_scores(p[1][v][3], 5) for p in ps], 3)])
    trained = ["trained state, step 300", "300"] + [_cell([probe_success(r.theta, PROBE_IN, lv) for r in rs], 3) for lv in range(1, LEVELS + 1)] \
        + [_cell([probe_success(r.theta, AUDIT_IN, 5) for r in rs], 3)]
    rows.append(trained)
    return (
        f"The registry of the headline run under the paired gate: every promoted version keeps its parameters (the {N_PARAM} numbers, as served) and the "
        f"scores it had when it was promoted. Version 0 is the untrained policy. The columns are the mean success of the probe bank at each level and, "
        f"in the last column, the mean success at level 5 of the audit bank (inputs the gate never saw). The versions shown are 0, 1, 2, 4, 8, 12, 16, 20, 24 and "
        f"the last ({n_versions - 1}); the last row is the trainer's own state at step {STEPS}, which is not a version unless the last snapshot was promoted.\n\n"
        + _table(["version", "promoted at step", "probe level 1", "level 2", "level 3", "level 4", "level 5", "audit level 5"], rows, "rrrrrrrr")
        + f"\n\nAudits that failed after a promotion: {_int_cell([sum(1 for rec in p[0] if rec['audit']) for p in ps])}.\n\n" + RULE_TEXT)


def _bug_candidate(r):
    """From the snapshot at step BUG_AT, BUG_STEPS steps at that snapshot's level with the sign of the update reversed."""
    k = BUG_AT // EVERY - 1
    step, level, theta, _scores = r.snaps[k]
    rng = bookrng.generator(BUG)
    for _ in range(BUG_STEPS):
        theta, _train, _flat = train_step(theta, level, rng, verify_exact, ascent=-1.0)
    return theta, level


def rollback():
    rs = runs("spec", 0)
    rows = []
    out = []
    for r in rs:
        recs, reg, served = serve(r)
        k = BUG_AT // EVERY - 1
        v_before = recs[k]["served_before"]
        cand, level = _bug_candidate(r)
        sc = bank_scores(cand, PROBE_IN)
        au = bank_scores(cand, AUDIT_IN)
        inc = reg[v_before]
        g_paired = gate(sc, inc[2], level, paired_lower)
        g_unpaired = gate(sc, inc[2], level, unpaired_lower)
        audit = non_inferior(au, inc[3])
        # a forced promotion, then the audit's rollback: the served pointer returns to the previous version
        registry = list(reg[:v_before + 1]) + [(BUG_AT + BUG_STEPS, cand, sc, au)]
        pointer = len(registry) - 1
        if audit:
            pointer = pointer - 1
        out.append({"v_before": v_before, "level": level, "g_paired": g_paired, "g_unpaired": g_unpaired, "audit": audit,
                    "pointer": pointer, "same": registry[pointer][1] == inc[1] and registry[pointer][2] == inc[2],
                    "inc": [_mean_scores(inc[2], lv) for lv in range(1, LEVELS + 1)],
                    "cand": [_mean_scores(sc, lv) for lv in range(1, LEVELS + 1)],
                    "cand_audit": [_mean_scores(au, lv) for lv in range(1, LEVELS + 1)],
                    "inc_audit": [_mean_scores(inc[3], lv) for lv in range(1, LEVELS + 1)]})
    rows.append(["served version " + _int_cell([o["v_before"] for o in out]) + ", probe bank"] +
                [_cell([o["inc"][lv] for o in out], 3) for lv in range(LEVELS)])
    rows.append(["faulty candidate, probe bank"] + [_cell([o["cand"][lv] for o in out], 3) for lv in range(LEVELS)])
    rows.append(["paired lower bound, faulty minus served, probe bank"] +
                [_cell([paired_lower(bank_scores(_bug_candidate(r)[0], PROBE_IN)[lv], serve(r)[1][o["v_before"]][2][lv])[2]
                        for r, o in zip(rs, out)], 3) for lv in range(LEVELS)])
    rows.append(["served version, audit bank"] + [_cell([o["inc_audit"][lv] for o in out], 3) for lv in range(LEVELS)])
    rows.append(["faulty candidate, audit bank"] + [_cell([o["cand_audit"][lv] for o in out], 3) for lv in range(LEVELS)])
    fmt = lambda g: g[0] + (": " + g[1] if g[1] else "")
    return (
        f"The faulty candidate. Take the trainer's state at step {BUG_AT} (the snapshot of milestone 6, at level {_int_cell([o['level'] for o in out])}) and train it for "
        f"{BUG_STEPS} more steps with the loop of milestone 5 at that level, but with a sign error in the update: theta - eta x grad in place of theta + eta x grad "
        f"in both epochs of every step (the sampling, the advantages and the clipping are unchanged). The generator is {_seed_line(BUG)} "
        f"(restarted for this candidate). It is judged against the version served when the snapshot of step {BUG_AT} was judged.\n\n"
        + _table(["", "level 1", "level 2", "level 3", "level 4", "level 5"], rows, "lrrrrr")
        + f"\n\nThe paired gate says: {_text_cell([fmt(o['g_paired']) for o in out])}. The unpaired gate says: {_text_cell([fmt(o['g_unpaired']) for o in out])}. "
        f"If the candidate is served anyway (a hot fix that bypasses the gate), the audit, which applies the non-inferiority bars of the gate to the audit bank "
        f"against the previous version, first finds level {_int_cell([o['audit'] for o in out])} regressed (0 would mean none), and the rollback moves "
        f"the served pointer back to version {_int_cell([o['pointer'] for o in out])}; the parameters and the stored probe scores of that version are "
        f"identical to what they were before the candidate arrived: {_text_cell([str(o['same']) for o in out])}.\n\n" + RULE_TEXT)


# ------------------------------------------------------------------ milestone 8: three planted mistakes

def _final_level(r):
    return r.levels[-1]


def _last_mean(values, n=10):
    acc = 0.0
    for v in values[-n:]:
        acc += v
    return acc / n


def _advance_text(rs, k):
    return ", ".join(_int_cell([r.advances[i][0] for r in rs]) if len(rs[0].advances) > i else "-" for i in range(k))


def _adv_cells(rs, which, nd=2):
    n = min(len(r.advances) for r in rs)
    return ", ".join(_cell([r.advances[i][which] for r in rs], nd) for i in range(n))


def weak_verifier():
    rows = []
    for seed in range(RUNS):
        sp, wk = runs("spec", seed), runs("weak", seed)
        rows.append([seed, _int_cell([_final_level(r) for r in sp]), _cell([probe_success(r.theta, PROBE_IN, 5) for r in sp], 3),
                     _int_cell([_final_level(r) for r in wk]), _cell([_last_mean(r.train) for r in wk], 3),
                     _cell([probe_success(r.theta, PROBE_IN, _final_level(r)) for r in wk], 3),
                     _cell([probe_success(r.theta, PROBE_IN, 5) for r in wk], 3)])
    return (
        f"Mistake 1: the weak verifier. The system of milestones 1 to 7 with one change, the reward of every sampled answer is the weak verifier of milestone 1 "
        f"(every output digit has the parity of the target digit); the curriculum still reads the exact probe bank. Five seeds, run names "
        f"{', '.join('`' + _run_name(s) + '`' for s in range(RUNS))}, the same generators for both systems. 'Level of step 300' is the level the trainer used in "
        f"step {STEPS}; 'weak reward' is the mean reward the trainer was paid over steps {STEPS - 9} to {STEPS}; 'exact probe success' is the mean success of "
        f"the exact probe bank at the trainer's level, with the final policy.\n\n"
        + _table(["seed", "specified: level of step 300", "specified: probe level 5", "weak: level of step 300", "weak: weak reward, last 10 steps",
                  "weak: exact probe success at its level", "weak: probe level 5"], rows, "rrrrrrr") + "\n\n" + RULE_TEXT)


def train_reward_curriculum():
    rows, rows2 = [], []
    for seed in range(RUNS):
        sp, tr = runs("spec", seed), runs("train", seed)
        rows.append([seed, "probe success", _advance_text(sp, 4), _adv_cells(sp, 2), "-"])
        rows.append(["", "training success", _advance_text(tr, 4), _adv_cells(tr, 2), _adv_cells(tr, 3)])
        rows2.append([seed, _int_cell([sum(_flat_after(r, a[0]) for a in r.advances) for r in sp]),
                      _int_cell([sum(_flat_after(r, a[0]) for a in r.advances) for r in tr]),
                      _cell([probe_success(r.theta, PROBE_IN, 5) for r in sp], 3), _cell([probe_success(r.theta, PROBE_IN, 5) for r in tr], 3)])
    return (
        f"Mistake 2: a curriculum that reads the training reward. The system of milestones 1 to 7 with one change: the level is raised when the mean reward of the "
        f"{TASKS_PER_STEP * GROUP} answers sampled in the step just taken is at least {TAU:g}, instead of the probe success. Five seeds, as above. First table: the "
        f"steps at which the level was raised (the 1 to 2, 2 to 3, 3 to 4 and 4 to 5 advances), the probe success after the step at each advance, and for the "
        f"flawed curriculum the training success of the step at each advance.\n\n"
        + _table(["seed", "what the curriculum reads", "steps of the four advances", "probe success after those steps", "training success of those steps"],
                 rows, "rllll")
        + f"\n\nSecond table: the flat groups among the groups of the 10 steps after each of the four advances (of 160), and the probe success at level 5 "
        f"after {STEPS} steps.\n\n"
        + _table(["seed", "flat groups, specified", "flat groups, training-reward curriculum", "probe level 5, specified", "probe level 5, training-reward curriculum"],
                 rows2, "rrrrr") + "\n\n" + RULE_TEXT)


def unpaired_gate():
    rows = []
    for seed in range(RUNS):
        rs = runs("spec", seed)
        pa_ = _decisions(rs, paired_lower)
        un_ = _decisions(rs, unpaired_lower)
        rows.append([seed, _int_cell([sum(1 for q in p[0] if q["decision"] == "promote") for p in pa_]),
                     _int_cell([sum(1 for q in u[0] if q["decision"] == "promote") for u in un_]),
                     _cell([sum(q["lag"] for q in p[0]) / len(p[0]) for p in pa_], 4),
                     _cell([sum(q["lag"] for q in u[0]) / len(u[0]) for u in un_], 4),
                     _cell([_mean_scores(p[1][p[2]][2], 5) for p in pa_], 3), _cell([_mean_scores(u[1][u[2]][2], 5) for u in un_], 3)])
    return (
        f"Mistake 3: a gate that compares unpaired means. The snapshots of the five specified runs (milestone 6) are judged twice, once by the paired gate and "
        f"once by the same gate with the standard error of two independent means. 'Lag' is the mean over the 30 snapshots of the probe success of the snapshot "
        f"at the trainer's level minus that of the version being served at that moment (in the chain of the gate in question). 'Served level 5' is the probe success at "
        f"level 5 of the version served after the last snapshot.\n\n"
        + _table(["seed", "promotions, paired (of 30)", "promotions, unpaired", "lag, paired", "lag, unpaired", "served level 5, paired", "served level 5, unpaired"],
                 rows, "rrrrrrr") + "\n\n" + RULE_TEXT)


def fixed_level_control():
    rows = []
    for seed in range(RUNS):
        fx, sp = runs("fixed", seed), runs("spec", seed)
        rows.append([seed, _int_cell([sum(r.flat) for r in sp]), _int_cell([sum(r.flat) for r in fx]),
                     _cell([probe_success(r.theta, PROBE_IN, 5) for r in sp], 3), _cell([probe_success(r.theta, PROBE_IN, 5) for r in fx], 3)])
    return (
        f"The control without a curriculum: every step at level {LEVELS}, five seeds, the other settings those of the specified run.\n\n"
        + _table(["seed", "flat groups, curriculum (of 1200)", "flat groups, level 5 throughout", "probe level 5, curriculum", "probe level 5, level 5 throughout"],
                 rows, "rrrrr") + "\n\n" + RULE_TEXT)


# ------------------------------------------------------------------ the opening table and the bill

def opening():
    rows = []
    builds = (("the system as specified", "spec", paired_lower), ("the weak verifier (milestone 8, mistake 1)", "weak", paired_lower),
              ("a curriculum that reads the training reward (mistake 2)", "train", paired_lower),
              ("a gate that compares unpaired means (mistake 3)", "spec", unpaired_lower),
              ("no curriculum, level 5 from the first step", "fixed", paired_lower))
    for label, config, bound in builds:
        rs = runs(config, 0)
        ps = [serve(r, bound) for r in rs]
        rows.append([label, _cell([_last_mean(r.train) for r in rs], 3), _int_cell([_final_level(r) for r in rs]),
                     _cell([probe_success(r.theta, PROBE_IN, 5) for r in rs], 3),
                     _int_cell([sum(1 for q in p[0] if q["decision"] == "promote") for p in ps]),
                     _cell([_mean_scores(p[1][p[2]][2], 5) for p in ps], 3)])
    return (
        f"Five builds of the agent, all from the generator {_seed_line(_run_name(0))}, each trained for {STEPS} steps. 'Reward seen' is the mean reward the trainer was "
        f"paid for the answers of its last 10 steps (steps {STEPS - 9} to {STEPS}), with the verifier that build uses; 'level' is the level the trainer used in step "
        f"{STEPS}; 'probe level 5' is the mean success of the exact probe bank at level 5, for the trainer's final policy; 'promoted' counts the snapshots (of "
        f"{STEPS // EVERY}) that the build's gate admitted as the served version, and 'served level 5' is the probe success at level 5 of the version served after the "
        f"last snapshot. The first four rows are the system of milestones 1 to 7 and the three mistakes of milestone 8; the last row is a control.\n\n"
        + _table(["build", "reward seen", "level", "probe level 5", "promoted", "served level 5"], rows, "lrrrrr") + "\n\n" + RULE_TEXT)


def bill():
    rs = runs("spec", 0)
    r = rs[0]
    ps = [serve(x) for x in rs]
    n_promoted = _int_cell([len(p[1]) - 1 for p in ps])
    sampled = sum(TASKS_PER_STEP * GROUP * lv for lv in r.levels)
    old = sampled
    epochs = sum(EPOCHS * (TASKS_PER_STEP - f) * GROUP * lv for f, lv in zip(r.flat, r.levels))
    check = sum(PROBES * lv for lv in r.levels)
    per_bank = PROBES * sum(range(1, LEVELS + 1))
    snaps = per_bank * (STEPS // EVERY)
    audits = per_bank * (len(ps[0][1]) - 1)
    total = sampled + old + epochs + check + snaps + audits
    rows = [["sample the answers", "sum over steps of B x n_g x level", f"{sampled:,}", f"{100 * sampled / total:.1f}"],
            ["old probabilities of the sampled digits", "the same count", f"{old:,}", f"{100 * old / total:.1f}"],
            ["epoch gradients (tokens of non-flat groups)", "sum over steps of E x (B - flat) x n_g x level", f"{epochs:,}", f"{100 * epochs / total:.1f}"],
            ["curriculum check, every step", f"sum over steps of {PROBES} x level", f"{check:,}", f"{100 * check / total:.1f}"],
            ["score the snapshots on the probe bank", f"{STEPS // EVERY} x {PROBES} x (1 + 2 + 3 + 4 + 5)", f"{snaps:,}", f"{100 * snaps / total:.1f}"],
            ["audit the promotions on the audit bank", f"{n_promoted} x {PROBES} x 15", f"{audits:,}", f"{100 * audits / total:.1f}"],
            ["total", "", f"{total:,}", "100.0"]]
    theta_bytes = 8 * N_PARAM
    version_bytes = theta_bytes + 8 * 2 * LEVELS * PROBES
    return (
        f"Work counted in token evaluations for the headline run (the specified system, {STEPS} steps): one token evaluation is the three logits and the softmax "
        f"of one output position of one input (milestone 2). The gradient of an epoch evaluates the tokens of the groups that are not flat (their advantages are "
        f"nonzero) and nothing else; the numbers of flat groups per step are those of the run. The counts of the steps use the level of each step.\n\n"
        + _table(["stage", "count", "token evaluations", "share (%)"], rows, "llrr")
        + f"\n\nThe policy is {N_PARAM} doubles, {theta_bytes:,} bytes; a registry entry is the policy plus the stored scores of the probe bank "
        f"and of the audit bank ({2 * LEVELS * PROBES} doubles), {version_bytes:,} bytes, so {n_promoted} promoted versions plus version 0 hold "
        f"{(int(n_promoted) + 1) * version_bytes:,} bytes.\n\n" + RULE_TEXT)


# ------------------------------------------------------------------ the checklist

def acceptance_tests():
    rs = runs("spec", 0)
    zero = [0.0] * N_PARAM
    x0 = digits(PROBE_IN[0])
    # milestone 2 and 3
    rng = bookrng.generator(DEMO)
    us = ", ".join(f"{rng.uniform():.6f}" for _ in range(6))
    rec = {"groups": [], "epochs": []}
    new, train, flat = train_step(zero, 2, bookrng.generator(STEP_DEMO), verify_exact, rec)
    lines = _bank_lines()
    # milestone 5
    adv = [_int_cell([r.advances[k][0] for r in rs]) for k in range(4)]
    advp = [_cell([r.advances[k][2] for r in rs], 4) for k in range(4)]
    end = [_cell([probe_success(r.theta, PROBE_IN, lv) for r in rs], 3) for lv in range(1, LEVELS + 1)]
    fx = runs("fixed", 0)
    # milestone 6 and 7
    pa_ = _decisions(rs, paired_lower)
    un_ = _decisions(rs, unpaired_lower)
    rejected = [rec_["step"] for rec_ in pa_[0][0] if rec_["decision"] == "reject"]
    k40 = 3
    reg = pa_[0][1]
    inc = [v for v in reg if v[0] < 40][-1]
    cand, level = _bug_candidate(rs[0])
    sc = bank_scores(cand, PROBE_IN)
    au = bank_scores(cand, AUDIT_IN)
    v_before = pa_[0][0][BUG_AT // EVERY - 1]["served_before"]
    # milestone 8
    wk = runs("weak", 0)
    tr = runs("train", 0)
    rows = [
        [1, "SHA-256 prefix of the pool, probe and audit numbers", f"{_digest_numbers(POOL_IN)}, {_digest_numbers(PROBE_IN)}, {_digest_numbers(AUDIT_IN)}", "exact"],
        [1, "the first probe input: m, its digits, its target", f"{PROBE_IN[0]}, {_digits_text(x0)}, {_digits_text(target(x0))}", "exact"],
        [2, "success of any input at theta = 0, levels 1 to 5", ", ".join(f"{3.0 ** -lv:.6f}" for lv in range(1, 6)), "6 decimals"],
        [2, f"first six draws of {DEMO}", us, "6 decimals"],
        [3, "one step from theta = 0, level 2: mean reward, flat groups", f"{train:.4f}, {flat}", "exact"],
        [3, "tokens clipped, L1 change of theta, in epoch 1 and in epoch 2", f"{rec['epochs'][0]['clipped']}, {rec['epochs'][0]['delta_l1']:.6f}; {rec['epochs'][1]['clipped']}, {rec['epochs'][1]['delta_l1']:.6f}", "exact; 6 decimals"],
        [3, "mean probe success at level 2 after that step", f"{probe_success(new, PROBE_IN, 2):.6f}", "6 decimals"],
        [4, "identity of the probe bank (SHA-256 prefix)", hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16], "exact"],
        [5, "steps at which the level is raised, 1 to 2 .. 4 to 5", ", ".join(adv), "exact"],
        [5, "probe success after those steps", ", ".join(advp), "4 decimals"],
        [5, "probe success at levels 1 to 5, step 300", ", ".join(end), "3 decimals"],
        [5, "flat groups of 1200, curriculum and level 5 throughout", f"{_int_cell([sum(r.flat) for r in rs])}, {_int_cell([sum(r.flat) for r in fx])}", "exact"],
        [6, "snapshots promoted of 30, paired gate and unpaired gate", f"{_int_cell([sum(1 for q in p[0] if q['decision'] == 'promote') for p in pa_])}, {_int_cell([sum(1 for q in u[0] if q['decision'] == 'promote') for u in un_])}", "exact"],
        [6, "steps of the snapshots the paired gate refuses, and the cause", ", ".join(map(str, rejected)) + ": " + _text_cell([pa_[0][0][s // EVERY - 1]["cause"] for s in rejected[:1]]), "exact"],
        [6, "step 40 against the served version: level 1 paired and unpaired lower bound",
         f"{_cell([paired_lower(r.snaps[k40][3][0], inc[2][0])[2] for r in rs], 4)}, {_cell([unpaired_lower(r.snaps[k40][3][0], inc[2][0])[2] for r in rs], 4)}", "4 decimals"],
        [7, "faulty candidate: paired gate decision, first level the audit finds regressed, version served after rollback",
         f"{gate(sc, pa_[0][1][v_before][2], level)[0]}: {gate(sc, pa_[0][1][v_before][2], level)[1]}, {non_inferior(au, pa_[0][1][v_before][3])}, {v_before}", "exact"],
        [8, "weak verifier: level of step 300, weak reward, exact probe success at that level, probe level 5",
         f"{_int_cell([_final_level(r) for r in wk])}, {_cell([_last_mean(r.train) for r in wk], 3)}, {_cell([probe_success(r.theta, PROBE_IN, _final_level(r)) for r in wk], 3)}, {_cell([probe_success(r.theta, PROBE_IN, 5) for r in wk], 3)}", "3 decimals"],
        [8, "training-reward curriculum: steps of the advances, probe success after them",
         f"{_advance_text(tr, 4)}; {_adv_cells(tr, 2)}", "exact; 2 decimals"],
        [8, "unpaired gate: promotions of 30, lag, served level 5", f"{_int_cell([sum(1 for q in u[0] if q['decision'] == 'promote') for u in un_])}, {_cell([sum(q['lag'] for q in u[0]) / len(u[0]) for u in un_], 4)}, {_cell([_mean_scores(u[1][u[2]][2], 5) for u in un_], 3)}", "exact; 4 and 3 decimals"],
    ]
    return (
        "Each row is a number of a block above for the headline run (generator `project_agent.run0`), collected in the order of the milestones. 'Exact' means "
        "equal integers or equal text; a number with decimals is to be matched to the digits shown, which are the digits on which the run and its four nudged copies agree.\n\n"
        + _table(["milestone", "quantity", "value", "how to compare"], rows, "rlll"))
