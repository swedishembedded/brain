# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Policy Gradients".

Plain Python 3, standard library only, written from the chapter's own description. The decision process is the shop of
`mdps` and the learning machinery of `value_learning` is imported, never copied. The chapter starts from a one-day bandit made
of the empty shelf (four orders, each paying the day's reward plus the discounted optimal value of the next shelf), where
the exact gradient of the expected return, the single-sample estimators of it and their exact variance can all be printed.
It then checks the policy gradient theorem on the whole shop by comparing the formula with finite differences of the exact
return, runs REINFORCE with and without a learned baseline against a one-step actor-critic on sampled episodes, shows how
generalised advantage estimation trades variance against the bias of a critic, tabulates the clipped surrogate and runs the
clipped and the unclipped objective for many epochs on one batch. Each public function without a leading underscore that takes
no arguments returns a markdown string that the chapter embeds between `<!-- worked: policy_gradients.<name> -->` markers.
Underscore functions and the constants are shared with the chapter's figures, so a picture and the table beside it cannot
disagree.

Every number learned from samples goes through `stable` (the rule of `value_learning`): the run is repeated with the first
stored preference changed by each of the nudges of `stable.py`, and a digit is printed only if the run and its four copies
share it. Exact quantities (enumerations over the twelve outcomes of the bandit, linear systems) are deterministic and are
printed to the digits stated beside them.

Swedish Embedded AB implements solutions for agents that improve their own behaviour from the outcomes it produces. If your
team needs expertise in policy-gradient training and reinforcement learning from verifiable rewards, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng
import mdps
import stable
import value_learning as vl

# ------------------------------------------------------------------ the shared setting

GAMMA = mdps.GAMMA
N = mdps.N_STATES
PI_BEST = (1, 0, 0, 0)
V_STAR = mdps._evaluate_solve(PI_BEST, GAMMA)[0]        # the optimal values: the best policy is PI_BEST (chapter 1)
J_STAR = sum(V_STAR) / N
VARIANTS = vl.VARIANTS

# The one-day bandit: the empty shelf, four orders. y[a][d] = reward of the day + gamma V*(next shelf).
M_BANDIT = 4
THETA_B = (0.0, 1.0, 0.0, -1.0)         # the preferences at which the estimators are analysed
BASELINES = (0.0, 3.0, None, None, 12.0)  # None, None: the value J and the optimal baseline b*, filled in by _baselines()
MOMENT_DRAWS = 20000
ACCURACY = (0.5, 0.1)                   # relative root-mean-square errors that a gradient estimate must reach

# The shop with a tabular softmax policy.
NA = [mdps._n_actions(s) for s in range(N)]
THETA_C = ((0.0, 0.5, 0.0, -0.5), (0.0, 0.5, 0.0), (0.0, 0.5), (0.0,))   # preferences theta[s][a] of the theorem check
FD_BANDIT, FD_SHOP = 1e-6, 1e-5

RUN_DAYS, RUN_EPISODES, RUN_RUNS, RUN_CHECKS = 100, 200, 10, (1, 5, 10, 25, 50, 100, 200)
RUN_ETA, RUN_ALPHA_C = 0.05, 0.05
RUN_METHODS = (("reinforce", "REINFORCE"), ("baseline", "REINFORCE with baseline"), ("actor_critic", "actor-critic"))

GAE_LAMBDAS, GAE_ROLLOUTS, GAE_DAYS = (0.0, 0.5, 0.9, 1.0), 2000, 100

PPO_BATCH, PPO_EPOCHS, PPO_ETA, PPO_EPS = 64, 20, 0.5, 0.2
PPO_CHECKS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)
PPO_BATCHES = 200                       # independent batches of the closing investigation


# ------------------------------------------------------------------ small helpers


def _softmax(z):
    """The stable softmax of the pseudocode: subtract the largest, exponentiate, divide by the sum accumulated in order."""
    m = max(z)
    e = [math.exp(x - m) for x in z]
    total = 0.0
    for x in e:
        total += x
    return [x / total for x in e]


def _sample_index(p, rng):
    """Inverse CDF: one uniform(), the first index whose running sum of p exceeds it (the last one if rounding leaves a gap)."""
    u = rng.uniform()
    c = 0.0
    for a, x in enumerate(p):
        c += x
        if u < c:
            return a
    return len(p) - 1


def _cell(vals, nd):
    """vl._cell, with values that are zero up to the rounding noise of a nudged start (below 1e-9) taken as exactly zero, so a
    quantity that is 0 by construction at the start of a run is not reported unstable for the sign of its noise."""
    return vl._cell([v if abs(v) >= 1e-9 else 0.0 for v in vals], nd)


def _dot(a, b):
    acc = 0.0
    for x, y in zip(a, b):
        acc += x * y
    return acc


RULE_TEXT = ("Every number learned from samples is printed with the digits that the run and four copies of it share: the copies "
             "start with the first preference (theta[0][0] of the shop, theta_B[0] of the bandit) changed by +1e-12, -1e-12, +3e-12 "
             "and -3e-12 (a stability check applied to every trained number), and a digit is printed only if all five agree on it ('about x' keeps the "
             "leading digits that survive, 'unstable' means that none does). The copies use the same generator and so draw the "
             "same numbers; they differ only if a nudge flips a comparison. A run is a deterministic computation, so another "
             "implementation reproduces the printed digits.")


# ------------------------------------------------------------------ the one-day bandit


def _outcomes():
    """y[a][d], the value of ordering a on the empty shelf when the demand turns out d."""
    return [[mdps._step(0, a, d)[0] + GAMMA * V_STAR[mdps._step(0, a, d)[1]] for d in range(3)] for a in range(M_BANDIT)]


Y = _outcomes()
P_D = mdps.DEMAND
Q_BANDIT = [sum(p * y for p, y in zip(P_D, Y[a])) for a in range(M_BANDIT)]        # summed over d = 0, 1, 2 in order


def _bandit_j(theta):
    pi = _softmax(list(theta))
    return _dot(pi, Q_BANDIT)


def _bandit_gradient(theta):
    """The exact gradient pi_k (q_k - J) of J(theta) = sum_a pi_a q_a with respect to the preferences."""
    pi = _softmax(list(theta))
    j = _dot(pi, Q_BANDIT)
    return [pi[k] * (Q_BANDIT[k] - j) for k in range(M_BANDIT)], pi, j


def _score_norms(pi):
    """|| e_a - pi ||^2 for every a."""
    sq = _dot(pi, pi)
    return [1.0 - 2.0 * pi[a] + sq for a in range(M_BANDIT)]


def _outcome_probs(pi):
    return [[pi[a] * P_D[d] for d in range(3)] for a in range(M_BANDIT)]


def _baselines():
    """The five baselines of the moments table: 0, 3, J, the variance-minimising constant b* and 12."""
    g, pi, j = _bandit_gradient(THETA_B)
    w = _score_norms(pi)
    pr = _outcome_probs(pi)
    num = sum(pr[a][d] * Y[a][d] * w[a] for a in range(M_BANDIT) for d in range(3))
    den = sum(pr[a][d] * w[a] for a in range(M_BANDIT) for d in range(3))
    b_star = num / den
    return [0.0, 3.0, j, b_star, 12.0], j, b_star


def _total_variance(b, pi, g):
    """E || (y - b)(e_a - pi) ||^2 minus || g ||^2: the exact total variance of one single-sample estimate."""
    w = _score_norms(pi)
    pr = _outcome_probs(pi)
    second = sum(pr[a][d] * (Y[a][d] - b) ** 2 * w[a] for a in range(M_BANDIT) for d in range(3))
    return second - _dot(g, g)


def _exact_mean(b, pi):
    """E[(y - b)(e_a - pi)] by enumerating the twelve outcomes, as a vector."""
    out = [0.0] * M_BANDIT
    pr = _outcome_probs(pi)
    for a in range(M_BANDIT):
        for d in range(3):
            for k in range(M_BANDIT):
                out[k] += pr[a][d] * (Y[a][d] - b) * ((1.0 if k == a else 0.0) - pi[k])
    return out


def _estimate(a, d, b, pi):
    return [(Y[a][d] - b) * ((1.0 if k == a else 0.0) - pi[k]) for k in range(M_BANDIT)]


def _fmt(x, nd=4):
    return mdps._fmt(x, nd)


def one_day_bandit():
    g, pi, j = _bandit_gradient(THETA_B)
    g0, pi0, j0 = _bandit_gradient((0.0,) * M_BANDIT)
    # central differences of J(theta) in every preference
    worst = 0.0
    for k in range(M_BANDIT):
        up, dn = list(THETA_B), list(THETA_B)
        up[k] += FD_BANDIT
        dn[k] -= FD_BANDIT
        fd = (_bandit_j(up) - _bandit_j(dn)) / (2.0 * FD_BANDIT)
        worst = max(worst, abs(fd - g[k]))
    rows = []
    for a in range(M_BANDIT):
        rows.append([a] + [_fmt(Y[a][d], 4) for d in range(3)] + [_fmt(Q_BANDIT[a], 4), _fmt(pi0[a], 4), _fmt(pi[a], 4),
                                                                   _fmt(Q_BANDIT[a] - j, 4), _fmt(g[a], 4)])
    top = (
        "The empty shelf (s = 0) as a one-day decision. The action a is the number of units ordered, a = 0 .. 3. The demand D is "
        "0, 1 or 2 with probabilities 0.3, 0.4, 0.3. The outcome y(a, d) is the reward of the day plus gamma = 0.9 times the "
        "optimal value V*(s') of the shelf that is left, where V* is the exact solution of the previous chapter's system for the "
        "policy (1, 0, 0, 0) (6.7500, 9.2500, 11.1678, 11.6684 to four decimals) and the reward and the next shelf come from the "
        "rules of chapter 1. q(a) is the mean of y(a, d) over d = 0, 1, 2 in order. The policy is the softmax of four preferences "
        "theta, pi(a) = exp(theta_a) / sum_b exp(theta_b), computed in the stable form. J(theta) = sum_a pi(a) q(a), and the "
        "exact gradient is the vector with entries pi(k) (q(k) - J). Two preference vectors: theta = (0, 0, 0, 0) and "
        f"theta_B = (0, 1, 0, -1), the one the estimators of the chapter are analysed at; at theta = 0 the policy is uniform "
        f"and J = {_fmt(j0, 4)}."
    )
    table = vl._table(["a", "y(a, 0)", "y(a, 1)", "y(a, 2)", "q(a)", "pi(a) at 0", "pi(a) at theta_B", "q(a) - J", "dJ/dtheta_a"],
                      rows)
    tail = (
        f"At theta_B, J = {_fmt(j, 4)}. Central differences of J with the step {FD_BANDIT:g} in each preference agree with the "
        f"formula to {worst:.1e}, below the criterion 1e-6. The four gradient entries sum to {_fmt(sum(g), 4)}: a change that "
        f"raises every preference by the same amount changes nothing."
    )
    return "\n".join([top, "", table, "", tail])


# ------------------------------------------------------------------ single-sample estimates


def single_sample_estimates():
    name = "policy_gradients.draws"
    rng = bookrng.generator(name)
    g, pi, j = _bandit_gradient(THETA_B)
    samples = 8
    rows, acc0, accj = [], [0.0] * M_BANDIT, [0.0] * M_BANDIT
    for i in range(samples):
        ua = rng.uniform()
        # same inverse CDF as _sample_index, written out so the draw is printed
        c, a = 0.0, M_BANDIT - 1
        for k, x in enumerate(pi):
            c += x
            if ua < c:
                a = k
                break
        ud = rng.uniform()
        d = mdps._demand_from(ud)
        e0, ej = _estimate(a, d, 0.0, pi), _estimate(a, d, j, pi)
        for k in range(M_BANDIT):
            acc0[k] += e0[k]
            accj[k] += ej[k]
        rows.append([i, f"{ua:.4f}", a, f"{ud:.4f}", d, _fmt(Y[a][d], 4), _fmt(e0[1], 3), _fmt(ej[1], 3)])
    rows.append(["mean of 8", "", "", "", "", "", _fmt(acc0[1] / samples, 3), _fmt(accj[1] / samples, 3)])
    rows.append(["exact", "", "", "", "", "", _fmt(g[1], 3), _fmt(g[1], 3)])
    top = (
        f"Eight single-sample estimates of the gradient at theta_B = {THETA_B}, pi = ({', '.join(_fmt(p, 4) for p in pi)}), "
        f"J = {_fmt(j, 4)}. Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} ({bookrng.seed_of(name)}), "
        f"one generator for the whole table. Each sample draws, in this order, one uniform() ua for the action (inverse CDF: the "
        f"first a whose running sum of pi exceeds ua) and one uniform() ud for the demand (D = 0 if ud < 0.3, 1 if ud < 0.7, "
        f"otherwise 2). The estimate is (y - b)(e_a - pi), with e_a the vector that is 1 at a and 0 elsewhere; the table shows "
        f"its entry for the preference of the order of one unit (k = 1), for b = 0 and for b = J, and the mean of the eight "
        f"(summed in order, divisor 8) beside the exact value of that entry."
    )
    table = vl._table(["sample", "ua", "a", "ud", "d", "y", "estimate, b = 0", "estimate, b = J"], rows)
    return "\n".join([top, "", table])


# ------------------------------------------------------------------ the moments of the estimator


def estimator_moments():
    bs, j, b_star = _baselines()
    g, pi, _ = _bandit_gradient(THETA_B)
    labels = ["0", "3", "J", "b*", "12"]
    name = "policy_gradients.moments"
    rng = bookrng.generator(name)
    draws = []
    for _ in range(MOMENT_DRAWS):
        a = _sample_index(pi, rng)
        draws.append((a, mdps._demand_from(rng.uniform())))
    rows = []
    for label, b in zip(labels, bs):
        bias = max(abs(m - gk) for m, gk in zip(_exact_mean(b, pi), g))
        var = _total_variance(b, pi, g)
        ests = [_estimate(a, d, b, pi) for a, d in draws]
        mean = [0.0] * M_BANDIT
        for e in ests:
            for k in range(M_BANDIT):
                mean[k] += e[k]
        mean = [x / MOMENT_DRAWS for x in mean]
        sv = 0.0
        for e in ests:
            sv += sum((e[k] - mean[k]) ** 2 for k in range(M_BANDIT))
        sv /= MOMENT_DRAWS - 1
        assert bias < 1e-12
        rows.append([label, _fmt(b, 4), "below 1e-12", _fmt(var, 4), _fmt(mean[1], 4), _fmt(sv, 3)])
    top = (
        f"The estimator (y - b)(e_a - pi) at theta_B = {THETA_B} (pi = {', '.join(_fmt(p, 4) for p in pi)}, J = {_fmt(j, 4)}, "
        f"exact gradient entry for k = 1: {_fmt(g[1], 4)}), for five constant baselines b: 0, 3, J, b* and 12, where b* = "
        f"sum pi(a) p(d) y(a, d) |e_a - pi|^2 / sum pi(a) p(d) |e_a - pi|^2 = {_fmt(b_star, 4)}. 'Exact' columns enumerate "
        f"the twelve outcomes (a, d) with probability pi(a) p(d): the bias is the largest entry of |E[estimate] - gradient|, the "
        f"total variance is E|estimate|^2 - |gradient|^2 (the sum of the four entry variances). 'Sampled' columns use "
        f"{MOMENT_DRAWS} draws of (a, d) from the generator of seed {bookrng.seed_of(name):#018x} (one uniform() for the action, "
        f"then one for the demand, in this order per draw; the same draws serve every b): the mean of entry k = 1 (summed in "
        f"draw order, divisor {MOMENT_DRAWS}) and the sample total variance (divisor {MOMENT_DRAWS} - 1)."
    )
    table = vl._table(["b", "value", "exact bias", "exact total variance", "sampled mean, k = 1", "sampled total variance"],
                      rows, align="lrrrrr")
    return "\n".join([top, "", table])


def samples_for_accuracy():
    bs, j, b_star = _baselines()
    g, pi, _ = _bandit_gradient(THETA_B)
    gg = _dot(g, g)
    rows = []
    for label, b in zip(["0", "3", "J", "b*", "12"], bs):
        var = _total_variance(b, pi, g)
        rows.append([label] + [f"{math.ceil(var / (rho * rho * gg)):,}" for rho in ACCURACY])
    return "\n".join([
        f"The mean of N independent single-sample estimates has root-mean-square error sqrt(V/N), V the exact total variance of "
        f"the previous table. The smallest N with sqrt(V/N) at most rho |gradient| is ceil(V / (rho^2 |gradient|^2)), with "
        f"|gradient|^2 = {_fmt(gg, 5)} at theta_B.", "",
        vl._table(["baseline b", f"N for rho = {ACCURACY[0]}", f"N for rho = {ACCURACY[1]}"], rows, align="lrr")])


# ------------------------------------------------------------------ the theorem on the shop


def _pi_table(theta):
    return [_softmax(list(theta[s])) for s in range(N)]


def _chain(pis):
    """P_pi and r_pi of a stochastic policy: rows summed over the actions in increasing a."""
    P = [[0.0] * N for _ in range(N)]
    r = [0.0] * N
    for s in range(N):
        for a in range(NA[s]):
            r[s] += pis[s][a] * mdps.REW[s][a]
            for j in range(N):
                P[s][j] += pis[s][a] * mdps.PROB[s][a][j]
    return P, r


def _values(theta):
    pis = _pi_table(theta)
    P, r = _chain(pis)
    A = [[(1.0 if i == j else 0.0) - GAMMA * P[i][j] for j in range(N)] for i in range(N)]
    V, _ = mdps._solve(A, r)
    return V, pis, P


def _return_of(theta):
    """J(theta): the mean of V over the four start shelves."""
    V, _, _ = _values(theta)
    return sum(V) / N


def _visitation(P):
    """d = mu0^T (I - gamma P)^-1 for mu0 uniform over the four shelves, from the system (I - gamma P)^T d = mu0."""
    A = [[(1.0 if i == j else 0.0) - GAMMA * P[j][i] for j in range(N)] for i in range(N)]
    d, _ = mdps._solve(A, [1.0 / N] * N)
    return d


def _theorem_gradient(theta):
    V, pis, P = _values(theta)
    d = _visitation(P)
    grad = []
    for s in range(N):
        q = [mdps._q(s, a, V, GAMMA) for a in range(NA[s])]
        grad.append([d[s] * pis[s][a] * (q[a] - V[s]) for a in range(NA[s])])
    return grad, V, pis, d


def softmax_policy_gradient():
    theta = [list(row) for row in THETA_C]
    grad, V, pis, d = _theorem_gradient(theta)
    rows, worst = [], 0.0
    for s in range(N):
        for a in range(NA[s]):
            up = [row[:] for row in theta]
            dn = [row[:] for row in theta]
            up[s][a] += FD_SHOP
            dn[s][a] -= FD_SHOP
            fd = (_return_of(up) - _return_of(dn)) / (2.0 * FD_SHOP)
            diff = abs(fd - grad[s][a])
            worst = max(worst, diff)
            q = mdps._q(s, a, V, GAMMA)
            rows.append([s, a, _fmt(theta[s][a], 1), _fmt(pis[s][a], 4), _fmt(q, 4), _fmt(V[s], 4), _fmt(q - V[s], 4),
                         _fmt(d[s], 4), _fmt(grad[s][a], 5), _fmt(fd, 5)])
    top = (
        f"The shop (gamma = {GAMMA}) with a tabular softmax policy: pi(a | s) = softmax of the preferences theta[s][0 .. n_act[s] - 1]. "
        f"Preferences: theta[0] = {THETA_C[0]}, theta[1] = {THETA_C[1]}, theta[2] = {THETA_C[2]}, theta[3] = {THETA_C[3]}. The "
        f"start shelf is uniform, mu_0(s) = 1/4, and J(theta) = (1/4) sum_s V(s), with V the exact solution of "
        f"(I - gamma P_pi) V = r_pi, where row s of P_pi is sum_a pi(a | s) P(. | s, a) and r_pi(s) = sum_a pi(a | s) R(s, a) "
        f"(sums over a in increasing order; elimination with partial pivoting). d(s) is the discounted visitation, the solution "
        f"of (I - gamma P_pi)^T d = mu_0, so d(s) = sum_t gamma^t Pr(S_t = s) and the four values add to 1/(1 - gamma) = 10. "
        f"The theorem column is d(s) pi(a | s) (q(s, a) - V(s)), with q(s, a) = R(s, a) + gamma sum_j P(j | s, a) V(j). The "
        f"finite-difference column is (J(theta + h e) - J(theta - h e)) / (2h) with h = {FD_SHOP:g}, e the unit vector of the "
        f"entry, J recomputed from the linear system each time."
    )
    table = vl._table(["s", "a", "theta", "pi", "q(s, a)", "V(s)", "A = q - V", "d(s)", "theorem", "finite difference"], rows)
    tail = (
        f"J(theta) = {_fmt(sum(V) / N, 4)}. The largest difference between the two columns over the ten entries is {worst:.1e}, "
        f"below the criterion 1e-7."
    )
    return "\n".join([top, "", table, "", tail])


# ------------------------------------------------------------------ learning runs on the shop


def _episode_days(theta, rng):
    """One episode of RUN_DAYS days under the softmax policy: a start shelf from integer(4), then per day one uniform() for
    the action (inverse CDF) and one for the demand. Returns the list of (s, a, r, next)."""
    s = rng.integer(N)
    out = []
    for _ in range(RUN_DAYS):
        p = _softmax(theta[s])
        a = _sample_index(p, rng)
        r, nxt, _sold = mdps._step(s, a, mdps._demand_from(rng.uniform()))
        out.append((s, a, r, nxt))
        s = nxt
    return out


def _reinforce_update(theta, V, ep, use_baseline):
    """One REINFORCE update from one episode: reward-to-go returns, advantage G_t (- V(s_t) with a baseline), gradient
    summed over the days with the running power gamma^t, applied once at the end of the episode, then the critic."""
    ret, acc = [0.0] * len(ep), 0.0
    for t in range(len(ep) - 1, -1, -1):
        acc = ep[t][2] + GAMMA * acc
        ret[t] = acc
    grad = [[0.0] * NA[s] for s in range(N)]
    disc = 1.0
    for t, (s, a, _r, _nxt) in enumerate(ep):
        p = _softmax(theta[s])
        adv = ret[t] - (V[s] if use_baseline else 0.0)
        for k in range(NA[s]):
            grad[s][k] += disc * adv * ((1.0 if k == a else 0.0) - p[k])
        disc *= GAMMA
    for s in range(N):
        for k in range(NA[s]):
            theta[s][k] += RUN_ETA * grad[s][k]
    if use_baseline:
        for t, (s, _a, _r, _nxt) in enumerate(ep):
            V[s] += RUN_ALPHA_C * (ret[t] - V[s])


def _actor_critic_episode(theta, V, rng):
    """One episode of online actor-critic: every day, in order, the TD error delta = r + gamma V(next) - V(s) (from the critic
    as it stands), the actor step eta gamma^t delta (e_a - pi) on the preferences of s, then the critic step alpha_c delta."""
    s = rng.integer(N)
    disc = 1.0
    for _ in range(RUN_DAYS):
        p = _softmax(theta[s])
        a = _sample_index(p, rng)
        r, nxt, _sold = mdps._step(s, a, mdps._demand_from(rng.uniform()))
        delta = r + GAMMA * V[nxt] - V[s]
        for k in range(NA[s]):
            theta[s][k] += RUN_ETA * disc * delta * ((1.0 if k == a else 0.0) - p[k])
        V[s] += RUN_ALPHA_C * delta
        disc *= GAMMA
        s = nxt


def _run_names(method):
    return [f"policy_gradients.runs.{method}.run{i}" for i in range(RUN_RUNS)]


def _learn(method, name, nudge):
    """One run: RUN_EPISODES episodes from theta = 0 (its first entry nudged by `nudge`), the exact J(theta) after each, and
    the final greedy policy (smallest action with the largest preference, ties to the smaller)."""
    rng = bookrng.generator(name)
    theta = [[0.0] * NA[s] for s in range(N)]
    theta[0][0] += nudge
    V = [0.0] * N
    curve = []
    for _ in range(RUN_EPISODES):
        if method == "actor_critic":
            _actor_critic_episode(theta, V, rng)
        else:
            _reinforce_update(theta, V, _episode_days(theta, rng), method == "baseline")
        curve.append(_return_of(theta))
    greedy = tuple(max(range(NA[s]), key=lambda a, s=s: (theta[s][a], -a)) for s in range(N))
    return curve, greedy


_RUNS_CACHE = {}


def _all_runs(variants=VARIANTS):
    """{method: [per variant: (list of curves over runs, list of greedy policies)]}"""
    key = tuple(variants)
    if key not in _RUNS_CACHE:
        out = {}
        for method, _label in RUN_METHODS:
            out[method] = []
            for nudge in variants:
                curves, greedies = [], []
                for name in _run_names(method):
                    c, g = _learn(method, name, nudge)
                    curves.append(c)
                    greedies.append(g)
                out[method].append((curves, greedies))
        _RUNS_CACHE[key] = out
    return _RUNS_CACHE[key]


def _mean_sd(vals):
    n = len(vals)
    mean = 0.0
    for v in vals:
        mean += v
    mean /= n
    ss = 0.0
    for v in vals:
        ss += (v - mean) ** 2
    return mean, math.sqrt(ss / (n - 1))


def policy_gradient_runs():
    runs = _all_runs()
    rows = []
    for e in RUN_CHECKS:
        row = [e, e * RUN_DAYS]
        for method, _label in RUN_METHODS:
            stats = [_mean_sd([c[e - 1] for c in curves]) for curves, _g in runs[method]]
            row.append(_cell([m for m, _s in stats], 2) + " / " + _cell([s for _m, s in stats], 2))
        rows.append(row)
    final = ["greedy = (1, 0, 0, 0)", ""]
    for method, _label in RUN_METHODS:
        counts = [str(sum(1 for g in greedies if g == PI_BEST)) for _c, greedies in runs[method]]
        final.append(vl._same(counts) + f" of {RUN_RUNS}")
    rows.append(final)
    zero = [[0.0] * NA[s] for s in range(N)]
    seeds = vl._table(["run", "REINFORCE", "REINFORCE with baseline", "actor-critic"],
                      [[i] + [f"{bookrng.seed_of(_run_names(m)[i]):#018x}" for m, _l in RUN_METHODS] for i in range(RUN_RUNS)],
                      align="rrrr")
    top = (
        f"The shop, gamma = {GAMMA}, tabular softmax policy with preferences theta[s][a] starting at 0 (the uniform policy over "
        f"the legal orders, J = {_fmt(_return_of(zero), 4)}); the best policy (1, 0, 0, 0) has J = {_fmt(J_STAR, 4)}. An "
        f"episode is {RUN_DAYS} days: a start shelf from integer({N}), then per day one uniform() for the action (inverse CDF "
        f"over pi(. | s)) and one for the demand, in this order. Three learners, {RUN_RUNS} runs each of {RUN_EPISODES} "
        f"episodes, every run with its own generator (the SplitMix64 of the book seeded with seed_of of the name "
        f"policy_gradients.runs.<method>.run<i>, method = reinforce, baseline or actor_critic, i = 0 .. {RUN_RUNS - 1}; every "
        f"run restarts from its own seed, seeds below). REINFORCE: after an episode the return of day t is the reward-to-go "
        f"G_t = r_t + gamma G_(t+1) from G_{RUN_DAYS} = 0, the gradient is the sum over the days t = 0 .. {RUN_DAYS - 1} in order "
        f"of gamma^t (G_t - b) (e_(a_t) - pi(. | s_t)) added to the preferences of s_t (gamma^t is a running product, all "
        f"pi taken from the preferences as they stood at the start of the episode), and the preferences move by "
        f"eta = {RUN_ETA} times the sum, once per episode. The baseline is b = 0 for REINFORCE and b = V(s_t) for the second "
        f"method, a critic that starts at 0 and, after each episode's policy step, moves by alpha_c = {RUN_ALPHA_C} times "
        f"(G_t - V(s_t)) for every day in order (every visit, applied immediately). Actor-critic updates every day before the "
        f"next is drawn: delta = r + gamma V(s') - V(s) with the critic as it stands, theta[s] += eta gamma^t delta "
        f"(e_a - pi(. | s)) with pi from before the step, then V(s) += alpha_c delta, in this order; one policy step per day "
        f"and not per episode. J(theta) is recomputed exactly from the linear system after every episode. Each cell is the "
        f"mean / standard deviation of J over the {RUN_RUNS} runs (summed in run order; divisors {RUN_RUNS} and "
        f"{RUN_RUNS - 1}) after the stated number of episodes. The last row counts the runs whose greedy policy (the "
        f"largest preference of each shelf, the smaller action on a tie) is (1, 0, 0, 0)."
    )
    table = vl._table(["episodes", "days", "REINFORCE", "REINFORCE with baseline", "actor-critic"], rows, align="rrrrr")
    return "\n".join([top, "", table, "", RULE_TEXT, "", seeds])


def _mean_curve(method):
    """The mean over the runs of J after every episode, for the figure (no nudge)."""
    curves = _all_runs((0.0,))[method][0][0]
    return [sum(c[e] for c in curves) / len(curves) for e in range(RUN_EPISODES)]


# ------------------------------------------------------------------ generalised advantage estimation


def gae_by_hand():
    gamma, rewards, values, last = 0.9, (1.0, 0.0, 2.0), (0.5, 1.0, 0.2), 0.4
    nxt = list(values[1:]) + [last]
    deltas = [rewards[t] + gamma * nxt[t] - values[t] for t in range(3)]
    lams = (0.0, 0.5, 1.0)
    cols = {}
    for lam in lams:
        acc, adv = 0.0, [0.0] * 3
        for t in range(2, -1, -1):
            acc = deltas[t] + gamma * lam * acc
            adv[t] = acc
        cols[lam] = adv
    g3, ret = 0.0, [0.0] * 3
    for t in range(2, -1, -1):
        g3 = rewards[t] + gamma * (g3 if t < 2 else last)
        ret[t] = g3
    rows = [[t, _fmt(rewards[t], 1), _fmt(values[t], 1), _fmt(nxt[t], 1), _fmt(deltas[t], 3)] +
            [_fmt(cols[lam][t], 4) for lam in lams] + [_fmt(ret[t] - values[t], 4)] for t in range(3)]
    top = (
        f"Three days of one episode that was cut off after the third, gamma = {gamma}. r_t is the reward, V(s_t) the critic's value "
        f"of the day's state and V(s_(t+1)) the critic's value of the next state, which for the last day is the estimate of the "
        f"state where the episode was cut, {last}. delta_t = r_t + gamma V(s_(t+1)) - V(s_t). The advantage estimate for lambda "
        f"is computed backwards from the last day as A_t = delta_t + gamma lambda A_(t+1) with A_3 = 0. The last column is "
        f"G_t - V(s_t), with G_t = r_t + gamma G_(t+1) bootstrapped from {last} at the cut."
    )
    return "\n".join([top, "", vl._table(["t", "r_t", "V(s_t)", "V(s_(t+1))", "delta_t", "A, lambda = 0", "A, lambda = 0.5",
                                          "A, lambda = 1", "G_t - V(s_t)"], rows)])


def _gae_setting():
    theta = [list(r) for r in THETA_C]
    V, pis, P = _values(theta)
    V0, _p0, _P0 = _values([[0.0] * NA[s] for s in range(N)])      # the critic of the uniform policy: stale for theta
    return V, V0, pis, P


def _gae_exact_bias(a, lam, Vh, V, pis, P):
    """E[A_hat(0, a)] - A(0, a) for the infinite-horizon lambda-estimator with critic Vh, from the Markov chain:
    delta_0 + c P_a (I - c P_pi)^-1 u with c = gamma lambda and u = B_pi Vh - Vh (the mean one-step error per state)."""
    u = [sum(pis[s][b] * mdps._q(s, b, Vh, GAMMA) for b in range(NA[s])) - Vh[s] for s in range(N)]
    d0 = mdps._q(0, a, Vh, GAMMA) - Vh[0]
    c = GAMMA * lam
    M = [[(1.0 if i == j else 0.0) - c * P[i][j] for j in range(N)] for i in range(N)]
    w, _ = mdps._solve(M, u)
    mean = d0 + c * sum(mdps.PROB[0][a][j] * w[j] for j in range(N))
    return mean - (mdps._q(0, a, V, GAMMA) - V[0])


def advantage_estimates():
    V, V0, pis, P = _gae_setting()
    name = "policy_gradients.gae"
    rng = bookrng.generator(name)
    days = {}
    for a in range(M_BANDIT):
        days[a] = []
        for _ in range(GAE_ROLLOUTS):
            s, ep = 0, []
            for t in range(GAE_DAYS):
                act = a if t == 0 else _sample_index(pis[s], rng)
                r, nxt, _sold = mdps._step(s, act, mdps._demand_from(rng.uniform()))
                ep.append((s, r, nxt))
                s = nxt
            days[a].append(ep)

    def pooled_sd(Vh, lam):
        total = 0.0
        for a in range(M_BANDIT):
            ests = []
            for ep in days[a]:
                acc = 0.0
                for (s, r, nxt) in reversed(ep):
                    acc = r + GAMMA * Vh[nxt] - Vh[s] + GAMMA * lam * acc
                ests.append(acc)
            total += _mean_sd(ests)[1] ** 2
        return math.sqrt(total / M_BANDIT)

    rows = []
    for lam in GAE_LAMBDAS:
        bias = [_gae_exact_bias(a, lam, V0, V, pis, P) for a in range(M_BANDIT)]
        assert max(abs(_gae_exact_bias(a, lam, V, V, pis, P)) for a in range(M_BANDIT)) < 1e-12
        rows.append([f"{lam:g}", _fmt(pooled_sd(V, lam), 2), _fmt(pooled_sd(V0, lam), 2)] + [_fmt(b, 3) for b in bias] +
                    [_fmt(max(bias) - min(bias), 3), "below 1e-12"])
    adv = [mdps._q(0, a, V, GAMMA) - V[0] for a in range(M_BANDIT)]
    top = (
        f"Advantage estimates for the order a made on the empty shelf under the softmax policy of the theorem check (preferences "
        f"{THETA_C}); the exact advantage is A(0, a) = q(0, a) - V(0) = ({', '.join(_fmt(x, 4) for x in adv)}) for a = 0 .. 3. "
        f"Two critics: the exact V of the policy and the stale critic V0, the exact values of the uniform policy "
        f"(V0(0) = {_fmt(V0[0], 4)} against V(0) = {_fmt(V[0], 4)}). The estimate is A_hat = sum_(k < {GAE_DAYS}) "
        f"(gamma lambda)^k delta_k with delta_k = r_k + gamma Vhat(s_(k+1)) - Vhat(s_k) along a rollout that makes the order a on "
        f"day 0 and follows the policy afterwards, the critic's value of the state reached after the {GAE_DAYS}th day closing the "
        f"sum (computed backwards, A = delta + gamma lambda A). Sampled: {GAE_ROLLOUTS} rollouts for each a = 0 .. 3 in this order, "
        f"from the generator of seed {bookrng.seed_of(name):#018x}; each day after the first draws one uniform() for the action "
        f"(inverse CDF) and then one for the demand, and the first day only the demand; the same rollouts serve both critics and "
        f"every lambda. 'sd' is the root of the mean over the four orders of the variance of A_hat (divisor {GAE_ROLLOUTS} - 1). "
        f"'Bias' is exact, from the Markov chain (E[A_hat] - A for the infinite sum): the four columns are for the stale critic "
        f"and 'spread' is their largest minus smallest; the last column is the largest absolute bias of the exact critic."
    )
    table = vl._table(["lambda", "sd, exact critic", "sd, stale critic", "bias a = 0", "bias a = 1", "bias a = 2", "bias a = 3",
                       "spread", "exact critic"], rows)
    return "\n".join([top, "", table])


# ------------------------------------------------------------------ the clipped surrogate


def _clip(r, eps):
    return min(max(r, 1.0 - eps), 1.0 + eps)


def clip_cases():
    eps = PPO_EPS
    rows = []
    for adv in (1.0, -1.0):
        for r in (0.5, 0.9, 1.1, 1.5):
            raw, clipped = r * adv, _clip(r, eps) * adv
            used = min(raw, clipped)
            live = raw <= clipped
            rows.append([f"{adv:+.0f}", f"{r:.1f}", f"{raw:+.2f}", f"{clipped:+.2f}", f"{used:+.2f}",
                         "raw (A r)" if live else "clipped", f"{adv * r:+.2f}" if live else "0"])
    return "\n".join([
        f"The per-sample term min(r A, clip(r, 1 - eps, 1 + eps) A) with eps = {eps} (clip applied to the ratio r, "
        f"clip(r, lo, hi) = min(max(r, lo), hi)), for advantages A = +1 and -1 and four ratios. The winning term is the one "
        f"the minimum picks (the raw one on a tie); its derivative with respect to the log-probability that moves r is "
        f"A r when the raw term wins and 0 when the clipped term wins.", "",
        vl._table(["A", "r", "r A", "clip(r) A", "min", "term that wins", "gradient factor"], rows, align="rrrrrlr")])


# ------------------------------------------------------------------ PPO epochs on one batch


def _ppo_setup(name="policy_gradients.ppo_batch"):
    """The batch: PPO_BATCH samples drawn from pi_old = softmax(THETA_B) by the generator `name`. Returns the name, pi_old,
    the list of (a, d, y), the batch mean of y and the advantages."""
    rng = bookrng.generator(name)
    pi_old = _softmax(list(THETA_B))
    samples = []
    for _ in range(PPO_BATCH):
        a = _sample_index(pi_old, rng)
        d = mdps._demand_from(rng.uniform())
        samples.append((a, d, Y[a][d]))
    mean = 0.0
    for _a, _d, y in samples:
        mean += y
    mean /= PPO_BATCH
    adv = [y - mean for _a, _d, y in samples]
    return name, pi_old, samples, mean, adv


def _ppo_run(eps, nudge, batch_name="policy_gradients.ppo_batch"):
    """PPO_EPOCHS epochs of full-batch gradient ascent on the clipped surrogate (eps = None: the unclipped surrogate).
    Returns one record per epoch, taken at the start of the epoch (before its step) and one after the last step."""
    _name, pi_old, samples, _mean, adv = _ppo_setup(batch_name)
    j_old = _bandit_j(THETA_B)
    theta = list(THETA_B)
    theta[0] += nudge
    out = []
    for e in range(1, PPO_EPOCHS + 2):
        pi = _softmax(theta)
        ratios = [pi[a] / pi_old[a] for a, _d, _y in samples]
        surrogate = 0.0
        grad = [0.0] * M_BANDIT
        clipped = 0
        for (a, _d, _y), A, r in zip(samples, adv, ratios):
            raw = r * A
            used = raw if eps is None else min(raw, _clip(r, eps) * A)
            surrogate += used / PPO_BATCH
            if eps is None or raw <= _clip(r, eps) * A:
                for k in range(M_BANDIT):
                    grad[k] += A * r * ((1.0 if k == a else 0.0) - pi[k]) / PPO_BATCH
            else:
                clipped += 1
        kl = sum(pi_old[a] * math.log(pi_old[a] / pi[a]) for a in range(M_BANDIT))
        k1 = -sum(math.log(r) for r in ratios) / PPO_BATCH
        k3 = sum(r - 1.0 - math.log(r) for r in ratios) / PPO_BATCH
        out.append({"epoch": e, "theta": list(theta), "clipped": clipped, "dev": max(abs(r - 1.0) for r in ratios),
                    "kl": kl, "k1": k1, "k3": k3, "claimed": surrogate, "true": _bandit_j(theta) - j_old, "ratios": ratios,
                    "grad": grad})
        theta = [t + PPO_ETA * g for t, g in zip(theta, grad)]
    return out


def ppo_batch():
    name, pi_old, samples, mean, _adv = _ppo_setup()
    counts = {}
    for a, d, y in samples:
        counts[(a, d)] = counts.get((a, d), 0) + 1
    rows = [[a, d, n, _fmt(Y[a][d], 4), _fmt(Y[a][d] - mean, 4)] for (a, d), n in sorted(counts.items())]
    top = (
        f"One batch of {PPO_BATCH} samples from the behaviour policy pi_old = softmax{THETA_B} = "
        f"({', '.join(_fmt(p, 4) for p in pi_old)}). Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} "
        f"({bookrng.seed_of(name)}); per sample one uniform() for the action (inverse CDF) and then one for the demand "
        f"(D = 0 if below 0.3, 1 if below 0.7, otherwise 2), the samples in drawing order. The outcome is y(a, d) of the one-day "
        f"bandit and the advantage of a sample is y minus the batch mean {_fmt(mean, 4)} (summed in sample order, divisor "
        f"{PPO_BATCH}). The objective depends on a sample only through (a, A), so the table lists each outcome that occurs "
        f"with its count."
    )
    return "\n".join([top, "", vl._table(["a", "d", "samples", "y", "advantage A"], rows)])


def ppo_epochs():
    clipped_runs = [_ppo_run(PPO_EPS, n) for n in VARIANTS]
    plain_runs = [_ppo_run(None, n) for n in VARIANTS]
    rows_c, rows_u = [], []
    for e in PPO_CHECKS:
        c = [run[e - 1] for run in clipped_runs]
        u = [run[e - 1] for run in plain_runs]
        rows_c.append([e, vl._same([str(x["clipped"]) for x in c]), _cell([x["dev"] for x in c], 3), _cell([x["kl"] for x in c], 4),
                       _cell([x["claimed"] for x in c], 4), _cell([x["true"] for x in c], 4)])
        rows_u.append([e, _cell([x["dev"] for x in u], 3), _cell([x["kl"] for x in u], 4), _cell([x["claimed"] for x in u], 4),
                       _cell([x["true"] for x in u], 4)])
    top = (
        f"The batch of the previous table, {PPO_EPOCHS} epochs of gradient ascent on its surrogate L(theta) = (1/{PPO_BATCH}) "
        f"sum_i min(r_i A_i, clip(r_i, 1 - eps, 1 + eps) A_i), r_i = pi_theta(a_i) / pi_old(a_i), starting from theta_B = "
        f"{THETA_B}. An epoch is one step on the whole batch (one minibatch of {PPO_BATCH} samples in drawing order, no "
        f"shuffling): the ratios are recomputed from the current preferences, a sample whose raw term r A is not larger than its "
        f"clipped term contributes A r (e_a - pi_theta) / {PPO_BATCH} to the gradient (summed in sample order) and any other "
        f"sample contributes nothing, and the preferences move by eta = {PPO_ETA} times the gradient. The advantages stay "
        f"fixed. The first table is the clipped run, eps = {PPO_EPS}, the second the same run without the clip (every sample "
        f"contributes). Each row describes the preferences at the start of the stated epoch: 'clipped' counts the samples with "
        f"no gradient, 'dev' is the largest |r - 1|, 'KL' is the exact KL(pi_old || pi_theta) in nats, 'claimed' is the "
        f"surrogate's own value, which is 0 at the start since the advantages sum to 0, and 'true' is J(theta) - J(theta_B) "
        f"from the exact q of the bandit."
    )
    table_c = vl._table(["epoch", "clipped", "dev", "KL", "claimed", "true"], rows_c)
    table_u = vl._table(["epoch", "dev", "KL", "claimed", "true"], rows_u)
    j_old = _bandit_j(THETA_B)
    return "\n".join([top, "", "Clipped:", "", table_c, "", "Unclipped:", "", table_u, "", f"J(theta_B) = {_fmt(j_old, 4)}. " + RULE_TEXT])


def kl_estimators():
    runs = [_ppo_run(PPO_EPS, n) for n in VARIANTS]
    rows = []
    for e in (5, 10, 20):
        c = [run[e - 1] for run in runs]
        rows.append([e, _cell([x["kl"] for x in c], 5), _cell([x["k1"] for x in c], 5), _cell([x["k3"] for x in c], 5)])
    return "\n".join([
        f"The clipped run of the previous table. At the start of the stated epoch the exact KL(pi_old || pi_theta) over the four "
        f"orders, the estimate k1 = -(1/{PPO_BATCH}) sum_i ln r_i and the estimate k3 = (1/{PPO_BATCH}) sum_i (r_i - 1 - ln r_i) "
        f"from the batch's own samples (r_i as before; natural logarithms).", "",
        vl._table(["epoch", "exact KL", "k1", "k3"], rows)])


def ppo_clip_binding():
    run = _ppo_run(PPO_EPS, 0.0)
    first = next(r for r in run if r["clipped"] > 0)
    _name, pi_old, samples, _mean, adv = _ppo_setup()
    pi = _softmax(first["theta"])
    seen = {}
    for (a, d, _y), A in zip(samples, adv):
        seen.setdefault((a, d), [0, A])[0] += 1
    rows, kept, lost = [], 0.0, 0.0
    for (a, d), (n, A) in sorted(seen.items()):
        r = pi[a] / pi_old[a]
        raw, cl = r * A, _clip(r, PPO_EPS) * A
        live = raw <= cl
        mass = n * abs(A * r) / PPO_BATCH
        if live:
            kept += mass
        else:
            lost += mass
        rows.append([a, d, n, _fmt(A, 4), _fmt(r, 4), _fmt(raw, 4), _fmt(cl, 4), "raw" if live else "clipped",
                     _fmt(A * r, 4) if live else "0"])
    top = (
        f"The first epoch of the clipped run at which a sample is clipped, epoch {first['epoch']}: the preferences are "
        f"theta = ({', '.join(_fmt(t, 4) for t in first['theta'])}), pi_theta = ({', '.join(_fmt(p, 4) for p in pi)}), against "
        f"pi_old = ({', '.join(_fmt(p, 4) for p in pi_old)}). For every outcome of the batch: its count n, advantage A, ratio "
        f"r = pi_theta(a) / pi_old(a) (it depends on a only), the raw term r A, the clipped term clip(r, {1 - PPO_EPS:.1f}, "
        f"{1 + PPO_EPS:.1f}) A, the winner of the minimum and its gradient factor (A r when raw, else 0)."
    )
    tail = (
        f"{first['clipped']} of {PPO_BATCH} samples are clipped. The sum over the samples of |A r| / {PPO_BATCH} is "
        f"{_fmt(kept + lost, 4)}; the clipped samples carry {_fmt(lost, 4)} of it, which this epoch does not use."
    )
    return "\n".join([top, "", vl._table(["a", "d", "n", "A", "r", "r A", "clip(r) A", "wins", "gradient factor"], rows, align="rrrrrrrlr"),
                      "", tail])


def clip_over_batches():
    """The investigation of the chapter's last exercise: the clipped and the unclipped run on many independent batches."""
    rows = []
    for label, eps in (("clipped, eps = 0.2", PPO_EPS), ("unclipped", None)):
        below, mean, worst, kl = [], [], [], []
        for nudge in VARIANTS:
            true, kls = [], []
            for i in range(PPO_BATCHES):
                rec = _ppo_run(eps, nudge, f"policy_gradients.batches.batch{i}")[PPO_EPOCHS]    # after the 20th step
                true.append(rec["true"])
                kls.append(rec["kl"])
            below.append(sum(1 for t in true if t < 0.0))
            mean.append(sum(true) / PPO_BATCHES)
            worst.append(min(true))
            kl.append(sum(kls) / PPO_BATCHES)
        rows.append([label, vl._same([f"{b} of {PPO_BATCHES}" for b in below]), _cell(mean, 4), _cell(worst, 4), _cell(kl, 4)])
    top = (
        f"{PPO_BATCHES} independent batches of {PPO_BATCH} samples, batch i drawn exactly as in the batch table but by the "
        f"generator named policy_gradients.batches.batch<i> (seed_of of that name, i = 0 .. {PPO_BATCHES - 1}): per sample one "
        f"uniform() for the action (inverse CDF over pi_old = softmax{THETA_B}) and then one for the demand, the advantage "
        f"the sample's y minus the batch's mean y. On each batch the run of the epochs table starts from theta_B and takes "
        f"{PPO_EPOCHS} steps (eta = {PPO_ETA}), clipped (eps = {PPO_EPS}) or unclipped. 'Below' counts the batches whose true "
        f"change J(theta) - J(theta_B) after the {PPO_EPOCHS}th step is negative, 'mean' and 'worst' are the mean and the "
        f"smallest of that change over the batches (summed in batch order, divisor {PPO_BATCHES}), 'KL' the mean exact "
        f"KL(pi_old || pi_theta) after the {PPO_EPOCHS}th step. " + RULE_TEXT
    )
    return "\n".join([top, "", vl._table(["run", "below J(theta_B)", "mean true change", "worst true change", "mean KL"], rows,
                                         align="lrrrr")])


# ------------------------------------------------------------------ work


def work_of_updates():
    m = M_BANDIT
    softmax_ops = 4 * m - 1           # m subtractions, m exponentials, m - 1 additions, m divisions
    row_ops = 3 * m + 2               # (e_a - pi): m, the shared scale of 2 multiplications, one multiply and one add per entry
    rows = [
        ["softmax over m orders", f"m subtractions, m exponentials, m - 1 additions, m divisions: 4m - 1", softmax_ops],
        ["preference row step", "m subtractions (e_a - pi), the scale eta gamma^t delta (2 multiplications), m multiplications "
                                "and m additions: 3m + 2", row_ops],
        ["REINFORCE, one day", "return recursion 2, softmax, row step, running power gamma^t 1", 2 + softmax_ops + row_ops + 1],
        ["REINFORCE with baseline, one day", "the same, plus the advantage subtraction 1 and the critic step "
                                             "(G - V, alpha_c times, add) 3", 2 + softmax_ops + row_ops + 1 + 1 + 3],
        ["actor-critic, one day", "delta 3, softmax, row step, critic step 2, running power 1", 3 + softmax_ops + row_ops + 2 + 1],
    ]
    clipped_run, plain_run = _ppo_run(PPO_EPS, 0.0), _ppo_run(None, 0.0)

    def ppo_cost(run):
        total = 0
        for rec in run[:PPO_EPOCHS]:
            live = PPO_BATCH - rec["clipped"]
            # softmax once per epoch; per sample the ratio (1), r A and clip(r) A (2), and for a live sample the scale
            # A r / N (1) and a row of m subtractions, m multiplications and m additions
            total += softmax_ops + PPO_BATCH * 3 + live * (1 + 3 * m)
        return total

    clipped_total = sum(rec["clipped"] for rec in clipped_run[:PPO_EPOCHS])
    top = (
        "One operation is an addition, subtraction, multiplication, division, exponential or logarithm; comparisons, min, max, "
        "clamps and indexing are not counted. m is the number of actions of a state (m = 4 on the empty shelf). The PPO rows "
        f"count the batch of {PPO_BATCH} samples over the epochs 1 to {PPO_EPOCHS} of the unnudged run of the epoch table: the clipped "
        f"run has {clipped_total} clipped sample-epochs (samples with no gradient, summed over the {PPO_EPOCHS} epochs, the "
        f"'clipped' column of the epoch table at every epoch, not only the printed ones) and {PPO_BATCH * PPO_EPOCHS - clipped_total} "
        f"live ones; the unclipped run has {PPO_BATCH * PPO_EPOCHS} live ones."
    )
    run_rows = [
        [f"PPO, {PPO_EPOCHS} epochs, clipped (eps = {PPO_EPS})",
         "softmax once per epoch; per sample ratio 1, r A and clip(r) A 2; per unclipped sample the scale A r / N 1 and the "
         f"row 3m", f"{ppo_cost(clipped_run):,}"],
        [f"PPO, {PPO_EPOCHS} epochs, unclipped", "the same with every sample unclipped", f"{ppo_cost(plain_run):,}"],
    ]
    return "\n".join([top, "", vl._table(["update", "count", "FLOPs, m = 4"], [[a, b, f"{c:,}"] for a, b, c in rows] + run_rows,
                                         align="llr")])
