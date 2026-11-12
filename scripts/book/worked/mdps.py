# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Markov Decision Processes".

Plain Python 3, standard library only, written from the chapter's own description: a four-state
inventory problem (a shelf of at most three units, a demand of zero, one or two units a day, an
order of up to the free room on the shelf) with every transition probability and expected reward
printed; one trajectory with its returns under four discounts; the sums of a constant reward;
policy evaluation by solving the linear Bellman system (with the floating-point operations of the
elimination counted) and by sweeps; a sampled estimate of the same value; value iteration with
its stopping rule, the contraction bound on its number of sweeps and the greedy policy it ends in
(with the tie rule that decides between equal actions); the optimal policy as the discount grows;
policy iteration; and the work of a sweep against the work of a solve. Each public function
without a leading underscore that takes no arguments returns a markdown string that the chapter
embeds between `<!-- worked: mdps.<name> -->` markers. Underscore functions and the constants are
shared with the chapter's figures, so a picture and the table beside it cannot disagree.

Swedish Embedded AB implements solutions for sequential decision problems that are solved exactly
before they are learned. If your team needs expertise in Markov decision processes and dynamic
programming, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng

# ------------------------------------------------------------------ the inventory problem

CAP = 3                               # the shelf holds at most 3 units; a state is the stock in the morning
N_STATES = CAP + 1
DEMAND = (0.3, 0.4, 0.3)              # P(D = 0), P(D = 1), P(D = 2), the same every day
PRICE, UNIT, FEE, HOLD = 4.0, 2.0, 0.5, 1.0
GAMMA = 0.9                           # the discount of the main runs
TOL = 1e-8                            # stop when the largest change of a sweep is below this
TIE = 1e-9                            # actions within this of the best are tied

TRAJ_NAME, TRAJ_DAYS, TRAJ_START = "mdps.trajectory", 10, 2
MC_NAME, MC_EPISODES, MC_DAYS, MC_BLOCKS = "mdps.rollouts", 2000, 100, 5
SWITCH_PAIR = ((1, 0, 0, 0), (2, 0, 0, 0))   # order one, or order two, when the shelf is empty

NEVER = (0, 0, 0, 0)
GAMMAS = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def _n_actions(s):
    return CAP - s + 1                # an order a = 0 .. CAP - s fits on the shelf


def _model():
    """P[s][a][j] and R[s][a] for every legal pair, accumulated over the demand in increasing d."""
    P, R = [], []
    for s in range(N_STATES):
        Ps, Rs = [], []
        for a in range(_n_actions(s)):
            x = s + a
            nxt = [0.0] * N_STATES
            reward = 0.0
            for d, p in enumerate(DEMAND):
                sold = min(d, x)
                left = x - sold
                reward += p * (PRICE * sold - HOLD * left)
                nxt[left] += p
            reward -= UNIT * a + (FEE if a > 0 else 0.0)
            Ps.append(nxt)
            Rs.append(reward)
        P.append(Ps)
        R.append(Rs)
    return P, R


PROB, REW = _model()
PAIRS = sum(_n_actions(s) for s in range(N_STATES))


def _fmt(x, nd=3):
    s = f"{x:.{nd}f}"
    return s.replace("-0.000", "0.000") if float(s) == 0 else s


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _seed_text(name):
    seed = bookrng.seed_of(name)
    return f"seed {seed:#018x} ({seed})"


def _demand_from(u):
    """Inverse of the demand distribution: one uniform() gives D = 0, 1 or 2."""
    if u < DEMAND[0]:
        return 0
    return 1 if u < DEMAND[0] + DEMAND[1] else 2


def _step(s, a, d):
    """One day: (reward, next stock) for stock s, order a and demand d."""
    x = s + a
    sold = min(d, x)
    left = x - sold
    return PRICE * sold - HOLD * left - UNIT * a - (FEE if a > 0 else 0.0), left, sold


# ------------------------------------------------------------------ the algorithms


def _q(s, a, V, g):
    acc = 0.0
    for j in range(N_STATES):
        acc += PROB[s][a][j] * V[j]
    return REW[s][a] + g * acc


def _greedy(V, g):
    """The smallest action whose q is within TIE of the best, for every state."""
    pi = []
    for s in range(N_STATES):
        qs = [_q(s, a, V, g) for a in range(_n_actions(s))]
        best = max(qs)
        pi.append(next(a for a, q in enumerate(qs) if q >= best - TIE))
    return tuple(pi)


def _solve(A, b):
    """Gaussian elimination with partial pivoting; returns x and the FLOPs of the elimination
    (a multiply, an add, a subtract and a divide are one operation each; comparisons and row
    swaps are not counted)."""
    n = len(b)
    A = [row[:] + [b[i]] for i, row in enumerate(A)]
    flops = 0
    for k in range(n - 1):
        piv = max(range(k, n), key=lambda i: abs(A[i][k]))
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            flops += 1
            for j in range(k + 1, n + 1):
                A[i][j] -= f * A[k][j]
                flops += 2
            A[i][k] = 0.0
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = A[i][n]
        for j in range(i + 1, n):
            acc -= A[i][j] * x[j]
            flops += 2
        x[i] = acc / A[i][i]
        flops += 1
    return x, flops


def _policy_system(pi, g):
    """A = I - g P_pi (row s is the transition row of the action pi[s]) and r = R_pi."""
    A = [[(1.0 if i == j else 0.0) - g * PROB[i][pi[i]][j] for j in range(N_STATES)] for i in range(N_STATES)]
    r = [REW[s][pi[s]] for s in range(N_STATES)]
    return A, r


def _evaluate_solve(pi, g):
    A, r = _policy_system(pi, g)
    V, flops = _solve(A, r)
    return V, flops + 2 * N_STATES * N_STATES   # forming A: n^2 multiplies, n^2 subtracts


def _evaluate_sweeps(pi, g, tol=TOL):
    """V_{k+1}(s) = R(s, pi s) + g sum_j P V_k(j) from V_0 = 0; returns V, sweeps, deltas."""
    V = [0.0] * N_STATES
    deltas = []
    while True:
        new = [_q(s, pi[s], V, g) for s in range(N_STATES)]
        delta = max(abs(x - y) for x, y in zip(new, V))
        V = new
        deltas.append(delta)
        if delta < tol:
            return V, len(deltas), deltas


def _value_iteration(g, tol=TOL):
    """V_{k+1}(s) = max_a q(s, a; V_k) from V_0 = 0; returns V, sweeps, deltas, history."""
    V = [0.0] * N_STATES
    deltas, hist = [], [V[:]]
    while True:
        new = [max(_q(s, a, V, g) for a in range(_n_actions(s))) for s in range(N_STATES)]
        delta = max(abs(x - y) for x, y in zip(new, V))
        V = new
        deltas.append(delta)
        hist.append(V[:])
        if delta < tol:
            return V, len(deltas), deltas, hist


def _policy_iteration(pi, g):
    """Evaluate by solving, improve greedily, stop when the policy no longer changes."""
    trace = []
    while True:
        V, flops = _evaluate_solve(pi, g)
        new = _greedy(V, g)
        trace.append((pi, V, new, flops))
        if new == pi:
            return pi, V, trace
        pi = new


def _optimal(g):
    V, k, deltas, hist = _value_iteration(g)
    pi = _greedy(V, g)
    return pi, V, k, deltas


def _sweeps_bound(first_delta, g, tol=TOL):
    """Smallest k with g^(k-1) * first_delta < tol, the sweep at which the contraction bound
    guarantees a change below tol."""
    k = 1
    while g ** (k - 1) * first_delta >= tol:
        k += 1
    return k


def _sweep_flops(pairs=PAIRS, n=N_STATES):
    """Per pair: n multiplies and n - 1 adds for the sum, one multiply by g, one add of R."""
    return pairs * (2 * n + 1)


def _elimination_flops(n):
    """The operations of _solve for an n by n system, without pivoting swaps: the same loops."""
    total = 0
    for k in range(n - 1):
        total += (n - 1 - k) * (1 + 2 * (n - k))
    for i in range(n - 1, -1, -1):
        total += 2 * (n - 1 - i) + 1
    return total


def _policy_chain(pi):
    return [PROB[s][pi[s]] for s in range(N_STATES)]


def _pi_text(pi):
    return "(" + ", ".join(str(a) for a in pi) + ")"


# ------------------------------------------------------------------ the blocks


def inventory_mdp():
    rows = []
    for s in range(N_STATES):
        for a in range(_n_actions(s)):
            rows.append([s, a] + [_fmt(p, 1) for p in PROB[s][a]] + [_fmt(REW[s][a], 3)])
    out = [
        f"A shelf holds at most {CAP} units. In the morning there are s units on it (the state); the keeper orders a units "
        f"(the action, a = 0 .. {CAP} - s, so the shelf never overflows) and they arrive at once. During the day D units are "
        f"asked for, with P(D = 0) = {DEMAND[0]}, P(D = 1) = {DEMAND[1]}, P(D = 2) = {DEMAND[2]}, the same every day and independent "
        f"of everything before. The units sold are min(D, s + a) and the stock left over, which is tomorrow's state, is s + a minus "
        f"the units sold. The reward of the day is {PRICE:g} per unit sold, minus {UNIT:g} per unit ordered, minus {FEE:g} if an order "
        f"is placed at all (a > 0), minus {HOLD:g} per unit left on the shelf in the evening. R(s, a) is the mean of that reward over D, "
        f"summed over d = 0, 1, 2 in this order. P(j | s, a) is the probability that tomorrow's stock is j.",
        "",
        _table(["s", "a", "P(0|s,a)", "P(1|s,a)", "P(2|s,a)", "P(3|s,a)", "R(s,a)"], rows),
    ]
    return "\n".join(out)


def trajectory_returns():
    pi = (1, 0, 0, 0)
    rng = bookrng.generator(TRAJ_NAME)
    gammas = (0.0, 0.5, 0.9, 1.0)
    s = TRAJ_START
    steps = []
    for t in range(TRAJ_DAYS):
        a = pi[s]
        d = _demand_from(rng.uniform())
        r, left, sold = _step(s, a, d)
        steps.append((t, s, a, d, sold, left, r))
        s = left
    G = {g: [0.0] * (TRAJ_DAYS + 1) for g in gammas}
    for g in gammas:
        for t in range(TRAJ_DAYS - 1, -1, -1):
            G[g][t] = steps[t][6] + g * G[g][t + 1]
    rows = []
    for (t, s, a, d, sold, left, r) in steps:
        rows.append([t, s, a, d, sold, left, _fmt(r, 1)] + [_fmt(G[g][t], 3) for g in gammas])
    out = [
        f"Policy: order one unit when the shelf is empty, nothing otherwise, written (1, 0, 0, 0) with one action per stock 0 .. 3. "
        f"Start: s_0 = {TRAJ_START}. Generator: the book's SplitMix64, {_seed_text(TRAJ_NAME)}; one uniform() per day in order, "
        f"D = 0 if u < 0.3, D = 1 if u < 0.7, otherwise D = 2. The return G_t of day t is the sum over k = 0 .. {TRAJ_DAYS - 1} - t "
        f"of gamma^k r_(t+k); the days after day {TRAJ_DAYS - 1} are not simulated and count as reward 0 (so G_{TRAJ_DAYS} = 0 and "
        f"G_t = r_t + gamma G_(t+1)). The last four columns are G_t for gamma = 0, 0.5, 0.9 and 1.",
        "",
        _table(["t", "s_t", "a_t", "D", "sold", "s_(t+1)", "r_t", "G (0)", "G (0.5)", "G (0.9)", "G (1)"], rows),
    ]
    return "\n".join(out)


def discount_sums():
    horizons = (10, 100, 1000)
    rows = []
    for g in (0.5, 0.9, 0.99, 0.999):
        cells = [f"{(1 - g ** T) / (1 - g):.2f}" for T in horizons]
        rows.append([f"{g:g}", f"{1 / (1 - g):.0f}"] + cells)
    rows.append(["1", "none"] + [str(T) for T in horizons])
    out = [
        "A reward of 1 on every day. The sum of the first T discounted rewards is the sum over k = 0 .. T - 1 of gamma^k, "
        "which is (1 - gamma^T) / (1 - gamma) for gamma < 1 and T for gamma = 1; 1 / (1 - gamma) is the limit as T grows.",
        "",
        _table(["gamma", "1/(1-gamma)", "T = 10", "T = 100", "T = 1000"], rows),
    ]
    return "\n".join(out)


def evaluate_by_solving():
    g = GAMMA
    pols = (("never order", NEVER), ("order 1 when empty", (1, 0, 0, 0)), ("order 2 when empty", (2, 0, 0, 0)),
            ("order 3 when empty", (3, 0, 0, 0)))
    A, r = _policy_system((1, 0, 0, 0), g)
    V, _ = _solve(A, r)
    resid = max(abs(sum(A[i][j] * V[j] for j in range(N_STATES)) - r[i]) for i in range(N_STATES))
    sysrows = [[f"{A[i][j]:+.3f}" for j in range(N_STATES)] + [_fmt(r[i], 3), _fmt(V[i], 4)] for i in range(N_STATES)]
    rows = []
    for name, pi in pols:
        Vp, _ = _evaluate_solve(pi, g)
        rows.append([name, _pi_text(pi)] + [_fmt(v, 3) for v in Vp])
    out = [
        f"gamma = {g}. The system for the policy (1, 0, 0, 0): row s of A = I - gamma P_pi holds the row P(. | s, pi(s)) times "
        f"-gamma, plus 1 on the diagonal; the right side is r(s) = R(s, pi(s)); the solution is found by Gaussian elimination with partial "
        f"pivoting. The largest entry of |A V - r| is {resid:.1e} (below 1e-12).",
        "",
        _table(["row", "A[s,0]", "A[s,1]", "A[s,2]", "A[s,3]", "r(s)", "V(s)"], [[i] + sysrows[i] for i in range(N_STATES)]),
        "",
        f"The value of four policies (one action per stock 0 .. 3), each from its own system, gamma = {g}:",
        "",
        _table(["policy", "pi", "V(0)", "V(1)", "V(2)", "V(3)"], rows, align="llrrrr"),
    ]
    return "\n".join(out)


def evaluate_by_sweeps():
    g, pi = GAMMA, (1, 0, 0, 0)
    exact, _ = _evaluate_solve(pi, g)
    V, k, deltas = _evaluate_sweeps(pi, g)
    rows = []
    shown = (1, 2, 3, 5, 10, 20, 50, 100, k)
    for kk in shown:
        Vk = [0.0] * N_STATES
        for _ in range(kk):
            Vk = [_q(s, pi[s], Vk, g) for s in range(N_STATES)]
        err = max(abs(x - y) for x, y in zip(Vk, exact))
        bound = max(abs(x) for x in exact) * g ** kk
        rows.append([kk, _fmt(Vk[0], 4), _fmt(Vk[3], 4), f"{deltas[kk - 1]:.2e}", f"{err:.2e}", f"{bound:.2e}"])
    final_err = max(abs(x - y) for x, y in zip(V, exact))
    out = [
        f"Policy (1, 0, 0, 0), gamma = {g}, V_0 = 0, V_(k+1)(s) = R(s, pi(s)) + gamma * sum_j P(j | s, pi(s)) V_k(j) for every s "
        f"(each sum over j = 0 .. 3 in order, every state updated from the old vector). change_k is the largest |V_k(s) - V_(k-1)(s)|; "
        f"the iteration stops at the first k with change_k < {TOL:g}. error_k is the largest |V_k(s) - V(s)| against the solved system "
        f"and the last column is gamma^k times the largest |V(s)|.",
        "",
        _table(["k", "V_k(0)", "V_k(3)", "change_k", "error_k", "gamma^k max V"], rows),
        "",
        f"It stops after {k} sweeps with the largest error {final_err:.2e} against the solved system.",
    ]
    return "\n".join(out)


def monte_carlo_value():
    g = GAMMA
    pi = (1, 0, 0, 0)
    rng = bookrng.generator(MC_NAME)
    returns = []
    for _ in range(MC_EPISODES):
        s, disc, G = 0, 1.0, 0.0
        for _t in range(MC_DAYS):
            r, s, _ = _step(s, pi[s], _demand_from(rng.uniform()))
            G += disc * r
            disc *= g
        returns.append(G)
    mean = sum(returns) / len(returns)
    var = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(len(returns))
    exact, _ = _evaluate_solve(pi, g)
    block = MC_EPISODES // MC_BLOCKS
    bmeans = [sum(returns[i * block:(i + 1) * block]) / block for i in range(MC_BLOCKS)]
    out = [
        f"Policy (1, 0, 0, 0), gamma = {g}, start s_0 = 0. Generator: the book's SplitMix64, {_seed_text(MC_NAME)}; "
        f"{MC_EPISODES} episodes of {MC_DAYS} days one after the other, one uniform() per day (the demand rule of the trajectory), "
        f"each return G = sum over t of gamma^t r_t with the running power gamma^t multiplied once per day, days after the "
        f"{MC_DAYS}th ignored (they weigh less than {g ** MC_DAYS * max(exact):.1e} in total). The mean is the plain sum over the "
        f"episodes in order divided by {MC_EPISODES}, the standard deviation uses the divisor {MC_EPISODES} - 1. The {MC_BLOCKS} "
        f"block means are those of the consecutive blocks of {block} episodes.",
        "",
        _table(["quantity", "value"], [
            ["mean of the sampled returns", _fmt(mean, 3)],
            ["standard deviation of one return", _fmt(sd, 3)],
            ["standard error of the mean", _fmt(se, 3)],
            ["V(0) from the linear system", _fmt(exact[0], 3)],
            ["difference in standard errors", f"{(mean - exact[0]) / se:+.3f}"],
            ["block means"] + [", ".join(_fmt(b, 2) for b in bmeans)],
        ]),
    ]
    return "\n".join(out)


def value_iteration_run():
    g = GAMMA
    V, k, deltas, hist = _value_iteration(g)
    rows = []
    for kk in (0, 1, 2, 3, 4, 5, 10, 20, 50, 100, k):
        rows.append([kk] + [_fmt(v, 3) for v in hist[kk]] + ["" if kk == 0 else f"{deltas[kk - 1]:.2e}"])
    qrows = []
    pi = _greedy(V, g)
    for s in range(N_STATES):
        for a in range(_n_actions(s)):
            q = _q(s, a, V, g)
            qrows.append([s, a, _fmt(q, 4), "chosen" if pi[s] == a else ""])
    out = [
        f"gamma = {g}, V_0 = 0, V_(k+1)(s) = max over the legal a of [R(s, a) + gamma * sum_j P(j | s, a) V_k(j)], every state "
        f"updated from the old vector; stop at the first k with change_k < {TOL:g}, change_k being the largest |V_k(s) - V_(k-1)(s)|.",
        "",
        _table(["k", "V_k(0)", "V_k(1)", "V_k(2)", "V_k(3)", "change_k"], rows),
        "",
        f"It stops after {k} sweeps. The one-step lookahead q(s, a) = R(s, a) + gamma * sum_j P(j | s, a) V_{k}(j) of the last vector, "
        f"and the greedy policy: the smallest a whose q is within {TIE:g} of the largest q of its state.",
        "",
        _table(["s", "a", "q(s, a)", "greedy"], qrows, align="rrrl"),
        "",
        f"The greedy policy is {_pi_text(pi)}.",
    ]
    return "\n".join(out)


def greedy_tie():
    rows = []
    for s in (0, 1):
        for a in range(_n_actions(s)):
            rows.append([s, a, _fmt(REW[s][a], 3)])
    V0 = [0.0] * N_STATES
    pi = _greedy(V0, 0.0)
    out = [
        f"gamma = 0, so q(s, a) = R(s, a) whatever V is. The first rows of the table of the inventory problem, and the greedy "
        f"policy with the tie rule (smallest a within {TIE:g} of the best):",
        "",
        _table(["s", "a", "q(s, a) = R(s, a)"], rows),
        "",
        f"At s = 0 the actions 0 and 1 are tied at {_fmt(max(REW[0]), 3)}, so the greedy policy is {_pi_text(pi)}, which orders nothing.",
    ]
    return "\n".join(out)


def contraction_prediction():
    rows = []
    for g in (0.5, 0.9, 0.99):
        V, k, deltas, _ = _value_iteration(g)
        pi = _greedy(V, g)
        star, _ = _evaluate_solve(pi, g)
        err = max(abs(x - y) for x, y in zip(V, star))
        bound = g * deltas[-1] / (1 - g)
        kb = _sweeps_bound(deltas[0], g)
        late = deltas[-1] / deltas[-2]
        rows.append([f"{g:g}", kb, k, f"{late:.4f}", f"{deltas[0] / g:.3f}", f"{deltas[-1] / g ** k:.3f}", f"{err:.2e}", f"{bound:.2e}"])
    out = [
        f"V_0 = 0 and the stopping rule change_k < {TOL:g}, as above; change_1 = {max(max(r) for r in REW):.3f} for every gamma (the largest R(s, a)). "
        f"predicted is the smallest k with gamma^(k-1) * change_1 < {TOL:g}; actual is the sweep at which the run stopped. "
        f"The bound says change_k <= (change_1 / gamma) * gamma^k, whose constant is the column 'bound constant'; 'actual constant' "
        f"is change_k / gamma^k at the stop. error is the largest |V_k(s) - V*(s)|, V* being the solved value of the greedy policy "
        f"of V_k; bound is gamma * change_k / (1 - gamma).",
        "",
        _table(["gamma", "predicted k", "actual k", "change_k / change_(k-1)", "bound constant", "actual constant", "error", "bound"], rows),
    ]
    return "\n".join(out)


def policy_settles():
    rows = []
    for g in (0.5, 0.9, 0.95, 0.99):
        V, k, _, hist = _value_iteration(g)
        final = _greedy(V, g)
        pols = [_greedy(hist[i], g) for i in range(1, k + 1)]       # the greedy policy of V_1 .. V_k
        first = next(i for i in range(k) if all(p == final for p in pols[i:])) + 1
        path = []
        for i, p in enumerate(pols, start=1):
            if not path or path[-1][1] != p:
                path.append((i, p))
        rows.append([f"{g:g}", k, first, _pi_text(final), "; ".join(f"from sweep {i}: {_pi_text(p)}" for i, p in path)])
    out = [
        f"The greedy policy (smallest action within {TIE:g} of the best lookahead) of the vector V_k of value iteration, for "
        f"k = 1, 2, ..., up to the sweep at which the run stops (change_k < {TOL:g}). 'settled at' is the first k from which "
        f"the greedy policy of V_k never again differs from the final one.",
        "",
        _table(["gamma", "sweeps to stop", "settled at", "final policy", "greedy policy of V_k along the run"], rows, align="rrrll"),
    ]
    return "\n".join(out)


def _switch_gamma():
    """The discount at which ordering two units overtakes ordering one when empty, by bisection on
    V(0) of the two solved policies (60 halvings of [0.9, 0.95])."""
    lo, hi = 0.9, 0.95

    def gap(g):
        a, _ = _evaluate_solve(SWITCH_PAIR[0], g)
        b, _ = _evaluate_solve(SWITCH_PAIR[1], g)
        return b[0] - a[0]

    for _ in range(60):
        mid = (lo + hi) / 2
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def discount_sweep():
    rows = []
    for g in GAMMAS:
        pi, V, k, _ = _optimal(g)
        rows.append([f"{g:g}", _pi_text(pi), _fmt(V[0], 3), _fmt(V[3], 3), k, f"{1 / (1 - g):.1f}"])
    sw = _switch_gamma()
    a, _ = _evaluate_solve(SWITCH_PAIR[0], sw)
    b, _ = _evaluate_solve(SWITCH_PAIR[1], sw)
    out = [
        f"Value iteration from V_0 = 0 with the stopping rule change_k < {TOL:g}, then the greedy policy with the tie rule, for "
        f"each gamma. sweeps is the number of sweeps until the stop, and 1/(1-gamma) the effective horizon.",
        "",
        _table(["gamma", "greedy policy", "V(0)", "V(3)", "sweeps", "1/(1-gamma)"], rows, align="lrrrrr"),
        "",
        f"The two policies (1, 0, 0, 0) and (2, 0, 0, 0) have the same V(0) at gamma = {sw:.4f} (V(0) = {a[0]:.3f} and "
        f"{b[0]:.3f}); this is found by 60 halvings of the interval [0.9, 0.95] on the difference of their solved values, and "
        f"the second is better above it.",
    ]
    return "\n".join(out)


def policy_iteration_run():
    rows_all = []
    out = []
    for g in (0.9, 0.99):
        pi, V, trace = _policy_iteration(NEVER, g)
        rows = []
        for i, (p, Vp, new, flops) in enumerate(trace):
            rows.append([i, _pi_text(p)] + [_fmt(v, 3) for v in Vp] + [_pi_text(new)])
        out += [
            f"gamma = {g}, starting policy (0, 0, 0, 0): each round solves the system of the current policy and takes the greedy "
            f"policy of the solved values; the stop is a round whose greedy policy equals its own.",
            "",
            _table(["round", "policy", "V(0)", "V(1)", "V(2)", "V(3)", "greedy policy"], rows, align="llrrrrl"),
            "",
            f"{len(trace)} rounds ({len(trace)} solves), final policy {_pi_text(pi)}.",
            "",
        ]
        rows_all.append((g, len(trace)))
    return "\n".join(out).rstrip()


def work_per_sweep():
    n = N_STATES
    sweep = _sweep_flops()
    elim = _elimination_flops(n)
    A, r = _policy_system((1, 0, 0, 0), GAMMA)
    _, counted = _solve(A, r)
    assert counted == elim
    form = 2 * n * n
    pol_sweep = n * (2 * n + 1)
    g = GAMMA
    Vv, kvi, _, _ = _value_iteration(g)
    _, kev, _ = _evaluate_sweeps((1, 0, 0, 0), g)
    pi, V, trace = _policy_iteration(NEVER, g)
    rounds = len(trace)
    rows = [
        ["one sweep of value iteration", f"{PAIRS} pairs times {2 * n + 1}", sweep],
        ["one sweep of evaluating a fixed policy", f"{n} states times {2 * n + 1}", pol_sweep],
        ["forming I - gamma P_pi", f"2 times {n}^2", form],
        ["elimination and back substitution", "counted", elim],
        ["one round of policy iteration", f"{form} + {elim} + {sweep}", form + elim + sweep],
        [f"value iteration to the stop ({kvi} sweeps)", f"{kvi} times {sweep}", kvi * sweep],
        [f"policy iteration to the stop ({rounds} rounds)", f"{rounds} times {form + elim + sweep}", rounds * (form + elim + sweep)],
        [f"evaluating one policy by sweeps ({kev} sweeps)", f"{kev} times {pol_sweep}", kev * pol_sweep],
        ["evaluating one policy by solving", f"{form} + {elim}", form + elim],
    ]
    scale = []
    for ns in (4, 100, 10_000):
        sw = ns * 3 * (2 * ns + 1)
        el = _elimination_flops(ns)
        byt = ns * 3 * ns * 4
        scale.append([ns, f"{sw:,}", f"{el:,}", f"{byt:,}", f"{sw / byt:.2f}"])
    out = [
        f"Counts for the inventory problem (n = {n} states, {PAIRS} legal pairs), gamma = {g}, from the stopping rule "
        f"change_k < {TOL:g}. One operation is a multiply, an add, a subtract or a divide; comparisons are not counted. A pair costs "
        f"{n} multiplies and {n - 1} adds for the sum over j, one multiply by gamma and one add of R, that is {2 * n + 1}.",
        "",
        _table(["what", "count", "FLOPs"], rows, align="llr"),
        "",
        "The same counts for a dense problem with 3 actions in every state and n states (a transition table of n times 3 times n entries of "
        "4 bytes, read once per sweep):",
        "",
        _table(["n", "FLOPs per sweep", "FLOPs of one solve", "bytes of P", "FLOP per byte"], scale),
    ]
    return "\n".join(out)
