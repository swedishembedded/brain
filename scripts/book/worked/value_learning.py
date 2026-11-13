# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Learning Values from Experience".

Plain Python 3, standard library only, written from the chapter's own description. The decision process is the one of
the previous chapter (`mdps`, imported, never copied): its exact values are the ground truth that every learned estimate
is measured against. The chapter takes the tables away and lets the learner see sampled days only: a ten-day ledger turned
into estimates, the exact mean and spread of the Monte Carlo and the temporal-difference targets, learning curves of both
on the same sampled episodes, the conditions on a step-size schedule, first-visit against every-visit Monte Carlo, greedy-only learning that locks onto the worst
policy, the two ways out of it and the starting values of the optimistic one, SARSA and Q-learning against their exact fixed points, the maximisation bias of
Q-learning on a small example and the double estimator that removes it, linear temporal-difference learning against its
exact fixed point, a two-state example in which the weights grow without bound, and the work of an update. Each public
function without a leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: value_learning.<name> -->` markers. Underscore functions and the constants are shared with the chapter's
figures, so a picture and the table beside it cannot disagree.

Every number learned from samples goes through `stable`: the run is repeated with the first stored value changed by each
of the nudges of `stable.py` and a digit is printed only if the run and its four copies share it. The data of an
evaluation do not depend on the estimate, so the copies of an evaluation share one stream of episodes; a learner that
chooses its own actions draws its own stream in every copy.

Swedish Embedded AB implements solutions for agents that learn what a decision is worth from the days they have lived. If
your team needs expertise in temporal-difference learning and reinforcement learning from sampled experience, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import math

import bookrng
import mdps
import stable

# ------------------------------------------------------------------ the shared setting

PI = (1, 0, 0, 0)                      # the policy that is evaluated: order one unit when the shelf is empty
GAMMA = mdps.GAMMA
N = mdps.N_STATES
EXACT = mdps._evaluate_solve(PI, GAMMA)[0]            # V of PI from the linear system of the previous chapter
V_STAR = mdps._value_iteration(GAMMA)[0]              # the optimal values of the previous chapter
Q_STAR = [[mdps._q(s, a, V_STAR, GAMMA) for a in range(mdps._n_actions(s))] for s in range(N)]
TIE = mdps.TIE
VARIANTS = (0.0, *stable.NUDGES)       # the run itself, then four copies with the first stored value nudged

EVAL_DAYS, EVAL_EPISODES, EVAL_RUNS = 100, 200, 10
EVAL_CHECKS = (1, 2, 5, 10, 20, 50, 100, 200)
SCHEDULES = (("1/n", lambda n: n ** -1.0), ("n^-0.7", lambda n: n ** -0.7),
             ("0.5", lambda n: 0.5), ("0.1", lambda n: 0.1))
BEST = "n^-0.7"
REPEATS, REPEAT_EPISODES, REPEAT_CHECKS = 200, 100, (2, 5, 10, 20, 50, 100)

CONTROL_DAYS, CONTROL_RUNS, CONTROL_START = 20000, 10, 2
EPS_SCHEDULE = (1.0, 200.0, 0.05)      # eps_t = max(floor, eps0 / (1 + t / tau)) written (eps0, tau, floor)
OPTIMISTIC = 15.0                      # above the largest optimal lookahead, 11.668
PAIR_RUNS, PAIR_DAYS, PAIR_EPS = 10, 50000, 0.2

BIAS_ACTIONS, BIAS_MEAN, BIAS_SD, BIAS_ALPHA, BIAS_EPS = 8, -0.1, 1.0, 0.1, 0.1
BIAS_RUNS, BIAS_EPISODES = 1000, 300
RIGHT, LEFT = 0, 1                     # the action numbers at A: a tie goes to right, the better action

FA_POLICY = (3, 0, 0, 0)               # the one policy of the previous chapter that visits every shelf level
FA_RUNS, FA_DAYS = 3, 60000


# ------------------------------------------------------------------ small helpers

RULE_TEXT = ("Every number learned from samples is printed with the digits that the run and four copies of it share: the copies "
             "start with the first stored value (V(0), Q(0, 0), the first weight or Q(A, left), as the example has it) changed by "
             "+1e-12, -1e-12, +3e-12 and -3e-12 (a stability check applied to every trained number), and a digit is printed only if all five agree on it "
             "('about x' keeps the leading digits that survive, 'unstable' means that none does). The copies of an evaluation learn "
             "from the same episodes, because the data do not depend on the estimates; the copies of a learner that chooses its own "
             "actions use the same generator, so their draws are the same and they differ only if a nudge flips a comparison. "
             "A run is a deterministic computation, so another implementation reproduces the printed digits.")


def _table(headers, rows, align=None):
    return mdps._table(headers, rows, align)


def _seed_line(names):
    return ", ".join(f"{n.rsplit('.', 1)[1]} {bookrng.seed_of(n):#018x}" for n in names)


def _cell(vals, nd):
    """The digits of vals[0] that all of vals (the run and its copies) share, written with nd decimals."""
    text = stable.text(vals[0], vals[1:], lambda x: f"{x:.{nd}f}")
    return text.replace("-0.000", "0.000") if float(text.split()[-1]) == 0 else text


def _same(items):
    """The text of a discrete outcome when the run and its copies agree on it, 'unstable' otherwise."""
    return items[0] if all(i == items[0] for i in items) else "unstable"


def _rms(V, exact=EXACT):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(V, exact)) / len(exact))


def _greedy(q):
    """The smallest action whose value is within TIE of the largest (the tie rule of the previous chapter)."""
    best = max(q)
    return next(a for a, x in enumerate(q) if x >= best - TIE)


def _g4(x):
    """Four significant digits, written without an exponent."""
    return f"{float(f'{x:.4g}'):,.{max(0, 3 - int(math.floor(math.log10(abs(x)))))}f}" if x else "0"


def _sci(x):
    """Four significant digits in plain decimals for moderate sizes and in scientific notation beyond."""
    return f"{x:.4g}" if 0.01 <= abs(x) < 1000 else f"{x:.3e}"


def _pi_text(pi):
    return mdps._pi_text(pi)


# ------------------------------------------------------------------ evaluation from sampled days


def _episode(rng, policy, days):
    """One episode: a start state from integer(4), then per day one uniform() for the demand."""
    s = rng.integer(N)
    out = []
    for _ in range(days):
        r, nxt, _sold = mdps._step(s, policy[s], mdps._demand_from(rng.uniform()))
        out.append((s, r, nxt))
        s = nxt
    return out


def _first_visit_returns(ep):
    """The return of the first visit to each state of an episode, days after the last ignored."""
    G, returns = 0.0, [0.0] * len(ep)
    for t in range(len(ep) - 1, -1, -1):
        G = ep[t][1] + GAMMA * G
        returns[t] = G
    seen = {}
    for t, (s, _r, _nxt) in enumerate(ep):
        if s not in seen:
            seen[s] = (t, returns[t])
    return seen


def _eval_methods(episodes, nudge, schedules=SCHEDULES, checks=None):
    """Learn from the same episodes with Monte Carlo (first visit, step 1/n) and TD(0) under every schedule.

    Returns {method: [RMS error after each episode]} (or only at `checks`)."""
    est = {"MC": [nudge, 0.0, 0.0, 0.0]}
    cnt = {"MC": [0] * N}
    for name, _ in schedules:
        est[name] = [nudge, 0.0, 0.0, 0.0]
        cnt[name] = [0] * N
    curves = {m: [] for m in est}
    for e, ep in enumerate(episodes, start=1):
        for s, (_t, G) in _first_visit_returns(ep).items():
            cnt["MC"][s] += 1
            est["MC"][s] += (G - est["MC"][s]) / cnt["MC"][s]
        for name, f in schedules:
            V, n = est[name], cnt[name]
            for (s, r, nxt) in ep:
                n[s] += 1
                V[s] += f(n[s]) * (r + GAMMA * V[nxt] - V[s])
        if checks is None or e in checks:
            for m in est:
                curves[m].append(_rms(est[m]))
    return curves


def _run_names(prefix, count):
    return [f"value_learning.{prefix}.run{i}" for i in range(count)]


def _mean_curves(variants=VARIANTS):
    """Per variant, the mean over the runs of the RMS error after every episode, per method."""
    names = _run_names("evaluate", EVAL_RUNS)
    total = [None] * len(variants)
    first_day = 0
    for name in names:
        rng = bookrng.generator(name)
        episodes = [_episode(rng, PI, EVAL_DAYS) for _ in range(EVAL_EPISODES)]
        for ep in episodes:
            first_day = max(first_day, max(t for (t, _G) in _first_visit_returns(ep).values()))
        for v, nudge in enumerate(variants):
            c = _eval_methods(episodes, nudge)
            if total[v] is None:
                total[v] = {m: [0.0] * EVAL_EPISODES for m in c}
            for m in c:
                total[v][m] = [x + y for x, y in zip(total[v][m], c[m])]
    return [{m: [x / EVAL_RUNS for x in cs] for m, cs in t.items()} for t in total], first_day


# ------------------------------------------------------------------ the blocks: the ledger and the targets


def ledger_updates():
    pi, alpha = PI, 0.5
    rng = bookrng.generator(mdps.TRAJ_NAME)
    s, V = mdps.TRAJ_START, [0.0] * N
    rows, days = [], []
    for t in range(mdps.TRAJ_DAYS):
        a = pi[s]
        d = mdps._demand_from(rng.uniform())
        r, nxt, _sold = mdps._step(s, a, d)
        days.append((t, s, a, d, r, nxt))
        s = nxt
    for (t, s, a, d, r, nxt) in days:
        before = V[s]
        target = r + GAMMA * V[nxt]
        delta = target - before
        V[s] = before + alpha * delta
        rows.append([t, s, a, d, mdps._fmt(r, 1), nxt, mdps._fmt(before, 3), mdps._fmt(target, 3), mdps._fmt(delta, 3),
                     mdps._fmt(V[s], 3)])
    G = [0.0] * (len(days) + 1)
    for t in range(len(days) - 1, -1, -1):
        G[t] = days[t][4] + GAMMA * G[t + 1]
    first = {}
    for (t, s, *_rest) in days:
        first.setdefault(s, t)
    mc = [mdps._fmt(G[first[s]], 3) if s in first else "none" for s in range(N)]
    out = [
        f"The ten-day ledger of the previous chapter: policy (1, 0, 0, 0), start s_0 = {mdps.TRAJ_START}, generator the book's "
        f"SplitMix64 with {mdps._seed_text(mdps.TRAJ_NAME)}, one uniform() per day in order and the demand rule of that chapter. "
        f"The estimates start at V(s) = 0 for every s. Each day, in order, the target r + gamma V(s') is formed from the estimates as "
        f"they stand (gamma = {GAMMA}), the difference delta = target - V(s) is computed, and V(s) is replaced by V(s) + {alpha} * delta "
        f"before the next day is looked at; when s' = s the target uses the estimate of the day before.",
        "",
        _table(["t", "s", "a", "D", "r", "s'", "V(s)", "target", "delta", "new V(s)"], rows),
        "",
        f"After the ten updates V = ({', '.join(mdps._fmt(v, 3) for v in V)}). The Monte Carlo estimate from the same ledger uses "
        f"the first visit of each state, the returns of the previous chapter's table at gamma = {GAMMA}: "
        f"V(0), V(1), V(2), V(3) = {', '.join(mc)} (the state 3 is never visited). The exact values are "
        f"({', '.join(mdps._fmt(v, 3) for v in EXACT)}).",
    ]
    return "\n".join(out)


def _return_moments(policy=PI, gamma=GAMMA):
    """Exact mean and variance of the return from each state, by the Bellman equation for E[G] and for E[G^2]."""
    V, _ = mdps._evaluate_solve(policy, gamma)
    c = []
    for s in range(N):
        acc = 0.0
        for d, p in enumerate(mdps.DEMAND):
            r, nxt, _sold = mdps._step(s, policy[s], d)
            acc += p * (r * r + 2.0 * gamma * r * V[nxt])
        c.append(acc)
    A = [[(1.0 if i == j else 0.0) - gamma * gamma * mdps.PROB[i][policy[i]][j] for j in range(N)] for i in range(N)]
    M, _ = mdps._solve(A, c)
    return V, [math.sqrt(M[s] - V[s] ** 2) for s in range(N)]


def _td_target_moments(estimate, policy=PI, gamma=GAMMA):
    """Mean and standard deviation of the one-day target r + gamma estimate(s') for each state."""
    out = []
    for s in range(N):
        ys = []
        for d, p in enumerate(mdps.DEMAND):
            r, nxt, _sold = mdps._step(s, policy[s], d)
            ys.append((p, r + gamma * estimate[nxt]))
        mean = sum(p * y for p, y in ys)
        out.append((mean, math.sqrt(sum(p * (y - mean) ** 2 for p, y in ys))))
    return out


def target_spread():
    V, sd_mc = _return_moments()
    exact_t = _td_target_moments(V)
    high_t = _td_target_moments([v + 1.0 for v in V])
    zero_t = _td_target_moments([0.0] * N)
    rows = []
    for s in range(N):
        rows.append([s, mdps._fmt(V[s], 3), mdps._fmt(sd_mc[s], 3), mdps._fmt(exact_t[s][1], 3), mdps._fmt(zero_t[s][1], 3),
                     mdps._fmt(exact_t[s][0] - V[s], 3), mdps._fmt(high_t[s][0] - V[s], 3), mdps._fmt(zero_t[s][0] - V[s], 3)])
    out = [
        f"Policy (1, 0, 0, 0), gamma = {GAMMA}; every number is computed exactly from the tables of the previous chapter, none is "
        f"sampled. The Monte Carlo target of a state is the whole return G; its mean is V(s) and its standard deviation sd(G) comes "
        f"from E[G^2] = E[r^2] + 2 gamma E[r V(s')] + gamma^2 E[G'^2], a linear system with the matrix I - gamma^2 P_pi, minus V(s)^2. "
        f"The one-day target is y = r + gamma W(s') for an estimate W of the next state, with the three outcomes d = 0, 1, 2 of "
        f"the day as its only randomness; sd(y) is taken for W = V and for W = 0 (all values zero). The three last columns are "
        f"E[y] - V(s) for W = V, for W = V + 1 (every estimate one too high) and for W = 0; a number that rounds to zero is printed 0.000 whatever its sign.",
        "",
        _table(["s", "V(s)", "sd(G)", "sd(y), W = V", "sd(y), W = 0", "bias, W = V", "bias, W = V + 1", "bias, W = 0"], rows),
    ]
    return "\n".join(out)


def _repeat_stats(variants=VARIANTS):
    """For each variant, MC and TD(0) (n^-0.7) estimates of V(0) after each of the checks, over REPEATS independent repeats."""
    f = dict(SCHEDULES)[BEST]
    errs = [{m: {e: [] for e in REPEAT_CHECKS} for m in ("MC", "TD")} for _ in variants]
    for name in _run_names("repeat", REPEATS):
        rng = bookrng.generator(name)
        episodes = [_episode(rng, PI, EVAL_DAYS) for _ in range(REPEAT_EPISODES)]
        for v, nudge in enumerate(variants):
            mc, mcn = [nudge, 0.0, 0.0, 0.0], [0] * N
            td, tdn = [nudge, 0.0, 0.0, 0.0], [0] * N
            for e, ep in enumerate(episodes, start=1):
                for s, (_t, G) in _first_visit_returns(ep).items():
                    mcn[s] += 1
                    mc[s] += (G - mc[s]) / mcn[s]
                for (s, r, nxt) in ep:
                    tdn[s] += 1
                    td[s] += f(tdn[s]) * (r + GAMMA * td[nxt] - td[s])
                if e in REPEAT_CHECKS:
                    errs[v]["MC"][e].append(mc[0] - EXACT[0])
                    errs[v]["TD"][e].append(td[0] - EXACT[0])
    return errs


def bias_and_spread():
    errs = _repeat_stats()
    rows = []
    for e in REPEAT_CHECKS:
        cells = []
        for m in ("MC", "TD"):
            bias = [sum(x[m][e]) / REPEATS for x in errs]
            sd = [math.sqrt(sum((y - sum(x[m][e]) / REPEATS) ** 2 for y in x[m][e]) / (REPEATS - 1)) for x in errs]
            rmse = [math.sqrt(sum(y * y for y in x[m][e]) / REPEATS) for x in errs]
            cells += [_cell(bias, 3), _cell(sd, 3), _cell(rmse, 3)]
        rows.append([e] + cells)
    out = [
        f"{REPEATS} independent repeats, each with its own generator (run i has the generator named value_learning.repeat.run<i> for "
        f"i = 0 .. {REPEATS - 1}, so its seed is seed_of of that name: the first three are {_seed_line(_run_names('repeat', 3))}). A repeat "
        f"draws {REPEAT_EPISODES} episodes of {EVAL_DAYS} days in order, each as in the table of learning curves (a start state "
        f"from integer({N}), then one uniform() per day), and both methods learn from the same episodes: first-visit Monte Carlo with "
        f"the step 1/n of the state, and TD(0) with the step n^-0.7 of the state, n counting the updates of that state from 1, "
        f"both from the estimates 0. The table is the error of the estimate of V(0) = {mdps._fmt(EXACT[0], 3)} after the stated number of "
        f"episodes: bias is the mean of (estimate - V(0)) over the repeats, sd the standard deviation over the repeats with the divisor "
        f"{REPEATS - 1}, and rmse the square root of the mean of the squares of the errors (divisor {REPEATS}).",
        "",
        _table(["episodes", "MC bias", "MC sd", "MC rmse", "TD bias", "TD sd", "TD rmse"], rows),
    ]
    return "\n".join(out)


def learning_curves():
    variants, first_day = _mean_curves()
    methods = ["MC"] + [n for n, _ in SCHEDULES]
    rows = []
    for e in EVAL_CHECKS:
        rows.append([e, e * EVAL_DAYS] + [_cell([c[m][e - 1] for c in variants], 3) for m in methods])
    names = _run_names("evaluate", EVAL_RUNS)
    seeds = _table(["run", "generator name", "seed"],
                   [[i, n, f"{bookrng.seed_of(n):#018x}"] for i, n in enumerate(names)], align="rlr")
    out = [
        f"Policy (1, 0, 0, 0), gamma = {GAMMA}. A run is {EVAL_EPISODES} episodes of {EVAL_DAYS} days with its own generator, the "
        f"SplitMix64 of the book seeded with the seed of its name (below); an episode draws one integer({N}) for its start state and "
        f"then, for each of its {EVAL_DAYS} days in order, one uniform() for the demand (the action is the policy's, so nothing else "
        f"is drawn). Every method learns from the same episodes of a run. All estimates start at 0 and no state is terminal: "
        f"the update on the last day of an episode uses the estimate of the state reached, and the days after an episode are "
        f"ignored by Monte Carlo. Monte Carlo is first visit: after an episode the return G of the first visit to each state "
        f"(G_t = r_t + gamma G_(t+1) from G_{EVAL_DAYS} = 0) moves its estimate by (G - V)/n, n counting the updates of that state "
        f"from 1. TD(0) updates the state of every day, in the order of the days, immediately: V(s) <- V(s) + alpha_n "
        f"(r + gamma V(s') - V(s)) with the estimate of s' as it stands and n the number of updates of s so far including this one, "
        f"so the first update of a state uses alpha_1; the four schedules are alpha_n = 1/n, n^-0.7, and the constants 0.5 and 0.1. "
        f"The error of a method after an episode is the root mean square over the four states of V(s) - V*(s) against the solved "
        f"values of the previous chapter (divisor 4), and each entry is the mean over the {EVAL_RUNS} runs (summed in run order, "
        f"divisor {EVAL_RUNS}). The largest first-visit day seen in any episode of any run is {first_day}, so the days ignored by Monte "
        f"Carlo weigh at most {GAMMA ** (EVAL_DAYS - first_day) * max(EXACT):.1e} in a first-visit return.",
        "",
        _table(["episodes", "days", "MC", "TD 1/n", "TD n^-0.7", "TD 0.5", "TD 0.1"], rows),
        "",
        RULE_TEXT,
        "",
        seeds,
    ]
    return "\n".join(out)


EV_RUNS, EV_EPISODES, EV_DAYS_USED = 200, 200, 50
EV_CHECKS = (5, 10, 20, 50, 100, 200)


def _every_visit_runs():
    """Per run, after each episode: first-visit and every-visit estimates, and the visits of the episode to each state.

    Every-visit uses the return of each of the first EV_DAYS_USED days of an episode (the returns are those of the whole
    episode, so the days after it are the only ones ignored); both estimates are running means (step 1/n of the state)."""
    out = []
    for name in _run_names("everyvisit", EV_RUNS):
        rng = bookrng.generator(name)
        fv, fn = [0.0] * N, [0] * N
        ev, en = [0.0] * N, [0] * N
        per_episode = []
        for _ in range(EV_EPISODES):
            ep = _episode(rng, PI, EVAL_DAYS)
            G, returns = 0.0, [0.0] * len(ep)
            for t in range(len(ep) - 1, -1, -1):
                G = ep[t][1] + GAMMA * G
                returns[t] = G
            for s, (_t, g) in _first_visit_returns(ep).items():
                fn[s] += 1
                fv[s] += (g - fv[s]) / fn[s]
            visits = [0] * N
            for t in range(EV_DAYS_USED):
                s = ep[t][0]
                en[s] += 1
                visits[s] += 1
                ev[s] += (returns[t] - ev[s]) / en[s]
            per_episode.append((list(fv), list(ev), list(fn), list(en), visits))
        out.append(per_episode)
    return out


def _mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def every_visit_monte_carlo():
    runs = _every_visit_runs()
    _V, sd_g = _return_moments()
    rows = []
    for e in EV_CHECKS:
        f = [_rms(r[e - 1][0]) for r in runs]
        v = [_rms(r[e - 1][1]) for r in runs]
        m, sd = _mean_sd([y - x for x, y in zip(f, v)])
        rows.append([e, f"{sum(f) / EV_RUNS:.3f}", f"{sum(v) / EV_RUNS:.3f}", f"{m:+.3f}", f"{sd / math.sqrt(EV_RUNS):.3f}"])
    last = EV_EPISODES - 1
    rows_state = []
    for s in range(N):
        fe = [r[last][0][s] - EXACT[s] for r in runs]
        ee = [r[last][1][s] - EXACT[s] for r in runs]
        (fb, fsd), (eb, esd) = _mean_sd(fe), _mean_sd(ee)
        kish = []
        for r in runs:
            m = [ep[4][s] for ep in r]
            kish.append(sum(m) ** 2 / sum(x * x for x in m))
        rows_state.append([s, f"{sum(r[last][2][s] for r in runs) / EV_RUNS:.1f}", f"{sum(r[last][3][s] for r in runs) / EV_RUNS:.1f}",
                           f"{fb:+.3f}", f"{fsd:.3f}", f"{eb:+.3f}", f"{esd:.3f}",
                           f"{(sd_g[s] / fsd) ** 2:.0f}", f"{(sd_g[s] / esd) ** 2:.0f}", f"{sum(kish) / EV_RUNS:.0f}"])
    out = [
        f"First-visit and every-visit Monte Carlo on the same episodes: policy (1, 0, 0, 0), gamma = {GAMMA}, {EV_RUNS} independent "
        f"repeats (repeat i has the generator named value_learning.everyvisit.run<i>, i = 0 .. {EV_RUNS - 1}; the first seeds are "
        f"{_seed_line(_run_names('everyvisit', 2))}), each {EV_EPISODES} episodes of {EVAL_DAYS} days drawn as in the table of learning "
        f"curves. Both estimates start at 0 and are running means with the step 1/n, n counting the returns averaged for that "
        f"state from 1 (the first step is 1, so the start value never matters). First visit averages the return of the first "
        f"visit of an episode, every visit the return of each of the first {EV_DAYS_USED} days of an episode (days 0 .. "
        f"{EV_DAYS_USED - 1}, each return being the discounted sum to the end of the {EVAL_DAYS}-day episode). The first table is the "
        f"root mean square error over the four states after the stated number of episodes, as in the table of learning curves "
        f"(divisor 4), averaged over the {EV_RUNS} repeats; 'EV - FV' is the mean over the repeats of the difference of the two "
        f"errors of the same repeat and 'se' its standard error (the standard deviation of the differences with the divisor "
        f"{EV_RUNS - 1}, over the square root of {EV_RUNS}).",
        "",
        _table(["episodes", "first visit", "every visit", "EV - FV", "se"], rows),
        "",
        f"The second table is the state-by-state view after {EV_EPISODES} episodes. 'visits' is the mean over the repeats of the number "
        f"of returns averaged; bias and sd are the mean and the standard deviation (divisor {EV_RUNS - 1}) over the repeats of "
        f"V(s) - V*(s); 'equivalent n' is (sd(G) / sd)^2 with sd(G) the exact standard deviation of one return from the table of "
        f"target spreads, the number of independent returns that would give that spread; 'weight n' is the mean over the repeats of "
        f"(sum of m_i)^2 / (sum of m_i^2) over the episodes, m_i the number of every-visit returns that episode i contributed to "
        f"the state: the number of equally weighted episodes that would carry the same weights.",
        "",
        _table(["s", "visits FV", "visits EV", "bias FV", "sd FV", "bias EV", "sd EV", "equivalent n FV", "equivalent n EV",
                "weight n"], rows_state),
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ step sizes


def step_size_sums():
    rows = []
    for label, f in (("n^-1", lambda n: n ** -1.0), ("n^-0.7", lambda n: n ** -0.7), ("n^-0.5", lambda n: n ** -0.5),
                     ("0.1", lambda n: 0.1)):
        s1 = s2 = 0.0
        marks = {10: None, 1000: None, 1000000: None}
        for n in range(1, 1000001):
            a = f(n)
            s1 += a
            s2 += a * a
            if n in marks:
                marks[n] = (s1, s2)
        for n, (a, b) in marks.items():
            rows.append([label, f"{n:,}", _g4(a), _g4(b)])
    out = [
        "The partial sums of a step-size schedule: alpha_n is the step of the n-th update, n from 1, and the two sums are accumulated "
        "in increasing n from 0.0 in double precision (four significant digits are printed, which they reproduce).",
        "",
        _table(["alpha_n", "n", "sum of alpha_1 .. alpha_n", "sum of alpha_1^2 .. alpha_n^2"], rows, align="lrrr"),
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ control: exploration, SARSA, Q-learning


def _control_run(name, days, sarsa, eps, q0, nudge=0.0, power=0.7, start=CONTROL_START):
    """One stream of `days` days of SARSA or Q-learning; eps is (eps0, tau, floor). Returns the table and the counts."""
    rng = bookrng.generator(name)
    nact = [mdps._n_actions(s) for s in range(N)]
    Q = [[q0] * nact[s] for s in range(N)]
    Q[0][0] += nudge
    cnt = [[0] * nact[s] for s in range(N)]
    eps0, tau, floor = eps

    def choose(s, t):
        if rng.uniform() < max(floor, eps0 / (1.0 + t / tau)):
            return rng.integer(nact[s])
        return _greedy(Q[s])

    s = start
    a = choose(s, 0)
    for t in range(days):
        r, nxt, _sold = mdps._step(s, a, mdps._demand_from(rng.uniform()))
        a2 = choose(nxt, t + 1)
        y = r + GAMMA * (Q[nxt][a2] if sarsa else max(Q[nxt]))
        cnt[s][a] += 1
        Q[s][a] += cnt[s][a] ** -power * (y - Q[s][a])
        s, a = nxt, a2
    return Q, cnt


def exploration_lock_in():
    variants = (("greedy, Q starts at 0", (0.0, 1.0, 0.0), 0.0),
                ("epsilon-greedy, Q starts at 0", EPS_SCHEDULE, 0.0),
                (f"greedy, Q starts at {OPTIMISTIC:g}", (0.0, 1.0, 0.0), OPTIMISTIC))
    rows = []
    for k, (label, eps, q0) in enumerate(variants):
        res = [[_control_run(f"value_learning.exploration.v{k}.run{i}", CONTROL_DAYS, False, eps, q0, nudge)
                for i in range(CONTROL_RUNS)] for nudge in VARIANTS]
        policies, v0, one, q00 = [], [], [], []
        for runs in res:
            pols = [tuple(_greedy(Q[s]) for s in range(N)) for Q, _c in runs]
            policies.append("; ".join(f"{_pi_text(p)} x {pols.count(p)}" for p in sorted(set(pols))))
            v0.append(sum(mdps._evaluate_solve(p, GAMMA)[0][0] for p in pols) / CONTROL_RUNS)
            one.append(sum(c[0][1] for _Q, c in runs) / CONTROL_RUNS)
            q00.append([sum(Q[0][a] for Q, _c in runs) / CONTROL_RUNS for a in range(4)])
        rows.append([label, _same(policies), _cell(v0, 3), _cell(one, 1)] + [_cell([q[a] for q in q00], 3) for a in range(4)])
    out = [
        f"Q-learning on the inventory problem of the previous chapter, gamma = {GAMMA}, from the shelf s_0 = {CONTROL_START}, "
        f"{CONTROL_DAYS} days per run and {CONTROL_RUNS} runs; run i of variant k has the generator named "
        f"value_learning.exploration.v<k>.run<i> (k = 0, 1, 2 in the order of the rows, i = 0 .. {CONTROL_RUNS - 1}; the seeds of "
        f"the first run of each are {', '.join(f'{bookrng.seed_of(f'value_learning.exploration.v{k}.run0'):#018x}' for k in range(3))}). "
        f"Every day draws in this order: the demand uniform() of the day just played, then, for the action of the next day, one uniform() "
        f"u and, only when u < eps_t, one integer(m) over the m legal actions; eps_t = max({EPS_SCHEDULE[2]}, {EPS_SCHEDULE[0]:g} / "
        f"(1 + t / {EPS_SCHEDULE[1]:g})) for the action of day t in the second row and 0 in the other two (the uniform u is drawn "
        f"all the same). Otherwise the action is the greedy one, the smallest within {TIE:g} of the largest Q(s, .). The update of a "
        f"day is Q(s, a) <- Q(s, a) + n^-0.7 (r + gamma max Q(s', .) - Q(s, a)) with n the number of updates of that pair so far "
        f"including this one, applied after the next action has been chosen and before the next demand is drawn. The final greedy "
        f"policy of a run is the greedy action of Q in every state; 'V(0) of the final policy' is its exact value on an empty shelf "
        f"from the linear system (mean over the runs), 'days at s = 0 with a = 1' the mean number of updates of that pair, and the "
        f"last four columns are the mean over the runs of Q(0, 0 .. 3) after the last day.",
        "",
        _table(["variant", "final greedy policies of the runs", "V(0) of the final policy", "days at s = 0, a = 1",
                "Q(0,0)", "Q(0,1)", "Q(0,2)", "Q(0,3)"], rows, align="llrrrrrr"),
        "",
        f"The optimal values of the previous chapter are Q*(0, 0 .. 3) = {', '.join(mdps._fmt(q, 3) for q in Q_STAR[0])} and "
        f"V*(0) = {mdps._fmt(V_STAR[0], 3)}.",
    ]
    return "\n".join(out)


OPTIMISM_STARTS = (5.0, 11.7, OPTIMISTIC, 100.0)


def optimistic_starts():
    rows = []
    for q0 in OPTIMISM_STARTS:
        res = [[_control_run("value_learning.exploration.v2.run%d" % i, CONTROL_DAYS, False, (0.0, 1.0, 0.0), q0, nudge)
                for i in range(CONTROL_RUNS)] for nudge in VARIANTS]
        policies, one, top = [], [], []
        for runs in res:
            pols = [tuple(_greedy(Q[s]) for s in range(N)) for Q, _c in runs]
            policies.append("; ".join(f"{_pi_text(p)} x {pols.count(p)}" for p in sorted(set(pols))))
            one.append(sum(c[0][1] for _Q, c in runs) / CONTROL_RUNS)
            top.append(sum(max(Q[0]) for Q, _c in runs) / CONTROL_RUNS)
        rows.append([f"{q0:g}", _same(policies), _cell(one, 1), _cell(top, 2)])
    out = [
        f"The third variant of the lock-in table (greedy, no random action, Q starting at the stated value for every pair) with "
        f"four starting values; every start uses the same {CONTROL_RUNS} generators, run i the one named "
        f"value_learning.exploration.v2.run<i> (i = 0 .. {CONTROL_RUNS - 1}), so the rows differ only in the starting value, and "
        f"everything else (days {CONTROL_DAYS}, s_0 = {CONTROL_START}, the draw order, the tie rule, the step n^-0.7) is as there. "
        f"The row of {OPTIMISTIC:g} is the third row of that table. The last column is the mean over the runs of the largest "
        f"Q(0, .) after the last day, to be read against V*(0) = {mdps._fmt(V_STAR[0], 3)}.",
        "",
        _table(["start value", "final greedy policies of the runs", "days at s = 0, a = 1", "largest Q(0, .)"], rows, align="rlrr"),
    ]
    return "\n".join(out)


def _q_eps_star(eps):
    """The fixed point of the best epsilon-greedy policy: Q(s, a) = R + gamma sum_j P (1 - eps) max Q(j, .) + eps mean Q(j, .)."""
    Q = [[0.0] * mdps._n_actions(s) for s in range(N)]
    while True:
        new = []
        for s in range(N):
            row = []
            for a in range(mdps._n_actions(s)):
                acc = 0.0
                for j in range(N):
                    acc += mdps.PROB[s][a][j] * ((1.0 - eps) * max(Q[j]) + eps * sum(Q[j]) / len(Q[j]))
                row.append(mdps.REW[s][a] + GAMMA * acc)
            new.append(row)
        delta = max(abs(x - y) for rn, ro in zip(new, Q) for x, y in zip(rn, ro))
        Q = new
        if delta < 1e-12:
            return Q


def sarsa_and_q_learning():
    eps = (PAIR_EPS, 1.0, PAIR_EPS)
    qe = _q_eps_star(PAIR_EPS)
    means = {}
    for mode in (True, False):
        per = []
        for nudge in VARIANTS:
            acc = [[0.0] * mdps._n_actions(s) for s in range(N)]
            for i in range(PAIR_RUNS):
                Q, _c = _control_run(f"value_learning.pairs.{'sarsa' if mode else 'q'}.run{i}", PAIR_DAYS, mode, eps, 0.0, nudge)
                for s in range(N):
                    for a in range(len(Q[s])):
                        acc[s][a] += Q[s][a] / PAIR_RUNS
            per.append(acc)
        means[mode] = per
    rows = []
    for s in range(N):
        for a in range(mdps._n_actions(s)):
            rows.append([s, a, mdps._fmt(Q_STAR[s][a], 3), mdps._fmt(qe[s][a], 3),
                         _cell([m[s][a] for m in means[True]], 2), _cell([m[s][a] for m in means[False]], 2)])
    out = [
        f"A constant exploration rate eps = {PAIR_EPS} for every day (eps0 = floor = {PAIR_EPS}), Q starting at 0, {PAIR_DAYS} days per "
        f"run from s_0 = {CONTROL_START}, {PAIR_RUNS} runs of each algorithm (run i of SARSA has the generator named "
        f"value_learning.pairs.sarsa.run<i>, of Q-learning value_learning.pairs.q.run<i>, i = 0 .. {PAIR_RUNS - 1}; the first seeds are "
        f"{bookrng.seed_of('value_learning.pairs.sarsa.run0'):#018x} and {bookrng.seed_of('value_learning.pairs.q.run0'):#018x}), "
        f"draws and update order as in the table above, the step n^-0.7. Q* is the optimal lookahead of the previous chapter; "
        f"Q_eps is the fixed point of Q(s, a) = R(s, a) + gamma sum_j P(j | s, a) [(1 - eps) max_b Q(j, b) + eps mean_b Q(j, b)], "
        f"the value of the best policy that must explore with probability eps, iterated until the change is below 1e-12. "
        f"The last two columns are the mean over the {PAIR_RUNS} runs of Q(s, a) after the last day (divisor {PAIR_RUNS}).",
        "",
        _table(["s", "a", "Q*", "Q_eps", "SARSA", "Q-learning"], rows),
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ maximisation bias


def _expected_max_normal(k):
    """E[max of k independent N(0, 1)] = integral of k x phi(x) Phi(x)^(k-1) dx, by the trapezoid rule on [-10, 10], step 0.001."""
    h, total = 0.001, 0.0
    for i in range(20001):
        x = -10.0 + i * h
        phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        w = 0.5 if i in (0, 20000) else 1.0
        total += w * k * x * phi * Phi ** (k - 1)
    return total * h


def _bias_run(name, double, nudge):
    """One run of BIAS_EPISODES episodes; returns the 0/1 'left at A' list and the Q(A, left) after each episode."""
    rng = bookrng.generator(name)
    T = [[[0.0, 0.0], [0.0] * BIAS_ACTIONS], [[0.0, 0.0], [0.0] * BIAS_ACTIONS]]
    T[0][0][LEFT] += nudge

    def act(s):
        n = len(T[0][s])
        q = [T[0][s][i] + T[1][s][i] for i in range(n)] if double else T[0][s]
        if rng.uniform() < BIAS_EPS:
            return rng.integer(n)
        return _greedy(q)

    def update(s, a, r, nxt):
        k = (0 if rng.uniform() < 0.5 else 1) if double else 0
        y = r
        if nxt is not None:
            if double:
                y += GAMMA * T[1 - k][nxt][_greedy(T[k][nxt])]
            else:
                y += GAMMA * max(T[0][nxt])
        T[k][s][a] += BIAS_ALPHA * (y - T[k][s][a])

    left, qa = [], []
    for _ in range(BIAS_EPISODES):
        a = act(0)
        left.append(1 if a == LEFT else 0)
        if a == LEFT:
            update(0, LEFT, 0.0, 1)
            b = act(1)
            update(1, b, BIAS_MEAN + BIAS_SD * rng.normal(), None)
        else:
            update(0, RIGHT, 0.0, None)
        qa.append((T[0][0][LEFT] + T[1][0][LEFT]) / 2.0 if double else T[0][0][LEFT])
    return left, qa


def maximisation_bias():
    c = _expected_max_normal(BIAS_ACTIONS)
    rows_est = []
    for n in (1, 4, 25, 100):
        rows_est.append([n, mdps._fmt(BIAS_MEAN + c / math.sqrt(n), 3), mdps._fmt(BIAS_MEAN, 3)])
    names = [f"value_learning.maxbias.run{i}" for i in range(BIAS_RUNS)]
    blocks = ((1, 10), (11, 50), (51, 100), (101, 300))
    checks = (10, 50, 100, 300)
    cols = {}
    for double in (False, True):
        per = []
        for nudge in VARIANTS:
            runs = [_bias_run(n, double, nudge) for n in names]
            per.append(runs)
        cols[double] = per
    rows_left = []
    for lo, hi in blocks:
        row = [f"{lo} .. {hi}"]
        for double in (False, True):
            vals = [sum(sum(r[0][lo - 1:hi]) / (hi - lo + 1) for r in runs) / BIAS_RUNS for runs in cols[double]]
            row.append(_cell([100.0 * v for v in vals], 1))
        rows_left.append(row)
    rows_q = []
    for e in checks:
        row = [e]
        for double in (False, True):
            row.append(_cell([sum(r[1][e - 1] for r in runs) / BIAS_RUNS for runs in cols[double]], 3))
        rows_q.append(row)
    out = [
        f"The small problem. State A has two actions, right (action 0: reward 0, the episode ends) and left (action 1: reward 0, the next state is B). "
        f"State B has {BIAS_ACTIONS} actions, each ending the episode with a reward drawn from a Gaussian with mean {BIAS_MEAN} and "
        f"standard deviation {BIAS_SD}, the same for all of them. Terminal states have value 0 and gamma = {GAMMA}, so the true "
        f"Q(A, right) = 0 and Q(A, left) = {GAMMA} * {BIAS_MEAN} = {GAMMA * BIAS_MEAN:.2f}: right is better. "
        f"If the {BIAS_ACTIONS} values at B were each the mean of n independent rewards, they would be Gaussian with mean "
        f"{BIAS_MEAN} and standard deviation 1/sqrt(n), and the expected largest of {BIAS_ACTIONS} of them is {BIAS_MEAN} + c / sqrt(n) "
        f"with c = E[max of {BIAS_ACTIONS} standard Gaussians] = {c:.4f} (the integral of {BIAS_ACTIONS} x phi(x) Phi(x)^{BIAS_ACTIONS - 1} "
        f"by the trapezoid rule on [-10, 10] with step 0.001):",
        "",
        _table(["n", "expected largest estimate at B", "value of every action at B"], rows_est),
        "",
        f"Learning it: {BIAS_RUNS} runs of {BIAS_EPISODES} episodes, run i with the generator named value_learning.maxbias.run<i> "
        f"(i = 0 .. {BIAS_RUNS - 1}; run 0 has seed {bookrng.seed_of(names[0]):#018x}), the same generator name for both "
        f"algorithms. All values start at 0, the step is a constant {BIAS_ALPHA}, the exploration rate a constant {BIAS_EPS} "
        f"(one uniform() u per decision and, if u < {BIAS_EPS}, one integer over the actions), the greedy action the smallest "
        f"within {TIE:g} of the best. An episode chooses at A, updates the pair (A, chosen) when the step ends, and if it went left "
        f"chooses at B, draws one normal() and updates (B, chosen) with the reward alone. Q-learning's target at A is "
        f"gamma max_b Q(B, b). Double Q-learning keeps two tables, chooses by their sum, and for every update draws one uniform(), "
        f"after the transition and before the update, to pick the table k to change (k = 0 if it is below 0.5): its target "
        f"is gamma Q_other(B, argmax_b Q_k(B, b)). The tables of a run start at 0; the share of runs that went left in the episodes "
        f"of a block is a mean over the runs (divisor {BIAS_RUNS}) of the fraction of those episodes, in percent. A run that "
        f"always chose the best action and explored at {BIAS_EPS} would go left in {100 * BIAS_EPS / 2:g}% of the episodes.",
        "",
        _table(["episodes", "Q-learning goes left (%)", "double Q-learning goes left (%)"], rows_left),
        "",
        f"The estimate of Q(A, left) after the stated episode, mean over the runs (the true value is {GAMMA * BIAS_MEAN:.2f}; "
        f"for double Q-learning the mean of its two tables):",
        "",
        _table(["after episode", "Q-learning", "double Q-learning"], rows_q),
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ function approximation


def _features(name):
    if name == "one-hot":
        return [[1.0 if i == s else 0.0 for i in range(N)] for s in range(N)]
    if name == "linear":
        return [[1.0, float(s)] for s in range(N)]
    return [[1.0, float(s), float(s * s)] for s in range(N)]


FEATURE_SETS = ("one-hot", "linear", "quadratic")


def _stationary(policy):
    """The long-run share of days mu(s) of the chain of a policy: mu P = mu with the entries summing to 1."""
    P = mdps._policy_chain(policy)
    A = [[(1.0 if i == j else 0.0) - P[j][i] for j in range(N)] for i in range(N)]
    A[N - 1] = [1.0] * N
    d, _ = mdps._solve(A, [0.0] * (N - 1) + [1.0])
    return d


def _td_fixed_point(X, d, policy, gamma=GAMMA):
    """The weights w with sum_s mu(s) x(s) (r(s) + gamma x(s')^T w - x(s)^T w) = 0 in expectation, by one linear system."""
    k = len(X[0])
    P = mdps._policy_chain(policy)
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for s in range(N):
        rs = mdps.REW[s][policy[s]]
        for i in range(k):
            b[i] += d[s] * X[s][i] * rs
            for j in range(k):
                nxt = 0.0
                for t in range(N):
                    nxt += P[s][t] * X[t][j]
                A[i][j] += d[s] * X[s][i] * (X[s][j] - gamma * nxt)
    return mdps._solve(A, b)[0]


def _best_fit(X, d, V):
    """The weights minimising sum_s mu(s) (x(s)^T w - V(s))^2, by the normal equations."""
    k = len(X[0])
    A = [[sum(d[s] * X[s][i] * X[s][j] for s in range(N)) for j in range(k)] for i in range(k)]
    b = [sum(d[s] * X[s][i] * V[s] for s in range(N)) for i in range(k)]
    return mdps._solve(A, b)[0]


def _predict(X, w):
    return [sum(w[i] * X[s][i] for i in range(len(w))) for s in range(N)]


def _fa_alpha(t):
    return 0.02 / (1.0 + t / 1000.0) ** 0.75


def _linear_td_run(name, X, nudge):
    """Semi-gradient TD(0) on one stream of FA_DAYS days of the policy FA_POLICY from the shelf 0."""
    rng = bookrng.generator(name)
    w = [0.0] * len(X[0])
    w[0] += nudge
    s = 0
    for t in range(FA_DAYS):
        r, nxt, _sold = mdps._step(s, FA_POLICY[s], mdps._demand_from(rng.uniform()))
        v, vn = 0.0, 0.0
        for i in range(len(w)):
            v += w[i] * X[s][i]
        for i in range(len(w)):
            vn += w[i] * X[nxt][i]
        delta = r + GAMMA * vn - v
        step = _fa_alpha(t) * delta
        for i in range(len(w)):
            w[i] += step * X[s][i]
        s = nxt
    return w


def linear_td():
    V, _ = mdps._evaluate_solve(FA_POLICY, GAMMA)
    d = _stationary(FA_POLICY)
    rows_x = []
    for s in range(N):
        rows_x.append([s] + [", ".join(f"{x:g}" for x in _features(f)[s]) for f in FEATURE_SETS])
    rows, bounds = [], []
    for f in FEATURE_SETS:
        X = _features(f)
        wtd = _td_fixed_point(X, d, FA_POLICY)
        wbest = _best_fit(X, d, V)
        vtd, vbest = _predict(X, wtd), _predict(X, wbest)
        ve_td = sum(d[s] * (vtd[s] - V[s]) ** 2 for s in range(N))
        ve_best = sum(d[s] * (vbest[s] - V[s]) ** 2 for s in range(N))
        bounds.append((f, math.sqrt(ve_td), math.sqrt(ve_best) / math.sqrt(1.0 - GAMMA * GAMMA)))
        names = _run_names(f"linear.{f}", FA_RUNS)
        per = []
        for nudge in VARIANTS:
            acc = [0.0] * len(wtd)
            for nm in names:
                w = _linear_td_run(nm, X, nudge)
                acc = [a + x / FA_RUNS for a, x in zip(acc, w)]
            per.append(acc)
        rows.append([f, "TD fixed point", ", ".join(mdps._fmt(x, 3) for x in wtd), ", ".join(mdps._fmt(x, 3) for x in vtd),
                     f"{ve_td:.3f}"])
        rows.append([f, "best weighted fit", ", ".join(mdps._fmt(x, 3) for x in wbest), ", ".join(mdps._fmt(x, 3) for x in vbest),
                     f"{ve_best:.3f}"])
        rows.append([f, f"sampled, mean of {FA_RUNS} runs",
                     ", ".join(_cell([p[i] for p in per], 2) for i in range(len(wtd))),
                     ", ".join(_cell([_predict(X, p)[s] for p in per], 2) for s in range(N)), ""])
    out = [
        f"The policy of the table is (3, 0, 0, 0), order three units on an empty shelf, whose exact values from the previous chapter "
        f"are V = ({', '.join(mdps._fmt(v, 3) for v in V)}) at gamma = {GAMMA}. Its chain visits every shelf level, with the long-run "
        f"share of days mu = ({', '.join(mdps._fmt(x, 4) for x in d)}) (the solution of mu P = mu with the entries summing to 1). "
        f"The value of a state is approximated by x(s) . w with a feature vector x(s) per state; the three feature tables are printed in "
        f"full (one row per state):",
        "",
        _table(["s", "one-hot", "linear", "quadratic"], rows_x, align="rlll"),
        "",
        f"For each table: the TD fixed point, the weights at which the average of the semi-gradient update over the days, "
        f"sum_s mu(s) x(s) (r(s) + gamma x(s')^T w - x(s)^T w) with r(s) = R(s, pi(s)) and the average over s' taken with P, is zero "
        f"(one linear system); the weights that minimise the mu-weighted squared error sum_s mu(s) (x(s)^T w - V(s))^2 (the normal "
        f"equations); and the weights that semi-gradient TD(0) reaches. 'weighted error' is sum_s mu(s) (V_hat(s) - V(s))^2. "
        f"Sampled runs: {FA_RUNS} runs of {FA_DAYS} days from the shelf 0, run i of table f with the generator named "
        f"value_learning.linear.<f>.run<i> with f one-hot, linear or quadratic (for example one-hot run 0 has seed "
        f"{bookrng.seed_of('value_learning.linear.one-hot.run0'):#018x}, linear run 0 {bookrng.seed_of('value_learning.linear.linear.run0'):#018x}, "
        f"quadratic run 0 {bookrng.seed_of('value_learning.linear.quadratic.run0'):#018x}), one uniform() per day for the demand; weights start at 0 and "
        f"change every day in the order of the days, delta = r + gamma x(s')^T w - x(s)^T w computed from the weights before the "
        f"update, then w <- w + alpha_t delta x(s) with alpha_t = 0.02 / (1 + t / 1000)^0.75 for the day t counted from 0; "
        f"the weights shown are those after the last day, averaged over the runs (divisor {FA_RUNS}).",
        "",
        _table(["features", "weights from", "w", "V_hat(0 .. 3)", "weighted error"], rows, align="llllr"),
        "",
        "In the norm sqrt(sum_s mu(s) e(s)^2) of an error vector e, the TD fixed point is within 1 / sqrt(1 - gamma^2) times the best "
        f"weighted fit (the bound of Tsitsiklis and Van Roy for the on-policy case): {'; '.join(f'{f}: {a:.3f} against a bound of {b:.3f}' for f, a, b in bounds)}.",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ the two-state example that diverges


DIV_GAMMA, DIV_ALPHA, DIV_K = 0.9, 0.1, 10


def _diverge(updates, mode):
    """Weight w after each update: states 1 and 2 have features 1 and 2 (V_hat = w, 2 w), all rewards 0.

    'off': only the transition 1 -> 2 is ever updated. 'frozen': as 'off' but the target uses a copy of w refreshed every
    DIV_K updates. 'on': the episode 1 -> 2 -> end, both transitions updated in that order, one update per transition."""
    w, frozen = 1.0, 1.0
    out = [w]
    for k in range(updates):
        if mode == "on" and k % 2 == 1:
            y = 0.0                                   # the transition 2 -> end: the target is the reward 0
            w += DIV_ALPHA * (y - 2.0 * w) * 2.0
        else:
            ref = frozen if mode == "frozen" else w
            y = DIV_GAMMA * 2.0 * ref                  # the transition 1 -> 2 with reward 0
            w += DIV_ALPHA * (y - w) * 1.0
        if mode == "frozen" and (k + 1) % DIV_K == 0:
            frozen = w
        out.append(w)
    return out


def divergence():
    ks = (0, 10, 20, 50, 100, 200)
    off, frozen, on = _diverge(200, "off"), _diverge(200, "frozen"), _diverge(200, "on")
    rows = [[k, _sci(off[k]), _sci(frozen[k]), _sci(on[k])] for k in ks]
    out = [
        f"Two states with one weight w and the features x(1) = 1 and x(2) = 2, so V_hat(1) = w and V_hat(2) = 2w. Every reward is 0, so "
        f"the true values are 0 and w = 0 is exact. The transition is 1 -> 2, and from 2 the episode ends (terminal value 0). "
        f"gamma = {DIV_GAMMA}, alpha = {DIV_ALPHA}, w = 1 before the first update. 'off' updates only the transition 1 -> 2, again and "
        f"again: delta = 0 + gamma * 2w - w and w <- w + alpha delta * 1. 'frozen' does the same with the target computed from a copy "
        f"of w that is replaced by w after every {DIV_K} updates (the copy starts at 1). 'on' updates the whole episode, the update of "
        f"1 -> 2 followed by the update of 2 -> end (delta = 0 - 2w, w <- w + alpha delta * 2), one update per transition. The "
        f"table is w after k updates (k counts single updates, in the order stated). The arithmetic is exact to rounding and the "
        f"growth is a plain geometric sequence, so four significant digits are printed.",
        "",
        _table(["updates k", "off-policy, w", f"frozen target (copy every {DIV_K})", "on-policy, w"], rows),
        "",
        f"Per update, 'off' multiplies w by 1 + alpha (2 gamma - 1) = {1 + DIV_ALPHA * (2 * DIV_GAMMA - 1):.2f}; per round of "
        f"{DIV_K} updates, 'frozen' multiplies it by {(1 - DIV_ALPHA) ** DIV_K + 2 * DIV_GAMMA * (1 - (1 - DIV_ALPHA) ** DIV_K):.4f}; "
        f"per episode (two updates), 'on' multiplies it by {(1 + DIV_ALPHA * (2 * DIV_GAMMA - 1)) * (1 - 4 * DIV_ALPHA):.3f}.",
    ]
    return "\n".join(out)


def _divergence_curves(updates=100):
    return {m: _diverge(updates, m) for m in ("off", "frozen", "on")}


# ------------------------------------------------------------------ the work of an update


def work_of_updates():
    def linear_flops(d):
        # two dot products, target, delta, the schedule (divide, add, power, divide), alpha*delta, d multiplies, d adds
        return 2 * (2 * d - 1) + 2 + 1 + 4 + 1 + 2 * d

    rows = [
        ["tabular TD(0)", "target r + gamma V(s'): 1 multiply, 1 add; delta: 1 subtract; alpha_n: 1; alpha * delta and the add: 2", 6],
        ["tabular SARSA", "the same, Q(s', a') read in place of V(s')", 6],
        ["tabular Q-learning", "the same, with the maximum over the m actions (comparisons are not counted)", 6],
        ["linear TD(0), d = 2 (linear table)", "two dot products of 2d - 1, target 2, delta 1, alpha_t = 0.02 / (1 + t / 1000)^0.75 4 "
                                               "(divide, add, power, divide), alpha * delta 1, d multiplies and d adds for the weights", linear_flops(2)],
        ["linear TD(0), d = 3 (quadratic table)", "the same with d = 3", linear_flops(3)],
        ["linear TD(0), d = 4 (one-hot table)", "the same with d = 4 (against 6 for a table lookup, whose schedule n^-0.7 is one power)", linear_flops(4)],
    ]
    day_mc, visit_mc = 2 * EVAL_DAYS, 3 * N
    variants, _first = _mean_curves()
    first_below = {}
    for m in ("MC", BEST):
        first_below[m] = [next((e for e, x in enumerate(c[m], start=1) if x <= 1.0), None) for c in variants]
    rows_cost = []
    for m, label in (("MC", "first-visit Monte Carlo"), (BEST, "TD(0), alpha_n = n^-0.7")):
        e = _same([str(x) for x in first_below[m]])
        if e == "unstable" or e == "None":
            rows_cost.append([label, "unstable", "", ""])
            continue
        e = int(e)
        per_ep = day_mc + visit_mc if m == "MC" else 6 * EVAL_DAYS
        rows_cost.append([label, e, f"{e * EVAL_DAYS:,}", f"{e * per_ep:,} (at most)" if m == "MC" else f"{e * per_ep:,}"])
    out = [
        "One operation is a multiply, an add, a subtract or a divide; a power and a comparison count as above (a power for the "
        "schedule counts one operation, a comparison none). Per update:",
        "",
        _table(["update", "count", "FLOPs"], rows, align="llr"),
        "",
        f"Per episode of {EVAL_DAYS} days Monte Carlo spends 2 operations a day on the backward return G <- r + gamma G "
        f"({day_mc}) and at most 3 per state on the update (G - V)/n added to V ({visit_mc}); TD(0) spends 6 a day ({6 * EVAL_DAYS}). "
        f"The first episode after which the mean error of the table of learning curves is at most 1.0 (the mean over the {EVAL_RUNS} "
        f"runs of the root-mean-square error, as there), with the days and the operations of the updates that it took:",
        "",
        _table(["method", "episodes", "days of experience", "FLOPs of updates"], rows_cost, align="lrrr"),
    ]
    return "\n".join(out)
