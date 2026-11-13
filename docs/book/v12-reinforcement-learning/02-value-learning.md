<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learning Values from Experience

*How can an agent learn what states are worth without a model of the world?*

Suppose the printed tables of the previous chapter are gone. The keeper still has the shop, and the shop still writes one line a day into a notebook: the stock in the morning, the units ordered, the units that sold, what the day paid and tomorrow's stock. Nothing in the notebook says that demand is $0$, $1$ or $2$ with probabilities $0.3$, $0.4$ and $0.3$. Nothing says what a full shelf is worth. Yet the question of the last chapter stands: what is each shelf level worth to a keeper who orders one unit when the shelf is empty? Here is the notebook for ten days, the very trajectory of the previous chapter, with one update of an estimate written beside every line.

<!-- worked: value_learning.ledger_updates -->
The ten-day ledger of the previous chapter: policy (1, 0, 0, 0), start s_0 = 2, generator the book's SplitMix64 with seed 0xaa77e877a9f056b2 (12283542109326169778), one uniform() per day in order and the demand rule of that chapter. The estimates start at V(s) = 0 for every s. Each day, in order, the target r + gamma V(s') is formed from the estimates as they stand (gamma = 0.9), the difference delta = target - V(s) is computed, and V(s) is replaced by V(s) + 0.5 * delta before the next day is looked at; when s' = s the target uses the estimate of the day before.

| t | s | a | D | r | s' | V(s) | target | delta | new V(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0 | 0 | -2.0 | 2 | 0.000 | -2.000 | -2.000 | -1.000 |
| 1 | 2 | 0 | 0 | -2.0 | 2 | -1.000 | -2.900 | -1.900 | -1.950 |
| 2 | 2 | 0 | 0 | -2.0 | 2 | -1.950 | -3.755 | -1.805 | -2.853 |
| 3 | 2 | 0 | 0 | -2.0 | 2 | -2.853 | -4.567 | -1.715 | -3.710 |
| 4 | 2 | 0 | 1 | 3.0 | 1 | -3.710 | 3.000 | 6.710 | -0.355 |
| 5 | 1 | 0 | 0 | -1.0 | 1 | 0.000 | -1.000 | -1.000 | -0.500 |
| 6 | 1 | 0 | 0 | -1.0 | 1 | -0.500 | -1.450 | -0.950 | -0.975 |
| 7 | 1 | 0 | 2 | 4.0 | 0 | -0.975 | 4.000 | 4.975 | 1.512 |
| 8 | 0 | 1 | 0 | -3.5 | 1 | 0.000 | -2.139 | -2.139 | -1.069 |
| 9 | 1 | 0 | 0 | -1.0 | 1 | 1.512 | 0.361 | -1.151 | 0.937 |

After the ten updates V = (-1.069, 0.937, -0.355, 0.000). The Monte Carlo estimate from the same ledger uses the first visit of each state, the returns of the previous chapter's table at gamma = 0.9: V(0), V(1), V(2), V(3) = -4.400, -1.868, -6.012, none (the state 3 is never visited). The exact values are (6.750, 9.250, 11.168, 11.668).
<!-- /worked -->

Two things happened on those lines. The first is the rule of the update. On the first day the shelf holds two units, nothing sells and the day pays $-2$. The keeper keeps a guess $V(2)$ of what a shelf of two is worth, starting at $0$, and asks what the day suggests. The day paid $-2$ and left her on a shelf of two again, which her own guess values at $0$, so the day suggests $-2 + 0.9 \times 0 = -2$. The guess was $0$, the suggestion is $-2$, and she moves half the distance: $V(2) = -1$. The suggestion, $r + \gamma V(s')$, is the **target**; the gap between target and guess, here $-2.000$, is the **TD error** $\delta$; and the guess moves a fraction $\alpha$ of the gap, $V(s) \leftarrow V(s) + \alpha\,\delta$. The fraction is the step size (the learning rate of Volume III). No probability was used anywhere, only the line of the notebook.

The second thing is that the numbers are bad. After ten days $V(2)$ is $-0.355$ and $V(0)$ is $-1.069$, against the true $11.168$ and $6.750$. A second route to an estimate uses the same notebook and no guess at all: wait until the days that follow are known and use the discounted sum of what was actually paid, the return $G_t$ of the previous chapter. The return of the first day on a shelf of two is $-6.012$, so the keeper who averages returns would start from $-6.012$ for $V(2)$ and from $-4.400$ for $V(0)$. Both routes are far from the truth, for the same reason: ten days that happened to open with four idle mornings are not the shop. The interesting question is what each route does once there are thousands of lines, and which of them to trust when there are few.

Four ideas from earlier chapters are the tools of the answer. Bring them back from memory; the answers are at the end of the chapter.

1. (Volume XII, Chapter 1, "Markov Decision Processes", the Bellman equation.) Write the Bellman equation for the value $V^\pi(s)$ of a policy $\pi$, and say in a sentence what it expresses.
2. (Volume XII, Chapter 1, the value function.) The return of one trajectory is a random number. What is the value $V^\pi(s)$ in terms of returns, and which number is $V^\pi(0)$ for the policy $(1, 0, 0, 0)$ at $\gamma = 0.9$?
3. (Volume III, Chapter 7, "Optimizers: From SGD to AdamW", stochastic gradient descent.) A single example $y$ gives the loss $\ell(w) = \tfrac12 (y - w)^2$. Write one SGD step with rate $\eta$ and say what fraction of the way from $w$ to $y$ it moves $w$.
4. (Volume III, Chapter 7.) Apply that step to the examples $y_1, y_2, \dots$ one after another with the rate $\eta_n = 1/n$ for the $n$-th example, starting from any $w_0$. What is $w_n$?

## Averaging what actually happened

The shortest route to a value estimate is to take the definition literally. The value of a state is the expected return from it, so run the policy, record the return that followed each visit, and average. This is **Monte Carlo evaluation**. Chapter 1 did it once, for the empty shelf: $2000$ simulated lives, whose mean return agreed with the solved $V(0)$ to a hundredth of a standard error. What changes now is that the lives are not simulated from a table; they are the keeper's own days.

A shop never ends, but a notebook is cut into episodes. This chapter draws episodes of $100$ days, each from a start state chosen at random (the shop is opened on a shelf of $0$, $1$, $2$ or $3$ with equal probability, which is also the only way a policy that never visits a shelf of three will learn about it). The return of a day is the discounted sum of what the rest of the episode paid, with the days after the episode ignored, and the estimate of a state uses the return of the first visit to it in the episode (first-visit Monte Carlo; every visit also works, and a table after the learning curves compares the two). Averaging is done incrementally, because a running mean needs no list of past returns. If $V_{n-1}$ is the mean of the first $n - 1$ returns and $G_n$ the next, the mean of $n$ returns is

$$V_n = \frac{(n-1)\,V_{n-1} + G_n}{n} = V_{n-1} + \frac{1}{n}\,(G_n - V_{n-1}).$$

That is an update of the form met in the opening, with the step $1/n$ and the return as the target, and it is the answer to the fourth recall question: the SGD step on the loss $\tfrac12(G - V)^2$ is $V \leftarrow V + \eta\,(G - V)$, and with $\eta_n = 1/n$ it is the running mean.

```text
function mc_update(V: inout array<real>[n], count: inout array<int>[n],
                   episode: list<(int, real, int)>, gamma: real) -> none:
    require 0 <= gamma and gamma < 1
    G = 0.0
    first = zeros(real, [n])
    seen = zeros(int, [n])
    for t in len(episode) - 1 down to 0:
        G = episode[t][1] + gamma * G
        first[episode[t][0]] = G
        seen[episode[t][0]] = 1
    for s in 0 .. n:
        if seen[s] == 1:
            count[s] = count[s] + 1
            V[s] = V[s] + (first[s] - V[s]) / count[s]
```

Each day of the `episode` is the triple `(s, r, s')` of the state, the reward and the next state. Walking the days backwards computes every return from $G_t = r_t + \gamma G_{t+1}$, and because the loop ends at the earliest day, the value that remains in `first[s]` is the return of the first visit. The days after the last one are ignored (the loop starts from $G = 0$), which biases a first-visit return by at most $\gamma^{100 - t}$ times the largest value for a first visit on day $t$; the table of learning curves below states how large that is for the episodes it uses.

Monte Carlo is unbiased: every return it averages has the value as its mean, by definition. What it pays is spread. A return adds up a hundred days of random demand, and one return of one visit is a single draw from that distribution. The previous chapter could see it in the sampled standard deviation of $2.962$ for a return from the empty shelf; the exact spread follows from the same Bellman recursion that gave the exact mean. Squaring $G = r + \gamma G'$ and taking expectations gives $E[G^2] = E[r^2] + 2\gamma E[r\,V(s')] + \gamma^2 E[G'^2]$ (the cross term uses that $G'$ depends on the past only through $s'$), which is a linear system with the matrix $\mathbf{I} - \gamma^2\mathbf{P}_\pi$, solved as before, and the variance is $E[G^2] - V(s)^2$.

<!-- worked: value_learning.target_spread -->
Policy (1, 0, 0, 0), gamma = 0.9; every number is computed exactly from the tables of the previous chapter, none is sampled. The Monte Carlo target of a state is the whole return G; its mean is V(s) and its standard deviation sd(G) comes from E[G^2] = E[r^2] + 2 gamma E[r V(s')] + gamma^2 E[G'^2], a linear system with the matrix I - gamma^2 P_pi, minus V(s)^2. The one-day target is y = r + gamma W(s') for an estimate W of the next state, with the three outcomes d = 0, 1, 2 of the day as its only randomness; sd(y) is taken for W = V and for W = 0 (all values zero). The three last columns are E[y] - V(s) for W = V, for W = V + 1 (every estimate one too high) and for W = 0; a number that rounds to zero is printed 0.000 whatever its sign.

| s | V(s) | sd(G) | sd(y), W = V | sd(y), W = 0 | bias, W = V | bias, W = V + 1 | bias, W = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.750 | 2.891 | 1.260 | 2.291 | 0.000 | 0.900 | -6.750 |
| 1 | 9.250 | 2.891 | 1.260 | 2.291 | 0.000 | 0.900 | -6.750 |
| 2 | 11.168 | 3.671 | 2.337 | 3.873 | 0.000 | 0.900 | -8.168 |
| 3 | 11.668 | 4.550 | 3.046 | 3.873 | 0.000 | 0.900 | -9.668 |
<!-- /worked -->

The first column of spreads is the one of Monte Carlo. A return from the empty shelf has an exact standard deviation of $2.891$ (the $2.962$ of the previous chapter's $2000$ truncated draws is $1.6$ standard errors above it, where the standard error of a sample standard deviation is about $\sigma/\sqrt{2 \times 1999} = 0.046$, not the $0.066$ of the mean in that chapter, which is ordinary sampling noise), and from a full shelf $4.550$, against values of $6.750$ and $11.668$. To pin a value down to $\pm 0.1$ by averaging independent returns takes about $(2.891/0.1)^2 \approx 836$ of them for state $0$, and about $2071$ for state $3$ (the unrounded spread $4.5505$ gives $2070.7$, the printed $4.550$ gives $2070$).

## A target made of an estimate

Look again at the opening. The keeper did not wait for the return. She took one day, $r$ and $s'$, and replaced the rest of the return by her current guess of what $s'$ is worth. That replacement is the content of the Bellman equation. It says that $V^\pi(s) = E\,[\,r + \gamma V^\pi(s')\,]$, the mean over the demand of the day, with the true value of the next state inside. A single day produces one draw $y = r + \gamma V(s')$ of a random variable whose mean is the right-hand side, and the update

$$V(s) \leftarrow V(s) + \alpha\,\big(\,r + \gamma V(s') - V(s)\,\big) = V(s) + \alpha\,\delta$$

moves $V(s)$ a fraction $\alpha$ of the way to that draw. This is **temporal-difference learning**, TD(0) in its simplest form: it learns a value from the difference between two successive estimates of it, the one made before the day ($V(s)$) and the one made after ($r + \gamma V(s')$), and needs neither a model nor the end of the episode. The rule uses an estimate to improve an estimate, which is called **bootstrapping** (nothing to do with the resampling bootstrap of Volume I): the target is only as good as $V(s')$ is.

Think of what the update does on average. If $V$ is the vector of estimates, the expected target of state $s$ is the backup $\mathcal{B}_\pi V(s)$ of the previous chapter, so in expectation the update moves $V(s)$ from where it is by $\alpha\,(\mathcal{B}_\pi V(s) - V(s))$, a step toward the output of the contraction whose fixed point is $V^\pi$. The sweep of Chapter 1 made that step for every state at once, with the exact average over the demand; TD(0) makes it for one state and one sampled demand, and the noise of the sample is what the step size has to average away. Algorithms of this shape, root-finding from noisy evaluations, are stochastic approximation. They are the SGD of Volume III with a target that is not a label but another estimate.

The routine is short. The state of the shop, the reward and the next state come from `sample_day(s, a, rng)`, which draws one `rng.uniform()` for the demand, applies the demand rule of the previous chapter (demand $0$ if $u < 0.3$, $1$ if $u < 0.7$, otherwise $2$) and returns the reward of the day and the next stock.

```text
function td0_update(V: inout array<real>[n], count: inout array<int>[n],
                    s: int, r: real, next: int, gamma: real,
                    step: function(int) -> real) -> real:
    count[s] = count[s] + 1
    delta = r + gamma * V[next] - V[s]
    V[s] = V[s] + step(count[s]) * delta
    return delta

function td0_episode(V: inout array<real>[n], count: inout array<int>[n],
                     episode: list<(int, real, int)>, gamma: real,
                     step: function(int) -> real) -> none:
    for t in 0 .. len(episode):
        (s, r, next) = episode[t]
        td0_update(V, count, s, r, next, gamma, step)
```

Every day is applied before the next is looked at, in the order of the days, so a day that stays on the same shelf ($s' = s$) uses the estimate as the previous day left it. `count[s]` counts the updates of `s` including the current one, so the first update of a state uses `step(1)`. No state is terminal in this chapter's shop: the update on the last day of an episode bootstraps from the estimate of the state reached, like any other.

The target of TD is a different random variable from the one of Monte Carlo, with a different mean and a different spread, and the exact table above shows both. At estimates that are exactly right ($W = V$) the target has the right mean, and its standard deviation is only the spread of one day's demand: $1.260$ from the empty shelf, against $2.891$ for the return, a factor of $2.3$ smaller. If every estimate is one too high ($W = V + 1$), the mean of the target is wrong by exactly $\gamma \times 1 = 0.900$ in every state, because the error of $s'$ enters the target multiplied by $\gamma$. And at the starting estimates of $0$ the target of the empty shelf has a spread of $2.291$, for the keeper's $r$ alone, and a bias of $-6.750$: the update is pulling toward the reward of the day, not toward the value. Monte Carlo has no such bias and a large variance; TD has a small variance and a bias that is the error of its own estimates, shrunk by $\gamma$ and decreasing as the estimates improve.

So the two methods are in a contest between a bias that is paid off as the estimates improve and a variance that is paid off only by more samples, and which of them is ahead depends on the amount of data. That is a measurement, not an argument. Two hundred repeats of the same experiment, each with its own episodes, give the bias and the spread of the estimate of $V(0)$ at several amounts of data, with both methods learning from the same episodes.

<!-- worked: value_learning.bias_and_spread -->
200 independent repeats, each with its own generator (run i has the generator named value_learning.repeat.run<i> for i = 0 .. 199, so its seed is seed_of of that name: the first three are run0 0x2483ab8cea36f0d0, run1 0xe77ab99b801c5327, run2 0x5e31918a5773366c). A repeat draws 100 episodes of 100 days in order, each as in the table of learning curves (a start state from integer(4), then one uniform() per day), and both methods learn from the same episodes: first-visit Monte Carlo with the step 1/n of the state, and TD(0) with the step n^-0.7 of the state, n counting the updates of that state from 1, both from the estimates 0. The table is the error of the estimate of V(0) = 6.750 after the stated number of episodes: bias is the mean of (estimate - V(0)) over the repeats, sd the standard deviation over the repeats with the divisor 199, and rmse the square root of the mean of the squares of the errors (divisor 200).

| episodes | MC bias | MC sd | MC rmse | TD bias | TD sd | TD rmse |
|---|---:|---:|---:|---:|---:|---:|
| 2 | -0.224 | 1.772 | 1.782 | -2.694 | 0.856 | 2.826 |
| 5 | -0.160 | 1.232 | 1.239 | -1.811 | 0.625 | 1.915 |
| 10 | -0.022 | 0.862 | 0.860 | -1.198 | 0.435 | 1.275 |
| 20 | 0.002 | 0.648 | 0.647 | -0.720 | 0.320 | 0.788 |
| 50 | -0.009 | 0.418 | 0.417 | -0.287 | 0.193 | 0.346 |
| 100 | -0.012 | 0.271 | 0.271 | -0.129 | 0.134 | 0.185 |
<!-- /worked -->

The Monte Carlo column behaves as the theory says: its bias is zero within sampling error (the standard error of each bias is the spread divided by $\sqrt{200}$, $0.061$ at $10$ episodes), and its spread falls as $1/\sqrt{\text{episodes}}$ from $0.862$ at $10$ episodes to $0.271$ at $100$. The TD column shows the other half. At $10$ episodes its estimate is $1.198$ below the true value on average, a bias nearly three times its spread of $0.435$, which is itself half of Monte Carlo's $0.862$; the error is dominated by the bias, and its root mean square error of $1.275$ is larger than Monte Carlo's $0.860$. The bias shrinks as the estimates it bootstraps from improve ($-0.287$ at $50$ episodes, $-0.129$ at $100$), and between $20$ episodes (rmse $0.788$ against $0.647$) and $50$ (rmse $0.346$ against $0.417$) the two cross: from then on TD is the better estimator, with an error of $0.185$ against $0.271$ at $100$ episodes. TD has the smaller spread throughout, but at first the larger error: a bias that is paid off over time ends up cheaper than a variance that is paid off only by more samples, and the two cross between $20$ and $50$ episodes of $100$ days.

The same contest on all four states, with several step-size schedules and out to $200$ episodes, is the table of learning curves, each entry the mean over ten runs.

<!-- worked: value_learning.learning_curves -->
Policy (1, 0, 0, 0), gamma = 0.9. A run is 200 episodes of 100 days with its own generator, the SplitMix64 of the book seeded with the seed of its name (below); an episode draws one integer(4) for its start state and then, for each of its 100 days in order, one uniform() for the demand (the action is the policy's, so nothing else is drawn). Every method learns from the same episodes of a run. All estimates start at 0 and no state is terminal: the update on the last day of an episode uses the estimate of the state reached, and the days after an episode are ignored by Monte Carlo. Monte Carlo is first visit: after an episode the return G of the first visit to each state (G_t = r_t + gamma G_(t+1) from G_100 = 0) moves its estimate by (G - V)/n, n counting the updates of that state from 1. TD(0) updates the state of every day, in the order of the days, immediately: V(s) <- V(s) + alpha_n (r + gamma V(s') - V(s)) with the estimate of s' as it stands and n the number of updates of s so far including this one, so the first update of a state uses alpha_1; the four schedules are alpha_n = 1/n, n^-0.7, and the constants 0.5 and 0.1. The error of a method after an episode is the root mean square over the four states of V(s) - V*(s) against the solved values of the previous chapter (divisor 4), and each entry is the mean over the 10 runs (summed in run order, divisor 10). The largest first-visit day seen in any episode of any run is 19, so the days ignored by Monte Carlo weigh at most 2.3e-03 in a first-visit return.

| episodes | days | MC | TD 1/n | TD n^-0.7 | TD 0.5 | TD 0.1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 7.731 | 8.721 | 8.284 | 8.064 | 8.812 |
| 2 | 200 | 6.929 | 8.511 | 7.743 | 7.332 | 8.259 |
| 5 | 500 | 2.761 | 6.442 | 4.914 | 5.201 | 7.464 |
| 10 | 1000 | 1.852 | 5.603 | 3.258 | 3.216 | 6.576 |
| 20 | 2000 | 1.583 | 5.023 | 2.198 | 2.302 | 5.179 |
| 50 | 5000 | 0.883 | 4.133 | 0.860 | 1.529 | 2.515 |
| 100 | 10000 | 0.524 | 3.628 | 0.490 | 1.507 | 0.687 |
| 200 | 20000 | 0.358 | 3.251 | 0.307 | 2.019 | 0.790 |

Every number learned from samples is printed with the digits that the run and four copies of it share: the copies start with the first stored value (V(0), Q(0, 0), the first weight or Q(A, left), as the example has it) changed by +1e-12, -1e-12, +3e-12 and -3e-12 (a stability check applied to every trained number), and a digit is printed only if all five agree on it ('about x' keeps the leading digits that survive, 'unstable' means that none does). The copies of an evaluation learn from the same episodes, because the data do not depend on the estimates; the copies of a learner that chooses its own actions use the same generator, so their draws are the same and they differ only if a nudge flips a comparison. A run is a deterministic computation, so another implementation reproduces the printed digits.

| run | generator name | seed |
|---:|---|---:|
| 0 | value_learning.evaluate.run0 | 0x35e7a981db3ddf1f |
| 1 | value_learning.evaluate.run1 | 0x58eed793c4e5a7ba |
| 2 | value_learning.evaluate.run2 | 0x9a3d03f8f654ac19 |
| 3 | value_learning.evaluate.run3 | 0xce8f3ca2a1acd785 |
| 4 | value_learning.evaluate.run4 | 0x84d707418ece777a |
| 5 | value_learning.evaluate.run5 | 0xda7c8c5ef76c7c90 |
| 6 | value_learning.evaluate.run6 | 0x658a9fbbeefcd84f |
| 7 | value_learning.evaluate.run7 | 0xe0dddb1ba8603e72 |
| 8 | value_learning.evaluate.run8 | 0xb0ee325bf24bc757 |
| 9 | value_learning.evaluate.run9 | 0x65bcfee57eee6278 |
<!-- /worked -->

![Five curves of the error against the exact values on logarithmic axes, episodes of 100 days across from 1 to 200 and the error up from 0.2 to 12. Monte Carlo falls from about 7.7 to 0.36. TD(0) with the step n to the power minus 0.7 falls from 8.3 to 0.31 and crosses below Monte Carlo near 50 episodes. TD(0) with the step 1/n falls only to 3.3. The constant steps 0.5 and 0.1 fall quickly at first and then flatten at about 1.5 to 2 and at 0.7 to 0.8.](img/td-versus-monte-carlo.svg)

The first row of the table is the opening again: after one episode both methods are bad, because three of the four states have not been visited long enough (the start state of an episode is the only way into a shelf of two or three under this policy) and the estimates of unvisited states are still $0$. The Monte Carlo column then falls steadily, from $7.731$ to $0.358$ at $200$ episodes, and TD with the step $n^{-0.7}$ follows more slowly at first and ends lower, $0.307$. Every-visit Monte Carlo is the other way to use an episode: the keeper averages the return of every day on which the state was visited, not only the first, and an episode that starts on the empty shelf or the shelf of one spends most of its days there (about $34$ of its first $50$ on the empty shelf, in the table below) and offers that many returns instead of one. Whether those extra returns are worth anything is a question for a measurement, so the table below repeats the experiment with the same episodes, $200$ times.

<!-- worked: value_learning.every_visit_monte_carlo -->
First-visit and every-visit Monte Carlo on the same episodes: policy (1, 0, 0, 0), gamma = 0.9, 200 independent repeats (repeat i has the generator named value_learning.everyvisit.run<i>, i = 0 .. 199; the first seeds are run0 0xc2e3cef6f00177b4, run1 0x4e0d6efff0bce482), each 200 episodes of 100 days drawn as in the table of learning curves. Both estimates start at 0 and are running means with the step 1/n, n counting the returns averaged for that state from 1 (the first step is 1, so the start value never matters). First visit averages the return of the first visit of an episode, every visit the return of each of the first 50 days of an episode (days 0 .. 49, each return being the discounted sum to the end of the 100-day episode). The first table is the root mean square error over the four states after the stated number of episodes, as in the table of learning curves (divisor 4), averaged over the 200 repeats; 'EV - FV' is the mean over the repeats of the difference of the two errors of the same repeat and 'se' its standard error (the standard deviation of the differences with the divisor 199, over the square root of 200).

| episodes | first visit | every visit | EV - FV | se |
|---|---:|---:|---:|---:|
| 5 | 3.372 | 3.223 | -0.149 | 0.031 |
| 10 | 1.948 | 1.837 | -0.111 | 0.035 |
| 20 | 1.224 | 1.185 | -0.038 | 0.030 |
| 50 | 0.683 | 0.689 | +0.006 | 0.020 |
| 100 | 0.502 | 0.508 | +0.006 | 0.014 |
| 200 | 0.380 | 0.391 | +0.011 | 0.010 |

The second table is the state-by-state view after 200 episodes. 'visits' is the mean over the repeats of the number of returns averaged; bias and sd are the mean and the standard deviation (divisor 199) over the repeats of V(s) - V*(s); 'equivalent n' is (sd(G) / sd)^2 with sd(G) the exact standard deviation of one return from the table of target spreads, the number of independent returns that would give that spread; 'weight n' is the mean over the repeats of (sum of m_i)^2 / (sum of m_i^2) over the episodes, m_i the number of every-visit returns that episode i contributed to the state: the number of equally weighted episodes that would carry the same weights.

| s | visits FV | visits EV | bias FV | sd FV | bias EV | sd EV | equivalent n FV | equivalent n EV | weight n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200.0 | 6819.2 | +0.022 | 0.196 | -0.007 | 0.121 | 219 | 572 | 198 |
| 1 | 200.0 | 2997.5 | +0.022 | 0.212 | -0.002 | 0.123 | 186 | 557 | 191 |
| 2 | 78.1 | 111.5 | +0.028 | 0.424 | +0.070 | 0.472 | 75 | 60 | 61 |
| 3 | 50.0 | 71.9 | -0.041 | 0.683 | -0.016 | 0.752 | 44 | 37 | 39 |
<!-- /worked -->

The first table says that every visit is ahead for the first ten episodes (by $0.149$ and $0.111$ at $5$ and $10$, three to five standard errors of the difference), that the two are level from $20$ episodes on, and that after $50$ every visit is, if anything, a little behind ($+0.006$ at $50$ and $100$ episodes, $+0.011$ at $200$, each within about one standard error): at $100$ and $200$ episodes the differences are within sampling noise. The second table says where the advantage comes from and why it stops. The empty shelf and the shelf of one are visited thousands of times, and their spread falls from $0.196$ and $0.212$ to $0.121$ and $0.123$; but a shelf of two or three is visited only at the start of an episode (the policy never orders on it, so the stock only falls) and on the days that stay there, so every visit adds about forty per cent more returns ($111.5$ against $78.1$ for a shelf of two) and those are consecutive days of the same episode, whose returns share most of their terms. The spread of the estimate of a shelf of two does not fall ($0.424$ first visit, $0.472$ every visit), and of a shelf of three it rises ($0.683, 0.752$). In the units of independent returns the extra visits are worth less than they cost: $60$ equivalent returns for $111.5$ visits of a shelf of two, against $75$ for the $78.1$ first visits, and the 'weight n' column, the number of equally weighted episodes that the visit counts amount to, reproduces it ($61$ and $39$ against $60$ and $37$): the estimate weighs an episode by how many times it stood on the shelf, and unequal weights on nearly duplicate returns count as fewer samples. For the empty shelf the same column is $198$ while the equivalent number is $572$, because the days of one long episode that are far apart share almost nothing, and there every visit does carry new information. The root mean square error over the four states is dominated by the two rarely visited ones, which is why the gain of the first rows turns into a tie. Every visit is not unbiased in general, since it weighs an episode by how often the episode visited the state; the largest bias measured here is $+0.070$ for a shelf of two (two standard errors over $200$ repeats, against a spread of $0.472$), small beside the spread.

Every other column of the table of learning curves is the subject of the next section.

## Step sizes that converge and steps that do not

The update $V \leftarrow V + \alpha_n (y - V)$ has a free parameter, and the table above shows that the choice is not a detail. Two requirements pull against each other. The steps must be large enough, in total, to carry the estimate from wherever it started to wherever the answer is, however far that is; and they must become small enough, in total square, that the noise of the targets averages away instead of being passed on to the estimate. The classical statement of both is the pair of conditions of Robbins and Monro for the step $\alpha_n$ of the $n$-th update of one state:

$$\sum_{n \ge 1} \alpha_n = \infty \qquad\text{and}\qquad \sum_{n \ge 1} \alpha_n^2 < \infty.$$

If the first sum were finite, an estimate that started far from the value could never get there, whatever the data said: the total distance it can move is bounded by the sum of its steps times the size of the gaps. If the second were infinite, the noise would not die: each update adds a random disturbance of size proportional to $\alpha_n$, and the sum of the variances of those disturbances is proportional to $\sum \alpha_n^2$. Under both conditions, and with every state updated infinitely often, tabular TD(0) converges to $V^\pi$ with probability one when the rewards are bounded (Jaakkola, Jordan and Singh, and Tsitsiklis, both 1994, prove it for this family of updates; the book states the result and does not prove it). Here is what three schedules and a constant do to the two sums.

<!-- worked: value_learning.step_size_sums -->
The partial sums of a step-size schedule: alpha_n is the step of the n-th update, n from 1, and the two sums are accumulated in increasing n from 0.0 in double precision (four significant digits are printed, which they reproduce).

| alpha_n | n | sum of alpha_1 .. alpha_n | sum of alpha_1^2 .. alpha_n^2 |
|---|---:|---:|---:|
| n^-1 | 10 | 2.929 | 1.550 |
| n^-1 | 1,000 | 7.485 | 1.644 |
| n^-1 | 1,000,000 | 14.39 | 1.645 |
| n^-0.7 | 10 | 3.971 | 2.130 |
| n^-0.7 | 1,000 | 23.70 | 2.948 |
| n^-0.7 | 1,000,000 | 207.5 | 3.096 |
| n^-0.5 | 10 | 5.021 | 2.929 |
| n^-0.5 | 1,000 | 61.80 | 7.485 |
| n^-0.5 | 1,000,000 | 1,999 | 14.39 |
| 0.1 | 10 | 1.0000 | 0.1000 |
| 0.1 | 1,000 | 100.00 | 10.000 |
| 0.1 | 1,000,000 | 100,000 | 10,000 |
<!-- /worked -->

The step $1/n$ sits on the edge: its first sum grows like $\ln n$, which is slowly but without bound ($14.39$ at a million updates), and its second sum settles at $\pi^2/6 \approx 1.645$. The step $n^{-0.7}$ passes with room to spare, with a first sum that grows like $n^{0.3}$ and a second that converges to about $3.1$. The step $n^{-0.5}$ fails the second condition, since its second sum is the harmonic sum $14.39$ and growing. Any $n^{-p}$ with $p$ in $(1/2, 1]$ satisfies both. The constant step $0.1$ fails the second condition outright, with a sum of squares that is $0.01\,n$.

It is the table of learning curves that shows what those facts cost. The schedule $1/n$ is the one that is right for a mean of independent returns, where it gives the plain average. In TD it gives the plain average of all the targets so far, including the first ones, built on estimates of $V(s')$ that were $0$. With $\alpha_n = 1/n$ the target built on a bad guess at update 5 weighs exactly what the target built on a good one at update 500 weighs, so the early bias is forgotten at the speed of the harmonic series. After $200$ episodes of $100$ days its error is still $3.251$. The step $n^{-0.7}$ weighs recent targets more than old ones (the weight of target $k$ in the estimate after $n$ updates is $\alpha_k\prod_{j>k}(1 - \alpha_j)$, which for $\alpha_j = 1/j$ is exactly $1/n$ for every $k$ and for $n^{-0.7}$ is larger for the targets nearer to $n$), so the early bias leaves sooner, and it is the best column at the end of the table.

The constant steps are the opposite compromise. A constant $\alpha$ makes the estimate an exponentially weighted average of the targets, $V_n = (1 - \alpha)V_{n-1} + \alpha\,y_n$, which never stops listening to the newest target. For independent targets with spread $\sigma$ the spread of the estimate settles at $\sqrt{\alpha/(2 - \alpha)}\,\sigma$ (the stationary variance of that recursion is $\alpha^2\sigma^2/(1 - (1 - \alpha)^2)$) and does not shrink with more data. In return, an old error is forgotten at the geometric rate $1 - \alpha$. In the table the step $0.5$ keeps pace with $n^{-0.7}$ for the first ten episodes ($3.216$ against $3.258$ at $10$) and then stops at an error of about $1.5$ to $2.0$ that more data do not improve; the step $0.1$ is slower to arrive, reaches $0.687$ at $100$ episodes and is back at $0.790$ at $200$, a fluctuation and not a trend. Constant steps are what practice uses when the problem moves (the value of a state that changes with the world should be forgotten, and the second condition is an assumption of stationarity); a schedule that decays is what the theory asks for when it does not.

## From states to actions

A value of a state is not yet a decision. To choose an order the keeper needs what each order is worth, and with no table she cannot compute the lookahead $q(s, a) = R(s, a) + \gamma\sum_j P(j \mid s, a)\,V(j)$ of the previous chapter, because the sum over $j$ needs the probabilities. The way out is to learn the lookahead itself. Write $Q^\pi(s, a)$ for the expected return of taking the action $a$ in the state $s$ today and following the policy $\pi$ afterwards: it is $q(s, a)$ evaluated at $V^\pi$. Write $Q^\ast(s, a)$ for the same with the best policy afterwards, $q(s, a)$ evaluated at $V^\ast$; these are the numbers of the lookahead table of Chapter 1 (for an empty shelf $6.075$, $6.750$, $6.668$, $5.168$ for orders $0$ to $3$). With $Q^\ast$ in hand no model is needed to act: in every state take the action with the largest $Q^\ast$.

A pair $(s, a)$ followed by a reward and a next state gives the same kind of sample as before, and the Bellman equation for $Q^\ast$,

$$Q^\ast(s, a) = R(s, a) + \gamma\sum_j P(j \mid s, a)\,\max_{b}\,Q^\ast(j, b),$$

(the previous chapter's optimal equation, with $V^\ast(j) = \max_b Q^\ast(j, b)$ written out) says what its target should be. Taking one day of experience, $(s, a, r, s')$, the update of the estimate $Q$ with the target built from the best action in the next state is

$$Q(s, a) \leftarrow Q(s, a) + \alpha\,\big(\,r + \gamma\max_{b} Q(s', b) - Q(s, a)\,\big).$$

This is **Q-learning**. Its target does not mention the action that was taken in the next state, or even the policy that chose the action $a$: it estimates the value of acting greedily from $s'$ onward whatever was actually done, and under the step-size conditions of the previous section it converges to $Q^\ast$ as long as every pair keeps being tried (Watkins and Dayan proved it, and the two 1994 papers cited above give the general proof). Learning the values of one policy, the greedy one, from days generated by another, the one that explores, is called **off-policy** learning. The variant that learns the value of the policy that is actually generating the days replaces the maximum by the next action $a'$ the agent will take, the one it has already chosen: $Q(s, a) \leftarrow Q(s, a) + \alpha\,(r + \gamma Q(s', a') - Q(s, a))$. It is called **SARSA**, for the quintuple $(s, a, r, s', a')$ that one update uses, and it is **on-policy**: what it estimates is $Q^\pi$ for the policy $\pi$ that behaves, including its exploration.

The two differ only in one term of the target, and the difference is easy to state exactly when the behaviour is epsilon-greedy, that is, with a fixed probability $\epsilon_{\mathrm{ex}}$ it takes an action drawn uniformly from the legal ones and otherwise the greedy one (next section): the mean of SARSA's target, over the demand and over the behaviour's next action, is the right-hand side of the equation below, so a table that satisfies it no longer moves on average, and it is the fixed point of

$$Q_{\mathrm{ex}}(s, a) = R(s, a) + \gamma\sum_j P(j \mid s, a)\,\Big[(1 - \epsilon_{\mathrm{ex}})\max_b Q_{\mathrm{ex}}(j, b) + \epsilon_{\mathrm{ex}}\,\tfrac{1}{m_j}\sum_b Q_{\mathrm{ex}}(j, b)\Big],$$

where $m_j$ is the number of legal actions in $j$, the value of the best policy that has to explore with probability $\epsilon_{\mathrm{ex}}$; Q-learning's target does not depend on $\epsilon_{\mathrm{ex}}$ at all, so its fixed point is $Q^\ast$ whatever the rate is. Both can be checked against the shop.

```text
function greedy_action(q: list<real>, tie: real) -> int:
    best = max(q)
    a = 0
    while q[a] < best - tie:
        a = a + 1
    return a

function choose_action(Q: list<list<real>>, n_act: list<int>, s: int, t: int,
                       eps0: real, tau: real, floor: real, rng: Generator) -> int:
    eps = max(floor, eps0 / (1.0 + t / tau))
    u = rng.uniform()
    if u < eps:
        return rng.integer(n_act[s])
    return greedy_action(Q[s], 1e-9)

function td_control(n_act: list<int>, gamma: real, sarsa: bool, q0: real,
                    eps0: real, tau: real, floor: real, days: int, s0: int,
                    rng: Generator) -> list<list<real>>:
    require 0 <= gamma and gamma < 1
    Q = []
    count = []
    for s in 0 .. len(n_act):
        append [q0 for a in 0 .. n_act[s]] to Q
        append [0 for a in 0 .. n_act[s]] to count
    s = s0
    a = choose_action(Q, n_act, s, 0, eps0, tau, floor, rng)
    for t in 0 .. days:
        (r, next) = sample_day(s, a, rng)
        a2 = choose_action(Q, n_act, next, t + 1, eps0, tau, floor, rng)
        if sarsa:
            y = r + gamma * Q[next][a2]
        else:
            y = r + gamma * max(Q[next])
        count[s][a] = count[s][a] + 1
        Q[s][a] = Q[s][a] + (count[s][a] ^ (-0.7)) * (y - Q[s][a])
        s = next
        a = a2
    return Q
```

Every convention is in the routine. The behaviour is epsilon-greedy with $\epsilon_{\mathrm{ex}}(t) = \max(\text{floor},\ \epsilon_0/(1 + t/\tau))$ for the action of day $t$, which is greedy with probability $1 - \epsilon_{\mathrm{ex}}(t)$ and otherwise uniform over the legal actions (so the greedy action can also be drawn at random); one `uniform()` is drawn for every decision whether or not it is needed, and one `integer` only when exploring. The greedy action is the smallest one within $10^{-9}$ of the largest value, the tie rule of the previous chapter, so on an untouched table every action is tied and the smallest wins. The next action is chosen before the update of the current pair and the update is applied before the next day's demand is drawn. The step is $n^{-0.7}$ with $n$ the number of updates of that pair so far, current one included, so the first update of a pair uses $\alpha_1 = 1$. After a constant $\epsilon_{\mathrm{ex}} = 0.2$, $50\,000$ days per run, the mean of ten runs of each algorithm gives the following next to the two exact fixed points.

<!-- worked: value_learning.sarsa_and_q_learning -->
A constant exploration rate eps = 0.2 for every day (eps0 = floor = 0.2), Q starting at 0, 50000 days per run from s_0 = 2, 10 runs of each algorithm (run i of SARSA has the generator named value_learning.pairs.sarsa.run<i>, of Q-learning value_learning.pairs.q.run<i>, i = 0 .. 9; the first seeds are 0x48eb6ac2726882e4 and 0xb81eb7486a4f5337), draws and update order as in the table above, the step n^-0.7. Q* is the optimal lookahead of the previous chapter; Q_eps is the fixed point of Q(s, a) = R(s, a) + gamma sum_j P(j | s, a) [(1 - eps) max_b Q(j, b) + eps mean_b Q(j, b)], the value of the best policy that must explore with probability eps, iterated until the change is below 1e-12. The last two columns are the mean over the 10 runs of Q(s, a) after the last day (divisor 10).

| s | a | Q* | Q_eps | SARSA | Q-learning |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 6.075 | 4.856 | 4.75 | 6.08 |
| 0 | 1 | 6.750 | 5.514 | 5.44 | 6.77 |
| 0 | 2 | 6.668 | 5.395 | 5.29 | 6.65 |
| 0 | 3 | 5.168 | 3.919 | 3.85 | 5.24 |
| 1 | 0 | 9.250 | 8.014 | 7.94 | 9.25 |
| 1 | 1 | 8.668 | 7.395 | 7.29 | 8.70 |
| 1 | 2 | 7.168 | 5.919 | 5.82 | 7.24 |
| 2 | 0 | 11.168 | 9.895 | 9.87 | 11.17 |
| 2 | 1 | 9.168 | 7.919 | 8.03 | 9.21 |
| 3 | 0 | 11.668 | 10.419 | 10.25 | 11.65 |
<!-- /worked -->

Q-learning lands on $Q^\ast$: for an empty shelf $6.08$, $6.77$, $6.65$, $5.24$ against $6.075$, $6.750$, $6.668$, $5.168$, even though the days it learned from were drawn by a policy that picked a random order one day in five. SARSA does not: for the same row it has $4.75$, $5.44$, $5.29$, $3.85$, within $0.17$ of the exact $Q_{\mathrm{ex}}$ of $4.856$, $5.514$, $5.395$, $3.919$ in every row of the table, and between $1.1$ and $1.4$ below $Q^\ast$ (the exact $Q_{\mathrm{ex}}$ is $1.2$ to $1.3$ below it). That gap is not an estimation error. It is what exploring costs: a keeper who orders a random amount on one day in five really does earn about $1.2$ to $1.3$ less discounted value for each pair, and SARSA, which values the policy that is behaving, reports it. Q-learning reports what the keeper would earn after the exploring stopped. Which of the two is wanted depends on whether the agent is judged while it learns or after. Both rank the four orders on an empty shelf the same way, one unit first, two units second, none third, three units last, which is the order that matters for the decision.

## A greedy learner that never finds out

The previous section assumed that every pair keeps being tried. That is a property of the behaviour, not of the update, and a learner that acts greedily on its own estimates can arrange for it to fail. The opening problem of the previous chapter again: an empty shelf, a table of estimates all at $0$, and the rule "take the action with the largest estimate". All four actions are tied at $0$, the tie rule picks order $0$, the day pays $0$, and the update of $Q(0, 0)$ has the target $0 + 0.9 \times \max_b Q(0, b) = 0$ and leaves it at $0$. The estimate agrees with itself, and nothing in the data has ever contradicted it. The other three orders are never tried and never updated, so they stay at exactly $0$ as well, tied, and the smallest still wins.

<!-- worked: value_learning.exploration_lock_in -->
Q-learning on the inventory problem of the previous chapter, gamma = 0.9, from the shelf s_0 = 2, 20000 days per run and 10 runs; run i of variant k has the generator named value_learning.exploration.v<k>.run<i> (k = 0, 1, 2 in the order of the rows, i = 0 .. 9; the seeds of the first run of each are 0x893b0e5db0dd262d, 0x38c419c4b80e4d0e, 0xe2ff827a5c73daf9). Every day draws in this order: the demand uniform() of the day just played, then, for the action of the next day, one uniform() u and, only when u < eps_t, one integer(m) over the m legal actions; eps_t = max(0.05, 1 / (1 + t / 200)) for the action of day t in the second row and 0 in the other two (the uniform u is drawn all the same). Otherwise the action is the greedy one, the smallest within 1e-09 of the largest Q(s, .). The update of a day is Q(s, a) <- Q(s, a) + n^-0.7 (r + gamma max Q(s', .) - Q(s, a)) with n the number of updates of that pair so far including this one, applied after the next action has been chosen and before the next demand is drawn. The final greedy policy of a run is the greedy action of Q in every state; 'V(0) of the final policy' is its exact value on an empty shelf from the linear system (mean over the runs), 'days at s = 0 with a = 1' the mean number of updates of that pair, and the last four columns are the mean over the runs of Q(0, 0 .. 3) after the last day.

| variant | final greedy policies of the runs | V(0) of the final policy | days at s = 0, a = 1 | Q(0,0) | Q(0,1) | Q(0,2) | Q(0,3) |
|---|---|---:|---:|---:|---:|---:|---:|
| greedy, Q starts at 0 | (0, 0, 0, 0) x 9; (0, 0, 1, 0) x 1 | 0.000 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| epsilon-greedy, Q starts at 0 | (1, 0, 0, 0) x 10 | 6.750 | 12114.9 | 5.992 | 6.687 | 6.390 | 4.913 |
| greedy, Q starts at 15 | (1, 0, 0, 0) x 10 | 6.750 | 12029.9 | 6.686 | 6.791 | 6.545 | 5.138 |

The optimal values of the previous chapter are Q*(0, 0 .. 3) = 6.075, 6.750, 6.668, 5.168 and V*(0) = 6.750.
<!-- /worked -->

The first row is that learner. In nine of ten runs it ends with the greedy policy $(0, 0, 0, 0)$, the keeper of the opening who looks only at today, with a value of $0.000$ on an empty shelf where the optimum is $6.750$; in the tenth the policy also orders one unit on a shelf of two, which a negative reward there must have pushed below the start value of $0$, but the empty shelf is still locked. The mean of $Q(0, a)$ over the runs is $0.000$ for every order: no order has ever been ordered. The mean number of days on which one unit was ordered at the empty shelf is $0.0$. This is the case the previous chapter showed from the other side: there a tie at zero was a rule for the greedy policy of a known table; here it is a trap for a learner with no table, because the tied actions never get a chance to differ.

The second row is the standard repair, **epsilon-greedy** exploration: with probability $\epsilon_{\mathrm{ex}}$ take an action drawn uniformly at random, otherwise the greedy one. The rate cannot stay at $1$, which would never use what was learned, and cannot drop to $0$ at once; the schedule here is $\epsilon_{\mathrm{ex}}(t) = \max(0.05,\ 1/(1 + t/200))$ for the action of day $t$: purely random at the start, $0.5$ after $200$ days, $0.25$ after $600$, and at the floor of $0.05$ from day $3800$ on, where a twentieth of the days still explore. Every run finds the policy $(1, 0, 0, 0)$, with $V(0) = 6.750$, and has ordered one unit at the empty shelf on $12\,114.9$ of its $20\,000$ days on average, with estimated values of $5.992$, $6.687$, $6.390$ and $4.913$ for the four orders against $6.075$, $6.750$, $6.668$ and $5.168$ at the optimum.

The third row draws no random number after the start. Every value starts at $15$, above the largest value in the problem ($11.668$ for a full shelf), so every untried action looks better than anything that can be observed. The first update of a pair sets it to a reward plus $0.9$ times a value of at most $15$ (the first step is $\alpha_1 = 1$), which is below $15$, so a tried action drops below the untried ones and the greedy choice moves on to an action that has not been tried. The learner works through the actions of a state until their estimates have come down to what they are worth, and then the greedy rule is right: **optimistic initial values** turn the lack of evidence into a reason to explore. All ten runs find $(1, 0, 0, 0)$.

The start is a choice with consequences. Any start above $0$ breaks the lock of the first row. That row is stable because the target of the tried action, $0.9$ times the largest value of the state, is again the $0$ it started with; from a start above $0$ the same target is lower than the start, the tried action falls below the untried ones and the learner moves on. But the length of the exploring depends on the start. With the start values $5$, $11.7$, $15$ and $100$ the same ten runs end on the optimal $(1, 0, 0, 0)$ in $5$, $7$, $10$ and $4$ runs (Exercise 12); the others end on another policy, in most runs one that orders two units on an empty shelf (the table lists them all; at $11.7$ one run differs from the optimum only at a shelf of two), whose lookahead at the optimum is only $0.082$ below the best ($6.668$ against $6.750$), a gap that $20\,000$ days do not always resolve, and a start of $100$ is still coming down at the end of the run. The price of optimism is a bound on the values that the learner must know, and time spent coming down from it that grows with the bound.

<!-- worked: value_learning.optimistic_starts -->
The third variant of the lock-in table (greedy, no random action, Q starting at the stated value for every pair) with four starting values; every start uses the same 10 generators, run i the one named value_learning.exploration.v2.run<i> (i = 0 .. 9), so the rows differ only in the starting value, and everything else (days 20000, s_0 = 2, the draw order, the tie rule, the step n^-0.7) is as there. The row of 15 is the third row of that table. The last column is the mean over the runs of the largest Q(0, .) after the last day, to be read against V*(0) = 6.750.

| start value | final greedy policies of the runs | days at s = 0, a = 1 | largest Q(0, .) |
|---:|---|---:|---:|
| 5 | (1, 0, 0, 0) x 5; (2, 0, 0, 0) x 4; (2, 1, 0, 0) x 1 | 6992.6 | 6.34 |
| 11.7 | (1, 0, 0, 0) x 7; (1, 0, 1, 0) x 1; (2, 0, 0, 0) x 2 | 9794.4 | 6.69 |
| 15 | (1, 0, 0, 0) x 10 | 12029.9 | 6.79 |
| 100 | (1, 0, 0, 0) x 4; (2, 0, 0, 0) x 6 | 4226.8 | 8.85 |
<!-- /worked -->

These are two answers to one tension, the **exploration-exploitation** trade-off: every day spent on an action that looks worse than the best known is a day not earning the best known value, and every day spent on the best known is a day not learning whether a better one exists. Exploring costs the amount that Q-learning and SARSA disagree about in the previous section; not exploring costs, as the first row shows, the whole of the optimum. Neither extreme is right, and the schedules above are two of many ways to spend one on the other.

## Why the maximum of noisy estimates is too large

Q-learning's target takes a maximum over the estimates of the next state, and the estimates are noisy. A maximum is not a neutral operation on noisy numbers. If $\hat q_1, \dots, \hat q_k$ are unbiased estimates of values that are all equal to $\mu$, the expected maximum of the estimates is above $\mu$, because whichever estimate happened to be too high is the one that is picked. This is **maximisation bias**. For $k$ estimates that are Gaussian with the same mean and the spread $\sigma$ the expectation of the largest is $\mu + c_k\sigma$, where $c_k$ is the expected largest of $k$ standard Gaussians: $c_2 = 1/\sqrt{\pi} \approx 0.564$ (the maximum of two numbers is their mean plus half the absolute difference), and $c_8 = 1.4236$ by numerical integration.

A small problem makes it visible. State A has two actions, one of which ends the episode with reward $0$ and the other of which moves to state B, with reward $0$. State B has eight actions, each ending the episode with a Gaussian reward of mean $-0.1$ and standard deviation $1$. Every action at B is equally poor, worth $-0.1$ on average, so the move to B is worth $0.9 \times (-0.1) = -0.09$ and the right choice at A is the action that ends the episode, worth $0$. A learner whose estimates at B are means of $n$ samples sees a maximum that is not $-0.1$.

<!-- worked: value_learning.maximisation_bias -->
The small problem. State A has two actions, right (action 0: reward 0, the episode ends) and left (action 1: reward 0, the next state is B). State B has 8 actions, each ending the episode with a reward drawn from a Gaussian with mean -0.1 and standard deviation 1.0, the same for all of them. Terminal states have value 0 and gamma = 0.9, so the true Q(A, right) = 0 and Q(A, left) = 0.9 * -0.1 = -0.09: right is better. If the 8 values at B were each the mean of n independent rewards, they would be Gaussian with mean -0.1 and standard deviation 1/sqrt(n), and the expected largest of 8 of them is -0.1 + c / sqrt(n) with c = E[max of 8 standard Gaussians] = 1.4236 (the integral of 8 x phi(x) Phi(x)^7 by the trapezoid rule on [-10, 10] with step 0.001):

| n | expected largest estimate at B | value of every action at B |
|---|---:|---:|
| 1 | 1.324 | -0.100 |
| 4 | 0.612 | -0.100 |
| 25 | 0.185 | -0.100 |
| 100 | 0.042 | -0.100 |

Learning it: 1000 runs of 300 episodes, run i with the generator named value_learning.maxbias.run<i> (i = 0 .. 999; run 0 has seed 0x2c51f203f72a4978), the same generator name for both algorithms. All values start at 0, the step is a constant 0.1, the exploration rate a constant 0.1 (one uniform() u per decision and, if u < 0.1, one integer over the actions), the greedy action the smallest within 1e-09 of the best. An episode chooses at A, updates the pair (A, chosen) when the step ends, and if it went left chooses at B, draws one normal() and updates (B, chosen) with the reward alone. Q-learning's target at A is gamma max_b Q(B, b). Double Q-learning keeps two tables, chooses by their sum, and for every update draws one uniform(), after the transition and before the update, to pick the table k to change (k = 0 if it is below 0.5): its target is gamma Q_other(B, argmax_b Q_k(B, b)). The tables of a run start at 0; the share of runs that went left in the episodes of a block is a mean over the runs (divisor 1000) of the fraction of those episodes, in percent. A run that always chose the best action and explored at 0.1 would go left in 5% of the episodes.

| episodes | Q-learning goes left (%) | double Q-learning goes left (%) |
|---|---:|---:|
| 1 .. 10 | 6.8 | 5.8 |
| 11 .. 50 | 27.7 | 9.7 |
| 51 .. 100 | 53.8 | 12.0 |
| 101 .. 300 | 27.6 | 8.6 |

The estimate of Q(A, left) after the stated episode, mean over the runs (the true value is -0.09; for double Q-learning the mean of its two tables):

| after episode | Q-learning | double Q-learning |
|---|---:|---:|
| 10 | 0.001 | 0.000 |
| 50 | 0.037 | 0.000 |
| 100 | 0.047 | -0.005 |
| 300 | -0.024 | -0.030 |
<!-- /worked -->

The first table gives the size: with one sample per action at B the largest of eight estimates averages $1.324$ where every action is worth $-0.1$, with four samples $0.612$, with a hundred $0.042$, and it falls toward the truth only as $1/\sqrt{n}$. Q-learning's target at A is $0.9$ times that maximum, so $Q(A, \text{left})$ is inflated while the estimates at B are few, enough to make left look better than right. In the runs, the share of episodes that go left, which a learner that was right would hold at about $5\%$ (the exploration of $0.1$ spread over two actions), is $6.8\%$ in the first ten (ties at the start go to the better action, right, and left has to be tried by exploration first), then $27.7\%$ in episodes $11$ to $50$ and $53.8\%$ in $51$ to $100$, where the learner prefers the worse action about half the time, and $27.6\%$ over episodes $101$ to $300$ as the estimates at B sharpen. The estimate of $Q(A, \text{left})$, averaged over the runs, is above $0$ ($0.037$ after $50$ episodes and $0.047$ after $100$) where the truth is $-0.09$.

The cure is to stop using the same noise twice, once to pick the maximising action and once to value it. **Double Q-learning** keeps two independent estimates, $Q_0$ and $Q_1$, and for each update picks one of them at random to change; the action that maximises the next state is chosen by the table being updated and valued by the other one: $Q_k(s, a) \leftarrow Q_k(s, a) + \alpha\,(r + \gamma\,Q_{1-k}(s', \arg\max_b Q_k(s', b)) - Q_k(s, a))$. If $Q_k$ is lucky for some action the other table, whose noise is independent, has no reason to agree. The behaviour chooses by the sum of the two tables.

```text
function double_q_update(T: inout list<list<list<real>>>, s: int, a: int, r: real,
                         next: optional int, gamma: real, alpha: real,
                         rng: Generator) -> none:
    k = 0 if rng.uniform() < 0.5 else 1
    y = r
    if next is present:
        b = greedy_action(T[k][next], 1e-9)
        y = y + gamma * T[1 - k][next][b]
    T[k][s][a] = T[k][s][a] + alpha * (y - T[k][s][a])
```

One episode of the small problem, for both learners, is the routine below; `T[0]` is the single table of Q-learning and the sum of `T[0]` and `T[1]` chooses for double Q-learning. A state is `0` for A and `1` for B, the actions of A are `0` (right) and `1` (left), and `bias_update(T, double, s, a, r, next, gamma, alpha, rng)` is `double_q_update` when `double` is true (a `next` of `-1` standing for the end of the episode, `absent` there) and otherwise the update of `T[0]` alone with the target `r + gamma * max(T[0][next])`, or `r` at the end, drawing no number.

```text
function bias_choose(T: list<list<list<real>>>, double: bool, s: int,
                     eps: real, rng: Generator) -> int:
    q = T[0][s] + T[1][s] if double else T[0][s]
    u = rng.uniform()
    if u < eps:
        return rng.integer(len(q))
    return greedy_action(q, 1e-9)

function bias_episode(T: inout list<list<list<real>>>, double: bool, gamma: real,
                      alpha: real, eps: real, mu: real, sd: real,
                      rng: Generator) -> int:
    a = bias_choose(T, double, 0, eps, rng)
    if a == 0:
        bias_update(T, double, 0, 0, 0.0, -1, gamma, alpha, rng)
        return 0
    bias_update(T, double, 0, 1, 0.0, 1, gamma, alpha, rng)
    b = bias_choose(T, double, 1, eps, rng)
    r = mu + sd * rng.normal()
    bias_update(T, double, 1, b, r, -1, gamma, alpha, rng)
    return 1
```

The routine returns `1` when the episode went left, the quantity of the first table below, and the draws are in the order written: the choice at A, then the table draw of the update of A (double only), then the choice at B, the reward and the table draw of the update of B.

In the table the second column is the same runs with the same generator names. Double Q-learning goes left in $5.8\%$ of the first ten episodes, $9.7\%$ of episodes $11$ to $50$, $12.0\%$ of $51$ to $100$ and $8.6\%$ of $101$ to $300$, and its estimate of $Q(A, \text{left})$ is $0.000$ after $50$ episodes, $-0.005$ after $100$ and $-0.030$ after $300$, never clearly above $0$ and on its way to the truth of $-0.09$ (left is tried rarely, so it moves slowly). It is not unbiased: the estimate of the maximising action's value by an independent table is an underestimate in expectation, $E[Q_{1-k}(s', \arg\max_b Q_k(s', b))] \le \max_b E[Q_{1-k}(s', b)]$, and the bias is small and of the opposite sign. The cost is two tables and a coin, and an update that has half as many samples per table.

## Values that share weights

A table has one entry per state. The shop has four. A game board, an image or a conversation has more states than any memory could list, and then two things fail at once: there is no room for a table, and a state that has never been visited has no estimate at all, however many near-identical states have been. **Function approximation** replaces the table by a function with a fixed, small number of parameters, $\hat v(s; \mathbf{w})$, so that what is learned from one state moves the estimate of every state that resembles it. The simplest function is linear in a feature vector $\mathbf{x}(s)$ with $d$ numbers per state:

$$\hat v(s; \mathbf{w}) = \mathbf{w}^\top\mathbf{x}(s).$$

The table is a special case: with the one-hot features $\mathbf{x}(s) = \mathbf{e}_s$ the weight $w_s$ is the value of state $s$. This is the sense in which the chapter's three feature tables, printed below in full for the shop's four shelf levels, are one idea: the one-hot table (four numbers, four free values), a linear table $(1, s)$ (two numbers, so only a straight line of values over the shelf level) and a quadratic table $(1, s, s^2)$ (three numbers).

The update follows the opening's. The loss of one day is the squared gap between the target and the estimate, $\tfrac12\,(y - \hat v(s; \mathbf{w}))^2$ with $y = r + \gamma\,\hat v(s'; \mathbf{w})$. Its gradient with respect to $\mathbf{w}$, if $y$ is a constant, is $-(y - \hat v(s; \mathbf{w}))\,\mathbf{x}(s)$, and a step against it is

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\delta\,\mathbf{x}(s), \qquad \delta = r + \gamma\,\mathbf{w}^\top\mathbf{x}(s') - \mathbf{w}^\top\mathbf{x}(s).$$

This is the **semi-gradient** update: the gradient of the loss with the target treated as a constant, and "semi" because $y$ does depend on $\mathbf{w}$ through $\hat v(s'; \mathbf{w})$ and that dependence is deliberately not differentiated. The reason is what the target is for. Its job is to be a label, a stand-in for the true value of $s$; the Bellman equation says that the label is built from the value of $s'$, and the value of $s'$ is what the learner is trying to get right, not a quantity to be nudged to make this day's error smaller. Differentiating through $y$ would add the term $\gamma\,\mathbf{x}(s')$ to the direction of the step, pulling $\hat v(s')$ toward $\hat v(s)$ as well as the reverse, and the update would then descend a different objective, the squared Bellman error, whose minimiser is not $V^\pi$ when the demand is random (the gradient of a squared sampled error needs two independent samples of $s'$ to be unbiased). The semi-gradient step is not the gradient of any fixed loss, which is why its convergence is a separate theorem and not a consequence of SGD.

```text
function linear_td_update(w: inout array<real>[d], x: array<real>[d],
                          r: real, x_next: array<real>[d], gamma: real,
                          alpha: real) -> real:
    delta = r + gamma * dot(w, x_next) - dot(w, x)
    w = w + (alpha * delta) * x
    return delta
```

The `dot` accumulates in increasing index from $0.0$, both dot products use the weights as they stand before the update, and the update uses the step $\alpha$ of the day, here $\alpha_t = 0.02/(1 + t/1000)^{0.75}$ for the day $t$ counted from $0$, a decaying schedule of the family of the previous section. The experiment evaluates the policy "order three units on an empty shelf", whose exact values are in the previous chapter's table of four policies, because its chain is the only one of those policies that visits every shelf level and so makes the on-policy distribution cover all four states.

<!-- worked: value_learning.linear_td -->
The policy of the table is (3, 0, 0, 0), order three units on an empty shelf, whose exact values from the previous chapter are V = (1.268, 4.519, 6.807, 7.768) at gamma = 0.9. Its chain visits every shelf level, with the long-run share of days mu = (0.3009, 0.3246, 0.2456, 0.1289) (the solution of mu P = mu with the entries summing to 1). The value of a state is approximated by x(s) . w with a feature vector x(s) per state; the three feature tables are printed in full (one row per state):

| s | one-hot | linear | quadratic |
|---:|---|---|---|
| 0 | 1, 0, 0, 0 | 1, 0 | 1, 0, 0 |
| 1 | 0, 1, 0, 0 | 1, 1 | 1, 1, 1 |
| 2 | 0, 0, 1, 0 | 1, 2 | 1, 2, 4 |
| 3 | 0, 0, 0, 1 | 1, 3 | 1, 3, 9 |

For each table: the TD fixed point, the weights at which the average of the semi-gradient update over the days, sum_s mu(s) x(s) (r(s) + gamma x(s')^T w - x(s)^T w) with r(s) = R(s, pi(s)) and the average over s' taken with P, is zero (one linear system); the weights that minimise the mu-weighted squared error sum_s mu(s) (x(s)^T w - V(s))^2 (the normal equations); and the weights that semi-gradient TD(0) reaches. 'weighted error' is sum_s mu(s) (V_hat(s) - V(s))^2. Sampled runs: 3 runs of 60000 days from the shelf 0, run i of table f with the generator named value_learning.linear.<f>.run<i> with f one-hot, linear or quadratic (for example one-hot run 0 has seed 0x8d77fc02f0ec56de, linear run 0 0x986df8ff59d62547, quadratic run 0 0x34af634a8b7c0b92), one uniform() per day for the demand; weights start at 0 and change every day in the order of the days, delta = r + gamma x(s')^T w - x(s)^T w computed from the weights before the update, then w <- w + alpha_t delta x(s) with alpha_t = 0.02 / (1 + t / 1000)^0.75 for the day t counted from 0; the weights shown are those after the last day, averaged over the runs (divisor 3).

| features | weights from | w | V_hat(0 .. 3) | weighted error |
|---|---|---|---|---:|
| one-hot | TD fixed point | 1.268, 4.519, 6.807, 7.768 | 1.268, 4.519, 6.807, 7.768 | 0.000 |
| one-hot | best weighted fit | 1.268, 4.519, 6.807, 7.768 | 1.268, 4.519, 6.807, 7.768 | 0.000 |
| one-hot | sampled, mean of 3 runs | 1.19, 4.40, 6.66, 7.57 | 1.19, 4.40, 6.66, 7.57 |  |
| linear | TD fixed point | 1.764, 2.293 | 1.764, 4.057, 6.350, 8.643 | 0.293 |
| linear | best weighted fit | 1.722, 2.328 | 1.722, 4.050, 6.378, 8.706 | 0.292 |
| linear | sampled, mean of 3 runs | 1.76, 2.32 | 1.76, 4.08, 6.40, 8.72 |  |
| quadratic | TD fixed point | 1.253, 3.874, -0.564 | 1.253, 4.564, 6.748, 7.804 | 0.002 |
| quadratic | best weighted fit | 1.252, 3.875, -0.564 | 1.252, 4.564, 6.748, 7.806 | 0.002 |
| quadratic | sampled, mean of 3 runs | 1.48, 3.83, -0.49 | 1.48, 4.82, 7.16, 8.52 |  |

In the norm sqrt(sum_s mu(s) e(s)^2) of an error vector e, the TD fixed point is within 1 / sqrt(1 - gamma^2) times the best weighted fit (the bound of Tsitsiklis and Van Roy for the on-policy case): one-hot: 0.000 against a bound of 0.000; linear: 0.542 against a bound of 1.240; quadratic: 0.042 against a bound of 0.096.
<!-- /worked -->

Three things stand out. First, with the one-hot features the TD fixed point is the exact value, the best fit is the exact value and the error is $0.000$: a table is expressive enough. Second, with the two linear features the TD fixed point, $(1.764, 2.293)$, is not the best weighted fit, $(1.722, 2.328)$, and neither can fit the value, which bends where a line cannot: the weighted errors are $0.293$ and $0.292$. For the on-policy case the theory (Tsitsiklis and Van Roy) bounds the fixed point's error by $1/\sqrt{1 - \gamma^2}$ times that of the best fit in the norm weighted by how often each state is visited, and here the error of the fixed point is $0.542$ against a bound of $1.240$. The best fit is the best a line can do; TD(0) gets within a small factor of it without ever seeing $V$. Third, the sampled weights reach the neighbourhood of their fixed points but not all the way: $(1.76, 2.32)$ for the linear features, within $0.03$ of $(1.764, 2.293)$, $(1.19, 4.40, 6.66, 7.57)$ for the one-hot features against $(1.268, 4.519, 6.807, 7.768)$, within $0.2$, and for the quadratic features $(1.48, 3.83, -0.49)$ against $(1.253, 3.874, -0.564)$, which is still moving after $60\,000$ days. The quadratic table has features up to $9$ in size, so one update of size $\alpha\,\delta$ moves the estimate of a full shelf by $\alpha\,\delta\,\lVert\mathbf{x}\rVert^2 = 91\,\alpha\,\delta$; the schedule starts at $0.02$ and not $0.1$ for that reason (at $0.1$ that move is $9.1\,\delta$, more than the gap it is meant to close), and the features of very different scales leave a poorly conditioned problem for the step to cross, as in Volume III.

## Three ingredients that together diverge

Every tool of this chapter so far is safe alone. Function approximation with Monte Carlo targets is regression on returns. Bootstrapping in a table converges, by the contraction argument. Off-policy learning with a table also converges, which is Q-learning. The combination of all three, function approximation, bootstrapping and off-policy data (data that do not come from the distribution of the policy being evaluated), is not safe, and a two-state example shows how little is needed. There are two states, $1$ and $2$, with one weight $w$ and the features $x(1) = 1$ and $x(2) = 2$, so that $\hat v(1) = w$ and $\hat v(2) = 2w$. Every reward is $0$ and from state $2$ the episode ends, so every true value is $0$ and $w = 0$ is exactly right. The data contain only the transition from $1$ to $2$.

On that transition, $\delta = 0 + \gamma \cdot 2w - w = (2\gamma - 1)\,w$, and the update is $w \leftarrow w\,(1 + \alpha\,(2\gamma - 1))$. At $\gamma = 0.9$ and $\alpha = 0.1$ the factor is $1.08$. The weight grows geometrically, away from its exact value, whatever the starting point. The estimate of state $2$ is twice that of state $1$, so the target is $1.8$ times the weight: each update raises $w$ and with it the target, which rises $1.8$ times as fast as the estimate it is meant to correct, and the weight only chases it.

<!-- worked: value_learning.divergence -->
Two states with one weight w and the features x(1) = 1 and x(2) = 2, so V_hat(1) = w and V_hat(2) = 2w. Every reward is 0, so the true values are 0 and w = 0 is exact. The transition is 1 -> 2, and from 2 the episode ends (terminal value 0). gamma = 0.9, alpha = 0.1, w = 1 before the first update. 'off' updates only the transition 1 -> 2, again and again: delta = 0 + gamma * 2w - w and w <- w + alpha delta * 1. 'frozen' does the same with the target computed from a copy of w that is replaced by w after every 10 updates (the copy starts at 1). 'on' updates the whole episode, the update of 1 -> 2 followed by the update of 2 -> end (delta = 0 - 2w, w <- w + alpha delta * 2), one update per transition. The table is w after k updates (k counts single updates, in the order stated). The arithmetic is exact to rounding and the growth is a plain geometric sequence, so four significant digits are printed.

| updates k | off-policy, w | frozen target (copy every 10) | on-policy, w |
|---|---:|---:|---:|
| 0 | 1 | 1 | 1 |
| 10 | 2.159 | 1.521 | 0.1143 |
| 20 | 4.661 | 2.314 | 0.01305 |
| 50 | 46.9 | 8.142 | 1.947e-05 |
| 100 | 2.200e+03 | 66.29 | 3.791e-10 |
| 200 | 4.839e+06 | 4.395e+03 | 1.437e-19 |

Per update, 'off' multiplies w by 1 + alpha (2 gamma - 1) = 1.08; per round of 10 updates, 'frozen' multiplies it by 1.5211; per episode (two updates), 'on' multiplies it by 0.648.
<!-- /worked -->

![Three curves of the weight against the number of updates, 0 to 100, with the weight on a logarithmic axis from 1e-10 to 1e4. The off-policy weight rises as a straight line on this scale to about 2200 at 100 updates. The weight with a target that is refreshed every ten updates rises more slowly, to about 66. The on-policy weight falls on the same scale to about 4e-10.](img/weights-diverge.svg)

The third column is the same two states, the same weight, the same step, with one change: the episode is also followed through its second transition, so that the update of $2 \to \text{end}$ (target $0$, feature $2$) is applied as well. Then the pair of updates multiplies $w$ by $1.08 \times (1 - 4 \times 0.1) = 0.648$ and the weight goes to the true value, $0$. What failed in the first column was not the function, the bootstrap or the step alone but the data: the learner was never shown the transition out of state $2$, which would have told it that $\hat v(2) = 2w$ is wrong, and so nothing corrected the extrapolation that the shared weight forced on state $2$. The distribution of the updates was not that of the policy, which is the meaning of off-policy here.

The remedies that the deep value-learning methods adopted attack the instability with two devices. A **target network** is a copy of the weights that is used inside the target and refreshed only every $K$ updates, so that for $K$ updates the learner regresses toward a fixed label. A replay of stored transitions samples updates from a buffer of past days instead of the latest one, which breaks the correlation between successive updates and lets each day be used several times (the continual-learning chapter uses the same idea for a different purpose). The second column of the table shows what the first device does here, and it is a sobering result. With a copy refreshed every $10$ updates the weight is still multiplied by $1.5211$ per round, instead of $2.159$ per $10$ updates: slower by a factor of $1.4$ per round, and still growing without bound, $66.29$ after $100$ updates and $4.4 \times 10^3$ after $200$. A frozen label stops the target from moving during a round, which is what supervised regression needs; it does not supply the missing transition. Replay of these same data does not either, since the buffer holds only the transition from $1$ to $2$. The remedies make a learner behave more like regression on fixed labels, which helps when the instability is the moving target, and they do nothing for a data distribution that never tests the extrapolation.

What the triad leaves is a real limit. It is why a value learner on a large network needs care in how data are chosen, how targets are held and how fast the weights are allowed to move, and why the next chapter starts from another idea: to stop estimating what every action is worth, with a maximum and a bootstrap inside every target, and to change the probabilities of the policy directly in the direction of the returns that followed each action.

## What an update costs

Counting operations makes the comparisons of the chapter concrete. The count is per update, not per second: one operation is a multiply, an add, a subtract or a divide, a power for the schedule is one, and a comparison is not counted.

<!-- worked: value_learning.work_of_updates -->
One operation is a multiply, an add, a subtract or a divide; a power and a comparison count as above (a power for the schedule counts one operation, a comparison none). Per update:

| update | count | FLOPs |
|---|---|---:|
| tabular TD(0) | target r + gamma V(s'): 1 multiply, 1 add; delta: 1 subtract; alpha_n: 1; alpha * delta and the add: 2 | 6 |
| tabular SARSA | the same, Q(s', a') read in place of V(s') | 6 |
| tabular Q-learning | the same, with the maximum over the m actions (comparisons are not counted) | 6 |
| linear TD(0), d = 2 (linear table) | two dot products of 2d - 1, target 2, delta 1, alpha_t = 0.02 / (1 + t / 1000)^0.75 4 (divide, add, power, divide), alpha * delta 1, d multiplies and d adds for the weights | 18 |
| linear TD(0), d = 3 (quadratic table) | the same with d = 3 | 24 |
| linear TD(0), d = 4 (one-hot table) | the same with d = 4 (against 6 for a table lookup, whose schedule n^-0.7 is one power) | 30 |

Per episode of 100 days Monte Carlo spends 2 operations a day on the backward return G <- r + gamma G (200) and at most 3 per state on the update (G - V)/n added to V (12); TD(0) spends 6 a day (600). The first episode after which the mean error of the table of learning curves is at most 1.0 (the mean over the 10 runs of the root-mean-square error, as there), with the days and the operations of the updates that it took:

| method | episodes | days of experience | FLOPs of updates |
|---|---:|---:|---:|
| first-visit Monte Carlo | 42 | 4,200 | 8,904 (at most) |
| TD(0), alpha_n = n^-0.7 | 45 | 4,500 | 27,000 |
<!-- /worked -->

A tabular update is $6$ operations whatever the number of states: the cost is in the data, not the arithmetic. A linear update costs $6d + 6$, which for a one-hot table of four features is $30$ against the $6$ of a table lookup, the price of treating a table as a dense vector. On the shop the two estimators need about the same amount of experience to reach a mean error of $1.0$, $42$ episodes for Monte Carlo and $45$ for TD with $n^{-0.7}$, but Monte Carlo spends at most $8\,904$ operations on updates over those episodes and TD spends $27\,000$: TD updates every day, Monte Carlo only the first visit of each state. Counted in operations the tabular estimators are both a tiny cost beside the simulation that produces the days, and what separates them is the number of days they need and what each does with a day it has seen.

## Where brain keeps a value and where a policy learns nothing

Two of brain's crates contain the chapter's ideas in forms the next chapters will use. `crates/decide/src/value.rs` holds a critic, a small network $V(s)$ over a fixed feature vector, fitted to observed returns by squared error with an L2 penalty: Monte Carlo targets and function approximation, the pair of the sections on averaging and on shared weights. The same file has the estimator of Chapter 3, generalized advantage estimation, and its two unit tests pin the endpoints that this chapter has just worked out: with its parameter $\lambda = 0$ the advantage is the one-step TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, and with $\lambda = 1$ it is the discounted return minus the value. Between them the parameter trades bias for variance, the contrast of the table of bias and spread above. A third test makes the point about episodes made earlier in this chapter: an episode cut off by a step limit must bootstrap from the estimate of the last state instead of being told that its future was worth zero. Commit `6664f316d` (September 2026), which taught a control policy by reinforcement, added this value function and the estimator after a baseline that was one number for the whole batch proved too weak, and pinned them with those two identities and the truncation test. The crate's header says plainly why: a baseline that is one number for the whole batch cannot tell a winning step from a losing one in the same episode.

The exploration problem has a counterpart at the scale of language models. In `crates/rl/src/objective/advantage.rs` a group of two or more completions whose rewards are all equal teaches nothing and gets the advantage $0.0$ for every member, and `crates/rl/examples/autonomous_exploration.rs` records the consequence in its header: a group whose members all earn the same reward is dropped, so a policy at chance trains on a fraction of what it samples. That is the greedy row of the exploration table again, with the policy's own sampling in the place of $\epsilon_{\mathrm{ex}}$: a learner whose experience is all alike cannot tell which action was better. Chapter 3 is where the policy-gradient estimator that uses those advantages is derived.

## Check your understanding

1. Find the error: "TD(0) is an unbiased estimator of the value, because the Bellman equation says its target has mean $V^\pi(s)$."
2. At $\gamma = 0.9$ the off-policy two-state example multiplies the weight by $1 + \alpha(2\gamma - 1)$ per update. If the feature of state $2$ is $c$ instead of $2$, so that $\hat v(2) = c\,w$, for which values of $c$ does the weight grow, and what does the answer say about the role of the features?
3. SARSA and Q-learning are both run with the constant exploration rate $\epsilon_{\mathrm{ex}} = 0.2$ and both end with the same greedy policy. Their tables differ by about $1.3$ in every row. Which is closer to what the shop earns while it learns, and which to what it earns once exploration is switched off?
4. In the lock-in table the third row starts every value at $15$, and every run finds the best policy. Predict what happens if every value starts at $5$ instead: does the lock of the first row return, and what would you look at in the table to see how well it explores?
5. Why does the step $1/n$, which is optimal for a mean of independent returns, give a slow learner when used for TD(0)? What does a step of $n^{-0.7}$ change?
6. The semi-gradient update is not the gradient of any fixed loss. Name one consequence for how the book can claim its convergence, and one thing it does not lose.
7. Double Q-learning picks the maximising action with one table and values it with the other. Can it underestimate? Show with two actions whose true values are equal.

## Summary

Values can be learned from the days a shop writes down. Monte Carlo evaluation averages the returns that followed each visit; it is unbiased and its spread is that of a whole return, $2.891$ from an empty shelf where one day's spread is $1.260$. Temporal-difference learning replaces the rest of the return by the current estimate of the next state, an update that samples the Bellman equation with a bootstrapped target: lower variance and a bias that is the error of its own estimates, shrunk by $\gamma$. On the shop the bias costs it at first and pays off later: the root mean square error of its estimate of $V(0)$ against Monte Carlo's is $1.275$ to $0.860$ at $10$ episodes and $0.185$ to $0.271$ at $100$. The step size decides what the estimate keeps: the conditions $\sum\alpha = \infty$ and $\sum\alpha^2 < \infty$ give convergence, $1/n$ does so slowly in TD and a constant step does not converge at all but forgets.

Learning values of actions, Q-learning bootstraps from the best next action and learns $Q^\ast$ off-policy while SARSA learns the value of the policy that behaves, the two differing by exactly the cost of exploring. A greedy learner on a table of zeros never leaves the first action it tried, which is the myopic keeper of the previous chapter again; epsilon-greedy schedules and optimistic starting values are two ways out. The maximum over noisy estimates is biased upward, and a second independent estimate removes most of it. With shared weights the update becomes a semi-gradient, tabular learning is the one-hot special case, and the TD fixed point stays within a bounded factor of the best fit; but function approximation, bootstrapping and off-policy data together can send the weights to infinity, as a two-state example shows in a few lines, and the replay and target network built to tame it slow the growth in that example without stopping it.

All of this estimates what actions are worth and acts by taking a maximum. The next chapter goes the other way and adjusts the probabilities of the policy itself, with the returns of this chapter as the signal and the critic of the previous section as a baseline.

## References

- H. Robbins and S. Monro, "A stochastic approximation method", 1951.
- R. S. Sutton, "Learning to predict by the methods of temporal differences", 1988.
- C. J. C. H. Watkins and P. Dayan, "Q-learning", 1992.
- G. A. Rummery and M. Niranjan, "On-line Q-learning using connectionist systems", 1994.
- T. Jaakkola, M. I. Jordan and S. P. Singh, "On the convergence of stochastic iterative dynamic programming algorithms", 1994.
- J. N. Tsitsiklis, "Asynchronous stochastic approximation and Q-learning", 1994.
- L. Baird, "Residual algorithms: reinforcement learning with function approximation", 1995.
- J. N. Tsitsiklis and B. Van Roy, "An analysis of temporal-difference learning with function approximation", 1997.
- H. van Hasselt, "Double Q-learning", 2010.
- V. Mnih and others, "Human-level control through deep reinforcement learning", 2015.
- R. S. Sutton and A. G. Barto, "Reinforcement Learning: An Introduction", second edition, 2018.

## Exercises

1. **[Calculate]** The estimates are $V(2) = 0.5$ and $V(1) = 2.0$, the day is $s = 2$, $r = 3$, $s' = 1$, $\gamma = 0.9$ and $\alpha = 0.2$. Give the target, the TD error and the new $V(2)$.
2. **[Calculate]** An episode visits the states $0, 1, 0$ with the rewards $1, 0, 2$ and ends after the third day, $\gamma = 0.5$. Give the three returns, the first-visit return of state $0$ and the every-visit mean of its returns.
3. **[Calculate]** For which exponents $p$ does the schedule $\alpha_n = n^{-p}$ satisfy both step-size conditions? Give the limit of $\sum_n \alpha_n^2$ for $p = 1$ and show why it fails for $p = 1/2$.
4. **[Calculate]** A running mean with the constant step $\alpha$ of independent targets of standard deviation $\sigma$ has a stationary variance $\alpha\,\sigma^2/(2 - \alpha)$. Derive it from $V_n = (1 - \alpha)V_{n-1} + \alpha y_n$, and evaluate the standard deviation for $\alpha = 0.1$ and $\sigma = 1.260$.
5. **[Calculate]** A table has $Q(s', 0) = 4$, $Q(s', 1) = 6$ and $Q(s', 2) = 5$. A day gives $r = 1$ and the next action chosen by the behaviour is $a' = 2$. With $\gamma = 0.9$ give the SARSA target and the Q-learning target, and say which action the greedy policy would take.
6. **[Calculate]** Two actions have estimates that are independent Gaussians with mean $0$ and standard deviation $1$, and both have the true value $0$. Show that the expected maximum of the estimates is $1/\sqrt{\pi}$, and give the expected value of Q-learning's target for $\gamma = 0.9$ and $r = 0$.
7. **[Implement]** Reproduce `value_learning.ledger_updates`, `value_learning.target_spread` and `value_learning.step_size_sums`, including the exact second-moment system for the spread of the return.
8. **[Implement]** Reproduce `value_learning.learning_curves` and `value_learning.bias_and_spread`, using the generator names printed for the seeds, the start state and the demand draws in the order stated, and the first-visit and TD(0) update rules with the step counts as defined.
9. **[Implement]** Reproduce `value_learning.exploration_lock_in`, `value_learning.sarsa_and_q_learning` and `value_learning.maximisation_bias`, with the draw order of the decisions and of the update, the tie rule and the exact fixed point $Q_{\mathrm{ex}}$.
10. **[Implement]** Reproduce `value_learning.linear_td`, `value_learning.divergence` and `value_learning.work_of_updates`, including the TD fixed point from the linear system $\mathbf{A}\mathbf{w} = \mathbf{b}$ with $\mathbf{A} = \sum_s \mu(s)\,\mathbf{x}(s)(\mathbf{x}(s) - \gamma\sum_j P(j \mid s, \pi(s))\,\mathbf{x}(j))^\top$ and $\mathbf{b} = \sum_s \mu(s)\,\mathbf{x}(s)\,R(s, \pi(s))$.
11. **[Implement]** Reproduce `value_learning.every_visit_monte_carlo`, with the generator names for the seeds, the every-visit returns of days $0$ to $49$ of each episode and the first-visit return as in the table of learning curves. Then say, from the second table, for which states every visit gains and for which it does not, and why.
12. **[Investigate]** Reproduce `value_learning.optimistic_starts`, using the third variant's generator names for all four starts. Which start finds the optimal policy in the most runs, why does the largest start do worst, and what in the last column tells you so?

## Key terms

- **temporal-difference learning**: learning the value of a state from the difference between two successive estimates of it, by moving the estimate a fraction of the way toward the reward of one day plus the discounted estimate of the next state, without a model and without waiting for the end of the episode. <!-- concept: temporal-difference-learning -->
- **Q-learning**: learning the values of state-action pairs by moving each toward the day's reward plus the discounted largest value of the next state, whatever action was taken there, so that it estimates the best policy from days generated by another. <!-- concept: q-learning -->
- **exploration-exploitation**: the trade-off between taking the action that looks best, which earns the most now, and trying another, which may teach something better, since an agent that never explores can lock onto a worse action for ever. <!-- concept: exploration-exploitation -->
- **function approximation in reinforcement learning**: replacing the table of values by a function of a few parameters, so that learning from one state moves the estimates of similar states, at the price that bootstrapping, off-policy data and the approximation together can make the weights diverge. <!-- concept: function-approximation-in-rl -->

## Answers

**Recall questions.**

1. $V^\pi(s) = R(s, \pi(s)) + \gamma\sum_j P(j \mid s, \pi(s))\,V^\pi(j)$ for every state: the value of a state is what the day pays on average plus the discounted value of where the day leads, averaged over where it may lead.
2. $V^\pi(s)$ is the expected return, the mean over trajectories of the discounted sum of rewards that follow a visit to $s$ under $\pi$. For the policy $(1, 0, 0, 0)$ at $\gamma = 0.9$, $V^\pi(0) = 6.750$.
3. $w \leftarrow w - \eta\,\ell'(w) = w + \eta\,(y - w)$, a move of the fraction $\eta$ of the way from $w$ to $y$.
4. Each step is $w_n = w_{n-1} + \tfrac1n\,(y_n - w_{n-1})$, which is $w_n = ((n - 1)\,w_{n-1} + y_n)/n$, so $w_1 = y_1$ whatever $w_0$ was and $w_n = (y_1 + \dots + y_n)/n$ is the mean of the first $n$ examples.

**Check your understanding.**

1. The target has mean $V^\pi(s)$ only if the estimate $W(s')$ inside it is exactly $V^\pi(s')$. Otherwise the mean of the target is $V^\pi(s) + \gamma\,E[W(s') - V^\pi(s')]$, so the update is biased by $\gamma$ times the error of the estimate of the next state, which is what the table of target spreads and the bias column of the table of repeats show.
2. The step multiplies $w$ by $1 + \alpha(c\gamma - 1)$, which exceeds $1$ when $c\gamma > 1$, that is for $c > 1/0.9 = 1.11$ at $\gamma = 0.9$. For $c = 1$ (the same feature for both states, so no extrapolation) the weight shrinks, and for $c = 2$ it grows. The size of the feature of $s'$ relative to $s$ is the extrapolation the weight forces on the state that the data never correct.
3. SARSA's table is closer to what the shop earns while it learns, since it values the policy that explores; Q-learning's is closer to what it earns once the exploration stops, since it values the greedy policy.
4. The lock does not return. A start of $5$ is above $0$, so at the empty shelf an action that has been tried (order $0$, which pays nothing and stays there) falls to a target of $0.9 \times 5 = 4.5$ while the untried ones are still at $5$, and the greedy choice moves on to them: the learner explores. What changes is how long the optimism lasts, which decides whether the small gap between one and two units on an empty shelf ($0.082$ in lookahead) is resolved in $20\,000$ days. The final policies of the ten runs and the days at the empty shelf with one unit ordered show it: the table of starting values shows a start of $5$ ending on the optimal policy in $5$ runs, the others on a policy that orders two units.
5. With $1/n$ every target has the same weight, including the targets built early from estimates of $V(s')$ that were $0$, so the early bias is averaged out at the speed of the harmonic series, and $200$ episodes still leave an error of $3.251$. The step $n^{-0.7}$ gives later targets more weight than earlier ones while still satisfying both conditions.
6. Convergence cannot be inherited from SGD on a loss, so it has to be proved for the update's expected direction: for linear features on-policy it converges to a unique fixed point, the TD fixed point, which is not the minimiser of the squared error. What it keeps is the form of an ordinary step against the gradient for a given label, and with a target held fixed for a while (a target network) the regression on the fixed label is a true SGD problem.
7. Yes. Suppose two actions have the same true value and the two tables are independent and unbiased. The action that $Q_k$ picks carries no information about the noise of $Q_{1-k}$, so the expected value of $Q_{1-k}$ at that action is the common true value: no overestimate. When the true values differ, $Q_k$ sometimes picks the worse action, and $Q_{1-k}$ then values it at its lower true value on average, so the expected target is below the true maximum: an underestimate, usually small.

**Exercises.**

1. Target $= 3 + 0.9 \times 2.0 = 4.8$, $\delta = 4.8 - 0.5 = 4.3$, new $V(2) = 0.5 + 0.2 \times 4.3 = 1.36$.
2. $G_2 = 2$, $G_1 = 0 + 0.5 \times 2 = 1$, $G_0 = 1 + 0.5 \times 1 = 1.5$. The first-visit return of state $0$ is $G_0 = 1.5$; the every-visit mean of its returns ($G_0$ and $G_2$) is $(1.5 + 2)/2 = 1.75$.
3. For $p \in (1/2, 1]$. At $p = 1$, $\sum 1/n^2 = \pi^2/6 \approx 1.645$. At $p = 1/2$, $\sum \alpha_n^2 = \sum 1/n$, the harmonic series, which diverges (like $\ln n$), so the second condition fails; the first sum $\sum n^{-p}$ diverges for every $p \le 1$.
4. In stationarity $\mathrm{Var}(V_n) = (1 - \alpha)^2\,\mathrm{Var}(V_{n-1}) + \alpha^2\sigma^2$ with $\mathrm{Var}(V_n) = \mathrm{Var}(V_{n-1})$, so $\mathrm{Var} = \alpha^2\sigma^2/(1 - (1 - \alpha)^2) = \alpha^2\sigma^2/(\alpha(2 - \alpha)) = \alpha\sigma^2/(2 - \alpha)$. For $\alpha = 0.1$ and $\sigma = 1.260$ the standard deviation is $1.260 \times \sqrt{0.1/1.9} = 0.289$.
5. SARSA: $1 + 0.9 \times Q(s', 2) = 1 + 0.9 \times 5 = 5.5$. Q-learning: $1 + 0.9 \times \max = 1 + 0.9 \times 6 = 6.4$. The greedy policy takes action $1$ in $s'$, which has the largest value.
6. $\max(a, b) = (a + b)/2 + |a - b|/2$, and $a - b$ is Gaussian with mean $0$ and variance $2$, so $E|a - b| = \sqrt{2/\pi}\,\sqrt{2} = 2/\sqrt{\pi}$. Then $E[\max] = 0 + 1/\sqrt{\pi} \approx 0.5642$. The expected target is $0 + 0.9 \times 0.5642 = 0.508$, though the true value of the next state is $0$.
7. to 10. The checks are the tables of the blocks to the printed digits (the ten-day ledger in 7 has $V = (-1.069, 0.937, -0.355, 0.000)$ after the ten updates). In 8, the Monte Carlo column of `learning_curves` at $1$ episode is $7.731$ and the TD $n^{-0.7}$ column at $200$ episodes is $0.307$; in 10, the weight of the off-policy example after $100$ updates is $2200$ (the value $1.08^{100} = 2199.8$).
11. The checks are the first table at $5$ episodes (first visit $3.372$, every visit $3.223$, EV - FV $-0.149$ with se $0.031$) and the second at a shelf of three ($50.0$ and $71.9$ visits, sd $0.683$ and $0.752$, equivalent n $44$ and $37$). Every visit gains for the empty shelf and the shelf of one, which are visited many times per episode by days far enough apart to be nearly independent (equivalent n $572$ and $557$ against $219$ and $186$), and does not gain for a shelf of two or three, whose extra visits are the consecutive days of one stay on the shelf: the estimate then weighs an episode by the length of its stay, which is as good as the equally weighted episodes of the 'weight n' column ($61$ and $39$, against the equivalent n of $60$ and $37$). The error over the four states is dominated by those two states, so the gain at $5$ and $10$ episodes ($-0.149$, $-0.111$) is gone by $20$ ($-0.038$, se $0.030$) and the differences at $100$ and $200$ ($+0.006$, $+0.011$, se $0.014$ and $0.010$) are noise. Every visit is not unbiased in general; its largest measured bias is $+0.070$, for a shelf of two.
12. Every one of the four starts avoids the lock of the first row. The rows of the table: start $5$, $5$ runs on $(1, 0, 0, 0)$ (the others $(2, 0, 0, 0)$ four times and $(2, 1, 0, 0)$ once), $6992.6$ days at the empty shelf with one unit ordered; start $11.7$, $7$ runs ($(2, 0, 0, 0)$ twice, $(1, 0, 1, 0)$ once), $9794.4$ days; start $15$, $10$ runs, $12029.9$ days; start $100$, $4$ runs (the other six on $(2, 0, 0, 0)$), $4226.8$ days. A start only helps if it keeps the untried actions above the tried ones for long enough, and one that is far above every value takes so long to come down that the run ends before the actions are told apart: the last column, the largest estimate for the empty shelf, is $8.85$ for the start $100$ against $6.34$, $6.69$ and $6.79$ for the others, with the optimum at $6.750$.
