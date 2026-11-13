<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Markov Decision Processes

*How do we describe an agent that acts, sees consequences and wants long-term reward?*

A shelf holds at most three units of one product. Every morning the keeper sees how many are left and orders some more, which arrive at once; during the day a random number of customers each ask for one unit, up to two, and what is not sold stays on the shelf overnight. Selling a unit earns $4$, buying one costs $2$, placing an order at all costs a further $0.5$, and every unit left on the shelf in the evening costs $1$. The prices will be printed in full in a moment. What matters first is what a keeper who judges each order by what it earns today will do on an empty shelf.

Ordering one unit onto an empty shelf earns, on average, exactly nothing. The unit sells with probability $0.7$ and brings $4$ when it does, which is $2.8$; the unit that stays on the shelf with probability $0.3$ costs $1$, which is $0.3$ on average; the order itself cost $2$ and the fee $0.5$; and $2.8 - 0.3 - 2 - 0.5 = 0$. Ordering two units earns $-1.5$ and ordering three $-4.5$. So the keeper who looks only at today orders nothing, the shelf stays empty the next morning and every morning after it, and the shop earns nothing for ever. A keeper who counts what tomorrow's shelf is worth places the same one-unit order that earned nothing, and the shop, valued as the chapter will define at a discount of $0.9$, is worth $6.750$ from an empty shelf. One decision, tied at zero today, separates a dead shop from a living one.

The two-button game of Volume II hid nothing of this kind. Its buttons never changed, so the best button was the best button on every press, and a choice affected only the one reward it earned. Here a choice also changes the situation in which the next choice is made, and the reward for it may come days later. Describing that exactly, saying what it is worth to be in a situation and then computing the best thing to do in every one of them, is the subject of the chapter, and it is the frame in which every method of the rest of the volume is stated.

Four ideas from earlier volumes are needed first. Answer them from memory; the answers are at the end of the chapter.

1. (Volume II, Chapter 1, "The Learning Problem", reinforcement learning, overview.) In the two-button game, what did the agent's choice change, and what did it leave unchanged?
2. (Volume I, Chapter 10, "Probability: Reasoning Under Uncertainty", conditional probability.) A day's demand is $0$, $1$ or $2$ with probabilities $0.3$, $0.4$ and $0.3$. Given that at least one customer came, what is the probability that two did?
3. (Volume I, Chapter 10, expectation and its linearity.) With one unit on the shelf, the units sold are $\min(D, 1)$. What is its expectation, and what is the expectation of $4 \min(D, 1) - 2$?
4. (Volume I, Chapter 4, "Matrices as Linear Maps", solving a linear system.) Solve $x - 0.5y = 2$ and $-0.5x + y = 1$.

## A shop as a table of probabilities

Everything the keeper's world does can be written in two tables. Call the number of units on the shelf in the morning the **state** $s \in \{0, 1, 2, 3\}$, and the number ordered the **action** $a$, which may be anything from $0$ up to the room left, $3 - s$ (so ten pairs of state and action exist in all). The stock on the shelf during the day is $s + a$. The demand $D$ is $0$, $1$ or $2$ with probabilities $0.3$, $0.4$ and $0.3$, the same every day and independent of everything before; the units sold are $\min(D, s + a)$ and what is left over, $s + a$ minus the units sold, is tomorrow's state. The reward of the day is $4$ per unit sold, minus $2$ per unit ordered, minus $0.5$ if $a > 0$, minus $1$ per unit left over.

A row of probabilities and a reward per pair describe the future completely: where tomorrow's state will be, and what today paid. The first is a probability for each next state $j$, written $P(j \mid s, a)$, the conditional probability of tomorrow's state given today's state and action; the second is the mean reward $R(s, a)$ of the day over the demand, summed over $d = 0, 1, 2$ in that order.

<!-- worked: mdps.inventory_mdp -->
A shelf holds at most 3 units. In the morning there are s units on it (the state); the keeper orders a units (the action, a = 0 .. 3 - s, so the shelf never overflows) and they arrive at once. During the day D units are asked for, with P(D = 0) = 0.3, P(D = 1) = 0.4, P(D = 2) = 0.3, the same every day and independent of everything before. The units sold are min(D, s + a) and the stock left over, which is tomorrow's state, is s + a minus the units sold. The reward of the day is 4 per unit sold, minus 2 per unit ordered, minus 0.5 if an order is placed at all (a > 0), minus 1 per unit left on the shelf in the evening. R(s, a) is the mean of that reward over D, summed over d = 0, 1, 2 in this order. P(j | s, a) is the probability that tomorrow's stock is j.

| s | a | P(0|s,a) | P(1|s,a) | P(2|s,a) | P(3|s,a) | R(s,a) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.000 |
| 0 | 1 | 0.7 | 0.3 | 0.0 | 0.0 | 0.000 |
| 0 | 2 | 0.3 | 0.4 | 0.3 | 0.0 | -1.500 |
| 0 | 3 | 0.0 | 0.3 | 0.4 | 0.3 | -4.500 |
| 1 | 0 | 0.7 | 0.3 | 0.0 | 0.0 | 2.500 |
| 1 | 1 | 0.3 | 0.4 | 0.3 | 0.0 | 0.500 |
| 1 | 2 | 0.0 | 0.3 | 0.4 | 0.3 | -2.500 |
| 2 | 0 | 0.3 | 0.4 | 0.3 | 0.0 | 3.000 |
| 2 | 1 | 0.0 | 0.3 | 0.4 | 0.3 | -0.500 |
| 3 | 0 | 0.0 | 0.3 | 0.4 | 0.3 | 2.000 |
<!-- /worked -->

Read a row to see where it comes from. With $s = 1$ and $a = 0$ there is one unit on the shelf. It sells if $D \ge 1$, which has probability $0.7$, leaving $0$; with probability $0.3$ nothing is asked and the unit stays. That is the row $(0.7, 0.3, 0, 0)$. The mean number of units sold is $0.7$ and left over is $0.3$, so $R(1, 0) = 4 \times 0.7 - 1 \times 0.3 = 2.5$. Ordering a second unit, $a = 1$, makes the stock $2$: now $D = 0$ leaves two, $D = 1$ leaves one and $D = 2$ leaves none, which are the probabilities $0.3, 0.4, 0.3$ of the row $(1, 1)$, and the reward falls to $0.5$ because of the extra unit, the fee and the larger leftover.

A table of this kind is a **Markov decision process**: a set of states, an action set for each state, the probabilities $P(j \mid s, a)$ of the next state, the rewards $R(s, a)$, and a discount factor $\gamma$ that the next section introduces. The word Markov names a property of $P$ that the table has quietly assumed. Write $S_t$ and $A_t$ for the random state and action of day $t$ (days counted from $0$). The property says that, given $S_t = s$ and $A_t = a$, the conditional probability of $S_{t+1} = j$ is $P(j \mid s, a)$ whatever happened on any earlier day:

$$\Pr(S_{t+1} = j \mid S_t = s, A_t = a, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = P(j \mid s, a).$$

Conditioning on the whole history adds nothing once the present state is known. The shop satisfies it because the demand is fresh each day, so tomorrow's stock depends on today's stock and order and on a draw that has nothing to do with the past. If instead a busy day tended to be followed by another, the stock alone would not be a state: yesterday's demand would carry information about tomorrow's, and the repair is to put it into the state, $(\text{stock}, \text{yesterday's demand})$, which makes twelve states and restores the property. What a state is, is therefore a modelling decision, and the decision is "whatever the future depends on, given what the agent can do".

The loop that a decision process describes, one day after another, is short.

![A loop between two boxes. The agent, holding a policy that maps the state of the morning to an action, sends the action (units to order) to the environment. The environment draws the demand, moves to the next state according to P and pays the reward, and sends the next state and the reward back to the agent.](img/agent-environment-loop.svg)

The environment's side of the loop is the pair of tables; the agent's side is a rule for choosing the action, to which the chapter now turns.

## Policies and the sum of rewards that counts

A **policy** $\pi$ is such a rule: it says which action to take in each state. In this chapter a policy is deterministic, a list of one action per state, written $\pi = (\pi(0), \pi(1), \pi(2), \pi(3))$; the policy "order one unit when the shelf is empty and nothing otherwise" is $(1, 0, 0, 0)$. (A policy that draws its action at random from a distribution over the legal ones is the natural object of the policy-gradient chapter; the equations below extend to it by averaging over the draw.) Fixing a policy removes the agent's choices and leaves a plain **Markov chain** on the states, a random walk in which the next state is drawn from a fixed row of probabilities that depends only on the present state, whose row $s$ is the row $P(\cdot \mid s, \pi(s))$ of the table, the matrix $\mathbf{P}_\pi$.

![A transition graph of four circles labelled stock 0 to stock 3, each with the action the policy takes there, and an arrow for every positive probability. From stock 0 (order 1) the shelf stays at 0 with probability 0.7 or moves to 1 with 0.3; from stock 1 it moves to 0 with 0.7 or stays with 0.3; from stock 2 it goes to 0 with 0.3, to 1 with 0.4 or stays with 0.3; from stock 3 it goes to 1 with 0.3, to 2 with 0.4 or stays with 0.3.](img/inventory-chain.svg)

Run that chain and a **trajectory** appears: the states, actions and rewards in the order they happen, $s_0, a_0, r_0, s_1, a_1, r_1, \dots$, where the reward $r_t$ is the one paid for the action $a_t$ taken in $s_t$. Here is one, for the policy $(1, 0, 0, 0)$ starting from a shelf of two, with the demands drawn from the book's generator.

<!-- worked: mdps.trajectory_returns -->
Policy: order one unit when the shelf is empty, nothing otherwise, written (1, 0, 0, 0) with one action per stock 0 .. 3. Start: s_0 = 2. Generator: the book's SplitMix64, seed 0xaa77e877a9f056b2 (12283542109326169778); one uniform() per day in order, D = 0 if u < 0.3, D = 1 if u < 0.7, otherwise D = 2. The return G_t of day t is the sum over k = 0 .. 9 - t of gamma^k r_(t+k); the days after day 9 are not simulated and count as reward 0 (so G_10 = 0 and G_t = r_t + gamma G_(t+1)). The last four columns are G_t for gamma = 0, 0.5, 0.9 and 1.

| t | s_t | a_t | D | sold | s_(t+1) | r_t | G (0) | G (0.5) | G (0.9) | G (1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0 | 0 | 0 | 2 | -2.0 | -2.000 | -3.594 | -6.012 | -7.500 |
| 1 | 2 | 0 | 0 | 0 | 2 | -2.0 | -2.000 | -3.188 | -4.458 | -5.500 |
| 2 | 2 | 0 | 0 | 0 | 2 | -2.0 | -2.000 | -2.375 | -2.731 | -3.500 |
| 3 | 2 | 0 | 0 | 0 | 2 | -2.0 | -2.000 | -0.750 | -0.813 | -1.500 |
| 4 | 2 | 0 | 1 | 1 | 1 | 3.0 | 3.000 | 2.500 | 1.319 | 0.500 |
| 5 | 1 | 0 | 0 | 0 | 1 | -1.0 | -1.000 | -1.000 | -1.868 | -2.500 |
| 6 | 1 | 0 | 0 | 0 | 1 | -1.0 | -1.000 | 0.000 | -0.964 | -1.500 |
| 7 | 1 | 0 | 2 | 1 | 0 | 4.0 | 4.000 | 2.000 | 0.040 | -0.500 |
| 8 | 0 | 1 | 0 | 0 | 1 | -3.5 | -3.500 | -4.000 | -4.400 | -4.500 |
| 9 | 1 | 0 | 0 | 0 | 1 | -1.0 | -1.000 | -1.000 | -1.000 | -1.000 |
<!-- /worked -->

The first four days are a quiet stretch of nobody buying, and the shop loses $2$ a day on the two idle units. A comparison between this and another keeper's days needs one number, and the obvious one is the sum of the rewards. The sum is not enough for two reasons. It would let a reward ten years away count as much as one today, which no keeper believes; and for a shop that runs for ever it is not even defined, because a stream that pays $1$ every day adds up to infinity.

The fix is to count a reward of $k$ days ahead for $\gamma^k$ of its value, with a number $\gamma$ in $[0, 1)$ called the **discount factor**. The **return** from day $t$ is the discounted sum of everything that follows,

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k \ge 0} \gamma^k r_{t+k}.$$

Pulling the first term out gives the recurrence that all of the chapter rests on:

$$G_t = r_t + \gamma G_{t+1}.$$

The last four columns of the table were computed that way, backwards from the last day, with the days after day $9$ not simulated and counted as zero. At $\gamma = 0$ the return is the day's own reward, the keeper of the opening. At $\gamma = 1$ it is the plain sum, and the trajectory shows that discounting is not a harmless rescaling: the first return is $-7.500$ undiscounted, $-6.012$ at $\gamma = 0.9$ and $-3.594$ at $\gamma = 0.5$, because the early losses count in full while the later gains and losses shrink. For $\gamma < 1$ the return is always finite: if every reward is at most $R_{\max}$ in absolute value then $|G_t| \le R_{\max}\sum_k \gamma^k = R_{\max}/(1 - \gamma)$, the sum of a geometric series.

<!-- worked: mdps.discount_sums -->
A reward of 1 on every day. The sum of the first T discounted rewards is the sum over k = 0 .. T - 1 of gamma^k, which is (1 - gamma^T) / (1 - gamma) for gamma < 1 and T for gamma = 1; 1 / (1 - gamma) is the limit as T grows.

| gamma | 1/(1-gamma) | T = 10 | T = 100 | T = 1000 |
|---|---:|---:|---:|---:|
| 0.5 | 2 | 2.00 | 2.00 | 2.00 |
| 0.9 | 10 | 6.51 | 10.00 | 10.00 |
| 0.99 | 100 | 9.56 | 63.40 | 100.00 |
| 0.999 | 1000 | 9.96 | 95.21 | 632.30 |
| 1 | none | 10 | 100 | 1000 |
<!-- /worked -->

The reciprocal $1/(1 - \gamma)$ is the **effective horizon**, the number of days that carry most of the weight: at $\gamma = 0.9$ a constant reward is worth $10$ times a single one, and after $T = 10$ days the sum has reached $6.51$ of that; at $0.99$ it takes $T = 100$ days to reach $63.40$ of $100$. A second reading of $\gamma$ makes the same number concrete. Suppose that after every day the world ends with probability $1 - \gamma$. The shop is still running at day $k$ with probability $\gamma^k$, so the expected undiscounted sum of the rewards collected before the end is $\sum_k \gamma^k\,\mathbb{E}[r_k]$, the probability of an event being the expectation of its indicator: discounting is the same as a small chance, each day, that nothing further will be paid. A task that really ends, a game or a conversation, is covered by one more convention: a final state with no way out and reward $0$ (an absorbing state), after which every return stops growing. The shop is a continuing task, and the trajectory table's truncation at day $9$ is a choice made to print it, with the rest counted as zero.

The return is a random quantity, since the demands are random, and a policy should be judged by what it brings on average. That average is the value, and the idea that every state has one is the next step.

## What a state is worth

The **value function** of a policy $\pi$ gives each state the expected return of starting there and following the policy:

$$V^\pi(s) = \mathbb{E}\,[\,G_t \mid S_t = s\,], \qquad a_t = \pi(s_t) \text{ on every day.}$$

It is a number per state, and it does not depend on $t$ because the tables and the policy do not. Two keepers' policies are now comparable: the better one has the larger value in the states that matter.

To compute it, use the recurrence. Take the expectation of $G_t = r_t + \gamma G_{t+1}$ given $S_t = s$. By linearity of expectation the left side is the sum of two terms: the mean of $r_t$, which is $R(s, \pi(s))$ by definition, and $\gamma$ times the mean of $G_{t+1}$ given $S_t = s$. For the second, split on tomorrow's state. By the rule that an expectation is a probability-weighted average of conditional expectations (the conditional probability of Volume I, summed over $j$),

$$\mathbb{E}\,[\,G_{t+1} \mid S_t = s\,] = \sum_j P(j \mid s, \pi(s))\; \mathbb{E}\,[\,G_{t+1} \mid S_t = s, S_{t+1} = j\,],$$

and the Markov property says the future from day $t + 1$ depends on the past only through $S_{t+1}$, so the inner expectation is $V^\pi(j)$. The result is the **Bellman equation** for the policy:

$$V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{j} P(j \mid s, \pi(s))\, V^\pi(j) \qquad \text{for every state } s.$$

The value of a state is what the day pays plus the discounted value of where the day leads, averaged over where it may lead. That is four equations in four unknowns for the shop, and linear ones. In matrix form, with $\mathbf{r}_\pi$ the vector of the numbers $R(s, \pi(s))$ and $\mathbf{P}_\pi$ the matrix above, the vector $V^\pi$ solves

$$(\mathbf{I} - \gamma \mathbf{P}_\pi)\, V^\pi = \mathbf{r}_\pi.$$

The matrix is invertible whenever $\gamma < 1$, so the solution exists and is unique. Suppose $(\mathbf{I} - \gamma\mathbf{P}_\pi)x = 0$ and let $s$ be a state where $|x(s)|$ is largest. Then $x(s) = \gamma\sum_j P(j \mid s, \pi(s))\,x(j)$, and since the probabilities are non-negative and sum to $1$, $|x(s)| \le \gamma \max_j |x(j)| = \gamma |x(s)|$. With $\gamma < 1$ that forces $x(s) = 0$, so $x = 0$. For $\gamma = 1$ the argument fails for a good reason: every row of $\mathbf{I} - \mathbf{P}_\pi$ sums to zero, the matrix is singular, and the return is not defined.

For the policy $(1, 0, 0, 0)$ and $\gamma = 0.9$ the system is small enough to print. Row $s$ of $\mathbf{I} - \gamma \mathbf{P}_\pi$ is the row of the table for $(s, \pi(s))$, scaled by $-\gamma$, with $1$ added on the diagonal.

<!-- worked: mdps.evaluate_by_solving -->
gamma = 0.9. The system for the policy (1, 0, 0, 0): row s of A = I - gamma P_pi holds the row P(. | s, pi(s)) times -gamma, plus 1 on the diagonal; the right side is r(s) = R(s, pi(s)); the solution is found by Gaussian elimination with partial pivoting. The largest entry of |A V - r| is 8.9e-16 (below 1e-12). Every table of the chapter prints a number that rounds to zero as 0.000, whatever its sign.

| row | A[s,0] | A[s,1] | A[s,2] | A[s,3] | r(s) | V(s) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | +0.370 | -0.270 | +0.000 | +0.000 | 0.000 | 6.7500 |
| 1 | -0.630 | +0.730 | +0.000 | +0.000 | 2.500 | 9.2500 |
| 2 | -0.270 | -0.360 | +0.730 | +0.000 | 3.000 | 11.1678 |
| 3 | +0.000 | -0.270 | -0.360 | +0.730 | 2.000 | 11.6684 |

The value of four policies (one action per stock 0 .. 3), each from its own system, gamma = 0.9:

| policy | pi | V(0) | V(1) | V(2) | V(3) |
|---|---|---:|---:|---:|---:|
| never order | (0, 0, 0, 0) | 0.000 | 3.425 | 5.798 | 6.866 |
| order 1 when empty | (1, 0, 0, 0) | 6.750 | 9.250 | 11.168 | 11.668 |
| order 2 when empty | (2, 0, 0, 0) | 6.348 | 8.903 | 10.848 | 11.382 |
| order 3 when empty | (3, 0, 0, 0) | 1.268 | 4.519 | 6.807 | 7.768 |
<!-- /worked -->

The second table settles the opening. From an empty shelf the keeper who never orders collects $V(0) = 0.000$, as predicted, and the keeper who orders one unit when empty collects $6.750$; at $\gamma = 0.9$ ordering two units instead gives $6.348$, and three gives $1.268$. The values of the other states fall in line with intuition: a unit already on the shelf is worth $9.250$ to the second keeper, and two units $11.168$, and a full shelf only $11.668$, where the extra unit's holding cost bites.

An expectation is an average over what can happen, so it should be possible to check the table against what does happen. Run the policy $(1, 0, 0, 0)$ from an empty shelf for many days under random demand, discount the rewards and average the returns.

<!-- worked: mdps.monte_carlo_value -->
Policy (1, 0, 0, 0), gamma = 0.9, start s_0 = 0. Generator: the book's SplitMix64, seed 0xdf54ad9b776f4a6e (16092678251736943214); 2000 episodes of 100 days one after the other, one uniform() per day (the demand rule of the trajectory), each return G = sum over t of gamma^t r_t with the running power gamma^t multiplied once per day, days after the 100th ignored (they weigh less than 3.1e-04 in total). The mean is the plain sum over the episodes in order divided by 2000, the standard deviation uses the divisor 2000 - 1. The 5 block means are those of the consecutive blocks of 400 episodes.

| quantity | value |
|---|---:|
| mean of the sampled returns | 6.750 |
| standard deviation of one return | 2.962 |
| standard error of the mean | 0.066 |
| V(0) from the linear system | 6.750 |
| difference in standard errors | -0.004 |
| block means | 6.80, 6.57, 6.99, 6.80, 6.59 |
<!-- /worked -->

The average of $2000$ sampled returns, $6.750$, agrees with $V(0)$ from the linear system to within a hundredth of a standard error. That is luck of the draw, not precision: the block means beside it, each from $400$ episodes, spread across $6.57$ to $6.99$, which is what a standard error of $0.148$ per block predicts (the $0.066$ of the whole run times the square root of $5$). The two routes, solving a system and sampling many lives, compute the same number, and only the first is exact. Chapter 2 builds the second into a learning method for the case in which the tables are not known.

## Evaluating a policy by sweeping

Solving a system costs work that grows as the cube of the number of states, and for a table with millions of states that is out of reach. A cheaper route follows from reading the Bellman equation as an update. Start with every value at $0$ and replace each value by the right-hand side computed from the old values,

$$V_{k+1}(s) = R(s, \pi(s)) + \gamma \sum_j P(j \mid s, \pi(s))\, V_k(j), \qquad V_0 = 0.$$

One pass over all the states is a **sweep**, and each sweep reads the old vector and writes a new one. The map $\mathcal{B}_\pi$ that takes $V_k$ to $V_{k+1}$ has the true $V^\pi$ as its fixed point, by the Bellman equation. It also pulls any two vectors closer together. For two vectors $V$ and $W$ and a state $s$, the difference of the images is $\gamma\sum_j P(j \mid s, \pi(s))\,(V(j) - W(j))$, whose absolute value is at most $\gamma\max_j |V(j) - W(j)|$ because it is a weighted average with non-negative weights of numbers no larger than that in absolute value. Writing $\lVert x \rVert_\infty$ for the largest absolute entry of a vector,

$$\lVert \mathcal{B}_\pi V - \mathcal{B}_\pi W \rVert_\infty \le \gamma \lVert V - W \rVert_\infty.$$

A map with this property, for a number below $1$, is a **contraction**. Applied to $V_k$ and the fixed point it says the error shrinks by at least $\gamma$ per sweep, $\lVert V_k - V^\pi \rVert_\infty \le \gamma^k\lVert V_0 - V^\pi\rVert_\infty$, and applied to $V_k$ and $V_{k-1}$ it says the change shrinks the same way. The sweeps therefore converge, from any starting vector, to the one fixed point.

A program needs a stopping rule, and the rule it can compute is the size of the last change, $\lVert V_k - V_{k-1} \rVert_\infty$. Each later change is at most $\gamma$ times the one before, and the distance from $V_k$ to the fixed point is at most the sum of all later changes, so

$$\lVert V_k - V^\pi \rVert_\infty \le \sum_{i \ge 1} \gamma^{i} \lVert V_k - V_{k-1} \rVert_\infty = \frac{\gamma}{1 - \gamma} \lVert V_k - V_{k-1} \rVert_\infty.$$

A stop when the change falls below a tolerance $\varepsilon_{\mathrm{tol}}$ therefore guarantees an error of at most $\gamma\varepsilon_{\mathrm{tol}}/(1 - \gamma)$, which is nine times the tolerance at $\gamma = 0.9$. The routine, with the tolerance and the order of every sum fixed:

```text
function backup(P: array<real>[n, m, n], R: array<real>[n, m], V: array<real>[n],
                gamma: real, s: int, a: int) -> real:
    acc = 0.0
    for j in 0 .. n:
        acc = acc + P[s, a, j] * V[j]
    return R[s, a] + gamma * acc

function evaluate_sweeps(P: array<real>[n, m, n], R: array<real>[n, m], pi: list<int>,
                         gamma: real, tol: real) -> (array<real>[n], int):
    require 0 <= gamma and gamma < 1 and tol > 0
    V = zeros(real, [n])
    sweeps = 0
    loop:
        new = zeros(real, [n])
        change = 0.0
        for s in 0 .. n:
            new[s] = backup(P, R, V, gamma, s, pi[s])
            if abs(new[s] - V[s]) > change:
                change = abs(new[s] - V[s])
        V = new
        sweeps = sweeps + 1
        if change < tol:
            return (V, sweeps)
```

For the policy $(1, 0, 0, 0)$ at $\gamma = 0.9$ it runs as follows.

<!-- worked: mdps.evaluate_by_sweeps -->
Policy (1, 0, 0, 0), gamma = 0.9, V_0 = 0, V_(k+1)(s) = R(s, pi(s)) + gamma * sum_j P(j | s, pi(s)) V_k(j) for every s (each sum over j = 0 .. 3 in order, every state updated from the old vector). change_k is the largest |V_k(s) - V_(k-1)(s)|; the iteration stops at the first k with change_k < 1e-08. error_k is the largest |V_k(s) - V(s)| against the solved system and the last column is gamma^k times the largest |V(s)|.

| k | V_k(0) | V_k(3) | change_k | error_k | gamma^k max V |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 2.0000 | 3.00e+00 | 9.67e+00 | 1.05e+01 |
| 2 | 0.6750 | 4.2950 | 2.29e+00 | 7.37e+00 | 9.45e+00 |
| 3 | 1.2825 | 5.7125 | 1.42e+00 | 5.96e+00 | 8.51e+00 |
| 5 | 2.3213 | 7.1840 | 6.05e-01 | 4.48e+00 | 6.89e+00 |
| 10 | 4.1349 | 9.0531 | 2.91e-01 | 2.62e+00 | 4.07e+00 |
| 20 | 5.8382 | 10.7565 | 1.01e-01 | 9.12e-01 | 1.42e+00 |
| 50 | 6.7113 | 11.6297 | 4.29e-03 | 3.87e-02 | 6.01e-02 |
| 100 | 6.7498 | 11.6682 | 2.21e-05 | 1.99e-04 | 3.10e-04 |
| 174 | 6.7500 | 11.6684 | 9.10e-09 | 8.19e-08 | 1.27e-07 |

It stops after 174 sweeps with the largest error 8.19e-08 against the solved system.
<!-- /worked -->

The error against the solved system stays under the $\gamma^k$ bound at every printed sweep, and the run stops after $174$ sweeps with an error of $8.19 \times 10^{-8}$, below the nine times $10^{-8}$ that the stopping rule guarantees. For the same policy the elimination of the previous section and the iteration agree to $10^{-7}$. The direct solution is exact in one pass; the iteration is approximate and needs a number of sweeps that depends on $\gamma$ and not on the size of the problem, which is its advantage when the matrix is large and mostly zero.

The routine that solves the system is the elimination of Volume I with the matrix assembled from the table:

```text
function evaluate_policy(P: array<real>[n, m, n], R: array<real>[n, m], pi: list<int>,
                         gamma: real) -> array<real>[n]:
    require 0 <= gamma and gamma < 1
    A = identity(n)
    r = zeros(real, [n])
    for s in 0 .. n:
        for j in 0 .. n:
            A[s, j] = A[s, j] - gamma * P[s, pi[s], j]
        r[s] = R[s, pi[s]]
    return solve(A, r, true)
```

## The best policy and the equation it satisfies

The four policies above were chosen by hand. The best one maximises the value in every state at once, and the question is whether such a policy exists, since a policy that is best from one state might be poor from another. Define the **optimal value** $V^\ast(s)$ as the largest value any policy gives from $s$. If the keeper stood at the shelf and knew $V^\ast$ for the states tomorrow can reach, the best order today would be the one with the largest one-step lookahead,

$$q(s, a) = R(s, a) + \gamma \sum_j P(j \mid s, a)\, V(j),$$

with $V = V^\ast$: the reward today plus the discounted value of where the action leads. The best value of a state is then the best lookahead, and this is the Bellman equation of the optimal values:

$$V^\ast(s) = \max_{a}\; q(s, a) \quad \text{with } V = V^\ast.$$

It differs from the equation for a policy by one thing, a maximum over the actions in place of the single action of the policy, and the maximum makes it non-linear, so no elimination solves it. The idea of the sweep still applies. Let $\mathcal{B}$ take a vector $V$ to the vector with entries $\max_a q(s, a)$. Two facts make it behave like $\mathcal{B}_\pi$. The maximum does not spread differences: if $x_a$ and $y_a$ are two lists of numbers, $|\max_a x_a - \max_a y_a| \le \max_a |x_a - y_a|$, since at the action that gives the larger maximum the other list is no larger than its own maximum. So the argument of the previous section applies verbatim and $\mathcal{B}$ is a contraction of factor $\gamma$ too. And $\mathcal{B}$ is monotone: if $V \le W$ in every state then $\mathcal{B}V \le \mathcal{B}W$, because $\gamma \ge 0$ and the probabilities are non-negative.

**Value iteration** is the sweep of $\mathcal{B}$ from $V_0 = 0$: $V_{k+1}(s) = \max_a q(s, a)$ with $V = V_k$, repeated until the change is small. Because $\mathcal{B}$ is a contraction it has exactly one fixed point, which therefore is $V^\ast$, and the sweeps reach it at the rate $\gamma^k$. It remains to show that a policy attains $V^\ast$. Take the policy $\pi^\ast$ that picks, in every state, an action with the largest lookahead at $V^\ast$. Then $\mathcal{B}_{\pi^\ast}V^\ast = \mathcal{B}V^\ast = V^\ast$, so $V^\ast$ is also the fixed point of $\mathcal{B}_{\pi^\ast}$, which is to say it is the value of $\pi^\ast$. And no policy beats it: for any $\pi$ with value $U = V^\pi$ we have $U = \mathcal{B}_\pi U \le \mathcal{B}U$, and applying the monotone $\mathcal{B}$ over and over gives $U \le \mathcal{B}U \le \mathcal{B}^2U \le \dots$, which converges to $V^\ast$. One deterministic policy, read off from one vector, is best from every state.

Choosing in each state the action of the largest lookahead at a vector $V$ is the **greedy policy** of $V$. A tie, two actions with the same lookahead, needs a rule, because different choices give different policies with the same values; the chapter's rule is to take the smallest action whose lookahead is within $10^{-9}$ of the largest, so that equal actions are decided in favour of the smaller order and the printed policy is the same for every implementation.

```text
function greedy_policy(P: array<real>[n, m, n], R: array<real>[n, m], n_act: list<int>,
                       V: array<real>[n], gamma: real, tie: real) -> list<int>:
    pi = []
    for s in 0 .. n:
        q = [backup(P, R, V, gamma, s, a) for a in 0 .. n_act[s]]
        best = max(q)
        a = 0
        while q[a] < best - tie:
            a = a + 1
        append a to pi
    return pi

function value_iteration(P: array<real>[n, m, n], R: array<real>[n, m], n_act: list<int>,
                         gamma: real, tol: real) -> (array<real>[n], int):
    require 0 <= gamma and gamma < 1 and tol > 0
    V = zeros(real, [n])
    sweeps = 0
    loop:
        new = zeros(real, [n])
        change = 0.0
        for s in 0 .. n:
            best = backup(P, R, V, gamma, s, 0)
            for a in 1 .. n_act[s]:
                q = backup(P, R, V, gamma, s, a)
                if q > best:
                    best = q
            new[s] = best
            if abs(new[s] - V[s]) > change:
                change = abs(new[s] - V[s])
        V = new
        sweeps = sweeps + 1
        if change < tol:
            return (V, sweeps)
```

The shop at $\gamma = 0.9$, from $V_0 = 0$, with the stopping rule $\lVert V_k - V_{k-1}\rVert_\infty < 10^{-8}$:

<!-- worked: mdps.value_iteration_run -->
gamma = 0.9, V_0 = 0, V_(k+1)(s) = max over the legal a of [R(s, a) + gamma * sum_j P(j | s, a) V_k(j)], every state updated from the old vector; stop at the first k with change_k < 1e-08, change_k being the largest |V_k(s) - V_(k-1)(s)|.

| k | V_k(0) | V_k(1) | V_k(2) | V_k(3) | change_k |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| 1 | 0.000 | 2.500 | 3.000 | 2.000 | 3.00e+00 |
| 2 | 0.675 | 3.175 | 4.710 | 4.295 | 2.29e+00 |
| 3 | 1.282 | 3.782 | 5.597 | 5.713 | 1.42e+00 |
| 4 | 1.829 | 4.329 | 6.219 | 6.579 | 8.66e-01 |
| 5 | 2.321 | 4.821 | 6.732 | 7.184 | 6.05e-01 |
| 10 | 4.135 | 6.635 | 8.553 | 9.053 | 2.91e-01 |
| 20 | 5.838 | 8.338 | 10.256 | 10.757 | 1.01e-01 |
| 50 | 6.711 | 9.211 | 11.129 | 11.630 | 4.29e-03 |
| 100 | 6.750 | 9.250 | 11.168 | 11.668 | 2.21e-05 |
| 174 | 6.750 | 9.250 | 11.168 | 11.668 | 9.10e-09 |

It stops after 174 sweeps. The one-step lookahead q(s, a) = R(s, a) + gamma * sum_j P(j | s, a) V_174(j) of the last vector, and the greedy policy: the smallest a whose q is within 1e-09 of the largest q of its state.

| s | a | q(s, a) | greedy |
|---:|---:|---:|---|
| 0 | 0 | 6.0750 |  |
| 0 | 1 | 6.7500 | chosen |
| 0 | 2 | 6.6678 |  |
| 0 | 3 | 5.1684 |  |
| 1 | 0 | 9.2500 | chosen |
| 1 | 1 | 8.6678 |  |
| 1 | 2 | 7.1684 |  |
| 2 | 0 | 11.1678 | chosen |
| 2 | 1 | 9.1684 |  |
| 3 | 0 | 11.6684 | chosen |

The greedy policy is (1, 0, 0, 0).
<!-- /worked -->

The first sweep reproduces the one-day rewards, since $V_0 = 0$: $V_1(s) = \max_a R(s, a)$. After that the values climb as more days are counted, and it takes $174$ sweeps to settle. The lookahead table of the last vector explains the answer: on an empty shelf ordering one unit gives $6.7500$, two units $6.6678$, none $6.0750$ and three $5.1684$; on any other shelf ordering nothing is the largest. The optimal policy is $(1, 0, 0, 0)$, the one the keeper of the opening found, and $V^\ast$ is the vector $(6.750, 9.250, 11.168, 11.668)$ that the linear system gave for that policy.

A vector that is only close to $V^\ast$ gives a policy that may be slightly wrong, and the loss can be bounded. Suppose $\lVert V - V^\ast \rVert_\infty \le \epsilon$ and let $\pi$ be greedy for $V$, so that $\mathcal{B}_\pi V = \mathcal{B}V$. Then $\lVert V^\pi - V \rVert_\infty = \lVert \mathcal{B}_\pi V^\pi - V \rVert_\infty \le \gamma\lVert V^\pi - V \rVert_\infty + \lVert \mathcal{B}V - V \rVert_\infty$, and $\lVert \mathcal{B}V - V \rVert_\infty \le \lVert \mathcal{B}V - \mathcal{B}V^\ast \rVert_\infty + \lVert V^\ast - V \rVert_\infty \le (1 + \gamma)\epsilon$. So $\lVert V^\pi - V\rVert_\infty \le (1 + \gamma)\epsilon/(1 - \gamma)$ and, adding the distance from $V$ to $V^\ast$,

$$\lVert V^\pi - V^\ast \rVert_\infty \le \frac{2\epsilon}{1 - \gamma}.$$

At $\gamma = 0.9$ the stop above has $\epsilon \le 8.19 \times 10^{-8}$, so the greedy policy is within $1.64 \times 10^{-6}$ of optimal in every state. The bound also says that a near-tie, whichever way it is broken, costs very little, which is why a deterministic rule is enough.

At $\gamma = 0$ there is nothing to iterate, because the lookahead is the reward alone, and the rule for ties matters:

<!-- worked: mdps.greedy_tie -->
gamma = 0, so q(s, a) = R(s, a) whatever V is. The first rows of the table of the inventory problem, and the greedy policy with the tie rule (smallest a within 1e-09 of the best):

| s | a | q(s, a) = R(s, a) |
|---|---:|---:|
| 0 | 0 | 0.000 |
| 0 | 1 | 0.000 |
| 0 | 2 | -1.500 |
| 0 | 3 | -4.500 |
| 1 | 0 | 2.500 |
| 1 | 1 | 0.500 |
| 1 | 2 | -2.500 |

At s = 0 the actions 0 and 1 are tied at 0.000, so the greedy policy is (0, 0, 0, 0), which orders nothing.
<!-- /worked -->

This is the opening, seen from the other side. The myopic keeper's two best actions on an empty shelf are tied at $0.000$, the rule takes the smaller one, and the shelf stays empty.

## Policy iteration

Value iteration improves values and reads the policy off at the end. A second algorithm improves the policy and reads the values off in the middle. Start from any policy, solve its linear system, and replace the policy by the greedy policy of the solved values; repeat until the policy no longer changes. This is **policy iteration**.

```text
function policy_iteration(P: array<real>[n, m, n], R: array<real>[n, m], n_act: list<int>,
                          gamma: real, pi0: list<int>) -> (list<int>, array<real>[n], int):
    pi = pi0
    rounds = 0
    loop:
        V = evaluate_policy(P, R, pi, gamma)
        new = greedy_policy(P, R, n_act, V, gamma, 1e-9)
        rounds = rounds + 1
        if new == pi:
            return (pi, V, rounds)
        pi = new
```

The reason it works is the same pair of facts as before. Let $\pi'$ be greedy for $V^\pi$. In every state its lookahead is at least that of $\pi$, so $\mathcal{B}_{\pi'}V^\pi = \mathcal{B}V^\pi \ge \mathcal{B}_\pi V^\pi = V^\pi$. Apply the monotone $\mathcal{B}_{\pi'}$ repeatedly: $V^\pi \le \mathcal{B}_{\pi'}V^\pi \le \mathcal{B}_{\pi'}^2V^\pi \le \dots$, and the limit of this sequence is $V^{\pi'}$. So a round never makes any value smaller, and when it changes nothing, $V^\pi = \mathcal{B}V^\pi$ and $\pi$ is optimal. There are only finitely many deterministic policies (twenty-four here, the product of the numbers of actions $4 \cdot 3 \cdot 2 \cdot 1$), and a policy that changes strictly raises some value, so the loop ends.

<!-- worked: mdps.policy_iteration_run -->
gamma = 0.9, starting policy (0, 0, 0, 0): each round solves the system of the current policy and takes the greedy policy of the solved values; the stop is a round whose greedy policy equals its own.

| round | policy | V(0) | V(1) | V(2) | V(3) | greedy policy |
|---|---|---:|---:|---:|---:|---|
| 0 | (0, 0, 0, 0) | 0.000 | 3.425 | 5.798 | 6.866 | (2, 0, 0, 0) |
| 1 | (2, 0, 0, 0) | 6.348 | 8.903 | 10.848 | 11.382 | (1, 0, 0, 0) |
| 2 | (1, 0, 0, 0) | 6.750 | 9.250 | 11.168 | 11.668 | (1, 0, 0, 0) |

3 rounds (3 solves), final policy (1, 0, 0, 0).

gamma = 0.99, starting policy (0, 0, 0, 0): each round solves the system of the current policy and takes the greedy policy of the solved values; the stop is a round whose greedy policy equals its own.

| round | policy | V(0) | V(1) | V(2) | V(3) | greedy policy |
|---|---|---:|---:|---:|---:|---|
| 0 | (0, 0, 0, 0) | 0.000 | 3.556 | 6.271 | 7.880 | (2, 1, 0, 0) |
| 1 | (2, 1, 0, 0) | 62.850 | 64.850 | 67.350 | 68.181 | (2, 0, 0, 0) |
| 2 | (2, 0, 0, 0) | 79.623 | 82.047 | 84.123 | 84.894 | (2, 0, 0, 0) |

3 rounds (3 solves), final policy (2, 0, 0, 0).
<!-- /worked -->

From the policy that never orders, three solves are enough at $\gamma = 0.9$: the first round overshoots to ordering two units, because that is what the values of never ordering recommend, the second corrects to one unit and the third confirms it. At $\gamma = 0.99$ the path goes through $(2, 1, 0, 0)$ before it reaches $(2, 0, 0, 0)$. Three rounds against $174$ and $1813$ sweeps is not the whole comparison, since each round pays for a solve; the last section counts it.

## How the discount changes the answer

The discount $\gamma$ is part of the definition of the problem, not a setting of the algorithm, and a different $\gamma$ is a different problem with, in general, a different best policy. Running value iteration at several discounts shows how.

<!-- worked: mdps.discount_sweep -->
Value iteration from V_0 = 0 with the stopping rule change_k < 1e-08, then the greedy policy with the tie rule, for each gamma. sweeps is the number of sweeps until the stop, and 1/(1-gamma) the effective horizon.

| gamma | greedy policy | V(0) | V(3) | sweeps | 1/(1-gamma) |
|---|---:|---:|---:|---:|---:|
| 0 | (0, 0, 0, 0) | 0.000 | 2.000 | 2 | 1.0 |
| 0.3 | (1, 0, 0, 0) | 0.321 | 2.965 | 17 | 1.4 |
| 0.5 | (1, 0, 0, 0) | 0.750 | 3.968 | 28 | 2.0 |
| 0.7 | (1, 0, 0, 0) | 1.750 | 5.706 | 52 | 3.3 |
| 0.8 | (1, 0, 0, 0) | 3.000 | 7.404 | 83 | 5.0 |
| 0.9 | (1, 0, 0, 0) | 6.750 | 11.668 | 174 | 10.0 |
| 0.95 | (2, 0, 0, 0) | 14.511 | 19.674 | 355 | 20.0 |
| 0.99 | (2, 0, 0, 0) | 79.623 | 84.894 | 1813 | 100.0 |

The two policies (1, 0, 0, 0) and (2, 0, 0, 0) have the same V(0) at gamma = 0.9375 (V(0) = 11.250 and 11.250); this is found by 60 halvings of the interval [0.9, 0.95] on the difference of their solved values, and the second is better above it.
<!-- /worked -->

At $\gamma = 0$ the keeper of the opening orders nothing. For $\gamma$ from $0.3$ to $0.9$ it orders one unit on an empty shelf, and at $0.95$ and $0.99$ two units. The switch from one to two lies at $\gamma = 0.9375$, where the two policies have the same value $11.250$ on an empty shelf. The trade behind the switch is visible in the rows of the table: after an order of one unit the shelf is empty again the next morning with probability $0.7$ and a new order, with its fee, is due, while after an order of two units it is empty with probability $0.3$; the larger order buys fewer empty mornings but holds a unit at a cost of $1$ when demand is low, and what it saves lies in later days that count only when the horizon, $1/(1 - \gamma)$, is long enough. At $0.9$ the horizon is $10$ days and the smaller order wins; at $0.95$ it is $20$ and the larger does. The count of sweeps grows with the horizon, $17$ at $0.3$ and $1813$ at $0.99$, which the next section explains.

## Why the optimal policy appears long before the optimal values

The shop gives a second reason to trust a policy sooner than a value. Look at the greedy policy of each vector $V_k$ that value iteration produces, rather than only the last.

<!-- worked: mdps.policy_settles -->
The greedy policy (smallest action within 1e-09 of the best lookahead) of the vector V_k of value iteration, for k = 1, 2, ..., up to the sweep at which the run stops (change_k < 1e-08). 'settled at' is the first k from which the greedy policy of V_k never again differs from the final one.

| gamma | sweeps to stop | settled at | final policy | greedy policy of V_k along the run |
|---:|---:|---:|---|---|
| 0.5 | 28 | 1 | (1, 0, 0, 0) | from sweep 1: (1, 0, 0, 0) |
| 0.9 | 174 | 1 | (1, 0, 0, 0) | from sweep 1: (1, 0, 0, 0) |
| 0.95 | 355 | 4 | (2, 0, 0, 0) | from sweep 1: (1, 0, 0, 0); from sweep 4: (2, 0, 0, 0) |
| 0.99 | 1813 | 3 | (2, 0, 0, 0) | from sweep 1: (1, 0, 0, 0); from sweep 3: (2, 0, 0, 0) |
<!-- /worked -->

At $\gamma = 0.9$ the greedy policy of $V_1$ is already the final one, and the other $173$ sweeps only refine the values; at $0.95$ it settles from the fourth sweep and at $0.99$ from the third, against $355$ and $1813$ sweeps to the stop. A policy only needs the order of the lookaheads in each state to be right, while a value needs their sizes, and the ordering becomes stable once the gaps between them exceed the remaining error. That is the idea behind stopping value iteration early when only a policy is wanted, and it is what the bound of the previous section makes safe: the greedy policy of a vector within $\epsilon$ of $V^\ast$ is within $2\epsilon/(1 - \gamma)$ of optimal, whatever the gaps. Policy iteration leans on the same fact: it spends one exact solve per candidate policy and needs only as many rounds as the policy takes steps to settle.

## How many sweeps the contraction predicts

The contraction argument is also a forecast of the work. The change of the first sweep from $V_0 = 0$ is the largest one-day reward, $3.000$ here, and each later change is at most $\gamma$ times the one before, so the change of sweep $k$ is at most $3\gamma^{k-1}$. The stopping rule fires no later than the first $k$ at which that bound is below the tolerance, which gives the prediction. The runs below stop earlier than predicted, and the table shows by how much and why.

<!-- worked: mdps.contraction_prediction -->
V_0 = 0 and the stopping rule change_k < 1e-08, as above; change_1 = 3.000 for every gamma (the largest R(s, a)). predicted is the smallest k with gamma^(k-1) * change_1 < 1e-08; actual is the sweep at which the run stopped. The bound says change_k <= (change_1 / gamma) * gamma^k, whose constant is the column 'bound constant'; 'actual constant' is change_k / gamma^k at the stop. error is the largest |V_k(s) - V*(s)|, V* being the solved value of the greedy policy of V_k; bound is gamma * change_k / (1 - gamma).

| gamma | predicted k | actual k | change_k / change_(k-1) | bound constant | actual constant | error | bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 30 | 28 | 0.5000 | 6.000 | 1.500 | 5.59e-09 | 5.59e-09 |
| 0.9 | 187 | 174 | 0.9000 | 3.333 | 0.833 | 8.19e-08 | 8.19e-08 |
| 0.99 | 1944 | 1813 | 0.9900 | 3.030 | 0.813 | 9.83e-07 | 9.83e-07 |
<!-- /worked -->

The ratio of consecutive changes at the stop is exactly $\gamma$ to the digits shown, so the rate of the bound is the true rate: once the maximising actions stop changing the iteration is the linear one of a policy, and its slowest mode is the constant vector, which every matrix $\mathbf{P}_\pi$ with rows summing to $1$ leaves unchanged and the discount shrinks by $\gamma$ per sweep. What the bound misses is the constant in front. It uses $\text{change}_1/\gamma$, but the first changes are dominated by the quickly decaying parts of the iteration, and by the stop the real constant has fallen to about a quarter of the bound's: $0.833$ against $3.333$ at $\gamma = 0.9$, and $0.813$ against $3.030$ at $0.99$. A constant smaller by a factor $c$ is a fixed number of sweeps, $\ln c/\ln(1/\gamma)$, which is the $13$ sweeps of difference at $\gamma = 0.9$ ($187$ predicted, $174$ needed) and $131$ at $0.99$ ($1944$ and $1813$). The error after the stop equals the bound $\gamma\,\text{change}_k/(1 - \gamma)$ to three digits, so on this problem the guarantee of the stopping rule is tight as well.

![The largest change of the value vector in each sweep of value iteration, for gamma 0.5, 0.9 and 0.99, with both axes logarithmic (sweep number across, change up). Each solid curve falls steeply and ends at the dashed horizontal tolerance of 1e-8, after 28, 174 and 1813 sweeps; above each one a dashed curve of the same colour shows the geometric bound, three times gamma to the power k minus one, which is parallel and about a factor of four higher.](img/sweeps-to-converge.svg)

The figure shows the curves of the table: solid for the measured change, dashed for the contraction bound, and the horizontal line for the tolerance. For discounts near $1$ the rate is $\ln(1/\gamma) \approx 1 - \gamma$, so the sweeps to a fixed accuracy grow like the effective horizon $1/(1 - \gamma)$: the move from $\gamma = 0.5$ to $0.99$ multiplies the horizon by $50$ and the sweeps by $1813/28 \approx 65$, as $\ln 2/\ln(1/0.99) \approx 69$ would suggest.

## What a sweep costs

A sweep is a fixed amount of arithmetic, which can be counted before any machine is chosen. For each pair of state and action the lookahead is a dot product of a row of $P$ with the vector, $n$ multiplications and $n - 1$ additions, then one multiplication by $\gamma$ and one addition of $R$: $2n + 1$ operations, with a multiply, an add, a subtract and a divide counted as one each and comparisons not counted. A sweep over all the pairs of the shop is $10 \times 9 = 90$ FLOPs. Policy evaluation by sweeps touches one pair per state, $4 \times 9 = 36$ FLOPs per sweep. Solving the system costs the assembly of $\mathbf{I} - \gamma\mathbf{P}_\pi$ ($2n^2 = 32$ FLOPs) and the elimination with back substitution ($62$ FLOPs for $n = 4$, $46$ forward and $16$ back, counted operation by operation by the rule in the caption of the table below, which is not Volume I's multiply-add count), which is a little over $\tfrac{2}{3}n^3$ FLOPs for large $n$.

<!-- worked: mdps.work_per_sweep -->
Counts for the inventory problem (n = 4 states, 10 legal pairs), gamma = 0.9, from the stopping rule change_k < 1e-08. One operation is a multiply, an add, a subtract or a divide; comparisons are not counted. A pair costs 4 multiplies and 3 adds for the sum over j, one multiply by gamma and one add of R, that is 9. The elimination is counted as the routine that made the other tables runs it: below the pivot of column k each of the n - 1 - k rows costs one divide for its factor and then a multiply and a subtract (2 operations) for each of the n - k columns after the pivot column, the right-hand side included, the entry that becomes zero being set and not computed; back substitution costs 2 operations per already known unknown in a row and one divide per row. For n = 4 that is 46 forward and 16 back, 62 in all. Volume I counts a multiply-add as one operation and leaves divisions out, so its elimination counts are not comparable with these.

| what | count | FLOPs |
|---|---|---:|
| one sweep of value iteration | 10 pairs times 9 | 90 |
| one sweep of evaluating a fixed policy | 4 states times 9 | 36 |
| forming I - gamma P_pi | 2 times 4^2 | 32 |
| elimination and back substitution | counted | 62 |
| one round of policy iteration | 32 + 62 + 90 | 184 |
| value iteration to the stop (174 sweeps) | 174 times 90 | 15660 |
| policy iteration to the stop (3 rounds) | 3 times 184 | 552 |
| evaluating one policy by sweeps (174 sweeps) | 174 times 36 | 6264 |
| evaluating one policy by solving | 32 + 62 | 94 |

The same counts for a dense problem with 3 actions in every state and n states (a transition table of n times 3 times n entries of 4 bytes, read once per sweep):

| n | FLOPs per sweep | FLOPs of one solve | bytes of P | FLOP per byte |
|---|---:|---:|---:|---:|
| 4 | 108 | 62 | 192 | 0.56 |
| 100 | 60,300 | 681,550 | 120,000 | 0.50 |
| 10000 | 600,030,000 | 666,816,655,000 | 1,200,000,000 | 0.50 |
<!-- /worked -->

On four states the direct solve wins easily: $94$ FLOPs to evaluate a policy against $6264$ FLOPs for $174$ sweeps, and three rounds of policy iteration come to $552$ FLOPs against $15660$ FLOPs for value iteration. The balance reverses with size, because a sweep over a dense table costs about $2n^2$ per action while a solve costs about $\tfrac{2}{3}n^3$: sweeping a fixed policy is cheaper than solving for it while the number of sweeps is below about $n/3$. At $n = 10\,000$ states a dense sweep over three actions is $6.0 \times 10^{8}$ FLOPs and one solve is $6.7 \times 10^{11}$ FLOPs, more than a thousand such sweeps. Real decision processes are sparse, each action leading to a few next states, which makes a sweep cheaper still and a solve, whose fill-in destroys the sparsity, dearer.

The last column of the table is the arithmetic intensity of a sweep. Every entry of $P$ is read once, used in one multiplication and one addition, and discarded; with four-byte entries a sweep over the table moves $4$ bytes for $2$ FLOPs, $0.5$ FLOP per byte (a little more for small $n$, where the multiply by $\gamma$ and the add of $R$ of each pair are not negligible). With so few operations per byte a sweep is limited by how fast the table can be read and not by how fast it can be multiplied, which is why the count of bytes, $1.2 \times 10^{9}$ for the dense table at $n = 10\,000$, matters as much as the count of FLOPs. Counting both is the discipline of the roofline chapters of Volume IV, applied to a new algorithm.

## What brain does with these ideas

brain trains models that act, and the vocabulary of this chapter is the one its code is written in. The pieces that touch the chapter are an interface and a contrast.

The reward-driven interface of language-model training is a decision process with the reward at the end. `crates/promote/src/env.rs` defines an `Environment` that produces tasks and advances an episode by one turn at a time, `step` returning the observation to feed back and a `done` flag, with the single-turn default that any action ends the episode at once; its `Verifier` scores a completion and returns a `Reward`, a scalar `value` and a named breakdown. Read in the chapter's terms, the state is the prompt and the transcript so far, an action is the token ids of a turn, and all of the reward arrives on the last step. Then every position of the episode has the same return, the final reward (the chapter's $\gamma = 1$, with one non-zero $r_t$). `crates/rl/src/objective/advantage.rs` turns the rewards of a group of completions into a single scalar per completion, which "every one of its tokens is pushed with", and the ATIF adapter `crates/rl/src/atif.rs` broadcasts a trajectory's reward across every token it contributes. Chapters 3 and 4 explain how those scalars become gradients and why the reward is subtracted from the group's mean first.

`crates/solve` is the contrast, a goal-reaching decision problem solved without computing a value. Its `StateSpace` in `crates/solve/src/lib.rs` is a deterministic process, a move applied to a state giving the next state with probability one and a goal in place of a reward, and commit `199078b34` (23 September 2026) introduced it with the observation, in the crate's header, that walking away from the goal along invertible moves labels itself, so that "no value has to be bootstrapped, and no episode has to be scored". What is trained is a policy, and `crates/solve/src/rollout.rs` plays it greedily: it picks the move whose probability is strictly larger than the best so far, so on a tie the first, smallest-numbered allowed move wins, the same convention as the chapter's tie rule. A rollout that has not reached the goal after `max_steps` moves is reported stuck and counted against the policy, a horizon cap in the spirit of the truncated trajectory above. Chapter 5 returns to this crate and `crates/search`.

## Check your understanding

1. The first return in the trajectory table is $-7.500$ at $\gamma = 1$ and $-6.012$ at $\gamma = 0.9$. Where does the difference come from, and would discounting always move a return toward zero?
2. Find the error: "the policy $(1, 0, 0, 0)$ is optimal at $\gamma = 0.9$, so it is optimal for every discount".
3. A new keeper believes demand is bursty: a day with demand $2$ makes demand $2$ more likely tomorrow. Is the stock alone still a state of a Markov decision process? What would you add, and how many states result?
4. Value iteration at $\gamma = 0.99$ is stopped when the change is $10^{-6}$. A colleague says the values are within $10^{-6}$ of optimal. What is the correct guarantee, and what does it say about the greedy policy?
5. Why can policy iteration use an exact solve and stop after a handful of rounds while value iteration needs hundreds of sweeps, and when does the extra cost per round stop being worth it?
6. For $\gamma = 1$ the matrix $\mathbf{I} - \mathbf{P}_\pi$ is singular. Show it, and say what this means for the value of a policy in a task that never ends.
7. At $\gamma = 0.95$ the greedy policy of $V_4$ is already the final policy, yet value iteration runs $355$ sweeps. What did the other sweeps do, and what could one do to stop earlier if the policy is all that is wanted?

## Summary

A Markov decision process is a set of states, the actions available in each, the probabilities of the next state and the mean reward of each pair, and a discount; its defining property is that the next state depends on the past only through the present state and action. A policy chooses an action per state, the return is the discounted sum of the rewards that follow, and the value of a state is its expected return. Taking the expectation of $G_t = r_t + \gamma G_{t+1}$ gives the Bellman equation, a linear system for a policy, solved exactly by elimination or approximately by sweeps that contract by $\gamma$ per pass and stop with a guaranteed error of $\gamma/(1 - \gamma)$ times the last change. Replacing the policy's action by a maximum gives the optimal equation, which is a contraction too, so value iteration converges to the optimal value and the greedy policy of that vector is optimal; policy iteration reaches the same policy in a few rounds, each costing a solve. The discount belongs to the problem: on the shop the best order changes from none to one to two units as the horizon $1/(1 - \gamma)$ lengthens, and the sweeps to a fixed accuracy grow with it.

Everything here used the tables $P$ and $R$. A keeper rarely has them: what exists is a stream of days, and the update $V(s) \leftarrow r + \gamma V(s')$ computed from one day's experience in place of the sum over all next states is the thought on which the next chapter builds, together with the question of how often to try an action that looks worse.

## References

- R. Bellman, "Dynamic Programming", 1957.
- R. A. Howard, "Dynamic Programming and Markov Processes", 1960.
- M. L. Puterman, "Markov Decision Processes: Discrete Stochastic Dynamic Programming", 1994.
- R. S. Sutton and A. G. Barto, "Reinforcement Learning: An Introduction", second edition, 2018.

## Exercises

1. **[Calculate]** From the rules of the shop, derive the row $P(\cdot \mid 1, 1)$ and the reward $R(1, 1)$ of the table and check them against it.
2. **[Calculate]** A trajectory pays $2$, $0$, $5$ on three days and nothing after. Give $G_0$, $G_1$, $G_2$ for $\gamma = 0.5$ and for $\gamma = 0.9$ using $G_t = r_t + \gamma G_{t+1}$, and check $G_0$ against the sum $\sum_k \gamma^k r_k$.
3. **[Calculate]** A two-state chain with one action has $\mathbf{P} = \begin{pmatrix} 0.5 & 0.5 \\ 0.2 & 0.8 \end{pmatrix}$, rewards $\mathbf{r} = (1, 0)$ and $\gamma = 0.5$. Solve $(\mathbf{I} - \gamma\mathbf{P})V = \mathbf{r}$ for $V(0)$ and $V(1)$, and give the vectors after the first and the second sweep from the zero vector.
4. **[Calculate]** For $\gamma = 0.8$, rewards of at most $1$ in absolute value and a first change of $1$, give the bound on $|G_t|$, the smallest sweep $k$ at which the contraction bound guarantees a change below $10^{-3}$, and the error bound at that stop.
5. **[Calculate]** In one state the lookaheads of three actions are $2.0$, $2.0 + 5 \times 10^{-10}$ and $1.5$. Which action does the chapter's rule choose? Which if the second were $2.0 + 2 \times 10^{-9}$?
6. **[Calculate]** For a problem with $n = 5$ states and $2$ actions in each, give the FLOPs of one value-iteration sweep, of assembling and solving the system of a policy by the chapter's counts, and the number of sweeps of policy evaluation (each $n(2n + 1)$ FLOPs) that cost as much as one solve.
7. **[Implement]** Reproduce `mdps.inventory_mdp`, `mdps.trajectory_returns` and `mdps.discount_sums`, including the demand rule, the generator and the backward computation of the returns.
8. **[Implement]** Reproduce `mdps.evaluate_by_solving`, `mdps.evaluate_by_sweeps` and `mdps.monte_carlo_value`, with elimination with partial pivoting, the stopping rule on the largest change and the generator draws in the order stated.
9. **[Implement]** Reproduce `mdps.value_iteration_run`, `mdps.greedy_tie`, `mdps.policy_iteration_run`, `mdps.discount_sweep`, `mdps.policy_settles`, `mdps.contraction_prediction` and `mdps.work_per_sweep`, including the FLOP counts of the elimination.
10. **[Investigate]** Stop value iteration at $\gamma = 0.9$ with the tolerance $10^{-3}$ instead of $10^{-8}$. How many sweeps does it take, what is the guaranteed error, and is the greedy policy of the vector at the stop the same as before? Compare the gap between the best and second-best lookahead on an empty shelf with $2\epsilon/(1 - \gamma)$ and say whether the bound alone proves the answer.

## Key terms

- **Markov decision process**: a set of states, the actions available in each, the probabilities of the next state given a state and an action, the mean reward of each pair and a discount factor, with the property that the next state depends on the past only through the present state and action. <!-- concept: markov-decision-process -->
- **policy**: a rule that chooses an action in each state; in this chapter a list of one action per state. <!-- concept: policy -->
- **return**: the discounted sum of the rewards that follow a day, $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$, which obeys $G_t = r_t + \gamma G_{t+1}$. <!-- concept: return -->
- **discount factor**: the number $\gamma$ in $[0, 1)$ by which a reward one day later is multiplied, which keeps the return finite and sets an effective horizon of $1/(1 - \gamma)$ days. <!-- concept: discount-factor -->
- **value function**: the expected return of starting in each state and following a policy, written $V^\pi$ (and $V^\ast$ for the best policy). <!-- concept: value-function -->
- **Bellman equation**: the equation that says the value of a state is the reward of the day plus the discounted, averaged value of the next state, linear for a policy and with a maximum over actions for the optimal values. <!-- concept: bellman-equation -->
- **value iteration**: repeated sweeps of the optimal Bellman equation from a starting vector, each sweep replacing every value by the best one-step lookahead of the old values, until the change is below a tolerance. <!-- concept: value-iteration -->

## Answers

**Recall questions.**

1. The choice decided which reward the agent saw and so what it learned, but it left the buttons unchanged: each paid with the same probability on every press, and the choice did not alter what the next press would give. There was no state. Here the order changes tomorrow's stock, and so the odds and the reward of tomorrow's choices.
2. The probability of at least one customer is $0.4 + 0.3 = 0.7$ and of two is $0.3$, so the conditional probability is $0.3/0.7 = 3/7 \approx 0.4286$.
3. $\min(D, 1)$ is $0$ with probability $0.3$ and $1$ with probability $0.7$, so its expectation is $0.7$. By linearity $\mathbb{E}[4\min(D, 1) - 2] = 4 \times 0.7 - 2 = 0.8$.
4. The determinant is $1 - 0.25 = 0.75$, so $x = (2 \times 1 + 0.5 \times 1)/0.75 = 10/3 \approx 3.3333$ and $y = (1 \times 1 + 0.5 \times 2)/0.75 = 8/3 \approx 2.6667$. Check: $10/3 - 4/3 = 2$ and $-5/3 + 8/3 = 1$.

**Check your understanding.**

1. The first four days pay $-2$ each, $-8$ undiscounted; discounted by $0.9$ they are worth $-2(1 + 0.9 + 0.81 + 0.729) = -6.878$. The six later days sum to $+0.5$ undiscounted, and discounted they are worth $0.9^4 \times 1.319 = +0.865$ (the $G_4$ of the table), because the gain of the first of them is weighted before most of the losses. Together $-6.878 + 0.865 = -6.013$, the table's $-6.012$ to rounding. Discounting does not always move a return toward zero: it changes the weights of the days, so a stream of early gains followed by late losses becomes more positive and one of early losses followed by late gains more negative.
2. The best policy depends on $\gamma$: the discount table shows $(1, 0, 0, 0)$ at $0.3$ to $0.9$ but $(2, 0, 0, 0)$ at $0.95$ and $0.99$, and $(0, 0, 0, 0)$ at $0$. Optimality is a statement about one discount.
3. No: the stock does not carry yesterday's demand, which now predicts today's, so the next state depends on more than the present state and action. Adding yesterday's demand $\{0, 1, 2\}$ to the stock gives $4 \times 3 = 12$ states, and the next state's second coordinate is today's demand, which restores the property.
4. The guarantee is $\gamma/(1 - \gamma) \times 10^{-6} = 99 \times 10^{-6} \approx 9.9 \times 10^{-5}$ for the error of the values, about a hundred times the change. The greedy policy of the vector is within $2\epsilon/(1 - \gamma) \approx 2 \times 10^{-2}$ of optimal in every state, and exactly optimal if the gaps between the best and second-best lookaheads exceed that.
5. Each round of policy iteration solves for the policy's value exactly, so the improvement step sees the true lookaheads, and the policy sequence climbs strictly through at most a finite set; value iteration moves all values only a fraction $1 - \gamma$ closer to the optimum per sweep. The extra cost per round, a solve of about $\tfrac{2}{3}n^3$ FLOPs, stops being worth it when the number of states is so large that the solve costs more than the sweeps it saves, roughly when the sweeps needed exceed $n/3$, and when the table is sparse enough that elimination fills it in.
6. Every row of $\mathbf{P}_\pi$ sums to $1$, so $\mathbf{P}_\pi\mathbf{1} = \mathbf{1}$ and $(\mathbf{I} - \mathbf{P}_\pi)\mathbf{1} = 0$ with a non-zero vector: the matrix is singular. For a task that never ends and has no discount the sum of the rewards need not be finite, and the system has either no solution or a family of them, so the value is not defined; one must discount, end the task or switch to an average reward per day.
7. They refined the values, whose sizes were still moving, without changing which action has the largest lookahead in any state. If only the policy is wanted, stop when the greedy policy has been unchanged for several sweeps (a heuristic) or, with a guarantee, when $2\epsilon/(1 - \gamma)$ with $\epsilon = \gamma\,\text{change}_k/(1 - \gamma)$ is below the smallest gap between best and second-best lookahead, or finish with a policy-iteration round: evaluate the greedy policy exactly and check that it is stable.

**Exercises.**

1. With $s = 1$ and $a = 1$ the stock is $2$. Demand $0$ leaves $2$ (probability $0.3$), demand $1$ leaves $1$ ($0.4$), demand $2$ leaves $0$ ($0.3$), so the row is $P(0) = 0.3$, $P(1) = 0.4$, $P(2) = 0.3$, $P(3) = 0$. The mean units sold are $0 \times 0.3 + 1 \times 0.4 + 2 \times 0.3 = 1.0$ and the mean left over is $1.0$, so $R = 4 \times 1.0 - 1 \times 1.0 - 2 \times 1 - 0.5 = 0.5$, which is the table's row $(1, 1)$.
2. $\gamma = 0.5$: $G_2 = 5$, $G_1 = 0 + 0.5 \times 5 = 2.5$, $G_0 = 2 + 0.5 \times 2.5 = 3.25$; the sum is $2 + 0 + 0.25 \times 5 = 3.25$. $\gamma = 0.9$: $G_2 = 5$, $G_1 = 4.5$, $G_0 = 2 + 0.9 \times 4.5 = 6.05$; the sum is $2 + 0 + 0.81 \times 5 = 6.05$.
3. $\mathbf{I} - 0.5\mathbf{P} = \begin{pmatrix} 0.75 & -0.25 \\ -0.1 & 0.6 \end{pmatrix}$ with determinant $0.45 - 0.025 = 0.425$, so $V(0) = 0.6/0.425 = 24/17 \approx 1.4118$ and $V(1) = 0.1/0.425 = 4/17 \approx 0.2353$. The first sweep gives $\mathbf{r} + 0.5\mathbf{P}\mathbf{0} = (1, 0)$; then $\mathbf{P}(1, 0)^\top = (0.5, 0.2)$ and the second sweep gives $(1.25, 0.1)$, on its way to the solution.
4. $|G_t| \le 1/(1 - 0.8) = 5$. The change of sweep $k$ is at most $0.8^{k-1}$, which is below $10^{-3}$ for $k - 1 \ge \ln 1000/\ln 1.25 \approx 30.96$, so for $k - 1 = 31$ and $k = 32$ (and $0.8^{31} \approx 9.9 \times 10^{-4}$). The error at that stop is at most $0.8 \times 10^{-3}/0.2 = 4 \times 10^{-3}$.
5. The largest lookahead is $2.0 + 5 \times 10^{-10}$, and the actions within $10^{-9}$ of it are the first two, so the smallest, action $0$, is chosen. If the second is $2.0 + 2 \times 10^{-9}$, the first is $2 \times 10^{-9}$ below the best, outside the tolerance, and action $1$ is chosen.
6. A sweep over $10$ pairs is $10 \times (2 \times 5 + 1) = 110$ FLOPs. The elimination is $\sum_{k=0}^{3}(4 - k)(1 + 2(5 - k)) = 44 + 27 + 14 + 5 = 90$ plus the back substitution $n^2 = 25$, so $115$, and the assembly $2n^2 = 50$, together $165$. A sweep of policy evaluation costs $5 \times 11 = 55$ FLOPs, so $3$ sweeps cost the same as one assembly and solve.
7. to 9. The checks are the tables of the blocks, to the printed digits; in 9 the FLOP counts of the elimination are $46$ for the forward sweep and $16$ for the back substitution at $n = 4$.
10. With the tolerance $10^{-3}$ value iteration stops after $64$ sweeps, with a change of $9.83 \times 10^{-4}$ and a guaranteed error of $\gamma\,\text{change}/(1 - \gamma) = 8.84 \times 10^{-3}$. The greedy policy is still $(1, 0, 0, 0)$: the lookaheads on an empty shelf are $6.742$ for one unit and $6.660$ for two, a gap of $0.082$, against $2\epsilon/(1 - \gamma) = 0.177$ from the bound. The bound alone does not prove the policy right, because $0.177 > 0.082$; only the observation that the ordering of the lookaheads is stable (the policy settled at sweep 1) or a final exact evaluation of the policy does.
