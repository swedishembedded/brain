<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Policy Gradients

*How do we directly adjust a policy so that good actions become more likely?*

A keeper who has learned, to four decimals, what every order on an empty shelf is worth will take the largest and keep taking it. That rule has a defect that has nothing to do with the accuracy of the numbers. A policy that takes a largest entry is a step function of the table behind it. Raise the value of ordering two units from $6.6678$ to $6.7501$ and her behaviour jumps from one unit to two; change it by a millionth and nothing happens at all. There is no slope anywhere that says which way to move, and a value learner that wants a better policy has to rebuild the whole table and hope the largest entry lands in the right place. The previous chapter ended by showing how fragile that is: a maximum inside every target, a bootstrap inside every update, and weights that may run away.

The alternative is to let the keeper choose at random, with a probability for every order, and to change the probabilities. The expected return is then a smooth function of the numbers that set them, and a smooth function has a gradient. The whole chapter is about that gradient: what it is, how to estimate it from the days a shop actually lives without ever writing down a probability of demand, why the estimate is so noisy that a subtraction of one number can cut its variance twelvefold, how a learned value function turns it into an actor and a critic, and how to take several steps on one batch of experience without running away from the data that justified them. The last of these is the clipped objective on which much policy-gradient training of language models rests, and it is where brain's `crates/rl` lives.

Six ideas from earlier are the tools. Answer them from memory; the answers are at the end of the chapter.

1. (Volume XII, Chapter 1, "Markov Decision Processes", return.) Three days pay $1$, $1$ and $1$ and nothing after. With $\gamma = 0.5$, what are $G_2$, $G_1$ and $G_0$?
2. (Volume XII, Chapter 2, "Learning Values from Experience", temporal-difference learning.) Write the TD error $\delta$ of one day, and say what a positive $\delta$ tells the learner about the state it started from.
3. (Volume I, Chapter 10, "Probability: Reasoning Under Uncertainty", variance.) One draw has variance $\sigma^2$. What is the variance of the mean of $N$ independent draws, and how many draws does it take to halve the standard error?
4. (Volume I, Chapter 13, "Information, Entropy and Surprise", KL divergence.) For $p = (0.5, 0.5)$ and $q = (0.75, 0.25)$, compute $\mathrm{KL}(p \,\|\, q)$ in nats.
5. (Volume III, Chapter 3, "Softmax, Logits and Cross-Entropy", softmax.) For $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$ the cross-entropy loss of class $c$ has gradient $\mathbf{p} - \mathbf{e}_c$ with respect to $\mathbf{z}$. What is the gradient of $\ln p_c$ itself?
6. (Volume V, Chapter 11, "Decoding: Turning Probabilities into Text", log-probabilities.) A sampled text of three tokens had the probabilities $0.5$, $0.25$ and $0.5$ at its three steps. What is its log-probability?

## Four orders and one smooth return

Start with the smallest decision the shop contains: the order on an empty shelf. The keeper may order $a = 0, 1, 2$ or $3$ units, the demand $D$ is $0$, $1$ or $2$ with probabilities $0.3$, $0.4$ and $0.3$, and what the order is worth is the reward of the day plus the discounted value of the shelf it leaves, $y(a, d) = r + \gamma V^\ast(s')$, with the optimal values of Chapter 1 standing in for everything that comes after. That makes it a bandit: four doors, a random payoff behind each, and the day is over.

<!-- worked: policy_gradients.one_day_bandit -->
The empty shelf (s = 0) as a one-day decision. The action a is the number of units ordered, a = 0 .. 3. The demand D is 0, 1 or 2 with probabilities 0.3, 0.4, 0.3. The outcome y(a, d) is the reward of the day plus gamma = 0.9 times the optimal value V*(s') of the shelf that is left, where V* is the exact solution of the previous chapter's system for the policy (1, 0, 0, 0) (6.7500, 9.2500, 11.1678, 11.6684 to four decimals) and the reward and the next shelf come from the rules of chapter 1. q(a) is the mean of y(a, d) over d = 0, 1, 2 in order. The policy is the softmax of four preferences theta, pi(a) = exp(theta_a) / sum_b exp(theta_b), computed in the stable form. J(theta) = sum_a pi(a) q(a), and the exact gradient is the vector with entries pi(k) (q(k) - J). Two preference vectors: theta = (0, 0, 0, 0) and theta_B = (0, 1, 0, -1), the one the estimators of the chapter are analysed at; at theta = 0 the policy is uniform and J = 6.1653.

| a | y(a, 0) | y(a, 1) | y(a, 2) | q(a) | pi(a) at 0 | pi(a) at theta_B | q(a) - J | dJ/dtheta_a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.0750 | 6.0750 | 6.0750 | 6.0750 | 0.2500 | 0.1966 | -0.4117 | -0.0810 |
| 1 | 4.8250 | 7.5750 | 7.5750 | 6.7500 | 0.2500 | 0.5344 | 0.2633 | 0.1407 |
| 2 | 3.5510 | 6.8250 | 9.5750 | 6.6678 | 0.2500 | 0.1966 | 0.1811 | 0.0356 |
| 3 | 1.0015 | 5.5510 | 8.8250 | 5.1684 | 0.2500 | 0.0723 | -1.3184 | -0.0954 |

At theta_B, J = 6.4867. Central differences of J with the step 1e-06 in each preference agree with the formula to 1.3e-09, below the criterion 1e-6. The four gradient entries sum to 0.0000: a change that raises every preference by the same amount changes nothing.
<!-- /worked -->

Two features of the table matter later. The order of nothing pays $6.0750$ whatever the demand: its payoff has no spread at all. The others pay between $1.0015$ and $9.5750$, and the means $q(a)$ differ by less than the spread of any one payoff, so a single day tells the keeper very little about which door is better. The best door is the order of one unit, at $6.7500$.

A **softmax policy** gives every order a number $\theta_a$, its preference, and chooses $a$ with probability $\pi(a) = e^{\theta_a} / \sum_b e^{\theta_b}$, the softmax of Volume III. The expected payoff is $J(\theta) = \sum_a \pi(a)\,q(a)$, and since $\pi$ is a smooth function of $\theta$, so is $J$. The derivative of a softmax output is $\partial \pi(a) / \partial \theta_k = \pi(a)\,(\mathbb{1}[a = k] - \pi(k))$, where $\mathbb{1}[\cdot]$ is $1$ when its condition holds and $0$ otherwise (it is the gradient of the fifth recall question times $\pi(a)$). Then

$$\frac{\partial J}{\partial \theta_k} = \sum_a q(a)\,\pi(a)\,(\mathbb{1}[a = k] - \pi(k)) = \pi(k)\,\big(q(k) - J\big).$$

The preference of an order rises exactly when its mean payoff is above the policy's own average $J$, and by an amount proportional to how likely the order already is. The printed column agrees with central differences of $J$ to $1.3 \times 10^{-9}$, and the four entries add to zero, because raising every preference equally changes no probability.

This is the exact gradient, and it uses $q(a)$, which the keeper does not have. What she has is one payoff per day from the door she chose. The question of the chapter is how to follow this gradient from those payoffs alone.

## The same gradient from single samples

Write the gradient as a sum over doors and then multiply and divide each term by $\pi(a)$:

$$\begin{aligned}
\nabla_\theta J &= \sum_a q(a)\,\nabla_\theta \pi(a) = \sum_a \pi(a)\,q(a)\,\frac{\nabla_\theta \pi(a)}{\pi(a)} \\
&= \sum_a \pi(a)\,q(a)\,\nabla_\theta \ln \pi(a).
\end{aligned}$$

The last form is an expectation: draw a door $a$ from $\pi$, draw its payoff $y$ (whose mean given $a$ is $q(a)$), and the vector $y\,\nabla_\theta \ln \pi(a)$ has the exact gradient as its mean. That is the **likelihood-ratio trick**, and it is the whole mechanism of this chapter: the derivative of a probability is the probability times the derivative of its logarithm, so a gradient of an expectation becomes an expectation of a gradient, which a sample can estimate. The vector $\nabla_\theta \ln \pi(a)$ is the **score** of the sampled door. For a softmax it is the gradient of the fifth recall question, $\mathbf{e}_a - \boldsymbol{\pi}$, and it asks for nothing but the policy's own probabilities. Nothing about the demand, the rewards or the shelf enters the estimate except the one payoff that happened.

One draw therefore gives the estimate $\hat{\mathbf{g}} = y\,(\mathbf{e}_a - \boldsymbol{\pi})$. Here are eight of them, at the preferences $\theta_B = (0, 1, 0, -1)$ that this chapter's bandit examples use from now on, showing the entry for the preference of the one-unit order (the others behave alike).

<!-- worked: policy_gradients.single_sample_estimates -->
Eight single-sample estimates of the gradient at theta_B = (0.0, 1.0, 0.0, -1.0), pi = (0.1966, 0.5344, 0.1966, 0.0723), J = 6.4867. Generator: the book's SplitMix64, seed 0xb4220d5c5d44f79a (12979951766388144026), one generator for the whole table. Each sample draws, in this order, one uniform() ua for the action (inverse CDF: the first a whose running sum of pi exceeds ua) and one uniform() ud for the demand (D = 0 if ud < 0.3, 1 if ud < 0.7, otherwise 2). The estimate is (y - b)(e_a - pi), with e_a the vector that is 1 at a and 0 elsewhere; the table shows its entry for the preference of the order of one unit (k = 1), for b = 0 and for b = J, and the mean of the eight (summed in order, divisor 8) beside the exact value of that entry.

| sample | ua | a | ud | d | y | estimate, b = 0 | estimate, b = J |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7056 | 1 | 0.5805 | 1 | 7.5750 | 3.527 | 0.507 |
| 1 | 0.7303 | 1 | 0.1926 | 0 | 4.8250 | 2.246 | -0.774 |
| 2 | 0.9388 | 3 | 0.2653 | 0 | 1.0015 | -0.535 | 2.932 |
| 3 | 0.9140 | 2 | 0.3743 | 1 | 6.8250 | -3.648 | -0.181 |
| 4 | 0.5526 | 1 | 0.5666 | 1 | 7.5750 | 3.527 | 0.507 |
| 5 | 0.4959 | 1 | 0.9882 | 2 | 7.5750 | 3.527 | 0.507 |
| 6 | 0.4406 | 1 | 0.3114 | 1 | 7.5750 | 3.527 | 0.507 |
| 7 | 0.5079 | 1 | 0.9993 | 2 | 7.5750 | 3.527 | 0.507 |
| mean of 8 |  |  |  |  |  | 1.962 | 0.564 |
| exact |  |  |  |  |  | 0.141 | 0.141 |
<!-- /worked -->

The column for $b = 0$ is the estimate just described, and it is a poor guide. Every draw of the one-unit order pushes its preference up, by $3.527$ for a payoff of $7.575$ and still by $2.246$ for the draw of sample $1$, whose payoff, $4.825$, is below the policy's average of $6.4867$; every draw of another order pushes the preference down, including sample $3$, whose payoff of $6.825$ is close to the average. The sign of the estimate follows which door was drawn, not whether it was a good draw, because the payoffs are mostly large and positive whichever door is drawn, and each is multiplied by a score whose sign is set by the door. The mean of the eight is $1.962$, fourteen times the exact $0.141$. The estimate is unbiased, and the next section shows what that is worth when the noise is this large. The column for $b = J$ is the same estimate after one subtraction, and it is the subject of the next section.

## A number that can be subtracted for free

The score has mean zero. Summing its probability-weighted values,

$$\begin{aligned}
\mathbb{E}[\,\mathbf{e}_a - \boldsymbol{\pi}\,] &= \sum_a \pi(a)\,(\mathbf{e}_a - \boldsymbol{\pi}) \\
&= \boldsymbol{\pi} - \Big(\sum_a \pi(a)\Big)\boldsymbol{\pi} = \mathbf{0}.
\end{aligned}$$

(The same fact is $\sum_a \nabla\pi(a) = \nabla 1 = 0$, which holds for every policy, not only a softmax.) So any number $b$ that does not depend on which door was drawn can be subtracted from the payoff without changing the mean of the estimate: $\mathbb{E}[(y - b)(\mathbf{e}_a - \boldsymbol{\pi})] = \mathbb{E}[y\,(\mathbf{e}_a - \boldsymbol{\pi})] - b\,\mathbf{0}$. This is **baseline subtraction**, and the number $b$ is the baseline. Its only constraint is independence from the action; it may depend on everything that came before. A baseline that peeked at the action would no longer cancel.

The mean is untouched; the variance is not. The squared length of the estimate is $(y - b)^2\,\lVert \mathbf{e}_a - \boldsymbol{\pi}\rVert^2$, so its expectation is a quadratic in $b$,

$$\mathbb{E}\,\lVert \hat{\mathbf{g}}_b \rVert^2 = \sum_{a, d} \pi(a)\,p(d)\,(y(a, d) - b)^2\,w_a,$$

with $w_a = \lVert \mathbf{e}_a - \boldsymbol{\pi}\rVert^2 = 1 - 2\pi(a) + \lVert\boldsymbol{\pi}\rVert^2$, and the total variance is that minus $\lVert \nabla J \rVert^2$, which does not depend on $b$. Setting the derivative in $b$ to zero shows the lowest point is the weighted average of the payoffs,

$$b^\ast = \frac{\sum_{a,d} \pi(a)\,p(d)\,w_a\,y(a, d)}{\sum_{a,d} \pi(a)\,p(d)\,w_a},$$

and with all weights equal it would be the plain mean $J$. The twelve outcomes $(a, d)$ can be enumerated, so for this bandit everything is exact. The table does it for five baselines, and checks the exact numbers against $20\,000$ sampled draws.

<!-- worked: policy_gradients.estimator_moments -->
The estimator (y - b)(e_a - pi) at theta_B = (0.0, 1.0, 0.0, -1.0) (pi = 0.1966, 0.5344, 0.1966, 0.0723, J = 6.4867, exact gradient entry for k = 1: 0.1407), for five constant baselines b: 0, 3, J, b* and 12, where b* = sum pi(a) p(d) y(a, d) |e_a - pi|^2 / sum pi(a) p(d) |e_a - pi|^2 = 6.2987. 'Exact' columns enumerate the twelve outcomes (a, d) with probability pi(a) p(d): the bias is the largest entry of |E[estimate] - gradient|, the total variance is E|estimate|^2 - |gradient|^2 (the sum of the four entry variances). 'Sampled' columns use 20000 draws of (a, d) from the generator of seed 0xed79ffb853303dd2 (one uniform() for the action, then one for the demand, in this order per draw; the same draws serve every b): the mean of entry k = 1 (summed in draw order, divisor 20000) and the sample total variance (divisor 20000 - 1).

| b | value | exact bias | exact total variance | sampled mean, k = 1 | sampled total variance |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0000 | below 1e-12 | 27.3334 | 0.1464 | 27.318 |
| 3 | 3.0000 | below 1e-12 | 9.1417 | 0.1445 | 9.145 |
| J | 6.4867 | below 1e-12 | 2.2887 | 0.1422 | 2.311 |
| b* | 6.2987 | below 1e-12 | 2.2663 | 0.1423 | 2.287 |
| 12 | 12.0000 | below 1e-12 | 22.8033 | 0.1386 | 22.849 |
<!-- /worked -->

![The total variance of one policy-gradient estimate on the one-day bandit plotted against the constant baseline b, from -1 to 14: a parabola with its minimum of 2.27 at b* = 6.30. The points for b = 0 (27.33), b equal to J = 6.49 (2.29) and b = 12 (22.80) are marked.](img/baseline-variance.svg)

Three things stand out. All five baselines have the same mean, to $10^{-12}$ (the sampled means of the entry $k = 1$ lie within $0.006$ of the exact $0.1407$, well inside the sampling error of $20\,000$ draws, whose standard error is at most $\sqrt{27.33 / 20\,000} = 0.037$ since an entry's variance cannot exceed the total; the same draws serve every baseline, so the five columns err together). The variance is another matter: $27.33$ with no baseline, $2.29$ with the policy's own value $J$, a factor of $11.9$. The optimal constant, $b^\ast = 6.2987$, is only a hair better than $J = 6.4867$, $2.266$ against $2.289$. The weights $w_a$ pull it down, since the likely door has the smallest $w_a$ and the unlikely ones the largest, but the variance is nearly flat at the bottom of its parabola. And a bad constant is nearly as bad as none: $b = 12$ is not biased, but its variance, $22.80$, is within a fifth of the $27.33$ of $b = 0$. A baseline pays only when it sits near where the payoffs are.

How much that is worth is counted in samples. The mean of $N$ independent estimates has error $\sqrt{V/N}$, so the smallest $N$ that brings the root-mean-square error to a fraction $\rho$ of the gradient's own length is $V / (\rho^2 \lVert\nabla J\rVert^2)$.

<!-- worked: policy_gradients.samples_for_accuracy -->
The mean of N independent single-sample estimates has root-mean-square error sqrt(V/N), V the exact total variance of the previous table. The smallest N with sqrt(V/N) at most rho |gradient| is ceil(V / (rho^2 |gradient|^2)), with |gradient|^2 = 0.03671 at theta_B.

| baseline b | N for rho = 0.5 | N for rho = 0.1 |
|---|---:|---:|
| 0 | 2,979 | 74,456 |
| 3 | 997 | 24,902 |
| J | 250 | 6,235 |
| b* | 247 | 6,174 |
| 12 | 2,485 | 62,116 |
<!-- /worked -->

To know the direction of this gradient to within half its own length takes $2\,979$ days with no baseline and $250$ days with $J$; to within a tenth, $74\,456$ against $6\,235$. The baseline does not change what the estimator estimates. It changes how many days of experience a keeper must live to know it, by the same factor of twelve.

Subtracting $J$ turns the payoff into $y - J$, whose mean for door $a$ is $q(a) - J$. That difference is the **advantage** of the door: how much better than the policy's average it is. The exact gradient above is $\pi(k)$ times the advantage of door $k$, and the estimate $(y - J)(\mathbf{e}_a - \boldsymbol{\pi})$ is the advantage of one draw times a score. Advantages are signed: positive for an action that beats the policy's average, negative for one that does not, and zero on average under the policy. A policy gradient with advantages says what the exact gradient said, raise what is better than average and lower what is worse, and does it from samples.

With the baseline the keeper's estimate of the earlier table has the sign that a reader would give it: $+0.507$ for a draw of the one-unit order that paid $7.575$, $-0.774$ for the draw of the same order that paid $4.825$. The next section asks what replaces the single number $J$ in a shop whose days are not independent.

## From one day to a whole life: the policy gradient theorem

The bandit lasted one day. A real episode is a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$, and the policy acts in every state on the way. The return that the policy should maximise is the expected discounted sum from the start distribution $\mu_0$ (the chapter uses a uniform start over the four shelves, as Chapter 2 did):

$$J(\theta) = \mathbb{E}_\tau \Big[\sum_{t \ge 0} \gamma^t r_t\Big], \qquad \pi_\theta(a \mid s) = \text{softmax of } \theta[s].$$

A shelf's day has a random demand, and so does a trajectory have a probability. Write the demand of day $t$ as $d_t$; the state $s_{t+1}$ and the reward $r_t$ are functions of $(s_t, a_t, d_t)$. Then the probability of a life up to day $T - 1$ is

$$P_\theta(\tau) = \mu_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\,p(d_t).$$

Only one family of factors contains $\theta$: the policy's choices. The start distribution and the demand do not. That is the single fact the theorem rests on. The likelihood-ratio trick of the previous section applies to the whole life at once, $\nabla P_\theta = P_\theta \nabla \ln P_\theta$, and the logarithm of a product is a sum in which every factor free of $\theta$ has a zero derivative:

$$\nabla_\theta \ln P_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t).$$

So $\nabla J = \mathbb{E}\big[\,(\sum_t \gamma^t r_t)\,(\sum_k \nabla \ln \pi_\theta(a_k \mid s_k))\,\big]$: the whole return, times the sum of every score of the life. That is already an estimator, and a poor one, because it multiplies the score of day $k$ by rewards paid on days before it, which the choice of day $k$ cannot have caused. They add variance, and they add no mean. Fix a day $k$ and a reward $r_t$ with $t < k$. Condition on everything up to the state $s_k$: the reward is then fixed, and what remains random is the action, whose score has mean zero by the identity of the baseline section. So $\mathbb{E}[r_t \nabla \ln \pi(a_k \mid s_k)] = 0$ for $t < k$, those terms can be dropped, and what is left of day $k$ is the sum of the rewards from day $k$ on, which factors as $\gamma^k G_k$ with $G_k = \sum_{t \ge k} \gamma^{t-k} r_t$ the return of Chapter 1:

$$\nabla_\theta J = \mathbb{E}\Big[\sum_{k \ge 0} \gamma^k\,G_k\,\nabla_\theta \ln \pi_\theta(a_k \mid s_k)\Big].$$

(Taking $T \to \infty$ is harmless for $\gamma < 1$.) The same argument shows that a baseline $b(s_k)$ that depends on the state alone can be subtracted from $G_k$: given $s_k$ it is a constant. The mean of $G_k$ given $(s_k, a_k)$ is the action value $q^\pi(s_k, a_k)$ of Chapter 1 (the one-step lookahead evaluated at $V^\pi$), and grouping the days by the state they visit gives a statement about states and actions only. Let $d_\pi(s) = \sum_{t \ge 0} \gamma^t \Pr(S_t = s)$ be the **discounted visitation** of state $s$, the expected discounted number of days spent there (the letter $d$ with a state as its argument; the demand of a day, $d_t$ or the $d$ of $y(a, d)$, is a different quantity). Then the **policy gradient theorem** says

$$\begin{aligned}
\nabla_\theta J &= \sum_s d_\pi(s) \sum_a q^\pi(s, a)\,\nabla_\theta \pi_\theta(a \mid s) \\
&= \sum_s d_\pi(s) \sum_a \pi_\theta(a \mid s)\,A^\pi(s, a)\,\nabla_\theta \ln \pi_\theta(a \mid s),
\end{aligned}$$

where $A^\pi(s, a) = q^\pi(s, a) - V^\pi(s)$ is the advantage, now of an action in a state; the second form subtracts $V^\pi(s)$, a baseline free to subtract because $\sum_a \nabla\pi(a \mid s) = 0$. For the tabular softmax the preferences of different states do not interact, and the gradient with respect to the preference of action $a$ in state $s$ reduces to

$$\frac{\partial J}{\partial \theta[s][a]} = d_\pi(s)\,\pi_\theta(a \mid s)\,A^\pi(s, a).$$

It is the bandit's $\pi(k)(q(k) - J)$ again, weighted by how much of the shop's discounted time is spent in the state.

One worry deserves an answer, because it is the reason the theorem was hard to find. The value of the policy depends on $\theta$ through two channels: it changes which action is taken in each state, and it changes which states are visited, and that second channel is the distribution $d_\pi$ itself. The theorem has a term for the first and no term for the second. The derivation shows why that is right. The visitation was never a factor that had to be differentiated; it appeared only in the last step, as a way of regrouping an expectation that the likelihood-ratio trick had already differentiated. The dependence of the visit probabilities on $\theta$ is carried by the scores of the earlier days: the probability of being in a state on day $k$ is a product of earlier policy factors and earlier demand factors, and the derivative of that product is the sum of the earlier scores, which are the terms of the earlier days in the sum that the theorem already contains. Nothing is missing; nothing is counted twice. The check that does not rely on the argument is numeric. Compute the exact return $J(\theta)$ of the shop by solving the linear system of Chapter 1 for a stochastic policy, differentiate it by finite differences (which see both channels), and compare with the theorem, which sees only one.

<!-- worked: policy_gradients.softmax_policy_gradient -->
The shop (gamma = 0.9) with a tabular softmax policy: pi(a | s) = softmax of the preferences theta[s][0 .. n_act[s] - 1]. Preferences: theta[0] = (0.0, 0.5, 0.0, -0.5), theta[1] = (0.0, 0.5, 0.0), theta[2] = (0.0, 0.5), theta[3] = (0.0,). The start shelf is uniform, mu_0(s) = 1/4, and J(theta) = (1/4) sum_s V(s), with V the exact solution of (I - gamma P_pi) V = r_pi, where row s of P_pi is sum_a pi(a | s) P(. | s, a) and r_pi(s) = sum_a pi(a | s) R(s, a) (sums over a in increasing order; elimination with partial pivoting). d(s) is the discounted visitation, the solution of (I - gamma P_pi)^T d = mu_0, so d(s) = sum_t gamma^t Pr(S_t = s) and the four values add to 1/(1 - gamma) = 10. The theorem column is d(s) pi(a | s) (q(s, a) - V(s)), with q(s, a) = R(s, a) + gamma sum_j P(j | s, a) V(j). The finite-difference column is (J(theta + h e) - J(theta - h e)) / (2h) with h = 1e-05, e the unit vector of the entry, J recomputed from the linear system each time.

| s | a | theta | pi | q(s, a) | V(s) | A = q - V | d(s) | theorem | finite difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0 | 0.2350 | 0.1463 | 0.1625 | -0.0163 | 2.9361 | -0.01122 | -0.01122 |
| 0 | 1 | 0.5 | 0.3875 | 0.6618 | 0.1625 | 0.4992 | 2.9361 | 0.56795 | 0.56795 |
| 0 | 2 | 0.0 | 0.2350 | 0.2293 | 0.1625 | 0.0668 | 2.9361 | 0.04608 | 0.04608 |
| 0 | 3 | -0.5 | 0.1425 | -1.2779 | 0.1625 | -1.4404 | 2.9361 | -0.60281 | -0.60281 |
| 1 | 0 | 0.0 | 0.2741 | 3.1618 | 2.0718 | 1.0900 | 3.0383 | 0.90763 | 0.90763 |
| 1 | 1 | 0.5 | 0.4519 | 2.2293 | 2.0718 | 0.1575 | 3.0383 | 0.21624 | 0.21624 |
| 1 | 2 | 0.0 | 0.2741 | 0.7221 | 2.0718 | -1.3497 | 3.0383 | -1.12387 | -1.12387 |
| 2 | 0 | 0.0 | 0.3775 | 4.7293 | 3.4799 | 1.2494 | 2.6177 | 1.23474 | 1.23474 |
| 2 | 1 | 0.5 | 0.6225 | 2.7221 | 3.4799 | -0.7578 | 2.6177 | -1.23474 | -1.23474 |
| 3 | 0 | 0.0 | 1.0000 | 5.2221 | 5.2221 | 0.0000 | 1.4079 | 0.00000 | 0.00000 |

J(theta) = 2.7341. The largest difference between the two columns over the ten entries is 1.1e-10, below the criterion 1e-7.
<!-- /worked -->

The two columns agree to $1.1 \times 10^{-10}$, below the criterion of $10^{-7}$, in all ten entries; the entry of the single order on a full shelf is exactly zero, because a policy with one action has no preference to change. The visitation $d_\pi$ adds up to $10 = 1/(1 - \gamma)$, as it must, and it weights the states unequally: the shelf of one unit, which the policy reaches through every sale, carries $3.04$ of those ten days and the full shelf only $1.41$.

The factor $\gamma^k$ in the theorem deserves a remark, because most implementations leave it out and brain's language-model code has none. It is what makes the formula the gradient of the discounted return from the start distribution: with it, the day $k$ weighs $\gamma^k$, the same as in the objective. Without it every day of the life weighs the same, which for $\gamma < 1$ is the gradient of a different objective; with $\gamma = 1$ and an episode that ends, as for a completion of a language model, the factor is $1$ and nothing is dropped. Here the factor is kept, and the table above is the gradient of $J$ exactly.

## REINFORCE and its learning curve

The theorem is a formula for the gradient; an algorithm needs the next step, which is the one the likelihood-ratio trick already took: replace the expectation by a sample. Draw an episode under the policy, compute the return from every day on, and move every preference in the direction of its own score times the return. This is **REINFORCE**, the Monte Carlo policy gradient. With a baseline it reads, per episode,

$$\hat{\mathbf{g}} = \sum_{t=0}^{T-1} \gamma^t\,(G_t - b(s_t))\,\nabla_\theta \ln \pi_\theta(a_t \mid s_t), \qquad \theta \leftarrow \theta + \eta\,\hat{\mathbf{g}}.$$

Up to the truncation of the episode, whose later days weigh less than $3 \times 10^{-5}$, the estimator is unbiased for the gradient of the return by the derivation above, for any state-dependent baseline. In pseudocode (the episode lists the day `(s, a, r, next)` for every day, `theta[s]` is the list of preferences of state `s`, and `softmax` is the routine of Volume III over a list):

```text
function sample_index(p: array<real>[m], rng: Generator) -> int:
    u = rng.uniform()
    c = 0.0
    for a in 0 .. m:
        c = c + p[a]
        if u < c:
            return a
    return m - 1

function reinforce_episode(theta: list<list<real>>, V: array<real>[n],
                           episode: list<(int, int, real, int)>, gamma: real,
                           eta: real, alpha_c: real, use_baseline: bool) -> none:
    require 0 <= gamma and gamma < 1 and eta > 0 and alpha_c > 0
    T = len(episode)
    ret = zeros(real, [T])
    acc = 0.0
    for t in T - 1 down to 0:
        acc = episode[t][2] + gamma * acc
        ret[t] = acc
    grad = zeros_like(theta)
    disc = 1.0
    for t in 0 .. T:
        (s, a, r, next) = episode[t]
        p = softmax(theta[s])
        adv = ret[t] - (V[s] if use_baseline else 0.0)
        for k in 0 .. len(theta[s]):
            grad[s][k] = grad[s][k] + disc * adv * ((1.0 if k == a else 0.0) - p[k])
        disc = disc * gamma
    for s in 0 .. n:
        for k in 0 .. len(theta[s]):
            theta[s][k] = theta[s][k] + eta * grad[s][k]
    if use_baseline:
        for t in 0 .. T:
            (s, a, r, next) = episode[t]
            V[s] = V[s] + alpha_c * (ret[t] - V[s])
```

Every convention a reader could guess wrong is fixed here. The probabilities `p` are taken from the preferences as they stood at the start of the episode, since no step is applied until the loop over days is done. The power $\gamma^t$ is a running product, `disc`. The gradient is the sum over the days of the episode, not the mean (so a longer episode takes a larger step), and the step is applied once per episode. The returns are reward-to-go, truncated at the last day of the episode: the days after it count as zero, as in Chapter 2. With a baseline, the critic `V` is the one of Chapter 2 learned by Monte Carlo, every visit, with a constant step $\alpha_c$; it is updated after the policy step, so the baseline of an episode is never fitted to that episode's own returns (which would make it depend on the actions taken).

Three learners run on the whole shop, each ten times from its own seed, with all conventions of the table's heading. The two REINFORCE learners are the algorithm above, with $b = 0$ and with the learned $b = V$. The third, taken up in the next section, is the one-step actor-critic. The table prints the exact return $J(\theta)$, recomputed from the linear system after every episode, as mean and standard deviation over the runs.

<!-- worked: policy_gradients.policy_gradient_runs -->
The shop, gamma = 0.9, tabular softmax policy with preferences theta[s][a] starting at 0 (the uniform policy over the legal orders, J = 2.6785); the best policy (1, 0, 0, 0) has J = 9.7090. An episode is 100 days: a start shelf from integer(4), then per day one uniform() for the action (inverse CDF over pi(. | s)) and one for the demand, in this order. Three learners, 10 runs each of 200 episodes, every run with its own generator (the SplitMix64 of the book seeded with seed_of of the name policy_gradients.runs.<method>.run<i>, method = reinforce, baseline or actor_critic, i = 0 .. 9; every run restarts from its own seed, seeds below). REINFORCE: after an episode the return of day t is the reward-to-go G_t = r_t + gamma G_(t+1) from G_100 = 0, the gradient is the sum over the days t = 0 .. 99 in order of gamma^t (G_t - b) (e_(a_t) - pi(. | s_t)) added to the preferences of s_t (gamma^t is a running product, all pi taken from the preferences as they stood at the start of the episode), and the preferences move by eta = 0.05 times the sum, once per episode. The baseline is b = 0 for REINFORCE and b = V(s_t) for the second method, a critic that starts at 0 and, after each episode's policy step, moves by alpha_c = 0.05 times (G_t - V(s_t)) for every day in order (every visit, applied immediately). Actor-critic updates every day before the next is drawn: delta = r + gamma V(s') - V(s) with the critic as it stands, theta[s] += eta gamma^t delta (e_a - pi(. | s)) with pi from before the step, then V(s) += alpha_c delta, in this order; one policy step per day and not per episode. J(theta) is recomputed exactly from the linear system after every episode. Each cell is the mean / standard deviation of J over the 10 runs (summed in run order; divisors 10 and 9) after the stated number of episodes. The last row counts the runs whose greedy policy (the largest preference of each shelf, the smaller action on a tie) is (1, 0, 0, 0).

| episodes | days | REINFORCE | REINFORCE with baseline | actor-critic |
|---:|---:|---:|---:|---:|
| 1 | 100 | 3.05 / 0.47 | 2.79 / 0.37 | 3.25 / 0.49 |
| 5 | 500 | 3.90 / 1.14 | 3.60 / 0.85 | 4.49 / 0.58 |
| 10 | 1000 | 4.58 / 1.50 | 4.89 / 0.77 | 5.43 / 0.57 |
| 25 | 2500 | 5.59 / 1.33 | 6.52 / 0.97 | 7.05 / 0.36 |
| 50 | 5000 | 7.37 / 1.39 | 8.06 / 0.99 | 8.47 / 0.27 |
| 100 | 10000 | 8.33 / 1.02 | 8.79 / 0.43 | 9.05 / 0.38 |
| 200 | 20000 | 8.62 / 1.19 | 9.23 / 0.42 | 9.42 / 0.10 |
| greedy = (1, 0, 0, 0) |  | 5 of 10 | 5 of 10 | 9 of 10 |

Every number learned from samples is printed with the digits that the run and four copies of it share: the copies start with the first preference (theta[0][0] of the shop, theta_B[0] of the bandit) changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the nudges of stable.py), and a digit is printed only if all five agree on it ('about x' keeps the leading digits that survive, 'unstable' means that none does). The copies use the same generator and so draw the same numbers; they differ only if a nudge flips a comparison. A run is a deterministic computation, so another implementation reproduces the printed digits.

| run | REINFORCE | REINFORCE with baseline | actor-critic |
|---:|---:|---:|---:|
| 0 | 0x0f5928bc90f5dccb | 0x1783dc9d1ebe4641 | 0xb8257742ff0fc0fb |
| 1 | 0x82e4671b6e731ca7 | 0x35984fffcb6a8b8e | 0x7a60aae1f02ae4cd |
| 2 | 0x6f02893e99791587 | 0xf60462659e77dc2f | 0x8f8c2611bf797363 |
| 3 | 0x2d6736f2e2fc30fa | 0xd22e0f546471058e | 0xbbe0a1fbb9a1d12a |
| 4 | 0x92493361e69ec06d | 0x58cdb0dc8bbfab42 | 0xb9a1c0598d57cd30 |
| 5 | 0x0506df984cfecf50 | 0xe07a7e2361486842 | 0xd6bf55d452a7a5a9 |
| 6 | 0x3342a06475520b25 | 0x25b58ffc17796986 | 0xa61983cc6efec67a |
| 7 | 0xf32b92d032b057f2 | 0xabe45c39574037f7 | 0xadc197cc4e858a95 |
| 8 | 0x9447ce3b276c5458 | 0x18e72a65fa38fa81 | 0x12fc816bf46b3497 |
| 9 | 0x085bb2de63cd0b2d | 0x2eb06de8687d1c8d | 0x08afc7e2aea0aab3 |
<!-- /worked -->

![Mean exact return against the number of episodes on a logarithmic axis from 1 to 200 for three learners. All three start from the uniform policy's 2.68 and stand between 2.8 and 3.3 after the first episode. Actor-critic rises fastest, reaching about 9.4; REINFORCE with a baseline reaches about 9.2; plain REINFORCE reaches about 8.6, below the dashed line at the best policy's return of 9.71.](img/policy-gradient-runs.svg)

The uniform policy starts at $J = 2.6785$ and the best policy has $9.7090$. After $200$ episodes, twenty thousand days, plain REINFORCE is at $8.62$ with a spread of $1.19$ between runs, and the same algorithm with a learned baseline is at $9.23$ with a spread of $0.42$. The baseline does what the variance table promised: the runs agree with one another much better, and the mean is higher. It does not make the learner fast. Both REINFORCE learners need tens of thousands of days to find a policy that the exact computation of Chapter 1 finds in three solves, and in half the runs of both the greedy policy is still not the optimal one at episode $200$. That is the price of the sampling that the policy gradient lives on: the gradient is correct on average, and a thousand days are a long time to average over.

## A critic that replaces the return

REINFORCE waits for the end of the episode to learn what a day was worth, and a day's return adds up a hundred days of random demand. The previous chapter already built the remedy: a value function that predicts the return, learned by temporal differences. Use it twice. As a baseline, which is all that the previous section asked of it. And as a replacement for the sampled tail of the return, by the same trick that turned Monte Carlo into TD: write the advantage of the action just taken as the TD error,

$$\hat{A}_t = \delta_t = r_t + \gamma\,\hat v(s_{t+1}) - \hat v(s_t).$$

If the critic $\hat v$ is exactly $V^\pi$, then $\mathbb{E}[\delta_t \mid s_t, a_t] = q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$: by the Bellman equation the mean of $r + \gamma V^\pi(s')$ is the action value. The estimate then has the right mean and, since it contains one day of randomness in place of a hundred, a small variance. If the critic is wrong, the estimate is biased, by the part of its error that differs between the actions' successors; that is the bias of TD that Chapter 2 measured, now inside a gradient.

A learner that maintains both a policy and a value function, updating the first with the TD error of the second, is an **actor-critic**: the policy is the actor and the value function the critic. The shop's version updates after every day. In pseudocode, with the same conventions:

```text
function actor_critic_episode(theta: list<list<real>>, V: array<real>[n], days: int,
                              gamma: real, eta: real, alpha_c: real, rng: Generator) -> none:
    require 0 <= gamma and gamma < 1 and eta > 0 and alpha_c > 0 and days > 0
    s = rng.integer(n)
    disc = 1.0
    for t in 0 .. days:
        p = softmax(theta[s])
        a = sample_index(p, rng)
        (r, next) = sample_day(s, a, rng)
        delta = r + gamma * V[next] - V[s]
        for k in 0 .. len(theta[s]):
            theta[s][k] = theta[s][k] + eta * disc * delta * ((1.0 if k == a else 0.0) - p[k])
        V[s] = V[s] + alpha_c * delta
        disc = disc * gamma
        s = next
```

The order of events matters and is part of the definition: the action is drawn, then the demand, then $\delta$ is computed from the critic as it stands, then the actor moves along the score of the action just taken (with $p$ from before the step), and then the critic moves by its own TD step. The third column of the learning table is this loop, with $\eta = 0.05$ and $\alpha_c = 0.05$ as for the others. It reaches $7.05$ after $25$ episodes, where plain REINFORCE has $5.59$ with a spread nearly four times larger, $9.05$ after a hundred, and $9.42$ after two hundred with a spread of $0.10$ between runs; in $9$ of $10$ runs the greedy policy is the optimal one.

The table does not separate two causes. The actor-critic differs from REINFORCE in how often the policy moves, a step after every day (a hundred per episode) against one step per episode, each along the freshest policy, and in what it multiplies the score by, one day's TD error of a learned critic against a sampled return of up to a hundred days. Either may account for part of the gap, and the table cannot say how much; Exercise 9 holds the number of policy steps equal. What the table does show is the effect that matters in practice: the whole life of an episode need not be known before a day teaches the policy something.

## How far to look ahead: generalised advantage estimation

REINFORCE uses the whole remaining return, $G_t - \hat v(s_t)$, which is unbiased for any critic and has the variance of a hundred days. The actor-critic uses one TD error, $\delta_t$, which has the variance of one day and the bias of the critic. Between them lie estimates that use $n$ days of reward and the critic after that. **Generalised advantage estimation** weights all of them at once with one number $\lambda \in [0, 1]$:

$$\hat{A}_t^{\lambda} = \sum_{k \ge 0} (\gamma\lambda)^k\,\delta_{t+k},$$

which is computed backwards as $\hat{A}_t = \delta_t + \gamma\lambda\,\hat{A}_{t+1}$.

The endpoints are the two estimators already met. With $\lambda = 0$ only $\delta_t$ remains. With $\lambda = 1$ the sum telescopes: $\sum_k \gamma^k (r_{t+k} + \gamma \hat v(s_{t+k+1}) - \hat v(s_{t+k}))$ keeps every reward, discounted, and the critic's terms cancel in pairs except the first, so the sum is $G_t - \hat v(s_t)$, REINFORCE with the critic as baseline. The routine:

```text
function gae(rewards: array<real>[T], values: array<real>[T], last_value: real,
             gamma: real, lam: real) -> array<real>[T]:
    require 0 <= gamma and gamma < 1 and 0 <= lam and lam <= 1
    adv = zeros(real, [T])
    acc = 0.0
    for t in T - 1 down to 0:
        next = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next - values[t]
        acc = delta + gamma * lam * acc
        adv[t] = acc
    return adv
```

`last_value` is the critic's estimate of the state where the batch was cut: $0$ if the task really ended, the critic's value if the episode was only truncated (Chapter 2 gave the reason). A three-day example checks the endpoints by hand.

<!-- worked: policy_gradients.gae_by_hand -->
Three days of one episode that was cut off after the third, gamma = 0.9. r_t is the reward, V(s_t) the critic's value of the day's state and V(s_(t+1)) the critic's value of the next state, which for the last day is the estimate of the state where the episode was cut, 0.4. delta_t = r_t + gamma V(s_(t+1)) - V(s_t). The advantage estimate for lambda is computed backwards from the last day as A_t = delta_t + gamma lambda A_(t+1) with A_3 = 0. The last column is G_t - V(s_t), with G_t = r_t + gamma G_(t+1) bootstrapped from 0.4 at the cut.

| t | r_t | V(s_t) | V(s_(t+1)) | delta_t | A, lambda = 0 | A, lambda = 0.5 | A, lambda = 1 | G_t - V(s_t) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0 | 0.5 | 1.0 | 1.400 | 1.4000 | 1.4684 | 2.4116 | 2.4116 |
| 1 | 0.0 | 1.0 | 0.2 | -0.820 | -0.8200 | 0.1520 | 1.1240 | 1.1240 |
| 2 | 2.0 | 0.2 | 0.4 | 2.160 | 2.1600 | 2.1600 | 2.1600 | 2.1600 |
<!-- /worked -->

The first column with $\lambda = 0$ is $\delta_t$; the last two, for $\lambda = 1$, are $G_t - \hat v(s_t)$ computed independently, and they agree in every row. The middle column mixes: $\hat{A}_1 = -0.820 + 0.9 \times 0.5 \times 2.160 = 0.152$.

What does the dial trade? Take the shop under the stochastic policy of the theorem check, make a given order on the empty shelf on day $0$, follow the policy for a hundred days, and compute the estimate with two critics: the exact $V^\pi$, and the exact values of the uniform policy, which is what a critic that has not yet caught up with the actor looks like. The spread of the estimate over many rollouts is sampled; the bias is exact, from the Markov chain.

<!-- worked: policy_gradients.advantage_estimates -->
Advantage estimates for the order a made on the empty shelf under the softmax policy of the theorem check (preferences ((0.0, 0.5, 0.0, -0.5), (0.0, 0.5, 0.0), (0.0, 0.5), (0.0,))); the exact advantage is A(0, a) = q(0, a) - V(0) = (-0.0163, 0.4992, 0.0668, -1.4404) for a = 0 .. 3. Two critics: the exact V of the policy and the stale critic V0, the exact values of the uniform policy (V0(0) = -0.2229 against V(0) = 0.1625). The estimate is A_hat = sum_(k < 100) (gamma lambda)^k delta_k with delta_k = r_k + gamma Vhat(s_(k+1)) - Vhat(s_k) along a rollout that makes the order a on day 0 and follows the policy afterwards, the critic's value of the state reached after the 100th day closing the sum (computed backwards, A = delta + gamma lambda A). Sampled: 2000 rollouts for each a = 0 .. 3 in this order, from the generator of seed 0x88bc19c249e89df0; each day after the first draws one uniform() for the action (inverse CDF) and then one for the demand, and the first day only the demand; the same rollouts serve both critics and every lambda. 'sd' is the root of the mean over the four orders of the variance of A_hat (divisor 2000 - 1). 'Bias' is exact, from the Markov chain (E[A_hat] - A for the infinite sum): the four columns are for the stale critic and 'spread' is their largest minus smallest; the last column is the largest absolute bias of the exact critic.

| lambda | sd, exact critic | sd, stale critic | bias a = 0 | bias a = 1 | bias a = 2 | bias a = 3 | spread | exact critic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.09 | 1.97 | 0.039 | 0.106 | 0.296 | 0.450 | 0.412 | below 1e-12 |
| 0.5 | 2.44 | 2.35 | 0.144 | 0.187 | 0.302 | 0.406 | 0.261 | below 1e-12 |
| 0.9 | 4.01 | 3.98 | 0.300 | 0.311 | 0.339 | 0.366 | 0.067 | below 1e-12 |
| 1 | 5.56 | 5.56 | 0.385 | 0.385 | 0.385 | 0.385 | 0.000 | below 1e-12 |
<!-- /worked -->

With the exact critic every $\lambda$ is unbiased (the last column checks this to $10^{-12}$), and the standard deviation grows with $\lambda$ from $2.09$ to $5.56$: the more real days enter, the more demand noise does. With the stale critic the picture splits. Its bias at $\lambda = 0$ is $(0.039, 0.106, 0.296, 0.450)$ for the four orders, and what misleads the actor is the difference between them, $0.412$ from smallest to largest: the stale critic is too low on the empty shelf and too high on the shelf of two units, so the orders that lead to fuller shelves look better than they are, and a policy gradient built on it favours them. At $\lambda = 1$ the bias is the same $0.385$ for every order, which is a constant shift, and by the baseline argument a constant shift of the advantage of every action in a state leaves the gradient alone. In between, $\lambda = 0.9$ cuts the harmful spread to $0.067$ at a standard deviation of $4.0$, against $5.56$ for the full return. That is the use of $\lambda$: with a critic you trust, small; with one you do not, large; and most training sits at $0.95$ or so, trading a little of the critic's bias for most of its variance. brain keeps this estimator in `crates/decide/src/value.rs`, as the last section of Chapter 2 mentioned; its tests pin the two endpoints, $\lambda = 0$ (the TD error) and $\lambda = 1$ (the return minus the baseline), and the bootstrap from a truncated episode.

## Reusing a batch without running away

A day of experience is expensive: for a language model it is a sampled completion, scored by a checker. After the estimate of the gradient has been computed from a batch, one step along it uses the batch once and throws it away. Several steps on the same batch would be cheaper per day, and the plain policy gradient gives no licence for them. The estimate is an expectation under the policy that produced the batch; after one step the policy is another one, the actions in the batch are no longer sampled from it, and the estimate has no meaning.

There is an identity that repairs the first part of this. If the batch was drawn from $\pi_{\mathrm{old}}$ but the policy being evaluated is $\pi_\theta$, then reweighting each sample by the **importance ratio** $\rho = \pi_\theta(a \mid s) / \pi_{\mathrm{old}}(a \mid s)$ turns the expectation under $\pi_{\mathrm{old}}$ into the expectation under $\pi_\theta$: $\sum_a \pi_{\mathrm{old}}(a)\,\rho(a)\,f(a) = \sum_a \pi_\theta(a)\,f(a)$. The surrogate

$$L(\theta) = \frac{1}{N}\sum_{i=1}^{N} \rho_i(\theta)\,\hat A_i, \qquad \rho_i = \frac{\pi_\theta(a_i \mid s_i)}{\pi_{\mathrm{old}}(a_i \mid s_i)},$$

can be evaluated at any $\theta$ from the fixed batch. At $\theta = \theta_{\mathrm{old}}$ every ratio is $1$, and since $\nabla \rho_i = \rho_i\,\nabla \ln \pi_\theta(a_i \mid s_i)$, its gradient is $\frac{1}{N}\sum_i \hat A_i\,\nabla \ln \pi(a_i \mid s_i)$: exactly the policy gradient estimate. So $L$ is a function whose gradient at the starting point is the right one and which can be climbed for many steps.

It is only right near the start. Away from $\theta_{\mathrm{old}}$ two things fail, and neither is in the identity: the advantages were computed under the old policy, and the states the policy visits change when the policy does, which the batch never sees. Worse, nothing in $L$ pushes back. Gradient ascent on it will take the ratio of a sample with a positive advantage as high as the preferences allow. The next table shows how it fails in practice: one batch of $64$ samples from $\pi_{\mathrm{old}} = \mathrm{softmax}(\theta_B)$, with the batch mean subtracted as the baseline.

<!-- worked: policy_gradients.ppo_batch -->
One batch of 64 samples from the behaviour policy pi_old = softmax(0.0, 1.0, 0.0, -1.0) = (0.1966, 0.5344, 0.1966, 0.0723). Generator: the book's SplitMix64, seed 0xb673cd583c1bfd66 (13147077516073893222); per sample one uniform() for the action (inverse CDF) and then one for the demand (D = 0 if below 0.3, 1 if below 0.7, otherwise 2), the samples in drawing order. The outcome is y(a, d) of the one-day bandit and the advantage of a sample is y minus the batch mean 6.8372 (summed in sample order, divisor 64). The objective depends on a sample only through (a, A), so the table lists each outcome that occurs with its count.

| a | d | samples | y | advantage A |
|---|---:|---:|---:|---:|
| 0 | 0 | 4 | 6.0750 | -0.7622 |
| 0 | 1 | 4 | 6.0750 | -0.7622 |
| 0 | 2 | 6 | 6.0750 | -0.7622 |
| 1 | 0 | 5 | 4.8250 | -2.0122 |
| 1 | 1 | 13 | 7.5750 | 0.7378 |
| 1 | 2 | 14 | 7.5750 | 0.7378 |
| 2 | 0 | 3 | 3.5510 | -3.2862 |
| 2 | 1 | 3 | 6.8250 | -0.0122 |
| 2 | 2 | 6 | 9.5750 | 2.7378 |
| 3 | 0 | 1 | 1.0015 | -5.8357 |
| 3 | 1 | 3 | 5.5510 | -1.2862 |
| 3 | 2 | 2 | 8.8250 | 1.9878 |
<!-- /worked -->

The fix that **proximal policy optimisation** (PPO) made is to flatten $L$ wherever a sample has already moved as far as its advantage can justify. Fix a half-width $\epsilon$ (here $\epsilon = 0.2$) and replace the term of every sample by

$$\min\big(\rho_i\,\hat A_i,\ \mathrm{clip}(\rho_i,\, 1 - \epsilon,\, 1 + \epsilon)\,\hat A_i\big),$$

with $\mathrm{clip}(\rho, \mathrm{lo}, \mathrm{hi}) = \min(\max(\rho, \mathrm{lo}), \mathrm{hi})$.

The clip is applied to the ratio, not to the term. The minimum takes the smaller of the raw and the clipped term, which makes the objective a pessimistic bound on $L$. What that does is clearest case by case (the tables write the ratio $\rho$ as `r`).

<!-- worked: policy_gradients.clip_cases -->
The per-sample term min(r A, clip(r, 1 - eps, 1 + eps) A) with eps = 0.2 (clip applied to the ratio r, clip(r, lo, hi) = min(max(r, lo), hi)), for advantages A = +1 and -1 and four ratios. The winning term is the one the minimum picks (the raw one on a tie); its derivative with respect to the log-probability that moves r is A r when the raw term wins and 0 when the clipped term wins.

| A | r | r A | clip(r) A | min | term that wins | gradient factor |
|---:|---:|---:|---:|---:|---|---:|
| +1 | 0.5 | +0.50 | +0.80 | +0.50 | raw (A r) | +0.50 |
| +1 | 0.9 | +0.90 | +0.90 | +0.90 | raw (A r) | +0.90 |
| +1 | 1.1 | +1.10 | +1.10 | +1.10 | raw (A r) | +1.10 |
| +1 | 1.5 | +1.50 | +1.20 | +1.20 | clipped | 0 |
| -1 | 0.5 | -0.50 | -0.80 | -0.80 | clipped | 0 |
| -1 | 0.9 | -0.90 | -0.90 | -0.90 | raw (A r) | -0.90 |
| -1 | 1.1 | -1.10 | -1.10 | -1.10 | raw (A r) | -1.10 |
| -1 | 1.5 | -1.50 | -1.20 | -1.50 | raw (A r) | -1.50 |
<!-- /worked -->

![The term of the clipped surrogate against the ratio r from 0 to 2 for advantages +1 and -1, with the raw term r A dashed. For A = +1 the term rises along the raw line to r = 1.2 and is flat at 1.2 beyond it; for A = -1 it falls along the raw line to r = 0.8, is flat at -0.8 below it, and for r above 1.2 follows the raw line downward. Vertical lines mark r = 0.8 and r = 1.2.](img/clipped-surrogate.svg)

With a positive advantage the term grows with $\rho$ only until $1 + \epsilon$ and is flat beyond it: a sample that has already raised its action's probability by a fifth gets no more push from its own advantage. With a negative advantage the term falls as $\rho$ falls only until $1 - \epsilon$. The other two quadrants are the ones that make the minimum worth having rather than a plain clip of the ratio. A positive advantage with a ratio below $1 - \epsilon$ (row $r = 0.5$, $A = +1$: raw $0.50$, clipped $0.80$): the raw term is smaller, it wins, and its gradient $A\rho$ survives, so a mistake that has made a good action less likely is corrected. A negative advantage with a ratio above $1 + \epsilon$ (row $r = 1.5$, $A = -1$): the raw term, $-1.50$, again is smaller and again survives. The clip removes a sample's gradient only where the sample has moved in the direction its advantage asked for, and never where it has moved the wrong way.

The algorithm is a loop around that objective. Collect a batch from the current policy, compute advantages and keep $\pi_{\mathrm{old}}$ fixed; then for a number of epochs, step along the gradient of the clipped objective. The version below is the one the tables use: the whole batch is one minibatch, taken in the order drawn, and the advantages stay fixed during the epochs. (Production PPO splits the batch into several minibatches and reshuffles them every epoch; with the generator of Volume I that is one `shuffle` per epoch, and nothing else of the arithmetic changes.)

```text
function ppo_epochs(theta: array<real>[m], pi_old: array<real>[m], actions: list<int>,
                    adv: list<real>, eps: real, eta: real, epochs: int) -> array<real>[m]:
    require eps > 0 and eta > 0 and len(actions) == len(adv) and len(actions) > 0
    nb = len(actions)
    for e in 0 .. epochs:
        pi = softmax(theta)
        grad = zeros(real, [m])
        for i in 0 .. nb:
            a = actions[i]
            r = pi[a] / pi_old[a]
            raw = r * adv[i]
            clipped = min(max(r, 1.0 - eps), 1.0 + eps) * adv[i]
            if raw <= clipped:
                for k in 0 .. m:
                    grad[k] = grad[k] + adv[i] * r * ((1.0 if k == a else 0.0) - pi[k]) / nb
        theta = theta + eta * grad
    return theta
```

The `if raw <= clipped` is the minimum: when the raw term is the smaller, or equal, its derivative with respect to the preferences is $\hat A\,\rho\,(\mathbf{e}_a - \boldsymbol\pi)$, since $\rho$ is a function of $\theta$ whose derivative is $\rho$ times the score; when the clipped term is the smaller it is locally constant and the sample contributes nothing. Twenty epochs, $\eta = 0.5$, $\epsilon = 0.2$, from $\theta_B$ on the batch above, beside the same run without the clip:

<!-- worked: policy_gradients.ppo_epochs -->
The batch of the previous table, 20 epochs of gradient ascent on its surrogate L(theta) = (1/64) sum_i min(r_i A_i, clip(r_i, 1 - eps, 1 + eps) A_i), r_i = pi_theta(a_i) / pi_old(a_i), starting from theta_B = (0.0, 1.0, 0.0, -1.0). An epoch is one step on the whole batch (one minibatch of 64 samples in drawing order, no shuffling): the ratios are recomputed from the current preferences, a sample whose raw term r A is not larger than its clipped term contributes A r (e_a - pi_theta) / 64 to the gradient (summed in sample order) and any other sample contributes nothing, and the preferences move by eta = 0.5 times the gradient. The advantages stay fixed. The first table is the clipped run, eps = 0.2, the second the same run without the clip (every sample contributes). Each row describes the preferences at the start of the stated epoch: 'clipped' counts the samples with no gradient, 'dev' is the largest |r - 1|, 'KL' is the exact KL(pi_old || pi_theta) in nats, 'claimed' is the surrogate's own value, which is 0 at the start since the advantages sum to 0, and 'true' is J(theta) - J(theta_B) from the exact q of the bandit.

Clipped:

| epoch | clipped | dev | KL | claimed | true |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0 | 0.110 | 0.0021 | 0.0338 | 0.0230 |
| 3 | 14 | 0.204 | 0.0074 | 0.0622 | 0.0431 |
| 4 | 14 | 0.230 | 0.0102 | 0.0701 | 0.0521 |
| 5 | 18 | 0.254 | 0.0132 | 0.0710 | 0.0602 |
| 6 | 18 | 0.268 | 0.0139 | 0.0737 | 0.0600 |
| 8 | 14 | 0.295 | 0.0159 | 0.0782 | 0.0590 |
| 10 | 24 | 0.328 | 0.0209 | 0.0817 | 0.0661 |
| 15 | 18 | 0.393 | 0.0288 | 0.0844 | 0.0751 |
| 20 | 18 | 0.449 | 0.0378 | 0.0861 | 0.0828 |

Unclipped:

| epoch | dev | KL | claimed | true |
|---|---:|---:|---:|---:|
| 1 | 0.000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.110 | 0.0021 | 0.0338 | 0.0230 |
| 3 | 0.204 | 0.0074 | 0.0630 | 0.0431 |
| 4 | 0.285 | 0.0152 | 0.0882 | 0.0606 |
| 5 | 0.353 | 0.0246 | 0.1100 | 0.0758 |
| 6 | 0.411 | 0.0350 | 0.1289 | 0.0890 |
| 8 | 0.504 | 0.0573 | 0.1598 | 0.1106 |
| 10 | 0.574 | 0.0801 | 0.1837 | 0.1272 |
| 15 | 0.688 | 0.1332 | 0.2250 | 0.1546 |
| 20 | 0.754 | 0.1790 | 0.2523 | 0.1704 |

J(theta_B) = 6.4867. Every number learned from samples is printed with the digits that the run and four copies of it share: the copies start with the first preference (theta[0][0] of the shop, theta_B[0] of the bandit) changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the nudges of stable.py), and a digit is printed only if all five agree on it ('about x' keeps the leading digits that survive, 'unstable' means that none does). The copies use the same generator and so draw the same numbers; they differ only if a nudge flips a comparison. A run is a deterministic computation, so another implementation reproduces the printed digits.
<!-- /worked -->

Read the two tables against each other. The two runs take the same first two steps, since every ratio is inside the band at the start of epochs $1$ and $2$. At the start of epoch $3$ the clip binds for the first time, and here is what it does to each outcome of the batch:

<!-- worked: policy_gradients.ppo_clip_binding -->
The first epoch of the clipped run at which a sample is clipped, epoch 3: the preferences are theta = (-0.1605, 1.1480, 0.0996, -1.0871), pi_theta = (0.1564, 0.5788, 0.2029, 0.0619), against pi_old = (0.1966, 0.5344, 0.1966, 0.0723). For every outcome of the batch: its count n, advantage A, ratio r = pi_theta(a) / pi_old(a) (it depends on a only), the raw term r A, the clipped term clip(r, 0.8, 1.2) A, the winner of the minimum and its gradient factor (A r when raw, else 0).

| a | d | n | A | r | r A | clip(r) A | wins | gradient factor |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 0 | 4 | -0.7622 | 0.7955 | -0.6064 | -0.6098 | clipped | 0 |
| 0 | 1 | 4 | -0.7622 | 0.7955 | -0.6064 | -0.6098 | clipped | 0 |
| 0 | 2 | 6 | -0.7622 | 0.7955 | -0.6064 | -0.6098 | clipped | 0 |
| 1 | 0 | 5 | -2.0122 | 1.0830 | -2.1793 | -2.1793 | raw | -2.1793 |
| 1 | 1 | 13 | 0.7378 | 1.0830 | 0.7990 | 0.7990 | raw | 0.7990 |
| 1 | 2 | 14 | 0.7378 | 1.0830 | 0.7990 | 0.7990 | raw | 0.7990 |
| 2 | 0 | 3 | -3.2862 | 1.0318 | -3.3906 | -3.3906 | raw | -3.3906 |
| 2 | 1 | 3 | -0.0122 | 1.0318 | -0.0126 | -0.0126 | raw | -0.0126 |
| 2 | 2 | 6 | 2.7378 | 1.0318 | 2.8248 | 2.8248 | raw | 2.8248 |
| 3 | 0 | 1 | -5.8357 | 0.8561 | -4.9958 | -4.9958 | raw | -4.9958 |
| 3 | 1 | 3 | -1.2862 | 0.8561 | -1.1011 | -1.1011 | raw | -1.1011 |
| 3 | 2 | 2 | 1.9878 | 0.8561 | 1.7017 | 1.7017 | raw | 1.7017 |

14 of 64 samples are clipped. The sum over the samples of |A r| / 64 is 1.2472; the clipped samples carry 0.1326 of it, which this epoch does not use.
<!-- /worked -->

The first three outcomes are the three of the zero-unit order: its probability has fallen from $0.1966$ to $0.1564$, a ratio of $0.7955$, just below $0.8$, and every sample of that order has a negative advantage, so these $14$ samples have gone as far down as the band allows and stop pushing. The rest of the batch carries on. The clipped samples hold $0.1326$ of a total gradient mass of $1.2472$ in this epoch. The surrogate's own claim, the column "claimed", is its value on the batch; the column "true" is the real change of $J$, computed from the exact $q$ of the bandit, which a learner never sees. For the clipped run the claim and the truth stay close, $0.0861$ against $0.0828$ at the start of epoch $20$. For the unclipped run they separate: it claims $0.2523$ and delivers $0.1704$, and the policy has moved $0.1790$ nats from $\pi_{\mathrm{old}}$, against $0.0378$ for the clipped one.

An honest reading of the table is not that the clip wins. On this batch of $64$ samples the unclipped run gains more, because the batch estimates the direction well. What the clip buys is a bounded error in the claim. The surrogate is only a faithful model of $J$ near $\pi_{\mathrm{old}}$; the unclipped objective gives no signal to stop moving, while the clipped one has stopped listening to the samples that have moved enough. On a worse batch the unclipped run would have the same appetite for travel and a worse direction.

There is also a limit to what the clip bounds, and it is a place where textbooks are often loose. The clipped run's ratios are not held to $[0.8, 1.2]$: the largest $\lvert\rho - 1\rvert$ at epoch $20$ is $0.449$. Removing a sample's gradient stops that sample from pushing, but the other samples go on pushing, and because the policy shares its parameters across actions the ratio of a clipped one still moves. The clipped objective is a brake on individual pushes and not a trust region. A practitioner who wants a bound on the policy's travel measures it and stops, and the measure is a KL divergence. The exact $\mathrm{KL}(\pi_{\mathrm{old}} \,\|\, \pi_\theta)$ needs all probabilities; from the batch itself, the samples were drawn from $\pi_{\mathrm{old}}$, so two sample estimates exist, both from the ratios alone:

$$k_1 = -\frac{1}{N}\sum_i \ln \rho_i, \qquad k_3 = \frac{1}{N}\sum_i (\rho_i - 1 - \ln \rho_i).$$

Both have the right mean (the second because $\mathbb{E}[\rho - 1] = 0$ under $\pi_{\mathrm{old}}$). Near $\rho = 1$ the terms of $k_1$ are $-\ln\rho \approx 1 - \rho$, of either sign and first order in the distance from $1$, while those of $k_3$ are $\rho - 1 - \ln\rho \approx (\rho - 1)^2/2$: never negative, and second order, so for the same mean $k_3$ has a much smaller spread.

<!-- worked: policy_gradients.kl_estimators -->
The clipped run of the previous table. At the start of the stated epoch the exact KL(pi_old || pi_theta) over the four orders, the estimate k1 = -(1/64) sum_i ln r_i and the estimate k3 = (1/64) sum_i (r_i - 1 - ln r_i) from the batch's own samples (r_i as before; natural logarithms).

| epoch | exact KL | k1 | k3 |
|---|---:|---:|---:|
| 5 | 0.01322 | 0.02954 | 0.01464 |
| 10 | 0.02090 | 0.03942 | 0.02283 |
| 20 | 0.03775 | 0.06215 | 0.04122 |
<!-- /worked -->

On these $64$ samples $k_3$ stays within $11$ percent of the exact $0.0132$, $0.0209$ and $0.0378$ (it reads $0.01464$, $0.02283$, $0.04122$) while $k_1$ is between $1.6$ and $2.2$ times as large. A stopping rule can use $k_3$ and abandon the remaining epochs when it exceeds a target.

## What an update costs

Work is counted in samples and arithmetic, not time. The samples are in the tables of the baseline section: $2\,979$ days against $250$ for the same accuracy of direction on the bandit, twelve times as many without the baseline. The arithmetic of one tabular update has a short count under the rule that an addition, subtraction, multiplication, division, exponential or logarithm is one operation and comparisons are free.

<!-- worked: policy_gradients.work_of_updates -->
One operation is an addition, subtraction, multiplication, division, exponential or logarithm; comparisons, min, max, clamps and indexing are not counted. m is the number of actions of a state (m = 4 on the empty shelf).

| update | count | FLOPs, m = 4 |
|---|---|---:|
| softmax over m orders | m subtractions, m exponentials, m - 1 additions, m divisions: 4m - 1 | 15 |
| preference row step | m subtractions (e_a - pi), the scale eta gamma^t delta (2 multiplications), m multiplications and m additions: 3m + 2 | 14 |
| REINFORCE, one day | return recursion 2, softmax, row step, running power gamma^t 1 | 32 |
| REINFORCE with baseline, one day | the same, plus the advantage subtraction 1 and the critic step (G - V, alpha_c times, add) 3 | 36 |
| actor-critic, one day | delta 3, softmax, row step, critic step 2, running power 1 | 35 |
| PPO, 20 epochs, clipped (eps = 0.2) | softmax once per epoch; per sample ratio 1, r A and clip(r) A 2; per unclipped sample the scale A r / N 1 and the row 3m | 16,672 |
| PPO, 20 epochs, unclipped | the same with every sample unclipped | 20,780 |
<!-- /worked -->

A day of any of the three learners costs about as much as another, $32$ to $36$ operations for the four orders of the empty shelf, and the PPO epochs cost $16\,672$ operations in all with the clip and $20\,780$ without, since a clipped sample skips its row of the gradient. On a table that is nothing next to the simulation. For a language model the arithmetic is dominated by something else: a policy-gradient step on $T$ tokens is a forward pass and a backward pass of the network, about $6P$ FLOPs per token for $P$ parameters (Volume V, Chapter 10), and the samples that feed it cost a forward per generated token, about $2P$ each. The clip saves nothing there: the weights of the clipped tokens are zero but they still travel through the backward pass. Doing $K$ epochs on a batch multiplies the $6PT$ of the update by $K$ and leaves the sampling cost alone, which is the reason to do several: a day of experience is paid for once in samples and several times in updates, and the samples are the dear part.

## What brain's `crates/rl` trains through

The reduction that makes policy-gradient training of a language model cheap to implement is already in the chapter. The score of a sampled token under a softmax is $\mathbf{e}_a - \boldsymbol\pi$, the negative of the gradient of the cross-entropy loss against that token, which every model of the engine already computes in its backward pass. The gradient of the clipped term for one token is therefore the cross-entropy gradient times a number, $\hat A\rho$ when the raw term wins and $0$ when it does not. `crates/rl/src/objective/grpo_term.rs` is that number: `token_term_banded` takes the old and the new log-probability of the token (the ratio is $e^{\text{new} - \text{old}}$), the advantage and the clip band, and returns the weight, the loss it is the derivative of, and whether the clip bound. The model is told the weights through its `set_loss_weights`, runs one ordinary backward pass and receives the gradient of the clipped objective. Supervised fine-tuning is the special case $\hat A = 1$, $\rho = 1$. Commit `53b9ee08f` (6 September 2026) introduced the objective as this reduction, "the keystone reduction" in its own message, and in the code and tests it left behind it is true as stated: the weight comes out of differentiating `min(r*A, clip(r)*A)` with `r = exp(new - old)` treated as a function of the parameters, and the gradient test `crates/rl/tests/grpo_gradcheck.rs` differentiates the real objective by finite differences, with one precaution that the shape of the clip explains. The clip is piecewise constant in the ratio, so a token whose ratio sits within a margin of $1 \pm \epsilon$ can flip between clipped and unclipped under an arbitrarily small perturbation; the test asserts that every sampled ratio is clear of both boundaries rather than loosening its tolerance to hide the jump.

The baseline in `crates/rl` is a group mean and not a critic. `crates/rl/src/objective/advantage.rs` takes the rewards of a group of completions to the same prompt and returns, for each, the reward minus the group's mean (the default, `MeanCentered`) or that divided by the group's standard deviation plus $10^{-4}$ (`ZScore`). A group whose rewards are all equal gets advantage $0$ for every member, since it carries no comparison; a group of one gets $1.0$, the rejection-sampling case in which the kept completion is simply imitated. The subtraction is the chapter's baseline with the prompt playing the state. One subtlety follows from the baseline section's condition that the baseline not depend on the action: the mean of a group of $n_{\mathrm{g}}$ includes the completion being scored. Write the mean as $r_i/n_{\mathrm{g}}$ plus the other $n_{\mathrm{g}} - 1$ rewards over $n_{\mathrm{g}}$; the others are independent of completion $i$ and drop out by the zero-mean score, so the expected gradient is $(1 - 1/n_{\mathrm{g}})$ times the true one. The direction is right and the scale shrinks by a known factor. A leave-one-out mean removes even that, and Chapter 4 returns to how group-relative advantages are built from this.

The ratio and the clip come from `pg_batch.rs` (commit `c3cb16bbf`, 1 November 2026, took the batch step out of the GRPO trainer so that other objectives could share it), which both the GRPO sampler (`crates/rl/src/objective/grpo.rs`) and the external trainer (`crates/rl/src/objective/external.rs`, which learns from rollouts collected outside brain) train through. The GRPO options give the band a wider upper edge than lower: `clip_eps_high` defaults to $0.28$ above the lower half-width `clip_eps` of the configuration, so the band is $[1 - \epsilon,\ 1 + 0.28]$ and a low-probability token that turns out well can still gain mass; the external trainer's default is symmetric, $0.2$ on both sides. The tests of `grpo_term.rs` replay the chapter's table: a ratio of $1.5$ with a positive advantage gives weight $0$, the same ratio with a negative advantage keeps the weight $-1.5$, and a ratio of $1.25$ is clipped by the symmetric band but not by the $0.2/0.28$ one. What a step measured is reported in the terms of this chapter: the fraction of clipped tokens, the range of the ratio and its largest deviation from $1$, and the k3 estimate of the KL to a reference policy (that term is the subject of Chapter 4). Commit `6baeeaf3d` (2 November 2026) put the KL, the clip fraction and the mean ratio into the training log line of the external trainer, which had shown only the loss, the gradient norm and the learning rate; the objective had already measured them for its record, and the line costs no extra pass.

Three pieces of history show the chapter's ideas failing and being repaired.

The ratio must be taken in the distribution that sampled. Knowledge entry #290 and commit `d1807c4ed` (16 October 2026) record that GRPO's trainer scored the untempered softmax while the sampler drew at temperature $T$, so that the first update already had a ratio away from $1$: the entry measured a ratio off by up to $12.3\,\%$ at $T = 1.5$, and with a band of $0.2$ a confident token with a positive advantage starts outside the band and is clipped at once. The code now sets the logit scale to $1/T$ before the rollout and recomputes every row's behaviour log-probability in the training forward (`RatioBasis::Trainer`, the default; the old basis survives for ablations). It is the chapter's importance ratio with its definition enforced: $\pi_\theta / \pi_{\mathrm{old}}$ makes sense only when both are the distribution of the same sampler.

A loss and its gradient must be one function. Commit `af96cb3b6` (8 November 2026) found that the ratio's exponent was capped at $20$ in the forward value while the backward was handed the weight of the uncapped objective, so past the cap the reported loss was flat while the gradient was $e^{20}\hat A$. The exponential now continues along its tangent at the cap (`bounded_exp` in `grpo_term.rs`), so that the objective is differentiable once and the weight is exactly the derivative of the loss; the tests compare the term with an FP64 reference and with central differences at gaps from ordinary to $400$, on both sides of the clip edges. It is the habit of the gradient check of Volume III applied to a clipped, capped objective.

And clipping is not a trust region, as the table above showed. Commit `3a1a1cb57` (20 September 2026), in the control-policy trainer built on `crates/decide` (`PolicyConfig::target_kl` in `crates/decide/src/policy.rs`, the loop in `crates/sdk/src/control.rs`), added the guard the chapter describes: the k3 estimate of the drift of the policy from the one that collected the batch is checked before every minibatch step, and the remaining minibatch steps and epochs are abandoned when it exceeds the target (default $0.02$). The comment above the check explains the placement: a check that fires only after the whole pass is a report, not a guard.

A smaller story belongs beside these, because it is about testing a gradient method. Commit `58e7fc84b` (8 November 2026) repaired a test that asserted that the probability of a stop token alone would rise after twelve optimiser steps, which, by its own comment, failed on about a third of random seeds. The two completions now share their prefix and differ only at the stop position, with opposite advantages, so every shared position cancels and a small step must raise $\log P(\text{stop}) - \log P(\text{continue})$ for any weights, since the first-order change is a positive semidefinite quadratic form. The test asserts what the gradient guarantees and no more.

## Check your understanding

1. Find the error: "Subtracting a baseline changes the policy gradient estimator, so the baseline must be chosen to keep it unbiased."
2. A colleague subtracts, from every payoff, the mean payoff $q(a)$ of the door that was drawn, calling it the perfect baseline. What is the mean of the resulting estimator, and why? Would a baseline computed from the drawn demand $d$ alone be allowed?
3. In the learning table plain REINFORCE and REINFORCE with a baseline both have the optimal greedy policy in five of ten runs at $200$ episodes, yet their mean returns differ by $0.61$. What can account for that, given that the return is of the stochastic policy and the count is of its most likely actions?
4. A critic is off by the same amount $c$ in every state. What happens to the actor-critic gradient, and what if the error is $+c$ in one state and $0$ elsewhere?
5. A PPO run reports a clipped fraction of $0$ for twenty epochs. Give two different reasons, and say which diagnostic of the chapter tells them apart.
6. Why is $k_3$ a better stopping signal than $k_1$, although both estimate the same KL?
7. The importance ratio of a sample is $\pi_\theta(a)/\pi_{\mathrm{old}}(a)$. What goes wrong if the batch was sampled at temperature $1.5$ and the sampler's log-probabilities are used as $\pi_{\mathrm{old}}$, while the trainer computes $\pi_\theta$ at temperature $1$? Which tokens are affected most?
8. With a group of $n_{\mathrm{g}} = 2$ completions and mean-centred advantages, what is the factor by which the expected gradient shrinks, and what does a larger group do to it?

## Summary

A policy that chooses at random with smooth probabilities has an expected return that is a smooth function of its parameters, and the likelihood-ratio trick turns its gradient into an expectation of a return times the score of the actions taken, which samples can estimate without a model of the world. The estimate is unbiased and very noisy: on the shop's one-day bandit the single-sample estimate with no baseline has total variance $27.33$, and a baseline equal to the policy's own value cuts it to $2.29$ without changing its mean, because the score has mean zero. Applied to a whole life, the same trick and the observation that only the policy's own factors contain the parameters give the policy gradient theorem, a sum over states weighted by their discounted visitation of the advantage of each action times the gradient of its log-probability, which finite differences of the exact return confirm to $10^{-10}$ with no term for the change of the state distribution. Estimated by sampled returns it is REINFORCE; with a learned value function in place of the return it is an actor-critic, whose generalised advantage estimate trades the critic's bias for the sampled days' variance with one number $\lambda$. To use a batch for several steps the surrogate reweights it by the importance ratio, and PPO's clip removes the gradient of samples that have moved in the direction their advantage asked for, a brake that keeps the surrogate's claim close to the truth and does not bound the ratio, so a KL estimate watches the policy's travel.

All of this was written for a policy small enough that every preference sits in a table. The next chapter points it at a language model: the actions are tokens, the episode is a completion, and the reward is a checker's verdict or a learned preference that arrives only at the end. The same pieces are there, advantage as the reward minus a group's mean, the clipped ratio and a KL term that keeps the model near the one it started from, together with the questions of where the reward comes from and what the penalty is for.

## References

- R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning", 1992.
- R. S. Sutton, D. McAllester, S. Singh and Y. Mansour, "Policy gradient methods for reinforcement learning with function approximation", 1999.
- J. Schulman, P. Moritz, S. Levine, M. Jordan and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation", 2015.
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford and O. Klimov, "Proximal policy optimization algorithms", 2017.
- R. S. Sutton and A. G. Barto, "Reinforcement Learning: An Introduction", second edition, 2018.

## Exercises

1. **[Calculate]** For $\theta = (0, 0, 0, 0)$ on the one-day bandit, give $\pi$, the exact gradient $\pi(k)(q(k) - J)$ for each $k$, and the single-sample estimate $y\,(\mathbf{e}_a - \boldsymbol\pi)$ for the outcome $(a, d) = (2, 2)$, with $y$ from the table of the bandit. Check that the exact gradient sums to zero.
2. **[Calculate]** At $\theta_B$ compute $\lVert \mathbf{e}_a - \boldsymbol\pi\rVert^2$ for $a = 0, 1, 2, 3$ from $\pi = (0.1966, 0.5344, 0.1966, 0.0723)$ and explain, from the formula for $b^\ast$, why $b^\ast$ lies below $J$.
3. **[Calculate]** A group of $n_{\mathrm{g}} = 4$ completions earns rewards $(1, 0, 0, 1)$. Give the mean-centred and the $z$-score advantages (the standard deviation with divisor $n_{\mathrm{g}}$), and the factor $1 - 1/n_{\mathrm{g}}$ by which a mean-centred group scales the expected gradient.
4. **[Calculate]** Three days pay $0, 1, 3$ with critic values $V(s_t) = 1, 1, 2$, a last value $0.5$ for the state where the episode was cut and $\gamma = 0.8$. Compute $\delta_t$ and the advantage estimates for $\lambda = 0$, $0.5$ and $1$ by the backward recursion, and check the last against $G_t - V(s_t)$.
5. **[Calculate]** A sample has $A = -2$ and ratio $\rho = 0.7$ under $\epsilon = 0.2$. Give the raw and the clipped term, the one the minimum picks and the gradient factor. Repeat for $A = +2$.
6. **[Implement]** Reproduce `policy_gradients.one_day_bandit` and `policy_gradients.single_sample_estimates`: the outcome table from the rules of Chapter 1 with $V^\ast$ solved exactly, the stable softmax, the exact gradient with the finite-difference check, and the eight draws in the stated order (one `uniform()` for the action by inverse CDF, one for the demand).
7. **[Implement]** Reproduce `policy_gradients.estimator_moments` and `policy_gradients.samples_for_accuracy`: the enumeration of the twelve outcomes for the exact bias and total variance, $b^\ast$, and the sampled columns from the generator with the draw order stated.
8. **[Implement]** Reproduce `policy_gradients.softmax_policy_gradient`: the exact return of a stochastic policy by the linear system, the discounted visitation $d$ from $(\mathbf{I} - \gamma\mathbf{P}_\pi)^\top d = \mu_0$, the theorem's gradient and the central differences. Then drop the $d(s)$ factor and say by how much the gradient is wrong, which entry is wrong most, and why.
9. **[Implement]** Reproduce `policy_gradients.policy_gradient_runs` for REINFORCE with and without the baseline, and the actor-critic, with the generators, draw orders and update orders stated; then give REINFORCE with the baseline the same number of policy steps as the actor-critic: draw the episode as before, then apply the term of each day on its own, in day order, each with the preferences as they stand when it is applied, the return $G_t$ and the critic update unchanged (a hundred steps per episode). Report the table of this variant and whether the gap in the table closes.
10. **[Implement]** Reproduce `policy_gradients.advantage_estimates` for the four values of $\lambda$ and both critics, with the exact bias from the Markov chain and the sampled spread from the stated rollouts.
11. **[Implement]** Reproduce `policy_gradients.ppo_batch`, `ppo_epochs`, `ppo_clip_binding` and `kl_estimators`, then change the run to four shuffled minibatches of $16$ per epoch (one `shuffle` of the sample indices per epoch with the generator of the batch, continued after the draws) and report the clipped counts and the KL.
12. **[Investigate]** Run the clipped and the unclipped PPO of the chapter on $200$ independent batches of $64$ samples from $\pi_{\mathrm{old}} = \mathrm{softmax}(\theta_B)$, batch $i$ ($i = 0, \dots, 199$) drawn as in the batch table by the generator named `policy_gradients.batches.batch<i>` (seed of that name), with the advantages, $\eta = 0.5$ and $\epsilon = 0.2$ of the chapter, and for each batch compute the true change $J(\theta) - J(\theta_B)$ after the twentieth step. On how many batches does each variant end below $J(\theta_B)$, and how far? What does your answer say about when the clip matters?

## Key terms

- **policy-gradient theorem**: the statement that the gradient of the expected discounted return with respect to the policy's parameters is a sum over states, weighted by their discounted visitation, of each action's value times the gradient of its probability, with no term for the change of the state distribution. <!-- concept: policy-gradient-theorem -->
- **REINFORCE**: the Monte Carlo policy gradient, which samples a life under the policy and moves the parameters along the score of every action taken times the return that followed it. <!-- concept: reinforce -->
- **baseline subtraction**: subtracting from the return a number that does not depend on the action taken, which leaves the mean of the gradient estimate unchanged and can reduce its variance greatly. <!-- concept: baseline-subtraction -->
- **advantage**: the value of an action in a state minus the value of the state, $q^\pi(s, a) - V^\pi(s)$: how much better the action is than the policy's average there. <!-- concept: advantage -->
- **actor-critic**: a learner that keeps a policy (the actor) and a value function (the critic) and updates the policy along the score of its actions times the critic's temporal-difference error. <!-- concept: actor-critic -->
- **PPO**: proximal policy optimisation, which takes several gradient steps on one batch of experience under a surrogate that weights each sample by the ratio of the new to the old policy's probability and clips the term of a sample that has moved far enough in the direction its advantage asked for. <!-- concept: ppo -->

## Answers

**Recall questions.**

1. $G_2 = 1$, $G_1 = 1 + 0.5 \times 1 = 1.5$ and $G_0 = 1 + 0.5 \times 1.5 = 1.75$.
2. $\delta = r + \gamma V(s') - V(s)$. A positive $\delta$ says the day went better than the estimate of $s$ expected, since the reward and the discounted estimate of where it led together exceed what the learner believed $s$ was worth, and the estimate of $s$ should rise.
3. The variance of the mean is $\sigma^2/N$, so the standard error is $\sigma/\sqrt{N}$ and it takes $4N$ draws to halve it.
4. $\mathrm{KL}(p \,\|\, q) = 0.5 \ln(0.5/0.75) + 0.5 \ln(0.5/0.25) = 0.5 \times (-0.4055) + 0.5 \times 0.6931 = 0.1438$ nats.
5. The gradient of $\ln p_c$ is $\mathbf{e}_c - \mathbf{p}$, the negative of the cross-entropy gradient.
6. $\ln 0.5 + \ln 0.25 + \ln 0.5 = -0.6931 - 1.3863 - 0.6931 = -2.7726$, which is $\ln 0.0625$: log-probabilities of the steps add.

**Check your understanding.**

1. The baseline does not change the mean of the estimator at all, as long as it does not depend on the action: the score has mean zero. It is chosen for its variance, not for bias, and the only requirement it must meet is independence from the action.
2. The mean is $\mathbf{0}$ at every $\theta$: given the door, $\mathbb{E}[y - q(a) \mid a] = 0$, so every term of the expectation vanishes and the signal is subtracted along with the baseline. The baseline depends on the action, so it no longer factors out of the expectation. A baseline computed from the demand $d$ alone would be allowed, because in the bandit the demand is drawn independently of the door, so its product with the zero-mean score still has mean zero.
3. The count asks only which action has the largest preference in each state, while the return $J$ of the stochastic policy also depends on how peaked the policy is. Two runs with the same greedy policy can differ in $J$ by how much probability they leave on the other orders. The baseline's higher mean ($9.23$ against $8.62$) and smaller spread ($0.42$ against $1.19$) are consistent with its runs having moved further along the same direction and with fewer runs stuck far from it.
4. A constant error $c$ in every state changes $\delta$ by $(\gamma - 1)c$ for every action of every state, a constant that the zero-mean score removes from the expected gradient; it adds only a little variance. An error of $+c$ in one state $s^\ast$ changes $\delta$ by $-c$ for every action taken in $s^\ast$ (harmless, the same for all its actions) and by $+\gamma c$ for the transitions into $s^\ast$, which differ between the actions of the states that lead there: those actions' advantages are shifted by different amounts, which is a bias in the gradient.
5. Either every ratio stayed inside the band because the step size was small, or the advantages were all $0$ (a batch of groups with equal rewards) so there was nothing to clip. The count of samples with a non-zero advantage tells them apart: it is $0$ in the second case and not in the first. The largest $\lvert\rho - 1\rvert$ would also show a small step.
6. Each term of $k_3$, $\rho - 1 - \ln\rho$, is non-negative, so a batch cannot report a negative KL and cannot stop on noise of the wrong sign, and its terms are small where $\rho$ is near $1$; the terms of $k_1 = -\ln\rho$ have either sign and are larger in size, so its spread is larger for the same mean.
7. The ratio is no longer $1$ at the first step: the numerator is the untempered probability and the denominator the flattened one. A confident token has a larger probability at temperature $1$ than at $1.5$, so its ratio starts above $1$, and if it is above $1 + \epsilon$ and the advantage is positive the token is clipped before any update has moved it. This is the failure recorded in knowledge entry #290.
8. With $n_{\mathrm{g}} = 2$ the factor is $1 - 1/2 = 1/2$: the expected gradient has half the length it would have with a baseline independent of the sample. A larger group shrinks the deficit: $1 - 1/n_{\mathrm{g}}$ tends to $1$.

**Exercises.**

1. $\pi = (0.25, 0.25, 0.25, 0.25)$ and $J = 6.1653$; the gradient is $0.25 \times (6.0750 - 6.1653, 6.7500 - 6.1653, 6.6678 - 6.1653, 5.1684 - 6.1653) = (-0.0226, 0.1462, 0.1256, -0.2492)$, which adds to $0$ up to rounding. For $(a, d) = (2, 2)$, $y = 9.5750$ and $\mathbf{e}_2 - \boldsymbol\pi = (-0.25, -0.25, 0.75, -0.25)$, so the estimate is $(-2.394, -2.394, 7.181, -2.394)$.
2. $\lVert \mathbf{e}_a - \boldsymbol\pi\rVert^2 = 1 - 2\pi(a) + \lVert\boldsymbol\pi\rVert^2$ with $\lVert\boldsymbol\pi\rVert^2 = 0.3681$: $a = 0, 2$: $0.9749$; $a = 1$: $0.2993$; $a = 3$: $1.2235$. The weights of $b^\ast$ are $\pi(a)\,p(d)$ times these, and the likely door $a = 1$, whose payoffs are the highest ($7.575$ for two of its three demands), has the smallest weight per unit of probability, so the doors that pay less count for more in $b^\ast$ than in $J$ and $b^\ast = 6.2987$ lies below $J = 6.4867$.
3. Mean $0.5$; mean-centred advantages $(0.5, -0.5, -0.5, 0.5)$; the standard deviation is $0.5$, so the $z$-scores are $(1, -1, -1, 1)$ up to the $10^{-4}$ in the denominator. The factor is $1 - 1/4 = 0.75$.
4. $\delta_0 = 0 + 0.8 \times 1 - 1 = -0.2$, $\delta_1 = 1 + 0.8 \times 2 - 1 = 1.6$, $\delta_2 = 3 + 0.8 \times 0.5 - 2 = 1.4$. $\lambda = 0$: $(-0.2, 1.6, 1.4)$. $\lambda = 0.5$ (factor $0.4$): $\hat A_2 = 1.4$, $\hat A_1 = 1.6 + 0.4 \times 1.4 = 2.16$, $\hat A_0 = -0.2 + 0.4 \times 2.16 = 0.664$. $\lambda = 1$ (factor $0.8$): $\hat A_2 = 1.4$, $\hat A_1 = 1.6 + 1.12 = 2.72$, $\hat A_0 = -0.2 + 2.176 = 1.976$. Check: $G_2 = 3 + 0.4 = 3.4$, $G_1 = 1 + 2.72 = 3.72$, $G_0 = 0 + 2.976 = 2.976$; minus the values: $1.4$, $2.72$, $1.976$.
5. $A = -2$, $\rho = 0.7$: raw $-1.4$, clipped $0.8 \times (-2) = -1.6$; the minimum is $-1.6$, the clipped term, so the gradient factor is $0$. For $A = +2$: raw $1.4$, clipped $1.6$; the minimum is the raw term $1.4$ and the gradient factor is $A\rho = 1.4$.
6. and 7. The checks are the tables of the blocks to the printed digits. 8. The check is the table of the block. With $d(s)$ replaced by $1$ every entry of state $s$ is divided by $d(s)$, so an entry is wrong by its theorem value times $1/d(s) - 1$. The two entries of the shelf of two units are wrong most, by $\mp 0.76$ ($1.235$ becomes $0.472$), because they are the largest entries and $d = 2.62$ there; next come the third order of the shelf of one unit ($-1.124$ becomes $-0.370$, an error of $0.75$) and the first ($0.908$ becomes $0.299$). The gradient keeps its signs, and what changes is the proportion between states: each is shrunk by its own visitation, between $2.6$ and $3.0$ for the three shelves that have a choice.
9. to 11. The checks are the tables of the blocks to the printed digits; the variations of 9 and 11 are expected to give different numbers, which the reader reports.
12. The table below is the check.

<!-- worked: policy_gradients.clip_over_batches -->
200 independent batches of 64 samples, batch i drawn exactly as in the batch table but by the generator named policy_gradients.batches.batch<i> (seed_of of that name, i = 0 .. 199): per sample one uniform() for the action (inverse CDF over pi_old = softmax(0.0, 1.0, 0.0, -1.0)) and then one for the demand, the advantage the sample's y minus the batch's mean y. On each batch the run of the epochs table starts from theta_B and takes 20 steps (eta = 0.5), clipped (eps = 0.2) or unclipped. 'Below' counts the batches whose true change J(theta) - J(theta_B) after the 20th step is negative, 'mean' and 'worst' are the mean and the smallest of that change over the batches (summed in batch order, divisor 200), 'KL' the mean exact KL(pi_old || pi_theta) after the 20th step. Every number learned from samples is printed with the digits that the run and four copies of it share: the copies start with the first preference (theta[0][0] of the shop, theta_B[0] of the bandit) changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the nudges of stable.py), and a digit is printed only if all five agree on it ('about x' keeps the leading digits that survive, 'unstable' means that none does). The copies use the same generator and so draw the same numbers; they differ only if a nudge flips a comparison. A run is a deterministic computation, so another implementation reproduces the printed digits.

| run | below J(theta_B) | mean true change | worst true change | mean KL |
|---|---:|---:|---:|---:|
| clipped, eps = 0.2 | 6 of 200 | 0.0600 | -0.0198 | 0.0253 |
| unclipped | 12 of 200 | 0.1337 | -0.9140 | 0.4010 |
<!-- /worked -->

The unclipped run gains more on average, and it is also the one that sometimes ends far below where it began: on the batches whose estimate of the direction is poor it takes the same twenty confident steps in a wrong direction, and nothing in its objective tells it to stop, so its worst batch loses more than forty times what the clipped run's worst loses, and on average it ends about sixteen times as far from $\pi_{\mathrm{old}}$ in KL. The clip matters on those batches, and costs gain on the good ones.
