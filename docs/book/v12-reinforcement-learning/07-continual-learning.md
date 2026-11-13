<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Continual Learning and Promotion

*How can a model keep learning from new experience without forgetting, and how do we decide a new version is better?*

A gradient step has no memory. It looks at the rows in front of it, finds the direction in which the loss on those rows falls fastest, and moves every number of the model that way; nothing in the computation knows what the model could do a minute ago. For most of this book that was harmless, because the rows were all the rows there were. This chapter begins when they are not. A network that classifies 400 of 400 test points of one task is trained for 50 steps at a step size of 0.2 on a second, and it then classifies none of the first correctly: the second task did not make the network worse at the first by a little, it made it a perfect inverter of it.

The volume has met this situation twice already without naming it. The language model of Chapter 4 was fine-tuned on demonstrations and nothing in the loss kept it from drifting away from what pretraining had taught, which is why the next steps of that chapter put a penalty against the starting model into the objective. The router of Chapter 6 was trained once on a log of the old router's decisions, and a router in service meets new requests every week. Both want the same loop: take in new experience, train, and replace the model in use only when the new one has earned it. That loop has two halves with different difficulties. Keeping the old skills while learning a new one is an optimisation problem, and the first half of this chapter counts the ways to attack it against each other on a network small enough to print. Deciding that the result is better is a statistical problem, and the second half is a gate: a frozen set of questions that every candidate must answer, a paired comparison with the model in use, a rule for what a regression is worth when there are many things that could regress, and a loop that wires them together. brain's `crates/promote` and `crates/audit` are the engine's version of that gate and that loop, and the chapter ends with them.

Six things from earlier chapters are the tools. Answer them from memory; the answers are at the end.

1. (Volume XII, Chapter 4, "Reinforcement and Preference Learning for Language Models", supervised fine-tuning.) Two conversations in one batch have 7 and 5 supervised targets and summed losses $27.1824$ and $26.8443$. What is the loss per supervised token of the whole batch?
2. (Volume I, Chapter 12, "Statistics: Learning Facts from Samples", paired test.) Two models answer the same 40 questions and disagree on 10 of them, and model B is right on 9 of the 10. What is the one-sided exact sign-test p-value for "B is better"?
3. (Volume I, Chapter 12, multiple comparisons.) Twenty independent tests are each run at the $5\%$ level and none of the differences is real. What is the chance that at least one is declared different, and what level does Bonferroni's correction give each test?
4. (Volume II, Chapter 5, "Generalization, Overfitting and Honest Evaluation", validation set.) Why may the number that decides whether a model is kept not be computed on rows the model was trained on?
5. (Volume II, Chapter 4, "Maximum Likelihood, Priors and Regularization", L2 regularization.) The loss $\tfrac12(\theta - 3)^2 + \tfrac{\lambda}{2}(\theta - 1)^2$ pulls $\theta$ towards 3 and towards the anchor 1. For $\lambda = 1$, where is its minimum?
6. (Volume II, Chapter 3, "Classification and Logistic Regression", binary cross-entropy.) A logistic output with logit $z = 2$ is scored against the label $y = 1$. What are $p$, the derivative of the loss with respect to $z$, and $p(1 - p)$?

## The same question with the opposite answer

The network is small enough to print. It takes a point $(x_0, x_1)$ of the plane through two hidden layers of six tanh units and a logistic output, 67 numbers in all, and classifies it by the sign of its logit. Two tasks use the same two inputs. In task A the points are in a cluster around $x_0 = -1.5$ and the class is 1 when $x_1$ is above the line $x_1 = 0$. In task B the cluster is around $x_0 = +1.5$ and the class is 1 when $x_1$ is *below* it. The tasks agree on the question and disagree on the answer, and a network that has seen both can tell them apart by $x_0$, which is a function it is easy to represent (a reference run that trains on both at once, the last row of the table of remedies below, gets 387 of 400 right on A and 398 of 400 on B). What the first block fixes is everything a reader needs to rebuild it: the architecture, the draw order of the initial numbers and of the data, and the training rule.

<!-- worked: continual_learning.world_rule -->
Network: x = (x0, x1), h1 = tanh(W1 x + b1) with 6 units, h2 = tanh(W2 h1 + b2) with 6 units, z = w3 . h2 + b3, p = sigmoid(z); the loss of a row is softplus(z) - y z, the mean over the rows is the loss, and a row is classified as 1 when z > 0 (a logit of exactly 0 is class 0). The 67 numbers are stored in the order W1 (row-major, 12), b1 (6), W2 (row-major, 36), b2 (6), w3 (6), b3 (1); a gradient is the gradient of the mean loss, the rows' gradients added in row order and then divided by n.

Initialisation: generator `continual_learning.init` (seed 0x24a60d34586711b5, 2640812749971984821). A normal() call that finds a held value returns it without drawing. The 12 numbers of W1 are 0.5 times 12 successive `normal()` calls, then the 36 numbers of W2 are 0.5 times 36 more, both row-major (48 calls, 24 pairs, nothing is left held); every bias and w3 and b3 are 0. Because w3 is 0 the starting network outputs the logit 0 on every input, so it answers class 0 everywhere.

| initial numbers | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|---:|
| W1 row 0 | 0.3208 | 0.4941 |  |  |  |  |
| W1 row 1 | -0.2987 | 0.4646 |  |  |  |  |
| W1 row 2 | -0.4424 | 0.5368 |  |  |  |  |
| W1 row 3 | -0.1230 | 0.6257 |  |  |  |  |
| W1 row 4 | -0.6085 | 0.4757 |  |  |  |  |
| W1 row 5 | 0.0188 | -0.3021 |  |  |  |  |
| W2 row 0 | 0.3534 | 0.5854 | -0.4540 | 0.4127 | 0.0957 | -0.8870 |
| W2 row 1 | -0.8979 | 0.1098 | 0.3061 | -0.6952 | -0.0938 | -0.2220 |
| W2 row 2 | 0.6753 | 0.4299 | -0.6925 | -0.2493 | -0.0405 | 1.2562 |
| W2 row 3 | 0.2582 | -0.1134 | -0.1705 | 1.0618 | -0.3497 | -0.0665 |
| W2 row 4 | 0.0966 | -0.0020 | 0.4599 | -0.3364 | -1.1285 | 0.0308 |
| W2 row 5 | 0.9247 | -0.2575 | -1.0230 | -0.1643 | 0.5337 | 0.2350 |

Data: generator `continual_learning.data` (seed 0x7136b019b0096f7f, 8157901399402311551). A normal() call that finds a held value returns it without drawing. One generator for everything, in this order: the 100 training rows of A, the 100 of B, the 400 test rows of A, the 400 of B, then the 100 bank rows of A and the 100 of B (the bank of the probe-bank section). A row draws `u = uniform()`, then `z1 = normal()`, then `z0 = normal()` (the second call returns the held value, so a row consumes three raw values); y = 1 if u < 0.5, s = +1 if y = 1 and -1 otherwise, x0 = centre + 0.5 z0 and x1 = sign * s * (0.3 + 0.7 |z1|). Task A has centre -1.5 and sign +1 (y = 1 above the line x1 = 0), task B has centre +1.5 and sign -1 (y = 1 below it): the same question with the opposite answer, in another cluster.

| row | x0 | x1 | y |
|---|---:|---:|---:|
| A row 0 | -2.0342 | 0.3641 | 1 |
| A row 1 | -1.4924 | 0.5815 | 1 |
| A row 2 | -2.0659 | 0.4703 | 1 |
| B row 0 | 1.5128 | 1.5959 | 0 |
| B row 1 | 2.0360 | -1.3679 | 1 |
| B row 2 | 2.1082 | -1.1631 | 1 |

Counts of y = 1: A train 55 of 100, B train 42 of 100, A test 188 of 400, B test 211 of 400.
<!-- /worked -->

The initial network answers class 0 everywhere because its output layer is zero. Training on the 100 rows of A for 300 steps of full-batch gradient descent at the step size $0.5$ gives the numbers of the next block. The mean loss on the 100 rows of A is down to $0.0015$ and all 400 test rows of A are classified correctly; none of B's are, which is expected, since B is A with the answer reversed. The gradient used throughout is the backpropagation of Volume III, and it is checked once, at the trained numbers, against central differences.

<!-- worked: continual_learning.pretrained -->
Training on A from the initial numbers above: full-batch gradient descent, step size 0.5, 300 steps, the mean loss over the 100 rows of A. Numbers of the run are printed with the digits that the run and its four nudged copies share (the rule of `stable.py`); a count is out of 400 test rows. The analytic gradient agrees with central differences (step 1e-6) on all 67 numbers, at the trained numbers: the largest absolute difference is below 1e-6, the criterion.

| trained on A | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|---:|
| W1 row 0 | 0.082 | 0.852 |  |  |  |  |
| W1 row 1 | -0.194 | 0.645 |  |  |  |  |
| W1 row 2 | -0.304 | 0.623 |  |  |  |  |
| W1 row 3 | 0.000 | 1.656 |  |  |  |  |
| W1 row 4 | -0.266 | 0.765 |  |  |  |  |
| W1 row 5 | -0.047 | -1.221 |  |  |  |  |
| b1 | 0.132 | -0.060 | -0.078 | -0.054 | -0.195 | 0.012 |
| W2 row 0 | 0.608 | 0.773 | -0.295 | 0.820 | 0.254 | -1.174 |
| W2 row 1 | -1.146 | -0.101 | 0.119 | -1.120 | -0.296 | 0.078 |
| W2 row 2 | 0.409 | 0.283 | -0.797 | -0.635 | -0.149 | 1.548 |
| W2 row 3 | 0.504 | 0.118 | 0.046 | 1.487 | -0.120 | -0.361 |
| W2 row 4 | -0.134 | -0.126 | 0.375 | -0.672 | -1.222 | 0.290 |
| W2 row 5 | 0.793 | -0.345 | -1.092 | -0.360 | 0.456 | 0.385 |
| b2 | -0.017 | -0.019 | 0.101 | 0.071 | 0.102 | 0.041 |
| w3 | 1.880 | -1.526 | -1.303 | 1.565 | -0.999 | -0.357 |
| b3 | 0.008 |  |  |  |  |  |

| quantity | value |
|---|---:|
| mean loss on the 100 rows of A | 0.0015 |
| correct on A test (of 400) | 400 |
| correct on B test (of 400) | 0 |
<!-- /worked -->

Call these numbers $\boldsymbol\theta_A$. Now do what a fine-tuning run does: keep training the same network, but on task B alone. The next block counts, out of 400 test rows of each task, what the network gets right after a number of steps, for three step sizes. The figure after it draws the same runs every 5 steps.

<!-- worked: continual_learning.forgetting_grid -->
Starting from the numbers trained on A, gradient descent on the 100 rows of B alone (mean loss over them), with the step size and for the number of steps shown. Counts are correct classifications out of 400 test rows of each task; step 0 is the network trained on A.

| step size | steps on B | A correct | B correct |
|---|---:|---:|---:|
| 0.05 | 0 | 400 | 0 |
|  | 10 | 394 | 87 |
|  | 20 | 374 | 158 |
|  | 50 | 330 | 179 |
|  | 100 | 267 | 187 |
|  | 200 | 0 | 400 |
| 0.2 | 0 | 400 | 0 |
|  | 10 | 347 | 176 |
|  | 20 | 295 | 184 |
|  | 50 | 0 | 400 |
|  | 100 | 0 | 400 |
|  | 200 | 0 | 400 |
| 0.5 | 0 | 400 | 0 |
|  | 10 | 182 | 293 |
|  | 20 | 0 | 400 |
|  | 50 | 0 | 400 |
|  | 100 | 0 | 400 |
|  | 200 | 0 | 400 |
<!-- /worked -->

![Test accuracy on task A (solid lines) and on task B (dashed lines) of the network trained on A while it is trained on B alone, against the number of steps, for the step sizes 0.05 (blue), 0.2 (orange) and 0.5 (green): accuracy on A falls to zero in every case, by 20 steps at the largest step size and by 200 at the smallest, while accuracy on B rises to 1.](img/forgetting-curves.svg)

**Catastrophic forgetting** is what the table and the figure show: a model that is trained on a new task loses, abruptly and almost completely, what it had learned of an old one that is no longer in its training data. Three things are worth reading off. The damage follows how far the descent has gone and not the number of steps: gradient descent with a small step size approximates a flow in the time step size times steps, and in the grid the first count of zero is at the product $0.5 \times 20 = 10$, at $0.2 \times 50 = 10$ and at $0.05 \times 200 = 10$, while at a product of 5 the network at the step size $0.5$ still has 182 of A and at $0.05$ after 100 steps 267. It is not gradual: at the step size $0.2$ the network is still right on 295 rows of A after 20 steps and on none after 50, and the end of each curve in the figure is a cliff. And a count of zero is not the same as ignorance. A classifier that is right on none of 400 rows with two classes has put every row on the wrong side, which carries as much information as being right on all of them; the network has not forgotten the question, it has been taught the reverse answer, and a test that treated "zero correct" as "knows nothing" would misread it. For the gate that is built below the distinction is the whole point: a candidate that is perfectly wrong on an old skill is as different from the incumbent as a candidate can be.

## What the first-order and the second-order terms say

Why does the loss on A rise when no step of the descent looks at A? Write $L_A(\boldsymbol\theta)$ for the mean loss on the rows of A and $\mathbf d = \boldsymbol\theta - \boldsymbol\theta_A$ for how far the numbers have moved. Volume I's Taylor expansion of the loss around $\boldsymbol\theta_A$ is

$$L_A(\boldsymbol\theta_A + \mathbf d) \approx L_A(\boldsymbol\theta_A) + \nabla L_A^{\top}\mathbf d + \tfrac12\,\mathbf d^{\top} H_A\,\mathbf d,$$

where $H_A$ is the matrix of second derivatives. After training on A the gradient $\nabla L_A$ is nearly zero (the loss is a minimum, or close to one), so the first-order term has nothing to say and the damage is the second-order term: the loss rises as the quadratic form of $H_A$ in the displacement. A direction in which $H_A$ is large is a direction in which A's loss is steep, and a direction in which it is small is one the network can move along for free. Forgetting is the step of B's gradient moving the numbers along steep directions of A. That also says what a defence has to look like: keep $\mathbf d$ small in the directions in which $H_A$ is large.

For a logistic output the matrix has a neat form. The loss of one row is $\mathrm{softplus}(z) - yz$ and the derivative of the logit with respect to number $i$ is $g_i$, so the second derivative of the row's loss is $p(1-p)\,g_i g_j + (p - y)\,\partial^2 z/\partial\theta_i\partial\theta_j$. The second term has the residual $p - y$ in front, which is small when the model fits, and what remains is

$$F_{ij} = \frac{1}{n}\sum_{\text{rows}} p\,(1-p)\; g_i\, g_j ,$$

the **Fisher information** of the model on those rows. It has a second reading that gives it its name: if the label is drawn from the model's own probability $p$, the derivative of the log-likelihood with respect to the logit is $y - p$, whose mean square is $p(1-p)$, so $F$ is the average squared gradient of the log-likelihood with the label averaged out. This is the *true* Fisher, with the label drawn under the model; the *empirical* Fisher uses the observed label and has $(y - p)^2$ in place of $p(1-p)$, which is smaller when the model fits. This chapter uses the true one, per row, with the divisor $n$ the number of rows, and its diagonal is computable exactly in one pass because each row's gradient of the logit $\mathbf g$ is one backpropagation. The next block computes it for the network trained on A over the 100 rows of A.

<!-- worked: continual_learning.fisher_diagonal -->
The exact Fisher diagonal of the network trained on A over the 100 rows of A (true Fisher: the label is averaged out under the model's own probability, which for a logistic output gives p (1 - p) times the squared gradient of the logit; per row, divisor 100). Its sum and its largest entries; entries are printed with the digits all five runs share.

| quantity | value |
|---|---:|
| sum of the 67 entries | 0.0386 |
| largest entry | 0.0091 |
| number of entries below 1e-4 | 38 |
| entry W1[3,0] | 0.0091 |
| entry W1[5,0] | 0.0063 |
| entry b1[3] | 0.0037 |
| entry b1[5] | 0.0026 |
| entry W1[4,0] | 0.0015 |
<!-- /worked -->

The sum of its 67 diagonal entries is $0.0386$ and 38 of them are below $10^{-4}$: the model is so sure of A that $p(1 - p)$ is nearly zero on every row, so that the information about A is concentrated in a few first-layer numbers that read $x_0$ (the entries `W1[3,0]` and `W1[5,0]`) and their biases. Whether the quadratic form predicts the damage can be asked of the run itself. The next block measures the rise of the mean loss on A along the naive run at the step size $0.2$, and puts beside it $\tfrac12\mathbf d^{\top}F\mathbf d$ (the full matrix) and $\tfrac12\sum_i F_{ii}d_i^2$ (the diagonal only).

<!-- worked: continual_learning.anchor_prediction -->
The run with step size 0.2 on B alone. F is the exact Fisher information of the network trained on A, taken over the 100 rows of A: F_ij = (1/100) sum over the rows of p (1 - p) g_i g_j with g the gradient of the logit; 'diagonal' keeps only i = j. With d = (numbers after the steps) - (numbers trained on A), the quadratic prediction of the rise of the mean loss on the 100 rows of A is (1/2) d^T F d (full) or (1/2) sum_i F_ii d_i^2 (diagonal), against the measured rise.

| steps on B | length of d | measured rise | full prediction | diagonal prediction |
|---|---:|---:|---:|---:|
| 1 | 0.491 | 0.002 | 0.001 | 0.000 |
| 2 | 1.012 | 0.010 | 0.003 | 0.001 |
| 5 | 2.113 | 0.143 | 0.012 | 0.005 |
| 10 | 2.998 | 0.319 | 0.026 | 0.008 |
| 20 | 3.688 | 0.534 | 0.039 | 0.012 |
<!-- /worked -->

It does not. After a single step the measured rise is $0.002$ against a prediction of $0.001$, which is of the right order for a displacement of length $0.491$; after 5 steps, at a displacement of $2.113$, the loss has risen by $0.143$ and the quadratic form says $0.012$; at 20 steps the figures are $0.534$ and $0.039$, a factor of 14. The expansion is a local statement, and the steps at which forgetting happens are not local: the displacement is a few units long and the tanh and sigmoid saturate along the way. The Fisher form still has the sign and the ordering right (the full matrix predicts more than the diagonal, and both grow with the distance), and it is a useful *direction* of protection even where it is not a measure. The section on anchors below puts it to work and shows how far that goes.

## Mixing old rows back in

The most direct defence against forgetting what the network learned from A rows is to keep training on A rows. **Experience replay** keeps a buffer of old examples and mixes them into the training of the new task, so that every step's gradient is partly about the past. Two choices define it and both must be stated. The first is the *mixing ratio*, how many old rows are used for each new one. Here the loss of a step is the mean over the $n_{\text{new}}$ new rows and the $m$ buffered rows together, all weighted alike, so that the old rows' share of the gradient is $m/(n_{\text{new}} + m)$ and the ratio is $m/n_{\text{new}}$. The second is the *selection rule*: which $m$ of the old rows are kept. The old rows of a stream arrive one at a time and nobody knows in advance how many there will be, which is the problem **reservoir sampling** solves. Keep the first $m$ rows; then for row number $i$ (counting from 0) draw $j$ uniformly from $0 \ldots i$ and, if $j < m$, let the new row take slot $j$ of the buffer. Row $i$ is then kept with probability $m/(i+1)$ when it arrives, and each of the rows that it could displace survives with probability $1 - 1/(i+1)$, so by induction after $n$ rows every row of the stream is in the buffer with the same probability $m/n$: the buffer is a uniform sample of everything seen, however long ago.

```text
function reservoir_fill(n_items: int, cap: int, rng: Generator) -> list<int>:
    require cap >= 1 and n_items >= 1
    keep = [0, 1, ..., min(cap, n_items) - 1]
    for i in cap .. n_items:
        j = rng.integer(i + 1)
        if j < cap:
            keep[j] = i
    return keep
```

The block uses it on the 100 rows of A in their order, for six buffer sizes, with the generator restarted for each.

<!-- worked: continual_learning.replay_table -->
Replay: the loss of every step is the mean over the 100 rows of B and the m buffered rows of A (all weighted alike), step size 0.2, 200 steps, from the numbers trained on A. The buffer is Algorithm R over the 100 rows of A in their order with generator `continual_learning.reservoir` (seed 0x7abe008bf83b7bfd, 8844507319367334909). Each m starts the generator afresh; the first m rows fill the buffer and row i (0-based) replaces slot `rng.integer(i + 1)` when that is below m, one draw for each i = m .. 99. For m = 10 the buffer holds the A rows 15, 96, 61, 95, 74, 18, 93, 69, 39, 14 (slot order). A step lists the 100 rows of B in their order and then the buffer rows in slot order, and sums their gradients in that order.

| buffer m | old rows per new row | rows per step | A correct | B correct |
|---|---:|---:|---:|---:|
| 0 | 0.00 | 100 | 0 | 400 |
| 5 | 0.05 | 105 | 47 | 400 |
| 10 | 0.10 | 110 | 220 | 400 |
| 20 | 0.20 | 120 | 387 | 397 |
| 50 | 0.50 | 150 | 389 | 395 |
| 100 | 1.00 | 200 | 392 | 394 |
<!-- /worked -->

The table is the first count of what a remedy costs. With no buffer the network learns B and loses A completely. With 5 old rows (a ratio of $0.05$) A is still almost gone, 47 of 400; with 10 it is 220. At 20 old rows, a fifth of the number of rows of B, the network gets 387 of A and 397 of B right, and past that the gain is small: 389 and 395 at 50 rows, 392 and 394 with the whole of A. The curve has a knee, and the knee is at the point where the buffer holds enough of A to define the boundary in its cluster, not at a particular fraction of the new data. The cost is arithmetic, and it is exact: each step computes the gradient of $n_{\text{new}} + m$ rows instead of $n_{\text{new}}$, so a buffer of 20 raises the work of a step by a fifth and a buffer of 100 doubles it. The memory is $m$ rows. What replay does not need is the old model, the old loss or any assumption about the architecture; that, and its simplicity, is why it is the remedy that brain's continual reader settled on (the last sections come back to why).

## Holding the numbers near where they were

A second family of defences needs no old rows at all, only the old numbers. Add to the loss a penalty on the distance from $\boldsymbol\theta_A$, so that the descent on B is paid for every number it moves. The simplest anchor is the L2 penalty of Volume II with the old numbers in place of zero: $\tfrac{\lambda}{2}\sum_i(\theta_i - \theta_{A,i})^2$, whose gradient is $\lambda(\theta_i - \theta_{A,i})$, the same for every number. The sections above said what a better penalty would do, which is to charge more for the directions in which A's loss is steep. **Elastic weight consolidation** takes the quadratic form literally and uses the diagonal of the Fisher information as the stiffness: $\tfrac{\lambda}{2}\sum_i F_i(\theta_i - \theta_{A,i})^2$, with the gradient $\lambda F_i(\theta_i - \theta_{A,i})$ and $F$ the exact diagonal of the previous block. Both penalties need a copy of $\boldsymbol\theta_A$ (67 numbers) and EWC needs $F$ besides (another 67).

```text
function anchored_step(t: array<real>[P], rows: list<(array<real>[2], int)>, eta: real, lam: real,
                       anchor: array<real>[P], stiff: array<real>[P]) -> array<real>[P]:
    require eta > 0 and lam >= 0
    g = loss_gradient(t, rows)
    u = zeros(real, [P])
    for i in 0 .. P:
        u[i] = t[i] - eta * (g[i] + lam * stiff[i] * (t[i] - anchor[i]))
    return u
```

Here `loss_gradient(t, rows)` is the gradient of the mean cross-entropy of the network over the rows. For the L2 anchor `stiff` is all ones and for EWC it is the Fisher diagonal. A caution that belongs to the penalty and not to the statistics: a gradient step on a quadratic penalty is stable only while the step size times the stiffness is below 2, so $\eta\lambda < 2$ for the L2 anchor and $\eta\lambda F_i < 2$ for each entry of EWC's. The grid below stays inside that, and says by how much for EWC's largest value.

<!-- worked: continual_learning.anchor_table -->
An anchor on the old numbers, the same protocol: step size 0.2, 200 steps on the 100 rows of B. The L2 anchor adds (lambda/2) sum_i (t_i - tA_i)^2 to the mean loss, so its gradient is lambda (t_i - tA_i), for all 67 numbers. EWC adds (lambda/2) sum_i F_i (t_i - tA_i)^2 with the exact Fisher diagonal F of the previous block, gradient lambda F_i (t_i - tA_i). For a step to be stable the stiffest direction needs step size times stiffness below 2: here 0.2 * lambda <= 0.6 for the L2 anchor, and 0.2 * lambda * max F = 1.812 for EWC at lambda = 1000.

| anchor | lambda | A correct | B correct |
|---|---:|---:|---:|
| L2 | 0.1 | 76 | 400 |
| L2 | 0.3 | 207 | 382 |
| L2 | 1 | 396 | 117 |
| L2 | 3 | 399 | 70 |
| EWC | 10 | 0 | 400 |
| EWC | 100 | 0 | 400 |
| EWC | 1000 | 0 | 400 |
<!-- /worked -->

The L2 anchor shows the trade that every anchor makes, and it is a clean one: at $\lambda = 0.1$ the anchor is too weak (A: 76 of 400, B: 400), at $\lambda = 0.3$ the two tasks share the network at 207 and 382, and from $\lambda = 1$ the anchor wins, A at 396 and B at 117 of 400, and at $\lambda = 3$ B has all but gone, 70. There is no setting at which both are good. One reading is that the L2 anchor cannot tell the numbers that carry A from the numbers that do not, so that the only way it can protect A is to hold every number, B's included, near where it was.

EWC, which was built to tell them apart, protects nothing here: at none of the three values tried ($\lambda = 10, 100, 1000$, the last at $0.91$ of the largest stable value, $2/(0.2 \max F) = 1104$) is A above zero. The Fisher block says why, in the way this toy happens to be built. The information about A sits on a handful of numbers (38 of the 67 have an entry below $10^{-4}$), the stiffness there is capped by $\eta\lambda F_i < 2$ (the largest, at $\lambda = 1000$, is $1.812$), and every number that A hardly reads is left free for B to use. That is consistent with the network re-routing B through the free numbers while the protected ones are no obstacle, though the block does not test that explanation. This is a result about this network, this anchor and this grid and not a verdict on the method; what it does show is that the premise of EWC, a quadratic summary of A's loss, is only as good as the summary, and the previous section measured that this one under-predicts the rise by an order of magnitude at the distances forgetting travels.

## A change that can be switched off

A third approach is to let B change a small part of the network and keep the rest untouched, and to keep that part separate. A **low-rank adapter** (Volume XIII, Chapter 4 develops them) does it by adding to one of the frozen matrices $W$ a product of two thin ones, $W + \mathbf u\mathbf v^{\top}$: for a $6 \times 6$ matrix that is a rank-one matrix, whose 36 entries are determined by the 12 numbers of $\mathbf u$ and $\mathbf v$ (Volume I's low-rank approximation). Only $\mathbf u$ and $\mathbf v$ are trained, and the 67 numbers of the base never change. The next block adds one to the second matrix $W_2$ of the network trained on A and trains it on B, with $\mathbf u$ starting at zero so that the first step changes nothing, and counts three ways of serving the result.

<!-- worked: continual_learning.adapter_table -->
A rank-one adapter on W2: the matrix used is W2 + u v^T with u, v in R^6 (12 trained numbers; the 67 of the base are frozen). u starts at 0 and v from 0.5 times 6 `normal()` calls of generator `continual_learning.adapter` (seed 0xab1ea159a6a3f4da, 12330470236206920922). A normal() call that finds a held value returns it without drawing. (v starts at -0.4242, -0.2667, -0.4449, 0.0059, -0.2604, 0.5664). Gradient descent on the 100 rows of B only, step size 0.5, 200 steps; the gradient of u is G_W2 v and that of v is G_W2^T u, with G_W2 the gradient of the loss with respect to the matrix in use. 'Routed' serves the base where x0 < 0 and the adapter where x0 >= 0, by the row's own x0 and for every test row of both tasks (the router does not see the task label, so a row of A with x0 >= 0 reaches the adapter).

| served | A correct | B correct |
|---|---:|---:|
| adapter off (the base alone) | 400 | 0 |
| adapter on, always | 0 | 400 |
| routed by the sign of x0 (every test row of either task) | 398 | 400 |
<!-- /worked -->

An adapter does not protect A if it is always on: a rank-one change to one matrix is enough to flip the answers of this network on A, and the second row is the naive result again, 0 of 400. What it provides is *isolation*. With the adapter off the model is the base, bit for bit, 400 of 400 on A, and the whole of B's change lives in 12 numbers that can be stored, removed, replaced or applied alone. In the third row a router (here the sign of $x_0$, a stand-in for whatever knows which task a request belongs to) sends A's rows to the base and B's to the adapter, and both tasks are served at 400 of 400. The forgetting is not cured, it is made *addressable*: the old behaviour is a file that was never touched. This is the property brain's pool is built on, and it comes with a bill that the next block adds up.

## Five remedies and what each one costs

<!-- worked: continual_learning.remedy_summary -->
The remedies of the last blocks side by side (step size 0.2, 200 steps on B, from the numbers trained on A; the joint row is the reference trained from the initial numbers with step size 0.5). 'Rows per step' is the number of rows whose gradient is computed in every step; 'kept besides the new numbers' counts the numbers that must be stored to apply the remedy.

| remedy | rows per step | kept besides the new numbers | A correct | B correct |
|---|---:|---:|---:|---:|
| train on B alone | 100 | 0 | 0 | 400 |
| replay, m = 20 | 120 | 60 (20 rows of 3 numbers) | 387 | 397 |
| replay, m = 100 | 200 | 300 | 392 | 394 |
| L2 anchor, lambda = 0.3 | 100 | 67 (a copy of the old numbers) | 207 | 382 |
| EWC, lambda = 1000 | 100 | 134 (old numbers and F) | 0 | 400 |
| adapter, always on | 100 | 12 (and the 67 kept) | 0 | 400 |
| joint training from the start (600 steps on 200 rows) | 200 | 100 rows of A | 387 | 398 |
<!-- /worked -->

Read as a purchase. Replay buys the best joint accuracy per unit of complexity and pays in rows per step and in rows stored. The L2 anchor pays nothing per step and returns a compromise that no setting makes good for both tasks. EWC here pays for two stored vectors and bought nothing at the three values tried. An adapter always on is naive training in a smaller space, and the isolation it offers is exploited only by something that knows when not to apply it. None of the four is the joint training of the last row, which needs every old row and a run from the start, and which is the reference that sets what "no forgetting" can mean for this network: 387 of 400 on A and 398 of 400 on B.

The measurements have a use beyond ranking, which is why the second half of the chapter exists: each of them is a candidate, a network that someone proposes to serve in place of $\boldsymbol\theta_A$, and to know that replay with a buffer of 20 is better than training on B alone one needs a decision rule and not a glance at a table.

## A bank of questions that outlives every candidate

A candidate is judged by the questions it is asked, and the questions have to be the same for everyone: for the candidate, for the incumbent it is compared with, and for the candidate of next month. A **probe bank** is that set of questions: a fixed, versioned collection of evaluation items grouped by the capability they test, each with the one answer that counts as correct, frozen before any candidate is trained and never used as training data. Four properties make it a bank and not a pile of test rows. The items carry their *capability*, so that a regression can be named ("it lost task A") and not only counted. They are *disjoint from every row any candidate trains on*, by construction and not by luck, because a probe that was trained on measures memorisation and flatters every number downstream. They are *frozen*, which means that the bank has an identity, a digest of its own contents, that a report cites, so that two scores are comparable exactly when they cite the same digest and a changed probe is a different bank. And their answers are *checkable by a program*, so that no second model's opinion enters a number that decides a promotion.

The bank of this chapter's toy has two capabilities, A and B, and 100 probes of each, drawn after every training and test row.

<!-- worked: continual_learning.probe_bank -->
The probe bank of the toy has two capabilities and 100 probes each, frozen before any candidate is trained: capability A is the 100 bank rows of A and capability B the 100 bank rows of B (the last 200 draws of the data generator, after the 1,000 rows above), disjoint from the training and test rows by construction (separate draws). A probe is `capability|index|x0|x1|y` with x0 and x1 written with 6 decimals; the identity of the bank is the SHA-256 of the 200 lines joined by a line feed (no final one), in the order A then B.

| item | value |
|---|---:|
| probes in capability A / B | 100 / 100 |
| y = 1 in A / in B | 47 / 60 |
| first probe | `A|0|-1.946365|-0.561638|0` |
| identity of the bank (first 16 hex digits of SHA-256) | da691930399928f7 |
| identity after flipping the label of probe A|0 | be251d4089e6d3ff |
<!-- /worked -->

Flipping the label of one probe in 200 changes the digest completely, which is the property that makes a digest a version.

## Comparing a candidate with the incumbent, probe by probe

The comparison has the structure of the paired test of Volume I. Score the incumbent and the candidate on the *same* probes, in the same order, and look at the probes where they differ. On each probe the pair of outcomes is one of four: both right, both wrong, only the candidate right (a **win**) or only the incumbent right (a **loss**); a probe that both get right or both get wrong is a tie. Ties carry no information about which is better, and the exact sign test of Volume I, which this chapter uses throughout, drops them: with $w$ wins and $l$ losses it computes $p = \Pr[\mathrm{Binomial}(w + l, \tfrac12) \ge w]$, one-sided, for the hypothesis "no better", and it is $1$ when $w + l = 0$. A probe whose score is not a number (a verifier that could not score it) must be neither a win nor a loss, and counts as a tie, which is what a comparison with a NaN does in most languages. A small fact about the test fixes how many probes a gate needs: five wins and no losses give $p = 2^{-5} = 0.03125$, which is below $0.05$, and four give $0.0625$, which is not, so a probe set that cannot supply five discordant probes cannot produce a significant result however good the candidate is. With 8 discordant probes the fewest wins that reach it are 7.

One test is not a decision. A candidate that wins on its own task and has destroyed an old one is not better, and a significant gain that is one probe in a thousand is not worth the replacement. The gate of this chapter is the conjunction of four bars, checked in a fixed order, and the first one that fails names the cause. In the pseudocode of the book:

```text
function gate_candidate(own_c: array<real>[n], own_i: array<real>[n], blocks: list<(real, real)>,
                        alpha: real, min_effect: real, pooled: real, block: real)
                       -> (text, optional text):
    require n >= 1
    w = 0
    l = 0
    for k in 0 .. n:
        if own_c[k] > own_i[k]:
            w = w + 1
        else if own_c[k] < own_i[k]:
            l = l + 1
    p = 1.0
    if w + l > 0:
        p = sum(comb(w + l, i) for i in w ..= w + l) / 2^(w + l)
    effect = mean(own_c) - mean(own_i)
    anchor = 0.0
    worst = 0.0
    if len(blocks) > 0:
        anchor = mean([i for (c, i) in blocks]) - mean([c for (c, i) in blocks])
        worst = max([i - c for (c, i) in blocks])
    if p > alpha:
        return ("reject", "not significant")
    if effect < min_effect:
        return ("reject", "effect too small")
    if anchor > pooled:
        return ("reject", "anchor regressed")
    if worst > block:
        return ("reject", "block regressed")
    return ("promote", none)
```

`comb(n, k)` is the binomial coefficient, `mean` and `max` of a list are its mean and largest element, and a block is a pair (candidate mean, incumbent mean) over the probes of one earlier episode.

Each argument is one of the bars. `own_c` and `own_i` are the 0/1 outcomes of the two models on the probes of the capability the candidate was trained for; `blocks` holds, for each earlier capability, the pair (the candidate's mean, the incumbent's mean) on its probes. The four bars are the significance of the gain ($p \le \alpha$, here $0.05$), a floor under its size (the mean gain on the own probes at least $0.15$), the *pooled* anchor (the mean over blocks of the incumbent's score minus the candidate's must be at most a budget, here $0.05$) and the *block* bar (no single block may have dropped by more than $0.20$). The decision is the same whatever the order of the bars, since promotion needs all of them; the order only chooses which cause is reported, and every number is computed either way. The next block runs the candidates of the last section through it, with capability B as the target and capability A as the one earlier block, so that the pooled and block bars see the same drop.

![The promotion gate: the candidate and the incumbent are scored on the same frozen probes and pass four bars in order (significance of the gain, size of the gain, pooled anchor drop, worst block drop); a candidate that passes all four is promoted, and the first bar that fails rejects it and names the cause.](img/promotion-gate.svg)

<!-- worked: continual_learning.gate_on_candidates -->
Each candidate of the previous blocks is compared with the incumbent (the network trained on A) on the probe bank. Per probe the outcome is 1 (right) or 0; on the 100 probes of capability B (the target) a win is a probe the candidate gets right and the incumbent wrong, a loss the reverse, and ties (both right or both wrong) are left out of the sign test. p is the one-sided exact P(Binomial(wins + losses, 1/2) >= wins), 1 when there is no discordant probe. 'Effect' is mean(candidate) - mean(incumbent) on B and 'anchor drop' is mean(incumbent) - mean(candidate) on the 100 probes of capability A. The four bars in this order: p <= 0.05, effect >= 0.15, anchor drop <= 0.05 (the pooled bar; with one earlier block it is that block's drop) and the same drop <= 0.2 (the block bar); the first bar that fails names the cause. The candidates are the ones of the remedy table: 'naive' is B alone (step size 0.2, 200 steps), 'replay m', 'L2 lambda', 'EWC lambda' as before, 'adapter on' the rank-one adapter always on and 'joint' the network trained on both from the start.

| candidate | wins on B | losses on B | p | effect on B | anchor drop on A | decision |
|---|---:|---:|---:|---:|---:|---|
| naive | 100 | 0 | 7.89e-31 | 1.000 | 1.000 | reject: anchor regressed |
| replay 20 | 100 | 0 | 7.89e-31 | 1.000 | 0.020 | promote |
| replay 100 | 99 | 0 | 1.58e-30 | 0.990 | 0.020 | promote |
| L2 0.3 | 92 | 0 | 2.02e-28 | 0.920 | 0.520 | reject: anchor regressed |
| L2 1 | 22 | 0 | 2.38e-07 | 0.220 | 0.020 | promote |
| EWC 1000 | 100 | 0 | 7.89e-31 | 1.000 | 1.000 | reject: anchor regressed |
| adapter on | 100 | 0 | 7.89e-31 | 1.000 | 1.000 | reject: anchor regressed |
| joint | 99 | 0 | 1.58e-30 | 0.990 | 0.040 | promote |
<!-- /worked -->

The table reads in a way the previous one could not. Training on B alone wins all 100 discordant probes of B ($p = 7.9 \times 10^{-31}$) and is rejected: its anchor drop is $1.000$, every probe of A lost. The same is true of EWC at $\lambda = 1000$ and of the adapter left on. The L2 anchor at $\lambda = 0.3$ is rejected too, for an anchor drop of $0.52$, while the one at $\lambda = 1$ is promoted: it wins 22 probes of B and loses none ($p = 2.4 \times 10^{-7}$, effect $0.22$) and gives up $0.02$ of A, inside the budget. Replay with 20 and with 100 old rows and the joint training are promoted, with anchor drops of $0.02$, $0.02$ and $0.04$. Look at what the gate did not use: it never asked how each candidate was trained. It saw outcomes on frozen probes, so it applies unchanged to a candidate that came from a remedy nobody has invented yet. And look at what a *pooled* comparison would have done with the naive candidate: it gains 100 probes on B and loses 100 on A, a net change of zero, and a single test over the bank would have called it no better, without saying why. The probe bank's capabilities are what let the gate say *which* capability is lost.

## Many blocks, one decision

The toy has one earlier capability. A system that keeps learning has as many as it has learned, and the anchor bars then meet a problem that the paired test alone does not have. Take one earlier block that drops by $d$ while the other $k - 1$ hold. The pooled mean of the blocks drops by $d/k$, so the pooled bar with its budget of $0.02$ stops seeing a drop of $d$ as soon as $k \ge d/0.02$: with $d = 0.30$, at 15 blocks, and with $d = 1$, a block that lost everything, at 50. Longer runs protect themselves less, exactly backwards. This is the finding that brain's `crates/promote` records (the section on it below), and the next block is the arithmetic.

<!-- worked: continual_learning.pooled_blindness -->
One earlier block drops by d, the other k - 1 earlier blocks hold. The pooled mean of the blocks drops by d/k. 'Pooled bar fires' is d/k > 0.02 (the default budget of brain's gate); 'block bar fires' is d > 0.20 (the pre-registered per-block bar). Exact arithmetic; the pooled bar stops seeing a drop of d once k >= d / 0.02, which is 15 blocks for d = 0.30 and 50 for d = 1.00.

| drop d | earlier blocks k | pooled drop d/k | pooled bar fires | block bar fires |
|---|---:|---:|---:|---:|
| 0.30 | 1 | 0.3000 | yes | yes |
| 0.30 | 2 | 0.1500 | yes | yes |
| 0.30 | 5 | 0.0600 | yes | yes |
| 0.30 | 10 | 0.0300 | yes | yes |
| 0.30 | 15 | 0.0200 | no | yes |
| 0.30 | 20 | 0.0150 | no | yes |
| 0.30 | 50 | 0.0060 | no | yes |
| 0.30 | 100 | 0.0030 | no | yes |
| 1.00 | 1 | 1.0000 | yes | yes |
| 1.00 | 2 | 0.5000 | yes | yes |
| 1.00 | 5 | 0.2000 | yes | yes |
| 1.00 | 10 | 0.1000 | yes | yes |
| 1.00 | 15 | 0.0667 | yes | yes |
| 1.00 | 20 | 0.0500 | yes | yes |
| 1.00 | 50 | 0.0200 | no | yes |
| 1.00 | 100 | 0.0100 | no | yes |
<!-- /worked -->

The repair is the block bar, which looks at each block alone. But then a gate has $k$ chances to see a regression, and with $k$ large it will see some that are not there. This is the multiple-comparisons problem of Volume I in a new position: if the rule is "block the candidate if any earlier block is significantly worse", twenty independent tests at $0.05$ would block a harmless candidate with probability $1 - 0.95^{20} = 0.6415$ and Bonferroni's level $0.05/20 = 0.0025$ would bring it to $0.0488$. The next block measures it on simulated paired probes, for 20 blocks of 48 probes, comparing four kinds of rule.

<!-- worked: continual_learning.many_blocks -->
20 earlier blocks of 48 probes each. Generator `continual_learning.blocks` (seed 0x4ba17307593c160b, 5449763499494086155) One generator; the 400 repeats of the first population (candidate equal to the incumbent) are drawn first, then the 400 of the second (block 0 truly worse); a repeat draws its 20 blocks in order and a block its 48 probes in order, one `uniform()` per probe. A probe is 'both right' if u < 0.70, 'both wrong' if u < 0.90, 'only the candidate right' (a win) if u < 0.95 and otherwise 'only the incumbent right' (a loss); in the second population the thresholds of block 0 are 0.55, 0.65 and 0.70 (wins 0.05, losses 0.30, a true mean drop of 0.25). A block's mean difference is (wins - losses)/48. The lower bound of the non-inferiority rows is mean - 1.644854 * sqrt(s^2/48) with s^2 the unbiased variance of the 48 differences (divisor 47) and the Gaussian value 1.644854; the sign tests are one-sided exact on the discordant probes of a block, and the share is of the 400 repeats in which the rule blocks the candidate. For comparison, independent continuous tests would give 1 - 0.95^20 = 0.6415 and 1 - (1 - 0.0025)^20 = 0.0488.

| rule that blocks the candidate | share blocked, equal candidate | share blocked, block 0 worse by 0.25 |
|---|---:|---:|
| any block significantly worse at 0.05 (sign test, one-sided) | 0.1900 | 0.9100 |
| the same at 0.05/20 = 0.0025 (Bonferroni) | 0.0000 | 0.4175 |
| fixed bar: some block drops by more than 0.2 | 0.0000 | 0.6950 |
| non-inferiority: some block's lower bound at or below -0.05 | 1.0000 | 1.0000 |
| non-inferiority: some block's lower bound at or below -0.10 | 1.0000 | 1.0000 |
| non-inferiority: some block's lower bound at or below -0.15 | 0.7150 | 0.9975 |
| non-inferiority: some block's lower bound at or below -0.20 | 0.1375 | 0.9825 |
| non-inferiority: some block's lower bound at or below -0.25 | 0.0125 | 0.9275 |
<!-- /worked -->

Read the first two rows. An exact sign test is discrete, with a handful of discordant probes per block, so its real size at the nominal $0.05$ is far below $0.05$ and the rule blocks an equal candidate in $0.19$ of the repeats and not in $0.64$; Bonferroni's level removes the false blocks altogether (zero of 400) and lowers the share of a real regression of $0.25$ that is caught from $0.91$ to $0.4175$. The correction trades one error for the other, and the table shows the exchange rate. The third row is the fixed bar of brain's gate: a block that drops by more than $0.20$ blocks the candidate, with no inference at all. It blocks no equal candidate in 400 and catches the real regression in $0.695$ of the repeats, because the observed drop of a block with a true drop of $0.25$ is itself noisy.

The last five rows turn the fixed bar into a *test*. Let $\bar d$ be the mean paired difference of a block, $s^2$ the unbiased variance of the paired differences (divisor $n - 1$) and $\mathrm{SE} = \sqrt{s^2/n}$ the paired standard error. A candidate is **non-inferior** on a block within a **non-inferiority margin** $\delta$ when the lower end of its one-sided $95\%$ interval, $\bar d - 1.644854\cdot\mathrm{SE}$, lies above $-\delta$: the hypothesis "this block is at least $\delta$ worse" is rejected. The Gaussian value is used rather than a $t$ quantile with 47 degrees of freedom, which is a little larger and would lower every bound a little. A candidate is cleared only if every block is non-inferior; this is an intersection of the $k$ hypotheses and needs no correction of the level to keep its error at $0.05$, because to be promoted wrongly the candidate has to be wrongly cleared on *all* blocks at once, though it pays for the guarantee in power. The table shows the price as a function of $\delta$. A margin of $0.05$ or $0.10$ is below what 48 probes can resolve, since a block of 48 with a tenth of its probes discordant has a standard error of about $0.046$: the lower bound of a candidate that is exactly equal sits near $-0.075$ and the gate blocks it in every one of the 400 repeats, which makes the margin a gate that cannot say yes. At $\delta = 0.20$ the equal candidate is blocked in $0.1375$ of the repeats and a block that truly dropped by $0.25$ in $0.9825$, and at $\delta = 0.25$ in $0.0125$ and $0.9275$. A margin has a unit (a fraction of a block's probes), a direction (it is about *being worse*) and a floor set by the standard error: it must be larger than about $1.6$ standard errors of a block before a candidate that did not change can satisfy it on one block, and since all 20 blocks must clear it, a good deal more: at $\delta = 0.15$, about 3.3 standard errors, the equal candidate is still blocked in $0.715$ of the repeats.

The next block shows the same arithmetic on one block with $0, 1, 2, \ldots$ lost probes and no wins, with brain's fixed bar beside it.

<!-- worked: continual_learning.margin_example -->
One block of 48 probes where the candidate wins none and loses 'lost' of them (so 'lost' discordant probes and the rest ties). Mean difference m = -lost/48, s^2 = (lost - 48 m^2)/47, standard error sqrt(s^2/48), one-sided lower bound m - 1.644854 * SE. A candidate is non-inferior within the margin delta when the lower bound is above -delta; the last column is brain's fixed bar (a drop of at most 0.20, no inference).

| probes lost | mean difference | standard error | lower bound | non-inferior, delta 0.10 | non-inferior, delta 0.20 | fixed bar passes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0.0000 | yes | yes | yes |
| 1 | -0.0208 | 0.0208 | -0.0551 | yes | yes | yes |
| 2 | -0.0417 | 0.0291 | -0.0896 | yes | yes | yes |
| 4 | -0.0833 | 0.0403 | -0.1496 | no | yes | yes |
| 6 | -0.1250 | 0.0482 | -0.2043 | no | no | yes |
| 8 | -0.1667 | 0.0544 | -0.2561 | no | no | yes |
| 10 | -0.2083 | 0.0592 | -0.3058 | no | no | no |
<!-- /worked -->

One lost probe of 48 is a drop of $0.0208$, and it already exceeds a pooled budget of $0.02$: with a budget that small a gate is a gate against *any* regression of a one-block bank. That is a reason for the toy's reader to use $0.05$ and for brain's reader to sample blocks and widen the bar to match, as a later section shows.

## Giving the old version back

A promotion is a decision about what is served from now on, and it can be wrong in ways the gate could not see: the probes were a sample, an audit was thinner than the bank, or the world changed after the decision. A **rollback** returns to the previous incumbent. It needs three things and each is a design choice: the previous incumbent must still exist and be byte-identical to what was served (the artefact was kept and never edited), the pointer to what is served must move atomically (a temporary file and a rename, so that a crash leaves one version and never half of two), and something must notice (the audit of the next sections). It costs the storage of one more version and nothing per request. Rollback is also not a substitute for the gate: a gate that promotes carelessly and rolls back often wastes every training run it throws away, and the data it learned from is already in the stream.

brain's reader takes the other design and says so. Its run directory is documented as having no best-versus-latest distinction and no rollback in the reader itself, because "Nothing ungated ever lands in this directory": an adapter is written only after the gate promoted it against frozen probes and the blocks of earlier episodes, so the latest state is held to be the best the reader has evidence for. That is a reasoned position and a limited one. The evidence is the audit, and the audit of an unbounded stream is bounded (below), so a regression smaller than the bar, or one that lands between two audits of the block it hurts, is promoted and stays until someone withdraws it. The reader reports rather than reverts: it keeps a retention matrix of what earlier episodes score now against what they scored when learned (`worst_drop` is the largest fall), and its acceptance block refuses to report a headline when that is bad. Withdrawing an adapter means changing what is served, and the served incumbent belongs to the `Learner` and not to the reader, so the choice between withdrawing it, retraining it and accepting the drop is the learner's release policy, and a caller that wants a rollback reads `worst_drop` and the ledger row of the promotion that caused it.

## A loop that reads, trains and promotes

Put the pieces in a loop and you have a **continual reader**: a process that ingests an unbounded stream of new experience, one *episode* at a time, decides for each whether to keep what it learned, trains a candidate, promotes it only through the gate, and can say at any moment what it is still able to do. Four filters in order of cost keep most of the stream away from the expensive step. A *screen* refuses text with no structure in it at no model cost. A *reach* test spends one forward pass: an episode the incumbent already predicts has nothing to teach and is dropped as known, and one it cannot predict at all is deferred, since the same text can be out of reach now and inside it later. An *evidence* test refuses an episode that is too small to have produced a significant result, which with the sign test above means fewer than about a dozen probes. Only then comes the expensive step: train a candidate, score two models on the episode's probes and on the earlier episodes' probes, and gate. In the pseudocode of the book, with the model-specific parts left as functions of the chapter's toy:

```text
function read_episode(inc: array<real>[P], ep: record, bank: list<int>, eps: list<record>,
                      use_replay: bool, use_gate: bool, pooled: real, rng: Generator)
                     -> (array<real>[P], text):
    if mean_loss(inc, ep.train) <= 0.20:
        return (inc, "already known")
    rows = ep.train
    if use_replay and len(bank) > 0:
        rows = rows + replay_rows([eps[j].train for j in bank], 60, rng)
    cand = gd(inc, rows, 0.2, 200)
    own_c = correct(cand, ep.probes)
    own_i = correct(inc, ep.probes)
    blocks = [(mean(correct(cand, eps[j].probes)), mean(correct(inc, eps[j].probes))) for j in bank]
    (decision, cause) = gate_candidate(own_c, own_i, blocks, 0.05, 0.15, pooled, 0.20)
    if use_gate and decision == "reject":
        return (inc, "rejected: " + cause)
    return (cand, "promoted")
```

Here `bank` lists the promoted episodes in promotion order, `replay_rows(pools, m, rng)` draws $m$ rows without replacement from the concatenated rows by a partial Fisher-Yates (`j = i + rng.integer(len - i)` for $i = 0, \ldots, m-1$; the pool is never smaller than $m$ here, and a pool of exactly $m$ is still shuffled), `a + b` on two lists is their concatenation, `mean_loss` and `correct` are the mean cross-entropy and the 0/1 outcomes of the network above, and `gd(t, rows, eta, steps)` is `steps` applications of the plain step of this chapter. An arm without the gate promotes whatever it trains; the table below notes what the gate would have said.

The toy stream has four episodes, one per cluster, at $x_0 = -3, -1, 1, 3$, with the answer above the line $x_1 = 0$ being class 1 in episodes 0 and 3 and class 0 in episodes 1 and 2. Episodes are 60 training rows and 48 probes (the probes of an episode are frozen when it is ingested and never trained on), and 400 further test rows measure what the incumbent can do afterwards.

<!-- worked: continual_learning.reader_world -->
Four episodes in a fixed order, from generator `continual_learning.stream` (seed 0x9e137ab42c018112, 11390582796801114386). A normal() call that finds a held value returns it without drawing. One generator, episode after episode; an episode draws its 60 training rows, then its 48 probes, then its 400 test rows, each row as in the first block (a row draws `uniform()`, `normal()`, `normal()`). Episode k has the centre -3.0, -1.0, +1.0, +3.0 and the sign +1, -1, -1, +1 for k = 0 .. 3 (so the answer above x1 = 0 is class 1 in episodes 0 and 3 and class 0 in episodes 1 and 2). The starting incumbent is the network of the first block with the same initial numbers (generator `continual_learning.init`), answering class 0 everywhere.

| episode | centre | sign | y = 1 in train (of 60) | y = 1 in probes (of 48) | y = 1 in test (of 400) |
|---|---:|---:|---:|---:|---:|
| 0 | -3.0 | +1 | 33 | 20 | 200 |
| 1 | -1.0 | -1 | 30 | 17 | 199 |
| 2 | +1.0 | -1 | 28 | 25 | 218 |
| 3 | +3.0 | +1 | 33 | 15 | 184 |
<!-- /worked -->

Five arms read it, from the same initial numbers: no gate and no replay (the naive reader), replay without the gate, the gate without replay, replay with the gate, and replay with the gate with the pooled budget raised from $0.05$ to $0.10$. Replay draws as many old rows as the episode has own rows (a ratio of $1.0$, 60 rows).

<!-- worked: continual_learning.reader_arms -->
The four arms on the stream of the previous block. Every arm starts from the same initial numbers and restarts the generator `continual_learning.replay` (seed 0xb99c0ac8b93b28bc, 13374576850600929468); a candidate is trained from the incumbent for 200 steps of size 0.2 on the episode's own rows and, in the replay arms, as many rows drawn without replacement from the rows of the promoted episodes (ratio 1.0, 60 rows: a partial Fisher-Yates, `rng.integer(len - i)` for i = 0 .. 59, in the order drawn; a pool of exactly 60 still does the shuffle, and a pool is never smaller since an episode is held only after it is promoted). Before training, triage drops an episode whose mean loss under the incumbent on its own rows is at most 0.2 ('already known'). The gate is the one of the previous block: the sign test at 0.05 on the 48 own probes, effect at least 0.15, the pooled anchor drop over all promoted episodes' probes at most 0.05 and the worst block drop at most 0.2. An arm without the gate promotes whatever it trains. The outcome of each episode:

| arm | episode | outcome |
|---|---|---|
| naive | 0 | promoted |
|  | 1 | promoted (the gate would say: anchor regressed) |
|  | 2 | already known |
|  | 3 | promoted (the gate would say: block regressed) |
| replay | 0 | promoted |
|  | 1 | promoted (the gate would say: anchor regressed) |
|  | 2 | promoted |
|  | 3 | promoted (the gate would say: anchor regressed) |
| gate | 0 | promoted |
|  | 1 | rejected: anchor regressed |
|  | 2 | rejected: anchor regressed |
|  | 3 | already known |
| replay+gate | 0 | promoted |
|  | 1 | rejected: anchor regressed |
|  | 2 | promoted |
|  | 3 | rejected: anchor regressed |
| replay+gate, pooled budget 0.10 | 0 | promoted |
|  | 1 | promoted |
|  | 2 | promoted |
|  | 3 | promoted |

Correct classifications on the 400 test rows of each episode after the last episode, the number of promoted episodes, and backward transfer (the mean over the promoted episodes except the last of the test accuracy now minus the accuracy at its own promotion, in fractions of 400; none when fewer than two were promoted):

| arm | episode 0 | episode 1 | episode 2 | episode 3 | promoted | backward transfer |
|---|---:|---:|---:|---:|---:|---:|
| naive | 363 | 3 | 0 | 400 | 3 | -0.5425 |
| replay | 365 | 378 | 369 | 391 | 4 | -0.0525 |
| gate | 400 | 0 | 0 | 400 | 1 | none |
| replay+gate | 395 | 204 | 393 | 55 | 2 | -0.0125 |
| replay+gate, pooled budget 0.10 | 365 | 378 | 369 | 391 | 4 | -0.0525 |
| joint on all four (800 steps of size 0.5, 240 rows of episodes 0 to 3 in order), range over the run and its four nudged copies | 382 to 387 | 367 to 384 | 375 to 394 | 373 to 381 | 4 | - |
<!-- /worked -->

The table says more than the earlier ones, and what it says is not that a gate improves accuracy. The naive reader promotes three episodes and ends with 3 of 400 right on episode 1 and 0 on episode 2, a backward transfer of $-0.542$; the gate would have rejected two of its three promotions, for an anchor regression and for a block regression. The gate without replay is stable and stuck: it promotes episode 0, then rejects episodes 1 and 2 because each would wreck the first, and episode 3 is dropped by triage as already known, since episode 3 asks what episode 0 asked, in another cluster. It keeps 400 of 400 on episode 0 and learns nothing else. Replay without the gate learns all four, at 365, 378, 369 and 391 of 400 on the four episodes and a backward transfer of $-0.052$, near the joint reference of 387, 373, 377 and 379. The interesting row is replay with the gate at the budget of $0.05$: it promotes two episodes and refuses two for an anchor regression, and ends *worse* than replay alone on episodes 1 and 3 (204 and 55) and better on 0 and 2 (395 and 393), with a backward transfer of $-0.013$. The gate did not fail. It saw that candidate 1, trained with replay, had cost the first episode's probes more than $0.05$ of their score, and refused to trade that for the new episode. The last arm shows what that bar was: at a pooled budget of $0.10$ the gate promotes all four and the arm becomes the one without a gate. A gate is a set of dials, and what it expresses is how much regression the owner will pay for how much learning; it cannot choose them. What it can do is make the choice visible, with a named cause for every refusal.

A reader that keeps refusing is telling its owner something, and what depends on why. When the promote rate over a recent window falls below a floor, the reader can ask one expensive question: train a single network jointly on everything promoted so far and score it on the same probes, which bounds what any schedule could have achieved. If the joint network passes and the sequential one does not, the capacity exists and the schedule lost it (*interference*; the remedy is more rehearsal and a gentler step). If neither passes, the capacity is not there (*saturation*; the remedy is more capacity, an adapter, and more rehearsal would only spend the budget on old material). The toy's gate-only arm is the first case: the joint reference gets 387, 373, 377 and 379 of 400 and the gate-only arm 400, 0, 0 and 400. A fourth case, where the joint network scores *below* the sequential one, is not a diagnosis but a sign that the measurement is broken, and the right action there is none.

The last cost left to count is the audit. Checking every earlier episode on every decision is correct and quadratic: with $N$ promoted episodes of 48 probes, a decision decodes $2 \times 48 \times N$ probes, the factor two for the two arms. A stream cannot afford that, so brain's reader fixes the budget and lets the *claim* change shape: not "nothing was forgotten" but "any regression of more than $d$ on any earlier episode is detected within $L$ episodes".

<!-- worked: continual_learning.audit_cost -->
Probe decodes spent on re-checking the bank in one episode, counted for both arms of the gate (the candidate and the incumbent). 'Full' re-checks every earlier episode's 48 probes: 2 x 48 x N decodes for N promoted episodes. 'Bounded' is the schedule of brain's `audit::schedule` with its default budget: 192 decodes per arm a tick, blocks of 16 probes, so 12 blocks a tick, of which 3 go to the canary and 9 to a strict round-robin; every episode is revisited within ceil(N / 9) ticks.

| episodes in the bank N | full re-check, decodes per episode read | bounded re-check, decodes per episode read | ticks to revisit every episode |
|---|---:|---:|---:|
| 10 | 960 | 384 | 2 |
| 100 | 9600 | 384 | 12 |
| 1000 | 96000 | 384 | 112 |
| 10000 | 960000 | 384 | 1112 |
<!-- /worked -->

The table counts decodes and ticks, not seconds. With the default budget of 192 decodes per arm a tick in blocks of 16 probes the audit costs $2 \times 192 = 384$ decodes however long the run is, where the full re-check costs 960 at ten episodes and 960 000 at ten thousand; what degrades is the number of ticks to come round to every episode, $\lceil N/9 \rceil$, from 2 at ten episodes to 1112 at ten thousand. The price of the smaller sample is paid in the bar: a block of 16 probes has a standard error up to $0.5/\sqrt{16} = 0.125$, so the block bar has to be $\max(0.20, 1/\sqrt{16}) = 0.25$ to keep ordinary wobble from tripping it.

## How `crates/promote` decides

`crates/promote` is the gate of this chapter without a model: its source says it is a crate whose three pieces "none of which knows what a model is". `crates/promote/src/stats.rs` holds the exact one-sided paired sign test, `crates/promote/src/gate.rs` the gate, and `crates/promote/src/env.rs` the seam through which a task family supplies prompts and a deterministic reward that a stored run artefact re-derives byte for byte. It exists because of a cycle. Before it, the gate lived in `crates/rl` and computed its significance bar with a statistic from `crates/bench`, which links every model it benchmarks, and a model crate is one of those; the dependency `brain-qwen3 -> brain-rl -> brain-bench -> brain-qwen3` meant that a model crate could not expose "gate this candidate adapter" at all. Commit `71c2a9ef1` (September 2026) says an earlier attempt found the cycle by adding the dependency and reading the dependency tree, and fixed it by moving the arithmetic, which needed no model, below every model: the gate, the sign test and the environment seam now live in a leaf crate, and `rl` and `bench` re-export them so that no caller changed. The layering is a check and not a comment: `scripts/gates/check-crate-layers.sh` lists `brain-promote` and `brain-audit` among the leaf crates and fails the build if the dependency closure of one ever reaches the model layer.

The gate itself is the one in the pseudocode above, with a fifth bar the chapter's toy leaves out. `crates/promote/src/gate.rs` takes paired per-task scores for the candidate and the incumbent, an anchor score for each, a completion entropy for each and, optionally, per-block anchor pairs. It checks, in this order, the sign test (`alpha`), the effect size (`min_effect_size`), the pooled anchor (`anchor_budget`), the per-block anchor (`max_block_drop`, off unless a caller supplies blocks and a finite bar) and a degeneracy bar: the candidate's mean completion entropy must not fall below a fraction (`min_entropy_ratio`) of the incumbent's, which catches a candidate that wins by collapsing onto one completion, something a win count cannot see. The defaults are $\alpha = 0.05$, a minimum effect of $0.02$, an anchor budget of $0.02$ and an entropy ratio of $0.5$, and the code says of the minimum effect, the anchor budget and the entropy ratio that they are "deliberately conservative starting points a caller is expected to tune per regime, not tuned science". The report carries every number whichever bar rejected, and `worst_block` and its drop are `Option`: commit `cc3ee4be4` (September 2026, with knowledge #170) found that the gate reported a worst-block drop of 0.0 when it had no blocks, which read as "every block held", and that other results did the same for quantities nobody measured. The rule it left is that something unmeasured is `None`, or NaN where a type cannot change, and in the sign test a NaN is a tie: neither `c > b` nor `c < b` holds, so the pair is excluded and never a win.

The per-block bar has a short history that is this chapter's section on blindness, in code. Commit `bc0ac6653` added `max_block_drop` after a recorded run in which one anchor block fell from $0.90$ to $0.67$ and another from $0.35$ to $0.06$ while the pooled mean stayed in budget, with the bar off by default so that no existing caller's decision changed. Commit `d6f7adc0f` then armed it for document learning and derived its value: a block mean over `MIN_HELD_OUT_PROBES` $= 48$ pass-or-fail probes has a standard error of at most $0.5/\sqrt{48} = 0.072$, and the bar `MAX_BLOCK_DROP` $= 0.20$ is about $2.8$ of them out. Its test is a pair on purpose, in `crates/promote/src/document.rs`: twenty earlier cycles at $0.9$ with one dropping by $0.30$, which the pooled bar alone promotes and the block bar rejects naming block 3, and a block that moves by $0.06$, inside one standard error, which must still promote, because "a bar that can only ever fire is not a bar". `document_gate_config` differs from the default in exactly two numbers, a minimum effect of $0.15$ in place of $0.02$ and the armed bar, and `crates/audit` uses it unchanged for its own gate (`reader_gate_config` is that function and nothing else, to avoid a second place for the thresholds to drift).

The rest of `crates/promote/src/document.rs` is what makes a probe set honest. A fact batch is a list of `{fact, probe_question, expected_answer}` triples; `FactBatch::new` refuses an empty batch, a blank field, two triples with the same normalised probe question, and a probe question that appears inside a fact that will be trained on, and `train_probe_split` refuses fewer than 48 probes. A row's identity is derived from its own normalised text, with no seed and no split tag, so that "the probe split is disjoint from the training split" is a statement that can fail. The expected answer is stored as a SHA-256 digest of its normalised text, never as the text, and the verifier compares the digest of the first line of the decoded completion: exact match with no latitude beyond case and whitespace, reward $1$ or $0$, no model judging a model. And a batch verdict is split per fact, since twenty facts promoted together can mean fifteen landed and five did not; a fact has landed only if *every* one of its probes passed.

One more piece of `crates/promote/src/gate.rs` is a control rather than a bar. `GatePolicy::CoinFlip` lets a coin decide what carries forward while the real gate still runs and is still recorded, so that a gated run which is not separated from its coin by more than seed noise has a decorative gate; its coin is the SplitMix64 finalizer of the seed and the step, so a null run reproduces like any other (commit `42da6292d`).

## How `crates/audit` reads

`crates/audit` is "the model-free core of a continual reader", and the model-driven parts are four methods of a trait, `Learner`: the loss of a text under the incumbent, the training of a candidate from rows, the scoring of the two arms on probes (one method, because a decode yields a score and a completion entropy together and two methods would decode twice), and the joint oracle. The binding to a real model is `crates/rl/src/reader.rs`, and its comments are specific about what the seam protects: training starts from what is currently served and never from a previous candidate, which may be one the gate refused; both arms are decoded from files on disk, because what is scored must be what would be served; and a promotion is the candidate file becoming the incumbent file. The orchestration lives in `crates/audit/src/reader.rs`, and one `Reader::step` is the loop above, in the same order: the screen, one forward pass, the freezing of the episode's probes, the evidence floor, the draw of the audit blocks (before training, so that both arms see the same earlier episodes), the training of a candidate on the episode's rows plus a rehearsal draw, the scoring of both arms on the episode's probes and the blocks, the gate with its block bar re-derived for the sample size, and then, for a promotion, the adapter joins the pool, the episode enters the reservoir and joins the audit rotation.

The modules of the crate are the sections of this chapter, each with the decision that shaped it. *The episode* (`stream.rs`) is one bounded stretch of one document, never spanning two, and its identity is the digest of its own bytes, so that a restated document costs one forward pass and not a training run; the order of documents is a seeded permutation of the sorted paths, so that the seed is the only source of order. *The bank* (`bank.rs`) freezes, for an episode, probes of the kind "withheld line continuation": some lines are withheld, the lines before them are the prompt, and the answer is exact. Two separations make the numbers mean something: a second draw taken uniformly from the lines the selector did *not* want (the blind probes), so that a selector that prefers easy lines shows up as a gap between selected and blind pass rates, and a zero-shot baseline frozen at ingest, since documents are not equally hard and an absolute score is not comparable between two episodes. A probe whose answer appears inside any row that would be trained on cannot be one. Commit `bf1ec4162` records how that rule was first applied: a repeated line was grounds to refuse the whole episode, and on real directories that refused almost everything, because the repeats were markdown table rules and closing braces; the line is now dropped from the held set and rejoins training, to a fixed point since a dropped line can make another one reachable, and the episode is refused only when nothing survives (`EveryProbeInTrainedRows`).

*Triage* (`triage.rs`) is the four filters. Its screen asks how much deflate finds beyond the text's own symbol entropy, with a floor of $0.15$ and a minimum length of 640 characters, because a raw compression ratio cannot tell random base64 from a manual page (both compress towards $\log_2(\text{symbols})/8$) and the entropy-relative measure can. The reach filter defers an episode with loss at or above $4.0$ and drops one at or below $0.20$; the evidence floor is `MIN_EPISODE_PROBES` $= 12$, chosen because five clean wins are the fewest that can reach $p \le 0.05$ and twelve leaves room to lose a few. *The reservoir* (`reservoir.rs`) is Algorithm R as above, over *promoted episodes* (default capacity 256), with the draws made stateless, a SHA-256 of the seed and a counter turned into an exactly uniform integer by rejection sampling (no modulo bias), so that a reservoir written to disk resumes exactly; only a promoted episode is ever offered, so rehearsal never replays what the gate refused; and a duplicate does not advance the count it is sampled against. Its module comment calls rehearsal "a measured position rather than a preference", and commit `9d1a17c61` repeats it. The measurements behind the comment are in the engine's self-improvement roadmap: the twelve-cycle GRPO study with no replay and no rehearsal pool, one seed, ended at an accuracy of $0.271$ against $0.354$ for the untrained base on its first task alone (not a like-for-like baseline, since the base's score on the other eleven tasks was not measured), and the teacher-forced regime with a rehearsal pool reached $0.815$ and $0.932$ on two seeds. The two arms differ in the training objective as well as in rehearsal, so the pair supports the comment only as far as that, and I did not re-run either for the book. The rehearsal ratio of the reader is the chapter's, rows drawn per own row, with the default $1.0$.

*The schedule* (`schedule.rs`) is the audit of the last section. Its default is a budget of 192 decodes a tick per arm, blocks of 16 probes, so 12 blocks, of which 3 go to a *canary* (episodes whose last score sat nearest the pass-or-fail boundary, redrawn every 32 ticks) and 9 to a strict *rotation*; only rotation makes the coverage claim, which is why nothing is allowed to influence it, and `detection_latency` reports $\lceil N/9\rceil$ ticks. `block_drop_bar` is $\max(0.20, 1/\sqrt{m})$ for blocks of $m$ probes. *The retention matrix* (`retention.rs`) has $R[j][j]$, what an episode scored when it was promoted, and $R[k][j]$ from the *incumbent* arm after later episodes, since a refused candidate describes a model nobody ran. It is sparse because the audit samples, and **backward transfer** is the mean of $R[\text{now}][j] - R[j][j]$ over the episodes revisited, negative when earlier episodes are worse than when learned. Its method `bwt` returns `None` when nothing has been revisited, and the module's first point is the distinction the book's toy table also draws: a backward transfer of nothing is not a backward transfer of zero (commit `fe828e45b`). *The pool* (`pool.rs`) keeps the adapters on disk, with a working set of at most 8 resident, a margin before one displaces another, a dwell during which a newcomer is safe, and a retirement that renames a file into an archive and never deletes it (the reader retires an adapter that nothing has asked for in 4096 episodes, and brings one back from the archive when the audit plans its block again, since retiring takes an adapter out of the pool and not out of the audit): rare is not dead. And *the growth diagnosis* (`growth.rs`) is the interference and saturation table of the last section, with a window of 64 episodes, a floor of $0.25$ on the promote rate, a pass line of $0.60$ and, in commit `4f149f126`, a fourth row added during implementation and then a fifth for the case in which both arms fail and the oracle still scores below the run it bounds.

*The controls and the acceptance block* (`arms.rs`, `acceptance.rs`) answer the question the toy cannot, whether the reader's numbers mean anything. A reader reports a promote rate, a retention matrix and a battery delta, and a system that learned nothing can produce all three. So there are arms: shuffled labels, in which each episode is gated against another episode's probes and the promote rate should collapse (the verdict is inconclusive, never a pass, when the real rate was too low for a collapse to show); order permutation; the spread across seeds; and injections that must be refused. A run is *defensible* only when every arm passed, and the eight clauses of `Acceptance` include that the null-gate comparison separates by more than the seed spread, that the per-block bar was armed, that a battery nobody trained on and nobody gated on did not regress, that the audit's decodes stayed within a share of the total, and that two runs at one seed are byte-identical, with a claimed effect larger than the seed repeat's difference. Its own tests state the asymmetry: a run that learned little but kept its controls is accepted, and a spectacular run with a control that could not answer is not.

Four behaviours of these two crates were found while this chapter was written and are fixed in commits `d3e40f2d2` and `142537db9`, and each is now part of what the reader does. The growth diagnosis produces an action, `RaiseRehearsal` or `Grow`, and `Reader::step` stores it in the ledger row; `RaiseRehearsal` is applied, raising the reader's rehearsal ratio for the next episode by the growth step (a factor of $1.5$) up to a ceiling (default $4.0$, `GrowthConfig.rehearsal_ceiling`) and keeping it across a resume, while `Grow` is a recommendation for the caller, since the capacity of the learner and the cost of adding to it are the campaign's policy and not the reader's. The pool's selection takes a demand score per adapter, and the reader gives it to the adapters that were asked for in the episode, one for each re-checked block and one for the new adapter, so the pool decides residency every episode. The canary of the schedule spends attention on blocks near the pass-or-fail boundary because the reader sets each block's priority from its last score: highest at the pass line of $0.60$, falling linearly to the far end of the range. And `bwt_dense`, like `Retention::bwt`, returns `None` for fewer than two tasks, so that nothing earlier to have forgotten is unmeasured and never zero, which is the rule the rest of the crate and knowledge #170 exist for; the study report and the SDK's arm report carry the `Option` (commit `45645b77b`).

## Check your understanding

1. Find the error: "After training on B for 200 steps the network classifies none of the 400 rows of A correctly, so it has forgotten everything it knew about task A and is no better than a coin."
2. Predict: the EWC run of the chapter is repeated with $\lambda = 2000$ at the same step size. Without running it, what does the stability condition say will happen, and why?
3. Explain why a gate that promotes only when all four bars pass reaches the same decision whatever the order of the bars, and what the order does change.
4. Find the error: "A good candidate is blocked if any of 20 blocks looks worse at the $5\%$ level, so one in twenty good candidates is blocked."
5. Predict: a reader has 100 earlier blocks and one of them drops from $0.9$ to $0$. Does a pooled budget of $0.02$ see it, and does a block bar of $0.20$?
6. Explain why the probes of an episode have to be frozen before training and why the rows they were drawn from must not be trained on.
7. A reader has promoted three episodes, none has been revisited by the audit, and its backward transfer is reported as $0.000$. What is wrong, and what should it report?
8. The gate-only arm of the chapter ends at $400, 0, 0, 400$ on the four episodes and the joint reference at $387, 373, 377, 379$. With a pass line of $0.60$ on the fraction of probes right, what does the growth diagnosis say, and what does it recommend?

## Summary

A model that is trained on a new task forgets the old one because nothing in the gradient knows about it: on a network of 67 numbers, 50 steps on a second task took 400 of 400 on the first to zero, by distance and not by time, and a quadratic summary of the old loss (the Fisher information) is a local statement that under-predicts the damage by an order of magnitude at the distance it happens. Four remedies were counted against each other. Replay, a reservoir-sampled buffer mixed into every step, held both tasks at 387 of 400 and 397 of 400 with a buffer of 20 rows at a cost of a fifth more work a step. An L2 anchor gave a trade with no good setting, EWC with the exact Fisher diagonal protected nothing on this network at the three stable values tried, and an adapter was not a defence but an isolation that something else has to exploit.

Deciding that a candidate is better needs a probe bank (frozen, versioned, disjoint from training, grouped by capability), a paired comparison with the incumbent, and a gate of four bars checked in order: the exact sign test on the discordant probes, a floor under the size of the gain, a pooled anchor and a block bar. A pooled mean goes blind as blocks accumulate ($d/k$), a union of block tests flags harmless candidates unless it is corrected or replaced by a margin, and a non-inferiority margin is a test that cannot be satisfied below about 1.6 standard errors of a block. A continual reader is the loop around all of it, with triage before the expensive step, replay inside it, a bounded audit after it and controls that ask whether its own numbers mean anything. Its gate is a set of dials, not a source of accuracy: at the same replay, a pooled budget of $0.05$ promoted two episodes and one of $0.10$ promoted four.

The limits are the probes. A gate can be exactly right about a bank that does not contain the capability that matters, and no amount of statistics in the decision repairs a missing question. The next chapter, the volume's project, uses this gate on an agent that trains on verified rewards.

## References

- J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", 2017.
- M. McCloskey and N. J. Cohen, "Catastrophic interference in connectionist networks: the sequential learning problem", 1989.
- D. Lopez-Paz and M. Ranzato, "Gradient episodic memory for continual learning", 2017.
- J. S. Vitter, "Random sampling with a reservoir", 1985.
- E. J. Hu et al., "LoRA: low-rank adaptation of large language models", 2021.
- S. J. Wellek, "Testing statistical hypotheses of equivalence and noninferiority", 2010.
- J. Martens, "New insights and perspectives on the natural gradient method", 2020.

## Exercises

1. **[Calculate]** A candidate wins on 14 probes, loses on 3 and ties on 83. Give the discordant pairs, the one-sided exact sign-test p-value and the effect (mean candidate minus mean incumbent) over the 100 probes, and say which bars of the gate of this chapter it passes with $\alpha = 0.05$ and a minimum effect of $0.15$.
2. **[Calculate]** A bank has $k$ earlier blocks and one of them has dropped by $0.40$. At which smallest $k$ does the pooled bar with a budget of $0.02$ stop firing, and what does the block bar of $0.20$ do at that $k$?
3. **[Calculate]** A logistic output has the logit $z = \ln 4$ on a row whose gradient of the logit is $(1, 2)$ for two numbers. Give $p$, the contribution of the row to the Fisher diagonal, and the contribution it would make to the empirical Fisher if its label is 1.
4. **[Calculate]** A reservoir of capacity 10 has been offered 40 rows. Give the probability that a given row of the 40 is in it, the probability that row 25 (counting from 0) replaced something when it arrived, and the probability that, once in, it is still there after 15 more rows (rows 26 to 40 of the stream, offered after the 40 of the first part).
5. **[Calculate]** A schedule has a budget of 96 decodes a tick per arm, blocks of 12 probes and a canary of 2 blocks. Give the blocks per tick, the rotating blocks, the number of ticks to revisit 50 episodes, the standard error bound of a block mean and the block bar.
6. **[Calculate]** A block of 48 probes has 3 wins and 9 losses. Give the mean difference, the unbiased variance of the paired differences, the standard error and the one-sided lower bound with $z = 1.644854$, and say whether the candidate is non-inferior at $\delta = 0.20$ and at $\delta = 0.25$.
7. **[Implement]** Reproduce `continual_learning.world_rule` and `continual_learning.pretrained`: the generators and their draw orders, the 67 initial numbers, the 300 steps of descent at the step size $0.5$, and the gradient check against central differences with the criterion $10^{-6}$.
8. **[Implement]** Reproduce `continual_learning.forgetting_grid`, `continual_learning.fisher_diagonal` and `continual_learning.anchor_prediction`, to the digits printed.
9. **[Implement]** Reproduce `continual_learning.replay_table`, `continual_learning.anchor_table` and `continual_learning.adapter_table`, including the reservoir draws and the stability product for EWC.
10. **[Implement]** Reproduce `continual_learning.probe_bank` and `continual_learning.gate_on_candidates`, including the digest of the bank, and add a column that reports the cause for each rejected candidate if the four bars were checked in the opposite order.
11. **[Implement]** Reproduce `continual_learning.many_blocks`, `continual_learning.margin_example` and `continual_learning.pooled_blindness`, and add the rows for the margins $0.30$ and $0.35$.
12. **[Implement]** Reproduce `continual_learning.reader_world`, `continual_learning.reader_arms` and `continual_learning.audit_cost`: the stream's draw order, the five arms and the replay draws.
13. **[Investigate]** Repeat the EWC runs with the Fisher computed at a network trained on A for 40 steps instead of 300, so that it has not saturated, with stable values of $\lambda$. Does the premise of EWC survive when the Fisher is large, and what does the answer say about when the method can be expected to help?
14. **[Investigate]** Change the toy's blocks to 200 probes each and repeat `many_blocks` for the margins of the table. Predict first from the standard error where the equal candidate stops being blocked, and compare.

## Key terms

- **catastrophic forgetting**: the loss, abrupt and nearly complete, of what a model had learned of an earlier task when it is trained on a new one whose data no longer contains the old. <!-- concept: catastrophic-forgetting -->
- **experience replay**: mixing stored examples of what was learned before into the training of the new task, at a stated ratio of old rows to new and by a stated rule for which old rows are kept. <!-- concept: experience-replay -->
- **promotion gate**: the rule that decides whether a candidate may replace the model in use, by comparing the two on frozen probes with a paired test and requiring the gain to be significant, large enough and bought without a regression of what the incumbent could already do. <!-- concept: promotion-gate -->
- **probe bank**: a fixed, versioned set of evaluation items grouped by capability, each with a checkable answer, frozen before any candidate is trained and never used as training data, that every candidate and the incumbent are scored on. <!-- concept: probe-bank -->
- **continual reader**: a process that takes in an unbounded stream of new experience episode by episode, trains a candidate on each, promotes it only through the gate, and can report what it is still able to do. <!-- concept: continual-reader -->

## Answers

**Recall questions.**

1. The total of the summed losses is $27.1824 + 26.8443 = 54.0267$ over $7 + 5 = 12$ supervised tokens, so the loss per token of the batch is $4.5022$.
2. $\Pr[\mathrm{Binomial}(10, \tfrac12) \ge 9] = (\binom{10}{9} + \binom{10}{10})/2^{10} = 11/1024 = 0.0107$.
3. $1 - 0.95^{20} = 0.6415$; Bonferroni tests each at $0.05/20 = 0.0025$.
4. Rows that the model trained on measure how well it fit them, and a model can fit them by memorising; the held-out rows measure what the fit generalises to, which is the quantity a decision to keep the model is about.
5. The gradient is $(\theta - 3) + \lambda(\theta - 1) = 0$, so $\theta = (3 + \lambda)/(1 + \lambda) = 2$ for $\lambda = 1$: half-way between the pull and the anchor.
6. $p = \sigma(2) = 0.8808$; the derivative of $\mathrm{softplus}(z) - yz$ with respect to $z$ is $p - y = -0.1192$; and $p(1 - p) = 0.8808 \times 0.1192 = 0.1050$.

**Check your understanding.**

1. Zero of 400 on a two-class task means every row is classified wrongly, which is a perfect inversion and as informative as 400 of 400; the network has been taught the reverse answer and not lost the structure. It is not a coin (a coin would score about 200), and in the toy the 12 adapter numbers can be switched off to give 400 of 400 again.
2. The stiffness of the largest entry is $\eta\lambda F_{\max} = 0.2 \times 2000 \times F_{\max}$, twice the printed $1.812$ at $\lambda = 1000$, so $3.624$, above 2: the update for that number multiplies its distance from the anchor by $1 - 3.624 = -2.624$ each step, so it oscillates with growing amplitude and the run diverges.
3. The decision is the conjunction of four conditions, and a conjunction does not depend on the order in which its terms are evaluated. The order picks which failed bar is reported as the cause, which is why the toy's table of causes would change if the bars were reordered and its decisions would not.
4. If the 20 tests were independent and continuous, a harmless candidate would be blocked with probability $1 - 0.95^{20} = 0.6415$, not $0.05$; the exact sign test on 48 probes is discrete and gives $0.19$ in the chapter's simulation. Neither is one in twenty.
5. The pooled drop is $0.9/100 = 0.009$, below the budget $0.02$, so the pooled bar does not fire; the block bar sees the drop of $0.9$, above $0.20$, and rejects, naming the block.
6. The probes are the yardstick, and a yardstick that is chosen or shaped after training measures the training. If an answer is also in a row that is trained on, the model can pass the probe by recall of the row, so the number would be about memorisation and not about what the episode taught. Freezing the probes before training, and disjointness by construction, is what makes a gain on them evidence.
7. A backward transfer with no earlier episode revisited is not a measurement, and reporting $0.000$ reads as perfect retention, the most flattering number there is. It should report none (and the coverage: three episodes carry a baseline, none has been looked at again).
8. The joint network passes ($\ge 0.60$ on every episode) and the sequential run does not, so the capacity exists and the schedule lost it: interference. The recommended action is more rehearsal (the code names a rise of the cap by a factor of $1.5$) and not another adapter.

**Exercises.**

1. There are $17$ discordant pairs with $14$ wins, so $p = (\binom{17}{14} + \binom{17}{15} + \binom{17}{16} + \binom{17}{17})/2^{17} = (680 + 136 + 17 + 1)/131072 = 0.006363$. The effect is $(14 - 3)/100 = 0.11$. The significance bar passes and the effect bar ($0.15$) fails, so the cause is "effect too small".
2. The pooled drop is $0.40/k$ and stops exceeding $0.02$ at $k = 20$ (at exactly $0.02$ it does not fire, since the bar is strict); the block bar sees $0.40 > 0.20$ at every $k$ and fires.
3. $p = 4/5 = 0.8$, so the row contributes $p(1 - p) g_i^2 = 0.16 \times (1, 4) = (0.16, 0.64)$ to the diagonal. With the label 1 the empirical contribution is $(y - p)^2 g_i^2 = 0.04 \times (1, 4) = (0.04, 0.16)$, a quarter of it, since $(y-p)^2 = 0.04$ against $p(1-p) = 0.16$.
4. Each of the 40 rows is in the reservoir with probability $10/40 = 0.25$. Row 25 (the 26th row) replaced something with probability $10/26 = 0.3846$. A resident survives each later offer $i$ (the $(i+1)$-th row) with probability $1 - 1/(i+1)$, so over rows 26 to 40 it survives with $\prod_{i=26}^{40}\frac{i}{i+1} = 26/41 = 0.634$.
5. $96/12 = 8$ blocks, of which $\max(8 - 2, 1) = 6$ rotate; $\lceil 50/6 \rceil = 9$ ticks; a block mean over 12 pass-or-fail probes has a standard error at most $0.5/\sqrt{12} = 0.1443$ and the bar is $\max(0.20, 1/\sqrt{12}) = \max(0.20, 0.2887) = 0.2887$.
6. $\bar d = (3 - 9)/48 = -0.125$; the differences have $12$ non-zero values, so $s^2 = (12 - 48 \times 0.125^2)/47 = (12 - 0.75)/47 = 0.2394$, $\mathrm{SE} = \sqrt{0.2394/48} = 0.0706$ and the lower bound is $-0.125 - 1.644854 \times 0.0706 = -0.2411$. It is not non-inferior at $\delta = 0.20$ ($-0.2411 \le -0.20$) and is at $\delta = 0.25$.
7. The blocks of the chapter, to the digits printed; the draw orders are in the blocks.
8. The blocks of the chapter; the quadratic prediction of the damage is the full form $\tfrac12\mathbf d^{\top}F\mathbf d$ with $F$ the average of $p(1-p)\mathbf g\mathbf g^{\top}$.
9. The blocks of the chapter; the largest stability product of EWC is $0.2 \times 1000 \times \max F = 1.812$.
10. The digest of the bank is printed in the block; with the order reversed the cause column of the rejected candidates changes only where a candidate fails more than one bar (for instance a candidate with an anchor drop above $0.20$ is reported as "block regressed" and not "anchor regressed").
11. For the margin $0.30$ and $0.35$ the equal candidate is blocked in none or almost none of the repeats; the rows are computed by the same lower-bound rule as the table.
12. The blocks of the chapter; each arm restarts the generator `continual_learning.replay`.
13. Open: the quantity to report is the table of the anchor block for the less-saturated Fisher at each stable $\lambda$, with its stability product; the chapter's claim is limited to the anchor taken at 300 steps.
14. With 200 probes a block's standard error for a tenth of the probes discordant is $\sqrt{0.1/200} = 0.0224$, and the lower bound of an equal candidate sits near $-0.037$; so margins of $0.05$ should begin to clear equal candidates, which the repeated table can confirm.
