<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learning-Rate Schedules and Gradient Clipping

*Why should the step size change during training, and what do we do when a gradient explodes?*

## The problem

Chapter 7 ended with an optimizer, AdamW, and a learning rate $\eta$ that is part of it. Every table of that chapter used one $\eta$ for the whole run, and the chapter found the best constant by trying a grid. This chapter asks what that constant hides. Take the same network (the $2 \to 8 \to 1$ network of Chapter 2 on the 200 quadrant points of Chapter 7, 33 parameters), the same AdamW with decoupled decay $0.01$, 600 steps of batches of 20, and run it five times, once for each constant rate in a small grid.

<!-- worked: schedules.constant_rates -->
The network, data and starting weights are those of Chapter 7 (generator `optimizers.mlp`, seed 0xc234bb64614333c5; starting loss 0.7471). AdamW with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and decoupled decay lambda = 0.01; 600 steps of batch size 20; batches from the generator `schedules.batches` (seed 0xb0cdb4f232361fb9): shuffle the list 0 .. 199, cut it into consecutive batches of 20, reshuffle at the end of each epoch. Entries: the mean loss over all 200 examples after 2 updates; the largest loss over steps 1 to 50 (every step recorded); the loss after 100 updates; the largest loss over steps 301 to 600 (recorded every 10 steps); the mean loss at steps 550, 560, ..., 600 (the tail).

| constant eta | loss at step 2 | worst, steps 1-50 | step 100 | worst, steps 301-600 | tail |
|---|---:|---:|---:|---:|---:|
| 0.03 | 0.7203 | 0.7385 | 0.3220 | 0.1228 | 0.0933 |
| 0.1 | 0.7070 | 0.7434 | 0.1366 | 0.0863 | 0.0622 |
| 0.3 | 0.6851 | 0.8236 | 0.0900 | 0.3160 | 0.1665 |
| 1 | 1.9382 | 1.9382 | 0.2279 | 0.6031 | 0.2013 |
| 3 | 2.5987 | 6.7296 | 0.7668 | 1.5205 | 0.8388 |
<!-- /worked -->

Read the columns from left to right. At $\eta = 3$ the loss after the second update is $2.5987$, more than three times the starting loss of $0.7471$, and the worst loss in the first 50 steps is $6.7296$: the run is thrown far away from where it started, in its first steps. At the other end, $\eta = 0.03$ is calm early (worst $0.7385$, below the starting loss) but slow: after 100 steps its loss is $0.3220$ where $\eta = 0.3$ has $0.0900$. Then look at the late columns. The rates that are fast early are not good late: $\eta = 0.3$ is ahead at step 100 and then has the worse tail, $0.1665$ against $0.0622$ for $\eta = 0.1$, and its worst loss over the second half of the run is $0.3160$, a sign of a rate that keeps kicking the weights out of the valley they have found. The best constant, $0.1$, is a compromise: it is not the fastest at step 100 ($0.1366$) and it is not the quietest at the end.

A schedule is the way out of the compromise: use a rate that is small at the start, large in the middle, small again at the end. That is the first half of this chapter. The second half is about a different failure of the same step. A rate is multiplied by a gradient, and the gradient is not under our control: one batch with an unusual example can produce a gradient a hundred times the usual size, and a step along it, however well tuned the rate, throws the weights somewhere useless. **Gradient clipping** limits the size of what the rate is multiplied by: it shortens a gradient that is too long and leaves the others alone. Both are small additions to AdamW, a few lines each, and between them they decide whether the long runs of the later volumes finish.

## Before you begin

Six questions on ideas from earlier chapters, to answer from memory. The answers are at the end of the chapter.

1. ("Optimizers: From SGD to AdamW", AdamW and bias correction.) With bias correction, how large is AdamW's first update of a weight, in units of $\eta$, when the first gradient is $g_1 = 0.02$? And when $g_1 = 200$?
2. ("Linear Regression and the First Descent", Volume II, the learning rate.) Gradient descent on a quadratic whose largest curvature is $\lambda_{\max} = 40$ uses a fixed rate. What is the largest rate for which the error shrinks along every direction?
3. ("Vectors", Volume I, the norm.) What are the norm of $\mathbf{v} = (3, 4)$ and the norm of $0.2\,\mathbf{v}$, and what does multiplying a vector by a positive number do to its direction?
4. ("Complex Numbers, Frequencies and the Fourier Transform", Volume I, the cosine.) What are the values of $\tfrac12(1 + \cos\pi u)$ at $u = 0$, $\tfrac12$ and $1$?
5. ("Functions", Volume I, piecewise functions and the schedule.) A warmup of 10 steps to the peak $0.1$ uses the rate $0.1\,(t+1)/10$ at step $t$. What is the rate at $t = 4$, and at which step is the peak first reached?
6. ("Optimizers: From SGD to AdamW", the second moment.) After one step, what is Adam's corrected second moment $\hat v_1$ in terms of the first gradient, and what does that make the first update?

## Why one rate cannot be right for the whole run

*What it is for: to find, with numbers, the two reasons the best step changes during training, so that the shape of a schedule has a purpose before it has a formula.*

**Late in the run the rate sets a floor.** Chapter 7 derived what a constant rate does with noisy gradients: on one weight with curvature $\lambda$ and minibatch noise of variance $s^2$, the squared error does not go to zero but to a stationary value $V = \eta s^2/(\lambda(2 - \eta\lambda))$. For a small $\eta$ this is proportional to $\eta$: a lower rate gives a lower floor. The price of a lower rate is that the start is forgotten more slowly.

<!-- worked: schedules.noise_vs_rate -->
SGD on one weight with a quadratic loss of curvature lambda = 1 and minibatch noise of variance s^2 = 2 per step (the setting of Chapter 7's stationary variance). Columns: the stationary mean square error V = eta s^2 / (lambda (2 - eta lambda)); its square root; the number of steps in which the average error shrinks by a factor 100, ln(100) / (-ln(1 - eta lambda)).

| eta | V | sqrt(V) | steps to forget the start (factor 100) |
|---|---:|---:|---:|
| 0.5 | 0.6667 | 0.8165 | 6.6 |
| 0.25 | 0.2857 | 0.5345 | 16.0 |
| 0.125 | 0.1333 | 0.3651 | 34.5 |
| 0.0625 | 0.0645 | 0.2540 | 71.4 |
| 0.03125 | 0.0317 | 0.1782 | 145.1 |
<!-- /worked -->

Halving the rate roughly halves $V$ and roughly doubles the number of steps. So the best rate for the beginning of a run, where the error is far above any floor and speed is what counts, is large, and the best rate for the end, where the error sits at the floor and the floor is what counts, is small. A schedule spends the large rates first.

A second reason pulls the same way. Chapter 2 of Volume II showed that gradient descent on a quadratic is stable only when $\eta < 2/\lambda_{\max}$. A bowl of largest curvature $40$ accepts rates below $0.05$, one of curvature $4$ accepts rates below $0.5$. A network that moves from a wide, flat region into a narrower valley has met a larger $\lambda_{\max}$, and a rate that was safe on the plateau is then above the stability limit: the weights oscillate across the valley instead of down it. (This chapter does not measure the curvature of the network's loss; it takes the stability rule as the reason a late rate must be lower.)

**Early in the run Adam's first steps are too large.** The reason is specific to the rule of Chapter 7. Its first update is $\eta\,g_1/(|g_1| + \epsilon_{\mathrm{den}}) \approx \eta\,\mathrm{sign}(g_1)$: every weight moves by about $\eta$ whatever its gradient (the first recall question). The later steps are smaller because the first moment averages gradients that partly cancel. To see how much smaller, take a coordinate whose gradient is pure noise, $g_1, g_2, \dots$ independent with mean $0$ and variance $\sigma^2$. The first moment is $m_t = (1 - \beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$, so its variance is

$$
\operatorname{Var}(m_t) = (1-\beta_1)^2\sigma^2\sum_{i=1}^{t}\beta_1^{2(t-i)} = (1-\beta_1)^2\sigma^2\,\frac{1 - \beta_1^{2t}}{1 - \beta_1^2} = \sigma^2\,\frac{1-\beta_1}{1+\beta_1}\,(1 - \beta_1^{2t}),
$$

using the geometric sum of Volume I. Bias correction divides by $1 - \beta_1^t$, so the corrected first moment has the root mean square

$$
\mathrm{RMS}(\hat m_t) = \sigma\,\frac{\sqrt{\frac{1-\beta_1}{1+\beta_1}\,(1-\beta_1^{2t})}}{1 - \beta_1^t}.
$$

At $t = 1$ this is exactly $\sigma$ (a corrected average of one gradient is that gradient), and for large $t$ it tends to $\sigma\sqrt{(1-\beta_1)/(1+\beta_1)}$, which is $0.2294\,\sigma$ for $\beta_1 = 0.9$. So on a noise-dominated coordinate the first step is $1/0.2294 = 4.36$ times the size of the steady-state step. There is a second effect: the denominator $\sqrt{\hat v_t}$ is the root of an average of only $t$ squared gradients, an unbiased but noisy estimate. The table measures both, with 4000 simulated coordinates.

<!-- worked: schedules.adam_early_steps -->
4000 independent coordinates, each with gradients g_1, g_2, ... drawn from the standard normal distribution by the generator `schedules.early_steps` (seed 0xd14ff3aa2dcddff8): coordinate by coordinate, 300 calls of normal() each, in time order. Adam with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and bias correction. Columns: the exact RMS of the corrected first moment, sqrt((1 - beta1)/(1 + beta1) (1 - beta1^(2t))) / (1 - beta1^t) (unit gradient variance); the measured RMS of the whole Adam ratio mhat / (sqrt(vhat) + epsilon) over the coordinates (its sampling error is about 1.2%); the measured relative spread (standard deviation over mean) of sqrt(vhat) across the coordinates (a few percent of sampling error); the learning rate, as a fraction of its final value, that would keep the typical step at its steady size (the limit divided by the first column); and, for comparison, the linear warmup t/50. The limit of the first column for large t is sqrt((1 - beta1)/(1 + beta1)) = 0.2294.

| t | RMS of mhat (exact) | RMS of the Adam ratio (measured) | spread of sqrt(vhat) (measured) | rate that keeps the step steady | linear ramp t/50 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | 0.775 | 0.2294 | 0.0200 |
| 2 | 0.7081 | 0.7157 | 0.527 | 0.3240 | 0.0400 |
| 3 | 0.5795 | 0.5838 | 0.423 | 0.3959 | 0.0600 |
| 5 | 0.4521 | 0.4548 | 0.327 | 0.5074 | 0.1000 |
| 10 | 0.3301 | 0.3304 | 0.222 | 0.6949 | 0.2000 |
| 20 | 0.2592 | 0.2602 | 0.157 | 0.8850 | 0.4000 |
| 50 | 0.2306 | 0.2297 | 0.100 | 0.9949 | 1.0000 |
| 100 | 0.2294 | 0.2288 | 0.070 | 1.0000 | 1.0000 |
| 300 | 0.2294 | 0.2297 | 0.041 | 1.0000 | 1.0000 |
<!-- /worked -->

The first column is the formula, the second is the measured root mean square of the whole Adam ratio $\hat m/(\sqrt{\hat v} + \epsilon_{\mathrm{den}})$, which agrees with it to the sampling error (the denominator's noise changes little on average). The first step is $4.4$ times the steady one, the step is still $1.4$ times it at $t = 10$, and the steady value is reached near $t = 50$. The third column shows the other defect: after one step $\sqrt{\hat v}$ has a spread of $0.775$ of its own mean (it is $|g_1|$), after 10 steps $0.222$, after 100 steps $0.070$. A weight whose $\sqrt{\hat v}$ happens to be small by chance takes a step far above $\eta$. Both effects fade as $t$ grows, so the cure is a rate that starts low and rises: the fourth column is the rate, as a fraction of its final value, that would hold the typical step constant, and the last column is the straight line $t/50$ for comparison.

## Warmup

*What it is for: to keep the first updates of AdamW from throwing the weights away before its averages are calibrated.*

**Learning-rate warmup** is a rate that rises from a small value to the peak $\eta_{\max}$ over the first $W$ steps. The form used throughout this book, the one Volume I wrote as a piecewise function, is linear:

$$
\eta_t = \eta_{\max}\,\frac{t + 1}{W}, \qquad 0 \le t < W,
$$

with $t$ the 0-based index of the step (so the first update uses $\eta_{\max}/W$ and step $W - 1$ is the first at the peak; Adam's own step number is $t + 1$). Compare it with the fourth column of the table above, the rate that would hold the step constant. It is not the exact cure: the straight line $t/50$ is below the needed rate at every step before $50$ and meets it there. That is deliberate. The needed curve was computed for a coordinate of pure noise; a coordinate whose gradient is a consistent signal behaves differently, and the line, which is lower than the needed rate at every step before $W$, errs on the side of a smaller step while $\hat v$ is still being calibrated. The length $W$ is the one number to choose, and the next table varies it, at a peak ten times above the best constant of the opening table.

<!-- worked: schedules.warmup_lengths -->
The network, data and starting weights are those of Chapter 7 (generator `optimizers.mlp`, seed 0xc234bb64614333c5; starting loss 0.7471). AdamW with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and decoupled decay lambda = 0.01; 600 steps of batch size 20; batches from the generator `schedules.batches` (seed 0xb0cdb4f232361fb9): shuffle the list 0 .. 199, cut it into consecutive batches of 20, reshuffle at the end of each epoch. The learning rate is eta(t) = 3 (t + 1) / W for t < W and 3 afterwards; W = none means 3 from the first step. Entries: the mean loss over all 200 examples after the stated number of updates; 'worst' is the largest of the losses after 1, 2, ..., 50 updates.

| warmup W | step 1 | step 2 | step 3 | step 5 | step 10 | step 20 | step 50 | worst, 1-50 | step 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 5.716 | 2.599 | 3.964 | 6.730 | 2.774 | 1.040 | 0.618 | 6.730 | 0.767 |
| 10 | 0.824 | 0.755 | 0.682 | 0.994 | 0.582 | 0.929 | 0.428 | 1.525 | 0.713 |
| 30 | 0.743 | 0.695 | 0.716 | 0.792 | 0.485 | 0.963 | 0.763 | 0.970 | 0.780 |
| 60 | 0.737 | 0.696 | 0.692 | 0.715 | 0.626 | 0.208 | 0.247 | 0.737 | 0.632 |
| 120 | 0.739 | 0.711 | 0.687 | 0.685 | 0.639 | 0.482 | 0.286 | 0.739 | 0.683 |
<!-- /worked -->

![Mean loss over all 200 examples against step for the first 60 steps of AdamW at a peak rate of 3: with no warmup the loss rises from 0.7471 to 5.7 after one step and reaches 6.7 at step 5, staying above 2 until about step 17; with a warmup of 10 steps it peaks near 1.5; with 30 steps it stays near the starting loss and rises again after step 55; with 60 steps it stays at or below the starting loss for the first 50 steps.](img/warmup-spike.svg)

With no warmup the loss after the first update is $5.716$, $7.7$ times the starting loss, and the worst loss of the first 50 steps is $6.730$. With a warmup of 10 steps the worst is $1.525$, with 30 steps $0.970$, with 60 steps $0.737$ and with 120 steps $0.739$: from $W = 60$ the loss never rises above the starting value in the first 50 steps. The price of a long warmup is the steps spent at a low rate. The rate averages $\eta_{\max}/2$ over the warmup, so a warmup of $W$ steps gives up the equivalent of $W/2$ full-rate steps, 30 of the 600 for $W = 60$.

Warmup is not a replacement for the rest of the schedule. The row "warmup 60, then constant 3" of the schedule comparison below shows a run that is calm for 50 steps and then spikes again, because a peak of 3 is simply too high for the rest of the run. Warmup removes the early failure; the late one needs decay.

## Decay

*What it is for: to bring the rate down as the run settles, so that the floor of the noise falls with it.*

After the peak the rate falls. Volume I gave one decay, the exponential $\eta_t = \eta_{\max}\gamma^{t-W}$. There are four common shapes, and they differ in how they spend the steps. With $W = 30$ steps of warmup to $\eta_{\max} = 0.3$, a floor of $\eta_{\min} = 0.003$ and a run of $T = 600$ steps, for $t \ge W$:

- **Step decay** multiplies the rate by $\gamma$ after every fixed number $S$ of steps: $\eta_t = \eta_{\max}\gamma^{\lfloor (t-W)/S\rfloor}$ (here $\gamma = 0.3$ and $S = 150$).
- **Exponential decay** multiplies it at every step, $\eta_t = \eta_{\max}\gamma^{t-W}$ (here $\gamma = 0.99$).
- **Inverse square root** decay, $\eta_t = \eta_{\max}\sqrt{W/(t+1)}$, equals $\eta_{\max}$ at the end of the warmup and falls as one over the root of the step count, which is slow: it has lost only half its value at $t = 4W - 1$.
- **Cosine decay**, derived next.

Every one is kept from going below the floor $\eta_{\min}$ by a $\max$, as in Volume I.

### The cosine schedule

The decay that most large runs use comes from Volume I's cosine. The function $c(u) = \tfrac12(1 + \cos\pi u)$ goes from $c(0) = 1$ to $c(1) = 0$ as $u$ goes from $0$ to $1$: it is half a period of the cosine, shifted and scaled so that it starts at 1 and ends at 0. Let $u = (t - W)/(T - W)$ be the fraction of the decay that has been done, from $0$ at the end of the warmup to $1$ at the end of the run. The **cosine schedule** is

$$
\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min})\,\tfrac12\bigl(1 + \cos\pi u\bigr), \qquad u = \frac{t - W}{T - W}, \quad W \le t \le T,
$$

and $\eta_t = \eta_{\min}$ for $t > T$. It starts at the peak and ends at the floor, as it must. What distinguishes it from a straight line between the same two points is its slope, $\mathrm{d}c/\mathrm{d}u = -\tfrac\pi2\sin\pi u$:

<!-- worked: schedules.cosine_values -->
The decay factor c(u) = (1 + cos(pi u))/2 of the cosine schedule, a straight line 1 - u for comparison, and the slope dc/du = -(pi/2) sin(pi u).

| u | c(u) | linear 1 - u | slope dc/du |
|---|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 0.0000 |
| 0.1 | 0.9755 | 0.9000 | -0.4854 |
| 0.25 | 0.8536 | 0.7500 | -1.1107 |
| 0.5 | 0.5000 | 0.5000 | -1.5708 |
| 0.75 | 0.1464 | 0.2500 | -1.1107 |
| 0.9 | 0.0245 | 0.1000 | -0.4854 |
| 1 | 0.0000 | 0.0000 | 0.0000 |
<!-- /worked -->

The slope is zero at both ends and steepest in the middle, where it is $\pi/2 = 1.5708$ times that of the line. So the rate leaves the peak gently (at $u = 0.1$ the decay factor is still $0.9755$, against $0.9$ for the line), falls fastest in the middle, and arrives at the floor gently (at $u = 0.9$ it is $0.0245$, against $0.1$). A line has a kink at each join; the cosine joins the warmup's peak and the floor with no change of slope. Its mean over the decay is exactly the midpoint of the two rates, because the average of $\tfrac12(1+\cos\pi u)$ over $u \in [0, 1]$ is $\tfrac12$.

There is no theorem that the cosine is the best decay; it is a smooth shape with no parameter other than its two ends, and it is the one the engine of this book uses. The next table lists all of the shapes of this section, and a sixth, a hold, which comes below, on the same warmup, peak and length.

<!-- worked: schedules.shape_table -->
Peak eta_max = 0.3, floor eta_min = 0.003, warmup W = 30 steps (eta(t) = 0.3 (t + 1)/30 for t < 30), total T = 600 steps, t counted from 0. After the warmup: step decay 0.3 * 0.3^floor((t - 30)/150); exponential 0.3 * 0.99^(t - 30); inverse square root 0.3 sqrt(30/(t + 1)); cosine eta_min + (eta_max - eta_min)(1 + cos(pi u))/2 with u = (t - 30)/(T - 30) (and, in the last row, u = (t - 330)/(T - 330) after a hold of 300 steps at the peak); every decaying row is floored at eta_min. The last column is the mean rate over the 600 steps.

| schedule | t = 0 | t = 29 | t = 30 | t = 150 | t = 300 | t = 450 | t = 599 | mean rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constant | 0.30000 | 0.30000 | 0.30000 | 0.30000 | 0.30000 | 0.30000 | 0.30000 | 0.30000 |
| step decay (x 0.3 every 150) | 0.01000 | 0.30000 | 0.30000 | 0.30000 | 0.09000 | 0.02700 | 0.00810 | 0.11362 |
| exponential (0.99 per step) | 0.01000 | 0.30000 | 0.30000 | 0.08981 | 0.01989 | 0.00440 | 0.00300 | 0.05781 |
| inverse square root | 0.01000 | 0.30000 | 0.29512 | 0.13372 | 0.09471 | 0.07737 | 0.06708 | 0.11172 |
| cosine to the floor | 0.01000 | 0.30000 | 0.30000 | 0.26869 | 0.16376 | 0.05092 | 0.00300 | 0.15192 |
| warmup, hold 300, cosine | 0.01000 | 0.30000 | 0.30000 | 0.30000 | 0.30000 | 0.17729 | 0.00301 | 0.22617 |
<!-- /worked -->

![Learning rate against step from 0 to 600 for six schedules: a constant line at 0.3; a step decay with drops at steps 180, 330 and 480; an exponential curve reaching the floor near step 490; an inverse-square-root curve that is still at 0.067 at step 599; a cosine from the peak at step 30 to the floor at 600; and a warmup, a hold at the peak until step 330, and a cosine to the floor. All start at 0.01 and reach 0.3 at step 29.](img/schedule-shapes.svg)

The last column is the mean rate, which is a measure of how much total step length each schedule spends: the exponential spends little ($0.0578$), the inverse square root and the step decay about $0.11$, the plain cosine $0.152$, the held shape $0.226$.

### The floor

All of these end at a floor $\eta_{\min} > 0$, never at zero. A cosine to zero has a long tail of useless steps, and each of them costs a full forward and backward pass.

<!-- worked: schedules.floor_waste -->
Cosine schedule with peak 1, warmup 30, T = 600, as a function of the floor (a fraction of the peak). Columns: the rate of the last step; how many of the 570 decay steps use a rate below 0.01 (below a hundredth of the peak); how many use a rate below 0.05.

| floor | rate at step 599 | steps below 0.01 | steps below 0.05 |
|---|---:|---:|---:|
| 0 | 0.000008 | 36 | 81 |
| 0.01 | 0.010008 | 0 | 73 |
| 0.1 | 0.100007 | 0 | 0 |
<!-- /worked -->

With a floor of zero, 36 of the 570 decay steps (about one in sixteen) use a rate below a hundredth of the peak, and 81 use a rate below a twentieth; at a floor of $0.01$ of the peak none is below a hundredth. A weight moves by roughly the rate times $\hat m/\sqrt{\hat v}$ per step, so a step at a hundredth of the peak moves it about a hundredth as far as a step at the peak, and the 36 steps are 36 passes over the data spent not moving. The usual floor is a tenth of the peak, or a hundredth. Volume I cited the same reason from the engine's own comment, that a rate that reaches exactly zero stops training some steps before the run ends.

### Comparing the shapes

A table of numbers is not a comparison; a run is. The next table trains the network with five of the schedules, at a peak rate of 3 (ten times the best constant, to put the schedules where they are needed), over five batch orders.

<!-- worked: schedules.schedule_comparison -->
The network, data and starting weights are those of Chapter 7 (generator `optimizers.mlp`, seed 0xc234bb64614333c5; starting loss 0.7471). AdamW with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and decoupled decay lambda = 0.01; 600 steps of batch size 20; batches from the generator `schedules.batches` (seed 0xb0cdb4f232361fb9): shuffle the list 0 .. 199, cut it into consecutive batches of 20, reshuffle at the end of each epoch. The same run on five batch orders (generators `schedules.batches`, `schedules.batches.1` .. `schedules.batches.4`; seeds 0xb0cdb4f232361fb9, 0x5ff2320386ad95d9, 0xc2b97e6fd340df09, 0xe7cf0642e13ddd4d, 0x8ea4660c06fe3ddc). Peak 3 (ten times the rate that is best for a constant), floor 0.03, T = 600. Entries are means over the five orders: the largest loss over steps 1 to 50; the largest loss over steps 301 to 600 (recorded every 10 steps); the tail (mean loss at steps 550, 560, ..., 600); and the range of the tail over the orders.

| schedule | worst, steps 1-50 | worst, steps 301-600 | tail (mean of 5) | tail range |
|---|---:|---:|---:|---:|
| constant 0.1 (the best constant) | 0.749 | 0.098 | 0.0745 | 0.0622 to 0.0816 |
| constant 3 | 6.446 | 1.717 | 0.8437 | 0.7044 to 1.0408 |
| warmup 60, then constant 3 | 1.052 | 1.979 | 0.8235 | 0.6680 to 0.9910 |
| cosine from step 0 (no warmup) | 6.446 | 0.515 | 0.0924 | 0.0813 to 0.1028 |
| warmup 60, linear to 0.03 | 1.052 | 0.677 | 0.0975 | 0.0870 to 0.1055 |
| warmup 60, cosine to 0.03 | 1.052 | 0.383 | 0.0894 | 0.0818 to 0.1009 |
<!-- /worked -->

![Mean loss over five batch orders against step, from step 50 to 600: a constant rate of 0.1 falls smoothly from 0.26 to about 0.07; a constant rate of 3 stays between 0.4 and 1.0 with spikes throughout; the cosine from step 0 and the warmup followed by a cosine both start at 0.7 to 1.0 and fall to about 0.09 by step 600.](img/schedule-loss-curves.svg)

Three things stand out. The early failure belongs to warmup alone: at this peak every row without a warmup has an early worst of $6.446$, every row with one has $1.052$. The late failure belongs to decay alone: "warmup 60, then constant 3" cures the start and has a late worst of $1.979$ and a tail of $0.8235$, which is no better than the constant $3$, while the three decaying rows reach tails of $0.0894$ to $0.0975$. And the cosine is slightly the best of the decays with a warmup ($0.0894$ against $0.0975$ for the straight line, and a late worst of $0.383$ against $0.677$), with ranges over the five orders that overlap, so it is a small difference on one problem and not a ranking.

What the table does not say is that the schedule beats a well-tuned constant: the best constant, $0.1$, has a tail of $0.0745$ and the best schedule here $0.0894$. The comparison that does show the value of a schedule varies the peak, not the shape.

<!-- worked: schedules.peak_robustness -->
The network, data and starting weights are those of Chapter 7 (generator `optimizers.mlp`, seed 0xc234bb64614333c5; starting loss 0.7471). AdamW with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and decoupled decay lambda = 0.01; 600 steps of batch size 20; batches from the generator `schedules.batches` (seed 0xb0cdb4f232361fb9): shuffle the list 0 .. 199, cut it into consecutive batches of 20, reshuffle at the end of each epoch. The five batch orders of the schedule comparison. For each peak rate: a constant rate, and a linear warmup over 60 steps followed by a cosine to one hundredth of the peak at step 600. Entries are means over the five orders of the tail (mean loss at steps 550, 560, ..., 600) and of the largest loss over steps 1 to 50.

| peak | constant: tail | constant: worst 1-50 | warmup + cosine: tail | warmup + cosine: worst 1-50 |
|---|---:|---:|---:|---:|
| 0.03 | 0.0922 | 0.740 | 0.1214 | 0.747 |
| 0.1 | 0.0745 | 0.749 | 0.0770 | 0.747 |
| 0.3 | 0.1183 | 0.881 | 0.0703 | 0.745 |
| 1 | 0.2484 | 2.105 | 0.0710 | 0.742 |
| 3 | 0.8437 | 6.446 | 0.0894 | 1.052 |
<!-- /worked -->

A constant rate is good only near $0.1$: a factor of 3 up ($0.3$) its tail is $0.1183$, a factor of 10 up it is $0.2484$, a factor of 30 up it is $0.8437$ with an early worst of $6.446$. With warmup and cosine the tail stays between $0.0703$ and $0.0894$ over a range of peaks from $0.1$ to $3$, a factor of 30, and the early worst stays at the starting loss until the largest peak. The schedule does not find a better rate; it makes a wide range of peak rates safe, so that the rate does not have to be found by a search to within a factor of 2. Its low end shows the cost: at a peak of $0.03$ the schedule is slower than the constant, because it spends most of its steps below the peak.

### Warmup, hold, decay

The common shape in the engine and in most large runs combines the pieces. A warmup of $W$ steps, a hold at the peak for $T_{\mathrm{hold}}$ steps, then a cosine decay of $T_{\mathrm{cool}}$ steps to the floor, with $T = W + T_{\mathrm{hold}} + T_{\mathrm{cool}}$ and $u = (t - W - T_{\mathrm{hold}})/T_{\mathrm{cool}}$:

```text
function schedule_rate(t: int, peak: real, floor: real,
                       warmup: int, hold: int, total: int) -> real:
    require 0 <= t, "t is the 0-based step index"
    require warmup >= 0 and hold >= 0 and warmup + hold < total, "the decay needs at least one step"
    require 0.0 < floor and floor <= peak, "the floor is positive and not above the peak"
    if t < warmup:
        return peak * (t + 1) / warmup           # linear warmup; step warmup - 1 is the first at the peak
    if t < warmup + hold:
        return peak                              # hold
    u = min(1.0, (t - warmup - hold) / (total - warmup - hold))
    return floor + (peak - floor) * 0.5 * (1.0 + cos(pi * u))
```

With `hold = 0` this is the plain warmup-then-cosine curve, and with `warmup = 0` it starts at the peak. The pseudocode and the Python module that produces this chapter's tables agree on every step of the tables above (the module's `cosine` is this function).

## The step budget

*What it is for: to see what the total number of steps $T$ does to every rate of a schedule, and what it costs when $T$ is wrong.*

A cosine over the whole run depends on $T$ in every step after the warmup, because $u$ divides by $T - W$. A run planned for 400 steps and one planned for 800 use different rates from step 41 on:

<!-- worked: schedules.budget_independence -->
Peak 1, floor 0.1, warmup 40 steps. 'cosine' decays from step 40 to the floor at step T; 'hold, then cooldown of 100' holds the peak until step T - 100 and then runs the same cosine over its last 100 steps. Rows: the rate at steps 100, 200, 300 and 399 of a run planned for T = 400 steps and of one planned for T = 800.

| planned T | shape | t = 100 | t = 200 | t = 300 | t = 399 |
|---|---:|---:|---:|---:|---:|
| 400 | cosine | 0.9397 | 0.6281 | 0.2607 | 0.1000 |
| 400 | hold, then cooldown of 100 | 1.0000 | 1.0000 | 1.0000 | 0.1002 |
| 800 | cosine | 0.9862 | 0.9051 | 0.7642 | 0.5890 |
| 800 | hold, then cooldown of 100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
<!-- /worked -->

At step 200, the run planned for 400 steps is at $0.6281$ of the peak and the one planned for 800 at $0.9051$. If a run planned for 400 steps is continued, its rate at step 400 is already the floor, and everything after is trained at $0.1$ of the peak: the run has annealed itself against the short total, and extending it does not undo that. The held shape has no such dependence: its rates are the same, exactly the peak, for every step before the cooldown begins, whatever the total. That is the property Volume I's account of the engine's `LrSchedule` named: the held peak keeps every rate before the cooldown independent of the budget. A run can be stopped at any step of the hold, a cooldown of fixed length begun from that checkpoint, and the result is a finished model at that budget.

The price is a small loss against a cosine planned for exactly that budget. The next table trains each budget both ways, over the five batch orders, at the peak $0.3$ that suits the network.

<!-- worked: schedules.budget_runs -->
The network, data and starting weights are those of Chapter 7 (generator `optimizers.mlp`, seed 0xc234bb64614333c5; starting loss 0.7471). AdamW with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and decoupled decay lambda = 0.01; 600 steps of batch size 20; batches from the generator `schedules.batches` (seed 0xb0cdb4f232361fb9): shuffle the list 0 .. 199, cut it into consecutive batches of 20, reshuffle at the end of each epoch. Peak 0.3, floor 0.003, warmup 30 steps; five batch orders as in the schedule comparison. Each run is trained for its own budget B: 'cosine' decays from step 30 to the floor at step B; 'hold, then cooldown' holds the peak until step B - 100 and then runs the cosine over the last 100 steps. Entries: the tail (mean loss at the six steps B - 50, B - 40, ..., B), mean and range over the five orders.

| budget B | shape | tail (mean of 5) | tail range |
|---:|---|---:|---:|
| 300 | cosine over the whole run | 0.0796 | 0.0751 to 0.0831 |
| 300 | hold, then cooldown of 100 | 0.0856 | 0.0822 to 0.0888 |
| 450 | cosine over the whole run | 0.0722 | 0.0686 to 0.0748 |
| 450 | hold, then cooldown of 100 | 0.0768 | 0.0673 to 0.0845 |
| 600 | cosine over the whole run | 0.0677 | 0.0626 to 0.0714 |
| 600 | hold, then cooldown of 100 | 0.0713 | 0.0559 to 0.0887 |
<!-- /worked -->

The full cosine is slightly better at each budget ($0.0796$ against $0.0856$ at 300 steps, $0.0677$ against $0.0713$ at 600 steps), by less than the spread of the held runs over the orders at the two larger budgets. What the held shape buys is that the three held runs differ only in the step at which the cooldown starts: one long run with a hold, and three cooldowns branched from its checkpoints, produce all three finished models, where the cosine needs three separate runs.

The opposite error is a schedule longer than the run. A fine-tuning run of 40 steps with a peak rate of $2 \times 10^{-4}$ and a warmup left at 100 steps never leaves the warmup:

<!-- worked: schedules.warmup_longer_than_run -->
A fine-tuning run of 40 steps with peak rate 2e-4. First row: a linear warmup of 100 steps (rate 2e-4 (t + 1)/100 for t < 100). Second row: warmup of max(1, 40 // 20) = 2 steps, then a cosine over the remaining steps to a tenth of the peak. Columns: the rate at step index 7 (the eighth update), the rate of the last update, the largest rate reached as a fraction of the peak, and the mean rate over the 40 updates as a fraction of the peak.

| schedule | rate, 8th update | rate, last update | largest / peak | mean / peak |
|---|---:|---:|---:|---:|
| warmup 100, run of 40 steps | 1.600e-05 | 8.000e-05 | 0.40 | 0.205 |
| warmup 2 (a twentieth of the run), cosine to a tenth | 1.924e-04 | 2.031e-05 | 1.00 | 0.571 |
<!-- /worked -->

The rate of the eighth update is $1.6 \times 10^{-5}$, the last update of the run is at $0.4$ of the peak and the mean rate over the run is $0.205$ of it. The engine's knowledge #280 records this case: a fine-tune of a 27-billion-parameter model, run for 40 steps with its default warmup of 100, was at a rate of $1.6 \times 10^{-5}$ at step 8 (the first row's number) with the loss still between 1.5 and 1.7, and was stopped; with the warmup set to 4 steps the same run took the loss from $1.7469$ to $0.0746$. The second row is the engine's own default rule for the warmup (`max(1, steps // 20)`, a twentieth of the run, in `crates/qwen3/src/finetune.rs`), which scales with the budget. The lesson is the section's: a schedule's lengths are fractions of the run, never absolute numbers carried over from another run.

## Gradient clipping

*What it is for: to bound the size of one step against a gradient that is, for one batch, far larger than usual.*

Set the rate aside. The update of AdamW multiplies the rate by a ratio built from the gradient, and a gradient that is a hundred times too large for one batch is a problem the rate cannot see, because the rate does not depend on the gradient. Take the clean run of the earlier tables (warmup of 30 steps to $0.3$, cosine to $0.003$, 600 steps) and change one thing: at step index 300 the gradient of the batch is multiplied by 100, as if that batch had carried a loss weight of 100 by a data-pipeline mistake. The first table below measures the ordinary gradient of the run, to say what "a hundred times" is relative to.

### The global gradient norm

A model's gradient has one entry for every parameter, spread over many tensors of different shapes (the weight and bias of every layer). The **global gradient norm** is the Euclidean norm of all of them taken as one vector: square every entry of every tensor, add them all, and take the square root,

$$
\lVert\mathbf{g}_t\rVert = \sqrt{\sum_{k}\ \sum_{i}\ \bigl(g_{t}^{(k)}[i]\bigr)^2},
$$

with $k$ running over the tensors and $i$ over the entries of one. The statistics of this number over a run, for the clean run, are:

<!-- worked: schedules.gradient_norm_statistics -->
The network, data and starting weights are those of Chapter 7 (generator `optimizers.mlp`, seed 0xc234bb64614333c5; starting loss 0.7471). AdamW with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and decoupled decay lambda = 0.01; 600 steps of batch size 20; batches from the generator `schedules.batches` (seed 0xb0cdb4f232361fb9): shuffle the list 0 .. 199, cut it into consecutive batches of 20, reshuffle at the end of each epoch. Learning rate: warmup 30 steps to 0.3, then a cosine to 0.003 at step 600. No clip. The norm is the Euclidean norm of the 33-component minibatch gradient of each step (600 values; the first five are 0.307, 0.577, 0.274, 0.483, 0.313).

| statistic | gradient norm |
|---|---:|
| minimum | 0.0174 |
| median | 0.1789 |
| 90th percentile | 0.3373 |
| 99th percentile | 0.5312 |
| maximum | 0.7037 |

| part of the run | median | maximum |
|---|---:|---:|
| steps 0 to 99 | 0.2422 | 0.6118 |
| steps 100 to 299 | 0.1910 | 0.6485 |
| steps 300 to 599 | 0.1557 | 0.7037 |
<!-- /worked -->

The median is $0.1789$ and the largest of 600 values $0.7037$: ordinary steps have a norm within a factor 4 of the median. The batch with a gradient 100 times larger has a norm near $18$ (the next table gives it), about a hundred times the median and 26 times the largest ordinary value. A rule that acts on the norm can tell the two apart.

### Per-element and global clipping

There are two natural ways to bound a gradient. **Per-element clipping** replaces every entry by $\max(-\tau, \min(\tau, g_i))$. **Global clipping**, the rule that the rest of this chapter means by gradient clipping, multiplies the whole gradient by one number, the clip coefficient

$$
c_t = \min\Bigl(1,\ \frac{\tau_{\mathrm{clip}}}{\lVert\mathbf{g}_t\rVert + \epsilon_{\mathrm{clip}}}\Bigr), \qquad \mathbf{g}_t \leftarrow c_t\,\mathbf{g}_t,
$$

with $\tau_{\mathrm{clip}}$ the threshold and $\epsilon_{\mathrm{clip}} = 10^{-6}$ a constant that keeps the quotient finite when the gradient is exactly zero. When $\lVert\mathbf{g}_t\rVert + \epsilon_{\mathrm{clip}} \le \tau_{\mathrm{clip}}$ the coefficient is $1$ and the gradient is untouched; otherwise the new norm is $\tau_{\mathrm{clip}}\lVert\mathbf{g}_t\rVert/(\lVert\mathbf{g}_t\rVert + \epsilon_{\mathrm{clip}})$, which is the threshold to within $\epsilon_{\mathrm{clip}}$ relative.

The two clips differ in what they keep. Global clipping multiplies by a positive scalar, and a positive scalar changes the length of a vector and nothing else: $\mathbf{g}'$ points exactly where $\mathbf{g}$ does (the cosine of the angle between them is $1$), and its norm is $\min(\lVert\mathbf{g}\rVert, \tau_{\mathrm{clip}})$ up to $\epsilon_{\mathrm{clip}}$. Per-element clipping multiplies each coordinate by its own factor, $\min(1, \tau/|g_i|)$, and different factors turn the vector: only when every coordinate is clipped, or none, is the result a multiple of the original.

<!-- worked: schedules.clip_geometry -->
Two clips with threshold 1 on four gradients in two dimensions. Per-element: each coordinate is replaced by max(-1, min(1, g_i)). Global: g is multiplied by min(1, 1/(||g|| + 10^-6)). 'cos' is the cosine of the angle between the original and the clipped vector.

| g | ||g|| | per-element | its norm | cos | global | its norm | cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| (4, 1) | 4.1231 | (1.0000, 1.0000) | 1.4142 | 0.8575 | (0.9701, 0.2425) | 1.0000 | 1.0000 |
| (3, 3) | 4.2426 | (1.0000, 1.0000) | 1.4142 | 1.0000 | (0.7071, 0.7071) | 1.0000 | 1.0000 |
| (0.6, 0.3) | 0.6708 | (0.6000, 0.3000) | 0.6708 | 1.0000 | (0.6000, 0.3000) | 0.6708 | 1.0000 |
| (-5, 2) | 5.3852 | (-1.0000, 1.0000) | 1.4142 | 0.9191 | (-0.9285, 0.3714) | 1.0000 | 1.0000 |
<!-- /worked -->

![A two-dimensional plot with a dashed circle of radius 1 and a dashed square of half-width 1 around the origin. The gradient (4, 1) is drawn from the origin to the right, at 14 degrees above the horizontal. The global clip is a short arrow along the same line ending on the circle at (0.970, 0.243); the per-element clip is an arrow to the corner (1, 1) of the square, at 45 degrees.](img/clip-geometry.svg)

For $\mathbf{g} = (4, 1)$ per-element clipping gives $(1, 1)$, a vector at 45 degrees where the gradient was at 14, and the cosine with the original is $0.8575$: the step now goes along a different direction, one that has taken weight from the large coordinate and given it to the small one. Global clipping gives $(0.9701, 0.2425)$, the same direction with norm 1. For $(3, 3)$ the per-element clip happens to keep the direction (both coordinates are clipped by the same factor), and it still leaves the norm at $\sqrt 2$, above the threshold. That is the second difference: per-element clipping bounds each entry, not the norm. In $n$ dimensions it allows a norm up to $\tau\sqrt{n}$, which for a model of $10^9$ parameters and $\tau = 1$ is $31\,623$ (computed: $\sqrt{10^9}$); a threshold on each entry says little about the length of the step. Global clipping bounds the length and keeps the direction, which is why it is the clip used by gradient methods for networks.

### Over all tensors at once

The sum under the square root runs over every parameter of the model, not tensor by tensor. Clipping each tensor to the threshold on its own is the per-element clip again, one level up, and it turns the gradient in the same way:

<!-- worked: schedules.clip_tensors -->
Three tensors of one model's gradient, threshold tau = 1. 'per tensor' clips each tensor to norm 1 on its own; 'global' computes the norm over all seven numbers at once (sum of squares 26.04, norm 5.1029, one factor 0.1960).

| tensor | gradient | norm | own factor | per tensor | global |
|---|---:|---:|---:|---:|---:|
| layer 1 (weights) | [3.0, 4.0] | 5.0000 | 0.2000 | [0.6000, 0.8000] | [0.5879, 0.7839] |
| layer 2 (weights) | [0.6, 0.8] | 1.0000 | 1.0000 | [0.6000, 0.8000] | [0.1176, 0.1568] |
| bias | [0.0, 0.0, 0.2] | 0.2000 | 1.0000 | [0.0000, 0.0000, 0.2000] | [0.0000, 0.0000, 0.0392] |

Cosine between the concatenated original and the concatenated clipped gradient: per tensor 0.8287, global 1.0000. Norm after the clip: per tensor 1.4283, global 1.0000.
<!-- /worked -->

The first layer's gradient has norm $5$, the second layer's $1$ and the bias's $0.2$. Clipped tensor by tensor the first layer is brought to norm 1 and the other two are left alone, which changes the ratio of the first layer's gradient to the others' by a factor of 5. Clipped globally, all three are multiplied by $0.1960$ and keep their ratios. The concatenated gradient keeps its direction exactly (cosine $1.0000$ against $0.8287$), and the norm is the threshold ($1.0000$ against $1.4283$).

Written for a list of tensors with arbitrary shapes, and in place:

```text
function global_norm_clip(grads: list<array<real>>, tau: real) -> real:
    require tau > 0.0, "the threshold must be positive"
    total = 0.0
    for k in 0 .. len(grads):                    # one pass over every tensor: the norm needs all of them
        for index in indices(grads[k].shape):
            total = total + grads[k][index] ^ 2
    norm = sqrt(total)
    c = min(1.0, tau / (norm + 0.000001))        # the clip coefficient, one scalar for every tensor
    for k in 0 .. len(grads):
        for index in indices(grads[k].shape):
            grads[k][index] = c * grads[k][index]
    return norm                                  # the unclipped norm, for logging
```

It returns the unclipped norm because the norm is the number a run must watch. A note on the loop structure: the first double loop must finish over every tensor before the second starts, so the clip cannot be applied to the first layer's gradient before the backward pass has produced the last layer's. That is a constraint on the training loop and it has a price, in the section on brain below.

### Clipping and Adam

Where does the clip enter AdamW? Before the moments: the clipped gradient is what is averaged into $\mathbf{m}$ and $\mathbf{v}$,

```text
function train_step(theta: inout array<real>[P], g: inout array<real>[P],
                    m: inout array<real>[P], v: inout array<real>[P], t: int,
                    schedule: function(int) -> real, tau: real,
                    beta1: real, beta2: real, eps: real, lam: real) -> real:
    require t >= 0, "t is the 0-based step index"
    require tau > 0.0, "use tau = infinity for no clip"
    total = 0.0
    for i in 0 .. P:
        total = total + g[i] * g[i]              # the global norm over all P entries
        # (the sum is over every parameter of the model, as in global_norm_clip)
    norm = sqrt(total)
    c = min(1.0, tau / (norm + 0.000001))
    for i in 0 .. P:
        g[i] = c * g[i]                          # clip first: the moments see the clipped gradient
    adam_step(theta, g, m, v, t + 1, schedule(t), beta1, beta2, eps, lam, true)
    return norm                                  # the unclipped norm, to be logged
```

Here `theta`, `g`, `m` and `v` are one flat array of all the parameters, and `adam_step` is Chapter 7's function with `decoupled = true`; the schedule is a function of the 0-based step index and Adam's step number is `t + 1`. A fact from Chapter 7 now tells us something unexpected. Adam is unchanged by a constant factor on the gradient: $\hat m/\sqrt{\hat v}$ is the same if every gradient is multiplied by the same number. So a clip that multiplies this step's gradient by $c_t$ does nothing if $c_t$ is the same at every step, and the engine's test shows it: a gradient of four entries of $2.0$ (norm 4) with a threshold of $1$ is multiplied by $1/4$, and the first AdamW step is the same $\eta$ as without a clip. The clip matters exactly when $c_t$ changes from step to step, that is, when it fires on some steps and not on others, which is the case of an outlier. What it changes then is the moments, and the one-weight table shows how.

<!-- worked: schedules.clip_adam_one_weight -->
One weight, gradients 1, 1, 1, 1, 100, 1, 1, 1, 1, 1 (the fifth is the outlier). Adam with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, bias correction; 'step' is mhat / (sqrt(vhat) + epsilon), the update in units of the learning rate. Right-hand block: the same with the gradient clipped to [-2, 2] (for one weight, the global-norm clip with threshold 2 is the same operation).

| t | g | mhat | sqrt(vhat) | step | clipped g | mhat | sqrt(vhat) | step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 1 | 1.0000 | 1.0000 | 1.0000 | 1 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 1 | 1.0000 | 1.0000 | 1.0000 | 1 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 1 | 1.0000 | 1.0000 | 1.0000 | 1 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 100 | 25.1752 | 44.7750 | 0.5623 | 2 | 1.2442 | 1.2654 | 0.9833 |
| 6 | 1 | 20.0157 | 40.8656 | 0.4898 | 1 | 1.1921 | 1.2251 | 0.9731 |
| 7 | 1 | 16.3708 | 37.8267 | 0.4328 | 1 | 1.1553 | 1.1954 | 0.9664 |
| 8 | 1 | 13.6720 | 35.3765 | 0.3865 | 1 | 1.1280 | 1.1727 | 0.9619 |
| 9 | 1 | 11.6033 | 33.3466 | 0.3480 | 1 | 1.1071 | 1.1547 | 0.9588 |
| 10 | 1 | 9.9754 | 31.6290 | 0.3154 | 1 | 1.0907 | 1.1401 | 0.9566 |
<!-- /worked -->

Four steady gradients of $1$ give the steady step $1.0\,\eta$. The fifth, $100$, goes into the averages unclipped: $\hat m$ jumps to $25.18$ and $\sqrt{\hat v}$ to $44.78$, so the step at the outlier is $0.5623\,\eta$ (the normalisation limits it) and, more importantly, the next steps, whose gradients are ordinary again, are $0.4898\,\eta$, $0.4328\,\eta$, $0.3865\,\eta$ and down to $0.3154\,\eta$ at the tenth step: the outlier has inflated $\sqrt{\hat v}$ and the ordinary gradients are now small relative to it. The second moment forgets at the rate $\beta_2 = 0.999$ per step, with a memory of about $1/(1 - \beta_2) = 1000$ steps, so these steps recover slowly. With the gradient clipped to $2$ the fifth step is $0.9833\,\eta$ and the following ones are $0.97\,\eta$ and then $0.96\,\eta$, which is the steady behaviour with a small kick. In one dimension the direction of a step cannot be wrong; in 33 dimensions the outlier is also a wrong direction, and the first moment carries it for about $1/(1 - \beta_1) = 10$ steps.

### An outlier batch

Now the network with the outlier. Three runs differ only in the clip: none, a threshold of $1$ (above every ordinary norm of the table, so it never fires on an ordinary step) and a threshold of $0.2$ (near the median, so it fires often).

<!-- worked: schedules.outlier_batch -->
The network, data and starting weights are those of Chapter 7 (generator `optimizers.mlp`, seed 0xc234bb64614333c5; starting loss 0.7471). AdamW with beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 and decoupled decay lambda = 0.01; 600 steps of batch size 20; batches from the generator `schedules.batches` (seed 0xb0cdb4f232361fb9): shuffle the list 0 .. 199, cut it into consecutive batches of 20, reshuffle at the end of each epoch. Learning rate: warmup 30 steps to 0.3, cosine to 0.003 at step 600. At step index 300 (the 301st update) the minibatch gradient is multiplied by 100, as if one batch had carried a loss weight of 100; nothing else differs between the rows. The same run without an outlier has a gradient norm of 0.185 at that step and a loss of 0.0700 at step 600. Entries: the mean loss over all 200 examples after the stated number of updates; the last column is the largest loss over steps 301 to 400.

| clip | step 299 | step 300 | step 301 | step 305 | step 320 | step 350 | step 400 | step 600 | worst, 301-400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no clip | 0.0805 | 0.0789 | 0.1984 | 1.5704 | 0.5495 | 0.1323 | 0.1114 | 0.0862 | 1.943 |
| clip tau = 1 | 0.0805 | 0.0789 | 0.0905 | 0.1633 | 0.0902 | 0.0889 | 0.0835 | 0.0701 | 0.177 |
| clip tau = 0.2 | 0.0904 | 0.0860 | 0.0815 | 0.0779 | 0.1007 | 0.0905 | 0.0716 | 0.0642 | 0.107 |

The size of Adam's step in units of the learning rate: the root mean square over the 33 parameters of mhat / (sqrt(vhat) + epsilon), averaged over the stated steps (step indices).

| clip | norm at the outlier | steps 200-299 | at the outlier | steps 301-320 | steps 351-450 | steps 501-600 |
|---|---:|---:|---:|---:|---:|---:|
| no clip | 18.528 | 0.1223 | 1.2310 | 0.7724 | 0.1233 | 0.0750 |
| clip tau = 1 | 18.528 | 0.1223 | 0.3537 | 0.2052 | 0.1401 | 0.0817 |
| clip tau = 0.2 | 27.099 | 0.1502 | 0.1495 | 0.1298 | 0.1448 | 0.1088 |
<!-- /worked -->

![Mean loss over all 200 examples against step, from step 280 to 420. With no clip the loss rises from 0.08 at step 301 to 1.94 near step 307 and decays back to about 0.11 by step 400; with a clip of 1 it rises only to 0.18; with a clip of 0.2 it does not rise.](img/clip-outlier.svg)

Without a clip, the loss goes from $0.0789$ at step 300 to $1.943$ at its worst, about 25 times the loss before the batch, is still above the clipped run's at step 400 ($0.1114$ against $0.0835$) and ends at $0.0862$ where the same run without an outlier ends at $0.0700$. With a clip of $1$ the worst loss over steps 301 to 400 is $0.177$ and the run ends at $0.0701$: the clip fires exactly once, at the outlier, so the trajectory is the clean one plus one bounded step. The second table shows why in Adam's own terms: the step at the outlier is $1.2310\,\eta$ unclipped and $0.3537\,\eta$ clipped, the following twenty steps average $0.7724\,\eta$ against $0.2052\,\eta$ (the usual step of this phase of the run is $0.12\,\eta$), and over steps 351 to 450 the unclipped run's steps are about 12% smaller than the clipped one's ($0.1233$ against $0.1401$), which is what the inflated $\hat v$ of the one-weight table would produce (the weights of the two runs differ, so this is consistent with it, not a proof).

### Choosing the threshold

The threshold is not a universal constant: the norm of a gradient depends on how the loss is normalised (a mean over the batch or a sum), on the number of parameters and on the model. A threshold is read from the run's own norm statistics. Two rules follow from the tables. First, a threshold above the ordinary norms never touches an ordinary step and catches the outliers: here $\tau = 1$ is above the maximum of the ordinary norms and about 5.6 times the median. Second, a threshold that fires on many ordinary steps is not obviously harmful for Adam, because a clip that fires on every step multiplies by nearly the same factor each time and Adam cancels a constant factor; the table below includes such thresholds.

<!-- worked: schedules.clip_threshold_table -->
The run of the previous tables (warmup 30 steps to 0.3, cosine to 0.003, 600 steps) on the five batch orders of the schedule comparison, with the global-norm clip at each threshold tau, once as it is and once with the outlier of the next table (the gradient of step index 300 multiplied by 100). Columns: the share of the 3000 steps of the clean runs in which the clip changed the gradient; the tail (mean loss at steps 550, 560, ..., 600) of the clean runs; the largest loss over steps 301 to 400 of the runs with the outlier; their tail. All entries are means of the five orders.

| tau | steps clipped (clean) | tail (clean) | worst, 301-400 (outlier) | tail (outlier) |
|---|---:|---:|---:|---:|
| no clip | 0.0% | 0.0677 | 1.632 | 0.0919 |
| 0.1 | 82.0% | 0.0592 | 0.135 | 0.0592 |
| 0.2 | 42.0% | 0.0608 | 0.107 | 0.0607 |
| 0.4 | 6.2% | 0.0662 | 0.119 | 0.0662 |
| 1 | 0.0% | 0.0677 | 0.172 | 0.0677 |
| 5 | 0.0% | 0.0677 | 0.722 | 0.0705 |
<!-- /worked -->

A threshold of $5$, above every ordinary norm, protects only partly: the batch with the norm of $18.5$ is clipped to $5$, still $28$ times the median, and the worst loss is $0.722$, against $0.172$ at $\tau = 1$. The thresholds $0.1$ and $0.2$ clip $82\%$ and $42\%$ of the ordinary steps and the tail of the clean run is not worse ($0.0592$ and $0.0608$ against $0.0677$); on this small problem a run with a clip that fires very often loses nothing, and I would not generalise from one problem and five batch orders whose tails spread by about $0.01$. The rule for a new run:

1. Log the global norm of every step ($\lVert\mathbf{g}_t\rVert$ is returned by the clip above for that purpose).
2. Read its median, 90th and 99th percentile and maximum from a short clean run, as in the statistics table.
3. Set the threshold a few times above the 99th percentile, so that the clip fires on well under one step in a hundred. The classic recipe, from the paper that introduced norm clipping for recurrent networks, takes a threshold from the mean norm over a large number of updates.
4. Keep logging it and look at the share of clipped steps: a clip that fires on most steps is acting as a rescaling of the gradient, not as an outlier guard, and a norm that grows over the run says the model is moving toward an instability.

A conventional value for large language-model training is a threshold of $1.0$ on the gradient of a mean loss, as in the engine's fine-tuning defaults below, but it is a convention that suits that normalisation. A related convention: the norm must be measured on the same gradient the optimizer will use. If $K$ micro-batch gradients are accumulated and averaged by $1/K$ before the step, the norm that is clipped is the norm of the averaged gradient, not of the sum; in knowledge #181 the device clip measured the norm of the raw sum, which made it $K$ times too lax.

## Making it work

*What it is for: to count what a clip costs per step, and to compute the norm accurately enough to be reproducible.*

**Work.** A clip is one reduction over every gradient entry and one multiplication per entry. For a model of $P$ parameters in 32-bit floats:

<!-- worked: schedules.clip_work -->
Work for a model of P = 1,000,000,000 parameters in 32-bit floats, per optimizer step. The sum of squares is a reduction (the final sum of a few hundred partials is negligible).

| operation | FLOPs | bytes | what it does |
|---|---:|---:|---|
| sum of squares of all gradients | 2,000,000,000 | 4,000,000,000 | read g; 1 multiply and 1 add per element |
| scale by the coefficient, own pass | 1,000,000,000 | 8,000,000,000 | read g, write g |
| scale folded into the optimizer's read of g | 1,000,000,000 | 0 | one more multiply in a pass that already reads g |
| AdamW update itself (Chapter 7) | 16,000,000,000 | 28,000,000,000 | reads w, g, m, v; writes w, m, v |

Relative to the AdamW update (16P FLOPs, 28P bytes): the norm and the scaling in its own pass add (2 + 1)P FLOPs and (4 + 8)P bytes, which is 0.1875 of the update's FLOPs and 0.4286 of its bytes. With the scale folded into the optimizer's read of g, the norm pass is the only extra traffic, 0.1429 of the update's bytes, and the extra FLOPs are the same 0.1875 of the update's.
<!-- /worked -->

The norm reads $4P$ bytes and does $2P$ FLOPs (a multiplication and an addition per entry); the scaling is $P$ more FLOPs, and either its own read-and-write pass ($8P$ bytes) or, if it is folded into the optimizer's single pass over the gradient, no extra bytes. The AdamW update itself is $16P$ FLOPs and $28P$ bytes (Chapter 7), so the norm and a folded scaling add $0.1875$ of the update's FLOPs and $0.1429$ of its bytes, and a separate scaling pass would raise the bytes to $0.4286$ of them (the table's own sums). None of this depends on the batch size: it is a per-step cost, shared by every example of the batch, whereas the work of the forward and backward passes grows with the batch.

**Accuracy and determinism.** The norm is a sum of $P$ squares, and $P$ is large. A single 32-bit accumulator that adds squares in order loses the low bits of every small term once the sum is large:

<!-- worked: schedules.norm_accumulation -->
The sum of squares of N = 262,144 gradient entries: entry i is the 32-bit rounding of 0.01 * z_i, with z_i from normal() of the generator `schedules.norm_sum` (seed 0x20f29e3aabc0523a), i = 0 .. N - 1 in order. Every 32-bit operation (the square, each addition) is rounded to the nearest 32-bit float.

| method | sum of squares | relative error |
|---|---:|---:|
| exact (compensated sum of the 32-bit squares' double-precision values) | 2.618922169e+01 | 0 |
| one 32-bit accumulator, in order | 2.618758965e+01 | 6.23e-05 |
| chunks of 1024 in 32 bits, chunk sums added in 64 bits | 2.618922170e+01 | 5.34e-10 |
| one 64-bit accumulator, in order | 2.618922169e+01 | 1.12e-13 |
<!-- /worked -->

A relative error of $6.2 \times 10^{-5}$ in the norm is harmless to the clip coefficient at this size (the threshold is not a precise number), and it grows with the number of terms. The usual repair is to sum in chunks and add the chunk sums in a wider format: chunks of $1024$ in 32 bits and the chunk sums in 64 bits give $5.3 \times 10^{-10}$. A second reason to fix the order of the sum is reproducibility: a norm that differs in its last bit, because a parallel reduction adds in an order that depends on how many threads were free, multiplies every weight update by a slightly different coefficient, and two runs of the same seed then differ.

## Where this appears in brain

**The schedule.** `crates/model/src/train.rs` holds `LrSchedule`, whose `at(step)` is this chapter's `schedule_rate` with the same four pieces (warmup, hold, cosine, floor); Volume I, Chapter 2 told that story (commit `cb2136638`) and this chapter does not retell it. What this chapter adds is how a run chooses its lengths. `LoraHyper` in `crates/qwen3/src/finetune.rs` is the recipe for a LoRA fine-tune: a decay to a tenth of the peak, a warmup of `max(1, steps / 20)` steps when none is given (a twentieth of the run), a weight decay of $0.1$ and a global clip of $1.0$, with `decay_iters` set to the run's step count. Knowledge #305 records the same defaults.

**The clip.** The doc of `crates/optim/src/lib.rs` calls the optimizer "AdamW with optional global grad-norm clipping, matching `torch.optim.AdamW` + `torch.nn.utils.clip_grad_norm_`". The one definition of the number every gradient is multiplied by is `optim::grad_multiplier(raw_sum_sq, clip, extra_scale)`: the accumulation scale $1/K$ times the clip coefficient $c_t$ of the scaled gradient, with $\epsilon_{\mathrm{clip}} = 10^{-6}$ exactly as in this chapter. The device runs it as three stages. `crates/kernels/wgsl/gradnorm_part.wgsl` is the reduction: each parameter tensor is split over $\lceil \mathrm{numel}/8192\rceil$ workgroups, at most 512 (`paramstore::gradnorm_parts` in `crates/paramstore/src/lib.rs`), each writing one partial sum of squares. `crates/kernels/wgsl/clip_coef_wg.wgsl` folds the partials of every tensor into the coefficient. And `crates/kernels/wgsl/adamw.wgsl` multiplies the gradient by the coefficient as it reads it (`g = grad[i] * scale * coef[0]`), which is the "folded into the optimizer's pass" row of the work table: the separate scaling stage was removed, and a clipped step went from $3P + 1$ dispatches to $2P + 1$ for $P$ parameter tensors (the module's own account of the change). `crates/optim/src/gradnorm.rs` reads the partials back and adds them in 64 bits, so the gradients never leave the device, and the host path `sum_sq_f64` in `crates/backend-cpu/src/par.rs` adds fixed 16K-element chunks in a fixed order (commit `4b56d9428`, 20 October 2026; the doc comment of `sum_sq_f64` gives this chapter's reproducibility reason: a grad-norm "that differs in its last bit scales every weight update differently").

The tests state the chapter's facts. `adamw_with_clip_matches_hand_computation` in `crates/optim/src/lib.rs` sets four gradients of $2.0$ (norm 4, clipped by $1/4$ to $0.5$) and asserts that the first AdamW step is exactly $\eta$ (weights $1.0$ to $0.9$ at $\eta = 0.1$): the scale invariance of the section on clipping and Adam. `cooperative_gradnorm_matches_the_serial_path` runs the reduction at sizes that need several workgroups per tensor and several tensors per step against the serial kernel, `scale_and_clip_follow_torch_on_device_and_host` in `crates/optim/tests/it/offload_parity.rs` compares both paths with an independent reference, and the `adamw_q8` and `adamw_bf16` parity tests in the same directory run their device kernels against the host reference with a clip.

**Story 1: the reduction that ran on one thread** (commit `dba351816`, 30 July 2026). The first norm kernel, `crates/kernels/wgsl/gradnorm_sq.wgsl` (still in the tree, and still opening with `if (gidx != 0u) { return; }`), computed each tensor's sum of squares on a single thread of the device, a loop of `numel` dependent loads. The work count of this chapter says what the stage should cost: $4P$ bytes and $2P$ FLOPs, $0.143$ and $0.125$ of the optimizer's own update. The commit measured what it cost in place, on a GPT training run:

<!-- perf-number: historical measurement, commit dba351816 -->
In July 2026, on one P40 card, over five profiled steps of a six-layer model of width 768, the serial norm was $87.2\%$ of all GPU time of the step, at $0.023\%$ of the card's memory bandwidth; the cooperative reduction named above brought the stage to $0.3\%$ of the step, $2122$ times less time for the stage.

That is one setup on one day, and its lesson is not the number: a cost that is negligible by the work count is not negligible if the kernel does not use the machine's width, and a clip is a reduction over everything. The same message reports a numerical observation, which the accumulation table above shows in miniature: against a 64-bit oracle the serial 32-bit sum was off by $2.34 \times 10^{-3}$ relative at 4.19 million elements, where the tree was off by $2.4 \times 10^{-7}$.

**Story 2: a global norm is a barrier** (commit `614e891d5`, 21 October 2026). Layer-streamed full fine-tuning keeps only a window of layers on the device and updates a layer as soon as its gradient is complete, so that the gradient need not be stored for the whole model. A global clip is in conflict with that: the coefficient $c_t$ needs the norm of every gradient, and the first layers to be updated are finished before the last ones have been computed. `ClipPass` in `crates/model/src/resident_ft.rs` offers two resolutions. `TwoPass` "measures the global norm in a first backward pass, updates in a second: exact, one extra backward". `Running` updates in the only backward pass, "clipping each layer by the norm accumulated so far: never harder than the global clip, equal to it when nothing clips". The second claim is checkable in one line: the norm accumulated so far is at most the global norm, so $\min(1, \tau/\text{running}) \ge \min(1, \tau/\text{global})$ for every layer. The trade is the one the work counts predict: the exact clip costs a whole extra backward pass, and the approximation costs nothing but clips each layer by a coefficient that is too lax. The commit message says the norm is reduced on the device in both cases, by `gradnorm_part` and a bf16 twin.

**Story 3: a schedule longer than the run** (knowledge #280). Told above: a warmup of 100 steps inside a 40-step run, a rate of $1.6 \times 10^{-5}$ at the eighth step and a loss that did not move, until the warmup was set to 4 steps. The engine's answer is not a better default number but a default that is a fraction of the run, the `steps / 20` rule above. A related record, knowledge #344, added the global gradient norm of every clipped step to the step log (`step N loss ... grad_norm ... lr ...` in `crates/model/src/train.rs`), with the entry's reason that a run whose clip is eating the signal should be visible in its own log: the logging rule of the threshold section.

## Check your understanding

Answers are at the end of the chapter.

1. **Predict.** A constant rate of $0.03$ and one of $0.3$ are both run for 600 steps on the network of this chapter. From the first table, which has the lower loss at step 100, which has the lower tail, and what does the pair say about choosing one constant?
2. **Explain.** Adam's first update has size $\eta$ on every weight whatever the gradient. Using the table of early steps, say how large the typical step is at $t = 1$ and at $t = 10$ for a noise-dominated weight, relative to its steady size, and why a linear warmup is the cure and a smaller constant rate is not the same cure.
3. **Find the error.** "A cosine schedule over $T$ steps lets us extend the run: just continue training and the rate stays at the floor." What is wrong, using the budget table, and what shape avoids the problem?
4. **Predict.** A gradient has entries $(6, 8, 0)$ and the threshold is $5$. Compute the result of the per-element clip and of the global clip, the norm of each and which of them points where the gradient pointed.
5. **Find the error.** "To clip a model's gradient, clip each layer's gradient to the threshold: that way no layer is more than the threshold and the largest layers are the ones affected." Use the tensor table to explain what is wrong with the claim.
6. **Explain.** Clipping the gradient to half its size at every step of an Adam run changes nothing at all. Why, and what does that say about when a clip changes an Adam run?
7. **Predict.** A model has $2 \times 10^9$ parameters in 32-bit floats. How many bytes does the norm read and how many FLOPs does it do, and how many more bytes than the AdamW update does a clip add when the scaling is folded into the optimizer's pass?
8. **Explain.** A run logs a global gradient norm with a median of $0.4$ and a 99th percentile of $0.9$. A teammate proposes a threshold of $0.1$. What will the clip do to the run, and what threshold would you choose first?

## Exercises

1. **[Calculate]** For the schedule with $\eta_{\max} = 0.2$, $\eta_{\min} = 0.02$, warmup $W = 8$, no hold and total $T = 108$, compute $\eta_t$ at $t = 0, 7, 8, 58, 108$ by hand, and the mean rate over the decay (it is the midpoint of the two rates).
2. **[Calculate]** For $\beta_1 = 0.9$ compute $\mathrm{RMS}(\hat m_t)/\sigma$ at $t = 1, 2, 3$ from the formula of this chapter, and the rate that holds the typical step at its steady size. Check against the first rows of the early-step table.
3. **[Calculate]** A gradient is $\mathbf{g} = (12, -5, 0, 0)$ and $\tau_{\mathrm{clip}} = 6.5$. Compute the global clip with $\epsilon_{\mathrm{clip}} = 10^{-6}$ to six decimals and compare the result with the per-element clip at $6.5$ (norm and cosine with $\mathbf{g}$).
4. **[Calculate]** For three tensors with gradients $[6, 8]$, $[0, 0, 0]$ and $[3, 4]$ and a threshold of $5$, compute the global clip and the per-tensor clip, and the cosine between the original and each result.
5. **[Calculate]** Do the one-weight Adam table by hand for the first six steps with gradients $1, 1, 1, 1, 100, 1$ and no clip ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$): $\hat m$, $\hat v$ and the step in units of $\eta$ at $t = 5$ and $t = 6$.
6. **[Calculate]** For $P = 7 \times 10^9$ parameters in 32-bit floats compute the FLOPs and bytes of the norm, of a separate scaling pass and of AdamW, and the percentages of the clip over the update for the folded and the separate scaling.
7. **[Implement]** Implement `schedule_rate` of this chapter and reproduce `schedules.shape_table` (the six schedules, the marks $t = 0, 29, 30, 150, 300, 450, 599$ and the mean rate) to five decimals, and `schedules.cosine_values`.
8. **[Implement]** Implement the global-norm clip for a list of tensors, the per-element clip and the per-tensor clip, and reproduce `schedules.clip_geometry` and `schedules.clip_tensors` to four decimals.
9. **[Implement]** Implement the one-weight Adam of the clipping section, and reproduce `schedules.clip_adam_one_weight`. Then change the outlier to $10$ and to $1000$ and report the step at the outlier and the tenth step, with and without the clip at $2$.
10. **[Implement]** Using the book's generator (Volume I, Chapter 17) with the seed and the draw order printed in the block, reproduce `schedules.adam_early_steps` to its stated tolerance: the exact column to four decimals and the measured columns to within a few percent.
11. **[Investigate]** Train the network of this chapter with AdamW at a peak of $3$ and warmups of $W = 40, 60, 80$ steps over several batch orders. Which is the shortest warmup for which the worst loss of the first 50 steps stays within $0.01$ of the starting loss? Is it the same for every order? Relate your answer to the table of early steps.
12. **[Investigate]** Repeat the outlier experiment with factors $10$, $30$, $100$ and $1000$ and thresholds $0.5$, $1$, $5$. For each pair report the worst loss over the 100 steps after the outlier and say for which factors a threshold above the largest ordinary norm protects the run and for which it does not.

## Summary

One learning rate cannot be right for a whole run. Late in a run the rate sets the floor of the noise, which is proportional to $\eta$ (Chapter 7's stationary variance, halved by halving the rate at the price of twice the steps), and a narrower valley lowers the largest stable rate. Early in an Adam run the steps are too large: the first is $4.4$ times its steady size on a noise-dominated weight, and the second moment is an average of too few squares to be trusted. **Learning-rate warmup**, $\eta_t = \eta_{\max}(t+1)/W$, cures the early part: at a peak ten times too high, no warmup puts the loss at $5.716$ after one step and at a worst of $6.730$, while a warmup of 60 steps stays at the starting loss. Decay cures the late part: exponential, step and inverse-square-root decays, and the **cosine schedule**, half a period of Volume I's cosine, $\eta_{\min} + (\eta_{\max} - \eta_{\min})\tfrac12(1 + \cos\pi u)$, which leaves the peak and arrives at the floor with zero slope. The floor is positive so that no steps are wasted at a vanishing rate; warmup, hold and decay is the common shape; and its held peak keeps every rate before the cooldown independent of the total number of steps, where a cosine over the run depends on it in every step. The value of the combination is robustness more than a better optimum: a constant is good only near its best rate, while warmup and cosine keep the tail between $0.070$ and $0.089$ over a factor 30 of peak rates.

**Gradient clipping** bounds the step against rare large gradients. The **global gradient norm** is the norm of every parameter's gradient as one vector, computed over all tensors at once; multiplying the whole gradient by $c_t = \min(1, \tau_{\mathrm{clip}}/(\lVert\mathbf{g}_t\rVert + \epsilon_{\mathrm{clip}}))$ keeps its direction and bounds its length, where a per-element or per-tensor clip turns it and, per element, bounds only the entries. With Adam a clip that scales every step by the same factor changes nothing, so a clip matters when it fires on some steps only: an outlier batch of norm 18 in a run whose steps have a median of $0.18$ raises the loss to $1.943$ without a clip, to $0.177$ with a threshold of $1$ (the maximum of the ordinary norms is $0.7037$), and leaves Adam's second moment inflated for hundreds of steps. The threshold is read from the run's norm statistics, a few times above the 99th percentile, with the norm logged every step. The work is one reduction over all gradients, $2P$ FLOPs and $4P$ bytes, plus a scaling that is free if folded into the optimizer's read, and the norm needs a wide accumulator and a fixed order to be accurate and reproducible. The limits: the schedules were compared on one small problem, a clip's threshold depends on the loss normalisation, and the global norm forces every gradient to exist before any update, which costs an extra backward pass in a layer-streamed trainer. The next chapter, Initialization, turns to the other end of the same problem: where training starts, and why the starting weights decide whether the gradients of a deep network are the right size at all.

## Key terms

- **learning-rate warmup**: a learning rate that rises from a small value to its peak over the first $W$ steps, $\eta_t = \eta_{\max}(t+1)/W$, so that the first updates of an adaptive optimizer, whose averages are not yet calibrated, are small. <!-- concept: learning-rate-warmup -->
- **cosine schedule**: a decay of the learning rate from its peak to a floor along half a period of the cosine, $\eta_{\min} + (\eta_{\max} - \eta_{\min})\tfrac12(1 + \cos\pi u)$, which starts and ends with zero slope. <!-- concept: cosine-schedule -->
- **gradient clipping**: multiplying the gradient by $\min(1, \tau_{\mathrm{clip}}/(\lVert\mathbf{g}\rVert + \epsilon_{\mathrm{clip}}))$ so that its length never exceeds a threshold and its direction is kept, as a guard against rare very large gradients. <!-- concept: gradient-clipping -->
- **global gradient norm**: the Euclidean norm of the gradient of every parameter of the model, all tensors taken together as one vector, the number a clip compares with its threshold. <!-- concept: global-gradient-norm -->

## Answers

**Before you begin.**

1. Both are $1\,\eta$ (to within $\epsilon_{\mathrm{den}}$): at $t = 1$ the corrected moments are $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the update is $\eta\,g_1/(|g_1| + 10^{-8}) \approx \eta$ for $0.02$ and for $200$ alike (for the tiny gradient the $\epsilon_{\mathrm{den}}$ term is $5 \times 10^{-7}$ of it).
2. $\eta < 2/\lambda_{\max} = 2/40 = 0.05$.
3. $\lVert\mathbf{v}\rVert = \sqrt{9 + 16} = 5$ and $\lVert 0.2\,\mathbf{v}\rVert = 1$. A positive factor changes the length by that factor and leaves the direction unchanged.
4. $c(0) = 1$, $c(\tfrac12) = \tfrac12$ and $c(1) = 0$.
5. $0.1\cdot 5/10 = 0.05$; the peak is first reached at $t = 9$ ($0.1\cdot 10/10$).
6. $\hat v_1 = g_1^2$ (bias correction divides $0.001\,g_1^2$ by $0.001$), so $\sqrt{\hat v_1} = |g_1|$ and the first update is $\hat m_1/\sqrt{\hat v_1} = \mathrm{sign}(g_1)$, of size $\eta$.

**Check your understanding.**

1. At step 100 the rate $0.3$ has the lower loss ($0.0900$ against $0.3220$); at the end $0.03$ has the lower tail than $0.3$ ($0.0933$ against $0.1665$). The high rate is fast and then noisy, the low rate is calm and then slow: no constant is best at both, and the best constant here, $0.1$, is a compromise between them.
2. For a noise-dominated weight the typical step at $t = 1$ is $1.0$ against the steady $0.2294$ (a factor 4.4) and at $t = 10$ it is $0.3301$ (1.4 times). A smaller constant rate lowers the first step but also every later one, so the steady steps are smaller than they need be for the whole run. A warmup is small only while the averages are poor and reaches the full rate when they are good.
3. The cosine over $T$ steps has reached the floor at step $T$ and extending the run trains the added steps at the floor, which has already annealed the run: the table has $0.1$ at step 399 for a run planned for 400 and $0.5890$ for one planned for 800. A hold-then-cooldown shape, whose rates before the cooldown do not depend on the total, avoids it: continue the hold and cool down later.
4. $\mathbf{g} = (6, 8, 0)$ has norm 10. Per-element at $5$: $(5, 5, 0)$, norm $7.071$, and it points at 45 degrees (the cosine with $\mathbf{g}$ is $0.9899$). Global: $c = 5/(10 + 10^{-6})$, $\mathbf{g}' \approx (3, 4, 0)$, norm $5$, same direction.
5. Clipping each tensor on its own applies a different factor to each (the first layer $0.2000$, the others $1.0000$ in the table), so the ratios of the layers' gradients change and the concatenated gradient is turned (cosine $0.8287$), and the norm is not bounded by the threshold ($1.4283$ against $1$). Only one factor over all tensors keeps the direction.
6. Adam is unchanged by a constant factor on the gradient ($\hat m$ and $\sqrt{\hat v}$ both scale by it, apart from the effect of $\epsilon_{\mathrm{den}}$), so scaling every step the same changes nothing. A clip changes an Adam run only through the steps where the coefficient differs from the others: an outlier clipped while the rest are not changes the averages' relative weights.
7. The norm reads $4P = 8 \times 10^9$ bytes and does $2P = 4 \times 10^9$ FLOPs. The AdamW update moves $28P = 56 \times 10^9$ bytes, so the folded clip adds $4P/28P = 14.3\%$ of the update's bytes, $8 \times 10^9$ bytes.
8. A threshold of $0.1$ is a quarter of the median ($0.4$): the clip would fire on nearly every step and act as a rescaling to a norm of $0.1$, not as a guard (and every step of a mean-loss gradient would be shortened by a factor of 4 or more, which for SGD is a rate cut and for Adam a nearly constant factor). A first choice is a few times the 99th percentile, say $2$, so that it fires on well under one step in a hundred; then read the share of clipped steps from the log.

## References

- Razvan Pascanu, Tomas Mikolov and Yoshua Bengio, *On the Difficulty of Training Recurrent Neural Networks*, 2013 (gradient norm clipping, and a threshold chosen from the mean gradient norm).
- Ilya Loshchilov and Frank Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, 2017 (the cosine decay of the learning rate).
- Ilya Loshchilov and Frank Hutter, *Decoupled Weight Decay Regularization*, 2019 (AdamW, used with these schedules).
