<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Proving Gradients Correct

*How do we know a backward pass is right without trusting the person who wrote it?*

## The problem

You have written the backward pass of a network, by hand, from the rules of the last two chapters. It runs, it returns arrays of the right shapes, and the loss goes down when you train with it. Is it right?

Train a small network with five different backward passes and look. The task is a regression a small network can fit: 24 examples with 5 inputs, 7 hidden units, 2 outputs, targets produced by a random three-unit teacher network. One backward pass is correct. The other four each contain one realistic mistake: the factor 2 of the squared error dropped from $d\mathbf{Y}$, the mean taken over the batch only instead of over the batch and the outputs, the batch sum of the first-layer weight gradient missing the last example, and the factor $1 - A^2$ of the tanh forgotten. Plain gradient descent, 2000 steps:

<!-- worked: gradient_checking.buggy_training -->
Setup: B = 24 examples, D = 5 inputs, H = 7 hidden units, O = 2 outputs, tanh, mean squared error. Seed 12102139546493177346; draws in order: X (24 x 5, row-major), V (5 x 3, row-major), Wt (3 x 2, row-major), W1 (7 x 5, row-major, each draw multiplied by 0.3), W2 (2 x 7, row-major, each draw multiplied by 0.3), all normal(); the biases start at 0. The targets are T = tanh(X V) Wt (tanh elementwise). Plain full-batch gradient descent in double precision, 2000 steps; each entry is the loss before the step with that index.

| backward pass used | step 0 | step 100 | step 500 | step 1000 | step 2000 |
|---|---:|---:|---:|---:|---:|
| correct backward, learning rate 0.10 | 1.8216 | 0.0523 | 0.0086 | 0.0020 | 0.0009 |
| correct backward, learning rate 0.05 | 1.8216 | 0.0879 | 0.0266 | 0.0079 | 0.0020 |
| dropped factor 2 in dY, learning rate 0.10 | 1.8216 | 0.0879 | 0.0266 | 0.0079 | 0.0020 |
| mean over B instead of B O, learning rate 0.10 | 1.8216 | 0.0323 | 0.0020 | 0.0008 | 0.0004 |
| batch sum of dW1 misses the last example, learning rate 0.10 | 1.8216 | 0.0521 | 0.0094 | 0.0026 | 0.0013 |
| tanh derivative omitted, learning rate 0.10 | 1.8216 | 0.1020 | 0.1086 | 0.1816 | 0.3434 |

At the starting weights, the gradient each buggy backward pass returns against the correct one (the 7 x 5 + 7 + 2 x 7 + 2 = 58 numbers):

| bug | cosine with the correct gradient | relative L2 error |
|---|---:|---:|
| dropped factor 2 in dY | 1.000000 | 0.500 |
| mean over B instead of B O | 1.000000 | 1.000 |
| batch sum of dW1 misses the last example | 0.999723 | 0.024 |
| tanh derivative omitted | 0.970227 | 0.265 |
<!-- /worked -->

Read the table the way you would read a training run. Nothing crashed, and three of the four buggy runs end within a factor of about two of the correct run's final loss of 0.0009. The run that dropped the factor 2 is *identical*, to every printed digit, to the correct code with half the learning rate: the bug is indistinguishable from a hyperparameter choice. The run with the wrong mean reaches a lower loss *sooner* than the correct one, because an extra factor of $O = 2$ in every gradient is a doubled learning rate, and nobody investigates a run that does better. The run that loses one example out of 24 in one layer's gradient ends at a loss of 0.0013 against 0.0009, which looks like noise between seeds; its gradient is wrong by only 2.4% in relative L2 norm at the start. Only the run without the tanh derivative misbehaves visibly (it falls to 0.10 and then climbs), and a person reading that curve will blame the learning rate, not the code.

The second table is the part that should worry you. The first two bugs have cosine similarity with the correct gradient of exactly $1.000000$: they point the right way and are the wrong length. Cosine similarity cannot see them.

So a loss curve is not an instrument for the correctness of a backward pass, and neither is a similarity between two gradient vectors. What we need is a test with three properties. It must not trust the code under test, so it cannot be built from the same derivation. It must be built from the forward pass alone, which is the one piece of code we can check by reading (it is the definition of what we want). And it must say *which entries* are wrong, because a failing test that cannot localise a bug costs days.

Chapter 4 already contains the idea: go back to the definition of the derivative. Nudge one parameter up and down by a small step $h$, run the forward pass twice, and divide the difference of the two losses by $2h$. That previewed check was used to confirm every rule of the last two chapters. This chapter makes it a precise instrument. How wrong is the estimate, and how do we choose $h$? What numbers do we compare, and with what tolerance? What does a check cost, and how do we afford it for a million parameters? Which inputs should we test on? Where does the instrument go blind? The answers decide whether a green check means anything.

You may use from earlier chapters: the backward pass and its cost (Chapters 4 and 5), the derivative and the central difference (Volume I, Chapter 7), the multivariable chain rule, the directional derivative and the Taylor expansion (Volume I, Chapter 8), and rounding error, unit round-off and cancellation (Volume I, Chapter 15).

## Before you begin

Five questions on ideas from earlier chapters, to answer from memory. The answers are at the end of the chapter. If one is hard, go back to the chapter named before reading on.

1. ("Derivatives: The Mathematics of Change", Volume I, Chapter 7.) Write the central difference estimate of $f'(x)$ with step $h$. How does its error shrink as $h$ shrinks?
2. ("Gradients, Jacobians and Many Variables", Volume I, Chapter 8, Taylor expansion.) Write the Taylor expansion of $f(x + h)$ about $x$ up to the term in $h^3$.
3. ("Floating Point and Numerical Error", Volume I, Chapter 15.) What is the unit round-off $u$ in 32-bit and in 64-bit arithmetic, and what does it bound?
4. ("Floating Point and Numerical Error", cancellation.) Two nearby numbers are each stored with relative error at most $u$. Why can their difference have a much larger relative error?
5. ("Backpropagation Derived from Scratch", Chapter 4.) A network has $P$ parameters. How many forward passes does a central difference gradient need, and what does the backward pass cost in comparison?

## Finite differences from Taylor

The check rests on one fact: the derivative can be estimated from function values. Any estimate of a derivative built from values of the function at nearby points is a **finite difference**. Two of them were defined in Volume I, Chapter 7, and are the two we need. For a function $f$ of one variable and a step $h > 0$,

$$
D_+(h) = \frac{f(x + h) - f(x)}{h}, \qquad
D_0(h) = \frac{f(x + h) - f(x - h)}{2h}.
$$

$D_+$ is the **forward difference**; $D_0$ is the **central difference**, which looks the same distance to both sides of $x$. For a loss $L$ of many parameters $\boldsymbol\theta$, apply them to the one-variable function $\varphi(t) = L(\boldsymbol\theta + t\,\mathbf{e}_i)$ that moves only parameter $i$. By the chain rule of Volume I, Chapter 8, $\varphi'(0) = \partial L / \partial \theta_i$, so

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\boldsymbol\theta + h\,\mathbf{e}_i) - L(\boldsymbol\theta - h\,\mathbf{e}_i)}{2h}.
$$

**How wrong is it?** Expand the function around $x$ (Volume I, Chapter 8):

$$
f(x \pm h) = f(x) \pm f'(x)\,h + \tfrac12 f''(x)\,h^2 \pm \tfrac16 f'''(x)\,h^3 + \tfrac1{24} f''''(x)\,h^4 \pm \dots
$$

For the forward difference only one expansion is used, and $f(x + h) - f(x) = f' h + \tfrac12 f'' h^2 + \dots$, so

$$
D_+(h) - f'(x) = \tfrac12 f''(x)\,h + \tfrac16 f'''(x)\,h^2 + \dots
$$

The leading error is proportional to $h$: the forward difference is **first order**. For the central difference the two expansions are subtracted. The terms with an even power of $h$ are equal in both and cancel; the odd ones have opposite signs and double:

$$
f(x + h) - f(x - h) = 2 f'(x)\,h + \tfrac13 f'''(x)\,h^3 + \dots
\quad\Longrightarrow\quad
D_0(h) - f'(x) = \tfrac16 f'''(x)\,h^2 + O(h^4).
$$

The central difference is **second order**: the error is proportional to $h^2$, because of a cancellation we get for free from symmetry. Cutting $h$ by a factor of 10 cuts the forward error by 10 and the central error by 100. The price is one extra evaluation of the function, since $D_+$ can reuse the value $f(x)$ that the check computes anyway.

Here the predictions are tested on a real loss. The function is the loss of Chapter 4's network along one weight, $W_1[2,2]$ (all other numbers as in that chapter), whose derivative from the backward pass we trust from the last two chapters. The second and third derivatives that appear in the error terms are estimated from differences of the same loss:

<!-- worked: gradient_checking.fd_orders -->
The loss of Chapter 4's network as a function of the single weight W1[2,2] (all other weights fixed at Chapter 4's values), in double precision. The reference derivative is the backward pass's 1.60223547. The second and third derivatives are L'' = 2.4443 and L''' = -2.4252, estimated by differences of L at steps 0.01 and 0.02.

| h | forward error | ratio to previous row | predicted L'' h / 2 | central error | ratio to previous row | predicted L''' h^2 / 6 |
|---|---:|---:|---:|---:|---:|---:|
| 1e-01 | 1.168e-01 |  | 1.222e-01 | 4.072e-03 |  | 4.042e-03 |
| 1e-02 | 1.218e-02 | 9.59 | 1.222e-02 | 4.041e-05 | 100.8 | 4.042e-05 |
| 1e-03 | 1.222e-03 | 9.97 | 1.222e-03 | 4.040e-07 | 100.0 | 4.042e-07 |
| 1e-04 | 1.222e-04 | 10.00 | 1.222e-04 | 4.040e-09 | 100.0 | 4.042e-09 |
<!-- /worked -->

The forward error divides by 10 each time $h$ does, and the central error by 100; the predicted columns, built from the leading term alone, agree to three or four digits once $h$ is small enough for the next term to be negligible (the forward error at $h = 0.1$ is off by about 4% from $\frac12 f'' h$ because of its $h^2$ term, $f'''h^2/6$). Both differences are *consistent* estimators of the gradient entry: an error that falls with $h$ at the predicted rate is itself evidence that the quantity being estimated is the right one.

If that were the whole story, the best strategy would be to take $h$ as small as the number format allows. It is not, and the next section says why.

## The step: truncation against round-off

Chapter 15 of Volume I derived the best step for a difference quotient of $e^x$. We repeat the derivation for the setting of a gradient check, because two things differ: the function is a loss computed by a long forward pass, and we ask for a tolerance we can set in advance.

The estimate has two enemies. **Truncation error** is what Taylor's expansion left out, $\frac16 L''' h^2$ for the central difference; it falls as $h$ falls. **Round-off error** comes from the arithmetic. The computed loss is not $L$ but $L$ plus an error $\mathrm{rnd}_L$ that the forward pass's roundings accumulated. A single rounding has relative error at most $u$ (Volume I, Chapter 15); a forward pass of many operations accumulates more, and we write $|\mathrm{rnd}_L| \approx \kappa\,u\,|L|$ with $\kappa \ge 1$ a factor that grows with the number of operations (a few for a tiny network, much more for a long sum). The two computed losses carry independent errors. Subtracting them is exact when they are close (Volume I, Chapter 15, Sterbenz), but the subtraction does not remove the errors, and the division by $2h$ magnifies them: the numerator is uncertain by about $2\,\mathrm{rnd}_L$, so the quotient is uncertain by $\mathrm{rnd}_L / h$. This is the catastrophic cancellation of Volume I, Chapter 15, in the shape that matters here: *the smaller the step, the fewer correct digits survive the subtraction*.

With $M_3$ a bound on $|L'''|$ near $\boldsymbol\theta$ along the direction of the step, the worst-case total error of the central difference is

$$
E(h) = \frac{M_3}{6}\,h^2 + \frac{\mathrm{rnd}_L}{h}.
$$

It falls, bottoms out and rises: one valley. Setting $E'(h) = \frac{M_3}{3} h - \mathrm{rnd}_L / h^2 = 0$ gives the best step and the error there:

$$
h^\ast = \left(\frac{3\,\mathrm{rnd}_L}{M_3}\right)^{1/3}, \qquad
E^\ast = \frac{M_3}{2}\,(h^\ast)^2 .
$$

At $h^\ast$ the two terms are $\frac{M_3}{6}h^{\ast 2}$ and $\frac{M_3}{3}h^{\ast 2}$: round-off already contributes twice as much as truncation (the stationarity condition says $\mathrm{rnd}_L/h^\ast = \frac{M_3}{3}h^{\ast 2}$, twice $\frac{M_3}{6}h^{\ast 2}$), which is why the valley's bottom lies a little to the right of where the two curves cross. Two consequences matter more than the constants. The best step scales as $u^{1/3}$ and the best achievable error as $u^{2/3}$: the central difference delivers about two thirds of the digits of the format.

Here is the curve on the same loss as before, along the same weight, with every operation of the forward pass rounded to 32 or to 64 bits (Volume I, Chapter 15, explains the rounding). All four columns are absolute errors against the backward pass's gradient $1.60223547$:

<!-- worked: gradient_checking.step_curve -->
Absolute error of the difference quotient of the loss along W1[2,2] (exact derivative 1.60223547), the loss and every operation of the forward pass rounded to the format. Conventions: tanh is the correctly rounded library tanh (rounded to 32 bits in the 32-bit columns); the perturbed weight w + h or w - h is rounded to the format before the loss is evaluated; the losses are 32-bit numbers in the 32-bit columns, but the quotient (L(w + h) - L(w - h)) / (2 h) is formed in double precision with the nominal 2 h (forward: (L(w + h) - L(w)) / h), and the error is its absolute difference from the double-precision backward gradient.

| h | 32-bit forward | 32-bit central | 64-bit forward | 64-bit central |
|---|---:|---:|---:|---:|
| 1e-1 | 1.17e-01 | 4.07e-03 | 1.17e-01 | 4.07e-03 |
| 1e-2 | 1.22e-02 | 4.18e-05 | 1.22e-02 | 4.04e-05 |
| 1e-3 | 1.25e-03 | 3.02e-06 | 1.22e-03 | 4.04e-07 |
| 1e-4 | 5.33e-04 | 6.26e-05 | 1.22e-04 | 4.04e-09 |
| 1e-5 | 1.13e-03 | 1.85e-03 | 1.22e-05 | 4.44e-11 |
| 1e-6 | 6.67e-02 | 3.69e-02 | 1.22e-06 | 7.22e-11 |
| 1e-7 | 7.82e-01 | 4.10e-01 | 1.22e-07 | 1.43e-09 |
| 1e-8 | 1.60e+00 | 1.60e+00 | 4.76e-09 | 7.94e-10 |
<!-- /worked -->

![Base-10 logarithm of the error of the forward (solid) and central (dashed) difference of a network's loss along one weight against minus the base-10 logarithm of the step h, in 32-bit (orange) and 64-bit (blue) arithmetic. In 64 bits the forward error falls with slope 1 to about 10 to the minus 8 near h = 10 to the minus 8, and the central error falls with slope 2 to about 10 to the minus 11 near h = 10 to the minus 5.5 before round-off raises both. The 32-bit curves share the 64-bit truncation lines for large h, leave them near h = 10 to the minus 2.5 (central) and 10 to the minus 3 (forward), reach their lowest points near h = 10 to the minus 2.7 (central) and 10 to the minus 4 (forward), and rise to the size of the derivative itself by h = 10 to the minus 8. Vertical marks show the predicted best central steps.](img/gradcheck-step-error.svg)

Both axes are logarithmic, so a straight line is a power law and its slope is the exponent. In 64 bits you can read the slopes 1 and 2 directly, then the valley at about $10^{-5}$ to $10^{-6}$, then the round-off branch. In 32 bits the valley sits at a step several hundred times larger and the best error is tens of thousands of times larger; once $h$ falls to the spacing of the stored weight (about $10^{-8}$ for a weight of size 0.1) the perturbed weight rounds back to the original and the numerator is exactly zero, so the error is the whole derivative, 1.60. Comparing with the formulas:

<!-- worked: gradient_checking.step_optimum -->
Central difference along W1[2,2], scanning h = 2^-1 ... 2^-30. Inputs: L = 0.742402, |L'''| = 2.4252 (as in fd_orders). Predicted: h* = (3 u L / |L'''|)^(1/3) and worst-case error |L'''| h*^2 / 2.

| format | u | predicted h* | predicted worst-case error | best h found | best error found |
|---|---:|---:|---:|---:|---:|
| 32-bit | 5.96e-08 | 3.80e-03 | 1.75e-05 | 2^-10 = 9.77e-04 | 1.59e-06 |
| 64-bit | 1.11e-16 | 4.67e-06 | 2.65e-11 | 2^-19 = 1.91e-06 | 7.95e-12 |
<!-- /worked -->

The predicted step is within a factor of 4 of the best step found and the best error found lies below the worst-case bound, as it should: the bound assumes every rounding is maximal and aligned. The ratio of the two predicted steps is $(2^{29})^{1/3} \approx 813$, which is the whole message for 32-bit checks: **a 32-bit gradient check needs a step of order $10^{-3}$ to $10^{-2}$ where a 64-bit one uses $10^{-5}$**, and it can only promise about four or five digits (the table's best error, $1.6 \times 10^{-6}$ on a gradient of 1.6, is a lucky minimum; the bound $1.75 \times 10^{-5}$ is the safe statement).

Three details of the arithmetic decide whether a 32-bit check works.

**The step you ask for is not the step you get.** The perturbed weight $w + h$ is stored in 32 bits, which holds only about seven digits. The loss sees $\mathrm{fl}(w + h) - \mathrm{fl}(w - h)$, not $2h$:

<!-- worked: gradient_checking.step_quantisation -->
The weight is w = 0.2000000030 (0.2 rounded to 32 bits). The two perturbed weights are rounded to 32 bits, and the step the loss actually saw is their difference.

| nominal h | nominal 2h | actual (w+h) - (w-h) | relative difference |
|---|---:|---:|---:|
| 1e-02 | 0.0200000000 | 0.0200000107 | 5.36e-07 |
| 1e-03 | 0.0020000000 | 0.0020000041 | 2.03e-06 |
| 1e-04 | 0.0002000000 | 0.0002000034 | 1.69e-05 |
| 1e-05 | 0.0000200000 | 0.0000199974 | 1.32e-04 |
<!-- /worked -->

At $h = 10^{-5}$ the quantisation alone changes the denominator by about $10^{-4}$ relative, and the numeric gradient inherits it. The remedy is to divide by the difference of the two stored values, not by $2h$, or to keep $h$ large.

**A long sum rounds every term.** A loss is usually a mean over many terms. Accumulating them in 32 bits adds the error of every addition to the numerator that the check subtracts. Take $L(w) = \frac1N\sum_k (a_k w + b_k)^2$ with $N = 4096$ terms, each computed in 32 bits, differing only in the accumulator that sums them:

<!-- worked: gradient_checking.accumulator_noise -->
L(w) = (1/N) sum_k (a_k w + b_k)^2 with N = 4096, w = 0.3000000119, h = 5e-3. Seed 5302511125843471603; draws in order: a_0 ... a_4095, then b_0 ... b_4095, each normal() rounded to 32 bits. w is 0.3 rounded to 32 bits, and the two evaluation points w + h and w - h are each rounded to 32 bits before use, so the quotient (L(w + h) - L(w - h)) / (2 h) is taken in 64 bits on 32-bit losses. Every product a_k w, sum with b_k and square inside a term is rounded to 32 bits; only the sum over k differs: the 32-bit accumulator adds the 4096 terms in order k = 0, 1, ... starting from 0, rounding every partial sum to 32 bits, and rounds the division by N to 32 bits; the 64-bit accumulator sums the (still 32-bit) terms exactly and divides by N in 64 bits. The exact derivative is 2/N sum a_k (a_k w + b_k) = 0.612077.

| sum over k | central difference | absolute error |
|---|---:|---:|
| 32-bit accumulator | 0.611889 | 1.87e-04 |
| 64-bit accumulator | 0.612076 | 7.15e-07 |
<!-- /worked -->

Summing in 64 bits (the terms are still 32-bit) cuts the error by a factor of about 260. This is why a checker's harness should accumulate its loss in the wider format even when the network runs in 32 bits: the quantity being differenced is the loss, and its round-off lands directly in the numerator of the central difference.

Over all 21 parameters of Chapter 4's network, in 32 bits, at several steps, with the two gates used later in this chapter (a pair $(\mathrm{atol}, \mathrm{rtol})$ passes an entry when $|\mathrm{ana} - \mathrm{num}| \le \mathrm{atol} + \mathrm{rtol}\,\max(|\mathrm{ana}|, |\mathrm{num}|)$; the next section derives it):

<!-- worked: gradient_checking.f32_needs_large_steps -->
Per-entry central differences of Chapter 4's loss on all 21 parameters, with every operation of the forward pass rounded to 32 bits, against the backward pass's gradient rounded to 32 bits. Relative error is |a - n| / max(|a|, |n|, 0.001). A gate (atol, rtol) passes an entry when |a - n| <= atol + rtol max(|a|, |n|); tanh is the correctly rounded library tanh rounded to 32 bits, the perturbed weight is rounded to 32 bits, and the quotient is formed in double precision with the nominal 2 h. The counts are entries passing.

| h | worst absolute error | worst relative error | loose gate (4e-3, 8e-2) | strict gate (1e-6, 1e-4) |
|---|---:|---:|---:|---:|
| 1e-05 | 7.03e-03 | 9.62e-01 | 20 of 21 | 0 of 21 |
| 1e-04 | 8.71e-04 | 7.30e-02 | 21 of 21 | 5 of 21 |
| 1e-03 | 7.86e-05 | 1.34e-02 | 21 of 21 | 14 of 21 |
| 5e-03 | 5.94e-05 | 4.44e-03 | 21 of 21 | 18 of 21 |
| 1e-02 | 2.17e-04 | 1.04e-02 | 21 of 21 | 18 of 21 |
| 5e-02 | 5.42e-03 | 1.34e-01 | 21 of 21 | 5 of 21 |
<!-- /worked -->

The strict gate is passed by no entry at $h = 10^{-5}$ and by most at $5 \times 10^{-3}$; the loose gate, whose absolute floor is larger than the noise, passes nearly all at every step shown. Exercise 6 asks for the same table in your language.

**Everything about the step is a property of the loss.** $M_3$, $L$ and $\kappa$ belong to the forward pass being checked, so the right step differs between tensors and models. It is found by measurement: compute the check at a ladder of steps, as in the table above, and choose a step on the flat part of the valley. A model whose check improves steadily as $h$ falls to $10^{-6}$ in 32 bits is not accurate; it is fooling you, and the next measurement will show the error rising again.

## Comparing two numbers

We now have two numbers per parameter entry: the **analytic** value from the backward pass, written $\mathrm{ana}$, and the **numeric** value from the central difference, written $\mathrm{num}$. How close is close enough? Three candidate rules each fail somewhere.

**Absolute error**, $|\mathrm{ana} - \mathrm{num}|$, is meaningless without a scale: an error of $0.2$ is a disaster for a gradient of $0.6$ and noise for a gradient of $250$. **Relative error** divides by the size of the quantity,

$$
\mathrm{err}_{\mathrm{rel}} = \frac{|\mathrm{ana} - \mathrm{num}|}{\max(|\mathrm{ana}|, |\mathrm{num}|)},
$$

which is scale-free: $0.2$ against $250$ is $8 \times 10^{-4}$. (Dividing by the larger of the two keeps the quantity symmetric and at most 1 when the signs agree.) But a relative error blows up where the true gradient is near zero, and a network has many such entries. Round-off puts a *floor* under the numeric value that does not depend on the size of the gradient: the noise of the numerator is $\mathrm{rnd}_L / h$ however small $\partial L / \partial\theta_i$ is. The relative error of an entry is therefore about $\mathrm{rnd}_L / (h\,|\partial L / \partial\theta_i|)$, unbounded as the gradient shrinks. In Chapter 4's network the entry $b_1[1]$ has a gradient of only $-0.000225$, and in 32 bits:

<!-- worked: gradient_checking.small_gradient_entries -->
32-bit per-entry central differences with h = 5e-3 on Chapter 4's network; the four entries with the smallest and the two with the largest gradient magnitude. Relative error is |a - n| / max(|a|, |n|).

| entry | analytic a | numeric n | absolute error | relative error |
|---|---:|---:|---:|---:|
| b1[1] | -0.000225 | -0.000221 | 4.44e-06 | 1.97e-02 |
| b1[3] | -0.064128 | -0.064141 | 1.25e-05 | 1.94e-04 |
| W1[1,0] | +0.095181 | +0.095189 | 7.19e-06 | 7.55e-05 |
| W1[3,0] | -0.095593 | -0.095606 | 1.25e-05 | 1.31e-04 |
| W1[2,2] | +1.602235 | +1.602238 | 2.94e-06 | 1.84e-06 |
| W1[2,1] | -2.135753 | -2.135694 | 5.93e-05 | 2.78e-05 |
<!-- /worked -->

The tiny gradient has a relative error of 2% from an absolute error of $4 \times 10^{-6}$, the same order as the absolute errors of the large entries ($3 \times 10^{-6}$ and $6 \times 10^{-5}$ for the last two rows), whose relative errors are 700 to 10,000 times smaller. A pure relative tolerance chosen to pass the large entries rejects the correct small one.

The standard fix is a **combined tolerance** with an absolute floor and a relative slope:

$$
|\mathrm{ana} - \mathrm{num}| \;\le\; \mathrm{atol} + \mathrm{rtol}\cdot\max(|\mathrm{ana}|, |\mathrm{num}|).
$$

The absolute tolerance $\mathrm{atol}$ is the noise floor, about the size of $E^\ast$ from the last section, so that entries whose gradient is at or below the noise level cannot fail on noise. The relative tolerance $\mathrm{rtol}$ is the fraction of the gradient we accept to be wrong, set above the truncation error. The rule is symmetric in the two arguments, which is why this chapter does not use the `close` of the pseudocode page (it scales by one argument only). In the pseudocode below it is the function `within_gate`. Here is how four rules judge six pairs of numbers. The first four pairs are correct gradients with numeric noise or a large scale; the last two are wrong, one by a factor of 2 and one dead:

<!-- worked: gradient_checking.compare_rules -->
| case | analytic a | numeric n | |a - n| | |a - n| / max(|a|, |n|) | abs 1e-4 | rel 1e-2 | atol 1e-4 + rtol 1e-2 | atol 4e-3 + rtol 8e-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| typical entry | 1.2293 | 1.22935 | 5.00e-05 | 0.00 | pass | pass | pass | pass |
| small true gradient, noisy numeric | 0.00022498 | 0.00029 | 6.50e-05 | 0.22 | pass | FAIL | pass | pass |
| true gradient zero, numeric noise | 0 | 3e-05 | 3.00e-05 | 1.00 | pass | FAIL | pass | pass |
| large gradient, relative noise | 250 | 250.2 | 2.00e-01 | 0.00 | FAIL | pass | pass | pass |
| dead gradient (analytic 0, true 2.2e-4) | 0 | 0.00022498 | 2.25e-04 | 1.00 | FAIL | FAIL | FAIL | pass |
| wrong by a factor 2 | 0.6 | 1.2 | 6.00e-01 | 0.50 | FAIL | FAIL | FAIL | FAIL |
<!-- /worked -->

Each single rule makes a mistake that the combined rule avoids: the absolute rule rejects the large entry, the relative rule rejects the entries near zero. And the combined rule has a price, visible in the fifth row. A *dead* gradient, exactly zero where the true value is $2.2 \times 10^{-4}$, passes the loose gate $(4 \times 10^{-3}, 8 \times 10^{-2})$, because $2.2 \times 10^{-4}$ is below its absolute floor. An absolute tolerance that forgives noise also forgives a missing gradient of the same size. A gate therefore needs a separate, structural test for a gradient that is *exactly* zero (every entry of a tensor, or analytic zero against a clearly nonzero numeric value), which is the topic of the section on what a check proves.

Why not just compare the whole gradient with one number? Two summaries are common and each has a blind spot. The **cosine** between the analytic and numeric gradient vectors measures the angle and ignores length: the first two bugs of the opening table score $1.000000$, so cosine alone cannot see a dropped or doubled scale (Volume I, Chapter 3 tells the real case that made the point; it is knowledge #2). The **relative L2 error**, $\lVert\mathrm{ana} - \mathrm{num}\rVert / \lVert\mathrm{num}\rVert$, catches scale (0.500 for the dropped 2) but pools every entry into one number, so a gradient that is wrong in a few entries and exact in the rest averages to something small: the bug that loses one example of 24 had a relative L2 error of 0.024 in the opening table. The entry-level gate, with a floor, is the check that cannot be averaged away.

## The algorithm

The gate and the central difference make the instrument. A **gradient check** is a test that compares the gradient the backward pass returns with finite-difference estimates of the same forward pass, entry by entry or along directions, under a stated step and tolerance. In pseudocode, with the gate first:

```text
function within_gate(analytic: real, numeric: real, atol: real, rtol: real) -> bool:
    require atol >= 0 and rtol >= 0, "tolerances are non-negative"
    return abs(analytic - numeric) <= atol + rtol * max(abs(analytic), abs(numeric))
```

```text
function gradient_check_entries(loss: function(array<real>[P]) -> real,
                                grad: array<real>[P], theta: array<real>[P],
                                indices: list<int>, h: real,
                                atol: real, rtol: real) -> list<int>:
    # grad is the backward pass's gradient at theta, computed once before the call
    require len(grad) == len(theta), "gradient and parameters differ in length"
    require h > 0, "the step must be positive"
    failing = []
    for i in indices:
        require 0 <= i < len(theta), "index outside the parameter vector"
        up = copy(theta)
        up[i] = theta[i] + h
        down = copy(theta)
        down[i] = theta[i] - h
        numeric = (loss(up) - loss(down)) / (2 * h)
        if not within_gate(grad[i], numeric, atol, rtol):
            append i to failing
    return failing
```

The function returns the failing entries, not a verdict, so that a failure says *where*. It changes no state: it works on copies. Pass it all $P$ indices for a full check. `loss` is the forward pass on a fixed batch; the batch must not change between the two calls and between the analytic and the numeric evaluation, or the comparison is of different functions (a fresh dropout mask, a freshly sampled batch).

**A scalar loss for any layer.** The loss is the end of the chain. To check a single layer whose output is an array $\mathbf{Y} = f(\mathbf{X}, \mathbf{W})$, use a *fixed random linear functional* of its output as the loss: $L = \sum_{ij} G_{ij}\,Y_{ij}$ with $\mathbf{G}$ drawn once and kept. Its gradient with respect to $\mathbf{Y}$ is $\mathbf{G}$ itself, so the backward pass is called with $d\mathbf{Y} = \mathbf{G}$, and the check tests the layer's whole vector-Jacobian product in one stroke. This is what Chapter 5 did for every rule; it needs no head and no target.

**What it costs.** Two forward passes per checked entry, $2P$ for all of them. In the work units of Chapter 4 (a multiply-add is two operations; matrix products only):

<!-- worked: gradient_checking.check_cost -->
Work counts multiplications and additions of the matrix products only (a multiply-add is two operations), as in Chapter 4: forward = 2 B D H + 2 B H O; backward = 2 O B H + 2 B O H + 2 H B D. The directional check uses 4 tensors, 4 directions each, 2 forward passes per direction.

| network | parameters P | forward work F | backward work | per-entry check: forward passes (2P) | per-entry work / backward work | directional work / backward work |
|---|---:|---:|---:|---:|---:|---:|
| tiny network of Chapter 4 | 21 | 64 | 80 | 42 | 34 | 25.6 |
| digit classifier | 203,530 | 26,017,792 | 26,345,472 | 407,060 | 401,997 | 31.6 |
| one wide layer pair | 134,238,208 | 2,147,483,648 | 3,221,225,472 | 268,476,416 | 178,984,277 | 21.3 |
<!-- /worked -->

For the 203,530-parameter classifier the full per-entry check costs about 400,000 backward passes. Nobody runs that for a large model, and nobody needs to: the check is a *test* on a small instance, with a model shrunk to a few dozen to a few thousand parameters, where $2P$ forward passes are cheap. For the entries that must be checked in a model too big for that, there are two economies: choose a subset, and contract many entries into one direction (the next section).

**Choosing the entries.** When not all of them can be checked, choose to cover *roles*, not a random sample. Take every entry of the small structurally interesting tensors (biases, scales, embeddings of a few rows, anything a backward pass folds or shares, below). For a large weight matrix take a deterministic stride (every $s$-th entry, so that every row and every column is touched) and include the corners, so that a wrong index or a boundary rule shows. Prefer entries with large $|\mathrm{ana}|$ where the noise floor is least significant, but always include some entries of every tensor, because a tensor that no check touches has no evidence at all.

## Directional checks, and what they miss

A full per-entry check is $2P$ passes; one pair of forward passes can check a whole tensor at once along a single direction. For a direction $\mathbf{v}$ the directional derivative of the loss (Volume I, Chapter 8) is $\nabla L\cdot\mathbf{v}$, the sum of $g_i v_i$ over all entries. The central difference along it estimates the same number:

$$
\nabla L \cdot \mathbf{v} \;\approx\; \frac{L(\boldsymbol\theta + h\mathbf{v}) - L(\boldsymbol\theta - h\mathbf{v})}{2h}.
$$

Take the direction with entries $\pm 1$ drawn at random, so that every entry is perturbed and no entry's sign is special. A **directional check** compares the backward pass's $\nabla L\cdot\mathbf{v}$, one number, with this estimate. It tests a tensor of $N$ entries with $2$ forward passes instead of $2N$.

```text
function gradient_check_directional(loss: function(array<real>[N]) -> real,
                                    grad: array<real>[N], w: array<real>[N],
                                    h: real, n_dirs: int, atol: real, rtol: real,
                                    rng: Generator) -> bool:
    # loss evaluates the model with this one tensor replaced by its argument
    require len(grad) == len(w), "gradient and parameters differ in length"
    require h > 0 and n_dirs >= 1, "a positive step and at least one direction"
    best_error = infinity
    best_ok = false
    for k in 0 .. n_dirs:
        v = zeros(real, [N])
        for j in 0 .. N:
            v[j] = 1.0 if rng.integer(2) == 1 else -1.0
        analytic = sum([grad[j] * v[j] for j in 0 .. N])
        up = [w[j] + h * v[j] for j in 0 .. N]
        down = [w[j] - h * v[j] for j in 0 .. N]
        numeric = (loss(up) - loss(down)) / (2 * h)
        error = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 0.001)
        if error < best_error:
            best_error = error
            best_ok = within_gate(analytic, numeric, atol, rtol)
    return best_ok
```

The "best of $n_{\mathrm{dirs}}$" selection exists for a reason. A random direction can be nearly orthogonal to the gradient, so that $\nabla L\cdot\mathbf{v}$ is close to zero and its relative error is large for the wrong reason (noise over a tiny number, as in the last section). Trying a few directions and keeping the best-conditioned one (the smallest relative error) avoids spurious failures of that kind. It has a price, which the table below measures: the same selection also lets a real partial bug through more often, and the more directions are tried the more often.

**Why it is stable, and what step it wants.** The directional estimate has a larger numerator than a single-entry one: the loss moves by about $h\,|\nabla L\cdot\mathbf{v}|$, and $\nabla L\cdot\mathbf{v}$ is a sum of $N$ terms, typically of size $\lVert\nabla L\rVert$ rather than one entry's. The round-off floor $\mathrm{rnd}_L / h$ is the same, so the signal-to-noise ratio is better, and a directional check tolerates a smaller $h$ than a per-entry one. But the step is longer than it looks: a direction of $N$ entries of size $\pm1$ has Euclidean length $\sqrt{N}$, so the weights move by $h\sqrt{N}$ in total. A tensor with 864 entries and $h = 5 \times 10^{-3}$ moves by $0.147$ in weight space, which may be well outside the region where the loss is locally smooth, and the truncation error grows with it. The step of a directional check must therefore *shrink as the tensor grows*, and be measured (the section on brain shows such a measurement).

**What it cannot see.** A directional check compares one number. Suppose the backward pass is *partly* wrong, say it drops a share of the gradient: the true gradient is $\mathbf{g}$, the computed one is $\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{m}$, and $\mathbf{m}$ is the missing part. The check measures $\tilde{\mathbf g}\cdot\mathbf v$ against $\mathbf g\cdot\mathbf v$, so the error it sees is $\mathbf{m}\cdot\mathbf{v}$, a *projection* of the missing part. For a random $\pm1$ direction that projection is a random number of typical size $\lVert\mathbf m\rVert$ with a sign that varies from direction to direction, and it can be small even when $\lVert\mathbf m\rVert$ is large: the contraction of $N$ numbers into one lets them cancel. A wholly wrong gradient fails in every direction. A partly wrong one fails in *most* directions but passes in some, and the best-of-$n_{\mathrm{dirs}}$ rule *selects the direction where the error is smallest*, which is the worst selection against exactly this bug.

Volume I, Chapter 8 built a toy of this effect on a linear model and showed best-of-4 accepting a gradient with about 0.31 of its size missing in 579 of 1000 problems. The same effect on this chapter's network, with the bug of the opening table that drops the last example of 3 from the first-layer batch sum (the missing part has a relative size 0.525 here), with the loose gate of a 32-bit check:

<!-- worked: gradient_checking.directional_miss -->
Bug: the batch sum of dW1 omits the last of B = 3 examples. One generator, seed 16685689773887133093, draws 1000 problems one after another (D = 5, H = 7, O = 2); for each, X, T, W1, b1, W2, b2 as in bug_catalogue, then four directions of 35 entries each (+1 when integer(2) is 1, otherwise -1), the first of the four being 'one random direction'. Directional comparison: eps = 0.0001 on W1 alone; per entry: h = 1e-5; gate 0.004 + 0.08 max(|a|, |n|), double precision.

| measure | result over 1000 random problems |
|---|---:|
| mean of ||missing part|| / ||gradient of W1|| | 0.525 |
| one random direction: bug accepted | 136 of 1000 |
| best of 4 random directions: bug accepted | 387 of 1000 |
| per-entry check: at least one of the 35 entries fails | 997 of 1000 |
| per-entry check: mean share of the 35 entries that fail | 0.695 |
<!-- /worked -->

A single direction lets the bug through in 136 problems of 1000, the best of four in 387, while the per-entry check fails for 997 of 1000 and, on average, 70% of the 35 entries disagree. The per-entry check wins because it compares the whole vector entry by entry; nothing contracts the missing part away. The practical rule is a division of labour. A directional check is the right tool for big tensors, where $2N$ passes are out of the question and a whole-tensor mistake (a wrong factor, a wrong kernel) fails every direction. A per-entry check is needed for every tensor that a backward pass *folds, shares, ties or accumulates*, where the characteristic bug is one contribution missing out of several: a weight used twice, a bias summed over a batch, a gradient added from two heads. For those tensors the per-entry check must also be run at a setting where the missing contribution is large: if the second use of a shared weight is scaled by a coefficient near zero, its share is below the noise and the check passes whether or not the contribution is there (knowledge #187).

## Choosing test dimensions

The check is a test on a small instance, so the sizes of that instance are a choice, and a bad choice hides bugs from a perfect checker. **Test-dimension choice** is the selection of the sizes (batch, features, hidden width, outputs, heads) of the model that a test runs on, so that every distinct role a size plays gets its own distinct number. When two sizes are equal, or a size is one, bugs that confuse them or depend on them are not wrong anymore: they compute the same function.

Four bugs and four sets of sizes show it. The bugs: a mean over the batch only, $1/B$ where $1/(BO)$ belongs; a bias gradient taken from one example instead of summed over the batch; $d\mathbf{A} = d\mathbf{Y}\mathbf{W}_2^\top$ instead of $d\mathbf{Y}\mathbf{W}_2$ (a transposed weight); and $d\mathbf{W}_1 = \mathbf{X}^\top d\mathbf{Z}_1$ instead of $d\mathbf{Z}_1^\top\mathbf{X}$ (a transposed product). For each set of sizes $(B, D, H, O)$ the table says what the check finds:

<!-- worked: gradient_checking.dims_hide -->
One generator, seed 6951451086357263854, draws the four problems one after another; for each, in the order X (row-major), T (row-major), W1 (row-major), b1, W2 (row-major), b2, each normal(), the last four multiplied by 0.5. Per-entry check in double precision: central difference h = 1e-5, gate |a - n| <= 1e-07 + 1e-05 max(|a|, |n|). "Shape error": the buggy formula's result has a shape that cannot be the parameter's, or the product does not exist.

| test sizes | mean over B instead of B O | bias gradient of example 0 only | dA = dY W2^T | dW1 = X^T dZ1 |
|---|---|---|---|---|
| all ones, (B, D, H, O) = (1, 1, 1, 1) | same gradient, invisible | same gradient, invisible | same gradient, invisible | same gradient, invisible |
| Chapter 4's network, (B, D, H, O) = (2, 3, 4, 1) | same gradient, invisible | caught (1 of 21 entries) | shape error | shape error |
| equal sizes, (B, D, H, O) = (2, 3, 3, 3) | caught (24 of 24 entries) | caught (3 of 24 entries) | caught (12 of 24 entries) | caught (6 of 24 entries) |
| distinct, all above 1, (B, D, H, O) = (3, 5, 7, 2) | caught (58 of 58 entries) | caught (2 of 58 entries) | shape error | shape error |
<!-- /worked -->

Read it row by row. With all sizes 1, *nothing* is detectable: every bug computes the same gradient as the correct one, because a $1\times 1$ matrix is its own transpose, a sum over one example is that example, and $B\,O = B$. A check on such an instance is a green light for any code. Chapter 4's own network, with all different sizes but $O = 1$, rejects both transposes by shape and catches the batch-sum bug, but the mean over $B$ instead of $B\,O$ is invisible, because with one output $B\,O = B$. The row of equal sizes catches all four, but only by luck of the values: with square matrices a transposed product is shape-legal, and it is the *numbers* that betray it (12 of 24 entries wrong in one case), not the shapes. A random weight matrix is not symmetric, but a weight that is initialised to the identity, to zero, or tied to its transpose would hide the bug again. The last row, with all sizes distinct and above 1, is the only one where each bug is rejected by shape or caught by value, and the transposes are rejected by a *shape error* before any numerics: the cheapest and most localising detection there is.

The rule that follows is short.

- Choose all dimensions distinct (small primes work well: $B = 3$, $D = 5$, $H = 7$, $O = 2$), none equal to 1, so no two roles are interchangeable and no sum has one term.
- Avoid products and quotients that coincide with another dimension (a width that equals the number of heads times the head size, a hidden size that is twice the input) when the architecture has a rule linking them; give the roles that the architecture might confuse numbers that differ.
- Use at least two examples, two outputs, two heads, two layers: whatever the model sums over must be summed over something.
- Initialise away from structure: no identity or zero weights, no equal rows, no all-equal inputs.
- If a gate must cover several distinct cases (two widths of attention, two kinds of block), make sure the test *visits one of each*; a test that happens to sample two of the same kind covers one.

Knowledge #4 records how this was learned the expensive way in brain, at a size where $\text{heads} = d_{kv} = 64$ and $\text{heads}\times d_{kv} = d_{\text{model}} = 4096$: any swap of two of those would have passed.

## Kinks

Everything so far assumed the loss is smooth, because the Taylor expansion of the first section does. Piecewise-linear functions break the assumption: the ReLU has no derivative at zero, a maximum has none at a tie, and a clamp has none at its two ends. The backward pass picks a convention there (the book's ReLU rule gives slope 0 at $x = 0$; a maximum sends the gradient to the first index). The check cannot validate a convention, and near a kink it fails for a reason that has nothing to do with the code.

The central difference of $\mathrm{relu}$ at $x$ with $|x| < h$ is $(\mathrm{relu}(x + h) - \mathrm{relu}(x - h))/(2h) = (x + h)/(2h)$: it climbs linearly from 0 at $x = -h$ through $\frac12$ at $x = 0$ to 1 at $x = h$ (the average of the slopes of the two sides), while the backward pass reports 0 or 1.

![The slope of ReLU against x: the backward pass's step from 0 to 1 at x = 0 (solid blue) and the central difference with step h = 0.1 (dashed orange), which is a straight ramp from 0 at x = -h to 1 at x = h passing through one half at x = 0. Vertical lines mark x = -h and x = h; the two curves agree outside them.](img/gradcheck-kink-ramp.svg)

At a distance more than $h$ from the kink the two agree; inside, they disagree by up to $\frac12$. With a step of $10^{-3}$:

<!-- worked: gradient_checking.kink_scan -->
Central difference of relu(x) at x = delta with step h = 0.001, against the slope 1 for x > 0 and 0 otherwise (double precision). A row agrees when |difference| < 1e-9.

| delta | analytic slope | central difference | |difference| | verdict |
|---|---:|---:|---:|---:|
| +1e-01 | 1 | 1.0000 | 0.0000 | agree |
| +1e-02 | 1 | 1.0000 | 0.0000 | agree |
| +2e-03 | 1 | 1.0000 | 0.0000 | agree |
| +1e-03 | 1 | 1.0000 | 0.0000 | agree |
| +5e-04 | 1 | 0.7500 | 0.2500 | DISAGREE |
| +1e-04 | 1 | 0.5500 | 0.4500 | DISAGREE |
| +1e-06 | 1 | 0.5005 | 0.4995 | DISAGREE |
| 0 | 0 | 0.5000 | 0.5000 | DISAGREE |
| -1e-04 | 0 | 0.4500 | 0.4500 | DISAGREE |
| -2e-03 | 0 | 0.0000 | 0.0000 | agree |
<!-- /worked -->

The agreement stops exactly when $\delta$ falls to $h$ (the row $\delta = 10^{-3}$ is the last that agrees; the rows at $5 \times 10^{-4}$ and below show the ramp: $0.75$, $0.55$, and $0.5$ at the kink itself). In a network this is not a curiosity but a certain event, because we do not choose which pre-activations land near zero. Chapter 4's network, with tanh replaced by ReLU, has a hidden unit that sits at the kink for the first example:

<!-- worked: gradient_checking.kink_network -->
Chapter 4's network with tanh replaced by ReLU (slope 1 for z > 0, otherwise 0). Hidden unit j has pre-activation Z1[b, j] for the two examples b = 0, 1. Per-entry check of the four hidden biases, central difference h = 1e-5, double precision, gate 1e-07 + 1e-05 max(|a|, |n|). Each pre-activation is the dot product of a row of W1 with the example, summed from index 0 upward starting at 0.0, plus b1[j]; an entry of size 1e-17 is a rounding residue whose sign depends on that order.

| hidden unit j | Z1[0, j] | Z1[1, j] | analytic db1[j] | central difference | verdict |
|---|---:|---:|---:|---:|---:|
| 0 | -2.000e-01 | +7.500e-01 | +0.432500 | +0.432500 | agree |
| 1 | +7.000e-01 | -3.000e-01 | +0.243000 | +0.243000 | agree |
| 2 | +2.776e-17 | -2.000e-01 | -0.648000 | -0.323998 | DISAGREE |
| 3 | -7.000e-01 | +9.500e-01 | +0.173000 | +0.173000 | agree |
<!-- /worked -->

Unit 2's pre-activation for the first example is $+2.8 \times 10^{-17}$, zero in exact arithmetic and a rounding residue in floating point. Which side of the kink the backward pass is on is decided by that residue, and the central difference reports half of the slope. The code is correct and the check fails for that unit, and for the three other units it passes.

The remedies are all about where we test, not how.

- **Measure the margin and resample.** For a ReLU after a linear layer, the preactivation of a unit moves by up to $h\sum_d |x_d|$ when the weights of its row are all perturbed by $\pm h$ (a directional check), and by $h\,|x_d|$ when one is (a per-entry check). Compute the smallest $|z|$ over the batch for every unit and require it to exceed that movement; if not, draw another batch.
- **Replace the discontinuity by a smooth stand-in for the check.** A smooth activation of the same shape (GELU, softplus) exercises the surrounding code; the kink's own rule is tested once, in isolation, by a unit test that evaluates it on either side.
- **Freeze discrete decisions.** A top-$k$ router, an argmax, a nearest-codebook assignment is piecewise constant, and a perturbation can flip it. Fix the decision for the duration of the sweep: choose $k$ equal to the number of experts so that every expert is selected, or record the assignment once and reuse it.
- **Never initialise at a kink.** This is a different failure, the one in knowledge #166: a parameter that *starts* exactly where a clamp's slope is zero receives no gradient and never moves. The check, run at random weights, would not see it. It belongs in initialisation, not in testing.

## A catalogue of bugs

Now put the pieces together. Take the bugs that have appeared and a few more, inject each into the backward pass of a small network with distinct sizes ($B = 3$, $D = 5$, $H = 7$, $O = 2$), and see what each of two checks says, in 64 bits where the tolerances can be tight:

<!-- worked: gradient_checking.bug_catalogue -->
Network B = 3, D = 5, H = 7, O = 2 (58 parameters), tanh, mean squared error, double precision. Seed 9546417083664895460; draws in order: X (row-major), T (row-major), W1 (row-major), b1, W2 (row-major), b2, each normal(), the last four multiplied by 0.5. Per-entry check: central difference with h = 1e-5 on every parameter. Directional check: per tensor 4 directions of +-1 entries (seed 10285673115552927475; for W1, b1, W2, b2 in turn, four directions each, an entry is +1 when integer(2) is 1 and -1 otherwise), step eps = 1e-4, the best of the four kept by relative error. The 16 directions are drawn once, before any row is evaluated, and every row that is evaluated uses these same 16; a row rejected by the shape check is not evaluated, so it uses none. Both use the gate |a - n| <= 1e-07 + 1e-05 max(|a|, |n|). The cosine and relative L2 error compare the 58 analytic values with the correct backward pass's.

| bug injected in the backward pass | cosine with correct gradient | relative L2 error | per-entry check: entries failing | directional check, best of 4 |
|---|---:|---:|---:|---|
| none (control) | 1.000000 | 0.000 | 0 of 58 | pass |
| dropped factor 2: dY = E / (B O) | 1.000000 | 0.500 | 58 of 58 | FAIL: W1, b1, W2, b2 |
| mean over B only: dY = 2 E / B | 1.000000 | 1.000 | 58 of 58 | FAIL: W1, b1, W2, b2 |
| tanh derivative omitted: dZ1 = dA | 0.943953 | 0.402 | 42 of 58 | FAIL: W1, b1 |
| tanh derivative from Z1: 1 - Z1^2 | 0.593759 | 1.383 | 42 of 58 | FAIL: W1, b1 |
| transposed weight: dA = dY W2^T | - | - | rejected by the shape check | - |
| transposed product: dW1 = X^T dZ1 | - | - | rejected by the shape check | - |
| bias gradient of one example: db2 = dY[0] | 0.999063 | 0.044 | 2 of 58 | FAIL: b2 |
| batch sum misses last example in dW1 | 0.918779 | 0.405 | 35 of 58 | FAIL: W1 |
<!-- /worked -->

Row by row, the table is a summary of the chapter. The control passes both checks, which says that the checks do not cry wolf. The first two bugs are scale errors: perfect cosine, large relative L2, every entry wrong, and every directional comparison wrong: any check that compares *values* sees them, and cosine alone does not. The tanh derivative bugs corrupt only what is upstream of the mistake: 42 of 58 entries (the first layer's 35 weights and 7 biases) fail while the second layer's 16 pass, as in Chapter 4. The two transposes are rejected by the shape check, which is detection by distinct dimensions, not by numerics; the previous section showed what they do when shapes agree. The bias gradient of a single example is the *small* bug: cosine 0.999, relative L2 0.044, and only 2 of 58 entries disagree, those of $\mathbf{b}_2$; both checks see it, and a check that pools entries could miss it. The last bug, the missing example, fails the per-entry check on 35 of 58 entries and the directional check on the first-layer weights; the section above measured how often the directional check misses it in general.

## What a check proves

A passing check establishes one thing: *the gradient the backward pass returned is the derivative of the forward pass that the check ran, at the weights and batch it ran on, to within the tolerance*. Every word of that is a limit.

**The forward you wrote, not the forward you meant.** A finite difference of a wrong objective is the derivative of that wrong objective. If the loss weights a term by the wrong coefficient, if two loss terms are attached to the wrong tensors, if the forward pass computes a plausible but different function, the backward pass can be its exact derivative and the check passes. In brain's vector-quantised image model (VQGAN) the check distinguishes the codebook term from the commitment term (swapping them zeroes the codebook's gradient), but it cannot say which of the two a scale $\beta$ multiplies: the check differentiates whatever is emitted. That half of the model was pinned by reading the reference implementation's source (knowledge #3). The check validates the *derivative*, never the *objective*; the objective needs a reference, a published value or a golden output.

**Whatever the tolerance forgives.** The gate's absolute floor forgives any wrong gradient smaller than $\mathrm{atol}$. Brain has two GELU functions, an exact one built from the error function and a tanh approximation, whose derivatives differ everywhere by a small amount, and a model that borrowed the derivative of one for the other would train on the wrong gradient:

<!-- worked: gradient_checking.tolerance_blindness -->
The two GELU functions are gelu(x) = x Phi(x) with Phi the standard Gaussian distribution function, and the tanh form 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))). Over the 801 points x = -4.00, -3.99, ..., 4.00 the largest difference between the two derivatives is 8.68e-04. The derivatives are Phi(x) + x phi(x) (phi the standard Gaussian density) and, for the tanh form with t = tanh(sqrt(2/pi)(x + 0.044715 x^3)), 0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2), tanh the library tanh. Each gate is asked whether one derivative, taken as the analytic value, agrees with the other, taken as the numeric one.

| gate | atol | rtol | points where the wrong derivative is rejected |
|---|---:|---:|---:|
| brain's 32-bit gate | 0.004 | 0.08 | 0 of 801 |
| a 64-bit gate | 1e-07 | 1e-05 | 766 of 801 |
<!-- /worked -->

The difference never exceeds $8.68 \times 10^{-4}$, which is below the absolute floor of the 32-bit gate, so the gate accepts the wrong derivative at *every* point *for any relative tolerance*: the two terms of the gate add, and the absolute term alone already covers the gap. A 64-bit gate rejects the mismatch at 766 of 801 points. The lesson is not that the loose gate is wrong (32-bit arithmetic needs it), but that a tolerance defines what the check can see, and that anything smaller must be pinned some other way (here, by a test that the two kernels are *not* interchangeable).

**Dead and untested gradients.** Analytic zero against a numeric value below the absolute floor passes (the fifth row of the comparison table), and a parameter that no check touches has no evidence. Two structural tests close those holes: no parameter that must train has a gradient of exactly zero everywhere, and a report that lists analytic exactly 0 against a clearly nonzero numeric value is a failure whatever the gate says.

**One point, one backend.** The check holds at one set of weights and one batch. A bug that depends on a branch the batch did not exercise (a mask that is all ones, a sequence with no padding, a path taken only for long inputs) is not tested. And a kernel may be right on the device the test ran on and wrong on another: a reduction written for a device with cooperative workgroups can come back all zeros on a device without them, with no error. A check that ran on one backend certifies that backend. In brain every gradient check is run on every backend (knowledge #5); later volumes, on computation, return to why backends diverge.

## Making it work

A short checklist, in the order a check is built.

1. Build the smallest model that exercises every layer type, with distinct sizes, in a format you can switch (64-bit for the first run).
2. Choose the loss: the model's own, or a fixed random linear functional of its output for a layer under test. Fix the batch and any randomness.
3. Run the per-entry check in 64 bits with $h \approx 10^{-5}$ and a tight gate. Anything that fails is a bug or a kink; localise with the entry label.
4. Move to the format the model runs in. Measure the check at a ladder of steps, pick a step on the flat part of the valley, accumulate the loss in the wider format, and set $\mathrm{atol}$ at the noise floor and $\mathrm{rtol}$ above the truncation error.
5. For tensors too large to check per entry, use directional checks with a step that shrinks with the tensor, and add per-entry checks of the folded, shared and small tensors.
6. Run the structural tests (no all-zero gradient where a gradient is due; every parameter covered) and pin by other means whatever the gate cannot see.
7. Run it on every backend you ship.

## Where this appears in brain

`crates/gradcheck` is brain's gradient check, and `crates/gradcheck/src/lib.rs` its core. It was introduced as a replacement for an external framework used as an oracle (commit `afbbae41d`, 24 June 2026, whose message calls it a "directional finite-difference check" and reports the first language model's backward validated on all 29 of its tensors).

**The harness.** A model is checked through the trait `CheckModel`: seven methods (`param_names`, `read_weight`, `write_weight`, `read_grad`, `loss`, `zero_grads`, `backward`). The check never looks inside a model: it writes a perturbed weight, calls `loss`, reads a gradient. A blanket implementation covers every model that implements the engine's `Model` trait, so a new architecture needs a few lines to enter, namely a small configuration and a fixed batch (`check_gpt`, `check_qwen`, `check_autoencoder`); models whose structure does not fit (two parameter stores, a seeded objective rather than a loss) get a bespoke harness. A comment at the head of each model's module states which parameters the check covers and which are frozen, because a green result covers only what it touched. T5's check uses the loss of this chapter's section on scalar losses, $L = \langle \mathbf{r}, \text{output}\rangle$ with a fixed random $\mathbf{r}$, since the encoder has no head.

**The comparison.** `Check::within` is this chapter's gate, $|a - n| \le \mathrm{atol} + \mathrm{rtol}\cdot\max(|a|, |n|)$, with the doc comment's reason: a pure relative metric is ill-conditioned for a gradient near zero. The shared tolerance $(4 \times 10^{-3}, 8 \times 10^{-2})$ is used by the `brain gradcheck` command and by the assertion helper every model check goes through. The reported relative error divides by $\max(|a|, |n|, 10^{-3})$, a floor. `Report::dead_gradients` flags analytic exactly 0.0 against a numeric value above $10^{-3}$, and `zero_grad_params` returns every parameter whose gradient came back all zero: both are the structural tests of the last section (commit `83990db00`, 10 August 2026, generalised them from one model's test, noting that the absolute floor lets a dead gradient pass).

**Step choices.** `directional_check` takes a step (the doc comment says $5 \times 10^{-3}$ suits 32-bit, the value of the formula $(3u)^{1/3} = 5.6 \times 10^{-3}$ of this chapter), a number of directions (4 is typical), and keeps the best direction by relative error. `elementwise_check` is the per-entry check, and `elementwise_check_at` takes an explicit set of indices, used for a strided sample of a 16384-entry weight whose exhaustive check would need over thirty thousand forward passes of a multi-layer model (commit `d8cf72130`, 22 September 2026). A grep for this book (November 2026) of the literal step arguments in `crates/gradcheck/src` finds about forty uses of $5 \times 10^{-3}$, about a dozen of $5 \times 10^{-4}$, and a handful each of $10^{-2}$, $10^{-3}$ and smaller values: the right step is a per-model measurement, as the section on steps said. The T5 module records that measurement. Because a $\pm1$ direction over $N$ entries has length $\sqrt{N}$, its largest tensor has 864 entries, and at $5 \times 10^{-3}$ that is a step of 0.147 in weight space, so the module uses $5 \times 10^{-4}$, and its doc comment records the maximum relative error over all 17 tensors against the step, on one GPU backend and on the CPU backend:

| step $h$ | $5 \times 10^{-3}$ | $2 \times 10^{-3}$ | $10^{-3}$ | $5 \times 10^{-4}$ | $2 \times 10^{-4}$ | $10^{-4}$ |
|---|---:|---:|---:|---:|---:|---:|
| GPU backend | 2.92e-3 | 4.81e-4 | 2.21e-4 | 5.54e-4 | 3.12e-3 | 1.57e-3 |
| CPU backend | 2.90e-3 | 4.61e-4 | 2.27e-4 | 3.05e-4 | 6.85e-4 | 2.82e-3 |

This is the valley of the figure above, measured on a real model: truncation dominates above $2 \times 10^{-3}$, round-off below $2 \times 10^{-4}$, and the floor is at $10^{-3}$ to $5 \times 10^{-4}$. (These numbers are the doc comment's record, gated by the test `t5_eps_plateau` only loosely; they were not re-measured for this book.) The module's per-entry check uses a step of $10^{-2}$, twenty times larger, for the reason of the section on directional checks: one entry has no $\sqrt{N}$ amplification, so the loss difference is $h\,|\partial L/\partial w_i|$ and 32-bit round-off bites sooner. The harness's `loss` accumulates its host-side dot product in 64 bits for the reason of the accumulator table (commit `4171bccb0`, 4 August 2026).

**Three stories.**

*The check that cannot see a third missing.* The same commit, `4171bccb0`, added the per-entry check. A weight shared across T5's blocks, the relative-position bias, has its gradient contributions summed over blocks; deleting that sum left a third of the gradient missing, and the best-of-4 directional check reported a relative error of $6.2 \times 10^{-4}$ for it at one seed and $5.3 \times 10^{-2}$ at another, both inside the gate. The per-entry check over the tensor's 24 entries fails when the sum is deleted. This is the effect measured above; Volume I, Chapter 8 told the story in the context of its own toy. Knowledge #187 found the same blindness on a second model, in which the shared weights were two convolutions of a vision tower whose second use is added scaled by a learned coefficient; at the coefficient 0.7 the directional check caught the missing use for one of the two tensors and passed the other, and at the checkpoint's own coefficient of about $5.6 \times 10^{-3}$ the second use is below the resolution of any check (`crates/gradcheck/src/sam_hd.rs`).

*The kinks that made a check flicker.* The check of the ModernBERT decision head (commit `d8cf72130`, 22 September 2026) ran a directional check over a 16384-entry weight that feeds a ReLU. Three back-to-back runs on one GPU backend gave a failing, then two passing results for the identical seeded check; the analytic value was bit-identical across them, and the numeric one moved. The module's analysis is this chapter's: with a $\pm1$ perturbation of all 16384 weights, every one of 256 output units shifts, and some are close enough to zero to cross their kink within the step, so the numerator contains the unit's ramp. The cure was the two the chapter lists: a per-entry check on a strided sample (one entry moves one pre-activation, so at most one kink is ever crossed), and a smaller step. The commit message records that the signature of the failure was a non-monotonic sweep over the step: an error that does not fall smoothly as $h$ falls.

*The gradient that was silently zero on one backend.* The ArcFace training graph (commit `5523e9d12`, 4 August 2026) has a PReLU whose learned slope has a gradient that is a per-channel reduction. The kernel for it uses a workgroup barrier, and the kernel's own header records that the CPU backend's code generator mis-executes that construct: the slope gradient came back all zeros while the gradient of the input stayed correct. There is no crash, the network trains to a plausible loss, and a gate run only on the GPU passes it. The training graph therefore selects a barrier-free kernel on devices that lack cooperative reductions, and a test asserts that no slope gradient is all zero. The policy that came out of it is knowledge #5: run every gradient check on both backends. The Makefile's `test/parity` target runs the whole gradcheck suite under the CPU device, then on the GPU and, where there is one, the CUDA device (`scripts/gates/parity-gate.sh`), and the release gate `test/full` includes it since commit `2c32bfe72` (11 September 2026); before that commit the CPU run was a target that people had to remember.

## Check your understanding

Answers are at the end of the chapter.

1. **Explain.** In 64 bits the central difference of a smooth loss at $h = 10^{-5}$ is more accurate than at $h = 10^{-10}$, though $10^{-10}$ is far smaller. Using the two error terms, say why, and estimate both errors for $M_3 = 1$, $|L| = 1$, $u = 1.1 \times 10^{-16}$.
2. **Predict.** A gate uses only a relative tolerance of 1% (no absolute floor). What happens to a correct backward pass at a gradient entry whose true value is exactly 0, and why does adding an absolute floor repair it?
3. **Find the error.** "The cosine between my analytic and numeric gradients is 0.99999, so the backward pass is right." Name a bug that gives cosine $1.000000$, and say what to compute besides.
4. **Find the error.** "I tested the layer with $B = 1$ and $D = H = O = 4$ and all gradients matched, so the backward pass is correct." Which bugs does this test hide, and why?
5. **Predict.** A ReLU unit has pre-activation $z = 3 \times 10^{-4}$ on one example, and the check uses $h = 10^{-3}$ on its bias. What does the central difference report for that example's contribution, and what does the backward pass report? What should you do?
6. **Explain.** A directional check on a large weight matrix passes. Why does this not prove that the gradient of a weight used at two places in the network includes both uses, and what test would?
7. **Explain.** Name two kinds of backward-pass error that a passing check with the 32-bit gate cannot detect, and for each say what detects it.
8. **Predict.** Use the last table of the section on step size (the counts of entries within each gate over all 21 parameters): which of $h = 10^{-5}$ and $h = 5 \times 10^{-3}$ do you expect to pass the strict gate $(10^{-6}, 10^{-4})$ on more entries, and why does the loose gate hide the difference?

## Exercises

1. **[Calculate]** For $f(x) = x^3$ at $x = 2$ and $h = 0.1$, compute the forward and the central difference by hand. Compare their errors with $\frac12 f'' h + \frac16 f''' h^2$ and $\frac16 f''' h^2$.

2. **[Calculate]** Compute $h^\ast = (3u)^{1/3}$ and $E^\ast = \frac12 h^{\ast 2}$ (with $M_3 = |f| = 1$) for 16-bit half precision, $u = 2^{-11}$, and for 32-bit, $u = 2^{-24}$. What relative accuracy of a gradient can a half-precision check promise?

3. **[Calculate]** Apply the gate $(\mathrm{atol}, \mathrm{rtol}) = (4 \times 10^{-3}, 8 \times 10^{-2})$ to $\mathrm{ana} = 0.020$, $\mathrm{num} = 0.026$ and then the gate $(10^{-4}, 10^{-2})$. Which pass? Is the first passing result evidence that the gradient is right?

4. **[Calculate]** For $B = 32$, $D = 100$, $H = 200$, $O = 10$ (Chapter 4's two-layer network), count the parameters, the forward work and the backward work in multiply-add operations, and the work of a full per-entry central-difference check. How many backward passes is that?

5. **[Calculate]** A ReLU unit has pre-activation $z = 2 \times 10^{-4}$. For $h = 10^{-3}$ compute the central difference of $\mathrm{relu}$ and compare it with the backward pass. At what $h$ would the two agree?

6. **[Implement]** In your language, implement the loss of Chapter 4's network and the forward and central differences along $W_1[2,2]$, and reproduce the 64-bit columns of `gradient_checking.step_curve` and the table `gradient_checking.fd_orders`. If your language has 32-bit floats and lets you round after every operation, reproduce the 32-bit columns to their order of magnitude (they depend on rounding the perturbed weight and every operation, with the quotient formed in double precision and divided by the nominal $2h$; use the library $\tanh$) and locate the minimum of the central error. Reproduce also `gradient_checking.f32_needs_large_steps`.

7. **[Implement]** Implement the backward pass with the nine injectable bugs of `gradient_checking.bug_catalogue`, the per-entry check with the 64-bit gate, and the best-of-4 directional check, using the generator of Volume I, Chapter 17 with the seeds and draw orders printed in the table. Reproduce the cosine, the relative L2 error and the number of failing entries of every row.

8. **[Implement]** Reproduce `gradient_checking.dims_hide`: four problems drawn one after another from one generator, four bugs, and the per-entry check. Then add a fifth set of sizes of your own with all sizes distinct and a sixth where $H = O = D$ and the weights are symmetric; what does the transposed-product bug do on the sixth?

9. **[Implement]** Reproduce `gradient_checking.directional_miss` (1000 problems from one generator). Then vary the number of directions (1, 4, 16, 64) and the number of dropped examples (1 or 2 of 3), and report how often the bug is accepted for each. State why choosing the best of more directions does not make the check reliable, and what $n$ directions cost in forward passes ($2n$) compared with the per-entry check of 35 entries (70).

10. **[Investigate]** Replace tanh by ReLU in your implementation and scan the margin: shift the first-layer bias of one hidden unit so that its pre-activation on one example is $\delta$, for $\delta = 10^{-1}, 10^{-2}, \dots, 10^{-6}$, and check the unit's bias with $h = 10^{-5}$ in 64 bits. At which $\delta$ does the check start to fail, and how does that compare with $h$? Then run the directional check on the first-layer weights and find the largest perturbation of the pre-activations over the batch; is $\delta$ above it?

11. **[Investigate]** Reproduce `gradient_checking.accumulator_noise` for $N = 16, 256, 4096, 65536$ and plot the error of the 32-bit accumulator against $N$. Does it grow like $N$, like $\sqrt{N}$, or neither? Explain with the error bound of summation from Volume I, Chapter 15.

## Summary

A backward pass that is slightly wrong still trains: a dropped factor of 2 is a learning rate, a missing example is noise, and a cosine of 1 sees neither. The instrument that needs no trust in the code under test is the definition of the derivative. A **finite difference** estimates a derivative from function values; the forward difference has error proportional to $h$ and the **central difference** error proportional to $h^2$, as Taylor's expansion shows and the loss of a small network confirms (ratios of 10 and 100 per decade). The error has a second source: the loss is computed with rounding, and dividing the difference of two nearby losses by $2h$ magnifies it, so the total error has a valley with its bottom at $h^\ast = (3\,\mathrm{rnd}_L / M_3)^{1/3}$, a step about 800 times larger in 32 bits than in 64 and an accuracy of only about two thirds of the format's digits. A long loss should be accumulated in the wider format, and the stored perturbed weights, not $2h$, set the true step. Comparing two numbers needs both a floor and a slope: the **relative error** blows up at small gradients, the absolute error has no scale, and the gate $|\mathrm{ana} - \mathrm{num}| \le \mathrm{atol} + \mathrm{rtol}\max(|\mathrm{ana}|, |\mathrm{num}|)$ has a floor at the noise and a slope above the truncation; cosine sees direction only and L2 averages entries away. A **gradient check** does this for every entry, at a cost of $2P$ forward passes (about 400,000 backward passes for a 200,000-parameter classifier), so it runs on a small model; one pair of passes per random direction checks a whole tensor and misses a partly missing gradient, which is why folded, shared and small tensors need the per-entry check. **Test-dimension choice** decides what a check can see: with a size of 1 or two equal sizes whole bug classes compute the same function as the correct code, so sizes are distinct, above 1, and uncorrelated with the architecture's rules. Kinks, ties and discrete decisions make a correct backward pass fail the check; tests keep a margin larger than the perturbation, smooth the function, or freeze the decision. A passing check proves the derivative of the forward that was run, at one batch and one set of weights, to the gate's tolerance, on the backend that ran it: not the objective, not what the tolerance forgives, not dead or uncovered gradients, not another backend.

In brain, `crates/gradcheck` implements this: a trait with seven methods, the gate as `Check::within`, directional and per-entry checks with a step measured per model, structural tests for dead gradients, and a policy of running every check on every backend, shaped by the failures of its history.

Limits: a check tells whether a backward pass is the derivative of a forward pass, not whether training will succeed. For that we need to choose how to use the gradient, which is the subject of the next chapter, Optimizers: From SGD to AdamW.

## Key terms

- **finite difference**: an estimate of a derivative built from values of the function at nearby points, such as the forward difference $(f(x+h) - f(x))/h$. <!-- concept: finite-difference -->
- **central difference**: the finite difference $(f(x+h) - f(x-h))/(2h)$, whose error is proportional to $h^2$ because the first error term cancels by symmetry. <!-- concept: central-difference -->
- **relative error**: the size of the difference between two numbers divided by the larger of their sizes, which is independent of scale but unbounded near zero. <!-- concept: relative-error -->
- **gradient check**: a test that compares the gradient returned by a backward pass with finite-difference estimates of the same forward pass, entry by entry or along random directions, under a stated step and tolerance. <!-- concept: gradient-check -->
- **test-dimension choice**: choosing the sizes of the model a test runs on (batch, features, widths, heads) all different and above 1, so that bugs that confuse two sizes or depend on a size of 1 change the result. <!-- concept: test-dimension-choice -->

## Answers

**Before you begin.**

1. $D_0(h) = (f(x + h) - f(x - h))/(2h)$; its error is $\frac16 f'''(x)h^2 + \dots$, proportional to $h^2$, so a tenth of the step gives a hundredth of the error.
2. $f(x + h) = f + f' h + \frac12 f'' h^2 + \frac16 f''' h^3 + \dots$
3. $u = 2^{-24} \approx 5.96 \times 10^{-8}$ in 32 bits and $2^{-53} \approx 1.11 \times 10^{-16}$ in 64 bits; it bounds the relative error of rounding one number to the nearest representable one.
4. The subtraction is exact, but the absolute errors of the two operands, about $u|f|$ each, stay while the difference itself is small, so the relative error of the difference is about $u|f|/|f(x+h) - f(x)|$, large when the difference is small.
5. A central difference gradient needs $2P$ forward passes; the backward pass costs about two forward passes, independent of $P$ (Chapter 4's cost table: 1.01 to 1.5 times the forward work for the first layer-pair network).

**Check your understanding.**

1. The error is $\frac{M_3}{6}h^2 + \mathrm{rnd}_L / h$ with $\mathrm{rnd}_L \approx u|L|$. At $h = 10^{-5}$ truncation is $\frac16 \cdot 10^{-10} \approx 1.7 \times 10^{-11}$ and round-off $1.1 \times 10^{-16}/10^{-5} \approx 1.1 \times 10^{-11}$, total about $2.8 \times 10^{-11}$. At $h = 10^{-10}$ truncation is negligible ($1.7 \times 10^{-21}$) but round-off is $1.1 \times 10^{-6}$. Smaller steps lose digits in the subtraction faster than truncation falls.
2. The numeric value is nonzero noise, say $3 \times 10^{-5}$, and the relative error against an analytic 0 is $|0 - 3 \times 10^{-5}|/3 \times 10^{-5} = 1$, a relative error of 1, the largest possible. An absolute floor larger than the noise makes the right side $\mathrm{atol} + \dots \ge 3 \times 10^{-5}$, so the correct entry passes.
3. A scale bug, such as a dropped factor 2, has cosine exactly 1.000000 and relative L2 error 0.5. Also compute the relative L2 error, and above all the per-entry gate; the cosine alone cannot see length.
4. With $B = 1$ a sum over the batch is a single term, so a bias gradient from one example, or a missing sum, is invisible; with $D = H = O$ a transposed product is shape-legal, and with a size of 1 a transposition is a no-op. Equal sizes also hide any swap of two of the dimensions. Use distinct sizes, none equal to 1.
5. The central difference of $\mathrm{relu}$ at $z = 3 \times 10^{-4}$ with $h = 10^{-3}$ is $(z + h)/(2h) = 0.65$; the backward pass reports slope 1. The code is right and the test point is wrong: choose inputs so that $|z|$ exceeds the movement of the pre-activation under the perturbation (here more than $10^{-3}$), or test a smooth stand-in.
6. A directional check compares one number per direction, a projection of the gradient, and a missing share can project to nearly nothing; keeping the best of several directions selects such a projection. A per-entry check of that tensor, at a setting where the second use contributes visibly, would catch it.
7. (a) A wrong objective: the derivative of the wrong forward is correct, so the check passes; a reference value or golden output detects it. (b) A gradient wrong by less than $\mathrm{atol}$, or exactly zero where the true derivative is below it: a tighter gate where the arithmetic allows, a test that the gradient is nonzero, or a separate test pinning the two functions apart. (Also acceptable: an error on a backend or a branch the test did not run.)
8. $h = 5 \times 10^{-3}$: at $h = 10^{-5}$ the table reports 0 of 21 entries within the strict gate (the quantisation and round-off errors are largest there) and 18 of 21 at $5 \times 10^{-3}$. The loose gate passes 20 and 21 because its absolute floor $4 \times 10^{-3}$ is larger than the noise at either step.

## References

- Higham, N. J., *Accuracy and Stability of Numerical Algorithms*, second edition, 2002.
- Griewank, A. and Walther, A., *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, second edition, 2008.
