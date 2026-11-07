<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Linear Regression and the First Descent

*How do we find the straight line that best explains our measurements, and what does "best" mean?*

## The problem

The previous chapter ended with a promise and a complaint. The promise: a model
is a function with open parameters, and learning is choosing them with data. The
complaint: the only way we had to choose was to try thousands of lines on a grid
and keep the one with the smallest "average off by", and the cost of that grows
as $m^k$ with $k$ parameters. The workshop that built the temperature sensor
still wants an answer for a sensor that reads $1.75$ V, and it would like to
get it without computing $58{,}081$ scores.

There are really three questions hiding in that complaint, and this chapter
answers them in order.

1. **What does "best" mean?** The average of absolute differences was chosen
   because it is easy to read, not because it is right. We need a number that
   scores a choice of parameters, with a name, a definition and a reason.
2. **Can the best line be computed instead of searched for?** For a line with a
   score of the right kind, yes: it is the solution of a short system of
   equations, and the same derivation works for any number of features.
3. **What do we do when that cannot be done?** When there are too many
   parameters or too much data for a direct solution, we need a method that
   improves a guess step by step. That method, which gets its name in its own
   section below, is the one that will train every network in this book. We study it here on the one
   problem where we can see all of it: the line.

The data are the ten sensor rows of chapter 1 (their generator and seed are given there, in the section "Too small, too big").

<!-- worked: learning_problem.sensor_table -->
N = 10 measured rows (sorted by voltage; the source that produced them and the order of its draws are written out in the section "Too small, too big").

| row n | voltage v (V) | temperature y (deg C) |
|---|---:|---:|
| 1 | 0.42 | 13.4 |
| 2 | 0.95 | 32.8 |
| 3 | 1.60 | 52.2 |
| 4 | 1.63 | 55.5 |
| 5 | 1.80 | 57.5 |
| 6 | 1.90 | 58.9 |
| 7 | 1.96 | 61.7 |
| 8 | 2.19 | 65.0 |
| 9 | 2.30 | 66.8 |
| 10 | 2.68 | 70.6 |
<!-- /worked -->

You may use from earlier chapters: the model, its parameters and the idea of a
score for a choice of parameters (chapter 1 of this volume); the transpose, the
matrix inverse and elimination (Volume I, chapter 4); convexity (chapter 7);
the gradient and the condition number (chapter 8); gradient flow and the
stability of Euler's method (chapter 9); and z-scores (chapter 12).

## Before you begin

Six questions to answer from memory. The answers are at the end of the chapter.
If one is hard, go back to the chapter named before reading on.

1. (Chapter 1 of this volume, "The Learning Problem".) The score of chapter 1 for the line $f(v) = 24v + 9$ was $4.29$ degrees. If we change the line to $f(v) = 20v + 18$, which things change: the ten rows, the parameters, the score?
2. (Volume I, chapter 8, "Gradients, Jacobians and Many Variables", the gradient.) For $L(w_1, w_2) = w_1^2 + 3 w_1 w_2$, what is the gradient at $(1, 1)$, and in which direction does $L$ fall fastest?
3. (Volume I, chapter 4, "Matrices as Linear Maps", transpose and shapes.) A matrix $\mathbf{X}$ has shape $[10, 2]$ and $\mathbf{y}$ has shape $[10]$. What are the shapes of $\mathbf{X}^\top$, $\mathbf{X}^\top\mathbf{X}$ and $\mathbf{X}^\top\mathbf{y}$?
4. (Volume I, chapter 4, "Matrices as Linear Maps", elimination.) Solve $2 w_1 + w_2 = 4$, $w_1 + 3 w_2 = 7$ by subtracting a multiple of one equation from the other, and say what has to be true of the pivot for this to work.
5. (Volume I, chapter 7, "Derivatives: The Mathematics of Change", convexity.) Why does a convex function make "stop where the slope is zero" a safe rule for finding the lowest point?
6. (Volume I, chapter 9, "Integrals and Differential Equations", Euler's method.) For $dx/dt = -\lambda x$ with $\lambda > 0$, one Euler step of size $h$ is $x \leftarrow x - h\lambda x$. For which $h$ does $\lvert x\rvert$ shrink at every step?

## A score with a name: the loss function

Chapter 1 scored a line by $E(\theta) = \frac{1}{N}\sum_n \lvert f_\theta(\mathbf{x}_n) - y_n\rvert$. That is one choice among many, and the choice matters, because the line that scores best depends on what the score rewards. Write the idea in general. For one example, a **loss function** is a rule $\ell(\hat{y}, y)$ that takes a prediction and the label it should have matched and returns a single non-negative number, zero for a perfect prediction and larger the worse it is. The loss of a model on a dataset is the average of the per-example losses,

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{n=1}^{N} \ell\bigl(f_\theta(\mathbf{x}_n),\, y_n\bigr).
$$

The notation page's $L$ is the loss for one example and $\mathcal{L}$ the loss
averaged over a dataset; from here on, "the loss" means $\mathcal{L}$, a function
of the parameters only, since the data are fixed. Learning, in the sense of
chapter 1, is now a precise task: find $\theta$ that makes $\mathcal{L}(\theta)$
small. The absolute difference of chapter 1 is $\ell(\hat{y}, y) = \lvert\hat{y} - y\rvert$.
The loss this chapter builds on is the square of the difference,

$$
\ell(\hat{y}, y) = (\hat{y} - y)^2,
\qquad
\text{and its average}
\qquad
\mathrm{MSE}(\theta) = \frac{1}{N}\sum_{n=1}^{N} \bigl(f_\theta(\mathbf{x}_n) - y_n\bigr)^2 .
$$

The average is called the **mean squared error**. Its square root, the root mean
squared error, is in the unit of the label (degrees, not degrees squared) and is
the number to quote when a person has to read the result.

### A first computation

Take chapter 1's first attempt, $w = 24$, $b = 9$, and score it with the new
rule. Each row's difference is squared, and the squares are averaged.

<!-- worked: linear_regression.mse_first_attempt -->
Attempt 1 (w = 24, b = 9) with squared errors, computed from the unrounded predictions.

| v | label y | prediction | prediction - y | squared |
|---|---:|---:|---:|---:|
| 0.42 | 13.4 | 19.08 | +5.68 | 32.2624 |
| 0.95 | 32.8 | 31.80 | -1.00 | 1.0000 |
| 1.60 | 52.2 | 47.40 | -4.80 | 23.0400 |
| 1.63 | 55.5 | 48.12 | -7.38 | 54.4644 |
| 1.80 | 57.5 | 52.20 | -5.30 | 28.0900 |
| 1.90 | 58.9 | 54.60 | -4.30 | 18.4900 |
| 1.96 | 61.7 | 56.04 | -5.66 | 32.0356 |
| 2.19 | 65.0 | 61.56 | -3.44 | 11.8336 |
| 2.30 | 66.8 | 64.20 | -2.60 | 6.7600 |
| 2.68 | 70.6 | 73.32 | +2.72 | 7.3984 |
|  |  |  | total | 215.3744 |
|  |  |  | MSE = total / 10 | 21.5374 |
<!-- /worked -->

The average absolute difference of this line was $4.29$; its MSE is $21.54$ and
its root is $4.64$ degrees. Look at the column of squares: the four rows where the line is
off by more than $5$ degrees contribute $146.85$ of the total $215.37$ (68 percent), and the row
it gets nearly right ($1.00$ off) contributes $1.0$. That
is the first property of this loss, and the reason it is a choice and not a
formality.

### Why square the difference

Four reasons, in the order they matter for what follows.

**It punishes large misses much harder than small ones.** A prediction that is
off by $2$ costs $4$; off by $4$ it costs $16$; off by $8$ it costs $64$. Two
models can have the same average absolute error and very different squared
error. Four misses of $2$ each give an average absolute error of $2$ and an
average squared error of $4$. One miss of $8$ among three exact predictions
also gives an average absolute error of $2$ (that is $8/4$), but an average
squared error of $64/4 = 16$. Squaring says: one large error is worse than
several small ones adding up to the same total. For a sensor, where a reading
that is $8$ degrees off is a failure and one that is $2$ off is a nuisance,
this is usually what we want.

**It is smooth.** The absolute value has a corner at zero, where the slope
jumps from $-1$ to $+1$ and the derivative of chapter 7 does not exist. The
square $(\hat{y} - y)^2$ has a derivative $2(\hat{y} - y)$ everywhere, which
shrinks to zero as the prediction approaches the label. A method that follows
slopes (gradient descent, below) takes smaller steps as it nears a good fit,
without being told to.

**It is convex in the parameters of a line.** For a model that is a line, the
prediction is a linear function of $(w, b)$, the square of a linear function is
a bowl (a curve or surface with one lowest point that rises on every side), and an average of bowls is a bowl (the chord inequality of chapter 7
can be added term by term, so a sum of convex functions is convex). So the MSE of a line has one lowest point and no other valleys. We
check this by computing the curvature below.

**It has a probabilistic reading.** Suppose the label is the model's value plus
noise from a Gaussian distribution (chapter 11) with standard deviation
$\sigma$: $y = f_\theta(\mathbf{x}) + \varepsilon$. The density of the label is
$\frac{1}{\sqrt{2\pi\sigma^2}}\exp\bigl(-\frac{(y - \hat{y})^2}{2\sigma^2}\bigr)$,
and its negative logarithm,

$$
-\log p(y \mid \mathbf{x}) = \frac{(y - \hat{y})^2}{2\sigma^2} + \tfrac{1}{2}\log(2\pi\sigma^2),
$$

is the squared difference, scaled, plus a number that does not depend on
$\theta$. So minimising the squared error is the same as choosing the
parameters under which the observed labels were most probable, if the noise is
Gaussian. The chapter "Maximum Likelihood, Priors and Regularization" makes
that precise and shows what changes when the noise is not Gaussian. For now it
is the reason the squared error is not an arbitrary choice: noise that is the
sum of many small independent effects is nearly Gaussian (the central limit
theorem of chapter 11), and the thermometer's error is that kind.

### What each loss does when there is nothing to learn

A fair test of a loss is the simplest model of all: one that ignores the voltage
and always answers the same number $c$. Which $c$ does each loss prefer?

<!-- worked: linear_regression.constant_model -->
A model with no feature at all: it answers one constant c for every voltage. Both losses over the ten labels of the sensor table. The median of ten labels is the mean of the 5th and 6th smallest; the candidates c are the median and the mean at full precision (shown to 2 decimals), and 50 and 60.

| c | mean absolute error | MSE |
|---|---:|---:|
| 50.0 | 14.200 | 287.444 |
| median 58.20 | 11.160 | 298.268 |
| mean 53.44 | 12.384 | 275.610 |
| 60.0 | 11.380 | 318.644 |
<!-- /worked -->

The squared error is smallest at the mean of the labels, $53.44$, and the
absolute error is smallest at the median, $58.20$ (the midpoint of the two
middle labels). Both facts have one-line proofs. The derivative of
$\frac{1}{N}\sum_n (c - y_n)^2$ is $\frac{2}{N}\sum_n (c - y_n)$, which is zero
at $c = \bar{y}$, the sample mean of chapter 12. The absolute loss has slope
$+1$ for each label below $c$ and $-1$ for each above it, and the total slope
changes sign where half of the labels are on each side, the median. This is the
cleanest statement of what the squared error asks for: **the prediction that
is right on average**, which is the sample mean when there is nothing to
condition on and a conditional mean once the features enter.

## The loss of a line

Return to the model $\hat{y} = w v + b$ and the loss

$$
\mathrm{MSE}(w, b) = \frac{1}{N}\sum_{n=1}^{N}\bigl(w v_n + b - y_n\bigr)^2 .
$$

This is a function of two numbers, the same kind of object as the two-knob loss
of chapter 8, and every tool of that chapter applies. In particular the loss
can be pictured as a surface above the $(w, b)$ plane, with the best line at the
lowest point. We will not compute the lowest point by trying. The chapter 8
facts are enough: at the lowest point of a smooth function the gradient is zero.

The partial derivatives come from the chain rule of chapter 7. Write
$r_n = \hat{y}_n - y_n = w v_n + b - y_n$ for the difference between prediction and label on row
$n$: the **residual** of the row. Every residual in this chapter is
"prediction minus label", in the formulas and in the tables. (The residual of
chapter 3, what is left of a vector after a projection, is the label side minus the
prediction side, the opposite sign; the sign changes no length and no right
angle, and it is fixed here once so that the gradient and the tables agree.) Then

$$
\frac{\partial\,\mathrm{MSE}}{\partial b} = \frac{2}{N}\sum_n r_n ,
\qquad
\frac{\partial\,\mathrm{MSE}}{\partial w} = \frac{2}{N}\sum_n r_n v_n ,
$$

because the derivative of $r_n^2$ is $2 r_n$ times the derivative of $r_n$, and
$r_n$ changes by $1$ per unit of $b$ and by $v_n$ per unit of $w$.

### Setting the slope to zero: the exact line

Both partial derivatives must vanish. The first condition, $\sum_n r_n = 0$,
reads $\sum_n (w v_n + b - y_n) = 0$, that is $N b + w\sum_n v_n = \sum_n y_n$.
Divide by $N$ and write $\bar{v}$ and $\bar{y}$ for the means:

$$
b = \bar{y} - w\,\bar{v}.
$$

The best line passes through the point of means $(\bar{v}, \bar{y})$, whatever its
slope is. Substitute this into the second condition $\sum_n r_n v_n = 0$. Now
$r_n = w(v_n - \bar{v}) - (y_n - \bar{y})$, so the condition becomes
$\sum_n \bigl[w(v_n - \bar{v}) - (y_n - \bar{y})\bigr] v_n = 0$. Because
$\sum_n (v_n - \bar{v}) = 0$ and $\sum_n (y_n - \bar{y}) = 0$, replacing the last
factor $v_n$ by $v_n - \bar{v}$ changes nothing, and

$$
w = \frac{S_{vy}}{S_{vv}},
\qquad
S_{vv} = \sum_n (v_n - \bar{v})^2,
\qquad
S_{vy} = \sum_n (v_n - \bar{v})(y_n - \bar{y}).
$$

The slope is the covariance of voltage and temperature divided by the variance
of voltage (both unnormalised: the factors of $N - 1$ cancel), the quantities of chapter 12. For
the sensor:

<!-- worked: linear_regression.closed_form_line -->
Gradient entries and the sums below are zero up to rounding noise: a reproduction agrees when their size is below 1e-9, whatever digits it prints.

| quantity | value |
|---|---:|
| mean of v | 1.7430 |
| mean of y | 53.4400 |
| S_vv = sum (v - mean v)^2 | 3.8754 |
| S_vy = sum (v - mean v)(y - mean y) | 100.8538 |
| slope w = S_vy / S_vv | 26.0240 |
| intercept b = mean y - w mean v | 8.0801 |
| MSE at (w, b) | 13.1481 |
| dMSE/dw at (w, b) | 3.8e-15 |
| dMSE/db at (w, b) | 3.2e-15 |
<!-- /worked -->

The two gradient entries at the solution are zero to the rounding of the
arithmetic (below $10^{-9}$ in size, as the header of the table says; the digits printed after that are noise), as they must be. We should also check that the
point is a minimum and not a maximum or a saddle: the second derivatives are
$\partial^2 M/\partial b^2 = 2$, $\partial^2 M/\partial w^2 = \frac{2}{N}\sum_n v_n^2$
and the cross derivative is $2\bar{v}$. The determinant of this Hessian
(chapter 8) is $\frac{4}{N}\sum v_n^2 - 4\bar{v}^2 = \frac{4}{N} S_{vv}$, positive unless all the
voltages are equal. A Hessian with positive diagonal and positive determinant is positive definite, so
the loss is a bowl, convex in the sense of chapter 7, and the solution is its
one and only lowest point.

This procedure, choosing the parameters that minimise the sum of squared
differences between predictions and labels, is **least squares**. The line it
produces is the least-squares line, and the closed form above (the line through the
means with slope $S_{vy}/S_{vv}$) is known to anyone who has drawn a line through
a scatter plot in a spreadsheet.

The residuals of the least-squares line:

<!-- worked: linear_regression.line_residuals -->
The least-squares line and its residuals r = prediction - label, the sign convention of the whole chapter. The sum of residuals is zero up to rounding noise (agreement: size below 1e-9, whatever digits it prints).

| v | label y | prediction | residual (prediction - y) |
|---|---:|---:|---:|
| 0.42 | 13.4 | 19.01 | +5.61 |
| 0.95 | 32.8 | 32.80 | +0.00 |
| 1.60 | 52.2 | 49.72 | -2.48 |
| 1.63 | 55.5 | 50.50 | -5.00 |
| 1.80 | 57.5 | 54.92 | -2.58 |
| 1.90 | 58.9 | 57.53 | -1.37 |
| 1.96 | 61.7 | 59.09 | -2.61 |
| 2.19 | 65.0 | 65.07 | +0.07 |
| 2.30 | 66.8 | 67.94 | +1.14 |
| 2.68 | 70.6 | 77.82 | +7.22 |
|  |  | sum of residuals | +1.6e-14 |
<!-- /worked -->

They sum to zero (the first gradient condition, $\sum r_n = 0$, in numbers). The
residuals at both ends are positive (the line is above the labels) and in the middle mostly
negative (below them): a curved pattern that a line cannot follow. The picture draws each residual as a
vertical segment from the point to the line. The loss is the average of the
squares of those segment lengths.

![The ten measured rows as dots, the least-squares line drawn solid, and the vertical segment from each dot to the line. The segments are the residuals: longest at the highest voltage and at the lowest, where the line overshoots, shorter in the middle, where it mostly undershoots.](img/least-squares-line.svg)

### How it compares with the grid search

Score the lines of chapter 1, and the new line, with both losses.

<!-- worked: linear_regression.rescored_attempts -->
The same lines as the previous chapter, scored by the average absolute error E, by the mean squared error and by its square root (in degrees). Weights are shown to 3 decimals, scores to 2.

| line | w | b | E (deg C) | MSE (deg C^2) | sqrt(MSE) (deg C) |
|---|---:|---:|---:|---:|---:|
| attempt 1 (hand) | 24.000 | 9.000 | 4.29 | 21.54 | 4.64 |
| attempt 2 (hand) | 20.000 | 15.000 | 5.82 | 40.03 | 6.33 |
| attempt 3 (hand) | 20.000 | 18.000 | 4.22 | 27.55 | 5.25 |
| best of the grid | 27.250 | 7.000 | 2.77 | 14.85 | 3.85 |
| least squares | 26.024 | 8.080 | 2.81 | 13.15 | 3.63 |
<!-- /worked -->

Two observations. First, the least-squares line has the smallest MSE of any
line in the table, $13.15$ against $14.85$ for the grid's best, and it did not need
the $58{,}081$ evaluations of the grid: it is a handful of sums. Second, it does
*not* have the smallest average absolute error: the grid search, which minimised
that loss, reaches $2.77$ and the least-squares line has $2.81$. Each method is
best at the loss it was built for. The grid's line lets the highest-voltage row be
$9.4$ degrees off in exchange for six rows within $1.7$; the least-squares line
has no miss above $7.22$, at the price of smaller misses spread over more rows, and
that is the trade the squared error asks for: one large miss costs more than several small ones.

## Many features: the design matrix and the normal equations

A line has one feature. A real model has many (the houses of chapter 1 had
three), and a derivation that has to be redone by hand for every number of
features is of little use. The same result, derived once in matrix language, covers
all of them, and it is the form every later chapter uses.

### Stacking the model into a matrix

Take $N$ examples with $d$ features each. As in chapter 1, stack the feature
vectors as the rows of $\mathbf{X}$ with shape $[N, d]$. A model that is a
weighted sum of the features, $\hat{y} = w_1 x_1 + \dots + w_d x_d$, predicts all
$N$ rows at once as the matrix-vector product of chapter 4,

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}, \qquad \mathbf{X}: [N, d],\ \mathbf{w}: [d],\ \hat{\mathbf{y}}: [N].
$$

The line has an intercept $b$ that multiplies no feature. The trick that removes
the special case is to give every example one extra feature that is always $1$:
$\mathbf{x} = (1, v)$. Then $w_0 \cdot 1 + w_1 v$ is the line with intercept
$b = w_0$ and slope $w = w_1$. The matrix built this way, with a first column of
ones, is the **design matrix** of the model. From here on the parameter vector is
$\mathbf{w} = (w_0, w_1)$, intercept first, and every model in this chapter has
the form $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$.

With this notation the loss is a norm (chapter 3): the vector of residuals is
$\mathbf{X}\mathbf{w} - \mathbf{y}$ and

$$
\mathrm{MSE}(\mathbf{w}) = \frac{1}{N}\lVert \mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2
= \frac{1}{N}(\mathbf{X}\mathbf{w} - \mathbf{y})^\top(\mathbf{X}\mathbf{w} - \mathbf{y}).
$$

### The gradient, derived once

Expand the product, using $(\mathbf{X}\mathbf{w})^\top = \mathbf{w}^\top\mathbf{X}^\top$
(the transpose rule of chapter 4):

$$
N\cdot\mathrm{MSE} = \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w}
- 2\,\mathbf{w}^\top\mathbf{X}^\top\mathbf{y} + \mathbf{y}^\top\mathbf{y}.
$$

(The two cross terms are equal because each is a single number and a number
equals its own transpose.) Write $\mathbf{A} = \mathbf{X}^\top\mathbf{X}$, a
matrix of shape $[d, d]$ that is symmetric, and $\mathbf{c} = \mathbf{X}^\top\mathbf{y}$,
a vector of shape $[d]$. We need the gradient of $\mathbf{w}^\top\mathbf{A}\mathbf{w}$
and of $\mathbf{w}^\top\mathbf{c}$ with respect to $\mathbf{w}$, entry by entry, as in chapter 8.

- $\mathbf{w}^\top\mathbf{c} = \sum_j w_j c_j$, whose derivative with respect to $w_k$ is $c_k$. The gradient is $\mathbf{c}$.
- $\mathbf{w}^\top\mathbf{A}\mathbf{w} = \sum_{i}\sum_{j} A_{ij} w_i w_j$. The derivative with respect to $w_k$ collects the terms with $i = k$ and the terms with $j = k$: $\sum_j A_{kj} w_j + \sum_i A_{ik} w_i = 2(\mathbf{A}\mathbf{w})_k$, using $A_{ik} = A_{ki}$. The gradient is $2\mathbf{A}\mathbf{w}$.

Putting the pieces together and dividing by $N$,

$$
\nabla_{\mathbf{w}}\,\mathrm{MSE} = \frac{2}{N}\bigl(\mathbf{X}^\top\mathbf{X}\mathbf{w} - \mathbf{X}^\top\mathbf{y}\bigr)
= \frac{2}{N}\,\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}).
$$

The second form says what the gradient is in words: the residual vector, sent
back through the transpose of the design matrix. Entry $j$ is the dot product of
column $j$ of $\mathbf{X}$ (the $j$-th feature across all examples) with the
residuals, scaled by $2/N$. For the line this reproduces the two sums of the
previous section: the column of ones gives $\frac{2}{N}\sum r_n$ and the column
of voltages gives $\frac{2}{N}\sum r_n v_n$. A formula like this should always
be checked against the definition of a derivative before it is trusted, and the
check is cheap: nudge each parameter by a small $h$ and divide the change in the
loss by $2h$.

<!-- worked: linear_regression.gradient_check -->
At attempt 1 (b = 9, w = 24): the formula (2/N) X^T (X w - y) against central differences with step 1e-5.

| component | formula | finite difference |
|---|---:|---:|
| dMSE/db (intercept) | -5.21600 | -5.21600 |
| dMSE/dw (slope) | -10.66028 | -10.66028 |
<!-- /worked -->

### The normal equations

Setting the gradient to zero gives the **normal equations**,

$$
\mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}.
$$

This is a system of $d$ linear equations in the $d$ unknowns $\mathbf{w}$: the
matrix $\mathbf{X}^\top\mathbf{X}$ has shape $[d, d]$, however many examples there
are, so the system's size depends on the number of features and not on the
number of rows. If $\mathbf{X}^\top\mathbf{X}$ has an inverse (chapter 4), the
solution is $\mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$. That
formula is the standard way to write the answer. It is not the standard way to
compute it: chapter 4 showed that elimination solves the system without
forming the inverse, at about a third of the cost, and with fewer rounding errors.
We solve the system.

For the sensor, with the design matrix of ones and voltages:

<!-- worked: linear_regression.design_matrix_system -->
X has shape [10, 2]: a column of ones and the column of voltages. First three rows: [1, 0.42], [1, 0.95], [1, 1.60].

```text
X^T X  (shape [2, 2])                 X^T y  (shape [2])
[   10.0000    17.4300 ]            [   534.4000 ]
[   17.4300    34.2559 ]            [  1032.3130 ]
```

Elimination (Volume I, chapter 4):

```text
R2 <- R2 - 1.7430 R1     [ 0  3.8754 | 100.8538 ]
back substitution: w1 = 100.8538 / 3.8754 = 26.0240
                   w0 = (534.4000 - 17.4300 * 26.0240) / 10 = 8.0801
```

Closed form of the previous section: w = 26.0240, b = 8.0801.
<!-- /worked -->

The entries of $\mathbf{X}^\top\mathbf{X}$ are the sums the one-dimensional
derivation needed: $N$, $\sum v_n$ and $\sum v_n^2$ (and the right-hand side holds
$\sum y_n$ and $\sum v_n y_n$). The single elimination step produces, in the
corner, $\sum v^2 - N\bar{v}^2 = S_{vv}$, and on the right $S_{vy}$: the same two
numbers, so the same line. The two derivations are one derivation, and the
matrix form is the one that does not change when the features do.

The algorithm, in the book's pseudocode, is three short loops and a call to the
elimination of chapter 4:

```text
function fit_least_squares(X: array<real>[N, d], y: array<real>[N])
        -> array<real>[d]:
    require N >= d and d > 0, "need at least as many examples as features"
    A = zeros(real, [d, d])             # X^T X
    c = zeros(real, [d])                # X^T y
    for i in 0 .. d:
        for j in 0 .. d:
            for n in 0 .. N:
                A[i, j] = A[i, j] + X[n, i] * X[n, j]
        for n in 0 .. N:
            c[i] = c[i] + X[n, i] * y[n]
    return solve(A, c, true)            # partial pivoting, chapter 4; vanishing-pivot test as in the text
```

Forming $\mathbf{A}$ costs $N d^2$ multiply-adds and solving costs about $d^3/3$.
Both numbers are tabulated at the end of the next section, where they matter.

The `solve` of chapter 4 refuses a matrix only when a pivot is exactly zero, the
right test in exact arithmetic. In floating point a singular
$\mathbf{X}^\top\mathbf{X}$ leaves a pivot of rounding noise instead of a zero, so
wherever this chapter's tables need to detect a vanishing pivot they call
`solve(A, c, true)` with this test in place of the exact one. The third argument
`true` is the pivoting rule defined on the pseudocode page (partial pivoting): in
each column the pivot is the entry of largest absolute value at or below the
diagonal of the partly eliminated matrix, brought to the diagonal by a row swap.
The test is applied to each pivot after its swap: the pivot counts as zero when
its absolute value is below $10^{-12}$ times the larger of $1$ and the
largest absolute entry of the matrix $\mathbf{A}$ as it was passed to the solver,
and elimination stops at the first pivot that counts as zero.

### The same answer as a picture: projection

The normal equations have a second reading, geometric, that explains the name.
Think of the $d$ columns of $\mathbf{X}$ as $d$ vectors in $N$-dimensional space
(chapter 3), and the label vector $\mathbf{y}$ as one more vector in that space.
The vectors $\mathbf{X}\mathbf{w}$ obtainable by choosing $\mathbf{w}$ fill a flat
sheet through the origin: the **column space** of $\mathbf{X}$ (a line if $d = 1$, a plane if
$d = 2$). The loss is the squared distance between $\mathbf{y}$ and the point
$\mathbf{X}\mathbf{w}$ of that sheet, so least squares asks for the point on the
sheet nearest to $\mathbf{y}$.

Chapter 3 found the nearest point on a line: the residual, what is left of
$\mathbf{y}$, is at a right angle to the line. The same must hold for a sheet:
the residual vector $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y}$ (the entries $r_n$ above, prediction minus label; its negative is chapter 3's) is at a right angle
to every column of $\mathbf{X}$, that is, $\mathbf{X}^\top\mathbf{r} = \mathbf{0}$. Written out,
$\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) = \mathbf{0}$, which is the
normal equations rearranged. "Normal" is the old word for "at a right angle", and
the equations say that the residual is normal to the column space. So the fitted
vector $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ is the projection of $\mathbf{y}$
onto the column space, the same idea as chapter 3 with a sheet in place of a line.

Numbers make the picture checkable. For the sensor ($N = 10$, so these are
vectors in ten dimensions that nobody can draw), compute the residual vector and take its dot
products:

<!-- worked: linear_regression.projection_check -->
The fitted vector y_hat = X w and the residual r = X w - y (prediction minus label), as vectors in 10 dimensions. The two dot products with the columns of X and the last one are zero up to rounding noise (agreement: size below 1e-9, whatever digits it prints); the other rows are shown to 4 decimals.

| quantity | value |
|---|---:|
| sum of residuals  (ones column . r) | -6.2e-14 |
| sum of v * residual  (voltage column . r) | -6.2e-14 |
| y . y | 31314.4400 |
| y_hat . y_hat | 31182.9587 |
| r . r | 131.4813 |
| y_hat . y_hat + r . r | 31314.4400 |
| y_hat . r | -2.2e-12 |
<!-- /worked -->

The residual is orthogonal to the column of ones (its entries sum to zero) and to
the column of voltages, to rounding; it is also orthogonal to the fitted vector
itself. The right angle gives a theorem of Pythagoras in ten dimensions:
$\lVert\mathbf{y}\rVert^2 = \lVert\hat{\mathbf{y}}\rVert^2 + \lVert\mathbf{r}\rVert^2$,
the last three rows of the table. And $\lVert\mathbf{r}\rVert^2 = 131.48$ is $N$ times
the MSE, so the minimal loss is the squared length of what the model cannot reach.

## When the normal equations break

The recipe needs $\mathbf{X}^\top\mathbf{X}$ to be invertible. That fails in an
instructive way. Suppose the workshop records the voltage twice, once in volts and
once in millivolts, and feeds the model both. The third column of
$\mathbf{X}$ is exactly $1000$ times the second. The model
$w_1 v + w_2 (1000 v)$ depends only on the combination $w_1 + 1000 w_2$, so
infinitely many weights give identical predictions and none can be called the
best. In the language of chapter 4 the columns are dependent, the rank of
$\mathbf{X}$ is smaller than its number of columns, and elimination finds a
pivot that is zero.

A second case is more common and more dangerous: two features that are *almost*
copies of each other, for example the same voltage read by two meters that
disagree by a few millivolts. No pivot is exactly zero, so the solver returns an
answer. But the answer is not trustworthy.

<!-- worked: linear_regression.collinear_features -->
Case A: the third column is the voltage in millivolts, exactly 1000 v.

The millivolt column is computed as 1000.0 * v in floating point. Elimination with partial pivoting on X^T X (the pivot of each column is the entry of largest size at or below the diagonal of the partly eliminated matrix; threshold 3.4e-05 = 1e-12 times the largest absolute entry of X^T X as formed): the first two pivots are 1.743e+04 and -2.223, and the third is rounding noise, below the threshold, so X^T X has no inverse. The rank is the number of eigenvalues of X^T X larger than 1e-9 times the largest eigenvalue: rank 2 of 3. Every (w1, w2) with w1 + 1000 w2 = 26.0240 gives the same predictions, for example (26.0240, 0) and (0, 0.0260240).

Case B: the third column is a second meter reading the same voltage, differing by a few millivolts. Generator: the book's SplitMix64, seed 0xe44a83d6174254b9 (16450105544460620985). For the ten rows in table order, one `rng.normal()` each, the second meter reads v + 0.003 * rng.normal() rounded to 4 decimals. Labels of the second run: row 5 raised by 0.5.

| row n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| meter 1 (V) | 0.42 | 0.95 | 1.60 | 1.63 | 1.80 | 1.90 | 1.96 | 2.19 | 2.30 | 2.68 |
| meter 2 (V) | 0.4172 | 0.9544 | 1.6006 | 1.6271 | 1.7989 | 1.9017 | 1.9598 | 2.1917 | 2.2971 | 2.6812 |

| labels | b | w (meter 1) | w' (meter 2) | prediction at 1.75 V on both |
|---|---:|---:|---:|---:|
| original | 8.05 | 99.0 | -72.9 | 53.62 |
| row 5 + 0.5 | 8.08 | 109.3 | -83.2 | 53.67 |

Eigenvalues of X^T X in case B: 77.512, 1.000, 2.631 x 10^-5; condition number 2.946 x 10^6. Weights are shown to 1 decimal (2 for b); they are the solution of the normal equations by elimination with partial pivoting and the same threshold, accurate to far more digits than shown; the eigenvalues come from a symmetric eigensolver such as `jacobi_eigen`.
<!-- /worked -->

In case B, change one label by half a degree (well inside the thermometer's
noise) and the two weights move by about ten units each, in opposite directions:
the first is large and positive ($99.0$, then $109.3$), the second large and negative ($-72.9$, then $-83.2$), and their sum,
which is what the predictions depend on, stays at $26.1$, near the slope $26.0$ of the
single-feature line. The predictions at $1.75$ V are nearly the same in both
runs; the individual weights are meaningless. The cause is the condition number
of chapter 8. The eigenvalues of $\mathbf{X}^\top\mathbf{X}$ are the squares of
the singular values of $\mathbf{X}$ (chapter 6), so forming $\mathbf{X}^\top\mathbf{X}$
squares the conditioning of the problem: the table's condition number of
millions comes from a design matrix whose own condition number is its square
root, in the thousands. The smallest eigenvalue measures how well the data pin
down the weights along the direction $(w_1, w_2) \propto (1, -1)$, the direction in which the two features cancel, and it is tiny.

Three remedies, in order of how often they are the right one. **Remove the
redundancy** (drop one meter, or combine the two into their average).
**Prefer small weights** when the data cannot decide between large ones, which
is the idea of regularization, developed in the chapter on maximum likelihood.
**Do not form $\mathbf{X}^\top\mathbf{X}$** at all and use a method that works
on $\mathbf{X}$ directly: the decompositions of chapter 6 and the iterative
method of the next section are of this kind.

### Why an iterative method is still needed

Even when $\mathbf{X}^\top\mathbf{X}$ is perfectly healthy, the direct route has
a cost that grows quickly. Forming it takes $N d^2$ multiply-adds and solving it
takes about $d^3/3$; storing it takes $d^2$ numbers. One pass over the data to
compute a gradient takes only $2Nd$ (one product with $\mathbf{X}$ for the
residuals and one with $\mathbf{X}^\top$ for the gradient).

<!-- worked: linear_regression.cost_comparison -->
Multiply-adds for the normal equations (forming X^T X, then elimination) against one gradient-descent step, and the memory of X^T X in f32.

| case | N | d | form X^T X: N d^2 | solve: d^3 / 3 | one descent step: 2 N d | X^T X in GiB |
|---|---:|---:|---:|---:|---:|---:|
| sensor line | 10 | 2 | 40.00 | 2.67 | 40.00 | 1.49 x 10^-8 |
| house prices | 100,000 | 20 | 4.00 x 10^7 | 2,666.67 | 4.00 x 10^6 | 1.49 x 10^-6 |
| tabular model | 1,000,000 | 1,000 | 1.00 x 10^12 | 3.33 x 10^8 | 2.00 x 10^9 | 0.0037 |
| wide model | 1,000,000 | 10,000 | 1.00 x 10^14 | 3.33 x 10^11 | 2.00 x 10^10 | 0.3725 |
| a 7B-parameter model | 1,000,000,000 | 7,000,000,000 | 4.90 x 10^28 | 1.14 x 10^29 | 1.40 x 10^19 | 1.83 x 10^11 |
<!-- /worked -->

For the sensor everything is instantaneous. For a model with a thousand features
and a million rows, forming the matrix costs the same as five hundred gradient
passes. For a model with ten thousand features the matrix alone is $0.37$ GiB
and solving it costs $3.3 \times 10^{11}$. The last row is a language model of
seven billion parameters, where it does not even make sense to ask: $\mathbf{X}^\top\mathbf{X}$ would
hold more numbers than any machine has memory for. (A network is not linear in its
parameters, so there are no normal equations to write down; the row shows only
what the size would do.) What always remains possible is the gradient, whose cost
and memory grow linearly with the number of parameters. The rest of the chapter
is the method that uses only it.

## Gradient descent

Volume I met the idea twice. Chapter 8 stepped from a point against the gradient
and watched the loss fall, and chapter 9 recognised the step as Euler's method
applied to the gradient flow $d\mathbf{w}/dt = -\nabla\mathcal{L}(\mathbf{w})$. Now
we name it and use it. **Gradient descent** is the iteration

$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\,\nabla\mathcal{L}(\mathbf{w}_k),
$$

started from some initial $\mathbf{w}_0$ and repeated. At each step the parameters
move a distance proportional to the slope in the direction in which the loss falls
fastest, since the gradient is the direction of steepest ascent. The positive
number $\eta$ that scales the step is the **learning rate**. It is the one number
of the method that the person choosing the method must pick, and the next two
sections explain how much depends on it. (It is called a rate because, in the
flow picture of chapter 9, it is the step of time; it is not a speed in any
everyday sense.)

For the MSE of a linear model the gradient is the formula just derived, so one
step is

$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \frac{2\eta}{N}\,\mathbf{X}^\top(\mathbf{X}\mathbf{w}_k - \mathbf{y}).
$$

The algorithm, with the shape of every array:

```text
function mse_and_gradient(X: array<real>[N, d], y: array<real>[N],
                          w: array<real>[d]) -> (real, array<real>[d]):
    r = zeros(real, [N])                 # residuals: prediction - label
    loss = 0.0
    for n in 0 .. N:
        p = 0.0
        for j in 0 .. d:
            p = p + X[n, j] * w[j]
        r[n] = p - y[n]
        loss = loss + r[n] * r[n]
    g = zeros(real, [d])
    for j in 0 .. d:
        for n in 0 .. N:
            g[j] = g[j] + X[n, j] * r[n]
        g[j] = 2.0 * g[j] / N
    return (loss / N, g)

function gradient_descent(X: array<real>[N, d], y: array<real>[N],
                          w0: array<real>[d], eta: real,
                          max_steps: int, tol: real) -> array<real>[d]:
    require N > 0 and d > 0, "need data"
    require eta > 0.0, "the learning rate must be positive"
    w = copy(w0)
    for k in 0 .. max_steps:
        (loss, g) = mse_and_gradient(X, y, w)
        norm = 0.0
        for j in 0 .. d:
            norm = norm + g[j] * g[j]
        if sqrt(norm) < tol:             # stopping rule 1, discussed below
            return w
        for j in 0 .. d:
            w[j] = w[j] - eta * g[j]
    return w
```

Each step costs one pass over the data, $2Nd$ multiply-adds, and needs
only $O(d)$ memory beyond the data. Nothing in it asks for an inverse, and nothing
in it is specific to a line: replace the model and the gradient routine, and the
loop is the loop that trains a network (Volume III).

### Three learning rates on the sensor

Start where a person would: with the constant guess that the temperature is
$30$ degrees at every voltage, that is $(w, b) = (0, 30)$. Run the loop on
the ten rows with three learning rates and print the weights and the loss as it
goes. (In the vector convention of this chapter the start is $\mathbf{w} = (w_0, w_1) = (30, 0)$; the tables print the pair as (slope, intercept) = $(w, b)$ = $(w_1, w_0)$.) First the curvature of the bowl, which the next section turns into a rule. The eigenvalues of the symmetric matrix $\mathbf{H}$ below, and of the larger matrices of later tables, come from `jacobi_eigen(H, 20)`, the symmetric eigenproblem routine of Volume I, chapter 6 (for a $2 \times 2$ matrix the trace and the determinant give them as well).

<!-- worked: linear_regression.curvature -->
| quantity | value |
|---|---:|
| Hessian (2/N) X^T X | [[2.0000, 3.4860], [3.4860, 6.8512]] |
| eigenvalues | 8.6724 and 0.1787 |
| condition number | 48.52 |
| largest stable learning rate 2 / lambda_max | 0.2306 |
| best learning rate eta* = 2 / (lambda_max + lambda_min) | 0.225959 |
| loss at the start (w, b) = (0, 30) minus the minimum | 811.9 |
<!-- /worked -->

<!-- worked: linear_regression.gd_three_rates -->
Gradient descent from (w, b) = (0, 30), the constant guess of 30 degrees, on the ten sensor rows, unscaled voltage. Row k shows the weights after k updates w <- w - eta * gradient (k = 0 is the start), with learning rates 0.01, 0.2 and 0.24, and the MSE there. Weights and MSE values are shown to 2 decimals (thousands separated by commas), and numbers of 10^6 or more in size as a mantissa with 3 significant digits times a power of ten. The eta = 0.24 column diverges (it grows by a factor of about 1.08 per step), so its late rows agree only in their leading digits. The minimum is at (26.02, 8.08) with MSE 13.1481.

| step k | (w, b), eta=0.01 | MSE | (w, b), eta=0.2 | MSE | (w, b), eta=0.24 | MSE |
|---|---:|---:|---:|---:|---:|---:|
| 0 | (0.00, 30.00) | 825.04 | (0.00, 30.00) | 825.04 | (0.00, 30.00) | 825.04 |
| 1 | (1.02, 30.47) | 704.71 | (20.38, 39.38) | 485.71 | (24.45, 41.25) | 940.14 |
| 2 | (1.95, 30.89) | 604.29 | (6.30, 30.80) | 300.22 | (-0.72, 26.64) | 1,076.61 |
| 5 | (4.30, 31.93) | 392.12 | (16.32, 32.62) | 107.74 | (31.25, 39.33) | 1,652.03 |
| 10 | (7.08, 33.07) | 216.46 | (15.37, 27.19) | 57.42 | (-8.41, 13.00) | 3,508.90 |
| 20 | (10.09, 34.03) | 114.80 | (18.96, 21.54) | 33.80 | (-34.78, -8.91) | 16,568.60 |
| 50 | (12.57, 33.53) | 87.28 | (23.66, 12.60) | 15.47 | (-548.01, -287.86) | 1.81 x 10^6 |
| 100 | (13.83, 31.41) | 75.07 | (25.64, 8.81) | 13.21 | (-28,596.63, -14,945.33) | 4.52 x 10^9 |
| 200 | (15.83, 27.59) | 56.44 | (26.01, 8.10) | 13.15 | (-7.16 x 10^7, -3.74 x 10^7) | 2.83 x 10^16 |
<!-- /worked -->

The three columns behave in three different ways.

**Too small ($\eta = 0.01$).** The loss falls at every step and never overshoots,
but after $200$ steps it is still at $56.44$ against a minimum of $13.15$, and
the slope has reached $15.83$ of its final $26.02$. The path is safe and
inefficient, which for a model with millions of parameters means days of training
instead of hours.

**Good ($\eta = 0.2$).** The loss falls fast at first and then settles; by step
$100$ it is within $0.07$ of the minimum, and by step $200$ the weights are within $0.02$ of
the exact line. Early on the path overshoots (the slope
is $20.38$ after one step and $6.30$ after two), a sign that the step is large
for one of the two directions of the bowl.

**Too large ($\eta = 0.24$).** The loss rises. It is $940$ after one step, $3{,}509$
after ten, and about $4.5 \times 10^9$ after a hundred. The slope swings from side
to side with growing amplitude and the weights run away. This is not an accident
of this data set: the next section shows it is the unavoidable behaviour for any
learning rate above a limit that we can compute in advance. A loss that
rises is the signal.

The picture draws the same three runs on the contours of the loss: the paths
across the long valley. The $\eta = 0.01$ path creeps. The $\eta = 0.2$ path
zigzags across the valley with swings that shrink while it moves along the valley
floor. The $\eta = 0.24$ path (only seven steps shown) swings across the valley
by more at each step and leaves the frame.

![Contours of the mean squared error of the sensor line in the plane of slope w and intercept b, drawn as long narrow ellipses sloping down to the right, with the minimum at (26.02, 8.08). Three paths start at (0, 30): eta 0.01 moves slowly along the valley, eta 0.2 zigzags across the valley while advancing to the minimum, and eta 0.24 swings across the valley with growing amplitude and leaves the plot.](img/descent-paths.svg)

## How large a step: the stability limit

Why did $0.2$ work and $0.24$ not? The loss of a linear model is exactly a
quadratic, so the question has an exact answer, the same one as in chapters 8 and
9 with one new ingredient: the Hessian of the MSE is known.

The gradient is $\frac{2}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ and the
minimiser $\mathbf{w}^\star$ satisfies the normal equations
$\mathbf{X}^\top\mathbf{X}\mathbf{w}^\star = \mathbf{X}^\top\mathbf{y}$. Subtract:

$$
\nabla\,\mathrm{MSE}(\mathbf{w}) = \frac{2}{N}\mathbf{X}^\top\mathbf{X}(\mathbf{w} - \mathbf{w}^\star) = \mathbf{H}\,\mathbf{e},
\qquad
\mathbf{H} = \frac{2}{N}\mathbf{X}^\top\mathbf{X},
\quad
\mathbf{e} = \mathbf{w} - \mathbf{w}^\star .
$$

$\mathbf{H}$ is the Hessian of chapter 8 (it is the matrix of second derivatives
we computed for the line, in matrix form), and it does not depend on $\mathbf{w}$.
One step of descent therefore acts on the error as

$$
\mathbf{e}_{k+1} = \mathbf{e}_k - \eta\mathbf{H}\mathbf{e}_k = (\mathbf{I} - \eta\mathbf{H})\,\mathbf{e}_k .
$$

$\mathbf{H}$ is symmetric, so by chapter 6 it has orthogonal eigenvectors
$\mathbf{u}_i$ with eigenvalues $\lambda_i$ (positive, when $\mathbf{X}^\top\mathbf{X}$
is invertible). Write the error as $\mathbf{e} = \sum_i \xi_i\mathbf{u}_i$. Then
$(\mathbf{I} - \eta\mathbf{H})\mathbf{e} = \sum_i (1 - \eta\lambda_i)\xi_i\mathbf{u}_i$: each
component of the error is multiplied, independently of the others, by
$1 - \eta\lambda_i$ at every step. After $k$ steps it is
$(1 - \eta\lambda_i)^k \xi_i$.

**The stability bound.** For the error to vanish along every direction, every
factor must have absolute value below $1$: $\lvert 1 - \eta\lambda_i\rvert < 1$,
which for $\eta > 0$ means $\eta\lambda_i < 2$. The strictest is the largest
eigenvalue:

$$
\eta < \frac{2}{\lambda_{\max}} .
$$

This is the stability limit of Euler's method from chapter 9, with the largest
curvature of the bowl playing the part of the rate $\lambda$. Above it the factor
$1 - \eta\lambda_{\max}$ is below $-1$, and the component of the error along
the steepest direction is multiplied by more than $1$ in size and changes sign at
every step: it alternates and grows, which is the swinging of the $\eta = 0.24$ run.

Everything is in the table for our bowl: $\lambda_{\max} = 8.6724$, so the limit is
$2/8.6724 = 0.2306$, and $0.24$ lies above it and $0.2$ below. The factors for the
two eigen-directions, and the number of steps that the run needs to bring the loss
to within $0.001$ of its minimum (the header of the table defines the count exactly):

<!-- worked: linear_regression.stability_table -->
Curvatures lambda_max = 8.6724, lambda_min = 0.1787; 2 / lambda_max = 0.2306. Steps k: the smallest k with loss(w_k) - minimum < 0.001, from (w, b) = (0, 30); a run whose loss exceeds 1e12 above the minimum is reported as diverging, and one that needs more than 200,000 steps as such. Bound: ceil(ln(0.001 / gap_0) / (2 ln(worst factor))) with gap_0 = 811.9 the starting loss minus the minimum. The row marked eta* uses the unrounded best rate eta* = 2 / (lambda_max + lambda_min) = 0.225959; every other row uses the rate as printed.

| learning rate | |1 - eta lambda_max| | |1 - eta lambda_min| | worst factor | bound from the worst factor | steps measured |
|---|---:|---:|---:|---:|---:|
| 0.0100 | 0.9133 | 0.9982 | 0.9982 | 3,803 | 3,184 |
| 0.0500 | 0.5664 | 0.9911 | 0.9911 | 758 | 635 |
| 0.1000 | 0.1328 | 0.9821 | 0.9821 | 378 | 316 |
| 0.2000 | 0.7345 | 0.9643 | 0.9643 | 187 | 157 |
| 0.225959 (eta*) | 0.9596 | 0.9596 | 0.9596 | 166 | 166 |
| 0.2300 | 0.9947 | 0.9589 | 0.9947 | 1,271 | 1,260 |
| 0.2400 | 1.0814 | 0.9571 | 1.0814 | none | diverges |
<!-- /worked -->

Read the table by columns. The factor along the steep direction,
$\lvert 1 - \eta\lambda_{\max}\rvert$, is small for moderate $\eta$ and rises to $1$
at the limit and beyond it. The factor along the flat direction,
$\lvert 1 - \eta\lambda_{\min}\rvert$, is within $0.05$ of $1$ for *every* allowed
learning rate, because $\lambda_{\min} = 0.179$ is tiny: no allowed step size
makes progress fast along the valley floor. The worst of the two factors sets the
speed. It is smallest, and the descent fastest, when the two are equal in
size: $1 - \eta\lambda_{\min} = -(1 - \eta\lambda_{\max})$, which gives

$$
\eta^\star = \frac{2}{\lambda_{\max} + \lambda_{\min}},
\qquad
\text{worst factor} = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\kappa - 1}{\kappa + 1},
$$

where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number of chapter
8. For the sensor, $\kappa = 48.5$ and the best rate is $\eta^\star = 0.225959$ (the
table's row marked $\eta^\star$ uses it unrounded; the rounded $0.2260$ would give the factors
$0.9600$ and $0.9596$ and one step more), with a factor of
$0.9596$ per step: even with the best learning rate the error shrinks only by
$4\%$ per step, because the bowl is $49$ times more curved in one direction than
in the other. The last column of the table also shows how narrow the useful range
is. At $0.2300$, just under the limit, the steep direction's factor is $0.9947$ and the run needs
$1{,}260$ steps; at $0.2400$ it diverges. Past $\eta^\star$ the steep direction oscillates
and sets the pace. In practice one picks a rate somewhat below the limit rather
than the optimum, which is not known without the Hessian; for a model with millions of parameters nobody
knows it.

## Convergence

A sequence of parameter vectors **converges** when it approaches a limit: here,
when $\mathbf{w}_k \to \mathbf{w}^\star$ and the loss approaches its minimum. For
the quadratic loss of this chapter the analysis above says exactly how. The
excess of the loss over its minimum is, since a quadratic equals its second-order
expansion,

$$
\mathcal{L}(\mathbf{w}) - \mathcal{L}(\mathbf{w}^\star) = \tfrac{1}{2}\,\mathbf{e}^\top\mathbf{H}\,\mathbf{e} = \tfrac{1}{2}\sum_i \lambda_i \xi_i^2 ,
$$

so each component of the excess shrinks by $(1 - \eta\lambda_i)^2$ per step.
Late in the run the fast components have died, and the slowest one dominates: the
excess falls by the same ratio $\gamma^2$ at every step, with
$\gamma = \max_i\lvert 1 - \eta\lambda_i\rvert$. A sequence that shrinks by a fixed
factor per step is called **geometric** (in optimisation, *linear*, because its
logarithm is a straight line against the step number, as the figure below shows).

<!-- worked: linear_regression.convergence_rate -->
Gradient descent at learning rate 0.2 from (w, b) = (0, 30). With gap_k the loss at step k minus the minimum, the table lists gap_k and gap_(k+1) / gap_k. The factors are |1 - eta lambda| = 0.7345 and 0.9643; the predicted late ratio is the square of the larger, 0.9298.

| step k | loss - minimum | ratio of the next gap to this one |
|---|---:|---:|
| 0 | 8.119e+02 | 0.5820 |
| 1 | 4.726e+02 | 0.6075 |
| 2 | 2.871e+02 | 0.6436 |
| 5 | 9.459e+01 | 0.7934 |
| 10 | 4.427e+01 | 0.9165 |
| 50 | 2.324e+00 | 0.9298 |
| 100 | 6.099e-02 | 0.9298 |
| 300 | 2.892e-08 | 0.9298 |
<!-- /worked -->

At $\eta = 0.2$ the ratio of consecutive excesses starts at $0.58$, while the fast
direction (factor $0.7345$, squared $0.5395$) is still alive, and by step $50$ it
has risen to $0.9298$, the square of the slow factor ($0.9643$), and stays there, to four digits. The theory
predicts the number of steps: the excess is at most $\gamma^{2k}$ times its starting value,
so it is below a tolerance $\tau$ once
$k \ge \ln(\tau/\text{excess}_0)\,/\,(2\ln\gamma)$. That is the column "bound from the worst factor" of the
stability table, and the measured steps are always below it (they coincide when
the starting error lies along the slow direction). The count grows with $\kappa$:
for $\gamma = (\kappa - 1)/(\kappa + 1) \approx 1 - 2/\kappa$ the step count is
proportional to $\kappa$.

### When to stop

A program has to decide to stop, since the iteration never reaches the minimum
exactly. Three stopping rules are in use, usually together.

- **A step limit.** Stop after a fixed number of steps. It guarantees the program ends and is the only rule that does not need the loss to behave.
- **A small gradient.** Stop when $\lVert\nabla\mathcal{L}\rVert < \tau_g$. At the minimum the gradient is zero, so a small gradient means we are near a critical point. For our quadratic we can say how near: $\lVert\nabla\mathcal{L}\rVert = \lVert\mathbf{H}\mathbf{e}\rVert \ge \lambda_{\min}\lVert\mathbf{e}\rVert$, so $\lVert\mathbf{e}\rVert \le \lVert\nabla\mathcal{L}\rVert/\lambda_{\min}$.
- **A small change in the loss.** Stop when $\lvert\mathcal{L}_{k} - \mathcal{L}_{k+1}\rvert < \tau_\ell$. This is the easiest rule to write, and the one that can mislead.

Run the last two on the same problem with three learning rates. A step count in the table is the number of updates made when the rule fires, the `k` at which `gradient_descent` above returns.

<!-- worked: linear_regression.stopping_criteria -->
Two stopping rules from (w, b) = (0, 30). Loss-change rule: stop at the first k >= 1 for which |loss(w_(k-1)) - loss(w_k)| < 1e-6. Gradient rule: stop at the first k >= 0 for which the Euclidean norm of the gradient at w_k is below 1e-3. The step count is k, the number of updates made. For the second, the distance to the minimum is at most 1e-3 / lambda_min = 0.0056.

| learning rate | loss-change rule: steps | loss - minimum | distance to minimum | gradient rule: steps | loss - minimum | distance to minimum |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 216 | 1.31 x 10^-5 | 0.0121 | 238 | 2.64 x 10^-6 | 0.0054 |
| 0.02 | 1,866 | 1.39 x 10^-4 | 0.0394 | 2,412 | 2.78 x 10^-6 | 0.0056 |
| 0.002 | 15,462 | 0.0014 | 0.1250 | 24,151 | 2.80 x 10^-6 | 0.0056 |
<!-- /worked -->

The gradient rule stops with the loss under $3 \times 10^{-6}$ above its minimum and the weights within
$0.0056$ of the truth for all three learning rates, as the bound promised, at
the price of more steps. The loss-change rule is cheaper and its quality depends on
$\eta$: at $\eta = 0.2$ the final error is $0.012$ away, and at $\eta = 0.002$ it is $0.125$
away with a loss $0.0014$ above its minimum, about five hundred times worse than the gradient rule
for the same nominal tolerance. The reason is in the slow direction: there the
loss falls by a fraction $2\eta\lambda_{\min}$ of the remaining excess each step, so
a per-step change of $\tau_\ell$ means an excess of about
$\tau_\ell/(2\eta\lambda_{\min})$: for $\eta = 0.002$ that is about $1400$ times
the tolerance. A small step size makes the loss change small *because the steps are small*, not because the
minimum is near. The loss-change rule is to be trusted only when the
learning rate is large enough for the loss change to mean something, and the
gradient rule is safer; but either one is only as good as the conditioning, and
no stopping rule makes a badly conditioned problem fast.

## Feature scaling

The condition number $\kappa = 48.5$ of the sensor line came from the data, not
from the model, and the data can be rewritten. Two things are wrong with the
design matrix $[1, v]$.

First, the columns are not at a right angle: the voltages are all positive and
centred near $1.7$, so the column of voltages is largely a copy of the column of
ones. (Their cosine, chapter 3, is $\sum v/\sqrt{N\sum v^2} = 0.942$.) A change
in the intercept and a change in the slope can nearly cancel in their effect on
the predictions, which is the long valley: the near-collinearity of the previous
section, in a mild form. **Centring** the feature (subtracting its mean) makes the column
of voltages sum to zero and so orthogonal to the column of ones.

Second, the scales are different: the column of ones has entries $1$ and the
voltages entries near $2$. Had the workshop logged millivolts, the entries would
be in the thousands, and the curvature along the slope would be a million times
the one along the intercept. Dividing by the standard deviation puts every
feature on the same scale.

Together these are the z-score of chapter 12, $z = (v - \bar{v})/s$, with $s$ the
standard deviation of the voltages (divisor $N - 1$). A feature rewritten this way
is called **standardised** (or z-scored), and the standardised features are what
is fitted. With the design matrix $[1, z]$ the matrix
$\mathbf{X}^\top\mathbf{X}$ is diagonal (its entries are $N$ and $N - 1$), so
$\mathbf{H} = \mathrm{diag}(2,\ 2(N-1)/N)$ and its two curvatures are almost equal.

<!-- worked: linear_regression.scaling_conditioning -->
Same ten rows, same model, three ways of writing the voltage (z-score z = (v - mean) / s with mean 1.743 and standard deviation s = 0.656, divisor N - 1 = 9; millivolts are 1000 v). Best learning rate eta* = 2 / (lambda_max + lambda_min), used unrounded (shown to 6 significant digits), and steps k, the smallest k with loss(w_k) - minimum < 0.001, from (w, b) = (0, 30) in the units of the column (for z-scores, the weights of the column of ones and of z are started at 30 and 0); a run is cut off at 200,000 steps. The predicted bound is ceil(ln(0.001 / gap_0) / (2 ln((kappa - 1) / (kappa + 1)))), shown with 3 significant digits when it is 10^6 or more.

| feature | lambda_max | lambda_min | condition number | best eta | steps measured | steps predicted (bound) |
|---|---:|---:|---:|---:|---:|---:|
| volts, raw | 8.672 | 0.1787 | 48.518 | 0.225959 | 166 | 166 |
| millivolts, raw | 6.851 x 10^6 | 0.2263 | 3.028 x 10^7 | 2.91920 x 10^-7 | > 200,000 | 1.03 x 10^8 |
| z-scored | 2.000 | 1.800 | 1.111 | 0.526316 | 3 | 3 |
<!-- /worked -->

Volts, millivolts and z-scores describe the *same ten rows*, and the same line
fits all three. Their conditioning could hardly differ more. In millivolts the
condition number is $3.0 \times 10^{7}$, the stable learning rate is below
$10^{-6}$, and the run is not within tolerance after $200{,}000$ steps (the bound says
$1.0 \times 10^{8}$). In volts it is $48.5$ and takes $166$ steps. After z-scoring it is $1.11$ and the loss is
within tolerance in three. The comparison of the two cases that can be run is in the picture.

![Base-ten logarithm of the loss above its minimum against iteration number, for gradient descent at its best learning rate from the same start. The raw-volts line falls slowly and crosses the line marking a gap of 0.001 near step 166; the z-scored line drops almost vertically to the floor of rounding noise within a few steps.](img/scaling-convergence.svg)

Nothing was lost in the change: a fit in z-score units can be mapped back.
Since $\hat{y} = w_0 + w_1(v - \bar{v})/s$, the slope in volts is $w_1/s$ and the
intercept $w_0 - w_1\bar{v}/s$.

<!-- worked: linear_regression.scaled_fit_matches -->
| fit | slope | intercept | prediction at 1.75 V |
|---|---:|---:|---:|
| z-scored: (w1, w0) = (17.0770, 53.4400) |  |  | 53.6222 |
| mapped back: w1 / s, w0 - w1 m / s | 26.0240 | 8.0801 | 53.6222 |
| raw volts, closed form | 26.0240 | 8.0801 | 53.6222 |
<!-- /worked -->

One rule has to be kept for later chapters: **the mean and standard deviation
come from the training rows and are then fixed**. A new sensor reading is scaled
with the same $\bar{v}$ and $s$, not with its own, since a single reading has no
meaningful standard deviation and a statistic computed on the rows a model is
later judged on leaks information about them. Chapter "Generalization,
Overfitting and Honest Evaluation" names this.

## Polynomial features: linear regression in disguise

The word *linear* in linear regression is about the parameters, not the
features. The model $\hat{y} = w_0 + w_1 v + w_2 v^2$ is a parabola in $v$, but it
is a weighted sum of fixed features $1$, $v$ and $v^2$ and so is exactly the model
$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ with the design matrix whose columns
are $1$, $v$ and $v^2$. Everything in this chapter applies unchanged: the same
normal equations, the same gradient, the same stability bound. Chapter 1's
parabola, found there by a grid search of $241{,}865$ settings, comes out of one
$3 \times 3$ elimination.

<!-- worked: linear_regression.polynomial_fit -->
Polynomial features of the voltage (columns 1, v, v^2, ...), fitted by the normal equations on the ten rows. Weights are (w0, w1, ...) from the constant term up. The fresh rows are the 2000 rows of the source of the previous chapter, drawn from the book's SplitMix64 with seed 0x0fe67df3a852f39e (1145741640666903454). Each row is made by two draws in this order: v = 0.4 + 2.4 * rng.uniform(), then z = rng.normal(); y = -6 + 50 v - 8 v^2 + 1.2 z computed from the unrounded v; then v is rounded to 2 decimals and y to 1. The fits are not repeated on them: the weights found on the ten rows are scored on the fresh rows. MSE is shown to 3 decimals. cond(X^T X) is the largest eigenvalue over the smallest; the last column builds the columns 1, z, z^2, ... from the z-score z = (v - mean) / s of v (mean and standard deviation s of the ten voltages, divisor 9) instead of from v.

| model | weights | MSE, 10 known rows | MSE, 2000 fresh rows | cond(X^T X), raw v | cond(X^T X), z-scored v |
|---|---:|---:|---:|---:|---:|
| line | (8.08, 26.02) | 13.148 | 14.084 | 48.52 | 1.11 |
| parabola | (-6.41, 49.52, -7.72) | 0.599 | 1.733 | 1,444.73 | 7.97 |
| cubic | (-4.31, 43.48, -3.35, -0.92) | 0.544 | 1.965 | 115,038.33 | 83.20 |
<!-- /worked -->

The parabola is $(w_0, w_1, w_2) = (-6.41, 49.52, -7.72)$, close to the grid's
$(-6, 50, -8)$ and it cut the error on the fresh rows from $14.08$ for the line to
$1.73$. The cubic fits the ten known rows a little better than the parabola
($0.544$ against $0.599$), as any larger family must, and it does *worse* on the
fresh rows ($1.97$ against $1.73$): the extra freedom has begun to fit the noise,
the pattern of chapter 1 that the generalization chapter develops. The last two
columns show the other lesson of the section: powers of $v$ are strongly correlated, and the
condition number of $\mathbf{X}^\top\mathbf{X}$ grows from $49$ for the line to
$1.4 \times 10^3$ for the parabola and $1.2 \times 10^5$ for the cubic. Applying
the z-score to $v$ before taking its powers brings them to $1.1$, $8.0$ and $83$.
Feature scaling is not a courtesy: it decides whether the method works.

Fitting a *power law* is another disguise, and the one brain's scaling studies
use. If the loss of a model falls with its size $n_{\mathrm{p}}$ (its number of parameters) as
$\mathcal{L}(n_{\mathrm{p}}) = \mathcal{L}_\infty + \zeta\, n_{\mathrm{p}}^{-\alpha}$ (a floor $\mathcal{L}_\infty$ plus a power
with scale $\zeta$ and exponent $\alpha$), then
$\log(\mathcal{L} - \mathcal{L}_\infty) = \log\zeta - \alpha\log n_{\mathrm{p}}$: a straight line in the logarithms of both
quantities, with slope $-\alpha$. For a candidate floor the problem is a
line fit. Seven noisy points are generated here from $\mathcal{L} = 0.5 + 10\,n_{\mathrm{p}}^{-0.3}$ with
$1\%$ noise (generator, seed and draw order in the header of the table), and a line is fitted for four candidate floors.
To compare the four lines we need one number for how straight the points are in the logarithms. The line leaves a
residual on each of the seven points (in log space); $\mathrm{SS}_{\mathrm{res}}$ is the sum of their squares. The
best a model with no slope could do is the mean of the log-losses, and $\mathrm{SS}_{\mathrm{tot}}$ is the sum of the
squared deviations from it. The ratio
$R^2 = 1 - \mathrm{SS}_{\mathrm{res}}/\mathrm{SS}_{\mathrm{tot}}$ is $1$ for a line through every point and
$0$ for a line no better than that mean (the metrics chapter treats it properly):

<!-- worked: linear_regression.power_law_log_fit -->
Seven (size n_p, loss) points generated from loss = 0.5 + 10 n_p^-0.3 with multiplicative noise of one percent. Generator: the book's SplitMix64, seed 0x1377ca06b334a36f (1402811939074384751). For the sizes in the order listed, one `rng.normal()` each, the loss is (0.5 + 10 n_p^-0.3) * (1 + 0.01 * rng.normal()), shown to 4 decimals.

| n_p | loss |
|---|---:|
| 1,000 | 1.7667 |
| 3,000 | 1.4174 |
| 10,000 | 1.1280 |
| 30,000 | 0.9460 |
| 100,000 | 0.8212 |
| 300,000 | 0.7278 |
| 1,000,000 | 0.6573 |

For each candidate floor, an ordinary least-squares line (closed form of the chapter, natural logarithms, unrounded losses) of log(loss - floor) against log n_p (slope = -alpha, intercept = log zeta, so the scale is zeta = exp(intercept)); R^2 = 1 - SS_res / SS_tot of that line, with SS_res the sum of the squared differences between the seven values of log(loss - floor) and the line, and SS_tot the sum of squares of log(loss - floor) about its mean:

| candidate floor | alpha | scale zeta | R^2 of the log-space fit |
|---|---:|---:|---:|
| 0.00 | 0.1431 | 4.423 | 0.97663 |
| 0.25 | 0.1910 | 5.326 | 0.98824 |
| 0.50 | 0.3014 | 10.153 | 0.99978 |
| 0.60 | 0.4228 | 24.634 | 0.98411 |
<!-- /worked -->

Only a floor near the truth, $\mathcal{L}_\infty = 0.5$, gives a straight line in the logarithms:
$R^2$ (the share of the variance of the log-losses that the line explains) is $0.99978$ there and lower for the others,
and the fit recovers $\alpha = 0.301$ (truth $0.3$) and $\zeta = 10.2$ (truth $10$). The
loss is no longer the loss in degrees: least squares in log space minimises the
squares of *relative* errors, so every size counts equally rather than the
largest losses dominating. That is a modelling decision, made by the transformation, and
is worth making deliberately.

## Making it work

**Include the column of ones.** A line through the origin is a different model
and usually a wrong one. The most common bug in a hand-written regression is
a missing intercept column, and the residuals' failing to sum to zero is the
symptom: a least-squares fit with an intercept always has residuals that sum to zero.

**Solve; do not invert.** Use elimination on the normal equations (chapter 4).
Check that the solver reports a vanishing pivot instead of returning numbers, and
check the condition number before trusting individual weights.

**Check the gradient.** If you write the gradient by hand for a new model,
compare it with finite differences on a small case, as the table above did. A
descent with a wrong gradient still runs and still lowers the loss for a while,
and then stalls somewhere meaningless.

**Watch the loss, not the weights.** A rising loss means the learning rate is
above the limit (or, for models that are not quadratic, that the step overshoots
a feature of the surface); halve the rate. A loss that falls but very slowly means
bad conditioning or a rate that is too small: look at the scales of the features
before touching the rate.

**Scale the features, then start with the rate $1/\lambda_{\max}$.** For z-scored
features of a linear model, $\lambda_{\max}$ is of order $1$ times a small
constant, so a rate of $0.1$ to $0.5$ is a safe first try. For larger models the
reasoning does not transfer unchanged, and the next volume's optimizers chapter
asks what to do instead.

**Common mistakes.**

- Taking a stopping rule on loss change as a guarantee of being near the minimum.
- Comparing the loss of a fit in z-score units with one in raw units as if the weights were comparable: the weights change with the scale; the predictions and the loss do not.
- Reading individual weights as importance when features are correlated, as the collinearity table showed.
- Using the learning rate that worked on one scale of data on another: millions of times too large, or too small, with no change to the code.

## Where this appears in brain

brain is mostly about models that no formula can solve, so exact least squares
is rare in it, but the closed form turns up wherever a small fit is needed
beside a big model, and the descent step lives at the heart of its optimiser.

**Fitting a scaling law in log space.** `crates/bench/src/scaling.rs` trains the
same task at several model sizes, measures the final loss at each size and fits
a floor plus a power of the size, the $\mathcal{L}_\infty + \zeta\, n_{\mathrm{p}}^{-\alpha}$ of this chapter (its code calls the floor
`E`, the scale `A` and the size `N`), to the points. Its `fit_power_law` does what the
previous section did: it searches 64 candidate floors $E$ between zero and the
smallest loss observed, and for each runs an ordinary least-squares line in the
logarithms, `ols`, which computes the closed form of this chapter (slope
the cross sum over the sum of squares, $S_{vy}/S_{vv}$ with generic names, intercept through the means) and keeps the floor with the best
$R^2$. The reasoning is in its doc comment: with only a handful of points a
three-parameter fit optimised jointly is unstable, so one parameter is fixed by
a coarse search and the other two are found exactly. The module's test
`fits_a_known_power_law` builds the loss $0.5 + 10\,n_{\mathrm{p}}^{-0.3}$ from four noiseless
points and checks that $\alpha$ is recovered to within $0.05$. Commit `f6069a024`
(24 June 2026) introduced the harness. Its default sweep trains three model sizes,
and that is worth a reader's caution drawn from this chapter's arithmetic: a fit
with a free floor, a scale and an exponent has three numbers for three points, so
a score near $1$ on the default sweep says the fit can pass through the points,
not that the loss follows a power law. The module's own table of fitted laws is only evidence
when it is given more sizes than free numbers.

**The same closed form with an error bar.** `crates/bench/src/metrics.rs`
holds `ols_slope_ci`, which fits a line by the same two formulas and adds the
standard error of the slope, $\sqrt{(\mathrm{SS}_{\mathrm{res}}/(n-2))/S_{vv}}$ (in this chapter's names), and a $95\%$
interval from a small table of $t$ values (chapter 12). It came with commit
`05446791a` (6 September 2026), the multi-cycle continual-learning harness, and
`crates/rl/src/continual.rs` calls it on a twelve-point series of a plasticity
measure. Its doc comment states the lesson of this chapter's data-size theme: a
twelve-point, single-seed series cannot support a significance test, so the
interval is a diagnostic that is reported and never used to pass or fail a model,
and an interval containing zero means "no trend distinguishable at this
resolution", which is not the claim that there is no trend.

**Normal equations in a forecasting baseline.** The autoregressive baseline in
`crates/fcbench/src/baselines.rs` predicts each value of a series from its $p$
predecessors with a linear model. `fit_ar` builds the design matrix implicitly (a
column of ones and $p$ lagged copies of the series), accumulates $\mathbf{X}^\top\mathbf{X}$
and $\mathbf{X}^\top\mathbf{y}$ row by row, as the pseudocode above does, and
calls `solve` in `crates/fcbench/src/util.rs`, a Gaussian elimination with row
swaps that returns `None` when a pivot vanishes. The caller treats `None` as the
rank-deficient case of this chapter: for a constant (differenced) series, every lag
column is equal to the column of ones, the matrix is singular, and the baseline
falls back to a random walk instead of returning weights. The test
`solve_reports_singular` feeds it the matrix of ones. The pivot test is an
absolute threshold of $10^{-12}$ on `f32` numbers, so it catches exact
singularity and not the near-singular case of the two-meter table: the matrix is
accumulated in single precision, where the squared conditioning of the normal
equations is the more serious risk. Commit `5ce448795` (21 July 2026) added the
baseline with the forecasting subsystem.

**The plain gradient step.** The update $\mathbf{w} \leftarrow \mathbf{w} - \eta\,\mathbf{g}$
is brain's `Rule::SgdMomentum` in `crates/optim/src/rule.rs` with momentum zero:
the code path does `w -= lr * g` when weight decay is also zero, applied to
the gradient of a batch's loss (the gradient of an average over a few rows is a
sample mean of chapter 12, an estimate of the full gradient that costs a few rows
instead of $N$; that variant is stochastic gradient descent, and the optimizers
chapter develops it and the momentum and adaptive rules that surround it). The notable
test in `crates/optim/src/streamed.rs`,
`master_keeps_updates_a_low_precision_value_would_lose`, is a lesson about the
learning rate that this chapter's formulas alone do not give: it runs a hundred
plain steps with $\eta = 10^{-4}$ and gradient $1$ on a weight of $1.0$ stored in
bfloat16. The spacing of bfloat16 numbers near $1$ is $7.8 \times 10^{-3}$, so
each step of $10^{-4}$ rounds away and the stored weight never moves; with an
`f32` master copy that accumulates the steps, the weight reaches $0.99$ as the
arithmetic of this chapter says it should. A step size $\eta g$ far below the
spacing of the stored number is a step that does not happen. Commit `b7d283e1b`
(20 October 2026) introduced the streamed optimiser state with these rules.

## Check your understanding

Answers are at the end of the chapter.

1. **Explain.** A colleague says that the least-squares line cannot be wrong because it is "the best line". In what sense is it best, and what would make a different line better?
2. **Predict.** Model A makes errors of $3, 3, 3, 3$ on four rows and model B makes errors of $0, 0, 0, 12$. Which has the smaller average absolute error and which the smaller MSE?
3. **Find the error.** "The normal equations need $(\mathbf{X}^\top\mathbf{X})^{-1}$, so the first step of a regression program is to compute that inverse." Why is that not the right way to write the program, even if the inverse exists?
4. **Explain.** After fitting a line with an intercept, the residuals sum to $0.0000$. Which of the equations of the chapter says this, and what would it mean if they summed to $3.2$?
5. **Predict.** A least-squares problem has $\lambda_{\max} = 10$ and $\lambda_{\min} = 0.1$. What learning rates are stable, which is best, and what is the worst factor per step at the best rate?
6. **Find the error.** "The loss fell for 40 steps and then began to rise, so gradient descent has found the minimum and is now moving away." Give a more likely explanation and the first thing to try.
7. **Explain.** A model is trained with a stopping rule "stop when the loss changes by less than $10^{-6}$". The learning rate is then divided by $100$ in a later run and the final loss is worse. Why?
8. **Predict.** The voltage feature is replaced by its value in millivolts. What happens to the exact least-squares line, to the condition number of $\mathbf{X}^\top\mathbf{X}$ and to the largest stable learning rate of gradient descent?

## Exercises

1. **[Calculate]** For the line $w = 20$, $b = 15$ (attempt 2 of chapter 1) compute the residuals (prediction minus label) on rows 1, 5 and 10 of the sensor table, their squares, and compare the average of the three squares with the line's MSE from `linear_regression.rescored_attempts`. Why is an average over three rows not a good guide to the average over ten?
2. **[Calculate]** For the four rows $(v, y) = (0, 1), (1, 3), (2, 4), (3, 8)$ compute $\bar{v}$, $\bar{y}$, $S_{vv}$, $S_{vy}$, the least-squares slope and intercept, and the MSE. Then build $\mathbf{X}^\top\mathbf{X}$ and $\mathbf{X}^\top\mathbf{y}$ and solve by elimination to confirm.
3. **[Calculate]** Derive the least-squares estimate for a model with no intercept, $\hat{y} = w v$, from $\partial\,\mathrm{MSE}/\partial w = 0$. Apply it to the four rows of exercise 2 and compare the MSE with the model with an intercept. Which is smaller, and must it be?
4. **[Calculate]** For $\mathbf{X}$ with columns $(1, 1, 1)$ and $(1, 2, 3)$ and $\mathbf{y} = (2, 2, 5)$, compute $\mathbf{X}^\top\mathbf{X}$, its eigenvalues $\lambda_i$ of $\mathbf{H} = \frac{2}{3}\mathbf{X}^\top\mathbf{X}$, the stability limit and the best learning rate. Start from $\mathbf{w} = (0, 0)$ and do two steps of gradient descent with that best rate by hand.
5. **[Calculate]** Show that for a standardised feature (its z-score with divisor $N - 1$) and an intercept, $\mathbf{X}^\top\mathbf{X}$ is diagonal and find its entries for $N$ rows. What is the condition number as a function of $N$?
6. **[Implement]** Implement `fit_least_squares` (normal equations with your own elimination) and reproduce `linear_regression.closed_form_line` and `linear_regression.design_matrix_system` from the ten rows of `learning_problem.sensor_table`. Use `solve(A, c, true)` with the vanishing-pivot test of the section on the normal equations. Then break it on purpose with the millivolt column of `linear_regression.collinear_features`, case A, and make your solver report the vanishing pivot.
7. **[Implement]** Implement `gradient_descent` with a finite-difference check of its gradient, and reproduce `linear_regression.gradient_check` and the rows $k = 0, 1, 2, 5, 10$ of `linear_regression.gd_three_rates` from the start $(w, b) = (0, 30)$.
8. **[Implement]** Compute the eigenvalues of $\mathbf{H} = \frac{2}{N}\mathbf{X}^\top\mathbf{X}$ with `jacobi_eigen(H, 20)` (Volume I, chapter 6, "The algorithms"; for the $2\times 2$ case the trace and determinant give them as well), then reproduce the two factor columns of `linear_regression.stability_table` that come from them and its "steps measured" column, using the unrounded $\eta^\star$ in its marked row. The same routine, applied to $\mathbf{X}^\top\mathbf{X}$ ($3 \times 3$ in case B of `linear_regression.collinear_features`, up to $4 \times 4$ for the cubic of `linear_regression.polynomial_fit`), gives their eigenvalue and condition-number columns.
9. **[Implement]** Reproduce `linear_regression.scaling_conditioning` for volts and z-scores (best rate used unrounded), and `linear_regression.scaled_fit_matches`: fit in z-score units, map the weights back and verify that the prediction at $1.75$ V is unchanged.
10. **[Investigate]** Fit the parabola of `linear_regression.polynomial_fit` by gradient descent on the raw $(1, v, v^2)$ columns with the largest stable learning rate. How many steps does it need to reach the closed-form loss to within $0.001$, and how many with z-scored $v$? Report both condition numbers.
11. **[Investigate]** Replace the squared loss by the absolute loss in gradient descent on the sensor line (the slope of $\lvert r\rvert$ is $\mathrm{sign}(r)$). Run it with a fixed learning rate and with one that is divided by $k$ at step $k$. What happens near the minimum, and how does it relate to the smoothness argument for the squared loss?
12. **[Investigate]** Create $N = 20$ rows from a line with noise of your own, and fit with the two-meter design of `linear_regression.collinear_features`, case B, using several values of the disagreement between the meters (0.1, 0.01, 0.001). Plot the weights against the disagreement and the condition number against the disagreement, and say how they scale.

## Summary

A **loss function** turns "how far off" into a number the parameters can be
chosen to minimise, and the loss of the model on the data is the average of
the loss per example. The **mean squared error** squares each prediction error: it
punishes large misses much more than small ones, it is smooth, it is a convex bowl
in the parameters of a linear model, and it is exactly what a Gaussian noise
model asks for. **Least squares** chooses the parameters that minimise it. For a
line the slope is $S_{vy}/S_{vv}$ and the line passes through the means. For any
number of features, setting the gradient $\frac{2}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ to
zero gives the **normal equations** $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$, solved
by elimination; geometrically the fitted vector is the projection of the labels
onto the column space of the design matrix and the residual is at a right angle to
it. The equations fail, or give meaningless weights, when features are collinear, and
their cost grows with the square of the number of features.

**Gradient descent** removes both limits: it steps against the gradient by a
**learning rate** $\eta$, costs one pass over the data per step and stores only the
weights. For a quadratic loss with Hessian eigenvalues $\lambda_i$, each error
component shrinks by $1 - \eta\lambda_i$ per step, so the method is stable only
for $\eta < 2/\lambda_{\max}$ and fastest at $2/(\lambda_{\max} + \lambda_{\min})$, where
the error shrinks by $(\kappa - 1)/(\kappa + 1)$ per step. The iteration
**converges** geometrically, with a speed set by the condition number $\kappa$.
Stopping rules on the gradient and on the loss change differ in reliability, and
rescaling features with z-scores can change $\kappa$ from tens of millions to
about one without changing the model. Features may be nonlinear functions of the
data (powers, logarithms) while the model stays linear in its parameters.

The limits are in the model. A line has one straight answer to every question; a
quadratic loss has one minimum and a closed form, and none of that survives a
model like a neural network. What survives is the loop: compute the loss and its
gradient, step, check that the loss fell. What we have not done is explain why the
squared error is the right loss for *every* problem. For a yes-or-no answer it is
the wrong one, as the next chapter, *Classification and Logistic Regression*,
shows; it needs a model whose output is a probability, and a loss that fits it.

## Key terms

- **loss function**: a rule that scores one prediction against its label with a non-negative number, zero for a perfect prediction; its average over the data is the quantity learning minimises. <!-- concept: loss-function -->
- **mean squared error**: the average over the examples of the squared difference between prediction and label, $\frac{1}{N}\sum_n (\hat{y}_n - y_n)^2$. <!-- concept: mean-squared-error -->
- **least squares**: the method of choosing parameters that minimise the sum of the squared differences between predictions and labels. <!-- concept: least-squares -->
- **normal equations**: the system $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$ obtained by setting the gradient of the squared error to zero; it says the residual is orthogonal (normal) to every column of the design matrix. <!-- concept: normal-equations -->
- **gradient descent**: the iteration that repeatedly moves the parameters a small step against the gradient of the loss, $\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla\mathcal{L}(\mathbf{w})$. <!-- concept: gradient-descent -->
- **learning rate**: the positive number $\eta$ that scales the step of gradient descent; too small is slow, above $2/\lambda_{\max}$ the iteration diverges on a quadratic loss. <!-- concept: learning-rate -->
- **convergence**: the approach of the iterates to the minimiser and of the loss to its minimum; for a quadratic loss it is geometric, with a speed set by the learning rate and the condition number. <!-- concept: convergence -->

## Answers

**Before you begin.**

1. The parameters change (from $(24, 9)$ to $(20, 18)$), and so does the score. The ten rows do not change: they are the fixed data, so the score is a function of the parameters only.
2. The gradient is $(2w_1 + 3w_2,\ 3w_1)$, which is $(5, 3)$ at $(1, 1)$. $L$ falls fastest in the direction $-(5, 3)$, the negative of the gradient.
3. $\mathbf{X}^\top$ has shape $[2, 10]$, $\mathbf{X}^\top\mathbf{X}$ has shape $[2, 2]$ and $\mathbf{X}^\top\mathbf{y}$ has shape $[2]$.
4. Subtract $\tfrac12$ of the first equation from the second: $2.5\,w_2 = 5$, so $w_2 = 2$ and $w_1 = 1$. The pivot (the entry we divide by) must be nonzero, here $2$ and then $2.5$; a zero pivot means the rows are dependent or need to be swapped.
5. A convex function has a single valley: every point with zero slope is a global minimum, so there is no worse critical point to get stuck at.
6. $\lvert 1 - h\lambda\rvert < 1$ exactly when $0 < h < 2/\lambda$.

**Check your understanding.**

1. It is best in the sense of the squared error: it has the smallest MSE of all lines on these rows. A different loss, such as the absolute error, can prefer a different line (the grid's line had a smaller average absolute error, $2.77$ against $2.81$), and a line that is best on these ten rows is not guaranteed to be the best on new ones.
2. Both have an average absolute error of $3$ ($12/4 = 3$). Model A has MSE $(4 \cdot 9)/4 = 9$ and model B has MSE $144/4 = 36$: the squared error prefers A, the model without a large miss.
3. Computing the inverse costs about $d^3$ against $d^3/3$ for elimination and is less accurate, and the inverse is not what is wanted: only its product with $\mathbf{X}^\top\mathbf{y}$ is. The program should call a solver on the system, and check for a vanishing pivot.
4. It is the first gradient condition, $\partial\mathrm{MSE}/\partial b = \frac{2}{N}\sum r_n = 0$, which holds because the design matrix has a column of ones. A sum of $3.2$ would mean the fit is not a least-squares solution: the intercept is missing, the solve failed, or gradient descent has not converged.
5. Stable: $0 < \eta < 2/10 = 0.2$. Best: $2/(10 + 0.1) = 0.198$. The worst factor is $(\kappa - 1)/(\kappa + 1)$ with $\kappa = 100$, that is $99/101 = 0.980$ per step in the error, $0.961$ in the loss.
6. The most likely explanation is a learning rate above the stability limit (the slow growth of the oscillation in the steep direction makes the loss fall at first and rise later, as in the $\eta = 0.24$ run); an exact minimum would have the loss stay flat, not rise. Halve the learning rate and run again.
7. The loss falls by a fraction of about $2\eta\lambda_{\min}$ of its excess per step in the slow direction, so a per-step change below $10^{-6}$ occurs at an excess of $10^{-6}/(2\eta\lambda_{\min})$, which is $100$ times larger at the smaller rate: the run stops earlier in terms of distance from the minimum.
8. The least-squares line is unchanged as a function (the slope in mV is $1/1000$ of the slope in volts and the predictions are the same). The condition number of $\mathbf{X}^\top\mathbf{X}$ grows enormously (to $3.0 \times 10^7$ for the sensor, in the table) and the largest stable rate drops to $2/\lambda_{\max}$ of the new, much larger, curvature (below $10^{-6}$).

## References

- Legendre, A.-M., *Nouvelles methodes pour la determination des orbites des cometes*, 1805, and Gauss, C. F., *Theoria motus corporum coelestium*, 1809 (the method of least squares).
- Cauchy, A.-L., "Methode generale pour la resolution des systemes d'equations simultanees", 1847 (gradient descent).
- Golub, G. H. and Van Loan, C. F., *Matrix Computations*, fourth edition, 2013 (normal equations, conditioning and their alternatives).
- Hastie, T., Tibshirani, R. and Friedman, J., *The Elements of Statistical Learning*, second edition, 2009 (linear regression and least squares).
- Nocedal, J. and Wright, S. J., *Numerical Optimization*, second edition, 2006 (convergence rate of steepest descent on a quadratic).
