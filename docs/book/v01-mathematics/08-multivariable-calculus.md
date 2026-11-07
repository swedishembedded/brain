<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Gradients, Jacobians and Many Variables

*When a quantity depends on millions of numbers, which way should we move them all to make it smaller?*

## The problem

Chapter 7 gave us a tool for one knob: the derivative says which way to turn it
and how strongly the score responds. A real model has many knobs, and they are
turned together. Take the smallest case that shows the new difficulty, a model
with two knobs $w_1$ and $w_2$. It sees an input made of two numbers
$(x_1, x_2)$ and predicts $w_1 x_1 + w_2 x_2$. Two examples are given: the input
$(1, 1)$ with target $3$, and the input $(1, 3)$ with target $7$. The score is
half the sum of the squared errors,

$$
L(w_1, w_2) = \tfrac{1}{2}\Big[(w_1 + w_2 - 3)^2 + (w_1 + 3 w_2 - 7)^2\Big].
$$

Chapter 2 would call $L$ a function of two inputs. Its output is a score and its
inputs are the knobs. We cannot draw a curve for it, but we can tabulate it on a
grid, one row per value of $w_2$ and one column per value of $w_1$.

<!-- worked: multivariable.loss_grid -->
L(w1, w2) = 1/2 [(w1 + w2 - 3)^2 + (w1 + 3 w2 - 7)^2], w2 down the left, w1 across the top.

| w2 \ w1 | -1.0 | 0.0 | 1.0 | 2.0 | 3.0 |
|---|---:|---:|---:|---:|---:|
| 4.0 | 8.0 | 13.0 | 20.0 | 29.0 | 40.0 |
| 3.0 | 1.0 | 2.0 | 5.0 | 10.0 | 17.0 |
| 2.0 | 4.0 | 1.0 | 0.0 | 1.0 | 4.0 |
| 1.0 | 17.0 | 10.0 | 5.0 | 2.0 | 1.0 |
| 0.0 | 40.0 | 29.0 | 20.0 | 13.0 | 8.0 |
<!-- /worked -->

Read the table as a landscape. The score is $0$ at $(w_1, w_2) = (1, 2)$, which
is the setting that predicts both targets exactly ($1 + 2 = 3$ and
$1 + 6 = 7$), and it grows in every direction away from it. The growth is not
the same in every direction. Along the row $w_2 = 2$ the score climbs
$0, 1, 4$ in each direction from the minimum; along the column $w_1 = 1$ it
climbs $0, 5, 20$. Along the diagonal that runs from the top left to the bottom
right it hardly climbs at all: $(0, 3)$ scores $2$ and $(2, 1)$ scores $2$, while
$(2, 3)$ scores $10$. The picture is a long narrow valley.

Standing at one point of this landscape with only the ability to evaluate $L$,
which way is downhill, and how far should we walk? One knob at a time is no
longer enough: a step changes both knobs, and the total change in the score
depends on how the two effects combine. This chapter builds the tools to answer
the question for any number of knobs. The **partial derivative** measures the
effect of one knob with the others held still. The **gradient** collects them
and points uphill. The **Jacobian** does the same for a function with many
outputs, and with it the chain rule becomes a product of tables of slopes.
The **Hessian** measures how the slope itself bends, and the **Taylor
expansion** puts the pieces together into a local model of the function. At the
end we use the model to see why a narrow valley makes walking downhill slow.

You may use from earlier chapters: the derivative and the chain rule (Chapter
7), vectors and the dot product (Chapter 3), and matrices as linear maps
(Chapter 4).

## Before you begin

Six questions on ideas from earlier chapters, to answer from memory. The
answers are at the end of the chapter. If one is hard, go back to the chapter
named before reading on.

1. (Chapter 7, derivative.) For $L(w) = (2w - 3)^2$, what is $L'(0.5)$?
2. (Chapter 7, chain rule.) What is the derivative of $(3x + 1)^2$ at $x = 1$?
3. (Chapter 3, dot product and Cauchy-Schwarz.) For $\mathbf{a} = (3, 4)$ and $\mathbf{b} = (2, 1)$, compute $\mathbf{a}\cdot\mathbf{b}$ and $\lVert\mathbf{a}\rVert$. What does Cauchy-Schwarz say about $\lvert\mathbf{a}\cdot\mathbf{u}\rvert$ for a unit vector $\mathbf{u}$, and when is the bound reached?
4. (Chapter 4, matrix-vector product.) Compute $\mathbf{W}\mathbf{x}$ for $\mathbf{W} = \begin{pmatrix}1 & 2\\ 0 & 3\end{pmatrix}$ and $\mathbf{x} = (2, 1)$. If $\mathbf{W}$ has shape $[m, n]$, what shape must $\mathbf{x}$ have and what shape is the result?
5. (Chapter 4, transpose.) If $\mathbf{A}$ has shape $[2, 3]$ and $\mathbf{B}$ has shape $[3, 4]$, what are the shapes of $\mathbf{A}\mathbf{B}$ and $\mathbf{B}^\top\mathbf{A}^\top$, and how are the two matrices related?
6. (Chapter 2, polynomials.) Evaluate $p(x) = 1 + x + \tfrac{1}{2}x^2 + \tfrac{1}{6}x^3$ at $x = 1$. What number from Chapter 1 does it come close to?

## Partial derivatives: one knob at a time

The first tool is the one we already have. Stand at $\mathbf{p} = (0.5, 1.5)$,
where $L = 2.5$. Freeze $w_2$ at $1.5$ and nudge only $w_1$ by $h$, then do the
reverse. This is the nudging experiment of Chapter 7, run once per knob.

<!-- worked: multivariable.freeze_and_nudge -->
At p = (0.5, 1.5): L = 2.5000. One knob nudged by h, the other frozen.

| knob nudged | h | L(p + h e) | change in L | change per unit of h |
|---|---:|---:|---:|---:|
| w1 | 0.1 | 2.210000 | -0.290000 | -2.9000 |
| w2 | 0.1 | 1.850000 | -0.650000 | -6.5000 |
| w1 | 0.01 | 2.470100 | -0.029900 | -2.9900 |
| w2 | 0.01 | 2.430500 | -0.069500 | -6.9500 |
| w1 | 0.001 | 2.497001 | -0.002999 | -2.9990 |
| w2 | 0.001 | 2.493005 | -0.006995 | -6.9950 |
<!-- /worked -->

The last column settles near $-3$ for $w_1$ and near $-7$ for $w_2$, closer the
smaller the nudge, exactly as the single-knob table of Chapter 7 settled near its
derivative. These two limits are the **partial derivatives** of $L$ at
$\mathbf{p}$, written

$$
\frac{\partial L}{\partial w_1}(\mathbf{p}) = \lim_{h\to 0}\frac{L(p_1 + h,\, p_2) - L(p_1, p_2)}{h},
\qquad
\frac{\partial L}{\partial w_2}(\mathbf{p}) = \lim_{h\to 0}\frac{L(p_1,\, p_2 + h) - L(p_1, p_2)}{h}.
$$

The curly $\partial$ reminds us that other inputs exist and are being held
fixed. The definition is Chapter 7's definition applied to a one-variable slice
of $L$, so every rule of that chapter (sums, products, the chain rule) applies
unchanged, with the frozen variables treated as constants. Freezing $w_2 = 1.5$
turns $L$ into a function of $w_1$ alone,

$$
L(w_1, 1.5) = \tfrac{1}{2}\big[(w_1 - 1.5)^2 + (w_1 - 2.5)^2\big],
\qquad
\frac{d}{dw_1}L(w_1, 1.5) = (w_1 - 1.5) + (w_1 - 2.5) = 2w_1 - 4,
$$

and freezing $w_1 = 0.5$ gives $L(0.5, w_2) = \tfrac{1}{2}\big[(w_2 - 2.5)^2 + (3w_2 - 6.5)^2\big]$ with
derivative $(w_2 - 2.5) + 3(3w_2 - 6.5) = 10w_2 - 22$. Evaluated at $\mathbf{p}$
these are $-3$ and $-7$.

<!-- worked: multivariable.partial_formulas -->
Residuals at p = (0.5, 1.5): rA = -1.00, rB = -2.00.

| quantity | value |
|---|---:|
| slice with w2 frozen at 1.5: derivative 2 w1 - 4 at w1 = 0.5 | -3.00 |
| slice with w1 frozen at 0.5: derivative 10 w2 - 22 at w2 = 1.5 | -7.00 |
| rA + rB (first partial from the residuals) | -3.00 |
| rA + 3 rB (second partial from the residuals) | -7.00 |
<!-- /worked -->

The two ways of computing agree, and the second row of the formulas suggests how
to do it once and for all. Write the **residual** of example $k$ as
$r_k = \mathbf{x}_k\cdot\mathbf{w} - t_k$ (prediction minus target), so that
$L = \tfrac{1}{2}\sum_k r_k^2$. Since $\partial r_k/\partial w_j = x_{kj}$ (only
the $j$-th term of the dot product involves $w_j$), the chain rule of Chapter 7
gives

$$
\frac{\partial L}{\partial w_j} = \sum_k r_k\, x_{kj}.
$$

Each residual is multiplied by the input that its knob saw, and the products are
added: a knob moves the score in proportion to how much it contributed to the
errors. At $\mathbf{p}$ the residuals are $-1$ and $-2$, and the sum for
$j = 2$ is $(-1)(1) + (-2)(3) = -7$.

## The gradient: all the slopes together

Collect the partial derivatives into one vector. For a function $L$ of $n$
inputs $\mathbf{w} = (w_1, \dots, w_n)$ the **gradient** is

$$
\nabla L(\mathbf{w}) = \begin{pmatrix} \partial L/\partial w_1 \\ \vdots \\ \partial L/\partial w_n \end{pmatrix},
$$

a column vector with one entry per input, the same shape as $\mathbf{w}$. That
shape is not a convenience: it is what lets us add the gradient to the
parameters later, entry by entry. (The notation page writes
$\nabla_{\mathbf{w}} L$ when several variables are in play.) For our loss the
formula of the last section reads $\nabla L = \sum_k r_k\,\mathbf{x}_k$, the
residual-weighted sum of the inputs. Here it is at six points, beside a central
difference taken one knob at a time with the other frozen (Chapter 7):

<!-- worked: multivariable.gradient_check_points -->
Central differences with h = 1e-5, one knob at a time. The last column is the largest absolute difference of the two gradient entries; it is round-off, so its digits depend on h and are compared by order of magnitude.

| point w | L(w) | gradient (formula) | central differences | largest difference |
|---|---:|---:|---:|---:|
| (0.0, 0.0) | 29.000 | (-10.000, -24.000) | (-10.000, -24.000) | 2.01e-10 |
| (0.5, 1.5) | 2.500 | (-3.000, -7.000) | (-3.000, -7.000) | 1.35e-10 |
| (1.0, 2.0) | 0.000 | (0.000, 0.000) | (0.000, 0.000) | 0.00e+00 |
| (2.0, 0.0) | 13.000 | (-6.000, -16.000) | (-6.000, -16.000) | 1.94e-10 |
| (3.0, 4.0) | 40.000 | (12.000, 28.000) | (12.000, 28.000) | 4.54e-10 |
| (-1.0, 3.0) | 1.000 | (0.000, 2.000) | (0.000, 2.000) | 5.75e-11 |
<!-- /worked -->

The two columns agree to about $10^{-10}$, which is the check of Chapter 7
applied once per entry. At the minimum $(1, 2)$ the gradient is the zero
vector: no knob can lower the score. So far the gradient is a bookkeeping device.
What makes it the central object of the subject is what it says about moving in
every direction at once.

## Moving in any direction: the directional derivative

The partial derivatives describe moves along the two axes. A real step has
components in both knobs. Pick a unit vector $\mathbf{u}$ (length $1$, Chapter 3)
as the direction and walk a distance $h$ along it from $\mathbf{p}$. How does the
score change? Define the **directional derivative**

$$
D_{\mathbf{u}}L(\mathbf{p}) = \lim_{h\to 0}\frac{L(\mathbf{p} + h\mathbf{u}) - L(\mathbf{p})}{h},
$$

the rate of change of $L$ per unit distance in direction $\mathbf{u}$. With
$\mathbf{u} = (1, 0)$ it is $\partial L/\partial w_1$. For other directions we
have no formula yet, so we derive one. Split the change over a step
$h\mathbf{u} = (hu_1, hu_2)$ into two one-knob moves: first change $w_1$ with
$w_2$ at its old value, then change $w_2$.

$$
L(p_1 + hu_1,\, p_2 + hu_2) - L(p_1, p_2)
= \underbrace{\big[L(p_1 + hu_1,\, p_2 + hu_2) - L(p_1,\, p_2 + hu_2)\big]}_{\text{only } w_1 \text{ moved}}
+ \underbrace{\big[L(p_1,\, p_2 + hu_2) - L(p_1,\, p_2)\big]}_{\text{only } w_2 \text{ moved}}.
$$

Each bracket is the change of a one-variable function over a small step, so by
Chapter 7 it is the slope times the step, up to an error that shrinks faster
than the step: the second is $\partial_2 L\cdot hu_2$, and the first is
$\partial_1 L\cdot hu_1$ with the slope read at $w_2 = p_2 + hu_2$ rather than
$p_2$. When the partial derivatives are continuous (the case for every function
in this chapter) that slope is, for small $h$, as close to its value at $p_2$ as
we please. Dividing by $h$ and letting it shrink,

$$
D_{\mathbf{u}}L(\mathbf{p}) = \frac{\partial L}{\partial w_1}u_1 + \frac{\partial L}{\partial w_2}u_2
= \nabla L(\mathbf{p})\cdot\mathbf{u}.
$$

The same argument, one knob at a time, gives $\nabla L\cdot\mathbf{u}$ for any
number of inputs. The slope in every direction is the dot product of the
gradient with that direction. One vector, computed once, answers the question for
all the directions. Eight directions at $\mathbf{p} = (0.5, 1.5)$, and three
special ones, compared with a central difference along each:

<!-- worked: multivariable.directional_scan -->
At p = (0.5, 1.5) the gradient is (-3.00, -7.00) with length 7.6158. Angles are measured anticlockwise from the w1 axis; the perpendicular direction is (-g2, g1)/|g|, the gradient direction turned a quarter turn anticlockwise. Central difference along u: (L(p + h u) - L(p - h u)) / 2h with h = 1e-5.

| unit direction u | u | grad . u | central difference along u |
|---|---:|---:|---:|
| 0 deg | (1.000, 0.000) | -3.0000 | -3.0000 |
| 45 deg | (0.707, 0.707) | -7.0711 | -7.0711 |
| 90 deg | (0.000, 1.000) | -7.0000 | -7.0000 |
| 135 deg | (-0.707, 0.707) | -2.8284 | -2.8284 |
| 180 deg | (-1.000, 0.000) | 3.0000 | 3.0000 |
| 225 deg | (-0.707, -0.707) | 7.0711 | 7.0711 |
| 270 deg | (0.000, -1.000) | 7.0000 | 7.0000 |
| 315 deg | (0.707, -0.707) | 2.8284 | 2.8284 |
| along the gradient | (-0.394, -0.919) | 7.6158 | 7.6158 |
| against the gradient | (0.394, 0.919) | -7.6158 | -7.6158 |
| perpendicular to the gradient | (0.919, -0.394) | 0.0000 | 0.0000 |
<!-- /worked -->

The formula and the experiment agree on every row. The pattern across the rows
is the content of the next paragraph.

**Why the gradient points uphill.** Among all unit directions, which one makes
$\nabla L\cdot\mathbf{u}$ largest? Chapter 3 proved the Cauchy-Schwarz
inequality, $\mathbf{a}\cdot\mathbf{b} \le \lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert$,
with equality exactly when one vector is a positive multiple of the other. With
$\lVert\mathbf{u}\rVert = 1$,

$$
\nabla L\cdot\mathbf{u} \;\le\; \lVert\nabla L\rVert\,\lVert\mathbf{u}\rVert = \lVert\nabla L\rVert,
$$

and the bound is reached when $\mathbf{u} = \nabla L/\lVert\nabla L\rVert$. Three
conclusions follow. The **gradient direction is the direction of steepest
ascent**, and the steepest rate is the length $\lVert\nabla L\rVert$. The
negative gradient is the direction of steepest descent, with rate
$-\lVert\nabla L\rVert$ (the same bound applied to $-\mathbf{u}$). And a direction
perpendicular to the gradient has slope zero. In the table the largest number
among the eight sampled directions is $7.0711$, at $225$ degrees, and the
gradient direction reaches $7.6158$, which is $\lVert\nabla L\rVert$ at this
point. A brute-force search confirms that no direction does better:

<!-- worked: multivariable.cauchy_schwarz_gradient -->
Each direction is u = (cos t, sin t) with t = 2 pi times one uniform draw of the book's generator (Volume I, chapter 17), seed 18266694207942227462; 100000 draws in order.

| quantity | value |
|---|---:|
| length of the gradient | 7.6158 |
| largest grad . u over 100000 random unit u | 7.6158 |
| smallest grad . u over the same u | -7.6158 |
<!-- /worked -->

This is the answer to the question of the chapter, in the local sense that a
derivative always gives: to lower the score fastest per unit of distance
travelled, move against the gradient. A subtlety is worth marking now. Steepest
is measured in the ordinary length of Chapter 3. If the knobs have different
scales (one measured in metres, one in kilometres), a different measure of
length picks a different "steepest" direction. In all that follows we use the
ordinary length.

## Contour lines and the gradient

A **contour line** of $L$ is the set of points where $L$ has one fixed value,
the way a line on a hiking map joins points of equal height. For our loss the
contours are ellipses around the minimum, and they are the picture of the
valley.

![Contour lines of the two-knob loss, five ellipses around the minimum at (1, 2), long and thin and tilted. At five marked points an arrow shows the gradient, scaled by 0.06. Each arrow is perpendicular to the contour through its point and points toward larger loss.](img/contour-gradient.svg)

Each arrow in the figure is the gradient of the formula above, scaled down so
that it fits. Two facts are visible: the arrows cross the contours at right
angles, and they point away from the minimum. The first is a consequence of the
formula for the directional derivative. If we walk along a contour, the score does
not change, so its rate of change in the direction of travel is zero. If
$\mathbf{c}(t)$ is a path along the contour, then $L(\mathbf{c}(t))$ is constant,
and its derivative, which is $\nabla L\cdot\mathbf{c}'(t)$ by the formula of the
last section, is zero. The velocity $\mathbf{c}'(t)$ of the path is the tangent of
the contour, so $\nabla L$ is perpendicular to it (the dot product is zero,
Chapter 3). Numerically, a short step of length $h$ along the tangent at
$\mathbf{p}$ changes the score by far less than the same step along the
gradient:

<!-- worked: multivariable.contour_tangent -->
At p = (0.5, 1.5), tangent direction (0.9191, -0.3939) (perpendicular to the gradient), gradient direction (-0.3939, -0.9191).

| step length | change in L along tangent | per unit step | change in L along gradient | per unit step |
|---|---:|---:|---:|---:|
| 0.1 | +1.724e-03 | +0.0172 | +8.199e-01 | +8.1985 |
| 0.01 | +1.724e-05 | +0.0017 | +7.674e-02 | +7.6740 |
| 0.001 | +1.724e-07 | +0.0002 | +7.622e-03 | +7.6216 |
<!-- /worked -->

Along the tangent the change per unit step is not exactly zero for a finite step
(the contour curves away from its tangent line), but it falls in proportion to
the step length, from $0.017$ to $0.0002$, while the change per unit step along the
gradient settles at $\lVert\nabla L\rVert = 7.6158$. The valley shape also
explains the length of the arrows: they are long across the narrow direction,
where the contours are close together and the score changes quickly, and short
along the valley, where the contours are far apart.

## A first look: stepping against the gradient

We now have a rule for choosing a direction. Try it. From a point $\mathbf{w}$,
move to $\mathbf{w} - \eta\,\nabla L(\mathbf{w})$, where $\eta > 0$ is a step
size we pick, and repeat. Chapter 7 contained the one-knob version ("move against
the sign of the slope"). Start from $(0, 0)$, where $L = 29$, with a small step
size $\eta = 0.05$ and a larger one $\eta = 0.18$.

<!-- worked: multivariable.first_steps -->
Start at w = (0, 0). Each step moves w by -eta times the gradient.

| step | w (eta = 0.05) | L | gradient there | w (eta = 0.18) | L |
|---|---:|---:|---:|---:|---:|
| 0 | (0.0000, 0.0000) | 29.0000 | (-10.000, -24.000) | (0.000, 0.000) | 29.00 |
| 1 | (0.5000, 1.2000) | 5.0500 | (-4.200, -10.000) | (1.800, 4.320) | 34.98 |
| 2 | (0.7100, 1.7000) | 0.8821 | (-1.780, -4.160) | (-0.158, -0.432) | 42.18 |
| 3 | (0.7990, 1.9080) | 0.1567 | (-0.770, -1.724) | (2.010, 4.780) | 50.88 |
| 4 | (0.8375, 1.9942) | 0.0303 | (-0.348, -0.708) | (-0.355, -0.951) | 61.36 |
| 5 | (0.8549, 2.0296) | 0.0083 | (-0.172, -0.284) | (2.257, 5.336) | 74.01 |
| 6 | (0.8635, 2.0438) | 0.0043 | (-0.098, -0.108) | (-0.598, -1.574) | 89.26 |
| 7 | (0.8684, 2.0492) | 0.0035 | (-0.066, -0.034) | (2.551, 6.010) | 107.66 |
| 8 | (0.8717, 2.0509) | 0.0033 | (-0.053, -0.004) | (-0.894, -2.324) | 129.85 |
<!-- /worked -->

The small step size behaves well for a while. After one step the score falls from
$29$ to $5.05$ and after four it is $0.03$. Then something changes: from step 5
on, the score barely moves ($0.0083$, $0.0043$, $0.0035$, $0.0033$) although the
knobs, $(0.87, 2.05)$ at step 8, are still some way from the minimum $(1, 2)$. The
path has found the floor of the valley and is crawling along it. The larger step
size does worse than crawling: the score *rises* at every step ($29$, $34.98$,
$42.18$, and $129.85$ at step 8), and the knobs jump from one side of the valley
to the other. The direction was right and the distance was wrong.

The reason is that the gradient is a local fact. It says which way is downhill at
$\mathbf{p}$, not how long the slope will last. Walk along the line
$\mathbf{w} = \mathbf{p} - t\nabla L(\mathbf{p})$ from $\mathbf{p} = (0.5, 1.5)$
and watch the score:

<!-- worked: multivariable.step_length_scan -->
From p = (0.5, 1.5) along the line w = p - t g, g = (-3.00, -7.00).

| t | w | L(w) |
|---|---:|---:|
| 0.0000 | (0.500, 1.500) | 2.5000 |
| 0.0200 | (0.560, 1.640) | 1.4752 |
| 0.0500 | (0.650, 1.850) | 0.4450 |
| 0.0858 | (0.757, 2.101) | 0.0118 |
| 0.1200 | (0.860, 2.340) | 0.4072 |
| 0.1700 | (1.010, 2.690) | 2.4082 |
| 0.2500 | (1.250, 3.250) | 9.1250 |

The parabola L(p - t g) is lowest at t = (g . g) / (g^T H g) = 58.0000 / 676.0000 = 0.0858.
<!-- /worked -->

Along the line the score falls, reaches a lowest value near $t = 0.086$, and then
rises, so the best distance exists and is not "as far as possible". Where it is
depends on how quickly the slope itself changes along the line. That quantity,
the curvature, is the subject of the last sections. For now, two things have been
established: the negative gradient is the right direction to start in, and
choosing the distance needs a second piece of information. (Volume II names this
rule gradient descent, gives it a chapter of its own, and the later chapters of
Volume III study how to choose its step sizes.)

## Vector functions and the Jacobian

Not every function ends in a single score. A layer of a network takes a vector of
numbers and produces another vector. Call such a function
$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, with $m$ output entries
$f_1, \dots, f_m$ and $n$ inputs. Each output entry is itself a function of
$n$ variables with its own gradient. We stack the $m$ gradients, written as rows,
into one table, the **Jacobian**:

$$
\mathbf{J}_{\mathbf{f}}(\mathbf{x}) =
\begin{pmatrix}
\partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\
\vdots & & \vdots \\
\partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n
\end{pmatrix},
\qquad (\mathbf{J}_{\mathbf{f}})_{ij} = \frac{\partial f_i}{\partial x_j},
$$

of shape $[m, n]$: **one row per output, one column per input**. This is the
convention the backpropagation chapter uses, and the shape rule to remember is
that the Jacobian of a map from $\mathbb{R}^n$ to $\mathbb{R}^m$ has the shape of
the matrix of that map, outputs by inputs. For a single output ($m = 1$) it is
one row, the transpose of the gradient column.

A concrete case. Let $\mathbf{f}(x, y) = (x^2 y,\; 5x + \sin y)$, two inputs and
two outputs, at the point $(1, 2)$. The partial derivatives by the rules of
Chapter 7, with the other variable frozen, are $\partial f_1/\partial x = 2xy$,
$\partial f_1/\partial y = x^2$, $\partial f_2/\partial x = 5$ and
$\partial f_2/\partial y = \cos y$.

<!-- worked: multivariable.jacobian_2x2 -->
f(x, y) = (x^2 y, 5x + sin y) at (1, 2): f = (2.0000, 5.9093). Central differences with h = 1e-6, one input at a time.

| Jacobian | formula | central differences |
|---|---:|---:|
| row 0 (output 0) | (4.0000, 1.0000) | (4.0000, 1.0000) |
| row 1 (output 1) | (5.0000, -0.4161) | (5.0000, -0.4161) |
<!-- /worked -->

**The Jacobian is the best linear approximation.** A matrix is a linear map
(Chapter 4), and a Jacobian is the linear map that best imitates $\mathbf{f}$
near the point. The argument is the one that gave the directional derivative,
applied to each output entry. For a small step $\mathbf{d}$ we found
$f_i(\mathbf{x} + \mathbf{d}) \approx f_i(\mathbf{x}) + \nabla f_i\cdot\mathbf{d}$.
The right-hand side is the dot product of the $i$-th row of $\mathbf{J}$ with
$\mathbf{d}$. Stacking all $m$ rows,

$$
\mathbf{f}(\mathbf{x} + \mathbf{d}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}_{\mathbf{f}}(\mathbf{x})\,\mathbf{d}.
$$

The curve is replaced by its tangent, a surface by its tangent plane, a vector
function by the affine map whose matrix is the Jacobian. Shrink a step of
direction $(0.6, -0.8)$ taken from $(1, 2)$ and compare the actual change in
$\mathbf{f}$ with $\mathbf{J}\mathbf{d}$:

<!-- worked: multivariable.linear_approximation_error -->
Step d = s (0.6, -0.8) from (1, 2), length s. Actual change of f against J d.

| s | f(p + d) - f(p) | J d | length of the difference | difference / s^2 |
|---|---:|---:|---:|---:|
| 1 | (1.07200, 3.02274) | (1.60000, 3.33292) | 6.12e-01 | 0.6124 |
| 0.1 | (0.15731, 0.33035) | (0.16000, 0.33329) | 3.99e-03 | 0.3986 |
| 0.01 | (0.01598, 0.03330) | (0.01600, 0.03333) | 3.79e-05 | 0.3793 |
| 0.001 | (0.00160, 0.00333) | (0.00160, 0.00333) | 3.77e-07 | 0.3774 |
<!-- /worked -->

The last column is nearly constant: the error shrinks like $s^2$, as the length
$s$ of the step, so the error shrinks much faster than the step itself, which is
what "best linear approximation" means. (The Taylor section below explains why the
error has this size.) The approximation is useful only for small steps; at
$s = 1$ the two columns disagree in the first digit.

**A linear layer has a constant Jacobian.** The map $\mathbf{y} = \mathbf{W}\mathbf{x}$
with $\mathbf{W}$ of shape $[m, n]$ has entries $y_i = \sum_j W_{ij}x_j$. Only the
term with $j$ involves $x_j$, so
$\partial y_i/\partial x_j = W_{ij}$, and therefore

$$
\mathbf{J}_{\mathbf{y}} = \mathbf{W}.
$$

The Jacobian of the map $\mathbf{x}\mapsto\mathbf{W}\mathbf{x}$ is the matrix
itself, at every point $\mathbf{x}$, because a linear map is already its own best
linear approximation. Adding a bias $\mathbf{b}$ changes nothing: a constant has
derivative $0$.

<!-- worked: multivariable.linear_layer_jacobian -->
y = W x with W of shape [2, 3] at x = (1.0, 2.0, -1.0): y = (-3.50, 7.00). Central differences with h = 1e-3.

| Jacobian row | row of W | central differences |
|---|---:|---:|
| row 0 | (1.0000, -2.0000, 0.5000) | (1.0000, -2.0000, 0.5000) |
| row 1 | (0.0000, 3.0000, -1.0000) | (0.0000, 3.0000, -1.0000) |

The same central-difference Jacobian at x = (10, -4, 7) differs from the one above by at most 2.22e-12.
<!-- /worked -->

This one-line result is the reason the matrix multiplication of Chapter 4 sits at
the centre of the backward pass: the derivative of a layer with respect to its
input is the layer's own weight matrix, with the nonlinearity's slope placed
beside it (next section).

## The multivariable chain rule

A network is a composition of maps (Chapter 2). How does a change at the input
propagate to the output? Start with the smallest case where the one-variable
rule of Chapter 7 is not enough: one input $x$ and one output $z$ with *two*
intermediate quantities, $z = y_1 y_2$ with $y_1 = x^2$ and $y_2 = \sin x$. A
nudge to $x$ changes $z$ along two routes, through $y_1$ and through $y_2$, and
the two effects add:

$$
\frac{dz}{dx} = \frac{\partial z}{\partial y_1}\frac{dy_1}{dx} + \frac{\partial z}{\partial y_2}\frac{dy_2}{dx}.
$$

<!-- worked: multivariable.chain_path_sum -->
| path | factors | contribution |
|---|---|---:|
| through y1 = x^2 | 0.9320 * 2.4000 | 2.236894 |
| through y2 = sin x | 1.4400 * 0.3624 | 0.521795 |
| sum of the paths |  | 2.758689 |
| central difference of z(x) = x^2 sin x at 1.2 |  | 2.758689 |
<!-- /worked -->

The sum of the two paths equals the central difference of $z(x) = x^2 \sin x$ at
$x = 1.2$. The right-hand side of the formula is a row of two partial derivatives
times a column of two derivatives, a dot product. It is a matrix product in
disguise, and the general statement says so.

**The rule, derived.** Let $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$ and
$\mathbf{g}:\mathbb{R}^m\to\mathbb{R}^p$, and let $\mathbf{h}(\mathbf{x}) =
\mathbf{g}(\mathbf{f}(\mathbf{x}))$. Fix $\mathbf{x}$ and take a small step
$\mathbf{d}$. Writing the best linear approximation of $\mathbf{f}$ with its
error, $\mathbf{f}(\mathbf{x} + \mathbf{d}) = \mathbf{f}(\mathbf{x}) +
\mathbf{J}_{\mathbf{f}}\mathbf{d} + \mathbf{r}_f$, where $\mathbf{r}_f$ (the
remainder) is smaller than $\mathbf{d}$ by a factor that tends to $0$ with
$\lVert\mathbf{d}\rVert$. The step that $\mathbf{g}$ sees is
$\mathbf{e} = \mathbf{J}_{\mathbf{f}}\mathbf{d} + \mathbf{r}_f$, which is about as
small as $\mathbf{d}$ (it is at most a fixed multiple of $\lVert\mathbf{d}\rVert$).
Apply the linear approximation of $\mathbf{g}$ at $\mathbf{y} = \mathbf{f}(\mathbf{x})$:

$$
\mathbf{h}(\mathbf{x} + \mathbf{d})
= \mathbf{g}(\mathbf{y} + \mathbf{e})
= \mathbf{g}(\mathbf{y}) + \mathbf{J}_{\mathbf{g}}\,\mathbf{e} + \mathbf{r}_g
= \mathbf{h}(\mathbf{x}) + \mathbf{J}_{\mathbf{g}}\mathbf{J}_{\mathbf{f}}\,\mathbf{d}
+ \underbrace{\mathbf{J}_{\mathbf{g}}\mathbf{r}_f + \mathbf{r}_g}_{\text{remainders}}.
$$

Both remainders shrink faster than $\lVert\mathbf{d}\rVert$ ($\mathbf{J}_{\mathbf{g}}$
is a fixed matrix, so $\mathbf{J}_{\mathbf{g}}\mathbf{r}_f$ does too, and
$\mathbf{r}_g$ does because $\mathbf{e}$ is proportional to $\mathbf{d}$ in size).
So $\mathbf{h}(\mathbf{x} + \mathbf{d}) \approx \mathbf{h}(\mathbf{x}) +
(\mathbf{J}_{\mathbf{g}}\mathbf{J}_{\mathbf{f}})\mathbf{d}$, and since the Jacobian
is the only matrix that fits that role,

$$
\boxed{\;\mathbf{J}_{\mathbf{h}}(\mathbf{x}) = \mathbf{J}_{\mathbf{g}}\big(\mathbf{f}(\mathbf{x})\big)\;\mathbf{J}_{\mathbf{f}}(\mathbf{x})\;}
$$

This is the **multivariable chain rule**: the Jacobian of a composition is the
product of the Jacobians, the outer one *evaluated at the inner function's
output* (the same care as in Chapter 7). The shapes check themselves:
$[p, m]\,[m, n] = [p, n]$, the inner size $m$ is shared and disappears, exactly
as in the matrix product of Chapter 4. Reading entry $(i, j)$ of the product
gives $\partial h_i/\partial x_j = \sum_k (\partial g_i/\partial y_k)(\partial y_k/\partial x_j)$,
the sum over routes that opened this section. The one-variable chain rule is the
case where every Jacobian is $[1, 1]$.

The remainder argument can be watched. Compose $\mathbf{g}(a, b) = ab$ after the
function $\mathbf{f}$ of the previous section, and compare the actual change of
the composition with $(\mathbf{J}_{\mathbf{g}}\mathbf{J}_{\mathbf{f}})\mathbf{d}$:

<!-- worked: multivariable.chain_remainder -->
g(a, b) = a b composed after f(x, y) = (x^2 y, 5x + sin y) at (1, 2). J_g = [5.9093, 2.0000], J_f = [4.0000, 1.0000; 5.0000, -0.4161], product [33.6372, 5.0770].

| s | actual change | (J_g J_f) d | error | error / s^2 |
|---|---:|---:|---:|---:|
| 0.1 | 1.642267 | 1.612071 | 3.02e-02 | 3.020 |
| 0.01 | 0.161537 | 0.161207 | 3.30e-04 | 3.302 |
| 0.001 | 0.016124 | 0.016121 | 3.33e-06 | 3.329 |
<!-- /worked -->

The error falls like $s^2$ again, as the derivation promised.

**A chain of three.** A layer's pre-activation, its nonlinearity and its loss:
$\mathbf{u} = \mathbf{W}\mathbf{x}$ (input of width $2$, output of width $3$),
$\mathbf{h} = \tanh(\mathbf{u})$ entry by entry, and
$L = \tfrac{1}{2}\lVert\mathbf{h} - \mathbf{t}\rVert^2$. Each link has a Jacobian, and
the figure shows their shapes.

![A chain of four boxes, x of size 2, u = W x of size 3, h = tanh(u) of size 3 and the loss L of size 1, joined by arrows labelled with the Jacobians W of shape 3 by 2, diag of one minus h squared of shape 3 by 3 and the row h minus t of shape 1 by 3. A note states that dL/dx is the product of the three Jacobians with shapes 1 by 3, 3 by 3, 3 by 2, giving 1 by 2.](img/jacobian-chain.svg)

The tanh acts on each entry alone, so its Jacobian is diagonal: entry $i$ of the
output depends only on entry $i$ of the input, with slope $1 - h_i^2$ (Chapter 7).
The loss has the single row $(\mathbf{h} - \mathbf{t})^\top$. By the chain rule
the Jacobian of the whole map from $\mathbf{x}$ to $L$ is the product of the three.
At $\mathbf{x} = (0.6, -0.4)$, with a fixed $\mathbf{W}$ and target $\mathbf{t}$:

<!-- worked: multivariable.chain_jacobian_product -->
W = [0.5, -1.0; 1.0, 0.5; -0.5, 2.0], target t = (0.2, -0.3, 0.4), L = (1/2)|h - t|^2. x = (0.6, -0.4), u = W x = (0.7000, 0.4000, -1.1000), h = tanh(u) = (0.6044, 0.3799, -0.8005), L = 1.033521. Central differences of L(x) with h = 1e-5.

| factor | shape | value |
|---|---|---:|
| dL/dh = (h - t)^T | [1, 3] | [0.4044, 0.6799, -1.2005] |
| dh/du = diag(1 - h^2) | [3, 3] | [0.6347, 0.0000, 0.0000; 0.0000, 0.8556, 0.0000; 0.0000, 0.0000, 0.3592] |
| du/dx = W | [3, 2] | [0.5000, -1.0000; 1.0000, 0.5000; -0.5000, 2.0000] |
| (dL/dh)(dh/du) | [1, 3] | [0.2567, 0.5818, -0.4312] |
| (dL/dh)(dh/du)(du/dx) | [1, 2] | [0.9257, -0.8282] |
| central differences of L(x) | [2] | (0.9257, -0.8282) |
<!-- /worked -->

The product of the three matrices equals the gradient found by finite differences
on $L(\mathbf{x})$. The result has shape $[1, 2]$, one row, which is the
transpose of the gradient column.

**Which way to multiply.** The product can be evaluated in either order, and the
shapes decide the cost. From the left, multiplying the row $[1, 3]$ by $[3, 3]$ and
then by $[3, 2]$ only ever combines a row with a matrix. From the right, we would
build the $[3, 2]$ matrix first, which is a full matrix product, and at the widths of a real
network that is the difference between tiny and impossible. Taking the transpose
of the row result turns it into the gradient column and reverses the order of the
product (Chapter 4: $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$):

$$
\nabla_{\mathbf{x}} L = \big(\mathbf{J}_3\mathbf{J}_2\mathbf{J}_1\big)^\top
= \mathbf{J}_1^\top\,\mathbf{J}_2^\top\,\mathbf{J}_3^\top .
$$

Start with the gradient at the loss, $\mathbf{J}_3^\top = \mathbf{h} - \mathbf{t}$,
and push it backwards through the transpose of each link:

<!-- worked: multivariable.chain_transposed -->
| step (backwards) | formula | result |
|---|---|---:|
| gradient at h | h - t | (0.4044, 0.6799, -1.2005) |
| gradient at u | (1 - h^2) * (gradient at h), elementwise | (0.2567, 0.5818, -0.4312) |
| gradient at x | W^T (gradient at u) | (0.9257, -0.8282) |
<!-- /worked -->

Each backward step is a matrix-vector product with a transposed Jacobian: the
diagonal one is an elementwise multiplication by $1 - h_i^2$, and the one that
belongs to the linear layer is $\mathbf{W}^\top$. The gradient of the score with
respect to every input of a long chain has been obtained from vectors only, at
the cost of one pass through the links. This is the computation the
backpropagation chapter of Volume III names and generalises; it needs nothing
beyond what we derived here.

## Second derivatives: the Hessian

The step-length table of the first look had a lowest point at a particular $t$.
To predict where, we need to know how quickly the slope changes. In one variable
that is the second derivative (Chapter 7). In many variables there are several
slopes, each of which can change in several directions, so there are $n \times n$
**second partial derivatives**, collected in the **Hessian** matrix

$$
\mathbf{H}(\mathbf{w}) = \Big(\frac{\partial^2 L}{\partial w_i\,\partial w_j}\Big)_{ij}, \qquad
\frac{\partial^2 L}{\partial w_i\,\partial w_j} = \frac{\partial}{\partial w_i}\Big(\frac{\partial L}{\partial w_j}\Big),
$$

of shape $[n, n]$: row $i$ is the gradient of the $i$-th first partial, so the
Hessian is the Jacobian of the gradient. For our loss we already have
$\partial L/\partial w_j = \sum_k r_k x_{kj}$, and $\partial r_k/\partial w_i = x_{ki}$, so

$$
\frac{\partial^2 L}{\partial w_i\,\partial w_j} = \sum_k x_{ki}\,x_{kj}, \qquad
\mathbf{H} = \mathbf{X}^\top\mathbf{X} = \begin{pmatrix} 2 & 4 \\ 4 & 10 \end{pmatrix},
$$

where $\mathbf{X}$ is the matrix of the two inputs, one example per row. The
Hessian of this loss is the same at every point, because $L$ is a quadratic
function. A loss with a nonlinearity inside it has a Hessian that changes from
point to point, as in this example, $f(x, y) = e^x\cos y$:

<!-- worked: multivariable.hessian_example -->
f(x, y) = e^x cos y. Central second differences (f(+h,+h) - f(+h,-h) - f(-h,+h) + f(-h,-h)) / 4h^2 with h = 1e-4.

| point | Hessian (formulas) | central second differences | H[0][1] - H[1][0] |
|---|---:|---:|---:|
| (0.0, 0.0) | [1.0000, 0.0000; 0.0000, -1.0000] | [1.0000, 0.0000; 0.0000, -1.0000] | 0.0 |
| (0.0, 1.0) | [0.5403, -0.8415; -0.8415, -0.5403] | [0.5403, -0.8415; -0.8415, -0.5403] | 0.0 |
| (0.5, 2.0) | [-0.6861, -1.4992; -1.4992, 0.6861] | [-0.6861, -1.4992; -1.4992, 0.6861] | 0.0 |
<!-- /worked -->

The central second differences in the table agree with the formulas, and the last
column shows that the matrix is symmetric: $\partial^2 f/\partial x\,\partial y =
\partial^2 f/\partial y\,\partial x$. This holds for every function whose second
partial derivatives are continuous, and it is why the Hessian can be stored as a
symmetric table. (It also halves the work of computing one by hand.)

**Curvature in a direction.** The Hessian answers the question of the first look.
Walk from $\mathbf{p}$ in a direction $\mathbf{u}$ and let
$\varphi(t) = L(\mathbf{p} + t\mathbf{u})$ be the score along the way. We found
$\varphi'(t) = \sum_i \partial_i L(\mathbf{p} + t\mathbf{u})\,u_i$, the dot product
of the gradient at the current point with $\mathbf{u}$. Differentiate once more,
using the chain rule on each $\partial_i L$:

$$
\varphi''(t) = \sum_i u_i\sum_j \frac{\partial^2 L}{\partial w_j\,\partial w_i}\,u_j
= \mathbf{u}^\top\mathbf{H}\,\mathbf{u}.
$$

The number $\mathbf{u}^\top\mathbf{H}\mathbf{u}$ is the **curvature** of the
landscape along $\mathbf{u}$: how fast the slope changes as we walk. (A *quadratic
form*: a vector, a matrix and the same vector.) For our bowl:

<!-- worked: multivariable.hessian_of_loss -->
H = X^T X = [2, 4; 4, 10], trace 12, determinant 4. The two numbers lambda with det(H - lambda I) = 0 are 11.6569 and 0.3431; their ratio is 33.97.

| unit direction u | u | curvature u^T H u |
|---|---:|---:|
| along w1 | (1.0000, 0.0000) | 2.0000 |
| along w2 | (0.0000, 1.0000) | 10.0000 |
| along (1,1)/sqrt2 | (0.7071, 0.7071) | 10.0000 |
| along (1,-1)/sqrt2 | (0.7071, -0.7071) | 2.0000 |
| steepest curvature direction | (0.3827, 0.9239) | 11.6569 |
| flattest curvature direction | (0.9239, -0.3827) | 0.3431 |
<!-- /worked -->

The curvature is $2$ along the first knob and $10$ along the second, and it
depends strongly on direction. It is largest, $11.66$, along a direction close to
the second knob and smallest, $0.343$, along a direction roughly perpendicular to
it. These two extreme values are the two numbers $\lambda$ for which
$\mathbf{H} - \lambda\mathbf{I}$ stops being invertible (Chapter 4), that is,
$\det(\mathbf{H} - \lambda\mathbf{I}) = (2 - \lambda)(10 - \lambda) - 16 =
\lambda^2 - 12\lambda + 4 = 0$, which gives $\lambda = 6 \pm \sqrt{32}$. They are the
eigenvalues of Chapter 6, and the two directions are its eigenvectors: for a
symmetric matrix the extreme values of $\mathbf{u}^\top\mathbf{H}\mathbf{u}$ over
unit vectors are the extreme eigenvalues. The ratio of the largest to the smallest,
here $33.97$, measures how far from round the bowl is. It decides much of what
follows.

**The best step along a line.** On the line $\mathbf{p} - t\,\mathbf{g}$ with
$\mathbf{g} = \nabla L(\mathbf{p})$, the score is a quadratic polynomial in $t$,
$\varphi(t) = L(\mathbf{p}) - t\,\mathbf{g}\cdot\mathbf{g} + \tfrac{1}{2}t^2\,\mathbf{g}^\top\mathbf{H}\mathbf{g}$
(the slope at $t = 0$ is $-\lVert\mathbf{g}\rVert^2$ and the curvature is
$\mathbf{g}^\top\mathbf{H}\mathbf{g}$; the next section derives where this expression comes
from). Its lowest point, found by setting $\varphi'(t) = 0$ as in Chapter 7, is at

$$
t^\star = \frac{\mathbf{g}\cdot\mathbf{g}}{\mathbf{g}^\top\mathbf{H}\mathbf{g}} = \frac{58}{676} = 0.0858,
$$

which is the minimum seen in the step-length table. The numerator is the squared
slope, the denominator the curvature along the gradient: steep slopes want long
steps and large curvature wants short ones.

## Taylor expansions: a local model of a function

Both the linear approximation of the Jacobian section and the curvature of the
Hessian section say the same thing: near a point, a smooth function looks like a
simple one. The **Taylor expansion** makes this exact and tells us what to add
for more accuracy. We start in one variable, where the idea is cleanest, and
which is also the unfinished business of Chapter 7: its numerical-differentiation
section used the expansion without deriving it.

**One variable: the polynomial that matches.** We want a polynomial
$p_n(h) = c_0 + c_1 h + c_2 h^2 + \dots + c_n h^n$ that imitates $f(a + h)$ for
small $h$. The natural requirement is that it agrees with $f$ at $a$ in value and
in the first $n$ derivatives. Differentiate $p_n$ at $h = 0$: the value is $c_0$,
the first derivative is $c_1$, the second derivative is $2c_2$, the third is
$3\cdot 2\,c_3$, and in general the $k$-th derivative at $0$ is $k!\,c_k$ (the
power rule, applied $k$ times to $h^k$, leaves the constant $k!$; every other
term vanishes at $0$). Matching $f^{(k)}(a) = k!\,c_k$ gives

$$
f(a + h) \approx p_n(h) = f(a) + f'(a)\,h + \frac{f''(a)}{2!}h^2 + \frac{f'''(a)}{3!}h^3 + \dots + \frac{f^{(n)}(a)}{n!}h^n.
$$

This is the **Taylor expansion** of order $n$. Order $0$ is the constant $f(a)$,
order $1$ is the tangent line of Chapter 7, and order $2$ adds the curvature:
$\tfrac{1}{2}f''(a)h^2$, the term that explained the error of the tangent line
there. For $f = e^x$ at $a = 0$ every derivative is $1$ (Chapter 7), so
$c_k = 1/k!$ and $p_n(x) = 1 + x + x^2/2 + x^3/6 + \dots$, the polynomial that
Before-you-begin question 6 evaluated at $1$.

<!-- worked: multivariable.taylor_exp_partial_sums -->
Order-n Taylor polynomial of e^x at 0, evaluated at x = 1 (e = 2.71828183).

| order n | coefficient of x^n | value at x = 1 | e minus value |
|---|---:|---:|---:|
| 0 | 1/1 | 1.00000000 | 1.72e+00 |
| 1 | 1/1 | 2.00000000 | 7.18e-01 |
| 2 | 1/2 | 2.50000000 | 2.18e-01 |
| 3 | 1/6 | 2.66666667 | 5.16e-02 |
| 4 | 1/24 | 2.70833333 | 9.95e-03 |
| 5 | 1/120 | 2.71666667 | 1.62e-03 |
| 6 | 1/720 | 2.71805556 | 2.26e-04 |
| 7 | 1/5040 | 2.71825397 | 2.79e-05 |
| 8 | 1/40320 | 2.71827877 | 3.06e-06 |
<!-- /worked -->

![The curve e to the x on the interval from -2 to 2 with its Taylor polynomials of order 1, 2 and 3 about 0. All four curves touch at x = 0. The order 1 line leaves the curve first, the parabola of order 2 follows it longer, and the cubic of order 3 stays close to the curve over most of the interval, parting from it only at the right end near x = 2.](img/taylor-exp.svg)

Each added term brings the polynomial closer to the curve over a wider interval,
and each is accurate near $a$ and unreliable far from it. At $x = 1$ the sums
$2, 2.5, 2.667, 2.708, \dots$ approach $e = 2.71828$, and the order-8 polynomial
is within $3\times10^{-6}$.

**How big is the error?** Look at the error of orders $1$, $2$ and $3$ for steps
$h$ that are halved repeatedly.

<!-- worked: multivariable.taylor_error_order -->
Error of the order-n Taylor polynomial of e^x at 0, evaluated at x = h; in brackets, how much the error fell from the previous (twice as large) h.

| h | order 1 | order 2 | order 3 |
|---|---:|---:|---:|
| 0.4 | 9.18e-02 | 1.18e-02 | 1.16e-03 |
| 0.2 | 2.14e-02 (4.3x) | 1.40e-03 (8.4x) | 6.94e-05 (16.7x) |
| 0.1 | 5.17e-03 (4.1x) | 1.71e-04 (8.2x) | 4.25e-06 (16.3x) |
| 0.05 | 1.27e-03 (4.1x) | 2.11e-05 (8.1x) | 2.63e-07 (16.2x) |

Predicted fall factors: 4x, 8x, 16x for orders 1, 2, 3.
<!-- /worked -->

The order-1 error falls by about $4$ each time $h$ is halved, the order-2 by about
$8$ and the order-3 by about $16$: the error of the order-$n$ polynomial is
proportional to $h^{n+1}$, one power of $h$ beyond the last term we kept. The
reason is a short argument. Let $r(h) = f(a + h) - p_n(h)$ be the error. By the
choice of the coefficients, $r$ and its first $n$ derivatives are $0$ at $h = 0$,
and since $p_n$ has degree $n$, its $(n+1)$-th derivative is $0$, so
$r^{(n+1)}(h) = f^{(n+1)}(a + h)$. Suppose $\lvert f^{(n+1)}\rvert \le M$ near $a$.
A quantity that starts at $0$ and changes at a rate of at most $s$ cannot be more
than $s\lvert h\rvert$ away from $0$ after a distance $h$. So the derivative
$r^{(n)}$, which starts at $0$ and has slope at most $M$, satisfies
$\lvert r^{(n)}(t)\rvert \le M\lvert t\rvert$. The next one down, $r^{(n-1)}$,
starts at $0$ with slope at most $M\lvert t\rvert$ and so stays within $M h^2/2$
(the area under the bound $M\lvert t\rvert$ from $0$ to $h$; the chapter on
integrals makes "area under a rate gives the change" exact). Repeating $n + 1$
times,

$$
\lvert f(a + h) - p_n(h)\rvert \le M\,\frac{\lvert h\rvert^{n+1}}{(n+1)!}.
$$

The error is of order $h^{n+1}$, and its size is set by the first dropped
derivative. The sharper statement is that the error is *approximately* the next
term, $\frac{f^{(n+1)}(a)}{(n+1)!}h^{n+1}$, as the next table confirms for $e^x$:

<!-- worked: multivariable.taylor_remainder_constant -->
Error at h = 0.01 against the first dropped term h^(n+1) / (n+1)!.

| order n | error | h^(n+1) / (n+1)! | ratio |
|---|---:|---:|---:|
| 1 | 5.02e-05 | 5.00e-05 | 1.0033 |
| 2 | 1.67e-07 | 1.67e-07 | 1.0025 |
| 3 | 4.18e-10 | 4.17e-10 | 1.0020 |
| 4 | 8.35e-13 | 8.33e-13 | 1.0016 |
<!-- /worked -->

**What Chapter 7 relied on.** The forward difference is
$D_+(h) = \big(f(x + h) - f(x)\big)/h$. Insert the order-3 expansion
$f(x + h) = f + f'h + \tfrac{1}{2}f''h^2 + \tfrac{1}{6}f'''h^3 + \dots$:

$$
D_+(h) = f'(x) + \tfrac{1}{2}f''(x)\,h + \tfrac{1}{6}f'''(x)\,h^2 + \dots,
$$

so the error is first order in $h$ with coefficient $f''/2$. For the central
difference, expand both $f(x + h)$ and $f(x - h)$ (replace $h$ by $-h$: the odd
powers change sign) and subtract. The even terms cancel, leaving
$2f'h + \tfrac{1}{3}f'''h^3$, which divided by $2h$ is $f' + \tfrac{1}{6}f'''h^2$:
second order. These are the two error formulas that Chapter 7 took on trust. For
$e^x$ at $x = 1$ and $h = 0.1$, adding the terms one at a time reproduces the
measured error of the forward difference:

<!-- worked: multivariable.taylor_explains_numdiff -->
| estimate of the forward-difference error at h = 0.1 | value |
|---|---:|
| measured: (e^(1.1) - e) / 0.1 - e | 0.140560 |
| first term f'' h / 2 | 0.135914 |
| plus f''' h^2 / 6 | 0.140445 |
| plus f'''' h^3 / 24 | 0.140558 |
<!-- /worked -->

Each further term removes most of what is left: the gap between the measured error and the sum falls from $4.6\times10^{-3}$ to $1.2\times10^{-4}$ to $2\times10^{-6}$.

**Many variables.** To expand a function of $\mathbf{w}$ about $\mathbf{p}$ by a
step $\mathbf{d}$, reduce to one variable. The function
$\varphi(t) = L(\mathbf{p} + t\mathbf{d})$ of the distance $t$ along the line has
$\varphi(0) = L(\mathbf{p})$, $\varphi'(0) = \nabla L\cdot\mathbf{d}$ (the
directional derivative, unnormalised) and $\varphi''(0) = \mathbf{d}^\top\mathbf{H}\mathbf{d}$
(the curvature formula). The one-variable expansion of order $2$ at $t = 1$ is then

$$
\boxed{\;L(\mathbf{p} + \mathbf{d}) \approx L(\mathbf{p}) + \nabla L(\mathbf{p})\cdot\mathbf{d} + \tfrac{1}{2}\,\mathbf{d}^\top\mathbf{H}(\mathbf{p})\,\mathbf{d}\;}
$$

with an error that is third order in the size of $\mathbf{d}$. The three terms are
a number, a slope and a curvature: the height at $\mathbf{p}$, the tilted plane,
and the bowl. For a quadratic function such as our loss, all third derivatives
are zero, so the expansion is not an approximation but an identity:

<!-- worked: multivariable.quadratic_taylor_exact -->
Expansion about p = (0.5, 1.5): L(p) = 2.5000, g = (-3.00, -7.00).

| step d | L(p + d) exactly | first order | second order |
|---|---:|---:|---:|
| (1.0, 0.0) | 0.500000 | -0.500000 | 0.500000 |
| (-0.4, 0.3) | 1.730000 | 1.600000 | 1.730000 |
| (2.0, -1.5) | 10.250000 | 7.000000 | 10.250000 |
<!-- /worked -->

The second-order column equals the exact score, even for a step as long as
$(2, -1.5)$, while the first-order plane is off by three units. Setting
$\mathbf{d} = -t\mathbf{g}$ in the formula gives exactly the quadratic
$\varphi(t)$ of the last section, and so the best-step formula. For a function
that is not quadratic the expansion is local. Take $f = e^x\cos y$ about the
origin, where the gradient is $(1, 0)$ and the Hessian is $\mathrm{diag}(1, -1)$,
so that the expansion is $1 + x + \tfrac{1}{2}(x^2 - y^2)$. The curvature is
positive along $x$ and negative along $y$, which makes the origin a *saddle*, not
a bowl: a **saddle** is a point where the slope is zero but the surface curves up
along one direction and down along another, like the middle of a mountain pass. Step $t(0.5, 1)$ and halve $t$:

<!-- worked: multivariable.taylor2_errors -->
f(x, y) = e^x cos y about the origin, step t (0.5, 1). Gradient (1, 0), Hessian diag(1, -1); second-order polynomial 1 + x + (x^2 - y^2)/2.

| t | f exactly | error of first order | error of second order |
|---|---:|---:|---:|
| 0.4 | 1.124986 | 7.50e-02 | 1.50e-02 |
| 0.2 | 1.083141 | 1.69e-02 (4.4x) | 1.86e-03 (8.1x) |
| 0.1 | 1.046019 | 3.98e-03 (4.2x) | 2.31e-04 (8.1x) |
| 0.05 | 1.024034 | 9.66e-04 (4.1x) | 2.88e-05 (8.0x) |
<!-- /worked -->

Halving $t$ cuts the error of the first-order plane by about $4$ and the error of
the second-order model by about $8$: the same pattern, orders $2$ and $3$, as in
one variable.

## Conditioning: why a long valley is slow

The first look ended with two puzzles: a step size that crawled and one that
diverged. The quadratic model explains both exactly. For our loss the gradient is
$\nabla L(\mathbf{w}) = \mathbf{H}(\mathbf{w} - \mathbf{w}^\star)$ (the Taylor
expansion about the minimum $\mathbf{w}^\star$, with a zero gradient there,
differentiated). The step rule $\mathbf{w}\leftarrow\mathbf{w} - \eta\nabla L$
then acts on the error $\mathbf{e} = \mathbf{w} - \mathbf{w}^\star$ as

$$
\mathbf{e}\;\leftarrow\;\mathbf{e} - \eta\,\mathbf{H}\mathbf{e} = (\mathbf{I} - \eta\mathbf{H})\,\mathbf{e}.
$$

Write the error as a sum of its parts along the two curvature directions,
$\mathbf{e} = a\,\mathbf{v}_1 + b\,\mathbf{v}_2$, where $\mathbf{H}\mathbf{v}_i = \lambda_i\mathbf{v}_i$.
Then $\mathbf{H}\mathbf{e} = a\lambda_1\mathbf{v}_1 + b\lambda_2\mathbf{v}_2$, and each
part is simply scaled:

$$
a \leftarrow (1 - \eta\lambda_1)\,a, \qquad b \leftarrow (1 - \eta\lambda_2)\,b.
$$

Two independent one-knob problems, each shrinking by its own factor per step. For
the error to die along both directions, both factors must have absolute value
below $1$. The steep direction, $\lambda_1 = 11.66$, sets the limit
$\eta < 2/\lambda_1 = 0.1716$: above it, $\lvert 1 - \eta\lambda_1\rvert > 1$ and
that part of the error is multiplied by more than $1$ in size, with alternating
sign, at every step. That is the divergence of $\eta = 0.18$ in the first look. The
flat direction, $\lambda_2 = 0.343$, has factor $1 - 0.343\eta$, which is within
$0.06$ of $1$ for every allowed $\eta$, so the error along the valley floor shrinks
slowly. That is the crawl of $\eta = 0.05$. Start with one unit of error in each
direction (the start is $\mathbf{w}^\star + \mathbf{v}_1 + \mathbf{v}_2$ for the two unit
eigenvectors, at distance $\sqrt2$ from the minimum) and count the steps $n$ until
the distance $\lVert\mathbf{w}_n - \mathbf{w}^\star\rVert$ is below $0.001$. The best step size is
exactly $2/(\lambda_1 + \lambda_2) = 2/12 = 1/6$, which the table prints as $0.1667$:

<!-- worked: multivariable.step_size_limit -->
Curvatures lambda1 = 11.6569, lambda2 = 0.3431; 2 / lambda1 = 0.1716; the step size 2/(lambda1 + lambda2) = 2/12 = 1/6 exactly (printed rounded). Start w = w* + v1 + v2 with v1, v2 the unit eigenvectors of H (distance sqrt 2 = 1.414); each step w <- w - eta grad L(w); steps = the first count n with distance |w_n - w*| < 0.001. 'never' means the distance exceeds 10^12 or 100000 steps pass.

| step size eta | |1 - eta lambda1| | |1 - eta lambda2| | worst factor per step | steps needed |
|---|---:|---:|---:|---:|
| 0.0200 | 0.7669 | 0.9931 | 0.9931 | 1004 |
| 0.0500 | 0.4172 | 0.9828 | 0.9828 | 400 |
| 0.1000 | 0.1657 | 0.9657 | 0.9657 | 198 |
| 1/6 (best, 0.1667) | 0.9428 | 0.9428 | 0.9428 | 124 |
| 0.1700 | 0.9817 | 0.9417 | 0.9817 | 374 |
| 0.1800 | 1.0982 | 0.9382 | 1.0982 | never (diverges) |
<!-- /worked -->

A step size whose worst factor is above $1$ never reaches the target, shown as "never" (the distance has grown past $10^{12}$ or $100000$ steps have passed). Small step sizes are slow because the flat direction takes many steps; step sizes
near the limit are slow too, or fail outright, because the steep direction
oscillates; the best step size balances the two factors, $1 - \eta\lambda_2 =
-(1 - \eta\lambda_1)$, giving $\eta^\star = 2/(\lambda_1 + \lambda_2)$ and the
shrink factor
$(\lambda_1 - \lambda_2)/(\lambda_1 + \lambda_2) = (\kappa - 1)/(\kappa + 1)$ where
$\kappa = \lambda_1/\lambda_2$ is the ratio of the largest to the smallest
curvature, called the **condition number** of the bowl. Even the best step size is
only as good as $\kappa$ allows:

<!-- worked: multivariable.conditioning_compare -->
| bowl | curvatures | ratio (condition number) | best step 2/(l1 + l2) | shrink factor (k-1)/(k+1) | steps until distance < 0.001 |
|---|---:|---:|---:|---:|---:|
| round bowl | 3.000 and 3.000 | 1.00 | 0.3333 | 0.0000 | 1 |
| mild | 4.000 and 1.000 | 4.00 | 0.4000 | 0.6000 | 15 |
| this chapter's bowl | 11.657 and 0.343 | 33.97 | 0.1667 | 0.9428 | 124 |
| severe | 10.000 and 0.010 | 1000.00 | 0.1998 | 0.9980 | 3628 |
<!-- /worked -->

A round bowl ($\kappa = 1$) is solved in one step. Our bowl, $\kappa = 33.97 \approx 34$, needs
$124$ steps at the best step size, and a bowl with a ratio of $1000$ needs
thousands. The picture shows what that looks like on the contour map: the path
jumps back and forth across the narrow direction, a little less each time, while
creeping along the valley.

![The elongated bowl of the chapter with forty steps of size 0.15 starting at (0, 3.5). The path zigzags from side to side across the narrow direction, each swing smaller than the last, while advancing slowly along the valley floor toward the minimum at (1, 2), which it has not yet reached.](img/conditioning-path.svg)

<!-- worked: multivariable.zigzag_path -->
Steps of size 0.15 from (0.0, 3.5).

| step | w | L(w) | distance to the minimum (1, 2) |
|---|---:|---:|---:|
| 0 | (0.0000, 3.5000) | 6.25000 | 1.8028 |
| 1 | (-0.6000, 1.8500) | 3.63250 | 1.6070 |
| 2 | (-0.0300, 3.0350) | 2.15282 | 1.4602 |
| 3 | (-0.3420, 2.1005) | 1.31198 | 1.3458 |
| 4 | (0.0003, 2.7549) | 0.83025 | 1.2527 |
| 5 | (-0.1528, 2.2223) | 0.55080 | 1.1740 |
| 10 | (0.2054, 2.3891) | 0.15167 | 0.8848 |
| 20 | (0.5202, 2.2020) | 0.04655 | 0.5206 |
| 30 | (0.7165, 2.1176) | 0.01616 | 0.3069 |
| 40 | (0.8329, 2.0692) | 0.00562 | 0.1809 |
<!-- /worked -->

After $40$ steps the path is still $0.18$ away from the minimum. This is the
reason that optimisation of real models is not just "move against the gradient":
a loss with elongated contours punishes every simple rule for choosing step sizes.
Volume II asks how to choose $\eta$ when the Hessian is known. Volume III
chapters on optimizers study what to do when it is not, with methods that
average gradients over past steps, or scale each knob by its own step size, in order to cope
with a large $\kappa$. The Hessian itself has one entry per pair of knobs, and for
a model with millions of knobs that is millions squared entries, which cannot be
stored; every method that wants curvature information has to get it indirectly.

## Making it work

- **The gradient has the shape of the parameters.** A parameter of shape $[n, k]$
  has a gradient of shape $[n, k]$; if a derivative comes out with another shape,
  a transpose or a sum is missing.
- **Jacobians are outputs by inputs.** Fix the convention and check each product
  by its shapes: $[p, m]\,[m, n]$. Wrong-order products rarely have compatible
  shapes, and when they do, the shapes of the result tell on them.
- **Do not form a Jacobian you do not need.** The product of a long chain of
  Jacobians has the cost of its bracketing. For a scalar loss, multiply from the
  loss end and keep a vector.
- **Check a gradient entry by entry, and then in random directions.** A central
  difference per knob costs $2n$ evaluations of the loss; a difference along one
  random direction costs two and checks the dot product $\nabla L\cdot\mathbf{v}$.
  Use the first on small tensors, the second on large ones, and read the
  next section before trusting the second alone.
- **Choose the finite-difference step for the number format.** A central
  difference in 32-bit arithmetic has a best step near $4\times10^{-3}$ (Chapter
  7). A difference along a direction $\mathbf{v}$ of $n$ entries of size $1$ moves
  every entry at once, so the step it takes has length $h\sqrt{n}$.
- **Curvature, not slope, limits the step.** The slope tells which way to go; the
  largest curvature caps how far: $\eta < 2/\lambda_{\max}$ on a quadratic.

## Where this appears in brain

**The backward pass is a product of transposed Jacobians.** For a linear layer,
the derivative of the loss with respect to the input is $\mathbf{W}^\top$ times
the derivative with respect to the output, which is what we derived. The kernel
`crates/kernels/wgsl/matmul_dx.wgsl` documents it in its header for the batched,
row-per-example layout: `dX[m, k] = sum_n dY[m, n] * W[n, k]`, the matrix-vector
product with $\mathbf{W}^\top$ for each row $m$ of the batch. Its neighbour
`crates/kernels/wgsl/matmul_dw.wgsl` computes
`dW[n, k] += sum_m dY[m, n] * X[m, k]`, the gradient with respect to the weight
matrix itself, and the `+=` there is the sum over routes of the chain-rule
section (a weight used twice collects both contributions). The chapter on
backpropagation in Volume III builds the whole backward pass from these
pieces: [Backpropagation Derived from Scratch](../v03-neural-networks/04-backpropagation.md).

**The gradient checker perturbs knobs one at a time, or along a direction.**
`crates/gradcheck/src/lib.rs` has the two checks of the previous section. Its
`directional_check` picks a random direction $\mathbf{v}$ of $\pm 1$ entries per
parameter tensor and compares the analytic $\nabla L\cdot\mathbf{v}$, the formula
of this chapter's directional-derivative section, with a central difference
$(L(\mathbf{w} + h\mathbf{v}) - L(\mathbf{w} - h\mathbf{v}))/2h$. Its
`elementwise_check` perturbs one entry at a time and compares each partial
derivative, at a cost of two evaluations of the loss per entry.

**Story: a gradient with a third missing passed the check.** In August 2026,
commit `4171bccb0` added `elementwise_check` after finding that the directional
check could not see a particular bug. The doc comment of `directional_check`
(which the commit corrected) records it: in T5, a weight shared across blocks had its
gradient contributions summed across blocks by a fold, and deleting the fold left
$33\%$ of that gradient missing, yet the directional check reported a relative
error of $6.2\times10^{-4}$ for it and passed. The check keeps the best of several
random directions, and the dot product of a gradient with a $\pm 1$ direction is a
single number in which a missing share can be small by luck, and the best-of-$n$
selection picks the luckiest. Knowledge entry #187 later saw the same blindness on
a second weight and fixed the practice: a weight read at more than one site gets a
per-entry check. The mechanism is the one of this chapter: a directional derivative
is the gradient contracted onto one direction, a single number, while the
per-entry check compares the whole vector. A miniature reproduces it. For a
squared-error loss with $24$ knobs and $6$ examples, drop the contribution of one
example from the "analytic" gradient (a mis-folded share), then apply a best-of-4
random $\pm1$ directional check and a per-entry check with brain's tolerances
($4\times10^{-3}$ absolute, $8\times10^{-2}$ relative), over $1000$ random problems:

<!-- worked: multivariable.gradcheck_toy -->
One generator (Volume I, chapter 17), seed 10372158813470712557, draws the 1000 problems one after another: for each, X row by row, then t, then w, all with normal(), then the four directions, each entry +1 when integer(2) is 1 and -1 otherwise (all four are always drawn). The loss is L(w) = 1/2 sum_k (x_k . w - t_k)^2; the "analytic" gradient omits example 0 (the first row of X); the numeric value is a central difference of the true loss (all six examples) with h = 1e-4. A comparison passes when |a - n| <= atol + rtol max(|a|, |n|) with atol = 4e-3 and rtol = 8e-2, a the analytic and n the numeric value (this is not the close(a, b, ...) of the pseudocode page, which scales by |b| only); the directional check accepts the bug when any of its four directions passes; the per-entry check compares each of the 24 entries with the same gate and a central difference per entry (h = 1e-4).

| check | result over 1000 random problems |
|---|---:|
| relative size of the missing part, mean of ||missing|| / ||gradient|| | 0.310 |
| best-of-4 random +-1 directions: bug accepted (check passes) | 579 of 1000 |
| per-entry check: at least one entry fails | 984 of 1000 |
| per-entry check: mean share of the 24 entries that fail | 0.729 |
<!-- /worked -->

The missing part is about a third of the gradient in size, and the directional
check lets the bug through for more than half of the problems, while the per-entry
check fails for almost all of them. (The toy is an illustration of the mechanism,
not a measurement of brain's checker.)

**Curvature without the Hessian.** `crates/modelopt/src/sensitivity.rs` estimates
how much the loss would rise if a weight tensor were stored at lower precision
(a tier of rounding). Its documentation says that it uses the second-order Taylor
expansion of the loss around the trained weights, in which the increase is
$\tfrac{1}{2}\sum_j F_j(w_j - Q(w_j))^2$ for rounding $Q$: the quadratic term of this
chapter's expansion with the Hessian replaced by a diagonal estimate $F$ built
from squared gradients, and the first-order term dropped because the average
gradient is near zero at a trained minimum. The step is the one derived above,
$\tfrac{1}{2}\mathbf{d}^\top\mathbf{H}\mathbf{d}$ with $\mathbf{d}$ the rounding
error, and the Hessian is never formed: only its diagonal is approximated, because a full
matrix of a model's size cannot be stored. The module arrived with the
mixed-precision search in commit `e68da7686` (November 2026).

## Check your understanding

Answers are at the end of the chapter.

1. **Explain.** At a point, $\partial L/\partial w_1 = -3$ and $\partial L/\partial w_2 = -7$. A friend says the score therefore falls by $10$ per unit of distance along the diagonal direction $(1, 1)$. What is wrong, and what is the rate per unit of distance along that direction?
2. **Find the error.** "The gradient of $f(x, y) = x^2 + y^2$ at $(3, 4)$ is $(6, 8)$. Since $f$ is smallest at the origin, the gradient points toward the origin." Where does the argument go wrong?
3. **Predict.** A function $\mathbf{f}:\mathbb{R}^5\to\mathbb{R}^3$ is followed by a scalar function $g:\mathbb{R}^3\to\mathbb{R}$. What are the shapes of $\mathbf{J}_{\mathbf{f}}$, of $\mathbf{J}_g$ and of the Jacobian of the composition? What is the shape of $\nabla_{\mathbf{x}}(g\circ\mathbf{f})$ and how is it computed from $\nabla g$?
4. **Find the error.** "For $L = g(\mathbf{f}(\mathbf{x}))$ the Jacobian is $\mathbf{J}_{\mathbf{f}}\mathbf{J}_g$ because $\mathbf{f}$ is applied first." Use shapes to show it cannot be right.
5. **Predict.** The order-2 Taylor polynomial of $e^x$ at $0$ is $1 + x + x^2/2$. Roughly how large is its error at $x = 2$, and at $x = 0.2$? Use the error formula.
6. **Explain.** On the chapter's bowl, why does $\eta = 0.17$ converge (slowly) while $\eta = 0.18$ diverges, although the two differ by less than $6\%$?
7. **Find the error.** "A directional gradient check with one random direction passed, so the backward pass is correct." Which kind of bug can this pass, and what check would catch it?
8. **Predict.** A quadratic bowl has curvatures $100$ and $1$. What step size is best, what is the shrink factor per step, and roughly how many steps reduce the error by $1000$?

## Exercises

1. **[Calculate]** For $f(x, y) = x^2 y + \sin(xy)$, find $\nabla f$ and the Hessian by hand. At $(1, 0)$, evaluate the gradient and the directional derivative along $(3, 4)/5$, and state the direction of steepest ascent and its rate. Check by central differences.

2. **[Calculate]** For $f(x, y) = x^2 + 4y^2$, find the contour through $(2, 1)$ and its tangent direction there. Verify that the tangent is perpendicular to $\nabla f$, and compute $f$ at the point $(2, 1) + 0.01\,\hat{\mathbf{t}}$ to see the second-order change along the tangent.

3. **[Calculate]** Compute the Jacobian of the polar-to-Cartesian map $(r, \theta)\mapsto(r\cos\theta, r\sin\theta)$ at $(2, \pi/2)$, and the determinant of that $2\times2$ matrix. Use the linear approximation to estimate the change in the output when $r$ grows by $0.1$ and $\theta$ by $0.05$, and compare with the exact change.

4. **[Calculate]** Let $\mathbf{f}(x_1, x_2) = (x_1 + x_2,\; x_1 x_2)$ and $g(a, b) = a^2 + b$. Find $\mathbf{J}_{\mathbf{f}}$ and $\mathbf{J}_g$, multiply them to get the gradient of $g\circ\mathbf{f}$ at $(1, 2)$, and check against differentiating the composed formula directly.

5. **[Calculate]** Write the Taylor polynomials of orders $1$ to $3$ of $\ln(1 + x)$ at $0$, evaluate them at $x = 0.1$ and compare with the exact value. Then state the predicted error from the first dropped term and compare.

6. **[Calculate]** For $L(\mathbf{w}) = \tfrac{1}{2}\lVert\mathbf{X}\mathbf{w} - \mathbf{t}\rVert^2$ with $\mathbf{X} = \begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$ and $\mathbf{t} = (1, 2, 4)$, find $\nabla L$ and $\mathbf{H}$ at $\mathbf{w} = (0, 0)$, the extreme curvatures from $\det(\mathbf{H} - \lambda\mathbf{I}) = 0$, the step size limit $2/\lambda_{\max}$ and the best step along the gradient from $(0, 0)$.

7. **[Implement]** Implement the loss, its gradient formula and a central difference per knob, and reproduce `multivariable.gradient_check_points` (central differences with $h = 10^{-5}$; its last column is round-off, so compare it by order of magnitude). Then implement the directional derivative both ways and reproduce `multivariable.directional_scan`, including the three special directions (the perpendicular one is $(-g_2, g_1)/\lVert\mathbf{g}\rVert$).

8. **[Implement]** Implement the step rule $\mathbf{w}\leftarrow\mathbf{w} - \eta\nabla L$ and reproduce `multivariable.first_steps` (both step sizes) and `multivariable.step_size_limit`, which also needs the two curvature numbers from the quadratic equation of the Hessian section and the exact step size $1/6$ (not the rounded $0.1667$).

9. **[Implement]** Implement the Jacobian by central differences, one input at a time (steps $h = 10^{-6}$ for `jacobian_2x2`, $10^{-3}$ for `linear_layer_jacobian`), and use it to reproduce `multivariable.jacobian_2x2` and `multivariable.linear_layer_jacobian`. Then build the three-link chain with a matrix product and reproduce `multivariable.chain_jacobian_product` and `multivariable.chain_transposed`.

10. **[Implement]** Implement Taylor polynomials of $e^x$ and reproduce `multivariable.taylor_exp_partial_sums` and `multivariable.taylor_error_order`. Then implement the second-order multivariable model of $e^x\cos y$ and reproduce `multivariable.taylor2_errors`.

11. **[Investigate]** On the quadratic with curvatures $1$ and $\kappa$, run the step rule at the best step size for $\kappa = 1, 2, 5, 10, 100, 1000$ and record the steps needed to reduce the error by $1000$. How does the count grow with $\kappa$? Compare with $\ln(1000)/\ln\frac{\kappa+1}{\kappa-1}$.

12. **[Investigate]** Reproduce `multivariable.gradcheck_toy` and vary the number of random directions (1, 4, 16, 64) and the share of the gradient that is missing (drop one, two or three of the six examples). When does the directional check become reliable? What does that say about the cost of the two checks?

## Summary

We asked which way to move many knobs to lower a score, and answered in layers.
The **partial derivative** is the one-knob slope with the others frozen. The
**gradient** collects them into a vector of the same shape as the knobs. The
slope in any unit direction $\mathbf{u}$, the **directional derivative**, is
$\nabla L\cdot\mathbf{u}$, which we derived by splitting a step into one-knob
steps; Cauchy-Schwarz then shows that the gradient points in the direction of
steepest ascent with rate $\lVert\nabla L\rVert$, and since a contour line has
constant $L$, the gradient is perpendicular to it. For a function with several
outputs the **Jacobian**, of shape outputs by inputs, is the matrix of all the partial
derivatives and the best linear approximation near a point; a linear layer
$\mathbf{y} = \mathbf{W}\mathbf{x}$ has the Jacobian $\mathbf{W}$. The
**multivariable chain rule** says that the Jacobian of a composition is the
product of the Jacobians, derived from the linear approximations and their
remainders. For a scalar loss it is evaluated from the loss end with transposed
Jacobians, so the gradient comes out of a pass over vectors. The **Hessian** is
the matrix of second partial derivatives; the curvature in the direction
$\mathbf{u}$ is $\mathbf{u}^\top\mathbf{H}\mathbf{u}$. The **Taylor expansion**,
derived first in one variable with an error of the order of the first dropped
term (which Chapter 7's error formulas needed), is in several variables
$L(\mathbf{p}) + \nabla L\cdot\mathbf{d} + \tfrac{1}{2}\mathbf{d}^\top\mathbf{H}\mathbf{d}$,
exact for a quadratic. Applied to a quadratic bowl it shows that the step size is
capped by the largest curvature and the progress along the flat direction is
set by the smallest, so a bowl with elongated contours makes simple stepping slow.

The limits are the limits of calculus with a local view: everything here is a
statement about a neighbourhood of one point, and the steepest direction depends
on how length is measured. The model of the loss as a quadratic is exact only for
squared errors of linear models; real networks have loss landscapes with many
valleys and saddles. And we have never integrated: sums over examples, or over
time, are still sums of finitely many terms. The next chapter, *Integrals and
Differential Equations*, adds the inverse operation of the derivative, adding up
infinitely many small pieces, and uses the gradient to define the continuous path
that stepping against it approximates.

## Key terms

- **partial derivative**: the derivative of a function of several variables with respect to one of them while all the others are held fixed, $\partial L/\partial w_j$. <!-- concept: partial-derivative -->
- **gradient**: the vector of all the partial derivatives of a scalar function, the same shape as its inputs; it points in the direction of steepest ascent and its length is the steepest rate of change. <!-- concept: gradient -->
- **directional derivative**: the rate of change of a function per unit distance in a chosen unit direction $\mathbf{u}$, equal to $\nabla L\cdot\mathbf{u}$. <!-- concept: directional-derivative -->
- **Jacobian**: the matrix of shape outputs by inputs whose entry $(i, j)$ is $\partial f_i/\partial x_j$; it is the best linear approximation of a vector function near a point. <!-- concept: jacobian -->
- **Hessian**: the symmetric matrix of second partial derivatives of a scalar function; $\mathbf{u}^\top\mathbf{H}\mathbf{u}$ is the curvature along the unit direction $\mathbf{u}$. <!-- concept: hessian -->
- **Taylor expansion**: the polynomial that matches a function's value and first derivatives at a point, such as $f(a) + f'(a)h + \tfrac{1}{2}f''(a)h^2$, whose error is of the order of the first term dropped. <!-- concept: taylor-expansion -->
- **multivariable chain rule**: the Jacobian of a composition of vector functions is the product of their Jacobians, the outer one evaluated at the inner function's output. <!-- concept: multivariable-chain-rule -->
- **condition number**: for a bowl-shaped function, the ratio of its largest to its smallest curvature; a large ratio means long, narrow contours and slow progress for a fixed step size. The floating-point chapter extends the idea to how strongly a computation amplifies small errors in its inputs. <!-- concept: condition-number -->

## Answers

**Before you begin**

1. $L'(w) = 2(2w - 3)\cdot 2 = 4(2w - 3)$, so $L'(0.5) = 4(1 - 3) = -8$.
2. $2(3x + 1)\cdot 3 = 6(3x + 1)$, which is $24$ at $x = 1$.
3. $\mathbf{a}\cdot\mathbf{b} = 3\cdot2 + 4\cdot1 = 10$ and $\lVert\mathbf{a}\rVert = 5$. Cauchy-Schwarz: $\lvert\mathbf{a}\cdot\mathbf{u}\rvert\le\lVert\mathbf{a}\rVert = 5$ for a unit $\mathbf{u}$, reached when $\mathbf{u}$ points along $\pm\mathbf{a}$.
4. $\mathbf{W}\mathbf{x} = (1\cdot2 + 2\cdot1,\; 0\cdot2 + 3\cdot1) = (4, 3)$. For $\mathbf{W}$ of shape $[m, n]$, $\mathbf{x}$ has $n$ entries and the result has $m$.
5. $\mathbf{A}\mathbf{B}$ is $[2, 4]$ and $\mathbf{B}^\top\mathbf{A}^\top$ is $[4, 2]$; they are transposes of each other, $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$.
6. $1 + 1 + 0.5 + 0.1667 = 2.6667$, close to $e = 2.71828$ (Chapter 1).

**Check your understanding**

1. The rate per unit of distance is the dot product of the gradient with a *unit* direction. The vector $(1, 1)$ has length $\sqrt{2}$, so $-3 - 7 = -10$ is the rate per unit of the parameter $t$ in $\mathbf{p} + t(1, 1)$, a walk of $\sqrt{2}$ in distance for each unit of $t$. Per unit of distance the rate is $(-3, -7)\cdot(1, 1)/\sqrt{2} = -10/\sqrt{2} = -7.07$, the $45$ degree row of the directional table at the same point.
2. The gradient points uphill. At $(3, 4)$ it is $(6, 8)$, which points away from the origin; the negative gradient $(-6, -8)$ points toward the minimum. (Here, because the contours are circles, the negative gradient points straight at the minimum; in general it does not, as the elongated bowl shows.)
3. $\mathbf{J}_{\mathbf{f}}$ is $[3, 5]$, $\mathbf{J}_g$ is $[1, 3]$ and the composition has $[1, 3]\,[3, 5] = [1, 5]$. The gradient has the shape of $\mathbf{x}$, $[5]$, and is $\mathbf{J}_{\mathbf{f}}^\top\nabla g$, a $[5, 3]$ matrix times a $3$-vector.
4. $\mathbf{J}_{\mathbf{f}}$ is $[m, n]$ and $\mathbf{J}_g$ is $[p, m]$. The product $\mathbf{J}_{\mathbf{f}}\mathbf{J}_g$ needs $n = p$ and in general does not exist; it would also give an $[m, m]$ result, not the $[p, n]$ of $L$'s Jacobian. The outer function's Jacobian comes first, in the order of the composition as it is written: $\mathbf{J}_g\mathbf{J}_{\mathbf{f}}$.
5. The first dropped term is $x^3/6$: at $x = 2$ about $8/6 = 1.33$ (the actual error is $e^2 - 5 = 2.39$, because $x = 2$ is not small and the next terms matter, as the figure shows); at $x = 0.2$ about $0.008/6 = 0.0013$ (the actual error is $0.00140$).
6. Each error component is multiplied by $1 - \eta\lambda$ per step. For $\lambda_1 = 11.66$ the factor is $1 - 0.17\cdot11.66 = -0.98$ at $\eta = 0.17$ (magnitude below $1$) and $1 - 0.18\cdot11.66 = -1.10$ at $\eta = 0.18$ (above $1$). The threshold is $2/\lambda_1 = 0.1716$, between them; the behaviour changes sharply at it.
7. A single direction, or the best of a few, tests the gradient only contracted onto one vector, a single number. A gradient with a share missing can contract to a small error by luck; in the toy, more than half of the problems passed. A per-entry check, which compares each partial derivative, catches it.
8. $\eta^\star = 2/(100 + 1) = 0.0198$, shrink factor $99/101 = 0.980$, steps $\ln 1000/\ln(101/99) = 345$.
