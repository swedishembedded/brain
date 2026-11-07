<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Integrals and Differential Equations

*How do we add up infinitely many small pieces, and how do we follow a system that changes continuously?*

## The problem

A car's speedometer is read once a second. During the first three seconds of a
test drive it showed these speeds, in metres per second:

<!-- worked: integrals.speed_readings -->
| time t (s) | speed v(t) (m/s) |
|---|---:|
| 0 | 0.0 |
| 1 | 3.0 |
| 2 | 4.0 |
| 3 | 3.0 |

Left readings (t = 0, 1, 2): 7.0 m. Right readings (t = 1, 2, 3): 10.0 m.
<!-- /worked -->

How far did the car travel? The reading at $t = 0$ is the speed at that instant,
not the speed for the whole second, so we have to choose how to use it. If we
pretend the car held each reading for the following second, it travelled
$0 + 3 + 4 = 7$ metres. If we pretend it held each reading for the *preceding*
second, it travelled $3 + 4 + 3 = 10$ metres. The true distance lies somewhere
near both and equal to neither, because the car was speeding up and slowing down
between the readings.

Chapter 7 went from a position to its rate of change: from distance travelled to
speed. This question runs the other way, from a rate to the accumulated total.
We know how fast the quantity changes at every moment and want to know how much
it changed. The same question appears wherever a continuous quantity must be
summed: the total probability under a bell curve, the average of a function over
all its inputs, the final state of a system whose rate of change we know. This
chapter builds the operation, called the integral, from the smallest case above,
and then uses it for the second half of the question: if the rate of change of a
quantity depends on the quantity itself, how do we follow it forward in time?
That is a differential equation, and the simplest method for following it is
the one that, in disguise, trains every neural network.

## Before you begin

Six questions on ideas from earlier chapters, to answer from memory. The answers
are at the end of the chapter. If one is hard, go back to the chapter named
before reading on.

1. (Chapter 7, derivative.) What is the derivative of $x^3$, and of $e^{2x}$?
2. (Chapter 7, limit.) What number does $(e^h - 1)/h$ approach as $h \to 0$, and why is that the reason $\tfrac{d}{dx}e^x = e^x$?
3. (Chapter 1, summation notation.) Evaluate $\sum_{i=0}^{3} 0.5\,i^2$.
4. (Chapter 8, gradient.) For $L(w_1, w_2) = w_1^2 + 3w_2^2$, what is the gradient at $(1, 1)$, and in which direction does stepping *against* it move the point?
5. (Chapter 2, compounding.) A balance of 100 earns 5 % a year, added once a year. What is it after 2 years, and what number is multiplied in each year?
6. (Chapter 8, Taylor expansion.) What is the order-2 Taylor polynomial of $e^x$ at $0$, and what size does the error have at $x = h$ for small $h$?

## Adding up rectangles

We want a method, not a guess, for turning speed readings into a distance. Take
the readings every $\Delta t$ seconds, so there are $n = 3/\Delta t$ of them on
$[0, 3]$. Distance is speed times time when the speed is constant, so over one
interval of length $\Delta t$ we use a single reading and multiply. Using the
reading at the left end of each interval gives the **left sum**,

$$
S_n^{\text{left}} = \sum_{i=0}^{n-1} v(t_i)\,\Delta t, \qquad t_i = i\,\Delta t,
$$

and using the right end gives the right sum, with $v(t_{i+1})$ in place of $v(t_i)$.
Each term is the area of a rectangle of height $v(t_i)$ and width $\Delta t$, so
the sum is an area, and this is a **Riemann sum**. The speed of the opening
problem is $v(t) = t(4 - t)$, which gives exactly the readings above. The picture
shows six readings, that is $\Delta t = 0.5$:

![The curve v(t) = t(4 - t) for t from 0 to 3 with six rectangles whose heights are the speed at the left end of each half-second. The first four rectangles lie below the curve and the last two stick out above it, so the rectangles together cover 8.125 square units while the area under the curve is 9.](img/riemann-rectangles.svg)

The rectangles total $8.125$ metres and the curve encloses $9$. The rectangles on
the rising part of the curve (up to $t = 2$) lie below it; the last two stick out
above it, because the speed has started to fall. What happens when the readings
come more often?

<!-- worked: integrals.riemann_shrinking -->
Distance travelled in 3 seconds from n readings of v(t) = t (4 - t).

| readings n | width h | left sum | right sum | right minus left | average |
|---|---:|---:|---:|---:|---:|
| 3 | 1.0000 | 7.0000 | 10.0000 | 3.0000 | 8.5000 |
| 6 | 0.5000 | 8.1250 | 9.6250 | 1.5000 | 8.8750 |
| 12 | 0.2500 | 8.5938 | 9.3438 | 0.7500 | 8.9688 |
| 30 | 0.1000 | 8.8450 | 9.1450 | 0.3000 | 8.9950 |
| 300 | 0.0100 | 8.9849 | 9.0150 | 0.0300 | 9.0000 |
| 3000 | 0.0010 | 8.9985 | 9.0015 | 0.0030 | 9.0000 |
<!-- /worked -->

Each row of the table has the same two features. The left sum is below the right
sum, and the exact area, $9$, lies between them (for a curve that rises and then
falls this is not guaranteed, but it holds here). And the gap between them is
exactly $\Delta t\,(v(3) - v(0)) = 3\Delta t$: the two sums contain the same
terms except that the left sum has the reading $v(0)$ where the right sum has
$v(3)$, so they differ by one column of height $v(3) - v(0)$. The gap shrinks in
proportion to the width, and both sums are squeezed towards $9$. This is the same device as the
limit of Chapter 7: we never reach zero width, but we can say what the values
approach and how far from it we still are.

**The integral.** For a function $f$ on $[a, b]$ whose Riemann sums approach one
number as the width goes to zero, that number is the **integral**,

$$
\int_a^b f(t)\,dt \;=\; \lim_{\Delta t \to 0} \sum_{i=0}^{n-1} f(t_i)\,\Delta t .
$$

The symbol $\int$ is an elongated S for "sum", $dt$ stands for the width $\Delta t$
once it has become infinitesimal, and $a$ and $b$ are the limits of integration.
For $v(t) = t(4 - t)$ on $[0, 3]$ the integral is $9$. We will derive that number
exactly from a formula in the next section; for now the table is the evidence.
(Continuous functions have an integral; so do functions with a finite number of
jumps. That covers everything in this book.)

**Signed area.** A speed is never negative but a velocity is: a car reversing has
negative velocity. A rectangle of negative height contributes negative area, so the
integral counts area above the axis as positive and below as negative. Take the
velocity $v(t) = t - 2$, which is backwards for the first two seconds and forwards
for the next two:

<!-- worked: integrals.signed_area -->
Integral of v(t) = t - 2 (a velocity) over each interval.

| interval (s) | integral (m, signed) |
|---|---:|
| 0 to 2 | -2.0000 |
| 2 to 4 | 2.0000 |
| 0 to 4 | 0.0000 |

Integral of |v(t)| over 0 to 4: 4.0000 m (the distance actually driven).
<!-- /worked -->

The car ends where it started: the net change in position, the integral of the
velocity, is $0$. The distance it actually drove is the integral of the absolute
value, $4$ metres. This distinction matters whenever positive and negative
contributions can cancel; the average of a quantity that is sometimes above and
sometimes below zero is the example the next chapters keep returning to.

Three properties follow from the sum and cost nothing to check. The integral of
$c\,f + g$ is $c$ times the integral of $f$ plus the integral of $g$ (sums split);
$\int_a^c = \int_a^b + \int_b^c$ (areas add along the axis); and reversing the
limits flips the sign, $\int_b^a = -\int_a^b$ (walking the axis backwards).

## The fundamental theorem: the integral and the derivative undo each other

Evaluating limits of sums is tedious. Chapter 7 gave us derivatives of whole
formulas without ever returning to the limit; can we do the same here? We can,
and the reason is a relationship between the two operations.

Fix a starting point $a$ and define the **accumulated area**
$A(x) = \int_a^x f(t)\,dt$: how much area has built up by position $x$. For a
speed, $A$ is the distance travelled so far. How fast does it grow? Move $x$ a
little to $x + h$. The change $A(x + h) - A(x)$ is the area of a thin strip over
$[x, x + h]$. The strip is a sliver of height about $f(x)$ and width $h$, so its
area is about $f(x)\,h$ and the rate of growth of $A$ is about $f(x)$. Take the
area under $f(t) = t(4 - t)$ at $x = 1$ ($f(1) = 3$):

<!-- worked: integrals.strip_over_width -->
A(x) = area under v from 0 to x. Strip beside x = 1, v(1) = 3.

| strip width h | A(x + h) - A(x) | (A(x + h) - A(x)) / h |
|---|---:|---:|
| 1.0 | 3.666667 | 3.666667 |
| 0.5 | 1.708333 | 3.416667 |
| 0.1 | 0.309667 | 3.096667 |
| 0.01 | 0.030100 | 3.009967 |
| 0.001 | 0.003001 | 3.001000 |
<!-- /worked -->

The strip area divided by its width approaches the height, $3$. The argument in
general: on the strip, $f$ stays between its smallest value $m_h$ and its
largest value $M_h$ over $[x, x + h]$, so the strip's area is between
$m_h h$ and $M_h h$, and dividing by $h$,

$$
m_h \;\le\; \frac{A(x + h) - A(x)}{h} \;\le\; M_h .
$$

For a continuous $f$, both $m_h$ and $M_h$ approach $f(x)$ as $h \to 0$, so the
middle does too, and that middle quantity is the difference quotient of Chapter 7:

$$
A'(x) = f(x).
$$

This is the **fundamental theorem of calculus**: the rate of change of the
accumulated area is the height of the function. Accumulating and differentiating
undo each other.

**Evaluating an integral from an antiderivative.** Call $F$ an **antiderivative**
of $f$ if $F' = f$. The area function $A$ is one. Suppose $F$ is another. Then
$F - A$ has derivative $f - f = 0$ everywhere, and a function whose slope is zero
everywhere cannot go anywhere: it is a constant $c$. So $F(x) = A(x) + c$, and the
constant cancels in a difference. Since $A(a) = 0$,

$$
\int_a^b f(t)\,dt = A(b) = F(b) - F(a).
$$

Any antiderivative will do, and finding one is differentiation read backwards. The
rules of Chapter 7 give a short list:

- The power rule gives $\tfrac{d}{dx}\,\tfrac{x^{n+1}}{n+1} = x^n$, so $x^{n+1}/(n+1)$ is an antiderivative of $x^n$ for any $n \ne -1$.
- The derivative of $\log x$ is $1/x$, which covers the case the power rule misses, $n = -1$.
- By the chain rule $\tfrac{d}{dx}\,e^{kx} = k e^{kx}$, so $e^{kx}/k$ is an antiderivative of $e^{kx}$.
- A sum or a constant multiple of functions has the same combination of antiderivatives.

The opening distance is now one line. An antiderivative of $4t - t^2$ is
$2t^2 - t^3/3$, so $\int_0^3 t(4 - t)\,dt = (18 - 9) - 0 = 9$, as the table
suggested. Here are six more, each compared with the area computed by Simpson's
rule (a method described below; the point is that it does not use antiderivatives):

<!-- worked: integrals.antiderivative_table -->
| f(x) | interval | antiderivative F | F(b) - F(a) | area by Simpson, 2000 panels | abs difference |
|---|---:|---:|---:|---:|---:|
| x^2 | 0 to 2 | x^3 / 3 | 2.666667 | 2.666667 | 0.00e+00 |
| x^3 | 0 to 2 | x^4 / 4 | 4.000000 | 4.000000 | 0.00e+00 |
| 3 x^2 + 2 x | 0 to 2 | x^3 + x^2 | 12.000000 | 12.000000 | 0.00e+00 |
| e^x | 0 to 2 | e^x | 6.389056 | 6.389056 | 3.46e-14 |
| e^(-x) | 0 to 2 | -e^(-x) | 0.864665 | 0.864665 | 4.77e-15 |
| 1 / x | 1 to 2 | log x | 0.693147 | 0.693147 | 2.00e-15 |
<!-- /worked -->

The two columns agree to the last printed digit.

**Making the Taylor remainder exact.** Chapter 8 bounded the error of a Taylor
polynomial by the argument that "a quantity that starts at $0$ and changes at a
rate of at most $s$ cannot move more than $s\lvert h\rvert$", and said that the
integrals chapter would make "area under a rate gives the change" exact. The
fundamental theorem is that statement: $\int_0^t g'(s)\,ds = g(t) - g(0)$, the change
in $g$ is the area under its rate. Let $r(h) = f(a + h) - p_n(h)$ be the error of
the order-$n$ polynomial. By the choice of the coefficients, $r$ and its first $n$
derivatives vanish at $0$, and $r^{(n+1)}(t) = f^{(n+1)}(a + t)$, which we bound by
$M$. Apply the theorem to $r^{(n)}$, which starts at $0$:

$$
r^{(n)}(t) = \int_0^t f^{(n+1)}(a + s)\,ds, \qquad \text{so } \lvert r^{(n)}(t)\rvert \le \int_0^t M\,ds = M t .
$$

(A function that is never larger has an area that is never larger.) Apply it again
to $r^{(n-1)}$, which also starts at $0$ and whose rate is $r^{(n)}$:
$\lvert r^{(n-1)}(t)\rvert \le \int_0^t M s\,ds = M t^2/2$. Each application
integrates one more power, $M t^k/k!$, and after $n + 1$ of them

$$
\lvert f(a + h) - p_n(h)\rvert \;\le\; M\,\frac{\lvert h\rvert^{n+1}}{(n+1)!},
$$

which is the bound of Chapter 8, now derived without a hand wave. Carrying out the
$n + 1$ integrations without bounding, one gets the remainder itself as a single
integral, $r(h) = \int_0^h f^{(n+1)}(a + s)\,\frac{(h - s)^n}{n!}\,ds$. We state
that form and check it, rather than derive it, for $e^x$ with $n = 2$:

<!-- worked: integrals.taylor_remainder_exact -->
Remainder of the order-2 Taylor polynomial of e^x at 0, evaluated at x = h.

| h | remainder e^h - (1 + h + h^2/2) | integral of e^s (h - s)^2 / 2 over 0 to h | bound M h^3 / 3!, M = e^h | remainder / bound |
|---|---:|---:|---:|---:|
| 1.0 | 2.182818e-01 | 2.182818e-01 | 4.530470e-01 | 0.482 |
| 0.5 | 2.372127e-02 | 2.372127e-02 | 3.434836e-02 | 0.691 |
| 0.25 | 2.775417e-03 | 2.775417e-03 | 3.343816e-03 | 0.830 |
| 0.1 | 1.709181e-04 | 1.709181e-04 | 1.841952e-04 | 0.928 |
<!-- /worked -->

The remainder and the integral agree to every digit shown. The last column also
shows how good the bound is: the ratio of the true error to $M h^3/3!$ climbs to
$1$ as $h$ shrinks, because over a short interval the third derivative is nearly
constant, so the bound is nearly the exact value.

## Why integrals matter for learning

Most integrals in machine learning are not distances, and the antiderivative
trick does not apply to them. Three appear again and again in later chapters.

**Expectations.** The average of a quantity $g(x)$ over a continuous variable $x$
that occurs with density $p(x)$ is $\int g(x)\,p(x)\,dx$, the sum over all $x$ of
the value times how likely it is. This is the expectation of the probability chapter
that follows, and every loss that is "an average over the data distribution" is one.

**Normalising a density.** A density must integrate to $1$ over everything it
could be. The bell curve $e^{-x^2/2}$ does not: its area is a number we can
compute, and dividing by it makes a density.

<!-- worked: integrals.normalising_constant -->
Area under exp(-x^2 / 2). The limit is sqrt(2 pi) = 2.506628.

| interval | area (Simpson, 4000 panels) | sqrt(2 pi) minus area |
|---|---:|---:|
| -1 to 1 | 1.711249 | 7.95e-01 |
| -2 to 2 | 2.392576 | 1.14e-01 |
| -3 to 3 | 2.499861 | 6.77e-03 |
| -4 to 4 | 2.506469 | 1.59e-04 |
| -6 to 6 | 2.506628 | 4.95e-09 |
| -8 to 8 | 2.506628 | 3.11e-15 |
<!-- /worked -->

The areas approach $\sqrt{2\pi} = 2.5066$, so $p(x) = e^{-x^2/2}/\sqrt{2\pi}$ has
total area $1$. (That the limit is exactly $\sqrt{2\pi}$ is a classical result we
will use and not derive.) Note what is missing: there is no formula for an
antiderivative of $e^{-x^2/2}$ in terms of the functions of Chapter 2. It is
known that none exists. The only way to a number is a numerical method.

**Areas under curves as scores.** The area under a receiver operating
characteristic curve, a standard summary of how well a classifier ranks, is a sum
of trapezoids over a few measured points. Volume II (*Measuring Models*) defines it; the point here is
that evaluation code integrates.

So we need to approximate integrals from function values, with a known error.

## Numerical integration

**Numerical integration** approximates $\int_a^b f$ by a weighted sum of function
values. The left sum is one such rule. Its error is large: halving $\Delta t$ halves
it. We can do better by choosing which piece of curve each panel represents.

**Trapezoid rule.** The left rectangle replaces the curve over a panel by a
horizontal line at its left height. Replace it by the straight chord joining its
two ends instead. The area under a chord is a trapezoid, its width times the
average of the two heights:

$$
T_n = \sum_{i=0}^{n-1} \frac{f(t_i) + f(t_{i+1})}{2}\,\Delta t
= \Delta t\Big(\tfrac12 f(t_0) + f(t_1) + \dots + f(t_{n-1}) + \tfrac12 f(t_n)\Big).
$$

It is the average of the left sum and the right sum, which is the last column of
the shrinking-width table: at $\Delta t = 0.5$ it gives $8.875$, closer to $9$ than
either alone. With unequally spaced readings the same idea reads
$\sum \tfrac12(f_i + f_{i+1})(t_{i+1} - t_i)$.

**Simpson's rule.** Replace the curve over a pair of panels by the parabola
through three points. Shift the pair so its middle is at $s = 0$, with panel
width $\Delta t$ and heights $f_0, f_1, f_2$ at $s = -\Delta t, 0, \Delta t$. The
parabola $c_0 + c_1 s + c_2 s^2$ through them has $c_0 = f_1$ and
$c_2 = (f_0 - 2f_1 + f_2)/(2\Delta t^2)$. Its area over $[-\Delta t, \Delta t]$
is $2\Delta t\,c_0 + \tfrac23 \Delta t^3 c_2$ (the odd term integrates to zero by
symmetry), which simplifies to

$$
\frac{\Delta t}{3}\,\bigl(f_0 + 4 f_1 + f_2\bigr).
$$

Summing over the $n/2$ pairs gives **Simpson's rule**, with weights
$1, 4, 2, 4, \dots, 4, 1$ times $\Delta t/3$ ($n$ even). How do the three compare?
Integrate $e^x$ on $[0, 1]$, where the exact answer is $e - 1$:

<!-- worked: integrals.rules_error_order -->
Error of each rule for the area under e^x on [0, 1] (exact value e - 1 = 1.718282); in brackets, how much the error fell when the number of panels n doubled.

| panels n | left rectangles | trapezoids | Simpson |
|---|---:|---:|---:|
| 2 | 3.94e-01 | 3.56e-02 | 5.79e-04 |
| 4 | 2.06e-01 (1.9x) | 8.94e-03 (4.0x) | 3.70e-05 (15.7x) |
| 8 | 1.05e-01 (2.0x) | 2.24e-03 (4.0x) | 2.33e-06 (15.9x) |
| 16 | 5.31e-02 (2.0x) | 5.59e-04 (4.0x) | 1.46e-07 (16.0x) |
| 32 | 2.67e-02 (2.0x) | 1.40e-04 (4.0x) | 9.10e-09 (16.0x) |
| 64 | 1.34e-02 (2.0x) | 3.50e-05 (4.0x) | 5.69e-10 (16.0x) |

Predicted fall factors: 2x, 4x, 16x.
<!-- /worked -->

Doubling the number of panels cuts the left error by $2$, the trapezoid error by
$4$ and Simpson's by $16$. We say the rules have order $1$, $2$ and $4$: the error
shrinks like $\Delta t^p$. The reasons are Taylor expansions again. Over one panel
the left rectangle drops the slope term, so each panel is wrong by about
$\tfrac12 f'(t_i)\Delta t^2$, and over $n = (b - a)/\Delta t$ panels the errors add
to something proportional to $\Delta t$. The chord of the trapezoid is right to
first order, and what remains has a different sign on the two sides of each
panel and mostly cancels between neighbouring panels, leaving $\Delta t^2$.
Simpson's parabola is right through the quadratic term, and by symmetry through
the cubic one as well, leaving $\Delta t^4$. The last claim is visible exactly:

<!-- worked: integrals.rules_polynomials -->
Error (rule minus exact) of the area under x^k on [0, 1] with two panels.

| f(x) | left rectangles | trapezoids | Simpson |
|---|---:|---:|---:|
| x^0 | 0 (exact) | 0 (exact) | 0 (exact) |
| x^1 | -2.50e-01 | 0 (exact) | 0 (exact) |
| x^2 | -2.08e-01 | 4.17e-02 | 0 (exact) |
| x^3 | -1.88e-01 | 6.25e-02 | 0 (exact) |
| x^4 | -1.69e-01 | 8.12e-02 | 8.33e-03 |
<!-- /worked -->

With two panels, the left rule is exact only for constants, the trapezoid rule for
lines, and Simpson's rule for every polynomial up to degree three. (Degree three,
not two, is the lucky cancellation the symmetry gives.) A higher order is worth
function evaluations. To bring the error on $e^x$ below $10^{-6}$:

<!-- worked: integrals.evaluations_needed -->
Smallest power-of-two number of panels that brings the error on e^x over [0, 1] below 1e-6.

| rule | panels n | function evaluations | error |
|---|---:|---:|---:|
| left rectangles | 1048576 | 1048576 | 8.19e-07 |
| trapezoids | 512 | 513 | 5.46e-07 |
| Simpson | 16 | 17 | 1.46e-07 |
<!-- /worked -->

The left rule needs about a million evaluations, the trapezoid rule about five
hundred and Simpson's rule seventeen. That a smooth function can be integrated to
six digits from seventeen values is the reason this is a method and not a
curiosity. It holds only for smooth functions: the derivatives in the error terms
must exist and be moderate.

## Differential equations: a rate that depends on the state

Integration answers "given the rate as a function of time, what accumulates?" A
different situation is far more common. The rate depends on the quantity itself.
A cup of coffee loses heat faster the hotter it is. A bank balance grows faster
the larger it is. A learning rate that is cut by the same fraction every step
shrinks faster in absolute terms while it is large. Write $x(t)$ for the quantity;
its rate of change is a function of itself,

$$
\frac{dx}{dt} = f(t, x), \qquad x(0) = x_0 .
$$

This is an **ordinary differential equation** (ODE): an equation for an unknown
function $x(t)$ that contains the function and its derivative. "Ordinary" means
one independent variable, $t$, as opposed to partial derivatives in several. The
value $x_0$ at the start is the initial condition. The equation does not give $x$
at any time; it gives the slope at each point $(t, x)$ of the plane, and we must
find a curve that follows those slopes.

**The simplest case.** Let the rate be proportional to the quantity, with a decay
constant $k > 0$:

$$
\frac{dx}{dt} = -k\,x, \qquad x(0) = x_0 .
$$

The function $x_0 e^{-kt}$ solves it: by the chain rule its derivative is
$-k\,x_0 e^{-kt}$, which is $-k$ times the function. Is it the only solution? Suppose
$y$ is another and look at $y(t)e^{kt}$. Its derivative, by the product rule, is
$y' e^{kt} + k y\,e^{kt} = (y' + k y)e^{kt} = 0$, so $y e^{kt}$ is constant, equal to
its value $y(0)$ at $t = 0$, and $y = y(0)e^{-kt}$. The solution is unique and is
**exponential decay**: the continuous version of the schedule $\eta_0\gamma^{t}$ of
Chapter 2 with $\gamma = e^{-k}$, and the same shape as a cooling cup of coffee
(whose rate is proportional to the difference from room temperature, and so decays
the same way). With a negative sign replaced by a positive one, it is growth.

**Compounding as a limit, once more.** Chapter 1 met $e$ as the limit of
$(1 + 1/n)^n$, and Chapter 2 compounded a balance once a year. A balance that earns
at rate $r$ per year satisfies $dy/dt = r y$: the interest earned at each instant is
proportional to what is there. Compounding $n$ times a year multiplies the balance
by $1 + r/n$ every $1/n$ of a year. But that is "add the rate times the step",
$y \leftarrow y + (1/n)(r y)$, a step along the slope. Following the ODE by steps
along its slope and compounding are the same arithmetic:

<!-- worked: integrals.compounding_is_euler -->
Balance 100, rate 5 % a year, one year. The ODE dy/dt = 0.05 y has the exact answer 100 e^0.05 = 105.1271.

| Euler steps n | step h (years) | Euler result | 100 (1 + 0.05 / n)^n |
|---|---:|---:|---:|
| 1 | 1.000000 | 105.0000 | 105.0000 |
| 2 | 0.500000 | 105.0625 | 105.0625 |
| 12 | 0.083333 | 105.1162 | 105.1162 |
| 365 | 0.002740 | 105.1267 | 105.1267 |
| 1000000 | 0.000001 | 105.1271 | 105.1271 |
<!-- /worked -->

As the steps shrink, the result approaches the exact solution $100\,e^{rt}$ of the
ODE, and the two right-hand columns agree identically, since they compute the same
thing. Compounding yearly means a step of one year, which gives $105.0000$; the
true continuous growth gives $105.1271$.

## Euler's method

Most differential equations have no formula for their solution, so we need a way
to follow the slopes numerically. The idea is the one just used. At the current
point $(t, x)$ the equation gives the slope $f(t, x)$. Follow that slope for a
short time $h$, the **step size**, and then read the slope again:

$$
x_{i+1} = x_i + h\,f(t_i, x_i), \qquad t_{i+1} = t_i + h .
$$

This is **Euler's method**. It is the tangent-line approximation of Chapter 7 used
repeatedly, each time at the point where the previous tangent landed. As an
algorithm:

```text
euler(f: (real, real) -> real, t0: real, x0: real, h: real, n: int) -> array<real>[n + 1]
    require n >= 0
    x = zeros(real, [n + 1])
    x[0] = x0
    for i in 0 .. n - 1
        x[i + 1] = x[i] + h * f(t0 + i * h, x[i])
    return x
```

Take $dx/dt = -x$, $x(0) = 1$ and a step of $h = 0.5$. At each step the slope is
$-x_i$, so the new value is $x_i - 0.5\,x_i = 0.5\,x_i$:

<!-- worked: integrals.euler_by_hand -->
Euler's method for dx/dt = -x, x(0) = 1, step h = 0.5.

| step i | t | x_i | slope -x_i | exact e^(-t) | exact minus Euler |
|---|---:|---:|---:|---:|---:|
| 0 | 0.00 | 1.0000 | -1.0000 | 1.0000 | 0.00e+00 |
| 1 | 0.50 | 0.5000 | -0.5000 | 0.6065 | 1.07e-01 |
| 2 | 1.00 | 0.2500 | -0.2500 | 0.3679 | 1.18e-01 |
| 3 | 1.50 | 0.1250 | -0.1250 | 0.2231 | 9.81e-02 |
| 4 | 2.00 | 0.0625 |  | 0.1353 | 7.28e-02 |
<!-- /worked -->

The first value $0.5$ is below the exact $0.6065$, because the true curve bends
upward (its slope shrinks as $x$ falls) while the tangent keeps going straight
down. Each step makes the same kind of mistake and the errors carry forward.

**Local and global error.** The error of *one* step, started from the exact
value, is the **local error**. By the Taylor expansion of Chapter 8 the exact
solution is $x(t + h) = x(t) + h\,x'(t) + \tfrac12 h^2 x''(t) + \dots$, and the
Euler step reproduces the first two terms. So the local error is about
$\tfrac12 h^2 x''(t)$, of order $h^2$. To reach a fixed time $T$ we take $T/h$
steps; their errors accumulate, so the **global error** at time $T$ should be
about $(T/h)\cdot h^2$, of order $h$. Look at the decay equation up to $t = 2$,
halving $h$ each row:

<!-- worked: integrals.euler_error_order -->
Euler on dx/dt = -x from x(0) = 1 up to t = 2 (exact e^-2 = 0.135335). Local error: one step from the exact value at t = 0.

| step h | steps n | local error | h^2 / 2 | Euler x(2) | global error at t = 2 | global error / h |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 2 | 3.68e-01 | 5.00e-01 | 0.000000 | 1.35e-01 | 0.1353 |
| 0.5 | 4 | 1.07e-01 | 1.25e-01 | 0.062500 | 7.28e-02 (1.9x) | 0.1457 |
| 0.25 | 8 | 2.88e-02 | 3.12e-02 | 0.100113 | 3.52e-02 (2.1x) | 0.1409 |
| 0.125 | 16 | 7.50e-03 | 7.81e-03 | 0.118067 | 1.73e-02 (2.0x) | 0.1381 |
| 0.0625 | 32 | 1.91e-03 | 1.95e-03 | 0.126789 | 8.55e-03 (2.0x) | 0.1367 |
| 0.03125 | 64 | 4.83e-04 | 4.88e-04 | 0.131084 | 4.25e-03 (2.0x) | 0.1360 |

Predicted limit of the last column: t e^(-t) / 2 = 0.1353.
<!-- /worked -->

The local error falls by about $4$ when the step is halved (it follows $h^2/2$, as
$x''(0) = 1$), and the global error falls by about $2$: Euler's method has order
$1$, like the left rectangle. For this particular equation we can see it exactly.
Each step multiplies $x$ by $1 - h$, so after $t/h$ steps
$x = (1 - h)^{t/h} = \exp\bigl(\tfrac{t}{h}\ln(1 - h)\bigr)$, and by the Taylor
expansion $\ln(1 - h) = -h - \tfrac12 h^2 - \dots$, so
$x \approx e^{-t}\bigl(1 - \tfrac12 t h\bigr)$, an error of $\tfrac12 t e^{-t}\,h$:
the last column converges to its predicted limit $0.1353$.

![The exact curve e to the minus t from 1 towards 0, and the paths Euler's method takes with step 0.5, 1.5 and 2.5. The step 0.5 path follows the curve from above-the-line points that fall just short of it. The step 1.5 path jumps below zero at t = 1.5 and then returns, flipping sign at every step while shrinking. The step 2.5 path alternates in sign and grows, reaching minus 3.4 at t = 7.5.](img/euler-trajectories.svg)

**When a step is too large.** The figure shows something the error table hid. The
step $h = 1.5$ gives a path that flips sign every step although the true solution
never changes sign, and $h = 2.5$ explodes. On this equation each step multiplies
$x$ by $1 - h$ (more generally $1 - kh$ for $dx/dt = -kx$), so the iterates are
$(1 - kh)^i$, and whether they decay depends only on that factor:

<!-- worked: integrals.euler_stability -->
Euler on dx/dt = -x: each step multiplies x by 1 - h. Value after 10 steps from x(0) = 1.

| step h | factor 1 - h | x after 10 steps | behaviour |
|---|---:|---:|---|
| 0.5 | +0.50 | +0.0010 | decays, same sign as the true solution |
| 1.0 | +0.00 | +0.0000 | lands on 0 in one step |
| 1.5 | -0.50 | +0.0010 | decays, sign flips each step |
| 2.0 | -1.00 | +1.0000 | never decays, sign flips each step |
| 2.5 | -1.50 | +57.6650 | grows, sign flips each step |
<!-- /worked -->

For $0 < kh < 1$ the path decays like the truth. For $1 < kh < 2$ it still decays
but flips sign, an artefact the true solution does not have. At $kh = 2$ it never
decays, and beyond that the error is multiplied by more than $1$ in size each step
and grows without bound. The limit of **stability** is $h < 2/k$: how fast the
true solution decays sets the largest step the method tolerates.

**Gradient descent is Euler's method.** This should look familiar. Chapter 8 found
that stepping against the gradient on a quadratic bowl multiplies each curvature
direction of the error by $1 - \eta\lambda$ and is stable only for $\eta < 2/\lambda_{\max}$.
Take the one-knob bowl $L(w) = \tfrac12\lambda w^2$, whose gradient is $\lambda w$.
The rule $w \leftarrow w - \eta\,\nabla L(w)$ is
$w \leftarrow w - \eta\,\lambda w$, which is Euler's method for $dw/dt = -\lambda w$
with step $h = \eta$ and decay constant $k = \lambda$. Compare the iterates with the
exact solution of that ODE, $e^{-\lambda t}$, taken at the times $t = n\eta$ at
which the iterates apply:

<!-- worked: integrals.gd_is_euler -->
L(w) = 3 w^2 / 2, gradient 3 w, start w = 1. Gradient descent iterates beside the exact flow w(t) = e^(-3 t) at t = n eta.

| step n | descent, eta = 0.2 | flow at t = 0.2 n | descent, eta = 0.7 | flow at t = 0.7 n |
|---|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 0.4000 | 0.5488 | -1.1000 | 0.1225 |
| 2 | 0.1600 | 0.3012 | 1.2100 | 0.0150 |
| 3 | 0.0640 | 0.1653 | -1.3310 | 0.0018 |
| 4 | 0.0256 | 0.0907 | 1.4641 | 0.0002 |

The stability limit 2 / lambda = 0.6667.
<!-- /worked -->

With $\eta = 0.2$ the iterates follow the exact curve, a little low (the Euler
factor $0.4$ against $e^{-0.6} = 0.55$ per step) but decaying. With $\eta = 0.7$,
above the limit $2/\lambda = 0.667$, the exact curve decays to $0.0002$ while the
descent grows to $1.46$ with alternating sign. The step-size instability of
gradient descent is not a property of descent. It is the step-size instability of
Euler's method, and it is a numerical effect: the curve being followed is smooth
and convergent.

## Systems of equations and the gradient flow

A model has many knobs, so the state is a vector, $\mathbf{x}(t) \in \mathbb{R}^n$,
and the equation is $d\mathbf{x}/dt = \mathbf{f}(t, \mathbf{x})$ with
$\mathbf{f}:\mathbb{R}\times\mathbb{R}^n \to \mathbb{R}^n$. At every point of the
space $\mathbf{f}$ gives an arrow, the direction and speed with which the state
moves there. A function that assigns a vector to every point is a **vector
field**. Euler's method is unchanged: add $h\,\mathbf{f}$ to the vector. The
solution is a path that is tangent to the arrows everywhere.

A small example: $\mathbf{f}(x, y) = (-y, x)$, a field that points around the
origin, so the true solution from $(1, 0)$ is a circle of radius $1$ traversed once
every $2\pi$ time units. One Euler step turns $(x, y)$ into $(x - hy,\;y + hx)$,
whose squared length is $(1 + h^2)(x^2 + y^2)$. The radius grows by a factor
$\sqrt{1 + h^2} > 1$ at every step, for *every* step size:

<!-- worked: integrals.rotation_field -->
The rotation field (x, y)' = (-y, x) from (1, 0) for one full turn (t = 2 pi), n steps. The true radius stays 1.

| steps n | step h | Euler radius | (1 + h^2)^(n/2) | Heun radius |
|---|---:|---:|---:|---:|
| 20 | 0.3142 | 2.563159 | 2.563159 | 1.024621 |
| 40 | 0.1571 | 1.628225 | 1.628225 | 1.003048 |
| 100 | 0.0628 | 1.217748 | 1.217748 | 1.000195 |
| 1000 | 0.0063 | 1.019935 | 1.019935 | 1.000000 |
<!-- /worked -->

(The last column belongs to the better solver of the next section.) After one
turn with $100$ steps the Euler path has spiralled out to $1.2177$ times its
starting radius, and the table matches the formula $(1 + h^2)^{n/2}$. Euler's
method can fail through low accuracy, through instability at large steps or, as
here, through a systematic drift that no step size removes and only slows.

**The gradient flow.** Now choose the field to be the arrows pointing downhill on
a loss $L$:

$$
\frac{d\mathbf{w}}{dt} = -\nabla L(\mathbf{w}) .
$$

Its solution is the **gradient flow**: a path that at every instant moves in the
direction of steepest descent. Euler's method with step $\eta$ on this equation is
$\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla L(\mathbf{w})$, the stepping rule of
Chapter 8. Gradient descent is the Euler discretisation of the gradient flow, and
the learning rate is its step size.

Along a solution the loss can only fall. By the multivariable chain rule of
Chapter 8, $\tfrac{d}{dt}L(\mathbf{w}(t)) = \nabla L\cdot\tfrac{d\mathbf{w}}{dt}
= -\lVert\nabla L\rVert^2 \le 0$. A table along the flow of the chapter-8 bowl,
computed from its exact solution, confirms it:

<!-- worked: integrals.loss_rate_along_flow -->
Along the exact flow from (0, 0): the rate of change of L against minus the squared gradient length.

| t | w(t) | L(w(t)) | dL/dt by central difference | -|grad L|^2 |
|---|---:|---:|---:|---:|
| 0.0 | (0.0000, 0.0000) | 29.0000 | -676.0000 | -676.0000 |
| 0.1 | (0.5924, 1.4163) | 2.8213 | -65.6843 | -65.6843 |
| 0.3 | (0.8420, 1.9923) | 0.0301 | -0.6225 | -0.6225 |
| 0.8 | (0.8886, 2.0459) | 0.0025 | -0.0017 | -0.0017 |
<!-- /worked -->

The exact solution comes from the same split as Chapter 8. Write the error from the
minimum as $\mathbf{e} = a\,\mathbf{v}_1 + b\,\mathbf{v}_2$ along the curvature
directions. The equation $d\mathbf{e}/dt = -\mathbf{H}\mathbf{e}$ separates into
$\dot a = -\lambda_1 a$ and $\dot b = -\lambda_2 b$, two decay equations solved above:
$a(t) = a_0 e^{-\lambda_1 t}$ and $b(t) = b_0 e^{-\lambda_2 t}$. The flow is the
sum of two exponentials, one fast and one slow. For our bowl ($\lambda_1 = 11.66$,
$\lambda_2 = 0.343$) the steep direction is gone after a time of about
$1/\lambda_1 = 0.09$ and the valley floor takes about $1/\lambda_2 = 2.9$. The
condition number of Chapter 8 is the ratio of those two time scales. Gradient
descent with a fixed step $\eta$ is these two scales sampled every $\eta$:

<!-- worked: integrals.flow_vs_descent -->
Gradient descent with step eta after k steps, against the exact flow at t = eta k, both from (0, 0).

| eta | k | t | descent w | descent L | flow w(t) | flow L |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 0 | 0.00 | (0.0000, 0.0000) | 29.0000 | (0.0000, 0.0000) | 29.0000 |
| 0.05 | 2 | 0.10 | (0.7100, 1.7000) | 0.8821 | (0.5924, 1.4163) | 2.8213 |
| 0.05 | 4 | 0.20 | (0.8375, 1.9942) | 0.0303 | (0.7803, 1.8564) | 0.2775 |
| 0.05 | 8 | 0.40 | (0.8717, 2.0509) | 0.0033 | (0.8643, 2.0334) | 0.0059 |
| 0.05 | 20 | 1.00 | (0.8964, 2.0429) | 0.0022 | (0.8961, 2.0430) | 0.0022 |
| 0.18 | 0 | 0.00 | (0.0000, 0.0000) | 29.0000 | (0.0000, 0.0000) | 29.0000 |
| 0.18 | 2 | 0.36 | (-0.1584, -0.4320) | 42.1839 | (0.8577, 2.0226) | 0.0099 |
| 0.18 | 4 | 0.72 | (-0.3552, -0.9507) | 61.3635 | (0.8854, 2.0469) | 0.0026 |
| 0.18 | 8 | 1.44 | (-0.8942, -2.3244) | 129.8540 | (0.9107, 2.0370) | 0.0016 |
<!-- /worked -->

With $\eta = 0.05$ the steps track the flow at $t = 0.05k$ closely at
$k = 20$ ($L = 0.0022$ for both) and loosely in the early steps, where the descent
runs ahead of the flow: the gradient is largest at the start and shrinks as the
path moves, but a step uses the gradient at its start for its whole length, so it
travels farther than the flow does in the same time (the local error of Euler's method). With $\eta = 0.18$, above $2/\lambda_{\max} = 0.1716$, the descent
climbs to $L = 129.85$ while the flow it is supposed to follow is already at
$L = 0.0016$. The flow never diverges, because a bowl has a downhill direction
everywhere. Divergence is an artefact of the discretisation, and the picture shows
both on one field:

![Contours of the two-knob loss of Chapter 8 as ellipses elongated along a diagonal, with a grey arrow of fixed length at each grid point pointing downhill, towards the minimum at (1, 2). A smooth blue curve starts at (0, 3.5), drops steeply to the valley floor near (-0.2, 2.5) and then runs slowly along the floor towards the minimum. Orange dots joined by lines show gradient descent with step 0.15 from the same start: they swing from side to side across the valley, each swing smaller than the last, around the blue path.](img/gradient-flow-field.svg)

The smooth curve is the flow, and the dots are Euler's method with $h = 0.15$,
just below the limit: it follows the flow's route, but jumps across the valley in
the steep direction, where the step is nearly as long as the stability limit allows.
Volume II (*Linear Regression and the First Descent*) names the stepping rule
gradient descent and studies how to choose $\eta$.

## Better solvers: the same idea, a better slope

The integral and the ODE are not separate topics. If the slope does not depend on
$x$, the equation is $dx/dt = f(t)$, and the solution is the integral
$x(t) = x_0 + \int_0^t f(s)\,ds$. So Euler's method must be a rectangle rule, and
a better ODE solver should be a better quadrature rule. Check, with the speed of the
opening problem as the slope:

<!-- worked: integrals.euler_is_riemann -->
dx/dt = v(t) = t (4 - t), x(0) = 0, up to t = 3 with 6 steps of h = 0.5; exact answer 9.

| ODE solver | x(3) | quadrature rule | area |
|---|---:|---|---:|
| Euler | 8.125000 | left rectangles | 8.125000 |
| Heun | 8.875000 | trapezoids | 8.875000 |
| midpoint | 9.062500 | midpoint rectangles | 9.062500 |
<!-- /worked -->

Euler's method *is* the left Riemann sum. Now improve it as we improved the sum.
Over a step the exact change is the integral of the slope, $\int_t^{t+h} f(s, x(s))\,ds$,
which Euler approximates by $h$ times the slope at the start. Two rules do better
by sampling the slope at a second place, and each costs two evaluations of $f$
per step:

- **Heun's method** is the trapezoid rule: take the slope at the start, $k_1 = f(t, x)$; use it to predict where the step lands, $x + h k_1$; read the slope there, $k_2 = f(t + h, x + h k_1)$; and step with the average, $x + h\,(k_1 + k_2)/2$.
- The **midpoint method** reads the slope in the middle of the step instead: it takes a half Euler step, $k = f(t + h/2,\; x + \tfrac{h}{2}f(t, x))$, and steps with $x + h k$.

Both have local error of order $h^3$ and global error of order $h^2$. Since they
cost two evaluations per step, the fair comparison fixes the number of evaluations
of $f$:

<!-- worked: integrals.second_order_methods -->
dx/dt = -x, x(0) = 1, up to t = 2, with a fixed budget of evaluations of f (Euler uses one per step, Heun and midpoint two). Error at t = 2; in brackets, the fall when the budget doubled.

| evaluations of f | Euler | Heun | midpoint |
|---|---:|---:|---:|
| 8 | 3.52e-02 | 1.73e-02 | 1.73e-02 |
| 16 | 1.73e-02 (2.0x) | 3.44e-03 (5.0x) | 3.44e-03 |
| 32 | 8.55e-03 (2.0x) | 7.76e-04 (4.4x) | 7.76e-04 |
| 64 | 4.25e-03 (2.0x) | 1.85e-04 (4.2x) | 1.85e-04 |
| 128 | 2.12e-03 (2.0x) | 4.51e-05 (4.1x) | 4.51e-05 |
<!-- /worked -->

When the budget doubles, Euler's error halves and the two-evaluation methods'
error falls by about $4$. For this particular equation the two methods coincide:
both multiply by $1 - h + h^2/2$ at each step, the order-2 Taylor polynomial of
$e^{-h}$, which is why their error is of order $h^3$ per step. At a budget of $64$
evaluations the error is $4.25\times10^{-3}$ for Euler and $1.85\times10^{-4}$ for
Heun, about $23$ times smaller for the same work. On the rotation field above, one
turn with $100$ steps had the Euler radius at $1.2177$ and Heun's at $1.0002$.

Why does it matter here? Because in a generative model each evaluation of the
slope is a pass through a neural network and dominates the cost. A sampler for a
flow-matching or diffusion model is an ODE solver whose slope is a trained network
(Volume VIII, *Flow Matching and Rectified Flow* and *Samplers and Guidance*, derives them); the number of network passes is the budget, and the
solver's order decides how few steps give an acceptable result. Two details carry
over from this chapter. The step can be negative: the sampler runs *backwards* from
noise at $\sigma = 1$ to data at $\sigma = 0$.

<!-- worked: integrals.backward_in_time -->
Integrating from sigma = 1 down to 0 in n equal steps, x(1) = 1. Exact end values: 0 for dx/dsigma = 3 sigma^2, and 1 - 0.7 = 0.3 for dx/dsigma = 0.7.

| steps n | Euler, v = 3 sigma^2 | Heun, v = 3 sigma^2 | Euler, v = 0.7 |
|---|---:|---:|---:|
| 2 | -0.875000 | -0.125000 | 0.300000 |
| 4 | -0.406250 | -0.031250 | 0.300000 |
| 8 | -0.195312 | -0.007812 | 0.300000 |
| 16 | -0.095703 | -0.001953 | 0.300000 |
<!-- /worked -->

And when the slope is constant along the path, Euler's method is exact whatever
the number of steps, since each step adds the same multiple of the slope and the
sum of the step lengths is the total length. The last column is $0.3$, the exact
end value, for every $n$. Where the slope varies, the second-order step is far
better: on $3\sigma^2$, Heun's error at $n = 8$ is $0.0078$ against $0.195$.

## Making it work

- **Check the stability limit before the accuracy.** An explicit method such as Euler's is stable only for $h < 2/k$ on a decay of rate $k$. A solution may be smooth, with no accuracy reason for small steps, and the step is still capped by the fastest decay anywhere in the system. Equations with a fast and a slow time scale are called **stiff**, and the bowl of this chapter ($\lambda_1/\lambda_2 = 34$) is the smallest example.
- **When the fast part is linear, solve it exactly.** Over one step the equation $dx/dt = -B(x - x_\infty)$, with $B$ and $x_\infty$ held fixed, has the exact solution $x_\infty + (x - x_\infty)e^{-hB}$ (shift by $x_\infty$ and use the decay solution above). This **exponential step** is stable for every $h$ and exact when $B$ and $x_\infty$ really are constant. Euler uses $1 - hB$ in place of $e^{-hB}$:

<!-- worked: integrals.exponential_step -->
One step of size h on dx/dt = -x multiplies x by this factor: Euler uses 1 - h, the exact solution over the step is e^(-h).

| step h | Euler factor 1 - h | exact factor e^(-h) |
|---|---:|---:|
| 0.5 | +0.5000 | 0.6065 |
| 1.0 | +0.0000 | 0.3679 |
| 2.0 | -1.0000 | 0.1353 |
| 2.5 | -1.5000 | 0.0821 |
| 5.0 | -4.0000 | 0.0067 |
| 10.0 | -9.0000 | 0.0000 |
<!-- /worked -->

  The exponential factor stays in $(0, 1)$ however large the step, while the Euler factor passes $-1$ at $h = 2$ and keeps falling.
- **Pick the order by the cost of a slope.** Evaluate the slope as few times as the accuracy allows. A cheap slope favours small Euler steps; a slope that costs a network pass favours a higher-order method and larger steps.
- **Smoothness is an assumption.** The order of a rule is a statement about functions with moderate derivatives. A function with a jump or a kink in a panel breaks it and brings every rule back to order $1$ or lower in that panel. Special functions can beat their order: the trapezoid rule applied to $e^{-x^2/2}$ over $[-8, 8]$ is more accurate with sixteen panels than Simpson's rule (Exercise 11).
- **Grids do not scale with dimension.** A grid of $10$ points per axis needs $10^d$ evaluations in $d$ dimensions: $10^3$ at $d = 3$, $10^6$ at $d = 6$ and $10^{10}$ at $d = 10$. Expectations over the millions of dimensions of an image are estimated by averaging over random samples, not by quadrature; the probability and statistics chapters build that tool.
- **Keep the sums honest.** A sum of a million terms in 32-bit arithmetic loses digits; the floating-point chapter explains how, and the chapters that use integrals inherit its remedies.

## Where this appears in brain

**The flow-matching sampler is Euler's method.** `crates/diffusion/src/scheduler.rs`
documents the model of the process in its header: the noisy latent is a straight
line between a data point $x_0$ and noise $\varepsilon$, $x_\sigma = (1-\sigma)x_0 + \sigma\varepsilon$,
the denoising network predicts the velocity $v = dx/d\sigma$, and sampling
"integrates that ODE backwards from $\sigma_{\max}$ toward 0 with explicit Euler
steps". Its `step` computes `dt = sigmas[i + 1] - sigmas[i]` (negative, as in the
backward-in-time table above) and returns `sample + dt * model_output`, which is the
Euler step of this chapter verbatim. The scheduler was introduced as the first piece
of a new crate for model-agnostic diffusion machinery in commit `5f7ae6dc2`
(July 2026), as pure host arithmetic with no device dependency. Its test
`constant_velocity_integrates_exactly_to_minus_v` in
`crates/diffusion/tests/it/scheduler.rs` is this chapter's constant-slope fact as a
gate: with a constant velocity the steps telescope, the sum of the step lengths is
$\sigma_N - \sigma_0 = -1$, and eight steps must return the start minus $v$
exactly. The heavier samplers in `crates/diffusion/src/flowsolvers.rs` are
multistep methods that reuse earlier slopes to gain order; the test
`first_step_is_euler` there records the consequence that the very first step of
both is a plain Euler step, because no earlier slope exists yet.

**A spiking neuron is an ODE with a stability limit.** The leaky
integrate-and-fire membrane of `crates/neuro` obeys a relaxation equation of the
exponential-decay kind. `crates/neuro/src/lif.rs` documents `dt_over_tau`, the step
in units of the membrane time constant, as "must be in `(0, 1]`: at `a > 1` the
Euler update overshoots the resting potential and the membrane oscillates". That is
the factor $1 - kh$ of the stability table: for $kh > 1$ the sign flips. The
story of this kernel is the lesson of the "Making it work" list. The first version
(commit `36bee10b7`, September 2026) used forward Euler and was gated on the closed
form of the Euler recurrence, a geometric sequence, compared with the kernel's
output with no tolerance fitted. The next day commit `3945b43c6` made synapses
conductances, which makes the leak grow with the input: the equation became
$\tau\,dv/dt = A - Bv$ with $B$ increasing with the total conductance. The commit
message records the consequence: a forward step of fixed size "diverges exactly when
the input is strongest", and it replaced the update by the exact solution over the
tick. In `crates/kernels/wgsl/lif_step.wgsl` the line is
`v_inf + (v[i] - v_inf) * exp(-a * bb)`, the exponential step above with
`bb` as $B$. The oracle `analytic_v` in `crates/neuro/src/lif.rs` was updated to the
exponential form, and the test `membrane_matches_the_closed_form_for_a_constant_input`
in `crates/neuro/tests/it/dynamics.rs` compares the device with it for $64$ ticks to
within $10^{-6}$. (The header comment of the kernel still describes the earlier
forward-Euler update and its $(1-a)^k$ closed form; the code below it is the
exponential step.)

**Evaluation metrics integrate with the trapezoid rule.**
`crates/survival/src/brier.rs` has `integrated_brier`, the squared-error score of
a survival prediction averaged over a window of follow-up time. It computes the
score on a grid of times and sums `0.5 * (s[0] + s[1]) * (t[1] - t[0])` over
neighbouring pairs, the trapezoid rule with unequal panels, then divides by the
window length. The grid is whatever the follow-up times are, which is why the
formula takes each panel's own width. The function arrived with the crate in
commit `19a4c4b7f` (October 2026), whose message says the numbers are held to
reference data; `crates/survival/tests/reference.rs` compares `integrated_brier`
with a stored reference value. The restricted mean survival time in
`crates/survival/src/rmst.rs` is also an area under a curve: `curve_rmst` integrates
a survival curve known at whole years, "linear between them", by the same trapezoid
sum (commit `dc09a817c`, October 2026). Neither code needs an antiderivative, as a
curve that comes from data has none.

## Check your understanding

Answers are at the end of the chapter.

1. **Predict.** For a speed that is *decreasing* over the whole interval, is the left sum an underestimate or an overestimate of the distance? What is the gap between left and right sums in terms of the first and last reading?
2. **Explain.** A velocity has integral $0$ over $[0, 4]$. A friend says the car did not move. What is true and what is not?
3. **Find the error.** "The derivative of $x^2$ is $2x$, so an antiderivative of $x^2$ is $2x$, and $\int_0^2 x^2\,dx = 4$." Find the mistake and the correct value.
4. **Predict.** A trapezoid rule with $100$ panels has error $10^{-5}$ on a smooth function. Roughly what is the error with $200$ panels? Simpson's rule, which had error $10^{-9}$ at $100$ panels, with $200$?
5. **Predict.** Euler's method on $dx/dt = -4x$ is used with $h = 0.6$. What is the factor per step, and what does the path do? What is the largest stable step, and the largest step for which the path has the right sign?
6. **Explain.** Why does gradient descent with step $\eta = 0.18$ on the chapter's bowl diverge while the gradient flow it approximates converges?
7. **Find the error.** "Heun's method needs two evaluations per step, so it is twice as expensive as Euler and not worth it." Use the budget table to say what is wrong.
8. **Explain.** Why is the compound-interest formula $100(1 + r/n)^n$ an Euler method for $dy/dt = ry$? What does the limit $n \to \infty$ correspond to?

## Exercises

1. **[Calculate]** For $v(t) = t^2$ on $[0, 2]$ compute the left and right sums with $n = 4$, the trapezoid sum, and the exact area from an antiderivative. Verify that the gap between left and right is $\Delta t\,(v(2) - v(0))$.
2. **[Calculate]** Evaluate $\int_0^1 (x^3 + e^{2x})\,dx$ exactly, and differentiate your antiderivative to check it. Then find $\int_1^e \tfrac{1}{x}\,dx$.
3. **[Calculate]** Apply Simpson's rule with $n = 2$ to $\int_0^2 x^3\,dx$ and $\int_0^2 x^4\,dx$. Compare with the exact values and explain which is exact.
4. **[Calculate]** Apply Euler's method by hand to $dx/dt = -2x$, $x(0) = 1$ with $h = 0.25$ for four steps, and compare with $e^{-2}$. State the stability limit and the largest step for which the sign is right.
5. **[Calculate]** Show that one Heun step for $dx/dt = -x$ multiplies $x$ by $1 - h + h^2/2$, and that the difference from $e^{-h}$ is about $h^3/6$ (use the order-3 Taylor expansion).
6. **[Calculate]** For $L(w) = \tfrac12\lambda w^2$ write the gradient flow and its solution, and gradient descent with step $\eta$. For $\lambda = 3$ find the largest $\eta$ for which descent converges and the one for which it jumps to $0$ in one step.
7. **[Calculate]** Use the trapezoid rule with unequal panels on the readings $f(0) = 1$, $f(1) = 3$, $f(3) = 4$, $f(7) = 4$ to estimate $\int_0^7 f$, and divide by the window length to obtain the average value.
8. **[Implement]** Implement the left, right and trapezoid sums and reproduce `integrals.speed_readings`, `integrals.riemann_shrinking` and `integrals.rules_error_order` (add Simpson's rule), then `integrals.rules_polynomials`.
9. **[Implement]** Implement Euler's method, Heun's method and the midpoint method for a scalar ODE and reproduce `integrals.euler_by_hand`, `integrals.euler_error_order`, `integrals.euler_stability` and `integrals.second_order_methods`.
10. **[Implement]** Implement gradient descent on the chapter-8 loss and the exact gradient flow (diagonalise the Hessian; write the error in its two curvature directions) and reproduce `integrals.flow_vs_descent` and `integrals.loss_rate_along_flow`. Then reproduce `integrals.rotation_field`.
11. **[Investigate]** Integrate $e^{-x^2/2}$ over $[-8, 8]$ with the trapezoid rule and with Simpson's rule for $n = 8, 16, 32, 64$. Why is the trapezoid rule so accurate here, and why does Simpson's rule, which is higher order for a generic function, not help? (Hint: Simpson's rule is a weighted combination of two trapezoid sums with different widths. Look at what the integrand and its derivatives do at the two ends.)
12. **[Investigate]** Run Euler's method and Heun's method on $dx/dt = -50x$ with $h = 0.01, 0.02, 0.03, 0.04, 0.05$ up to $t = 1$. Find where each becomes unstable, and compare with $2/k$ and with the stability factor of Heun's method, $1 - kh + (kh)^2/2$. What does the answer say about how much a second-order method buys on a stiff problem?

## Summary

We asked how to add up infinitely many small pieces and how to follow a system
that changes continuously. A Riemann sum, the sum of rectangle areas
$\sum f(t_i)\,\Delta t$, brackets a total; its limit as the width goes to zero is
the **integral** $\int_a^b f(t)\,dt$, a signed area. The **fundamental theorem of
calculus** says the rate of change of the accumulated area $\int_a^x f$ is
$f(x)$, so the integral of $f$ is $F(b) - F(a)$ for any antiderivative $F$, which
turns the derivative rules of Chapter 7 into integration rules and makes the
Taylor remainder of Chapter 8 exact. When no formula exists, as for a bell curve,
**numerical integration** replaces the curve by a polynomial in each panel:
rectangles (error order $1$), trapezoids (order $2$) and Simpson's parabolas
(order $4$), verified by how the error falls when the panel count doubles. An
**ordinary differential equation** gives the rate of change of a quantity in terms
of the quantity. **Euler's method** follows the slope in steps of size $h$: it is
a left rectangle rule when the slope depends only on time, it is compounding for
$dy/dt = ry$, its global error is of order $h$ (local $h^2$), and it is unstable
for steps beyond $2/k$ on a decay of rate $k$. Gradient descent is Euler's method
on the gradient flow $d\mathbf{w}/dt = -\nabla L$, so the learning-rate limit of
Chapter 8 is Euler's stability limit. Better solvers (Heun, midpoint) sample the
slope twice per step for order $2$, and the same idea runs the samplers of
generative models.

The limits are the limits of a smooth world: every error statement assumed
bounded derivatives, and integration by grids is hopeless in many dimensions. The
quantities we integrated were also certain; a loss is an average over data we do
not fully know. The next chapter, *Probability: Reasoning Under Uncertainty*,
puts numbers on that uncertainty and then uses the integral to define its
averages: the expectation of a continuous variable, and the density that integrates
to $1$.

## Key terms

- **integral**: the limit of the Riemann sums $\sum f(t_i)\,\Delta t$ as the width goes to zero, written $\int_a^b f(t)\,dt$; the signed area between the graph of $f$ and the axis. <!-- concept: integral -->
- **fundamental theorem of calculus**: the rate of change of the accumulated area $\int_a^x f(t)\,dt$ is $f(x)$, so $\int_a^b f = F(b) - F(a)$ for any antiderivative $F$ with $F' = f$. <!-- concept: fundamental-theorem-of-calculus -->
- **ordinary differential equation**: an equation $dx/dt = f(t, x)$ that gives the rate of change of an unknown function of one variable in terms of the function itself; its solution is a curve that follows the prescribed slopes from a given starting value. <!-- concept: ordinary-differential-equation -->
- **Euler's method**: the rule $x_{i+1} = x_i + h\,f(t_i, x_i)$ that follows the slope of an ODE for a short step $h$ and repeats; with a gradient as the slope it is gradient descent. <!-- concept: euler-method -->
- **numerical integration**: approximating an integral by a weighted sum of function values, such as rectangles, trapezoids or Simpson's rule, whose error falls as a fixed power of the panel width. <!-- concept: numerical-integration -->

## Answers

**Before you begin**

1. $3x^2$ (power rule) and $2e^{2x}$ (chain rule: $e^{2x}$ times the derivative $2$ of the inside).
2. It approaches $1$. Then $\tfrac{d}{dx}e^x = e^x\cdot\lim_{h\to0}(e^h - 1)/h = e^x$.
3. $0.5\,(0 + 1 + 4 + 9) = 7$.
4. $\nabla L = (2w_1, 6w_2) = (2, 6)$ at $(1, 1)$. Stepping against it moves towards smaller $w_1$ and much smaller $w_2$, in direction $(-2, -6)$.
5. $100\cdot 1.05^2 = 110.25$; the factor $1.05$ each year.
6. $1 + x + x^2/2$, and the error is about $h^3/6$ (the first dropped term).

**Check your understanding**

1. An overestimate: each rectangle at the left end is higher than the curve over that interval, since the speed is falling. The right sum underestimates. The gap is $\Delta t\,(v(\text{first}) - v(\text{last}))$, the difference of the two end readings times the width.
2. The *net* displacement is zero: the car ended where it started. It may have driven forwards and backwards; the distance is the integral of $\lvert v\rvert$ (here $4$ for $v = t - 2$). The velocity being zero on average does not make the car stationary.
3. $2x$ is the derivative of $x^2$, not an antiderivative of it. The antiderivative of $x^2$ is $x^3/3$ (its derivative is $x^2$), so $\int_0^2 x^2\,dx = 8/3$, not $4$.
4. Trapezoid has order $2$: doubling the panels divides the error by about $4$, so about $2.5\times10^{-6}$. Simpson has order $4$, a factor $16$: about $6\times10^{-11}$.
5. The factor is $1 - kh = 1 - 2.4 = -1.4$: the path flips sign and grows by $1.4$ each step. The stable limit is $2/k = 0.5$ and the sign-correct limit is $1/k = 0.25$.
6. The flow is a smooth curve that decays everywhere, because the bowl slopes downward towards the minimum. The descent is Euler's method with $h = \eta$, and for a decay of rate $\lambda_{\max} = 11.66$ Euler is stable only for $\eta < 2/\lambda_{\max} = 0.1716$. At $0.18$ the steep-direction factor is $1 - 0.18\times11.66 = -1.098$, so the error in that direction grows each step.
7. Compare at equal *evaluations*, not equal steps. At $64$ evaluations Heun's error is about $23$ times smaller than Euler's and it improves by $4$ per doubling of the budget instead of $2$. Each Heun step is twice the price but far more accurate, so it wins whenever the target accuracy is more than modest.
8. Each Euler step on $dy/dt = ry$ with $h = 1/n$ is $y \leftarrow y + h r y = y(1 + r/n)$, and $n$ steps give $y_0(1 + r/n)^n$. Letting $n \to \infty$ makes $h \to 0$: the Euler path approaches the exact solution $y_0 e^{r t}$, continuous compounding.
