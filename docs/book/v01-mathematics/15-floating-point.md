<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Floating Point and Numerical Error

*Why does a computer get arithmetic slightly wrong, and when does slightly wrong become completely wrong?*

## The problem

Ask any programming language for $0.1 + 0.2$ and compare the answer with
$0.3$:

<!-- worked: floating.surprises -->
| expression (64-bit floating point) | result |
|---|---|
| 0.1 + 0.2 | 0.30000000000000004 |
| 0.1 + 0.2 == 0.3 | False |
| 0.1 + 0.2 - 0.3 | 5.55e-17 |
| 0.1 * 3 | 0.30000000000000004 |
| sum of ten copies of 0.1 | 1.0 |
| (0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3) | False |
| the number stored for 0.1, exactly | 0.1000000000000000055511151231257827021182 |
<!-- /worked -->

Nothing is broken. The machine did exactly what it was built to do, and the
rest of this chapter explains what that is. But the consequences are real. The
two sides of a comparison that school arithmetic says are equal are not; adding
the same three numbers in a different order gives a different answer; and the
number stored for $0.1$ is not $0.1$ at all but a close neighbour of it.

A single such error is harmless: it is in the seventeenth digit. Machine
learning does not do one operation. A training run does billions, and an error
too small to see in one of them can still decide the result of all of them.
Here is the smallest example. A running total of a million small steps, each
one tenth, is one hundred thousand. We add the stored value of $0.1$ one
million times in several ways:

<!-- worked: floating.million_tenths -->
The sum of 10^6 copies of the stored value of 0.1 (exact sum of the stored 32-bit copies: 100000.001490).

| method | computed sum | relative error |
|---|---:|---:|
| 64-bit, left to right | 100000.00000133288 | 1.33e-11 |
| 32-bit, left to right | 100958.344 | 9.58e-03 |
| 32-bit, pairwise | 100000 | 1.49e-08 |
| 32-bit, compensated (Kahan) | 100000 | 1.49e-08 |
| 32-bit terms added in a 64-bit accumulator | 100000 | 1.49e-08 |
<!-- /worked -->

In 64-bit arithmetic the error is $10^{-11}$ and nobody would notice. In 32-bit
arithmetic, adding left to right, the answer is off by almost one percent:
$100958$ instead of $100000$. The loss of accuracy is not random noise; it is
a systematic drift, and it comes from nothing more than the rule "round each
addition to the nearest representable number". Yet two other methods, which
add exactly the same 32-bit numbers, are right to the last digit the format can
show.

Two questions follow, and they organise this chapter. What exactly are the
numbers a computer has, and how big is the error that each operation commits?
And given that every operation errs a little, how do we write computations so
that the errors stay small, and how do we recognise the computations in which
they cannot? Chapter 1 met the first failures: a product of probabilities that
underflowed to zero at 46 factors, and $(1 + 1/n)^n$ that fell to exactly $1$
at $n = 10^{16}$. Chapter 7 found that a finite-difference derivative has a best
step. This chapter explains all three, and finishes the techniques that
those chapters postponed: log-sum-exp and compensated summation.

## Before you begin

Six questions on ideas from earlier chapters, to answer from memory. The
answers are at the end of the chapter. If one is hard, go back to the chapter
named before reading on.

1. (Chapter 1, real numbers.) How many real numbers lie between $0.1$ and $0.2$? A computer stores a number in a fixed number of bits: what does that imply about which real numbers it can hold?
2. (Chapter 1, exponents and scientific notation.) Write $0.00032$ and $5400$ in scientific notation. What is $2^{-3}$, and what is $2^{10}$?
3. (Chapter 1, logarithms count digits and bits.) How many bits are needed to write each of the whole numbers from $0$ to $1023$? What does $\lfloor \log_2 x \rfloor$ tell you about the binary representation of a number $x \ge 1$?
4. (Chapter 1, why products of probabilities become sums of logarithms.) A product of $n$ factors, each $0.1$, underflowed in 32-bit arithmetic at $n = 46$. What did the chapter do about it, and which problem did it say was left open?
5. (Chapter 7, numerical differentiation.) Write the central difference estimate of $f'(x)$. Its error shrinks like $h^2$ as $h$ shrinks, yet the chapter reported that making $h$ very small makes the estimate worse. Why?
6. (Chapter 8, conditioning.) What is the condition number of a bowl-shaped function, and what does a large one do to gradient descent?

## Numbers with a floating point

### Scientific notation, in base 2

Chapter 1 wrote very large and very small numbers in scientific notation:
$0.000314 = 3.14 \times 10^{-4}$. A number has two parts: a **significand**
$3.14$ that carries the digits, and an exponent $-4$ that says where the
decimal point goes. The same idea works in any base, and a computer uses
base 2. Every positive real number $x$ can be written

$$
x = m \times 2^{e}, \qquad 1 \le m < 2,
$$

with a whole-number exponent $e$ and a significand $m$ that has the binary
form $1.b_1 b_2 b_3 \dots$ (a one, a point, then bits). For example
$6.5 = 110.1_2$ (that is $4 + 2 + \tfrac12$), which is $1.101_2 \times 2^2 =
1.625 \times 4$. The numbers from $2^e$ up to (not including) $2^{e+1}$ all
share the exponent $e$. We call this range the **binade** of $e$; the binade of
$2$ is $[4, 8)$ and the binade of $-1$ is $[\tfrac12, 1)$.

A **floating-point number** keeps the idea but limits it. It stores a sign $s$,
an exponent $e$ from a fixed range $e_{\min} \le e \le e_{\max}$ and a
significand with a fixed number $p$ of bits, the **precision**: one leading $1$
(not stored, because it is always there) and $p - 1$ fraction bits.

$$
x = (-1)^s \times \left(1 + \frac{f}{2^{p-1}}\right) \times 2^{e}, \qquad
f \in \{0, 1, \dots, 2^{p-1} - 1\}.
$$

The word "floating" refers to the point: the number keeps $p$ significant
digits wherever it is, rather than a fixed number of digits after the point.
A number near a billion and a number near a billionth are stored with the same
relative detail. That is the right design for science and for machine
learning, where scales vary enormously, and it is the root of everything that
follows: the format has a fixed number of *significant* digits, so its
absolute resolution changes with the size of the number.

### A format small enough to see

Real formats have billions of numbers, so we first study a toy one. Take
$p = 3$ (so the significand is $1.00$, $1.01$, $1.10$ or $1.11$ in binary:
$1$, $1.25$, $1.5$ or $1.75$) and exponents from $-2$ to $1$. Listing every
positive number it can hold, binade by binade:

<!-- worked: floating.toy_grid -->
A toy format: significand 1.xx (p = 3 bits), exponents -2 to 1. 16 positive normal numbers, largest 3.5.

| binade | spacing in the binade | the numbers |
|---|---:|---|
| [1/4, 1/2) | 0.0625 | 0.25, 0.3125, 0.375, 0.4375 |
| [1/2, 1) | 0.125 | 0.5, 0.625, 0.75, 0.875 |
| [1, 2) | 0.25 | 1, 1.25, 1.5, 1.75 |
| [2, 4) | 0.5 | 2, 2.5, 3, 3.5 |
| below 0.25 (subnormal) | 0.0625 | 0, 0.0625, 0.125, 0.1875 |
<!-- /worked -->

Four numbers per binade (always $2^{p-1}$ of them), and the gap between
neighbours is the same inside a binade and **doubles** from one binade to the
next: $0.0625$ near $0.3$, $0.125$ near $0.7$, $0.25$ near $1.5$, $0.5$ near
$3$. In general the gap in the binade of $e$ is $2^{e-(p-1)}$. The picture
plots the gap against the number.

![The numbers of a toy format with a 3-bit significand, drawn as points at the gap to their next neighbour. The gap is 0.0625 for the numbers from 0.25 to 0.5, doubles to 0.125 from 0.5 to 1, 0.25 from 1 to 2 and 0.5 from 2 to 3.5. Zero and the three numbers below 0.25 (orange) keep the smallest gap, 0.0625. A rug of tick marks under the axis shows where the numbers sit on the real line.](img/toy-number-grid.svg)

Two consequences matter throughout.

First, the *absolute* gap grows with the size of the number, but the
*relative* gap does not. In the binade of $e$ the gap divided by the number is
between $2^{-(p-1)}/2 = 2^{-p}$ (at the top of the binade, number close to
$2^{e+1}$) and $2^{-(p-1)}$ (at the bottom, number $2^e$). With $p = 3$ the
relative gap lies between $0.125$ and $0.25$ for every normal number.

Second, below the smallest normal number $2^{e_{\min}} = 0.25$ the picture
shows orange points: zero and three more numbers with the *same* gap as the
lowest binade. These are **subnormal** numbers, a device that lets the format
degrade gently towards zero instead of jumping from $0.25$ to $0$. We return
to them when we reach the edges of the range.

### The real formats

The 32-bit format has $p = 24$: one sign bit, 8 bits for the exponent and 23
fraction bits. The exponent is stored with an offset (a bias of $127$) so that
the stored field is a plain unsigned integer $E$ and the exponent is
$e = E - 127$ for $1 \le E \le 254$. The two extreme fields are reserved: $E =
0$ is zero and the subnormals, $E = 255$ is infinity and "not a number".
The 64-bit format has $p = 53$, 11 exponent bits and 52 fraction bits. Using
Python's `struct` module to look at the bits of some 32-bit numbers:

<!-- worked: floating.fp32_bit_patterns -->
| real number | hex | s | exponent field | fraction field | value stored |
|---|---|---|---|---|---|
| 1 | 3f800000 | 0 | 01111111 (127) | 00000000000000000000000 | 1 |
| 0.5 | 3f000000 | 0 | 01111110 (126) | 00000000000000000000000 | 0.5 |
| -2 | c0000000 | 1 | 10000000 (128) | 00000000000000000000000 | -2 |
| 0.1 | 3dcccccd | 0 | 01111011 (123) | 10011001100110011001101 | 0.10000000149011611938 |
| 3.14159265358979 | 40490fdb | 0 | 10000000 (128) | 10010010000111111011011 | 3.1415927410125732422 |
| 2^-126 | 00800000 | 0 | 00000001 (1) | 00000000000000000000000 | 1.17549435082e-38 |
| 2^-149 | 00000001 | 0 | 00000000 (0) | 00000000000000000000001 | 1.40129846432e-45 |
| largest | 7f7fffff | 0 | 11111110 (254) | 11111111111111111111111 | 3.40282346639e+38 |
| inf | 7f800000 | 0 | 11111111 (255) | 00000000000000000000000 | inf |
| nan | 7fc00000 | 0 | 11111111 (255) | 10000000000000000000000 | nan |
<!-- /worked -->

Read the row for $0.1$. The stored exponent field is $123$, so $e = 123 -
127 = -4$ and the number lies in $[\tfrac1{16}, \tfrac18)$. The fraction
field `10011001100110011001101` gives a significand of about $1.6$. The value
actually stored is $0.10000000149\ldots$ and not $0.1$. The reason is that
$0.1$ has no finite binary expansion: $0.1 = 0.0001\,1001\,1001\,1001\ldots_2$,
a pattern that repeats forever, just as $\tfrac13 = 0.333\ldots$ does in
decimal. The format keeps the first 24 significant bits and rounds the rest,
and the discarded tail `1001...` is more than half a unit in the last place,
so the last kept bit is rounded up from `100` to `101`. Only fractions whose
denominator is a power of two ($\tfrac12$, $\tfrac38$, $6.5$) are stored
exactly. The rows for $2^{-126}$ and $2^{-149}$ are the smallest normal and the
smallest subnormal numbers; the row for "largest" has all fraction bits set and
the largest finite exponent field, $254$.

The parameters of the two formats, with the machine epsilon that the next
section defines:

<!-- worked: floating.format_parameters -->
| quantity | 32-bit | 64-bit |
|---|---:|---:|
| significand bits p | 24 | 53 |
| exponent bits | 8 | 11 |
| exponents of normal numbers | -126 to 127 | -1022 to 1023 |
| u = 2^-p | 5.960e-08 | 1.110e-16 |
| eps = 2^(1-p) | 1.192e-07 | 2.220e-16 |
| eps found by halving until 1 + eps/2 == 1 | 1.192e-07 | 2.220e-16 |
| decimal digits (p log10 2) | 7.22 | 15.95 |
| smallest normal number | 1.175e-38 | 2.225e-308 |
| smallest subnormal number | 1.401e-45 | 4.941e-324 |
| largest number | 3.403e+38 | 1.798e+308 |
<!-- /worked -->

A 32-bit number carries about $7$ decimal digits, a 64-bit number about $16$.
Smaller formats with 16 bits, and the 8-bit formats, follow the same layout
with fewer exponent and fraction bits; Volume IV (the chapter on number formats)
builds them from the ideas here.

## Rounding

### Rounding to the nearest number

Almost no real number is a floating-point number, so every result must be
replaced by a nearby representable one. The rule used by virtually all hardware
is **round to nearest, ties to even**: choose the representable number
closest to the exact value; if two are equally close, choose the one whose
last fraction bit is $0$. We write $\mathrm{fl}(x)$ for the stored number
that results from the exact real number $x$. The toy format shows the rule at
work, including two exact ties:

<!-- worked: floating.toy_rounding -->
Rounding to the toy format. u = 2^-3 = 0.125; 1.125 and 1.375 are exact ties.

| x | nearest toy number | relative error | error / u |
|---|---:|---:|---:|
| 0.3 | 0.3125 | 4.167e-02 | 0.333 |
| 1.1 | 1 | 9.091e-02 | 0.727 |
| 1.125 | 1 | 1.111e-01 | 0.889 |
| 1.375 | 1.5 | 9.091e-02 | 0.727 |
| 2.9 | 3 | 3.448e-02 | 0.276 |
| 3.33333 | 3.5 | 5.000e-02 | 0.400 |
| 0.1 | 0.125 | 2.500e-01 | 2.000 |
<!-- /worked -->

Look at the last column, which divides the relative error by $u = 2^{-p} =
0.125$. Every entry in the first six rows is at most $1$. The seventh, $0.1$,
is $2$: it lies below the smallest normal number $0.25$, in the subnormal
range, where the relative gap is not bounded.

The bound is the heart of the chapter, so we derive it. Let $x$ be a positive
number in the binade of $e$, inside the normal range. The gap between
representable numbers there is $2^{e-p+1}$, and the nearest one is at most
half a gap away:

$$
|\mathrm{fl}(x) - x| \le \tfrac12\, 2^{e-p+1} = 2^{e-p}.
$$

Since $x \ge 2^e$, dividing,

$$
\frac{|\mathrm{fl}(x) - x|}{|x|} \le \frac{2^{e-p}}{2^{e}} = 2^{-p} =: u .
$$

The quantity $u$ is the **unit round-off**: the largest relative error that
rounding one number can commit. Equivalently,

$$
\mathrm{fl}(x) = x\,(1 + \delta), \qquad |\delta| \le u .
$$

For the 32-bit format $u = 2^{-24} = 5.96 \times 10^{-8}$ and for the 64-bit
format $u = 2^{-53} = 1.11 \times 10^{-16}$. Rounding exact real numbers to
32 bits and measuring:

<!-- worked: floating.rounding_error_table -->
Rounding exact real numbers to the 32-bit format, ties to even. u = 2^-24 = 5.960e-08.

| x | nearest 32-bit number | absolute error | relative error | error / u |
|---|---:|---:|---:|---:|
| 0.1 | 0.1000000015 | 1.49e-09 | 1.49e-08 | 0.250 |
| 1/3 | 0.3333333433 | 9.93e-09 | 2.98e-08 | 0.500 |
| pi | 3.141592741 | 8.74e-08 | 2.78e-08 | 0.467 |
| 100.1 | 100.0999985 | 1.53e-06 | 1.52e-08 | 0.256 |
| 0.001 | 0.001000000047 | 4.75e-11 | 4.75e-08 | 0.797 |
| 16777217 | 16777216 | 1.00e+00 | 5.96e-08 | 1.000 |
| 5e-39 | 4.999999675e-39 | 3.25e-46 | 6.50e-08 | 1.090 |

The last row lies in the subnormal range, where the bound u no longer holds.
<!-- /worked -->

Every normal case lies below $u$ (the column "error / $u$" is at most $1$), and
the integer $16777217 = 2^{24} + 1$ shows the bound exactly attained: above
$2^{24}$ the gap of the 32-bit format is $2$, so not every whole number can be
stored. This is the floating-point twin of Chapter 1's warning that a 32-bit
integer wraps around: a 32-bit float never wraps, but past $2^{24}$ it skips
integers.

### Machine epsilon, and the two conventions

The **machine epsilon** $\varepsilon_{\mathrm{mach}}$ is the distance from $1$
to the next larger representable number. The number $1$ is the bottom of its
binade ($e = 0$), so that gap is $2^{0-(p-1)}$:

$$
\varepsilon_{\mathrm{mach}} = 2^{1-p}, \qquad u = \tfrac12\,\varepsilon_{\mathrm{mach}} = 2^{-p}.
$$

For 32 bits $\varepsilon_{\mathrm{mach}} = 2^{-23} = 1.19 \times 10^{-7}$, for 64
bits $2^{-52} = 2.22 \times 10^{-16}$. The last row of the parameter table found
these by a loop: halve $\varepsilon$ until $1 + \varepsilon/2$ rounds back to
$1$.

Two quantities, two names, and a convention to be careful about. Programming
languages and libraries (the `EPSILON` constant of most of them) use
$\varepsilon_{\mathrm{mach}} = 2^{1-p}$, the gap at $1$. Error analysis uses
$u = 2^{-p}$, the rounding error bound, which is half of it. Chapter 7 wrote
"unit round-off $\varepsilon$" in its table of best steps; the values there
($1.11 \times 10^{-16}$ and $5.96 \times 10^{-8}$) are $u$. From here on this
book writes $u$ for the bound on one rounding and $\varepsilon_{\mathrm{mach}}$ for the gap at
$1$, and when it quotes a library's epsilon it says so.

That $1 + u$ rounds to $1$ shows why Chapter 1's $(1 + 1/n)^n$ collapsed. At
$n = 10^{16}$, $1/n = 10^{-16} < u = 1.11 \times 10^{-16}$: the exact sum lies
below the midpoint between $1$ and its neighbour, and rounds back to $1$, so
the whole expression is $1^n = 1$.

### Ties go to the even neighbour

Why ties to even, rather than always up? Always rounding half up would make
every tie round the same way, and a computation with many ties would drift in
one direction. Rounding ties to the even neighbour sends half of them up and
half down. With 32 bits:

<!-- worked: floating.ties_to_even -->
| exact value | rounds to | last bit of the result |
|---|---:|---:|
| 1 + 2^-24 | 1 | 0 |
| 1 + 3 * 2^-24 | 1.0000002384 | 0 |
| 2^24 + 1 | 16777216 | 0 |
| 2^24 + 3 | 16777220 | 0 |
| 1 + 2^-24 + 2^-40 | 1.0000001192 | 1 |
<!-- /worked -->

$1 + 2^{-24}$ sits exactly between $1$ and $1 + 2^{-23}$ and goes to $1$, whose
last bit is $0$. $1 + 3 \cdot 2^{-24}$ sits between $1 + 2^{-23}$ (last bit
$1$) and $1 + 2^{-22}$ and goes up. The integer $2^{24} + 1$ is a tie between
$2^{24}$ and $2^{24} + 2$ and goes down; $2^{24} + 3$ goes up. The last row
adds a tiny excess to a tie, which makes the number strictly closer to the
upper neighbour: no longer a tie, so it rounds up.

To trust the rule rather than a library's implementation of it, the worked
module also contains a second, independent rounder written from the text above
using exact fractions, and compares it with the hardware conversion:

<!-- worked: floating.rounding_agreement -->
Exact rounding of 20000 random doubles to 32 bits against the hardware conversion: 0 disagreements.
<!-- /worked -->

### Arithmetic: one rounding per operation

Computers do not add real numbers; they add stored numbers. The standard
requires that for $a, b$ stored and an operation $\circ \in \{+, -, \times,
\div\}$ (and square root) the result is the *correctly rounded* exact result:

$$
\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta), \qquad |\delta| \le u,
$$

provided the exact result is in the normal range. This is the only fact about
the machine that error analysis needs: **each operation commits one relative
error of size at most $u$**, and the errors of different operations are
different. We call the discrepancy between the exact value and the stored one
the **rounding error**. Everything that follows is a study of how rounding
errors add up, cancel and are magnified.

The same fact explains the opening puzzle. The stored $0.1$ is slightly above
$0.1$, and $0.2$ is exactly twice it. The sum of the two is exact in the
real numbers but lies exactly halfway between two neighbours:

<!-- worked: floating.why_point_three -->
| quantity | value |
|---|---|
| stored 0.1 | 0.100000000000000005551115123126 |
| stored 0.2 (exactly twice the stored 0.1) | 0.200000000000000011102230246252 |
| their exact sum | 0.300000000000000016653345369377 |
| nearest number below it | 0.299999999999999988897769753748 |
| nearest number above it | 0.300000000000000044408920985006 |
| distance to the lower / to the upper | 2.7756e-17 / 2.7756e-17 |
<!-- /worked -->

The exact sum is equidistant ($2.78 \times 10^{-17}$ on each side) from the
number printed as $0.29999999999999999$ and the one printed as
$0.30000000000000004$, so the tie rule decides, and it picks the upper
neighbour. The stored $0.3$ is the lower one. The two results differ by a
single unit in the last place, the smallest possible difference. Equality of
floating-point results is therefore a fragile thing, and the right question is
never "are they equal" but "are they within an error bound I can justify".

## The edges of the range

The relative error bound $u$ holds only for results inside the representable
range. Outside it, other things happen. Chapter 1 met the symptoms; here is the
mechanism.

### Overflow, infinity and not-a-number

An exact result larger than the largest finite number (about $3.4 \times
10^{38}$ for 32 bits, $1.8 \times 10^{308}$ for 64 bits) cannot be stored. When
rounding would take it past the largest number, the result is $+\infty$ (or
$-\infty$). This is **overflow**. The boundary is itself a rounding question:

<!-- worked: floating.overflow_edge -->
Largest 32-bit number: 3.4028234663852886e+38. Decimal inputs near the top:

| decimal input | rounded to 32 bits |
|---|---:|
| 3.4028235e38 | 3.402823466e+38 |
| 3.40282356e38 | 3.402823466e+38 |
| 3.40282357e38 | inf |
| 1e39 | inf |
<!-- /worked -->

A decimal that is closer to the largest number than to the first number past it
(which would be $2^{128}$) rounds to the largest; one beyond the midpoint
becomes infinity. Infinity then behaves sensibly in arithmetic, mostly:
$1/\infty = 0$, $\infty + 1 = \infty$. But some operations have no sensible
answer, and the result is **NaN**, "not a number": $\infty - \infty$, $0
\times \infty$, $0/0$. NaN is contagious: any operation with a NaN gives a
NaN, so a single bad value spreads through everything computed from it.

<!-- worked: floating.special_values -->
| expression (64-bit) | result |
|---|---|
| 1e308 * 10 | inf |
| -1e308 * 10 | -inf |
| 1 / inf | 0 |
| inf - inf | nan |
| 0 * inf | nan |
| nan + 1 | nan |
| nan == nan | False |
| nan != nan | True |
| nan < 1  and  nan > 1 | False  and  False |
| max(nan, 1) | nan |
| max(1, nan) | 1 |
| sum of [1, nan, 2] | nan |
| sorted([3, nan, 1, 2]) | [3.0, nan, 1.0, 2.0] |
| inf > 1e308 | True |
<!-- /worked -->

The last rows are the dangerous ones. Every comparison with NaN, including
`nan == nan`, is false, so tests that look like "not less than" or "equals" can
be silently wrong. A maximum over a list is then order dependent: Python's
`max(nan, 1)` is `nan` but `max(1, nan)` is `1`, because the comparison it
relies on is false either way. Sorting a list that contains a NaN gives a list
that is not sorted at all. Code that must survive a NaN has to decide
explicitly where it goes; the case of a sampler choosing the next token was
Chapter 10's story, and a later section points at it.

### Underflow and subnormal numbers

At the other end, a result smaller than the smallest normal number $2^{e_{\min}}$
cannot keep $p$ significant bits. A format that simply returned $0$ for such
results would have a hole: two different numbers $x \ne y$ could have $x - y
= 0$. Instead the format uses subnormal numbers: below $2^{e_{\min}}$ the gap
stops shrinking and stays at its smallest value $2^{e_{\min}-p+1}$, so the
numbers become evenly spaced between $0$ and $2^{e_{\min}}$ (the orange points of the
toy picture). Differences of nearby numbers in that region are multiples of
the tiny gap and therefore exactly representable, so $x - y = 0$ only when $x
= y$. Computing with 32 and 64 bits:

<!-- worked: floating.gradual_underflow -->
| format | exact value | stored | spacing at that size | spacing / value |
|---|---:|---:|---:|---:|
| 32-bit | 2^-126 | 1.17549e-38 | 1.40e-45 | 1.19e-07 |
| 32-bit | 2^-127 | 5.87747e-39 | 1.40e-45 | 2.38e-07 |
| 32-bit | 2^-140 | 7.17465e-43 | 1.40e-45 | 0.00195 |
| 32-bit | 2^-149 | 1.4013e-45 | 1.40e-45 | 1 |
| 32-bit | 2^-150 | 0 | - | - |
| 64-bit | 2^-1022 | 2.22507e-308 | 4.94e-324 | 2.22e-16 |
| 64-bit | 2^-1023 | 1.11254e-308 | 4.94e-324 | 4.44e-16 |
| 64-bit | 2^-1050 | 8.28905e-317 | 4.94e-324 | 5.96e-08 |
| 64-bit | 2^-1074 | 4.94066e-324 | 4.94e-324 | 1 |
| 64-bit | 2^-1075 | 0 | - | - |
<!-- /worked -->

Read the last column. In the normal range it is about $u$ to $2u$; as the
numbers fall into the subnormal range it grows from $10^{-7}$ to $2 \times
10^{-3}$ to $1$: the number $2^{-149}$ has *no* significant digits beyond "it
is not zero". Half of the smallest subnormal rounds, by the tie rule, to zero.
A result that falls below the smallest subnormal is stored as $0$. This is
**underflow**: an exact nonzero result becoming zero, or losing most of its
digits on the way there.

So the representable range has a lower cliff (about $10^{-45}$ for 32 bits, with
precision lost gradually from about $10^{-38}$) and an upper cliff, and
Chapter 1's table of $e^x$ and of the product of $0.1$'s marked both. The
techniques of the rest of the chapter are mostly about staying away from the
cliffs without distorting the answer.

## Errors accumulate: summation

Sums are the commonest operation in machine learning: dot products, losses,
means, gradients accumulated over a batch. They are also where small errors
pile up.

### The bound

Add $x_1, \dots, x_n$ left to right. With $s_1 = x_1$ and $s_k =
\mathrm{fl}(s_{k-1} + x_k)$, each addition obeys the model of the previous
section: $s_k = (s_{k-1} + x_k)(1 + \delta_k)$ with $|\delta_k| \le u$.
Unrolling (and ignoring products of two $\delta$'s, which are of size $u^2$),
the first number $x_1$ and the second $x_2$ each pass through $n - 1$
additions, $x_3$ through $n - 2$, and so on, so that

$$
s_n \approx \sum_{i=1}^{n} x_i\,(1 + \theta_i), \qquad |\theta_i| \le (n - i + 1)\,u \le (n-1)\,u,
$$

and therefore

$$
\left| s_n - \sum_{i=1}^n x_i \right| \;\lesssim\; (n-1)\,u \sum_{i=1}^n |x_i| .
$$

The error of a left-to-right sum can grow in proportion to the *number of
terms* times the unit round-off, relative to the sum of the absolute values.
For $10^6$ terms in 32 bits $(n-1)u$ is about $0.06$, six percent. The bound is
a worst case that needs every rounding to push the same way. For random data
the errors partly cancel and the typical error is closer to $\sqrt{n}\,u$. For
data like our $0.1$'s, it is not random at all: once the total is large, each
addition of the same $0.1$ is rounded to the same grid in the same direction,
and the errors add up like the worst case. Here are the measurements.

<!-- worked: floating.sum_methods -->
Relative error of a 32-bit sum of n terms, against the exact sum of the stored terms.

| terms | n | exact sum | worst-case bound (n-1)u | left to right | pairwise | Kahan |
|---|---|---:|---:|---:|---:|---:|
| copies of 0.1 | 10^3 | 1.0000e+02 | 6.0e-05 | 9.55e-06 | 1.49e-08 | 1.49e-08 |
| copies of 0.1 | 10^4 | 1.0000e+03 | 6.0e-04 | 9.71e-05 | 1.49e-08 | 1.49e-08 |
| copies of 0.1 | 10^5 | 1.0000e+04 | 6.0e-03 | 1.44e-04 | 1.49e-08 | 1.49e-08 |
| copies of 0.1 | 10^6 | 1.0000e+05 | 6.0e-02 | 9.58e-03 | 1.49e-08 | 1.49e-08 |
| uniform in (0, 1) | 10^3 | 5.1676e+02 | 6.0e-05 | 5.48e-08 | 6.33e-08 | 5.48e-08 |
| uniform in (0, 1) | 10^4 | 5.0072e+03 | 6.0e-04 | 6.59e-07 | 2.34e-08 | 2.34e-08 |
| uniform in (0, 1) | 10^5 | 5.0032e+04 | 6.0e-03 | 1.04e-07 | 2.63e-08 | 2.63e-08 |
<!-- /worked -->

The left-to-right column for $0.1$'s grows with $n$ and reaches almost one
percent at a million terms (the bound is six). On random data the error is
much smaller, and there is no systematic drift. Pairwise summation and
compensated summation, to which we now turn, do not grow with $n$ at all on
the $0.1$'s.

### Two cures that cost nothing in precision

**Pairwise summation** adds the terms in a tree: sum the first half, sum the
second half, add the two results. Each term then passes through only about
$\lceil \log_2 n \rceil$ additions instead of up to $n - 1$, so the same
unrolling gives the bound

$$
\left| s_n - \sum_i x_i \right| \;\lesssim\; \lceil \log_2 n \rceil\, u \sum_{i=1}^n |x_i|,
$$

which is $20u$ for a million terms instead of $10^6 u$. The algorithm, with
the layer-by-layer form that needs no recursion:

```text
function pairwise_sum(x: array<f32>[N]) -> f32:
    require N > 0, "a sum of nothing has no first term"
    layer = copy(x)                         # array<f32>[N]
    while len(layer) > 1:
        m = len(layer) // 2
        next = zeros(f32, [m + len(layer) % 2])
        for i in 0 .. m:
            next[i] = layer[2 * i] + layer[2 * i + 1]
        if len(layer) % 2 == 1:
            next[m] = layer[len(layer) - 1]
        layer = next
    return layer[0]
```

(For the $0.1$'s the exact doubling at each level makes pairwise summation look
better than it is in general; the random rows of the table show its more
typical behaviour: its advantage over left to right depends on the data, from
none at $n = 10^3$ to a factor of about 28 at $n = 10^4$.)

**Compensated summation** takes a different route: keep the running total, and
a second number that remembers what each addition lost. It is due to Kahan.
To derive it, look at one addition $t = \mathrm{fl}(s + y)$ where $s$ is the
running total and $y$ the new term, with $|s| \ge |y|$. The total $t$ differs
from the exact $s + y$ by the rounding error $r = (s + y) - t$, and there is a
classical fact (the Fast2Sum theorem of Dekker and Knuth, which we use
without proof): when $|s| \ge |y|$, the subtraction $t - s$ is computed exactly
and gives the part of $y$ that actually made it into the total, and the second
subtraction $(t - s) - y$ is exact too, and equals $-r$. So with two
more subtractions the program obtains exactly what the addition lost. Kahan's
idea is to feed that loss into the next addition rather than throw it away:

```text
function kahan_sum(x: array<f32>[N]) -> f32:
    require N > 0, "a sum of nothing has no first term"
    s = 0.0                        # running total
    c = 0.0                        # what has been lost so far, as a negative number
    for i in 0 .. N:
        y = x[i] - c               # the term, corrected by the previous loss
        t = s + y                  # rounded: low bits of y are lost here
        c = (t - s) - y            # algebraically 0; in floating point it is -(lost part)
        s = t
    return s
```

A trace shows it working. Start at $s = 1$ in 32 bits and add $x = 4 \times
10^{-8}$ eight times. Half the gap above $1$ is $u = 5.96 \times 10^{-8}$, so
one $x$ alone is smaller than half a gap and is lost entirely by an ordinary
addition:

<!-- worked: floating.kahan_trace -->
Start at s = 1 in 32 bits and add x = 4.000000e-08 eight times (half the gap above 1 is 5.960e-08, so one such x alone is lost). Columns show the part above 1.

| step | y = x - c | Kahan s - 1 | new c | left to right s - 1 | exact sum - 1 |
|---|---:|---:|---:|---:|---:|
| 1 | 4.000e-08 | 0.0000e+00 | -4.000e-08 | 0.0000e+00 | 4.0000e-08 |
| 2 | 8.000e-08 | 1.1921e-07 | 3.921e-08 | 0.0000e+00 | 8.0000e-08 |
| 3 | 7.907e-10 | 1.1921e-07 | -7.907e-10 | 0.0000e+00 | 1.2000e-07 |
| 4 | 4.079e-08 | 1.1921e-07 | -4.079e-08 | 0.0000e+00 | 1.6000e-07 |
| 5 | 8.079e-08 | 2.3842e-07 | 3.842e-08 | 0.0000e+00 | 2.0000e-07 |
| 6 | 1.581e-09 | 2.3842e-07 | -1.581e-09 | 0.0000e+00 | 2.4000e-07 |
| 7 | 4.158e-08 | 2.3842e-07 | -4.158e-08 | 0.0000e+00 | 2.8000e-07 |
| 8 | 8.158e-08 | 3.5763e-07 | 3.763e-08 | 0.0000e+00 | 3.2000e-07 |
<!-- /worked -->

The left-to-right column never moves: $1 + 4 \times 10^{-8}$ rounds to $1$
every time, eight times in a row, and the total error is $3.2 \times 10^{-7}$,
nearly three gaps' worth. The compensated total follows the exact sum to within half a
gap (a gap is $1.19 \times 10^{-7}$) throughout. At step 1 the whole of $x$ is lost, and
$c$ records it as $-4 \times 10^{-8}$; at step 2 the term to be added is $x - c
= 8 \times 10^{-8}$, which is now more than half a gap and gets added (as a
whole gap, $1.19 \times 10^{-7}$), and $c$ becomes the small positive number
left over by that overshoot.

Kahan summation has an error bound of about $(2u + O(n u^2)) \sum_i |x_i|$,
independent of $n$ to first order, at the price of four additions per term in
place of one. Two practical notes. The step $c = (t - s) - y$ is zero in real
arithmetic, so a compiler or a language mode that is allowed to treat addition
as associative (the "fast math" modes) may simplify it away and silently turn
the algorithm into the naive sum. And the simplest alternative often
suffices: add the 32-bit terms into a 64-bit accumulator, which has 29 more bits
to absorb the low-order losses. The last row of the million-tenths table does
exactly that and is as accurate as the other two. Brain's cosine does this, as
the "in brain" section shows.

The figure compares the three on the opening problem, for $n$ from $10$ to
$10^6$.

![Relative error of a 32-bit sum of n copies of 0.1 against the number of terms n, both axes logarithmic. The worst-case bound (n minus 1) times u is a straight line rising from 10 to the minus 6 to 0.06. The left-to-right sum follows it about a factor of 10 to 100 lower, ending near 0.01 at a million terms. The pairwise and Kahan sums coincide, stay near the unit round-off u at about 10 to the minus 7.5 or below, and do not grow.](img/summation-error.svg)

Chapter 1 left two loose ends of exactly this kind, and they can be closed now:
the limit $(1 + 1/n)^n$ that collapsed to $1$, and the exercise of adding $\log
0.1$ a large number of times.

<!-- worked: floating.ch1_loose_ends -->
The limit (1 + 1/n)^n of chapter 1, in 64 bits:

| n | (1 + 1/n)^n directly | exp(n * log1p(1/n)) | stable minus e |
|---|---:|---:|---:|
| 10^8 | 2.718281798347 | 2.718281814868 | -1.36e-08 |
| 10^12 | 2.718523496037 | 2.718281828458 | -1.36e-12 |
| 10^15 | 3.035035206549 | 2.718281828459 | -1.33e-15 |
| 10^16 | 1.000000000000 | 2.718281828459 | 0.00e+00 |

The sum of logs of chapter 1, exact sum -2302585.1250:

| 32-bit sum of 10^6 copies of log 0.1 | computed | relative error |
|---|---:|---:|
| left to right | -2276958.25 | 1.11e-02 |
| pairwise | -2302585 | 5.43e-08 |
| Kahan | -2302585 | 5.43e-08 |
<!-- /worked -->

Computing $(1 + 1/n)^n$ as $\exp(n \log(1 + 1/n))$ with the library function
`log1p` (which computes $\log(1 + t)$ accurately even when $t$ is tiny, as we
see in the next section) recovers $e$ at every $n$, even $10^{16}$. Its error
column is just the truncation error $e/(2n)$ of the limit itself. And the sum
of a million copies of $\log 0.1$ in 32 bits is off by about one percent when
added left to right, but exact to the format's precision when added by either
careful method.

## Cancellation

Addition of positive numbers is benign: the bound above is proportional to the
sum of the absolute values, which equals the absolute value of the result. When
the terms have mixed signs, the result can be far smaller than $\sum |x_i|$,
and the same absolute error is then a large *relative* error. The extreme case
is subtracting two numbers that are nearly equal.

### Losing digits to a subtraction

Evaluate $(1 + x) - 1$ for smaller and smaller $x$. In exact arithmetic the
answer is $x$.

<!-- worked: floating.one_plus_x -->
In 64 bits. The first error column compares (1 + x) - 1 with x; the other two compare e^x - 1 computed naively and with `expm1` against the true value.

| x | (1 + x) - 1 | relative error | exp(x) - 1: error | expm1(x): error |
|---|---:|---:|---:|---:|
| 1e-1 | 0.10000000000000009 | 8.33e-16 | 7.75e-16 | 1.68e-17 |
| 1e-4 | 9.999999999998899e-05 | 1.10e-13 | 4.33e-13 | 3.45e-18 |
| 1e-8 | 9.99999993922529e-09 | 6.08e-09 | 1.11e-08 | 6.76e-17 |
| 1e-12 | 1.000088900582341e-12 | 8.89e-05 | 8.89e-05 | 2.42e-17 |
| 1e-15 | 1.1102230246251565e-15 | 1.10e-01 | 1.10e-01 | 9.16e-17 |
| 1e-16 | 0.0 | 1.00e+00 | 1.00e+00 | 5.00e-17 |
| 1e-17 | 0.0 | 1.00e+00 | 1.00e+00 | 5.00e-18 |
<!-- /worked -->

The first error column is what we want to understand. At $x = 10^{-1}$ the
relative error is $8 \times 10^{-16}$, the full 16 digits of the format. At
$10^{-8}$ it is $6 \times 10^{-9}$: only 8 digits are right. At $10^{-12}$ four digits
survive. At $10^{-16}$ there are none: $1 + x$ rounds to $1$ and the answer is
$0$.

The reason is visible in the steps. The number $a = \mathrm{fl}(1 + x)$ is
stored with an error of up to $u$ (absolute, because $a \approx 1$). The
subtraction $a - 1$ then removes the leading digits, which were identical,
and what is left is the tiny difference $x$ *plus that error*:

$$
\mathrm{fl}(1 + x) - 1 = x + \eta, \qquad |\eta| \le u, \qquad
\text{relative error} \;\le\; \frac{u}{x} .
$$

A relative error $u/x$ means about $\log_{10}(1/x)$ decimal digits are lost: 8
digits for $x = 10^{-8}$. We call this **catastrophic cancellation**:
subtracting two nearly equal numbers, each carrying a rounding error, so that
the leading digits cancel and the digits that survive are mostly error.

The subtraction itself is not guilty. There is a theorem (Sterbenz) that if $y/2 \le x \le 2y$ then $x - y$ is computed
*exactly*, with no rounding at all. A test on random 32-bit numbers:

<!-- worked: floating.sterbenz -->
Subtracting two 32-bit numbers x, y: when y/2 <= x <= 2y the stored difference was inexact in 0 of 100000 random cases; when x is 4 to 5 times y it was inexact in 71411 of 100000.
<!-- /worked -->

Within a factor of two the stored difference was never inexact; outside it
usually was. So cancellation does not create errors; it *exposes* errors
committed earlier, by promoting the lowest digits of the operands to the highest
digits of the result. If the operands are exact (the integers $10^8 + 1$ and
$10^8$ stored exactly) the subtraction is perfectly accurate: benign
cancellation. It is catastrophic only when the operands were already rounded.

### Curing it: reformulate before the subtraction

Since the damage comes from an earlier rounding, the cure is to avoid forming
the two nearly equal numbers at all. Three standard examples.

**Functions with a built-in subtraction.** The functions $e^x - 1$ and $\log(1 +
x)$ for small $x$ are so common that libraries provide them directly, as `expm1`
and `log1p`, computed from series that never form $1 + x$ or $e^x$. The last two
columns of the table compare: `exp(x) - 1` loses digits exactly like $(1 +
x) - 1$, while `expm1` is accurate to about $10^{-17}$ for every $x$.

**The quadratic formula.** The roots of $x^2 + bx + c = 0$ are $\bigl(-b \pm
\sqrt{b^2 - 4c}\bigr)/2$. When $b$ is large and positive, $\sqrt{b^2 - 4c}
\approx b$, and the root with $-b + \sqrt{\cdot}$ subtracts two nearly equal
numbers. The cure uses the fact that the product of the roots is $c$: compute
the large root $q = -(b + \sqrt{b^2 - 4c})/2$ (no subtraction: both terms are
negative) and then the small root as $c/q$.

<!-- worked: floating.quadratic -->
The smaller-magnitude root of x^2 + b x + 1 = 0, in 64 bits.

| b | (-b + sqrt(b^2 - 4)) / 2 | c / q with q = -(b + sqrt(b^2 - 4)) / 2 | true root | error of the first | error of the second |
|---|---:|---:|---:|---:|---:|
| 1e2 | -0.0100010002 | -0.0100010002 | -0.0100010002 | 1.22e-13 | 2.17e-17 |
| 1e4 | -0.0001000000011 | -0.000100000001 | -0.000100000001 | 1.12e-09 | 5.65e-17 |
| 1e6 | -1.000007614e-06 | -1e-06 | -1e-06 | 7.61e-06 | 8.89e-17 |
| 1e8 | -7.450580597e-09 | -1e-08 | -1e-08 | 2.55e-01 | 7.91e-17 |
<!-- /worked -->

At $b = 10^8$ the textbook formula is wrong in the very first digit ($-7.45
\times 10^{-9}$ for a true root of $-1.0 \times 10^{-8}$, an error of 25 percent);
the second formula is right to 16 digits in every row.

**The variance.** Chapter 10 defined the variance as the mean of squared
deviations. A formula that looks cheaper, $\mathbb{E}[X^2] - \mathbb{E}[X]^2$,
needs only a running sum of $x$ and of $x^2$ and no second pass over the data.
When the mean $\mu$ is large compared with the spread $\sigma$, the two terms
are both about $\mu^2$ and their difference is $\sigma^2$: the ratio of the
largest to the final value is $\mu^2/\sigma^2$, and about that many digits are
lost to cancellation. Compare three methods on $1000$ numbers equal to an
offset plus a standard normal variable, so that the true variance is about $1$:

<!-- worked: floating.variance -->
Variance of 1000 values offset + a standard normal, population form (divide by n).

| format | offset | true variance | E[X^2] - E[X]^2 | its error | two-pass error | Welford error |
|---|---:|---:|---:|---:|---:|---:|
| 32-bit | 10^1 | 0.931372 | 0.931366 | 6.41e-06 | 2.70e-07 | 2.42e-07 |
| 32-bit | 10^2 | 1.064524 | 1.06055 | 3.74e-03 | 5.56e-08 | 3.92e-07 |
| 32-bit | 10^3 | 0.902852 | 1.5625 | 7.31e-01 | 9.69e-07 | 4.83e-07 |
| 32-bit | 10^4 | 1.032536 | 32 | 3.00e+01 | 1.23e-05 | 1.11e-04 |
| 64-bit | 10^4 | 0.968590 | 0.96859 | 2.07e-07 | 5.02e-16 | 3.65e-13 |
| 64-bit | 10^8 | 0.976609 | 14 | 1.33e+01 | 1.31e-15 | 1.28e-09 |
| 64-bit | 10^9 | 1.094648 | 896 | 8.18e+02 | 1.75e-15 | 8.35e-09 |
<!-- /worked -->

In 32 bits the one-pass formula has lost everything by an offset of $10^3$
($\mu^2/\sigma^2 = 10^6$: six of the 7 digits the format holds are gone, leaving about one), and returns a variance
of $1.56$ where the truth is $0.90$; at $10^4$ it returns $32$. In 64 bits the
same collapse happens at $10^8$ and $10^9$. The **two-pass** method, which first
computes the mean and then sums $(x_i - \bar{x})^2$, subtracts the mean *before*
squaring, so that the numbers being squared are of the size of the spread; it
is accurate to $10^{-5}$ or better in every row in 32 bits and to about $10^{-15}$ in 64 bits. **Welford's** one-pass method keeps
a running mean and a running sum of squared deviations from the current mean:

```text
function variance_welford(x: array<f32>[N]) -> f32:
    require N > 0, "the variance of no values is undefined"
    mean = 0.0                      # running mean of the first k values
    m2 = 0.0                        # running sum of squared deviations from the running mean
    for k in 1 .. N + 1:
        d = x[k - 1] - mean
        mean = mean + d / k
        m2 = m2 + d * (x[k - 1] - mean)
    return m2 / N                   # population variance
```

It never forms a square of a large number, so it needs only one pass and
survives offsets that destroy the one-pass formula (at offset $10^8$ in 64 bits its
error is $10^{-9}$ where the one-pass formula's is $13$). On this data it is less
accurate than two-pass (the running mean carries rounding errors of its own),
which is the usual trade: two-pass when the data can be read twice, Welford when
it cannot.

## Conditioning of a computation

Cancellation is one instance of a general question: *given a small relative
error in the input, how large a relative error can there be in the output, even
with perfect arithmetic?* Chapter 8 defined the condition number of a bowl as the
ratio of its largest to smallest curvature, and the number told us how hard
descent would be. Here the idea is the same in spirit and applies to any function
of one variable.

Perturb the input $x$ to $x(1 + \epsilon)$ with $|\epsilon|$ tiny. By the
first-order Taylor expansion of Chapter 8, $f\bigl(x(1 + \epsilon)\bigr) \approx
f(x) + f'(x)\,x\epsilon$, so the relative change of the output is

$$
\frac{f\bigl(x(1+\epsilon)\bigr) - f(x)}{f(x)} \approx \frac{x f'(x)}{f(x)}\,\epsilon .
$$

The factor in front is the **relative condition number** of $f$ at $x$,
$\kappa_f(x) = \bigl|x f'(x)/f(x)\bigr|$. It is the amplification factor from relative input
error to relative output error. Computed in 60-digit arithmetic with an input
change of $10^{-12}$:

<!-- worked: floating.condition_numbers -->
Relative condition number |x f'(x) / f(x)| against the measured ratio (relative change of the output) / (relative change of the input) for an input change of 1e-12, computed with 60-digit arithmetic.

| f | x | |x f'(x) / f(x)| | measured ratio |
|---|---:|---:|---:|
| sqrt(x) | 2 | 0.5 | 0.5 |
| x^10 | 2 | 10 | 10 |
| 1/x | 7 | 1 | 1 |
| exp(x) | 30 | 30 | 30 |
| log(x) | 1.001 | 1000.5 | 1000.5 |
| log(x) | 2 | 1.4427 | 1.4427 |
| x - 1 | 1.001 | 1001 | 1001 |
| x - 1 | 1.00000001 | 1e+08 | 1e+08 |
| exp(x) - 1 | 1e-08 | 1 | 1 |
<!-- /worked -->

The measured ratio equals the formula to every digit shown. Reading the table: $\sqrt{x}$ has $\kappa =
\tfrac12$ (it *damps* relative error), $x^{10}$ has $\kappa = 10$, $e^{30}$ has
$\kappa = 30$ (the exponential amplifies in proportion to its argument), and
$\log x$ near $1$ has $\kappa = 1/|\log x|$, which is $1000$ at $x = 1.001$.
Most important for this chapter, $x - 1$ has $\kappa = x/|x - 1|$, which is
$10^8$ at $x = 1 + 10^{-8}$. That number is exactly the digit loss of the
previous section: an input rounded with relative error $u$ gives an output with
relative error $\kappa u = 10^8 u \approx 10^{-8}$. Cancellation is
ill-conditioning of the subtraction.

Two consequences organise how we think about numerical error.

**A floor.** Rounding the input to a stored number is already a relative
perturbation of size $u$, so even a perfect algorithm that sees only the stored
input cannot do better than about $\kappa_f(x)\,u$. An ill-conditioned problem
(large $\kappa$) is one where no method can give more than $\log_{10}(1/(\kappa u))$
good digits. Nothing in the program can be cured; the question itself is
fragile. Chapter 8's long-valley bowls were exactly the case in which
the answer to a question depends delicately on the data.

**A standard.** An algorithm for computing $f$ is **stable** when its error is
about $\kappa_f(x)\,u$, as good as the floor, and **unstable** when it is much
worse. The two sentences "the problem is ill-conditioned" and "the method is
unstable" are different statements and must not be confused. Compare:

- $f(x) = e^x - 1$ at $x = 10^{-8}$ has $\kappa = 1.0$ (last row of the table): a
  perfectly well-conditioned problem. The method "compute $e^x$, subtract $1$" has
  relative error $\approx u/x = 10^{-8}$ (see the one-plus-$x$ table): unstable.
  `expm1` is stable, with error near $u$.
- The small root of the quadratic with $b = 10^8$ is also well-conditioned
  (the root is about $-c/b$, and a relative error in $b$ gives the same relative
  error in the root). The textbook formula is unstable; $c/q$ is stable.
- $f(x) = x - 1$ at $x = 1 + 10^{-8}$ has $\kappa = 10^8$. Nothing helps, because
  the input $x$ was already rounded.

Conditioning composes by the chain rule of Chapter 7. For $h = g \circ f$ we have
$\kappa_h(x) = \kappa_g\bigl(f(x)\bigr)\,\kappa_f(x)$: the relative changes
multiply, just as the derivatives do. A calculation made of many
well-conditioned steps is well-conditioned; a single ill-conditioned step makes the
whole of it ill-conditioned *if its input is already rounded*, and the catastrophic
cancellation of this chapter is what that looks like. The standard remedy is the one
used above: rearrange the computation so that the ill-conditioned step is avoided, or
so that its input is exact.

## Numerical differentiation: the best step

Chapter 7 found that a finite-difference derivative has a best step and
postponed the derivation. We now have every ingredient. We estimate $f'(x)$ by

$$
D_{\mathrm{f}}(h) = \frac{f(x + h) - f(x)}{h}, \qquad
D_{\mathrm{c}}(h) = \frac{f(x + h) - f(x - h)}{2h},
$$

the forward and the central difference. Two errors compete.

**Truncation error** comes from the mathematics and shrinks with $h$.
The Taylor expansion of Chapter 8 gives
$D_{\mathrm{f}}(h) - f'(x) \approx \tfrac12 f''(x)\,h$ and $D_{\mathrm{c}}(h) -
f'(x) \approx \tfrac16 f'''(x)\,h^2$.

**Round-off error** comes from the arithmetic and grows as $h$ shrinks. The
function values $f(x \pm h)$ are computed with a relative error of order $u$,
so each carries an absolute error up to $u|f|$. The subtraction of the two nearby values is itself exact
(Sterbenz), but it passes those two errors through, so the numerator is
uncertain by up to $2u|f|$; the division then magnifies that uncertainty
by $1/h$ (forward) or $1/(2h)$ (central):

$$
\text{round-off}_{\mathrm{f}} \le \frac{2u|f|}{h}, \qquad
\text{round-off}_{\mathrm{c}} \le \frac{u|f|}{h} .
$$

The total worst-case error is the sum, a function with one valley:

$$
E_{\mathrm{f}}(h) = \frac{M_2}{2}\,h + \frac{2u|f|}{h}, \qquad
E_{\mathrm{c}}(h) = \frac{M_3}{6}\,h^2 + \frac{u|f|}{h},
$$

where $M_2$ and $M_3$ bound $|f''|$ and $|f'''|$ near $x$. Setting the
derivative of each in $h$ to zero:

$$
\frac{M_2}{2} - \frac{2u|f|}{h^2} = 0 \;\Rightarrow\; h_{\mathrm{f}}^* = 2\sqrt{\frac{u|f|}{M_2}},
\qquad
\frac{M_3}{3}\,h - \frac{u|f|}{h^2} = 0 \;\Rightarrow\; h_{\mathrm{c}}^* = \left(\frac{3u|f|}{M_3}\right)^{1/3}.
$$

At those steps the two error terms are of the same size, and the total is
$E_{\mathrm{f}}^* = 2\sqrt{u|f|M_2}$ for the forward difference and
$E_{\mathrm{c}}^* = \tfrac12 M_3 (h_{\mathrm{c}}^*)^2$ for the central one.
Two rules follow. The best step is $h \sim \sqrt{u}$ for a forward and $h \sim u^{1/3}$ for a
central difference, times a scale set by the function (for functions whose
value and derivatives are of the same size, a number of order $1$, as in the
rule of Chapter 7). And the best achievable error is of order $\sqrt{u}$ and
$u^{2/3}$: the forward difference delivers only *half* of the digits of the
format, and the central difference two thirds.

Measured, for $f = e^x$ at $x = 1$ with the function and all arithmetic rounded to
each format (the $h$ actually used is $\mathrm{fl}(1 + h) - 1$, so that the
step in the denominator is the one the arithmetic took):

<!-- worked: floating.numdiff_precision -->
Error of the difference quotient for the derivative of e^x at x = 1 (exactly e), with the function evaluated and the arithmetic done in each format.

| h | 32-bit forward | 32-bit central | 64-bit forward | 64-bit central |
|---|---:|---:|---:|---:|
| 1e-1 | 1.41e-01 | 4.53e-03 | 1.41e-01 | 4.53e-03 |
| 1e-2 | 1.36e-02 | 3.83e-05 | 1.36e-02 | 4.53e-05 |
| 1e-3 | 1.47e-03 | 1.59e-06 | 1.36e-03 | 4.53e-07 |
| 1e-4 | 1.62e-03 | 4.31e-04 | 1.36e-04 | 4.53e-09 |
| 1e-5 | 1.98e-02 | 7.91e-03 | 1.36e-05 | 5.59e-11 |
| 1e-6 | 3.17e-02 | 5.16e-02 | 1.36e-06 | 9.07e-11 |
| 1e-7 | 1.28e+00 | 2.82e-01 | 1.38e-07 | 1.96e-11 |
| 1e-8 | 2.72e+00 | 2.72e+00 | 9.92e-09 | 5.17e-09 |
| 1e-9 | 2.72e+00 | 2.72e+00 | 9.47e-09 | 8.06e-08 |
| 1e-10 | 2.72e+00 | 2.72e+00 | 1.32e-06 | 8.98e-07 |
| 1e-11 | 2.72e+00 | 2.72e+00 | 3.24e-05 | 1.02e-05 |
| 1e-12 | 2.72e+00 | 2.72e+00 | 1.91e-04 | 1.20e-04 |
| 1e-13 | 2.72e+00 | 2.72e+00 | 1.72e-03 | 2.08e-04 |
| 1e-14 | 2.72e+00 | 2.72e+00 | 7.17e-03 | 7.17e-03 |
| 1e-15 | 2.72e+00 | 2.72e+00 | 8.17e-02 | 1.86e-02 |
<!-- /worked -->

![Base-10 logarithm of the error of the forward (solid) and central (dashed) difference estimates of the derivative of e to the x at x = 1 against minus the base-10 logarithm of the step h, in 32-bit (orange) and 64-bit (blue) arithmetic. In 64 bits the central error falls with slope 2 to about 10 to the minus 11 near h = 10 to the minus 5.5 and then rises; the forward error falls with slope 1 to a few times 10 to the minus 8 near 10 to the minus 8. The 32-bit curves reach their minimum at much larger steps and are many orders of magnitude worse, and the 32-bit curves are flat at 2.7 once h is below about 6 times 10 to the minus 8 because the step vanishes. Vertical marks show the predicted best steps.](img/numdiff-precision.svg)

For the 64-bit curves the picture matches Chapter 7's: truncation error falls
(slope 1 for forward, slope 2 for central), round-off error rises (slope $-1$), and
the minimum lies near the predicted step. In 32 bits the same shapes appear but the
valley sits where $h$ is thousands of times larger, and below $h \approx 6 \times 10^{-8}$ (that is, $u$)
the step vanishes completely: $1 + h$ rounds to $1$, so the numerator is $0$
(the quotient is $0$ or $0/0$) and the error is the whole derivative $e = 2.72$. A scan of $h = 2^{-1}, \dots, 2^{-44}$ finds the minimum for each
combination and compares it with the formulas ($M_2 = M_3 = |f| = e$):

<!-- worked: floating.numdiff_optimum -->
Best step for the derivative of e^x at x = 1, scanning h = 2^-1 ... 2^-44. Predicted: forward h* = 2 sqrt(u), error 2e sqrt(u); central h* = (3u)^(1/3), error e h*^2/6 + u e / h*.

| format | difference | predicted h* | best h found | predicted worst-case error | best error found |
|---|---:|---:|---:|---:|---:|
| 32-bit | forward | 4.88e-04 | 2.44e-04 | 1.33e-03 | 4.68e-04 |
| 32-bit | central | 5.63e-03 | 3.91e-03 | 4.31e-05 | 1.04e-05 |
| 64-bit | forward | 2.11e-08 | 1.49e-08 | 5.73e-08 | 3.67e-08 |
| 64-bit | central | 6.93e-06 | 3.81e-06 | 6.53e-11 | 9.94e-12 |
<!-- /worked -->

The predicted steps are within a factor of two of the best found, and the
best error found is between 1.6 and 6.6 times below the *worst-case* prediction,
which is what an analysis that assumes every rounding error is of maximal size
and aligned should deliver. This is why a 32-bit gradient check needs a step of about
$5 \times 10^{-3}$ where a 64-bit one uses about $10^{-5}$: they are
$(3u)^{1/3}$ for $u = 2^{-24}$ and $u = 2^{-53}$, which are $5.6 \times 10^{-3}$
and $6.9 \times 10^{-6}$. The 32-bit central difference can give only 5 good
digits at best, in the ideal case; a derivative check in that format must be read with a
tolerance of at least that size, and in practice larger because a loss is a sum
of many terms, each rounded.

## Log-sum-exp

We now come to the technique that Chapter 1 promised. A model that outputs a
score $z_i$ for each of $n$ alternatives (words of a vocabulary, say) needs, to turn
scores into probabilities and to compute a loss, the quantity

$$
\mathrm{lse}(z_1, \dots, z_n) = \log \sum_{i=1}^{n} e^{z_i},
$$

the **log-sum-exp** of the scores. (It is the logarithm of a sum of
exponentials; in Volume III it is the denominator of the softmax and the heart
of cross-entropy.) The formula as written is a trap, for the reason given by
Chapter 1's table of $e^x$: $e^{z}$ overflows in 32 bits for $z$ above about $88.7$ and underflows
to zero for $z$ below about $-104$, so scores that are entirely reasonable make the sum
infinite or zero.

<!-- worked: floating.lse_table -->
| z | 32-bit naive | 32-bit shifted | 64-bit naive | 64-bit shifted | exact |
|---|---:|---:|---:|---:|---:|
| [1, 2, 3] | 3.4076059 | 3.4076059 | 3.407606 | 3.407606 | 3.407606 |
| [88, 88, 88] | inf | 89.09861 | 89.098612 | 89.098612 | 89.098612 |
| [89, 89, 89] | inf | 90.09861 | 90.098612 | 90.098612 | 90.098612 |
| [1000, 1000, 1000] | inf | 1001.0986 | inf | 1001.0986 | 1001.0986 |
| [1000, 0, -1000] | inf | 1000 | inf | 1000 | 1000 |
| [-1000, -1000, -1000] | -inf | -998.90137 | -inf | -998.90139 | -998.90139 |
| [-800, -801] | -inf | -799.68677 | -inf | -799.68674 | -799.68674 |
<!-- /worked -->

The "32-bit naive" column fails already at $z = 88$ (three terms of $e^{88}$
are each finite, but their sum exceeds the largest 32-bit number), and the
64-bit one at $z = 1000$. For three scores of $-1000$ the sum underflows to
$0$ and the logarithm of $0$ is $-\infty$. The answer in every case is an
ordinary number: $1001.0986$, $-998.9014$.

The repair rests on one identity. For any constant $c$,

$$
\log \sum_i e^{z_i} = \log \Bigl( e^{c} \sum_i e^{z_i - c} \Bigr) = c + \log \sum_i e^{z_i - c} .
$$

(Factor $e^c$ out of every term; the logarithm of a product is the sum of
logarithms, Chapter 1.) The two sides are equal for every $c$, so $c$ is
free to be chosen for arithmetic's sake. Choose $c = \max_i z_i =: m$. Then
every exponent $z_i - m$ is at most $0$, the largest term is exactly
$e^0 = 1$, and the sum lies in $[1, n]$:

$$
\mathrm{lse}(z) = m + \log \sum_{i=1}^{n} e^{z_i - m}, \qquad 1 \le \sum_i e^{z_i - m} \le n .
$$

No term can overflow (each is at most $1$). The sum cannot be zero, since it
contains the term $1$, so the logarithm is safe. Terms that underflow to $0$
are those smaller than about $10^{-45}$ relative to the largest, and omitting
them changes a sum that is at least $1$ in the 45th decimal place: that is
harmless. The shifted column of the table agrees with the exact column in
every row, in both formats.

The same derivation gives a sanity check worth remembering. Since the sum lies
in $[1, n]$,

$$
\max_i z_i \;\le\; \mathrm{lse}(z) \;\le\; \max_i z_i + \log n :
$$

log-sum-exp is a smooth maximum, never below the largest score and never more
than $\log n$ above it. The algorithm:

```text
function logsumexp(z: array<f32>[N]) -> f32:
    require N > 0, "the log-sum-exp of nothing is undefined"
    m = z[0]
    for i in 1 .. N:
        if z[i] > m:
            m = z[i]
    s = 0.0
    for i in 0 .. N:
        s = s + exp(z[i] - m)
    return m + log(s)
```

It reads the data twice, once for the maximum and once for the sum. (A
variant that keeps a running maximum and rescales the running sum when the
maximum changes needs one pass; the flash-attention chapter of Volume V builds it.)

Two uses close Chapter 1's story. First, Chapter 1 showed that a probability of
a long text, a product of 400 factors of $0.1$, underflows, and that its logarithm,
$-921$, is fine. It then noted that *adding* probabilities, which a mixture of models needs,
cannot be done with the product rule for logarithms. Log-sum-exp is the missing
rule: $\log(a + b) = \mathrm{lse}(\log a, \log b)$. A text with log-likelihoods
$-921$, $-950$ and $-1100$ under three models with mixture weights $0.5$, $0.3$,
$0.2$:

<!-- worked: floating.mixture_loglik -->
A text has log-likelihood -921, -950, -1100 under three models with mixture weights 0.5, 0.3, 0.2.

| step | value |
|---|---|
| log w_k + log p_k | -921.6931, -951.2040, -1101.6094 |
| sum of the probabilities, then log | sum = 0.0, log = -inf |
| log-sum-exp (shift by the largest) | -921.693147 |
| one pair at a time with log1p | -921.693147 |
| exact (60-digit arithmetic) | -921.693147 |
<!-- /worked -->

The probabilities are all of order $10^{-400}$ and the direct sum is exactly zero:
the log is $-\infty$. The shifted version gives $-921.693$, and so does the
pairwise form $\log(e^a + e^b) = \max(a, b) + \mathrm{log1p}\bigl(e^{-|a - b|}\bigr)$, which
is the two-term case of the same shift and uses `log1p` for the small correction.

Second, the loss of a classifier. The cross-entropy of the right answer $t$ is
$-\log \mathrm{softmax}(z)_t = \mathrm{lse}(z) - z_t$: never form the
probabilities, then take a logarithm; the log of the softmax is the score minus
the log-sum-exp.

<!-- worked: floating.cross_entropy_row -->
Cross-entropy -log softmax(z)[t] = lse(z) - z[t] for z = [1000.0, 998.0, 990.0, 0.0, -5.0], t = 1; exact value 2.12696800.

| format | unshifted lse - z[t] | shifted lse - z[t] |
|---|---:|---:|
| 32-bit | inf | 2.1269531 |
| 64-bit | inf | 2.126968 |
<!-- /worked -->

The unshifted version overflows for scores of order $1000$, and the shifted
version is right in both formats (to 32-bit precision in the first). The scores
of a language model are rarely that large, but the same trap appears in other
guises (for example when scores are multiplied by a temperature or a scale
before the exponential).

## Making it work

- **Compare with a tolerance, derived from the problem.** Never test floating-point
  results for equality. A tolerance is a few times $\kappa\,u$ for a stable
  computation, or $(n-1)u$ times the sum of absolute values for a naive sum.
- **Know your digits.** Seven digits in 32 bits, sixteen in 64. Count the digits a
  step will lose: $\log_{10}$ of the ratio of the largest quantity formed to the
  final result is a lower bound on the digits gone.
- **Subtract late and only exact things.** Rewrite so that nearly equal quantities
  are never subtracted when they are rounded results: `expm1`, `log1p`, the
  two-pass variance, the stable quadratic.
- **Sum with care.** Use a wider accumulator, pairwise summation, or compensated
  summation for any long reduction in 32 bits or less; check that no optimisation
  level removes the compensation.
- **Work in log space and shift by the maximum.** Products of probabilities become sums of
  logs; sums of probabilities become log-sum-exp; never exponentiate an unshifted
  score.
- **Make the finite-difference step depend on the format.** About
  $(3u)^{1/3}$ times a scale for central differences.
- **Keep clamps out of formulas.** A floor like `max(p, 1e-12)` protects a division
  or a logarithm but changes the function. The story in the next section shows what it
  costs.

## Where this appears in brain

**Accumulating in 64 bits.** `crates/model/src/hostmath.rs` holds the shared
`cosine` and `l2_normalize`, which accumulate the dot product and the squared norms
in 64-bit floating point even though their inputs are 32-bit; Chapter 3 told the story of
commit `f94c9dcda`, which put them there. The doc comment of `cosine` says that a
512-dimensional embedding is gated at seven digits of cosine and that a 32-bit dot product
loses the last two. The same experiment, run here on pairs of 512-dimensional
vectors with a cosine near $0.995$ and 200 random trials:

<!-- worked: floating.cosine_accumulation -->
Cosine of 200 pairs of 512-dimensional vectors (the second is the first plus noise of one tenth of its standard deviation, cosine near 0.995), against the exact value.

| accumulation | mean abs. error | max abs. error |
|---|---:|---:|
| 32-bit products and sums | 2.91e-07 | 1.10e-06 |
| 64-bit sums of the same 32-bit inputs, result rounded to 32 bits | 1.42e-08 | 2.97e-08 |
<!-- /worked -->

The mean error of the 32-bit accumulation is about five units of $u$ and its
worst case is $1.1 \times 10^{-6}$: the cosine is right to about $6$ digits instead of
$7$, so one digit is lost in the typical case (the doc comment says two). With a 64-bit
accumulator the remaining error is the rounding of the final result to 32 bits. The accumulator width is the cheapest of this
chapter's cures, and this function takes it.

**Log-sum-exp in the loss and the host.** The cross-entropy kernel
`crates/kernels/wgsl/ce_value.wgsl` computes, per row, a first pass for the maximum
of the logits, a second for $\sum e^{z - m}$, and writes `(mx + log(sum)) - logits[target]`:
exactly $\mathrm{lse}(z) - z_t$. `crates/model/src/logprobs.rs`
(`row_logprob`, whose doc comment says "numerically-stable") and the host `softmax`
in `crates/model/src/hostmath.rs` use the same shift. The kernel starts its running
maximum at `-3.4e38`, just above the most negative finite 32-bit number, rather than $-\infty$, so that any
finite logit replaces it. Another piece in the same spirit is
`log_ndtr` in `crates/model/src/hostmath.rs` (with its kernel twin in
`crates/kernels/wgsl/lib/normal.wgsl`), the logarithm of the normal distribution's
cumulative probability, which stays finite deep into the tail where the probability itself
would underflow; commit `3f8386bd2` (October 2026) added it, and its test
`log_ndtr_is_accurate_from_the_far_tail_to_the_upper_half` compares against
values computed in arbitrary precision.

**Story: a floor that deleted the gradient.** `crates/rlcd/src/scoring.rs` computes a
decision loss, a cross-entropy against a target distribution over a few options. The
first version floored every probability at $10^{-12}$ before taking its logarithm
and its reciprocal, the usual defence against $\log 0$. Commit `1bd342f7b`
(September 2026) records the consequence, and knowledge #155 teaches it. The
gradient of cross-entropy with respect to a score is $p - \text{target}$ at every
gap; with the floor it was not. Reproducing the commit's example, a target of
$(0.8, 0.2)$ over two scores $(-\text{gap}, 0)$:

<!-- worked: floating.clamped_gradient -->
Gradient of the loss with respect to the first score, for scores (-gap, 0) and target (0.8, 0.2). The last column is the closed form p - target.

| gap | p of the first option | gradient with the floor | p - target |
|---|---:|---:|---:|
| 10 | 4.54e-05 | -8.000e-01 | -0.800 |
| 27 | 1.88e-12 | -8.000e-01 | -0.800 |
| 30 | 9.36e-14 | -7.486e-02 | -0.800 |
| 100 | 3.72e-44 | -2.976e-32 | -0.800 |
| 400 | 1.92e-174 | -1.532e-162 | -0.800 |
<!-- /worked -->

Up to a gap of about $27$ ($p$ above the floor of $10^{-12}$) the gradient is right. At
a gap of $30$ it is $-0.075$ instead of $-0.8$, and at $100$ it is $-3 \times
10^{-32}$. The model that has become *confidently wrong* receives no gradient with
which to recover, and the loss plateaus at $-\log 10^{-12}$ with nothing to say why.
The knowledge entry adds the lesson about testing: the finite-difference check could not
see it, because the floor flattens the loss by the same amount that it flattens the
analytic gradient, so the two agree and are both wrong. The fix is this chapter's
technique rather than another patch: `softmax_with_log` in the same file computes
the softmax and its logarithm in 64 bits from one pass shifted by the maximum
(its doc comment notes that $p$ underflows at a gap near $745$, where
Chapter 1's table put the 64-bit limit of $e^x$, while $\log p$ is merely
$-745$ and exact), and `focal_slope_times_p` forms $p$ times the slope as a single
quantity so that the division by $p$ is never performed. The regression test is
`cross_entropy_keeps_its_gradient_when_a_prediction_saturates`, which checks the
closed form $p - \text{target}$ at gaps up to $400$.

**Story: an exact identity that was unusable.** The chunked recurrence of the
Gated DeltaNet layers (a model family built in Volume VI) needs the inverse of
$\mathbf{I} - \mathbf{A}$ for a strictly lower triangular matrix $\mathbf{A}$ of
size 64 whose entries are bounded by $1$. Such a matrix is nilpotent
($\mathbf{A}^{64} = \mathbf{0}$), so
$(\mathbf{I} - \mathbf{A})^{-1} = \sum_k \mathbf{A}^k = \prod_m
(\mathbf{I} + \mathbf{A}^{2^m})$: an exact identity, and the product needs far fewer
dispatches than forward substitution (the commit message counts 17 against 64). An
optimisation replaced forward substitution with it, every test passed, and a real
27-billion-parameter checkpoint began
producing repeated garbage on long prompts. Commit `970137320` (September
2026) explains why. When the entries of $\mathbf{A}$ approach $1$ in magnitude (as
they do for repetitive text, where keys are nearly parallel) the powers
$\mathbf{A}^{32}$ reach a largest entry near $10^{17}$ while the true inverse has
entries of size $1$; recovering a result of size $1$ from intermediates of size
$10^{17}$ needs about 24 significant digits, and the format has 7. The comment in
`crates/model/src/gdn.rs` says so, and notes that a 64-bit version of the product
fails the same way a little further along: "the instability is in the
reformulation, not in the precision it is evaluated at". A stand-in for the
situation, with all entries of $\mathbf{A}$ equal to $-a$:

<!-- worked: floating.nilpotent_inverse -->
The inverse of I - A for the 64 x 64 strictly lower triangular matrix A whose entries below the diagonal all equal -a, computed two ways in each format. Largest entry of any power of A used by the product form, largest entry of the true inverse, and the largest error of each computation.

| a | largest entry among the powers of A | largest entry of the inverse | 32-bit, product of powers | 32-bit, substitution | 64-bit, product of powers | 64-bit, substitution |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 4.92e+00 | 1 | 2.20e-06 | 4.52e-09 | 3.62e-15 | 7.77e-18 |
| 0.5 | 1.42e+09 | 1 | 8.00e+01 | 0.00e+00 | 1.49e-07 | 0.00e+00 |
| 0.98 | 2.44e+17 | 1 | 1.72e+10 | 2.58e-08 | 6.40e+01 | 4.48e-17 |
<!-- /worked -->

For $a = 0.1$ (standing in for the near-orthogonal keys of random weights) the product of powers is
accurate to about $2 \times 10^{-6}$ in 32 bits. For $a = 0.98$ its largest intermediate is $2.4
\times 10^{17}$ and the result is wrong by $1.7 \times 10^{10}$ in 32 bits and by
$64$ in 64 bits, while forward substitution never forms a power and stays at the
rounding level in both. The problem (inverting $\mathbf{I} - \mathbf{A}$) is
well-conditioned; one algorithm for it is unstable. The test
`gdn_chunk_fwd_matches_host_oracle_with_near_parallel_keys` in
`crates/model/tests/it/gdn_chunk_fwd.rs` pins the correlated case, and knowledge #129 records
the general lesson about testing: random weights sample a benign corner, where the
entries are small and every way of forming the inverse agrees.

**Choosing the step of a gradient check.** `crates/gradcheck/src/lib.rs` uses
central differences with $\epsilon \approx 5 \times 10^{-3}$ for 32-bit models (the
doc comment of `directional_check` says that value "suits fp32"), the value this
chapter derives: $(3u)^{1/3} = 5.6 \times 10^{-3}$. The checks of host reference
models in 64 bits (`check_wan`, for example) use $10^{-5}$, close to
$6.9 \times 10^{-6}$. The doc comment of `elementwise_check` explains why that
function wants a larger $\epsilon$: a step on a single entry changes the loss by only
$\epsilon\,|\partial L/\partial w_i|$, so the numerator of the difference quotient is
small next to the rounding error of the loss, and "fp32 cancellation bites sooner". That
is the round-off term $u|f|/h$ of the derivation. The same cancellation is behind
knowledge #40: a tiny test configuration shrank the inputs of an L2-normalisation to
about $10^{-6}$, so that a sum of squares of such numbers, far below $10^{-6}$, was added to an epsilon of
$10^{-6}$ (the kernel computes `inverseSqrt(sum + eps)`, as in
`crates/kernels/wgsl/l2norm_scale.wgsl`) and the epsilon, not the data, set the
output: an absorption of small by large. The check passed with numeric gradients of exactly
zero, and the harness in `crates/gradcheck/src/lib.rs` now initialises those two
tensors at a scale that survives.

**NaN and sampling.** Chapter 10 told the story of commit `072148692`, in
which a sampler panicked when a logit was NaN because every comparison with
NaN is false. The rows of this chapter's special-values table (`max(nan, 1)`
against `max(1, nan)`, and the unsorted list) are the same fact in miniature.

## Check your understanding

1. **Explain.** In 64-bit arithmetic every addition has relative error at most $1.1 \times 10^{-16}$, yet $0.1 + 0.2 \ne 0.3$. Which numbers were rounded, and why is the difference exactly one unit in the last place?
2. **Predict.** In 32-bit arithmetic, $s = 16777216$ ($=2^{24}$). What are $s + 1$ and $s + 2$? What would a loop that adds $1.0$ to a 32-bit counter one time after another do when it reached $s$?
3. **Find the error.** "To get the variance of data near $10^9$ I use $\mathbb{E}[X^2] - \mathbb{E}[X]^2$ in 64-bit arithmetic. Each operation is accurate to $10^{-16}$, so the answer is accurate to about $10^{-16}$." Where is the mistake, and what does the table show for offset $10^9$?
4. **Explain.** The subtraction $x - 1$ at $x = 1 + 10^{-8}$ is computed exactly by the machine, yet its condition number is $10^8$ and its result can be badly wrong. Reconcile these two facts.
5. **Predict.** A central difference of $e^x$ at $x = 1$ in 64 bits is computed with $h = 10^{-4}$ and with $h = 10^{-10}$. Which is more accurate, and roughly by how much? What step would you choose in 32 bits?
6. **Find the error.** A colleague computes $\mathrm{lse}(z)$ by subtracting the *mean* of $z$ instead of the maximum, arguing that this also centres the exponents. For which input does this still fail?
7. **Explain.** Why does the compensated summation $c = (t - s) - y$ compute something nonzero even though it is algebraically $0$, and what happens if a compiler is permitted to treat addition as associative?

## Exercises

1. **[Calculate]** Take the format with $p = 4$ (significand $1.xxx$), exponents $-3$ to $2$. How many positive normal numbers does it have? What are the gap above $1$, the machine epsilon, the unit round-off and the largest number? Round $5/7$ and $1.0625$ (a tie) to it and give the relative errors in units of $u$.
2. **[Calculate]** Decode by hand the 32-bit patterns `c0a00000` and `3e800000` (sign, exponent field, fraction field) and encode $0.15625$. Check against the rows of `floating.fp32_bit_patterns`.
3. **[Calculate]** Evaluate the worst-case bound $(n-1)u$ and the pairwise bound $\lceil \log_2 n \rceil u$ for $n = 1000$ in 32 bits and in 64 bits. At what $n$ does the left-to-right bound reach $1$ (no digit guaranteed) in 32 bits?
4. **[Calculate]** For $f(x) = 1 - \cos x$ at $x = 10^{-3}$ compute $\kappa_f$ and say whether the problem is well-conditioned. How many digits does the formula $1 - \cos x$ lose in 64 bits, and what stable rewriting (use $\cos x = 1 - 2\sin^2(x/2)$) avoids the loss?
5. **[Calculate]** Compute $\mathrm{lse}(0, 0, 0, 0)$ and $\mathrm{lse}(5, 5)$ by hand and check against the two bounds of the section. Then show that $\mathrm{lse}(z + c) = \mathrm{lse}(z) + c$ for every constant $c$.
6. **[Calculate]** For the central difference of $f(x) = \sin x$ at $x = 1$ in 32 bits, take $|f| = \sin 1$ and $M_3 = |f'''(1)| = \cos 1$ and compute the best step $h^*$ and the best error from the formulas of the section.
7. **[Implement]** Emulate 32-bit arithmetic in your language (round every result to the nearest 32-bit float, using a cast or `struct`). Reproduce `floating.million_tenths` and `floating.sum_methods` for the $0.1$'s (the uniform rows depend on a particular random stream, so reproduce only their pattern: left to right worst, pairwise and Kahan best).
8. **[Implement]** Implement Kahan summation as in the pseudocode and reproduce `floating.kahan_trace`. Then compile or run it with the most aggressive optimisation your language offers and say whether the trace changes.
9. **[Implement]** Implement the naive and the stable evaluations of `floating.one_plus_x` (use your language's `expm1` for the stable one) and of `floating.quadratic`, and reproduce their error columns to one significant digit.
10. **[Implement]** Implement the unshifted and shifted log-sum-exp in 32 and 64 bits and reproduce `floating.lse_table` and `floating.mixture_loglik`.
11. **[Investigate]** Add $1.0$ repeatedly to a 32-bit zero. Where does the sum stop changing, and why? Repeat with $0.1$ in place of $1.0$ and with Kahan summation. Explain each stopping point from the gap of the format.
12. **[Investigate]** Choose a function that is not $e^x$ (for example $\tanh$ or $\log$) and a point, and find the best central-difference step in 32 and 64 bits by scanning $h = 2^{-j}$ as in `floating.numdiff_optimum`. Compare with $(3u|f|/M_3)^{1/3}$ using a numerical estimate of the third derivative, and say when the constant matters.

## Summary

A computer stores a number as a sign, an exponent and a significand of fixed
precision $p$. Representable numbers are spaced by a gap that doubles in each
binade, so the gap relative to the number is constant, between $2^{-p}$ and $2^{1-p}$. The
machine epsilon $\varepsilon_{\mathrm{mach}} = 2^{1-p}$ is the gap at $1$, and the unit
round-off $u = 2^{-p}$ bounds the relative error of rounding to nearest, which
every arithmetic operation commits once. Beyond the ends of the range numbers
overflow to infinity or underflow, gradually through subnormals, to zero, and
operations with no answer produce a NaN that poisons all that it touches.

From that single rule follow the failure modes of the chapter. Summation errors
grow with the number of terms (bound $(n-1)u$) and are cured by pairwise
summation, a wider accumulator or compensated summation. Subtracting two
nearly equal rounded numbers exposes earlier rounding errors as lost leading
digits: catastrophic cancellation, which is the relative condition number
$|xf'(x)/f(x)|$ of the subtraction showing itself, and which stable algorithms avoid by
rewriting. A finite-difference derivative has a best step balancing truncation
against round-off, $\sim u^{1/2}$ forward and $\sim u^{1/3}$ central, which is the step
brain's gradient checker uses. And working with logarithms of products and
log-sum-exp of sums, shifted by the maximum, keeps probabilities away from the
cliffs of the range.

The limits are stated by the conditioning floor: when a problem is ill-conditioned
no algorithm recovers digits the input never had, and the work is in recognising
which kind of difficulty one has. The chapter's results make arithmetic predictable, but
they do not tell us how much arithmetic a computation needs. That is the
question of the next chapter: counting operations and memory, so as to predict how long a
computation will take, and why some algorithms (such as the fast transform of Chapter 14) are
worth their complexity.

## Key terms

- **floating-point number**: a number stored as a sign, an exponent in a fixed range and a significand of fixed precision $p$ bits, $(-1)^s(1 + f/2^{p-1})2^e$; it keeps the same number of significant digits at every scale, so the absolute gap between neighbours doubles with each binade. <!-- concept: floating-point-number -->
- **machine epsilon**: the gap between $1$ and the next larger floating-point number, $2^{1-p}$ ($1.19 \times 10^{-7}$ in 32 bits, $2.22 \times 10^{-16}$ in 64 bits); twice the unit round-off $u = 2^{-p}$, which is the bound on the relative error of one rounding. <!-- concept: machine-epsilon -->
- **rounding error**: the difference between an exact result and the stored number nearest to it; for round to nearest, ties to even, its relative size is at most $u$ for results inside the normal range, and every arithmetic operation commits one. <!-- concept: rounding-error -->
- **catastrophic cancellation**: the loss of leading digits when two nearly equal numbers that carry rounding errors are subtracted; the subtraction is exact but promotes the earlier errors to the leading digits of a much smaller result, losing about $\log_{10}$ of the ratio of the operands to the result. <!-- concept: catastrophic-cancellation -->
- **overflow and underflow**: overflow is an exact result too large for the format, stored as infinity; underflow is a nonzero result too small, stored with fewer and fewer significant digits (subnormal numbers) and finally as zero. <!-- concept: overflow-and-underflow -->
- **log-sum-exp**: the function $\log \sum_i e^{z_i}$, computed stably as $m + \log \sum_i e^{z_i - m}$ with $m = \max_i z_i$ so that no exponential overflows and the sum is at least $1$; it is the logarithm of a sum of probabilities given their logarithms, and the denominator of the log of a softmax. <!-- concept: log-sum-exp -->
- **compensated summation**: a way of adding many numbers in which a second variable carries the rounding error of each addition into the next, so that the error of the total stays near $2u$ times the sum of absolute values however many terms there are (Kahan's algorithm). <!-- concept: compensated-summation -->

## Answers

**Before you begin**

1. Infinitely many: between any two different reals there is another (their average). A computer has only finitely many bit patterns, so it can hold only a finite set of reals; every other number has to be replaced by a stored neighbour.
2. $0.00032 = 3.2 \times 10^{-4}$ and $5400 = 5.4 \times 10^{3}$. $2^{-3} = \tfrac18$ and $2^{10} = 1024$.
3. Ten bits, since $2^{10} = 1024$ patterns: $0$ to $1023$. For $x \ge 1$, $\lfloor \log_2 x \rfloor$ is the position of the leading $1$ bit, i.e. the exponent $e$ with $2^e \le x < 2^{e+1}$.
4. It took logarithms, so that the product became a sum of logarithms ($-921$ for 400 factors of $0.1$) that never leaves the range. It left open how to *add* probabilities, which the product rule does not cover, and it said the log-sum-exp trick would be the answer.
5. $\bigl(f(x+h) - f(x-h)\bigr)/(2h)$. Its truncation error is of order $h^2$, but the two function values are rounded and nearly equal, so their difference is mostly rounding error, and dividing by $2h$ magnifies it as $h$ shrinks: the total error has a minimum at some $h$ and rises again to the left of it.
6. The ratio of its largest to its smallest curvature, which measures how elongated the valley is. A large ratio makes gradient descent with a fixed step slow, because the steep direction limits the step while the flat direction needs many steps.

**Check your understanding**

1. The decimals $0.1$ and $0.2$ are rounded to stored numbers (each slightly above its value; $0.2$ is exactly twice the stored $0.1$), and then their exact sum, $0.3000000000000000166\ldots$, lies exactly halfway between two adjacent stored numbers, so the addition rounds, by the tie rule, to the upper one. The decimal $0.3$ is stored as the lower neighbour. Each rounding is within $u$; the difference is one gap, the smallest possible.
2. $2^{24} + 1$ is a tie between $2^{24}$ and $2^{24} + 2$ (the gap there is $2$), and it rounds to the even one, $2^{24}$. So $s + 1 = s$ and $s + 2 = 16777218$. A loop adding $1.0$ one at a time stops at $16777216$ and then stays there forever.
3. Each operation is accurate in the *relative* sense, but the two terms are both about $10^{18}$, so their absolute error is about $10^{18} \times 10^{-16} = 100$, while the difference they produce is about $1$. The relative error of the result is therefore of order $100$. The table for offset $10^9$ shows a computed variance of $896$ against a true $1.09$, a relative error of $818$.
4. The subtraction is exact (Sterbenz) *given* the stored $x$; its result is the difference of the stored operands. But the stored $x$ is already rounded, with a relative error up to $u$, i.e. an absolute error up to $u$ on the number $1.00000001$; the difference is $10^{-8}$, so that error is a relative error $10^8 u$ in the result, which is what $\kappa = 10^8$ says. The subtraction faithfully passes on the earlier error.
5. At $h = 10^{-4}$ the error is about $4.5 \times 10^{-9}$ (truncation). At $h = 10^{-10}$ it is about $9 \times 10^{-7}$ (round-off $u e/h$): the larger step is a hundred times more accurate here. In 32 bits the best step is near $(3u)^{1/3} = 5.6 \times 10^{-3}$, and the best error is of order $10^{-5}$.
6. Whenever the scores are far apart: for $z = (1000, -1000)$ the mean is $0$, the exponents stay at $\pm 1000$, and $e^{1000}$ overflows. Only the maximum guarantees that the largest exponent is $0$ and every other is negative.
7. In floating point the subtraction $t - s$ recovers the part of $y$ that was actually added, exactly (when $|s| \ge |y|$), and subtracting $y$ then gives minus the part that was *not*, the rounding error of the addition. If the compiler may treat addition as associative it may simplify $(t - s) - y$ to $0$, so $c$ is always $0$ and the algorithm turns into the plain left-to-right sum.
