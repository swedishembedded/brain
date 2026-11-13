<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Bits and Number Formats

*How many bits does a number need, and what do we lose with each bit we remove?*

A trained network is a long list of numbers, and the list has to live
somewhere: in memory, on a disk, in the registers of a chip. Every number is a
pattern of bits, and how many bits we give it decides three things at once:
how much memory the model takes, how many bytes must travel to the arithmetic
units each time it is used, and how closely the stored number matches the
number training produced.

Take one weight, $w = 0.0731$, and store it six ways. The first five are
floating-point formats, with 32, 16, 16, 8 and 8 bits; the last is an 8-bit
integer that stands for a multiple of a shared scale. For the integer, the
weight belongs to a group of eight weights whose largest magnitude is $0.9$:
the group is $0.0731$, $-0.412$, $0.0098$, $0.9$, $-0.236$, $0.0515$, $0.1604$
and $-0.0302$.

<!-- worked: number_formats.one_weight -->
The weight is w = 0.0731 (first entry of the group [0.0731, -0.412, 0.0098, 0.9, -0.236, 0.0515, 0.1604, -0.0302]); fp32 is the binary32 rounding of w, the others round w once, ties to even. The int8 row uses the scale Delta = max|group| / 127 = 0.00708661 and the code q = round(w / Delta) = 10, decoded as Delta q. Bit strings are sign, exponent, fraction (int8: the two's complement pattern).

| format | bits | hex | bit pattern | decoded value | error | relative error |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 32 | 0x3D95B574 | 0 01111011 00101011011010101110100 | 0.073100001 | 6.20e-10 | 8.48e-09 |
| fp16 | 16 | 0x2CAE | 0 01011 0010101110 | 0.073120117 | 2.01e-05 | 2.75e-04 |
| bf16 | 16 | 0x3D96 | 0 01111011 0010110 | 0.073242188 | 1.42e-04 | 0.00195 |
| E4M3 | 8 | 0x19 | 0 0011 001 | 0.0703125 | -0.00279 | 0.0381 |
| E5M2 | 8 | 0x2D | 0 01011 01 | 0.078125 | 0.00503 | 0.0687 |
| int8 with scale | 8 | 0x0A | 00001010 | 0.070866142 | -0.00223 | 0.0306 |
<!-- /worked -->

The 32-bit pattern gives back $0.073100001$: the
error is in the ninth digit. Two 16-bit formats with the same number of bits
land on different numbers, $0.07312$ and $0.07324$, because they divide their
16 bits differently. The two 8-bit floating-point formats are off by 4% and
7%, and the 8-bit integer by 3%. Nothing here is a bug. Each format can hold
only so many different numbers, and a stored value is the nearest one it has.

Why is that acceptable? A model with seven billion parameters is $28 \times
10^9$ bytes in 32 bits and $7 \times 10^9$ bytes in 8, and a computer that
generates text reads every weight once for every word it produces, so a
quarter of the bytes is a quarter of the traffic. But a format that is too
coarse corrupts the result, and a format with the wrong *shape* fails in a
different way: some lose the small numbers, some the large ones, some the
precision of the middle.

This chapter opens Volume IV, which is about how machines carry out the
arithmetic of Volumes I to III. Its first job is to say what a machine
actually holds. We start with whole numbers in binary (including negatives,
which need a trick), then read floating point as a layout of bits and derive
the 16-bit and 8-bit formats from the same rule, then look at integers that
stand for scaled numbers, then at converting between all of them, and finally
count the bytes. Volume I, Chapter 15 derived how 32- and 64-bit floating
point rounds and where it fails; this chapter builds on that and does not
repeat its derivations.

The chapter leans on Volume I's account of floating point and on one number from the optimizer chapter of Volume III.

1. (Volume I, Chapter 1, integers.) How many different patterns can 8 bits
   take? What range of whole numbers do they cover if all are non-negative, and
   if half are used for negatives?
2. (Volume I, Chapter 15, floating-point numbers.) A normal floating-point
   number is $(-1)^s (1 + f/2^{p-1}) 2^{e}$. Name its three stored parts. Why does
   the gap between neighbouring numbers double from one binade to the next?
3. (Volume I, Chapter 15, machine epsilon.) What is the machine epsilon of the 32-bit
   format, and how does it relate to the unit round-off $u$ and the significand
   precision $p$?
4. (Volume I, Chapter 15, rounding and the edges of the range.) What happens to
   a result larger than the largest finite number, and what to one smaller than
   the smallest normal number? Which neighbour does a value exactly halfway
   between two numbers go to?
5. (Volume III, Chapter 7, optimizer state.) How many bytes of training memory
   per parameter does AdamW need when everything is stored in 32 bits, and how
   many gigabytes is that for a model with $7 \times 10^9$ parameters?

## Whole numbers in bits

### Unsigned numbers: place values

Every other format in the chapter is built from bit patterns, so we first fix
what a pattern of bits means as a whole number. A **bit** is a digit that is 0 or 1. A **binary representation** writes a
non-negative whole number as bits, the way decimal writes it as digits: each
position has a place value, a power of the base, and the number is the sum of
the place values of the positions that hold a 1. For bits $b_{n-1}\dots b_1
b_0$,

$$
x = \sum_{i=0}^{n-1} b_i\, 2^i .
$$

To find the bits of a number, divide by 2 repeatedly. Each division removes
one place value, and its remainder is the lowest bit of what was left:

<!-- worked: number_formats.unsigned_binary -->
Repeated division of 37 by 2: the remainders, read from last to first, are the bits.

| k | k // 2 | k mod 2 (next bit, lowest first) |
|---|---:|---:|
| 37 | 18 | 1 |
| 18 | 9 | 0 |
| 9 | 4 | 1 |
| 4 | 2 | 0 |
| 2 | 1 | 0 |
| 1 | 0 | 1 |

Reading the remainders from the bottom up gives 100101; check: 1 + 4 + 32 = 37.
<!-- /worked -->

With $n$ bits there are $2^n$ patterns, and as unsigned numbers they are
exactly $0, 1, \dots, 2^n - 1$ (Volume I, Chapter 1 tabulated the ranges for
8, 16, 32 and 64 bits). A computer adds two such patterns the way you add by
hand, column by column with a carry, but it has only $n$ columns. The carry out
of the top column has nowhere to go and is dropped, which is the same as
taking the result modulo $2^n$:

<!-- worked: number_formats.unsigned_wrap -->
| operation on 8-bit unsigned | exact result | stored (result mod 256) | stored pattern |
|---|---:|---:|---:|
| 200 + 55 | 255 | 255 | 11111111 |
| 200 + 56 | 256 | 0 | 00000000 |
| 250 + 10 | 260 | 4 | 00000100 |
| 255 + 1 | 256 | 0 | 00000000 |
| 5 - 6 | -1 | 255 | 11111111 |
| 16 * 16 | 256 | 0 | 00000000 |
<!-- /worked -->

An unsigned 8-bit counter at 255 goes to 0 when one is added, and $5 - 6$ is
255. Nothing signals the wrap; it is what the arithmetic is. The bit
operations of `pseudocode.md` (`and`, `or`, `xor`, `not`, `>>`, `<<`) work on the
same patterns, in widths of 8, 16, 32 and 64 bits, and this chapter uses them
throughout.

### Negative numbers: two's complement

A network's weights and gradients are negative half the time, so a format needs
negatives that the same adder handles without special cases. Suppose a pattern $N(x)$ is to stand for $-x$. The test of a good
representation of $-x$ is that adding it to $x$ gives zero, so that subtraction
is just addition. Because addition on $n$ bits is addition modulo $2^n$, the
requirement is

$$
x + N(x) \equiv 0 \pmod{2^n}, \qquad\text{so}\qquad N(x) = 2^n - x .
$$

That is all of the design: the pattern of $-x$ is $2^n - x$. Reading it back,
a pattern $a$ in the upper half, $a \ge 2^{n-1}$, is taken to be the negative
number $a - 2^n$ (the pseudocode's `signed(a, n)`), and a pattern in the lower
half is itself. This is **two's complement**. The top bit is then a sign
bit, because it is 1 exactly for the negative numbers, and it carries the place
value $-2^{n-1}$ instead of $+2^{n-1}$:
$a = -b_{n-1}2^{n-1} + \sum_{i<n-1} b_i 2^i$.

How do we compute $2^n - x$ without a subtraction? Write $2^n - x = (2^n - 1 -
x) + 1$. The number $2^n - 1$ is the pattern of all ones, and subtracting $x$
from all ones never needs a borrow, because every column is $1 - b_i$ with
$b_i \in \{0, 1\}$: it flips each bit. So negation is **invert, then add one**:

<!-- worked: number_formats.twos_complement -->
For each magnitude m: its 8-bit pattern, that pattern inverted (every bit flipped), the inverted pattern plus one (mod 256), the pattern of -m computed directly as (-m) mod 256, the signed reading of the inverted-plus-one pattern, and the sum of the patterns of m and of inverted-plus-one, mod 256.

| m | pattern of m | inverted | inverted + 1 | pattern of -m = (-m) mod 256 | read as signed | sum mod 256 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 00000000 | 11111111 | 00000000 | 00000000 | 0 | 0 |
| 1 | 00000001 | 11111110 | 11111111 | 11111111 | -1 | 0 |
| 5 | 00000101 | 11111010 | 11111011 | 11111011 | -5 | 0 |
| 64 | 01000000 | 10111111 | 11000000 | 11000000 | -64 | 0 |
| 127 | 01111111 | 10000000 | 10000001 | 10000001 | -127 | 0 |
| 128 | 10000000 | 01111111 | 10000000 | 10000000 | -128 | 0 |
<!-- /worked -->

The last two columns are the proof by example: inverted-plus-one, read as a
signed number, is the negative of $m$, and the two patterns add to $0
\bmod 256$. The table also shows the one oddity of the format. The range is
$-2^{n-1}$ to $2^{n-1} - 1$, asymmetric, because zero takes one of the
non-negative patterns. The magnitude $128$ is the pattern `10000000`, which
reads as $-128$, so $-(-128) = -128$ in 8 bits. A representation with a
separate sign bit and a magnitude (sign-magnitude) would be symmetric, but it
would have two zeros and would need a different adder for subtraction;
two's complement has neither problem.

Overflow is the same wrap as before, now visible through the sign:

<!-- worked: number_formats.twos_complement_arithmetic -->
| int8 addition | exact | pattern a | pattern b | pattern of the sum (low 8 bits) | stored value | result |
|---|---:|---:|---:|---:|---:|---:|
| 100 + 27 | 127 | 01100100 | 00011011 | 01111111 | 127 | exact |
| 100 + 28 | 128 | 01100100 | 00011100 | 10000000 | -128 | wrapped |
| -100 + -28 | -128 | 10011100 | 11100100 | 10000000 | -128 | exact |
| -100 + -29 | -129 | 10011100 | 11100011 | 01111111 | 127 | wrapped |
| -5 + 3 | -2 | 11111011 | 00000011 | 11111110 | -2 | exact |
| 64 + 64 | 128 | 01000000 | 01000000 | 10000000 | -128 | wrapped |

Widening to 16 bits copies the sign bit into the new high bits:

| x | 8-bit pattern | 16-bit pattern |
|---|---:|---:|
| 5 | 00000101 | 0000000000000101 |
| -5 | 11111011 | 1111111111111011 |
| -128 | 10000000 | 1111111110000000 |
<!-- /worked -->

$100 + 28$ should be $128$, but the pattern `10000000` reads as $-128$: the sum of two positive numbers came out negative, which is the signature of overflow. Detecting that the result is
wrong is the program's job, as it is for the 32-bit parameter count that wrapped
in Volume I, Chapter 1. The lower table shows how a narrow integer is widened:
**sign extension** copies the sign bit into the new high bits, and the value is
unchanged. It matters in this chapter because sums of 8-bit products are
accumulated in 32-bit integers.

## Floating point as a layout of bits

A floating-point format is a rule for turning a pattern of bits into a number.
If the rule is written down once, every format of the chapter, from 32 bits to
8, is the same rule with different field widths. Volume I, Chapter 15 wrote a normal floating-point number as $(-1)^s (1 +
f/2^{p-1})\,2^{e}$. A machine stores three fields next to each other, from the top bit
down: one **sign** bit $s$, an **exponent field** $E$ of $n_e$ bits and a
**fraction field** $f$ of $n_m$ bits. The significand has $p = n_m + 1$
bits, because the leading 1 is implied and not stored. The picture shows the layouts
of the formats of this chapter, one box per bit.

![The bit layouts of fp32 (1 sign, 8 exponent, 23 fraction bits), fp16 (1, 5, 10), bf16 (1, 8, 7), E4M3 (1, 4, 3), E5M2 (1, 5, 2) and an int8 two's complement code (1 sign bit and 7 further bits), drawn as rows of equally wide boxes aligned at the top bit, with sign boxes yellow, exponent boxes blue and fraction boxes grey.](img/bit-layouts.svg)

The exponent field is an unsigned integer, so to represent negative exponents it is stored with an
offset, the **bias** $2^{n_e-1} - 1$: the stored field is $E = e + \mathrm{bias}$. The rule for
the value of the pattern $(s, E, f)$ has three cases, where $E$ is compared with the all-ones
field $2^{n_e} - 1$:

- **Normal numbers**, $0 < E < 2^{n_e} - 1$: $\;(-1)^s\,(1 + f/2^{n_m})\,2^{E - \mathrm{bias}}$.
- **Zero and subnormal numbers**, $E = 0$: $\;(-1)^s\,(f/2^{n_m})\,2^{1 - \mathrm{bias}}$. There
  is no implied 1, and the exponent is that of the smallest normal number,
  $1 - \mathrm{bias}$ and not $-\mathrm{bias}$. The choice is what makes the format continuous: the largest subnormal
  $(2^{n_m}-1)/2^{n_m} \cdot 2^{1-\mathrm{bias}}$ and the smallest normal number $2^{1-\mathrm{bias}}$ differ by one gap $2^{-n_m} \cdot 2^{1-\mathrm{bias}}$,
  the same gap as between any two neighbours of the lowest normal binade. This is the gradual
  underflow of Volume I, Chapter 15.
- **Infinity and NaN**, $E = 2^{n_e} - 1$: infinity when $f = 0$ and not-a-number otherwise (the convention of the
  IEEE standard for 32-bit and 16-bit floating point). One format below changes this rule.

Written once as an algorithm, with `finite_only` marking the one deviation:

```text
function decode_float(code: u32, ne: int, nm: int, finite_only: bool) -> real:
    require 1 <= ne and 1 <= nm and 1 + ne + nm <= 32, "unsupported split"
    require code < 2^(1 + ne + nm), "code wider than the format"
    bias = 2^(ne - 1) - 1
    s = code >> (ne + nm)
    E = (code >> nm) and (2^ne - 1)
    f = code and (2^nm - 1)
    if E == 2^ne - 1 and (finite_only == false or f == 2^nm - 1):
        if finite_only or f != 0:
            return nan
        return -infinity if s == 1 else infinity
    if E == 0:
        v = (f / 2^nm) * 2^(1 - bias)
    else:
        v = (1 + f / 2^nm) * 2^(E - bias)
    return -v if s == 1 else v
```

Everything the next table lists follows from this function and the split $(n_e, n_m)$:
the largest finite number (the largest normal exponent with all fraction bits set, $(2 - 2^{-n_m})\,2^{\mathrm{bias}}$ for an
IEEE-style format), the smallest normal number $2^{1 - \mathrm{bias}}$, the smallest subnormal $2^{1 - \mathrm{bias} - n_m}$, the machine epsilon
$\varepsilon = 2^{-n_m}$ (the gap between 1 and the next number, $2^{1-p}$ of Volume I, Chapter 15), the number of decimal digits
$p \log_{10} 2$, and the **decades** of range, $\log_{10}$ of the largest over the smallest positive number.

<!-- worked: number_formats.float_formats -->
Everything below is computed from the bit split n_e / n_m by the decoding rule of the chapter (bias = 2^(n_e-1) - 1, eps = 2^-n_m, digits = (n_m + 1) log10 2, decades = log10(max / smallest subnormal)). fp32, fp16, bf16 and E5M2 use the IEEE-style rule (the all-ones exponent field is infinity and NaN); E4M3 sets the decoder's finite_only flag: its all-ones exponent field 1111 is an ordinary exponent, only the pattern with fraction 111 as well is NaN, and there is no infinity, so its largest finite number is (1 + 6/8) * 2^8 = 448 (the IEEE-style rule would stop at (1 + 7/8) * 2^7 = 240). Decoding was checked against the machine's own binary16 for all 65536 codes, and encoding against its binary32 and binary16 on 2000 random values.

| format | bits | exponent / fraction bits | bias | largest finite | smallest normal | smallest subnormal | eps = 2^-n_m | decimal digits | decades of range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 32 | 8 / 23 | 127 | 3.403e+38 | 1.18e-38 | 1.40e-45 | 1.19e-07 | 7.22 | 83.4 |
| fp16 | 16 | 5 / 10 | 15 | 65504 | 6.10e-05 | 5.96e-08 | 9.77e-04 | 3.31 | 12.0 |
| bf16 | 16 | 8 / 7 | 127 | 3.390e+38 | 1.18e-38 | 9.18e-41 | 0.00781 | 2.41 | 78.6 |
| E4M3 | 8 | 4 / 3 | 7 | 448 | 0.0156 | 0.00195 | 0.125 | 1.20 | 5.4 |
| E5M2 | 8 | 5 / 2 | 15 | 57344 | 6.10e-05 | 1.53e-05 | 0.25 | 0.90 | 9.6 |
<!-- /worked -->

The row for fp32 repeats Volume I, Chapter 15: the **fp32 format** is 1 sign bit, 8 exponent bits with bias $127$ and 23 fraction bits,
about 7 digits and 83 decades. Its rounding theory (round to nearest, ties to even, error at
most $u = \varepsilon/2$, subnormals, overflow) is all that chapter, and every format
below obeys it with its own $n_m$. What changes down the table is how the bits are split.
Each bit taken from the fraction doubles the gap between neighbours; each bit taken from the exponent
*squares* the range, because a field of $n_e$ bits spans $2^{n_e}$ binades. The 16-bit formats show the two
choices side by side.

## Sixteen bits, split two ways: fp16 and bf16

Both formats have 16 bits and one sign bit. The **fp16 format** (the IEEE half
precision) spends 5 bits on the exponent and 10 on the fraction: a precision
of 11 bits, $\varepsilon = 2^{-10} \approx 10^{-3}$, but only 5 exponent bits,
which give the range $6 \times 10^{-8}$ to $65504$. The **bf16 format** spends 8 bits on the exponent and 7 on the fraction: precision 8 bits,
$\varepsilon = 2^{-7} \approx 0.008$, and the exponent field *of fp32*, so the same bias, the same
range of about $10^{-38}$ to $3 \times 10^{38}$ and 79 decades. bf16 is fp32 with 16 of the
23 fraction bits removed.

Put the same numbers into both and see what each does with them:

<!-- worked: number_formats.range_vs_precision -->
Each x is first rounded to fp32, then once to the narrower format, ties to even. 'rel' is |stored - fp32 value| / |fp32 value|.

| x | fp32 | fp16 | rel | bf16 | rel |
|---|---:|---:|---:|---:|---:|
| 1/3 | 0.33333334 | 0.33325195 | 2.4e-04 | 0.33398438 | 0.002 |
| 1000.1 | 1000.1 | 1000 | 1.0e-04 | 1000 | 1.0e-04 |
| 70000 | 70000 | inf | - | 70144 | 0.0021 |
| 3e-5 | 2.9999999e-05 | 2.9981136e-05 | 6.3e-04 | 3.0040741e-05 | 0.0014 |
| 1e-8 | 9.9999999e-09 | 0 | - | 1.0011718e-08 | 0.0012 |
| 1e-40 | 9.9999461e-41 | 0 | - | 9.1835496e-41 | 0.082 |
<!-- /worked -->

Row by row. $1/3$ and $1000.1$ test precision: fp16 is about eight times closer to $1/3$
than bf16 ($2.4 \times 10^{-4}$ against $0.002$), as the two epsilons promise. $70000$ is
above fp16's largest finite number $65504$, so it becomes infinity; bf16 holds it
with the same error as everything else. $3 \times 10^{-5}$ is below fp16's smallest normal
number $6.1 \times 10^{-5}$: it is stored as a subnormal, with fewer significant bits, and its
error is a little larger. $10^{-8}$ is below half of fp16's smallest subnormal ($2.98 \times 10^{-8}$) and is
stored as zero. bf16 keeps it, and even $10^{-40}$, below its smallest normal number, with
subnormal precision. Neither format is better; each is a different trade of precision
for range, and the trade has consequences for training.

### Why training uses bf16

The gradients of a deep network are spread over many orders of magnitude: some entries are of order 1, others $10^{-9}$ (Volume III, Chapter 12 showed how gradients vanish or explode with depth). A narrow format must hold all of them. We want to know what happens to a spread of gradient magnitudes when they are stored in fp16.
As a stand-in we draw gradient magnitudes log-uniformly over ten decades. (A synthetic spread is a
stand-in for the real thing, whose shape depends on the network; the arithmetic of the table does not.)

fp16 can recover some of the lost gradients with loss scaling. The
backward pass is linear in the loss: if the loss is multiplied by a constant $S$, every gradient
is multiplied by $S$ (chain rule, Volume III, Chapter 4). Run the backward pass on $S L$, so that
all gradients are shifted up by a factor $S$ into the range fp16 can hold, then divide the gradients by $S$
in a wider format before the update. The cost is that $S$ must be large enough to lift the small
gradients out of the subnormal range and small enough not to overflow the large ones, and the right $S$ changes during training.

<!-- worked: number_formats.loss_scaling -->
10000 stand-in gradient magnitudes g_i = 10^(-12 + 10 u_i), u_i = uniform() drawn in order i = 0, 1, ... from the generator seeded 13609293853419212352 (seed_of 'number_formats.loss_scaling'); the scaled values S g_i are rounded to the format once, ties to even. 'zero' is the share stored as 0, 'subnormal' the share stored with fewer than the full significand bits, 'inf' the share that overflows, 'kept' the share stored as a nonzero finite number; 'mean rel. error' is the mean of |stored - S g| / (S g) over the kept values. Shares are printed with two decimals (each is a count out of 10000, so two decimals are exact).

| format | scale S | zero | subnormal | inf | kept | mean rel. error of kept |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 1 | 44.26% | 33.07% | 0.00% | 55.74% | 0.0424 |
| fp16 | 2^8 | 20.15% | 32.47% | 0.00% | 79.85% | 0.0303 |
| fp16 | 2^12 | 8.32% | 32.95% | 0.00% | 91.68% | 0.0256 |
| fp16 | 2^16 | 0.00% | 29.13% | 0.00% | 100.00% | 0.00997 |
| fp16 | 2^20 | 0.00% | 17.17% | 0.00% | 100.00% | 7.22e-04 |
| fp16 | 2^24 | 0.00% | 5.53% | 4.20% | 95.80% | 1.92e-04 |
| bf16 | 1 | 0.00% | 0.00% | 0.00% | 100.00% | 0.00141 |
<!-- /worked -->

Without scaling, almost half of the stand-in gradients are stored as zero in fp16, and another third are
subnormal, kept with fewer significant bits (the "kept" column counts them, and their error is why the mean
error is large). Raising $S$ rescues them: at $2^{16}$ nothing is lost to zero, and at $2^{24}$ the largest
gradients start to overflow to infinity (4.20%). There is a window of workable scales, and a training
run has to stay inside it. With bf16 the whole distribution is stored as it is, with no scale: the range is
that of fp32, so there is no window to stay in. The price is precision: a mean error of 0.0014, about twice fp16's 0.0007 at $S = 2^{20}$, where nothing is lost. That is why bf16 training keeps its sums in 32 bits (see the sections below).
This is the reason bf16 is the usual training format for large models and fp16 survives where the
numbers are known to stay inside its range, as in serving. The mechanics of mixed-precision training (which tensors use which format, how
the scale is adjusted) belong to the later volume on training large models (Volume XIII). Nothing here depends on it: all
that is needed is that a master copy is a 32-bit copy of the weights to which the updates are added, with the narrow copy rebuilt from it for the arithmetic.

## Eight bits: E4M3 and E5M2

With 8 bits the same two choices must be made with almost nothing to spend, and the usual answer is
two formats. They are named by their split: **E4M3** has 4 exponent and 3 fraction bits; **E5M2**
has 5 exponent and 2 fraction bits. Together they are the **fp8 formats**. Both follow the decoding rule above. E5M2
is exactly fp16 with the low 8 bits cut off: same sign, same 5-bit exponent field, same bias 15, 2 fraction bits
instead of 10. E4M3 trades one exponent bit for one fraction bit, giving more precision and less range.

E4M3 has one more difference, and it is a decision about what to do with a nearly useless pattern. With 4 exponent bits the all-ones field has
only 16 patterns (8 for each sign, $f = 0 \dots 7$). In the IEEE convention they would hold two infinities and 14 NaNs, almost a whole binade
spent on values that are never wanted. The E4M3 of the published definition of the format (the 8-bit formats of
Micikevicius et al., 2022, and the Open Compute Project's specification) gives up infinity and keeps *one* NaN pattern, the one with exponent and
fraction all ones. Every other pattern with the top exponent field is an ordinary number:

<!-- worked: number_formats.special_codes -->
| format | hex | sign exponent fraction | value | kind |
|---|---|---|---:|---|
| E4M3 | 0x00 | 0 0000 000 | 0 | zero |
| E4M3 | 0x01 | 0 0000 001 | 0.00195312 | subnormal |
| E4M3 | 0x07 | 0 0000 111 | 0.0136719 | subnormal |
| E4M3 | 0x08 | 0 0001 000 | 0.015625 | normal |
| E4M3 | 0x38 | 0 0111 000 | 1 | normal |
| E4M3 | 0x7E | 0 1111 110 | 448 | normal |
| E4M3 | 0x7F | 0 1111 111 | nan | NaN |
| E4M3 | 0xFF | 1 1111 111 | nan | NaN |
| E4M3 | 0x80 | 1 0000 000 | 0 | zero |
| E5M2 | 0x00 | 0 00000 00 | 0 | zero |
| E5M2 | 0x01 | 0 00000 01 | 1.52588e-05 | subnormal |
| E5M2 | 0x03 | 0 00000 11 | 4.57764e-05 | subnormal |
| E5M2 | 0x04 | 0 00001 00 | 6.10352e-05 | normal |
| E5M2 | 0x3C | 0 01111 00 | 1 | normal |
| E5M2 | 0x7B | 0 11110 11 | 57344 | normal |
| E5M2 | 0x7C | 0 11111 00 | inf | infinity |
| E5M2 | 0x7D | 0 11111 01 | nan | NaN |
| E5M2 | 0x80 | 1 00000 00 | 0 | zero |
| fp16 | 0x0001 | 0 00000 0000000001 | 5.96046e-08 | subnormal |
| fp16 | 0x03FF | 0 00000 1111111111 | 6.09756e-05 | subnormal |
| fp16 | 0x0400 | 0 00001 0000000000 | 6.10352e-05 | normal |
| fp16 | 0x3C00 | 0 01111 0000000000 | 1 | normal |
| fp16 | 0x7BFF | 0 11110 1111111111 | 65504 | normal |
| fp16 | 0x7C00 | 0 11111 0000000000 | inf | infinity |
| fp16 | 0x7E00 | 0 11111 1000000000 | nan | NaN |
| fp16 | 0xFC00 | 1 11111 0000000000 | -inf | infinity |

The ordinary normal-number rule applied to E4M3's reserved pattern (exponent 1111, fraction 111) would give (1 + 7/8) * 2^8 = 480; the format spends that one pattern on NaN, so its largest finite value is 448.
<!-- /worked -->

Read the E4M3 rows. The largest finite number is `0x7E`, exponent field 1111 and fraction 110: $(1 + 6/8) \cdot 2^{15-7} = 448$.
With the IEEE rule the top field would be reserved and the largest number would be $(1 + 7/8)\,2^{14-7} = 240$; the convention nearly doubles the
range (to 448) at the cost of infinity. `0x7F` (and `0xFF`) is NaN, and the line under the table shows why this needs saying: the
ordinary rule would give that pattern the value $480$, so any decoder that treats it as a number
is wrong in a way no value check will notice. An E4M3 conversion that overflows therefore has no infinity to return. It returns NaN, or, in
a **saturating conversion**, the largest finite number $\pm 448$. Which one a program wants is a choice, and every
conversion below states it. E5M2 keeps the IEEE convention: `0x7C` is infinity, `0x7D` to `0x7F` are NaNs, the
largest finite number is `0x7B`, $1.75 \cdot 2^{15} = 57344$.

The 8-bit formats have *few* numbers: 126 positive finite values in E4M3, 123 in E5M2. Their subnormals are a
visible share of them:

<!-- worked: number_formats.subnormals -->
| format | smallest normal | subnormal spacing = smallest subnormal | positive subnormals 2^n_m - 1 | positive finite values | share that are subnormal | values below this round to 0 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 1.18e-38 | 1.40e-45 | 8,388,607 | 2,139,095,039 | 0.3922% | 7.01e-46 |
| fp16 | 6.10e-05 | 5.96e-08 | 1,023 | 31,743 | 3.2228% | 2.98e-08 |
| bf16 | 1.18e-38 | 9.18e-41 | 127 | 32,639 | 0.3891% | 4.59e-41 |
| E4M3 | 0.0156 | 0.00195 | 7 | 126 | 5.5556% | 9.77e-04 |
| E5M2 | 6.10e-05 | 1.53e-05 | 3 | 123 | 2.4390% | 7.63e-06 |
<!-- /worked -->

E4M3 has seven positive subnormal values, evenly spaced by $2^{-9} = 0.00195$ below $2^{-6}$: 5.6% of its positive numbers.
Everything smaller than half the smallest subnormal ($2^{-10}$) rounds to zero. In E4M3 the whole usable range is five decades (448 over $2^{-9}$ is $2.3 \times 10^5$),
so a tensor whose values are mostly small but include a large one cannot be stored in it as it is.

### A scale makes a narrow range usable

A format with five decades of range can still hold a tensor whose values span many more, if the tensor is first divided by a
number that moves its largest value to the top of the format. Take a block of eight values, whose largest magnitude is 3000.
Dividing by $s = 3000/448$ puts it at 448; the block is stored as the E4M3 codes of $x/s$ together with $s$, and a value is read back as $s$ times its decoded code.

<!-- worked: number_formats.fp8_block_scale -->
One block of 8 values, amax = 3000. The scale is s = amax / 448 = 6.696429; a value is stored as the E4M3 code of x / s and read back as s times the decoded code. 'E4M3 raw' stores x directly, saturating at 448. 'E5M2 raw' stores x directly in E5M2 (largest finite 57344). Rounding: nearest, ties to even.

| x | E4M3 raw | rel. error | code of x / s | E4M3 with scale | rel. error | E5M2 raw | rel. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 448 | 0.85 | 0x7E | 3000 | 0 | 3072 | 0.024 |
| -1200 | -448 | 0.63 | 0xF3 | -1178.6 | 0.018 | -1280 | 0.067 |
| 87.5 | 88 | 0.0057 | 0x55 | 87.054 | 0.0051 | 80 | 0.086 |
| 0.9 | 0.875 | 0.028 | 0x21 | 0.94169 | 0.046 | 0.875 | 0.028 |
| 12 | 12 | 0 | 0x3E | 11.719 | 0.023 | 12 | 0 |
| -0.04 | -0.039062 | 0.023 | 0x83 | -0.039237 | 0.019 | -0.039062 | 0.023 |
| 450 | 448 | 0.0044 | 0x68 | 428.57 | 0.048 | 448 | 0.0044 |
| -2300 | -448 | 0.81 | 0xFB | -2357.1 | 0.025 | -2048 | 0.11 |
<!-- /worked -->

Stored directly, the values 3000, -1200 and -2300 are clamped to $\pm 448$: errors of 63% to 85%, and unlike rounding error that is a loss of the number itself. With the scale every error is a few percent, at the price of one extra number per block that must be stored in
a wider format. E5M2's range already holds 3000, but it has half of E4M3's precision (one fraction bit fewer, so a relative error twice as large). The 8-bit floating-point formats, then, are always used with a scale, and one scale per block of values is the cost that
the memory table below will include.

## The density of numbers

The tables say the same thing in numbers; two pictures show the shape. The first counts the representable positive
numbers in every power of ten. A floating-point format has $2^{n_m}$ numbers in every binade, so the count per decade
is the same $2^{n_m} \log_2 10$ across the whole normal range: the plateau height is the format's precision and the
plateau length its range.

![For bf16, fp16, E5M2 and E4M3, the base-10 logarithm of how many representable positive numbers lie in each decade, plotted against the decade. bf16 is a flat line near 2.6 from -38 to 38, fp16 a plateau near 3.5 from -4 to 4 with a steep ramp below it, E5M2 and E4M3 low plateaus near 1.1 and 1.4.](img/format-density.svg)

bf16's plateau is long and low, fp16's short and high: the same 16 bits, one
spending them on reach and the other on resolution. Each curve falls at its left end, in the subnormal range, where the
spacing stops shrinking and the number of values per decade falls with the decade.

The second picture plots the largest relative error of rounding a number, against the number. For each pair of
neighbouring representable numbers the error is largest at their midpoint, where rounding errs by half the gap.

![For bf16, fp16, E5M2 and E4M3, the base-10 logarithm of the largest relative rounding error in each binade, plotted against the base-10 logarithm of the number. Each format is a flat line (bf16 near -2.4, fp16 near -3.3, E5M2 near -1, E4M3 near -1.2) over its normal range, and rises steeply to 0 in its subnormal range, where relative error reaches 100%.](img/rounding-error.svg)

Inside the normal range the relative error is constant: $u = 2^{-(n_m+1)}$, 0.39% for bf16 and 0.049% for fp16. The
absolute error grows with the number, so that every number keeps the same count of correct digits. That is the
property that makes floating point the format of choice for weights and activations, whose scale varies by orders of
magnitude from tensor to tensor. At the left the line rises to 100%: the smallest subnormal's gap is as large as
the number, and anything below half of it is stored as zero.

## Integers as a number format: a code and a scale

The cheapest format of all stores a small integer for each number and one shared number that says what the integer stands for. Return to the last row of the first table: a weight stored as the 8-bit pattern `00001010`, the integer 10, with a
group scale of $0.00708661$. The stored integer is a **code** $q$ and the number it stands for is
$\Delta q$, where $\Delta$ is a scale shared by a group of values. An **integer-quantized format** is a
storage format of this kind: small integer codes, such as 8-bit or 4-bit two's complement numbers, plus the scale (and sometimes an offset, the zero point, below)
needed to read them as numbers. How to choose scales well, group sizes, and how to find the numbers worth
protecting is the subject of Chapter 12 of this volume, on quantization; this section
describes the format plainly so that the byte counts and kernels of the chapters between can be read.

The simplest choice of scale maps the largest magnitude of the group to the largest code, $\Delta = \max_i |x_i| / q_{\max}$, with
$q_{\max} = 127$ for 8 bits, and the code of a value is the rounded quotient:

```text
function quantize_symmetric(x: array<real>[n], qmax: int) -> (delta: real, q: array<int>[n]):
    require n > 0 and qmax >= 1, "empty group or no codes"
    amax = max([abs(x[i]) for i in 0 .. n])
    require amax > 0, "an all-zero group has no scale"
    delta = amax / qmax
    q = zeros(int, [n])
    for i in 0 .. n:
        q[i] = clamp(round_even(x[i] / delta), -qmax, qmax)
    return (delta, q)
```

Why $127$ and not $128$? The range $-127 \dots 127$ is symmetric around zero, so negating a weight is always a valid code (the asymmetry of
two's complement is not used), zero decodes to exactly zero, and one scale serves both signs. The error of
rounding a code is at most half a step, $\Delta/2$: unlike a float's, an integer format's error is *absolute* and the same for
every value in the group, so a small value is represented with a large relative error. That is why the scale must
belong to a small group. One large weight sets the step for everyone:

<!-- worked: number_formats.int8_group_scale -->
Eight weights [0.12, -0.05, 0.31, -0.22, 2.4, 0.18, -0.27, 0.09]. One scale for all eight: Delta = 2.40 / 127 = 0.018898. Two scales, one per group of four: Delta_A = 0.31 / 127 = 0.002441 and Delta_B = 2.40 / 127 = 0.018898. Code q = round(w / Delta) (ties to even), decoded Delta q.

| w | q (one scale) | decoded | abs. error | q (group scale) | decoded | abs. error |
|---|---:|---:|---:|---:|---:|---:|
| 0.12 | 6 | 0.11339 | 0.0066 | 49 | 0.11961 | 3.9e-04 |
| -0.05 | -3 | -0.056693 | 0.0067 | -20 | -0.048819 | 0.0012 |
| 0.31 | 16 | 0.30236 | 0.0076 | 127 | 0.31 | 0 |
| -0.22 | -12 | -0.22677 | 0.0068 | -90 | -0.21969 | 3.1e-04 |
| 2.4 | 127 | 2.4 | 0 | 127 | 2.4 | 0 |
| 0.18 | 10 | 0.18898 | 0.009 | 10 | 0.18898 | 0.009 |
| -0.27 | -14 | -0.26457 | 0.0054 | -14 | -0.26457 | 0.0054 |
| 0.09 | 5 | 0.094488 | 0.0045 | 5 | 0.094488 | 0.0045 |

Mean absolute error over the four weights of the first group: 0.00693 with one scale, 0.00047 with two. Over all eight: 0.00583 against 0.00260.
<!-- /worked -->

The 2.40 forces a step of 0.0189 on all eight weights, and the four small ones, whose largest is 0.31, are stored with an error near 0.007 each: 13% of the value 0.05 and 6% of 0.12. Splitting the eight into two groups of four, each with its own scale, cuts the mean absolute error over the four small weights from 0.0069 to 0.0005, and the outlier is still exact. The price is one more scale to store. This
trade between the size of the group and the error is the heart of weight quantization, and brain's int8 and int4 formats both
use a group of 32 along the reduction axis.

When the values are not symmetric around zero, a scale alone wastes codes. A value range $[-1, 6]$ stored with the symmetric
rule would never use the codes from $-127$ to $-22$. The remedy is an offset, the zero point $q_0$: the code that stands for the number zero, so that
$x \approx \Delta\,(q - q_0)$. With the range $[x_{\min}, x_{\max}]$ spread over the 256 unsigned codes $0 \dots 255$,
$\Delta = (x_{\max} - x_{\min})/255$ and $q_0 = \text{round\_even}(-x_{\min}/\Delta)$:

<!-- worked: number_formats.int8_zero_point -->
Six values [-1.0, -0.3, 0.0, 1.7, 4.2, 6.0] with range [-1.0, 6.0]. Symmetric: Delta = 6 / 127 = 0.047244, q = round(x / Delta), decoded Delta q. With a zero point: Delta = (6 - (-1)) / 255 = 0.027451, q_0 = round(1 / Delta) = 36, q = clamp(round(x / Delta) + q_0, 0, 255), decoded Delta (q - q_0).

| x | q symmetric | decoded | abs. error | q with zero point | decoded | abs. error |
|---|---:|---:|---:|---:|---:|---:|
| -1.0 | -21 | -0.99213 | 0.0079 | 0 | -0.98824 | 0.012 |
| -0.3 | -6 | -0.28346 | 0.017 | 25 | -0.30196 | 0.002 |
| 0.0 | 0 | 0 | 0 | 36 | 0 | 0 |
| 1.7 | 36 | 1.7008 | 7.9e-04 | 98 | 1.702 | 0.002 |
| 4.2 | 89 | 4.2047 | 0.0047 | 189 | 4.2 | 0 |
| 6.0 | 127 | 6 | 0 | 255 | 6.0118 | 0.012 |
<!-- /worked -->

The step shrinks from 0.0472 to 0.0275, and the largest error from 0.017 to 0.012. The zero point has a cost the symmetric rule does not have: every
product of two such numbers needs correction terms for the offsets.
Weights are close to symmetric and use no zero point; activations after a one-sided function often use one.

### Four bits, two to a byte

Memory is addressed in bytes, so a 4-bit code needs a rule for sharing a byte. The usual one puts the even-indexed
code in the low four bits and the odd-indexed one in the high four, using the bit operations of the pseudocode. The 4-bit range is
$-8 \dots 7$ in two's complement; the symmetric rule uses $\pm 7$ (the scale is $\max|x|/7$), for the same reason as 127.

```text
function pack_int4(q: array<int>[n]) -> array<u8>[n // 2]:
    require n mod 2 == 0, "an odd count does not fill the last byte"
    out = zeros(u8, [n // 2])
    for j in 0 .. n // 2:
        require -8 <= q[2 * j] <= 7 and -8 <= q[2 * j + 1] <= 7, "code outside the 4-bit range"
        lo = pattern(q[2 * j], 4)
        hi = pattern(q[2 * j + 1], 4)
        out[j] = u8(lo or (hi << 4))
    return out

function unpack_int4(b: u8) -> (int, int):
    return (signed(b and 0xF, 4), signed(b >> 4, 4))
```

<!-- worked: number_formats.int4_pack -->
Eight weights [0.5, -0.2, 0.7, 0.1, -0.7, 0.3, 0.0, -0.4]; Delta = max|w| / 7 = 0.1, q = round(w / Delta) in -7 .. 7. Byte b holds the even-index code in its low four bits and the odd-index code in its high four bits: byte = (lo and 0xF) or ((hi and 0xF) << 4).

| byte | even element | q | low nibble | odd element | q | high nibble | byte (binary) | hex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | w[0] = 0.5 | 5 | 0101 | w[1] = -0.2 | -2 | 1110 | 11100101 | 0xE5 |
| 1 | w[2] = 0.7 | 7 | 0111 | w[3] = 0.1 | 1 | 0001 | 00010111 | 0x17 |
| 2 | w[4] = -0.7 | -7 | 1001 | w[5] = 0.3 | 3 | 0011 | 00111001 | 0x39 |
| 3 | w[6] = 0.0 | 0 | 0000 | w[7] = -0.4 | -4 | 1100 | 11000000 | 0xC0 |

Unpacking (low = byte and 0xF, high = byte >> 4, each read as a 4-bit two's complement number) returns the codes [5, -2, 7, 1, -7, 3, 0, -4]; they equal the codes [5, -2, 7, 1, -7, 3, 0, -4]: True. The eight weights now take 4 bytes plus one scale.
<!-- /worked -->

Byte `0xE5` holds the codes 5 (low nibble `0101`) and $-2$ (high nibble `1110`). The masks `and 0xF` matter: without them, the sign
extension of a negative code would spill ones into the neighbour's bits. Unpacking reads each nibble back through `signed(., 4)`. A group of eight
weights is now four bytes and a scale.

The 4-bit integer code is the narrowest format of the chapter, and the least forgiving: it has 15 usable levels, so the group size, the scale and the
rounding rule decide whether the result is usable. The memory table below counts what the scales add.

## Converting between formats

A model is stored in one format, multiplied in another and accumulated in a third, so every arrow between formats is a rounding step, and each has to be defined. Two directions behave differently.

**Widening is exact.** Every finite value of fp16, bf16, E4M3 and E5M2 is also a value of fp32, so moving a number to a wider format changes nothing. The same holds for the
relations the formats were designed with: bf16 is the top half of fp32, and E5M2 the top byte of fp16, which the table checks on every code.

<!-- worked: number_formats.format_relations -->
| relation | cases checked | violations |
|---|---:|---:|
| every finite value of fp16, bf16, E4M3, E5M2 is an fp32 value | 129270 | 0 |
| bf16 code c decodes like the fp32 pattern c << 16 | 65536 | 0 |
| E5M2 code c decodes like the fp16 pattern c << 8 | 256 | 0 |
| the finite positive codes of each format increase with their values | 4 | 0 |
| encode(decode(c)) = c for every finite code, negative zero excepted | 129266 | 0 |
| the negative-zero code decodes to 0 and 0 encodes as the positive-zero code (sign of zero is dropped) | 4 | 0 |
<!-- /worked -->

The last row says that the finite positive codes are ordered like their values, so two positive numbers can be compared as integers. (For negative numbers the order reverses, and NaN has no place in it.)

**Narrowing rounds once.** Going to a narrower format, the result is the nearest representable number, ties to the even code,
with the three edge cases above: overflow, underflow and the sign. The algorithm below is the whole conversion for any format of this chapter; it needs only
the split $(n_e, n_m)$, the finite-only flag and a choice of what to do on overflow (`saturate`).

```text
function encode_float(x: real, ne: int, nm: int, finite_only: bool, saturate: bool) -> u32:
    require 1 <= ne and 1 <= nm and 1 + ne + nm <= 32, "unsupported split"
    bias = 2^(ne - 1) - 1
    all_ones = 2^ne - 1
    if finite_only:
        largest = (all_ones << nm) + 2^nm - 2     # the code of the largest finite number
    else:
        largest = (all_ones << nm) - 1
    if x != x:                                    # not-a-number
        return (2^(ne + nm) - 1) if finite_only else ((all_ones << nm) or (1 << (nm - 1)))
    s = u32(1) if x < 0 else u32(0)       # -0 keeps its sign in hardware; here it is +0
    sign = s << (ne + nm)
    overflow = false
    if abs(x) == infinity:
        overflow = true
    else if x == 0:
        return sign
    else:
        a = abs(x)
        e = max(exponent_of(a), 1 - bias)          # subnormals share the lowest exponent
        n = round_even(a / 2^(e - nm))             # the number of steps of size 2^(e - nm)
        if n == 2^(nm + 1):                        # rounded up into the next binade
            e = e + 1
            n = 2^nm
        if n < 2^nm:                               # a subnormal number or zero
            return sign or u32(n)
        code = u32(((e + bias) << nm) + (n - 2^nm))
        if code <= largest:
            return sign or code
        overflow = true
    if saturate:
        return sign or u32(largest)
    if finite_only:
        return sign or u32(2^(ne + nm) - 1)        # no infinity: NaN
    return sign or u32(all_ones << nm)             # infinity
```

The division `a / 2^(e - nm)` is exact for a binary floating-point `a`, and the rounding is the only inexact step. When $n$ rounds up to $2^{n_m+1}$ the fraction overflows into
the exponent, which is exactly what rounding $1.111\ldots_2$ up to $10.000\ldots_2$ should do. The *tie* at the top of the range decides the overflow
boundary: in fp16 $65519$ stores as $65504$ but $65520$ is halfway to the next (non-existent) number and rounds to even, which is infinity. In E4M3 the same
rule puts the limit of an unsaturated conversion at 464, halfway between 448 and the would-be 480; the saturating conversion clamps everything above to 448.

**The bf16 shortcut.** For bf16 the conversion collapses into integer arithmetic, because bf16 is the top 16 bits of fp32 and the
rounding position is the boundary between two 16-bit halves. *Truncation* takes `bits(x) >> 16`. Rounding to nearest even adds a bias that carries
into the kept bits exactly when the discarded half is past the middle:

```text
function f32_to_bf16(x: f32) -> u16:
    b = bits(x)
    if (b and 0x7FFFFFFF) > 0x7F800000:          # NaN: keep it a NaN
        return u16((b >> 16) or 0x0040)
    return u16((b + 0x7FFF + ((b >> 16) and 1)) >> 16)
```

Why it works. Let the discarded low half of $b$ be $d \in [0, 65535]$. Adding $\mathtt{0x7FFF} = 32767$ carries into bit 16 when $d + 32767 \ge 65536$, that is
$d \ge 32769$: more than half. A tie is $d = 32768$; there the added lowest kept bit $\ell$ decides: $32768 + 32767 + \ell \ge 65536$ only when
$\ell = 1$. So a tie rounds up when the kept number is odd and stays when it is even, ties to even. A carry out of the fraction bits goes into the exponent field, which is the correct
rounding up to the next binade, and the largest numbers carry into infinity as they should.

How much does it matter whether we truncate or round? Truncation saves one addition. We measure what it costs:

<!-- worked: number_formats.bf16_rounding_bias -->
100000 values x_i = f32(normal()) drawn in order from the generator seeded 17662302740837105179 (seed_of 'number_formats.bf16_rounding_bias'), each rounded to bf16. 'mean change of |x|' is the mean of (|y| - |x|) / |x|; 'worst relative error' is the largest |y - x| / |x|; 'moved toward zero' is the share with |y| < |x|; the last two columns are the relative error of the sums of |x_i| and of x_i^2. The integer rule agreed with the exact rounding rule on 2000 of the first 2000 values.

| rounding | mean change of |x| | worst relative error | moved toward zero | error of sum |x| | error of sum x^2 |
|---|---:|---:|---:|---:|---:|
| round to nearest even | +0.0004% | 0.3889% | 49.9% | +0.0010% | +0.0029% |
| truncate (drop the low 16 bits) | -0.2812% | 0.7743% | 100.0% | -0.2820% | -0.5651% |
<!-- /worked -->

Truncation moves *every* value toward zero, so the mean magnitude drops by 0.28% and the sum of squares (the square of a norm) by 0.57%, a systematic loss
of the same sign on every element. Its worst-case error is also twice as large (0.77% against 0.39%), since the discarded part is a full unit in the last place instead of
half. Round to nearest even errs in both directions equally often (49.9% of values move toward zero), and its errors average out: the mean change is 0.0004%. The size of one error is not what separates them: a bias adds up with the number of terms, while errors of random sign grow more slowly.

### Narrow sums stop growing

An arithmetic operation in a format rounds its exact result to that format, and when
the result is much larger than the addend the addend can disappear. Adding 1.0 repeatedly to a running total
in a narrow format stops at a number we can predict: once the total reaches $2^p$ the gap between neighbours is 2, and $2^p + 1$ lies exactly halfway between $2^p$ and $2^p + 2$, a tie that goes to the even neighbour, the total itself.

<!-- worked: number_formats.accumulator_stall -->
Adding 1.0 repeatedly to a running total that starts at 0, rounding to the format after each addition (ties to even); the loop stops when an addition changes nothing.

| format | significand bits p = n_m + 1 | predicted stall 2^p | total where it stalls | additions made |
|---|---:|---:|---:|---:|
| fp16 | 11 | 2^11 = 2048 | 2048 | 2048 |
| bf16 | 8 | 2^8 = 256 | 256 | 256 |
| E4M3 | 4 | 2^4 = 16 | 16 | 16 |
| E5M2 | 3 | 2^3 = 8 | 8 | 8 |
| fp32 (one step) | 24 | 2^24 = 16777216 | 16777216 | 16777216 + 1 stores 16777216 |
<!-- /worked -->

An fp16 counter stops at 2048, a bf16 counter at 256, an E4M3 counter at 16, and every addition after that is silently lost. This is why dot products and sums of narrow numbers are accumulated in 32 bits (the significand of a product of two fp16 or bf16 numbers has at most 22 bits, which fp32's 24 hold), and only the result is rounded back.

The same effect decides whether a weight update takes effect. A bf16 weight at $w = 1$ has a gap of $2^{-7} = 0.0078$, and an update smaller than half of that is absorbed:

<!-- worked: number_formats.update_absorbed -->
The weight is w = 1. The update u is added in exact arithmetic and the sum is rounded once to the format (ties to even); the cell is the change that was stored, (stored sum) - 1.

| update u | fp32 | fp16 | bf16 |
|---|---:|---:|---:|
| 0.0005 | 4.9996e-04 | 9.7656e-04 | 0 |
| 0.001 | 0.001 | 9.7656e-04 | 0 |
| 0.002 | 0.002 | 0.0019531 | 0 |
| 0.004 | 0.0039999 | 0.0039062 | 0.0078125 |
| 0.005 | 0.005 | 0.0048828 | 0.0078125 |
| 0.01 | 0.01 | 0.0097656 | 0.0078125 |
| gap above 1 = 2^-n_m | 1.192e-07 | 9.766e-04 | 0.007812 |
<!-- /worked -->

An update of 0.001 leaves a bf16 weight at 1 exactly: the learning rate times the gradient can be too small for the format, for every step of
training, and the weight never moves. (The row for 0.0005 in fp16 is the opposite oddity: the update rounds up to a full gap, nearly double its size.) Training in a narrow
format therefore keeps the weights, or a master copy of them, in 32 bits, or rounds stochastically (rounds up with a probability equal to the fraction of a gap that the update covers, so that the average update survives); the later volume on training large models works both out.

## Bytes and bandwidth

The reason to put up with all of the above is the number of bytes, so we count them. A weight stored in $b$ bytes costs $b$ bytes of memory and, each time it is used, $b$ bytes of traffic. Generating one token with a dense model at batch size 1 reads each
weight once and does 2 FLOPs with it (a multiply and an add), so the work per token is $2P$ FLOPs and $Pb$ bytes: $2/b$ FLOPs per byte (the FLOPs per byte of a computation is its arithmetic divided by the bytes it moves; Chapter 3 builds the roofline on it). The scales of the
block and group formats are charged in full.

<!-- worked: number_formats.memory_per_format -->
A dense model of P = 7,000,000,000 parameters. Producing one token at batch size 1 reads every weight once and does 2 FLOPs per weight (a multiply and an add), so the bytes moved per token are P times the bytes per parameter and the FLOPs per byte are 2 divided by the bytes per parameter. GB = 10^9 bytes.

| storage of the weights | bytes per parameter | GB for the weights = bytes read per token | fraction of fp32 | FLOPs per byte |
|---|---:|---:|---:|---:|
| fp32 | 4.0000 | 28.000 | 1.000 | 0.50 |
| fp16 or bf16 | 2.0000 | 14.000 | 0.500 | 1.00 |
| E4M3 with an f32 scale per 128 x 128 block | 1.0002 | 7.002 | 0.250 | 2.00 |
| int8 with an f32 scale per 32 values | 1.1250 | 7.875 | 0.281 | 1.78 |
| int4 with an f32 scale per 32 values | 0.6250 | 4.375 | 0.156 | 3.20 |
<!-- /worked -->

For $7 \times 10^9$ parameters the weights alone go from 28 GB in fp32 to 14 GB in a 16-bit format, to about 7 GB in 8 bits and 4.4 GB in 4 bits with a scale
for every 32 values. The scales are not free: they turn "4 bits" into 5, and "8 bits" into 9. Because one token needs one pass over the
weights, the bytes per token are the table's second column, and the arithmetic per byte rises as the bytes fall: the FLOPs per byte rise as $2/b$, from 0.5 for fp32 to 3.2 for the 4-bit format, and a device limited by its memory (Chapter 3 of this volume gives the test) gains exactly as the bytes fall. How a number of seconds follows depends on the device's bandwidth, and is for that chapter and for the profiling chapter.

Training stores more than the weights. Volume III, Chapter 7 counted 16 bytes per parameter for AdamW in 32 bits:

<!-- worked: number_formats.training_bytes -->
Bytes of training memory per parameter for AdamW (two moment slots, each in the moment format), as the sum of the value, the gradient, the moments and the optional master copy; int8-block moments cost 1 + 4/64 = 1.0625 bytes each. The last column is the total for 7,000,000,000 parameters in decimal GB.

| policy | value | gradient | moments (2 slots) | master | total bytes | GB for 7e9 |
|---|---:|---:|---:|---:|---:|---:|
| all fp32 | 4 | 4 | 8 | 0 | 16 | 112.0 |
| bf16 value and gradient, fp32 moments, fp32 master | 2 | 2 | 8 | 4 | 16 | 112.0 |
| bf16 value and gradient, bf16 moments, fp32 master | 2 | 2 | 4 | 4 | 12 | 84.0 |
| bf16 value and gradient, int8-block moments, fp32 master | 2 | 2 | 2.125 | 4 | 10.125 | 70.9 |
| bf16 everything, no master | 2 | 2 | 4 | 0 | 8 | 56.0 |
<!-- /worked -->

The second row is the surprise of mixed precision: with a 32-bit master copy and 32-bit moments, a bf16 model needs *as many* parameter bytes as the 32-bit one, 16 per
parameter. The saving of that configuration is in the activations and the bandwidth of the matrix products, not in the parameter memory. Only narrower moments, or dropping the master copy,
reduce the bill, and each of those costs precision somewhere else; the later volume on training large models works the trade out.

## Rules that keep narrow formats safe

The sections above leave five working rules, each of which a bug has taught someone.

- **Store narrow, compute wide.** Weights are stored in the narrowest format that preserves the result, decoded as they are loaded,
  and multiplied and summed in a wider one. Sums and dot products accumulate in 32 bits, as the stalls above require.
- **Scales are stored wider than the codes they scale.** A block scale is a 32-bit number (or 16-bit), because an error in a
  scale is an error in every value of its block.
- **State every conversion's overflow rule.** Infinity, NaN or saturation: the choice is part of the format of a pipeline, and with E4M3 it is the only
  choice there is.
- **NaN is a pattern, not a value.** Comparing NaN to anything is false, and a NaN written to a narrow format may become a number if the
  conversion does not guard the pattern (the bf16 conversion above keeps the NaN's quiet bit for that reason).
- **Check a decode on every code.** An 8-bit format has 256 codes and a 16-bit one 65536: a decoder can be tested against an independent
  reference exhaustively, not by examples, and the NaN pattern of E4M3 is exactly the kind of case an example misses.

## How brain stores and converts these formats

**The conversions.** `crates/model/src/half.rs` is the host side of brain's 16-bit tiers. `f32_to_bf16`
is this chapter's shortcut: the doc comment states why rounding is not truncation ("Truncation always rounds toward zero, so every packed weight would be silently biased low in
magnitude") and the code adds `0x7FFF` plus the lowest kept bit before shifting. Its tests pin the rule at the boundaries: a tie
that goes down, a tie that goes up and a value just above the middle. One of them records a testing lesson: the first tie, `0x3F808000`, would give the same answer under truncation "by luck", so the test that actually separates the two rules is the one just above the half. `f32_to_f16` delegates to the
`half` crate instead of repeating the work, and the file says why (a well-tested library beats a second hand-written conversion on the host, where one can be called, unlike inside a shader). The same bf16 rule is in CUDA in
`crates/kernels-cuda/cu/cast_f32_bf16.cu` (`(b + 0x7fff + ((b >> 16) & 1)) >> 16`, with NaN mapped to `0x7fc0`). The bf16 storage tier arrived with commit `4b8230dae` (13 August 2026), whose message
describes the host pack, the WGSL decode of two bf16 values per 32-bit word and the end-to-end round-trip test; the f16 tier followed in commit `3ee8a158b`, and the device decode of f16 is checked against an independent
reference for every one of the 65536 patterns (`f16_decode_matches_an_independent_reference_for_every_possible_bit_pattern` in `crates/kernels/src/template.rs`).

**The 8-bit formats.** `crates/model/src/fp8.rs` defines `Fp8Format::{E4M3, E5M2}` with exactly this chapter's conventions: the doc comments give E4M3 as "no infinity, max finite 448" and E5M2 as "has infinities, max finite 57344",
`max_finite` returns those numbers, and `encode` rounds to the nearest finite code, ties to the even code, saturating (NaN keeps a NaN code). The reference decoders are in
`crates/checkpoint/src/safetensors.rs`: `e4m3fn_to_f32` returns NaN only for exponent and fraction all ones, and `e5m2_to_f32` returns infinity for exponent 31 and fraction 0. The block scale of the chapter is
`quantize_weight_fp8`: one 32-bit scale per tile of the weight, the tile's largest magnitude divided by `max_finite`; the checkpoint format brain reads uses one scale per 128 by 128 block
(`dequant_block128`), and the CUDA activation quantisers do the same per 128 values (`crates/kernels-cuda/cu/add_rms_quant_e4m3.cu` computes `s = amax(|xn| over 128) / 448` and stores `e4m3(xn / s)`,
rounded to nearest even and saturating, through the hardware instruction `cvt.rn.satfinite.e4m3x2.f32` in `crates/kernels-cuda/cu/e4m3_quant_common.cuh`).

Two stories come from this code. First, decoding. In commit `068d2ff53` (19 August 2026) a profile of importing a real 27-billion-parameter FP8 checkpoint showed that the E4M3
byte decode was the dominant cost: it recomputed each value, with calls to `powi`, for tens of millions of elements, though an 8-bit format has only 256 possible bytes. The fix
is the table `LUT: [f32; 256]` that `e4m3fn_to_f32` indexes today, bit-identical to the formula it memoizes.

<!-- perf-number: historical measurement, commit 068d2ff53 -->
The commit message reports, for one tensor of that checkpoint in August 2026, a decode time falling from about 5.0 s to about 0.41 s on the machine it was run on, a dated observation of one setup. The lesson that does not depend on it is the work count: a format with 256 values needs 256 evaluations of the formula, not one per element.

Second, the reserved NaN. In commit `e49c0a169` (5 September 2026) the device-side decode of E4M3 was written as a few integer and bit-cast operations (`f8e4m3_decode_expr` in `crates/kernels/src/template.rs`). Its comment records
that the multiplication that rebuilds a normal number "would otherwise compute a perfectly ordinary finite value (480.0) for this exact byte", the 480 of the table above, so the decode has a third branch that overrides the NaN pattern. The decode was checked
byte for byte against an independent reference over all 256 bytes of each format
(`f8e4m3_decode_matches_an_independent_reference_for_every_possible_byte` and its E5M2 sibling).

**The integer formats.** `crates/model/src/int8.rs` quantizes a weight in groups of `GROUP = 32` values along the reduction axis, with `max|block| / 127` as the scale and four 8-bit codes to a 32-bit word. Its module comment gives the reason the group is small, the outlier argument of this chapter,
and `crates/model/src/int4.rs` does the same with a scale of `max|block| / 7` ("7, not 15": the symmetric range, for the reason int8 uses 127), eight nibbles per 32-bit word with nibble `b` in bits `[4b, 4b + 4)`, which is this chapter's byte rule continued to eight
codes. (That quantizer rounds with Rust's `round`, which sends ties away from zero, where this chapter uses ties to even; the two differ only on exact ties.)

**The byte accounting.** `crates/paramstore/src/dtype.rs` has `Destination::bytes`: 4 bytes per element for fp32, 2 for bf16 and fp16, and for `Int8Group { group }` one byte per element plus 4 bytes per group (`numel + 4 * ceil(numel / group)`), the 1.125 of the memory table for a
group of 32. `crates/paramstore/src/precision.rs` has the training bill of the last table: `ParamPrecision::bytes` is the sum of the value, the gradient, the moment slots in the state format and the master copy,
and its comment says why the scales are counted: "so a budget built on it cannot under-report the way an 'elements x 4' model does".

## Check your understanding

Answers are at the end of the chapter.

1. **Find the error.** "To negate a two's complement number, flip all its bits." Apply it to $5 = $ `00000101` in 8 bits and say what number the result is.
2. **Predict.** In int8 two's complement, what is $100 + 100$, as stored? And in an unsigned 8-bit register?
3. **Explain.** A gradient of $10^{-9}$ is stored in fp16 and in bf16. Using the bit splits, say what each does with it, and what fp16 needs to keep it.
4. **Predict.** A running total in fp16 starts at 0 and adds 1.0 five thousand times. What does it hold at the end, and why?
5. **Find the error.** "bf16 is fp32 with the low 16 bits discarded, so converting is a shift, and the shift is exact enough." What is wrong, and in which direction does the result err?
6. **Explain.** Why is the largest finite E4M3 number 448 and not 240, and what did the format give up for it?
7. **Predict.** A group of 4-bit weights has largest magnitude $0.35$. What is the scale, and what do the weights $0.12$ and $-0.35$ decode to?
8. **Explain.** One weight in a group of 32 is ten times larger than the rest. What happens to the other 31 under an int8 code with one scale for the group, and what is the remedy?

## Summary

A number format is a rule that reads a bit pattern. Whole numbers use the binary place values, wrap modulo $2^n$, and represent negatives in two's complement, the pattern $2^n - x$, which is the bits inverted plus one and lets one adder serve
both signs. Floating point splits the pattern into sign, exponent field and fraction field, and one decoding rule covers fp32, fp16, bf16 and both 8-bit formats: each fraction bit halves the gap
between numbers, each exponent bit squares the range. Of the two 16-bit formats, fp16 has a range of 12 decades and 3.3 digits, bf16 79 decades and 2.4 digits, which is why bf16 trains without loss
scaling and fp16 does not. The 8-bit formats E4M3 (448, one NaN and no infinity) and E5M2 (57344, IEEE special values) are usable only with a scale per block. Integer-quantized formats store small codes and a shared scale (and sometimes a zero point), with
constant absolute error, so the scale must belong to a small group, and four-bit codes pack two to a byte with shifts and masks. Narrowing rounds once and the rounding rule matters: truncation makes the bf16 error one-sided and
twice as large, accumulators in narrow formats stop growing at $2^p$, and an update smaller than half a gap vanishes. The bill is in bytes: 28 GB for seven billion parameters in fp32, 14 in a 16-bit format, 7 and 4.4 in the
8-bit and 4-bit formats with scales, and a mixed-precision training run with a master copy and 32-bit moments saves no parameter memory at all.

The limits are the chapter's own: the formats are described from their bit definitions, but how to choose a scale from data, how much precision a given layer can lose, and how to train in
low precision are open questions that later chapters and volumes answer. The next chapter turns from the numbers to the machine that computes with them, a CPU with its caches and vector registers,
because a format that moves fewer bytes only helps if the machine is waiting for bytes.

## Exercises

1. **[Calculate]** Convert 83 to binary by repeated division by 2. Then form the 8-bit two's complement pattern of $-83$ by invert-and-add-one, and check that the two patterns add to 0 modulo 256.
2. **[Calculate]** Decode by hand, with the rule of this chapter: the fp16 pattern `0x4900`, the bf16 pattern `0xC0A0` and the E4M3 pattern `0xB4`. Give each as a decimal number.
3. **[Calculate]** A hypothetical IEEE-style format has 3 exponent bits and 4 fraction bits. Give its bias, largest finite number, smallest normal number, smallest subnormal number, $\varepsilon$ and decades of range, as in the format table.
4. **[Calculate]** Convert the fp32 patterns `0x3F808000`, `0x3F818000` and `0x3F808001` to bf16 with round to nearest even by the integer rule, and by truncation. Which of the three distinguishes the two rules?
5. **[Calculate]** A model has $13 \times 10^9$ parameters. Give its weight memory in decimal GB in bf16, in E4M3 with one 32-bit scale per 128 by 128 block, and in 4-bit codes with one 16-bit scale per group of 64.
6. **[Implement]** Implement `decode_float` and `encode_float` from this chapter in your language and reproduce `number_formats.float_formats` and `number_formats.special_codes`. Check that `encode_float(decode_float(c)) == c` for every finite code of the four small formats except the negative-zero code of each, which decodes to 0 and so encodes as the positive-zero code (the exhaustive test of `number_formats.format_relations`).
7. **[Implement]** Implement two's complement negation, addition with wrap and sign extension and reproduce `number_formats.twos_complement`, `number_formats.twos_complement_arithmetic` and `number_formats.unsigned_wrap`.
8. **[Implement]** Implement the SplitMix64 stream of Volume I, Chapter 17, the bf16 shortcut and truncation, and reproduce `number_formats.bf16_rounding_bias` (the seed and draw order are printed with it). Then reproduce `number_formats.accumulator_stall` and `number_formats.update_absorbed`.
9. **[Implement]** Implement symmetric and zero-point quantization, the per-block E4M3 scale and the int4 packing, and reproduce `number_formats.int8_group_scale`, `number_formats.int8_zero_point`, `number_formats.fp8_block_scale` and `number_formats.int4_pack`.
10. **[Investigate]** Reproduce `number_formats.loss_scaling` and then change the spread of the stand-in gradients to ten decades centred on $10^{-4}$ instead of $10^{-7}$. Find the smallest power of two $S$ for which fewer than 1% of the gradients become zero in fp16, and the largest one for which none overflow. Is there a window at all? What does your answer say about a fixed scale for a training run?
11. **[Investigate]** There are six ways to split 7 bits between exponent and fraction (at least one bit each). For each, with the finite-only convention and with the IEEE convention, compute the largest number, the number of decades of range and the number of decimal digits using your `decode_float`. Which splits have both more range and more precision than E5M2? Which is E4M3?
12. **[Investigate]** For $u = 2^{-k}$, $k = 1, 2, \dots$, use your `encode_float` to find the largest $k$ for which $1 + u$ rounds to a number other than 1, in fp16 and in bf16, and explain the answer from the tie rule. Then find an update that is stored as nearly twice its size, and say what that does to a sum of many small updates.

## Key terms

- **binary representation**: writing a non-negative whole number as bits, each with a place value that is a power of 2, so that $n$ bits give the $2^n$ numbers $0 \dots 2^n - 1$ and arithmetic on them wraps modulo $2^n$. <!-- concept: binary-representation -->
- **two's complement**: the representation of negative whole numbers in which the pattern of $-x$ is $2^n - x$ (the bits of $x$ inverted, plus one), so that the same addition works for both signs, the top bit is the sign, and the range is $-2^{n-1} \dots 2^{n-1} - 1$. <!-- concept: twos-complement -->
- **fp32 format**: the 32-bit floating-point layout of 1 sign bit, 8 exponent bits (bias 127) and 23 fraction bits, with about 7 decimal digits and a range of 83 decades. <!-- concept: fp32-format -->
- **fp16 format**: the 16-bit floating-point layout of 1 sign bit, 5 exponent bits (bias 15) and 10 fraction bits: about 3.3 digits, but a largest finite number of only 65504 and a smallest subnormal of $6 \times 10^{-8}$. <!-- concept: fp16-format -->
- **bf16 format**: the 16-bit floating-point layout of 1 sign bit, 8 exponent bits (the exponent field of fp32) and 7 fraction bits: fp32's range with 2.4 digits, obtained from an fp32 by dropping the low 16 bits (with rounding). <!-- concept: bf16-format -->
- **fp8 format**: either of the two 8-bit floating-point layouts, E4M3 (4 exponent, 3 fraction bits, no infinity, one NaN pattern, largest finite number 448) and E5M2 (5 exponent, 2 fraction bits, IEEE infinities and NaNs, largest finite number 57344), used with a scale per block of values. <!-- concept: fp8-format -->
- **integer-quantized format**: a storage format that keeps each number as a small integer code (8 or 4 bits) and recovers it as a shared scale times the code (less a zero point where the range is not symmetric). <!-- concept: integer-quantized-format -->

## Answers

**Recall questions.**

1. $2^8 = 256$ patterns: $0 \dots 255$ unsigned, $-128 \dots 127$ with half used for negatives.
2. The sign $s$, the exponent $e$ and the fraction $f$ (the significand has $p - 1$ stored bits plus an implied leading 1). Within a binade the gap is $2^{e-(p-1)}$, so entering the next binade doubles $e$'s contribution and doubles the gap while the number also roughly doubles: the relative gap stays between $2^{-p}$ and $2^{-(p-1)}$.
3. $\varepsilon = 2^{-23} = 1.19 \times 10^{-7}$, with $p = 24$: $\varepsilon = 2^{1-p}$, twice the unit round-off $u = 2^{-p}$.
4. A result beyond the largest finite number overflows to infinity (when it is past the halfway point to the next would-be number); a result below the smallest normal number is stored as a subnormal with fewer significant bits, and below half the smallest subnormal as zero. A value exactly halfway goes to the neighbour with the even fraction.
5. $4 + 4 + 8 = 16$ bytes per parameter (value, gradient, two moments); $16 \times 7 \times 10^9 = 112$ GB.

**Check your understanding.**

1. Flipping gives `11111010`, which reads as $-6$, not $-5$: the missing step is adding one (`11111011` is $-5$). Flipping gives $-x - 1$.
2. $200$ does not fit in 8 signed bits; the pattern is $200 \bmod 256 = 200$, `11001000`, which reads as $200 - 256 = -56$. In an unsigned register the result is 200, which fits.
3. fp16's smallest subnormal is $6 \times 10^{-8}$ and anything below half of it, $3 \times 10^{-8}$, becomes zero, so $10^{-9}$ is stored as 0. bf16 has fp32's exponent field: $10^{-9}$ is a normal number there, stored with an error of at most 0.4%. fp16 needs a loss scale $S$ that lifts $10^{-9}$ above its subnormal range, for example $S = 2^{16}$ gives $6.6 \times 10^{-5}$, near its smallest normal number, with $S$ small enough that the large gradients do not exceed 65504.
4. 2048. At 2048 the gap in fp16 is 2, $2048 + 1$ is a tie between 2048 and 2050 and goes to the even fraction, 2048; every later addition is lost the same way.
5. The shift always rounds toward zero (the discarded bits are dropped), so every value is stored with a smaller magnitude: a bias of about $-0.28\%$ on normal data in the chapter's measurement, and a worst case of one full unit in the last place, twice the error of rounding to nearest. Add the half-way bias `0x7FFF` and the kept bit before shifting.
6. With the IEEE rule the all-ones exponent field is reserved for infinity and NaN, and the largest number is $(1 + 7/8)\,2^{7} = 240$. E4M3 uses that field for ordinary numbers, leaving one NaN pattern, so the largest is $(1 + 6/8)\,2^{8} = 448$. It gave up infinity and all but one NaN pattern, so an overflow has no infinity to become.
7. $\Delta = 0.35/7 = 0.05$. $0.12/0.05 = 2.4$ rounds to 2, which decodes to $0.10$ (error 0.02); $-0.35$ has code $-7$ and decodes to exactly $-0.35$.
8. The scale is set by the large weight, $\Delta = 10 m/127$ where $m$ is the typical magnitude, so the step is $0.079\,m$ and a typical weight is rounded with an absolute error up to $0.04\,m$ (4% of its size), and weights smaller than $0.04\,m$ are stored as zero. The remedy is a smaller group (or a separate scale for the outlier), at the price of one more stored scale.
