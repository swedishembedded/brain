<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Pseudocode Conventions

The algorithms in this book are written in a small pseudocode language
rather than in any real programming language. It is meant to be translated
line by line into whatever language you use. For that to work, every
construct has to mean exactly one thing. This page defines them. An
algorithm in the book uses nothing that is not defined here; a chapter that
needs a new operation adds it to this page in the same change.

## Values and types

Every value has a type, and the book always states it when it matters.

| Type | Meaning |
|---|---|
| `int` | a mathematical integer: arithmetic on `int` values is exact, so `2^64` is a legal value even where your language's native integer is narrower (a 64-bit language computes `2^64 mod n` as `(2^64 - n) mod n`) |
| `u64` | an unsigned 64-bit integer, `0 .. 2^64 - 1`; see "Unsigned 64-bit integers" below |
| `f32`, `f16`, `bf16` | floating-point numbers of the stated format (Volume IV) |
| `real` | a floating-point number whose format does not matter to the algorithm; implement it as `f32` unless told otherwise. A literal such as `0.0` takes the type of the variable it is stored in |
| `complex` | a pair of `real` numbers, see "Complex numbers" below |
| `bool` | `true` or `false` |
| `array<T>[d0, d1, ...]` | a tensor of element type `T` with the given shape |
| `function(T) -> U` | a function taken as an argument or returned (a `function(real) -> real` maps a number to a number; with several arguments the types are listed, `function(real, real) -> real`) |
| `optional T` | either a value of type `T` or nothing. `x is present` is true when it holds a value and false otherwise; the value is only read after that test |
| `text` | a string of characters. Its **bytes** are those of its UTF-8 encoding, each an `int` in `0 .. 255`, taken in order: `for b in the bytes of s:` visits them first to last (the text `"ab"` has the bytes `97, 98`) |
| `(T, U)` | a pair (tuple) of a `T` and a `U`; an argument may have a tuple type, `a: (real, int)`, and `a[0]`, `a[1]` read its first and second component (0-indexed like an array) |
| `Generator` | the state of a random number generator, see "Randomness" |
| `list<T>` | an ordered, growable list of values of type `T`, indexed from 0: `L[i]` reads element `i` (`0 <= i < len(L)`; outside is an error), `len(L)` is its length, `append x to L` adds at the end. The elements may be arrays of different shapes (a network's weight matrices) |
| `none` | the value of a function that returns nothing |

A **record** groups named fields into one value:

```text
record View:
    storage: array<real>[S]
    offset: int
```

`v.offset` reads a field, and a record value is built by naming the type and listing its fields in order (`View(storage, offset)`). A field of a record variable is assigned with `v.offset = 3`; this changes that variable only (a record is a value, so `w = v` makes `w` an independent copy), and a function changes a record argument only when it is marked `inout`. A function that returns a record names the record type in its signature, `-> View`, and every field is set on every path before the `return`. A function may return several values as a tuple,
`-> (t: real, r: array<real>[n])`, written `return (t, r)` and received as
`(t, r) = f(x)`. Single letters such as `T` in `array<T>[n]` stand for any
element type, the same one everywhere in the signature.

Dimension names in a signature, such as `N` in `x: array<real>[N]`, are
variables bound to the actual dimensions when the function is called. Using
one name twice (`x: array<real>[N], y: array<real>[N]`) requires that the two
dimensions are equal, as if the first line of the function were a `require`.

## Arrays

Arrays are **indexed from 0**. An array of shape `[d0, d1, d2]` holds
`d0 * d1 * d2` elements, and `a[i, j, k]` requires
`0 <= i < d0`, `0 <= j < d1`, `0 <= k < d2`. Reading or writing outside
those bounds is an error, never a silent wrap or clamp. A literal is written
`[1, 2, 3]` for a vector and `[[1, 2], [3, 4]]` for a matrix whose rows are
the inner lists.

Unless a chapter says otherwise, arrays are stored in **row-major order**:
the last index varies fastest. Element `a[i, j, k]` of an array of shape
`[d0, d1, d2]` lives at offset `(i * d1 + j) * d2 + k` from the start of
its storage. Volume I, chapter "Arrays, Shapes, Strides and Broadcasting",
derives this and the general stride formula.

`a.shape` is the shape (a list of integers), `a.shape[0]` its first
dimension, and `len(a)` the same as `a.shape[0]`. `numel(a)` is the number
of elements, the product of the dimensions; `numel(s)` of a list `s` of
integers is the product of the list (1 for the empty list), which is the
element count of an array of shape `s`. `zeros(T, [d0, d1])` creates an
array of element type `T` filled with zero, and `identity(n)` the `n` by `n`
matrix with ones on the diagonal and zeros elsewhere.
`augment(A, y)` appends the vector `y: [m]` to `A: [m, n]` as a new last
column and returns the array of shape `[m, n + 1]`.

**Slices.** An index position in `a[...]` holds either an integer or a
range `start:stop` or `start:stop:step`. A range is half-open: `2:5` is
`2, 3, 4`; a missing `start` is 0, a missing `stop` is the dimension, a
missing `step` is 1, and a lone `:` is the whole axis. **An integer drops its
axis; a range keeps it.** So for `a` of shape `[3, 4]`:

| expression | shape | what it is |
|---|---|---|
| `a[1, 2:4]` | `[2]` | the two elements at `j = 2, 3` of row 1 |
| `a[1, :]` | `[4]` | row 1 as a vector |
| `a[1:2, :]` | `[1, 4]` | row 1 as a matrix of one row |
| `a[:, 2]` | `[3]` | column 2 as a vector |
| `a[::2, :]` | `[2, 4]` | rows 0 and 2 |

A range with a step `k > 1` has `ceil((stop - start) / k)` elements (`0:5:2` is `0, 2, 4`, three elements; with `stop` the dimension when it is missing). A slice is a **view**: it refers to the same storage, so writing through it
changes the original. Where a copy is needed, the pseudocode says
`copy(...)`: `copy(x)` returns a new array (or list) of the same shape and
values that shares no storage with `x`.

`rows(A, idx)` gathers rows: for `A` of shape `[N, ...]` and a list `idx` of integers, each in `0 .. N-1`, it returns a new array of shape `[len(idx), ...]` whose row `m` is a copy of `A[idx[m], ...]`. The indices may repeat and need not be sorted; an index outside `0 .. N-1` is an error. (A slice cannot express this, because a slice selects a regular range.)

A list comprehension `[expression for i in 0 .. n]` builds the vector of the
expression's values for `i = 0, 1, ..., n - 1`, and `[expression for i in 0 ..
n if condition]` keeps only the values where the condition holds. The same form iterates over the elements of a list, `[expression for x in L]` (and `[expression for x in L if condition]`), in list order. `[]` is the
empty list and `append x to L` adds `x` at the end of the list `L`; `len(L)`
is its length. `for index in indices(shape)` visits every index tuple of an
array of that shape, in row-major order (the last index varies fastest).

## Arithmetic on arrays

Arithmetic operators apply to scalars. When an operator is applied to whole
arrays, the pseudocode says so explicitly with a loop or with one of these
named operations:

| Operation | Meaning |
|---|---|
| `matmul(A, B)` | matrix product; `A: [m, k]`, `B: [k, n]` gives `[m, n]` |
| `transpose(A)` | swaps the last two axes |
| `sum(a)`, `mean(a)` | sum and arithmetic mean of all elements of `a` (`mean` divides by `numel(a)`; `a` must not be empty) |
| `sum(a, axis)`, `mean(a, axis)` | the same along one axis, which is removed from the shape |
| `max(a)`, `min(a)` | largest and smallest element of the whole array |
| `max(a, axis)`, `min(a, axis)` | along one axis |
| `argmax(a)`, `argmin(a)` | index of the largest (smallest) element of a vector; on a tie the **first** index wins |
| `sort(a)` | a new vector with the elements of `a` in ascending order; equal elements keep their original relative order |
| `argsort(a)`, `argsort(a, descending)` | the vector of indices that puts `a` in ascending (or descending) order; ties keep their original relative order, so `argsort` is stable |
| `percentile(a, q)` | for a vector `a` with `m >= 1` elements and a fraction `q` in `[0, 1]`: sort `a` ascending into `v`, let `pos = q * (m - 1)`, `lo = floor(pos)`, `hi = min(lo + 1, m - 1)`, and return `v[lo] + (pos - lo) * (v[hi] - v[lo])` (linear interpolation between neighbouring order statistics; Volume I, chapter 12 and the project of chapter 18) |
| `median(a)` | `percentile(a, 0.5)`: the middle element of an odd count, the average of the two middle elements of an even count. A chapter that needs the lower or upper middle element of an even count says so and does not call it `median` |
| `sd(a)` | the sample standard deviation of a vector `a` with `m >= 2` elements, with divisor `m - 1`: `sqrt(sum((a[i] - mean(a))^2 for i in 0 .. m) / (m - 1))`. A standardiser that divides by `m` says so in its signature and does not call `sd` |
| `correlation(a, b)` | the Pearson correlation of two vectors of equal length `m >= 2`: `sum((a[i] - mean(a)) * (b[i] - mean(b))) / sqrt(sum((a[i] - mean(a))^2) * sum((b[i] - mean(b))^2))`, a `real` in `[-1, 1]`; it is an error (`require`) when either vector is constant, because a denominator is then zero. The divisor of the covariance cancels, so none is stated |
| `map(f, a)` | applies the scalar function `f` to every element |
| `a + b`, `a * b` on arrays | elementwise, and **only** when the shapes are equal or the text states the broadcast |

`sort a ascending` and `sort a descending` as statements sort the array
`a` in place with the same tie rule.

Broadcasting is never implicit. When a smaller array is combined with a
larger one, the text writes the rule: for example "add `bias: [D]` to every
row of `x: [T, D]`".

## Scalar operations

| Operation | Meaning |
|---|---|
| `+`, `-`, `*` | addition, subtraction, multiplication |
| `/` | division: real division, even when both operands are `int` (`7 / 2` is `3.5`); to halve an `int` size or index use `//` (`N // 2`), never `/` |
| `a // b` | integer division for `int` operands, rounded toward minus infinity (`7 // 2` is `3`, `-7 // 2` is `-4`) |
| `a mod n` | remainder after division by `n > 0`, always in `0 .. n - 1`, so that `a = (a // n) * n + (a mod n)` (`-7 mod 2` is `1`) |
| `a ^ b` | power: `a` to the exponent `b`. It is never exclusive-or (that is `xor`). `a ^ 0` is 1 for every `a`, including `0 ^ 0 = 1`, because it is an empty product |
| `abs(x)` | absolute value; for a `complex` number its modulus |
| `sign(x)` | **+1 if `x >= 0`, otherwise -1**. In particular `sign(0) = +1`, unlike the sign function of mathematics, which gives 0 there. Algorithms that need the convention say so (the rotation formulas of Jacobi's method in Chapter 6 divide by `abs(theta) + ...` and need a nonzero rotation when `theta = 0`) |
| `min(a, b, ...)`, `max(a, b, ...)` | smallest and largest of two or more scalars |
| `floor(x)`, `ceil(x)` | the largest integer not above `x`, the smallest integer not below `x`; both return an `int` |
| `sqrt`, `exp`, `log`, `sin`, `cos` | the usual functions; `log` is the natural logarithm and `sin`, `cos` take radians |
| `log1p(x)` | `log(1 + x)` for `x > -1`, computed so that it stays accurate when `x` is tiny (Volume I, Chapter 15); where `1 + x` would round to 1, `log1p(x)` still returns about `x` |
| `log2(x)`, `log10(x)` | logarithms to base 2 and 10 |
| `round(x, n)` | `x` rounded to `n >= 0` decimal places, **ties away from zero**: take the exact decimal value of `x` as written or computed, find the two multiples of `10^(-n)` that enclose it, and pick the nearer one; when `x` is exactly halfway, pick the one farther from zero (`round(2.5, 0) = 3`, `round(-2.5, 0) = -3`, `round(0.125, 2) = 0.13`). It returns a `real`. The book avoids a printed value that depends on this rule: where a table is rounded, the underlying quantity is not a tie, or the text says which way it was rounded. A binary floating-point number is rarely exactly halfway, so a reproduction compares unrounded values with a tolerance (see "Numerical tolerance") before it rounds |
| `sigmoid(z)` | the logistic function $1/(1 + e^{-z})$, computed as `1 / (1 + exp(-z))` for `z >= 0` and as `exp(z) / (1 + exp(z))` for `z < 0`, so that `exp` never overflows; the result lies in $(0, 1)$ and `sigmoid(0) = 0.5` (Volume II, chapter 3) |
| `softplus(z)` | $\log(1 + e^z)$, computed as `max(z, 0) + log1p(exp(-abs(z)))`, which is exact in real arithmetic and cannot overflow; `softplus(0) = log(2)` (Volume II, chapter 3) |
| `tanh(x)` | the hyperbolic tangent $(e^x - e^{-x})/(e^x + e^{-x})$, computed as `2 * sigmoid(2 * x) - 1`; odd, in $(-1, 1)$, `tanh(0) = 0` (Volume III, chapter 2) |
| `relu(x)` | `max(0, x)`: `x` when `x > 0` and `0` otherwise, so `relu(0) = 0`; its slope is taken as `1` for `x > 0` and `0` for `x <= 0` (Volume III, chapter 2) |
| `gelu(x)` | the exact Gaussian-error linear unit `0.5 * x * (1 + erf(x / sqrt(2)))`, that is `x` times the standard Gaussian cumulative distribution at `x`; the tanh approximation is always named `gelu_tanh` (Volume III, chapter 2) |
| `silu(x)` | `x * sigmoid(x)` (Volume III, chapter 2) |
| `erf(x)` | the error function $\frac{2}{\sqrt\pi}\int_0^x e^{-s^2}ds$, odd, with `erf(0) = 0` and limits $\pm 1$; the standard Gaussian cumulative distribution is $\Phi(z) = \frac12\bigl(1 + \mathrm{erf}(z/\sqrt2)\bigr)$. Use the library routine of your language (a reproduction agrees to the printed digits; its series is not needed) |
| `atan2(y, x)` | the angle in radians, in $(-\pi, \pi]$, of the point with horizontal coordinate `x` and vertical coordinate `y` (the angle from the positive horizontal axis, anticlockwise); `atan2(0, 0)` is `0` |
| `pi`, `infinity` | the constant $\pi$ and positive infinity (`-infinity` is its negative) |
| `and`, `or`, `not` | logical operations on `bool`; `and` and `or` stop at the first operand that decides the result |
| `x if c else y` | `x` when the condition `c` is true, otherwise `y`; only the chosen operand is evaluated |

**Arithmetic in a stated format.** When a text says an expression is evaluated in `f32` (or another format) *one operation at a time*, each `+`, `-`, `*`, `/` and each call of `sqrt`, `exp`, `log` or `log1p` rounds its exact result, computed from the already rounded inputs, to the nearest number of the format, ties to even. Negation, `abs`, `min`, `max` and comparisons are exact and are not rounding steps. A result beyond the largest finite number becomes `infinity`; tiny results are kept as subnormal numbers (nothing is flushed to zero); `log(0)` is `-infinity`, `1 / infinity` is `0`, and `infinity - infinity` is not a number. Volume II, Chapter 3 uses this to show how a naive loss fails.

Comparison operators are `==`, `!=`, `<`, `<=`, `>`, `>=`. Every ordering comparison (`<`, `<=`, `>`, `>=`) with a not-a-number operand is false, so `v < best` is false when `v` is not a number, and an algorithm that keeps the lowest value seen never keeps a not-a-number one. A **chained comparison** `a <= b < c` means `a <= b and b < c`, with `b` evaluated once; it is only used with the operators `<` and `<=` pointing the same way. Operators bind in
the usual order of arithmetic (`^` first, then `*`, `/`, `//`, `mod`, then
`+`, `-`, then comparisons, then `not`, `and`, `or`). The bitwise operations
below are always fully parenthesised in the book, so they need no precedence
rule.

## Unsigned 64-bit integers

A value of type `u64` is an integer in `0 .. 2^64 - 1`.

- `+`, `-` and `*` **wrap around**: the result is the true result taken
  modulo $2^{64}$. So `0xFFFFFFFFFFFFFFFF + 1` is `0`, and `0 - 1` is
  `0xFFFFFFFFFFFFFFFF`. A language whose integers are wider than 64 bits
  reproduces this by taking `mod 2^64` after each operation, which is why
  the book writes `mod 2^64` in the places where wrapping matters, even
  though a `u64` already does it.
- `a xor b`, `a and b`, `a or b` act on the 64 bits independently
  (`1` where the bits differ, are both 1, or at least one is 1);
  `not a` flips every bit (`2^64 - 1 - a`).
- `a >> k` and `a << k` shift the 64 bits right or left by `k` places
  (`0 <= k < 64`): bits shifted out are lost and zeros are shifted in. So
  `a >> k` is `a // 2^k`, and `a << k` is `(a * 2^k) mod 2^64`.
- `a // b` and `a mod b` are the integer division and remainder of the
  non-negative values.
- Integer literals may be written in hexadecimal, `0x9E3779B97F4A7C15`.
- `u64(x)` converts a non-negative `int` below `2^64` to `u64`; to turn
  the top 53 bits of a `u64` into a `real`, divide `a >> 11` by `2^53`.

`reverse_bits(i, b)` is the integer whose `b`-bit binary form is the form of
`i` read backwards (`reverse_bits(1, 3)` is `4`, since `001` becomes `100`).

## Complex numbers

A `complex` number is a pair of `real` numbers, its real part `z.re` and
imaginary part `z.im`; `complex(a, b)` builds the pair, which the book writes
$a + b\,\mathrm{i}$. The imaginary unit is written `1i` in pseudocode, so that
the letter `i` stays free as an ordinary index: `2.0 + 3.0 * 1i` is
`complex(2.0, 3.0)`, and `1i * 1i` is `complex(-1.0, 0.0)`. Zero is
`complex(0.0, 0.0)`, which is what `zeros(complex, [N])` fills with.

| Operation | Meaning |
|---|---|
| `z + w`, `z - w` | componentwise |
| `z * w` | `complex(z.re*w.re - z.im*w.im, z.re*w.im + z.im*w.re)`; a `real` is a complex number with imaginary part 0 |
| `z / w` | `z * conj(w) / (w.re^2 + w.im^2)`; `w` must not be zero |
| `conj(z)` | `complex(z.re, -z.im)` |
| `abs(z)` | `sqrt(z.re^2 + z.im^2)` |
| `exp(z)` | `exp(z.re) * complex(cos(z.im), sin(z.im))` |
| `atan2(z.im, z.re)` | the phase (argument) of `z` |

## Control flow

```text
for i in 0 .. n:          # i = 0, 1, ..., n-1 (n excluded)
for i in 0 ..= n:         # i = 0, 1, ..., n   (n included)
for i in n-1 down to 0:   # reverse order
parallel for i in 0 .. n: # iterations are independent and may run at once
while condition:
loop:                     # repeat until a return or break inside
if condition: ... else if condition: ... else: ...
break                     # leave the innermost loop
continue                  # start the next iteration of the innermost loop
return value
```

A range `a .. b` may start anywhere, not only at 0: `for i in r + 1 .. m` visits `r + 1, ..., m - 1`, and it is empty when `a >= b`.

`else if` is `else` followed by an `if`; at most one branch runs. A loop
over an empty range (`for i in 0 .. 0`) runs zero times.

A function may call itself (**recursion**); a recursive function states the case in which it returns without calling itself, and
every call works on a strictly smaller input, so that the calls end. Volume II builds decision trees this way.

A `parallel for` promises that no iteration reads anything another
iteration writes. If an algorithm needs iterations to combine results, it
uses an explicit reduction (Volume IV), never a shared variable.

Row operations on a matrix `M` are written as statements on slices: `M[r, :] = M[r, :] - c * M[p, :]` subtracts `c` times row `p` from row `r`, and `M[r, :] = M[r, :] / d` divides row `r` by `d`. The sentence forms "subtract `c` times row `p` from row `r`" and "divide row `r` by `d`" mean exactly these statements; each is a loop over the columns of the row.

`swap a, b` exchanges the values of two variables or two array elements or
two slices of equal shape (`swap x[i], x[j]`, `swap M[r, :], M[p, :]`). The
sentence forms "swap `a[i]` and `a[j]`" and "swap rows `r` and `p` of `M`"
mean the same. `print "text", value` writes to the program's output, and
`record x in the run's output` stores `x` so that it can be shown afterwards.

## Functions and mutation

```text
function name(arg1: type, arg2: type) -> type:
    ...
```

A signature too long for one line continues on the following lines, indented; a parameter of function type is written with the types of its arguments and result, `score: function(array<real>[M, d], array<real>[M]) -> real`, and the function passed in must accept exactly those arguments.

Arguments are read-only unless marked `inout`, in which case the function
may change them and the caller sees the change. A function never changes
anything that is not an argument marked `inout` or a value it created.
There is no hidden global state; anything an algorithm needs, including a
random number generator, is passed in.

## Routines defined in Volume I

Three numerical routines are used by later chapters and defined, with their
code, in Volume I. A chapter that calls one names it and states the arguments;
it does not redefine it.

- **Gaussian elimination.** `solve(A, y, partial)` of Volume I, chapter 4
  ("Solving systems: Gaussian elimination", "The algorithms") returns the
  `x` with `A x = y` for a square `A: [n, n]`, and stops with an error when
  `A` is singular. The third argument is the **pivoting rule**: `true` picks,
  in each column, the row at or below the diagonal with the largest absolute
  value in that column (partial pivoting); `false` takes the first row whose
  entry is not zero. Every use in Volume II passes `true`.
- **Symmetric eigenproblem.** `jacobi_eigen(S, sweeps)` of Volume I, chapter 6
  ("The algorithms") returns `(values, V)` for a symmetric `S`: the eigenvalues
  on the diagonal at the end (in no particular order, so a caller sorts them
  descending with the columns of `V`) and the unit eigenvectors as the columns
  of `V`. A call states the number of sweeps (`20` is ample for the matrices of
  the book).
- **Newton's method.** Volume I, chapter 7 ("Newton's method: using the tangent
  to jump") derives `x_next = x - f(x) / fprime(x)` for a root of `f`, and
  `x_next = x - g(x) / gprime(x)` for a minimum of a function with first
  derivative `g` and second derivative `gprime`. One iteration is one such
  update; a chapter states the starting value and the stopping rule.

## Randomness

Random numbers come from an explicit generator `rng` of type `Generator`,
the SplitMix64 of Volume I, chapter "Randomness on a Deterministic Machine".
Its state is a `u64`, equal to the seed before the first draw, plus a held
Gaussian value (initially empty). The calls are:

| Call | Meaning |
|---|---|
| `rng.next()` | the next raw `u64`: add `0x9E3779B97F4A7C15` to the state (mod $2^{64}$), then mix the new state; one call consumes one raw value |
| `rng.uniform()` | `(r >> 11) / 2^53` for one raw value `r`: a `real` in $[0, 1)$ |
| `rng.integer(n)` | an `int` in `0 .. n-1` (`0, ..., n-1`), the remainder `r mod n` of the first raw value `r >= 2^64 mod n` (rejection sampling) |
| `rng.normal()` | a sample of $\mathcal{N}(0, 1)$ by Box-Muller. A call with no held value draws `u1 = rng.uniform()` then `u2 = rng.uniform()`, sets `R = sqrt(-2 * log(1 - u1))`, holds `R * sin(2 * pi * u2)` and returns `R * cos(2 * pi * u2)`; the next call returns the held value without drawing anything |
| `shuffle(a, rng)` | permutes the array `a` in place: for `i` from `n-1` down to `1`, `j = rng.integer(i + 1)` and `swap a[i], a[j]` |
| `Generator(s)` | a new generator whose state is the `u64` `s` before its first draw (a seed `s` below $2^{64}$; there is no held Gaussian value yet) |
| `derive_seed(run_seed, name)` | the seed of the stream called `name` under a run seed: `h = mix(run_seed)`, then `h = mix(h xor b)` for each byte `b` of the UTF-8 name in order, and `h` is the result in `0 .. 2^64 - 1`, where `mix` is the three scrambling lines of the SplitMix64 step without the state advance (Volume I, chapter 17, which gives the code and printed values). `Generator(derive_seed(s, "init"))` is the stream "init" of run seed `s` |

Two runs with generators created from the same seed produce the same numbers.
`os_entropy()` returns a `u64` taken from the operating system, the only
nondeterministic call in the book (its name keeps it apart from the entropy of
a distribution, `entropy(p)`, Chapter 13). The chapter on randomness gives the
derivation of each rule and checks it on printed values; every worked example
that draws random numbers prints its seed and the order of its draws so that
you can reproduce it.

## Errors

`require condition, "message"` stops the algorithm with an error if the
condition is false. Algorithms check their inputs this way at the start:
shapes, ranges and any assumption the derivation relied on. An algorithm
never silently corrects bad input. How the error is reported (an exception,
an error code) is up to your language.

## Numerical tolerance

Two floating-point results are compared with a tolerance, never with `==`,
unless the algorithm needs an exact test and says so (for example a pivot
that is exactly zero, or two integers). `close(a, b, rtol, atol)` (arguments always passed by position, in this
order) is true
when $\lvert a - b \rvert \le \mathrm{atol} + \mathrm{rtol} \cdot \lvert b \rvert$
for every element; it also applies to two scalars. Each chapter states the
tolerance its tests use and explains where it comes from.

When a worked example is a printed table, a reproduction agrees with it when
every value matches after rounding to the digits printed (a difference below
half a unit in the last printed place), unless the text states another
tolerance.

## A complete example

```text
function softmax_rows(z: array<real>[T, V]) -> array<real>[T, V]:
    require T > 0 and V > 0, "softmax of an empty array"
    p = zeros(real, [T, V])
    parallel for t in 0 .. T:
        m = max(z[t, :])                      # subtract the row maximum
        s = 0.0                               # so exp() cannot overflow
        for v in 0 .. V:
            p[t, v] = exp(z[t, v] - m)
            s = s + p[t, v]
        for v in 0 .. V:
            p[t, v] = p[t, v] / s
    return p
```

Volume III explains every line of it, including why subtracting the
maximum changes nothing mathematically and everything numerically.
