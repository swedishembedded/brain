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
| `u8`, `u16`, `u32` | unsigned integers of 8, 16 and 32 bits, `0 .. 2^n - 1`; see "Fixed-width integers and bit patterns" below |
| `f32`, `f16`, `bf16` | floating-point numbers of the stated format (Volume IV, chapter 1) |
| `real` | a floating-point number whose format does not matter to the algorithm; implement it as `f32` unless told otherwise. A literal such as `0.0` takes the type of the variable it is stored in |
| `complex` | a pair of `real` numbers, see "Complex numbers" below |
| `bool` | `true` or `false` |
| `array<T>[d0, d1, ...]` | a tensor of element type `T` with the given shape |
| `function(T) -> U` | a function taken as an argument or returned (a `function(real) -> real` maps a number to a number; with several arguments the types are listed, `function(real, real) -> real`). A function is passed by writing its name, `map(tanh, a)`, and called through the argument's name |
| `optional T` | either a value of type `T` or nothing. `x is present` is true when it holds a value and false otherwise; the value is only read after that test |
| `text` | a string of characters. Its **bytes** are those of its UTF-8 encoding, each an `int` in `0 .. 255`, taken in order: `for b in the bytes of s:` visits them first to last (the text `"ab"` has the bytes `97, 98`) |
| `(T, U)`, `(T, U, V)` | a pair or triple (tuple) of a `T` and a `U` (and a `V`); an argument may have a tuple type, `a: (real, int)`, and `a[0]`, `a[1]`, `a[2]` read the components (0-indexed like an array). A function may return a tuple of unnamed components, `-> (real, int, int)`, received as `(x, i, j) = f(...)` |
| `Generator` | the state of a random number generator, see "Randomness" |
| `list<T>` | an ordered, growable list of values of type `T`, indexed from 0: `L[i]` reads element `i` (`0 <= i < len(L)`; outside is an error), `len(L)` is its length, `append x to L` adds at the end. The elements may be arrays of different shapes (a network's weight matrices) |
| `tensor` | a name for `array<real>` of any shape, used where an algorithm treats values of different shapes alike (a graph's node values); `t.shape` is its shape and `numel(t)` its number of elements |
| `operation` | a value that names one differentiable operation of a graph (an input, a parameter, a matrix product, a `tanh`, ...) and carries the constants it needs; a chapter lists the operations it uses, and `op is input` tests which kind a value is |
| `bytes` | an ordered sequence of `u8` values (a serialised checkpoint, the input of a hash); `len(b)` is its length, `b[i]` is byte `i` (0-indexed, `0 <= i < len(b)`) and `append x to b` adds one at the end, as for a `list`. Floats written into bytes are stored as their IEEE 754 bit patterns (little-endian, 8 bytes for a double) so that they read back to the same bits |
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

**Assigning to a slice.** A slice may stand on the left of `=`: `A[i:j, :] = B` copies the elements of `B` into the elements of `A` that the slice selects and changes no other element of `A`. The right side is evaluated first, and its shape must equal the shape of the slice (an integer index drops its axis on both sides alike: `A[r, :] = v` needs a vector `v` of the width of row `r`, and `O[:, a:b] = Oh` needs `Oh` of shape `[T, b - a]`); a shape that differs is an error. The slice is a view, so the write changes the array that the slice was cut from, and `X[c:c + n, :] = matmul(A, W)` fills rows `c .. c + n - 1` of `X`.

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
| `sum(a, axis)`, `mean(a, axis)` | the same along one axis, which is removed from the shape. The keyword form `sum(a, axis=0)` means the same as `sum(a, 0)` |
| `max(a)`, `min(a)` | largest and smallest element of the whole array |
| `max(a, axis)`, `min(a, axis)` | along one axis |
| `argmax(a)`, `argmin(a)` | index of the largest (smallest) element of a vector; on a tie the **first** index wins |
| `sort(a)` | a new vector with the elements of `a` in ascending order; equal elements keep their original relative order |
| `argsort(a)`, `argsort(a, descending)` | the vector of indices that puts `a` in ascending (or descending) order; ties keep their original relative order, so `argsort` is stable |
| `percentile(a, q)` | for a vector `a` with `m >= 1` elements and a fraction `q` in `[0, 1]`: sort `a` ascending into `v`, let `pos = q * (m - 1)`, `lo = floor(pos)`, `hi = min(lo + 1, m - 1)`, and return `v[lo] + (pos - lo) * (v[hi] - v[lo])` (linear interpolation between neighbouring order statistics; Volume I, chapter 12 and the project of chapter 18) |
| `median(a)` | `percentile(a, 0.5)`: the middle element of an odd count, the average of the two middle elements of an even count. A chapter that needs the lower or upper middle element of an even count says so and does not call it `median` |
| `sd(a)` | the sample standard deviation of a vector `a` with `m >= 2` elements, with divisor `m - 1`: `sqrt(sum((a[i] - mean(a))^2 for i in 0 .. m) / (m - 1))`. A standardiser that divides by `m` says so in its signature and does not call `sd` |
| `correlation(a, b)` | the Pearson correlation of two vectors of equal length `m >= 2`: `sum((a[i] - mean(a)) * (b[i] - mean(b))) / sqrt(sum((a[i] - mean(a))^2) * sum((b[i] - mean(b))^2))`, a `real` in `[-1, 1]`; it is an error (`require`) when either vector is constant, because a denominator is then zero. The divisor of the covariance cancels, so none is stated |
| `all(c)`, `any(c)` | for an array or list `c` of `bool`: true when every (some) element is true; `all` of an empty list is true and `any` of an empty list is false |
| `dot(a, b)` | `sum(a[i] * b[i] for i in 0 .. len(a))` for two vectors of equal length (an error otherwise); the scalar $\mathbf{a}\cdot\mathbf{b}$ |
| `norm(x)` | the Euclidean length `sqrt(sum(x[i]^2 for i in 0 .. len(x)))` of a vector, also written $\lVert\mathbf{x}\rVert$; for a list of tensors the square root of the sum of the squares of every entry of every tensor (the global norm) |
| `geometric_mean(a)` | `exp(mean(log(a)))` for a vector `a` of positive numbers |
| `sha256(b)` | the SHA-256 digest (FIPS 180-4) of the `bytes` `b`, as 32 bytes; written as 64 lowercase hexadecimal digits when printed. The book prints only a prefix of it to compare two runs (Volume III, Chapter 13) |
| `power_iteration(W, iters, v0)` | an estimate of the largest singular value of a matrix `W: [m, n]`: starting from the vector `v0` of length `n`, repeat `iters` times: multiply `v` by `W`, multiply the result by the transpose of `W`, and divide by its norm to get the new `v`; return `norm(W v)` at the end. `v0` is used as given, not normalised first (the division after the first round does it, so `iters >= 1` and `v0` must not be zero); normalising `v0` first would change nothing in the result. The result is the spectral norm of Volume I, Chapter 6 (Eigenvalues, Rotations and Low-Rank Structure); this is a different function from the eigenvalue `power_iteration` of that chapter |
| `onehot(c, C)` | the vector of length `C` with `1.0` at index `c` and `0.0` elsewhere (`0 <= c < C`) |
| `softmax(z)`, `lse(z)` | for a vector `z` of length `C >= 1`: `m = max(z)`, `lse(z) = m + log(sum(exp(z[i] - m)))` and `softmax(z)[i] = exp(z[i] - m) / sum(exp(z[j] - m))`, the stable forms of Volume III, chapter 3 (`logsumexp` is a synonym of `lse`); applied to a matrix `softmax(Z)` works on each row |
| `map(f, a)` | applies the scalar function `f` (a function passed by name, `map(silu, Zg)`) to every element of the array `a` and returns an array of the same shape. It is a different thing from the table type `map<K, V>` of the Volume V section below, which has type arguments in angle brackets |
| `a + b`, `a - b`, `a * b`, `a / b` on arrays | elementwise (the Hadamard product $\odot$ for `*`), and **only** when the shapes are equal or the text states the broadcast; a chain such as `dHd * Zu * map(silu_slope, Zg)` is evaluated left to right on arrays of one shape. `matmul` is the only operation that multiplies matrices |

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
| `fma(a, b, c)` | the fused multiply-add of Volume IV, Chapter 9: `a * b + c` formed exactly and rounded **once** to the stated format (ties to even), where the expression `a * b + c` evaluated one operation at a time rounds twice (the product, then the sum); on `f32` operands `fma(a, b, c)` is the `f32` nearest to the exact value of `a * b + c` |
| `floor(x)`, `ceil(x)` | the largest integer not above `x`, the smallest integer not below `x`; both return an `int` |
| `sqrt`, `exp`, `log`, `sin`, `cos` | the usual functions; `log` is the natural logarithm and `sin`, `cos` take radians |
| `sinh(x)`, `asinh(x)` | the hyperbolic sine $(e^x - e^{-x})/2$ and its inverse $\ln(x + \sqrt{x^2 + 1})$, the library functions `sinh` and `asinh` of the C math library; both are odd, increasing and defined for every real `x`, and `sinh(asinh(x)) = x` to rounding |
| `log1p(x)` | `log(1 + x)` for `x > -1`, computed so that it stays accurate when `x` is tiny (Volume I, Chapter 15); where `1 + x` would round to 1, `log1p(x)` still returns about `x` |
| `log2(x)`, `log10(x)` | logarithms to base 2 and 10 |
| `round(x, n)` | `x` rounded to `n >= 0` decimal places, **ties away from zero**: take the exact decimal value of `x` as written or computed, find the two multiples of `10^(-n)` that enclose it, and pick the nearer one; when `x` is exactly halfway, pick the one farther from zero (`round(2.5, 0) = 3`, `round(-2.5, 0) = -3`, `round(0.125, 2) = 0.13`). It returns a `real`. The book avoids a printed value that depends on this rule: where a table is rounded, the underlying quantity is not a tie, or the text says which way it was rounded. A binary floating-point number is rarely exactly halfway, so a reproduction compares unrounded values with a tolerance (see "Numerical tolerance") before it rounds |
| `sigmoid(z)` | the logistic function $1/(1 + e^{-z})$, computed as `1 / (1 + exp(-z))` for `z >= 0` and as `exp(z) / (1 + exp(z))` for `z < 0`, so that `exp` never overflows; the result lies in $(0, 1)$ and `sigmoid(0) = 0.5` (Volume II, chapter 3) |
| `softplus(z)` | $\log(1 + e^z)$, computed as `max(z, 0) + log1p(exp(-abs(z)))`, which is exact in real arithmetic and cannot overflow; `softplus(0) = log(2)` (Volume II, chapter 3) |
| `tanh(x)` | the hyperbolic tangent $(e^x - e^{-x})/(e^x + e^{-x})$ as the **correctly rounded library function** `tanh` of the stated format (in double precision, the `tanh` of the C math library or Python's `math.tanh`; in a stated 32-bit format, that value rounded to 32 bits); odd, in $(-1, 1)$, `tanh(0) = 0` (Volume III, chapter 2). As a remark, mathematically $\tanh x = 2\,\mathrm{sigmoid}(2x) - 1$, but that composition rounds differently in the last digits, and tables that print noise-level digits (Volume III, chapters 4 and 6) were computed with the library function |
| `relu(x)` | `max(0, x)`: `x` when `x > 0` and `0` otherwise, so `relu(0) = 0`; its slope is taken as `1` for `x > 0` and `0` for `x <= 0` (Volume III, chapter 2) |
| `gelu(x)` | the exact Gaussian-error linear unit `0.5 * x * (1 + erf(x / sqrt(2)))`, that is `x` times the standard Gaussian cumulative distribution at `x`; the tanh approximation is always named `gelu_tanh` (Volume III, chapter 2) |
| `silu(x)` | `x * sigmoid(x)` (Volume III, chapter 2) |
| `silu_slope(z)` | the derivative of `silu` at `z`: with `s = sigmoid(z)`, `s * (1 + z * (1 - s))`; `silu_slope(0) = 0.5` (Volume V, Chapter 9) |
| `erf(x)` | the error function $\frac{2}{\sqrt\pi}\int_0^x e^{-s^2}ds$, odd, with `erf(0) = 0` and limits $\pm 1$; the standard Gaussian cumulative distribution is $\Phi(z) = \frac12\bigl(1 + \mathrm{erf}(z/\sqrt2)\bigr)$. Use the library routine of your language (a reproduction agrees to the printed digits; its series is not needed) |
| `acos(x)`, `angle(a, b)` | `acos(x)` is the angle in $[0, \pi]$ whose cosine is `x`, for `x` in `[-1, 1]` (clamp a quotient that rounding pushed just outside). `angle(a, b)` for two nonzero vectors is `acos(dot(a, b) / (norm(a) * norm(b)))` |
| `cos_sim(a, b)` | the cosine similarity of two nonzero vectors of equal length, `dot(a, b) / (norm(a) * norm(b))`, a `real` in `[-1, 1]` (clamp a quotient that rounding pushed just outside); it is an error (`require`) when either vector is zero, because the denominator is then zero (Volume I, chapter 3). `angle(a, b)` is its `acos` |
| `step(z)` | `1` if `z >= 0` and `0` otherwise, so `step(0) = 1` (Volume III, chapter 1) |
| `atan2(y, x)` | the angle in radians, in $(-\pi, \pi]$, of the point with horizontal coordinate `x` and vertical coordinate `y` (the angle from the positive horizontal axis, anticlockwise); `atan2(0, 0)` is `0` |
| `is_finite(x)` | true when `x` is neither `infinity`, `-infinity` nor not-a-number, false otherwise (so `is_finite(0.0 / 1.0)` is true and `is_finite(infinity - infinity)` is false); it is the test that a value can still be used in arithmetic (Volume III, chapter 12) |
| `pi`, `infinity`, `nan` | the constant $\pi$, positive infinity (`-infinity` is its negative) and the not-a-number value (`is_finite(nan)` is false, `nan == nan` is false). `exp(-infinity)` is `0` and `max(x, -infinity)` is `x`, so a running maximum can start at `-infinity` and a masked score of `-infinity` has weight 0; `-infinity - (-infinity)` is `nan`, so an algorithm that subtracts a running maximum tests for a row that has seen nothing (Volume V, Chapter 13) |
| `and`, `or`, `not` | logical operations on `bool`; `and` and `or` stop at the first operand that decides the result |
| `x if c else y` | `x` when the condition `c` is true, otherwise `y`; only the chosen operand is evaluated |

**Arithmetic in a stated format.** When a text says an expression is evaluated in `f32` (or another format) *one operation at a time*, each `+`, `-`, `*`, `/` and each call of `sqrt`, `exp`, `log` or `log1p` rounds its exact result, computed from the already rounded inputs, to the nearest number of the format, ties to even; a value stored into a variable of that format is rounded the same way (`w = w + h` stored in `f32` is the exact sum of the two `f32` numbers rounded to `f32`; a mixed expression says which format each part is evaluated in, for example a quotient of two `f32` losses formed in `f64`). Negation, `abs`, `min`, `max` and comparisons are exact and are not rounding steps. A result beyond the largest finite number becomes `infinity`; tiny results are kept as subnormal numbers (nothing is flushed to zero); `log(0)` is `-infinity`, `1 / infinity` is `0`, and `infinity - infinity` is not a number. Volume II, Chapter 3 uses this to show how a naive loss fails.

Comparison operators are `==`, `!=`, `<`, `<=`, `>`, `>=`; on `text` values `==` is true when both have the same characters in the same order (`scheme == "he"`). Every ordering comparison (`<`, `<=`, `>`, `>=`) with a not-a-number operand is false, so `v < best` is false when `v` is not a number, and an algorithm that keeps the lowest value seen never keeps a not-a-number one. A **chained comparison** `a <= b < c` means `a <= b and b < c`, with `b` evaluated once; it is only used with the operators `<` and `<=` pointing the same way. Operators bind in
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

## Fixed-width integers and bit patterns

The types `u8`, `u16` and `u32` behave like `u64` with $2^n$ in place of
$2^{64}$ (Volume IV, chapter 1): `+`, `-` and `*` wrap, taking the true result
modulo $2^n$; `a and b`, `a or b` and `a xor b` act on the $n$ bits
independently; `not a` flips all $n$ bits (`2^n - 1 - a`); `a >> k` is
`a // 2^k`; `u8(x)`, `u16(x)` and `u32(x)` convert an `int` in range, like `u64(x)`; and `a << k` is `(a * 2^k) mod 2^n`, so bits shifted past the top
are lost. Shift counts satisfy `0 <= k < n`. Literals may be written in
hexadecimal (`0x7FFF`) or binary (`0b0101`).

| Operation | Meaning |
|---|---|
| `pattern(x, n)` | the `n`-bit pattern of the integer `x`, that is `x mod 2^n`; for a negative `x` this is `2^n + x` (`pattern(-1, 8) = 255`) |
| `signed(a, n)` | the value of the `n`-bit pattern `a` read as a two's complement number: `a` if `a < 2^(n-1)`, otherwise `a - 2^n` (`signed(255, 8) = -1`, `signed(128, 8) = -128`) |
| `bits(x)`, `from_bits(a)` | for an `f32` value `x`, the `u32` holding its 32 stored bits (sign, exponent, fraction, in that order from the top); and the `f32` whose stored bits are the `u32` `a`. Each undoes the other. Volume IV, chapter 1 defines the layout |
| `f16_of(a)` | for a `u16` pattern `a`, the `f16` number whose stored bits (sign, 5 exponent bits, 10 fraction bits, from the top) are `a`, decoded with the rule of Volume IV, chapter 1 (so `f16_of(0x3C00) = 1.0` and `f16_of(0x0000) = 0.0`); a file stores it as two bytes, the low byte first, so the number held in bytes `b[0], b[1]` is `f16_of(u16(b[0]) or (u16(b[1]) << 8))` |
| `round_even(x)` | the integer nearest to `x`, **ties to the even integer** (`round_even(2.5) = 2`, `round_even(3.5) = 4`, `round_even(-2.5) = -2`, `round_even(0.4) = 0`); it returns an `int` and is the rule the book calls round to nearest even |
| `round_shift(x, n)` | for an `int` `x` and an `int` `n >= 0`: `round_even(x / 2^n)` with the quotient taken exactly (never through a rounded float), so the integer nearest to `x / 2^n`, ties to the even integer (`round_shift(5, 1) = 2`, `round_shift(7, 1) = 4`, `round_shift(-5, 1) = -2`, `round_shift(-7, 1) = -4`, `round_shift(6, 2) = 2`, `round_shift(x, 0) = x`); it is the rounded right shift of Volume IV, Chapter 13, and unlike `x >> n` on a negative `x` it does not round toward minus infinity |
| `exponent_of(x)` | for `x > 0`, the integer `e` with `2^e <= x < 2^(e+1)` (`exponent_of(0.75) = -1`, `exponent_of(8) = 3`); computed exactly, never as a rounded `floor(log2(x))` |

`clamp(x, l, u)` is `min(max(x, l), u)` (Volume III, chapter 2).

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

## Kernels (Volume IV)

A **kernel** is the program of one invocation of a data-parallel computation; a `launch` runs it for every invocation of a grid
of workgroups (Volume IV, Chapters 4 and 5).

```text
kernel scale_shift(x: array<f32>[N], y: inout array<f32>[N], scale: f32, shift: f32, N: int) workgroup n_wg:
    if g >= N: return                  # the guard for the idle tail of the grid
    y[g] = scale * x[g] + shift

launch scale_shift(x, y, 2.0, 1.0, N) over ceil(N / n_wg) workgroups
```

- The header names the inputs, the outputs (`inout`, as everywhere in this book) and the constants, and gives the
  workgroup size `n_wg` (a positive `int`). A kernel may also declare workgroup memory in the header,
  `shared tile: array<f32>[k]`.
- Inside the body three read-only `int` values are defined: the local index `l` of the invocation in its workgroup
  (`0 <= l < n_wg`), the index `w` of its workgroup, and the global index `g = w * n_wg + l`. A launch over a two-dimensional
  grid of `n_x` by `n_y` workgroups gives each workgroup the coordinates `w_x` and `w_y` and defines `w = w_y * n_x + w_x`.
- `return` ends this invocation only. The body of a kernel reads and writes its arguments and its `shared` arrays and nothing else.
- `launch name(arguments) over n workgroups` (or `over n_x by n_y workgroups`) runs the body once for every pair `(w, l)`.
  The workgroups run in any order, and so may the invocations of one workgroup; a result that depends on that order is a data
  race. A launch with no `barrier` is a `parallel for` over all `g`: no invocation reads an element that another writes.
- A `shared` array holds unspecified values when a workgroup starts; the kernel writes an element before any invocation reads it.
- `barrier` is a statement of the body. Every invocation of the workgroup that is still running must reach it, and none passes it
  until all have arrived, so the statements before it have finished, for every invocation of the workgroup, before any statement
  after it begins. A `return` before a `barrier` is legal only if every invocation of the workgroup takes it; a kernel in which
  some invocations of a workgroup reach the barrier and others do not is wrong, and the book says its result is undefined.
- `atomic_add(a, i, v)` (Volume IV, Chapter 7) sets `a[i] = a[i] + v` as one indivisible step. Calls on the same word
  from different invocations are served one at a time, in an order the device chooses and the program cannot see. The
  book uses it only to describe the obvious way of summing; its kernels, like the portable reference shaders of brain,
  do not use atomics.

**Fragments (Volume IV, Chapter 6).** A **fragment** is a small matrix that the invocations of one subgroup hold jointly: a value of type
`fragment<T>[R, C]` with `R` rows and `C` columns of element type `T`. The language does not say which invocation holds which
element, so a kernel never reads or writes one element of a fragment; it only passes whole fragments to the four operations below. A kernel
that uses fragments has a workgroup of exactly one subgroup (`workgroup n_sg`, 32 invocations in the book's model), and every invocation of
the subgroup executes each fragment operation together, so none sits inside a branch the invocations take differently.

| Operation | Meaning |
|---|---|
| `fragment_zero(T, R, C)` | a fragment of type `fragment<T>[R, C]` with every element zero |
| `fragment_load(src, r0, c0, R, C)` | for a two-dimensional `src` of element type `T`: the fragment whose element `[i, j]` is `src[r0 + i, c0 + j]`; the whole block `r0 .. r0 + R - 1` by `c0 .. c0 + C - 1` must lie inside `src` (there is no per-element guard) |
| `fragment_mad(a, b, c)` | for `a: fragment<f16>[R, K]`, `b: fragment<f16>[K, C]` and `c: fragment<f32>[R, C]`: the fragment `d: fragment<f32>[R, C]` with `d[i, j] = c[i, j] + sum_p a[i, p] * b[p, j]`; the book's model forms every product exactly and rounds the whole sum, `c[i, j]` included, once to `f32` (real units differ in how they align and truncate the sum, and the order of the `K` terms is not specified) |
| `fragment_store(f, dst, r0, c0)` | the inverse of `fragment_load`: `dst[r0 + i, c0 + j] = f[i, j]` for the whole block, which must lie inside `dst` |

The **host** program that runs a kernel uses six operations on a device (Volume IV, Chapter 5). The device keeps a queue of work and
runs it in the order it was enqueued.

| Operation | Meaning |
|---|---|
| `compile(source: text) -> kernel` | turns the text of a kernel into a kernel the device can run; done once per program, not once per launch |
| `create_buffer(bytes: int) -> buffer` | allocates that many bytes of device memory; the contents are unspecified |
| `upload(b: buffer, data)` | enqueues a copy of host values into the buffer; `data` is read when the call returns, so the host may reuse it |
| `dispatch(k: kernel, buffers: list<buffer>, params: list<u32>, n_grp: int)` | enqueues a launch of `k` over `n_grp` workgroups with the buffers bound in order and the parameters in a small constant block; returns at once, without waiting |
| `read_back(b: buffer, count: int) -> array<f32>[count]` | waits until every queued operation has finished, then copies the first `count` numbers of the buffer to the host |
| `release(b: buffer)` | gives the buffer's memory back; the buffer is not used again |

**Recording (Volume IV, Chapter 8).** `dispatch` hands the device one launch at a time. A host that has several launches ready
can record them first and hand them over together, with three more operations:

| Operation | Meaning |
|---|---|
| `begin_commands() -> commands` | starts an empty list of commands; only one list is open at a time |
| `record(c: commands, k: kernel, buffers: list<buffer>, params: list<u32>, n_grp: int)` | appends one launch, with the arguments of `dispatch`, to the list `c`; nothing reaches the device and nothing is bound or checked until the list is submitted |
| `submit(c: commands)` | closes the list and enqueues all of it as one **submission**: the launches run in the order they were recorded, each seeing the writes of those before it, and the call returns at once |

`dispatch(k, buffers, params, n_grp)` is `begin_commands`, one `record` and `submit`: one submission holding one launch.
`read_back` waits for every submission made before it, and a list that is still open (recorded, not submitted) is not part of
that wait: a program reads back only after it has submitted what the result depends on.

**Backends (Volume IV, Chapter 9).** A **backend** is a program that provides the host operations above for one kind of device;
code above it uses only these operations, and the book states the contract as an `interface` block, a list of operation
signatures that every backend implements with the meanings given here:

```text
interface Backend:
    capabilities() -> record
    compile(source: text) -> kernel
    create_buffer(bytes: int) -> buffer
    upload(b: buffer, data)
    dispatch(k: kernel, buffers: list<buffer>, params: list<u32>, n_grp: int)
    read_back(b: buffer, count: int) -> array<f32>[count]
    release(b: buffer)
```

`capabilities()` returns a record that describes the device and never changes while the backend lives. Its fields are
`max_workgroup` (the largest workgroup size a kernel may declare, an `int`), `shared_bytes` (the shared memory of one workgroup in
bytes), `subgroup` (the width of a subgroup, 0 when the device does not say) and `max_barriers` (the most barriers one kernel may contain: 0,
1, or `infinity`). A capability a backend cannot establish is reported as the value that forbids the feature (0 or `false`), never as the value that allows
it. `compile` either returns a kernel that computes what its source computes or reports an error that names the kernel; a backend never
substitutes an approximation. A backend may also provide `begin_commands`, `record` and `submit`.

**Measuring (Volume IV, Chapter 11).** Two operations let a host program read a clock and wait, and two let a command list carry the
device's own clock. They are a model of what every backend offers in some form, and the chapter's pseudocode uses nothing else to time work:

| Operation | Meaning |
|---|---|
| `now() -> real` | the host's clock, in seconds; only differences of two readings mean anything |
| `wait_idle()` | blocks until every submission made so far has finished on the device (the wait that `read_back` performs before it copies, without the copy); a list still open (recorded, not submitted) is not part of it |
| `record_stamp(c: commands) -> int` | appends to the list `c` a point at which the device writes its own clock, between the launches recorded before and after it, and returns the index of that stamp; stamps are numbered 0, 1, 2, ... in the order recorded |
| `stamp_span(c: commands, i: int, j: int) -> real` | the seconds of the device's clock between stamps `i` and `j` of the list `c` after it has run (`i <= j`); an error before `wait_idle()` has returned for the submission of `c` |

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

Four routines are used by later chapters and defined, with their
code, in Volume I. A chapter that calls one names it and states the arguments;
it does not redefine it.

- **Gaussian elimination.** `solve(A, y, partial)` of Volume I, chapter 4
  (its sections on Gaussian elimination and its pseudocode) returns the
  `x` with `A x = y` for a square `A: [n, n]`, and stops with an error when
  `A` is singular. The third argument is the **pivoting rule**: `true` picks,
  in each column, the row at or below the diagonal with the largest absolute
  value in that column (partial pivoting); `false` takes the first row whose
  entry is not zero. Every use in Volume II passes `true`.
- **Symmetric eigenproblem.** `jacobi_eigen(S, sweeps)` of Volume I, chapter 6
  (its pseudocode) returns `(values, V)` for a symmetric `S`: the eigenvalues
  on the diagonal at the end (in no particular order, so a caller sorts them
  descending with the columns of `V`) and the unit eigenvectors as the columns
  of `V`. A call states the number of sweeps (`20` is ample for the matrices of
  the book).
- **Newton's method.** Volume I, chapter 7 ("Newton's method: using the tangent
  to jump") derives `x_next = x - f(x) / fprime(x)` for a root of `f`, and
  `x_next = x - g(x) / gprime(x)` for a minimum of a function with first
  derivative `g` and second derivative `gprime`. One iteration is one such
  update; a chapter states the starting value and the stopping rule.
- **Sampling a table.** `sample_discrete(p, rng)` of Volume I, chapter 10 returns the index `k` of a draw from the probability table `p: array<real>[K]`: one `u = rng.uniform()`, then the first `k` whose running sum `p[0] + ... + p[k]`, accumulated in that order, exceeds `u` (the last index if rounding leaves none). It consumes exactly one uniform number.

## Routines defined in Volume III

Two conventions of Volume III are used in pseudocode without being operations of
the language, and are defined where they are taught.

- **Vector-Jacobian product.** `vjp(f, x, u)` of Volume III, chapter 4
  ("Vector-Jacobian products: never build the Jacobian") returns the row
  vector `u J_f(x)` for the operation `f` at the input `x` and an arriving
  gradient `u` with the shape of the output of `f`, without building `J_f`. It
  is what each operation's backward rule computes.
- **Flat parameter vector.** Volume III, chapter 14 (milestone 2, the checks
  of whole models) fixes the order of a model's parameters as one vector: the
  leaf layers in depth-first order, within a layer `W` before `b` (each
  row-major) and `gamma` before `beta`. Checkpoints, optimizer state and the
  gradient checks all use this order.

## Routines defined in Volume IV

- **Exact dot product.** `exact_dot(x, y)` of Volume IV, chapter 10 ("Reference-gated selection") returns the exact value of $\sum_i x_i y_i$ for two vectors of equal length whose entries are floating-point numbers, as a rational number or as a value rounded once to a format wider than the one under test. It is the reference that a candidate's rounded result is compared with, and it is never computed in the format being tested. Two vectors of different lengths are an error (`require`); two empty vectors give 0.

- **Memory transactions and flattened views.** `transactions(log, wg)` of Volume IV, chapter 14 (milestone 2) counts the 32-byte memory transactions of one phase of a workgroup from the per-invocation access logs, by the rule of Volume IV, chapter 5 (subgroups of 32, the j-th access of each member is one instruction, one transaction per distinct 32-byte block, loads and stores apart); `flat(t)` of the same chapter (milestone 1) is the list of the elements of a tensor view in the row-major order of its shape.
- **Selection helpers.** `digest`, `bucket_text`, `order_by_predicted_cost`, `gate_at` and `race` of Volume IV, chapter 10 are defined there, in pseudocode, before `select_kernel` uses them.

## Maps, queues and pieces of text (Volume V)

Volume V's algorithms count pairs, look pieces up and take the smallest of many candidates. Four more types and operations
serve them; each is defined by what it returns, never by how it is stored.

| Operation | Meaning |
|---|---|
| `map<K, V>` | a table from keys of type `K` to values of type `V`. `m[k]` reads the value of a key that is present (an error otherwise), `m[k] = v` sets it, `k in m` is true when the key is present, `len(m)` is the number of keys, and `get(m, k, d)` is `m[k]` when `k in m` and `d` otherwise. A literal empty map is written `map of K to V, empty` |
| `keys(m)` | the keys of `m` as a list in **ascending order**: integers by value, `text` by its characters' code points, and tuples component by component (first components first). A loop over `keys(m)` therefore visits the keys in a fixed order, so a rule such as "the largest count, ties to the first key visited" has one answer |
| tuple order | two tuples of the same shape compare lexicographically: `(a, b) < (c, d)` when `a < c`, or `a == c` and `b < d`. `==` is componentwise |
| tuples as keys | a tuple of integers of any length may be a map key, and so may a pair whose first component is a tuple. `tuple(L)` turns a list into a tuple with the same elements, `t[i:j]` is the slice of a tuple as a tuple (the half-open rule of lists, empty when `i >= j`) and `len(t)` its length. In key order a tuple that is a prefix of another comes first (`(1, 2) < (1, 2, 0)`) |
| `concat(a, b, ...)` | a new `list` holding the elements of `a`, then those of `b`, and so on (two vectors of lengths `m` and `n` give the vector of length `m + n` that holds the first and then the second); for `text` values, the text that joins them in order. A slice of a `list`, `L[i:j]`, is a new list of the elements `i .. j - 1` with the half-open rule of arrays (empty when `i >= j`) |
| `reverse(L)` | a new list with the elements of `L` in the opposite order |
| `s[i:j]`, `len(s)` for `text` | the characters (code points, not bytes) `i .. j - 1` of `s` as a `text`, and the number of characters of `s`; `s[i:j]` is the empty text when `i >= j` |
| `heap<T>` | a priority queue of tuples. `push(q, x)` adds `x`; `pop_min(q)` removes and returns the smallest element in tuple order (an error on an empty queue); `len(q)` is the number of elements |

## Routines of the vision chapters (Volume VII)

The routines of Volume VII's convolution chapters are written out where they are first used and are called by name afterwards.
`out_size`, `conv2d`, `im2col` and `col2im` are those of Chapter 2 (`conv2d(x, w, s, p, d, G)` takes the stride, padding, dilation and
number of groups by position). `maxpool2d(x, k, s)` and `maxpool2d_backward` of Chapter 3 take a window of side `k` every `s`
positions with no padding, and return the winners' rows and columns beside the output (the first maximum in row-major order wins a
tie, as `argmax` does); `residual_block`, `fold_bn_conv` and `fpn_top_down` are defined there, and `upsample2(x)` repeats each value
of a map into a 2 by 2 block, `y[c, i, j] = x[c, i // 2, j // 2]`.

The detection chapter (Chapter 4) writes out `iou`, `decode_v5`, `dfl_decode`, `dfl_loss`, `ciou`, `nms` and `unletterbox` where they are first used. A box is an `array<real>[4]` of corners `(x1, y1, x2, y2)` in pixels, with `x1 < x2` and `y1 < y2` for a non-empty box; its area is `max(x2 - x1, 0.0) * max(y2 - y1, 0.0)`, so a box with a negative side has area 0. `atan2(w, h)` with `h > 0` is the arctangent of `w / h`.

The dense-prediction chapter (Chapter 5) writes out `weighted_pixel_loss`, `pixel_shuffle`, `class_iou` and `align_scale_shift` where they are first used. `logits[:, y, x]` is the vector over the class axis of one pixel of an array of shape `[K, H, W]`; `class_iou` returns `nan` for a class that occurs in neither the truth nor the prediction, and the mean IoU is the `mean` of the entries for which `is_finite` is true.

The vision-transformer chapter (Chapter 6) writes out `patchify`, `shift_labels` and `cubic_taps` where they are first used. `zeros(int, [n, n])` is an integer array of zeros, `for k in -1 ..= 2` visits `-1, 0, 1, 2`, and `k_a(x)` is the cubic convolution kernel of Chapter 1 with parameter `a`: `(a + 2) * x^3 - (a + 3) * x^2 + 1` for `x <= 1` and `a * x^3 - 5 * a * x^2 + 8 * a * x - 4 * a` for `1 < x < 2`, taken at `x = abs(.)`, and `0` beyond 2. A list of tuples is built with `append (i, w) to taps`.

The promptable-segmentation chapter (Chapter 7) writes out `grow`, `fourier_encode`, `attend`, `two_way_layer` and `memory_slots` where they are first used. `layer_norm_rows(X)` is the LayerNorm of Volume III, Chapter 10 applied to every row of `X: [R, D]` with gain 1, offset 0 and epsilon 1e-5, and `softmax_rows` is the routine of the complete example at the end of this page. `W.sa.q` and its relatives are the fields of a record of weight matrices.

The contrastive-learning chapter (Chapter 8) writes out `class_vectors`, `classify`, `sigmoid_loss` and `perceptual_distance` where they are first used. `encode_caption(template, name)` is the text tower of the chapter followed by its projection and the scaling to length 1, the vector of the caption made from a template and a class name; a picture's vector `u` is the output of the image tower scaled to length 1; and `a * dot(...)` is a scalar times a scalar. `F[l][:, r, c]` is the vector over the channel axis at position `(r, c)` of the array `F[l]` of shape `[C, H, W]`. The symmetric loss of that chapter is `infonce` of Volume V, Chapter 14 with the cosines multiplied by a logit scale `a` instead of divided by a temperature.

The vision-language chapter (Chapter 9) writes out `splice`, `smart_resize`, `slice_grid`, `view_size`, `slice_size`, `crop_tiling` and `box_tokens` where they are first used. `round_even` is the integer rounding of ties to the even integer defined above and `smart_resize` calls it on a quotient with `/`; `slice_grid` returns `(0, 0)` for a picture that is not sliced (a grid always has at least one row and one column, so the pair is unambiguous), compares aspect ratios by `log`, and keeps the first grid of the smallest error, a later one only when its error is equal and it has more rows; `view_size` and `slice_size` take `real` sizes and truncate with `floor` (their values are positive, where truncation and `floor` agree); `splice` requires one picture row for every position whose id is the placeholder id, and reads the picture rows in order.

The document-reading chapter (Chapter 10) writes out `row_plan`, `tile_grid`, `prefix_allow`, `ngram_banned` and `edit_distance` where they are first used. A tile grid is a pair `(columns, rows)`; `row_plan` returns a `list` whose entries are `("P", view, token)` for a row of the encoder's output, `"N"` for the learned newline row and `"S"` for the learned separator row, where view 0 is the global view and view `1 + k` the `k`-th tile in row-major order; `tile_grid` returns `none` for a page that fits one tile. `abs` of a difference of two `real` numbers is compared with `<` and `==` exactly, as the arithmetic of the text states. `ngram_banned` takes the tokens generated so far as a `list<int>`; a `list` of `int` is compared with `==` element by element, and `t in L` is true when the list `L` holds an element equal to `t`.

The face chapter (Chapter 11) writes out `fit_similarity`, `decode_level`, `iou_incl` and `margin_logits` where they are first used. A landmark set is an `array<real>[5, 2]` of `(x, y)` pixel-centre coordinates in the order left eye, right eye, nose, left mouth corner, right mouth corner, with left and right taken in the picture. `fit_similarity` returns the least-squares similarity `(a, b, tx, ty)`, the map `(x, y)` to `(a * x - b * y + tx, b * x + a * y + ty)`, and requires at least two points, a source that is not a single point and a cross-covariance of non-negative determinant (a negative one means the best orthogonal part of the fit is a reflection, which no similarity expresses). `iou_incl` is the intersection over union in which a box from `x1` to `x2` counts `x2 - x1 + 1` pixels, and `decode_level` divides every coordinate by `det_scale` before the suppression that follows, which uses `nms` of Chapter 4 with `iou_incl` in place of `iou` and the order of equal scores reversed (among equal scores the higher index comes first). `margin_logits` returns the scaled cosines with the target's replaced by `s * cos(theta + m)` computed as `c * cos(m) - sn * sin(m)`, where `sn` is `sqrt(1 - c^2)` floored at `1.0e-6`; the loss is `lse(z) - z[y]`.

The vision project (Chapter 12) writes out `detector_view`, `segmenter_view`, `head_cells`, `detect_frame`, `admits`, `segment_box`, `read_memory`, `track_step` and `run_pipeline` where they are first used. Four helpers are defined by what they return: `cell_features(view)` is the `array<real>[256, 3]` of the means of the normalized colours of each `4 x 4` block of a `64 x 64` view (blocks in row-major order, the sum of a block taken row by row), `mask_cells(mask)` the `array<real>[256]` of the fraction of each block that a mask covers, `mean_colour(view, mask)` the mean colour over the pixels of a mask summed in row-major order, and `dilate3(on)` the `3 x 3` maximum of a `16 x 16` map of 0 and 1 clipped at the border. An inline choice is written `a if c else b`, `for (a, b) in L` takes the pair elements of a list of pairs, and `floor(x + 0.5)` is the rounding of the letterbox sizes (round half up, for `x >= 0`).

## Routines of the audio chapters (Volume IX)

The sound chapter (Chapter 1) writes out `apparent_frequency`, `sinc`, `bessel_i0`, `triangle_weight`, `kaiser_weight`, `cos2_weight` and `resample` where they are first used. `sinc(0)` is 1 (the limit), and `apparent_frequency` returns the pair `(frequency, sign)` with `r = fs / 2` counted as the lower branch (sign 1). In `resample` the input is zero outside `0 .. n - 1`, a weight is read only for a whole number `m` with `abs(m) <= reach`, the sum for one output runs over increasing `i`, and `(a + b - 1) // b` is the ceiling of `a / b` for a whole number `a` of either sign and `b > 0`, because `//` rounds toward minus infinity. The 16-bit write of the chapter is `round(clamp(x, -1.0, 1.0) * 32767.0, 0)` with ties away from zero, and the read is the code divided by `32768`.

The spectrogram chapter (Chapter 2) writes out `reflect_pad`, `stft_power`, `hz_to_mel`, `mel_to_hz`, `mel_filterbank` and `log_mel_spectrogram` where they are first used. `reflect_pad` repeats the signal about its first and last samples without repeating those samples (`out[p - 1 - i] = x[i + 1]`), and requires a pad smaller than the length. `stft_power` takes an already padded signal and a window of the frame length, produces `1 + (P - N) // H` frames, reduces `k * n` modulo `N` inside the exponent, and sums `n` upward; the bins are `0 ..= N // 2`. `hz_to_mel` and `mel_to_hz` are the HTK scale when `slaney` is false and the Slaney scale (linear below 1000 Hz, logarithmic above) when it is true. `mel_filterbank` places `M + 2` edges uniformly in mel between `f_min` and `f_max`, makes each triangle the lesser of its rising and falling lines clipped at zero, and with `normalise` scales it by `2 / (upper edge - lower edge)`. `log_mel_spectrogram` sums each mel energy over `k` upward and applies `compress` to it; the output is frames first.

The codec chapter (Chapter 3) writes out `residual_encode` and `residual_decode` where they are first used. `residual_encode` takes a vector `z: [d]` and a list of codebooks `books`, each of shape `[K, d]`, and returns one index per codebook: at each stage `argmin` of the squared distances (the lower index wins a tie), after which the chosen row is subtracted from the residual; `residual_decode` adds the chosen rows. The output length of a causal convolution of stride `s` on `L` samples is `(L + s - 1) // s`, and the book's trained residual quantizer is stage-wise Lloyd: starting centres are the first `K` rows of the residuals in the order of one `shuffle`, each round assigns every row to its nearest centre and replaces each centre that has members by their mean (a centre with none stays where it is).

The speech-recognition chapter (Chapter 4) writes out `ctc_collapse`, `ctc_forward`, `rnnt_forward` and `rnnt_greedy` where they are first used. Symbols are non-negative `int` ids, the blank is one of them, and `-1` is a value that no symbol takes. `ctc_forward` requires `T >= 1` and a non-empty label list, takes the table `P: [T, K]` of per-row probabilities and returns the probability of the label list; it is written in probabilities, and an implementation that trains computes the same recursion in logarithms, adding two log-probabilities with `lse` of the pair, where `lse([-infinity, -infinity])` is `-infinity`. `ctc_prefix_beam` is written out in the chapter, with `pre + (k,)` the tuple `pre` followed by `k`, `append (x, y) to L` adding a pair to a list, and `sort` of a list of pairs ordering by the first component and then the second (tuple order). `rnnt_forward` takes the probabilities `qb` of the blank and `qy` of the next label at every node of a lattice of `T` rows and `W = U + 1` label counts (the column `qy[:, U]` is never read) and returns the probability of the label list that has `U` labels. `rnnt_greedy` takes the encoder rows, a predictor `predictor_step(state, token)` that returns the new state and the predictor vector, a `joint(e, g)` that returns the probabilities of the symbols and the blank, and `max_symbols`; `argmax` takes the first index on a tie.

The speaker-embedding chapter (Chapter 5) writes out `time_delay_layer`, `weighted_stats`, `attentive_pool` and `equal_error_rate` where they are first used. `time_delay_layer` takes a kernel of odd size `k` and a dilation `d`, pads each channel by `d * (k - 1) // 2` frames on each side with `reflect_pad` of Chapter 2 (so it requires that pad to be smaller than `T`) and reads tap `m` of output `t` at the padded position `t + d * m`. `weighted_stats` takes time weights that need not be uniform and returns the weighted mean and `sqrt(max(variance, eps))` of every channel, the variance being the weighted mean of the squared deviations from the weighted mean (no divisor `T - 1`); a caller that wants the unbiased deviation of a uniform weighting says so. `attentive_pool` stacks the channels, their uniform means and their uniform deviations into `3 * C` rows, scores every frame of every channel through a kernel-1 layer, `relu`, `tanh` and a second kernel-1 layer, applies `softmax` over the frames of each channel and returns the `2 * C` weighted means followed by the weighted deviations. `equal_error_rate` takes the cosines of genuine and impostor pairs, tries every observed cosine as a threshold (a pair is accepted at or above it), and returns the threshold where the false rejection rate and the false acceptance rate are closest, the lowest threshold on a tie, with the mean of the two rates there.

The speech-synthesis chapter (Chapter 6) writes out `chain_draw`, `fsq_token` and `cfg_euler` where they are first used. `round_even(x)` is the nearest integer to `x`, a tie going to the even integer (`round_even(0.5)` is 0, `round_even(1.5)` is 2, `round_even(-0.5)` is 0), returned as an `int`; it is not `round(x, 0)`, whose ties go away from zero. `chain_draw` takes the logits of a vocabulary of `V` ids, the list of ids generated so far and one uniform `u` in `[0, 1)`, and applies in this order a mask of the last `window` ids (the end-of-speech id restored when `allow_eos`), a repetition penalty applied once for each distinct id of the history (a positive logit divided by `theta`, a negative one multiplied), the division by the temperature `tau`, top-k (entries strictly below the `k`-th largest are removed, so a tie at the threshold stays), top-p (the shortest prefix of the ranked probabilities whose sum reaches `p`) and an inverse-CDF draw that skips ids of probability 0. `fsq_token` returns the base-3 numeral of the eight digits `round_even(tanh(z[i]) * 0.999) + 1`, the first coordinate the least significant. `cfg_euler` takes two velocity functions, a grid from 0 (noise) to 1 (data) and the guidance scale `g`, and at each step moves `x` by the step times `un + g * (uc - un)`.

The music chapter (Chapter 7) writes out `hold_euler` where it is first used. It is `cfg_euler` on the uniform grid $t_i = i/n$ (so $n$ steps of $1/n$, the network evaluated at $t = 0, 1/n, \ldots, (n-1)/n$ and never at 1) with one addition: when a carried prefix `prev` of `P` steps is present, then before every evaluation the first `P` steps of `x` are set to `(1 - (1 - 0.000001) * t) * noise[:, 0:P] + t * prev`, the noise being the chunk's own starting noise, and after the last step they are set to `prev` exactly. `prev` is `optional`: `prev is present` is false for the first chunk of a song.

The omni-model chapter (Chapter 8) writes out `clip_positions` and `wave_length` where they are first used. `clip_positions` returns the list of position triples `(T, H, W)` of the rows of a clip in the order of the sequence (the picture rows of each group first, then the sound rows, chunk by chunk) and the counter after the clip, one more than the largest entry of any triple (or of `p0`, for an empty clip); it requires that the number of seconds is a multiple of the chunk length. `wave_length` composes `n = n * u` for each entry of `ups` and then `n = (n - 1) * r` for each entry of `rates`, in the order given, and requires `n >= 1` before every `(n - 1) * r`.

The project chapter of the volume (Chapter 9) writes out `synthesise`, `ctc_scaled`, `train_recogniser` and `voice_map` where they are first used, and uses `fft`, `hann_periodic` and `ctc_loss_and_gradient` without writing them out. `fft(x)` is the discrete Fourier transform $X[k] = \sum_n x[n]\,e^{-2\pi i k n / N}$ of a power-of-two length $N$, by radix-2 butterflies, all $N$ complex bins (the project reads bins $0 \ldots N/2$). `hann_periodic(N)` is the array $0.5 - 0.5\cos(2\pi n / N)$ for $n = 0 \ldots N - 1$. `ctc_loss_and_gradient(theta, sentence, g)` runs the forward pass of the network of milestone 3 on the sentence's stacked frames, calls `ctc_scaled` on the softmax rows and the label, adds the hand-written gradient of the loss with respect to `theta` (the recipe printed in that milestone) into `g`, and returns the loss. `synthesise` takes a text over the letters and the space, a speaker parameter and a `Generator`, and draws in the order its text states (one `integer(3)` for every symbol, one `uniform()`, then one `normal()` for every sample). `ctc_scaled` is `ctc_forward` of Chapter 4 with the table of every frame divided by its own sum, the logarithms of those sums added back into the loss, and it returns the loss and the soft alignment; a label equal to the blank is not rejected, and the recursion then treats that state as a blank (which is what the mistake of milestone 7 shows). `train_recogniser` is the minibatch loop of Volume III with the AdamW step and a `shuffle` of one persisting list of sentence indices at the start of every epoch (in place, so each epoch permutes the previous order), the gradient the mean over the batch. `voice_map` is ridge regression: the normal equations `(E^T E + lambda I) w = E^T y` over centred columns, solved by the `solve` of Volume I, chapter 4 with partial pivoting.

## Routines of the video chapters (Volume X)

The video chapter (Chapter 1) writes out `frame_start_ms` and `best_vector` where they are first used. `//` rounds toward minus infinity, and `frame_start_ms` takes it twice, first to whole nanoseconds and then to whole milliseconds. In `best_vector` the list `C` is the 25 pairs `(dx, dy)` with `dx` and `dy` in `-2 .. 2`, ordered by increasing `abs(dx) + abs(dy)`, then by increasing `dy`, then by increasing `dx`, the loop visits it in that order and only a strictly smaller sum replaces the best so far, and a pixel read outside the frame takes the nearest pixel inside it.

The video generation chapter (Chapter 2) writes out `denoise_clip` and `window_plan` where they are first used. A scalar times a row of an array (`mask[n] * x[n, :]`) scales every element of the row, and a sum of two rows of equal shape is elementwise. In `window_plan` the result is a list of triples (carried latent frames, new latent frames, the output pixel frame at which the window's first emitted frame goes); `ceil` of a quotient is the smallest integer not below it, and `min` and `max` of integers are integers.

The camera-geometry chapter (Chapter 3) writes out `look_at`, `rotation_vector_of`, `undistort`, `hartley`, `homography` and `planar_intrinsics` where they are first used. `cross(a, b)` of two vectors of length 3 is `[a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]`, and `unit(v)` is `v / norm(v)` for a nonzero vector. A scalar times a vector, a vector divided by a scalar and `-v` act on every element. `null_vector(rows)` takes a list of rows of equal length `n` (as vectors) and returns the unit vector `v` of length `n` that makes `norm(matmul(A, v))` smallest, with `A` the matrix of the rows: the right singular vector of the smallest singular value of Volume I, Chapter 6 (a matrix with fewer rows than columns is padded with zero rows to `n` rows, which changes no singular vector); its sign is not fixed by the definition, and a caller that needs it fixed says how. `inverse(A)` is the matrix inverse of Volume I, Chapter 4, an error for a singular `A`, and `solve(A, y, true)` is Gaussian elimination with partial pivoting as above. `distort` and `jacobian` are equations (6) and (7) of the chapter and `v_row(H, i, j)` is defined in its calibration section. `0 .. 3` in `for j in 0 .. 3` visits 0, 1, 2.

The structure-from-motion chapter (Chapter 4) writes out `eight_point`, `ransac_iterations`, `ransac_essential`, `levenberg_marquardt`, `schur_step` and `patchmatch_half_step` where they are first used. A list of equal-length lists or arrays is a matrix wherever a matrix is expected (`hartley`, `matmul`, `transpose`, `null_vector`) and a list of arrays is indexed like one, `a[i]` being the $i$-th array. `svd(M)` is the thin singular value decomposition of Volume I, Chapter 6, returning `(U, sigma, V)` with `M = U diag(sigma) V^T`, the columns of `U` and `V` orthonormal and `sigma` descending; the sign of a singular vector pair is not fixed by the definition, and a caller that needs it fixed says how. `zeros(real, [2, 2])` is a $2 \times 2$ array of zeros, `M[:, 0:2]` the first two columns of `M`, and `-g` the negated vector. `sampson(M, a, b)` is equation (3) of the chapter on the normalized points `a` and `b`. `rand_dir(rng)` draws $z = 2u - 1$ and $\varphi = 2\pi u'$ with two successive `rng.uniform()` calls and returns $(\sqrt{1 - z^2}\cos\varphi, \sqrt{1 - z^2}\sin\varphi, z)$; `face(n, d)` returns `-n` if `dot(n, d) > 0` and `n` otherwise. `inverse(A)` and `solve(A, y, true)` are those of Volume I, Chapter 4. In `patchmatch_half_step` a candidate is scored by the function passed as `cost`, `mod` is the pseudocode's remainder, and `r`, `n` and `c` are changed in place: a pixel reads only the neighbours of the other colour, which the same half-step does not write.

The learning-3D chapter (Chapter 5) writes out `align_similarity` and `fuse_pixel` where they are first used. `nearest_rotation(S)` of a $3 \times 3$ matrix `S` is the rotation matrix nearest to it in the Frobenius norm: with `(U, sigma, V) = svd(S)` it is `matmul(U, transpose(V))` when that has determinant $+1$, and otherwise `matmul(U, matmul(diag, transpose(V)))` with `diag` the diagonal matrix $(1, 1, -1)$ (a reflection is never returned); `trace(A)` is the sum of the diagonal of a square matrix and `mean(A, axis=0)` of an `[n, 3]` array is the vector of the column means. In `fuse_pixel`, `project(cam, X)` returns `(u, v, z)`, the continuous pixel coordinates and the camera depth of the world point `X` as in Chapter 3, `centre(cam)` is the camera centre $\mathbf{C}$, `world_ray(cam, u, v)` is $\mathbf{R}^\top((u - c_x)/f_x,\ (v - c_y)/f_y,\ 1)$ (not normalized), a vector sum or difference acts on the three components, `floor` of a non-negative `real` is an `int` used as an index, and `depth[j][ky, kx]` is the depth map of view `j` at row `ky` and column `kx`. The result is the fused depth of one pixel and the number of other views that agreed; the maps in `depth` are never written, so that every pixel of every view is fused from the same unfused maps.

The splatting chapter (Chapter 6) writes out `project_gaussian`, `build_tile_lists` and `composite_pixel` where they are first used. A scalar times a matrix and a sum of two matrices of the same shape are elementwise, `identity(n)` is the $n \times n$ identity matrix and `matmul(A, B)` of a matrix and a transposed matrix follows the rules of Volume I. `clamp_int(x, lo, hi)` is `min(max(x, lo), hi)` of integers; `arange(M)` is the list `[0, 1, ..., M - 1]`; `int(a)` of a `u32` is the integer of the same value; `u32(n)` of a non-negative integer below $2^{32}$ is as in the fixed-width section; and `bits(z)` of a depth stored as `f32` is its 32-bit pattern. `exclusive_scan(a)` of an integer vector is the vector whose entry $i$ is the sum of the entries before $i$ (so its first entry is 0; Volume IV, Chapter 7), and `radix_sort_pairs(keys, vals)` of a `u32` vector and a vector of equal length is the pair (sorted keys, the values permuted with them) produced by the stable least-significant-digit radix sort of Volume IV, Chapter 7 with four passes of 8-bit digits: keys that are equal keep the order of their positions in the input. The routine `project_gaussian` returns `false` and zeros for a Gaussian that is culled; in `composite_pixel` the colours and the background are real vectors, and `continue` and `break` act on the loop over the list.

The simulation chapter (Chapter 8) writes out `gather_tick`, `neural_tick` and `control_tick` where they are first used. A state that holds several arrays of the same length is written as a `map<text, array<real>[n]>` whose arrays are read and written by name (`S["v"][i]`); a count of ticks left is stored in a `real` array and compared with 0. `inout` marks the arrays a routine updates in place.

The dynamic-scene project (Chapter 9) writes out `floor_plane`, `ray_on_plane`, `fit_scene`, `fit_ar2` and `rollout` where they are first used. `for p in points` visits the elements of a list in order, `append k to on` adds at the end, and `len` is the length of a list; `-matmul(transpose(R), t)` is the negated matrix-vector product, a scalar times a vector acts on every element, and `Ga[j][2:5]` is the slice of three entries from index 2. `solve(A, y, true)` is the elimination of Volume I, Chapter 4 as above. In `fit_scene` the routines `view_gradient` and `mse_upstream` are those of Chapter 6, a camera is the record of Chapter 6's renderer, and "adam update" is the step of Volume III, Chapter 7 with the rates the chapter states; "zeros of the shapes of" is an array or list of zeros with those shapes.

## Routines of the forecasting chapters (Volume XI)

The time-series chapter (Chapter 1) writes out `autocorrelation`, `fit_ar`, `seasonal_ar_forecast`, `temporal_split` and `window_scale` where they are first used. `mean(x)` and `sum(... for i in 0 .. c)` are those of the arrays section above, `ys[0:n] = y` assigns a slice and `ys[n:n + H]` reads one, and `c / c[0]` divides every element of the array `c` by its first. `solve(A, b, true)` is the elimination of Volume I, chapter 4. In `fit_ar` the entries of `A` and `b` are accumulated row by row in increasing `t`, which fixes the order of every sum. In `seasonal_ar_forecast` the array `z` holds the seasonal differences of the past in its first `n - m` cells, and each step writes one forecast difference in the next cell and one forecast value in `ys`. `floor` of a product in `temporal_split` is the largest integer not above it, so `floor(120 * 0.7)` is 84 (single or double precision give the same value for the sizes of the chapter); the three lists it returns hold origins in increasing order, and a window in none of the three bands is not returned.

The probabilistic-forecast chapter (Chapter 2) writes out `pinball`, `crps_from_levels`, `fit_head`, `fit_quantile_line` and `conformal_halfwidth` where they are first used. `fit_head` takes `nu = infinity` for the Gaussian head, where the weight of every row is exactly 1.0; its sums over rows run in increasing `i`, the weights of a round are computed from the parameters at the start of the round and also weight the residuals of the variance at its end, and `n` is the number of rows. `fit_quantile_line` tries the pairs `(i, j)` with `i < j` in lexicographic order and keeps the first pair whose total pinball loss is strictly smaller than every earlier one. In `conformal_halfwidth` the product `(n + 1) * (1.0 - alpha)` is formed on exact decimals (for `alpha = 0.1` and `n = 19` it is exactly 18), because a binary product such as `20 * 0.9` can come out a rounding step above its integer value and `ceil` would then return one too many; `sort(scores)[k - 1]` is the `k`-th smallest score counting from 1.

Chapter 2 also needs three reference routines that a C math library does not have. `Phi(z)` is `0.5 * (1.0 + erf(z / sqrt(2.0)))`. `Phi_inv(p)` for `p` in `(0, 1)` is found by bisection on `Phi`, and agrees with a library inverse to better than 1e-11 for `p` between 1e-6 and `1 - 1e-6`. `t_cdf(x, nu)` is the distribution function of the standard Student-t for an even number `nu` of degrees of freedom, a finite sum, and `t_quantile(p, nu)` is the same bisection on `t_cdf`.

```text
function Phi_inv(p: real) -> real:
    require 0.0 < p and p < 1.0, "a probability strictly inside (0, 1)"
    lo = -40.0
    hi = 40.0
    for i in 0 .. 200:                       # 200 halvings of 80 reach the spacing of doubles
        mid = 0.5 * (lo + hi)
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

function t_cdf(x: real, nu: int) -> real:
    require nu >= 2 and nu mod 2 == 0, "an even number of degrees of freedom"
    u = nu / (nu + x ^ 2)
    total = 0.0
    term = 1.0                               # term k is (1 * 3 * ... * (2k - 1)) / (2 * 4 * ... * 2k) * u^k
    for k in 0 .. nu // 2:
        total = total + term
        term = term * (2 * k + 1) / (2 * k + 2) * u
    return 0.5 + x / (2.0 * sqrt(nu + x ^ 2)) * total

function t_quantile(p: real, nu: int) -> real:
    require 0.0 < p and p < 1.0, "a probability strictly inside (0, 1)"
    lo = -1.0e6
    hi = 1.0e6
    for i in 0 .. 200:
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
```

The evaluation chapter (Chapter 4) writes out `walk_forward`, `mase`, `skill`, `bootstrap_skill` and `dm_statistic` where they are first used. `walk_forward` takes a forecaster as a parameter of function type and gives it `y[0:o]` and the horizon, so a forecaster never sees a bar at or after its origin; `floor` in the number of origins is the largest integer not above the quotient. In `bootstrap_skill` one resample draws `ceil(K / blk)` starting positions, each one call `rng.integer(K - blk + 1)`, in order, lays the blocks end to end and keeps the first `K` indices; `rows(E_model, idx)` gathers the elements of a vector at the indices, in order and with repeats; the generator is shared by every call of a table, in the table's order. `dm_statistic` forms every autocovariance with the mean of all `K` values removed and the divisor `K`, weights lag `k` by `1 - k / (lag + 1)` when `bartlett` is true and by 1 otherwise, and refuses (`require`) a long-run variance that is not positive.

The survival chapter (Chapter 5) writes out `kaplan_meier`, `aalen_johansen`, `person_period`, `concordance`, `ipcw_brier` and `incidence_at` where they are first used. `distinct(a)` is the increasing list of the distinct values of the vector `a`, each once (`distinct([3.0, 1.0, 3.0, 2.0])` is `[1.0, 2.0, 3.0]`), and `u[r]` reads its element `r`. `len([i for i in 0 .. n if time[i] >= u[r]])` counts the subjects at risk at `u[r]`: a subject whose time equals `u[r]` is at risk there, whether it fails or is censored (failures happen before censorings at a tie). In `kaplan_meier` a time with no failure leaves `S` and the Greenwood sum `W` as they were, and a time at which every subject at risk fails (`n_r == d_r`) sets `S` to zero and leaves `W` unchanged. `concordance` called with `tstar = infinity` and all weights 1.0 is Harrell's index; with `tstar` finite and `weight[i] = 1.0 / G(time[i])^2` it is Uno's, where `G` is the step function of `kaplan_meier(time, 1 - event)`, read as the value at the largest `u[r]` not above its argument and 1.0 before the first; `1 - event` is the array with 1 where `event` is 0. In `person_period` the loop `for j in 1 .. last[i] + 1` visits `j = 1, ..., last[i]`. In `incidence_at` the piece `p` is the interval from `knots[p]` to `knots[p + 1]`, and a time past `knots[P]` is held at its value there because `dt` is then the full piece length for every piece.

The scoring chapter (Chapter 6) writes out `pav`, `isotonic_at` and `risk_coverage` where they are first used. In `pav` the examples are visited in the order of `argsort(score)`, so that equal scores keep their input order; the blocks are the first `m` entries of `sums` and `counts`, two neighbouring blocks are merged while the mean label of the earlier is strictly larger than that of the later, and `sums[m] = y[i]` stores the label as a `real`. `isotonic_at` returns the fitted value at the largest position `r` with `xs[r] <= z`, and at position 0 when there is none. In `risk_coverage` the examples are visited from the largest confidence to the smallest, equal confidences in input order, and `risk[k]` is the number of errors among the first `k + 1` examples divided by `k + 1`.

The forecasting project (Chapter 7) writes out `draw_series`, `draw_events`, `scaled_window`, `fit_map`, `recal_multipliers`, `scaled_quantile_loss`, `order_cost` and `alert_cost` where they are first used, and calls `window_scale`, `temporal_split`, `pinball`, `bootstrap_skill`, `pav` and `isotonic_at` of the earlier chapters of the volume. `Phi(z)` is the standard Gaussian distribution function of the scalar operations above and `Phi_inv(p)` its inverse for `p` in `(0, 1)`, the library routine of the language (any routine accurate to about 1e-12 reproduces the printed digits). In `draw_series` and `draw_events` each `rng.uniform()`, `rng.integer(200)` and `rng.normal()` is one call in the order written, the generator is shared by the 40 series in turn, and `5.0 ^ u` is the power of the scalar operations. In `window_scale` the argument may have any length `c >= 1`; it returns the mean and the population standard deviation (divisor `c`) plus `eps`. In `fit_map` the loops visit the series in order and then the origins in increasing order, the upper triangle of `A` is accumulated row by row and mirrored before the penalty is added to every diagonal entry but the first, and `solve(A, b, true)` is the elimination of Volume I, chapter 4 with partial pivoting. In `recal_multipliers`, `sort(u)` ascending, `ext[r]` is the `r`-th smallest value counting from 1 and `ext[0] = 0.0`, `ext[n + 1] = 1.0`; `floor(pos)` is at most `n` for a level below 1. In `alert_cost` a probability equal to the threshold alerts.

## Routines of the reinforcement-learning chapters (Volume XII)

The Markov decision process chapter (Chapter 1) writes out `backup`, `evaluate_policy`, `evaluate_sweeps`, `greedy_policy`, `value_iteration` and `policy_iteration` where they are first used. A problem is given by `P: array<real>[n, m, n]`, `R: array<real>[n, m]` and `n_act: list<int>`: the legal actions of state `s` are `0 .. n_act[s] - 1`, and entries of `P` and `R` for an action beyond them are never read. A policy is a `list<int>` of `n` actions, one per state. `max(q)` of a list is its largest element, `abs` is the scalar absolute value and `identity(n)` is the real identity matrix. Two lists are equal when they have the same length and equal elements in order, so `new == pi` compares policies. `solve(A, r, true)` is the elimination of Volume I, chapter 4 with partial pivoting. In `backup` the sum over `j` is accumulated in increasing `j` from `0.0`, and every state of a sweep is updated from the vector of the previous sweep, never from entries already updated in the same sweep. In `greedy_policy`, `best` is the largest lookahead of the state and the action kept is the smallest `a` whose lookahead is at least `best - tie`, so that equal actions are decided in favour of the smallest index. `loop:` is repeated until a `return` inside it.

The value-learning chapter (Chapter 2) writes out `mc_update`, `td0_update`, `td0_episode`, `greedy_action`, `choose_action`, `td_control`, `double_q_update`, `bias_choose`, `bias_episode` and `linear_td_update` where they are first used (`bias_update`, the one update both learners of the small problem share, is described beside them in the chapter). An episode is a `list<(int, real, int)>` of days, each the triple `(s, r, next)` of the state, the reward and the next state; `sample_day(s, a, rng)` returns the pair `(r, next)` of one day of the shop of Chapter 1, drawing exactly one `rng.uniform()` for the demand (demand 0 if it is below 0.3, 1 if it is below 0.7, otherwise 2). A table of values per state and action is a `list<list<real>>` whose row `s` has `n_act[s]` entries, and `step` is a `function(int) -> real` from the number of updates so far, counted from 1, to the step size. `dot(a, b)` accumulates the products in increasing index from `0.0`, `zeros(int, [n])` is an array of integer zeros, `0 if c else 1` is the conditional expression of the scalar operations, and `x is present` is the test of an `optional` value. `greedy_action` returns the smallest index whose value is at least the largest minus `tie`. In `choose_action` one `rng.uniform()` is drawn for every decision, and one `rng.integer(n_act[s])` more only when it falls below the exploration rate. In `td_control` the next action is chosen before the table is updated with the day, and the update is applied before the next demand is drawn; `count[s][a]` includes the current update. In `bias_choose` the two tables are added entry by entry (`T[0][s] + T[1][s]`, a list of the same length) before the greedy choice, and `rng.normal()` is the Gaussian draw of the generator.

The policy-gradient chapter (Chapter 3) writes out `sample_index`, `reinforce_episode`, `actor_critic_episode`, `gae` and `ppo_epochs` where they are first used. A tabular softmax policy is a `list<list<real>>` `theta` whose row `s` holds the preferences of the legal actions of state `s`, and `softmax(theta[s])` is the softmax above applied to that list; `zeros_like(theta)` is a list of lists of real zeros with the lengths of the rows of `theta`, and arithmetic on arrays such as `theta + eta * grad` is elementwise. `sample_index(p, rng)` draws one `rng.uniform()` and returns the first index whose running sum of `p` (accumulated in increasing index from `0.0`) exceeds it, the last index when rounding leaves a gap. An episode of `reinforce_episode` is a `list<(int, int, real, int)>` of days `(s, a, r, next)`. In all of them `gamma^t` is a running product `disc`, multiplied by `gamma` once per day, and the order of the multiplications in an update is the order written. `V[s]` is the critic of Chapter 2. In `ppo_epochs` the clip is applied to the ratio and the comparison `raw <= clipped` decides which term of the minimum is used, the raw term on a tie.

The language-model alignment chapter (Chapter 4) writes out `masked_mean_loss`, `reward_model_epoch`, `rlhf_iteration`, `dpo_loss`, `group_advantages` and `grpo_step` where they are first used. In `masked_mean_loss` the sum runs over the positions in increasing `t`, `lse(logits[t]) - logits[t][targets[t]]` is the cross-entropy of one position and the divisor is `max(count, 1)`, so that a batch with no supervised position has the loss `0.0`. A policy over the answers of one prompt is the `softmax` of a `theta: array<real>[m]` and `pi_ref` is the probability vector of the frozen reference. In `reward_model_epoch`, `chosen` and `rejected` list the answer indices of the pairs in order, and the gradient is accumulated in pair order and divided by the number of pairs. In `rlhf_iteration` the answers of the batch are drawn first, one `sample_index` each in batch order, from the policy `pi_old` that stays fixed for the iteration, the batch mean is summed in batch order and the clip is the one of `ppo_epochs`; with `in_loss` false the penalty `beta * log(pi_old[y] / pi_ref[y])` is subtracted from the reward, with `in_loss` true the weight `beta * (pi_ref[y] / pi[y] - 1.0)` is added to the weight of every sample, clipped or not. `group_advantages` uses the population standard deviation (the divisor is the group size, never one less), adds `1e-4` to it for the z-score, returns `1.0` for every member of a group of one and `0.0` for every member of a group whose standard deviation is below `1e-6`. `grpo_step` draws the group with one `sample_index` per member and divides by the group size.

The search chapter (Chapter 5) writes out `best_first`, `uct_search`, `puct_search`, `expert_round`, `map_elites`, `relabel_future`, `goal_q_sweeps` and `retrace_batch` where they are first used. A cell of a map is a pair `(row, column)` and may be a key of a `map` or a component of a `heap` entry; entries of a `heap` compare as tuples, so the second component of `best_first`'s entries, a counter that grows with every push, decides between equal priorities in favour of the earlier push and the components after it are never compared. `heap of (real, int, (int, int), real), empty` is an empty queue of such entries, `get(m, k, d)` and `k in m` are the operations of the map type, `keys(archive)` lists the cells of a map in ascending tuple order, and `x if c else y` is the conditional expression of the scalar operations. In `uct_search` and `puct_search` the nodes of a tree are numbered in the order they are created, node 0 being the root, and the arrays `pile`, `depth`, `parent`, `N`, `W`, `child` (and `P`) are indexed by node number; `parent[0]` is -1, and the move `m` of a node with `p` stones left is one of `1 ..= min(3, p)`. Both break a tie between children in favour of the smaller move, which the loops over `m = 1, 2, 3` in increasing order and the strict comparison `score > best_score` give. `log` is the natural logarithm, `softmax(theta[p][0 : k])` is the softmax of the first `k` entries of a row of a table of preferences (the slice of a list is half-open) and `sample_index(p, rng)` is the routine of Chapter 3. `visits / sum(visits)` divides every element of a list by a number and `softmax(...) - target[n]` is the elementwise difference of two lists of equal length. A list comprehension may run over an inclusive range, `[expression for m in 1 ..= k]`, which visits `m = 1, ..., k`. Text joined with `+` is the concatenation of the pieces, a number being written in decimal, and `Generator(seed_of(text))` is the generator of the worked example of that name. In `map_elites`, `fitness(genome)` and `cell_of(genome, k)` are the formulas of the chapter, `cell_of` returning the pair of cell indices `min(floor(k * d), k - 1)` of the two coordinates of the descriptor `d`; the normal draws of the mutation are consecutive calls of `rng.normal()`, genes in order. `retrace_batch` begins every walk at the goal `[0, 1, 2, 3, 4]` and redraws a move that equals the move just played. Arrays are references, so `map_elites` makes a new `genome` for every offer and `retrace_batch` stores `copy(s)`, not `s`, which the walk goes on swapping.

The decision-model chapter (Chapter 6) writes out `encode_state`, `train_head`, `fit_temperature`, `decide_action` and `ipw_value` where they are first used. A request is a `list<text>` of words and an `array<real>[4]` of fields, `vec` is a `map` from a word to its vector, and `norm` is the Euclidean length. In `encode_state` the word vectors are added in the order of the list, coordinate by coordinate, from `0.0`, and divided by the number of words before the length is taken. In `train_head` the logit of a route is the sum of the products of its weights and the inputs, accumulated in increasing input index from `0.0`, plus the bias; the gradients of a step are accumulated over the requests in order and divided by `n` at the update; `softmax(z)` is the one above and `1.0 if k == y[i] else 0.0` is the conditional expression of the scalar operations. `fit_temperature` calls `slope(a, Z, y)`, the mean over the requests, in order, of `sum(softmax(a * Z[i])[k] * Z[i, k] for k in 0 .. 4) - Z[i, y[i]]`, and returns the temperature `T = 1 / a`. `decide_action` returns the index of the route or `-1`; the comparison is not strict, so a probability exactly at the threshold routes, and `argmax` takes the first index on a tie. In `ipw_value` the sums run over the requests in order, a request contributes only when the new policy's route equals the logged route, and `n` is the number of requests of the log.

The continual-learning chapter (Chapter 7) writes out `reservoir_fill`, `anchored_step`, `gate_candidate` and `read_episode` where they are first used. `P` is the number of numbers of the chapter's network (67), `t` and `anchor` are arrays of `P` reals, `loss_gradient(t, rows)` is the gradient of the mean binary cross-entropy over the rows, the rows' gradients added in row order and then divided by the number of rows, and a row is a pair `(x, y)` of an `array<real>[2]` and a class `0` or `1`. In `reservoir_fill` the first `min(cap, n_items)` indices fill the buffer, one `rng.integer(i + 1)` is drawn for every later `i`, and the slot `j` it returns replaces the item only when `j < cap`. In `anchored_step` every new entry is computed from the old `t`, not from entries already updated. In `gate_candidate` a pair of outcomes is a win when `own_c[k] > own_i[k]`, a loss when `own_c[k] < own_i[k]` and otherwise a tie, which no count uses; `comb(n, k)` is the binomial coefficient, `mean` and `max` of a list are its mean and its largest element, a block is a pair `(candidate mean, incumbent mean)`, and with no blocks the anchor drop and the worst block drop are `0.0`. `read_episode` takes `ep` as a record with the fields `train` (rows) and `probes` (rows), `bank` as the list of promoted episodes' numbers in promotion order, and returns the network that is served afterwards and the outcome as text; `mean_loss(t, rows)` is the mean cross-entropy, `correct(t, rows)` the list of 1.0 or 0.0 for each row (the logit above 0 is class 1, a logit of exactly 0 class 0), `gd(t, rows, eta, steps)` is `steps` applications of the plain gradient step `t = t - eta * loss_gradient(t, rows)`, and `replay_rows(pools, m, rng)` concatenates the lists of rows of `pools` in order and draws `m` of them without replacement by a partial Fisher-Yates (for `i` in `0 .. m`, `j = i + rng.integer(len - i)` and swap the entries `i` and `j`; the first `m` entries are the draw; a pool of exactly `m` is still shuffled, and a pool is never smaller than `m` in the chapter), and `a + b` on two lists is their concatenation.

The project chapter (Chapter 8) writes out `digits`, `target_digits`, `verify_exact`, `verify_weak`, `position_probs`, `success`, `bank_success`, `train_step`, `train_agent`, `paired_lower`, `first_regressed`, `gate` and `serve` where they are first used. An answer is an `array<int>[ell]` of the first `ell` output digits, `level` is `ell` (`verify_exact` and `verify_weak` take it as an argument and score 0.0 for an answer of any other length), and `theta` is an `array<real>[240]` that every routine leaves unchanged (`train_step` works on `copy(theta)`). A pool, a bank of probe inputs and a bank of audit inputs are lists of the numbers `m` of the inputs; `bank_scores(theta, inputs)` is the list over the levels 1 to 5 of the arrays `[success(theta, digits(m), level) for m in inputs]`, in the order of `inputs`, so `bank_scores(...)[level - 1][k]` is the score of probe `k` at `level`. In `position_probs` the logit of a digit is accumulated from the constant weight in increasing input position `i`, in `train_step` the samples are in task-major order (a task's eight answers together, the tasks in the order drawn), the tokens of a sample in increasing position, every sum is accumulated in the order written from `0.0`, and the draws are made in the order written: the four `rng.integer(len(pool))` first, then one `sample_index` per output digit, answer by answer. `group_advantages(rewards, false)` is the routine of Chapter 4. In `paired_lower` the sums run in increasing `k`, `sum(d)` being accumulated from `0.0` in that order, and `1.644854` is the one-sided Gaussian point of Chapter 7. Text joined with `+` is the concatenation of the pieces, a number being written in decimal. `serve` keeps the registry of every promoted version and the position `served` in it; a version that fails its audit stays in the registry and is not served, and the version served after a rollback is the one served before.

## Routines of the training-at-scale chapters (Volume XIII)

The data chapter (Chapter 1) writes out `encode_record`, `bucket_batches`, `mix_repeats`, `minhash` and `epoch_order` where they are first used. `bytes of s` is the list of the UTF-8 bytes of the text `s`, each an `int` in `0 .. 255`, in order; `concat` of lists joins them, and `append (a and b) to mask` appends the `bool` of the conjunction. In `encode_record` the mask holds one flag per token, and the target of position `t` is supervised when `mask[t + 1]` is true. In `bucket_batches` `sort chunk ascending by L[i]` is the stable sort of the indices in `chunk` by their length, equal lengths keeping the order they have in `chunk`, a slice `order[a : b]` stops at the end of the list when `b` is larger, and the two `shuffle` calls draw from the same generator in the order written, the first for the list of indices and the second for the list of batches. In `mix_repeats` the first group is the dataset, whose entry of `q` is replaced by `1.0` minus the sum of the others; `floor(x + 0.5)` is the rounding half up of a positive number, a group with no records is listed once and takes no part in the comparison, the scale loop returns at the first `c` whose largest deviation is at most `0.01`, and when none is it returns the first scale with the smallest deviation (a later scale replaces it only when strictly smaller). In `minhash` `fingerprint(s)` is the first 8 bytes of the SHA-256 digest of the bytes of the text `s` read as one big-endian `u64` (the rule of `seed_of`), `mix` is the three scrambling lines of the SplitMix64 step without the state advance (the one of `derive_seed`), `seeds` are raw values drawn in order from one `Generator`, and the estimated Jaccard similarity of two signatures is the number of equal entries divided by their length. In `epoch_order` the product `u64(epoch + 1) * 0x9E3779B97F4A7C15` wraps modulo `2^64`, and `shuffle` is the one of the randomness section.

The mixed-precision chapter (Chapter 2) writes out `sr_round_bf16`, `sr_bits`, `mix32`, `counter_uniform`, `scale_step` and `next_scale` where they are first used. `sr_round_bf16(x, u)` takes a normal `x` (an exponent of at least -126) and a draw `u` in `[0, 1)`: it rounds the magnitude to the bf16 number below it, or to the one above it (away from zero) when `u < phi`, where `phi` is the fraction of the gap that the magnitude covers, and `sr_round_bf16(0.0, u)` is `0.0` for every `u`. `sr_bits(b, r)` adds the integer `r` in `0 .. 65535` to the `u32` pattern `b` (the `u32` addition wraps) and keeps the top 16 bits, so for a finite fp32 pattern it carries into the kept bits for exactly the number of values of `r` that the dropped 16 bits spell. `mix32` is the `u32` function written out in that chapter (its multiplications wrap modulo 2^32), and `counter_uniform(key, i)` is `(mix32(mix32(key xor i) xor 0x9E3779B9) >> 8) / 2 ^ 24`, a function of the key and the element index only. In `scale_step` a step is applied when its gradients are all finite, the count `good` is of applied steps in a row since the scale last changed, and the doubling takes place on the step that makes `good` equal `grow_after`; in `next_scale` the history is oldest first and `window` larger than its length means all of it.

The memory chapter (Chapter 3) writes out `block_forward`, `block_backward`, `recompute_backward`, `encode_second_moment`, `decode_second_moment`, `encode_first_moment`, `decode_first_moment` and `streamed_adamw_step` where they are first used. `tanh_slope(a)` is `1 - a ^ 2`, the derivative of `tanh` at a point whose output is `a`. In `recompute_backward` the entry `grads = a list of L entries, each assigned below` is a list of `L` pairs written by index, `grads[l] = (dW1, dW2)`, and `for g in (L - 1) // k down to 0` visits the segments last to first, the last one holding the remaining `L - s` blocks when `k` does not divide `L`. In the moment codecs `max(v)` of a block of non-negative values is its largest entry, `floor(x)` is the integer below `x`, `clamp(x, l, u)` is `min(max(x, l), u)`, the draw `u[i]` is a number in `[0, 1)` (`0.5` makes `floor(y + u[i])` the rounding to nearest, ties up), a block whose largest magnitude is `0.0` has the scale `0.0` and all codes `0`, and `x ^ 0.25` of a positive `x` is its fourth root. The one-element forms `dec_first(q, s)` and `dec_second(q, s)` of `streamed_adamw_step` are the elements of `decode_first_moment` and `decode_second_moment`: `(abs(q) / 127.0) ^ 2 * s` with the sign of `q` for the first moment and `(q / 255.0) ^ 4 * s` for the second. In `streamed_adamw_step` the blocks of 64 elements are those of the global grid, `bk * 64 .. bk * 64 + 63` (the last one shorter), `lo` and `hi` clip a block to the tile (a tile that cuts a block encodes its piece with the piece's own scale), a slice assignment `state.mq[lo : hi] = q` writes the codes of one block, and the keys use the `mix32` and `counter_uniform` of Chapter 2 with `key` a `u32`.

The parameter-efficient chapter (Chapter 4) writes out `lora_gradients`, `train_lora` and `bank_forward` where they are first used, and defines `adam_update` here: `adam_update(p, g, m, v, t, lr)` changes the arrays `p`, `m` and `v` in place, elementwise and in this order, `m = 0.9 * m + 0.1 * g`, `v = 0.999 * v + 0.001 * g * g`, `p = p - lr * (m / (1 - 0.9 ^ t)) / (sqrt(v / (1 - 0.999 ^ t)) + 1e-8)`, with the step count `t` starting at 1 and no weight decay. In `train_lora` the algorithm works on its own copy of the `A` it is given, the arrays `mA`, `vA`, `mB` and `vB` are the moments of `A` and `B`, which `adam_update` keeps between steps, and `loss_gradient(W)` is the gradient of the loss with respect to the whole weight. In `bank_forward` a slot `z >= len(ranks)` means no adapter (the value `4294967295` of the chapter's slot vector is one such), `A_bank[z][0:r, :]` is the first `r` rows of the padded matrix of slot `z` and the rows beyond are never read, `matmul(t, Bt_bank[z][0:r, :])` of a vector `t: [r]` and an array `[r, m]` is the vector `[m]` of the sums `t[0] * row0 + t[1] * row1 + ...` taken in the order of the rows, and `Y[i] = ...` assigns a whole row. The tables of the chapter also use the quantization rules of Volume IV, Chapter 12 in their stated form: groups of 32 consecutive entries of a row with one scale each; for 8 bits the scale `max(abs(g)) / 127`, for the even 4-bit format `max(abs(g)) / 7`, the integer code the quotient rounded half away from zero and clamped to the format's range (`-127 .. 127`, `-7 .. 7`); for NF4 the scale `max(abs(g)) / 1.0` and the code `argmin(abs(g[i] / scale - lut))`, the first index on a tie, over the 16 values of the codebook printed in the chapter; a group whose largest magnitude is below `1e-8` uses `1e-8` in its place.

The parallelism chapter (Chapter 5) writes out `average_gradients`, `ring_all_reduce`, `sharded_adam_step`, `tp_block_forward` and `pipeline_length` where they are first used. In `average_gradients` the products `counts[j] * grads[j]` are added into `total` in increasing `j` from zeros and the division is by `sum(counts)`, an `int` converted to `real`; a call with every count 1 is the plain average. In `ring_all_reduce` `copy(v[j])` is an independent copy of the array, `buf[j][a : b]` is the slice of elements `a` to `b - 1` of array `j` of the list, `mod` is the non-negative remainder defined above (`(j - s) mod R` is never negative), and in each step the list `msg` is filled before any array of `buf` is changed, so that the sends of a step read the state before the step; every sum of a reduce-scatter step is a single addition of the received chunk to the receiver's own copy. In `sharded_adam_step` the list `m` and the list `v` hold one array of `c` numbers for each rank, `adam_update` is the routine of Chapter 4 applied to the copy `piece` of a slice, `total[a : b] / R` divides each element by `R`, and the slice assignment into `w_new` stands for the all-gather. In `tp_block_forward` `W1[j * fk : (j + 1) * fk, :]` is rows `j * fk` to `(j + 1) * fk - 1` of `W1` and `W2[:, j * fk : (j + 1) * fk]` the same range of columns of `W2`, `tanh` is applied to every element of an array, the partial products are added into `S` in increasing `j` from zeros, and `X + S` is the elementwise sum. In `pipeline_length` every time is a whole number of units of one stage's forward work, the forward of a stage on a micro-batch takes 1 unit, its rebuild 1 and its backward 2, `t[s]` is the unit at which stage `s` is next free, `start = a if c else b` is the conditional expression of the scalar operations, and the sends take no time.

The distillation chapter (Chapter 6) writes out `topk_teacher`, `distill_gradient`, `distill_loss`, `sampled_gradient` and `token_advantages` where they are first used. `topk_teacher(z, k)` returns the ids of the `min(k, V)` largest entries of `z` (a tie to the lower id, by `argsort(z, descending)`) and their softmax renormalized over those entries only, computed as `exp(z[id] - z[ids[0]])` divided by the sum of these in the order of the ids. In `distill_gradient` and `distill_loss` a position with an empty `ids[t]` has no teacher and contributes nothing, the divisor `n_act` counts the others and is at least 1, and the sums over `k` are taken in increasing `k`. `sampled_gradient` returns the single-sample estimate of the gradient of the reverse divergence with respect to the logits, `-(logq[y] - log(p[y])) * (e_y - p)`. `token_advantages` clamps first and caps second, and `none` stands for an absent bound or cap; `clamp(x, l, u)` is `min(max(x, l), u)`.

The checkpoint chapter (Chapter 7) writes out `resume_step`, `fnv1a64` and `verifies` where they are first used. `RunState` is the record of the chapter: a record is a value, `s.w[j] = ...` assigns a field of the variable `s`, and `resume_step` returns the changed record. In `resume_step` `draw(seed, N, k)` is the routine of Chapter 1 (entry `k mod N` of `epoch_order(seed, N, k // N)`), `batch_gradient(X, y, idx, w)` is `(2 / B)` times the sum over the batch, in the order of `idx`, of `(dot(X[i], w) - y[i]) * X[i]`, with `dot` accumulating in increasing index from `0.0` and the sum taken in order from `0.0` and multiplied by `2 / B` after the sum, `sr_round_bf16` and `counter_uniform` and `mix32` are those of Chapter 2 and `u32(.)` converts a non-negative integer below 2^32. The powers `0.9 ^ s.t` and `0.999 ^ s.t` are repeated multiplications from `1.0`, one per update. In `fnv1a64` a `list<int>` of bytes is a list of integers in `0 .. 255`, `u64(b)` converts one to a `u64`, `xor` and the product act on 64 bits and the product wraps modulo 2^64 as in the fixed-width section. In `verifies` a file is a triple of its name, its length in bytes and its digest, `read(name)` returns the bytes of the file of that name and a function may be passed as an argument (`function(text) -> list<int>` is the type of one that takes a text and returns a list of integers); the files are checked in the order listed and the first mismatch of length or digest ends the check with `false`.

The project chapter (Chapter 8) writes out `layer_forward`, `layer_backward`, `example_gradient`, `batch_gradient`, `commit` and `restore` where they are first used. `Net` is the record of the network: for each layer `l` in `0 .. L - 1` (`L = 6`) the matrix `W[l]` of shape `[n_out, n_in]` and the vector `b[l]`, and for the layers 1 to 4 the adapter factors `A[l]` of shape `[r, n_in]` and `B[l]` of shape `[n_out, r]` (`A[l] is present` is false for the others), with the constants `r = 2` and `s = 1.0`; `g` is a `Net` of the same shapes that holds gradients, `zeros-like(net)` is one with every entry `0.0`, and `flat(g)` is its entries as one vector in the order of the chapter's tables: for the base run the 337 numbers of each layer's `W[l]` then `b[l]` in layer order, and for the adapter run (base frozen) the 128 adapter entries, `A[l]` then `B[l]` for each adapted layer in order, every matrix row-major, and for the planted mistake that trains the base as well the 465 numbers with the 128 adapter entries first and the 337 base numbers after them; `t = A x` in the adapter forward pass is the vector of the low-rank product, not the update count `t` of `AdapterRun`. In `layer_forward` and `layer_backward` the sums start at `b[l][o]` or at `0.0` as written and add in increasing index, `z if c else 0.0` is the conditional expression of the scalar operations, a layer's gradient entries receive one product per row so that the sum over a batch runs in the order of the rows, and `dx` is formed even for layer 0, where nothing reads it. In `example_gradient` `k` is the segment length or absent for keeping every vector, `kept` is a `map` from a vector index to the vector (`kept[l + 1] = cur` stores a copy), `//` is integer division and `for gi in (L - 1) // k down to 0` visits the segments last to first, the last one holding the remaining layers when `k` does not divide `L`. In `batch_gradient` row `b` of the batch belongs to rank `b mod R`, `X[i]` and `target[i]` are example `i` and its target in the stage's task, `padded(vecs)` appends `0.0` entries to each vector until the length is a multiple of `R`, `ring_all_reduce` is the routine of Chapter 5 (every rank's result is the same, so rank 0's is read), and the division is by the number of rows. `AdapterRun` is the record `(p, m, v, t, d)` of the chapter, a record being a value as above. In `commit` the bytes of a list of reals are each number as a 64-bit float, little-endian, `bytes of st.t and st.d as u64` are the two counts as 64-bit unsigned integers, little-endian, `meta_text` is the JSON text, keys sorted and no spaces, of the digest of the base, the update count and the length and `fnv1a64` digest (sixteen lower-case hex digits) of the two files, a rename moves every name that starts with the temporary directory's name to the same name under the generation's, and the two newest generations that verify are the ones whose meta parses and whose two files have the recorded length and digest. In `restore` the generation names sort newest first because the update count is written in four digits.

## Routines of the inference and serving chapters (Volume XIV)

The loading chapter (Chapter 1) writes out `pages_of`, `tile_bytes`, `stage`, `tile_quantum` and `plan_tiles` where they are first used. `lcm(a, b)` is the least positive integer that both `a` and `b` divide, for `a >= 1` and `b >= 1`. An `Encoding` is the record of the chapter: `elems` and `bytes` are the elements and bytes of one payload block (2 and 1 for 4-bit codes, 1 and 2 for bf16, 32 and 34 for a Q8_0 block), and `scales` lists the scale levels as triples `(kind, n, bits)`, `kind` being the text `"tensor"` (one scale for the whole tensor, `n` unused) or `"block"` (one scale for each `n` consecutive elements), each scale `bits` wide. `pages_of(a, b, page)` returns the first page, the last page and the number of pages that the bytes `a .. b - 1` of a file touch, pages being `page` bytes long and numbered from 0 at the first byte. `tile_bytes(enc, e)` counts the payload bytes and the scale bytes of the first `e` elements of a tensor, a partial block or group counting in full; `count * bits // 8` is an integer division of the product. `stage` adds the staging bytes of one tile for the step named by `op`, one of the texts `"Preserve"`, `"Repack"` and `"Convert"` (for `"Preserve"` the caller passes the stored encoding as `dst`). In `plan_tiles` the tile starts at the quantum and grows by one quantum while it stays inside the tensor and its staging bytes are at most `cap` (the caller passes the smaller of the configured tile bytes and the budget capacity); the result `(0, 0, 0)` means the plan is refused, and otherwise it is the elements per tile, the number of tiles (the last tile may be shorter) and the staging bytes of a full tile.

The residency chapter (Chapter 2) writes out `cost_score` and `choose_victims` where they are first used. A `map of text to Resident` is a map from the name of a model to its record, `keys of m` lists its keys in the order they were inserted, and `x sorted ascending by f(x)` is the stable sort of a list by the key `f`, equal keys keeping the order they had; `choose_victims` sorts twice, first by last use and then by score, so that equal scores are broken in favour of the earlier last use. `real(x)` converts an `int` to a `real`. The routine returns the victims taken so far and the deficit that remains after them; a remaining deficit of 0 is a plan and anything above 0 is not. The chapter's other computations (the free bytes of a budget, the peak of a set of phased parts, the headroom of a pool, and the loads of a window of weight groups, whose rules are printed beside each table) are closed-form or are enumerations of the rules that the table states.

The KV-cache chapter (Chapter 3) writes out `new_pool`, `alloc`, `incref`, `decref`, `append_row`, `reserve`, `release`, `fork`, `unshare_tail`, `locate`, `lookup_prefix`, `insert_chain`, `evict_leaves`, `quantize_row_clipped` and `make_room` where they are first used. `Pool` and `Table` are the records of the chapter: `Pool.free` is a `list<int>` used as a stack whose last element is the top, `pop_last(L)` removes the last element of a non-empty list and returns it, and `append b to L` pushes; `a list of n zeros` is the `list<int>` of `n` entries `0`, `copy of L` an independent copy of a list, and `t.blocks = []` makes the list empty. `return nothing` returns an absent `optional` value, `best = nothing` makes an `optional` variable absent, and a present optional tuple is read with `best[0]`, `best[1]`. `x in L` for a `list` is true when some element equals `x`, `remove v from L` removes the first element equal to `v`, and `remove k from m` removes the key `k` from a map. In `lookup_prefix` and `insert_chain` a key is `tuple(concat([parent], tokens))`, the parent's block id (`ROOT = -1` for a prompt's first block) followed by the block's token ids, so that two keys are equal only when the parent and every token agree; `stamp` maps a block to the value of `tick` at its last insertion or hit, and `tick` starts at 0. In `evict_leaves` `the set of key[0] over every key in cache` is the parents that some indexed block chains off, the candidates are visited in `keys(cache)` order and the smallest pair `(stamp[b], b)` wins, so a tie of stamps goes to the lower block id. In `quantize_row_clipped` `round_half_even(y)` is the integer nearest `y`, an exact half going to the even integer (`round_half_even(2.5) = 2`, `round_half_even(-0.5) = 0`), and `clamp(x, l, u)` is `min(max(x, l), u)`. In `make_room` `need_of(s)` is the number of blocks sequence `s` must take to store one more row, `ceil((len + 1) / B) - held`, and `blocks_of(s)` the blocks it holds; `the last s in running that is not in promoted, or the last s in running if every one is` names the newest running sequence that was not swapped in during the current step, falling back to the newest of all; the caller puts a sequence moved with `"swap out"` at the back of the swapped queue and one moved with `"recompute"` at the front of the waiting queue.

The batching chapter (Chapter 4) writes out `static_slots`, `continuous_slots`, `forward_work`, `forward_cost50`, `free_rows`, `run_burst`, `grow`, `run_pool` and `run_overload` where they are first used. `F(c, n)` is the FLOP count of Volume V, Chapter 12 for `n` new positions on a cache of `c` (an `int`: `2 n N_blk + 4 L H d_h (n c + n (n + 1) / 2) + 2 D V`, the head counted once), and `Bw`, `Bkv`, `D` and `V` are the weight bytes, the cache bytes per position, the width and the vocabulary of the chapter's decoder, all `int`. A slice is a triple `(start, rows, last)` whose `last` is true for the last slice of a prompt, the one whose pass computes the head. `forward_cost50` returns 50 times the cost in equivalent bytes, so every cost and every sum of costs is an exact `int`; a table divides by 50 only when it prints. `max(list)` is the largest element of a list of `int`, `min(a, b)` and `max(a, b)` of two `int` are the smaller and the larger, `argmin(free_at)` returns the first index of the smallest element, and `waiting[1 : len(waiting)]` and `concat` make new lists. `Seq`, `Req` and `Run` are records of the chapter, and `running[j].made = running[j].made + 1` assigns a field of element `j` of the list `running`, changing that element in place; a list comprehension over `running` (`[r for r in running if ...]`) builds a new list of the elements that pass the test, in order. In `run_burst` the policies are the texts `"whole"`, `"one"`, `"own"` and `"mixed"`, the sequence already decoding has the id 0 and wants `10^9` tokens (more than it can get before the run ends), and the entries 1 to `n_req` of `first` and `last` hold the iteration of each new request's first and last token. In `run_pool` the policies are `"reserve"`, `"recompute"` and `"none"`, `x // y` is the integer quotient rounded down, so `(c + blk) // blk` is the number of blocks of `c + 1` positions, and a run that finds the pool unable to back a decode step under `"none"` returns `(true, t, k, 0, 0, 0)` with the iteration `t` and the `k` sequences then decoding. In `run_overload` the policies are `"unbounded"`, `"depth"` and `"deadline"` (the last branch), a request whose start equals its arrival counts as started when a later arrival counts `q`, and `ok == false` is the test of a `bool`.

The speculative-decoding chapter (Chapter 5) writes out `verify_greedy`, `draft_proposals`, `speculative_greedy`, `marginal_window`, `update_estimate`, `residual`, `verify_sampled` and `prompt_lookup` where they are first used. A function may be passed as an argument, as in Volume XIII, Chapter 7: `draft` and `target` in `speculative_greedy` return each model's greedy token after a context (the largest probability, the lowest id on a tie), and the list comprehension that builds `best` stands for the rows of one target pass, row `j` scoring the context followed by the first `j` proposals. `append t to props` appends one element, `concat` joins lists, `props[0 : j]` is the first `j` elements, and two lists compare equal when they have the same length and the same entries in order. In `verify_sampled` `p[j]` is the distribution the target samples from at row `j` after its temperature and cuts, `q[j]` the distribution the proposal was drawn from (a one-hot row for a deterministic draft), `draw` is the routine of Volume V, Chapter 11, and the uniform numbers are drawn in the order written: one per tested proposal, then one for the draw that ends the round. In `marginal_window` the window is the longest run `j = 1 .. k` with `alpha ^ j >= c`, `alpha ^ j` formed by repeated multiplication from `1.0`; `update_estimate` leaves both counts unchanged for a round without proposals. In `prompt_lookup` `for i in m - n - 1 down to 0` visits the earlier starts latest first.

The interface chapter (Chapter 6) writes out `validate`, `for_serving`, `validate_served`, `classify` and `sort_reply` where they are first used. `ParamSpec`, `BlobSpec` and `ActionSpec` are the records of the chapter; `optional real` and `optional Value` fields are tested with `is present`. A `Value` is a JSON value (RFC 8259): `kind(v)` is `"int"` for a JSON number written without a fraction or exponent, `"float"` for any other JSON number, and `"bool"`, `"text"` or `"other"` otherwise; `number(v)` is a number as a `real`; `json(v)` is its compact JSON text with the keys of every object in ascending order; and a `text` stored into a `map<text, Value>` is the JSON string of that text. `type_ok(p, v)` is true when `p.ty` is `"str"` and `kind(v)` is `"text"`, `"int"` and `"int"`, `"float"` and `"int"` or `"float"`, `"bool"` and `"bool"`, or `"enum"` and `v` is a text in `p.values`. `copy(m)` of a map is a new map with the same keys and values. `bytes of v` for a text `v` is the list of its UTF-8 bytes. A bound `p.lo` or `p.hi` joined to a text with `+` is written as the engine writes a 64-bit float: without a fraction when it is whole (`1`, `30`), otherwise in the shortest decimal form that reads back to the same number (`0.5`). `missing_message(spec, p)` is `"action '" + spec.name + "' needs '" + p.name + "', which this machine has not configured: set $" + p.host_env + " (or pass '" + p.name + "' explicitly)"` when `p.host_env is present` and `"missing required param '" + p.name + "' for action '" + spec.name + "'"` otherwise. `blobs` maps an input's name to its medium. For texts, `starts_with(s, t)` and `ends_with(s, t)` are true when the first (last) `len(t)` characters of `s` are `t` (always true for an empty `t`), `contains(s, t)` is true when `t` occurs in `s` at some position, and `find(s, t)` is the position of the first character of the first occurrence of `t` in `s`, or -1 when there is none; `x in L` for a list is true when some element equals `x`. In `classify` `SLUGS` is the list `not-installed`, `failed-to-load`, `unsupported-capability`, `timeout`, `invalid-request`, `backend-failure`, `unavailable`, `nothing` is the absent value of an `optional` result, and the `loop` ends at a `return`. In `sort_reply` `capacity_counts(e)` is present when `e` contains `needs ` followed by a whole number, the text ` tokens, engine capacity is ` and a whole number (the first occurrence of `needs ` is the one read), and the rules are tried in the order written.

The operations chapter (Chapter 8) writes out `starts_with`, `trace_admits`, `nearest_rank`, `path_to_root`, `lca`, `hsm_step` and `controller_turn` where they are first used. `starts_with(s, p)` compares the characters of two texts (code points, by the slice rule of Volume V), so a target is matched by a prefix of its text and not of its `::`-separated segments; in `trace_admits` a directive is a pair of a target and a level from 1 to 5, the longest matching target decides and a tie in length keeps the first in the list, and with no match the event is off. `nearest_rank` takes a list already in ascending order and `ceil(q * n)` is computed exactly (the product of the `real` `q` and the `int` `n`, then the ceiling as an `int`). In the state-machine routines a state is a `text`, `parent` maps every state except the single top state to its parent, `s in parent` tests whether a key is present (the `in` of maps) and `c in up` whether the list `up` holds an element equal to `c`; `lca` requires that the two states share an ancestor, which a tree with one top state guarantees. `dispatch(s, ev)` returns a pair whose first component is `"handled"`, `"unhandled"` or `"tran"` and whose second is the target state when the first is `"tran"` (and ignored otherwise); `on_exit` and `on_entry` are the machine's actions, which return nothing and may change the machine's own fields (the controller's `pending_tick` and `want_idle`) but never its state or its queue. In `controller_turn` the record `Ctl` is changed in place (`inout`), `hsm_step(c.state, e, ...)` is called with the machine's `parent`, `dispatch`, `on_exit` and `on_entry`, `poll()` returns the text of an event or `""` when none is pending, and `cap` bounds the passes of the outer loop (brain uses the larger of the turn's token limit plus 8 and 100,000).

## Routines of the scientific and engineering methods chapters (Volume XV)

The porting chapter (Chapter 1) writes out `isolated_parity` and `identity_mismatches` where they are first used. In `isolated_parity` a stage is a `function(tensor) -> tensor`, `want[s]` is the golden's stored input to stage `s` and `want[s + 1]` its stored output, `rel_l2(a, b)` is the routine of Volume VI, Chapter 2 applied to the two tensors flattened in row-major order (`norm(a - b) / norm(b)`, an error for a reference of norm zero), the stages are visited in increasing `s`, and the result lists every stage whose difference exceeds its bound, in increasing order, empty when none does. In `identity_mismatches` an identity is a `list<(text, int)>` of field names and integer values, `r[0]` and `r[1]` are the components of a pair, the fields of `actual` are checked in the order listed, a field asked for that `recorded` does not hold is an error, and the result lists the names of the fields whose integers differ, in the order of `actual`.

The testing chapter (Chapter 2) writes out `attention_variant`, `blind_mutants`, `parity_failures`, `parity_test` and `run_conformance` where they are first used. `attention_variant(Q, K, V, m)` is one causal attention head for `m = 0` and one of six deliberately wrong copies for `m = 1 ..= 6`; a score that is not visible is `-infinity`, so `softmax` gives it weight 0 and every row (or, for `m = 2`, every column) has at least one visible entry; the dot products are summed in increasing index from `0.0`, and `transpose(softmax(transpose(scores)))` normalises each column. `blind_mutants` skips the variant `m = 5` when `dv != dk`, because its product does not exist there and that counts as caught, and returns the other variants whose largest absolute difference from the reference is at most `tol`, in increasing `m`. In `parity_failures` the cosine and the relative L2 error are computed on the flattened taps in index order, and a value that is not finite (an output of zeros gives `0 / 0`) is a failure, written as `not is_finite(c) or c < cos_floor` because a comparison with `nan` is false. `parity_test` takes the golden as a `map<text, array<real>[N]>` and a flag `present`, prints one skip line and returns no failures when the golden is absent and `prove` is false, and refuses an absent golden when `prove` is true, an empty list of taps and a tap name the golden does not hold; `name + ": " + f` joins texts. In `run_conformance` a `Subject` is whatever the interface under test needs (in the chapter, a function from logits, targets and weights to a loss and its gradient), a contract is a `function(Subject) -> bool`, `zeros(bool, [R, C])` is an array of `false`, the loops visit the subjects and then the contracts in list order, and every subject of `known_bad` must fail at least one contract.

The project chapter (Chapter 6) writes out `dot_lr`, `block_forward`, `block_backward`, `quantize_rows` and `int8_matvec` where they are first used. `dot_lr(a, b)` forms the first product, then adds each further product in increasing index, and is the order in which the chapter's 32-bit port rounds; `block_forward` handles one position and returns `Taps`, the record of the six arrays `n`, `gate`, `up`, `h`, `f` and `y`, and evaluated in 64 bits it is the reference, in 32 bits one operation at a time (with `exp` the library's 64-bit value rounded once and `eps` rounded to 32 bits before use) the port. `block_backward` uses `rms_norm` and `rms_norm_backward` of the transformer section below on one row at a time (`x[t : t + 1, :]`, shape `[1, D]`) and returns `Grads`, the record of the gradients of the inputs, the gain and the three matrices. In `quantize_rows` the codes are whole numbers from -127 to 127 held as `real`, `round(v, 0)` rounds half away from zero, and one scale per row is the call with `G = k`; `int8_matvec` adds each group's products from `0.0` in increasing index, multiplies the partial sum by the group's scale and adds the groups from `0.0` in increasing group index.

## Routines of the transformer chapters (Volume V)

Chapters 9 to 12 use four routines of the normalization and attention chapters without writing them out again. Each is defined here by what it returns.

- **`rms_norm(x, gamma, eps)`** of Volume III, chapter 10 (the pseudocode of its RMS normalization section) normalizes every row of `x: [R, D]`: `inv_std[i] = 1 / sqrt(mean(x[i, :]^2) + eps)`, `xhat[i, :] = x[i, :] * inv_std[i]` and `y[i, :] = gamma * xhat[i, :]`, in this order: when the call is evaluated one operation at a time in a stated format (above), `xhat` is rounded before it is multiplied by `gamma`, so `(x * gamma) * inv_std` is a different computation in `f32` (in 64 bits the two orders differ by rounding only). It returns the triple `(y, xhat, inv_std)`, and a caller that needs only the result writes `(y, _, _) = rms_norm(...)`, where `_` is a name whose value is not used.
- **`rms_norm_backward(dy, xhat, inv_std, gamma)`** returns `(dx, dgamma)` for the gradient `dy: [R, D]` of the loss with respect to `y`, using the `xhat` and `inv_std` that `rms_norm` returned: `dx[i, j] = inv_std[i] * (gamma[j] * dy[i, j] - xhat[i, j] * mean(gamma * dy[i, :] * xhat[i, :]))` and `dgamma[j] = sum(dy[i, j] * xhat[i, j] for i in 0 .. R)`. It is the rule of Volume V, Chapter 9.
- **`multi_head(X, Wq, Wk, Wv, Wo, H, causal)`** is the function of Volume V, Chapter 7, and **`multi_head_backward(dY, X, Wq, Wk, Wv, Wo, H, causal)`** its backward pass: `dOcat = matmul(dY, transpose(Wo))`; for each head `h` with columns `a .. b - 1`, `attention_backward` of that chapter, run on `dOcat[:, a:b]` with the head's queries, keys, values and weights recomputed from `X` and the column blocks of the matrices, gives the head's gradient of the input and the column blocks `dWq[:, a:b]`, `dWk[:, a:b]`, `dWv[:, a:b]`; `dX` is the sum of the heads' input gradients and `dWo = matmul(transpose(Ocat), dY)` with `Ocat` the concatenated head outputs. It returns `(dX, dWq, dWk, dWv, dWo)`.
- **`silu_slope`**, the derivative of `silu`, is in the table of scalar operations.
- **Routines written out in Chapter 14.** `init_gpt(name, V, T, D, F, L)` (the initialization of Chapter 10 as an algorithm: the order of the draws and the extra factor of the two output matrices), `mlm_corrupt(ids, special, mask_id, V, p_sel, rng)` (selection and the 80, 10 and 10 percent replacement of masked language modeling, with the draws it consumes), `infonce(Ahat, Bhat, tau)` and `recall_at_k(sims, relevant, k)` are defined in that chapter, in this pseudocode, where they are first used.
- **Routines written out in Volume VI, Chapter 6.** `select_top(score, n_sel)` (the positions of the `n_sel` largest scores in ascending order, a tie to the lower position), `sparse_attention_row(q, K, V, sel)` (one softmax over the scores of the listed keys only; with every position listed it is the dense row) and `lsa_read(...)` (the keys a query reads under latent sparse attention) are defined in that chapter, in this pseudocode, where they are first used.
- **Routines written out in Volume VI, Chapter 7.** `looped_forward(ids, E, block_params, Wc, n_loop)` (the embeddings of the ids, then `n_loop` applications of the same pre-norm block of Volume V, Chapter 9, then the gain-only RMS norm and the head on the last position), `exit_distribution(z)` (the exit probabilities `q` of the passes from the gate logits `z` of all passes but the last) and `exit_pass(q, threshold)` (the first pass whose cumulative probability reaches the threshold, the last pass when none does) are defined in that chapter, in this pseudocode, where they are first used.

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
| `rng.state` | the current `u64` state of the generator. `Generator(rng.state)` continues the stream exactly where `rng` stands, provided no Gaussian value is held (`uniform`, `integer` and `shuffle` never hold one; after an odd number of `normal` calls the held value is also state and must be saved with it). Volume III, Chapter 14 stores it in a checkpoint |
| `derive_seed(run_seed, name)` | the seed of the stream called `name` under a run seed: `h = mix(run_seed)`, then `h = mix(h xor b)` for each byte `b` of the UTF-8 name in order, and `h` is the result in `0 .. 2^64 - 1`, where `mix` is the three scrambling lines of the SplitMix64 step without the state advance (Volume I, chapter 17, which gives the code and printed values). `Generator(derive_seed(s, "init"))` is the stream "init" of run seed `s` |
| `seed_of(name)` | the seed of the worked example called `name` (a text such as `kernel_selection.ranking`): the first 8 bytes of the SHA-256 digest of the UTF-8 bytes of `name`, read as one big-endian `u64`. Every worked example that draws random numbers prints this seed beside its name; `Generator(seed_of(name))` is its generator. It is a rule for naming the book's own examples and is not `derive_seed`, which mixes a run seed with a stream name (Volume I, chapter 17) |

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
