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

Three numerical routines are used by later chapters and defined, with their
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
