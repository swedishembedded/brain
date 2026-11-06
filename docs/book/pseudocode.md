<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Pseudocode Conventions

The algorithms in this book are written in a small pseudocode language
rather than in any real programming language. It is meant to be translated
line by line into whatever language you use. For that to work, every
construct has to mean exactly one thing. This page defines them.

## Values and types

Every value has a type, and the book always states it when it matters.

| Type | Meaning |
|---|---|
| `int` | a signed integer large enough for any index or count in the algorithm |
| `f32`, `f16`, `bf16` | floating-point numbers of the stated format (Volume IV) |
| `real` | a floating-point number whose format does not matter to the algorithm; implement it as `f32` unless told otherwise |
| `bool` | `true` or `false` |
| `array<T>[d0, d1, ...]` | a tensor of element type `T` with the given shape |

## Arrays

Arrays are **indexed from 0**. An array of shape `[d0, d1, d2]` holds
`d0 * d1 * d2` elements, and `a[i, j, k]` requires
`0 <= i < d0`, `0 <= j < d1`, `0 <= k < d2`. Reading or writing outside
those bounds is an error, never a silent wrap or clamp.

Unless a chapter says otherwise, arrays are stored in **row-major order**:
the last index varies fastest. Element `a[i, j, k]` of an array of shape
`[d0, d1, d2]` lives at offset `(i * d1 + j) * d2 + k` from the start of
its storage. Volume I, chapter "Arrays, Shapes, Strides and Broadcasting",
derives this and the general stride formula.

`a.shape` is the shape, `a.shape[0]` its first dimension, and `len(a)` the
same as `a.shape[0]`. `zeros(T, [d0, d1])` creates an array filled with zero.

Slices use half-open ranges: `a[i, 2:5]` is the three elements at
`j = 2, 3, 4`. A slice is a **view**: it refers to the same storage, so
writing through it changes the original. Where a copy is needed, the
pseudocode says `copy(...)`.

## Arithmetic on arrays

Arithmetic operators apply to scalars. When an operator is applied to whole
arrays, the pseudocode says so explicitly with a loop or with one of these
named operations:

| Operation | Meaning |
|---|---|
| `matmul(A, B)` | matrix product; `A: [m, k]`, `B: [k, n]` gives `[m, n]` |
| `transpose(A)` | swaps the last two axes |
| `sum(a, axis)` | sum along one axis, which is removed from the shape |
| `max(a, axis)` | maximum along one axis |
| `map(f, a)` | applies the scalar function `f` to every element |
| `a + b`, `a * b` on arrays | elementwise, and **only** when the shapes are equal or the text states the broadcast |

Broadcasting is never implicit. When a smaller array is combined with a
larger one, the text writes the rule: for example "add `bias: [D]` to every
row of `x: [T, D]`".

## Control flow

```text
for i in 0 .. n:          # i = 0, 1, ..., n-1 (n excluded)
for i in 0 ..= n:         # i = 0, 1, ..., n   (n included)
for i in n-1 down to 0:   # reverse order
parallel for i in 0 .. n: # iterations are independent and may run at once
while condition:
if condition: ... else: ...
return value
```

A `parallel for` promises that no iteration reads anything another
iteration writes. If an algorithm needs iterations to combine results, it
uses an explicit reduction (Volume IV), never a shared variable.

## Functions and mutation

```text
function name(arg1: type, arg2: type) -> type:
    ...
```

Arguments are read-only unless marked `inout`, in which case the function
may change them and the caller sees the change. A function never changes
anything that is not an argument marked `inout` or a value it created.
There is no hidden global state; anything an algorithm needs, including a
random number generator, is passed in.

## Randomness

Random numbers come from an explicit generator: `rng.uniform()` returns a
`real` in $[0, 1)$, `rng.normal()` a sample of $\mathcal{N}(0, 1)$ and
`rng.integer(n)` an `int` in `0 .. n`. Two runs with generators created
from the same seed produce the same numbers. Volume I, chapter "Randomness
on a Deterministic Machine", shows how to build one.

## Errors

`require condition, "message"` stops the algorithm with an error if the
condition is false. Algorithms check their inputs this way at the start:
shapes, ranges and any assumption the derivation relied on. An algorithm
never silently corrects bad input.

## Numerical tolerance

Two floating-point results are compared with a tolerance, never with `==`.
`close(a, b, rtol, atol)` is true when
$\lvert a - b \rvert \le \mathrm{atol} + \mathrm{rtol} \cdot \lvert b \rvert$
for every element. Each chapter states the tolerance its tests use and
explains where it comes from.

## A complete example

```text
function softmax_rows(z: array<real>[T, V]) -> array<real>[T, V]:
    require T > 0 and V > 0, "softmax of an empty array"
    p = zeros(real, [T, V])
    parallel for t in 0 .. T:
        m = max(z[t, :], axis=0)              # subtract the row maximum
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
