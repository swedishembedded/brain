<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Arrays, Shapes, Strides and Broadcasting

*How are vectors, matrices and higher-dimensional arrays actually laid out in memory and indexed?*

## The problem

A model reads a batch of 32 colour images, each 224 pixels high and 224 wide
with three colour channels. As a mathematical object that batch is a table with
four indices: which image, which channel, which row, which column. A matrix has
two indices, and Chapter 4 gave those a great deal of structure. A batch of
images has four, a batch of token sequences has three, and an engine has to
store, slice, reorder and combine all of them.

But a computer has no tables. Memory is one long line of numbered cells, and a
four-index table has to be flattened onto it. Every operation that sounds
harmless in the mathematical language then turns into a question about that line:

1. **Where is the element with index $(b, c, i, j)$?** A program that wants one
   number must turn four indices into one address, and must do it the same way
   every time, in every kernel, or two pieces of code will disagree about which
   number is which. Nothing in the numbers themselves says how it was done.
2. **What does it cost to transpose?** Chapter 4 defined the transpose as
   swapping rows and columns, and noted that a program rarely needs a copy. How
   can a matrix be read as its own transpose without moving a single number?
3. **What can be reshaped, and what cannot?** A sequence of vectors of width 6
   can be seen as two vectors of width 3 per position (two "heads"). That is a
   change of shape. Moving the head axis in front of the position axis is also
   a change of shape. They give arrays of the same shape with different
   contents, and the program runs either way and produces a plausible wrong
   answer for one of them.
4. **How do arrays of different shapes combine?** A layer adds a bias vector of
   length $D$ to every one of $T$ rows. Chapter 4 wrote "add $\mathbf{b}$ to every
   row" in words. There should be one precise rule for this, so that code can
   do it without making $T$ copies of the bias, and so that mistakes are
   caught and not silently turned into something else.

This chapter answers all four with one idea, the stride, and one habit, the
distinction between the numbers in memory and the way they are read. Chapter 3
gave us vectors, Chapter 4 matrices. We now generalise both to arrays with any
number of axes, say exactly how they live in memory, and derive the operations
that move, reorder and combine them.

A small case shows what is at stake. The twelve numbers $0, 1, 2, \dots, 11$ sit
in memory in that order, and a program is told to treat them as a matrix $A$ of
3 rows and 4 columns. What is $A_{12}$, the entry in row 1, column 2 (counting
from 0)? If the rows were written out one after another, row 0 is
$0, 1, 2, 3$ and row 1 is $4, 5, 6, 7$, so the answer is 6. If the columns were
written out one after another, column 0 is $0, 1, 2$, column 1 is $3, 4, 5$,
column 2 is $6, 7, 8$ and the entry is the second number of that column, 7.
Both readings are reasonable, they give different matrices from the same twelve
numbers, and the numbers themselves do not say which one was meant. The rest of
the chapter turns that small ambiguity into a complete theory.

You may use from earlier chapters: vectors (Chapter 3), matrices, the transpose
and matrix multiplication (Chapter 4), and sums and products (Chapter 1). Not
assumed: anything about computer memory; it is explained here as far as the
book needs it. How memory speed depends on the order of reading is only touched
on here and is measured in Volume IV.

One note on indices. Elsewhere in mathematics indices start at 1. This chapter
is about addresses, where the first element is at offset 0, so, as in the
[pseudocode conventions](../pseudocode.md), every index here starts at **0**.

## Before you begin

Five questions on ideas from earlier chapters, to answer from memory. The
answers are at the end of the chapter. If one is hard, go back to the
chapter named before reading on.

1. (Chapter 1, sums.) Write $\sum_{k=0}^{2} i_k s_k$ out in full and evaluate it for
   $(i_0, i_1, i_2) = (1, 2, 3)$ and $(s_0, s_1, s_2) = (12, 4, 1)$.
2. (Chapter 2, functions.) What does it mean for a function to have an inverse,
   and what two properties must the function have for one to exist?
3. (Chapter 3, vectors.) A vector has the entries $(7, 8, 9)$. How many entries
   has it, and which is the entry with index 1 if the first entry has index 0?
4. (Chapter 4, matrices.) What is the transpose of
   $\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$, and what is its shape?
5. (Chapter 4, layers.) A batch $\mathbf{X}$ of $B$ inputs of width $d_{\text{in}}$
   is the matrix with the inputs as rows, and a layer has weights
   $\mathbf{W}$ of shape $[d_{\text{out}}, d_{\text{in}}]$. What are the shapes
   of $\mathbf{X}$ and of $\mathbf{X}\mathbf{W}^\top$, and which dimension is
   summed away in the product?

## Arrays with any number of axes

A vector is a list of numbers with one index. A matrix is a table with two. The
natural next step is a table with any number $R$ of indices, and the book needs
it constantly, so it gets a name.

A **tensor** is an array of numbers with $R \ge 0$ axes; $R$ is its **number
of axes** (the book never calls it the rank of the tensor, because Chapter 4
used rank for the number of independent columns of a matrix, a different
thing). A scalar is a tensor with no axes, a vector has one, a
matrix two. The word has a deeper meaning in physics and geometry, where it
also says how the numbers change when the coordinates change. In machine
learning it almost always means just this: a multi-axis array of numbers. The
book writes a tensor with more than two axes as $\mathsf{X}$, as in the
[notation page](../notation.md), and does not use any property of it beyond
what this chapter states.

The **shape** of a tensor is the list of the sizes of its axes,
$[d_0, d_1, \dots, d_{R-1}]$, outermost axis first (the book's convention).
The tensor has $d_0 d_1 \cdots d_{R-1}$ elements, and an index is a tuple
$(i_0, i_1, \dots, i_{R-1})$ with $0 \le i_k < d_k$. The shape $[3, 4]$ is a
matrix with 3 rows and 4 columns, so $A_{ij}$ is `A[i, j]` with $i$ the first
(row) index. A vector of length 12 has shape $[12]$, a scalar has shape $[\,]$,
and a *row* of a batch can be written $[1, 12]$, which is not the same shape as
$[12]$ even though it holds the same twelve numbers. Whether it matters is
decided by the broadcasting rule below.

Shapes carry meaning in the order of their letters, and the book uses the same
ones everywhere (see the notation page): $[B, T, D]$ is a batch of $B$
sequences of $T$ positions with a vector of width $D$ at each position, and
$[B, C, H, W]$ is a batch of images with $C$ channels, $H$ rows and $W$
columns. Nothing in a tensor records these meanings. The program holds numbers
and a shape; the meaning of each axis lives in the programmer's head and in the
names of the functions. That is why this chapter spends so long on exactly how
the axes are ordered.

## Memory is one long line

A machine stores numbers in cells numbered $0, 1, 2, \dots$. The cell number is
called an **address** or, relative to the start of the array's own storage, an
**offset**. We count offsets in elements, not bytes: a 32-bit number takes 4
bytes, so element offset 3 is byte offset 12, and the conversion is one
multiplication that every real program does last and the book does not mention
again until it matters.

An array with a shape of $12$ elements is easy to store: element $i$ at offset
$i$. The interesting question is a matrix of shape $[3, 4]$. It has twelve
elements and two indices, and memory has one. Whatever we choose, we are
choosing an ordering of the twelve index pairs $(i, j)$ into a single line.
Two choices come to mind at once.

- **Row by row.** Write the first row, then the second, then the third:
  $A_{00}, A_{01}, A_{02}, A_{03}, A_{10}, \dots$ The second index changes fastest.
- **Column by column.** Write the first column, then the second, and so on:
  $A_{00}, A_{10}, A_{20}, A_{01}, \dots$ The first index changes fastest.

The first is called **row-major layout**, the second **column-major layout**.
The book uses row-major throughout, and so do the pseudocode conventions, the
WGSL kernels of the case study and most of the machine learning world. Column-major
is the layout of Fortran, and therefore of the linear-algebra libraries
descended from it, which the case study meets at the end of this chapter.

Take the matrix whose entry $(i, j)$ is the number $10i + j$, so that the value
names the position. (The first digit is the row, the second the column, which
makes it easy to see where every number went.) Its two layouts:

<!-- worked: arrays.layout_3x4 -->
A = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]] [3, 4]; entry (i, j) holds the number 10 i + j.

Row-major memory: [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]

Column-major memory: [0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23]

| (i, j) | value | row-major i*4 + j | column-major i + 3*j |
|---|---:|---:|---:|
| (0, 0) | 0 | 0 | 0 |
| (0, 3) | 3 | 3 | 9 |
| (1, 0) | 10 | 4 | 1 |
| (1, 2) | 12 | 6 | 7 |
| (2, 1) | 21 | 9 | 5 |
| (2, 3) | 23 | 11 | 11 |

Memory distance from (i, j) to (i, j+1): 1 row-major, 3 column-major. From (i, j) to (i+1, j): 4 row-major, 1 column-major.
<!-- /worked -->

![A 3 by 4 matrix whose entry in row i, column j is the number 10 i plus j, coloured by row, above two strips of twelve cells showing its memory. In row-major order the strip reads 0 to 3, 10 to 13, 20 to 23 with each colour contiguous; in column-major order the colours interleave, reading 0, 10, 20, 1, 11, 21 and so on.](img/memory-layout.svg)

The picture makes the difference visible: in row-major memory each colour is
one unbroken run, and in column-major memory the three rows are woven together.
The last two lines of the block say the same thing in numbers. Moving one step
to the right in a row is one step in memory in row-major layout and a jump of
three in column-major layout, and moving one step down is the reverse. Which is
cheaper depends on what the code does next, a point the section on views returns to.

## The row-major offset, derived

We want a formula that turns an index into an offset. Derive it for two axes
first, then for any number.

**Two axes.** A matrix of shape $[m, n]$ has $m$ rows of $n$ elements each. In
row-major layout the rows are stored one after another, so row $i$ is preceded
by $i$ complete rows, that is by $i \cdot n$ elements. Within the row, element
$j$ is preceded by $j$ elements. The element $(i, j)$ therefore sits at
$$\mathrm{offset}(i, j) = i\,n + j.$$
Check with the picture: $(2, 1)$ in a $[3, 4]$ matrix is $2 \cdot 4 + 1 = 9$, the
tenth cell, holding the number 21. The size $m$ never appears: the first axis
is the outermost and nothing is stored before it that depends on how many rows
there are.

**Three axes.** A tensor of shape $[d_0, d_1, d_2]$ is $d_0$ matrices (called
planes or slices) of shape $[d_1, d_2]$ stored one after another. A plane has
$d_1 d_2$ elements, so plane $i_0$ starts at $i_0\,d_1 d_2$, and inside the plane
the element $(i_1, i_2)$ sits at the two-axis offset $i_1 d_2 + i_2$. Adding:
$$\mathrm{offset}(i_0, i_1, i_2) = i_0\,d_1 d_2 + i_1\,d_2 + i_2 = (i_0\,d_1 + i_1)\,d_2 + i_2.$$
The last form, a nested one, is how the pseudocode conventions wrote it, and it
shows the structure: first combine the first two indices into the number of
whole rows before the element, $i_0 d_1 + i_1$, then multiply by the row length
$d_2$ and add the last index.

**Any number of axes.** The pattern is that of a number written in a mixed
radix, like hours, minutes and seconds, where $(h, m, s)$ means
$(60h + m)\cdot 60 + s$ seconds. The nested form for $R$ axes is
$$\mathrm{offset}(i_0, \dots, i_{R-1}) = \Bigl(\dots\bigl((i_0\,d_1 + i_1)\,d_2 + i_2\bigr)\dots\Bigr)\,d_{R-1} + i_{R-1}.$$
Expanding the brackets gives a sum of one term per axis:
$$\mathrm{offset}(i_0, \dots, i_{R-1}) = \sum_{k=0}^{R-1} i_k\, s_k,
\qquad s_k = \prod_{m = k+1}^{R-1} d_m = d_{k+1}\, d_{k+2} \cdots d_{R-1}.$$
The number $s_k$ is the **stride** of axis $k$: how far apart, in elements, two
neighbouring entries along that axis are in memory. It is the product of the
sizes of all axes to the right of $k$ (an empty product is 1, so the last
stride is always 1: neighbours along the last axis are adjacent).

That is the formula the pseudocode page promised. It is worth checking on a
shape with three axes against both forms:

<!-- worked: arrays.nd_offsets -->
Shape [2, 3, 4] has 24 elements and row-major strides (12, 4, 1).

| index | nested (Horner) form | stride form |
|---|---|---|
| (0, 0, 0) | ((0*3 + 0)*4 + 0) = 0 | 0*12 + 0*4 + 0*1 = 0 |
| (0, 0, 3) | ((0*3 + 0)*4 + 3) = 3 | 0*12 + 0*4 + 3*1 = 3 |
| (0, 1, 0) | ((0*3 + 1)*4 + 0) = 4 | 0*12 + 1*4 + 0*1 = 4 |
| (0, 2, 3) | ((0*3 + 2)*4 + 3) = 11 | 0*12 + 2*4 + 3*1 = 11 |
| (1, 0, 0) | ((1*3 + 0)*4 + 0) = 12 | 1*12 + 0*4 + 0*1 = 12 |
| (1, 1, 1) | ((1*3 + 1)*4 + 1) = 17 | 1*12 + 1*4 + 1*1 = 17 |
| (1, 2, 3) | ((1*3 + 2)*4 + 3) = 23 | 1*12 + 2*4 + 3*1 = 23 |

All 24 offsets are distinct and cover exactly 0 to 23: True.

Going back: offset 17 = 1*12 + 5, then 5 = 1*4 + 1, so the index is (1, 1, 1); the function agrees: (1, 1, 1).

Column-major strides of the same shape are (1, 2, 6), so index (1, 2, 3) sits at offset 23 there and at 23 row-major (both are the last element, as they must be).
<!-- /worked -->

Three facts follow from the derivation, and each is used later.

**The offsets are a perfect fit.** The index $(i_0, \dots, i_{R-1})$ is a
mixed-radix numeral, and mixed-radix numerals are in one-to-one correspondence
with the integers $0, \dots, d_0 \cdots d_{R-1} - 1$. The check in the block
confirms it for 24 elements: no two indices share an offset, and no offset is
unused. That is also why the elements of a row-major array occupy exactly
$d_0 \cdots d_{R-1}$ consecutive cells and nothing more.

**The map can be undone.** Given an offset $o$, the first index is the integer
quotient $o \div s_0$, then take the remainder and divide by $s_1$, and so on:
$i_0 = o \div s_0$, $i_1 = (o \bmod s_0) \div s_1$, and so on. (Exactly the way
$3725$ seconds is $1$ hour, $2$ minutes and $5$ seconds: divide by 3600, then by 60.)
A GPU kernel that runs one thread per output element receives a flat
thread number and recovers its index this way; the section on the case study
shows a real one.

**The first size never matters.** $s_k$ involves only the sizes to the right,
so $d_0$ appears in no stride. A tensor can have its first axis enlarged or
shrunk, a batch of 32 images or a batch of 31, and every element that exists in
both keeps the same offset. That fact is why the batch axis is placed first, as
the later section on batches explains.

## Column-major, and why conventions matter

Column-major layout is the mirror image: the first index varies fastest. The
derivation repeats with the roles reversed. Element $(i, j)$ of an $[m, n]$
matrix is preceded by $j$ complete columns of $m$ elements each, and then by
$i$ elements of its own column:
$$\mathrm{offset}_{\text{col}}(i, j) = i + j\,m.$$
For $R$ axes the stride of axis $k$ is the product of the sizes to its **left**,
$s_k = d_0 d_1 \cdots d_{k-1}$, with $s_0 = 1$. The earlier block
printed it: for the $[2, 3, 4]$ shape the column-major strides are $(1, 2, 6)$.
Both layouts have the same form, $\sum_k i_k s_k$; they differ only in which
list of strides is used. That is the observation the rest of the chapter builds
on.

There is nothing wrong with either layout, and the choice is a convention. The
cost of a convention is paid when two pieces of code disagree about it, and the
failure is silent: both read the same twelve numbers, each with a perfectly
consistent idea of where the elements are, and the results differ.

The simplest case is a matrix handed from one convention to the other. A
$[2, 3]$ matrix stored row-major is the buffer $1, 2, 3, 4, 5, 6$. Someone who
reads that buffer as a column-major matrix of shape $[3, 2]$ (column $j$ is the
three numbers starting at $3j$) obtains the transpose of the matrix, with
no error and no warning:

<!-- worked: arrays.misread_and_swap -->
A = [[1, 2, 3], [4, 5, 6]] [2, 3], stored row-major: [1, 2, 3, 4, 5, 6].

Read that buffer as a column-major matrix of shape [3, 2] (column j is buf[3j:3j+3]): [[1, 4], [2, 5], [3, 6]]. That is A^T = [[1, 4], [2, 5], [3, 6]]: True. Same bytes, other convention, transposed matrix.

Row-major product Y = X W^T with X [2, 3] = [[1, 2, 3], [4, 5, 6]] and W [4, 3] = [[1, 0, 2], [0, 1, 1], [1, 1, 0], [2, 0, 1]]: Y flat = [7, 5, 3, 5, 16, 11, 9, 14].

The column-major routine gemm of the text is asked for the [M, N] = [4, 2] product W X^T, given W's buffer and X's buffer unchanged: M = 4, N = 2, K = 3, A = W's buffer with trans_a = true and lda = K = 3, B = X's buffer with ldb = K = 3, and ldc = M = 4. Its output buffer is [7, 5, 3, 5, 16, 11, 9, 14].

Equal to Y's row-major buffer: True. No operand was copied or transposed.
<!-- /worked -->

The first paragraph is the general fact. A row-major array of shape $[m, n]$
and a column-major array of shape $[n, m]$ have the *same* memory when one is
the transpose of the other: the $(i, j)$ entry of the first and the $(j, i)$ entry of the
second sit at the same offset, $i n + j$ in the first and $j + i n$ in the
second. Memory cannot tell the two readings apart, only the convention can.

The second paragraph of the block turns this into a tool. The routine in
question is the standard **general matrix multiply**, written here
`gemm(trans_a, M, N, K, A, lda, B, ldb, C, ldc)`. It computes the column-major
$[M, N]$ matrix $\mathbf{C} = \mathrm{op}(\mathbf{A})\,\mathbf{B}$, where
$\mathrm{op}$ is the identity, or the transpose when `trans_a` is true. Every
buffer is column-major, and the **leading dimension** of a buffer (`lda`,
`ldb`, `ldc`) is the distance in elements between the starts of consecutive
columns of the stored matrix: element $(i, j)$ of a stored matrix sits at
offset $i + j \cdot \mathrm{ld}$, and $\mathrm{ld}$ must be at least the
number of rows stored (any extra rows are padding, or belong to a larger
matrix of which this is a block). `A` is stored as an $[M, K]$ matrix when
`trans_a` is false and as a $[K, M]$ matrix, read transposed, when it is true;
`B` is stored $[K, N]$ and `C` is $[M, N]$ with leading dimension `ldc` $\ge M$.
Suppose such a library routine, as the linear-algebra libraries inherited from
Fortran provide, is all we have, and we want a row-major product $\mathbf{Y} =
\mathbf{X}\mathbf{W}^\top$ with $\mathbf{X}$ of shape $[m, k]$ and
$\mathbf{W}$ of shape $[n, k]$. Take the transpose of the whole equation. By the
rule of Chapter 4, $(\mathbf{X}\mathbf{W}^\top)^\top = \mathbf{W}\mathbf{X}^\top$, so
$$\mathbf{Y}^\top = \mathbf{W}\,\mathbf{X}^\top.$$
Now read the buffers through the column-major convention. The buffer of
$\mathbf{X}$, row-major of shape $[m, k]$, *is* a column-major matrix of shape
$[k, m]$ with leading dimension $k$, that is $\mathbf{X}^\top$ exactly as
needed. The buffer of $\mathbf{W}$ read the same way is $\mathbf{W}^\top$, which
the routine is asked to transpose back with a flag. In the routine's letters: $M = n$, $N = m$, $K = k$; `A` is the buffer of
$\mathbf{W}$ with `trans_a` true and `lda` $= k$ (the buffer, read column-major,
is $[k, n]$); `B` is the buffer of $\mathbf{X}$ with `ldb` $= k$; and the
output has `ldc` $= M = n$ (its stored matrix has $M = n$ rows). The routine writes a
column-major $[n, m]$ matrix, $\mathbf{Y}^\top$, and a column-major $[n, m]$ matrix
has the same memory as a row-major $[m, n]$ one: that is $\mathbf{Y}$. No operand was
copied or transposed; the product was merely *stated* with the operands in the
other order, and the buffer that comes out is already in the row-major layout
the caller wanted. Every row-major program that calls a column-major library
does this, and every one that forgets it, or applies it to one operand and not
the other, gets a transposed or scrambled result of a perfectly valid shape.
The routine's own checks do not protect against this: a buffer-length and leading-dimension check passes for every shape once the leading dimension is chosen large enough to satisfy it (Exercise 10 asks for both mistakes), while the numbers are wrong.

## Strides: the general description

Both layouts have the offset $\sum_k i_k s_k$ with a different list of strides.
That suggests asking what other lists of strides could describe, and the answer
is the most useful idea in this chapter. Instead of fixing the layout and
deriving the strides, take the strides as the *definition* of how an array
reads its storage.

A **view** of a flat storage is four things: the storage itself (a flat list of
numbers), a **shape** $[d_0, \dots, d_{R-1}]$, a list of **strides**
$(s_0, \dots, s_{R-1})$ in elements, and a starting **offset** $o_0$. Element
$(i_0, \dots, i_{R-1})$ of the view is the storage cell
$$\mathrm{offset}(i_0, \dots, i_{R-1}) = o_0 + \sum_{k=0}^{R-1} i_k\, s_k. \tag{1}$$
An array "stored row-major" is the special case $o_0 = 0$ and
$s_k = \prod_{m>k} d_m$. Row-major and column-major are two particular choices;
equation (1) accepts any, and many of them correspond to something useful.

A small example with unusual strides, because it shows that nothing in (1)
requires the strides to look like either layout. Take the storage
$0, 10, 20, 30, 40, 50, 60$ and the view of shape $[2, 3]$, strides $(1, 2)$ and
starting offset 1:

<!-- worked: arrays.storage_view_gather -->
Storage [0, 10, 20, 30, 40, 50, 60], offset 1, shape [2, 3], strides (1, 2):

| index | offset | value |
|---|---:|---:|
| (0, 0) | 1 + 0*1 + 0*2 = 1 | 10 |
| (0, 1) | 1 + 0*1 + 1*2 = 3 | 30 |
| (0, 2) | 1 + 0*1 + 2*2 = 5 | 50 |
| (1, 0) | 1 + 1*1 + 0*2 = 2 | 20 |
| (1, 1) | 1 + 1*1 + 1*2 = 4 | 40 |
| (1, 2) | 1 + 1*1 + 2*2 = 6 | 60 |

Materialised in row-major order: [10, 30, 50, 20, 40, 60].
<!-- /worked -->

Stepping down a row moves one cell to the right in storage, stepping along a
row moves two, and the whole thing begins at cell 1. This is a valid view, and
reading it out in row-major order produces the six numbers in the last line.
Any program that can evaluate (1) can read it. (The checkpoint reader of the
case study does precisely this, and this very example is one of its tests.)

A table of strides for the shapes the book uses most:

<!-- worked: arrays.strides_table -->
| kind | shape | elements | row-major strides | column-major strides |
|---|---|---:|---|---|
| vector | [12] | 12 | (1) | (1) |
| matrix | [3, 4] | 12 | (4, 1) | (1, 3) |
| row of a batch | [1, 12] | 12 | (12, 1) | (1, 1) |
| three axes | [2, 3, 4] | 24 | (12, 4, 1) | (1, 2, 6) |
| images [B, C, H, W] | [2, 3, 4, 5] | 120 | (60, 20, 5, 1) | (1, 2, 6, 24) |
| sequences [B, T, D] | [2, 5, 8] | 80 | (40, 8, 1) | (1, 2, 10) |
| a size-1 axis | [3, 1, 4] | 12 | (4, 4, 1) | (1, 3, 3) |

Each row-major stride is the product of all the sizes to its right; the last is always 1.

| shape | elements | bytes at 4 bytes each | MiB |
|---|---:|---:|---:|
| [32, 3, 224, 224] | 4,816,896 | 19,267,584 | 18.375 |
| [8, 512, 4096] | 16,777,216 | 67,108,864 | 64.000 |
<!-- /worked -->

Read the second table with the batch of images in mind: $[32, 3, 224, 224]$ is
a little over eighteen mebibytes of 32-bit numbers (18.375 MiB). Hardly small,
and still a tiny batch compared with what training uses; the point is that
every operation below has to avoid copying it if it can.

## Views: transpose and slicing for free

Equation (1) is the key to doing a number of operations without touching the
numbers. Each of the following builds a new view over the *same* storage by
changing only the shape, the strides and the offset.

**Transpose.** The transpose of a matrix of shape $[m, n]$ has shape $[n, m]$
and $B_{ji} = A_{ij}$. If $A$ has strides $(s_0, s_1)$ then
$$B_{ji}: \quad o_0 + j\, s_1 + i\, s_0,$$
which is a view with shape $[n, m]$ and strides $(s_1, s_0)$: **swap the two
entries of the shape and the two entries of the strides**. Nothing else
changes; no number moves. For the $[3, 4]$ example the strides $(4, 1)$ become
$(1, 4)$. A trace of every element of that view, with the offset computed from
(1) and the element of the original it must equal (the last line of the block
says whether a view is **contiguous**, meaning its elements sit in storage in
reading order with no gaps; the term is defined with its test below):

<!-- worked: arrays.transpose_view -->
A = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]] [3, 4], strides (4, 1), storage [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23].

B = transpose view of A: shape [4, 3], strides (1, 4), same storage, no element moved. Every element of B:

| index in B | offset | B value | which element of A |
|---|---:|---:|---|
| (0, 0) | 0*1 + 0*4 = 0 | 0 | A[0][0] = 0 |
| (0, 1) | 0*1 + 1*4 = 4 | 10 | A[1][0] = 10 |
| (0, 2) | 0*1 + 2*4 = 8 | 20 | A[2][0] = 20 |
| (1, 0) | 1*1 + 0*4 = 1 | 1 | A[0][1] = 1 |
| (1, 1) | 1*1 + 1*4 = 5 | 11 | A[1][1] = 11 |
| (1, 2) | 1*1 + 2*4 = 9 | 21 | A[2][1] = 21 |
| (2, 0) | 2*1 + 0*4 = 2 | 2 | A[0][2] = 2 |
| (2, 1) | 2*1 + 1*4 = 6 | 12 | A[1][2] = 12 |
| (2, 2) | 2*1 + 2*4 = 10 | 22 | A[2][2] = 22 |
| (3, 0) | 3*1 + 0*4 = 3 | 3 | A[0][3] = 3 |
| (3, 1) | 3*1 + 1*4 = 7 | 13 | A[1][3] = 13 |
| (3, 2) | 3*1 + 2*4 = 11 | 23 | A[2][3] = 23 |

B read in its own row-major order: [0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23]. The offsets it visits: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11].

Copy of B, materialised: storage [0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23], strides (3, 1); contiguous: True (the view itself: False).
<!-- /worked -->

![Two 3 by 4 grids showing the same twelve cells of storage, coloured by storage row. Left: reading A in row-major order visits cells in the order 0 to 11 along each row. Right: reading the transpose B = A^T, of shape 4 by 3 with strides 1 and 4, visits the same cells in the order 0, 1, 2 down each column, jumping four storage positions between visits in the same row of B.](img/stride-views.svg)

The picture is the same table read twice. The left reading visits storage in
order. The right reading, the transpose, visits it with jumps of four: the
elements of one row of $B$ are a column of $A$, which are four cells apart in
storage. (Compare the offset list in the block: $0, 4, 8, 1, 5, 9, \dots$.)
The multiplication of Chapter 4 by a transposed weight, which the layer
section wrote as $\mathbf{X}\mathbf{W}^\top$, is exactly this: the kernel
reads $\mathbf{W}$ with the strides of its transpose and never builds
$\mathbf{W}^\top$ (the case study shows the line of code). For more than two
axes the same move generalises to **permuting the axes**: any reordering of the
axes is the same reordering of the shape and of the strides. Transposing the
last two axes of a batch of matrices swaps the last two entries of both lists.

**Slicing.** Take rows $a$ up to $b$ (exclusive) with step $c$ of a view, that
is the sub-array of rows $a, a+c, a+2c, \dots$ below $b$. Element $i$ of the slice
is row $a + c\,i$ of the original, so its offset is
$o_0 + (a + c\,i)\,s_0 + \dots = (o_0 + a\,s_0) + i\,(c\,s_0) + \dots$: the new
starting offset is $o_0 + a s_0$, the stride is multiplied by the step, and the
size is $\lceil (b - a)/c \rceil$. The other axes are untouched. Slicing several
axes at once repeats the rule per axis. The book's pseudocode writes half-open
ranges `a[i, 2:5]`; this is how they are executed. A range keeps its axis, even
when it selects one element, and an integer index drops it (see the
pseudocode page), so a view that must keep the axis, as in the table below, is
written with a range of length one: `A[1:2, :]` has shape $[1, 4]$, while
`A[1, :]` has shape $[4]$.

<!-- worked: arrays.slices_are_views -->
Storage of A (row-major [3, 4]): [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23].

| view | shape | strides | start offset | elements in order | contiguous |
|---|---|---|---|---|---|
| A | [3, 4] | (4, 1) | 0 | [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23] | yes |
| A[:, 1:3] | [3, 2] | (4, 1) | 1 | [1, 2, 11, 12, 21, 22] | no |
| A[1:3, :] | [2, 4] | (4, 1) | 4 | [10, 11, 12, 13, 20, 21, 22, 23] | yes |
| A[::2, :] | [2, 4] | (8, 1) | 0 | [0, 1, 2, 3, 20, 21, 22, 23] | no |
| A[:, ::2] | [3, 2] | (4, 2) | 0 | [0, 2, 10, 12, 20, 22] | no |
| A[1:2, :] (a row) | [1, 4] | (4, 1) | 4 | [10, 11, 12, 13] | yes |
| A[:, 2:3] (a column) | [3, 1] | (4, 1) | 2 | [2, 12, 22] | no |
| transpose(A) | [4, 3] | (1, 4) | 0 | [0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23] | no |

Writing through a view changes the original:

after setting element 1 of the column view A[:, 2:3] to -1, A = [[0, 1, 2, 3], [10, 11, -1, 13], [20, 21, 22, 23]] (was [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]).
<!-- /worked -->

Each line is a different way to look at the storage of $A$, and none moved a
number. A row is a view of shape $[1, 4]$ at offset 4, a column is a view of shape
$[3, 1]$ at offset 2 whose elements are four cells apart, and taking every other
row keeps the stride $4$ doubled to $8$.

**Views share storage, so writes go through.** The last line of the block is
the price of not copying. The column $A[:, 2]$ is not a separate array: writing
$-1$ to its middle element wrote it into $A$. That is what the pseudocode
conventions mean by "a slice is a **view**; where a copy is needed, the pseudocode
says `copy(...)`". Sharing is the reason views are cheap and the reason a
careless write corrupts something else. The same sentence is the definition of
an **alias**: two names for the same cells.

### Contiguity, and when a copy is needed

A view is **contiguous** when its strides are the row-major strides of its
shape, so that reading it in row-major order walks its storage in order, with
no gaps. (An axis of size 1 is never stepped along, so its stride is
irrelevant: the row $A[1, :]$ above is contiguous at stride 4 in front of its
single row.) The `contiguous` column of the last block is the test. A whole
array is contiguous, a block of whole rows is contiguous, a block of columns is
not, nor is every second row, nor is a transpose.

Contiguity matters because a great deal of code assumes it: a routine that
takes "a pointer and a length" and walks the numbers in order is correct only
for contiguous input. Two ways out are available when an array is not. The
routine can take strides as parameters and handle any view (slower to write, and
sometimes slower to run, for the reason below), or the caller can **materialise**
the view, copying its elements in row-major order into new contiguous
storage. The last line of the transpose block does exactly that: the copy of the
transpose has storage $0, 10, 20, 1, 11, \dots$, strides $(3, 1)$, and is
contiguous. Materialising costs one pass over the data and memory for a second
copy; a view costs nothing now and may cost on every later read.

### Why reading order matters

Memory is not read one number at a time. Hardware fetches a cache line, a
small fixed block of consecutive addresses (64 bytes is typical), whenever any
byte of it is needed, and the next numbers in that line then arrive almost for
free. A program that walks memory with stride 1 uses every number it fetched; a
program that jumps by a large stride uses one number from each line. Counting
only the lines touched, with 4-byte numbers and 64-byte lines (an assumption about
typical hardware, not a law; Volume IV measures real machines):

<!-- worked: arrays.cache_lines -->
Reading 64 numbers of 4 bytes starting at a line boundary, with memory fetched in lines of 64 bytes (16 numbers). Distinct lines touched, by stride in elements:

| stride | lines touched | useful numbers per line |
|---|---:|---:|
| 1 | 4 | 16.0 |
| 2 | 8 | 8.0 |
| 4 | 16 | 4.0 |
| 8 | 32 | 2.0 |
| 16 | 64 | 1.0 |
| 17 | 64 | 1.0 |
| 64 | 64 | 1.0 |

One row of a [64, 64] matrix (stride 1) touches 4 lines; one column (stride 64) touches 64.
<!-- /worked -->

A row of a $[64, 64]$ row-major matrix is 4 lines; a column is 64, sixteen times
as many for the same number of elements, and every line fetched for the column
is almost entirely wasted. This is the practical content of the layout choice
and of the transpose view: a view makes the transpose free to *create*, but
reading it row by row is reading the original by column. An algorithm that
traverses the transposed array many times is often cheaper to run after one
materialising copy; an algorithm that reads it once is cheaper without. The
count of multiply-adds of Chapter 4 does not see this difference at all.

## Reshape, and the bug of reshaping when you meant to transpose

**Reshape** gives the same elements a new shape: a $[2, 3]$ array becomes
$[3, 2]$, or $[6]$, or $[6, 1]$. The definition is this: the elements of the
array, read in row-major order, are written back in row-major order into the new
shape. The count of elements must be the same ($2 \cdot 3 = 3 \cdot 2$). In
words, reshaping keeps the **line of numbers** and moves only the places at
which the line is cut into rows.

For a contiguous array the line of numbers is the storage, so reshape needs no
work: the new view has the same storage, the same offset and the row-major strides of the
new shape. That is the whole content of the term **reshape-and-view**: when it
can, reshape returns a view of the same storage. When the source is not
contiguous the elements are not in order in storage, and the simple answer is
to copy them into order first and then take the view. Either way the result
reads the same.

**Transposing is not reshaping.** Both can turn a $[2, 3]$ array into a $[3, 2]$
one, which is the root of the bug. Compare them:

<!-- worked: arrays.reshape_vs_transpose -->
A = [[1, 2, 3], [4, 5, 6]] [2, 3], storage [1, 2, 3, 4, 5, 6].

| operation | shape | result | storage order | work |
|---|---|---|---|---|
| reshape A to [3, 2] | [3, 2] | [[1, 2], [3, 4], [5, 6]] | [1, 2, 3, 4, 5, 6] | view |
| transpose A | [3, 2] | [[1, 4], [2, 5], [3, 6]] | [1, 4, 2, 5, 3, 6] | view (strides (1, 3)) |

The two results have the same shape and different contents: True.

Flattening the transposed view needs a copy (copy): reshape(transpose(A), [6]) = [1, 4, 2, 5, 3, 6], whereas reshape(A, [6]) = [1, 2, 3, 4, 5, 6].

Multiply each [3, 2] result by v = (1, 10): reshape gives [21, 43, 65], transpose gives [41, 52, 63].
<!-- /worked -->

The two results have the same shape and different contents. Reshape cut the line
$1, 2, 3, 4, 5, 6$ into rows of two, giving $\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}$.
Transpose kept every row of the original together as a *column*, giving
$\begin{pmatrix}1&4\\2&5\\3&6\end{pmatrix}$. A shape check cannot tell them
apart, and neither can the program that uses them: multiplying each by the same
$(1, 10)$ gave different answers in the block, and both ran.

A rule to remember: **reshape only splits or merges neighbouring axes.** It
never moves an axis past another. Any time an axis changes its position, that is
a permutation (a transpose of a general kind), and it must be written as one.

### The attention-heads bug, in numbers

The bug is rarely a $[2, 3]$ array. The most common real instance is
splitting the width of a sequence into heads. A transformer layer (Volume V)
holds, for a sequence of $T$ positions, a matrix $\mathbf{X}$ of shape $[T, D]$,
and wants to treat its width $D = H \cdot D_h$ as $H$ independent heads of
width $D_h$. The wanted result has shape $[H, T, D_h]$: for each head, the
sequence of that head's slice at every position. Here is a tiny version, with
$T = 2$, $H = 2$, $D_h = 3$ and entry $(t, c)$ equal to $10t + c$ so that the
value shows where it came from:

<!-- worked: arrays.split_heads -->
x is a sequence of T = 2 positions of width D = 6; entry (t, c) holds 10 t + c: x = [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]], storage [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]. Split D into H = 2 heads of Dh = 3.

Right: reshape to [T, H, Dh] = [2, 2, 3] (a view, strides (6, 3, 1)), then swap the first two axes to [H, T, Dh] (a view, strides (3, 6, 1)).

| head | rows of the head (one row per position) |
|---:|---|
| 0 | [[0, 1, 2], [10, 11, 12]] |
| 1 | [[3, 4, 5], [13, 14, 15]] |

Wrong: reshape x directly to [H, T, Dh] = [2, 2, 3] (a view, strides (6, 3, 1)):

| head | rows of the head |
|---:|---|
| 0 | [[0, 1, 2], [3, 4, 5]] |
| 1 | [[10, 11, 12], [13, 14, 15]] |

The wrong version hands head 0 the first two groups of position 0 (entries 0 to 5 split in threes) instead of the first group of every position.

Merging back: swap the axes back to [T, H, Dh] (strides (6, 3, 1), which is the row-major strides of [T, H, Dh]: True), then reshape to [T, D] (a view); the result equals x: True. Reshaping the [H, T, Dh] view itself would be a copy (contiguous False), and its elements [[0, 1, 2, 10, 11, 12], [3, 4, 5, 13, 14, 15]] are not x.
<!-- /worked -->

The correct recipe is two steps, each of a kind this chapter has defined.
First **split the last axis**: reshape $[T, D]$ to $[T, H, D_h]$, which cuts the
line of each row into groups of $D_h$ and does not move any axis. Then
**permute** the first two axes to get $[H, T, D_h]$, a view with swapped strides
that moves nothing. The wrong recipe is the one-step reshape of $[T, D]$ to
$[H, T, D_h]$ that has the same final shape. It cuts the line $0, 1, 2, 3, 4, 5, 10, \dots$
into rows of three and calls the first two rows "head 0": position 0's first
head, then position 0's *second* head, Head 0 is now all of position 0 (its two
real heads, laid out as if they were two positions) and head 1 is all of
position 1. The shapes agree, every later operation runs, and
the model learns from scrambled data. This is the bug the chapter warns about: not
a crash, a model that trains to something worse than it should.

The last paragraph of the block shows the way back, merging the heads into
the width again: permute to $[T, H, D_h]$ and reshape to $[T, D]$. It also shows that
reshaping the permuted $[H, T, D_h]$ view directly would have been a copy and
would have given $[[0,1,2,10,11,12],[3,4,5,13,14,15]]$, not $x$.

### Which reshapes are free?

When a reshape of a non-contiguous view needs a copy, and when it does not, has
a clean answer for the two operations that matter. **Splitting one axis** into
two is always free. If an axis has size $d = a \cdot b$ and stride $s$, then
element $i$ of the axis, written $i = p\,b + q$ with $0 \le q < b$, has offset
$i\,s = p\,(b\,s) + q\,s$, so the axis is replaced by two axes of sizes $(a, b)$ and
strides $(b s, s)$. Nothing about the other axes or about contiguity of the
original is needed. This is the first step of the heads recipe, and works on
sliced and strided inputs too:

<!-- worked: arrays.split_axis_rule -->
A non-contiguous view W = base[::2, :] of a [4, 6] array: shape [2, 6], strides (12, 1), contiguous: False.

Splitting its last axis (size 6, stride 1) into (2, 3) needs no copy. The new strides are (Dh*1, 1) = (3, 1) for the pieces, the first axis keeps stride 12: shape [2, 2, 3], strides (12, 3, 1).

Its elements equal those of a copied-and-reshaped array: True. Element (1, 1, 2) sits at offset 17 of the original storage.
<!-- /worked -->

**Merging two adjacent axes** into one is free only when the axes fit together
in storage. Merge axes $k$ and $k + 1$, sizes $d_k$ and $d_{k+1}$, strides $s_k$
and $s_{k+1}$. A merged index $j = i_k d_{k+1} + i_{k+1}$ has to correspond to
offset $j\,s_{k+1}$ (one stride, the inner one), so we need
$i_k s_k + i_{k+1} s_{k+1} = (i_k d_{k+1} + i_{k+1})\,s_{k+1}$ for every pair. That
holds for all pairs of indices exactly when
$$s_k = d_{k+1}\, s_{k+1}, \tag{2}$$
provided both axes have size at least 2. (An axis of size 1 only ever has
index 0, so its stride is never used and the condition cannot fail for it: such
an axis is dropped first, and the merged axis takes the stride of the other
axis.)
In a contiguous array this is the definition of the row-major strides, so
merging neighbours is always free there. In a transposed or sliced array it
usually fails, and the block shows the cases:

<!-- worked: arrays.merge_rule -->
| array | shape | strides | axes merged | s_k vs d_(k+1) s_(k+1) | result |
|---|---|---|---|---|---|
| contiguous [2, 3, 4] | [2, 3, 4] | (12, 4, 1) | 0, 1 | 12 vs 3*4 = 12 | view, merged stride 4 |
| contiguous [2, 3, 4] | [2, 3, 4] | (12, 4, 1) | 1, 2 | 4 vs 4*1 = 4 | view, merged stride 1 |
| swapped axes [3, 2, 4] | [3, 2, 4] | (4, 12, 1) | 0, 1 | 4 vs 2*12 = 24 | copy |
| swapped axes [3, 2, 4] | [3, 2, 4] | (4, 12, 1) | 1, 2 | 12 vs 4*1 = 4 | copy |
| A[:, 0:2] of [3, 4] | [3, 2] | (4, 1) | 0, 1 | 4 vs 2*1 = 2 | copy |
| A[::2, :] of [3, 4] | [2, 4] | (8, 1) | 0, 1 | 8 vs 4*1 = 4 | copy |
| A[:, 2:3] of [3, 4] | [3, 1] | (4, 1) | 0, 1 | an axis has size 1: drop it | view, merged stride 4 |

Every merge the rule allows reproduces the row-major reading exactly: True.
<!-- /worked -->

Rule (2) says "the outer axis steps over exactly the whole inner axis". A block
of columns, every second row, and the swapped axes all violate it, and for
those the reshape is a copy. This is the full content of the guidance above:
split anything, merge only what is adjacent in memory, and expect a copy when
you ask for more. The last row of the block is the size-1 case: the axis of
size 1 is dropped and the merged axis keeps the stride 4 of the other.

The two rules combine into one algorithm for any reshape, given in the section
on algorithms: it matches old axes to new axes by their running products, and
takes a view when every group of old axes can be merged by rule (2).

## Broadcasting

The fourth question of the problem was how to add a bias vector of length $D$ to
every row of a $[T, D]$ matrix. In Chapter 4's notation the book wrote
$\mathbf{1}\mathbf{b}^\top$ and said "add $\mathbf{b}$ to every row". Doing it
literally means building a $[T, D]$ matrix with $T$ copies of $\mathbf{b}$,
which is wasteful: the copies hold no new information. **Broadcasting** is the
rule that lets an operation treat a smaller array as if it were stretched to
the shape of a larger one, without building the stretched array.

### The rule

Two shapes **broadcast** together to a result shape by these steps:

1. **Align from the right.** Write the two shapes one above the other with their
   *last* axes together. If one has fewer axes, pad it on the left with axes of
   size 1.
2. **Compare axis by axis.** In each position the two sizes must either be equal
   or one of them must be 1. Otherwise the shapes do not broadcast: an error.
3. **The result takes the larger size** in each position (the size of the
   non-1 axis, or 1 if both are 1).

An axis of size 1 is "stretched": its single entry is used for every index along
the result's axis. Both shapes may be stretched in different axes, which is
how an outer sum arises. Cases, including the failures:

<!-- worked: arrays.broadcast_shapes -->
| shape a | shape b | result | note |
|---|---|---|---|
| [3, 4] | [4] | [3, 4] | ok |
| [3, 4] | [3, 1] | [3, 4] | ok |
| [3, 1] | [1, 4] | [3, 4] | ok |
| [2, 3, 4] | [3, 4] | [2, 3, 4] | ok |
| [2, 3, 4] | [2, 1, 4] | [2, 3, 4] | ok |
| [8, 1, 6, 1] | [7, 1, 5] | [8, 7, 6, 5] | ok |
| [] | [2, 3] | [2, 3] | ok |
| [3, 4] | [3] | error | axis 1 from the right: 4 vs 3 |
| [2, 3] | [4, 3] | error | axis 2 from the right: 2 vs 4 |
| [5, 1] | [5] | [5, 5] | ok |
<!-- /worked -->

Read the first row as the bias example: $[3, 4]$ and $[4]$ align the two sizes of 4, the
second is padded to $[1, 4]$, the 1 is stretched to 3, and the result is $[3, 4]$.
The case $[8, 1, 6, 1]$ with $[7, 1, 5]$ is padded to $[1, 7, 1, 5]$ and gives
$[8, 7, 6, 5]$, each shape stretching where the other is larger. The failures say
at which position from the right the sizes disagree: $[3, 4]$ with $[3]$
compares $4$ with $3$. That error is the *good* outcome of the rule, and it is the
reason alignment is from the right: the last axis is where vectors like the bias
live.

### No copies: stride zero

Broadcasting is a view too. To stretch an axis of size 1 to size $n$, give it
**stride 0**: every index along the axis then reads the same cell,
$\mathrm{offset} = o_0 + i\cdot 0 + \dots$. New axes padded on the left get stride
0 as well. So the bias of shape $[3]$ seen as $[2, 3]$ has strides $(0, 1)$, and
elementwise addition runs over the stretched shape with one stored copy:

<!-- worked: arrays.broadcast_values -->
X = [[1, 2, 3], [4, 5, 6]] [2, 3] and bias = [10, 20, 30] [3]. The bias aligned to the right axis of X is stretched over rows: as a view of shape [2, 3] its strides are (0, 1) (stride 0 on the stretched axis, the original stride 1 on the other), over the same three numbers of storage.

| index (i, j) | offset into bias | bias value | X value | sum |
|---|---:|---:|---:|---:|
| (0, 0) | 0 | 10 | 1 | 11 |
| (0, 1) | 1 | 20 | 2 | 22 |
| (0, 2) | 2 | 30 | 3 | 33 |
| (1, 0) | 0 | 10 | 4 | 14 |
| (1, 1) | 1 | 20 | 5 | 25 |
| (1, 2) | 2 | 30 | 6 | 36 |

X + bias = [[11, 22, 33], [14, 25, 36]] [2, 3].

Column [2, 1] = [100, 200] added to X: [[101, 102, 103], [204, 205, 206]]. Mixing the two kinds, shapes [2, 1] and [3] broadcast to [2, 3]: [[110, 120, 130], [210, 220, 230]].

Outer sum: [3, 1] + [1, 4] gives [3, 4]: [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]].
<!-- /worked -->

![Three 3 by 4 grids side by side: a column a of shape 3 by 1 with entries 10, 20, 30 stretched across four columns, a row b of shape 1 by 4 with entries 1, 2, 3, 4 stretched down three rows, and their elementwise sum, a 3 by 4 grid whose entry in row i, column j is a_i + b_j.](img/broadcast-stretch.svg)

The picture is the outer-sum case of the block: the solid cells are stored
numbers and the faded dashed cells are the same number read again. The result
has twelve entries from seven stored numbers. How many stored numbers each
broadcast actually touches:

<!-- worked: arrays.broadcast_cost -->
Symbol values: T = 7 and D = 5 in the first three rows; B = 2 and T = 4 in the last row.

| broadcast | view strides | elements seen | stored numbers touched |
|---|---:|---:|---:|
| bias [D] over [T, D] | (0, 1) | 35 | 5 |
| column [T, 1] over [T, D] | (1, 0) | 35 | 7 |
| scalar [] over [T, D] | (0, 0) | 35 | 1 |
| [B, 1, T] over [B, T, T] | (4, 0, 1) | 32 | 8 |
<!-- /worked -->

The stretched view sees $T D$ elements and touches $D$ numbers (or $T$, or 1):
the work is the elementwise add over the result, the *memory* is the small
operand.

### The explicit-broadcast policy

Broadcasting is convenient and dangerous in the same way: it makes mismatched
shapes work. The rule never reports an error for $[4, 1]$ against $[4]$,
because it is a legal broadcast to $[4, 4]$. A loss function that subtracts a
column of predictions of shape $[4, 1]$ from a vector of targets of shape $[4]$
therefore returns a perfectly valid $[4, 4]$ table of every prediction against
every target, and averages it:

<!-- worked: arrays.silent_outer_diff -->
Predictions p of shape [4, 1] = [1, 2, 3, 4] and targets t of shape [4] = [1, 2, 3, 4]; they agree exactly.

The shapes broadcast to [4, 4], so p - t is the table of all 16 differences:

[[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1], [3, 2, 1, 0]]

Mean squared error computed that way: 2.5. With p reshaped to [4] first: 0.0.
<!-- /worked -->

The predictions equal the targets exactly and the error computed is 2.5, not
zero; the program ran, the shapes were legal and a model trained against this
loss would be chasing the wrong number. The remedy is not to avoid
broadcasting but to be explicit about it, which is the policy of the book's
pseudocode:

- An operator on whole arrays applies **only** to arrays of equal shape, unless
  the text states the broadcast: "add `bias: [D]` to every row of `x: [T, D]`".
  The reader checks the stated broadcast against the rule above.
- A stretched axis is written with its size 1 spelled out, `[T, 1]` and `[1, D]`,
  and never left to be inferred by the right-alignment padding. The padded form
  is the one that produces the $[4, 4]$ surprise.
- Results of a broadcast are given their full shape in the text, so that a
  wrong shape is visible at a glance.

Engines make the same choice in a different way: a kernel is written for one
broadcast pattern and names it, as the case study shows, so a mismatched
shape is a mismatched kernel and not a valid program that computes a different
quantity.

## The batch dimension

Chapter 4's layer section stacked $B$ input vectors as the rows of a matrix and
computed all of them with one matrix product. That is the **batch dimension**:
an extra, normally first, axis of size $B$ that indexes independent examples, so
that a tensor of shape $[B, \dots]$ holds $B$ examples of shape $[\dots]$. An
operation is *batched* when its result for the whole tensor equals the result
for each example separately, stacked. Two reasons make this the way machine
learning works, one about hardware and one about learning; the second is
developed in later volumes and only previewed here.

**Hardware.** A weight matrix has to be fetched from memory to be used. A
product with one example reads all of $\mathbf{W}$ to do $d_{\text{out}}\,d_{\text{in}}$
multiply-adds, a ratio of about one multiply-add per number fetched. A product
with $B$ examples reads $\mathbf{W}$ once and does $B\,d_{\text{out}}\,d_{\text{in}}$
multiply-adds: the arithmetic per number fetched is $B$ times higher, and a device whose arithmetic is much faster than its memory needs that ratio. The
cost formula of Chapter 4 ($m n k$ multiply-adds) counts the operations and does not
see the fetch, which is why the batch dimension pays: many small independent
problems become one large one.

**Learning.** A training step estimates what to change from a sample of examples
rather than a single one, so the quantity being computed is naturally an
average over the batch axis. Volume II explains why this works.

### Where the batch axis goes

The first axis, and the reason is in the derivation: the stride of the batch
axis is the size of one example, $s_0 = d_1 \cdots d_{R-1}$, and every example
is a contiguous block in memory. Example $b$ is the view with offset
$b\,s_0$ and the remaining shape and strides, so it can be passed on to any
routine that expects one contiguous example without copying. And the first
size $d_0$ appears in no stride, so the same code works for $B = 1$ and
$B = 32$.

For a batch of sequences, $[B, T, D]$, a token $(b, t)$ is a row of $D$ contiguous
numbers at offset $(b T + t) D$. The linear layer of Chapter 4 does not care
about the sequence structure: each token is an independent row. Merging the
first two axes, $[B, T, D] \to [B T, D]$, satisfies rule (2) ($s_0 = T D = T \cdot s_1$),
so it is free, and the whole batch is one matrix product with $m = B T$ rows,
after which a reshape back restores $[B, T, D]$:

<!-- worked: arrays.image_layouts -->
Image batch with B = 2, C = 3, H = 2, W = 2; element value = its NCHW offset.

NCHW: shape [2, 3, 2, 2], strides (12, 4, 2, 1). NHWC: shape [2, 2, 2, 3], strides (12, 6, 3, 1).

The NHWC array is the same picture stored channels-last: its element (b, h, w, c) is the NCHW element (b, c, h, w). The three channel values of the pixel at image 1, row 1, column 0:

| channel | NCHW offset | NCHW value | NHWC offset | NHWC value |
|---|---:|---:|---:|---:|
| 0 | 14 | 14 | 18 | 14 |
| 1 | 18 | 18 | 19 | 18 |
| 2 | 22 | 22 | 20 | 22 |

In NCHW the three channels of one pixel are 4 apart; in NHWC they are adjacent.

Viewing the NCHW storage as NHWC without moving anything is the permuted view with axes (0, 2, 3, 1): shape [2, 2, 2, 3], strides (12, 2, 1, 4), contiguous False; its elements equal the NHWC array's: True. So a real NHWC layout needs a copy (True contiguous afterwards).

Sequence batch B = 2, T = 3, D = 4: shape [2, 3, 4], strides (12, 4, 1). Merging the first two axes to [B*T, D] = [6, 4] is a view; the row of token (b, t) is b*T + t:

| b | t | row b*T + t | offset of (b, t, 0) | value of row's first entry |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 4 | 4 |
| 0 | 2 | 2 | 8 | 8 |
| 1 | 0 | 3 | 12 | 12 |
| 1 | 1 | 4 | 16 | 16 |
| 1 | 2 | 5 | 20 | 20 |
<!-- /worked -->

The last table is that merge in numbers: the row of token $(b, t)$ in the
$[B T, D]$ view is $bT + t$, at offset $(bT + t) D$, the same cell as before.

### Images: channels first or channels last

The first part of the block above is a batch of images in the two layouts in
common use. In **NCHW** (the book's $[B, C, H, W]$; the N of the name counts the examples in the
batch, which the rest of the book calls $B$) the three
channels of a pixel are $H W$ cells apart, and each channel is a contiguous
plane. In **NHWC** ($[B, H, W, C]$) the channels of a pixel are adjacent, and
each pixel is a short contiguous vector. Which is better depends on the
operation: a convolution that treats each channel plane separately reads NCHW
in order, an operation across the channels of one pixel reads NHWC in order.
Neither is wrong. As with row- and column-major, the cost is the conversion
between them, and it is a true copy: the permuted view with axes $(0, 2, 3, 1)$ has
the NHWC elements but strides $(12, 2, 1, 4)$, it is not contiguous (the block says
so), and a routine that expects NHWC storage needs the copy.

The same applies to sequences: $[B, T, D]$ and $[T, B, D]$ both occur, and
exchange of them is a permutation with the same cost and the same
trap. Code that handles several models must say which layout every tensor
has, in a name or a comment, because the shape alone, as the heads bug showed, does not.

## The algorithms

Pseudocode for the operations above. A **view** is the record of equation (1).
Where the conventions say a slice is a view, these are the functions that make
it one.

```text
record View:
    storage: array<real>[S]       # shared flat numbers
    offset:  int                  # o_0
    shape:   array<int>[R]
    strides: array<int>[R]        # in elements, may be 0 or larger than the row

function row_major_strides(shape: array<int>[R]) -> array<int>[R]:
    strides = zeros(int, [R])
    run = 1
    for k in R-1 down to 0:
        strides[k] = run               # product of the sizes to the right
        run = run * shape[k]
    return strides

function offset_of(v: View, index: array<int>[R]) -> int:
    require len(index) == R, "index length differs from the number of axes"
    off = v.offset
    for k in 0 .. R:
        require 0 <= index[k] < v.shape[k], "index out of bounds"
        off = off + index[k] * v.strides[k]
    return off

function transpose_view(v: View, a: int, b: int) -> View:
    require 0 <= a < R and 0 <= b < R, "axis out of range"
    shape = copy(v.shape); strides = copy(v.strides)
    swap shape[a], shape[b]; swap strides[a], strides[b]
    return View(v.storage, v.offset, shape, strides)      # no number moves

function slice_view(v: View, start, stop, step: array<int>[R]) -> View:
    shape = zeros(int, [R]); strides = zeros(int, [R]); off = v.offset
    for k in 0 .. R:
        require step[k] >= 1 and 0 <= start[k] <= stop[k] <= v.shape[k], "bad slice"
        shape[k]   = (stop[k] - start[k] + step[k] - 1) // step[k]   # integer division
        strides[k] = v.strides[k] * step[k]
        off        = off + start[k] * v.strides[k]
    return View(v.storage, off, shape, strides)

function is_contiguous(v: View) -> bool:
    want = row_major_strides(v.shape)
    for k in 0 .. R:
        if v.shape[k] != 1 and v.strides[k] != want[k]:
            return false
    return true

function materialize(v: View) -> View:        # copy into new contiguous storage
    out = zeros(real, [numel(v.shape)])
    n = 0
    for index in indices(v.shape):            # every index tuple, in row-major order
        out[n] = v.storage[offset_of(v, index)]
        n = n + 1
    return View(out, 0, copy(v.shape), row_major_strides(v.shape))

function reshape(v: View, new_shape: array<int>[Q]) -> View:
    require numel(new_shape) == numel(v.shape), "element counts differ"
    if not is_contiguous(v):
        v = materialize(v)                    # a view is possible only in special cases
    return View(v.storage, v.offset, new_shape, row_major_strides(new_shape))

function reshape_view(v: View, new_shape: array<int>[Q]) -> (ok: bool, w: View):
    # A view with the new shape if the split and merge rules allow one, else ok = false.
    require numel(new_shape) == numel(v.shape), "element counts differ"
    require all dimensions of v.shape and new_shape are >= 1, "empty axes not handled"
    R = len(v.shape)
    old_d = [v.shape[k] for k in 0 .. R if v.shape[k] != 1]        # axes of size 1 are dropped:
    old_s = [v.strides[k] for k in 0 .. R if v.shape[k] != 1]      # their stride is never used
    new_strides = zeros(int, [Q])
    k = 0                                     # next unused old axis
    j = 0                                     # next unused new axis
    while j < Q:
        if new_shape[j] == 1:
            new_strides[j] = 1                # never stepped along, any value will do
            j = j + 1
            continue
        # a group: old axes k .. k2 and new axes j .. j2 whose sizes have the same product
        k2 = k; prod_old = old_d[k]
        j2 = j; prod_new = new_shape[j]
        while prod_old != prod_new:
            if prod_old < prod_new:
                k2 = k2 + 1; prod_old = prod_old * old_d[k2]
            else:
                j2 = j2 + 1; prod_new = prod_new * new_shape[j2]
        for a in k .. k2:                     # the old axes of the group must merge, by rule (2)
            if old_s[a] != old_d[a + 1] * old_s[a + 1]:
                return (false, v)
        inner = old_s[k2]                     # the group is read with the innermost old stride ...
        for b in j2 down to j:                # ... and the new axes split it, row-major
            new_strides[b] = inner
            inner = inner * new_shape[b]
        k = k2 + 1
        j = j2 + 1
    return (true, View(v.storage, v.offset, new_shape, new_strides))

function broadcast_shape(a: array<int>[P], b: array<int>[Q]) -> array<int>[max(P, Q)]:
    R = max(P, Q)
    out = zeros(int, [R])
    for k in 1 ..= R:                         # axes counted from the right
        x = a[P - k] if k <= P else 1
        y = b[Q - k] if k <= Q else 1
        require x == y or x == 1 or y == 1, "shapes do not broadcast"
        out[R - k] = max(x, y)
    return out

function broadcast_view(v: View, shape: array<int>[R]) -> View:
    lead = R - len(v.shape)
    require lead >= 0, "target has fewer axes"
    strides = zeros(int, [R])                 # padded axes: stride 0
    for k in 0 .. len(v.shape):
        if v.shape[k] == shape[lead + k]:
            strides[lead + k] = v.strides[k]
        else:
            require v.shape[k] == 1, "only an axis of size 1 can stretch"
            strides[lead + k] = 0
    return View(v.storage, v.offset, shape, strides)
```

Every function checks its assumptions with `require`. The two copy-producing
functions are `materialize` and, when it must, `reshape`; every other function
returns in time independent of the number of elements. `reshape` is deliberately
conservative: it copies whenever the input is not contiguous, even when
rules (2) and the axis split would have allowed a view, which is safe and keeps
the algorithm short. `reshape_view` is the complete rule. Try it on a
contiguous $[2, 3, 4]$ made into $[6, 4]$: the old sizes $2, 3, 4$ and the new
sizes $6, 4$ group as $\{2, 3\} \to \{6\}$ and $\{4\} \to \{4\}$; the first group
needs $s_0 = d_1 s_1$, that is $12 = 3 \cdot 4$, which holds, so the result is
a view with strides $(4, 1)$. On the swapped-axes view of the merge block
($[3, 2, 4]$, strides $(4, 12, 1)$) the same request groups $\{3, 2\} \to \{6\}$
and fails on $4 = 2 \cdot 12$, so it reports `ok = false` and a caller copies
with `materialize`. A real implementation calls `reshape_view` first and
`reshape` only when it fails.

## Making it work

**Say which layout a tensor is in.** The shape of a tensor does not record
whether its axes are $[B, C, H, W]$ or $[B, H, W, C]$, nor whether a buffer is
row- or column-major. Name the layout where a tensor is created and at every
boundary (a file, a kernel call, a library), and test a conversion by a round
trip: permute and permute back must return the original bitwise.

**A test must see the bug.** The reshape-for-transpose bug and the silent
broadcast both preserve every shape. A test on shapes cannot find them; a test
on *values* with distinguishable numbers (like $10t + c$ above, where each
number names its position) finds them immediately. Use numbers that name their
place whenever a test checks a layout.

**Also use dimensions that expose what the real ones hide.** A test with square
matrices accepts a transposed buffer. A test with every size a multiple of 64
never notices an alignment assumption. Make the sizes distinct and awkward
(2, 3, 5), which is the reason the chapter's examples use $[2, 3]$ and $[3, 4]$ and not
$[4, 4]$.

**Strides are in elements, offsets can be large.** A stride or an offset must
be stored in an integer big enough for the largest array: with 32-bit
integers an array of more than $2^{32}$ elements cannot be indexed, and the
failure, a wrapped offset, is silent. Check the element count when the shape is
created.

**Aliasing.** A write through a view changes everything that shares the storage.
Where a function must not change its input, it copies; where it may, the
conventions mark the argument `inout`. A function that returns a view of its
input extends the input's lifetime: the view is only valid while the storage is.

**A slice of a buffer can be illegal.** Some devices bind a slice of a buffer to
a kernel only if the slice starts at a multiple of a fixed number of bytes.
Whether a row boundary satisfies that depends on the row width, which is
arithmetic of exactly this chapter, and the case study's story is about it.

## Where this appears in brain

Every kernel in brain receives its tensors as flat buffers plus a small block of
parameters, and computes the offset formula of this chapter by hand, one index
at a time. Reading a few of them shows the whole chapter.

**The linear layer reads a transpose without building one.**
`crates/kernels/wgsl/matmul.wgsl` documents `x : [M, K]`, `W : [N, K]` (both
row-major) and `out[m, n] = sum_k x[m, k] * W[n, k]`. Each thread takes a flat
output number `idx`, and recovers its index with `row = idx / p.n` and
`col = idx % p.n`: the inverse of the offset map of the derivation above. It then
forms `x_base = row * p.k` and `w_base = col * p.k` (offsets of the starts of two
rows) and loops `x[x_base + i] * w[w_base + i]`. The weight is stored as
$[N, K]$ and the product wants $\mathbf{W}^\top$ of shape $[K, N]$; the kernel
reads the weight with the strides of that transpose, namely $1$ along $K$ and $K$
along $N$, and no transposed copy exists. That is Chapter 4's "the transpose is
usually free" in one line of WGSL.

**A layout change is a gather through the offset formula.**
`crates/kernels/wgsl/nchw_nlc.wgsl` converts an image batch from channels-first
to $[N, L = HW, C]$. Its header gives the decomposition of the output index and
the source offset directly: `n = idx/(hw*c); r0 = idx % (hw*c); l = r0/c; ch = r0 % c`
and `y[idx] = x[(n*c + ch)*hw + l]`. Those are the repeated divisions
of the inverse map, followed by equation (1) with the strides of the input.
The header also states the test it must pass: its inverse `nlc_nchw`
composed with it returns the input *bitwise*, because the operation is a
permutation. The same gather on a small size, with $C = 3$ and $HW = 4$, agrees with the
permuted view, which is the in-chapter check that the kernel and the strided
read are one operation:

<!-- worked: arrays.kernel_permute -->
Layout change [N, C, HW] to [N, HW, C] with N = 2, C = 3, HW = 4. One output element per index idx, which decomposes as n = idx / (HW C), r = idx mod (HW C), l = r / C, ch = r mod C and is read from (n C + ch) HW + l of the input. The first 15 outputs:

| idx | n | l | ch | source offset |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 4 |
| 2 | 0 | 0 | 2 | 8 |
| 3 | 0 | 1 | 0 | 1 |
| 4 | 0 | 1 | 1 | 5 |
| 5 | 0 | 1 | 2 | 9 |
| 6 | 0 | 2 | 0 | 2 |
| 7 | 0 | 2 | 1 | 6 |
| 8 | 0 | 2 | 2 | 10 |
| 9 | 0 | 3 | 0 | 3 |
| 10 | 0 | 3 | 1 | 7 |
| 11 | 0 | 3 | 2 | 11 |
| 12 | 1 | 0 | 0 | 12 |
| 13 | 1 | 0 | 1 | 16 |
| 14 | 1 | 0 | 2 | 20 |

The gathered output equals the row-major reading of the permuted view with axes (0, 2, 1): True. The permuted view's strides are (12, 1, 4).
<!-- /worked -->

The layout conversion is a real copy, not a view, because the kernel's consumers
need contiguous input.

**Broadcasting is one named kernel per pattern.**
`crates/kernels/wgsl/broadcast_add_hw.wgsl` adds `a : [N, C, H, 1]` and
`b : [N, C, 1, W]` into `y : [N, C, H, W]` with
`y = a[nc * H + h] + b[nc * W + w]`: the stride-0 view of the broadcast
section written out as index arithmetic. On a small size the formula and the
general broadcast of the earlier section agree, and the stretched views show
the stride 0 on the axes of size 1:

<!-- worked: arrays.broadcast_hw_kernel -->
a [1, 2, 2, 1] = [1, 2, 3, 4], b [1, 2, 1, 3] = [10, 20, 30, 40, 50, 60]; the broadcast shape is [1, 2, 2, 3].

As stretched views: a has strides (4, 2, 1, 0), b has strides (6, 3, 0, 1).

Result, flat: [11, 21, 31, 12, 22, 32, 43, 53, 63, 44, 54, 64].

The index formula y[idx] = a[nc*H + h] + b[nc*W + w] gives [11, 21, 31, 12, 22, 32, 43, 53, 63, 44, 54, 64]; equal: True.
<!-- /worked -->

Its neighbour `crates/kernels/wgsl/add_chan_bcast.wgsl` adds a per-image,
per-channel scalar `v : [N, C]` to a map. Its header explains why the
existing `bias_add` kernel does not do the job: the bias there is a `[C]`
vector shared by the whole batch, and at `N > 1` it would add the wrong image's context. That
sentence is the explicit-broadcast policy as an engineering fact: two
broadcasts that both look like "add a small array to a big one" are
different operations, and an engine that wishes to be correct gives each its own
kernel and its own name instead of inferring one from the shapes.

**A stride larger than the row.**
`crates/kernels/wgsl/attn_decode_scores.wgsl` reads a key cache of shape
$[\max T, n_{\text{kv}} \cdot d_h]$ with the row stride `kv_stride` passed in its
parameters, and writes scores into rows of stride `cap`, the maximum length, of
which only the first `t` entries are valid. The offsets are
`kb = j * p.kv_stride + kvh * hd` and `scores[h * p.cap + j]`: equation (1) with a
stride that is the capacity of a row and not the number of elements in it. The
cache is allocated for the longest sequence once, so that growing the sequence
never changes any element's offset (the "first size never matters" fact again).

**Column-major libraries, by swapping operands.**
`crates/backend-api/src/lib.rs` defines `LibraryGemm`, the description of a
product for a vendor library in the library's column-major convention, with
its documentation stating "Products over row-major data are stated by swapping the
operands, so no operand is ever copied or transposed". The code that builds one,
`crates/gpu-core/src/provider/library.rs`, has the derivation of this chapter
as a comment: "Row-major `y = x W^T` is column-major `y^T[n, m] = W[k, n]^T x^T[k, m]`",
with the fields `trans_a: true`, `m: n`, `n: m`, `lda: k`, `ldb: k`, `ldc: n`
(there `n` is the width of the row-major output, which is the number of rows $M$
of the column-major one, so `ldc` is $M$ as above). The
worked example `arrays.misread_and_swap` is this call on a small size. It
arrived with the library path in commit `8ed1c8345` (16 October 2026), and its
batched form carries strides between consecutive matrices of a batch
(`stride_a`, `stride_b`, `stride_c`), with `0` meaning one matrix shared by the
whole batch: the stride-0 broadcast, here for a weight shared across examples.

**Reading a foreign tensor through its strides.**
`crates/checkpoint/src/torchpt.rs` reads checkpoints written by a Python
framework, whose tensors are stored as a storage blob together with a
storage offset, a shape and strides: the view of equation (1) in a file.
Its `gather` function walks the index space in row-major order and
reads `t.offset + sum(i * s)`, so that a checkpointed transpose comes out
correctly. The test `noncontiguous_strided_view` in
`crates/checkpoint/tests/it/torchpt.rs` uses the storage
$0, 10, \dots, 60$, offset 1, shape $[2, 3]$ and strides $(1, 2)$ and expects
$10, 30, 50, 20, 40, 60$: exactly the example of the strides section, reproduced
above by `arrays.storage_view_gather`. The commit that added the reader,
`693cbb9aa` (15 July 2026), records that non-contiguous views were
materialised contiguous and validated against real files of the framework
with strided views, bit-exact.

**A story: 480 bytes.** Knowledge #7 records a rule paid for twice: a storage
binding must start at a multiple of 256 bytes, which is 64 four-byte numbers,
or the graphics layer rejects the call with a validation error. A row boundary of a
matrix of width $D$ is at $r D$ numbers, so it is aligned exactly when
$r D$ is a multiple of 64, that is when $r$ is a multiple of $64/\gcd(64, D)$.
For real model widths that are multiples of 64 the multiple is 1 and the rule
is invisible, which is the reason the bug hid.

On 10 August 2026 commit `8d69f8b11` fixed a decode path that bound a slice of a buffer starting at
`local_row * d` numbers, valid only when that product is a multiple
of 64; the commit message describes a deterministic validation failure on
hardware that enforces the full limit. Then commit `46d31484e` (4 September
2026) chased an intermittent failure of three tests under the full suite. A
test that lowered the tiling budget in a process-wide setting leaked into other
tests running concurrently. Under a budget of 128 numbers and a test model
width of 20, the tile height was $\lfloor 128/20 \rfloor = 6$ rows, putting a
tile boundary $6 \cdot 20 \cdot 4 = 480$ bytes into the buffer, 224 bytes past a
256-byte boundary. The fix floored every multi-tile row count to the alignment
granularity, `tile_row_align` in `crates/model/src/block.rs`, defined as
$64/\gcd(64, D)$, which is 16 rows for $D = 20$ (1280 bytes, five times 256), and
a test sweeps pairs of width and budget. The table of this arithmetic:

<!-- worked: arrays.alignment_arithmetic -->
A byte offset into a buffer must be a multiple of 256 bytes = 64 four-byte numbers. A row boundary r of a matrix with D columns is at r*D numbers; the smallest r > 0 that works is 64 / gcd(64, D).

| D | smallest aligned r | bytes at that r | bytes at r = 6 | remainder mod 256 |
|---|---:|---:|---:|---:|
| 20 | 16 | 1280 | 480 | 224 |
| 48 | 4 | 768 | 1152 | 128 |
| 64 | 1 | 256 | 1536 | 0 |
| 96 | 2 | 768 | 2304 | 0 |
| 100 | 16 | 6400 | 2400 | 96 |
| 128 | 1 | 512 | 3072 | 0 |

With D = 20 a tile boundary at row 6 sits 480 bytes in, which is 224 bytes past the last 256-byte boundary; row 16 sits at 1280 = 5 * 256 bytes.
<!-- /worked -->

The lesson is the one of this chapter in a small case: where a slice of an array
starts is a fact about its strides, a constraint a device can impose on it, and
an assumption like "real shapes happen to be multiples of 64" is a property of
today's sizes and not of the code. (Knowledge #7 also records the alternative
used where offsets are genuinely ragged: pass them in the kernel's own
parameters instead of binding a slice.)

## Check your understanding

Answers are at the end of the chapter.

1. **Predict.** For shape $[2, 3, 4]$, what are the row-major offset and the
   column-major offset of the index $(1, 2, 0)$?
2. **Find the error.** "To split a width of 12 into 3 heads of 4 for a sequence of
   $T = 4$ positions, reshape $[T, D] = [4, 12]$ to $[H, T, D_h] = [3, 4, 4]$: the
   element counts agree, so it is correct." What is wrong, and what is the right
   sequence of operations?
3. **Predict.** What are the shapes of $[5, 1] - [1, 3]$, of $[5] - [5, 1]$ and of
   $[2, 3, 4] + [3, 1]$? Which of them would you spell differently in the
   book's pseudocode, and why?
4. **Explain.** The transpose of a $[1000, 1000]$ matrix as a view takes no time
   at all. Why can summing the entries of that view row by row still be much
   slower than summing the original row by row, and what would you do if you had to
   do it many times?
5. **Find the error.** "$A[:, 1:3]$ of a $[3, 4]$ array is a block of two whole
   columns, so reshaping it to $[6]$ is free." Use rule (2) to say whether it is.
6. **Explain.** Why does the size of the first axis never appear in any row-major
   stride, and what does that allow a program to do with the batch axis?
7. **Predict.** The strides of a contiguous array of shape $[3]$ and of one of
   shape $[2, 1]$ are $(1)$ and $(1, 1)$. What are the strides of their broadcast
   views of shape $[2, 3]$?
8. **Find the error.** "A row-major buffer of a $[2, 3]$ matrix handed to a
   column-major routine as a $[2, 3]$ matrix gives the same matrix." What does
   the routine actually see, and when would the shapes hide the mistake?

## Exercises

1. **[Calculate]** For shape $[4, 5, 6]$, write down the row-major and
   column-major strides, the row-major offset of index $(2, 3, 1)$, and the index
   whose row-major offset is 100. Check that the offset of the last element is
   $4 \cdot 5 \cdot 6 - 1$.

2. **[Calculate]** A $[3, 4]$ row-major array has storage $0, 1, \dots, 11$. Write
   the shape, strides and starting offset of the view $A[1:3, 1:4]$, of
   $A[:, ::2]$ and of the transpose of $A[:, 1:3]$. Which are contiguous? List the
   elements of each in row-major order.

3. **[Calculate]** Show that a row-major buffer of shape $[m, n]$ read as a
   column-major matrix of shape $[n, m]$ is the transpose of the first, by
   comparing the two offset formulas. Then, for $\mathbf{X}$ of shape $[2, 3]$
   and $\mathbf{W}$ of shape $[4, 3]$, state the call a column-major product routine
   needs (which operand first, which is transposed, the leading dimensions) to
   produce the row-major $\mathbf{X}\mathbf{W}^\top$, and explain why a swap
   without the transpose flag would fail for these shapes.

4. **[Calculate]** Which pairs of these shapes broadcast, and to what? $[5, 1, 4]$
   with $[3, 1]$; $[5, 3, 4]$ with $[3, 4]$; $[5, 3, 4]$ with $[5, 4]$; $[2, 1]$ with
   $[1, 3, 1]$; $[6]$ with $[6, 1]$. For each that succeeds, give the strides of
   both operands as broadcast views if each is contiguous.

5. **[Calculate]** For a tensor of shape $[2, 3, 4]$ decide for each of these
   whether the reshape can be a view when the tensor is (a) contiguous, (b) the
   permuted view with axes $(1, 0, 2)$: to $[6, 4]$, to $[2, 12]$, to $[2, 3, 2, 2]$, to $[24]$.
   Use the split rule and rule (2).

6. **[Calculate]** A batch of $B = 8$ images of $C = 3$, $H = 32$, $W = 32$ is stored
   in NCHW. Give the strides, the offset of $(b, c, h, w) = (5, 2, 7, 9)$, and the
   offset of the same pixel when the batch is NHWC. How far apart in storage are
   the three channels of one pixel in each layout?

7. **[Calculate]** Using the alignment rule, find for $D = 20, 30, 48, 100$
   the smallest number of rows $r > 0$ whose boundary is 256-byte aligned for
   4-byte numbers, and for 2-byte numbers. State the rule in terms of the
   element size and of $D$.

8. **[Implement]** Implement the `View` record, `row_major_strides`, `offset_of`,
   `transpose_view`, `slice_view`, `is_contiguous` and `materialize` from the
   pseudocode in your language, with the `require` checks, and reproduce
   `arrays.layout_3x4`, `arrays.nd_offsets`, `arrays.transpose_view` and
   `arrays.slices_are_views` (take `A[1:2, :]` as a slice with start 1, stop 2),
   including the write-through example.

9. **[Implement]** Implement `reshape`, `broadcast_shape` and `broadcast_view`,
   and elementwise addition on broadcast views, and reproduce
   `arrays.reshape_vs_transpose`, `arrays.split_heads`, `arrays.broadcast_shapes`,
   `arrays.broadcast_values` and `arrays.silent_outer_diff`. Then implement
   `reshape_view`, make `reshape` try it first and copy only when it reports
   failure, and reproduce `arrays.split_axis_rule` and `arrays.merge_rule`;
   check the claim at the end of the last block by brute force on random
   shapes (sizes 1 to 5) and random strides (0 to 12): whenever `reshape_view`
   returns a view, its elements equal those of the copying `reshape`.

10. **[Implement]** Implement the column-major product routine `gemm` of the
    text and the swapped-operand call, and reproduce `arrays.misread_and_swap`.
    Then break it two ways. First: forget the transpose flag (`trans_a = false`,
    with `lda` raised to $M$ so that the routine's checks pass). Second: state
    the product in the unswapped order, with $\mathbf{X}$'s buffer as the first
    operand (transposed) and $\mathbf{W}$'s as the second, and read the output
    buffer as the row-major $\mathbf{Y}$. A shape check here means: each
    leading dimension is at least the number of rows stored, and each buffer is
    long enough. Report for which shapes each mistake passes the shape check.

11. **[Investigate]** Time a sum over every element of a large row-major
    matrix traversed by rows, and by columns (the same elements in the other order),
    for sizes from $64 \times 64$ to $4096 \times 4096$. Plot the ratio against the matrix
    size and explain it with the cache-line count of `arrays.cache_lines`. Where does
    the ratio stop growing, and what does that say about the size of your
    machine's cache? State your machine and your measurements as such, not as a
    property of the language.

12. **[Investigate]** Convert a batch of images from NCHW to NHWC and back with
    a permuted view and with `materialize`. Verify the round trip is bitwise exact
    and that the converted array is contiguous only after the copy. Then find
    one operation (a per-channel normalisation, say, and a per-pixel one) for
    each layout where reading is in order for one layout and strided for the other.

## Summary

A **tensor** is an array of numbers with any number of axes, and its **shape**
lists their sizes, outermost first. Memory is one line, so every tensor is
flattened onto it, and the **row-major layout** (last axis fastest) gives the
element $(i_0, \dots, i_{R-1})$ the offset $\sum_k i_k s_k$ with **strides**
$s_k = d_{k+1} \cdots d_{R-1}$, derived from the nesting of rows inside planes
inside batches; the map is a perfect fit onto $0, \dots, \prod d_k - 1$, can be
undone by repeated division, and does not depend on the first size. Column-major
layout has the same form with strides taken from the left, and the two
conventions meet in a transposed read: a row-major $[m, n]$ buffer is the
column-major $[n, m]$ transpose, which is also why a row-major product can be
posed to a column-major library by swapping the operands.

The general description is a **view**: storage, an offset, a shape and strides, with
the element at $o_0 + \sum_k i_k s_k$. Transpose swaps shape and strides,
slicing shifts the offset and multiplies strides by the step, and neither
moves a number; the cost is aliasing (writes go through) and a loss of
contiguity, which routines that walk memory in order require and which a copy
restores. **Reshape** keeps the line of numbers and re-cuts it: free on a
contiguous array, free for splitting any axis, free for merging axes that satisfy
$s_k = d_{k+1}s_{k+1}$ and a copy otherwise. It is not a transpose, and the confusion
of the two, shape-correct and value-wrong, is the classic bug, seen here as the
split of attention heads. **Broadcasting** aligns shapes from the right and
stretches axes of size 1 with stride 0, costing a view and no copy; because it
makes mismatched shapes work, the book's pseudocode states every broadcast
explicitly. The **batch dimension** is the first axis, where each example is a
contiguous block, the first size never enters a stride, and a $[B, T, D]$ batch merges
to $[B T, D]$ for free so that one matrix product serves all examples.

What this chapter does not give is a reason why the numbers in a matrix should be
what they are. Having fixed how matrices and tensors are stored and moved, the
next chapter returns to the matrices themselves and asks what hidden structure one
has: the directions a map only stretches (eigenvectors), the rotations, and
the way a large matrix is approximated by a sum of a few simple ones.

## Key terms

- **tensor**: an array of numbers with any number of axes (zero for a scalar, one for a vector, two for a matrix), described by its shape. <!-- concept: tensor -->
- **shape**: the list of the sizes of a tensor's axes, outermost first; the number of elements is their product. <!-- concept: shape -->
- **row-major layout**: storing a tensor in memory with the last index changing fastest, so that the element $(i_0, \dots, i_{R-1})$ sits at offset $\sum_k i_k s_k$ with $s_k$ the product of the sizes to its right. <!-- concept: row-major-layout -->
- **stride**: the distance in memory, counted in elements, between two neighbours along one axis; a view is described by one stride per axis, and stride 0 repeats one number. <!-- concept: stride -->
- **broadcasting**: the rule that aligns two shapes from the right and treats an axis of size 1 as stretched to match the other, by giving it stride 0, so that arrays of different shapes combine without copying. <!-- concept: broadcasting -->
- **reshape-and-view**: reshaping keeps a tensor's line of numbers in row-major order and cuts it into a new shape; a view is a new shape, strides and offset over the same storage, so a reshape of a contiguous array needs no copy. <!-- concept: reshape-and-view -->
- **batch dimension**: an extra, usually first, axis of a tensor that indexes independent examples processed together, so that each example is one contiguous block. <!-- concept: batch-dimension -->

## Answers

**Before you begin.**

1. $\sum_{k=0}^{2} i_k s_k = i_0 s_0 + i_1 s_1 + i_2 s_2 = 1\cdot 12 + 2\cdot 4 + 3\cdot 1 = 23$.
2. A function has an inverse when it is one-to-one (different inputs give
   different outputs) and onto its set of outputs (every output is produced by
   some input): then each output can be traced back to exactly one input.
3. Three entries; the entry with index 1 is 8.
4. $\begin{pmatrix}1&4\\2&5\\3&6\end{pmatrix}$, of shape $[3, 2]$.
5. $\mathbf{X}$ has shape $[B, d_{\text{in}}]$ and $\mathbf{X}\mathbf{W}^\top$ has
   shape $[B, d_{\text{out}}]$; the dimension $d_{\text{in}}$ is summed away.

**Check your understanding.**

1. Row-major strides $(12, 4, 1)$: $12\cdot 1 + 4\cdot 2 + 0 = 20$. Column-major strides
   $(1, 2, 6)$: $1 + 4 + 0 = 5$.
2. Reshape never moves an axis, it only re-cuts the line of numbers. Cutting the
   line of $[4, 12]$ into $[3, 4, 4]$ gives "head" 0 the first 16 numbers, which are
   the first one and a third of positions 0 and 1, not one head of every
   position. The right sequence is to reshape $[4, 12]$ to $[T, H, D_h] = [4, 3, 4]$
   (splitting the last axis) and then permute the first two axes to
   $[H, T, D_h] = [3, 4, 4]$.
3. $[5, 1] - [1, 3]$ gives $[5, 3]$; $[5] - [5, 1]$ aligns $[1, 5]$ with $[5, 1]$ and
   gives $[5, 5]$; $[2, 3, 4] + [3, 1]$ aligns $[1, 3, 1]$ with $[2, 3, 4]$ and gives
   $[2, 3, 4]$. The second is the dangerous one, and in the pseudocode every
   stretched axis would be written with its 1 (`[5, 1]` and `[1, 5]`) so that the
   reader sees the $[5, 5]$.
4. Reading the transposed view row by row steps through the original storage by
   1000 elements at a time, so every cache line fetched yields one useful
   number instead of sixteen. If the transposed array will be read many times,
   materialise it once into contiguous storage.
5. The view has shape $[3, 2]$ and strides $(4, 1)$. Rule (2) asks for
   $s_0 = d_1 s_1$, that is $4 = 2\cdot 1 = 2$, which fails, so the reshape needs a
   copy. A block of columns is not contiguous.
6. $s_k$ is a product of the sizes to the right of axis $k$, so $d_0$ is never
   in it. A tensor can therefore have more or fewer examples without changing
   the offset of any example that exists in both, and each example is a
   contiguous block at offset $b\,s_0$.
7. A view of the $[3]$ array stretched to $[2, 3]$ has strides $(0, 1)$. The
   $[2, 1]$ array stretched to $[2, 3]$ has strides $(1, 0)$: the size-1 axis gets
   stride 0, the other keeps its stride.
8. The row-major $[2, 3]$ buffer is the column-major matrix of shape $[3, 2]$,
   the transpose. Read as a column-major $[2, 3]$ matrix with leading dimension 2
   it is a scrambled matrix, a different one again. For a square matrix the same
   buffer is read as its transpose with the right shape, so no shape check
   fires.
