<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Eigenvalues, Rotations and Low-Rank Structure

*What hidden structure does a matrix have, and how can a big matrix be approximated by small ones?*

A weight matrix in a language model has $4096$ rows and $4096$ columns:
$16\,777\,216$ numbers, exactly sixty-four mebibytes in 32-bit floating point,
for one layer of dozens. Chapter 4 taught us what such a table does: it is a
linear map, it composes with other maps by multiplication, and it can be
inverted only if its rank is full. That describes what the table *is*. It says
nothing yet about what is *inside* it, and the practical questions that follow
all depend on the answer.

Chapter 4 found a single number, $ad - bc$, that gave the area of the image of
a unit square and vanished exactly when the map crushed the plane onto a line.
We want that number for matrices of any size, with its geometric meaning, and
we want to know whether it is the right thing to compute. Then there is the
matter of direction. Take
$\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. It sends
$(1, 0)$ to $(2, 1)$, which points somewhere else. But it sends $(1, 1)$ to
$(3, 3)$, the same direction, three times as long. If a matrix has such
directions, applying it a hundred times, which is what depth in a network and
recurrence in a sequence model amount to, becomes easy to predict: the
stretched directions grow or shrink by a fixed factor each time.

Some maps do not stretch anything. They keep every length and every angle and
turn the space as a rigid object, so they never amplify a rounding error and
their inverse costs nothing. Which matrices are these, how are they built, and
what do they do to a pair of coordinates that we turn by an angle? And behind
all of it sits the question that makes the chapter pay: does every matrix
reduce to such simple pieces, and can a big one be replaced by a small one? If
a table of $4096 \times 4096$ numbers were built from only $8$ independent
patterns, it could be stored with $8 \times (4096 + 4096) = 65\,536$ numbers,
two hundred and fifty-six times fewer, and a fine-tuning method could train
only those. Whether that is true of a given matrix is something the matrix
itself can tell us, through numbers called singular values.

Each answer is one idea from the earlier chapters used a little harder. The area number is the $ad - bc$ of Chapter 4, the
stretched directions come from solving $\mathbf{A}\mathbf{x} = \lambda
\mathbf{x}$ with the elimination of Chapter 4, the rigid maps are the matrices
whose columns are orthogonal unit vectors in the sense of Chapter 3, and the
approximation is a sum of simple matrices whose size is measured with the norm
of Chapter 3. Complex numbers belong to Chapter 14, and the few places that
would need one go around them.

Six ideas from earlier chapters carry the weight, one question each: matrix
products and the transpose, the inverse, rank, norms and orthogonality, the unit
circle, and the roots of a quadratic. An answer that will not come is a reason
to reread the chapter named before going further.

1. (Chapter 4, matrix multiplication and transpose.) For $\mathbf{A} =
   \begin{pmatrix}1&2\\0&1\end{pmatrix}$ and $\mathbf{B} =
   \begin{pmatrix}1&0\\1&1\end{pmatrix}$ compute $\mathbf{A}\mathbf{B}$ and
   $(\mathbf{A}\mathbf{B})^\top$, and state what $(\mathbf{A}\mathbf{B})^\top$
   equals in terms of $\mathbf{A}^\top$ and $\mathbf{B}^\top$.
2. (Chapter 4, inverse.) For $\begin{pmatrix}4&7\\2&6\end{pmatrix}$ compute
   $ad - bc$. Does the matrix have an inverse, and what does $ad - bc = 0$
   tell you about a $2 \times 2$ matrix?
3. (Chapter 4, rank.) What is the rank of $\begin{pmatrix}1&2\\2&4\end{pmatrix}$,
   and what does that matrix do to the unit square?
4. (Chapter 3, norm and orthogonality.) Compute the length of $(1, 2, 2)$ and
   decide whether $(1, 2, 2)$ and $(2, 1, -2)$ are orthogonal.
5. (Chapter 3, unit circle.) What are the cosine and sine of $90^\circ$, and
   what identity relates $\cos\theta$ and $\sin\theta$ for every $\theta$?
6. (Chapter 2, quadratics, four chapters back.) Solve $\lambda^2 - 4\lambda + 3
   = 0$.

## The determinant: how a map scales area

Chapter 4 ended its tour of the $2 \times 2$ inverse with a number,
$\Delta = ad - bc$, and the remark that $|\Delta|$ is the factor by which the
map scales area. Here is that claim on six maps. Each row maps the unit square
(corners $(0,0)$, $(1,0)$, $(1,1)$, $(0,1)$, in that order) and measures the
image by the shoelace formula: walk the four image corners in the same order
and add up $x_k y_{k+1} - x_{k+1} y_k$, then halve.

<!-- worked: decompositions.determinant_area -->
Unit square (0,0), (1,0), (1,1), (0,1) mapped by each matrix; the signed area of the image polygon, walked in that order, by the shoelace formula, next to ad - bc.

| map | matrix | ad - bc | det | signed area of image |
|---|---|---|---:|---:|
| stretch | [[3, 0], [0, 2]] | 3*2 - 0*0 | 6 | 6 |
| chapter 4 parallelogram | [[2, 1], [1/2, 3/2]] | 2*3/2 - 1*1/2 | 5/2 | 5/2 |
| shear | [[1, 1], [0, 1]] | 1*1 - 1*0 | 1 | 1 |
| swap axes | [[0, 1], [1, 0]] | 0*0 - 1*1 | -1 | -1 |
| collapse | [[1, 2], [1/2, 1]] | 1*1 - 2*1/2 | 0 | 0 |
| generic | [[1, 2], [3, 4]] | 1*4 - 2*3 | -2 | -2 |
<!-- /worked -->

The signed area of the image equals $ad - bc$ in every row. The first row
stretches the square into a $3 \times 2$ rectangle, area $6$. The shear moves
the top edge sideways without changing base or height, area $1$. The "collapse"
matrix has columns $(1, \tfrac12)$ and $(2, 1)$, one twice the other, so the
image is a segment, area $0$, the same flattened case as the right half of
Chapter 4's Figure 1.

One feature of the table needs a name. The number is **signed**: the swap
$\begin{pmatrix}0&1\\1&0\end{pmatrix}$ exchanges the two axes, the image of
$\mathbf{e}_1$ is now where $\mathbf{e}_2$ was, and walking the image corners
in the original order goes round the other way, giving $-1$. A negative sign
means the map flips orientation, as a mirror does. Its size, $1$, still says
the area is unchanged.

### Why the formula is the area

The image of the unit square is the parallelogram spanned by the two columns
$(a, c)$ and $(b, d)$ of $\begin{pmatrix}a&b\\c&d\end{pmatrix}$. Its corners
in order are $(0,0)$, $(a, c)$, $(a + b, c + d)$ and $(b, d)$. The shoelace sum
adds, corner by corner, $0$, then $a(c + d) - (a + b)c = ad - bc$, then
$(a + b)d - b(c + d) = ad - bc$, then $0$. Half of $2(ad - bc)$ is
$ad - bc$. So the formula is not a recipe to memorise but the area of the
parallelogram built from the columns, which Chapter 4 showed is what a matrix
does to the unit vectors.

The **determinant** of a square matrix $\mathbf{A}$, written $\det\mathbf{A}$,
is the signed factor by which the map $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$
scales area (in two dimensions), volume (in three) and the corresponding
measure in more dimensions, with the sign recording whether orientation is
kept or flipped. For $2 \times 2$ matrices $\det\mathbf{A} = ad - bc$.

The factor is the same for every shape, not only for the unit square. A map
is linear, so it takes the grid of tiny squares that covers any region to a
grid of equal tiny parallelograms, each scaled by the same number; adding
them up, the region's area is multiplied by $|\det\mathbf{A}|$. That is why a
single number describes the whole map.

### Rules that follow from the geometry

Four facts follow from the meaning, and each is checked below on one pair of
matrices.

- **Products multiply.** Applying $\mathbf{B}$ scales areas by $\det\mathbf{B}$,
  then applying $\mathbf{A}$ scales them by $\det\mathbf{A}$, so the composite
  scales them by the product: $\det(\mathbf{A}\mathbf{B}) =
  \det\mathbf{A}\,\det\mathbf{B}$. (The signs multiply the same way: two mirror
  flips make a rotation.)
- **Inverses divide.** $\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$ and the identity
  scales nothing, $\det\mathbf{I} = 1$, so $\det\mathbf{A}^{-1} =
  1/\det\mathbf{A}$. The formula can only be written when $\det\mathbf{A}
  \ne 0$, which is the invertibility condition of Chapter 4 in its final form.
- **The transpose changes nothing.** $\det\mathbf{A}^\top = \det\mathbf{A}$.
  For $2 \times 2$ this is $ad - cb = ad - bc$ at a glance; it holds for every
  size.
- **Scaling by $c$ scales volume by $c^n$.** Multiplying every column by $c$
  multiplies each of the $n$ edge lengths of the parallelepiped by $c$:
  $\det(c\,\mathbf{A}) = c^n \det\mathbf{A}$ for an $n \times n$ matrix.

<!-- worked: decompositions.determinant_rules -->
A = [[4, 7], [2, 6]], B = [[1, 2], [3, 4]].

| quantity | value |
|---|---:|
| det A | 10 |
| det B | -2 |
| det(A) * det(B) | -20 |
| A B | [[25, 36], [20, 28]] |
| det(A B) | -20 |
| det(A^T) | 10 |
| A^-1 | [[3/5, -7/10], [-1/5, 2/5]] |
| det(A^-1) | 1/10 |
| 1 / det A | 1/10 |
| det(3 A), A is 2 x 2 | 90 |
| 3^2 det A | 90 |
<!-- /worked -->

### Three dimensions and beyond

In three dimensions the columns span a slanted box, a **parallelepiped**, and
the determinant is its signed volume. The formula expands along the first row
into three $2 \times 2$ determinants, with signs that alternate $+, -, +$:

$$
\det\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}
= a\,(ei - fh) \;-\; b\,(di - fg) \;+\; c\,(dh - eg).
$$

The $2 \times 2$ determinant multiplying each entry is the determinant of what
remains after deleting that entry's row and column, the entry's **minor**.

<!-- worked: decompositions.determinant_3x3 -->
A = [[2, 0, 1], [1, 3, 2], [1, 1, 2]], expanded along row 0 with signs +, -, +.

| term | entry A[0][j] | minor (delete row 0 and column j) | det of minor | signed contribution |
|---|---:|---|---:|---:|
| column 0 | 2 | [[3, 2], [1, 2]] | 4 | 8 |
| column 1 | 0 | [[1, 2], [1, 2]] | 0 | 0 |
| column 2 | 1 | [[1, 3], [1, 1]] | -2 | -2 |

det A = 6; the cofactor routine gives 6.

The singular matrix of chapter 4, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]: det = 0.
Columns of A: a1 = (2, 1, 1), a2 = (0, 3, 1), a3 = (1, 2, 2); a2 x a3 = (4, 1, -3), and a1 . (a2 x a3) = 6, the volume of the parallelepiped they span.
<!-- /worked -->

The singular matrix of Chapter 4 with rows $(1,2,3)$, $(4,5,6)$, $(7,8,9)$ has
determinant $0$, as its rank of $2$ demands: its three columns lie in one
plane, so the box they span is flat.

Expansion along a row is the right definition and the wrong algorithm. Each
$n \times n$ determinant becomes $n$ determinants of size $n - 1$, so the
number of terms is $n! = n(n-1)\cdots 1$: six for $n = 3$, but $20! \approx
2.4 \times 10^{18}$ for $n = 20$. Elimination does the same job in about
$n^3/3$ multiply-adds. Adding a multiple of one row to another does not change
the parallelepiped's volume (it slides the box sideways, like the shear), and
swapping two rows flips its sign. After elimination the matrix is triangular
and the determinant is the product of the pivots, times $-1$ for each swap.

<!-- worked: decompositions.determinant_pivots -->
| step | pivot | matrix after clearing the column |
|---|---:|---|
| pivot 0 | 2 | [[2, 0, 1], [0, 3, 3/2], [0, 1, 3/2]] |
| pivot 1 | 3 | [[2, 0, 1], [0, 3, 3/2], [0, 0, 1]] |
| pivot 2 | 1 | [[2, 0, 1], [0, 3, 3/2], [0, 0, 1]] |

Product of the pivots: 6 (row swaps used: 0). Cofactor expansion: 6.
<!-- /worked -->

### The determinant in floating point

A determinant of zero means "not invertible", so it is tempting to test
invertibility by computing one and comparing it with zero. It is a poor test.
The matrix $0.1\,\mathbf{I}_n$ shrinks every vector to a tenth, loses no
information and is trivially invertible, yet its determinant is $0.1^n$.

<!-- worked: decompositions.determinant_scale -->
A = 0.1 I_n multiplies every vector by 0.1: invertible for every n. Its determinant is 0.1^n.

| n | det computed as a float64 product | log \|det\| = n log 0.1 |
|---|---:|---:|
| 10 | 1.00e-10 | -23.03 |
| 100 | 1.00e-100 | -230.3 |
| 300 | 1.00e-300 | -690.8 |
| 400 | 0.0 (underflow) | -921 |
<!-- /worked -->

At $n = 400$ the true value is $10^{-400}$, below the smallest number a 64-bit
float can hold (Chapter 15 gives the range), and the computed product is $0$.
The determinant mixes together how many dimensions there are and how big the
entries are, so it can be astronomically small or large for a perfectly healthy
matrix. Programs that need a determinant keep its logarithm (the sum of the
logarithms of the pivots' magnitudes, the last column), and programs that need
to know *how close to singular* a matrix is use the singular values defined
later in this chapter.

## Eigenvectors: directions a map only stretches

Back to $\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. Here is
what it does to unit vectors at several angles, with the angle and length of
the image.

<!-- worked: decompositions.eigen_directions -->
A = [[2, 1], [1, 2]] applied to unit vectors at several angles.

| angle of x | x | A x | angle of A x | length of A x |
|---|---:|---:|---:|---:|
| 0 | (1, 0) | (2, 1) | 26.57 | 2.236 |
| 30 | (0.866, 0.5) | (2.232, 1.866) | 39.9 | 2.909 |
| 45 | (0.7071, 0.7071) | (2.121, 2.121) | 45 | 3 |
| 60 | (0.5, 0.866) | (1.866, 2.232) | 50.1 | 2.909 |
| 90 | (0, 1) | (1, 2) | 63.43 | 2.236 |
| 135 | (-0.7071, 0.7071) | (-0.7071, 0.7071) | 135 | 1 |
<!-- /worked -->

Most directions are turned: $0^\circ$ goes to $26.57^\circ$, $90^\circ$ to
$63.43^\circ$, every image is pulled towards the $45^\circ$ line. Two
directions are not turned at all. At $45^\circ$ the image is also at
$45^\circ$, three times as long. At $135^\circ$ the image is at $135^\circ$
with length $1$: the vector $(-0.7071, 0.7071)$ is sent to itself. Figure 1
shows sixteen directions at once, with the whole unit circle and its image.

![The unit circle (grey) and its image under the matrix with rows (2, 1) and (1, 2), an ellipse tilted along the 45 degree line (blue). Sixteen points on the circle are joined to their images by thin lines. Two dashed lines through the origin, along (1, 1) and along (1, -1), are the directions on which the image stays on the same line: stretched by 3 along (1, 1) and left unchanged along (1, -1).](img/eigen-directions.svg)

**Figure 1.** The matrix $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ applied to the
unit circle. Directions off the two dashed lines are rotated towards the
$(1, 1)$ line; on the dashed lines a vector only changes length.

A nonzero vector $\mathbf{v}$ that a square matrix $\mathbf{A}$ sends to a
multiple of itself,

$$
\mathbf{A}\mathbf{v} = \lambda\,\mathbf{v},
$$

is an **eigenvector** of $\mathbf{A}$, and the number $\lambda$ is the matching
**eigenvalue**. (The words are half German: *eigen* means "own", the matrix's
own directions and own stretch factors.) An eigenvalue of $3$ stretches by
three, $1$ leaves the vector alone, $-2$ stretches by two and flips the
vector to point the other way, $0$ crushes it to nothing, and a number between
$-1$ and $1$ shrinks it. Any nonzero multiple of an eigenvector is an
eigenvector with the same eigenvalue, because both sides scale together;
we usually normalise to length $1$, and the *line* through the origin is the
real object.

### Finding them by hand for a $2 \times 2$

Rewrite $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ as
$(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$, since
$\lambda\mathbf{v} = \lambda\mathbf{I}\mathbf{v}$. We are looking for a
*nonzero* $\mathbf{v}$ that the matrix $\mathbf{A} - \lambda\mathbf{I}$ sends
to zero. Chapter 4 told us when that is possible: a matrix that crushes a
nonzero vector to zero is not invertible, and for a square matrix that means
its determinant is zero. So the eigenvalues are exactly the numbers $\lambda$ with

$$
\det(\mathbf{A} - \lambda\mathbf{I}) = 0.
$$

For $\mathbf{A} = \begin{pmatrix}a&b\\c&d\end{pmatrix}$ the left side is
$(a - \lambda)(d - \lambda) - bc = \lambda^2 - (a + d)\lambda + (ad - bc)$, the
**characteristic polynomial**, a quadratic in $\lambda$. The sum $a + d$ of
the diagonal entries is the **trace** of $\mathbf{A}$. So for every
$2 \times 2$ matrix the eigenvalues solve

$$
\lambda^2 - (\operatorname{trace}\mathbf{A})\,\lambda + \det\mathbf{A} = 0 .
$$

By the roots of a quadratic from Chapter 2 (a quadratic with roots
$\lambda_1, \lambda_2$ is $\lambda^2 - (\lambda_1 + \lambda_2)\lambda +
\lambda_1\lambda_2$), the two eigenvalues add up to the trace and multiply to
the determinant. For each root, solve $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v}
= \mathbf{0}$ by elimination; for a $2 \times 2$ the two rows are multiples of
each other, so one row is enough.

<!-- worked: decompositions.eigen_by_hand -->
A = [[2, 1], [1, 2]]: trace = 4, det = 3.
det(A - lambda I) = (2 - lambda)(2 - lambda) - 1 = lambda^2 - 4 lambda + 3 = 0.
Discriminant 4; lambda = (4 +- 2) / 2 = 3 or 1.

| lambda | solve (A - lambda I) v = 0 | A v | lambda v |
|---:|---|---|---|
| 3 | (2 - 3) v0 + v1 = 0, so v = (1, 1) | (3, 3) | (3, 3) |
| 1 | (2 - 1) v0 + v1 = 0, so v = (1, -1) | (1, -1) | (1, -1) |

Sum of eigenvalues 4 = trace 4; product 3 = det 3.
<!-- /worked -->

These are the two dashed lines of Figure 1. The check column shows
$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ in both rows, and the last line the
trace and determinant relations. The product of the eigenvalues equals the
determinant, which explains the first section from this side: the map scales
area by $3 \times 1$ because it scales one direction by $3$ and the other by
$1$.

The recipe does not need the matrix to mirror itself across its diagonal, as
$\begin{pmatrix}2&1\\1&2\end{pmatrix}$ does. For $\begin{pmatrix}4&1\\2&3\end{pmatrix}$ it works the same way:

<!-- worked: decompositions.eigen_nonsymmetric -->
A = [[4, 1], [2, 3]]: trace 7, det 10, lambda^2 - 7 lambda + 10 = 0.

| lambda | eigenvector v | A v | lambda v |
|---:|---|---|---|
| 5 | (1, 1) | (5, 5) | (5, 5) |
| 2 | (1, -2) | (2, -4) | (2, -4) |

Dot product of the two eigenvectors: -1; angle between them 108.4 degrees.
<!-- /worked -->

Both eigenvalues are again real and both eigenvectors are found, but now the
two eigenvector directions meet at $108.4^\circ$, not at a right angle. That
difference is the subject of the section on symmetric matrices below.

A matrix can also have no real eigenvalue at all. The quarter turn
$\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ has trace $0$ and determinant $1$, so
$\lambda^2 + 1 = 0$, which has no real solution:

<!-- worked: decompositions.rotation_has_no_real_eigenvalue -->
The quarter turn A = [[0, -1], [1, 0]]: trace 0, det 1, characteristic polynomial lambda^2 + 1; discriminant -4 < 0, so no real lambda.

| x | A x | angle between x and A x |
|---|---:|---:|
| (1, 0) | (0, 1) | 90 degrees |
| (1, 1) | (-1, 1) | 90 degrees |
| (-2, 3) | (-3, -2) | 90 degrees |
<!-- /worked -->

Every vector is turned by $90^\circ$, so none stays on its own line. The roots
exist if numbers such as $\sqrt{-1}$ are allowed; Chapter 14 introduces them
and shows that rotations always have such eigenvalues. This chapter stays with
real numbers, and the matrices it relies on most (the symmetric ones) never
have this problem.

### Why eigenvectors make repeated application easy

If a matrix has $n$ independent eigenvectors, put them as the columns of
$\mathbf{V}$ and the eigenvalues on the diagonal of $\boldsymbol{\Lambda}$.
Then $\mathbf{A}\mathbf{V} = \mathbf{V}\boldsymbol{\Lambda}$ (column $j$ of the
left side is $\mathbf{A}\mathbf{v}_j = \lambda_j\mathbf{v}_j$, which is column
$j$ of the right side), and since the columns are independent $\mathbf{V}$ has
an inverse:

$$
\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}.
$$

Read it right to left: express the input in the eigenvector directions
($\mathbf{V}^{-1}$), scale each coordinate by its eigenvalue
($\boldsymbol{\Lambda}$), and convert back ($\mathbf{V}$). The inner factors
cancel when the map is repeated, since $\mathbf{V}^{-1}\mathbf{V} =
\mathbf{I}$:

$$
\mathbf{A}^k = \mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1},
\qquad
\boldsymbol{\Lambda}^k = \mathrm{diag}(\lambda_1^k, \dots, \lambda_n^k).
$$

<!-- worked: decompositions.matrix_power -->
A = [[2, 1], [1, 2]], A = V diag(3, 1) V^-1 with V = [[1, 1], [1, -1]] (eigenvectors as columns).

| k | A^k by k multiplications | V diag(3^k, 1^k) V^-1 | equal |
|---:|---|---|---:|
| 1 | [[2, 1], [1, 2]] | [[2, 1], [1, 2]] | yes |
| 2 | [[5, 4], [4, 5]] | [[5, 4], [4, 5]] | yes |
| 5 | [[122, 121], [121, 122]] | [[122, 121], [121, 122]] | yes |
| 10 | [[29525, 29524], [29524, 29525]] | [[29525, 29524], [29524, 29525]] | yes |
<!-- /worked -->

Ten multiplications of $2 \times 2$ matrices give the same exact matrix as
$3^{10} = 59049$ and $1^{10} = 1$ pushed through the formula. For large $k$
the answer is dominated by the largest $|\lambda|$: a direction with
$|\lambda| > 1$ grows without bound under repetition, one with $|\lambda| < 1$
decays to zero. A system that applies the same matrix at every step, as a
recurrent network does, is stable only if its eigenvalues are no larger than
$1$ in magnitude; Volume V returns to this.

## Power iteration: finding the top eigenvector without algebra

The characteristic polynomial of a $4096 \times 4096$ matrix has degree
$4096$, and there is no formula for its roots. We need a method that uses only
what a machine does well, matrix-vector products. The idea comes from the
formula above: repeated application amplifies the direction with the largest
$|\lambda|$ more than any other.

Suppose $\mathbf{A}$ has eigenpairs $(\lambda_i, \mathbf{v}_i)$ with
$|\lambda_1| > |\lambda_2| \ge \dots$, and start from any vector. Write it in
the eigenvector directions, $\mathbf{x}_0 = c_1\mathbf{v}_1 + \dots +
c_n\mathbf{v}_n$. One application multiplies each term by its own eigenvalue,
so after $k$ applications

$$
\mathbf{A}^k\mathbf{x}_0 = \sum_i c_i\lambda_i^k\mathbf{v}_i
= \lambda_1^k\Big(c_1\mathbf{v}_1 + \sum_{i \ge 2} c_i
\Big(\frac{\lambda_i}{\lambda_1}\Big)^{k}\mathbf{v}_i\Big).
$$

Every ratio $\lambda_i/\lambda_1$ is smaller than $1$ in magnitude, so its
$k$-th power goes to zero, and what is left points along $\mathbf{v}_1$,
provided $c_1 \ne 0$ (the start is not exactly perpendicular to $\mathbf{v}_1$).
The slowest term to die is the second, so the **error shrinks by a factor
about $|\lambda_2|/|\lambda_1|$ per step**.

Computing $\mathbf{A}^k\mathbf{x}_0$ directly would grow like $\lambda_1^k$ and
overflow, so after every product we rescale the vector to length $1$. Scaling
does not change the direction, which is all we want. Once the direction has
converged, the eigenvalue is read off with the **Rayleigh quotient**
$\mathbf{x}\cdot\mathbf{A}\mathbf{x}$ for a unit vector $\mathbf{x}$: if
$\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$ and $\|\mathbf{x}\| = 1$ it equals
$\lambda\,\mathbf{x}\cdot\mathbf{x} = \lambda$.

```text
function power_iteration(A: array<real>[n, n], x0: array<real>[n], steps: int)
        -> (real, array<real>[n]):
    require A.shape[0] == A.shape[1] and A.shape[0] == x0.shape[0], "power_iteration: shapes"
    require norm(x0) > 0, "power_iteration: the start vector is zero"
    x = x0 / norm(x0)                      # broadcast: divide every element by the scalar
    for k in 0 .. steps:
        y = matvec(A, x)
        ny = norm(y)
        require ny > 0, "power_iteration: A sent the iterate to zero"
        x = y / ny
    lam = dot(x, matvec(A, x))             # Rayleigh quotient
    return (lam, x)
```

The tridiagonal matrix $\begin{pmatrix}2&1&0\\1&2&1\\0&1&2\end{pmatrix}$ has
eigenvalues $2 + \sqrt2 \approx 3.414$, $2$ and $2 - \sqrt2 \approx 0.586$,
and top eigenvector $(1, \sqrt2, 1)/2$. Started at $(1, 0, 0)$:

<!-- worked: decompositions.power_iteration_trace -->
A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]], x_0 = (1, 0, 0). The true top eigenpair is lambda_1 = 2 + sqrt 2 = 3.41421, v_1 = (1, sqrt 2, 1)/2 = (0.5, 0.70711, 0.5); the other eigenvalues are 2 and 0.585786.

| k | x_k | Rayleigh quotient x_k . A x_k | lambda_1 - Rayleigh | \|x_k - v_1\| |
|---:|---|---:|---:|---:|
| 0 | (1, 0, 0) | 2 | 1.41e+00 | 1.00e+00 |
| 1 | (0.89443, 0.44721, 0) | 2.8 | 6.14e-01 | 6.88e-01 |
| 2 | (0.77152, 0.61721, 0.1543) | 3.14286 | 2.71e-01 | 4.49e-01 |
| 3 | (0.67672, 0.67672, 0.29002) | 3.30841 | 1.06e-01 | 2.76e-01 |
| 4 | (0.60979, 0.6969, 0.37749) | 3.37605 | 3.82e-02 | 1.65e-01 |
| 5 | (0.56636, 0.70366, 0.42906) | 3.40088 | 1.33e-02 | 9.72e-02 |
| 6 | (0.53954, 0.70594, 0.45886) | 3.40961 | 4.60e-03 | 5.71e-02 |
| 7 | (0.52338, 0.70671, 0.47607) | 3.41263 | 1.58e-03 | 3.35e-02 |
| 8 | (0.51377, 0.70697, 0.48604) | 3.41367 | 5.44e-04 | 1.96e-02 |
| 9 | (0.50809, 0.70706, 0.49185) | 3.41403 | 1.87e-04 | 1.15e-02 |
| 10 | (0.50475, 0.70709, 0.49523) | 3.41415 | 6.40e-05 | 6.73e-03 |
<!-- /worked -->

Two columns of this table decay at different rates, and both rates are
predictable. The eigenvector error falls by about $0.587$ per step, in line with
$|\lambda_2|/|\lambda_1| = 2/3.414 = 0.586$. The eigenvalue estimate falls
faster, by about $0.34 = 0.586^2$ per step: the Rayleigh quotient of a vector
that is wrong by a small amount $\epsilon$ in direction is wrong by about
$\epsilon^2$. (For a symmetric matrix the first-order error cancels; we
observe this here and do not prove it.)

The rate depends on the ratio, and a ratio can be changed. Adding $s\mathbf{I}$
to $\mathbf{A}$ adds $s$ to every eigenvalue, since $(\mathbf{A} + s\mathbf{I})
\mathbf{v} = (\lambda + s)\mathbf{v}$, and leaves the eigenvectors alone. Here
are three shifts of the same matrix. The table's last two columns are defined
as follows: the predicted number of steps is $\ln 10^{-8}/\ln(|\lambda_2|/|\lambda_1|)$
(the steps for the error to fall from 1 to $10^{-8}$ at the predicted rate), and
the measured ratio per step is $(e_{20}/e_{10})^{1/10}$ with
$e_k = \|\mathbf{x}_k - \mathbf{v}_1\|$, the length of the difference to the
unit top eigenvector whose first entry is positive. (The shift $s = 2$ has
a predicted ratio $0.73880$ and a measured $0.73892$: the third eigenvalue
still contributes a little at iteration 10.)

<!-- worked: decompositions.power_iteration_rates -->
A + s I for the tridiagonal matrix above, started at x_0 = (1, 0, 0).

| shift s | eigenvalues of A + s I | \|lambda_2\| / \|lambda_1\| | measured error ratio per step | steps to reach error 1e-8 (predicted) |
|---:|---|---:|---:|---:|
| -1 | 2.414, 1, -0.4142 | 0.4142 | 0.4142 | 20.9 |
| 0 | 3.414, 2, 0.5858 | 0.5858 | 0.5858 | 34.4 |
| 2 | 5.414, 4, 2.586 | 0.7388 | 0.7389 | 60.8 |
<!-- /worked -->

![Three curves of the base-10 logarithm of the distance from the power-iteration iterate to the dominant eigenvector against iteration number from 0 to 40. They are straight lines, with slopes set by the ratio of the second to the first eigenvalue: 0.414 (steepest, reaching the round-off floor near 1e-15 at iteration 40), 0.586 and 0.739. Dashed predictions lie on top of the measured lines.](img/power-iteration-convergence.svg)

**Figure 2.** Power iteration on the three shifted matrices. On a logarithmic
scale geometric decay is a straight line whose slope is $\log_{10}$ of the
ratio $|\lambda_2|/|\lambda_1|$. The dashed predictions (the error at iteration
$5$ times the ratio to the power of the further steps) lie under the measured
lines; the steepest curve meets the round-off floor of 64-bit arithmetic.

Shifting up makes the ratio close to $1$ and iteration slow (more than
sixty steps to reach $10^{-8}$ instead of twenty-one). The convergence rate is a
property of the matrix, not of the code.

Power iteration fails in three ways, each visible in the derivation. If
$c_1 = 0$ the dominant term is absent in exact arithmetic, and the iteration
settles on the next direction (floating-point rounding usually reintroduces a
little of $\mathbf{v}_1$, which then slowly takes over). If $|\lambda_1| =
|\lambda_2|$ with different eigenvectors, no term wins and the iterate does not
settle. And each step costs one matrix-vector product, $n^2$ multiply-adds
for a square matrix, so the method is attractive when only the top few
directions are wanted from a large matrix and wasteful when all $n$ are.

## Symmetric matrices have orthogonal eigenvectors

The nonsymmetric example had eigenvectors at $108.4^\circ$; the symmetric one
had them at $90^\circ$. A square matrix is **symmetric** if $\mathbf{A} =
\mathbf{A}^\top$: entry $(i, j)$ equals entry $(j, i)$. Symmetric matrices are
common in machine learning, because the product $\mathbf{X}^\top\mathbf{X}$ of
any matrix with its own transpose is symmetric (its entry $(i, j)$ is the dot
product of columns $i$ and $j$ of $\mathbf{X}$, the same as entry $(j, i)$), and
so is any table of similarities between every pair of vectors in a set.

**Claim.** If $\mathbf{A}$ is symmetric and $\mathbf{v}, \mathbf{w}$ are
eigenvectors with different eigenvalues $\lambda \ne \mu$, then
$\mathbf{v}\cdot\mathbf{w} = 0$.

*Why.* Use the adjoint identity of Chapter 4 with $\mathbf{M} = \mathbf{A}$.
The number $(\mathbf{A}\mathbf{v})\cdot\mathbf{w}$ can be computed two ways:

$$
(\mathbf{A}\mathbf{v})\cdot\mathbf{w} = \lambda\,(\mathbf{v}\cdot\mathbf{w}),
\qquad
(\mathbf{A}\mathbf{v})\cdot\mathbf{w} = \mathbf{v}\cdot(\mathbf{A}^\top\mathbf{w})
= \mathbf{v}\cdot(\mathbf{A}\mathbf{w}) = \mu\,(\mathbf{v}\cdot\mathbf{w}).
$$

The first uses $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$; the second moves
$\mathbf{A}$ to the other side as $\mathbf{A}^\top$, which equals $\mathbf{A}$.
Subtracting, $(\lambda - \mu)(\mathbf{v}\cdot\mathbf{w}) = 0$, and since
$\lambda \ne \mu$ the dot product must be zero. For a nonsymmetric matrix the
step $\mathbf{A}^\top = \mathbf{A}$ is not available, which is why the angle
was $108.4^\circ$ there.

Two further facts hold for every real symmetric matrix and are stated without
proof here (the proof of the first needs the complex numbers of Chapter 14): its
eigenvalues are real, and when an eigenvalue is repeated one can still choose
orthogonal eigenvectors inside its eigenspace (the set of all eigenvectors
of that eigenvalue, with the zero vector). The conclusion is the
**spectral theorem**: an $n \times n$ symmetric matrix has $n$ real
eigenvalues and $n$ mutually orthogonal unit eigenvectors. Numerically, on a
$3 \times 3$ example:

<!-- worked: decompositions.symmetric_orthogonal -->
A = [[4, 1, 2], [1, 3, 0], [2, 0, 5]] (symmetric).

| lambda | unit eigenvector v | A v | lambda v |
|---:|---|---|---|
| 6.6691 | (0.6312, 0.172, 0.7563) | (4.209, 1.147, 5.044) | (4.209, 1.147, 5.044) |
| 3.476 | (0.3744, 0.7864, -0.4913) | (1.301, 2.734, -1.708) | (1.301, 2.734, -1.708) |
| 1.8549 | (0.6793, -0.5932, -0.432) | (1.26, -1.1, -0.8013) | (1.26, -1.1, -0.8013) |

Pairwise dot products of the eigenvectors: v0.v1 = 0, v0.v2 = 0, v1.v2 = 0.
Q^T Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].
Largest entry of |Q diag(lambda) Q^T - A|: 2.22e-15.
<!-- /worked -->

The three eigenvectors have pairwise dot product $0$ (to the printed
precision). The matrix $\mathbf{Q}$ whose columns are those eigenvectors
satisfies $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$, and
$\mathbf{Q}\,\mathrm{diag}(\lambda)\,\mathbf{Q}^\top$ reproduces $\mathbf{A}$ to
about $10^{-15}$. That property, $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$,
defines a very useful kind of matrix.

## Orthogonal matrices: maps that keep lengths and angles

Take three vectors $\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3$ that each have
length $1$ and are pairwise orthogonal, and make them the columns of
$\mathbf{Q}$. Look at the product $\mathbf{Q}^\top\mathbf{Q}$. Its entry
$(i, j)$ is row $i$ of $\mathbf{Q}^\top$ (which is column $i$ of $\mathbf{Q}$)
times column $j$ of $\mathbf{Q}$, that is the dot product $\mathbf{q}_i\cdot
\mathbf{q}_j$. That is $1$ when $i = j$ (the vector has length $1$) and $0$
otherwise (orthogonal), so

$$
\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}.
$$

A square matrix with this property, equivalently with orthonormal columns, is
an **orthogonal matrix**. Since $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ says
that $\mathbf{Q}^\top$ undoes $\mathbf{Q}$, it is the inverse:
$\mathbf{Q}^{-1} = \mathbf{Q}^\top$, and also $\mathbf{Q}\mathbf{Q}^\top =
\mathbf{I}$, so the *rows* are orthonormal too. Inverting an orthogonal matrix
costs nothing: it is a transpose, which Chapter 5 showed is a view over the
same numbers.

Here is a $3 \times 3$ example with rational entries, so that every check is
exact.

<!-- worked: decompositions.orthogonal_matrix -->
Q = (1/3) [[1, 2, 2], [2, 1, -2], [2, -2, 1]].

| quantity | value |
|---|---|
| column norms squared | 1, 1, 1 |
| column dot products (1,2), (1,3), (2,3) | 0, 0, 0 |
| Q^T Q | [[1, 0, 0], [0, 1, 0], [0, 0, 1]] |
| Q Q^T | [[1, 0, 0], [0, 1, 0], [0, 0, 1]] |
| x | (3, 0, 4) |
| Q x | (11/3, -2/3, 10/3) |
| \|x\|^2 and \|Q x\|^2 | 25 and 25 |
| y | (1, 2, -1) |
| x . y and (Q x) . (Q y) | -1 and -1 |
| det Q | -1 |
<!-- /worked -->

Why this matrix is special as a *map* follows from the same one-line
computation. For any two vectors, using the transpose of a product from
Chapter 4,

$$
(\mathbf{Q}\mathbf{x})\cdot(\mathbf{Q}\mathbf{y})
= (\mathbf{Q}\mathbf{x})^\top(\mathbf{Q}\mathbf{y})
= \mathbf{x}^\top\mathbf{Q}^\top\mathbf{Q}\,\mathbf{y}
= \mathbf{x}^\top\mathbf{y} = \mathbf{x}\cdot\mathbf{y}.
$$

An orthogonal matrix **preserves every dot product**. Taking $\mathbf{y} =
\mathbf{x}$ gives $\|\mathbf{Q}\mathbf{x}\|^2 = \|\mathbf{x}\|^2$: lengths are
preserved. Since the angle between two vectors is determined by their dot
product and their lengths (the cosine formula of Chapter 3), angles are
preserved. The table shows it: $(3, 0, 4)$ and its image both have squared
length $25$, and the pair $\mathbf{x}$, $\mathbf{y}$ has dot product $-1$
before and after. An orthogonal map moves the space as a rigid object.

Three consequences are used constantly.

- **The determinant is $+1$ or $-1$.** $\det(\mathbf{Q}^\top\mathbf{Q}) = 1$, the
  product rule gives $\det\mathbf{Q}^\top\det\mathbf{Q} = (\det\mathbf{Q})^2 =
  1$. The sign separates **rotations** ($+1$, orientation kept) from
  **reflections** ($-1$, orientation flipped); the example above has $-1$.
- **Products stay orthogonal.** $(\mathbf{Q}_1\mathbf{Q}_2)^\top(\mathbf{Q}_1
  \mathbf{Q}_2) = \mathbf{Q}_2^\top\mathbf{Q}_1^\top\mathbf{Q}_1\mathbf{Q}_2 =
  \mathbf{Q}_2^\top\mathbf{Q}_2 = \mathbf{I}$. Chains of rotations are rotations.
- **No error is amplified.** Multiplying by $\mathbf{Q}$ cannot make a vector,
  or a rounding error in it, longer. This is why numerical methods are built
  from orthogonal steps whenever they can be.

Putting the spectral theorem and this section together, a symmetric
$\mathbf{A}$ is

$$
\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top,
$$

the earlier $\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ with
$\mathbf{V}^{-1} = \mathbf{V}^\top$ because the eigenvector matrix is
orthogonal. In words: **rotate** the input so that the eigenvector
directions become the coordinate axes, **stretch** each axis by its
eigenvalue, and **rotate back**. Rotations can be written down explicitly,
and a rotate-stretch-rotate description exists for every matrix, symmetric or
not, square or not.

## Rotation matrices

Rotations are the orthogonal matrices with determinant $+1$, and in the plane
we can write them down. Turn the whole plane anticlockwise by an angle
$\theta$. A linear map is determined by what it does to the unit vectors
(Chapter 4), so we need only find the images of $\mathbf{e}_1$ and
$\mathbf{e}_2$, the columns of the matrix.

By the definition of cosine and sine from the unit circle (Chapter 3), the
point reached from $(1, 0)$ by walking an angle $\theta$ anticlockwise around
the circle is $(\cos\theta, \sin\theta)$. That is the image of $\mathbf{e}_1$.
The vector $\mathbf{e}_2 = (0, 1)$ is $\mathbf{e}_1$ turned by $90^\circ$. Turning
any vector $(x, y)$ by $90^\circ$ gives $(-y, x)$ (check with $(1, 0) \mapsto
(0, 1)$ and $(0, 1) \mapsto (-1, 0)$), so after the rotation by $\theta$ the
image of $\mathbf{e}_2$ is the image of $\mathbf{e}_1$ turned a further
$90^\circ$: $(-\sin\theta, \cos\theta)$. The columns give

$$
\mathbf{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}.
$$

<!-- worked: decompositions.rotation_matrix_table -->
R(theta) = [[cos theta, -sin theta], [sin theta, cos theta]], applied to e1 = (1, 0) and p = (2, 1), |p| = 2.2361.

| theta (degrees) | R(theta) | R e1 | R p | det R | \|R p\| |
|---:|---|---|---|---:|---:|
| 0 | [[1, 0], [0, 1]] | (1, 0) | (2, 1) | 1 | 2.2361 |
| 30 | [[0.866, -0.5], [0.5, 0.866]] | (0.866, 0.5) | (1.232, 1.866) | 1 | 2.2361 |
| 45 | [[0.7071, -0.7071], [0.7071, 0.7071]] | (0.7071, 0.7071) | (0.7071, 2.121) | 1 | 2.2361 |
| 90 | [[0, -1], [1, 0]] | (0, 1) | (-1, 2) | 1 | 2.2361 |
| 180 | [[-1, 0], [0, -1]] | (-1, 0) | (-2, -1) | 1 | 2.2361 |
<!-- /worked -->

The matrix is orthogonal for a reason visible in its entries: the columns have
length$^2$ $\cos^2\theta + \sin^2\theta = 1$ and dot product $-\cos\theta\sin\theta
+ \sin\theta\cos\theta = 0$. The determinant is $\cos^2\theta + \sin^2\theta =
1$. The table shows the length of $\mathbf{p} = (2, 1)$ unchanged at every angle.

### Composing rotations adds angles

Rotate by $\beta$ and then by $\alpha$. The total is a rotation by $\alpha +
\beta$, since turning a vector by two angles in a row is turning it by their
sum. In matrices, "$\beta$ first, then $\alpha$" is the product
$\mathbf{R}(\alpha)\mathbf{R}(\beta)$, and multiplying out the entries gives

$$
\mathbf{R}(\alpha)\mathbf{R}(\beta) =
\begin{pmatrix}
\cos\alpha\cos\beta - \sin\alpha\sin\beta & -(\sin\alpha\cos\beta + \cos\alpha\sin\beta) \\
\sin\alpha\cos\beta + \cos\alpha\sin\beta & \cos\alpha\cos\beta - \sin\alpha\sin\beta
\end{pmatrix}.
$$

Comparing with $\mathbf{R}(\alpha + \beta)$, whose first column is
$(\cos(\alpha + \beta), \sin(\alpha + \beta))$, reads off the angle-addition
formulas $\cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$ and
$\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$. They are
consequences of the geometry, not facts to memorise.

<!-- worked: decompositions.rotation_composition -->
| expression | value |
|---|---|
| R(20) R(25) | [[0.707107, -0.707107], [0.707107, 0.707107]] |
| R(25) R(20) | [[0.707107, -0.707107], [0.707107, 0.707107]] |
| R(45) | [[0.707107, -0.707107], [0.707107, 0.707107]] |
| cos 20 cos 25 - sin 20 sin 25 | 0.707107 |
| cos 45 | 0.707107 |
| sin 20 cos 25 + cos 20 sin 25 | 0.707107 |
| sin 45 | 0.707107 |
| largest \|R(20) R(25) - R(45)\| entry | 1.11e-16 |
| R(30)^T | [[0.86603, 0.5], [-0.5, 0.86603]] |
| R(-30) | [[0.86603, 0.5], [-0.5, 0.86603]] |
| R(30)^T R(30) | [[1, 0], [0, 1]] |
<!-- /worked -->

So $\mathbf{R}(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\alpha + \beta)$, the order
does not matter for planar rotations (addition commutes, unlike general
matrix products), and the inverse of a rotation is the rotation by the
opposite angle: $\mathbf{R}(-\theta) = \mathbf{R}(\theta)^\top$, as the last
rows show. (Rotations in three dimensions, about different axes, do not
commute; that is a story for a geometry text.)

### Rotating pairs of coordinates, and the offset that survives

A vector with many coordinates can be rotated in the same way by splitting it
into pairs and rotating each pair by its own angle. The block-diagonal matrix
with $\mathbf{R}(\theta_1), \mathbf{R}(\theta_2), \dots$ on its diagonal is
orthogonal, and it is the building block of rotary position embeddings, which
Volume V defines. A short derivation shows why anyone would want it. Take two
vectors $\mathbf{q}$ and $\mathbf{k}$ and rotate $\mathbf{q}$ by an angle
proportional to a position $p$, and $\mathbf{k}$ by the same rate times a
position $s$, so $\mathbf{R}(p\omega)\mathbf{q}$ and $\mathbf{R}(s\omega)
\mathbf{k}$. Their dot product is

$$
(\mathbf{R}(p\omega)\mathbf{q})\cdot(\mathbf{R}(s\omega)\mathbf{k})
= \mathbf{q}^\top\mathbf{R}(p\omega)^\top\mathbf{R}(s\omega)\mathbf{k}
= \mathbf{q}^\top\mathbf{R}(-p\omega)\mathbf{R}(s\omega)\mathbf{k}
= \mathbf{q}^\top\mathbf{R}\big((s - p)\omega\big)\mathbf{k}.
$$

It depends on the positions only through the **difference** $s - p$. The same
holds pair by pair, with a different rate $\omega$ per pair:

<!-- worked: decompositions.rotation_relative_position -->
q = (0.6, 0.8, 1.0, -0.5), k = (1.0, 0.5, -0.4, 0.9) in two coordinate pairs; pair 0 is rotated by position * 0.5 radians, pair 1 by position * 0.05 radians. The dot product of the rotated vectors:

| position of q | position of k | offset s - p | rotated q . rotated k |
|---|---:|---:|---:|
| 2 | 5 | 3 | -0.375577 |
| 10 | 13 | 3 | -0.375577 |
| 100 | 103 | 3 | -0.375577 |
| 7 | 3 | -4 | -1.56478 |
| 50 | 50 | 0 | 0.15 |

The same quantity computed with a single rotation by the offset, sum over pairs of q_pair . R(offset * omega) k_pair:

| offset | value |
|---|---:|
| 3 | -0.375577 |
| -4 | -1.56478 |
| 0 | 0.15 |
<!-- /worked -->

Moving both positions by the same amount (rows $(2, 5)$, $(10, 13)$ and
$(100, 103)$, all at offset $3$) changes nothing. The two ways of computing the
quantity, rotating both vectors and then taking the dot product, or rotating one
by the offset, agree digit for digit. Volume V builds a position encoding on
this property.

## The singular value decomposition

Symmetric matrices are rotate-stretch-rotate-back with the *same* rotation at
both ends. A general matrix, perhaps not even square, can be described with two
different rotations. Take $\mathbf{A} = \begin{pmatrix}3&0\\4&5\end{pmatrix}$
and push the unit circle through it (Figure 3). Whatever the matrix, the image
of a circle under a linear map is an ellipse (possibly flattened). Its two
axes are the directions of the longest and shortest images; the circle's
points that land on the axes are two perpendicular directions, $\mathbf{v}_1$
and $\mathbf{v}_2$. So the map takes the orthogonal pair $\mathbf{v}_1,
\mathbf{v}_2$ to an orthogonal pair of directions $\mathbf{u}_1, \mathbf{u}_2$,
stretched by $\sigma_1$ and $\sigma_2$:
$\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$. (In this chapter $\sigma_i$,
with an index, is always the $i$-th stretch factor, a singular value; it is
unrelated to the sigmoid $\sigma(x)$, which has an argument, and to the
standard deviation $\sigma$ of Chapters 11 and 12. Likewise
$\boldsymbol{\Sigma}$ here is the diagonal matrix of the $\sigma_i$, not a
covariance matrix.)

![Four panels showing the unit circle pushed through the matrix with rows (3, 0) and (4, 5) in three steps. Panel 1: the unit circle with two perpendicular right singular vectors v1 and v2. Panel 2: after the rotation V transpose the circle is unchanged and v1, v2 lie on the coordinate axes. Panel 3: after the stretch Sigma the circle is an axis-aligned ellipse with semi-axes 6.71 and 2.24. Panel 4: after the rotation U the ellipse is tilted, with its axes sigma1 u1 and sigma2 u2 drawn. A black dot follows the point (1, 0) through all four panels.](img/svd-circle-ellipse.svg)

**Figure 3.** The unit circle through $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}
\mathbf{V}^\top$ in three steps: rotate ($\mathbf{V}^\top$), stretch
($\boldsymbol{\Sigma}$), rotate ($\mathbf{U}$). Panels 1 and 2 use one scale,
panels 3 and 4 another. Every point is computed from the factors in the
table below; the black dot follows $(1, 0)$.

The picture has an algebraic statement. Stack the $\mathbf{v}_i$ as the columns
of $\mathbf{V}$, the $\mathbf{u}_i$ as the columns of $\mathbf{U}$, and put the
$\sigma_i$ on the diagonal of $\boldsymbol{\Sigma}$. Then $\mathbf{A}\mathbf{V}
= \mathbf{U}\boldsymbol{\Sigma}$, and since $\mathbf{V}$ is orthogonal,
$\mathbf{V}^{-1} = \mathbf{V}^\top$:

$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top .
$$

This is the **singular value decomposition** (SVD): an orthogonal
$\mathbf{V}^\top$ rotates, a diagonal $\boldsymbol{\Sigma}$ with non-negative
entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ stretches the coordinate axes,
and an orthogonal $\mathbf{U}$ rotates again. The $\sigma_i$ are the
**singular values**, the columns of $\mathbf{V}$ the right singular vectors and
the columns of $\mathbf{U}$ the left singular vectors.

### Computing it from a symmetric matrix

How do we find $\mathbf{v}_i$ and $\sigma_i$? The trick is to turn $\mathbf{A}$
into a symmetric matrix, for which we have the spectral theorem.
Consider $\mathbf{A}^\top\mathbf{A}$, the **Gram matrix** of $\mathbf{A}$ (the
table of dot products between its columns), which is symmetric as shown above. Its quadratic
form is a squared length: $\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x} =
(\mathbf{A}\mathbf{x})\cdot(\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|^2
\ge 0$. By the spectral theorem it has orthonormal eigenvectors
$\mathbf{v}_1, \dots, \mathbf{v}_n$ with eigenvalues $\lambda_i$, and applying
the squared-length identity to an eigenvector gives

$$
\|\mathbf{A}\mathbf{v}_i\|^2 = \mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_i
= \lambda_i\,\mathbf{v}_i^\top\mathbf{v}_i = \lambda_i .
$$

So $\lambda_i \ge 0$, and the length of the image of $\mathbf{v}_i$ is
$\sigma_i = \sqrt{\lambda_i}$. For two different eigenvectors,
$(\mathbf{A}\mathbf{v}_i)\cdot(\mathbf{A}\mathbf{v}_j) =
\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j = \lambda_j\,\mathbf{v}_i
\cdot\mathbf{v}_j = 0$: **the images are orthogonal as well.** Dividing each
image by its length, $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$ (when
$\sigma_i > 0$), gives an orthonormal set, and $\mathbf{A}\mathbf{v}_i =
\sigma_i\mathbf{u}_i$ holds by construction. The recipe:

1. Form $\mathbf{A}^\top\mathbf{A}$ and find its eigenvalues and orthonormal
   eigenvectors, sorted by decreasing eigenvalue. The eigenvectors are the
   $\mathbf{v}_i$.
2. Set $\sigma_i = \sqrt{\lambda_i}$.
3. Set $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$.

<!-- worked: decompositions.svd_by_hand -->
A = [[3, 0], [4, 5]], A^T A = [[25, 20], [20, 25]].
det(A^T A - lambda I) = (25 - lambda)^2 - 400 = lambda^2 - 50 lambda + 225 = 0, so lambda = (50 +- 40)/2 = 45 and 5.

| i | lambda_i | sigma_i = sqrt lambda_i | v_i | A v_i | u_i = A v_i / sigma_i | \|u_i\| |
|---:|---:|---:|---|---|---|---:|
| 1 | 45 | 6.7082 | (0.7071, 0.7071) | (2.121, 6.364) | (0.3162, 0.9487) | 1 |
| 2 | 5 | 2.2361 | (0.7071, -0.7071) | (2.121, -0.7071) | (0.9487, -0.3162) | 1 |

u_1 . u_2 = 0; v_1 . v_2 = 0.
U Sigma V^T = [[3, 0], [4, 5]].
sigma_1 sigma_2 = 15, |det A| = 15.
<!-- /worked -->

For $\mathbf{A} = \begin{pmatrix}3&0\\4&5\end{pmatrix}$ the Gram matrix
$\mathbf{A}^\top\mathbf{A}$ is $\begin{pmatrix}25&20\\20&25\end{pmatrix}$ and the
characteristic polynomial factors as $(\lambda - 45)(\lambda - 5)$, so
$\sigma_1 = \sqrt{45} = 3\sqrt5 \approx 6.708$ and $\sigma_2 = \sqrt5 \approx
2.236$. The right singular vectors are $(1, 1)/\sqrt2$ and $(1, -1)/\sqrt2$,
the left ones $(1, 3)/\sqrt{10}$ and $(3, -1)/\sqrt{10}$, and the product
$\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ returns $\mathbf{A}$. Two
cross-checks: the singular values multiply to $15 = |\det\mathbf{A}|$ (the
determinant of the factors is $\pm1 \cdot \sigma_1\sigma_2 \cdot \pm1$), and
Figure 3's panels can be read off the next table, where $\mathbf{V}^\top\mathbf{x}$
are the coordinates of $\mathbf{x}$ along the $\mathbf{v}_i$.

<!-- worked: decompositions.svd_rotate_stretch_rotate -->
Unit vectors x pushed through the three factors of A = U Sigma V^T for A = [[3, 0], [4, 5]]: V^T x are the coordinates of x along v_1, v_2; Sigma stretches them; U places them along u_1, u_2.

| x | V^T x | Sigma V^T x | U Sigma V^T x | A x |
|---|---|---|---|---|
| (1, 0) | (0.707, 0.707) | (4.74, 1.58) | (3, 4) | (3, 4) |
| (0.707, 0.707) | (1, 0) | (6.71, 0) | (2.12, 6.36) | (2.12, 6.36) |
| (0, 1) | (0.707, -0.707) | (4.74, -1.58) | (0, 5) | (0, 5) |
| (-0.707, 0.707) | (0, -1) | (0, -2.24) | (-2.12, 0.707) | (-2.12, 0.707) |
| (-1, 0) | (-0.707, -0.707) | (-4.74, -1.58) | (-3, -4) | (-3, -4) |
| (-0.707, -0.707) | (-1, 0) | (-6.71, 0) | (-2.12, -6.36) | (-2.12, -6.36) |
| (0, -1) | (-0.707, 0.707) | (-4.74, 1.58) | (0, -5) | (0, -5) |
| (0.707, -0.707) | (0, 1) | (0, 2.24) | (2.12, -0.707) | (2.12, -0.707) |
<!-- /worked -->

The last two columns agree: pushing $\mathbf{x}$ through the
three factors is the same as multiplying by $\mathbf{A}$. The $\sigma$'s have a
direct meaning as stretch factors, which a scan over every whole degree
confirms:

<!-- worked: decompositions.stretch_scan -->
|  | angle (degrees) | \|A x\| | singular value |
|---|---:|---:|---:|
| longest image | 45 | 6.7082 | 6.7082 |
| shortest image | 135 | 2.2361 | 2.2361 |

Scan of |A x| over x at every whole degree from 0 to 359. The longest image lies along v_1 (45 degrees); the shortest lies along the line of v_2, whose upper end is at 135 degrees.
<!-- /worked -->

The longest image of any unit vector is $\sigma_1$ and the shortest is
$\sigma_2$. In general, write $\mathbf{x} = \sum_i c_i\mathbf{v}_i$ with
$\sum c_i^2 = 1$ (the $\mathbf{v}_i$ are an orthonormal basis); then
$\|\mathbf{A}\mathbf{x}\|^2 = \sum_i \sigma_i^2c_i^2 \le \sigma_1^2$, with
equality at $\mathbf{x} = \mathbf{v}_1$. The largest stretch of a matrix, which
we call its **spectral norm**, is its first singular value.

### Any shape, and rank

Nothing above needed a square $\mathbf{A}$. For an $m \times n$ matrix,
$\mathbf{A}^\top\mathbf{A}$ is $n \times n$ and gives $n$ right singular
vectors; $\mathbf{A}\mathbf{v}_i$ lives in $\mathbb{R}^m$. If some $\lambda_i =
0$ then $\mathbf{A}\mathbf{v}_i = \mathbf{0}$: the map crushes that direction
and there is no $\mathbf{u}_i$ to divide out. Only the $r$ nonzero singular
values matter, so the compact (**thin**) form keeps $\mathbf{U}$ of shape
$[m, r]$, $\boldsymbol{\Sigma}$ of shape $[r, r]$ and $\mathbf{V}$ of shape
$[n, r]$; the **full** form completes them to square orthogonal matrices by
adding directions that multiply zeros. The number of nonzero singular values
is the rank, because each nonzero $\sigma_i$ is one independent direction that
survives the map and each zero is one that is crushed.

<!-- worked: decompositions.svd_shapes_and_rank -->
| matrix | singular values | rank = number of nonzero singular values |
|---|---|---|
| A = [[1, 0], [0, 1], [1, 1]] (shape [3, 2]) | (1.7321, 1) | 2 |
| B = [[1, 2], [2, 4]] (shape [2, 2], second row twice the first) | (5, 0) | 1 |
| C = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] (chapter 4's singular matrix) | (16.848, 1.0684, 0) | 2 |

For A the Gram matrix A^T A = [[2, 1], [1, 2]] has eigenvalues 3 and 1, so sigma = (sqrt 3, 1) = (1.7321, 1); u_1 = (0.4082, 0.4082, 0.8165), u_2 = (0.7071, -0.7071, 0).

| factor | shape (full SVD) | shape (thin SVD) |
|---|---|---|
| U | [3, 3] | [3, 2] |
| Sigma | [3, 2] | [2, 2] |
| V^T | [2, 2] | [2, 2] |
<!-- /worked -->

This gives the precise answer to the question that Chapter 4 left open: a
matrix of rank $3$ on paper may have a third singular value of $10^{-12}$ next to
a first of $1$, in which case it is, in Chapter 4's words, "rank 3 on paper and,
for every practical purpose, rank 2". The right way to say how close a matrix is to losing rank is to look
at its singular values.

## Low-rank approximation

The decomposition has a second reading. Multiply $\mathbf{U}\boldsymbol{\Sigma}
\mathbf{V}^\top$ out one singular triple at a time:

$$
\mathbf{A} = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top + \sigma_2\mathbf{u}_2
\mathbf{v}_2^\top + \dots + \sigma_r\mathbf{u}_r\mathbf{v}_r^\top .
$$

Each term $\mathbf{u}\mathbf{v}^\top$ is a column $[m, 1]$ times a row $[1, n]$,
an $[m, n]$ matrix whose every column is a multiple of $\mathbf{u}$: a
**rank-one matrix**, described by $m + n$ numbers instead of $mn$. A matrix is
a sum of rank-one pieces, ordered from the one that matters most. Keeping only
the first $r$ pieces gives the **truncated SVD**

$$
\mathbf{A}_r = \sum_{i=1}^{r}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top,
$$

a matrix of rank at most $r$. A **low-rank approximation** of $\mathbf{A}$ is
a matrix of small rank $r$ that is close to $\mathbf{A}$. The block shows the
pieces for our $2 \times 2$ example. Besides the largest stretch (the spectral
norm above), it reports the Frobenius norm $\|\cdot\|_F$ of the error, the
Euclidean norm of all the entries of a matrix read as one long vector, which
is defined properly below.

<!-- worked: decompositions.rank_one_pieces -->
A = [[3, 0], [4, 5]] = sigma_1 u_1 v_1^T + sigma_2 u_2 v_2^T.

| piece | matrix | rank |
|---|---:|---:|
| sigma_1 u_1 v_1^T | [[1.5, 1.5], [4.5, 4.5]] | 1 |
| sigma_2 u_2 v_2^T | [[1.5, -1.5], [-0.5, 0.5]] | 1 |
| sum | [[3, 0], [4, 5]] | 2 |
| A - A_1 | [[1.5, -1.5], [-0.5, 0.5]] | 1 |

Frobenius norm of A - A_1 = 2.2361 = sigma_2 = 2.2361; the largest stretch of A - A_1 is also 2.2361.
<!-- /worked -->

The first piece alone, $\mathbf{A}_1$, leaves an error whose size is exactly
$\sigma_2$. Whether $\mathbf{A}_1$ is a good approximation depends on how
$\sigma_2 = 2.236$ compares with $\sigma_1 = 6.708$; here it is not very good.

### A matrix with a low-rank core

To see an approximation worth having we need a matrix with structure. Define
an $8 \times 6$ table, $i = 0, \dots, 7$ and $j = 0, \dots, 5$, as three smooth
patterns of decreasing strength and a little deterministic noise:

$$
M_{ij} = \sum_{k=1}^{3} w_k \,\sin\!\Big(\frac{k\pi(i + \tfrac12)}{8}\Big)
\cos\!\Big(\frac{k\pi(j + \tfrac12)}{6}\Big)
\;+\; 0.05\cdot\frac{((7i + 13j) \bmod 11) - 5}{5},
$$

with $(w_1, w_2, w_3) = (6, 3, 1.5)$.

Each pattern is a rank-one matrix (a function of $i$ times a function of $j$),
so without the noise this table would have rank $3$. The singular values
reveal exactly that:

<!-- worked: decompositions.spectrum_table -->
The 8 x 6 table M (three smooth patterns plus noise, defined in the text).

| i | sigma_i | share of \|M\|_F^2 (%) | cumulative share (%) |
|---|---:|---:|---:|
| 1 | 20.79 | 76.16 | 76.157 |
| 2 | 10.404 | 19.07 | 95.228 |
| 3 | 5.2038 | 4.771 | 99.999 |
| 4 | 0.068053 | 0.000816 | 100 |
| 5 | 0.034737 | 0.0002126 | 100 |
| 6 | 0.019712 | 6.846e-05 | 100 |

|M|_F = 23.823, sqrt(sum sigma_i^2) = 23.823.
<!-- /worked -->

The first three singular values carry $99.999\%$ of the squared size of the
table, and the fourth is about seventy-six times smaller than the third. The left
half of Figure 4 plots them: a cliff after the third.

![Two panels. Left: the six singular values of the 8 by 6 table, about 20.8, 10.4, 5.2, then three values near zero (0.07, 0.03, 0.02), a cliff after the third. Right: base-10 logarithm of the relative error of the rank-r truncation for r from 0 to 5, falling from 0 to about -0.3, -0.66, then dropping to -2.5 at r = 3 and -3.1 at r = 5; circles showing the square root of the sum of dropped squared singular values lie exactly on the line.](img/singular-spectrum.svg)

**Figure 4.** Left: singular values of the $8 \times 6$ table. Right: relative
error $\|\mathbf{M} - \mathbf{M}_r\|_F/\|\mathbf{M}\|_F$ of the rank-$r$
truncation (line) next to the square root of the sum of the dropped squared
singular values (circles); the two are the same number.

### How good is the truncation: the error is the dropped singular values

To measure the distance between two matrices we need a norm for matrices. The
**Frobenius norm** is the Euclidean norm of Chapter 3 applied to the matrix read
as one long vector (Chapter 5's flat view): $\|\mathbf{A}\|_F = \sqrt{\sum_{ij}
A_{ij}^2}$. Multiplying by an orthogonal matrix does not change it. For
$\mathbf{Q}\mathbf{A}$: column $j$ of the product is $\mathbf{Q}\mathbf{a}_{:j}$,
which has the same length as $\mathbf{a}_{:j}$ (an orthogonal matrix
keeps lengths, shown above), and $\|\mathbf{A}\|_F^2$ is the sum of the squared column
lengths. For $\mathbf{A}\mathbf{Q}$: $\|\mathbf{A}\mathbf{Q}\|_F =
\|\mathbf{Q}^\top\mathbf{A}^\top\|_F = \|\mathbf{A}^\top\|_F =
\|\mathbf{A}\|_F$, since transposing only reorders the entries. Hence
$\|\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\|_F = \|\boldsymbol{\Sigma}\|_F
= \sqrt{\sum_i\sigma_i^2}$: **the squared size of a matrix is the sum of its
squared singular values**.

The truncation error $\mathbf{A} - \mathbf{A}_r = \sum_{i > r}\sigma_i\mathbf{u}_i
\mathbf{v}_i^\top$ is itself a singular value decomposition, with the singular
values $\sigma_{r+1}, \sigma_{r+2}, \dots$. Applying the two facts just
derived to it:

$$
\|\mathbf{A} - \mathbf{A}_r\|_F = \sqrt{\sigma_{r+1}^2 + \sigma_{r+2}^2 + \dots},
\qquad
\text{largest stretch of } (\mathbf{A} - \mathbf{A}_r) = \sigma_{r+1}.
$$

<!-- worked: decompositions.truncation_error -->
M is 8 x 6: storing it takes m n = 48 numbers; a rank-r approximation as two factors takes r (m + n) = r * 14.

| r | numbers stored r(m+n) | \|M - M_r\|_F | sqrt(sum of dropped sigma^2) | largest stretch of M - M_r | sigma_(r+1) |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 23.823 | 23.823 | 20.79 | 20.79 |
| 1 | 14 | 11.633 | 11.633 | 10.404 | 10.404 |
| 2 | 28 | 5.2044 | 5.2044 | 5.2038 | 5.2038 |
| 3 | 42 | 0.078908 | 0.078908 | 0.068053 | 0.068053 |
| 4 | 56 | 0.03994 | 0.03994 | 0.034737 | 0.034737 |
| 5 | 70 | 0.019712 | 0.019712 | 0.019712 | 0.019712 |
| 6 | 84 | 0 | 0 | 0 | 0 |
<!-- /worked -->

The table checks both: the measured Frobenius error equals the square root of the
dropped squares in every row, and the largest stretch of the residual equals
$\sigma_{r+1}$. Keeping $r = 3$ numbers of structure costs $3 \times 14 = 42$
numbers and leaves an error of $0.079$, against a table of norm $23.8$; keeping
$r = 2$ leaves $5.2$, which is the size of the dropped third pattern.

Is the truncated SVD the *best* rank-$r$ matrix, or just a convenient one? The
**Eckart-Young theorem** says it is the best, in both norms: no matrix of rank
at most $r$ is closer to $\mathbf{A}$ than $\mathbf{A}_r$, neither in Frobenius
norm nor in largest stretch. We do not prove the Frobenius case here. The
largest-stretch case has a short argument. Let $\mathbf{X}$ be any matrix of
rank at most $r$. It crushes at least an $(n - r)$-dimensional set of directions
to zero (this is the counting behind rank in Chapter 4: $r$ pivots leave $n - r$
free columns). Directions $\mathbf{v}_1, \dots, \mathbf{v}_{r+1}$ span an
$(r+1)$-dimensional set. Because $(n - r) + (r + 1) = n + 1 > n$, the two sets
share a nonzero unit vector $\mathbf{x}$. (Why: put a basis of each set as the
columns of one $n \times (n + 1)$ matrix. Elimination (Chapter 4) has at most $n$
pivots, so some column is free, which gives a combination of the basis
vectors, not all coefficients zero, that adds up to zero; moved to two sides,
a nonzero vector of the first set equals a vector of the second.) For it $\mathbf{X}\mathbf{x} =
\mathbf{0}$, so $\|(\mathbf{A} - \mathbf{X})\mathbf{x}\| = \|\mathbf{A}
\mathbf{x}\|$, and writing $\mathbf{x}$ in the $\mathbf{v}_i$ with only
$i \le r + 1$ present, $\|\mathbf{A}\mathbf{x}\|^2 = \sum_{i \le r+1}
\sigma_i^2c_i^2 \ge \sigma_{r+1}^2$. Every rank-$r$ matrix therefore misses
$\mathbf{A}$ by at least $\sigma_{r+1}$ in some direction, and the truncated SVD
misses by exactly that.

We can also test it. Here are four ways to build a rank-$r$ approximation of
$\mathbf{M}$: the truncated SVD; the projection of $\mathbf{M}$ onto the
span of its $r$ longest columns; the truncated SVD of $\mathbf{M}$ plus noise of
amplitude $0.3$ (each entry gets $0.6(u - \tfrac12)$ for a uniform $u$ in $[0, 1)$);
and the best of $2000$ random $r$-dimensional column spaces
onto which $\mathbf{M}$ is projected. The projections are onto subspaces of
$\mathbb{R}^8$, the space of the columns. The block states the generator and
the exact order of its draws, so that the last two columns can be reproduced
digit for digit.

<!-- worked: decompositions.eckart_young_check -->
Frobenius norm of M - X (the error) for rank-r matrices X built four ways (smaller is better), printed to 5 significant digits.

Column 2: orthonormalise (Gram-Schmidt, in order of decreasing column length) the r columns of M with the largest Euclidean norm, and project M onto their span in R^8. Columns 3 and 4 draw from ONE SplitMix64 generator with seed 8136057881202253862 (0x70E915894913E026), seeded once and never reseeded, used for r = 1, 2, 3, 4 in turn. For each r: first the 48 noise values 0.6 * (uniform() - 0.5), row by row (row 0 first), added to M for column 3; then 2000 random subspaces for column 4, each made of r vectors of 8 entries (uniform() - 0.5), drawn vector by vector and orthonormalised by Gram-Schmidt in the order drawn; M is projected on the span and the smallest error of the 2000 is kept.

| r | truncated SVD of M | project on the r longest columns | truncated SVD of M + noise | best of 2000 random subspaces |
|---|---:|---:|---:|---:|
| 1 | 11.633 | 14.091 | 11.642 | 13.564 |
| 2 | 5.2044 | 6.3407 | 5.2857 | 11.83 |
| 3 | 0.078908 | 0.11595 | 1.0555 | 7.8013 |
| 4 | 0.03994 | 0.055165 | 1.1696 | 6.5225 |
<!-- /worked -->

In every row the first column is the smallest. The projection onto the longest
columns stays within about a quarter of the optimum ($14.09$ against $11.63$ at
$r = 1$), and the SVD of the noisy copy is nearly as good while only the two
strongest patterns are needed ($5.29$ against $5.20$ at $r = 2$). At $r = 3$,
where the SVD finds the third pattern and the error drops to $0.079$, the
column projection gives $0.116$ and the noisy copy $1.06$, an error made of the
noise it carries. The random subspaces are far worse, as expected: among all
directions the good ones are rare, and the SVD finds them directly.

### What a rank-$r$ matrix costs

A rank-$r$ matrix does not need $mn$ stored numbers. Keep $\mathbf{A}_r =
\mathbf{B}\mathbf{A}'$ with $\mathbf{B}$ of shape $[m, r]$ (the columns
$\sigma_i\mathbf{u}_i$) and $\mathbf{A}'$ of shape $[r, n]$ (the rows
$\mathbf{v}_i^\top$). That is $r(m + n)$ numbers. It is smaller than $mn$
exactly when $r < mn/(m + n)$, which for a square $m = n$ matrix is $r < n/2$.
Applying it to a vector is also cheaper: $\mathbf{A}_r\mathbf{x} =
\mathbf{B}(\mathbf{A}'\mathbf{x})$ costs $r(m + n)$ multiply-adds instead of
$mn$, by the grouping argument of Chapter 4 (the intermediate vector has only
$r$ entries).

<!-- worked: decompositions.lora_parameter_counts -->
| W shape | r | m n | r (m + n) | ratio m n / r (m + n) | adapter size (% of W) |
|---|---:|---:|---:|---:|---:|
| [4096, 4096] | 8 | 16,777,216 | 65,536 | 256 | 0.391 |
| [4096, 4096] | 16 | 16,777,216 | 131,072 | 128 | 0.781 |
| [4096, 4096] | 64 | 16,777,216 | 524,288 | 32 | 3.12 |
| [4096, 11008] | 16 | 45,088,768 | 241,664 | 186.6 | 0.536 |
| [1024, 1024] | 8 | 1,048,576 | 16,384 | 64 | 1.56 |
<!-- /worked -->

For a $4096 \times 4096$ weight, rank $8$ is $0.39\%$ of the numbers. Two ideas
in later volumes use exactly this arithmetic, in different ways.

**Compressing a matrix that is already there.** If the singular values of a
trained weight fall quickly, the truncated SVD is the best small replacement,
and the error is the dropped singular values. Whether real weights have such a
spectrum is an empirical question for each model, not a theorem; the tool that
answers it is the plot of Figure 4.

**Learning a low-rank change.** Low-rank adaptation (LoRA, Volume XIII) leaves a pretrained
weight $\mathbf{W}$ alone and trains a correction $\Delta\mathbf{W} =
\mathbf{B}\mathbf{A}'$ of rank $r$ directly, rather than ever computing an
SVD: only $r(m + n)$ numbers are trained and stored. The correction starts at
$\mathbf{B} = \mathbf{0}$, so the adapted layer begins as exactly the original
one.

<!-- worked: decompositions.lora_zero_start -->
W [3, 4], A [2, 4], B [3, 2], x = (1, 2, 3, 4).

| quantity | value |
|---|---|
| W x | (30, 6, 5) |
| (W + B_0 A) x with B_0 = 0 | (30, 6, 5) |
| B A | [[0.5, -1, 0, 1], [1, 0, 0.5, 0], [1.5, -1, 0.5, 1]] |
| (W + B A) x | (32.5, 8.5, 10) |
| W x + B (A x), two thin products | (32.5, 8.5, 10) |
| singular values of B A | (2.624, 1.057, 0) |
| rank of B A (r = 2) | 2 |
<!-- /worked -->

The adapted product $(\mathbf{W} + \mathbf{B}\mathbf{A}')\mathbf{x}$ equals the
original plus $\mathbf{B}(\mathbf{A}'\mathbf{x})$, two thin products, and the
correction $\mathbf{B}\mathbf{A}'$ has $r = 2$ nonzero singular values. The
adapter is a bet that the change a task needs is low rank; the table of
singular values is how one would check that bet after training.

A last connection: data is often arranged as a matrix $\mathbf{X}$ with one
example per row. Then $\mathbf{X}^\top\mathbf{X}$ is symmetric, and its
eigenvectors are the right singular vectors of $\mathbf{X}$, because
$\mathbf{X}^\top\mathbf{X} = \mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top$.
The leading eigenvectors are the $r$ directions along which the data has the
most energy, and projecting onto them is the best rank-$r$ summary of the
examples in the Eckart-Young sense. This is principal component analysis,
met again in Volume II.

## Jacobi eigenvalues and the SVD as pseudocode

Power iteration was written out above; the functions below, in the book's
[pseudocode](../pseudocode.md), compute the eigenpairs of a symmetric matrix and the SVD;
the textbook cost of either is on the order of $n^3$ for a square matrix, so
they are for matrices up to a few thousand wide, not for the full weights of a
large model.

```text
function jacobi_eigen(S: array<real>[n, n], sweeps: int)
        -> (array<real>[n], array<real>[n, n]):
    require S.shape[0] == S.shape[1], "jacobi_eigen: not square"
    require close(S, transpose(S), 1e-12, 1e-12), "jacobi_eigen: not symmetric"
    A = copy(S)
    V = identity(n)
    scale = 0.0                                 # squared size of S; rotations do not change it
    for i in 0 .. n:
        for j in 0 .. n:
            scale = scale + A[i, j] * A[i, j]
    for sweep in 0 .. sweeps:
        off = 0.0
        for i in 0 .. n:
            for j in 0 .. n:
                if i != j:
                    off = off + A[i, j] * A[i, j]
        if off <= 1e-30 * scale:                # relative, so the test does not depend on the units of S
            break
        for p in 0 .. n - 1:
            for q in p + 1 .. n:
                if A[p, q] != 0.0:
                    theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                    t = sign(theta) / (abs(theta) + sqrt(theta * theta + 1.0))   # sign(0) = +1
                    c = 1.0 / sqrt(t * t + 1.0)
                    s = t * c
                    J = identity(n)             # a rotation in the (p, q) plane
                    J[p, p] = c
                    J[q, q] = c
                    J[p, q] = s
                    J[q, p] = -s
                    A = matmul(transpose(J), matmul(A, J))
                    V = matmul(V, J)
    values = [A[i, i] for i in 0 .. n]
    return (values, V)                          # eigenvectors are the columns of V
```

Each rotation $\mathbf{J}$ is chosen to make the entry $A_{pq}$ zero: with
$t = s/c$ the condition is $t^2 + 2\theta t - 1 = 0$, and the code takes its
smaller root. A rotation that zeroes one off-diagonal pair can disturb others, but
every rotation reduces the total squared size of the off-diagonal entries, so
sweeping over all pairs converges to a diagonal matrix. The convention
$\mathrm{sign}(0) = +1$ of the pseudocode page matters: when the two diagonal
entries are equal, as in $\begin{pmatrix}25&20\\20&25\end{pmatrix}$ below,
$\theta = 0$, the smaller root is $t = 1$ (a rotation by $45^\circ$), and a
sign function returning $0$ there would give $t = 0$, the identity rotation,
and a method that never makes progress. The product of the
rotations is the orthogonal matrix of eigenvectors, and because each step is a
rotation, rounding error is not amplified. Sort the values and columns in
decreasing order to match the chapter.

```text
function svd_via_gram(A: array<real>[m, n])
        -> (array<real>[m, n], array<real>[n], array<real>[n, n]):
    G = matmul(transpose(A), A)                 # [n, n], symmetric
    (lam, Vj) = jacobi_eigen(G, 60)
    order = argsort(lam, descending)            # indices from the largest eigenvalue down
    V = zeros(real, [n, n])
    sigma = zeros(real, [n])
    U = zeros(real, [m, n])                     # a column stays zero where sigma is 0
    for i in 0 .. n:
        V[:, i] = Vj[:, order[i]]               # the columns of Vj in the same order
        sigma[i] = sqrt(max(lam[order[i]], 0.0))
        if sigma[i] > 0.0:
            U[:, i] = matvec(A, V[:, i]) / sigma[i]     # broadcast by the scalar
    return (U, sigma, V)
```

```text
function svd_one_sided(A: array<real>[m, n], sweeps: int)
        -> (array<real>[m, n], array<real>[n], array<real>[n, n]):
    require m >= n, "svd_one_sided: transpose A first when m < n"
    W = copy(A)
    V = identity(n)
    sigma = zeros(real, [n])
    U = zeros(real, [m, n])
    for sweep in 0 .. sweeps:
        rotated = false
        for i in 0 .. n - 1:
            for j in i + 1 .. n:
                a = dot(W[:, i], W[:, i])
                b = dot(W[:, j], W[:, j])
                g = dot(W[:, i], W[:, j])
                if abs(g) > 1e-15 * sqrt(a * b):
                    rotated = true
                    zeta = (b - a) / (2.0 * g)
                    t = sign(zeta) / (abs(zeta) + sqrt(1.0 + zeta * zeta))   # sign(0) = +1
                    c = 1.0 / sqrt(1.0 + t * t)
                    s = c * t
                    for k in 0 .. m:            # rotate columns i and j of W by (c, s)
                        x = W[k, i]
                        y = W[k, j]
                        W[k, i] = c * x - s * y
                        W[k, j] = s * x + c * y
                    for k in 0 .. n:            # and the same rotation of V
                        x = V[k, i]
                        y = V[k, j]
                        V[k, i] = c * x - s * y
                        V[k, j] = s * x + c * y
        if not rotated:
            break
    for j in 0 .. n:
        sigma[j] = norm(W[:, j])                # the column lengths
    order = argsort(sigma, descending)
    Vs = zeros(real, [n, n])
    ss = zeros(real, [n])
    for j in 0 .. n:                            # reorder everything from the largest sigma down
        Vs[:, j] = V[:, order[j]]
        ss[j] = sigma[order[j]]
        if ss[j] > 0.0:
            U[:, j] = W[:, order[j]] / ss[j]
    return (U, ss, Vs)
```

`svd_one_sided` implements the idea behind $\mathbf{A}\mathbf{V} =
\mathbf{U}\boldsymbol{\Sigma}$ directly: rotate pairs of columns of $\mathbf{A}$
until all columns are mutually orthogonal; then the column lengths are the
singular values and the rotations multiplied together are $\mathbf{V}$. It
never forms $\mathbf{A}^\top\mathbf{A}$, which matters in the next section. (Both SVD
functions return $\mathbf{U}$ of shape $[m, n]$ with a zero column wherever
$\sigma_i = 0$: neither the thin nor the full form, but enough to rebuild
$\mathbf{A}$. The examples below call `svd_one_sided` with `sweeps = 60`; it needs
2 on the matrix of the next section.)

## Comparing decompositions without fooling yourself

**Never compare singular vectors elementwise.** The pair $(\mathbf{u}_i,
\mathbf{v}_i)$ can be negated together without changing $\mathbf{u}_i\mathbf{v}_i^\top$,
and when two singular values are equal any orthonormal pair in their plane is
valid. Two correct libraries can therefore return different $\mathbf{U}$ and
$\mathbf{V}$ for the same matrix. A test must compare what is unique: the
singular values, the reconstruction $\mathbf{U}\boldsymbol{\Sigma}
\mathbf{V}^\top$, the products $\mathbf{u}_i\mathbf{v}_i^\top$ for separated
$\sigma_i$, or the subspace spanned by the leading vectors. The same holds for
eigenvectors, whose sign and (for repeated eigenvalues) basis are free. The
tables of this chapter print each eigenvector or singular vector with the sign
that makes its first nonzero entry positive; a reproduction that gets the
opposite sign has not made a mistake. Entries such as the "largest entry of
$|\mathbf{Q}\,\mathrm{diag}(\lambda)\mathbf{Q}^\top - \mathbf{A}|$" of the
symmetric example and of the rotation-composition example are round-off: their
size (below $10^{-13}$) is what to compare, not their digits, which depend on
the order of the additions.

**Squaring the matrix squares the problem.** The recipe "eigenvalues of
$\mathbf{A}^\top\mathbf{A}$" is exact and clear, and it is also the way to lose
small singular values. Entries of $\mathbf{A}^\top\mathbf{A}$ are sums of
products, so quantities at relative size $\varepsilon$ in $\mathbf{A}$ appear at
$\varepsilon^2$ in the Gram matrix, below the resolution of floating point
(Chapter 15) for $\varepsilon$ around $10^{-8}$ in 64-bit arithmetic:

<!-- worked: decompositions.lauchli_squaring -->
A = [[1, 1], [1e-8, 0], [0, 1e-8]]. The exact singular values are sqrt(2 + 1e-16) and 1e-8.

| route | sigma_1 | sigma_2 |
|---|---:|---:|
| eigenvalues of A^T A (float64) | 1.4142136 | 0.00e+00 |
| one-sided Jacobi on A itself | 1.4142136 | 1.00e-08 |

In float64, A^T A = [[1, 1], [1, 1]]: the 1 + 1e-16 on its diagonal rounds to 1, the two rows become equal, and the small singular value is lost.
<!-- /worked -->

The two routes agree on $\sigma_1$. The Gram route reports $0$ for the second
singular value (the matrix looks rank 1 to it) while the true value is
$10^{-8}$, which the one-sided method returns because it only ever works with
$\mathbf{A}$. Production libraries use methods of the second kind. The Gram
route remains the right way to *understand* the SVD, and is acceptable when the
singular values of interest are well above the noise.

**Decide numerical rank from the spectrum, with a threshold stated in
advance.** For the $8 \times 6$ table, the singular values are $20.79, 10.40,
5.20, 0.068, 0.035, 0.020$. Counting the values larger than a tolerance
$\tau\,\sigma_1$ gives rank $3$ for any $\tau$ between $0.0033$ and $0.25$, a
wide window; if the values had decayed smoothly with no cliff, the rank would
depend on a choice, and the honest report is the whole spectrum. The
determinant gives no such window, as the $0.1\,\mathbf{I}_n$ example showed.

**Do only as much decomposition as you need.** Power iteration finds the top
direction with one matrix-vector product per step; finding the top $k$
(repeat with the found directions removed, or iterate on $k$ vectors at once)
costs far less than all $n$ for $k \ll n$. A full decomposition is cubic.

## Eigensolvers, rotations and adapters in brain

The brain engine uses a symmetric eigensolver, orthogonal rotations of
activations, low-rank adapters and rotary position embeddings. Four places show them, and two stories
show what they cost.

**The symmetric eigensolver.** `crates/model/src/eigen.rs` has one public
function, `symmetric_eigen(a, n)`, which returns the eigenvalues ascending and
the unit eigenvectors as rows of a flat array. It is the spectral theorem of
this chapter computed for matrices as wide as a model's hidden state:
Householder reflections (orthogonal matrices with determinant $-1$) reduce the
matrix to tridiagonal form, and a sequence of plane rotations, the same
$\mathbf{R}(\theta)$ as here, then drives it to diagonal, with every rotation
accumulated into the eigenvector matrix. The Jacobi sweeps of the pseudocode
above would make many passes of order $n^3$ each, while this pair of routines
(known as `tred2` and `tql2` from an old numerical library) makes one. Its test, `decomposes_symmetric_matrices`, checks
the chapter's three properties on a random $37 \times 37$ matrix and on a
deliberately degenerate one with a repeated eigenvalue and a zero block: every
returned pair satisfies $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ to
$10^{-10}$ times the largest eigenvalue magnitude (at least $1$), the eigenvalues ascend, and the
vectors are orthonormal, $\mathbf{v}_i\cdot\mathbf{v}_j$ equal to $1$ or $0$.
The degenerate case is the one the section on symmetric matrices flagged: with a
repeated eigenvalue the vectors are not unique, so the test checks the
properties and never compares against fixed vectors.

**Why it exists, and its story.** Commit `655f7927f` (26 October 2026) added the
module. Its message gives the reason in the vocabulary of this chapter: the
principal directions of hidden-state covariances need eigenvectors of matrices
as wide as a model's residual stream, and an earlier routine based on cyclic
Jacobi rotations made many cubic sweeps where the new one makes a single cubic
pass.

The same day, commit `a882fe8d3` added the program that uses it (`samples/text/spectral-lm/src/slice.rs`), a tool that
pools the second moment of a Qwen3 model's residual rows at every point of the
stream over calibration text, writes the eigenvectors with the largest
eigenvalue first, and scores the model sliced to its first $K$ directions. That
is the data-matrix $\mathbf{X}^\top\mathbf{X}$ reading of low-rank
approximation at the end of the previous section: keep the $K$ directions of
greatest energy.

**An orthogonal matrix applied to a whole model.**
`crates/qwen3/tests/it/slice.rs` builds a random orthonormal basis for the
test as the eigenvectors of a random symmetric matrix (using `symmetric_eigen`
as the source of an orthogonal $\mathbf{Q}$), then checks that rotating a
decoder's residual stream into that basis, with every direction kept, gives the
same logits. This is the orthogonality section at work on a real network:
every layer that reads the stream is multiplied by $\mathbf{Q}^\top$ on the
right and every layer that writes it by $\mathbf{Q}$, so $(\mathbf{W}
\mathbf{Q}^\top)(\mathbf{Q}\mathbf{x}) = \mathbf{W}\mathbf{x}$; once the
learned gains of the normalisation are folded into the layers that read the
stream, as the slicing does, what remains of the normalisation depends
only on length and is unchanged by $\mathbf{Q}$. Dropping directions then
makes the rotation a projection, which is exactly the truncation of the
low-rank section, and the test fixes the exact end of the scale (all
directions kept) so that any loss in a sliced model can only come from what was
dropped.

**Low-rank adapters.** `crates/model/src/lora.rs` holds the host-side LoRA
machinery. Its `Pair` is the factorisation of this chapter: `a` of shape
`[r, in]`, `b` of shape `[out, r]`, `Pair::new` draws `a` at random and sets
`b` to zero ("initial no-op", the table above), and `Pair::delta` adds
`scale * B * A` to a weight in `[out, in]` layout with `scale = alpha / rank`.
Applying the adapter at $\mathbf{B} = \mathbf{0}$ must reproduce the base
weights exactly, so the fold must give the same bits as a serial walk: it runs
the output rows in parallel but gives each row its own loop over the rank in
ascending order, and skips zero entries of `b`.
The same module reads adapters written by other tools in two shapes, and
distinguishes them on purpose: LoRA, a low-rank product $\Delta\mathbf{W} =
(\alpha/r)\mathbf{B}\mathbf{A}'$, and LoKr, a Kronecker product, which is
"full rank in general". Both are ways of describing a large change with few
numbers, and only the first is the one this chapter's approximation theorem is
about.

Commit `4fa58dabb` (10 August 2026) is the story of that module's origin: the
machinery of a pair of factors (initialisation, applying the delta, projecting
a weight gradient onto the two factors, the Adam moments) had been written
twice, almost verbatim, for two image models, and the commit hoisted it into
one place, deleting 257 lines for 178 added. The models keep only what differs:
which layers are targeted, how fused tensors are offset, and the initial
distribution of `a`, passed in as a closure so that existing seeds still produce
bit-identical adapters.

A property of the zero start is recorded as knowledge #86. A single-example
overfit of a tiny randomly initialised model with a LoRA adapter plateaued
at a nonzero loss that did not move with rank, learning rate or clipping, and
the control (the same model trained fully) fell to zero. The explanation:
the gradient reaching $\mathbf{A}$ passes through $\mathbf{B}$, so with
$\mathbf{B} = \mathbf{0}$ it is exactly zero and stays small while $\mathbf{B}$ is small. A
random model with a low-rank constraint can settle in a local minimum before
$\mathbf{A}$ moves, whereas a pretrained base rarely does. (The derivation is in
Volume III; for now the lesson is to run the full-training control before
suspecting the wiring.) Knowledge #305 records a question about size: brain's
default adapter rank is $8$ whereas published recipes for code fine-tuning
use far higher ranks, up to $256$ (the entry cites a study of LoRA against
full fine-tuning). How low a rank suffices is a property of the task, not a
number to hard-code, which is the reason a spectrum plot beats a guess.

**Rotations in the attention kernels.** `crates/kernels/wgsl/rope_base.wgsl`
applies a rotary embedding. Its header gives the rotation for channel $m$ and
its partner $m + \mathrm{head\_dim}/2$ at position $p$:

```text
out[m]      = x[m]*cos(theta) - x[m+half]*sin(theta)
out[m+half] = x[m+half]*cos(theta) + x[m]*sin(theta)
```

with angle $\theta_m = p \cdot \mathrm{base}^{-2m/\mathrm{head\_dim}}$. That
is $\mathbf{R}(\theta_m)$ applied to the pair $(x_m, x_{m+\mathrm{half}})$, with
the rotation rate $\omega_m = \mathrm{base}^{-2m/\mathrm{head\_dim}}$ falling
geometrically across pairs, and the relative-position property derived above
is what makes the query-key dot product in attention depend on the offset
between two tokens. The header also records a convention choice from the
chapter's section on pairs: the other kernel in the same directory,
`rope_train.wgsl`, rotates adjacent pairs $(2j, 2j+1)$ while this one pairs
$m$ with $m + \mathrm{half}$ to match imported weights; pairing is
a permutation of coordinates, and an orthogonal matrix again. The backward
kernel `rope_base_bwd.wgsl` is the shortest proof of this chapter's
orthogonality section in the repository: the gradient of a rotation is the
transposed, that is inverse, rotation, and the code multiplies by
$\mathbf{R}(\theta)^\top = \mathbf{R}(-\theta)$.
Finally `rope_base_yarn.wgsl` multiplies both cosine and sine by an
`attention_factor` before rotating. That scales the matrix by the factor, so
it is no longer orthogonal (lengths grow by exactly that factor), and a factor
of $1$ reproduces the plain kernel exactly.

**Low-rank structure in an index.** The test
`pca_projections_are_orthonormal_leading_directions` in
`crates/model/src/lsa/engine_tests.rs` (added with commit `6e4a59caa`,
4 November 2026) builds key vectors concentrated on the first four of sixteen
channels, runs the principal-direction fit in `crates/model/src/lsa/indexer.rs`
(which calls `symmetric_eigen` on the uncentred second moment), and checks that
the four returned columns have pairwise dot products $\delta_{ab}$ (orthonormal,
$\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$) and that almost nothing of each lies
outside the four channels where the data lives. That is the whole chapter
in one test: build structured data, extract directions with the symmetric
eigensolver, verify orthonormality, verify that they capture the low-rank core.

## Check your understanding

Answers are at the end of the chapter.

1. **Find the error.** "Since the determinant measures area scaling, $\det(\mathbf{A} + \mathbf{B}) = \det\mathbf{A} + \det\mathbf{B}$." Give a $2 \times 2$ counterexample with $\mathbf{A} = \mathbf{B} = \mathbf{I}$ and say which rule of the section does hold for products.
2. **Predict.** A $2 \times 2$ matrix has trace $5$ and determinant $6$. What are its eigenvalues, the eigenvalues of $\mathbf{A}^2$ and of $\mathbf{A}^{-1}$, and by what factor does it scale area?
3. **Explain.** Power iteration is run on a symmetric matrix with eigenvalues $10$ and $9.9$ (and smaller ones). Roughly how many iterations are needed to shrink the direction error by a factor of $10^6$, and why is the answer so different from the case of $10$ and $2$?
4. **Find the error.** "The matrix with columns $(1, 1)$ and $(1, -1)$ is orthogonal because its columns are orthogonal." Compute $\mathbf{Q}^\top\mathbf{Q}$ and say what the map does to the length of a vector.
5. **Predict.** Without a calculator, what is $\mathbf{R}(40^\circ)\mathbf{R}(50^\circ)$? What is $\mathbf{R}(40^\circ)^\top$?
6. **Predict.** A $2 \times 2$ matrix has singular values $6$ and $0.5$. What is $|\det|$, what is the longest image of a unit vector, what is its rank, and what does it do to the unit circle?
7. **Find the error.** "To get the best rank-1 approximation of a matrix, keep the column with the largest length and set the other columns to zero; Eckart-Young guarantees this is optimal." Which part is wrong, and which table in the chapter shows it?
8. **Explain.** For which ranks $r$ does the factored form $\mathbf{B}\mathbf{A}'$ of an $m \times n$ matrix use fewer numbers than the original? Evaluate for a $100 \times 100$ and a $4096 \times 4096$ matrix.

## Exercises

1. **[Calculate]** For $\begin{pmatrix}2&3\\1&4\end{pmatrix}$ and $\begin{pmatrix}2&4\\1&2\end{pmatrix}$ compute the determinant, the signed area of the image of the unit square by the shoelace formula, and say whether each is invertible. Then compute $\det\begin{pmatrix}1&2&0\\3&1&4\\0&2&1\end{pmatrix}$ by expansion along the first row and by elimination, and check that they agree.

2. **[Calculate]** Find the eigenvalues and eigenvectors of $\begin{pmatrix}5&2\\2&2\end{pmatrix}$ by the characteristic polynomial, check the trace and determinant relations, and check that the eigenvectors are orthogonal. Do the same for $\begin{pmatrix}2&3\\0&5\end{pmatrix}$ and show that the eigenvalues of a triangular matrix are its diagonal entries.

3. **[Calculate]** The matrix $\mathbf{A} = \begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}$ has eigenvectors $(1, 1)$ and $(1, -1)$. Find the eigenvalues, write $\mathbf{A}^k$ in closed form with $\mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1}$, and find the limit of $\mathbf{A}^k$ as $k \to \infty$. What does the limit do to $(1, 0)$?

4. **[Calculate]** Is $\begin{pmatrix}0.6&0.8\\-0.8&0.6\end{pmatrix}$ orthogonal? Is it a rotation or a reflection, and by what angle (use $\cos\theta = 0.6$)? Prove that a product of two orthogonal matrices is orthogonal, and that a $2\times 2$ orthogonal matrix with determinant $-1$ times one with determinant $-1$ is a rotation.

5. **[Calculate]** Compute $\mathbf{R}(60^\circ)\mathbf{R}(30^\circ)$ from the matrices and from $\mathbf{R}(90^\circ)$. Set $\alpha = \beta = \theta$ in the angle-addition formulas to derive $\cos 2\theta$ and $\sin 2\theta$, and check them at $\theta = 30^\circ$.

6. **[Calculate]** Find the singular value decomposition of $\begin{pmatrix}2&2\\-1&1\end{pmatrix}$ by hand with $\mathbf{A}^\top\mathbf{A}$: the eigenvalues, $\sigma_i$, the $\mathbf{v}_i$ and $\mathbf{u}_i$. Verify $\sigma_1\sigma_2 = |\det\mathbf{A}|$ and describe the ellipse the unit circle becomes.

7. **[Calculate]** The matrix $\begin{pmatrix}1&2\\2&4\\3&6\end{pmatrix}$ has rank $1$. Write it as $\mathbf{u}\mathbf{v}^\top$, compute its single singular value, and state the number of parameters of the factored form against the original. For a $1000 \times 500$ matrix at rank $10$, compute the number of stored numbers and the ratio to $mn$.

8. **[Implement]** Implement `power_iteration` and reproduce `decompositions.power_iteration_trace` and `decompositions.power_iteration_rates` (the three shifts, the start vector $(1, 0, 0)$, the measured per-step ratio taken between iterations $10$ and $20$). Plot $\log_{10}$ of the error (or, without a plotting library, print the slope between iterations 10 and 20) and compare the slope with $\log_{10}(|\lambda_2|/|\lambda_1|)$: for the three shifts it is $-0.3828$, $-0.2323$ and $-0.1314$.

9. **[Implement]** Implement `jacobi_eigen` and `svd_via_gram`, and reproduce `decompositions.symmetric_orthogonal` and `decompositions.svd_by_hand`. Then implement `svd_one_sided` (with 60 sweeps) and reproduce `decompositions.lauchli_squaring`, including the failure of the Gram route. Both examples have equal diagonal entries (or equal squared column lengths), so they test the $\mathrm{sign}(0) = +1$ rule.

10. **[Implement]** Implement $\mathbf{R}(\theta)$, the block rotation of pairs and the check that rotated dot products depend only on the offset. Reproduce `decompositions.rotation_matrix_table`, `decompositions.rotation_composition` and `decompositions.rotation_relative_position`.

11. **[Implement]** Build the $8 \times 6$ table $\mathbf{M}$ of the text, and reproduce `decompositions.spectrum_table` and `decompositions.truncation_error` (the Frobenius error must equal the root of the dropped squares, and the largest stretch of the residual must equal $\sigma_{r+1}$). Then reproduce the first two columns of `decompositions.eckart_young_check`. The last two use the generator that Chapter 17 builds, seeded and drawn in the order printed under the block; once you have that generator, reproduce them digit for digit, and until then check that the noisy-copy column is larger than the first and that no random subspace you draw yourself beats the first column.

12. **[Investigate]** (a) Start power iteration exactly at a vector orthogonal to $\mathbf{v}_1$ (for the tridiagonal matrix, $(1, 0, -1)$) and watch the iterate for $100$ steps: what does it converge to, and what does round-off do? (b) Try a matrix whose two largest eigenvalues are $+3$ and $-3$. (c) Change the noise amplitude of the $8 \times 6$ table from $0.05$ to $0.5$ and to $2$ and find where the cliff in the spectrum disappears. At what noise level would you stop calling the table "rank 3"? Justify your choice from the spectrum.

## Summary

The **determinant** of a square matrix is the signed factor by which it scales
area or volume: $ad - bc$ for $2 \times 2$, the signed volume of the box on the
columns in three dimensions, zero exactly when the map flattens space. Products
multiply, inverses divide, the transpose changes nothing, and the number is
better computed from the pivots of elimination than from cofactors, and not
at all as a test of invertibility in floating point, where it underflows for
healthy matrices. An **eigenvector** of a matrix is a direction it only
stretches, and the **eigenvalue** is the stretch factor; they solve
$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$, sum to the trace and multiply to
the determinant, make repeated application $\mathbf{A}^k = \mathbf{V}
\boldsymbol{\Lambda}^k\mathbf{V}^{-1}$ trivial, and for a large matrix are found
by **power iteration**, whose error shrinks by $|\lambda_2|/|\lambda_1|$ per
step. A symmetric matrix has real eigenvalues and orthogonal eigenvectors,
the second proved from the adjoint identity of Chapter 4.

An **orthogonal matrix** has orthonormal columns, satisfies
$\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ (the entry $(i, j)$ is a dot product of
columns), so its inverse is its transpose and it preserves every dot product,
hence every length and angle; its determinant is $\pm1$. A **rotation matrix**
in the plane is $\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta
\end{pmatrix}$, built from the images of the unit vectors on the unit circle;
rotations compose by adding angles, and rotating pairs of coordinates by angles
proportional to position makes a dot product depend only on the offset. The
**singular value decomposition** $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}
\mathbf{V}^\top$ exists for every matrix, of any shape: rotate, stretch
by the **singular values** $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$,
rotate again; its number of nonzero values is the rank, its first value is the
largest stretch. Writing it as a sum of rank-one pieces and keeping the first
$r$ is the **low-rank approximation**; its error is exactly the dropped singular
values (in Frobenius norm the root of the sum of their squares, in the largest
stretch the next one), no other rank-$r$ matrix does better (Eckart-Young),
and storing it takes $r(m + n)$ numbers instead of $mn$. That is the arithmetic behind low-rank
adaptation of weights and behind keeping the leading directions of activations.

The limits are practical. Singular vectors are unique only up to sign and
repeated values, so tests compare reconstructions; computing the SVD through
$\mathbf{A}^\top\mathbf{A}$ loses small singular values; the full decomposition
is cubic; and whether a given matrix is close to low rank is a fact about the
matrix that its spectrum reveals, never an assumption. Everything in the
chapter described a matrix as it is. What we cannot yet do is describe how a
quantity *changes* when its inputs move a little, which is what learning
needs. The next chapter starts from the simplest case, a function of one
variable, and builds the derivative.

## Key terms

- **determinant**: the signed factor by which a square matrix scales area or volume ($ad - bc$ for a $2 \times 2$); zero exactly when the matrix crushes space and has no inverse. <!-- concept: determinant -->
- **eigenvalue**: the number $\lambda$ by which a matrix stretches one of its eigenvectors, $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$; the roots of $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$. <!-- concept: eigenvalue -->
- **eigenvector**: a nonzero vector that a matrix sends to a multiple of itself, so that the map only stretches (or flips) it. <!-- concept: eigenvector -->
- **orthogonal matrix**: a square matrix whose columns are orthonormal, so that $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$, its inverse is its transpose, and it preserves lengths and angles. <!-- concept: orthogonal-matrix -->
- **rotation matrix**: an orthogonal matrix with determinant $+1$; in the plane $\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$, and rotations compose by adding their angles. <!-- concept: rotation-matrix -->
- **singular value decomposition**: the factorisation $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ of any matrix into a rotation, a stretch by non-negative singular values and a second rotation. <!-- concept: singular-value-decomposition -->
- **low-rank approximation**: replacing a matrix by one of small rank, best obtained by keeping the largest singular values; it stores $r(m + n)$ numbers instead of $mn$ and its error is the dropped singular values. <!-- concept: low-rank-approximation -->

## Answers

**Recall questions.**

1. $\mathbf{A}\mathbf{B} = \begin{pmatrix}3&2\\1&1\end{pmatrix}$ and $(\mathbf{A}\mathbf{B})^\top = \begin{pmatrix}3&1\\2&1\end{pmatrix}$, which equals $\mathbf{B}^\top\mathbf{A}^\top$ (the order reverses).
2. $4\cdot 6 - 7 \cdot 2 = 10 \ne 0$, so the inverse exists. A $2 \times 2$ matrix with $ad - bc = 0$ has no inverse: it crushes the plane onto a line (or a point).
3. Rank $1$: the second row is twice the first. It sends the unit square to a segment of the line through $(1, 2)$, area $0$.
4. $\sqrt{1 + 4 + 4} = 3$. The dot product $1\cdot 2 + 2\cdot 1 + 2\cdot(-2) = 0$, so yes, orthogonal.
5. $\cos 90^\circ = 0$, $\sin 90^\circ = 1$; $\cos^2\theta + \sin^2\theta = 1$ for every $\theta$ because the point $(\cos\theta, \sin\theta)$ lies on the unit circle.
6. $(\lambda - 1)(\lambda - 3) = 0$, so $\lambda = 1$ or $3$.

**Check your understanding.**

1. With $\mathbf{A} = \mathbf{B} = \mathbf{I}_2$, $\det(\mathbf{A} + \mathbf{B}) = \det(2\mathbf{I}) = 4$ but $\det\mathbf{A} + \det\mathbf{B} = 2$. The rules that hold are for products, $\det(\mathbf{A}\mathbf{B}) = \det\mathbf{A}\det\mathbf{B}$, and for scaling, $\det(c\mathbf{A}) = c^n\det\mathbf{A}$ (here $2^2 \cdot 1 = 4$).
2. The eigenvalues solve $\lambda^2 - 5\lambda + 6 = 0$: $2$ and $3$. $\mathbf{A}^2$ has $4$ and $9$; $\mathbf{A}^{-1}$ has $\tfrac12$ and $\tfrac13$. Area is scaled by $\det = 6$.
3. The error shrinks by $0.99$ per step, so $0.99^k = 10^{-6}$ gives $k = \ln(10^{-6})/\ln(0.99) \approx 1375$ iterations. For $10$ and $2$ the ratio is $0.2$ and $k = \ln(10^{-6})/\ln 0.2 \approx 9$. The speed depends on the ratio $|\lambda_2|/|\lambda_1|$, which is close to $1$ in the first case, not on the size of the eigenvalues.
4. $\mathbf{Q}^\top\mathbf{Q} = \begin{pmatrix}2&0\\0&2\end{pmatrix} = 2\mathbf{I}$: the columns are orthogonal but have length $\sqrt2$, not $1$. The map multiplies every length by $\sqrt2$. Orthogonal columns are not enough; they must be orthonormal.
5. $\mathbf{R}(40^\circ)\mathbf{R}(50^\circ) = \mathbf{R}(90^\circ) = \begin{pmatrix}0&-1\\1&0\end{pmatrix}$, and $\mathbf{R}(40^\circ)^\top = \mathbf{R}(-40^\circ)$.
6. $|\det| = 6 \times 0.5 = 3$. The longest image of a unit vector has length $6$ (the shortest $0.5$). Rank $2$ (both nonzero). The unit circle becomes an ellipse with semi-axes $6$ and $0.5$.
7. Keeping the longest column and zeroing the rest is not the truncated SVD, and Eckart-Young only guarantees the truncated SVD is optimal. The second column of the `eckart_young_check` table, at $r = 1$, shows projecting onto the longest column giving $14.09$ against $11.63$ for the SVD at $r = 1$.
8. $r(m + n) < mn$ means $r < mn/(m + n)$. For $100 \times 100$: $r < 50$ (at $r = 50$ the counts are equal, $10\,000$). For $4096 \times 4096$: $r < 2048$.
